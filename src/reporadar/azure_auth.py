"""Microsoft Entra ID tokens for Azure OpenAI, from the Azure CLI: no API key, no SDK.

Azure OpenAI's v1 API takes the same Chat Completions body RepoRadar already sends, at
``{endpoint}/openai/v1/chat/completions``, with an Entra token in the same ``Authorization:
Bearer`` header an OpenAI key goes in. So keyless Azure is a transport change rather than a new
client, and the one new capability it needs is a token. This module gets one the way
azure-identity's ``AzureCliCredential`` does -- by running ``az account get-access-token`` --
without adding azure-identity's dependency tree to every plugin install.

Established live on 2026-09-13 against a resource with key authentication disabled (PLANS item
17): both the ``cognitiveservices.azure.com`` and ``ai.azure.com`` audiences were accepted on all
three host forms below, so the audience the v1 OpenAPI spec and the SDK examples name is used.

Two properties are security-relevant, not tidiness:

* **Tokens go only to Azure hosts.** ``.reporadar.yml`` is committed to repositories, and a
  config written into someone else's repository must not be able to send the user's Entra token
  anywhere else. :func:`chat_completions_url` refuses any endpoint that is not an ``https``
  resource URL under the three Azure OpenAI host suffixes.
* **The tenant is validated before it reaches ``az``.** It also comes from committed config, and
  on Windows ``az`` is a batch file run through ``cmd.exe``, where an unquoted metacharacter in an
  argument is a command. Only the characters a tenant id or domain can contain are accepted.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from typing import Any
from urllib.parse import urlparse

AUDIENCE = "https://cognitiveservices.azure.com"

# The Azure OpenAI / Foundry resource host forms, each a single subdomain label under the suffix.
# `cognitiveservices.azure.com` is undocumented for the v1 API but accepted it in the live probe,
# and it is the endpoint `az cognitiveservices account show` prints, so users will paste it.
ALLOWED_HOST_SUFFIXES = (
    ".openai.azure.com",
    ".services.ai.azure.com",
    ".cognitiveservices.azure.com",
)

# Refresh this long before expiry, so a token never expires between being handed out and the
# request that carries it -- the gate makes up to 50 sequential calls.
REFRESH_MARGIN_SECONDS = 300

# `az account get-access-token` has been measured at 10-15 s on some CLI versions (Azure CLI issues
# #29871, #29329), which is what broke the 10 s timeouts other tools used. Generous on purpose: a
# token fetch is paid once per ~hour, and a false timeout costs the whole gate.
AZ_TIMEOUT_SECONDS = 60

_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?")
_TENANT = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9.-]{0,252}[A-Za-z0-9])?")
_ALLOWED_PATHS = ("", "/openai", "/openai/v1")

_lock = threading.Lock()
_cache: dict[str, tuple[str, float]] = {}


class AzureAuthError(Exception):
    """Azure OpenAI could not be reached keylessly. The message says what to do about it."""


def chat_completions_url(endpoint: str) -> str:
    """The v1 Chat Completions URL for *endpoint*, or AzureAuthError if it is not an Azure resource.

    Accepts what users actually paste: ``https://<name>.openai.azure.com``, with or without a
    trailing slash or an ``/openai/v1`` suffix, the ``services.ai.azure.com`` and
    ``cognitiveservices.azure.com`` forms, and a bare host. Returns the normalised URL, built from
    the validated host alone, so nothing else from the configured string reaches the request.
    """
    raw = (endpoint or "").strip()
    if not raw:
        raise AzureAuthError(
            "No Azure OpenAI endpoint. Set azure_openai.endpoint in .reporadar.yml to your "
            "resource URL, e.g. https://<resource>.openai.azure.com"
        )
    parsed = urlparse(raw if "://" in raw else f"https://{raw}")
    host = (parsed.hostname or "").lower()
    suffix = next((s for s in ALLOWED_HOST_SUFFIXES if host.endswith(s)), None)
    label = host[: -len(suffix)] if suffix else ""
    problems = []
    if parsed.scheme != "https":
        problems.append("it is not https")
    if suffix is None or not host.isascii() or not _LABEL.fullmatch(label):
        problems.append(
            "its host is not an Azure OpenAI resource (<resource>"
            + " or <resource>".join(ALLOWED_HOST_SUFFIXES)
            + ")"
        )
    if parsed.username or parsed.password:
        problems.append("it carries credentials")
    try:
        port = parsed.port
    except ValueError:
        port = -1
    if port not in (None, 443):
        problems.append("it names a port")
    if parsed.path.rstrip("/") not in _ALLOWED_PATHS or parsed.query or parsed.fragment:
        problems.append("it has a path; use the resource URL itself")
    if problems:
        # The configured value is quoted so the user can see the typo -- it is not a secret, and
        # the point of refusing is that it never becomes a request.
        raise AzureAuthError(
            f"Refusing Azure OpenAI endpoint {raw!r}: {'; '.join(problems)}. Tokens are only "
            f"sent to Azure resource hosts."
        )
    return f"https://{host}/openai/v1/chat/completions"


def validate_tenant(tenant: str) -> str:
    """*tenant* stripped, or AzureAuthError if it could be anything but a tenant id or domain."""
    value = (tenant or "").strip()
    if value and not _TENANT.fullmatch(value):
        raise AzureAuthError(
            f"azure_openai.tenant {value!r} is not a tenant id or domain; only letters, digits, "
            f"'.' and '-' are allowed."
        )
    return value


def az_executable() -> str | None:
    """The Azure CLI, found the way azure-identity finds it. On Windows `az` is `az.cmd`."""
    if sys.platform == "win32":
        return shutil.which("az.cmd") or shutil.which("az")
    return shutil.which("az")


def get_token(tenant: str = "") -> str:
    """An Entra access token for Azure OpenAI, cached until shortly before it expires.

    Held under a lock across the `az` call itself: fifty gate calls arriving while the first token
    is being fetched should wait for that one fetch, not start fifty `az` processes.
    """
    key = validate_tenant(tenant)
    with _lock:
        cached = _cache.get(key)
        if cached is not None and cached[1] - REFRESH_MARGIN_SECONDS > time.time():
            return cached[0]
        token, expires_at = _fetch(key)
        _cache[key] = (token, expires_at)
        return token


def clear_cache() -> None:
    with _lock:
        _cache.clear()


def _fetch(tenant: str) -> tuple[str, float]:
    az = az_executable()
    if az is None:
        raise AzureAuthError(
            "The Azure CLI (`az`) is not on PATH. Install it and run `az login`. If you installed "
            "it after starting your editor, restart the editor: a server it launched still has "
            "the PATH from before the install."
        )
    command = [az, "account", "get-access-token", "--resource", AUDIENCE, "--output", "json"]
    if tenant:
        command += ["--tenant", tenant]
    try:
        done = subprocess.run(
            command,
            # Never the caller's stdin. Under the MCP server that is the editor's JSON-RPC pipe,
            # and a child sharing it deadlocks on Windows (see reporadar.delegate).
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=AZ_TIMEOUT_SECONDS,
            env={**os.environ, "AZURE_CORE_NO_COLOR": "true"},
            # Not the repository: a command lookup relative to the working directory must never
            # find something the repository put there.
            cwd=os.environ.get("SYSTEMROOT") if sys.platform == "win32" else "/",
        )
    except subprocess.TimeoutExpired:
        raise AzureAuthError(
            f"`az account get-access-token` did not answer within {AZ_TIMEOUT_SECONDS} s."
        ) from None
    except OSError as exc:
        raise AzureAuthError(f"Could not run the Azure CLI at {az}: {exc}") from exc
    if done.returncode != 0:
        raise AzureAuthError(_explain(done.stderr, tenant))
    try:
        data = json.loads(done.stdout)
        token = str(data["accessToken"])
    except (ValueError, KeyError, TypeError):
        raise AzureAuthError(
            "The Azure CLI returned a token response RepoRadar could not read."
        ) from None
    if not token:
        raise AzureAuthError("The Azure CLI returned an empty token.")
    return token, _expires_at(data)


def _expires_at(data: dict[str, Any]) -> float:
    """When the token expires, as epoch seconds.

    ``expires_on`` (epoch, UTC) is what current CLIs return. The older ``expiresOn`` is a naive
    LOCAL datetime, and parsing it is a timezone bug waiting to happen, so a response carrying
    only that is cached briefly instead of trusted.
    """
    try:
        return float(data["expires_on"])
    except (KeyError, TypeError, ValueError):
        return time.time() + REFRESH_MARGIN_SECONDS + 300


def _explain(stderr: str, tenant: str) -> str:
    text = (stderr or "").strip()
    login = f"az login --tenant {tenant}" if tenant else "az login"
    if "AADSTS50076" in text or "multi-factor" in text.lower():
        return (
            "Azure requires multi-factor authentication for this token. Run "
            f"`{login}` in a terminal to sign in interactively"
            + ("" if tenant else ", naming the tenant that holds the resource with --tenant")
            + "."
        )
    if "AADSTS70043" in text or "AADSTS700082" in text or "expired" in text.lower():
        return f"Your Azure sign-in has expired. Run `{login}` in a terminal."
    if "az login" in text:
        return f"Not signed in to Azure. Run `{login}` in a terminal."
    first = text.splitlines()[0] if text else "no error output"
    return f"`az account get-access-token` failed: {first[:300]}"
