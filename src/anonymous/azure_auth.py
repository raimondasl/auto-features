"""Microsoft Entra ID tokens for Azure OpenAI, from the Azure CLI: no API key, no SDK.

Azure OpenAI's v1 API takes the same Chat Completions body Anonymous already sends, at
``{endpoint}/openai/v1/chat/completions``, with an Entra token in the same ``Authorization:
Bearer`` header an OpenAI key goes in. So keyless Azure is a transport change rather than a new
client, and the one new capability it needs is a token. This module gets one the way
azure-identity's ``AzureCliCredential`` does -- by running ``az account get-access-token`` --
without adding azure-identity's dependency tree to every plugin install.

Established live on 2026-09-13 against a resource with key authentication disabled (PLANS item
17): both the ``cognitiveservices.azure.com`` and ``ai.azure.com`` audiences were accepted on all
three host forms below, so the audience the v1 OpenAPI spec and the SDK examples name is used.

Two properties are security-relevant, not tidiness:

* **Tokens go only to Azure hosts.** ``.anonymous.yml`` is committed to repositories, and a
  config written into someone else's repository must not be able to send the user's Entra token
  anywhere else. :func:`chat_completions_url` refuses any endpoint that is not an ``https``
  resource URL under the three Azure OpenAI host suffixes.
* **The tenant is validated before it reaches ``az``.** It also comes from committed config, and
  on Windows ``az`` is a batch file run through ``cmd.exe``, where an unquoted metacharacter in an
  argument is a command. Only the characters a tenant id or domain can contain are accepted.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from typing import Any
from urllib.parse import urlparse

from anonymous.executables import find_on_path

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

# How long a failed fetch is answered from memory. Long enough for the callers queued behind it;
# shorter than anyone takes to switch to a terminal and complete `az login`.
FAILURE_TTL_SECONDS = 15.0

_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?")
_TENANT = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9.-]{0,252}[A-Za-z0-9])?")
_ALLOWED_PATHS = ("", "/openai", "/openai/v1")

_lock = threading.Lock()
_cache: dict[str, tuple[str, float]] = {}
_failures: dict[str, tuple[str, float]] = {}  # tenant -> (message, monotonic deadline)


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
            "No Azure OpenAI endpoint. Set azure_openai.endpoint in .anonymous.yml to the "
            "Endpoint the Azure portal shows for the resource, e.g. "
            "https://<resource>.openai.azure.com or https://<resource>.cognitiveservices.azure.com"
        )
    try:
        parsed = urlparse(raw if "://" in raw else f"https://{raw}")
        host = (parsed.hostname or "").lower()
    except ValueError:  # "Invalid IPv6 URL" -- raised, not reported, by urlparse
        raise AzureAuthError(
            f"Refusing Azure OpenAI endpoint {raw!r}: it is not a URL. Tokens are only sent to "
            f"Azure resource hosts."
        ) from None
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
    """The Azure CLI, as an absolute path. On Windows `az` is `az.cmd`.

    Never looked up in the working directory, which is the repository being profiled: on Windows
    ``shutil.which`` searches there first, and a committed ``az.exe`` would receive the call and
    the token it returns. See :mod:`anonymous.executables`.
    """
    if sys.platform == "win32":
        return find_on_path("az.cmd") or find_on_path("az")
    return find_on_path("az")


def get_token(tenant: str = "") -> str:
    """An Entra access token for Azure OpenAI, cached until shortly before it expires.

    Held under a lock across the `az` call itself: fifty gate calls arriving while the first token
    is being fetched should wait for that one fetch, not start fifty `az` processes. A failure is
    remembered for :data:`FAILURE_TTL_SECONDS` for the same reason -- callers queued behind a
    fetch that failed get its answer rather than each paying for another -- and no longer, so a
    user who has just run `az login` is not told to run it again.
    """
    key = validate_tenant(tenant)
    with _lock:
        cached = _cache.get(key)
        if cached is not None and cached[1] - REFRESH_MARGIN_SECONDS > time.time():
            return cached[0]
        failed = _failures.get(key)
        if failed is not None and failed[1] > time.monotonic():
            raise AzureAuthError(failed[0])
        try:
            token, expires_at = _fetch(key)
        except AzureAuthError as exc:
            _failures[key] = (str(exc), time.monotonic() + FAILURE_TTL_SECONDS)
            raise
        _failures.pop(key, None)
        _cache[key] = (token, expires_at)
        return token


def forget(tenant: str = "") -> None:
    """Drop *tenant*'s cached token, so the next call asks `az` again.

    For a token Azure refused. A long-lived server would otherwise keep sending it for up to an
    hour after the user fixed the cause -- signed in to the right account, or into the tenant
    that holds the resource -- which is exactly what the refusal message tells them to do.
    """
    with _lock:
        _cache.pop((tenant or "").strip(), None)
        _failures.pop((tenant or "").strip(), None)


def clear_cache() -> None:
    with _lock:
        _cache.clear()
        _failures.clear()


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
        done = _run(command, timeout=AZ_TIMEOUT_SECONDS)
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
            "The Azure CLI returned a token response Anonymous could not read."
        ) from None
    if not token:
        raise AzureAuthError("The Azure CLI returned an empty token.")
    return token, _expires_at(data)


def _run(command: list[str], *, timeout: float) -> subprocess.CompletedProcess[str]:
    """Run *command* to completion, or kill it and everything it started after *timeout*.

    Not ``subprocess.run(timeout=)``: that kills only the process it started, and `az` is a
    wrapper -- ``az.cmd`` runs ``python.exe`` through ``cmd.exe`` on Windows, a shell script runs
    it elsewhere -- whose child inherits the output pipes. ``run`` then waits, with no timeout at
    all, for pipes the surviving child still holds, so a stalled `az` stalled Anonymous for as
    long as it liked, under a lock every other Azure call was queued on.
    """
    windows = sys.platform == "win32"
    process = subprocess.Popen(
        command,
        # Never the caller's stdin. Under the MCP server that is the editor's JSON-RPC pipe, and a
        # child sharing it deadlocks on Windows (see anonymous.delegate).
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={**os.environ, "AZURE_CORE_NO_COLOR": "true"},
        # Not the repository, so nothing `az` itself looks up relative to its working directory
        # comes from there. (The program path is already absolute: see az_executable.)
        cwd=os.environ.get("SYSTEMROOT") if windows else "/",
        start_new_session=not windows,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_tree(process)
        with contextlib.suppress(subprocess.TimeoutExpired, OSError, ValueError):
            process.communicate(timeout=5)  # reap; a child that still holds a pipe is abandoned
        raise
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def _kill_tree(process: subprocess.Popen[str]) -> None:
    if sys.platform == "win32":
        # By absolute path, for the reason az_executable is. `/T` takes the children with it,
        # which only works while the parent is still alive -- so before `kill`, not after.
        taskkill = os.path.join(
            os.environ.get("SYSTEMROOT", r"C:\Windows"), "System32", "taskkill.exe"
        )
        with contextlib.suppress(OSError, subprocess.SubprocessError):
            subprocess.run(
                [taskkill, "/T", "/F", "/PID", str(process.pid)],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=10,
            )
    else:
        with contextlib.suppress(OSError):
            os.killpg(process.pid, signal.SIGKILL)
    with contextlib.suppress(OSError):
        process.kill()


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
