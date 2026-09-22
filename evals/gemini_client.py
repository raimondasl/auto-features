"""A standard-library client for the Gemini REST API, used by `third_judge.py` and nothing else.

`evals/PREREG-third-judge.md` registers a third judge from a third vendor. The product knows no
Gemini provider, and this study must not add one. So the calls go through `urllib` and `json`
here, inside `evals/`. `pyproject.toml`, `uv.lock` and `src/reporadar` stay untouched.

Three endpoints, and only `generate` costs money:

    generate(body, model, api_key)       POST .../models/<model>:generateContent   (paid)
    get_model(model, api_key)            GET  .../models/<model>                    (free)
    count_tokens(request, model, key)    POST .../models/<model>:countTokens        (free)

The key travels only in the `x-goog-api-key` header. It never enters a URL, an exception
message, a log line or a returned object. Every error text passes through `scrub` before it
leaves this module.

`generate` returns a typed `Outcome` and never raises for an HTTP or network failure. The kinds
map onto the registration's failure classes as follows. `third_judge.classify` makes the final
call, because only it can parse the answer text.

    text        HTTP 200, a candidate whose finishReason is STOP
    finish      HTTP 200, a candidate with any other finishReason (or none)
    blocked     HTTP 200, promptFeedback.blockReason set and no candidate
    malformed   HTTP 200 with neither a candidate nor a block reason, or an undecodable body
    rate_limit  HTTP 429, with the delay from google.rpc.RetryInfo when the error carries one
    key         HTTP 401 or 403, or a 400 whose error reason is API_KEY_INVALID
    transport   HTTP 408, 500, 502, 503 or 504, a timeout, a connection error or a cut-off body
    fatal       any other status

`malformed` is a 200 the server meant to answer, so `third_judge.classify` files it with the
content failures as a `parse` failure, not with the transport ones. Resending the same prompt is
what a garbled answer asks for, and the registration gives a parse failure one retry, not three.
"""

from __future__ import annotations

import http.client
import json
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

HOST = "https://generativelanguage.googleapis.com"
API_VERSION = "v1beta"
KEY_HEADER = "x-goog-api-key"

TEXT = "text"
FINISH = "finish"
BLOCKED = "blocked"
MALFORMED = "malformed"
RATE_LIMIT = "rate_limit"
KEY = "key"
TRANSPORT = "transport"
FATAL = "fatal"
KINDS = (TEXT, FINISH, BLOCKED, MALFORMED, RATE_LIMIT, KEY, TRANSPORT, FATAL)

TRANSPORT_STATUSES = frozenset({408, 500, 502, 503, 504})
KEY_STATUSES = frozenset({401, 403})

# A model id is spliced into a URL path. Anything outside this alphabet is refused, so an id
# read from a file can never add a path segment or a query string.
MODEL_ID = re.compile(r"[a-z0-9][a-z0-9.\-]*")
API_VERSIONS = re.compile(r"v[0-9]+(?:beta|alpha)?[0-9]*")

# (status, body bytes). Raises one of TRANSPORT_ERRORS for a network failure.
Transport = Callable[[urllib.request.Request, float], tuple[int, bytes]]

# What a transport may raise for a network failure. `resp.read()` raises
# `http.client.IncompleteRead` when a connection drops mid-body, and that is an
# `HTTPException`, not an `OSError`. Both are transport outcomes, never a crash.
TRANSPORT_ERRORS = (OSError, http.client.HTTPException)


@dataclass
class Outcome:
    """One attempt, as the API answered it. Nothing here holds the key."""

    kind: str
    http_status: int | None
    text: str = ""
    finish_reason: str | None = None
    block_reason: str | None = None
    prompt_feedback: dict[str, Any] | None = None
    model_version: str | None = None
    response_id: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    latency: float = 0.0
    raw: Any = None
    retry_delay: float | None = None
    error: str = ""

    def record(self) -> dict[str, Any]:
        """The attempt as it is stored in a cache file."""
        return {
            "kind": self.kind,
            "http_status": self.http_status,
            "finish_reason": self.finish_reason,
            "block_reason": self.block_reason,
            "prompt_feedback": self.prompt_feedback,
            "model_version": self.model_version,
            "response_id": self.response_id,
            "usage": self.usage,
            "latency": round(self.latency, 3),
            "retry_delay": self.retry_delay,
            "error": self.error,
            "raw": self.raw,
        }


def scrub(text: str, api_key: str | None) -> str:
    """*text* with every occurrence of the key replaced. Applied to anything that leaves."""
    if api_key:
        text = text.replace(api_key, "<key>")
    return text


def _url(model: str, action: str, api_version: str) -> str:
    if not MODEL_ID.fullmatch(model):
        raise ValueError(f"not a model id: {model!r}")
    if not API_VERSIONS.fullmatch(api_version):
        raise ValueError(f"not an API version: {api_version!r}")
    return f"{HOST}/{api_version}/models/{model}{action}"


def _urlopen(req: urllib.request.Request, timeout: float) -> tuple[int, bytes]:
    """The default transport. An HTTP error status is an answer, not an exception."""
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 -- fixed host
            return int(resp.status), resp.read()
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read()
        except Exception:  # noqa: BLE001 -- an unreadable error body is still a status
            body = b""
        return int(exc.code), body


def _request(url: str, api_key: str, data: dict[str, Any] | None) -> urllib.request.Request:
    headers = {KEY_HEADER: api_key}
    body = None
    if data is not None:
        headers["Content-Type"] = "application/json"
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
    return urllib.request.Request(
        url, data=body, headers=headers, method="POST" if data is not None else "GET"
    )


def _decode(body: bytes, api_key: str | None = None) -> Any:
    """The body as JSON, with the key scrubbed from its text first. None when undecodable."""
    try:
        return json.loads(scrub(body.decode("utf-8"), api_key))
    except (UnicodeDecodeError, ValueError):
        return None


def _error(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict) and isinstance(payload.get("error"), dict):
        return dict(payload["error"])
    return {}


def _details(err: dict[str, Any]) -> list[dict[str, Any]]:
    return [d for d in (err.get("details") or []) if isinstance(d, dict)]


def retry_delay(payload: Any) -> float | None:
    """Seconds from the error's google.rpc.RetryInfo detail, or None when it carries none."""
    for d in _details(_error(payload)):
        if str(d.get("@type", "")).endswith("google.rpc.RetryInfo"):
            m = re.fullmatch(r"\s*([0-9]+(?:\.[0-9]+)?)s\s*", str(d.get("retryDelay", "")))
            if m:
                return float(m.group(1))
    return None


def key_invalid(payload: Any) -> bool:
    """True when a 400's error names API_KEY_INVALID, in an ErrorInfo reason."""
    return any(d.get("reason") == "API_KEY_INVALID" for d in _details(_error(payload)))


def visible_text(candidate: dict[str, Any]) -> str:
    """The answer text: every part whose `thought` is not true, concatenated in order."""
    parts = ((candidate.get("content") or {}).get("parts")) or []
    return "".join(
        str(p.get("text", ""))
        for p in parts
        if isinstance(p, dict) and p.get("thought") is not True and "text" in p
    )


def outcome_of(status: int, body: bytes, latency: float, api_key: str | None) -> Outcome:
    """Classify one HTTP answer. Pure, so recorded fixture responses can be replayed."""
    payload = _decode(body, api_key)
    raw: Any = payload if payload is not None else scrub(body.decode("utf-8", "replace"), api_key)
    base: dict[str, Any] = {"http_status": status, "latency": latency, "raw": raw}
    if status == 200:
        if not isinstance(payload, dict):
            return Outcome(MALFORMED, error="undecodable body", **base)
        usage = dict(payload.get("usageMetadata") or {})
        feedback = payload.get("promptFeedback")
        common: dict[str, Any] = {
            **base,
            "usage": usage,
            "model_version": payload.get("modelVersion"),
            "response_id": payload.get("responseId"),
            "prompt_feedback": feedback if isinstance(feedback, dict) else None,
        }
        candidates = [c for c in (payload.get("candidates") or []) if isinstance(c, dict)]
        if candidates:
            cand = candidates[0]
            reason = cand.get("finishReason")
            kind = TEXT if reason == "STOP" else FINISH
            return Outcome(kind, text=visible_text(cand), finish_reason=reason, **common)
        block = (feedback or {}).get("blockReason") if isinstance(feedback, dict) else None
        if block:
            return Outcome(BLOCKED, block_reason=str(block), **common)
        return Outcome(MALFORMED, error="neither a candidate nor a block reason", **common)
    err = _error(payload)
    # Scrub before the cut: a key straddling character 300 would survive a truncate-then-scrub.
    message = scrub(str(err.get("message") or ""), api_key)[:300]
    if status == 429:
        return Outcome(RATE_LIMIT, retry_delay=retry_delay(payload), error=message, **base)
    if status in KEY_STATUSES or (status == 400 and key_invalid(payload)):
        return Outcome(KEY, error=message, **base)
    if status in TRANSPORT_STATUSES:
        return Outcome(TRANSPORT, error=message, **base)
    return Outcome(FATAL, error=message, **base)


def generate(
    body: dict[str, Any],
    model: str,
    api_key: str,
    timeout: float = 300,
    *,
    api_version: str = API_VERSION,
    transport: Transport | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> Outcome:
    """One generateContent call. Paid. Never raises for an HTTP or network failure."""
    req = _request(_url(model, ":generateContent", api_version), api_key, body)
    send = transport or _urlopen
    start = clock()
    try:
        status, raw = send(req, timeout)
    except TRANSPORT_ERRORS as exc:
        return Outcome(
            TRANSPORT,
            None,
            latency=clock() - start,
            error=scrub(f"{type(exc).__name__}: {exc}", api_key)[:300],
        )
    return outcome_of(status, raw, clock() - start, api_key)


def get_model(
    model: str,
    api_key: str,
    timeout: float = 60,
    *,
    api_version: str = API_VERSION,
    transport: Transport | None = None,
) -> tuple[int | None, Any]:
    """models.get. Free. Returns (HTTP status or None, decoded body or a scrubbed error)."""
    req = _request(_url(model, "", api_version), api_key, None)
    try:
        status, raw = (transport or _urlopen)(req, timeout)
    except TRANSPORT_ERRORS as exc:
        return None, scrub(f"{type(exc).__name__}: {exc}", api_key)[:300]
    return status, _decode(raw, api_key)


def count_tokens(
    request: dict[str, Any],
    model: str,
    api_key: str,
    timeout: float = 60,
    *,
    api_version: str = API_VERSION,
    transport: Transport | None = None,
) -> tuple[int | None, Any]:
    """countTokens for one full request. Free. Returns (status, decoded body or error)."""
    wrapped = {"generateContentRequest": {"model": f"models/{model}", **request}}
    req = _request(_url(model, ":countTokens", api_version), api_key, wrapped)
    try:
        status, raw = (transport or _urlopen)(req, timeout)
    except TRANSPORT_ERRORS as exc:
        return None, scrub(f"{type(exc).__name__}: {exc}", api_key)[:300]
    return status, _decode(raw, api_key)
