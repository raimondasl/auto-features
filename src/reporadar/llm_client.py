"""Shared LLM transport for Ollama, the Anthropic Messages API, OpenAI and Azure OpenAI.

A single ``complete(prompt, cfg)`` entry point used by every LLM-backed feature
(suggestions, triage/reranking). Retries transient failures with backoff and
raises a typed :class:`LLMError` on failure — never silently returns empty.
"""

from __future__ import annotations

import json
import math
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NoReturn
from urllib.parse import urlparse

from reporadar.credentials import resolve_api_key

# One message per provider, naming every place a key can come from. The question a user
# actually has at this point is "I set it -- why can you not see it", and an error that
# names only one of the three sources answers a different question.
_NO_KEY = {
    "openai": (
        "No OpenAI API key. Set suggestions.openai_api_key in .reporadar.yml (it may be "
        "${OPENAI_API_KEY}), export OPENAI_API_KEY, or run `rr auth` to store one."
    ),
    "claude": (
        "No Claude API key. Set suggestions.claude_api_key in .reporadar.yml (it may be "
        "${ANTHROPIC_API_KEY}), export ANTHROPIC_API_KEY, or run `rr auth --provider "
        "claude` to store one."
    ),
}


class LLMError(Exception):
    """Raised when an LLM call fails (misconfiguration or exhausted retries)."""


class LLMUnavailable(LLMError):
    """A failure every further call would repeat -- no credential, a refused token, a missing
    deployment. Stages that call once per paper stop on this instead of paying for the same
    refusal fifty times, and the reason reaches the user as the stage's warning."""


class LLMRateLimited(LLMError):
    """The provider's rate limit or quota refused this call, after any wait it asked for."""


class RateLimitBreaker:
    """Turns a run of rate-limited papers into :class:`LLMUnavailable`.

    One 429 is a paper worth skipping. Several in a row is a quota, and trying each remaining
    paper spends its retries and waits on an answer already known -- up to minutes per paper with
    nothing reported, in a tool call a client cancels after 180 s of silence.
    """

    LIMIT = 3

    def __init__(self) -> None:
        self.consecutive = 0

    def record(self, exc: Exception | None) -> None:
        """Note one paper's outcome: None for success. Re-raises what should end the stage."""
        if isinstance(exc, LLMUnavailable):
            raise exc
        if not isinstance(exc, LLMRateLimited):
            self.consecutive = 0
            return
        self.consecutive += 1
        if self.consecutive >= self.LIMIT:
            raise LLMUnavailable(
                f"stopped after {self.consecutive} papers in a row were rate-limited: {exc}"
            ) from exc


def _call_ollama(prompt: str, model: str, url: str, timeout: int) -> str:
    """Call the Ollama /api/generate endpoint."""
    payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode("utf-8")
    req = urllib.request.Request(
        f"{url.rstrip('/')}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return str(data.get("response", ""))


# Models that answered 400 "temperature is deprecated for this model", learned at runtime.
#
# The alternative designs are both worse. A hardcoded list of Claude 5 model ids goes stale the
# moment a model ships and encodes a guess about models nobody here has called. Retrying on
# every call pays a wasted request forever -- and 400s still consume request-rate budget, which
# this project has repeatedly hit. Discovering it from the API and remembering costs exactly
# one extra request per model per process.
#
# Process-local on purpose: it is a cache of an API fact, not configuration, so it must not
# outlive a process that might be talking to a different endpoint or a changed API.
_REJECTS_TEMPERATURE: set[str] = set()
_TOKEN_CAP: dict[str, str] = {}
"""Which output-cap parameter a model wants, learned from its own 400 and remembered.

OpenAI renamed `max_tokens` to `max_completion_tokens` for its reasoning models and
refuses the old name rather than accepting both. Learned rather than switched on a
model-name prefix, because that hard-codes a naming convention this API has already
changed once and would need editing for every model released after this line."""


MIN_CACHEABLE_CHARS = 4000
"""Below this the split is not worth making. Anthropic's minimum cacheable prefix is
model-dependent (512-4096 tokens) and a prefix under it is silently NOT cached -- no error,
no `cache_creation_input_tokens`, just a request carrying two blocks instead of one for no
reason. 4000 characters is comfortably above the ceiling of that range at this codebase's
measured density (~1.4 chars/token on repository context)."""


def _cache_split(prompt: str, marker: str | None) -> Any:
    """The message content: one string by default, two blocks when *marker* earns a breakpoint.

    Split on a MARKER rather than a character offset because `complete` redacts the prompt
    before dispatch, and redaction changes its length -- an offset computed by the caller
    would then land mid-token and cache the wrong prefix, silently, with the paper's text
    inside the cached half. Searching for the boundary is immune to that.

    Returns the plain string unchanged whenever the split cannot be made, so a request that
    would not benefit is byte-identical to one built before this existed.
    """
    if not marker:
        return prompt
    head, sep, tail = prompt.partition(marker)
    if not sep or len(head) < MIN_CACHEABLE_CHARS:
        return prompt
    return [
        {"type": "text", "text": head, "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": sep + tail},
    ]


def _call_claude(
    prompt: str,
    api_key: str,
    model: str,
    timeout: int,
    max_tokens: int,
    cache_split_on: str | None = None,
) -> str:
    """Call the Anthropic Messages API and return the concatenated text blocks.

    **`temperature=0`, and it was missing until 2026-09-01.** Without it the Anthropic default
    of 1.0 applied, so every call on this path was a *sample*: the actionability gate, HyDE's
    hypotheses, the repo summary, typed anchors, and the second judge. Two consequences were
    measured before the fix:

    * **NR-53** — the second judge disagreed with *itself* on 8.4% of label decisions across a
      redraw of 200 papers (score-level agreement 0.806).
    * **NR-54** — re-running the shipped config against a byte-identical frozen pool moved
      net@2 by **sd 1.44 per case**, with only 10 of 37 cases reproducing exactly. That is 35%
      of the paired variance in a frozen-pool arm, so removing it tightens the benchmark's
      resolution from ±0.78 to about ±0.63.

    Every caller here wants one determinate answer about a fixed input, so none of them loses
    anything. HyDE is the only arguable case and it is not one: the four hypotheses are made
    diverse by the *prompt* asking for four different abstracts in a single response, not by
    sampling across calls.

    **`_call_ollama` is deliberately left alone.** It exposes temperature through a different
    field (`options`), no measured arm has ever used it, and widening this change to a path
    nothing measures would be scope the evidence does not cover. Noted here rather than left
    for someone to discover as an inconsistency.

    Runs before and after this differ by construction. Frozen pools and cached judge verdicts
    are unaffected — the pool fingerprint does not cover temperature, and the judge cache is
    keyed by prompt and model — but a *gate* comparison spanning the change is confounded.

    **It does not cover every model, and the first version of this claimed it did.** The
    **Claude 5 family rejects the parameter outright** — `claude-sonnet-5` and `claude-opus-5`
    answer `400 "temperature is deprecated for this model"` — while Claude 4.x accepts it. The
    shipped gate runs `claude-haiku-4-5` and was fine, which is exactly why the change looked
    verified: NR-55's two runs exercised the gate and never touched the Sonnet judge, and the
    judge path broke silently until NR-56 tried to use it and got 155 straight 400s.

    So a rejected `temperature` is retried without it — narrowly, only on a 400 whose body names
    `temperature`, because a blanket retry would swallow rate limits and quota errors as if they
    were parameter problems — and **the rejection is remembered in `_REJECTS_TEMPERATURE`**, so
    a model pays that extra request once per process rather than on every call.

    **A consequence worth stating: the judge cannot be made deterministic this way.** NR-53
    measured `claude-sonnet-5` disagreeing with itself on 8.4% of label decisions, and since
    that model refuses the parameter, that figure is a standing property of the instrument
    rather than something a setting can remove.
    """
    body: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": _cache_split(prompt, cache_split_on)}],
    }
    if model not in _REJECTS_TEMPERATURE:
        body["temperature"] = 0
    try:
        return _post_claude(body, api_key, timeout)
    except urllib.error.HTTPError as exc:
        if exc.code != 400 or "temperature" not in body:
            raise
        detail = ""
        try:
            detail = exc.read().decode("utf-8", "replace")
        except Exception:  # noqa: BLE001 -- an unreadable body is just an unknown 400
            raise exc from None
        if "temperature" not in detail:
            raise LLMError(f"LLM HTTP 400: {detail[:200]}") from exc
        _REJECTS_TEMPERATURE.add(model)
        body.pop("temperature")
        return _post_claude(body, api_key, timeout)


def _post_claude(body: dict[str, Any], api_key: str, timeout: int) -> str:
    payload = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    content = data.get("content", [])
    parts = [block.get("text", "") for block in content if block.get("type") == "text"]
    return "\n".join(parts)


_OPENAI_URL = "https://api.openai.com/v1/chat/completions"

# More request-shape facts learned from a model's own 400, like `_REJECTS_TEMPERATURE` and
# `_TOKEN_CAP` above. Measured 2026-09-13 on Azure (PLANS item 17): gpt-5.6-luna returns logprobs
# at `reasoning_effort: none` -- despite Azure's docs -- but refuses `top_logprobs` above 5 with
# "must be less than or equal to 5", and a non-reasoning model can refuse `reasoning_effort`.
#
# Keyed by `_Target.key(model)`: the bare model name for OpenAI, exactly as before, and
# resource-qualified for Azure. The same name behaves differently on the two -- gpt-5.6-luna
# refuses `temperature: 0` on OpenAI and accepted it on Azure -- so one must not teach the other.
_REJECTS_EFFORT: set[str] = set()
_TOP_LOGPROBS_CAP: dict[str, int] = {}
_TOP_LOGPROBS_LIMIT = re.compile(r"top_logprobs.*?less than or equal to (\d+)", re.I | re.S)

# A 429 that names its own wait is obeyed, up to this. Azure enforces requests-per-minute over
# 1-10 s windows and says how long in `retry-after-ms`; the fixed half-second backoff alone spent
# the gate's retries inside the same window and dropped the paper. A longer wait is not slept
# through: it is a quota rather than a window, the paper fails at once, and RateLimitBreaker stops
# the stage when that keeps happening. Sleeping a capped minute per retry per paper had turned a
# 50-paper gate against an exhausted quota into 100 silent minutes.
RETRY_AFTER_CAP_SECONDS = 20.0


@dataclass(frozen=True)
class _Target:
    """Where an OpenAI-shaped Chat Completions request goes, and how it authenticates.

    OpenAI and Azure OpenAI take the same body and the same `Authorization: Bearer` header; they
    differ only in the URL and in what the bearer is -- a stored key, or an Entra token from
    `az login`. So the gate and the fine-scale rescore share one transport, not two copies.
    """

    url: str
    label: str
    bearer: Callable[[], str]
    azure_host: str = ""
    forget_token: Callable[[], None] | None = None

    def key(self, model: str) -> str:
        return f"azure:{self.azure_host}:{model}" if self.azure_host else model


def _openai_target(cfg: Any, provider: str) -> _Target:
    if provider != "azure_openai":
        api_key = resolve_api_key("openai", cfg)
        if not api_key:
            raise LLMUnavailable(_NO_KEY["openai"])
        return _Target(_OPENAI_URL, "OpenAI", lambda: api_key)

    from reporadar import azure_auth

    try:
        # Validated before anything else happens, so a refused endpoint never costs a token fetch
        # and never becomes a request.
        url = azure_auth.chat_completions_url(getattr(cfg, "azure_endpoint", ""))
        tenant = azure_auth.validate_tenant(getattr(cfg, "azure_tenant", ""))
    except azure_auth.AzureAuthError as exc:
        raise LLMUnavailable(str(exc)) from None
    host = urlparse(url).hostname or ""

    def bearer() -> str:
        try:
            return azure_auth.get_token(tenant)
        except azure_auth.AzureAuthError as exc:
            raise LLMUnavailable(str(exc)) from None

    return _Target(
        url,
        f"Azure OpenAI ({host})",
        bearer,
        azure_host=host,
        forget_token=lambda: azure_auth.forget(tenant),
    )


def _model_for(cfg: Any, target: _Target, section: str) -> str:
    """The `model` field: a deployment name on Azure, a model name on OpenAI."""
    if not target.azure_host:
        return str(getattr(cfg, "openai_model", "gpt-4o-mini"))
    deployment = str(getattr(cfg, "azure_deployment", "") or "").strip()
    if not deployment:
        raise LLMUnavailable(
            f"No Azure OpenAI deployment. Set {section}.azure_deployment to the deployment name "
            f"you chose in Azure — not the model name; Azure sends it as `model`."
        )
    return deployment


def _call_openai(
    prompt: str,
    target: _Target,
    model: str,
    timeout: int,
    max_tokens: int,
    cache_split_on: str | None = None,
    effort: str = "",
) -> str:
    """One OpenAI-shaped chat completion, returning the message text.

    The gate and the triage stage needed a Claude key until this existed, while the fine-scale
    rescore needed an OpenAI one — so a working installation required two accounts from two
    vendors, for two calls that ask nearly the same question. That is a real barrier to anyone
    installing this, and nothing measured depends on which vendor answers the gate.

    *cache_split_on* is accepted and ignored. OpenAI matches the prefix server-side, so there
    is no annotation to add: the caller passes a marker to `complete` and the discount applies
    on this path without the request differing at all. Rejecting the argument here instead
    would make one provider's caching a caller's problem.

    `temperature=0` is sent, and dropped on the retry if the model rejects it, exactly as the
    Claude path does — the reasoning models refuse the parameter and the rejection is
    remembered per process rather than paid for on every call.
    """
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    if effort:
        body["reasoning_effort"] = effort
    return _message_text(_post_adaptive(target, body, timeout), target)


def _post_adaptive(target: _Target, body: dict[str, Any], timeout: int) -> dict[str, Any]:
    """POST *body*, adapting it to whatever this model refuses; return the parsed response.

    Each adaptation is learned from the message the API actually returned and remembered for the
    process. Measured 2026-09-07 against gpt-5.6-luna: `max_tokens` is refused outright ("use
    'max_completion_tokens' instead") and `temperature: 0` is refused separately ("only the
    default (1) value is supported"), so a single retry that pops one of them still fails on the
    other. Guessing the modern shape by model-name prefix would be worse: it hard-codes a naming
    convention this API has already changed once.

    Shared by the gate and the fine-scale rescore. The rescore used to send `max_tokens`,
    `temperature: 0` and `top_logprobs: 20` with no retry at all, so it failed on any reasoning
    model -- on OpenAI's own API as much as on Azure.
    """
    key = target.key(str(body["model"]))
    _apply_learned(body, key)
    for _ in range(4):  # at most one adaptation of each kind
        try:
            return _post_json(target, body, timeout)
        except urllib.error.HTTPError as exc:
            if exc.code != 400:
                raise
            try:
                detail = exc.read().decode("utf-8", "replace")
            except Exception:  # noqa: BLE001 -- an unreadable body is just an unknown 400
                raise exc from None
            if _is_content_filter(detail):
                # Not a parameter problem and not transient: the same prompt is blocked again.
                # Said as what it is, so a thin digest can be traced to a filter rather than read
                # as the model's judgement.
                raise LLMError(
                    f"{target.label}'s content filter blocked this prompt, so the paper was not "
                    f"scored: {detail[:200]}"
                ) from exc
            limit = _TOP_LOGPROBS_LIMIT.search(detail)
            if "max_tokens" in detail and "max_tokens" in body:
                _TOKEN_CAP[key] = "max_completion_tokens"
            elif "temperature" in detail and "temperature" in body:
                _REJECTS_TEMPERATURE.add(key)
            elif "reasoning_effort" in detail and "reasoning_effort" in body:
                if _refuses_value(detail):
                    # The parameter is supported and this VALUE is not ("does not support 'none'
                    # with this model"). Dropping it would silently run the model at its default
                    # effort -- medium on gpt-5-mini, which spends the gate's small token cap on
                    # reasoning and fails every paper with an empty answer far from the cause.
                    raise LLMUnavailable(
                        f"{target.label} refused reasoning_effort {body['reasoning_effort']!r} "
                        f"for {body['model']!r}. Set suggestions.openai_reasoning_effort (the "
                        f"gate) or triage.finescale.reasoning_effort to a value it lists: "
                        f"{detail[:300]}"
                    ) from exc
                _REJECTS_EFFORT.add(key)
            elif limit and int(body.get("top_logprobs", 0)) > int(limit.group(1)):
                _TOP_LOGPROBS_CAP[key] = int(limit.group(1))
            else:
                raise LLMError(f"LLM HTTP 400: {detail[:200]}") from exc
            _apply_learned(body, key)
    return _post_json(target, body, timeout)


def _refuses_value(detail: str) -> bool:
    """Whether a 400 refuses a parameter's value rather than the parameter itself."""
    try:
        error = json.loads(detail).get("error")
    except (ValueError, AttributeError):
        error = None
    if isinstance(error, dict) and error.get("code") in ("unsupported_value", "invalid_value"):
        return True
    return "Unsupported value" in detail or "does not support" in detail


def _apply_learned(body: dict[str, Any], key: str) -> None:
    if _TOKEN_CAP.get(key) == "max_completion_tokens" and "max_tokens" in body:
        body["max_completion_tokens"] = body.pop("max_tokens")
    if key in _REJECTS_TEMPERATURE:
        body.pop("temperature", None)
    if key in _REJECTS_EFFORT:
        body.pop("reasoning_effort", None)
    cap = _TOP_LOGPROBS_CAP.get(key)
    if cap is not None and int(body.get("top_logprobs", 0)) > cap:
        body["top_logprobs"] = cap


def _post_json(target: _Target, body: dict[str, Any], timeout: int) -> dict[str, Any]:
    req = urllib.request.Request(
        target.url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    # Unredirected: urllib copies ordinary headers onto a redirect, to whatever host and scheme
    # the Location names. The endpoint check guarantees only the first hop is Azure.
    req.add_unredirected_header("Authorization", f"Bearer {target.bearer()}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if target.azure_host and exc.code in (401, 403, 404):
            if exc.code in (401, 403) and target.forget_token is not None:
                target.forget_token()  # so signing in again takes effect without a restart
            raise LLMUnavailable(_azure_refusal(exc, target, str(body.get("model")))) from exc
        raise
    return data if isinstance(data, dict) else {}


def _azure_refusal(exc: urllib.error.HTTPError, target: _Target, deployment: str) -> str:
    """The three Azure refusals a keyless setup actually hits, each in terms of its fix."""
    try:
        detail = exc.read().decode("utf-8", "replace")[:200]
    except Exception:  # noqa: BLE001
        detail = ""
    if exc.code == 403:
        why = (
            "your account has no data-plane role on this resource. Grant it 'Cognitive Services "
            "OpenAI User' — Owner and Contributor are not enough, they carry no data actions. A "
            "new assignment can take about 5 minutes to apply."
        )
    elif exc.code == 401:
        why = (
            "the Entra token was refused. If the resource is in another tenant, set "
            "azure_openai.tenant and run `az login --tenant <tenant>`."
        )
    else:
        why = (
            f"there is no deployment named {deployment!r} on this resource. azure_deployment is "
            f"the name you gave the deployment in Azure, not the model's name."
        )
    return f"{target.label} HTTP {exc.code}: {why} ({detail})"


def _is_content_filter(detail: str) -> bool:
    try:
        error = json.loads(detail).get("error") or {}
    except (ValueError, AttributeError):
        return '"content_filter"' in detail
    if not isinstance(error, dict):
        return '"content_filter"' in detail
    inner = error.get("innererror") or error.get("inner_error") or {}
    return error.get("code") == "content_filter" or (
        isinstance(inner, dict) and inner.get("code") == "ResponsibleAIPolicyViolation"
    )


def _message_text(data: dict[str, Any], target: _Target) -> str:
    choices = data.get("choices") or []
    if not choices:
        raise LLMError(f"{target.label} returned no choices")
    finish = choices[0].get("finish_reason")
    if finish == "content_filter":
        raise LLMError(
            f"{target.label}'s content filter withheld the answer, so the paper was not scored."
        )
    # An empty string is a real answer to nothing, and the callers parse JSON out of this.
    # Returning it would surface as a parse error somewhere far from the cause.
    text = (choices[0].get("message") or {}).get("content") or ""
    if not text.strip():
        raise LLMError(f"{target.label} returned an empty message (finish_reason={finish or '?'})")
    return str(text)


def _call_openai_top_logprobs(
    prompt: str, target: _Target, model: str, timeout: int, top_k: int, effort: str = ""
) -> list[tuple[str, float]]:
    """Return ``[(token, probability)]`` alternatives at the answer's FIRST token.

    Used by the fine-scale actionability rescore, which needs the score
    *distribution* rather than the sampled score — reading the expectation over the
    digit tokens is what turns a near-binary gate into a continuous one
    (see reporadar/finescale.py and evals/RESULTS.md). Anthropic's API exposes no
    logprobs, so this path is OpenAI-shaped only: OpenAI, or Azure OpenAI.

    Deliberately urllib rather than the ``openai`` SDK: nothing else in the shipped
    package needs that dependency, and the request is one POST.
    """
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 4,
        "temperature": 0,
        "logprobs": True,
        "top_logprobs": top_k,
    }
    if effort:
        body["reasoning_effort"] = effort
    data = _post_adaptive(target, body, timeout)
    choices = data.get("choices") or []
    if not choices:
        raise LLMError(f"{target.label} returned no choices")
    if choices[0].get("finish_reason") == "content_filter":
        raise LLMError(
            f"{target.label}'s content filter withheld the answer, so the paper was not scored."
        )
    content = (choices[0].get("logprobs") or {}).get("content") or []
    if not content:
        raise LLMError(
            f"{target.label} returned no logprobs (model or account may not support them)"
        )
    return [
        (alt.get("token", ""), math.exp(alt["logprob"]))
        for alt in content[0].get("top_logprobs", [])
        if "logprob" in alt
    ]


def top_logprobs(
    prompt: str, cfg: Any, *, top_k: int = 20, max_retries: int = 2
) -> list[tuple[str, float]]:
    """First-token ``[(token, probability)]`` alternatives from an OpenAI-shaped model.

    *cfg* needs ``openai_api_key`` (or ``OPENAI_API_KEY`` in the environment),
    ``openai_model`` and ``timeout`` -- or, with ``provider: azure_openai``, the mirrored
    ``azure_endpoint``/``azure_tenant`` and an ``azure_deployment``. Raises :class:`LLMError` on
    failure — never returns an empty list to mean "no signal", because a caller that cannot tell
    those apart would score a failed call as a confident zero.
    """
    patterns = getattr(cfg, "redact", None)
    if patterns:
        from reporadar.privacy import compile_patterns, redact

        prompt = redact(prompt, compile_patterns(list(patterns)))

    provider = str(getattr(cfg, "provider", "") or "openai")
    if provider not in ("openai", "azure_openai"):
        # Not quietly OpenAI: a key would go to a vendor the config never named.
        raise LLMUnavailable(
            f"triage.finescale.provider {provider!r} cannot return logprobs; use openai or "
            f"azure_openai."
        )
    target = _openai_target(cfg, provider)
    model = _model_for(cfg, target, "triage.finescale")
    timeout = getattr(cfg, "timeout", 30)
    effort = str(getattr(cfg, "reasoning_effort", "") or "")

    last: Exception | None = None
    for attempt in range(max_retries + 1):
        delay = 0.0
        try:
            return _call_openai_top_logprobs(prompt, target, model, timeout, top_k, effort)
        except LLMError:
            raise
        except urllib.error.HTTPError as exc:
            if exc.code != 429 and exc.code < 500:
                raise LLMError(f"{target.label} HTTP {exc.code}: {exc}") from exc
            last = exc
            delay = _retry_after(exc, target.label)
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            last = exc
        if attempt < max_retries:
            time.sleep(max(0.5 * (2**attempt), delay))
    _raise_exhausted(f"{target.label} logprob call", max_retries, last)


def _retry_after(exc: urllib.error.HTTPError, label: str = "LLM") -> float:
    """Seconds a 429/5xx asks us to wait, from `retry-after-ms` or `retry-after`; 0 if unsaid.

    Raises :class:`LLMRateLimited` for a wait longer than RETRY_AFTER_CAP_SECONDS rather than
    sleeping any of it.
    """
    headers = exc.headers
    if headers is None:
        return 0.0
    for name, scale in (("retry-after-ms", 0.001), ("retry-after", 1.0)):
        raw = headers.get(name)
        if raw is None:
            continue
        try:
            seconds = float(raw) * scale
        except (TypeError, ValueError):
            continue
        if seconds > RETRY_AFTER_CAP_SECONDS:
            raise LLMRateLimited(
                f"{label} is rate-limited or out of quota: HTTP {exc.code}, asked to wait "
                f"{seconds:.0f} s"
            ) from exc
        return max(seconds, 0.0)
    return 0.0


def _raise_exhausted(what: str, max_retries: int, last: Exception | None) -> NoReturn:
    message = f"{what} failed after {max_retries + 1} attempts: {last}"
    if isinstance(last, urllib.error.HTTPError) and last.code == 429:
        raise LLMRateLimited(message) from last
    raise LLMError(message) from last


def _dispatch(prompt: str, cfg: Any, max_tokens: int, cache_split_on: str | None = None) -> str:
    provider = getattr(cfg, "provider", "ollama")
    timeout = getattr(cfg, "timeout", 30)
    if provider == "claude":
        api_key = resolve_api_key("claude", cfg)
        if not api_key:
            raise LLMUnavailable(_NO_KEY["claude"])
        model = getattr(cfg, "claude_model", "claude-haiku-4-5")
        return _call_claude(prompt, api_key, model, timeout, max_tokens, cache_split_on)
    if provider in ("openai", "azure_openai"):
        target = _openai_target(cfg, provider)
        model = _model_for(cfg, target, "suggestions")
        effort = str(getattr(cfg, "openai_reasoning_effort", "") or "")
        return _call_openai(prompt, target, model, timeout, max_tokens, cache_split_on, effort)
    if provider == "ollama":
        url = getattr(cfg, "ollama_url", "http://localhost:11434")
        model = getattr(cfg, "ollama_model", "llama3.2")
        return _call_ollama(prompt, model, url, timeout)
    raise LLMError(f"Unknown LLM provider: {provider!r}")


def complete(
    prompt: str,
    cfg: Any,
    *,
    max_tokens: int = 300,
    max_retries: int = 2,
    base_delay: float = 0.5,
    cache_split_on: str | None = None,
) -> str:
    """Run one completion. Retries transient failures; raises LLMError on failure.

    *cfg* is any object exposing ``provider`` plus the provider's fields
    (``claude_api_key``/``claude_model`` or ``ollama_url``/``ollama_model``,
    and ``timeout``) — e.g. a SuggestionsConfig or TriageConfig.

    If *cfg* carries a non-empty ``redact`` list (config mirrors ``privacy.redact``
    onto it at load time), those terms are stripped from the prompt here — at the
    last point before it leaves the process, so no call site can route around it.

    *cache_split_on* is a marker string; when the prompt contains it and the part before it
    is long enough to be worth caching, the Claude request sends two content blocks with a
    cache breakpoint between them instead of one string. The rendered token sequence is
    unchanged — measured with `count_tokens`, 4112 either way — so this changes what the call
    is billed, never what the model reads. Omitted, the request body is byte-identical to one
    built before this parameter existed. Ignored by non-Claude providers.
    """
    patterns = getattr(cfg, "redact", None)
    if patterns:
        from reporadar.privacy import compile_patterns, redact

        prompt = redact(prompt, compile_patterns(list(patterns)))

    last: Exception | None = None
    for attempt in range(max_retries + 1):
        delay = 0.0
        try:
            return _dispatch(prompt, cfg, max_tokens, cache_split_on)
        except LLMError:
            raise  # config errors are not transient — don't retry or wrap
        except urllib.error.HTTPError as exc:
            if exc.code != 429 and exc.code < 500:
                raise LLMError(f"LLM HTTP {exc.code}: {exc}") from exc
            last = exc
            delay = _retry_after(exc)
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            last = exc
        if attempt < max_retries:
            time.sleep(max(base_delay * (2**attempt), delay))
    _raise_exhausted("LLM call", max_retries, last)
