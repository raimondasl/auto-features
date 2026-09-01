"""Shared LLM transport for Ollama and the Anthropic Messages API.

A single ``complete(prompt, cfg)`` entry point used by every LLM-backed feature
(suggestions, triage/reranking). Retries transient failures with backoff and
raises a typed :class:`LLMError` on failure — never silently returns empty.
"""

from __future__ import annotations

import json
import math
import os
import time
import urllib.error
import urllib.request
from typing import Any


class LLMError(Exception):
    """Raised when an LLM call fails (misconfiguration or exhausted retries)."""


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


def _call_claude(prompt: str, api_key: str, model: str, timeout: int, max_tokens: int) -> str:
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
        "messages": [{"role": "user", "content": prompt}],
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


def _call_openai_top_logprobs(
    prompt: str, api_key: str, model: str, timeout: int, top_k: int
) -> list[tuple[str, float]]:
    """Return ``[(token, probability)]`` alternatives at the answer's FIRST token.

    Used by the fine-scale actionability rescore, which needs the score
    *distribution* rather than the sampled score — reading the expectation over the
    digit tokens is what turns a near-binary gate into a continuous one
    (see reporadar/finescale.py and evals/RESULTS.md). Anthropic's API exposes no
    logprobs, so this path is OpenAI-only.

    Deliberately urllib rather than the ``openai`` SDK: nothing else in the shipped
    package needs that dependency, and the request is one POST.
    """
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 4,
            "temperature": 0,
            "logprobs": True,
            "top_logprobs": top_k,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    choices = data.get("choices") or []
    if not choices:
        raise LLMError("OpenAI returned no choices")
    content = (choices[0].get("logprobs") or {}).get("content") or []
    if not content:
        raise LLMError("OpenAI returned no logprobs (model or account may not support them)")
    return [
        (alt.get("token", ""), math.exp(alt["logprob"]))
        for alt in content[0].get("top_logprobs", [])
        if "logprob" in alt
    ]


def top_logprobs(
    prompt: str, cfg: Any, *, top_k: int = 20, max_retries: int = 2
) -> list[tuple[str, float]]:
    """First-token ``[(token, probability)]`` alternatives from an OpenAI model.

    *cfg* needs ``openai_api_key`` (or ``OPENAI_API_KEY`` in the environment),
    ``openai_model`` and ``timeout``. Raises :class:`LLMError` on failure — never
    returns an empty list to mean "no signal", because a caller that cannot tell
    those apart would score a failed call as a confident zero.
    """
    patterns = getattr(cfg, "redact", None)
    if patterns:
        from reporadar.privacy import compile_patterns, redact

        prompt = redact(prompt, compile_patterns(list(patterns)))

    api_key = getattr(cfg, "openai_api_key", "") or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise LLMError("No OpenAI API key configured (set openai_api_key or OPENAI_API_KEY)")
    model = getattr(cfg, "openai_model", "gpt-4o-mini")
    timeout = getattr(cfg, "timeout", 30)

    last: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return _call_openai_top_logprobs(prompt, api_key, model, timeout, top_k)
        except LLMError:
            raise
        except urllib.error.HTTPError as exc:
            if exc.code != 429 and exc.code < 500:
                raise LLMError(f"OpenAI HTTP {exc.code}: {exc}") from exc
            last = exc
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            last = exc
        if attempt < max_retries:
            time.sleep(0.5 * (2**attempt))
    raise LLMError(f"OpenAI logprob call failed after {max_retries + 1} attempts: {last}")


def _dispatch(prompt: str, cfg: Any, max_tokens: int) -> str:
    provider = getattr(cfg, "provider", "ollama")
    timeout = getattr(cfg, "timeout", 30)
    if provider == "claude":
        api_key = getattr(cfg, "claude_api_key", "") or os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise LLMError("No Claude API key configured (set claude_api_key or ANTHROPIC_API_KEY)")
        model = getattr(cfg, "claude_model", "claude-haiku-4-5")
        return _call_claude(prompt, api_key, model, timeout, max_tokens)
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
) -> str:
    """Run one completion. Retries transient failures; raises LLMError on failure.

    *cfg* is any object exposing ``provider`` plus the provider's fields
    (``claude_api_key``/``claude_model`` or ``ollama_url``/``ollama_model``,
    and ``timeout``) — e.g. a SuggestionsConfig or TriageConfig.

    If *cfg* carries a non-empty ``redact`` list (config mirrors ``privacy.redact``
    onto it at load time), those terms are stripped from the prompt here — at the
    last point before it leaves the process, so no call site can route around it.
    """
    patterns = getattr(cfg, "redact", None)
    if patterns:
        from reporadar.privacy import compile_patterns, redact

        prompt = redact(prompt, compile_patterns(list(patterns)))

    last: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return _dispatch(prompt, cfg, max_tokens)
        except LLMError:
            raise  # config errors are not transient — don't retry or wrap
        except urllib.error.HTTPError as exc:
            if exc.code != 429 and exc.code < 500:
                raise LLMError(f"LLM HTTP {exc.code}: {exc}") from exc
            last = exc
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            last = exc
        if attempt < max_retries:
            time.sleep(base_delay * (2**attempt))
    raise LLMError(f"LLM call failed after {max_retries + 1} attempts: {last}")
