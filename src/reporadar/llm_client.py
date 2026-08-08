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


def _call_claude(prompt: str, api_key: str, model: str, timeout: int, max_tokens: int) -> str:
    """Call the Anthropic Messages API and return the concatenated text blocks."""
    payload = json.dumps(
        {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
    ).encode("utf-8")
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
