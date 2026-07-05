"""Shared LLM transport for Ollama and the Anthropic Messages API.

A single ``complete(prompt, cfg)`` entry point used by every LLM-backed feature
(suggestions, triage/reranking). Retries transient failures with backoff and
raises a typed :class:`LLMError` on failure — never silently returns empty.
"""

from __future__ import annotations

import json
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
    """
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
