"""LLM-powered suggestions via Ollama or Claude APIs."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from reporadar.profiler import RepoProfile


def _build_prompt(paper: dict[str, Any], profile: RepoProfile) -> str:
    """Build a prompt for generating suggestions."""
    keywords = ", ".join(term for term, _ in profile.keywords[:10])
    domains = ", ".join(profile.domains[:5]) if profile.domains else "general"
    anchors = ", ".join(profile.anchors[:10]) if profile.anchors else "none"

    return (
        f"You are a research assistant helping a developer whose project uses: {anchors}.\n"
        f"Their project domains: {domains}.\n"
        f"Key topics: {keywords}.\n\n"
        f"Paper title: {paper.get('title', 'Unknown')}\n"
        f"Abstract: {paper.get('abstract', '')[:500]}\n\n"
        f"Give exactly 3 concise, actionable suggestions for how this paper could be "
        f"applied to the developer's project. Each suggestion should be 1-2 sentences. "
        f"Format as a numbered list (1. 2. 3.)."
    )


def _parse_suggestions(text: str, max_suggestions: int = 3) -> list[str]:
    """Parse numbered suggestions from LLM output."""
    suggestions: list[str] = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        # Match numbered items: "1.", "1)", "- ", "* "
        for prefix_len in range(1, 4):
            prefix = line[:prefix_len]
            if prefix.rstrip(".):") != prefix and prefix[0].isdigit():
                suggestion = line[prefix_len:].strip().lstrip(".):").strip()
                if suggestion:
                    suggestions.append(suggestion)
                break
        else:
            if line.startswith(("- ", "* ")):
                suggestion = line[2:].strip()
                if suggestion:
                    suggestions.append(suggestion)

    return suggestions[:max_suggestions]


def _call_ollama(
    prompt: str,
    model: str,
    url: str,
    timeout: int,
) -> str:
    """Call the Ollama /api/generate endpoint."""
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{url.rstrip('/')}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data.get("response", "")


def _call_claude(
    prompt: str,
    api_key: str,
    model: str,
    timeout: int,
) -> str:
    """Call the Anthropic Messages API."""
    payload = json.dumps({
        "model": model,
        "max_tokens": 300,
        "messages": [{"role": "user", "content": prompt}],
    }).encode("utf-8")

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

    # Extract text from content blocks
    content = data.get("content", [])
    parts = [block.get("text", "") for block in content if block.get("type") == "text"]
    return "\n".join(parts)


def generate_llm_suggestions(
    paper: dict[str, Any],
    profile: RepoProfile,
    config: Any,
) -> list[str]:
    """Generate suggestions using an LLM (Ollama or Claude).

    *config* should be a SuggestionsConfig instance.
    Raises on any error (caller should handle fallback).
    """
    prompt = _build_prompt(paper, profile)
    max_suggestions = getattr(config, "max_suggestions", 3)
    timeout = getattr(config, "timeout", 30)
    provider = getattr(config, "provider", "ollama")

    if provider == "claude":
        api_key = getattr(config, "claude_api_key", "")
        if not api_key:
            import os
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise ValueError("No Claude API key configured")
        model = getattr(config, "claude_model", "claude-sonnet-4-20250514")
        raw = _call_claude(prompt, api_key, model, timeout)
    else:
        # Default: Ollama
        ollama_url = getattr(config, "ollama_url", "http://localhost:11434")
        ollama_model = getattr(config, "ollama_model", "llama3.2")
        raw = _call_ollama(prompt, ollama_model, ollama_url, timeout)

    return _parse_suggestions(raw, max_suggestions)
