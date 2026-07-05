"""LLM-powered suggestions via Ollama or Claude APIs."""

from __future__ import annotations

from typing import Any

from reporadar.llm_client import complete
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


def generate_llm_suggestions(
    paper: dict[str, Any],
    profile: RepoProfile,
    config: Any,
) -> list[str]:
    """Generate suggestions using an LLM (Ollama or Claude).

    *config* should be a SuggestionsConfig instance. Raises ``LLMError`` on any
    failure (caller should handle fallback).
    """
    prompt = _build_prompt(paper, profile)
    max_suggestions = getattr(config, "max_suggestions", 3)
    raw = complete(prompt, config, max_tokens=300)
    return _parse_suggestions(raw, max_suggestions)
