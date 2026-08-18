"""LLM-powered suggestions via Ollama or Claude APIs."""

from __future__ import annotations

from typing import Any

from reporadar.llm_client import complete
from reporadar.profiler import RepoProfile


def _build_prompt(paper: dict[str, Any], profile: RepoProfile) -> str:
    """Build a prompt for generating suggestions.

    The reader is the repository's **maintainer**, and saying so is the whole content of
    this prompt's first line. It used to open "a research assistant helping a developer
    whose project uses: {anchors}", which reads the profile as a dependency list and the
    reader as somebody building *on top of* it -- so on minimap2, whose only detected anchor
    is `cython`, the ideas were addressed to the owner of "your minimap2 Python wrapper",
    and on repositories whose own paper surfaced they proposed adopting the very tool the
    reader maintains. The framing is not cosmetic: the same mistake in the GATE's prompt is
    why minimap2's own accuracy paper was scored 0 with the reason "not novel methods that
    could be applied to this repository's Python wrapper/bindings".
    """
    keywords = ", ".join(term for term, _ in profile.keywords[:10])
    domains = ", ".join(profile.domains[:5]) if profile.domains else "general"
    anchors = ", ".join(profile.anchors[:10]) if profile.anchors else "none"
    prose = (getattr(profile, "prose", "") or "").strip()
    prose_line = f"What the project is, in its own words: {prose}\n" if prose else ""

    return (
        f"You are a research assistant advising the MAINTAINERS of a software repository "
        f"about a paper. They own and modify this codebase; they do not merely use it.\n"
        f"{prose_line}"
        f"Its dependencies/libraries: {anchors}.\n"
        f"Its domains: {domains}.\n"
        f"Key topics: {keywords}.\n\n"
        f"Paper title: {paper.get('title', 'Unknown')}\n"
        f"Abstract: {paper.get('abstract', '')[:500]}\n\n"
        f"Give exactly 3 concise, actionable suggestions for what the maintainers could "
        f"CHANGE IN THIS REPOSITORY in light of this paper -- a method to implement, a "
        f"component to add, a comparison to run. Do not suggest adopting the project "
        f"itself, and do not address the reader as a user of it. Each suggestion should "
        f"be 1-2 sentences. Format as a numbered list (1. 2. 3.)."
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
