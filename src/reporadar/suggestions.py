"""Template-based action suggestions derived from paper abstracts."""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Each pattern maps a regex (applied to the abstract) to a suggestion template.
# The template can use {match} to interpolate the captured text.
_BENCHMARK_RE = re.compile(
    r"\b(?:benchmark|evaluate[ds]? on|tested on)\s+([A-Z][\w\s\-]{2,30})",
    re.IGNORECASE,
)
_DATASET_RE = re.compile(
    r"\b(?:dataset|corpus)\s+(?:called|named)?\s*([A-Z][\w\-]{2,20})",
    re.IGNORECASE,
)
_OUTPERFORM_RE = re.compile(
    r"\boutperforms?\s+([A-Z][\w\s\-]{2,30})",
    re.IGNORECASE,
)
_PROPOSE_RE = re.compile(
    r"\b(?:we (?:propose|introduce|present))\s+(?:a |an )?"
    r"([a-zA-Z][\w\s\-]{3,40}?)(?:\s*[,.]|\s+that|\s+which|\s+for)",
    re.IGNORECASE,
)
_LOSS_RE = re.compile(
    r"\b(?:loss function|objective|optimizer|regulariz(?:er|ation))"
    r"\s+(?:called|named|based on)?\s*([A-Z][\w\-]{2,20})",
    re.IGNORECASE,
)
_SOTA_RE = re.compile(
    r"\b(?:state[- ]of[- ]the[- ]art|SOTA)"
    r"\s+(?:on|for|in)\s+([a-zA-Z][\w\s\-]{3,30})",
    re.IGNORECASE,
)
_OPENSOURCE_RE = re.compile(
    r"\b(?:open[- ]source[d]?|release[ds]?|available at)\s",
    re.IGNORECASE,
)
_MODULAR_RE = re.compile(
    r"\b(?:plug[- ]?in|drop[- ]?in|module|component)"
    r"\s+(?:that |which )?(?:can be|is )?(?:easily )?(?:added|integrated|applied)",
    re.IGNORECASE,
)

SUGGESTION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (_BENCHMARK_RE, "Add evaluation on {match} (mentioned as a benchmark in this paper)"),
    (_DATASET_RE, "Consider using the {match} dataset for evaluation"),
    (_OUTPERFORM_RE, "Compare your approach against {match} as a baseline"),
    (_PROPOSE_RE, "Explore integrating the proposed {match} into your pipeline"),
    (_LOSS_RE, "Try swapping your optimizer/loss for {match}"),
    (_SOTA_RE, "This paper claims SOTA on {match} — worth checking methodology"),
    (_OPENSOURCE_RE, "Code/data may be publicly available — check the paper for links"),
    (_MODULAR_RE, "This describes a modular component — consider adding it as a feature flag"),
]

# Maximum suggestions per paper
MAX_SUGGESTIONS = 3


def generate_suggestions(paper: dict[str, Any]) -> list[str]:
    """Generate templated action suggestions for a paper.

    Scans the abstract for known patterns and returns up to
    MAX_SUGGESTIONS ideas. Each suggestion is clearly labeled as an idea
    and grounded in the abstract text.
    """
    abstract = paper.get("abstract", "")
    title = paper.get("title", "")
    text = f"{title}. {abstract}"

    suggestions: list[str] = []
    seen_templates: set[str] = set()

    for pattern, template in SUGGESTION_PATTERNS:
        if len(suggestions) >= MAX_SUGGESTIONS:
            break

        match = pattern.search(text)
        if match:
            # Avoid duplicate suggestion types
            if template in seen_templates:
                continue
            seen_templates.add(template)

            if match.lastindex and match.lastindex >= 1:
                matched_text = match.group(1).strip().rstrip(".,;:")
                suggestion = template.format(match=matched_text)
            else:
                suggestion = template
            suggestions.append(suggestion)

    return suggestions


def enrich_papers_with_suggestions(
    papers: list[dict[str, Any]],
    config: Any | None = None,
    profile: Any | None = None,
) -> list[dict[str, Any]]:
    """Add a 'suggestions' key to each paper dict.

    When *config* has ``provider`` set to ``"ollama"`` or ``"claude"``
    and a *profile* is given, uses LLM-powered suggestions. Falls back
    to template-based suggestions on any error.
    """
    provider = getattr(config, "provider", "template") if config else "template"
    use_llm = provider in ("ollama", "claude") and profile is not None

    if provider in ("ollama", "claude") and profile is None:
        logger.warning(
            "suggestions.provider is %r but no repo profile was supplied; "
            "falling back to template suggestions.",
            provider,
        )

    llm_failed = False
    for paper in papers:
        if use_llm and not llm_failed:
            try:
                from reporadar.llm_suggestions import generate_llm_suggestions

                assert profile is not None  # guaranteed by use_llm
                paper["suggestions"] = generate_llm_suggestions(paper, profile, config)
                continue
            except Exception as exc:
                # Surface the failure once (don't spam per-paper), then fall
                # back to templates for the rest of this batch.
                logger.warning(
                    "LLM suggestions (provider=%r) failed: %s. "
                    "Falling back to template suggestions.",
                    provider,
                    exc,
                )
                llm_failed = True
        paper["suggestions"] = generate_suggestions(paper)
    return papers
