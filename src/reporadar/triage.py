"""Repo-aware LLM triage: score whether a paper could genuinely improve THIS repo.

The heuristic ranker surfaces topic-adjacent papers; benchmarking against an
agentic Opus baseline showed the real gap is *precision* — RepoRadar ranks many
non-actionable papers into its top tier. Triage adds an LLM actionability score
(0-3) per candidate, grounded in the repo profile, so the digest can gate its
"Top Picks" on genuine applicability (abstaining when nothing qualifies) instead
of the miscalibrated raw-score threshold.

Failures never fabricate a score: a paper whose LLM call errors or returns a
malformed verdict is simply skipped, so it can't be silently ranked as a 0.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from reporadar.llm_client import LLMError, complete
from reporadar.profiler import RepoProfile

logger = logging.getLogger(__name__)

# 0-3 actionability; >= ACTIONABLE_THRESHOLD counts as a genuine "Top Pick".
ACTIONABLE_THRESHOLD = 2

_RUBRIC = """\
You are a senior research engineer deciding whether a paper gives you a method
you could use to IMPROVE THIS repository's OWN code — not whether it is on a
related topic. Score it for THIS repository on a 0-3 scale:
  0 = unrelated, or not applicable to this repository at all.
  1 = related topic, but NO concrete change you could make to THIS code. This is
      the correct score when the paper is any of:
        - a measurement/survey/empirical study that describes a phenomenon but
          proposes no technique to change this code;
        - aimed at a DIFFERENT layer than this repo implements (e.g. a concern
          this codebase deliberately delegates to a dependency or extension);
        - about application-level USAGE or misuse of the tech, not this repo's
          own internals;
        - general tooling (a linter, benchmark, or audit) that applies to any
          project rather than improving this one;
        - a wholesale alternative system to adopt instead, not a technique you
          graft into the existing code.
  2 = proposes a specific method/technique that plugs into THIS repository's
      existing code, AND you can name the concrete component it changes and how.
  3 = directly addresses a known limitation or core capability of THIS repo with
      a strong, specific, implementable improvement to its own code.

Decisive test: before scoring 2 or 3 you MUST be able to name, in one phrase, the
specific file/module/component of THIS repository the method would change and the
change itself. If you cannot, the score is at most 1. Topical adjacency, "could
inspire", and "generally useful for projects like this" are score 1, not 2.

Be strict: for most papers, 0 or 1 is the correct, expected answer. Reserve 2-3
for a method that plugs into this codebase.
Respond with ONLY a JSON object:
{"score": 0|1|2|3, "reason": "<one phrase: the change, or why it does not apply>"}"""


def build_triage_prompt(paper: dict[str, Any], profile: RepoProfile) -> str:
    keywords = ", ".join(term for term, _ in profile.keywords[:12]) or "n/a"
    domains = ", ".join(profile.domains[:5]) if profile.domains else "general"
    anchors = ", ".join(profile.anchors[:12]) if profile.anchors else "none"
    return (
        f"{_RUBRIC}\n\n"
        f"# Repository\n"
        f"Dependencies/libraries: {anchors}\n"
        f"Domains: {domains}\n"
        f"Key topics: {keywords}\n\n"
        f"# Candidate paper\n"
        f"Title: {paper.get('title', 'Unknown')}\n"
        f"Abstract: {paper.get('abstract', '')[:1500]}\n\n"
        f"Score this paper for the repository above."
    )


def _parse_verdict(raw: str) -> tuple[int, str]:
    """Parse {"score", "reason"} from LLM output. Raises on anything malformed."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object in triage response: {raw[:150]}")
    data = json.loads(match.group(0))
    if "score" not in data:
        raise ValueError(f"triage response missing 'score': {raw[:150]}")
    score = int(data["score"])
    if score not in (0, 1, 2, 3):
        raise ValueError(f"triage score out of range 0-3: {score}")
    return score, str(data.get("reason", "")).strip()


def score_actionability(
    paper: dict[str, Any], profile: RepoProfile, llm_cfg: Any
) -> tuple[int, str]:
    """Return (score 0-3, one-line reason). Raises LLMError/ValueError on failure.

    *llm_cfg* is any object with the LLM transport fields (provider, model,
    timeout) — e.g. the SuggestionsConfig.
    """
    raw = complete(build_triage_prompt(paper, profile), llm_cfg, max_tokens=200)
    return _parse_verdict(raw)


def rerank_by_actionability(papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reorder papers so higher LLM actionability (``llm_score``) comes first.

    The listwise half of triage: the heuristic ranker decides the initial order,
    but a paper it buried can still be the most *actionable* one. Sorting by
    ``llm_score`` (with ``score_total`` as the tiebreak) floats those to the head
    so the Top-Picks window/gate sees them instead of cutting them off. Untriaged
    papers (``llm_score`` missing/None) sort after every scored paper, keeping
    their relative ``score_total`` order. Returns a new list; input is untouched.
    """

    def key(paper: dict[str, Any]) -> tuple[int, float]:
        llm = paper.get("llm_score")
        primary = llm if isinstance(llm, int) else -1
        return (primary, float(paper.get("score_total", 0.0)))

    return sorted(papers, key=key, reverse=True)


def triage_papers(
    papers: list[dict[str, Any]],
    profile: RepoProfile,
    llm_cfg: Any,
    *,
    top_k: int = 15,
) -> dict[str, dict[str, Any]]:
    """Score the top-*top_k* papers. Returns ``{arxiv_id: {llm_score, llm_reason}}``.

    A paper whose scoring fails is omitted (never scored 0), so downstream
    tiering treats "couldn't judge" as "not a confident Top Pick", not as a
    confident rejection.
    """
    out: dict[str, dict[str, Any]] = {}
    for paper in papers[:top_k]:
        arxiv_id = paper.get("arxiv_id")
        if not arxiv_id:
            continue
        try:
            score, reason = score_actionability(paper, profile, llm_cfg)
        except (LLMError, ValueError, KeyError, TypeError) as exc:
            logger.warning("Triage failed for %s: %s", arxiv_id, exc)
            continue
        out[arxiv_id] = {"llm_score": score, "llm_reason": reason}
    return out
