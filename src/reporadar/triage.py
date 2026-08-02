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
You are a senior research engineer deciding whether a paper can be used to
IMPROVE a specific software repository — not merely whether it is on a related
topic. Score it for THIS repository on a 0-3 scale:
  0 = unrelated or not applicable to this repository.
  1 = same general topic, but no concrete, actionable improvement to this code.
  2 = proposes a method/technique that could plausibly be integrated to improve
      this repository, with a concrete implementation path.
  3 = directly addresses a known limitation or core capability of this
      repository with a strong, specific, implementable improvement.

Be strict — reserve 2 and 3 for papers whose METHOD could actually be applied to
this code. A low score is the correct answer when the paper is not useful.
Respond with ONLY a JSON object: {"score": 0|1|2|3, "reason": "<one sentence>"}"""


def build_triage_prompt(paper: dict[str, Any], profile: RepoProfile) -> str:
    """The gate's prompt: the rubric, the repository, then one candidate paper.

    The repository half carries ``profile.prose`` when the profile has any. Without it the
    repo is described only by what it *contains* — its libraries and its frequent terms —
    which measured at 366 characters mean against the 6,375 the eval judge reads about the
    same repo before labelling the papers this gate is graded on. Prose is what says what
    the project is FOR, and the keyword half regularly gets that wrong (ColBERT profiles as
    "web APIs" because it depends on flask).

    Prose is additive, not a replacement: whether dropping the keyword block helps is a
    separate, open question. See evals/RESULTS.md.
    """
    keywords = ", ".join(term for term, _ in profile.keywords[:12]) or "n/a"
    domains = ", ".join(profile.domains[:5]) if profile.domains else "general"
    anchors = ", ".join(profile.anchors[:12]) if profile.anchors else "none"
    prose = getattr(profile, "prose", "").strip()
    prose_block = f"\nWhat this project is, in its own words:\n{prose}\n" if prose else ""
    return (
        f"{_RUBRIC}\n\n"
        f"# Repository\n"
        f"Dependencies/libraries: {anchors}\n"
        f"Domains: {domains}\n"
        f"Key topics: {keywords}\n"
        f"{prose_block}\n"
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
