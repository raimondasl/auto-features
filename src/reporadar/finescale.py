"""Fine-scale actionability rescore — the second stage that orders the gate's score-2 band.

The 0-3 triage gate (:mod:`reporadar.triage`) is near-binary in practice: almost
everything it admits scores 2, score 3 is rare, and *within* that score-2 band the
fraction of genuinely actionable papers ranged from 0.00 to 1.00 across a 22-repo
benchmark. Two repos could admit ten papers each with near-identical gate scores while
one band was 100% useful and the other 50% — indistinguishable at gate time. That made
the digest's Top Picks a coin flip on the band-heavy repos, and no cheaper signal fixed
it: not the gate's own score, not the heuristic ranker, not a threshold on either.

What fixes it is asking the *same* question on a finer scale and reading the answer's
**distribution** rather than its sample. The model emits one digit 0-9; the score is the
expectation over that digit's token distribution, so a paper the model considers "mostly
7, maybe 8" scores 7.3 rather than collapsing to a single rung. On the benchmark this
ordered the score-2 band at ROC-AUC 0.84 and, gated at the threshold below, took mean
net@2 from +1.91 to +2.91 — rescuing every net-negative repo while costing one paper
each on two net-positive ones. Full write-up and the four rejected alternatives:
evals/RESULTS.md, "Ranking the score-2 band".

Two consequences worth stating plainly:

* **This path is OpenAI-only.** Reading a token distribution needs logprobs, which the
  Anthropic API does not expose. It is therefore opt-in (``triage.finescale.enabled``)
  and needs ``OPENAI_API_KEY``; with it off, RepoRadar behaves exactly as before.
* **The probability map is frozen, and it is tied to this prompt.** SLOPE/INTERCEPT were
  fitted once against :data:`RUBRIC` and :func:`triage.repo_context_block`. Editing
  either without refitting silently decalibrates the threshold, which is why the repo
  block is shared with the gate rather than duplicated here.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any

from reporadar.llm_client import LLMError, top_logprobs
from reporadar.profiler import RepoProfile
from reporadar.triage import repo_context_block

logger = logging.getLogger(__name__)

# The 0-9 rubric. Its bands are the 0-3 gate's bands widened, so the two stages are
# asking one question at two resolutions rather than two different questions.
RUBRIC = """\
You are a senior research engineer scoring whether a paper can be used to IMPROVE a
specific software repository - not merely whether it is on a related topic. Score 0-9:
  0-1 = unrelated or not applicable to this repository.
  2-3 = same general topic, but no concrete, actionable improvement to this code.
  4-6 = proposes a method that could plausibly be integrated to improve this repository,
        with a concrete implementation path. Higher = clearer path, stronger evidence.
  7-9 = directly addresses a known limitation or core capability of this repository with
        a strong, specific, implementable improvement. Higher = more specific, stronger.
Use the full scale - distinguish shades within each band, and be strict: a low score is
the correct answer when the paper is not useful to this code."""

# Frozen two-parameter logistic mapping the 0-9 expectation to P(actionable), fitted on
# 219 judge-labelled papers across 22 repos (evals/exp_finescale.py + exp_features.py).
# Honest generalisation estimate came from leave-one-repo-out cross-fitting; these are
# the all-22 coefficients, which is the artefact that ships.
#
# Do not tune these to move a benchmark number: they are a measurement, and the whole
# point of a global map is that no per-repo knob exists to turn.
SLOPE = 0.9674915191842209
INTERCEPT = -5.809648743203093

# Show/abstain threshold, derived from the metric rather than fitted: under
# net@2 a shown paper is worth 3p - 2, so it pays iff p > 2/3. Not a tuning knob either.
SHOW_THRESHOLD = 2.0 / 3.0

_DIGIT = re.compile(r"^\s*(\d)\s*$")


def probability(expectation: float) -> float:
    """P(the paper is genuinely actionable for this repo) from a 0-9 expectation."""
    return 1.0 / (1.0 + math.exp(-(SLOPE * expectation + INTERCEPT)))


def build_prompt(paper: dict[str, Any], profile: RepoProfile, summary: Any = None) -> str:
    """The fine-scale prompt: the 0-9 rubric, the repository, then one candidate."""
    return (
        f"{RUBRIC}\n\n"
        f"# Repository\n"
        f"{repo_context_block(profile, summary)}\n"
        f"# Candidate paper\n"
        f"Title: {paper.get('title', 'Unknown')}\n"
        f"Abstract: {paper.get('abstract', '')[:1500]}\n\n"
        f"Respond with ONLY a single digit 0-9."
    )


def digit_expectation(alternatives: list[tuple[str, float]]) -> float | None:
    """Expectation over the digit tokens in a first-token distribution.

    Non-digit alternatives are dropped and the remainder renormalised, so stray
    formatting tokens neither shift the score nor shrink it toward zero. Returns None
    when the distribution contains no digit at all — the caller must treat that as
    "unknown", never as a low score.
    """
    weights: dict[int, float] = {}
    for token, prob in alternatives:
        m = _DIGIT.match(token)
        if m:
            digit = int(m.group(1))
            weights[digit] = weights.get(digit, 0.0) + prob
    total = sum(weights.values())
    if total <= 0:
        return None
    return sum(d * p for d, p in weights.items()) / total


def score_paper(
    paper: dict[str, Any], profile: RepoProfile, llm_cfg: Any, summary: Any = None
) -> tuple[float, float]:
    """Return ``(expectation 0-9, probability)``. Raises LLMError/ValueError on failure."""
    alternatives = top_logprobs(build_prompt(paper, profile, summary), llm_cfg)
    expectation = digit_expectation(alternatives)
    if expectation is None:
        raise ValueError("fine-scale response carried no digit token")
    return expectation, probability(expectation)


def score_papers(
    papers: list[dict[str, Any]],
    profile: RepoProfile,
    llm_cfg: Any,
    *,
    summary: Any = None,
) -> dict[str, dict[str, float]]:
    """Score each paper. Returns ``{arxiv_id: {finescale, finescale_p}}``.

    A paper whose call fails is **omitted**, never scored 0 — the same rule the 0-3 gate
    follows, and for the same reason: "could not judge" and "judged not useful" must not
    be the same value downstream. The caller decides what an omission means; see
    :func:`enough_scored`.
    """
    out: dict[str, dict[str, float]] = {}
    for paper in papers:
        arxiv_id = paper.get("arxiv_id")
        if not arxiv_id:
            continue
        try:
            expectation, p = score_paper(paper, profile, llm_cfg, summary)
        except (LLMError, ValueError, KeyError, TypeError) as exc:
            logger.warning("Fine-scale scoring failed for %s: %s", arxiv_id, exc)
            continue
        out[arxiv_id] = {"finescale": expectation, "finescale_p": p}
    return out


def threshold_for_run(
    papers: list[dict[str, Any]], configured: float = SHOW_THRESHOLD
) -> float | None:
    """The gate threshold for an already-stored run, or None if it was not fine-scaled.

    For readers that arrive after ``rr update`` — the archive, notifications, the
    watcher — "did this run get fine-scaled?" is a property of the stored data, not
    something the caller should have to remember and pass along. Scores are persisted
    only when the gate applies (see cli.update), so the presence of any
    ``finescale_p`` answers it exactly.
    """
    return configured if any(p.get("finescale_p") is not None for p in papers) else None


def enough_scored(scored: int, attempted: int, min_fraction: float = 0.5) -> bool:
    """Whether the fine-scale gate should be applied to this run at all.

    A band paper with no fine-scale score does not become a Top Pick — that is the rule
    the benchmark measured, and it is the right one when a handful of calls fail. But
    the same rule applied to a run where the *whole stage* failed (a bad key, a network
    outage) would silently empty the digest's Top Picks and report nothing wrong. So a
    run that scored less than *min_fraction* of what it attempted skips the gate
    entirely and says so, rather than abstaining by accident.
    """
    if attempted <= 0:
        return False
    return scored / attempted >= min_fraction
