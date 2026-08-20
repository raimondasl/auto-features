"""Where one paper stopped, and why — the read-only answer to "why is my paper not there".

`rr queries` prints what this repository asks for and `rr digest` prints what came back, and
between them sits a pipeline with seven places a paper can be dropped: collection, ranking,
the gate's window, the gate's verdict, the fine-scale rescore, the digest window, and muting.
Until now the only way to find out which one had claimed a given paper was to read the code.

That gap is not academic. On the 2026-08-20 re-baseline, four of the 25 benchmark repositories
returned **nothing at all**, and every one of them had actionable papers sitting in its pool —
ruff's had nine. A zero in `net@2` looks identical whether the digest correctly abstained or
silently discarded nine good papers, so the metric could not tell anyone which had happened.
This is the command that can.

**It re-uses the digest's own functions rather than re-deriving its rules.** `digest_window`
and `categorize_papers` are called here exactly as `cli.digest` calls them, so an explanation
cannot describe a decision the digest did not make. Re-implementing the tier rules would have
been shorter and would have been the C-9/C-12/C-14 defect for the fourth time: one invariant,
two implementations, and the copy drifts silently because both look right.

**Everything here is read-only and offline.** No API calls, no writes, no cost. What it cannot
know is the configuration a *past* run used — the store records scores, not settings — so the
explanation is computed with the config in force now and says so when that could matter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from reporadar.config import RepoRadarConfig
from reporadar.digest import categorize_papers, digest_window
from reporadar.paper_id import dedup_id

# The pipeline's drop points, in the order a paper meets them. `rr why` reports the FIRST one
# that claimed the paper, because that is the only one a user can act on: a paper the queries
# never returned has no gate verdict to explain.
STAGE_ORDER = (
    "collected",
    "ranked",
    "gate",
    "finescale",
    "window",
    "digest",
)


@dataclass
class Step:
    """One pipeline stage as it treated this paper."""

    name: str
    outcome: str  # "pass" | "stop" | "skip" | "info"
    detail: str


@dataclass
class Explanation:
    arxiv_id: str
    found: bool
    title: str = ""
    steps: list[Step] = field(default_factory=list)
    verdict: str = ""
    stopped_at: str = ""
    # What a user could change. Empty when the paper made it, or when nothing would help.
    remedy: str = ""
    paper: dict[str, Any] | None = None
    run_id: int | None = None


def _fmt(value: Any, digits: int = 3) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def _components(paper: dict[str, Any]) -> str:
    """The score components that actually contributed, named."""
    parts = []
    for key, label in (
        ("keyword_score", "keyword"),
        ("category_score", "category"),
        ("recency_score", "recency"),
        ("embedding_score", "embedding"),
        ("citation_score", "citations"),
        ("specter_score", "specter"),
        ("community_score", "community"),
        ("attention_score", "attention"),
    ):
        v = paper.get(key)
        if v is not None:
            parts.append(f"{label} {float(v):.2f}")
    return ", ".join(parts) or "no components recorded"


def explain_paper(
    arxiv_id: str,
    scored: list[dict[str, Any]],
    cfg: RepoRadarConfig,
    *,
    cited_ids: frozenset[str] | None = None,
    run_id: int | None = None,
    known_to_store: bool = True,
) -> Explanation:
    """Walk the pipeline for one paper and report the first stage that stopped it.

    *scored* is a run's papers as :meth:`PaperStore.get_scores_for_run` returns them —
    already ordered by the key the pipeline ranks on.
    """
    wanted = dedup_id(arxiv_id.strip())
    ex = Explanation(arxiv_id=wanted, found=False, run_id=run_id)

    match = next((p for p in scored if dedup_id(str(p.get("arxiv_id", ""))) == wanted), None)

    # ── 1. collection ───────────────────────────────────────────────────────────────
    if match is None:
        ex.steps.append(
            Step(
                "collected",
                "stop",
                "not in this run's candidate pool" if known_to_store else "not in the store at all",
            )
        )
        ex.stopped_at = "collected"
        ex.verdict = (
            "The pipeline never saw this paper: no query returned it, so nothing "
            "downstream had a chance to judge it."
        )
        cats = ", ".join(cfg.arxiv.categories) or "(none)"
        ex.remedy = (
            f"`rr queries` prints the queries this repository sends. Retrieval is scoped to "
            f"arxiv.categories = {cats}; a paper outside those categories cannot be reached "
            f"by the keyword channel however well it matches. A paper on bioRxiv or in a "
            f"journal only is not reachable at all unless `europepmc` is in `sources`."
        )
        return ex

    ex.found = True
    ex.paper = match
    ex.title = str(match.get("title") or "")
    rank = next(i for i, p in enumerate(scored, 1) if p is match)
    query = match.get("matched_query") or "—"
    ex.steps.append(Step("collected", "pass", f"matched query: {query}"))

    # ── 2. ranking ──────────────────────────────────────────────────────────────────
    rrf = match.get("rrf_score")
    ranked = (
        f"#{rank} of {len(scored)} · score {_fmt(match.get('score_total'))} ({_components(match)})"
    )
    if rrf is not None:
        ranked += f" · RRF {float(rrf):.4f}"
    ex.steps.append(Step("ranked", "pass", ranked))

    # ── 3. the gate ─────────────────────────────────────────────────────────────────
    triage_on = cfg.triage.enabled
    llm = match.get("llm_score")
    threshold = cfg.triage.min_actionable
    if not triage_on:
        ex.steps.append(Step("gate", "skip", "triage.enabled is false — no gate ran"))
    elif llm is None:
        ex.steps.append(
            Step(
                "gate",
                "stop",
                f"not scored — the gate reads the top {cfg.triage.top_k} and this paper "
                f"ranked #{rank}",
            )
        )
    else:
        reason = str(match.get("llm_reason") or "").strip()
        suffix = f" — {reason}" if reason else ""
        verdict = "pass" if llm >= threshold else "stop"
        ex.steps.append(Step("gate", verdict, f"score {llm}/3 (needs ≥{threshold}){suffix}"))

    # ── 4. the fine-scale rescore ───────────────────────────────────────────────────
    fs_on = cfg.triage.finescale.enabled
    fs_p = match.get("finescale_p")
    at_band = llm is not None and llm == threshold
    if not fs_on:
        ex.steps.append(Step("finescale", "skip", "triage.finescale.enabled is false"))
    elif not at_band:
        ex.steps.append(Step("finescale", "skip", f"only the score-{threshold} band is rescored"))
    elif fs_p is None:
        ex.steps.append(
            Step(
                "finescale", "stop", "no probability recorded — an unscored band paper is unproven"
            )
        )
    else:
        bar = cfg.triage.finescale.threshold
        ok = fs_p >= bar
        ex.steps.append(
            Step(
                "finescale",
                "pass" if ok else "stop",
                f"P(actionable) {fs_p:.3f} (needs ≥{bar:.3f})",
            )
        )

    # ── 5 & 6. the window and the tiers, from the digest's own functions ────────────
    top_n = cfg.output.top_n
    tri = threshold if triage_on else None
    window, _withdrawn = digest_window(
        list(scored), top_n, triage_threshold=tri, rerank=cfg.triage.rerank
    )
    in_window = any(p is match for p in window)
    ex.steps.append(
        Step(
            "window",
            "pass" if in_window else "stop",
            f"inside the {top_n}-paper digest window"
            if in_window
            else f"outside the {top_n}-paper window — it can never be displayed",
        )
    )

    top, maybe, muted = categorize_papers(
        list(scored),
        top_n=top_n,
        triage_threshold=tri,
        rerank=cfg.triage.rerank,
        finescale_threshold=cfg.triage.finescale.threshold if fs_on else None,
        cited_ids=cited_ids,
    )
    tier = (
        "Top Picks"
        if any(p is match for p in top)
        else "Maybe relevant"
        if any(p is match for p in maybe)
        else "Muted"
        if any(p is match for p in muted)
        else "not shown"
    )
    ex.steps.append(Step("digest", "pass" if tier == "Top Picks" else "info", tier))

    # ── the single-sentence answer ──────────────────────────────────────────────────
    ex.stopped_at, ex.verdict, ex.remedy = _verdict(
        ex, match, tier, rank, llm, threshold, triage_on, fs_on, at_band, fs_p, cfg, in_window
    )
    return ex


def _verdict(
    ex: Explanation,
    paper: dict[str, Any],
    tier: str,
    rank: int,
    llm: int | None,
    threshold: int,
    triage_on: bool,
    fs_on: bool,
    at_band: bool,
    fs_p: float | None,
    cfg: RepoRadarConfig,
    in_window: bool,
) -> tuple[str, str, str]:
    """The first stage that claimed the paper, in one sentence, with what would change it."""
    if paper.get("withdrawn_in"):
        return (
            "muted",
            f"It is shown as withdrawn ({paper['withdrawn_in']}) and muted before tiering, "
            "whatever its scores say.",
            "",
        )
    if paper.get("already_cited"):
        return (
            "muted",
            "This repository's own README, CITATION file or bibliography already cites it, "
            "so it is muted rather than recommended back.",
            "",
        )
    if tier == "Top Picks":
        return ("", f"It is in the digest as a Top Pick (rank #{rank}).", "")

    # The tier came from `categorize_papers` itself and is ground truth. Explaining a
    # different outcome from the gate score alone is how the first version of this function
    # reported "so it is muted" under a trace that said `not shown`: with `triage.rerank`
    # on, a low gate score demotes a paper *out of the window*, so it is not muted, it is
    # gone. Reading the tier first and using the scores only to say WHY is the fix.
    if not in_window:
        if triage_on and cfg.triage.rerank and llm is not None and llm < threshold:
            return (
                "window",
                f"The gate scored it {llm}, and with triage.rerank on that reorders the "
                f"digest by gate score — which pushed it out of the "
                f"{cfg.output.top_n}-paper window entirely. It is not shown at all, not "
                f"even muted.",
                f"It ranked #{rank} on the heuristic score, so raising output.top_n alone "
                "would not recover it; the gate's verdict is what moved it.",
            )
        return (
            "window",
            f"It ranked #{rank}, outside the {cfg.output.top_n}-paper digest window, so it "
            "was dropped before tiering and cannot be displayed.",
            f"Raising output.top_n above {rank} would bring it into the window.",
        )

    if triage_on and llm is None:
        return (
            "gate",
            f"It stopped before the gate: the gate reads the top {cfg.triage.top_k} ranked "
            f"papers and this one ranked #{rank}, so it was never judged. An ungated paper "
            "reaches Maybe at best — unproven is not endorsed.",
            f"Raising triage.top_k above {rank} would put it in front of the gate, at one "
            "extra LLM call per additional paper.",
        )
    if triage_on and llm is not None and llm == 0:
        return (
            "gate",
            "The gate scored it 0 — not applicable to this repository — so it is muted.",
            "",
        )
    if triage_on and llm is not None and 1 <= llm < threshold:
        return (
            "gate",
            f"The gate scored it {llm}, below the {threshold} a Top Pick needs, so it sits "
            "under Maybe relevant.",
            "",
        )
    if fs_on and at_band and (fs_p is None or fs_p < cfg.triage.finescale.threshold):
        got = "no probability was recorded" if fs_p is None else f"it scored {fs_p:.3f}"
        return (
            "finescale",
            f"The gate scored it {llm}, and a paper at the band must also clear "
            f"P(actionable) ≥ {cfg.triage.finescale.threshold:.3f}; {got}, so it dropped to "
            "Maybe relevant.",
            "The threshold is not a tuning knob: net@2 values a shown paper at 3p−2, so "
            "showing pays exactly above p = 2/3.",
        )
    if not triage_on:
        return (
            "digest",
            f"With the gate off, tiering falls back to the heuristic score "
            f"({_fmt(paper.get('score_total'))} against a {cfg.output.top_n}-paper window), "
            f"and it landed in {tier}.",
            "Setting triage.enabled: true replaces the heuristic threshold with a judged "
            "one; the benchmark measured the ungated digest at net@2 −8.12.",
        )
    return ("digest", f"It is in the digest under {tier}.", "")
