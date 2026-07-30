"""Score the ranker against the ratings you have already given (Feature 11).

RepoRadar ships several competing ranking components — RRF hybrid retrieval, SPECTER2
similarity, LLM triage, S2 recommendations, community and attention signals, plus a
feedback loop that tunes weights automatically. Without instrumentation, "did that
help?" is unanswerable, so every one of them is enabled on faith. This module turns
the ratings and stars already in the store into a labeled test set and makes a ranking
change falsifiable on *your* data.

Three methodological choices decide whether the numbers mean anything. Each is a
limitation stated out loud rather than an assumption buried in the code:

**1. Judgments are incomplete, so unjudged papers are removed, not counted wrong.**
A user rates tens of papers out of a corpus of thousands. Ranking the whole corpus and
treating everything unrated as irrelevant would drive precision@10 to nearly zero and
say nothing — the top of the list is mostly papers you never judged. Instead the
ranking is *condensed*: unjudged papers are dropped and the metrics computed over what
remains. That answers a narrower but real question — "among the papers you judged,
does the ranker put the good ones first?" — which is exactly what has to improve for a
ranking change to be worth keeping. It does **not** measure whether the ranker finds
good papers it never surfaced; nothing offline can.

**2. The judged set is not a random sample.** You can only rate what a digest showed
you, and the digest was chosen by an earlier version of this ranker. So these metrics
are conditioned on that selection, and a config that would have surfaced *different*
papers is judged only on the ones you happened to see. That is real selection bias, it
cannot be corrected offline, and :func:`format_report` prints it every time rather
than leaving it in the docs.

**3. Recency is scored as of when each paper was collected**, not as of now. Against
today's clock every stored paper is old, recency collapses to 0 for all of them, and
the component silently disappears from the comparison. Using each paper's ``first_seen``
reproduces the score the ranker actually produced at digest time.

With few labels the metrics are noisy, so :func:`compare_configs` reports a bootstrap
interval on the *difference* instead of inviting a decision from two point estimates
that differ in the third decimal.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from reporadar.config import RankingConfig, RepoRadarConfig
from reporadar.metrics import mrr, ndcg_at_k, precision_at_k, recall_at_k
from reporadar.profiler import RepoProfile
from reporadar.ranker import rank_papers
from reporadar.store import PaperStore

# Below this many judgments the metrics are dominated by noise. Not a hard error: a
# user with 8 ratings should still be able to look, as long as the report says so.
MIN_JUDGMENTS = 20

# Bootstrap resamples for the comparison interval. 2000 is plenty for a 90% interval
# and still runs in well under a second on a few hundred labels.
BOOTSTRAP_SAMPLES = 2000

# Fixed seed: the same data must give the same interval, or a CI gate built on this
# would flap for no reason.
BOOTSTRAP_SEED = 20260730


@dataclass
class Judgments:
    """Binary relevance labels derived from stored ratings and stars."""

    labels: dict[str, int]
    from_ratings: int
    from_stars: int
    ignored_neutral: int

    @property
    def n_relevant(self) -> int:
        return sum(self.labels.values())

    @property
    def n_judged(self) -> int:
        return len(self.labels)


def load_judgments(store: PaperStore) -> Judgments:
    """Turn stored ratings and stars into binary labels.

    Follows the convention ``feedback.compute_adjusted_weights`` already uses, so the
    thing being measured and the thing being tuned agree: rating >= 4 is relevant,
    <= 2 is not, and 3 is dropped rather than forced to a side. A star counts as
    relevant, but an **explicit rating always wins over an implicit star** — the same
    precedence the recommender uses, because ``rr open`` stars papers as a side effect
    of opening them and that is a much weaker signal than a rating.
    """
    ratings = store.get_all_ratings()
    labels: dict[str, int] = {}
    ignored = 0
    for arxiv_id, rating in ratings.items():
        if rating >= 4:
            labels[arxiv_id] = 1
        elif rating <= 2:
            labels[arxiv_id] = 0
        else:
            ignored += 1
    from_ratings = len(labels)

    from_stars = 0
    for arxiv_id in store.get_starred_papers():
        if arxiv_id not in labels and arxiv_id not in ratings:
            labels[arxiv_id] = 1
            from_stars += 1

    return Judgments(
        labels=labels,
        from_ratings=from_ratings,
        from_stars=from_stars,
        ignored_neutral=ignored,
    )


def _first_seen_map(store: PaperStore, arxiv_ids: list[str]) -> dict[str, datetime]:
    """When each paper entered the store, for faithful recency scoring."""
    out: dict[str, datetime] = {}
    for arxiv_id in arxiv_ids:
        paper = store.get_paper(arxiv_id)
        stamp = (paper or {}).get("first_seen")
        if not stamp:
            continue
        try:
            when = datetime.fromisoformat(stamp)
        except ValueError:
            continue
        out[arxiv_id] = when if when.tzinfo else when.replace(tzinfo=UTC)
    return out


def specter_scores_leave_one_out(store: PaperStore, arxiv_ids: list[str]) -> dict[str, float]:
    """SPECTER2 similarity scored without letting a paper into its own query.

    **This exists because the obvious implementation leaks the labels.** SPECTER2's
    query is the centroid of the papers you starred or rated 4-5 — which is exactly the
    set :func:`load_judgments` marks *relevant*. Score a judged paper against that
    centroid and its own vector is inside its own query, so every relevant paper gets an
    inflated similarity **by construction**. Measured with purely random 768-d vectors
    carrying zero information: mean score 0.81 for relevant versus 0.20 for irrelevant,
    and ``rr eval --compare`` duly reported nDCG 0.555 -> 1.000, 90% interval
    [+0.180, +0.751], "B is better (interval excludes zero)". A confident false positive
    recommending a weight change on the strength of noise — the exact failure the
    interval exists to prevent.

    The centroid is a mean of unit vectors, so removing one is arithmetic rather than a
    rebuild: ``(S - u_i) / (n - 1)``. Papers that are not seeds keep the full centroid.

    Production does not need this — there the candidates are newly fetched papers, not
    the rated ones — which is why it lives here rather than in ``specter``.
    """
    try:
        import numpy as np

        from reporadar.specter import (
            _MIN_POOL,
            _MIN_SPREAD,
            SPECTER_DIM,
            load_or_fetch_vectors,
        )
    except ImportError:  # pragma: no cover - numpy ships with the package
        return {}

    seeds = list(store.get_starred_papers())
    seeds += [aid for aid, rating in store.get_all_ratings().items() if rating >= 4]
    seeds = [s for s in dict.fromkeys(seeds) if ":" not in s]
    if not seeds:
        return {}

    wanted = [a for a in dict.fromkeys(list(seeds) + list(arxiv_ids)) if ":" not in a]
    vectors = load_or_fetch_vectors(store, wanted, offline=True)

    units: dict[str, Any] = {}
    for arxiv_id, vec in vectors.items():
        if vec.size != SPECTER_DIM:
            continue  # never mix dimensionalities with a stale MiniLM vector
        norm = float(np.linalg.norm(vec))
        if norm > 0.0:
            units[arxiv_id] = vec / norm

    seed_units = [units[s] for s in seeds if s in units]
    if not seed_units:
        return {}
    total = np.sum(np.vstack(seed_units), axis=0)
    n_seeds = len(seed_units)
    seed_set = {s for s in seeds if s in units}

    sims: dict[str, float] = {}
    for arxiv_id in arxiv_ids:
        unit = units.get(arxiv_id)
        if unit is None:
            continue
        if arxiv_id in seed_set:
            if n_seeds < 2:
                continue  # its own vector is the entire query; nothing honest to say
            query = (total - unit) / (n_seeds - 1)
        else:
            query = total / n_seeds
        qn = float(np.linalg.norm(query))
        if qn > 0.0:
            sims[arxiv_id] = float(np.dot(unit, query) / qn)

    # Same guards as specter.specter_scores: min-max manufactures a full-range signal
    # from nothing, so it only applies when the pool can support it.
    if len(sims) < _MIN_POOL:
        return {}
    lo, hi = min(sims.values()), max(sims.values())
    if hi - lo < _MIN_SPREAD:
        return {}
    scores = {k: (v - lo) / (hi - lo) for k, v in sims.items()}
    # Neutral fill for papers with no vector, matching specter.score_papers: omitting
    # them would drop the component from their average and favour papers S2 has never
    # heard of, since min-max always puts some real paper at 0.0.
    neutral = sum(scores.values()) / len(scores)
    return {a: scores.get(a, neutral) for a in arxiv_ids}


def stored_signals(
    store: PaperStore, arxiv_ids: list[str], ranking_cfg: RankingConfig
) -> dict[str, Any]:
    """Rebuild the optional ranking inputs from data already in the store.

    Without this, ``rr eval`` could not measure the components it exists to make
    falsifiable: every optional score would arrive as ``None``, the absent-is-not-zero
    rule would (correctly) leave them out of the weighted sum, and a ``--compare`` that
    changed only ``w_specter`` would report "NOT SHOWN" on every corpus — a harness that
    looks like it works and answers nothing.

    Everything here is read from SQLite: citation edges, cached SPECTER2 vectors, HF
    upvotes, HN points, withdrawal flags. **No network.** A signal the user's runs never
    fetched stays absent rather than becoming a zero, so the eval measures exactly the
    components they actually populated.
    """
    signals: dict[str, Any] = {}

    if ranking_cfg.w_citation_proximity > 0:
        try:
            from reporadar.citation_graph import build_seed_set, find_citation_links

            seeds = build_seed_set(store)
            if seeds:
                links = find_citation_links(store.get_citations_for(arxiv_ids), seeds)
                if links:
                    signals["citation_proximity"] = {citing: 1.0 for citing in links}
        except Exception:
            pass

    if ranking_cfg.w_specter > 0:
        try:
            # Leave-one-out, and cache-only. Scoring a judged paper against a centroid
            # that contains it leaks the labels; fetching from S2 mid-eval would change
            # the inputs between the two sides of a --compare.
            specter = specter_scores_leave_one_out(store, arxiv_ids)
            if specter:
                signals["specter"] = specter
        except Exception:
            pass

    if ranking_cfg.w_community > 0:
        try:
            from reporadar.sources.hf_papers import normalize_upvotes

            cached = store.get_enrichments(arxiv_ids)
            community = normalize_upvotes(
                {aid: int(e.get("upvotes") or 0) for aid, e in cached.items()}
            )
            if community:
                signals["community"] = community
        except Exception:
            pass

    if ranking_cfg.w_attention > 0:
        try:
            from reporadar.signals.hn import normalize_points

            rows = store.get_signals(arxiv_ids, "hn")
            points = {aid: int(row["value"] or 0) for aid, row in rows.items() if row["value"]}
            attention = normalize_points(points)
            if attention:
                signals["attention"] = attention
        except Exception:
            pass

    try:
        # Not gated on a weight: integrity is a hard penalty, not an opt-in component,
        # so an eval that ignored it would score a ranking the digest would never show.
        flags = store.get_signals(arxiv_ids, "withdrawn")
        withdrawn = {aid for aid, row in flags.items() if row["value"]}
        if withdrawn:
            signals["withdrawn"] = withdrawn
    except Exception:
        pass

    return signals


def rank_judged_papers(
    store: PaperStore,
    judgments: Judgments,
    profile: RepoProfile,
    cfg: RepoRadarConfig,
    ranking_cfg: RankingConfig | None = None,
) -> list[dict[str, Any]]:
    """Re-score every judged paper under *ranking_cfg* and return them best-first.

    Re-scoring is the point: reusing the ``score_total`` frozen in ``paper_scores``
    would make ``--compare`` meaningless, since both configs would read back the same
    stored numbers.
    """
    papers = [p for p in (store.get_paper(a) for a in judgments.labels) if p]
    if not papers:
        return []
    arxiv_ids = [p["arxiv_id"] for p in papers]
    active = ranking_cfg or cfg.ranking
    ranked = rank_papers(
        papers,
        profile,
        active,
        cfg.queries,
        cfg.arxiv.categories,
        cfg.arxiv.lookback_days,
        now_by_id=_first_seen_map(store, arxiv_ids),
        **stored_signals(store, arxiv_ids, active),
    )
    if active.hybrid:
        # RRF fusion with a BM25 lexical ranking, exactly as cli.update stage 8b does.
        # It is pure local text scoring, so unlike LLM triage it *can* be reproduced
        # here — and leaving it out would have made `hybrid` silently unmeasurable.
        try:
            from reporadar.retrieval import hybrid_reorder

            ranked = hybrid_reorder(ranked, papers, profile)
        except Exception:
            pass
    return ranked


def _labels_in_rank_order(ranked: list[dict[str, Any]], labels: dict[str, int]) -> list[int]:
    """The condensed label list: judged papers only, in the ranker's order.

    Unjudged papers are *removed* rather than scored 0 — see the module docstring.
    """
    return [labels[r["arxiv_id"]] for r in ranked if r["arxiv_id"] in labels]


def evaluate(
    store: PaperStore,
    judgments: Judgments,
    profile: RepoProfile,
    cfg: RepoRadarConfig,
    ranking_cfg: RankingConfig | None = None,
    k: int = 10,
) -> dict[str, Any]:
    """Metrics for one ranking config over the judged papers."""
    ranked = rank_judged_papers(store, judgments, profile, cfg, ranking_cfg)
    ordered = _labels_in_rank_order(ranked, judgments.labels)
    total_relevant = sum(judgments.labels.values())
    return {
        "k": k,
        "n_judged": len(ordered),
        "n_relevant": total_relevant,
        "precision@k": precision_at_k(ordered, k),
        "recall@k": recall_at_k(ordered, k, total_relevant),
        "ndcg@k": ndcg_at_k(ordered, k),
        "mrr": mrr(ordered),
        "_labels": ordered,
        "_ranked_ids": [r["arxiv_id"] for r in ranked],
    }


def _metric_value(labels: list[int], metric: str, k: int, total_relevant: int) -> float:
    if metric == "precision@k":
        return precision_at_k(labels, k)
    if metric == "recall@k":
        return recall_at_k(labels, k, total_relevant)
    if metric == "mrr":
        return mrr(labels)
    return ndcg_at_k(labels, k)


def bootstrap_delta(
    ranked_ids_a: list[str],
    ranked_ids_b: list[str],
    labels: dict[str, int],
    metric: str = "ndcg@k",
    k: int = 10,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float]:
    """A 90% bootstrap interval for ``metric(B) - metric(A)``.

    With a few dozen labels a 0.05 nDCG gap is well inside the noise, so two point
    estimates invite a decision the data cannot support; an interval straddling zero
    says "not shown" out loud instead.

    Resamples the judged **papers** with replacement and re-derives each config's
    ranking over the sample, preserving that config's relative order. Resampling rank
    *positions* instead — the obvious-looking shortcut — scrambles the order, so both
    configs collapse to their base rate and *every* interval straddles zero: a
    statistical no-op that reads as rigor. A test pins this by requiring that a perfect
    ranking and a reversed one come out distinguishable.
    """
    ids = [aid for aid in ranked_ids_a if aid in labels]
    if not ids or not ranked_ids_b:
        return {"delta": 0.0, "low": 0.0, "high": 0.0, "significant": 0.0}

    pos_a = {aid: i for i, aid in enumerate(ranked_ids_a)}
    pos_b = {aid: i for i, aid in enumerate(ranked_ids_b)}

    def labels_for(sample: list[str], pos: dict[str, int]) -> list[int]:
        """The sample in one config's ranked order (unranked ids sort last)."""
        return [labels[aid] for aid in sorted(sample, key=lambda a: pos.get(a, len(pos)))]

    total = sum(labels.values())
    observed = _metric_value(labels_for(ids, pos_b), metric, k, total) - _metric_value(
        labels_for(ids, pos_a), metric, k, total
    )

    rng = random.Random(seed)
    n = len(ids)
    deltas: list[float] = []
    for _ in range(samples):
        sample = [ids[rng.randrange(n)] for _ in range(n)]
        la, lb = labels_for(sample, pos_a), labels_for(sample, pos_b)
        # Same multiset either way, so one relevant-count serves both sides.
        relevant = sum(la)
        deltas.append(
            _metric_value(lb, metric, k, relevant) - _metric_value(la, metric, k, relevant)
        )
    deltas.sort()
    low = deltas[int(0.05 * len(deltas))]
    high = deltas[min(int(0.95 * len(deltas)), len(deltas) - 1)]
    return {
        "delta": observed,
        "low": low,
        "high": high,
        # An interval containing 0 means the data does not show a difference — which
        # is the honest verdict, not a failure to find one.
        "significant": 0.0 if low <= 0.0 <= high else 1.0,
    }


# Ranking knobs this harness cannot reproduce. `w_embedding` needs the repo's own
# embedding, which requires the optional sentence-transformers extra and re-encoding the
# profile; `w_citations` needs live citation counts that are only fetched at digest time.
# (`hybrid` used to be here and is now genuinely applied — see rank_judged_papers.)
# A config difference confined to these produces byte-identical rankings, and reporting
# that as "cannot tell the two apart" is indistinguishable from a genuine null result,
# so it is called out separately.
UNMODELLED_KNOBS = ("w_embedding", "w_citations")


def unmodelled_differences(config_a: RankingConfig, config_b: RankingConfig) -> list[str]:
    """Knobs that differ between the configs but that this harness cannot evaluate."""
    return [
        knob
        for knob in UNMODELLED_KNOBS
        if getattr(config_a, knob, None) != getattr(config_b, knob, None)
    ]


def compare_configs(
    store: PaperStore,
    judgments: Judgments,
    profile: RepoProfile,
    cfg: RepoRadarConfig,
    config_a: RankingConfig,
    config_b: RankingConfig,
    k: int = 10,
    metric: str = "ndcg@k",
) -> dict[str, Any]:
    """Evaluate two ranking configs on identical data and bound the difference."""
    result_a = evaluate(store, judgments, profile, cfg, config_a, k)
    result_b = evaluate(store, judgments, profile, cfg, config_b, k)
    return {
        "a": {m: v for m, v in result_a.items() if not m.startswith("_")},
        "b": {m: v for m, v in result_b.items() if not m.startswith("_")},
        "metric": metric,
        "unmodelled": unmodelled_differences(config_a, config_b),
        "bootstrap": bootstrap_delta(
            result_a["_ranked_ids"], result_b["_ranked_ids"], judgments.labels, metric, k
        ),
    }


# How far a metric may fall below a recorded baseline before it counts as a regression.
# Ranking metrics move a little every time a new rating lands, so an exact-equality gate
# would fail constantly and get switched off; this is the same "small differences are
# noise" stance the bootstrap interval takes.
REGRESSION_TOLERANCE = 0.02


def compare_to_baseline(
    result: dict[str, Any],
    snapshot: dict[str, Any],
    metric: str = "ndcg@k",
    tolerance: float = REGRESSION_TOLERANCE,
) -> dict[str, Any]:
    """Check a fresh measurement against a recorded snapshot.

    This is what makes ``--baseline`` more than a diary. Without it, snapshots pile up
    and nothing ever reads them, so the promised "CI catches a ranking regression" is a
    capability claimed in the docs and absent from the code.

    A snapshot taken at a different ``k`` is not comparable, and saying so is more
    useful than differencing two numbers that mean different things.
    """
    stored = snapshot.get("metrics") or {}
    if snapshot.get("k") != result.get("k"):
        return {
            "comparable": False,
            "reason": (
                f"baseline was measured at k={snapshot.get('k')}, this run at k={result.get('k')}"
            ),
        }
    if metric not in stored:
        return {"comparable": False, "reason": f"baseline has no {metric}"}

    delta = float(result[metric]) - float(stored[metric])
    return {
        "comparable": True,
        "metric": metric,
        "baseline": float(stored[metric]),
        "current": float(result[metric]),
        "delta": delta,
        "regressed": delta < -tolerance,
        "tolerance": tolerance,
        # Judgments accumulate between runs, so the two numbers are measured over
        # different label sets. Flag that rather than pretend they are comparable.
        "judgments_changed": snapshot.get("n_judged") != result.get("n_judged"),
        "baseline_label": snapshot.get("label"),
    }


def format_baseline_check(check: dict[str, Any]) -> str:
    """Render a baseline comparison, verdict first."""
    if not check.get("comparable"):
        return f"Baseline not comparable: {check.get('reason', 'unknown')}"
    label = f" ({check['baseline_label']})" if check.get("baseline_label") else ""
    lines = [
        f"{check['metric']} vs baseline{label}: {check['baseline']:.3f} -> "
        f"{check['current']:.3f} ({check['delta']:+.3f})"
    ]
    if check["regressed"]:
        lines.append(f"  -> REGRESSION: more than {check['tolerance']:.2f} below the baseline.")
    else:
        lines.append("  -> no regression.")
    if check["judgments_changed"]:
        lines.append(
            "  -> note: the judged set changed since the baseline, so some of this "
            "movement is new ratings rather than a ranking change."
        )
    return "\n".join(lines)


def degenerate_reason(judgments: Judgments) -> str | None:
    """Why these labels cannot rank anything, or ``None`` if they can.

    A user who only ever stars papers has no negative labels at all, so every metric
    reads 1.000 no matter how the ranker behaves — the most flattering possible output
    produced by the least informative possible data. Saying that plainly beats printing
    a perfect score.
    """
    if judgments.n_judged == 0:
        return "no judged papers"
    if judgments.n_relevant == judgments.n_judged:
        return (
            f"all {judgments.n_judged} judged papers are relevant, so every metric is "
            "1.000 whatever the ranker does. Rate some papers 1-2 to give it something "
            "to get wrong"
        )
    if judgments.n_relevant == 0:
        return (
            f"none of the {judgments.n_judged} judged papers are relevant, so every "
            "metric is 0.000 whatever the ranker does. Rate some papers 4-5"
        )
    return None


def format_report(result: dict[str, Any], judgments: Judgments) -> str:
    """Render one evaluation as plain ASCII (Windows consoles are cp1252)."""
    lines = [
        f"Judged papers: {judgments.n_judged} "
        f"({judgments.from_ratings} rated, {judgments.from_stars} starred-only, "
        f"{judgments.ignored_neutral} neutral 3-star ignored)",
        f"Relevant:      {judgments.n_relevant}",
        "",
    ]
    reason = degenerate_reason(judgments)
    if reason:
        # Headline, not a footnote: these numbers are worse than meaningless because
        # they look like a perfect score.
        lines += [f"DEGENERATE: {reason}.", ""]
    lines += [
        f"  precision@{result['k']}: {result['precision@k']:.3f}",
        f"  recall@{result['k']}:    {result['recall@k']:.3f}",
        f"  nDCG@{result['k']}:      {result['ndcg@k']:.3f}",
        f"  MRR:          {result['mrr']:.3f}",
    ]
    # Both @k metrics have ceilings below 1.0 that say nothing about ranking quality,
    # and a user reading 0.600 as "40% wrong" would draw exactly the wrong conclusion.
    # precision@k divides by the number of items actually in the list (which may be
    # fewer than k), so its ceiling must be computed the same way or the report
    # contradicts itself in the very next line.
    window = min(result["k"], result["n_judged"]) or 1
    p_ceiling = min(1.0, judgments.n_relevant / window)
    r_ceiling = min(1.0, result["k"] / judgments.n_relevant) if judgments.n_relevant else 0.0
    if p_ceiling < 1.0:
        lines.append(
            f"  (precision@{result['k']} maxes out at {p_ceiling:.3f} here: only "
            f"{judgments.n_relevant} of {result['n_judged']} ranked papers are relevant. "
            "nDCG accounts for this.)"
        )
    if r_ceiling < 1.0:
        lines.append(
            f"  (recall@{result['k']} maxes out at {r_ceiling:.3f}: {judgments.n_relevant} "
            f"relevant papers cannot all fit in the top {result['k']}. nDCG accounts for this.)"
        )
    if judgments.n_judged < MIN_JUDGMENTS:
        lines += [
            "",
            f"NOTE: below {MIN_JUDGMENTS} judgments these numbers are mostly noise.",
            "      Rate more papers (`rr rate <id> <1-5>`) before trusting a change.",
        ]
    lines += [
        "",
        "How to read this: it measures whether the ranker ORDERS the papers you",
        "judged well. It cannot measure papers it never showed you, and your",
        "ratings only cover what an earlier ranking already surfaced -- so treat",
        "these as a regression check on changes, not an absolute quality score.",
    ]
    return "\n".join(lines)


def format_comparison(comparison: dict[str, Any], judgments: Judgments) -> str:
    """Render an A/B comparison, leading with whether the difference is real."""
    a, b, boot = comparison["a"], comparison["b"], comparison["bootstrap"]
    metric = comparison["metric"]
    k = a["k"]
    lines = [
        f"Judged papers: {judgments.n_judged} ({judgments.n_relevant} relevant)",
        "",
        f"{'metric':<14} {'A':>9} {'B':>9} {'delta':>9}",
        f"{'-' * 44}",
    ]
    for name in (f"precision@{k}", f"recall@{k}", f"nDCG@{k}", "MRR"):
        key = {
            f"precision@{k}": "precision@k",
            f"recall@{k}": "recall@k",
            f"nDCG@{k}": "ndcg@k",
            "MRR": "mrr",
        }[name]
        lines.append(f"{name:<14} {a[key]:>9.3f} {b[key]:>9.3f} {b[key] - a[key]:>+9.3f}")

    lines += [
        "",
        f"{metric} difference (B - A): {boot['delta']:+.3f}  "
        f"90% interval [{boot['low']:+.3f}, {boot['high']:+.3f}]",
    ]
    if boot["significant"]:
        better = "B" if boot["delta"] > 0 else "A"
        lines.append(f"  -> {better} is better on this data (interval excludes zero).")
    else:
        lines.append(
            "  -> NOT SHOWN: the interval contains zero, so this data cannot tell the two apart."
        )
    if judgments.n_judged < MIN_JUDGMENTS:
        lines.append(
            f"  -> and with {judgments.n_judged} judgments, expect the interval to be wide."
        )
    for knob in comparison.get("unmodelled", ()):
        # Without this, a difference the harness cannot model reads as "cannot tell the
        # two apart" — indistinguishable from a genuine null result.
        lines.append(
            f"  -> WARNING: the configs differ in `{knob}`, which this harness "
            "does not reproduce. That difference is NOT reflected above."
        )
    return "\n".join(lines)
