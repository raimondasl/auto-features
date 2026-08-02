"""Seeded-preference benchmark — does personalization actually improve ranking?

Answers the question the Tier B harness structurally cannot: RepoRadar's
SPECTER2 (Feature 7) and citation-proximity (Feature 8) signals score candidates
against *papers the user liked*, but Tier B has no store and therefore no likes,
so those components never fire.

Method (per labeled Tier A case):

1. Split the case's gold papers: the first ``--seeds`` become "starred", the rest
   are **held out**.
2. Rank the pool *minus the seeds* with the baseline weights, then again with the
   component enabled.
3. Score both against the held-out gold with the standard metric suite.

Deterministic — no LLM judge. SPECTER2 vectors and reference lists come from
Semantic Scholar (free, keyless) and are cached in the per-case store, so a
re-run is offline and identical.

    uv run python evals/run_seeded_eval.py                    # all labeled cases
    uv run python evals/run_seeded_eval.py --case rag --seeds 5
    uv run python evals/run_seeded_eval.py --component specter
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

EVALS_DIR = Path(__file__).resolve().parent
if str(EVALS_DIR) not in sys.path:
    sys.path.insert(0, str(EVALS_DIR))
if str(EVALS_DIR.parent / "src") not in sys.path:
    sys.path.insert(0, str(EVALS_DIR.parent / "src"))

from harness import profile_case_repo, resolve_repo_dir  # noqa: E402
from metrics import evaluate_ranking  # noqa: E402
from seeded import (  # noqa: E402
    build_seeded_store,
    candidate_pool,
    citation_proximity_component,
    heldout_labels,
    specter_component,
    split_relevant,
)

from reporadar.config import QueriesConfig, RankingConfig  # noqa: E402
from reporadar.profiler import RepoProfile  # noqa: E402
from reporadar.ranker import rank_papers  # noqa: E402

FIXTURES_DIR = EVALS_DIR / "fixtures"
BENCHMARK_YAML = EVALS_DIR / "benchmark.yaml"
STORE_DIR = EVALS_DIR / ".work" / "seeded"
DEFAULT_SEEDS = 5
DEFAULT_K = 10
# Weights to sweep for the component under test. A single weight is NOT reportable:
# the response is step-shaped — the effect is ~0 below ~0.25 and saturates by ~0.5,
# so any one point is an arbitrary pick on a plateau. Note also that `w == w_keyword`
# does NOT mean equal influence: keyword_score spans a narrow band over a pool
# (~0.10-0.17) while a min-max-normalized component spans the full [0, 1], so equal
# nominal weights give the component several times the score *range*.
DEFAULT_WEIGHTS = (0.25, 0.5, 1.0)


def _rank(
    candidates: list[dict[str, Any]],
    profile: RepoProfile,
    categories: list[str],
    *,
    specter: dict[str, float] | None = None,
    proximity: dict[str, float] | None = None,
    weight: float = 1.0,
    w_keyword: float = 1.0,
    w_category: float = 0.5,
) -> list[dict[str, Any]]:
    """Rank with recency neutralized so the comparison is deterministic."""
    cfg = RankingConfig(
        w_keyword=w_keyword,
        w_category=w_category,
        w_recency=0.0,
        w_specter=weight if specter else 0.0,
        w_citation_proximity=weight if proximity else 0.0,
    )
    return rank_papers(
        candidates,
        profile,
        cfg,
        QueriesConfig(),
        categories,
        lookback_days=3650,
        specter=specter,
        citation_proximity=proximity,
    )


def _reference_rankings(
    candidates: list[dict[str, Any]],
    profile: RepoProfile,
    categories: list[str],
    labels: dict[str, int],
    k: int,
    deep_k: int,
) -> dict[str, dict[str, float]]:
    """Trivial baselines that show how hard the task actually is.

    Without these a reader can't tell a real capability from a fixture artifact:
    if a category-only ranker already scores 1.000, the case is testing "does the
    arXiv category match", not semantics.
    """
    category_only = _rank(candidates, profile, categories, w_keyword=0.0, w_category=1.0)
    keyword_only = _rank(candidates, profile, categories, w_keyword=1.0, w_category=0.0)
    # "Random": fixture-independent arbitrary order (sorted by id) with no scoring.
    arbitrary = [{"arxiv_id": p["arxiv_id"], "score_total": 0.0} for p in candidates]
    arbitrary.sort(key=lambda s: s["arxiv_id"])
    out = {}
    for name, ranked in (
        ("category-only", category_only),
        ("keyword-only", keyword_only),
        ("id-order", arbitrary),
    ):
        m = evaluate_ranking(ranked, labels, k)
        deep = evaluate_ranking(ranked, labels, deep_k)
        out[name] = {"ndcg@k": m["ndcg@k"], "ndcg@deep": deep["ndcg@k"], "map": m["map"]}
    return out


def run_case(
    name: str,
    pool: list[dict[str, Any]],
    profile: RepoProfile,
    categories: list[str],
    *,
    seeds: int,
    k: int,
    components: list[str],
    api_key: str | None,
    weights: tuple[float, ...] = DEFAULT_WEIGHTS,
    fresh: bool = False,
) -> dict[str, Any] | None:
    """Evaluate one case; returns per-component metrics or None if unusable."""
    seed_ids, heldout = split_relevant(pool, seeds)
    if not seed_ids or not heldout:
        print(f"  [{name}] too few labeled papers for a {seeds}-seed split — skipping.")
        return None

    # Sorted by id so ranking never inherits the fixture's gold-first ordering.
    candidates = sorted(candidate_pool(pool, seed_ids), key=lambda p: p["arxiv_id"])
    labels = heldout_labels(pool, seed_ids)
    # Also score at the depth where every held-out gold *could* be retrieved.
    # At k < n_heldout, recall is capped by construction and a component can reach
    # a perfect nDCG@k while still ranking several gold papers below the cut.
    deep_k = len(heldout)

    STORE_DIR.mkdir(parents=True, exist_ok=True)
    db_path = STORE_DIR / f"{name}.db"
    if fresh and db_path.exists():
        db_path.unlink()
    build_seeded_store(db_path, pool, seed_ids)

    base = evaluate_ranking(_rank(candidates, profile, categories), labels, k)
    base_deep = evaluate_ranking(_rank(candidates, profile, categories), labels, deep_k)
    by_id = {p["arxiv_id"]: p for p in pool}
    print(
        f"  [{name}] {len(seed_ids)} seeded / {len(heldout)} held out of "
        f"{len(pool)} papers  |  baseline nDCG@{k}={base['ndcg@k']:.3f} "
        f"nDCG@{deep_k}={base_deep['ndcg@k']:.3f} P@{k}={base['precision@k']:.3f}"
    )
    for sid in seed_ids:  # what "liked" actually meant, so junk seeds are visible
        paper = by_id.get(sid, {})
        cats = ",".join((paper.get("categories") or [])[:2])
        print(f"           seed: [{cats}] {str(paper.get('title'))[:64]}")

    refs = _reference_rankings(candidates, profile, categories, labels, k, deep_k)
    print(
        "           reference: "
        + "  ".join(f"{n} nDCG@{k}={v['ndcg@k']:.3f}" for n, v in refs.items())
    )

    out: dict[str, Any] = {
        "case": name,
        "n_seeds": len(seed_ids),
        "n_heldout": len(heldout),
        "pool_size": len(pool),
        "k": k,
        "deep_k": deep_k,
        "seed_titles": [str(by_id.get(s, {}).get("title")) for s in seed_ids],
        "baseline": base,
        "baseline_deep": base_deep,
        "reference": refs,
        "at_ceiling": base["ndcg@k"] >= 0.999,
        "components": {},
    }

    for component in components:
        if component == "specter":
            scores, diag = specter_component(db_path, candidates, api_key=api_key)
        elif component == "proximity":
            scores, diag = citation_proximity_component(db_path, candidates, api_key=api_key)
        else:  # pragma: no cover - argparse restricts the choices
            continue

        if not scores:
            print(f"        {component:9}: no signal — diagnostics: {diag}")
            out["components"][component] = {"active": False, "diagnostics": diag}
            continue

        key = "specter" if component == "specter" else "proximity"
        sweep: dict[str, Any] = {}
        for weight in weights:
            ranked = _rank(candidates, profile, categories, weight=weight, **{key: scores})
            m = evaluate_ranking(ranked, labels, k)
            deep = evaluate_ranking(ranked, labels, deep_k)
            sweep[str(weight)] = {
                "metrics": m,
                "ndcg@deep": deep["ndcg@k"],
                "delta_ndcg@k": m["ndcg@k"] - base["ndcg@k"],
                "delta_ndcg@deep": deep["ndcg@k"] - base_deep["ndcg@k"],
                "delta_precision@k": m["precision@k"] - base["precision@k"],
                "delta_map": m["map"] - base["map"],
            }
        # Component alone (no keyword/category) — the honest ablation.
        alone_ranked = _rank(
            candidates,
            profile,
            categories,
            weight=1.0,
            w_keyword=0.0,
            w_category=0.0,
            **{key: scores},
        )
        alone = evaluate_ranking(alone_ranked, labels, k)

        print(
            f"        {component:9}: "
            + "  ".join(
                f"w={w} nDCG@{k}={sweep[str(w)]['metrics']['ndcg@k']:.3f}"
                f"({sweep[str(w)]['delta_ndcg@k']:+.3f})"
                for w in weights
            )
        )
        print(
            f"           alone nDCG@{k}={alone['ndcg@k']:.3f}  "
            f"deep nDCG@{deep_k}={sweep[str(weights[-1])]['ndcg@deep']:.3f}"
            f"({sweep[str(weights[-1])]['delta_ndcg@deep']:+.3f})  diagnostics: {diag}"
        )
        out["components"][component] = {
            "active": True,
            "diagnostics": diag,
            "sweep": sweep,
            "alone": alone,
        }

    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--case", help="Only run this case.")
    parser.add_argument(
        "--seeds", type=int, default=DEFAULT_SEEDS, help="Gold papers to star as 'liked'."
    )
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="Cutoff for @k metrics.")
    parser.add_argument(
        "--component",
        action="append",
        choices=["specter", "proximity"],
        help="Component(s) to measure (default: both).",
    )
    parser.add_argument(
        "--weight",
        type=float,
        action="append",
        help="Component weight(s) to sweep (default 0.25, 0.5, 1.0).",
    )
    parser.add_argument(
        "--fresh", action="store_true", help="Discard the cached per-case store first."
    )
    parser.add_argument("-o", "--output", help="Write results JSON here.")
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be >= 1")
    if args.k < 1:
        parser.error("--k must be >= 1")
    weights = tuple(sorted(args.weight)) if args.weight else DEFAULT_WEIGHTS
    if any(w <= 0 for w in weights):
        parser.error("--weight values must be > 0")

    from run_judge_eval import load_dotenv  # reuse the same .env loading

    load_dotenv(EVALS_DIR / ".env")
    import os

    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY") or None

    components = args.component or ["specter", "proximity"]

    import yaml

    bench = yaml.safe_load(BENCHMARK_YAML.read_text(encoding="utf-8"))
    cases = [c for c in bench["cases"] if not c.get("negative_control")]
    if args.case:
        cases = [c for c in cases if c["name"] == args.case]
        if not cases:
            names = ", ".join(c["name"] for c in bench["cases"])
            print(f"No case {args.case!r}. Available: {names}")
            return 1

    print("=== RepoRadar seeded-preference benchmark ===")
    print(f"components={','.join(components)}  seeds={args.seeds}  k={args.k}")
    print("(deterministic: no LLM judge; S2 vectors/references cached per case)\n")

    results = []
    for case in cases:
        name = case["name"]
        fixture = FIXTURES_DIR / f"{name}.json"
        if not fixture.exists():
            print(f"  [{name}] no fixture — run build_fixtures.py first. Skipping.")
            continue
        pool = json.loads(fixture.read_text(encoding="utf-8"))
        # A duplicated id would corrupt the split silently (last-wins labels, a
        # "held out" paper that isn't in the pool), so reject it loudly.
        seen: dict[str, int] = {}
        for paper in pool:
            aid, label = paper["arxiv_id"], int(paper.get("label") or 0)
            if aid in seen and seen[aid] != label:
                print(f"  [{name}] duplicate arxiv_id {aid} with conflicting labels — skipping.")
                pool = []
                break
            seen[aid] = label
        if not pool:
            continue
        # The real mini-repo profile, exactly as Tier A uses it, so the baseline is
        # the ranking a user would actually get rather than a synthetic one.
        repo_dir = resolve_repo_dir(case)
        if not Path(repo_dir).is_dir():
            # Same defect as run_eval.py had: a case needs a fixture AND a mini-repo, and
            # only the fixture was checked. Building the 8 missing fixtures turned a clean
            # skip into a crash that discarded every result already computed. Fixed there;
            # this runner was missed.
            print(f"  [{name}] no mini-repo at {repo_dir} — skipping (fixture exists).")
            continue
        row = run_case(
            name,
            pool,
            profile_case_repo(repo_dir),
            case.get("expected_categories") or [],
            seeds=args.seeds,
            k=args.k,
            components=components,
            api_key=api_key,
            weights=weights,
            fresh=args.fresh,
        )
        if row:
            results.append(row)

    if not results:
        print("\nNo usable cases.")
        return 1

    print("\n--- summary (per component) ---")
    for component in components:
        rows = [r for r in results if r["components"].get(component, {}).get("active")]
        if not rows:
            print(f"  {component:9}: inactive on every case — nothing measured.")
            continue
        # Cases already at nDCG@k = 1.000 cannot show a gain; averaging their
        # structural zero in understates nothing and overstates the sample size, so
        # report the informative subset explicitly rather than a bare mean.
        informative = [r for r in rows if not r["at_ceiling"]]
        print(f"  {component}: {len(rows)} active, {len(informative)} with headroom")
        for weight in weights:
            deltas = [
                (r["case"], r["components"][component]["sweep"][str(weight)]["delta_ndcg@k"])
                for r in rows
            ]
            total = sum(d for _, d in deltas)
            top = max(deltas, key=lambda cd: abs(cd[1])) if deltas else ("-", 0.0)
            share = f"{abs(top[1]) / abs(total) * 100:.0f}% from {top[0]}" if total else "n/a"
            print(
                f"    w={weight}: mean d-nDCG@{args.k} {total / len(rows):+.4f}  "
                f"per-case {[f'{c}{d:+.3f}' for c, d in deltas]}  ({share})"
            )

    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
