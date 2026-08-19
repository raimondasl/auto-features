"""Is the gate's score-3 band separable, and does a rubric clause separate it? (~$0.03.)

    uv run python evals/probe_score3_band.py --probe finescale     # ~$0.01
    uv run python evals/probe_score3_band.py --probe rubric        # ~$0.02
    uv run python evals/probe_score3_band.py --report              # $0, re-reads results

Pre-registered in `evals/RESEARCH-scientific-software.md` section 9, before the first paid
call, bars and all.

On six scientific-software repositories the benchmark's central finding about the gate
inverted: papers it scored **3** were actionable 25/36 (0.694), while papers it scored 2
that then cleared the fine-scale rescore were actionable 28/29 (0.966). Eight of the nine
misses surviving the already-cited rule are score-3 papers, and they are one shape: a paper
that *applies* the repository -- fine-tunes it, benchmarks it, reports results obtained with
it -- and names it in the abstract, which the gate reads as direct relevance.

Two candidate fixes. This script is the ~$0.03 that chooses between them before anyone
spends $25 on a live 25-case run:

* **finescale** -- the rescore was fitted on the score-2 band and has never been run on a 3.
  Nothing in `finescale.py` is score-2-specific; the restriction lives in its caller. If the
  probability separates this band too, the fix is a policy change with no prompt risk.
* **rubric** -- one clause added to the 0-3 gate's rubric. `triage.repo_context_block` is
  NOT touched: the fine-scale probability map is fitted to those exact bytes.

Both score against the 69 GPT-5.5 verdicts already cached under
`evals/cache/judge/v1/gpt-5.5/scisoft-*/`. Nothing is re-judged, so the labels cannot drift
to suit the answer. Profiles and abstracts come from `evals/.work/scisoft/`, staged out of a
session scratchpad, so everything but the model calls is offline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from reporadar.config import SuggestionsConfig  # noqa: E402
from reporadar.paper_id import dedup_id  # noqa: E402
from reporadar.profiler import RepoProfile  # noqa: E402

WORK = ROOT / "evals" / ".work" / "scisoft"
JUDGE = ROOT / "evals" / "cache" / "judge" / "v1" / "gpt-5.5"
RESULTS = WORK / "probe_results.json"

REPOS = ("minimap2", "openmm", "scvi-tools", "chgnet", "mace", "dscribe")
DOMAIN = {
    "minimap2": "bio",
    "openmm": "bio",
    "scvi-tools": "bio",
    "chgnet": "matsci",
    "mace": "matsci",
    "dscribe": "matsci",
}
ACTIONABLE = 2

# The clause under test, appended to the shipped rubric. It names the failure shape rather
# than the repositories it was found on: a paper that uses the project is evidence the
# project works, not a proposal to change it.
CLAUSE = """

A paper that APPLIES this repository -- using it, fine-tuning it, benchmarking it, or
reporting results obtained with it -- is not an improvement to it. Score those 0 or 1
however good the paper is, unless they also propose a change to this codebase. A paper
describing the repository's own published method is the same case: the code already
implements it."""


def _profile_blobs() -> dict[str, Any]:
    return json.loads((WORK / "profiles.json").read_text(encoding="utf-8"))


def profiles() -> dict[str, RepoProfile]:
    """The six repo profiles, cached at staging time so no clone is needed to re-run."""
    return {
        name: RepoProfile(
            keywords=[(term, weight) for term, weight in blob["keywords"]],
            anchors=blob["anchors"],
            domains=blob["domains"],
            prose=blob["prose"],
            corpus_phrases=blob["corpus_phrases"],
            cited_arxiv_ids=frozenset(blob["cited_arxiv_ids"]),
        )
        for name, blob in _profile_blobs().items()
    }


def papers() -> list[dict[str, Any]]:
    """Every judged paper, with the gate score and judge verdict it already carries.

    ``already_cited`` marks the four the shipped rule now removes, so every count below can
    be reported both ways rather than silently on one of them.
    """
    blobs = _profile_blobs()
    out: list[dict[str, Any]] = []
    for name in REPOS:
        cited = set(blobs[name]["cited_arxiv_ids"])
        digest = json.loads((WORK / "digests" / f"{name}.json").read_text(encoding="utf-8"))
        by_id: dict[str, dict[str, Any]] = {}
        tier_of: dict[str, str] = {}
        for tier in ("top_picks", "maybe_relevant", "muted"):
            for paper in digest.get(tier, []):
                by_id[paper["arxiv_id"]] = paper
                tier_of[paper["arxiv_id"]] = tier
        for verdict_file in sorted((JUDGE / f"scisoft-{name}").glob("*.json")):
            verdict = json.loads(verdict_file.read_text(encoding="utf-8"))
            # The judge cache keys on the id as the digest carried it, version suffix and
            # all (`judge.py::_cache_path` only sanitises characters), so both sides are
            # version-stripped before comparison rather than one of them.
            stem = dedup_id(verdict_file.stem)
            full_id = next((i for i in by_id if dedup_id(i) == stem), None)
            if full_id is None:
                print(f"  ! {name}: no digest entry for {verdict_file.stem}")
                continue
            paper = by_id[full_id]
            out.append(
                {
                    "repo": name,
                    "domain": DOMAIN[name],
                    "arxiv_id": full_id,
                    "title": paper["title"],
                    "abstract": paper.get("abstract", ""),
                    "gate": paper.get("llm_score"),
                    "finescale_p": paper.get("finescale_p"),
                    "judge": int(verdict["score"]),
                    "tier": tier_of[full_id],
                    "already_cited": dedup_id(full_id) in cited,
                }
            )
    return out


def _auc(scored: list[tuple[float, int]]) -> float:
    """ROC-AUC by rank, ties at half credit. n is small, so this is exact, not sampled."""
    pos = [s for s, label in scored if label]
    neg = [s for s, label in scored if not label]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def probe_finescale(rows: list[dict[str, Any]], cfg: Any) -> dict[str, Any]:
    """Score the gate-3 band with the shipped rescore, which has never seen a 3."""
    from reporadar.finescale import SHOW_THRESHOLD, score_paper

    profs = profiles()
    band = [r for r in rows if r["gate"] == 3 and not r["already_cited"]]
    actionable = sum(r["judge"] >= ACTIONABLE for r in band)
    print(
        f"\nProbe A - fine-scale over the gate-3 band: {len(band)} papers, {actionable} actionable"
    )
    out = []
    for row in band:
        try:
            expectation, p = score_paper(row, profs[row["repo"]], cfg)
        except Exception as exc:  # noqa: BLE001 - a failure is dropped, never scored 0
            print(f"  ! {row['arxiv_id']}: {exc}")
            continue
        out.append(
            {
                **{k: row[k] for k in ("repo", "domain", "arxiv_id", "title", "judge")},
                "expectation": expectation,
                "p": p,
                "shown_at_threshold": p >= SHOW_THRESHOLD,
            }
        )
        flag = "OK  " if row["judge"] >= ACTIONABLE else "MISS"
        verdict = "show" if p >= SHOW_THRESHOLD else "DROP"
        print(
            f"  {flag} {row['arxiv_id']:<14} judge={row['judge']}  E={expectation:.2f}  "
            f"P={p:.3f}  {verdict}  {row['title'][:50]}"
        )
    return {"probe": "finescale", "papers": out}


def probe_rubric(rows: list[dict[str, Any]], cfg: Any) -> dict[str, Any]:
    """Re-score every shown paper under the shipped rubric, and under it plus one clause.

    Goes through the SHIPPED ``score_actionability`` with the rubric passed in. A harness
    that rebuilds the prompt measures the harness, which this project has published twice.
    """
    from reporadar.triage import _RUBRIC, score_actionability

    profs = profiles()
    shown = [r for r in rows if r["tier"] == "top_picks" and not r["already_cited"]]
    actionable = sum(r["judge"] >= ACTIONABLE for r in shown)
    print(f"\nProbe B - rubric clause over {len(shown)} shown papers, {actionable} actionable")
    out = []
    for row in shown:
        scores: dict[str, int | None] = {}
        for arm, rubric in (("shipped", _RUBRIC), ("clause", _RUBRIC + CLAUSE)):
            try:
                scores[arm] = score_actionability(row, profs[row["repo"]], cfg, rubric=rubric)[0]
            except Exception as exc:  # noqa: BLE001
                print(f"  ! {row['arxiv_id']} [{arm}]: {exc}")
                scores[arm] = None
        out.append(
            {
                **{k: row[k] for k in ("repo", "domain", "arxiv_id", "title", "judge", "gate")},
                "shipped": scores["shipped"],
                "clause": scores["clause"],
            }
        )
        moved = "" if scores["shipped"] == scores["clause"] else "  <-- moved"
        flag = "OK  " if row["judge"] >= ACTIONABLE else "MISS"
        print(
            f"  {flag} {row['arxiv_id']:<14} judge={row['judge']}  shipped={scores['shipped']}"
            f"  clause={scores['clause']}{moved}  {row['title'][:42]}"
        )
    return {"probe": "rubric", "papers": out}


def report(blob: dict[str, Any]) -> None:
    from reporadar.finescale import SHOW_THRESHOLD

    print("\n" + "=" * 78)
    if "finescale" in blob:
        rows = blob["finescale"]["papers"]
        auc = _auc([(r["p"], r["judge"] >= ACTIONABLE) for r in rows])
        shown = [r for r in rows if r["shown_at_threshold"]]
        dropped = [r for r in rows if not r["shown_at_threshold"]]
        kept_ok = sum(r["judge"] >= ACTIONABLE for r in shown)
        dropped_miss = sum(r["judge"] < ACTIONABLE for r in dropped)
        total_miss = sum(r["judge"] < ACTIONABLE for r in rows)
        total_ok = sum(r["judge"] >= ACTIONABLE for r in rows)
        print(f"PROBE A  n={len(rows)}  AUC={auc:.3f}   bars: kill <0.60, win >=0.70")
        print(
            f"  at P>={SHOW_THRESHOLD:.3f}: shows {len(shown)} ({kept_ok} actionable), "
            f"drops {len(dropped)} of which {dropped_miss}/{total_miss} are misses "
            f"(bar: >=4)"
        )
        print(f"  cost: {total_ok - kept_ok}/{total_ok} actionable dropped  (bar: <=2)")
    if "rubric" in blob:
        rows = blob["rubric"]["papers"]
        ok = [r for r in rows if r["judge"] >= ACTIONABLE]
        miss = [r for r in rows if r["judge"] < ACTIONABLE]
        fixed = [r for r in miss if (r["clause"] or 0) < ACTIONABLE <= (r["shipped"] or 0)]
        lost = [r for r in ok if (r["clause"] or 0) < ACTIONABLE <= (r["shipped"] or 0)]
        print(f"PROBE B  n={len(rows)}  ({len(ok)} actionable / {len(miss)} not)")
        print(f"  misses no longer admitted: {len(fixed)}/{len(miss)}   (bar: >=5)")
        print(f"  actionable papers lost:    {len(lost)}/{len(ok)}   (bar: <=3, kill >5)")
        for r in fixed:
            print(
                f"    fixed  {r['arxiv_id']:<14} {r['shipped']}->{r['clause']}  {r['title'][:48]}"
            )
        for r in lost:
            print(
                f"    LOST   {r['arxiv_id']:<14} {r['shipped']}->{r['clause']}  {r['title'][:48]}"
            )
    print("=" * 78)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--probe", choices=["finescale", "rubric"], action="append", default=[])
    parser.add_argument("--report", action="store_true", help="re-read stored results, $0")
    parser.add_argument("--gate-model", default="claude-haiku-4-5")
    args = parser.parse_args()

    blob: dict[str, Any] = {}
    if RESULTS.exists():
        blob = json.loads(RESULTS.read_text(encoding="utf-8"))

    if args.probe:
        rows = papers()
        removed = sum(r["already_cited"] for r in rows)
        print(f"loaded {len(rows)} judged papers ({removed} removed by the already-cited rule)")
        for probe in args.probe:
            if probe == "finescale":
                from reporadar.config import FinescaleConfig

                blob["finescale"] = probe_finescale(rows, FinescaleConfig(enabled=True, timeout=60))
            else:
                cfg = SuggestionsConfig(provider="claude", claude_model=args.gate_model, timeout=60)
                blob["rubric"] = probe_rubric(rows, cfg)
        RESULTS.parent.mkdir(parents=True, exist_ok=True)
        RESULTS.write_text(json.dumps(blob, indent=1), encoding="utf-8")
        print(f"\nwrote {RESULTS}")

    if blob:
        report(blob)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
