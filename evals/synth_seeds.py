"""P3: can LLM-generated "uses" phrases seed a citation hop for repos with no bibliography?

The hop is the only channel with measured recall, and its reach is **21/48 = 44%** across the
22-case benchmark — because it seeds from arXiv ids the repo itself cites, and most repos
cite few or none. 27 targets sit in 12 cases the hop cannot serve: 15 where the bibliography
is empty, 10 where it is 1-3 ids, and 2 in `diffusion` where 10 real seeds still missed.

    uv run python evals/synth_seeds.py                # every case the hop underserves
    uv run python evals/synth_seeds.py --case crypto  # one

The idea rests on a distinction §3.2 measured but did not exploit. Asked what a repo *uses*,
an LLM is accurate and mostly names real papers (19/54 phrases matched nothing, against 45/54
for "lacks"). As **direct retrieval** those phrases recovered only 2/24 — they find what the
repo already implements, not what would improve it. But the hop does not need the target; it
needs an **anchor near the target in citation space**. "What this repo implements" is exactly
that anchor. So the phrases are used to find SEEDS, and the neighbourhood does the work.

Pre-registered in ROADMAP P3 before this ran:
  PREDICTION  >=8 of 27 (30%) unreached targets enter a pool of <=50k candidates per repo.
              Brackets: a real bibliography with >=7 seeds reaches 89%; the same phrases as
              direct queries reached 8%.
  KILL        <=2 of 27 (7%) — indistinguishable from the direct-query baseline, so the
              neighbourhood adds nothing over the phrases. Or any pool >50k/repo.

Seeds are ranked by how many distinct phrases found them, then by recency, then capped — a
paper several "uses" phrases agree on is more likely to sit in the repo's own subfield than
one a single phrase surfaced.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import diagnose_query_generation as qg  # noqa: E402
from build_hop_pool import resolve_targets  # noqa: E402
from diagnose_citation_hop import hop, seeds_for  # noqa: E402

from reporadar.config import SuggestionsConfig  # noqa: E402
from reporadar.llm_client import complete  # noqa: E402
from reporadar.profiler import _collect_text_corpus, profile_repo  # noqa: E402

EVALS = Path(__file__).resolve().parent
WORK = EVALS / ".work"
OUT = WORK / "synth_seeds.json"
PHRASE_CACHE = WORK / "synth_phrases.json"

N_PHRASES = 10
PER_PHRASE = 20  # arXiv results kept per phrase
SEED_CAP = 40  # hop() also caps at 60; this is the tighter, deliberate bound
POOL_CEILING = 50_000  # pre-registered: above this it is noise amplification


def phrases_for(case: str, cfg: SuggestionsConfig, cache: dict[str, list[str]]) -> list[str]:
    """The 'uses' phrases for a repo, cached so re-runs cost nothing.

    Reuses `diagnose_query_generation`'s exact prompt rather than a new one, so this is a
    measurement of the *same* phrases §3.2 scored at 2/24 as direct queries — the only
    variable changed is what they are used for.
    """
    if case in cache:
        return cache[case]
    repo = WORK / case
    profile = profile_repo(repo)
    docs = _collect_text_corpus(repo)
    prompt = qg.PROMPTS["uses"].format(
        n=N_PHRASES,
        anchors=", ".join(profile.anchors[:12]) or "none",
        domains=", ".join(profile.domains[:5]) or "general",
        keywords=", ".join(t for t, _ in profile.keywords[:12]) or "n/a",
        docs=(docs[0] if docs else "")[:1500],
    )
    raw = complete(prompt, cfg, max_tokens=400)
    start, end = raw.find("["), raw.rfind("]")
    if start < 0 or end < 0:
        raise ValueError(f"no JSON array in phrase response: {raw[:150]}")
    out = [str(p).strip() for p in json.loads(raw[start : end + 1]) if str(p).strip()]
    cache[case] = out[:N_PHRASES]
    PHRASE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    PHRASE_CACHE.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    return cache[case]


def _yymm(arxiv_id: str) -> int:
    """YYMM as an int, for both id schemes. 0 when neither parses.

    arXiv has two forms: `2201.03545` (post-2007) and `cs/0302037` (before it). Only the
    first was handled, and a bare float() on the second raised mid-sort — an old paper in
    the results is enough to kill the run, and the seed sets here deliberately reach back
    decades because a repo's foundational work is old.
    """
    digits = arxiv_id.split("/")[-1].split(".")[0]
    return int(digits[:4]) if digits[:4].isdigit() else 0


def citation_counts(ids: list[str]) -> dict[str, int]:
    """S2 citation count per id, for hub-ranking seeds. Missing ids are simply absent."""
    from reporadar.citations import _s2_batch_post, _s2_id

    out: dict[str, int] = {}
    for i in range(0, len(ids), 200):
        chunk = ids[i : i + 200]
        data = _s2_batch_post([_s2_id(x) for x in chunk], "externalIds,citationCount", None, 6, 5.0)
        time.sleep(3.0)
        for entry in data or []:
            if not entry:
                continue
            ax = ((entry.get("externalIds") or {}).get("ArXiv") or "").split("v")[0]
            if ax:
                out[ax] = entry.get("citationCount") or 0
    return out


def seeds_from_phrases(phrases: list[str], rank: str = "votes") -> tuple[list[str], int]:
    """Search each phrase QUOTED, rank candidates by phrase agreement. Returns (seeds, hits).

    Quoted because an unquoted space after a field prefix is OR on arXiv, not AND — the
    defect PR #62 fixed, which made every multi-word query an OR union (§3.6).

    *rank* selects which 40 of the matches become seeds:

      "votes"     phrase agreement, then arXiv's own ranking. What P3 ran first, and it
                  reached 3/27 because arXiv relevance has NO impact weighting (§3.0) —
                  the seeds had a MEDIAN OF 3 CITATIONS and a third had zero, against
                  1,210 median for a real bibliography's seeds.
      "citations" S2 citation count. A hop is only as good as whether its seeds are hubs:
                  neighbourhood size measured 26 papers/seed for phrase-ranked seeds
                  against 515 for bibliography seeds, a 20x gap. This ranks for the
                  property the mechanism actually needs.
    """
    votes: Counter[str] = Counter()
    order: dict[str, int] = {}
    nonzero = 0
    for phrase in phrases:
        ids = qg.arxiv_ids(f'all:"{phrase}"', max_results=PER_PHRASE)
        if ids:
            nonzero += 1
        # NOT `rank` — that is this function's parameter, and shadowing it made the
        # citation branch below unreachable: `rank` became an int, `rank == "citations"`
        # was always False, and a whole experiment silently re-ran the control. It was
        # caught only because the two runs produced byte-identical pools on 10 of 11 cases.
        for position, pid in enumerate(ids):
            votes[pid] += 1
            order.setdefault(pid, position)
        time.sleep(3.5)  # arXiv politeness; §3.4 records an IP block from sustained polling
    # phrase agreement first, then the phrase's own ranking, then recency as a tiebreak
    if rank == "citations":
        cites = citation_counts(sorted(votes))
        ranked = sorted(votes, key=lambda p: (-cites.get(p, 0), -votes[p], order[p]))
    else:
        ranked = sorted(votes, key=lambda p: (-votes[p], order[p], -_yymm(p)))
    return ranked[:SEED_CAP], nonzero


def run_case(
    case: str,
    targets: list[str],
    cfg: SuggestionsConfig,
    cache: dict,
    rank: str = "votes",
) -> dict | None:
    real = seeds_for(case)
    print(f"\n[{case}] {len(targets)} targets, {len(real)} real seeds", flush=True)
    try:
        phrases = phrases_for(case, cfg, cache)
    except Exception as exc:  # noqa: BLE001
        print(f"    ! phrase generation failed: {exc}")
        return None
    print(f"    phrases: {', '.join(phrases[:4])}{'...' if len(phrases) > 4 else ''}", flush=True)

    seeds, nonzero = seeds_from_phrases(phrases, rank)
    print(f"    {nonzero}/{len(phrases)} phrases matched papers -> {len(seeds)} seeds", flush=True)
    if not seeds:
        return {"case": case, "targets": targets, "seeds": 0, "pool": 0, "found": [], "failed": 0}

    b, f = hop(seeds, "references"), hop(seeds, "citations")
    failed = b.failed_chunks + f.failed_chunks
    if failed:
        # Same rule as build_hop_pool: a throttled undercount is not a smaller result.
        print(f"    !! {failed} chunk(s) failed — REFUSING to report {case}; re-run later")
        return None
    # Synthetic seeds are NOT the repo's own, so subtract both sets: a "recovered" target
    # must be genuinely discovered, not handed over as a seed.
    pool = (set(b.reached) | set(f.reached)) - set(seeds) - set(real)
    found = sorted(set(targets) & pool)
    print(
        f"    pool={len(pool):,}  recovered={len(found)}/{len(targets)}"
        f"{'  ' + ','.join(found) if found else ''}",
        flush=True,
    )
    return {
        "case": case,
        "targets": targets,
        "seeds": len(seeds),
        "pool": len(pool),
        "found": found,
        "failed": failed,
        "phrases_nonzero": nonzero,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case")
    ap.add_argument("--model", default="claude-haiku-4-5")
    ap.add_argument(
        "--rank",
        choices=("votes", "citations"),
        default="votes",
        help="how to pick 40 seeds from the phrase matches; see seeds_from_phrases",
    )
    args = ap.parse_args()
    qg._load_env()

    manifest = {}
    mf = WORK / "hop_pool" / "manifest.json"
    if mf.is_file():
        manifest = {r["case"]: r for r in json.loads(mf.read_text(encoding="utf-8"))}

    # Only cases the real-bibliography hop underserves; a case it already solves is not P3's.
    unreached = {}
    for case, tg in resolve_targets().items():
        if not tg:
            continue
        got = set(manifest[case]["targets_in_pool"]) if case in manifest else set()
        missing = [t for t in tg if t not in got]
        if missing and (WORK / case).is_dir():
            unreached[case] = missing
    if args.case:
        if args.case not in unreached:
            print(f"{args.case} has no unreached targets (or no clone)")
            return 1
        unreached = {args.case: unreached[args.case]}

    total = sum(len(v) for v in unreached.values())
    print(f"P3: {total} unreached targets across {len(unreached)} cases")
    print("pre-registered: PREDICT >=8/27 at <=50k/repo; KILL <=2/27 or any pool >50k\n")

    cfg = SuggestionsConfig(provider="claude", claude_model=args.model, timeout=90)
    cache = json.loads(PHRASE_CACHE.read_text(encoding="utf-8")) if PHRASE_CACHE.is_file() else {}
    rows = [r for c, t in unreached.items() if (r := run_case(c, t, cfg, cache, args.rank))]

    out_path = OUT if args.rank == "votes" else OUT.with_name(f"synth_seeds_{args.rank}.json")
    # MERGE, never clobber. A `--case` re-run used to overwrite the whole-set results with
    # one row — which is exactly what happened: a single failing retry of `diffusion`
    # replaced an 11-case run with `[]` and printed a KILL verdict against bars scoped to
    # 27 targets. build_hop_pool already merges its manifest; this did not reuse that.
    prior = (
        {r["case"]: r for r in json.loads(out_path.read_text(encoding="utf-8"))}
        if out_path.is_file()
        else {}
    )
    prior.update({r["case"]: r for r in rows})
    merged = [prior[k] for k in sorted(prior)]
    out_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    found = sum(len(r["found"]) for r in merged)
    biggest = max((r["pool"] for r in merged), default=0)
    partial = bool(args.case) or len(merged) < len(unreached)
    print("\n=== P3 RESULT ===")
    print(f"cases in results: {len(merged)}   this run: {len(rows)}/{len(unreached)}")
    print(f"targets reached : {found}/{total}")
    print(f"largest pool    : {biggest:,} (ceiling {POOL_CEILING:,})")
    print(f"\nPRE-REGISTERED: predict >=8, kill <=2, pools <={POOL_CEILING:,}")
    if partial:
        print("verdict: PARTIAL RUN — bars are scoped to all 27 targets, not reported")
    else:
        verdict = "KILL" if found <= 2 else ("MET" if found >= 8 else "BELOW PREDICTION")
        if biggest > POOL_CEILING:
            verdict += " + POOL CEILING BREACHED"
        print(f"verdict: {verdict}")
    print("\nreference: real bibliography >=7 seeds reaches 89%; these phrases as DIRECT")
    print("queries reached 8% (2/24). Written to", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
