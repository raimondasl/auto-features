"""Does Table 3's coverage depend on its one searcher? Post hoc, descriptive and $0. [NR-71]

Table 3 of the paper reports how many of 56 reference papers two retrieval channels reach:
hypothetical-document search (the best rank over four hypotheses within the top 1,000) reaches
34, the citation hop 21, and the two together 43. The 56 are one draw of one searcher (the `cli`
agent) on 20 repositories, filtered by the primary judge. A review asked whether the coverage
partly measures agreement with that searcher's procedure, since the channels were developed
while looking at those papers.

This asks the same question of papers other searchers found for the same 20 repositories: a
redraw of the same searcher, a second prompt version, the Opus 5 agent and the API baseline. The
papers come from the tracked witness set (`witness_set.json`), which keeps only papers the
primary judge scored 2 or more, so no verdict is bought. Only arXiv papers count, because the
index holds only arXiv papers; NR-46's per-source reach mixed in non-arXiv picks and pool
membership, and NUMBERS.md asked for this arXiv-only, same-cohort comparison before searcher
dependence could be discussed again.

Method, identical to Table 3's (`hyde_replication.py`):

* hypothetical-document search: the cached hypotheses (`.work/hyde_hypotheses.json`), the shipped
  encoder and the binary index; a paper is reached when its best rank over the four hypotheses is
  1,000 or better. Ties are broken in the paper's favour, as in Table 3, and also against it.
* the citation hop: membership in the hop pools of P1 (`.work/hop_pool/`, 11 repositories); in
  the other nine repositories the hop never ran, and their papers count as misses, as in Table 3.
* a paper missing from the index is left out of its set's denominator, as in Table 3.

It refuses unless Table 3's stored ranks still give 34, 21 and 43, and every fresh rank equals
NR-46's stored rank for the same paper. Table 3's own run of 25 August does not reproduce exactly:
re-encoding the same hypotheses today moves 13 of the 56 ranks and one paper across the cut, as
NR-46's run of 29 August already did. So the 56 are ranked afresh too, with the same encoder run
as every other set, and the difference is reported. Intervals are Wilson over
papers, as in Table 3, and a bootstrap over the 20 repositories. Each comparison set is also
compared with the 56 by a paired bootstrap over repositories.

Reading rule, written before this script was run but after a scratch computation of the same
comparison had been seen: it is descriptive. It would say coverage does not depend on agreement
with the searcher if the other searchers' pooled union reach lies inside Table 3's interval
[0.64, 0.86], and it would say any advantage belongs to the draw rather than the procedure if the
same searcher's redraw is reached no better than the other procedures. No interval that includes
zero is read as a difference.

    uv run python evals/searcher_reach.py           # ranks from the cache if present
    uv run python evals/searcher_reach.py --ranks   # recompute ranks (~5 min of CPU, no network)
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

EVALS = Path(__file__).resolve().parent
sys.path.insert(0, str(EVALS))
sys.path.insert(0, str(EVALS.parent / "src"))

from reporadar.paper_id import dedup_id, is_arxiv_id  # noqa: E402

WORK = EVALS / ".work"
REPLICATION = WORK / "hyde_replication.json"  # Table 3's stored ranks for the 56
HYP_CACHE = WORK / "hyde_hypotheses.json"
HOP_DIR = WORK / "hop_pool"
RANKS = WORK / "searcher_reach_ranks.json"
NR46_RANKS = WORK / "hyde_witness_ranks.json"  # NR-46's, 0-based
WITNESSES = EVALS / "witness_set.json"
OUT = EVALS / "searcher_reach.json"

TOP_1K = 1000
REFERENCE = "cli"  # the searcher whose draw became the 56
# Not compared: the system's own picks and the agents handed its picks (witness_set's SELF
# sources), and adoptions, which were judged against the repository as it was before adoption.
EXCLUDED = ("reporadar", "cli-v2-opus5-rr@30", "cli-v2-opus5-rrwide@30", "adoption")
SAME_SEARCHER = ("cli-redraw", "cli-redraw@30")
TABLE3 = {"papers": 56, "hyde": 34, "hop": 21, "union": 43}  # from its stored ranks
# The 56 under today's encoder, which equals NR-46's. Ties broken against the paper give 33 and
# 42, exactly Table 3's pessimistic 0.59 and 0.75; only the favourable tie rule moved.
EXPECTED_FRESH = {"hyde": 35, "hop": 21, "union": 43, "hyde_pess": 33, "union_pess": 42}
MIN_READ = 20  # a comparison set smaller than this is reported, not read
TABLE3_UNION_WILSON = (0.64, 0.86)
BOOTSTRAP_N = 5000
BOOTSTRAP_SEED = 20261007

Row = dict[str, Any]


def wilson(k: int, n: int, z: float = 1.959964) -> list[float] | None:
    if n == 0:
        return None
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(mid - half, 4), round(mid + half, 4)]


def reached(row: Row, channel: str) -> bool:
    """Whether one paper is reached by a channel, given its ranks and hop membership."""
    hyde = row["rank_opt"] <= TOP_1K
    hyde_pess = row["rank_pess"] <= TOP_1K
    if channel == "hyde":
        return hyde
    if channel == "hyde_pess":
        return hyde_pess
    if channel == "hop":
        return bool(row["in_hop"])
    if channel == "union":
        return hyde or bool(row["in_hop"])
    if channel == "union_pess":
        return hyde_pess or bool(row["in_hop"])
    raise ValueError(channel)


CHANNELS = ("hyde", "hop", "union", "hyde_pess", "union_pess")


def share(rows: Iterable[Row], channel: str) -> tuple[int, int]:
    rows = list(rows)
    return sum(reached(r, channel) for r in rows), len(rows)


def by_case(rows: list[Row]) -> dict[str, list[Row]]:
    out: dict[str, list[Row]] = {}
    for r in rows:
        out.setdefault(r["case"], []).append(r)
    return out


def bootstrap(
    rows: list[Row], cases: list[str], channel: str, ref: list[Row] | None = None
) -> list[float] | None:
    """Percentile interval of the pooled share over resampled repositories.

    With `ref`, the interval of (share of rows) minus (share of ref) on the same resample, which
    is the paired comparison. A resample where either set is empty is skipped.
    """
    rng = random.Random(BOOTSTRAP_SEED)
    a, b = by_case(rows), by_case(ref or [])
    draws: list[float] = []
    for _ in range(BOOTSTRAP_N):
        pick = [cases[rng.randrange(len(cases))] for _ in cases]
        ka = sum(reached(r, channel) for c in pick for r in a.get(c, []))
        na = sum(len(a.get(c, [])) for c in pick)
        if na == 0:
            continue
        if ref is None:
            draws.append(ka / na)
            continue
        kb = sum(reached(r, channel) for c in pick for r in b.get(c, []))
        nb = sum(len(b.get(c, [])) for c in pick)
        if nb == 0:
            continue
        draws.append(ka / na - kb / nb)
    if len(draws) < BOOTSTRAP_N // 2:
        return None
    draws.sort()
    return [round(draws[int(0.025 * len(draws))], 4), round(draws[int(0.975 * len(draws))], 4)]


def summarise(rows: list[Row], cases: list[str], ref: list[Row] | None) -> Row:
    out: Row = {"papers": len(rows), "repositories": len(by_case(rows))}
    for ch in CHANNELS:
        k, n = share(rows, ch)
        out[ch] = {"reached": k, "share": round(k / n, 4) if n else None}
    out["union"]["wilson95"] = wilson(*share(rows, "union"))
    for ch in ("hyde", "hop", "union"):
        out[ch]["repo_boot95"] = bootstrap(rows, cases, ch)
        if ref is not None:
            k, n = share(rows, ch)
            kr, nr = share(ref, ch)
            out[ch]["minus_reference"] = round(k / n - kr / nr, 4) if n and nr else None
            out[ch]["minus_reference_paired95"] = bootstrap(rows, cases, ch, ref)
    return out


def load_reference() -> list[tuple[str, str, int]]:
    """Table 3's 56 as (case, id, stored optimistic rank)."""
    rows = json.loads(REPLICATION.read_text(encoding="utf-8"))["rows"]
    return [(r["case"], dedup_id(r["target"]), int(r["hyde4-union"])) for r in rows]


def hop_pools() -> dict[str, set[str]]:
    pools: dict[str, set[str]] = {}
    for path in sorted(HOP_DIR.glob("*.jsonl")):
        ids = set()
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                ids.add(dedup_id(json.loads(line)["id"]))
        pools[path.stem] = ids
    return pools


def hop_targets() -> set[str]:
    """The ids P1 flagged as targets it reached, which is how Table 3 counted the hop."""
    out = set()
    for path in sorted(HOP_DIR.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("is_target"):
                    out.add(dedup_id(row["id"]))
    return out


def comparison_papers(cases: set[str]) -> dict[tuple[str, str], list[str]]:
    """Other searchers' arXiv witnesses on the reference cases, with their sources."""
    data = json.loads(WITNESSES.read_text(encoding="utf-8"))["witnesses"]
    out: dict[tuple[str, str], list[str]] = {}
    for case, papers in data.items():
        if case not in cases:
            continue
        for pid, meta in papers.items():
            if not is_arxiv_id(pid):
                continue
            sources = sorted(s for s in meta["sources"] if s not in EXCLUDED)
            if sources:
                out[(case, dedup_id(pid))] = sources
    return out


def compute_ranks(wanted: dict[str, set[str]]) -> dict[str, dict[str, list[int]]]:
    """[optimistic, pessimistic] best rank over the case's hypotheses, as hyde_replication ranks."""
    import numpy as np
    from hyde_replication import MODEL, _hamming, load_index
    from sentence_transformers import SentenceTransformer

    hyps = json.loads(HYP_CACHE.read_text(encoding="utf-8"))
    missing = sorted(c for c in wanted if c not in hyps)
    if missing:
        raise SystemExit(f"no cached hypotheses for {missing}; refusing rather than generating")
    index, raw_positions = load_index()
    positions = {dedup_id(k): v for k, v in raw_positions.items()}
    model = SentenceTransformer(MODEL)
    out: dict[str, dict[str, list[int]]] = {}
    for n, case in enumerate(sorted(wanted), start=1):
        here = {pid: positions[pid] for pid in sorted(wanted[case]) if pid in positions}
        best = {pid: [10**9, 10**9] for pid in here}
        for vec in model.encode(hyps[case], normalize_embeddings=True):
            d = _hamming(index, np.packbits(vec > 0))
            for pid, pos in here.items():
                opt = int((d < d[pos]).sum()) + 1
                pess = int((d <= d[pos]).sum())
                best[pid] = [min(best[pid][0], opt), min(best[pid][1], pess)]
        out[case] = best
        print(
            f"  [{n}/{len(wanted)}] {case:<12} {len(here)} of {len(wanted[case])} in the index",
            flush=True,
        )
    RANKS.write_text(json.dumps(out, indent=0), encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ranks", action="store_true", help="recompute ranks (~5 min of CPU)")
    args = ap.parse_args()
    out = build(recompute=args.ranks)
    OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    a = out["anchors"]
    print(
        f"Table 3 stored: {a['table3_stored_counts']}; fresh for the 56: "
        f"{a['fresh_counts_for_the_56']}; {a['the_56_ranks_moved_since_table3']} of 56 ranks moved"
    )
    for label, s in [("the 56", out["reference"]), *out["sets"].items()]:
        u = s["union"]
        print(
            f"  {label:<26} n={s['papers']:>3} in {s['repositories']:>2} repos  "
            f"hyde {s['hyde']['share']}  "
            f"hop {s['hop']['share']}  union {u['share']} {u.get('wilson95')}  "
            f"paired vs 56 {u.get('minus_reference')} {u.get('minus_reference_paired95')}"
        )
    print(f"not in index: {out['not_in_index']}")
    print(f"reading: {json.dumps(out['reading'])}")
    return 0


def build(recompute: bool = False) -> Row:
    """Everything the artifact holds, from the cached ranks unless told to recompute them."""
    reference = load_reference()
    cases = sorted({c for c, _, _ in reference})
    others = comparison_papers(set(cases))
    wanted: dict[str, set[str]] = {}
    for c, pid, _ in reference:
        wanted.setdefault(c, set()).add(pid)
    for c, pid in others:
        wanted.setdefault(c, set()).add(pid)
    ranks = (
        compute_ranks(wanted)
        if recompute or not RANKS.is_file()
        else json.loads(RANKS.read_text(encoding="utf-8"))
    )
    pools = hop_pools()

    def row(case: str, pid: str) -> Row | None:
        r = ranks.get(case, {}).get(pid)
        if r is None:
            return None  # not in the index: out of the denominator, as in Table 3
        return {
            "case": case,
            "id": pid,
            "rank_opt": r[0],
            "rank_pess": r[1],
            "in_hop": pid in pools.get(case, set()),
        }

    ref_rows = [r for c, pid, _ in reference if (r := row(c, pid)) is not None]

    # Two anchors. Table 3's stored ranks (25 August) must still give 34, 21 and 43. The fresh
    # ranks must equal NR-46's stored ranks (29 August) paper for paper, which shows the
    # hypotheses, encoder and index are the ones the record used. The 25 August run itself does
    # not reproduce exactly: re-encoding the same hypotheses today moves 13 of the 56 ranks and
    # one paper across the cut, as NR-46's run did. So the comparison uses fresh ranks for the
    # 56 too, and reports the difference instead of hiding it.
    stored = {(c, pid): rank for c, pid, rank in reference}
    stored_rows = [
        {**r, "rank_opt": stored[(r["case"], r["id"])], "rank_pess": 0} for r in ref_rows
    ]
    stored_counts = {ch: share(stored_rows, ch)[0] for ch in ("hyde", "hop", "union")}
    if len(ref_rows) != TABLE3["papers"] or stored_counts != {
        k: v for k, v in TABLE3.items() if k != "papers"
    }:
        raise SystemExit(f"Table 3's stored ranks no longer give {TABLE3}: {stored_counts}")
    if {r["id"] for r in ref_rows if r["in_hop"]} != hop_targets() & {r["id"] for r in ref_rows}:
        raise SystemExit("hop membership by id disagrees with P1's is_target flags")
    nr46 = json.loads(NR46_RANKS.read_text(encoding="utf-8"))
    off = [
        (c, pid, nr46[c][pid] + 1, r[0])
        for c, papers in ranks.items()
        for pid, r in papers.items()
        if pid in nr46.get(c, {}) and nr46[c][pid] + 1 != r[0]
    ]
    unanchored = sum(
        1 for c, papers in ranks.items() for pid in papers if pid not in nr46.get(c, {})
    )
    if off:
        raise SystemExit(f"fresh ranks differ from NR-46's stored ranks: {off[:5]}")
    moved = sum(1 for r in ref_rows if stored[(r["case"], r["id"])] != r["rank_opt"])
    got = {ch: share(ref_rows, ch)[0] for ch in ("hyde", "hop", "union", "hyde_pess", "union_pess")}
    if EXPECTED_FRESH and got != EXPECTED_FRESH:
        raise SystemExit(f"fresh counts for the 56 changed: {got} != {EXPECTED_FRESH}")

    ref_keys = {(r["case"], r["id"]) for r in ref_rows}
    # The witness set derives its `cli` source from the same gold set, so on these cases it
    # must be exactly the 56; anything else means the two artifacts have drifted apart.
    cli_keys = {k for k, ss in others.items() if REFERENCE in ss}
    if cli_keys != {(c, pid) for c, pid, _ in reference}:
        raise SystemExit("witness_set's cli papers on these cases are not Table 3's 56")
    sources = sorted({s for ss in others.values() for s in ss} - {REFERENCE})
    sets: dict[str, list[Row]] = {}
    not_indexed: dict[str, int] = {}
    for label in [*sources, "other searchers, pooled"]:
        members = [
            (c, pid)
            for (c, pid), ss in others.items()
            if (
                label in ss
                if label != "other searchers, pooled"
                else any(s != REFERENCE for s in ss)
            )
        ]
        rows = [r for c, pid in members if (r := row(c, pid)) is not None]
        not_indexed[label] = len(members) - len(rows)
        sets[label] = rows

    out: Row = {
        "_comment": (
            "NR-71. Post hoc and descriptive. Table 3's channels measured against other "
            "searchers' arXiv picks on the same 20 repositories, every paper judged >= 2 by "
            "GPT-5.5 (witness_set). Same hypotheses, encoder, index, hop pools and tie rule as "
            "Table 3, which it reproduces. Derived by evals/searcher_reach.py; pinned by "
            "tests/test_searcher_reach.py."
        ),
        "reference": {"label": "the 56 (Table 3), fresh ranks", **summarise(ref_rows, cases, None)},
        "anchors": {
            "table3_stored_counts": stored_counts,
            "fresh_counts_for_the_56": got,
            "the_56_ranks_moved_since_table3": moved,
            "ranks_equal_to_nr46": sum(len(p) for p in ranks.values()) - unanchored,
            "ranks_not_in_nr46": unanchored,
        },
        "sets": {},
        "sets_excluding_the_56": {},
        "not_in_index": not_indexed,
        "table3_union_wilson95": list(TABLE3_UNION_WILSON),
    }
    for label, rows in sets.items():
        if not rows:
            continue
        out["sets"][label] = summarise(rows, cases, ref_rows)
        fresh = [r for r in rows if (r["case"], r["id"]) not in ref_keys]
        if fresh:
            out["sets_excluding_the_56"][label] = summarise(fresh, cases, ref_rows)

    pooled = out["sets"]["other searchers, pooled"]["union"]["share"]
    readable = {k: v for k, v in out["sets"].items() if v["papers"] >= MIN_READ}
    out["reading"] = {
        "_comment": f"Sets under {MIN_READ} papers are reported but not read.",
        "pooled_union_inside_table3_interval": TABLE3_UNION_WILSON[0]
        <= pooled
        <= TABLE3_UNION_WILSON[1],
        "same_searcher_redraw_union": {
            k: v["union"]["share"] for k, v in readable.items() if k in SAME_SEARCHER
        },
        "other_procedures_union": {
            k: v["union"]["share"]
            for k, v in readable.items()
            if k not in SAME_SEARCHER and k != "other searchers, pooled"
        },
    }
    return out


if __name__ == "__main__":
    raise SystemExit(main())
