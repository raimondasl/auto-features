"""P4 stage 2: blind HyDE replication against the real 3.1M-vector arXiv index.

Authorised by stage 1 (`evals/verify_hyde_deps.py`), which passed 4/4. This asks the only
question that matters about RETRIEVAL_DESIGN Design 2: if the LLM writes *the abstract of
the paper it wishes existed* for a repo, embeds it, and searches all of arXiv, do the
benchmark's known-good papers come back?

**The check stage 1 did not think to make.** Its four checks establish that the index
exists, is fetchable, is fast and contains the targets. None of them establishes that a
vector we compute is comparable to the vectors in it — and if the publisher embedded, say,
title+abstract, or normalised differently, every query would be measuring nothing while
looking perfectly healthy. So this script refuses to run until it reproduces stored vectors
bit-for-bit. Measured: mxbai-embed-large-v1 over the **abstract alone**, L2-normalised,
binarised at >0, `np.packbits` — Hamming **0/1024** on 5 held-out papers.

**Blind protocol.** Hypotheses come from `assemble_repo_context()` — README excerpt,
manifests, file listing — and nothing else. The generator never sees the targets, the pool,
or the judge's verdicts. Hypotheses are cached so the protocol cannot drift between arms.

**Arms**, all searching the same index with the same encoder, differing only in query text:

  hyde4-union     4 hypothesis abstracts, best rank across the four (a union of 4 lists,
                  so its "top-100" spends up to 400 candidates — reported as such)
  hyde4-centroid  the mean of the 4 hypothesis embeddings, one query, one list
  hyde1           the first hypothesis alone — what one guess is worth
  readme          the README excerpt as the query (today's `w_embedding`, same index)
  keywords        the profile's keywords + anchors + domains (today's arXiv query text)

**Pre-registered, restated for the 22-case benchmark before running (2026-08-06).** P4 was
written against 24 targets and predicted >=8/24 (33%) in top-1k, median rank <5,000, and
crypto 2/2, with a kill at <=5/24 (21%). The benchmark now holds **48 targets**, so the same
fractions are **>=16/48 in top-1k** and **kill at <=10/48**. crypto 2/2 stands as written:
it is the specific claim that Design 2 covers what the citation hop cannot.

    uv run python evals/hyde_replication.py --build     # ~415 MB of range requests, $0
    uv run python evals/hyde_replication.py             # ~$0.20 of Haiku, then CPU

The index build fetches the `id` and `vector` columns only — 415 MB of a 2.5 GB dataset,
which is stage 1's C2 doing the work it was verified for.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import diagnose_query_generation as qg  # noqa: E402
from build_hop_pool import resolve_targets  # noqa: E402
from harness import WORK_DIR, assemble_repo_context  # noqa: E402
from verify_hyde_deps import DATASET, RangeFile, _shard_url  # noqa: E402

from reporadar.config import ProfilerConfig, SuggestionsConfig  # noqa: E402
from reporadar.llm_client import complete  # noqa: E402
from reporadar.profiler import profile_repo  # noqa: E402

EVALS = Path(__file__).resolve().parent
WORK = EVALS / ".work"
INDEX_DIR = WORK / "hyde_index"
HYP_CACHE = WORK / "hyde_hypotheses.json"
OUT = WORK / "hyde_replication.json"

MODEL = "mixedbread-ai/mxbai-embed-large-v1"
YEARS = range(1991, 2027)
N_HYPOTHESES = 4
TOP_1K = 1000
TOP_100 = 100
# Pre-registered gates, restated for 48 targets (see module docstring).
PREDICT_TOP1K = 16
KILL_TOP1K = 10
PREDICT_MEDIAN_RANK = 5000

HYDE_PROMPT = """\
You are shown a software repository. Write the ABSTRACTS of {n} different research papers
that do not necessarily exist, but which — if they did — would most improve this specific
codebase.

Each abstract must read exactly like an arXiv abstract in its field: the literature's terms
of art, the usual "we propose / we show / on benchmark X we obtain" register, 120-200 words.
Do not mention the repository, its name, or its files. Do not write a title. Do not hedge or
describe what you are doing — write the abstract itself.

Make the {n} genuinely different from each other: different subproblems of this repository,
not four phrasings of one idea.

Repository:
{context}

Respond with ONLY a JSON array of {n} strings.
"""


def _hamming(index: np.ndarray, query: np.ndarray, chunk: int = 250_000) -> np.ndarray:
    out = np.empty(index.shape[0], dtype=np.uint16)
    for start in range(0, index.shape[0], chunk):
        block = index[start : start + chunk]
        out[start : start + len(block)] = np.bitwise_count(block ^ query).sum(
            axis=1, dtype=np.uint16
        )
    return out


def build_index(refresh: bool = False) -> None:
    """Fetch id+vector for every shard. Resumable; skips shards already on disk."""
    import pyarrow.parquet as pq

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    total_fetched = 0
    for year in YEARS:
        vec_path = INDEX_DIR / f"{year}.npy"
        id_path = INDEX_DIR / f"{year}.ids"
        if vec_path.is_file() and id_path.is_file() and not refresh:
            continue
        t0 = time.perf_counter()
        fh = RangeFile(_shard_url(year))
        table = pq.ParquetFile(fh).read(columns=["id", "vector"])
        ids = [str(x) for x in table.column("id").to_pylist()]
        vecs = table.column("vector").to_pylist()
        arr = np.frombuffer(b"".join(vecs), dtype=np.uint8).reshape(len(vecs), -1)
        assert arr.shape[1] == 128, f"{year}: vector width {arr.shape[1]}, expected 128"
        assert len(ids) == arr.shape[0]
        np.save(vec_path, arr)
        id_path.write_text("\n".join(ids), encoding="utf-8")
        total_fetched += fh.bytes_fetched
        print(
            f"  {year}: {len(ids):>7,} rows  {fh.bytes_fetched / 1e6:6.1f} MB of "
            f"{fh.size / 1e6:6.1f} MB shard  {time.perf_counter() - t0:5.1f}s",
            flush=True,
        )
    if total_fetched:
        print(f"fetched {total_fetched / 1e6:.0f} MB total")


def load_index() -> tuple[np.ndarray, dict[str, int]]:
    shards = sorted(INDEX_DIR.glob("*.npy"))
    if not shards:
        raise SystemExit("no index — run with --build first")
    sizes = [np.load(p, mmap_mode="r").shape[0] for p in shards]
    total = sum(sizes)
    index = np.empty((total, 128), dtype=np.uint8)
    positions: dict[str, int] = {}
    at = 0
    for path, n in zip(shards, sizes, strict=True):
        index[at : at + n] = np.load(path)
        ids = (INDEX_DIR / f"{path.stem}.ids").read_text(encoding="utf-8").split("\n")
        assert len(ids) == n, f"{path.stem}: {len(ids)} ids for {n} vectors"
        for offset, pid in enumerate(ids):
            positions[pid] = at + offset
        at += n
    print(f"index: {total:,} vectors ({index.nbytes / 1e6:.0f} MB), {len(positions):,} unique ids")
    return index, positions


def encoder_reproduces_the_index(model: Any) -> tuple[bool, list[int]]:
    """Refuse to query an index whose vectors we cannot reproduce exactly.

    Five papers from the smallest shard, embedded here and compared bit-for-bit with what
    the publisher stored. Anything but 0 means our queries live in a different space than
    the index and every number downstream is noise wearing a lab coat.
    """
    import pyarrow.parquet as pq

    fh = RangeFile(_shard_url(1992))
    table = pq.ParquetFile(fh).read_row_group(0, columns=["vector", "abstract"])
    rows = table.to_pylist()[:5]
    emb = model.encode([r["abstract"] for r in rows], normalize_embeddings=True)
    dists = []
    for row, vec in zip(rows, emb, strict=True):
        mine = np.packbits(vec > 0).astype(np.uint8)
        theirs = np.frombuffer(row["vector"], dtype=np.uint8)
        dists.append(int(np.bitwise_count(mine ^ theirs).sum()))
    return all(d == 0 for d in dists), dists


def hypotheses_for(case: str, cfg: SuggestionsConfig, cache: dict[str, list[str]]) -> list[str]:
    if case in cache:
        return cache[case]
    ctx = assemble_repo_context(WORK_DIR / case)
    raw = complete(HYDE_PROMPT.format(n=N_HYPOTHESES, context=ctx[:6000]), cfg, max_tokens=2500)
    a, b = raw.find("["), raw.rfind("]")
    if a < 0 or b < 0:
        raise ValueError(f"no JSON array in response: {raw[:160]}")
    out = [str(x).strip() for x in json.loads(raw[a : b + 1]) if str(x).strip()]
    cache[case] = out[:N_HYPOTHESES]
    HYP_CACHE.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    return cache[case]


def _readme_text(case: str) -> str:
    ctx = assemble_repo_context(WORK_DIR / case)
    head = ctx.split("## README (excerpt)")
    return head[1].split("\n## ")[0].strip() if len(head) > 1 else ctx[:2000]


def _keyword_text(case: str) -> str:
    p = profile_repo(WORK_DIR / case, ProfilerConfig())
    return " ".join([t for t, _ in p.keywords[:12]] + p.anchors[:12] + p.domains[:5])


def _ranks(index: np.ndarray, query_bits: np.ndarray, wanted: dict[str, int]) -> dict[str, int]:
    """Optimistic rank (1-based) of each wanted id under one query.

    Ties are broken in the target's favour — `(dists < d).sum() + 1`. At the head of a
    1024-bit Hamming ranking exact ties are rare; deep in the tail they are not, so a
    reported median rank is a lower bound on the true one.
    """
    dists = _hamming(index, query_bits)
    return {pid: int((dists < dists[pos]).sum()) + 1 for pid, pos in wanted.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", action="store_true", help="fetch id+vector columns, then exit")
    ap.add_argument("--refresh", action="store_true", help="re-fetch shards even if cached")
    ap.add_argument("--model", default="claude-haiku-4-5", help="hypothesis generator")
    ap.add_argument("--report", action="store_true", help="re-read saved ranks, re-derive only")
    args = ap.parse_args()

    if args.build:
        build_index(args.refresh)
        return 0
    if args.report:
        report(json.loads(OUT.read_text(encoding="utf-8"))["rows"])
        return 0

    qg._load_env()
    from sentence_transformers import SentenceTransformer

    index, positions = load_index()
    model = SentenceTransformer(MODEL)
    ok, dists = encoder_reproduces_the_index(model)
    print(f"encoder reproduction: hamming {dists} / 1024 each")
    if not ok:
        print("REFUSING TO RUN: our vectors are not the index's vectors.")
        return 1

    targets = resolve_targets()
    cases = sorted(c for c, ids in targets.items() if ids)
    cache = json.loads(HYP_CACHE.read_text(encoding="utf-8")) if HYP_CACHE.is_file() else {}
    cfg = SuggestionsConfig(provider="claude", claude_model=args.model, timeout=180)

    arms = ("hyde4-union", "hyde4-centroid", "hyde1", "readme", "keywords")
    rows: list[dict[str, Any]] = []

    for case in cases:
        wanted = {t: positions[t] for t in targets[case] if t in positions}
        missing = [t for t in targets[case] if t not in positions]
        if missing:
            print(f"[{case}] {missing} not in index — excluded from this case's denominator")
        if not wanted:
            continue
        hyps = hypotheses_for(case, cfg, cache)
        texts = [*hyps, _readme_text(case), _keyword_text(case)]
        emb = model.encode(texts, normalize_embeddings=True)
        hyp_emb, readme_emb, kw_emb = emb[: len(hyps)], emb[-2], emb[-1]
        centroid = hyp_emb.mean(axis=0)
        centroid = centroid / max(float(np.linalg.norm(centroid)), 1e-12)

        per_hyp = [_ranks(index, np.packbits(v > 0), wanted) for v in hyp_emb]
        arm_ranks = {
            "hyde4-union": {t: min(r[t] for r in per_hyp) for t in wanted},
            "hyde4-centroid": _ranks(index, np.packbits(centroid > 0), wanted),
            "hyde1": per_hyp[0],
            "readme": _ranks(index, np.packbits(readme_emb > 0), wanted),
            "keywords": _ranks(index, np.packbits(kw_emb > 0), wanted),
        }
        for target in sorted(wanted):
            rows.append({"case": case, "target": target, **{a: arm_ranks[a][target] for a in arms}})
        best = arm_ranks["hyde4-union"]
        print(
            f"[{case:10}] {len(hyps)} hypotheses, {len(wanted)} targets  "
            f"hyde4 top-1k {sum(1 for r in best.values() if r <= TOP_1K)}/{len(wanted)}  "
            f"median rank {int(np.median(list(best.values()))):,}",
            flush=True,
        )

    OUT.write_text(
        json.dumps({"dataset": DATASET, "model": MODEL, "rows": rows}, indent=2), encoding="utf-8"
    )
    report(rows)
    return 0


def _hop_reached() -> set[str]:
    """Targets the citation hop actually reaches, from P1's persisted pools."""
    reached: set[str] = set()
    pool_dir = WORK / "hop_pool"
    for path in sorted(pool_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            row = json.loads(line)
            if row.get("is_target"):
                reached.add(row["id"])
    return reached


def report(rows: list[dict[str, Any]]) -> None:
    """Everything the run concludes, recomputed from saved ranks. No network, no LLM."""
    arms = ("hyde4-union", "hyde4-centroid", "hyde1", "readme", "keywords")
    n = len(rows)
    print(f"\n=== P4 stage 2 — {n} targets across {len({r['case'] for r in rows})} cases ===")
    print(f"{'arm':16} {'top-100':>10} {'top-1k':>10} {'median rank':>14}")
    summary: dict[str, dict[str, int]] = {}
    for arm in arms:
        ranks = [r[arm] for r in rows]
        hit100 = sum(1 for x in ranks if x <= TOP_100)
        hit1k = sum(1 for x in ranks if x <= TOP_1K)
        med = int(np.median(ranks))
        summary[arm] = {"top100": hit100, "top1k": hit1k, "median": med}
        print(f"{arm:16} {f'{hit100}/{n}':>10} {f'{hit1k}/{n}':>10} {med:>14,}")

    # `hyde4-union` is best-of-4, so its top-1k has already spent up to 4,000 candidates.
    # Comparing it with a single-query arm at 1,000 would be reading a bigger budget as a
    # better query. The single-budget comparison is the centroid; this is the other half.
    print(f"\nequal-candidate comparison — every arm at {N_HYPOTHESES}k candidates:")
    for arm in arms:
        wide = sum(1 for r in rows if r[arm] <= TOP_1K * N_HYPOTHESES)
        print(f"  {arm:16} {wide:>2}/{n} within {TOP_1K * N_HYPOTHESES:,}")

    for name in ("crypto", "systems"):
        sel = [r for r in rows if r["case"] == name]
        if sel:
            hits = sum(1 for r in sel if r["hyde4-union"] <= TOP_1K)
            print(f"\n{name}: {hits}/{len(sel)} in top-1k  ranks {[r['hyde4-union'] for r in sel]}")

    hop = _hop_reached()
    if hop:
        hyde = {r["target"] for r in rows if r["hyde4-union"] <= TOP_1K}
        allt = {r["target"] for r in rows}
        print(
            f"\nchannel overlap over these {len(allt)} targets — "
            f"hop {len(hop & allt)}, hyde(top-1k) {len(hyde)}, "
            f"union {len((hop & allt) | hyde)}, hyde-only {len(hyde - hop)}"
        )

    best = summary["hyde4-union"]
    crypto_hits = sum(1 for r in rows if r["case"] == "crypto" and r["hyde4-union"] <= TOP_1K)
    crypto_n = sum(1 for r in rows if r["case"] == "crypto")
    print(
        f"\nPRE-REGISTERED: >={PREDICT_TOP1K}/{n} in top-1k, median <{PREDICT_MEDIAN_RANK:,}, "
        f"crypto 2/2; KILL at <={KILL_TOP1K}/{n}"
    )
    aggregate = best["top1k"] >= PREDICT_TOP1K and best["median"] < PREDICT_MEDIAN_RANK
    crypto_ok = crypto_n > 0 and crypto_hits == crypto_n
    if aggregate and crypto_ok:
        verdict = "MET"
    elif aggregate:
        # Stated separately on purpose. The aggregate is the headline, but "covers the repos
        # the hop cannot" is the specific claim that made Design 2 worth building, and an
        # aggregate pass must not be allowed to absorb its failure.
        verdict = f"MET ON AGGREGATE, crypto sub-claim NOT met ({crypto_hits}/{crypto_n} in top-1k)"
    elif best["top1k"] <= KILL_TOP1K:
        verdict = "KILL — the REPORTED numbers do not replicate; P1-P3 proceed alone"
    else:
        verdict = "BELOW PREDICTION"
    print(f"verdict: {verdict}")


if __name__ == "__main__":
    raise SystemExit(main())
