"""P4 stage 1: verify Design 2's four load-bearing dependencies before building on them.

archive/RETRIEVAL_DESIGN.md Design 2 ("Wanted Poster") is the largest single recall candidate in the
project — REPORTED 8/24 in top-100 against 1/24 for TF-IDF, and it covers `crypto` and
`systems`, which are structural zeros for the citation hop. It is also the design whose
every dependency is explicitly unverified, in a project where one REPORTED estimate was off
by 10x and another turned out to be measuring a transport bug. So: verify first, build never
until it passes.

The four checks are pre-registered in ROADMAP P4 and implemented here verbatim:

  C1  the dataset exists, under a licence that permits this use
  C2  columnar range-fetch is real — one column of one shard, without the other 2.5 GB
  C3  query latency is within 4x of the reported 1.87 s over 3.1M vectors
  C4  the benchmark targets are actually in the index

**Thresholds, fixed before the first run.**

  C1  resolves on the Hub; licence in the permissive set below; >=3.0M rows; schema exposes
      a string id and a binary vector column.
  C2  reading `id` + `vector` for ONE row group transfers <=25% of the shard's bytes, and
      every vector decodes to exactly 128 bytes (1024 bits). The 25% bar is what makes the
      REPORTED "~370 MB one-time sync" possible at all: the full dataset is 2.5 GB on disk,
      so without column pruning that number is wrong by ~7x and the cost line of Design 2
      collapses.
  C3  <=4x1.87 = 7.48 s -> PASS. <=10x = 18.7 s -> DEGRADED (`--foundational` batch only,
      per the ROADMAP's own band). >18.7 s -> KILL.
  C4  every target whose arXiv YYMM precedes the dataset snapshot must be present, AND at
      most 4 targets missing in total. Targets newer than the snapshot are a recency
      boundary of a mirror that is re-published periodically, not a defect in the design —
      but they are reported, because a channel that cannot see this month's papers is a
      channel with a stated blind spot.

**Restated for the 22-case benchmark (2026-08-06), before running.** P4 was written when
the benchmark had 12 cases and 24 targets. It now has 22 cases and **48 targets**, and the
hop's measured reach is 44%, not the 75% Design 2 was calibrated against. C4 therefore
checks 48 ids, not 24. Nothing else changes: C1-C3 are properties of the index, not of the
benchmark.

    uv run python evals/verify_hyde_deps.py            # ~40 MB of range requests, $0
    uv run python evals/verify_hyde_deps.py --skip-latency

No LLM calls. No API keys. Nothing is written outside `evals/.work/`.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from build_hop_pool import resolve_targets  # noqa: E402

EVALS = Path(__file__).resolve().parent
WORK = EVALS / ".work"
ID_CACHE = WORK / "hyde_index_ids"
OUT = WORK / "verify_hyde_deps.json"

# PaperMatch's published index — the only Hub dataset matching Design 2's description
# (whole-arXiv, binary vectors, ~3.1M rows, mxbai-embed-large-v1 at ~670 MB of weights).
DATASET = "bluuebunny/arxiv_abstract_embedding_mxbai_large_v1_milvus_binary"
_HUB = "https://huggingface.co"
_VIEWER = "https://datasets-server.huggingface.co"

REPORTED_LATENCY_S = 1.87
LATENCY_PASS_S = 4 * REPORTED_LATENCY_S
LATENCY_DEGRADED_S = 10 * REPORTED_LATENCY_S
MIN_ROWS = 3_000_000
VECTOR_BYTES = 128  # 1024 bits, packed
COLUMNAR_MAX_FRACTION = 0.25
MAX_MISSING_TARGETS = 4
# Redistribution + commercial use, no share-alike. A non-commercial or unstated licence is
# a fail, not a footnote: the index would ship inside a tool other people run.
PERMISSIVE = {"apache-2.0", "mit", "bsd-3-clause", "cc0-1.0", "cc-by-4.0", "odc-by"}

_UA = {"User-Agent": "reporadar-evals/1.0 (P4 dependency verification)"}


def _get(url: str, *, headers: dict[str, str] | None = None, retries: int = 4) -> bytes:
    last: Exception | None = None
    for attempt in range(retries):
        req = urllib.request.Request(url, headers={**_UA, **(headers or {})})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return bytes(resp.read())
        except (urllib.error.URLError, TimeoutError, OSError) as exc:  # noqa: PERF203
            last = exc
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"GET failed after {retries} tries: {url}") from last


def _json(url: str) -> Any:
    return json.loads(_get(url).decode("utf-8"))


class RangeFile(io.RawIOBase):
    """A seekable read-only file over HTTP Range requests, counting what it transfers.

    This IS check 2. Parquet keeps its schema and per-column byte offsets in a footer, so a
    reader that can seek never has to touch the columns it was not asked for — but only if
    the server honours `Range`. If it does not, every "columnar" cost estimate in Design 2
    is really the cost of downloading the whole shard, and `bytes_fetched` will say so.
    """

    def __init__(self, url: str) -> None:
        super().__init__()
        self.url = url
        self._pos = 0
        self.bytes_fetched = 0
        self.requests = 0
        req = urllib.request.Request(url, headers=_UA, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as resp:
            # LFS objects report the real object size in x-linked-size; Content-Length on
            # the pointer redirect is not it.
            linked = resp.headers.get("x-linked-size")
            self.size = int(linked or resp.headers["Content-Length"])
            self.accept_ranges = (resp.headers.get("accept-ranges") or "").lower()

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self._pos, io.SEEK_END: self.size}[whence]
        self._pos = max(0, min(self.size, base + offset))
        return self._pos

    def read(self, size: int = -1) -> bytes:  # type: ignore[override]
        if size is None or size < 0:
            size = self.size - self._pos
        size = min(size, self.size - self._pos)
        if size <= 0:
            return b""
        end = self._pos + size - 1
        data = _get(self.url, headers={"Range": f"bytes={self._pos}-{end}"})
        self.bytes_fetched += len(data)
        self.requests += 1
        self._pos += len(data)
        return data

    def readinto(self, b: Any) -> int:  # type: ignore[override]
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def _shard_url(year: int) -> str:
    return f"{_HUB}/datasets/{DATASET}/resolve/main/data/{year}.parquet"


def _year_of(arxiv_id: str) -> int | None:
    """2106.09685 -> 2021. Old-style ids (cs/0123456) have no YYMM prefix and return None."""
    head = arxiv_id.split(".")[0]
    if len(head) != 4 or not head.isdigit():
        return None
    return 2000 + int(head[:2])


def check_exists() -> dict[str, Any]:
    print("C1  dataset exists, permissive licence, whole-arXiv scale")
    info = _json(f"{_HUB}/api/datasets/{DATASET}")
    licence = (info.get("cardData") or {}).get("license")
    licence = licence[0] if isinstance(licence, list) else licence
    viewer = _json(f"{_VIEWER}/info?dataset={DATASET}")
    default = viewer["dataset_info"]["default"]
    rows = int(default["splits"]["train"]["num_examples"])
    features = default["features"]
    has_id = features.get("id", {}).get("dtype") == "string"
    has_vec = features.get("vector", {}).get("dtype") == "binary"
    ok = bool(
        not info.get("disabled")
        and not info.get("gated")
        and licence in PERMISSIVE
        and rows >= MIN_ROWS
        and has_id
        and has_vec
    )
    print(f"    licence={licence}  rows={rows:,}  id={has_id} vector(binary)={has_vec}")
    print(f"    last modified {info.get('lastModified')}   gated={info.get('gated')}")
    return {
        "check": "exists+licence",
        "pass": ok,
        "licence": licence,
        "rows": rows,
        "last_modified": info.get("lastModified"),
        "has_id_column": has_id,
        "has_binary_vector_column": has_vec,
    }


def check_columnar(year: int = 2021) -> dict[str, Any]:
    print(f"C2  columnar range-fetch — one row group of data/{year}.parquet")
    import pyarrow.parquet as pq

    fh = RangeFile(_shard_url(year))
    pf = pq.ParquetFile(fh)
    after_meta = fh.bytes_fetched
    table = pf.read_row_group(0, columns=["id", "vector"])
    vectors = table.column("vector").to_pylist()
    lengths = {len(v) for v in vectors if v is not None}
    fraction = fh.bytes_fetched / max(fh.size, 1)
    ok = fraction <= COLUMNAR_MAX_FRACTION and lengths == {VECTOR_BYTES}
    print(
        f"    shard={fh.size / 1e6:.1f} MB  fetched={fh.bytes_fetched / 1e6:.1f} MB "
        f"({fraction:.1%}) in {fh.requests} range requests  accept-ranges={fh.accept_ranges}"
    )
    print(
        f"    row group 0: {table.num_rows:,} rows, vector lengths {sorted(lengths)} bytes "
        f"(footer+metadata cost {after_meta / 1e3:.1f} kB)"
    )
    return {
        "check": "columnar range-fetch",
        "pass": ok,
        "shard_bytes": fh.size,
        "fetched_bytes": fh.bytes_fetched,
        "fraction": round(fraction, 4),
        "range_requests": fh.requests,
        "accept_ranges": fh.accept_ranges,
        "row_group_rows": table.num_rows,
        "vector_lengths": sorted(lengths),
        "row_groups": pf.num_row_groups,
    }


def check_latency(rows: int, repeats: int = 3) -> dict[str, Any]:
    """Time a full Hamming scan over a synthetic index of the real shape.

    Synthetic on purpose. The scan is XOR + popcount + argpartition over a fixed-size uint8
    array; its cost depends on the array's shape and dtype and not at all on the bits in it,
    so random vectors measure the same thing as real ones without a 400 MB download. What
    this does NOT measure is the encode step (mxbai-embed-large-v1, ~670 MB of weights) —
    that is a separate cost, and stage 2 has to pay it before quoting an end-to-end number.
    """
    print(f"C3  query latency — Hamming scan over {rows:,} x {VECTOR_BYTES}B synthetic vectors")
    import numpy as np

    rng = np.random.default_rng(20260806)
    index = rng.integers(0, 256, size=(rows, VECTOR_BYTES), dtype=np.uint8)
    query = rng.integers(0, 256, size=VECTOR_BYTES, dtype=np.uint8)
    chunk = 250_000
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        dists = np.empty(rows, dtype=np.uint16)
        for start in range(0, rows, chunk):
            block = index[start : start + chunk]
            dists[start : start + len(block)] = np.bitwise_count(block ^ query).sum(
                axis=1, dtype=np.uint16
            )
        np.argpartition(dists, 100)[:100]
        times.append(time.perf_counter() - t0)
    best = min(times)
    if best <= LATENCY_PASS_S:
        band = "PASS"
    elif best <= LATENCY_DEGRADED_S:
        band = "DEGRADED"
    else:
        band = "KILL"
    print(
        f"    best of {repeats}: {best:.2f} s  (reported {REPORTED_LATENCY_S:.2f} s, "
        f"{best / REPORTED_LATENCY_S:.1f}x)  -> {band}"
    )
    print(f"    index resident: {index.nbytes / 1e6:.0f} MB")
    return {
        "check": "query latency",
        "pass": band == "PASS",
        "band": band,
        "best_seconds": round(best, 3),
        "all_seconds": [round(t, 3) for t in times],
        "ratio_to_reported": round(best / REPORTED_LATENCY_S, 2),
        "index_bytes": int(index.nbytes),
        "synthetic": True,
        "excludes_encode_step": True,
    }


def _ids_for_year(year: int, refresh: bool = False) -> set[str]:
    """Fetch ONLY the id column of one shard, over Range. Cached — ids do not change."""
    ID_CACHE.mkdir(parents=True, exist_ok=True)
    cached = ID_CACHE / f"{year}.txt"
    if cached.is_file() and not refresh:
        return set(cached.read_text(encoding="utf-8").split())
    import pyarrow.parquet as pq

    fh = RangeFile(_shard_url(year))
    ids = pq.ParquetFile(fh).read(columns=["id"]).column("id").to_pylist()
    out = {str(i) for i in ids if i}
    cached.write_text("\n".join(sorted(out)), encoding="utf-8")
    print(
        f"    {year}: {len(out):,} ids  ({fh.bytes_fetched / 1e6:.1f} MB of "
        f"{fh.size / 1e6:.1f} MB shard, {fh.bytes_fetched / max(fh.size, 1):.1%})"
    )
    return out


def coverage_verdict(targets: list[str], present: set[str], snapshot: str) -> dict[str, Any]:
    """Decide C4 from a target list and the ids actually in the index.

    Split out from the fetching so the rule is testable without the network, because the
    rule is the interesting part: a miss that PREDATES the snapshot is a hole in the index
    and fails the check, while a miss that POSTDATES it is the recency boundary of a mirror
    that is re-published on its own schedule. Collapsing the two into one count would let a
    genuinely incomplete index pass by blaming the calendar.
    """
    missing = [t for t in targets if t not in present]
    snap_ym = int(snapshot[:4]) * 100 + int(snapshot[5:7])

    def _ym(t: str) -> int:
        head = t.split(".")[0]
        return 2000 * 100 + int(head[:2]) * 100 + int(head[2:4])

    newer = [t for t in missing if _ym(t) >= snap_ym]
    older = [t for t in missing if _ym(t) < snap_ym]
    return {
        "check": "target coverage",
        "pass": not older and len(missing) <= MAX_MISSING_TARGETS,
        "targets": len(targets),
        "present": len(targets) - len(missing),
        "missing": missing,
        "missing_newer_than_snapshot": newer,
        "missing_older_than_snapshot": older,
        "snapshot": snapshot,
    }


def check_coverage(snapshot: str, refresh: bool = False) -> dict[str, Any]:
    print("C4  are the benchmark targets in the index?")
    targets = resolve_targets()
    flat = sorted({t for ids in targets.values() for t in ids})
    years = sorted({y for t in flat if (y := _year_of(t)) is not None})
    print(
        f"    {len(flat)} targets across {len([c for c in targets if targets[c]])} cases; "
        f"id column of {len(years)} year shards"
    )
    present: set[str] = set()
    for year in years:
        present |= _ids_for_year(year, refresh)
    out = coverage_verdict(flat, present, snapshot)
    out["index_ids"] = len(present)
    missing = out["missing"]
    print(f"    present {out['present']}/{len(flat)}   missing {len(missing)}")
    if missing:
        for case, ids in sorted(targets.items()):
            miss = [t for t in ids if t in missing]
            if miss:
                print(f"      {case:10} missing {miss}")
        print(
            f"    of those, {len(out['missing_newer_than_snapshot'])} postdate the "
            f"{snapshot[:7]} snapshot, {len(out['missing_older_than_snapshot'])} do not"
        )
    return out


GATE_MESSAGES = {
    "INCOMPLETE": "INCOMPLETE — a skipped check is not a passed check; the gate is not satisfied.",
    "OPEN": "GATE OPEN: stage 2 (blind HyDE-4 replication) is authorised by P4.",
    "CLOSED": "GATE CLOSED: P4's kill clause fires — Design 2 dies at zero build cost.",
}


def gate_verdict(results: list[dict[str, Any]]) -> str:
    """OPEN only on 4/4. A skipped check is INCOMPLETE, never a pass.

    P4's gate is "stage 2 only if 4/4 pass", and the cheapest way to fake that is to skip
    the expensive check and count 3/3. `--skip-latency` exists (C3 allocates ~400 MB), so
    the arithmetic has to refuse it explicitly rather than divide by whatever ran.
    """
    if any(r["pass"] is None for r in results):
        return "INCOMPLETE"
    return "OPEN" if all(r["pass"] for r in results) else "CLOSED"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-latency", action="store_true", help="skip C3 (allocates ~400 MB)")
    ap.add_argument("--refresh-ids", action="store_true", help="re-fetch cached id columns")
    ap.add_argument("--shard-year", type=int, default=2021, help="shard used for C2")
    args = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    c1 = check_exists()
    results.append(c1)
    print()
    results.append(check_columnar(args.shard_year))
    print()
    if args.skip_latency:
        print("C3  skipped (--skip-latency)")
        results.append({"check": "query latency", "pass": None, "band": "SKIPPED"})
    else:
        results.append(check_latency(max(int(c1["rows"]), MIN_ROWS)))
    print()
    results.append(check_coverage(str(c1["last_modified"]), args.refresh_ids))

    OUT.write_text(json.dumps({"dataset": DATASET, "checks": results}, indent=2), "utf-8")
    print("\n=== P4 stage 1 ===")
    for r in results:
        mark = {True: "PASS", False: "FAIL", None: "SKIP"}[r["pass"]]
        print(f"  {mark:4}  {r['check']}")
    decided = [r for r in results if r["pass"] is not None]
    passed = sum(1 for r in decided if r["pass"])
    print(f"\n{passed}/{len(decided)} checks pass (pre-registered gate: 4/4 before stage 2)")
    print(GATE_MESSAGES[gate_verdict(results)])
    print(f"\nwritten to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
