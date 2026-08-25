"""Stage-1 yield probe, $0: does a peS2o-backed corpus hold the off-arXiv value?

RepoRadar's dense channel searches an arXiv-only index. The repos it still loses on --
and the whole `bio-*`/`mat-*` cohort -- have literature that is not on arXiv, so the
standing question is whether a second, wider corpus is worth building. OpenScholar
(arXiv:2411.14199) released the largest open one: `OpenSciLM/OpenScholar-DataStore-V3`,
peS2o v3, MIT-licensed.

**The obvious version of this probe is refuted, and that is why this one exists.** The
first design asked whether the gold targets the shipped channels never reach are in that
datastore. They are already in *ours*: all 56 gold targets are present in the local 3.1M
index, at ranks up to 223,245. Those are **ranking** failures, not coverage failures, and
adding 45M documents to a corpus that already holds 100% of the targets can only add
competitors. That check would have returned "6/6 present" and meant nothing.

**Why the gold set cannot ask the real question.** A gold target is a baseline pick judged
>= 2, and `baseline.py`'s prompt demands `{"arxiv_id": ...}` -- so a non-arXiv paper can
never become one. Yet the benchmark plainly surfaces off-arXiv value: 159 non-arXiv papers
judged, **79 actionable**, and 11 cases (every `bio-*` and `mat-*`) have actionable
non-arXiv papers and *no* gold targets at all. `net@2` counts them; gold-target recall
cannot see them.

So this probe uses those 79 as ground truth. They are free, already judged, need no
baseline run, and carry no arXiv restriction -- the NR-34 stage-1 yield pattern rather than
the gold-target recall pattern.

**What a PASS would cost, stated up front.** The datastore is ~378 GB of passages against
the arXiv index's 432 MB (~875x), and its precomputed vectors come from OpenScholar's own
retriever, not `mxbai-embed-large-v1`. Adopting it means **re-embedding the corpus**, not
syncing an index -- so the Design 2 "reproduce a stored vector bit-identically" check does
not apply here; there are no vectors of ours to reproduce.

PRE-REGISTERED, written before the first run (evals/RESULTS.md keeps the record):

  Primary: the share of the 79 actionable non-arXiv papers that peS2o v3 would contain.
  Rule: open access, in S2ORC, published on or before the **2024-10** cutoff.

  I predict **50-70%**. bioRxiv preprints (57 of 79 are DOIs, mostly 10.1101) are open
  access and largely in S2ORC; the closed-access chemistry/materials journals in the
  `mat-*` cohort (10.1021, 10.1016, 10.1063) are excluded from peS2o by construction, and
  any 2025-or-later paper is past the cutoff.

  **KILL: below 50%.** At 875x the build cost and a full re-embed, a corpus that cannot
  hold half the known off-arXiv value is not worth building. 50-70% is **marginal**, not a
  pass: the value would be real but better source adapters would likely be the cheaper
  route to it. Only >= 70% justifies the next step (a rank probe, then a live net@2 arm).

    uv run python evals/openscholar_yield.py            # ~1 min, S2 batch, no LLM
    uv run python evals/openscholar_yield.py --report   # $0, cached resolution only
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from reporadar.paper_id import is_arxiv_id  # noqa: E402

EVALS = Path(__file__).resolve().parent
WORK = EVALS / ".work"
JUDGE = EVALS / "cache" / "judge" / "v1" / "gpt-5.5"
CACHE = WORK / "openscholar_yield.json"

# peS2o v3 covers papers up to this date (OpenScholar paper, §datastore).
PES2O_CUTOFF = "2024-10-31"
ACTIONABLE = 2
KILL_FRACTION = 0.50
PASS_FRACTION = 0.70


# Judge-cache filenames encode old-style arXiv ids with `_` where the id has `/`
# (`cs_0412098` for `cs/0412098`), because `/` cannot appear in a filename. Restore the
# separator and ask the SHARED rule -- `audit_product_divergence.py` fails the build on a
# hand-rolled `split("v")[0]`, and this module is the reason that guard exists (C-12).
def is_arxiv(stem: str) -> bool:
    """True if this judge-cache filename names an arXiv paper, in either id era."""
    restored = re.sub(r"^([a-z-]+(?:\.[A-Za-z-]+)?)_(\d{7})", r"\1/\2", stem)
    return is_arxiv_id(stem) or is_arxiv_id(restored)


S2_BATCH = "https://api.semanticscholar.org/graph/v1/paper/batch"
S2_FIELDS = "externalIds,title,isOpenAccess,openAccessPdf,publicationDate,year,corpusId"
# RESEARCH.md §3.4: sustained polling earned this machine a ~70-minute IP block. Request
# RATE is the lever. The batch endpoint is one request per 100 ids rather than one per id.
BATCH_SIZE = 100
BATCH_SLEEP = 3.0


def actionable_non_arxiv() -> list[dict[str, str]]:
    """Judged-actionable papers that did NOT come from arXiv -- the probe's ground truth."""
    out: list[dict[str, str]] = []
    for case_dir in sorted(p for p in JUDGE.iterdir() if p.is_dir()):
        for f in case_dir.glob("*.json"):
            stem = f.stem
            if is_arxiv(stem):
                continue
            try:
                verdict = json.loads(f.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if int(verdict.get("score", 0)) >= ACTIONABLE:
                out.append({"case": case_dir.name, "id": stem})
    return out


def s2_query_id(paper_id: str) -> str | None:
    """The id string the S2 graph API accepts, or None if we cannot address it."""
    if paper_id.startswith("doi_"):
        return "DOI:" + paper_id[4:].replace("_", "/", 1)
    if paper_id.startswith("ss_"):
        return paper_id[3:]
    return None  # e.g. iacr_* -- no S2 handle without a title lookup


def resolve(ids: list[str]) -> dict[str, dict[str, Any]]:
    """S2 batch lookup. Flat fields only, so the §3.5 nested-truncation defect cannot bite."""
    key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
    out: dict[str, dict[str, Any]] = {}
    for start in range(0, len(ids), BATCH_SIZE):
        chunk = ids[start : start + BATCH_SIZE]
        body = json.dumps({"ids": chunk}).encode()
        headers = {"Content-Type": "application/json"}
        if key:
            headers["x-api-key"] = key
        req = urllib.request.Request(f"{S2_BATCH}?fields={S2_FIELDS}", data=body, headers=headers)
        for attempt in range(4):
            try:
                data = json.loads(urllib.request.urlopen(req, timeout=90).read())
                break
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                if attempt == 3:
                    print(f"  batch {start // BATCH_SIZE} FAILED after 4 tries: {exc}")
                    data = [None] * len(chunk)
                    break
                time.sleep(5 * (attempt + 1))
        # S2 returns null for ids it cannot resolve, positionally aligned with the request.
        for qid, rec in zip(chunk, data, strict=False):
            out[qid] = rec or {}
        print(f"  resolved {min(start + BATCH_SIZE, len(ids))}/{len(ids)}")
        time.sleep(BATCH_SLEEP)
    return out


def in_pes2o(rec: dict[str, Any]) -> tuple[bool, str]:
    """peS2o v3 inclusion: open access, in S2ORC, published by the cutoff."""
    if not rec:
        return False, "unresolved by S2"
    if not rec.get("isOpenAccess") and not rec.get("openAccessPdf"):
        return False, "not open access"
    date = rec.get("publicationDate") or (f"{rec['year']}-12-31" if rec.get("year") else "")
    if not date:
        return False, "no publication date"
    if date > PES2O_CUTOFF:
        return False, f"after cutoff ({date})"
    return True, "included"


def spot_check(corpus_ids: list[str], limit: int = 5) -> list[tuple[str, str]]:
    """Confirm a few corpus ids really appear as `raw_id` in the datastore itself.

    The S2 rule above is peS2o's stated inclusion criterion, not a lookup in the artifact.
    This checks the artifact directly for a handful, so the primary number is anchored to
    something observed rather than only inferred.
    """
    results: list[tuple[str, str]] = []
    for cid in corpus_ids[:limit]:
        params = urllib.parse.urlencode(
            {
                "dataset": "OpenSciLM/OpenScholar-DataStore-V3",
                "config": "default",
                "split": "train",
                "where": f"\"raw_id\"='{cid}'",
                "limit": "1",
            }
        )
        try:
            raw = urllib.request.urlopen(
                f"https://datasets-server.huggingface.co/filter?{params}", timeout=90
            ).read()
            payload = json.loads(raw)
            n = payload.get("num_rows_total")
            results.append((cid, f"{n} passage(s)" if n else "not found in indexed shard"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            results.append((cid, f"server unavailable ({type(exc).__name__})"))
        time.sleep(1.0)
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description="peS2o/OpenScholar yield probe")
    ap.add_argument("--report", action="store_true", help="$0: use cached resolution")
    ap.add_argument("--no-spot-check", action="store_true", help="skip the HF datastore calls")
    args = ap.parse_args()

    papers = actionable_non_arxiv()
    if not papers:
        raise SystemExit(
            "no actionable non-arXiv papers in the judge cache: the extractor is broken, "
            "not the corpus. Reporting 0% here would be void read as null."
        )

    queryable = {p["id"]: s2_query_id(p["id"]) for p in papers}
    unaddressable = [pid for pid, q in queryable.items() if q is None]

    if args.report and CACHE.is_file():
        resolved = json.loads(CACHE.read_text(encoding="utf-8"))
    elif args.report:
        raise SystemExit(f"no cache at {CACHE}; run without --report first")
    else:
        todo = sorted({q for q in queryable.values() if q})
        print(f"Resolving {len(todo)} papers through the S2 batch endpoint...\n")
        resolved = resolve(todo)
        CACHE.write_text(json.dumps(resolved, indent=1), encoding="utf-8")

    rows = []
    for p in papers:
        q = queryable[p["id"]]
        rec = resolved.get(q, {}) if q else {}
        ok, why = (False, "no S2 handle (IACR)") if q is None else in_pes2o(rec)
        rows.append({**p, "query": q, "ok": ok, "why": why, "corpus": rec.get("corpusId")})

    n = len(rows)
    included = [r for r in rows if r["ok"]]
    frac = len(included) / n

    print("\nOpenScholar / peS2o v3 yield probe\n")
    print(f"  ground truth: {n} judged-actionable NON-arXiv papers")
    print("  (gold-target recall cannot see these: baseline.py demands an arxiv_id)\n")
    print(f"  would be in peS2o v3: {len(included)}/{n} = {100 * frac:.1f}%\n")

    print("  exclusions by reason:")
    for why, count in Counter(r["why"] for r in rows if not r["ok"]).most_common():
        print(f"    {why:28} {count:>3}")
    print()

    print(f"  {'cohort':16} {'n':>4} {'in peS2o':>9} {'share':>7}")
    by_case: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_case.setdefault(r["case"], []).append(r)
    for group, prefix in (("bio-*", "bio-"), ("mat-*", "mat-"), ("other", None)):
        sub = [
            r
            for c, rs in by_case.items()
            for r in rs
            if (c.startswith(prefix) if prefix else not c.startswith(("bio-", "mat-")))
        ]
        if sub:
            k = sum(1 for r in sub if r["ok"])
            print(f"  {group:16} {len(sub):>4} {k:>9} {100 * k / len(sub):>6.1f}%")
    print()

    if not args.no_spot_check:
        cids = [str(r["corpus"]) for r in included if r.get("corpus")]
        if cids:
            print("  spot-check against the datastore's own raw_id (HF datasets-server):")
            for cid, note in spot_check(cids):
                print(f"    corpusId {cid:>12}  {note}")
            print("    (the server indexes one shard of sixteen, so a miss is not an absence)")
            print()

    verdict = "PASS" if frac >= PASS_FRACTION else ("KILL" if frac < KILL_FRACTION else "MARGINAL")
    print(
        f"  PRE-REGISTERED: predicted 50-70%, kill below {100 * KILL_FRACTION:.0f}%, "
        f"pass at {100 * PASS_FRACTION:.0f}%"
    )
    print(f"  VERDICT: {verdict} at {100 * frac:.1f}%")
    if unaddressable:
        print(
            f"\n  note: {len(unaddressable)} paper(s) have no S2 handle and count as absent: "
            f"{unaddressable}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
