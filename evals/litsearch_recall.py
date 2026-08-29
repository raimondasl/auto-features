"""A recall-fidelity gauge for the dense index, from LitSearch. [PLANS item 4]

The binary-quantized arXiv index is the substrate under HyDE, which is the project's biggest
converted change (+1.36 net@2 end to end, its first p < 0.05). `verify_encoder` proves our
vectors reproduce the published ones bit-for-bit, so the two are comparable -- but nothing
today would notice if binarisation, column pruning, a bad yearly shard or an encoder swap
silently cost fifteen points of recall. The index is verified for *identity* and unmeasured
for *usefulness*.

LitSearch (arXiv:2407.18940) supplies 597 gold-labelled literature-search questions over
recent ML/NLP -- squarely inside this index's coverage. This embeds them with the shipped
encoder, searches the shipped index through the shipped `hyde.search_index`, and freezes
recall@5/@20/@100 so a later run can be compared against it.

**This is not a net@2 claim and must never be quoted as one.** A researcher asking "where can
I find work on X" is a different register from a repository that needs a paper to act on;
§5's register-mismatch finding is precisely that the two do not transfer. The number here
answers one question -- *does the index still retrieve what it retrieved before* -- and is
useless for any other.

Three things the measurement has to get right, each of which would otherwise turn a fidelity
gauge into something else:

* **An unanswerable query is not a miss.** LitSearch's corpus is S2ORC; ours is arXiv. 458 of
  its 574 gold papers carry an arXiv id and **456 of 456 distinct ones are in our index**, so
  498 of 597 queries are answerable. Scoring the other 99 as failures would measure corpus
  overlap, drift every time arXiv grows, and hide the thing being gauged. They are excluded
  and counted, never scored -- the void-not-null rule this project keeps paying to relearn.
* **The encoder is verified before anything is embedded.** If our vectors stopped matching
  the index, every query would return confident nonsense and the recall figure would drop --
  correctly, but for a reason a recall number cannot name. `verify_encoder` runs first and the
  measurement refuses to proceed without it, so a fidelity failure is reported as one.
* **Both query forms are measured.** `mxbai-embed-large-v1` is asymmetric: documents bare,
  queries behind "Represent this sentence for searching relevant passages: ". The index holds
  bare abstracts, so the prefix belongs on our side or nowhere -- and which of the two is a
  measurement rather than a preference.

    uv run python evals/litsearch_recall.py --build      # 50 KB + one S2 batch, $0
    uv run python evals/litsearch_recall.py --limit 60   # pilot, ~5 min
    uv run python evals/litsearch_recall.py              # full, ~50 min of CPU
    uv run python evals/litsearch_recall.py --check      # $0 gate; exits non-zero on a drop
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

EVALS = Path(__file__).resolve().parent
sys.path.insert(0, str(EVALS.parent / "src"))

from reporadar import hyde  # noqa: E402
from reporadar.paper_id import dedup_id  # noqa: E402

WORK = EVALS / ".work" / "litsearch"
INDEX = EVALS / ".work" / "hyde_index"
FROZEN = EVALS / "litsearch_recall.json"
QUERY_PARQUET = WORK / "query.parquet"
MAPPING = WORK / "corpusid_to_arxiv.json"

# The `query` config only: 50 KB. `corpus_clean` is 1.26 GB and `corpus_s2orc` 1.6 GB, and
# neither is needed -- we search OUR index, so all we want from LitSearch is the questions
# and which papers answer them. The corpusid -> arXiv mapping comes from the S2 batch API
# this repository already wraps, which is 574 ids in two requests.
QUERY_URL = (
    "https://huggingface.co/datasets/princeton-nlp/LitSearch/resolve/"
    "refs%2Fconvert%2Fparquet/query/full/0000.parquet"
)
PREFIX = "Represent this sentence for searching relevant passages: "
ARMS = {"bare": "", "prefixed": PREFIX}
KS = (5, 20, 100)
TOP_K = max(KS)

# "recall@k" here is: does at least one gold paper appear in the top k. For the 563 of 597
# queries with a single gold paper that IS recall, and it is how LitSearch's own paper reports
# it; for the 34 with several it is a hit rate, which is the more forgiving of the two. Named
# in full because a metric whose definition shifts with the row is how a gauge stops being
# comparable to itself.

# How far recall may fall before `--check` fails. The gauge exists to catch a fifteen-point
# collapse, not to police the third decimal: the index gains a shard every month and the
# denominator shifts slightly with it.
TOLERANCE = 0.05


def build() -> None:
    """Fetch the 597 queries and resolve their gold papers to arXiv ids. Idempotent."""
    WORK.mkdir(parents=True, exist_ok=True)
    if not QUERY_PARQUET.is_file():
        req = urllib.request.Request(QUERY_URL, headers={"User-Agent": "reporadar-eval"})
        with urllib.request.urlopen(req, timeout=120) as r:
            QUERY_PARQUET.write_bytes(r.read())
        print(f"fetched {QUERY_PARQUET.name} ({QUERY_PARQUET.stat().st_size} bytes)")
    if MAPPING.is_file():
        print(f"{MAPPING.name} present; not re-resolving")
        return

    from dotenv import load_dotenv

    from reporadar.citations import _s2_batch_post

    load_dotenv(EVALS / ".env")
    rows = load_queries()
    gold = sorted({c for r in rows for c in r["corpusids"]})
    key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY") or None
    mapping: dict[str, str | None] = {}
    for i in range(0, len(gold), 500):
        chunk = gold[i : i + 500]
        data = _s2_batch_post([f"CorpusId:{c}" for c in chunk], "externalIds", key, 4, 2.0)
        if data is None:
            raise SystemExit(f"S2 batch {i // 500} failed; re-run --build when it clears")
        for cid, rec in zip(chunk, data, strict=True):
            mapping[str(cid)] = ((rec or {}).get("externalIds") or {}).get("ArXiv")
    MAPPING.write_text(json.dumps(mapping, indent=0), encoding="utf-8")
    got = sum(1 for v in mapping.values() if v)
    print(f"resolved {len(mapping)} corpusids; {got} carry an arXiv id")


def load_queries() -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    return pq.read_table(QUERY_PARQUET).to_pylist()


def index_ids() -> set[str]:
    ids: set[str] = set()
    for f in sorted(INDEX.glob("*.ids")):
        ids.update(dedup_id(i) for i in f.read_text(encoding="utf-8").split("\n") if i)
    return ids


def answerable(rows, mapping, held) -> list[dict[str, Any]]:
    """Queries with at least one gold paper the index could return.

    The rest are not failures and are not scored. A query whose answer is not in the corpus
    is an unanswerable question; counting it as a miss would make this a coverage measure
    that drifts every time arXiv grows -- and would mask the fidelity signal it exists for.
    """
    out = []
    for r in rows:
        gold = {
            dedup_id(mapping[str(c)])
            for c in r["corpusids"]
            if mapping.get(str(c)) and dedup_id(mapping[str(c)]) in held
        }
        if gold:
            out.append({**r, "gold_arxiv": sorted(gold)})
    return out


def measure(limit: int | None) -> dict[str, Any]:
    rows = load_queries()
    mapping = json.loads(MAPPING.read_text(encoding="utf-8"))
    held = index_ids()
    scored = answerable(rows, mapping, held)
    if limit:
        scored = scored[:limit]

    model = hyde.load_encoder()
    ok, dists = hyde.verify_encoder(model)
    if not ok:
        raise SystemExit(
            f"encoder does not reproduce the index (Hamming {dists}, expected all 0). "
            "Recall would drop for a reason recall cannot name; refusing to measure."
        )

    out: dict[str, Any] = {
        "_comment": (
            "Recall of the shipped dense index on LitSearch (arXiv:2407.18940), as a "
            "REGRESSION GAUGE for the index and nothing else. Derived by "
            "evals/litsearch_recall.py; pinned by tests/test_litsearch_recall.py. Not a net@2 "
            "claim: researcher questions are a different register from repo->paper, which is "
            "the register-mismatch finding of the paper's section 5."
        ),
        "dataset": "princeton-nlp/LitSearch, query config, 597 rows",
        "encoder": hyde.MODEL_NAME,
        "encoder_verified": True,
        "encoder_hamming": dists,
        "index": {
            "shards": len(hyde.index_shards(INDEX)),
            "ids": len(held),
        },
        "coverage": {
            "queries_total": len(rows),
            "queries_answerable": len(answerable(rows, mapping, held)),
            "gold_corpusids": len({c for r in rows for c in r["corpusids"]}),
            "gold_with_arxiv_id": sum(1 for v in mapping.values() if v),
            "gold_in_index": len(
                {dedup_id(v) for v in mapping.values() if v and dedup_id(v) in held}
            ),
        },
        "n_scored": len(scored),
        "arms": {},
    }

    # Both arms in ONE pass, so the two are paired on identical queries by construction and
    # the per-query rank of each is recorded. The first version reported two aggregate
    # recalls and nothing else, which cannot answer whether a ten-query difference between
    # them is real -- and a gauge that cannot say which queries moved is a scalar with no
    # diagnosis attached.
    t0 = time.time()
    per_query: list[dict[str, Any]] = []
    for n, row in enumerate(scored, start=1):
        gold = set(row["gold_arxiv"])
        rec: dict[str, Any] = {
            "query_set": row["query_set"],
            "specificity": row["specificity"],
            "quality": row["quality"],
            "n_gold": len(gold),
        }
        for arm, prefix in ARMS.items():
            bits = hyde.encode_binary(model, [prefix + row["query"]])
            hits = hyde.search_index(INDEX, bits, top_k=TOP_K)
            found = [i for i, pid in enumerate(hits) if dedup_id(pid) in gold]
            # -1 is "not in the top 100", which is NOT rank 101: the search was capped, so
            # the true rank is unknown rather than large. Every consumer below treats it as
            # a miss and none of them averages it.
            rec[f"rank_{arm}"] = found[0] if found else -1
        per_query.append(rec)
        if n % 50 == 0:
            print(f"  {n}/{len(scored)}  {time.time() - t0:.0f}s")

    def hit(rank: int, k: int) -> bool:
        return 0 <= rank < k

    for arm, prefix in ARMS.items():
        ranks = [r[f"rank_{arm}"] for r in per_query]
        found_ranks = sorted(r for r in ranks if r >= 0)
        out["arms"][arm] = {
            "query_prefix": prefix or None,
            **{f"recall_at_{k}": round(sum(hit(r, k) for r in ranks) / len(ranks), 4) for k in KS},
            "found_in_top_100": len(found_ranks),
            "median_rank_when_found": (found_ranks[len(found_ranks) // 2] if found_ranks else None),
        }
        got = out["arms"][arm]
        print(f"  [{arm}] " + "  ".join(f"R@{k}={got[f'recall_at_{k}']:.3f}" for k in KS))

    # Paired McNemar between the arms: of the queries where the two disagree, how lopsided
    # is the split? Comparing two aggregate rates would ignore that they answer the same
    # 498 questions, which is the whole reason they are run together.
    out["prefix_comparison"] = {
        "_comment": (
            "Exact McNemar on the discordant pairs. The prefix is the documented retrieval "
            "form for this encoder; the index holds bare abstracts, so it belongs on the "
            "query side or nowhere. This is the measurement that decides, and at k=100 it "
            "decides against."
        )
    }
    for k in KS:
        b = [hit(r["rank_bare"], k) for r in per_query]
        p_ = [hit(r["rank_prefixed"], k) for r in per_query]
        only_bare = sum(1 for x, y in zip(b, p_, strict=True) if x and not y)
        only_pref = sum(1 for x, y in zip(b, p_, strict=True) if y and not x)
        n = only_bare + only_pref
        p_value = (
            min(
                1.0,
                2 * sum(math.comb(n, i) for i in range(min(only_bare, only_pref) + 1)) / 2**n,
            )
            if n
            else 1.0
        )
        out["prefix_comparison"][f"at_{k}"] = {
            "only_bare": only_bare,
            "only_prefixed": only_pref,
            "discordant": n,
            "p_value": round(p_value, 4),
            "significant_at_05": p_value < 0.05,
        }
        print(f"  prefix @{k}: bare-only {only_bare}, prefixed-only {only_pref}, p={p_value:.4f}")

    out["per_query"] = per_query
    out["seconds"] = round(time.time() - t0, 1)
    return out


def check() -> int:
    """$0. Compare a fresh measurement against the frozen one and fail on a real drop."""
    if not FROZEN.is_file():
        print(f"! {FROZEN.name} absent — run without --check first.")
        return 1
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    fresh = measure(None)
    bad = []
    for arm, v in frozen["arms"].items():
        for k in KS:
            was, now = v[f"recall_at_{k}"], fresh["arms"][arm][f"recall_at_{k}"]
            print(f"  {arm:<9} R@{k:<3} frozen {was:.4f}  now {now:.4f}  {now - was:+.4f}")
            if was - now > TOLERANCE:
                bad.append(f"{arm} R@{k}: {was:.4f} -> {now:.4f}")
    if bad:
        print("\n! dense-index recall regressed:\n  " + "\n  ".join(bad))
        return 1
    print(f"\nindex recall within {TOLERANCE:.0%} of the frozen values.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--build", action="store_true", help="$0: fetch queries, resolve gold ids.")
    ap.add_argument("--check", action="store_true", help="Gate: fail on a recall drop.")
    ap.add_argument("--limit", type=int, help="Score only the first N queries (a pilot).")
    args = ap.parse_args()

    if args.build:
        build()
        return 0
    if not QUERY_PARQUET.is_file() or not MAPPING.is_file():
        print("! run --build first (50 KB + one S2 batch, $0).")
        return 1
    if args.check:
        return check()

    out = measure(args.limit)
    if args.limit:
        print(f"\n(pilot over {args.limit} queries; artifact NOT written)")
        print(json.dumps(out["arms"], indent=1))
        return 0
    FROZEN.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"\nwrote {FROZEN.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
