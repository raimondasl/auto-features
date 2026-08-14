"""Build frozen, real-arXiv fixtures for the offline eval benchmark.

Run this once (it hits the arXiv API) to (re)generate evals/fixtures/<case>.json.
The offline benchmark then runs fully deterministically against these frozen
files with no network access and no API keys.

    uv run python evals/build_fixtures.py            # all cases
    uv run python evals/build_fixtures.py --case rag # one case

Gold papers (label 1) come from the case's own domain queries; distractors
(label 0) come from clearly different fields.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import arxiv

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import EVALS_DIR, load_benchmark  # noqa: E402  (needs evals/ on sys.path first)

from reporadar.paper_id import dedup_id  # noqa: E402

FIXTURES_DIR = EVALS_DIR / "fixtures"


def _normalize(result: arxiv.Result, label: int, query: str) -> dict[str, Any]:
    return {
        "arxiv_id": result.get_short_id(),
        "title": result.title.strip().replace("\n", " "),
        "abstract": result.summary.strip().replace("\n", " "),
        "categories": list(result.categories),
        "published": result.published.isoformat() if result.published else "",
        "authors": [a.name for a in result.authors],
        "url": result.entry_id,
        "label": label,
        "source_query": query,
    }


def _fetch(client: arxiv.Client, query: str, n: int, label: int) -> list[dict[str, Any]]:
    if n <= 0:
        return []
    search = arxiv.Search(
        query=query,
        max_results=n,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    papers = []
    try:
        for result in client.results(search):
            papers.append(_normalize(result, label, query))
    except Exception as exc:  # noqa: BLE001 — network/parse errors are non-fatal per query
        print(f"    ! query {query!r} failed: {exc}", file=sys.stderr)
    return papers


def build_case(client: arxiv.Client, case: dict[str, Any]) -> list[dict[str, Any]]:
    name = case["name"]
    print(f"[{name}] fetching...")
    pool: dict[str, dict[str, Any]] = {}

    # Gold first so a paper matching both gold and distractor stays labeled gold.
    per_gold = max(1, case.get("gold_n", 0) // max(1, len(case.get("gold_queries", []) or [1])))
    for q in case.get("gold_queries", []) or []:
        for p in _fetch(client, q, per_gold + 2, label=1):
            pool.setdefault(dedup_id(p["arxiv_id"]), p)

    gold_ids = set(pool.keys())
    per_dist = max(
        1, case.get("distractor_n", 0) // max(1, len(case.get("distractor_queries", []) or [1]))
    )
    for q in case.get("distractor_queries", []) or []:
        for p in _fetch(client, q, per_dist + 2, label=0):
            base = dedup_id(p["arxiv_id"])
            if base in gold_ids:
                continue  # ambiguous — skip to keep labels clean
            pool.setdefault(base, p)

    papers = list(pool.values())
    n_gold = sum(1 for p in papers if p["label"] == 1)
    n_dist = sum(1 for p in papers if p["label"] == 0)
    print(f"[{name}] {n_gold} gold + {n_dist} distractors = {len(papers)} papers")
    return papers


def main() -> None:
    parser = argparse.ArgumentParser(description="Build offline eval fixtures from arXiv.")
    parser.add_argument("--case", help="Only build this case (default: all).")
    args = parser.parse_args()

    bench = load_benchmark()
    FIXTURES_DIR.mkdir(exist_ok=True)
    client = arxiv.Client(page_size=50, delay_seconds=3.0, num_retries=3)

    for case in bench["cases"]:
        if args.case and case["name"] != args.case:
            continue
        papers = build_case(client, case)
        out = FIXTURES_DIR / f"{case['name']}.json"
        out.write_text(json.dumps(papers, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[{case['name']}] wrote {out.relative_to(EVALS_DIR.parent)}")


if __name__ == "__main__":
    main()
