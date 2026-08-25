"""P8: the repo's own stated wants — verbatim open-issue titles, no model in the loop.

Five purpose-statement arms all converge at +85..+95 net@2. Every one of them is derived from
the project's *documents*, which describe what it **is**. The rubric's score-3 band asks for
evidence that a paper "directly addresses a known limitation", and no document-derived arm
supplies that. An issue tracker states it outright, in the maintainers' and users' own words.

This is deliberately NOT the failed `improvement_areas` arm (+70, inside the band). Those
were LLM-inferred *and paraphrased*, and §5.3's supported diagnosis was paraphrase-vocabulary
loss — verbatim beat paraphrase by +21. So the one thing this must not do is paraphrase.
Titles go into the prompt exactly as written, including their typos and shouting.

Ranked by reactions rather than recency: a thumbs-up count is the closest free signal to
"people want this" that a tracker exposes, and it is the same quantity across every repo.

    uv run python evals/fetch_wants.py            # ~1 request per repo, free
    uv run python evals/fetch_wants.py --refresh  # re-fetch (trackers move)

Needs an authenticated `gh` (used for the search API's rate limit). Results are cached to
`.work/repo_wants.json`; the triage arm reads only the cache, so a run is reproducible even
if a tracker changes underneath it.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import yaml  # noqa: E402

EVALS = Path(__file__).resolve().parent
WORK = EVALS / ".work"
OUT = WORK / "repo_wants.json"
BENCH = EVALS / "benchmark.yaml"

TOP_N = 15
MAX_TITLE_CHARS = 120


def repo_slug(url: str) -> str | None:
    """https://github.com/pallets/flask -> pallets/flask"""
    if "github.com/" not in url:
        return None
    return url.split("github.com/", 1)[1].strip("/")


def fetch(slug: str) -> list[str]:
    """Top open-issue titles by reactions, verbatim.

    The search endpoint is used rather than `/issues` because only search can order by
    reactions, and reaction count is the ranking this arm is about.
    """
    query = f"repo:{slug} is:issue is:open sort:reactions-desc"
    res = subprocess.run(
        [
            "gh",
            "api",
            "-X",
            "GET",
            "search/issues",
            "-f",
            f"q={query}",
            "-f",
            f"per_page={TOP_N}",
            "--jq",
            ".items[].title",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        # Issue titles are the least ASCII data this project touches, and this arm exists
        # to keep them VERBATIM. Without `errors` a single bad byte leaves stdout unset and
        # the failure lands on `.splitlines()`, three lines below the cause.
        errors="replace",
    )
    if res.returncode != 0:
        raise RuntimeError((res.stderr or "").strip()[:200])
    titles = [t.strip()[:MAX_TITLE_CHARS] for t in res.stdout.splitlines() if t.strip()]
    return titles[:TOP_N]


def as_block(titles: list[str]) -> str:
    """The prompt block. Empty when a tracker is empty — an absent header, never a lying one.

    A `What users are asking for` heading over nothing would read to the model as "this
    project has no open wants", which is a different claim from "nobody has said".
    """
    if not titles:
        return ""
    lines = "\n".join(f"- {t}" for t in titles)
    return (
        f"\nWhat users are asking for, in their own words (open issues, most-reacted):\n{lines}\n"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh", action="store_true", help="re-fetch even if cached")
    args = ap.parse_args()

    bench = yaml.safe_load(BENCH.read_text(encoding="utf-8"))
    cases = {
        c["name"]: c["live_repo"]
        for c in bench["cases"]
        if isinstance(c, dict) and c.get("live_repo")
    }
    cache: dict[str, list[str]] = {}
    if OUT.is_file() and not args.refresh:
        cache = json.loads(OUT.read_text(encoding="utf-8"))

    for case, url in sorted(cases.items()):
        if case in cache and not args.refresh:
            continue
        slug = repo_slug(url)
        if not slug:
            print(f"[{case:10}] not a github url — skipping")
            continue
        try:
            titles = fetch(slug)
        except Exception as exc:  # noqa: BLE001
            print(f"[{case:10}] fetch failed: {exc}")
            continue
        cache[case] = titles
        print(f"[{case:10}] {len(titles):2} wants  e.g. {titles[0][:70] if titles else '(none)'}")
        WORK.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(cache, indent=2), encoding="utf-8")
        time.sleep(2)  # the search endpoint is rate-limited well below the REST one

    have = sum(1 for v in cache.values() if v)
    print(f"\n{have}/{len(cache)} repos have stated wants; written to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
