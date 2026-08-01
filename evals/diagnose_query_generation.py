"""Can an LLM emit search phrases that reach the papers TF-IDF queries miss?

**Costs money** (~$0.01 of Haiku plus 54 arXiv fetches per variant). The answer is already
recorded — this exists so the numbers can be re-derived, not because it needs re-running.

Two prompts, identical in every other respect, both given the repo profile plus its
documentation and never the target papers:

    uv run python evals/diagnose_query_generation.py --prompt uses
    uv run python evals/diagnose_query_generation.py --prompt lacks

Result on 2026-08-01:

    control (current TF-IDF queries)   0/24        --
    --prompt uses                      2/24  (8%)  19/54 phrases matched nothing
    --prompt lacks                     0/24  (0%)  45/54 phrases matched nothing

They fail for opposite reasons. `uses` emits real terms of art that describe what the repo
already implements; the valuable papers describe what it does not. `lacks` aims at the right
classes of work and phrases them as compounds no paper title contains — it emitted
"experience replay prioritization methods" (0 hits) where `uses` emitted "prioritized
experience replay" and found the paper. See `RESULTS.md` -> "Candidate-pool diagnosis",
negative result 2, including why this is not a prompt-tuning problem.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from reporadar.config import SuggestionsConfig  # noqa: E402
from reporadar.llm_client import complete  # noqa: E402
from reporadar.profiler import _collect_text_corpus, profile_repo  # noqa: E402

EVALS = Path(__file__).resolve().parent
WORK = EVALS / ".work"

# The 24 papers the Opus baseline recommended and the judge scored >= 2, which RepoRadar's
# own queries never fetched. Produced by diagnose_pool.py.
TARGETS = {
    "rag": ["2409.14683", "2404.02805", "2501.17788", "2304.01982", "2505.11471"],
    "cv": ["1704.04503", "2012.07177", "2201.03545"],
    "rl": ["1509.06461", "1511.05952", "2110.02034"],
    "peft": ["2404.03592", "2405.12130"],
    "diffusion": ["2410.05317", "2508.16211"],
    "graph": ["2202.13013", "2303.06147", "2111.14522"],
    "speech": ["2303.00747", "2211.17192", "2311.00430"],
    "crypto": ["1812.04959", "2405.18993"],
    "systems": ["1512.00727"],
}

_SHARED_RULES = """\
- 2-5 words each. Real terms of art only, no invented-sounding phrases.
- Use the terminology a PAPER TITLE would use.
- Favour method names over topic names.

Return ONLY a JSON array of {n} strings. No prose.

# Repository
Dependencies: {anchors}
Domains: {domains}
Key topics: {keywords}

# Documentation excerpt
{docs}
"""

PROMPTS = {
    "uses": "Emit {n} search phrases naming the techniques, algorithms and methods this "
    "codebase implements or would plausibly adopt.\n" + _SHARED_RULES,
    "lacks": "Think about what this codebase does NOT do well: performance and latency "
    "bottlenecks, accuracy or robustness limits, components a newer method has superseded, "
    "capabilities a competitor would have. Emit {n} search phrases naming the RESEARCH "
    "AREAS that address those gaps.\n"
    "- Name what would IMPROVE the codebase, NOT what it already implements.\n" + _SHARED_RULES,
}

_ID_RE = re.compile(r"<id>https?://arxiv\.org/abs/([^<v]+)")


def _load_env() -> None:
    env = EVALS / ".env"
    if not env.is_file():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def arxiv_ids(query: str, max_results: int = 50, tries: int = 4) -> list[str]:
    url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
        {"search_query": query, "start": 0, "max_results": max_results, "sortBy": "relevance"}
    )
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "reporadar-diagnostic/1.0"})
            with urllib.request.urlopen(req, timeout=90) as resp:
                return _ID_RE.findall(resp.read().decode("utf-8", "replace"))
        except Exception as exc:  # noqa: BLE001
            wait = 30 * (attempt + 1)
            print(f"      ! {type(exc).__name__}; backing off {wait}s", flush=True)
            time.sleep(wait)
    return []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompt", choices=sorted(PROMPTS), required=True)
    ap.add_argument("--phrases", type=int, default=6)
    ap.add_argument("--model", default="claude-haiku-4-5")
    args = ap.parse_args()

    _load_env()
    cfg = SuggestionsConfig(provider="claude", claude_model=args.model, timeout=60)
    out: dict[str, dict] = {}
    dead = total = 0

    for case, targets in TARGETS.items():
        repo = WORK / case
        if not repo.is_dir():
            continue
        profile = profile_repo(repo)
        prompt = PROMPTS[args.prompt].format(
            n=args.phrases,
            anchors=", ".join(profile.anchors[:15]) or "none",
            domains=", ".join(profile.domains[:5]) or "general",
            keywords=", ".join(t for t, _ in profile.keywords[:15]),
            docs=" ".join(_collect_text_corpus(repo))[:6000],
        )
        try:
            raw = complete(prompt, cfg, max_tokens=400)
            phrases = json.loads(re.search(r"\[.*\]", raw, re.DOTALL).group(0))
        except Exception as exc:  # noqa: BLE001
            print(f"[{case}] LLM failed: {exc}")
            continue
        phrases = [p for p in phrases if isinstance(p, str)][: args.phrases]
        print(f"\n[{case}] emitted: {phrases}", flush=True)

        found: set[str] = set()
        for phrase in phrases:
            ids = arxiv_ids(f'all:"{phrase}"')
            total += 1
            dead += 1 if not ids else 0
            hits = [t for t in targets if t in ids]
            found |= set(hits)
            mark = "  <-- " + ",".join(hits) if hits else ""
            print(f'      all:"{phrase}"'.ljust(58) + f"{len(ids):3d} hits{mark}", flush=True)
            time.sleep(5)  # arXiv blocks on sustained rate, not page size
        out[case] = {"phrases": phrases, "targets": targets, "found": sorted(found)}
        print(f"      => {len(found)}/{len(targets)} recovered", flush=True)

    n_targets = sum(len(v["targets"]) for v in out.values())
    n_found = sum(len(v["found"]) for v in out.values())
    print(f"\n=== --prompt {args.prompt} ===")
    print(f"recovered: {n_found}/{n_targets}  ({n_found / max(n_targets, 1):.0%})")
    print(f"phrases matching nothing: {dead}/{total}")
    print("\nrecorded 2026-08-01:  control 0/24 · uses 2/24 (19/54 dead) · lacks 0/24 (45/54 dead)")
    (WORK / f"diag_queries_{args.prompt}.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
