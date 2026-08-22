"""Do the excluded repositories still profile badly? ($0, judge-free, no network beyond git.)

    uv run python evals/blindspot_profiles.py

§14.2 excluded htslib, kallisto, tblite, kim-api and LAMMPS from cohort 3 — the compiled,
manifest-less population — and §15.6 records that as the standing caveat on every published
figure: *"this cohort is optimistic about scientific software in general."* §14.11 adds the
ecosystems that have never been profiled at all: R, Julia, Rust and Nextflow.

**The specific claims about them are stale, and that is why this is worth running.** §4 recorded
that `kallisto` loses its own name to `__kallisto__`, that `htslib` draws the citation id
`giab007` as a query, and that tblite/kim-api/LAMMPS are blind-spot exhibits with `doc/` unread.
Every one of those observations predates §10, which added `doc/` reading, release-note exclusion,
`setup.cfg` parsing, and MyST/badge stripping — **not** `environment.yml`, which §10.2 dropped
along with the R, Julia and Rust parsers. §10.4 says plainly that every
published number describes the pre-2026-08-19 profiler; §11 re-measured the 25 benchmark repos
and **these five were never re-profiled**.

So this asks a cheap, judge-free, falsifiable question: **which of §4's blind-spot claims survive
the current profiler?** A profile is local and free — no judge, no API, no paper pool. Nothing
here measures digest quality; it measures whether the pipeline can describe these repositories at
all, which is the precondition for anything else.

Clones are shallow and go to `.work/blindspot/`, kept apart from `.work/<case>` so a benchmark
case can never accidentally read one.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

EVALS = Path(__file__).resolve().parent
sys.path.insert(0, str(EVALS))
sys.path.insert(0, str(EVALS.parent / "src"))

from harness import WORK_DIR, clone_repo  # noqa: E402

from reporadar.config import ProfilerConfig  # noqa: E402
from reporadar.profiler import profile_repo  # noqa: E402

DEST = WORK_DIR / "blindspot"

# §14.2's five, plus one representative of each ecosystem §14.11 says has never been profiled.
# The `claim` is what §4 recorded about it under the OLD profiler, so each row is a testable
# statement rather than a vibe.
REPOS: tuple[tuple[str, str, str, str], ...] = (
    (
        "htslib",
        "https://github.com/samtools/htslib",
        "C, no manifest",
        "draws `giab007` as a query",
    ),
    (
        "kallisto",
        "https://github.com/pachterlab/kallisto",
        "C++, no manifest",
        "loses its own name to `__kallisto__`",
    ),
    ("tblite", "https://github.com/tblite/tblite", "Fortran", "blind-spot exhibit, doc/ unread"),
    (
        "kim-api",
        "https://github.com/openkim/kim-api",
        "C++/Fortran",
        "blind-spot exhibit, doc/ unread",
    ),
    ("lammps", "https://github.com/lammps/lammps", "C++, huge", "blind-spot exhibit, doc/ unread"),
    ("seurat", "https://github.com/satijalab/seurat", "R / CRAN", "never profiled"),
    ("diffeq-jl", "https://github.com/SciML/DifferentialEquations.jl", "Julia", "never profiled"),
    ("noodles", "https://github.com/zaeleus/noodles", "Rust", "never profiled"),
    ("nf-core-rnaseq", "https://github.com/nf-core/rnaseq", "Nextflow", "never profiled"),
)
PROSE_CHARS = 300  # the shipped default and what every benchmark run used


def summarise(name: str, repo: Path) -> dict[str, Any]:
    profile = profile_repo(repo, profiler_cfg=ProfilerConfig(prose_chars=PROSE_CHARS))
    kws = [k for k, _w in profile.keywords]
    return {
        "name": name,
        "keywords": kws,
        "n_keywords": len(kws),
        "anchors": list(profile.anchors),
        "domains": list(profile.domains),
        "prose": (profile.prose or "")[:200],
        # The single most load-bearing question: can the repo's own name be used as a query?
        # §5 established that a repository whose name is a good query is the strongest single
        # predictor of a usable digest.
        "own_name_in_keywords": any(name.split("-")[0].lower() in k.lower() for k in kws),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default="", help="comma-separated subset")
    args = ap.parse_args()
    wanted = {s.strip() for s in args.only.split(",") if s.strip()}
    rows: list[dict[str, Any]] = []

    for name, url, kind, claim in REPOS:
        if wanted and name not in wanted:
            continue
        dest = clone_repo(url, DEST / name)
        if dest is None:
            print(f"  {name:16} CLONE FAILED — excluded, never reported as an empty profile")
            continue
        try:
            row = summarise(name, dest)
        except Exception as exc:  # noqa: BLE001 - a profiler crash IS the finding here
            print(f"  {name:16} PROFILER RAISED: {type(exc).__name__}: {str(exc)[:80]}")
            rows.append({"name": name, "kind": kind, "claim": claim, "crashed": str(exc)[:200]})
            continue
        row.update(kind=kind, claim=claim)
        rows.append(row)
        print(
            f"  {name:16} {kind:18} {row['n_keywords']:2d} keywords, "
            f"{len(row['anchors']):2d} anchors, {len(row['domains'])} domains"
        )

    print("\n" + "=" * 96)
    print("CAN THE PIPELINE DESCRIBE THESE REPOSITORIES AT ALL?")
    print("=" * 96)
    print(f"  {'repo':16} {'kind':18} {'kw':>3} {'anchor':>6} {'name?':>6}  top keywords")
    for r in rows:
        if r.get("crashed"):
            print(f"  {r['name']:16} {r['kind']:18} PROFILER CRASHED")
            continue
        print(
            f"  {r['name']:16} {r['kind']:18} {r['n_keywords']:3d} {len(r['anchors']):6d} "
            f"{'yes' if r['own_name_in_keywords'] else 'NO':>6}  "
            f"{', '.join(r['keywords'][:5])[:52]}"
        )

    print("\n" + "=" * 96)
    print("§4's CLAIMS, RE-TESTED AGAINST THE CURRENT PROFILER")
    print("=" * 96)
    for r in rows:
        if r.get("crashed"):
            continue
        print(f"  {r['name']:16} claim: {r['claim']}")
        print(f"  {'':16} now:   {', '.join(r['keywords'][:8])[:76]}")

    dest_json = WORK_DIR / "blindspot_profiles.json"
    dest_json.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"\nWrote {dest_json}")
    print(
        "\n  Nothing here measures digest quality. It measures whether the pipeline can\n"
        "  DESCRIBE these repositories, which is the precondition for everything else and\n"
        "  the thing §14.2 excluded them for."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
