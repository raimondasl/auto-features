"""Stage-1, $0: what does letting the profiler read source code actually do to a profile?

`profiler.scan_source` is shipped and **no benchmark arm has ever enabled it**. NR-26
pointed at it — whatever benefit lived in its richer arm "tracks the extra *information* —
source code the profiler never reads" — and `ablate_docs`'s guard states the mechanism a
thin-docs remedy would use: such a repository is thin in prose but *has code*.

Before paying for a judged A/B, this asks the question the profile can answer for nothing:
**does source scanning add concepts, or does it drown them?** The profile is the input to
every downstream stage — queries, ranking, the gate prompt, the rescore prompt — so a
change that degrades it cannot be rescued later.

The reason to ask first is a single case. On `thin-lang` (108 characters of prose, an
entire compiler), prose alone yields `programming language`, `native binaries`,
`language compiles`; with scanning on, the top terms become `vscode`, `net`,
`child_process`. Those are real facts about the repository and they are the wrong
*register* — the implementation, which is even further from "what should this adopt" than
the prose register §5.2 already found wanting. One case is an anecdote; this is the sweep.

    uv run python evals/scan_source_probe.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from harness import EVALS_DIR, WORK_DIR, load_benchmark, resolve_repo_dir  # noqa: E402
from run_judge_eval import case_profile  # noqa: E402

# Terms that describe how a program is built rather than what it is about. Not a
# blocklist used by anything — a *measure*, so "source scanning adds implementation
# vocabulary" is a number rather than an impression from one case.
IMPL_MARKERS = (
    "import",
    "self",
    "def",
    "class",
    "return",
    "async",
    "await",
    "const",
    "async def",
    "std",
    "util",
    "utils",
    "config",
    "test",
    "tests",
    "init",
    "args",
    "kwargs",
    "err",
    "func",
    "impl",
    "struct",
    "enum",
    "buf",
    "ptr",
    "net",
    "os",
    "sys",
    "fmt",
    "log",
    "child_process",
    "vscode",
    "npm",
    "node_modules",
)


def _terms(profile: Any, n: int = 20) -> list[str]:
    return [str(k) for k, _ in list(getattr(profile, "keywords", []))[:n]]


def _impl_share(terms: list[str]) -> float:
    if not terms:
        return 0.0
    hits = sum(1 for t in terms if any(m == t or m in t.split() for m in IMPL_MARKERS))
    return hits / len(terms)


def main() -> int:
    cases = load_benchmark()["cases"]
    rows: list[dict[str, Any]] = []
    print(f"{'case':<12} {'kw off':>6} {'kw on':>6} {'kept':>5} {'impl off':>9} {'impl on':>8}")
    print("-" * 56)
    for case in cases:
        name = case["name"]
        # The live clone is what the benchmark profiles; skip cases not cloned yet rather
        # than silently profiling the wrong directory.
        repo = WORK_DIR / name
        if not repo.exists():
            repo = resolve_repo_dir(case)
        if not repo.exists():
            print(f"{name:<12}   (no clone — skipped)")
            continue
        off, on = case_profile(repo, scan_source=False), case_profile(repo, scan_source=True)
        t_off, t_on = _terms(off), _terms(on)
        kept = len(set(t_off) & set(t_on))
        row = {
            "case": name,
            "kw_off": len(getattr(off, "keywords", [])),
            "kw_on": len(getattr(on, "keywords", [])),
            "anchors_off": len(getattr(off, "anchors", [])),
            "anchors_on": len(getattr(on, "anchors", [])),
            "top_kept": kept,
            "impl_off": round(_impl_share(t_off), 3),
            "impl_on": round(_impl_share(t_on), 3),
            "terms_off": t_off[:10],
            "terms_on": t_on[:10],
        }
        rows.append(row)
        print(
            f"{name:<12} {row['kw_off']:>6} {row['kw_on']:>6} {kept:>4}/20 "
            f"{row['impl_off']:>9.2f} {row['impl_on']:>8.2f}"
        )

    if not rows:
        print("\nno clones found — nothing measured (this is a refusal, not a clean result)")
        return 1

    n = len(rows)
    mean = lambda k: sum(r[k] for r in rows) / n  # noqa: E731
    print("-" * 56)
    print(
        f"{'MEAN':<12} {mean('kw_off'):>6.1f} {mean('kw_on'):>6.1f} {mean('top_kept'):>4.1f}/20 "
        f"{mean('impl_off'):>9.2f} {mean('impl_on'):>8.2f}"
    )
    print(
        f"\n  top-20 keyword overlap: {mean('top_kept') / 20:.0%} — source scanning replaces "
        f"{1 - mean('top_kept') / 20:.0%} of the terms every downstream stage reads."
    )
    print(f"  implementation-vocabulary share: {mean('impl_off'):.0%} -> {mean('impl_on'):.0%}")
    out = WORK_DIR / "scan_source_probe.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nWrote {out.relative_to(EVALS_DIR.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
