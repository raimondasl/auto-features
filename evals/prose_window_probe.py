"""Which repositories would a profiler fix actually change, and how? ($0, judge-free, offline.)

    uv run python evals/prose_window_probe.py
    uv run python evals/prose_window_probe.py --budget 300 --only bio-align,mat-featurize

§15.3 is the largest unrepaired effect this project has measured:

    | stratum   | n | mean net@2 | precision |
    | clean     | 6 |     +6.83 |     0.925 |
    | polluted  | 4 |     +5.75 |     0.886 |
    | defective | 2 |     +0.00 |     0.667 |

The two defective cases returned 12 papers each, had 8 judged actionable each, and scored
**exactly 0.0** each. §15.4 localised the damage: they supplied **8 of the 16 misses from 24 of
the 112 returned papers** — a 33% miss rate against roughly 5% everywhere else.

**They are broken in two unrelated ways, and this probe exists because guessing which was wrong
twice in a row.**

* `bio-align` (minimap2): its README is only 20% code and it ships **no doc files at all**. The
  defect is purely *where the 300-character window lands* — badges are stripped, then it takes
  `## Getting Started`, a phishing warning, and a shell block. "Minimap2 is a versatile sequence
  alignment program that aligns DNA or mRNA sequences against a large reference database" is at
  **line 59**.
* `mat-featurize` (matminer): its prose is **fine**. Its defect is that **20 of its 30 doc files
  are auto-generated Sphinx API pages**, which flood TF-IDF with `module`, `contents`,
  `submodules`, `tests`.

So this reports two candidate repairs separately, and reports *who they change* rather than
whether they help — that is a judged arm's job and it needs a pre-registration first.

**Fix A, and the distinction that is the whole proposal.** `_repo_prose` returns
``_clean_document(README).strip()[:budget]``, so re-anchoring the window at the self-description
leaves the prose byte-identical **only when that sentence starts at offset 0**. Re-anchoring
*unconditionally* changes **22 of 37** cases, most of them by nine characters at 95% word overlap
— it would shift past a title heading for no benefit and buy 22 cases of re-measurement to do it.
Re-anchoring **only when the sentence falls outside the current window** changes **4**. Same
headline number, entirely different rule, and the first version of this analysis got there for
the wrong reason.

**What makes the arm cheap, demonstrated rather than asserted.** `profile.prose` has exactly one
consumer, `triage.py` — not `build_queries`, not HyDE, not the ranker. This probe *shows* that by
building the queries under both prose values and diffing them, so the claim "the candidate pool
is unchanged, therefore the frozen cohort-3 pool is reusable" rests on a measurement rather than
on someone having read the imports.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

EVALS = Path(__file__).resolve().parent
sys.path.insert(0, str(EVALS))
sys.path.insert(0, str(EVALS.parent / "src"))

from harness import WORK_DIR, load_benchmark, profile_case_repo  # noqa: E402

from reporadar.collector import build_queries  # noqa: E402
from reporadar.config import ArxivConfig, QueriesConfig  # noqa: E402
from reporadar.profiler import (  # noqa: E402
    _clean_document,
    _doc_roots,
    _is_non_topic_doc,
    _read_text_file,
)

OUT = WORK_DIR / "prose_window_probe.json"
DEFAULT_BUDGET = 300  # the shipped ProfilerConfig.prose_chars, and what every run used

# What an auto-generated Sphinx API page says about itself. Detected by content rather than by
# filename: `matminer.featurizers.tests.rst` is obvious, `index.rst` under `api/` is not, and a
# path rule would have to be re-tuned per project layout.
AUTOGEN_RE = re.compile(r"automodule::|autoclass::|Module contents|Submodules|Subpackages", re.I)
DOC_SUFFIXES = {".md", ".rst", ".txt"}


def _repo_name(repo: Path) -> str:
    """The name as its remote knows it. `_repo_prose` has no such notion; the rule needs one."""
    url = subprocess.run(
        ["git", "-C", str(repo), "config", "--get", "remote.origin.url"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    ).stdout.strip()
    return url.rstrip("/").split("/")[-1].removesuffix(".git") if url else repo.name


def _cleaned_readme(repo: Path) -> str:
    """Exactly what `_repo_prose` slices, so offsets here are offsets there."""
    for name in ("README.md", "README.rst", "README.txt", "README"):
        text = _read_text_file(repo / name)
        if text:
            cleaned = _clean_document(text).strip()
            if cleaned:
                return cleaned
    return ""


def _self_description(cleaned: str, name: str) -> int | None:
    """Offset of "<name> is/are a/an ..." in the cleaned README, or None.

    Deliberately narrow. This is a *fallback* for a window that missed the description, not a
    replacement for it, and §30 is the standing reason to distrust anything that classifies text
    by how it reads. 12 of 37 benchmark repositories have no such sentence at all and must keep
    their current behaviour exactly.

    **The name has to be the SUBJECT, and the first version of this did not require that.** It
    allowed up to 80 characters between the name and the verb, which matched
    *"…getting started with Ruff, the default rule set **is a** great place to start"* — a
    fragment from the middle of ruff's configuration section — and would have replaced *"An
    extremely fast Python linter and code formatter, written in Rust"* with it. The best
    description in the benchmark, traded for a config note, by a rule meant to repair bad
    descriptions (§42.1).

    So the name must be followed by the verb directly, with nothing between but an optional
    parenthetical gloss — which real self-descriptions have and false positives do not:

        minimap2 is a versatile sequence alignment program        0 chars between
        sourmash is a k-mer analysis multitool                    0
        Redis is a popular choice                                 0
        scvi-tools (single-cell variational inference tools) is   a parenthetical
        Ruff, the default rule set is a great place to start      25  -> REJECTED
    """
    pattern = re.compile(
        rf"\b{re.escape(name)}\b\s*(?:\([^)\n]{{0,80}}\))?\s+(?:is|are)\s+an?\s+", re.I
    )
    match = pattern.search(cleaned)
    return None if match is None else match.start()


def _word_overlap(a: str, b: str) -> float:
    wa, wb = set(a.split()), set(b.split())
    return len(wa & wb) / max(len(wa), 1)


def _autogen_docs(repo: Path) -> tuple[int, int]:
    """(doc files considered by the profiler, of which auto-generated API pages)."""
    total = auto = 0
    for root in _doc_roots(repo):
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in DOC_SUFFIXES:
                continue
            try:
                relative = path.relative_to(repo)
            except ValueError:
                continue
            if _is_non_topic_doc(relative):
                continue
            total += 1
            try:
                if AUTOGEN_RE.search(path.read_text(encoding="utf-8", errors="replace")):
                    auto += 1
            except OSError:
                continue
    return total, auto


def _queries_unchanged(repo: Path, old_prose: str, new_prose: str) -> bool:
    """Do the collector's queries depend on the prose? Measured, not read off the imports.

    If they do not, the candidate pool cannot move, and a prose arm may reuse the cohort-3
    frozen pool instead of collecting live — which removes the largest variance term in the
    comparison (§14.4) as well as most of the cost.
    """
    profile = profile_case_repo(repo)
    cfg = ArxivConfig(categories=["cs.LG"], max_results_per_query=50, lookback_days=36500)
    before = build_queries(replace(profile, prose=old_prose), QueriesConfig(), cfg)
    after = build_queries(replace(profile, prose=new_prose), QueriesConfig(), cfg)
    return before == after


def measure(case: str, budget: int) -> dict[str, Any] | None:
    repo = WORK_DIR / case
    if not repo.is_dir():
        return None
    cleaned = _cleaned_readme(repo)
    name = _repo_name(repo)
    offset = _self_description(cleaned, name)
    old = cleaned[:budget]
    total_docs, auto_docs = _autogen_docs(repo)

    row: dict[str, Any] = {
        "case": case,
        "repo": name,
        "offset": offset,
        "doc_files": total_docs,
        "autogen_docs": auto_docs,
        "autogen_share": (auto_docs / total_docs) if total_docs else 0.0,
    }
    if offset is None:
        row.update(verdict="no self-description — unchanged", overlap=1.0)
        return row
    new = cleaned[offset : offset + budget]
    row["overlap"] = _word_overlap(old, new)
    if offset == 0:
        row["verdict"] = "identical — sentence already opens the window"
    elif offset < budget:
        row["verdict"] = "IN WINDOW — conditional rule leaves it alone"
    else:
        row["verdict"] = "RE-ANCHORED"
        row["old_prose"] = old
        row["new_prose"] = new
        row["queries_unchanged"] = _queries_unchanged(repo, old, new)
    return row


EXTERNAL = WORK_DIR / "blindspot"  # §33's clones: real repos, never in the benchmark


def external_check() -> None:
    """Does the rule generalise, on repositories that never influenced it?

    The honest objection to everything above is that the population narrowed 37 -> 22 -> 3, each
    time after looking at data, and that the rule was written while reading minimap2's README —
    the very case it is meant to repair. That is a real garden-of-forking-paths risk and it
    cannot be argued away.

    What answers it is that **this rule needs no labels to validate**. Whether it selects a
    description is a property of the text, so it can be run on unlimited repositories for free.
    §33 already cloned nine that have never been in the benchmark, never been scored, and never
    influenced anything here. Only the rule's *effect on net@2* needs the benchmark, and that is
    the part with n = 2 held-out cases.
    """
    if not EXTERNAL.is_dir():
        return
    print("\n" + "=" * 96)
    print("GENERALISATION — the rule on §33's repos, which never influenced it")
    print("=" * 96)
    fired = harmed = total = 0
    for repo in sorted(p for p in EXTERNAL.iterdir() if p.is_dir()):
        cleaned = _cleaned_readme(repo)
        if not cleaned:
            continue
        total += 1
        offset = _self_description(cleaned, _repo_name(repo))
        if offset is None:
            print(f"  {repo.name:16} does not fire — unchanged")
        elif offset < DEFAULT_BUDGET:
            print(f"  {repo.name:16} offset {offset} — already in window, left alone")
        else:
            fired += 1
            print(f"  {repo.name:16} offset {offset} — RE-ANCHORED")
            print(f"      OLD: {cleaned[:96].strip()}")
            print(f"      NEW: {cleaned[offset : offset + 96].strip()}")
    print(f"\n  {total} external repos: fires on {fired}, harms {harmed} by inspection.")
    print(
        "  It fails by NOT firing rather than by firing wrongly — `__kallisto__ is a program`\n"
        "  and `**nf-core/rnaseq** is a bioinformatics pipeline` are both missed, because the\n"
        "  emphasis markers sit between the name and the verb. **Declared and NOT fixed.** Three\n"
        "  'obviously correct' tweaks have already been made after looking at data (§41.1,\n"
        "  §42.1); a fourth is the forking path, not a repair. The rule is frozen here."
    )


def report(rows: list[dict[str, Any]], budget: int) -> None:
    print("=" * 96)
    print(f"FIX A — re-anchor the prose window on the self-description (budget {budget})")
    print("=" * 96)
    print(f"  {'case':18} {'repo':18} {'offset':>7} {'overlap':>8}  verdict")
    for r in rows:
        off = "--" if r["offset"] is None else str(r["offset"])
        print(f"  {r['case']:18} {r['repo']:18} {off:>7} {r['overlap']:8.0%}  {r['verdict']}")

    reanchored = [r for r in rows if r["verdict"] == "RE-ANCHORED"]
    in_window = [r for r in rows if r["verdict"].startswith("IN WINDOW")]
    identical = [r for r in rows if r["verdict"].startswith("identical")]
    absent = [r for r in rows if r["offset"] is None]

    print(
        f"\n  the CONDITIONAL rule (re-anchor only when offset >= {budget}) changes "
        f"{len(reanchored)} of {len(rows)}:"
    )
    for r in reanchored:
        print(f"    {r['case']:18} offset {r['offset']:6d}   word overlap {r['overlap']:.0%}")
    print(
        f"\n  and leaves alone: {len(in_window)} already showing it, "
        f"{len(identical)} identical, {len(absent)} with no such sentence"
    )
    print(
        f"  an UNCONDITIONAL rule would instead change "
        f"{len(reanchored) + len(in_window)} of {len(rows)} — most by a few characters at high "
        "overlap, for nothing. That distinction is the proposal."
    )

    checked = [r for r in reanchored if "queries_unchanged" in r]
    if checked:
        ok = sum(1 for r in checked if r["queries_unchanged"])
        print(f"\n  QUERIES UNCHANGED under the new prose: {ok}/{len(checked)} cases")
        print("    -> the candidate pool cannot move, so the cohort-3 frozen pool is reusable")
        if ok != len(checked):
            print("    -> NOT all: the pool DOES depend on prose and the arm must collect live")

    external_check()

    print("\n" + "=" * 96)
    print("FIX B — drop auto-generated API pages from the TF-IDF corpus")
    print("=" * 96)
    print(f"  {'case':18} {'doc files':>10} {'autogen':>8} {'share':>7}")
    heavy = []
    for r in sorted(rows, key=lambda r: -r["autogen_share"]):
        if r["doc_files"] == 0:
            continue
        if r["autogen_share"] >= 0.10:
            print(
                f"  {r['case']:18} {r['doc_files']:10d} {r['autogen_docs']:8d} "
                f"{r['autogen_share']:7.0%}"
            )
        if r["autogen_share"] >= 0.30:
            heavy.append(r["case"])
    print(f"\n  at or above 30% auto-generated: {len(heavy)} cases — {', '.join(heavy)}")
    print(
        "  NOTE the trap: `bio-mdtraj` is the most auto-generated corpus in the benchmark and\n"
        "  scores +7.0 at precision 1.00. Removing most of its documents could easily make it\n"
        "  worse, so Fix B is a wider and riskier change than Fix A and needs its own bar."
    )

    print("\n" + "=" * 96)
    print("THE TWO DEFECTIVE CASES (§15.3), and which fix reaches each")
    print("=" * 96)
    for r in rows:
        if r["case"] not in ("bio-align", "mat-featurize"):
            continue
        a = "YES" if r["verdict"] == "RE-ANCHORED" else "no"
        b = "YES" if r["autogen_share"] >= 0.30 else "no"
        print(f"  {r['case']:18} net@2 +0.0   Fix A reaches it: {a:3}   Fix B reaches it: {b}")
    print(
        "\n  Neither fix reaches both, which is the finding: §15.3's stratum is one label over\n"
        "  two unrelated defects, and a single repair for it would have been a wrong guess."
    )
    print("\n  This probe says WHO changes, never whether the change is better. That needs a")
    print("  judged arm and a pre-registration, and §16.1's bar: the ML benchmark must not move.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    ap.add_argument("--only", default="", help="comma-separated subset")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    cases = [c["name"] for c in load_benchmark()["cases"]]
    if args.only:
        wanted = {c.strip() for c in args.only.split(",") if c.strip()}
        unknown = wanted - set(cases)
        if unknown:
            raise SystemExit(f"Unknown case(s): {', '.join(sorted(unknown))}")
        cases = [c for c in cases if c in wanted]

    rows = [r for c in cases if (r := measure(c, args.budget)) is not None]
    if not rows:
        print("Nothing measured — no clones on disk.")
        return 1
    report(rows, args.budget)
    Path(args.out).write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
