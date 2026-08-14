"""Where does the benchmark stop measuring the product? ($0, no network, no LLM.)

    uv run python evals/audit_product_divergence.py

Two of this project's published corrections have the identical shape — **one invariant,
two implementations, one of them fixed**:

* **C-9** the arXiv-grammar-to-keywords bridge, hand-rolled at five call sites. Fixing the
  shared function fixed nothing on its own; three callers kept their own copy.
* **C-12** the version-strip before a source merge. `cli.py` was corrected, `evals/harness.py`
  was not, and the S2 A/B counted six papers twice.

Both were found by accident, months apart, while looking for something else. This script
looks for the shape on purpose, in three passes:

1. **Wiring** — every place an arXiv id is normalised, a source merge is deduped, or the
   digest window is derived, read out of the AST with whichever of the competing rules it
   uses. The id survey covers **every** module this project wrote, not the pipeline: scoping
   it to the five files C-9 and C-12 lived in reported clean while three rules coexisted
   across eight modules it never opened.
2. **Configuration** — the shipped defaults against the configuration the benchmark's
   headline actually runs. `arxiv.lookback_days` already drifted this way for a month
   (14 days shipped, all-time measured); a difference is fine, an *undeclared* one is not.
3. **Blast radius** — how much any of it touched a published number, read off the recorded
   runs in ``evals/results/`` rather than argued.

Everything here is free and offline. It is meant to be run before believing a benchmark
number, in the same spirit as the $0 stage-1 probes that precede a paid experiment.
"""

from __future__ import annotations

import ast
import collections
import json
import re
import sys
from pathlib import Path
from typing import Any, NamedTuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reporadar.config import (  # noqa: E402
    ArxivConfig,
    OutputConfig,
    ProfilerConfig,
    RankingConfig,
    TriageConfig,
)
from reporadar.paper_id import dedup_id  # noqa: E402

# Modules that participate in the collect -> rank -> gate -> show pipeline on either side.
# The merge and digest-window passes are about how THIS pipeline behaves, so they are
# scoped to it deliberately.
PIPELINE_MODULES = (
    ("src", "reporadar", "cli.py"),
    ("src", "reporadar", "digest.py"),
    ("evals", "harness.py"),
    ("evals", "run_eval.py"),
    ("evals", "run_judge_eval.py"),
)


def all_modules() -> list[Path]:
    """Everything this project wrote, on either side of the line.

    The id-normalisation survey uses this rather than `PIPELINE_MODULES`, and the reason is
    the finding that produced it: the first version of that survey looked at the five files
    C-9 and C-12 happened to live in, and a sweep on 2026-08-15 then turned up three
    competing rules across eight product modules it had never opened — the MCP server, two
    signal collectors, three source adapters. A survey scoped to where you last found a bug
    reports clean about everywhere you did not look.

    `evals/.work/` holds cloned benchmark repositories, which are other people's source.
    """
    files = sorted((ROOT / "src" / "reporadar").rglob("*.py"))
    files += sorted((ROOT / "evals").glob("*.py"))
    return [p for p in files if ".work" not in p.parts]


class Divergence(NamedTuple):
    """One invariant, and what each side does about it."""

    name: str
    product: str
    benchmark: str
    status: str  # "shared" | "declared" | "DIVERGENT"
    note: str


# ── pass 1: wiring ─────────────────────────────────────────────────────────


def _norm_calls(path: Path) -> list[tuple[int, str, str]]:
    """Every arXiv-id normalisation in *path*: (line, rule, source text).

    Two rules exist. `dedup_id` version-strips ids it recognises and leaves the rest
    alone; `split("v")[0]` truncates at the first lowercase ``v``, whatever that is. On a
    modern id (`2605.23815v1`) they agree, which is why the split lived so long.
    """
    found: list[tuple[int, str, str]] = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.Call):
            continue
        rendered = ast.unparse(node)
        # `cli.py` imports the helper and re-exports it as `_dedup_id`; both are the
        # shared rule, and a check that missed the alias would report a clean file dirty.
        if isinstance(node.func, ast.Name) and node.func.id in ("dedup_id", "_dedup_id"):
            found.append((node.lineno, "dedup_id", rendered))
    # `x.split("v")[0]` parses as Subscript(Call(...)), not Call — walk for it separately.
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.Subscript):
            continue
        rendered = ast.unparse(node)
        if re.search(r"\.split\('v'\)\[0\]", rendered):
            found.append((node.lineno, "split-v", rendered))
    return sorted(found)


# Membership tests on a raw `arxiv_id` that are NOT the C-12 defect, by the name of the
# container they test against. Each is a lookup into a mapping built from the *same* list
# in the *same* process, where both sides carry byte-identical ids and normalising would
# only add a rule. Declared here rather than special-cased in the checker, so adding one
# is a visible decision.
SAME_VINTAGE_CONTAINERS = {
    "papers_by_id": "dict built from this run's own collected papers",
    "by_id": "dict built from the pool being ranked",
    "exported": "ids the store itself recorded as exported, compared to store ids",
}


def _raw_merges(path: Path) -> list[tuple[int, str, str]]:
    """Merges still comparing a raw ``arxiv_id`` against a set — the C-12 defect.

    The shape is ``p["arxiv_id"] not in <container>``; the repaired form wraps the
    subscript in ``dedup_id``. Same-vintage lookups are reported separately rather than
    silently dropped: the reason each one is exempt is a claim worth being able to read.
    """
    out: list[tuple[int, str, str]] = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.Compare) or len(node.ops) != 1:
            continue
        if not isinstance(node.ops[0], (ast.NotIn, ast.In)):
            continue
        if not ast.unparse(node.left).endswith("['arxiv_id']"):
            continue
        container = ast.unparse(node.comparators[0])
        kind = "declared" if container in SAME_VINTAGE_CONTAINERS else "RAW MERGE"
        out.append((node.lineno, kind, ast.unparse(node)))
    return out


# The digest's ordering: rerank by `llm_score`, drop withdrawn papers, THEN cut to
# `output.top_n`. Two callers need it — `digest.categorize_papers` to tier, and
# `cli.update` to decide which papers are worth a fine-scale call — and the rule they must
# agree on is subtle enough to re-derive wrongly: a withdrawn paper leaves before the cut,
# so each one pulls the next paper up into the window. `digest_window` is the one home.
WINDOW_HELPER = "digest_window"
# The rule's two halves in source form. A second implementation would need both, so a file
# outside the helper containing either is the thing to look at.
WINDOW_FRAGMENTS = ('withdrawn_in")][:', "withdrawn_in')][:")


def _hand_rolled_windows(path: Path) -> list[tuple[int, str]]:
    """Re-implementations of the digest's drop-withdrawn-then-cut rule.

    `digest.py` is where the rule lives, so it is exempt by construction; any other file
    slicing a list it has just filtered on ``withdrawn_in`` has grown a second copy.
    """
    if path.name == "digest.py":
        return []
    out: list[tuple[int, str]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.lstrip().startswith("#"):
            continue
        if any(fragment in line for fragment in WINDOW_FRAGMENTS):
            out.append((line_no, line.strip()))
    return out


def _window_callers(path: Path) -> list[int]:
    """Lines where *path* calls the shared window helper."""
    return [
        node.lineno
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == WINDOW_HELPER
    ]


def pass_wiring() -> list[Divergence]:
    print("=" * 78)
    print("1. WIRING — one invariant, how many implementations?")
    print("=" * 78)

    by_rule: collections.Counter[str] = collections.Counter()
    by_module: collections.Counter[str] = collections.Counter()
    offenders: list[str] = []
    modules = all_modules()
    print(f"\n  arXiv-id normalisation, across all {len(modules)} modules:")
    for path in modules:
        # `paper_id.py` IS the rule; this file keeps a copy of the OLD rule on purpose,
        # because pass 3 reports the ids the two disagree on and cannot do that with
        # only one of them. Both exemptions are mirrored in tests/test_paper_id.py.
        if path.name in ("paper_id.py", "audit_product_divergence.py"):
            continue
        rel = path.relative_to(ROOT).as_posix()
        for line, rule, text in _norm_calls(path):
            by_rule[rule] += 1
            by_module[rel] += 1
            if rule != "dedup_id":
                offenders.append(f"{rel}:{line}  {text[:52]}")
    print(f"     {by_rule['dedup_id']} call(s) via the shared rule, across {len(by_module)} files")
    for line in offenders:
        print(f"   ! {line}")
    print(
        f"\n  -> {by_rule['split-v']} hand-rolled (the enforcing guard is tests/test_paper_id.py)"
    )

    print("\n  Membership tests on a raw arXiv id:")
    raw = []
    for parts in PIPELINE_MODULES:
        path = ROOT.joinpath(*parts)
        for line, kind, text in _raw_merges(path):
            flag = " ! " if kind == "RAW MERGE" else "   "
            if kind == "RAW MERGE":
                raw.append((parts, line, text))
            print(f"  {flag}{'/'.join(parts[-2:])}:{line:<5} [{kind:10}] {text[:52]}")
    print(f"\n  -> {len(raw)} unrepaired merge(s) on raw ids (the C-12 defect)")

    print("\n  The digest window (drop withdrawn, THEN cut to top_n):")
    hand_rolled: list[tuple[tuple[str, ...], int, str]] = []
    callers = 0
    for parts in PIPELINE_MODULES:
        path = ROOT.joinpath(*parts)
        for line in _window_callers(path):
            callers += 1
            print(f"     {'/'.join(parts[-2:])}:{line:<5} [{WINDOW_HELPER:10}] shared")
        for line, text in _hand_rolled_windows(path):
            hand_rolled.append((parts, line, text))
            print(f"   ! {'/'.join(parts[-2:])}:{line:<5} [re-derived] {text[:48]}")
    print(f"\n  -> {callers} caller(s) share it, {len(hand_rolled)} re-derive it")

    out = []
    if hand_rolled:
        out.append(
            Divergence(
                "digest window",
                f"digest.{WINDOW_HELPER}",
                f"{len(hand_rolled)} re-derived site(s)",
                "DIVERGENT",
                "a copy that forgets withdrawn-before-the-cut is off by one paper per "
                "withdrawal, and only on runs where a retraction lands in the top slots",
            )
        )
    if by_rule["split-v"]:
        out.append(
            Divergence(
                "arxiv id normalisation",
                "collector.dedup_id",
                f"{by_rule['split-v']} bare split('v')[0] call sites",
                "DIVERGENT",
                "the two rules disagree on old-style ids; see pass 3 for the population",
            )
        )
    if raw:
        out.append(
            Divergence(
                "source merge dedup",
                "dedup_id(p['arxiv_id'])",
                f"raw id at {len(raw)} site(s)",
                "DIVERGENT",
                "C-12, unfixed here",
            )
        )
    return out


# ── pass 2: configuration ──────────────────────────────────────────────────

# What the benchmark's headline command actually sets, from evals/RESULTS.md:
#   --rr-pool 50 --rr-rerank --rr-all-time --rr-hybrid --rr-sweep --rr-finescale --rr-hyde
BENCHMARK_HEADLINE: dict[str, Any] = {
    "arxiv.lookback_days": 36500,  # --rr-all-time
    "arxiv.sort_by": "relevance",  # --rr-all-time
    "arxiv.max_results_per_query": 50,
    "ranking.w_recency": 0.0,  # --rr-all-time
    "ranking.w_keyword": 1.0,
    "ranking.w_category": 0.5,
    "ranking.absent_category": "omit",
    "profiler.prose_chars": 300,  # --rr-prose-chars default
    "triage.min_actionable": 2,  # --rr-min-actionable default
    "triage.top_k": 50,  # --rr-pool 50
    "output.top_n": 10,  # the harness cuts the returned set at 10
    "triage.finescale.threshold": 2 / 3,  # --rr-finescale-threshold default
}

# Differences that are deliberate and have a reason. Anything NOT listed here and not
# equal is an undeclared divergence, which is the whole point of the pass.
DECLARED: dict[str, str] = {
    # `triage.top_k` was declared here on 2026-08-14 and un-declared the same day: this
    # audit is what noticed the shipped 15 had never been in an experiment, the depth arms
    # measured 50 at +1.00/case over it, and the default moved to match. Removing the entry
    # is the required half of that — a test asserts every DECLARED field still differs, so a
    # stale exemption fails rather than quietly covering the next drift.
    "output.top_n": (
        "the benchmark cuts the returned set at 10 to hold digest size fixed while "
        "testing selection; the product shows up to 15."
    ),
}


def _product_defaults() -> dict[str, Any]:
    triage = TriageConfig()
    return {
        "arxiv.lookback_days": ArxivConfig().lookback_days,
        "arxiv.sort_by": ArxivConfig().sort_by,
        "arxiv.max_results_per_query": ArxivConfig().max_results_per_query,
        "ranking.w_recency": RankingConfig().w_recency,
        "ranking.w_keyword": RankingConfig().w_keyword,
        "ranking.w_category": RankingConfig().w_category,
        "ranking.absent_category": RankingConfig().absent_category,
        "profiler.prose_chars": ProfilerConfig().prose_chars,
        "triage.min_actionable": triage.min_actionable,
        "triage.top_k": triage.top_k,
        "output.top_n": OutputConfig().top_n,
        "triage.finescale.threshold": triage.finescale.threshold,
    }


def config_divergences() -> list[Divergence]:
    """Fields where the shipped default and the measured configuration disagree."""
    product = _product_defaults()
    out = []
    for key, measured in BENCHMARK_HEADLINE.items():
        shipped = product[key]
        same = (
            abs(shipped - measured) < 1e-9
            if isinstance(shipped, float) and isinstance(measured, float)
            else shipped == measured
        )
        if same:
            continue
        out.append(
            Divergence(
                key,
                repr(shipped),
                repr(measured),
                "declared" if key in DECLARED else "DIVERGENT",
                DECLARED.get(key, "not declared — the benchmark is not measuring the default"),
            )
        )
    return out


def pass_config() -> list[Divergence]:
    print("\n" + "=" * 78)
    print("2. CONFIGURATION — is the benchmark measuring the shipped defaults?")
    print("=" * 78)
    divs = config_divergences()
    if not divs:
        print("\n  every compared field agrees.")
        return []
    print(f"\n  {'field':<32} {'shipped':>12} {'measured':>12}  status")
    for d in divs:
        print(f"  {d.name:<32} {d.product:>12} {d.benchmark:>12}  {d.status}")
    for d in divs:
        print(f"\n  {d.name}: {d.note}")
    return [d for d in divs if d.status == "DIVERGENT"]


# ── pass 3: blast radius ───────────────────────────────────────────────────

_MODERN = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")


def _split_v(arxiv_id: str) -> str:
    return arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id


def _recorded_runs() -> list[tuple[str, list[dict[str, Any]]]]:
    runs = []
    for path in sorted((ROOT / "evals" / "results").glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, list) and data and isinstance(data[0], dict):
            runs.append((path.name, data))
    return runs


def pass_blast_radius() -> None:
    print("\n" + "=" * 78)
    print("3. BLAST RADIUS — how much of this reached a published number?")
    print("=" * 78)

    runs = _recorded_runs()
    ids: set[str] = set()
    n_records = n_unscored = n_ungated = 0
    dup_groups: list[tuple[str, str, str]] = []

    for fname, cases in runs:
        for case in cases:
            if not isinstance(case, dict):
                continue
            returned = case.get("returned") or {}
            top10 = returned.get("reporadar_top10") or []
            for rec in top10:
                aid = rec.get("arxiv_id") or ""
                if aid:
                    ids.add(aid)
                n_records += 1
                # PRESENT-and-null means the gate ran and failed on this paper — the one
                # state where the two tiering rules disagree. Absent means it never ran.
                if "llm_score" in rec and rec["llm_score"] is None:
                    n_unscored += 1
                if "finescale_p" in rec and rec["finescale_p"] is None:
                    n_ungated += 1
            counted = collections.Counter(
                dedup_id(r["arxiv_id"]) for r in top10 if r.get("arxiv_id")
            )
            for base, n in counted.items():
                if n > 1:
                    dup_groups.append((fname, str(case.get("case")), base))

    print(f"\n  {len(runs)} recorded runs, {n_records} returned top-10 papers, {len(ids)} unique")

    print(
        "\n  a) papers the gate ran on and failed to score (product shows them, "
        "benchmark does not):"
    )
    print(f"     {n_unscored}")
    print(f"  b) band papers the rescore failed on: {n_ungated}")

    non_modern = sorted(a for a in ids if not _MODERN.match(a) and ":" not in a)
    disagree = sorted(a for a in ids if dedup_id(a) != _split_v(a))
    print(f"\n  c) old-style arXiv ids in judged pools: {len(non_modern)}")
    for a in non_modern[:8]:
        print(f"       {a:<24} dedup_id -> {dedup_id(a):<20} split-v -> {_split_v(a)}")
    print(f"     of those, the two rules disagree on: {len(disagree)}")

    print(f"\n  d) surviving duplicate groups in a returned top-10: {len(dup_groups)}")
    for fname, case, base in dup_groups[:10]:
        print(f"       {fname[-28:]:<30} {case:<10} {base}")


def main() -> int:
    findings = pass_wiring()
    findings += pass_config()
    pass_blast_radius()

    print("\n" + "=" * 78)
    if findings:
        print(f"UNDECLARED DIVERGENCES: {len(findings)}")
        for d in findings:
            print(f"  ! {d.name}: product={d.product}  benchmark={d.benchmark}")
            print(f"      {d.note}")
    else:
        print("No undeclared divergence between the product and the benchmark.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
