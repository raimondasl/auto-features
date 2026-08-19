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

1. **Wiring** — every place an arXiv id is normalised, a source merge is deduped, the
   digest window is derived, or the digest declines a paper the repository already cites,
   read out of the AST with whichever of the competing rules it uses. The id survey covers
   **every** module this project wrote, not the pipeline: scoping it to the five files C-9
   and C-12 lived in reported clean while three rules coexisted across eight modules it
   never opened.
2. **Configuration** — the shipped defaults against the configuration the benchmark's
   headline actually runs. `arxiv.lookback_days` already drifted this way for a month
   (14 days shipped, all-time measured); a difference is fine, an *undeclared* one is not.
   This pass is **exhaustive over every config leaf and over both shipped surfaces**, for
   the reason its first version demonstrates: it compared twelve hand-listed fields against
   the dataclass defaults, reported clean, and was asking the wrong object. `rr init` writes
   a *template*, and where the template sets a value that is what a user runs — a scope
   chosen by whoever last edited the list is the C-14b defect wearing a different hat.
3. **Blast radius** — how much any of it touched a published number, read off the recorded
   runs in ``evals/results/`` rather than argued.

Everything here is free and offline. It is meant to be run before believing a benchmark
number, in the same spirit as the $0 stage-1 probes that precede a paid experiment.
"""

from __future__ import annotations

import ast
import collections
import dataclasses
import json
import re
import sys
from pathlib import Path
from typing import Any, NamedTuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# This script's own findings are prose, and the prose has em-dashes in it. A Windows
# console defaults to cp1252, which renders every one of them as a replacement character
# — including inside the reasons a declared divergence gives for itself, which is the part
# a reader most needs to be able to read.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import yaml  # noqa: E402

from reporadar.config import (  # noqa: E402
    RepoRadarConfig,
    default_config_yaml,
    measured_config_yaml,
)
from reporadar.paper_id import dedup_id  # noqa: E402

# Modules that participate in the collect -> rank -> gate -> show pipeline on either side.
# The merge and digest-window passes are about how THIS pipeline behaves, so they are
# scoped to it deliberately.
PIPELINE_MODULES = (
    ("src", "reporadar", "cli.py"),
    ("src", "reporadar", "pipeline.py"),
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


def _rel(path: Path) -> str:
    """*path* for display, tolerating a path outside the repository.

    Both surveys that walk `all_modules()` are mutation-tested by pointing them at a
    synthetic tree that HAS the defect, and a bare `relative_to(ROOT)` raises there -- so
    the checkers would be untestable for a reason that has nothing to do with what they
    check.
    """
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


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

# The tiering entry point, and the keyword that carries the already-cited rule into it.
TIER_HELPER = "categorize_papers"
CITED_KWARG = "cited_ids"

# Wiring divergences that are known, deliberate, and argued -- the pass-1 counterpart of
# DECLARED below. An entry here is a promise that somebody decided, not that nobody looked,
# and `tests/test_eval_product_divergence.py` checks each one still describes reality.
DECLARED_WIRING: dict[str, str] = {
    "already-cited tiering": (
        "The product declines to recommend a paper the repository's own README, CITATION "
        "file or docs already cite (`profiler.cited_arxiv_ids_of` -> `categorize_papers"
        "(cited_ids=...)`), and the benchmark does not: `run_judge_eval.py` derives "
        "RepoRadar's Top Picks itself and never calls the tiering helper, so no published "
        "number is affected -- but from 2026-08-18 the product shows one fewer paper than "
        "the benchmark would score it on, on any repository that cites its own work. The "
        "rule was measured on six scientific-software repositories, where the repo's own "
        "paper was a gate-score-3 Top Pick on five of them and the judge scored it 1 every "
        "time (evals/RESEARCH-scientific-software.md). Closing it means teaching the "
        "harness the same rule, which changes what the benchmark measures and therefore "
        "needs a paired run -- so it belongs with the scientific-software cohort, not "
        "ahead of it. Until then this is a divergence in the product's favour, and the "
        "honest place for it is here rather than in a PR description nobody greps."
    ),
}


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


def _tier_callers(path: Path) -> list[tuple[int, bool]]:
    """Calls to the tiering helper in *path*, and whether each carries the cited-ids rule.

    An explicit ``cited_ids=None`` counts as omitting it: passing the argument is not the
    invariant, applying the rule is, and a literal None is how a caller opts out while
    reading as though it complied.
    """
    out: list[tuple[int, bool]] = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            continue
        if node.func.id != TIER_HELPER:
            continue
        passes = any(
            kw.arg == CITED_KWARG
            and not (isinstance(kw.value, ast.Constant) and kw.value.value is None)
            for kw in node.keywords
        )
        out.append((node.lineno, passes))
    return out


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
        rel = _rel(path)
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

    # Papers the repository already cites. Scoped to `all_modules()` rather than to
    # PIPELINE_MODULES on purpose: `archive.py` and `watcher.py` both tier, neither is in
    # that tuple, and a survey scoped to where the last bug was found reports clean about
    # everywhere it never looked -- the C-14b argument this file is built on.
    print("\n  Papers the repository already cites (product-only rule):")
    tier_missing: list[str] = []
    tier_ok = 0
    bench_tiering: list[str] = []
    for path in all_modules():
        if path.name == "audit_product_divergence.py":
            continue
        # `is_product` comes from the path itself rather than from ROOT, so the checker can
        # be pointed at a synthetic tree that HAS the defect -- the only way to know it can
        # still fail. A checker that cannot fail reads as a clean bill of health.
        rel = _rel(path)
        is_product = path.parts[-3:-1] == ("src", "reporadar")
        for line, passes in _tier_callers(path):
            if not is_product:
                bench_tiering.append(f"{rel}:{line}")
                print(f"   ! {rel}:{line:<5} [benchmark ] tiers through the product helper")
            elif passes:
                tier_ok += 1
                print(f"     {rel}:{line:<5} [{CITED_KWARG:10}] passed")
            else:
                tier_missing.append(f"{rel}:{line}")
                print(f"   ! {rel}:{line:<5} [omitted   ] tiers without the rule")
    print(
        f"\n  -> {tier_ok} product caller(s) apply it, {len(tier_missing)} omit it; "
        f"{len(bench_tiering)} benchmark caller(s)"
    )
    for name, reason in DECLARED_WIRING.items():
        print(f"\n  declared  {name}: {reason}")

    out = []
    if tier_missing:
        out.append(
            Divergence(
                "already-cited tiering",
                f"{TIER_HELPER}({CITED_KWARG}=...)",
                f"{len(tier_missing)} caller(s) tier without it",
                "DIVERGENT",
                "a caller that omits it counts a Top Pick the digest does not show, which "
                "is how `rr notify` and a digest come to disagree about the same run",
            )
        )
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
#
# "The shipped default" is not one object. There are two surfaces and they can disagree:
#
#   1. the dataclass default — what a field is worth when the yml omits it, and
#   2. `default_config_yaml()` — what `rr init` WRITES into `.reporadar.yml`.
#
# Where the template sets a value, that value is what a user runs, and the dataclass
# default is dead text. The first version of this pass compared twelve hand-listed fields
# against surface 1 and reported clean; `ranking.w_embedding` is 0.0 there and **1.5** in
# the template, so the audit was clean about a field on which the product and the benchmark
# disagree by the largest ranking weight in the file. Both halves of that failure are
# structural — the wrong surface, and a hand-written list — so both are closed here:
# every leaf is compared, and the comparison uses the effective value.


def config_leaves() -> dict[str, Any]:
    """Every leaf field of the config tree, dotted, with its dataclass default.

    Recursive, so `triage.finescale.threshold` and `hooks.email.smtp_port` are leaves like
    any other. Nothing here is hand-listed: adding a field to any config dataclass adds it
    to this dict, which is what makes the coverage check below unforgeable.

    The recursion is an inner function on purpose — a module-level self-call would make
    this unpatchable in the mutation test that proves the coverage check can fail.
    """

    def walk(obj: Any, prefix: str) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for f in dataclasses.fields(obj):
            value = getattr(obj, f.name)
            key = f"{prefix}{f.name}"
            if dataclasses.is_dataclass(value):
                out.update(walk(value, key + "."))
            else:
                out[key] = value
        return out

    return walk(RepoRadarConfig(repo_path="."), "")


def _flatten_yaml(text: str) -> dict[str, Any]:
    def flatten(node: dict[str, Any], prefix: str = "") -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in node.items():
            if isinstance(value, dict):
                out.update(flatten(value, f"{prefix}{key}."))
            else:
                out[f"{prefix}{key}"] = value
        return out

    return flatten(yaml.safe_load(text) or {})


def template_values() -> dict[str, Any]:
    """What `rr init` writes into `.reporadar.yml`, flattened to the same dotted keys.

    Parsed rather than duplicated, so the template and this audit cannot drift apart —
    the failure mode the audit exists to catch, applied to the audit itself.
    """
    return _flatten_yaml(default_config_yaml())


def measured_preset() -> dict[str, Any]:
    """What `rr init --measured` writes: the configuration we tell users reaches +5.42.

    Kept honest by :func:`preset_divergences` below rather than by intent. A recommended
    configuration is a claim about a measurement, and claims in this project decay the
    same way code does — `arxiv.lookback_days` shipped at 14 days for a month while every
    headline ran all-time. The difference here is that the claim is machine-checkable
    against the measurement it cites, so it fails instead of aging.
    """
    return _flatten_yaml(measured_config_yaml())


def effective_shipped() -> dict[str, Any]:
    """What a user actually runs after `rr init`: the template, falling back to the class."""
    return {**config_leaves(), **template_values()}


# What the benchmark's headline command actually sets, from evals/RESULTS.md:
#   --rr-pool 50 --rr-rerank --rr-all-time --rr-hybrid --rr-sweep --rr-finescale --rr-hyde
#
# The rule for what belongs here: the benchmark's code path READS the field. A field the
# benchmark reads at its default value still belongs here — that is a measurement, and
# recording it means a future change to that default gets caught rather than assumed
# harmless. Fields the benchmark never reaches, or always overrides per case, go in
# NOT_UNDER_TEST with the reason.
BENCHMARK_HEADLINE: dict[str, Any] = {
    "sources": ["arxiv"],  # --sources arxiv
    "arxiv.lookback_days": 36500,  # --rr-all-time
    "arxiv.sort_by": "relevance",  # --rr-all-time
    "arxiv.max_results_per_query": 50,
    "queries.seed": [],  # harness builds queries from the profile alone
    "queries.bigrams": "verified",  # --rr-bigrams default
    "ranking.w_keyword": 1.0,
    "ranking.w_category": 0.5,
    "ranking.absent_category": "omit",  # --rr-absent-category default
    "ranking.w_recency": 0.0,  # --rr-all-time
    # run_judge_eval._rank builds RankingConfig(w_keyword=, w_category=, w_recency=,
    # absent_category=) and leaves every other weight at the dataclass default, so these
    # are measured at 0.0 — not un-measured.
    #
    # `w_embedding` is the exception, and it moved here on 2026-08-16 only because a run
    # was paid for. Two paired draws put 1.5 at +1.00/case over 0.0 [NR-38], but a delta
    # is not a headline: this dict records what the published run SET, so flipping it on
    # the strength of the delta would have asserted a run that never happened — the C-17
    # error inside the file whose job is preventing it. The headline was re-measured
    # instead (`…-wemb1.5-20260815T225831Z.json`, 25 cases, window 15, `--baseline cli`,
    # +5.72 against the baseline's +1.56, paired +4.16, sign p = 0.0004), and this is
    # that run's value.
    "ranking.w_embedding": 1.5,
    "ranking.w_citations": 0.0,
    "ranking.w_citation_proximity": 0.0,
    "ranking.w_specter": 0.0,
    "ranking.w_community": 0.0,
    "ranking.w_attention": 0.0,
    "ranking.withdrawn_penalty": 0.1,
    "ranking.category_weights": {},
    "ranking.hybrid": True,  # --rr-hybrid (applied by hybrid_reorder, not the flag)
    "output.top_n": 15,  # --rr-window default since 2026-08-15
    "profiler.scan_source": False,  # harness.profile_case_repo default
    "profiler.typed_anchors": False,  # --rr-typed-anchors default; the measured arm sets it
    "profiler.prose_chars": 300,  # --rr-prose-chars default
    "suggestions.provider": "claude",  # the gate's SuggestionsConfig(provider="claude")
    "suggestions.claude_model": "claude-haiku-4-5",  # --rr-triage-model default
    "suggestions.timeout": 30,
    "triage.enabled": True,  # the gate runs in every headline
    "triage.top_k": 50,  # --rr-pool 50
    "triage.min_actionable": 2,  # --rr-min-actionable default
    "triage.rerank": True,  # --rr-rerank
    "triage.finescale.enabled": True,  # --rr-finescale
    "triage.finescale.openai_model": "gpt-4o-mini",  # --rr-finescale-model default
    "triage.finescale.timeout": 60,
    "triage.finescale.threshold": 2 / 3,  # --rr-finescale-threshold default
    "triage.finescale.min_success_fraction": 0.5,  # enough_scored() default
    "hyde.enabled": True,  # --rr-hyde
    "hyde.model": "mixedbread-ai/mxbai-embed-large-v1",  # hyde.discover(model_name=) default
    "hyde.n_hypotheses": 4,  # --rr-hyde-hypotheses default
    "hyde.top_k": 100,  # --rr-hyde-top-k default
    "hyde.verify_encoder": True,  # --rr-hyde-skip-verify not passed
}

# Fields whose SHIPPED DEFAULT the benchmark does not put under test, each with the reason.
# Two distinct cases, both of which mean "no number here says anything about this value":
# the benchmark never reaches the code that reads it, or it always supplies a per-case
# value so the default is never the thing being measured.
NOT_UNDER_TEST: dict[str, str] = {
    "repo_path": "per case — the benchmark points at its own mini-repos",
    "arxiv.categories": (
        "per case, from the benchmark YAML. The shipped [cs.LG, cs.CL] is therefore "
        "unmeasured, and it is a domain guess a fresh install applies to any repository"
    ),
    "queries.exclude": "per case — `harness.rank_pool(exclude=)` supplies the case's list",
    "queries.redact": "mirror of privacy.redact, populated at load; not settable directly",
    "suggestions.redact": "mirror of privacy.redact, populated at load",
    "triage.finescale.redact": "mirror of privacy.redact, populated at load",
    "privacy.redact": "redaction is measured by rr audit's drift guard, not by net@2",
    "output.digest_path": "the benchmark reads returned papers from the store, not a file",
    "semantic_scholar.api_key": "credential",
    "openalex.email": "credential/politeness header",
    "openalex.api_key": "credential",
    "suggestions.claude_api_key": "credential",
    "triage.finescale.openai_api_key": "credential",
    "enrichment.hf_token": "credential",
    "hooks.email.password": "credential",
    "enrichment.provider": "enrichment runs after selection; changes no returned paper",
    "hooks.on_digest": "notification, downstream of every measured number",
    "hooks.slack_webhook_url": "notification",
    "hooks.discord_webhook_url": "notification",
    "hooks.email.smtp_host": "notification",
    "hooks.email.smtp_port": "notification",
    "hooks.email.from_addr": "notification",
    "hooks.email.to": "notification",
    "hooks.email.username": "notification",
    "hooks.email.use_tls": "notification",
    "profiler.max_files": "only read when profiler.scan_source is on, which no arm sets",
    "profiler.source_extensions": "only read when profiler.scan_source is on",
    # The loader mirrors the gate's LLM credentials onto ProfilerConfig so `profile_repo`
    # can reach `complete()` without a signature change at ten call sites. They are read
    # only when profiler.typed_anchors is on, and `case_profile` mirrors the same values
    # from the environment for the arm that sets it.
    "profiler.provider": "mirrored from suggestions; read only when typed_anchors is on",
    "profiler.claude_api_key": "mirrored from suggestions; read only when typed_anchors is on",
    "profiler.claude_model": "mirrored; equals profiler.typed_anchors_model by construction",
    "profiler.timeout": "mirrored from suggestions; read only when typed_anchors is on",
    "profiler.redact": "mirrored from privacy.redact, as for suggestions and queries",
    "profiler.typed_anchors_model": "the --rr-typed-anchors arm sets it to the same Haiku",
    "suggestions.ollama_model": "the benchmark's gate is provider='claude'",
    "suggestions.ollama_url": "the benchmark's gate is provider='claude'",
    "suggestions.max_suggestions": "governs `rr suggest`; triage_papers does not read it",
    "feedback.enabled": "star-feedback reweighting has no benchmark arm at all",
    "feedback.min_ratings": "feedback has no benchmark arm",
    "feedback.learning_rate": "feedback has no benchmark arm",
    "recommendations.enabled": (
        "the S2 recommendations channel is measured through --sources, not this flag"
    ),
    "recommendations.limit": "see recommendations.enabled",
    "recommendations.max_seeds": "see recommendations.enabled",
    "signals.integrity": (
        "no eval run has ever carried a withdrawn_in field, so the digest window's "
        "drop-withdrawn half is exercised by tests and by the product, never by a number"
    ),
    "signals.hackernews": "w_attention is 0.0 in every arm, so the signal cannot move a rank",
    "hyde.index_dir": "path — the benchmark points at its own synced index",
    "hyde.stale_after_days": "staleness warning only; does not change which ids come back",
}

# Differences that are deliberate and have a reason. Anything NOT listed here and not
# equal is an undeclared divergence, which is the whole point of the pass. A test asserts
# every entry here still differs, so a stale exemption fails rather than quietly covering
# the next drift.
#
# The five stage flags below are the reason this file's previous version reported "every
# compared field agrees" while the product shipped with none of the pipeline the paper
# measures. They are declared, not fixed, because each has a cost a default cannot assume
# — and declaring them is what makes "what does a keyless install actually score?" a
# question with a written-down answer instead of an assumption.
DECLARED: dict[str, str] = {
    "triage.enabled": (
        "the gate needs an LLM key (or a local Ollama) and costs ~$0.01/run; a default "
        "that fails without a credential is worse than one that under-delivers. But the "
        "ungated digest IS the configuration measured at mean net@2 **-8.12** over the 25 "
        "repositories, net-negative on 19 of them, so this is the single largest gap "
        "between the shipped product and every number published about it. (This entry "
        "quoted -11 until 2026-08-16. That figure was a FOUR-case July mean, one of them a "
        "negative control, and quoting it as a benchmark headline is the C-17 error "
        "itself — surviving here, uncorrected, in the file whose purpose is catching it.)"
    ),
    "suggestions.provider": (
        "'template' is the keyless default, and cli.update requires ollama|claude before "
        "it will gate at all — so `triage.enabled: true` alone is a no-op that prints one "
        "info line. Enabling the gate takes TWO fields, which is worth knowing"
    ),
    "triage.finescale.enabled": (
        "needs a SECOND vendor's key (OpenAI, for token logprobs Anthropic does not "
        "expose). Cannot be a default at any price"
    ),
    "hyde.enabled": (
        "needs `rr sync-index` and ~1.1 GB on disk plus the `hyde` extra. Opt-in is right; "
        "that every headline includes it is the part to keep visible"
    ),
    "ranking.hybrid": (
        "BM25-RRF fusion, and the ONE flag here with no cost excuse — retrieval.py is "
        "dependency-free plain Python. It stays off in the DEFAULT and on in the preset, "
        "and that difference is now closed by scope rather than by measurement: the "
        "default is a configuration whose own header tells users to replace it, measured "
        "at -8.12, and we do not spend experiments tuning a configuration we recommend "
        "against. This entry previously promised 'the out-of-the-box arm' would settle it. "
        "That arm ran (-8.12) and reading the ungated path afterwards showed it could not "
        "settle anything: `hybrid_reorder` changes the order and leaves `score_total` "
        "intact, while the ungated tier admits on `score_total >= 0.5`, so fusion pulls "
        "lower-scoring papers into the window where they fail the threshold — shrinking "
        "the shown set. At precision 0.379 each shown paper is worth 3p-2 = -0.86, so "
        "shrinking helps for reasons unrelated to ranking. It would be a finding about "
        "the 0.5 tier rule wearing a finding about fusion. The question that IS worth "
        "money moved to where fusion actually ships: it has been on in every headline "
        "since PR #30 and had never been ablated inside the MEASURED configuration, whose "
        "keep decision rests on NR-11's pre-rescore argument. That ablation ran on "
        "2026-08-16 and came back +0.00 net@2/case, CI [-1.00, +0.96], inside the 0.74 "
        "floor, while changing 25/25 returned sets — and the gate-free measure puts 8.80 "
        "actionable papers in front of the gate with fusion against 8.72 without. So the "
        "preset carries a component we cannot show earns its place [NR-35]. It stays "
        "because every published number was measured with it and this file's job is to "
        "reproduce that configuration; dropping it on an unresolved result would trade a "
        "documented uncertainty for an undocumented divergence"
    ),
    "ranking.w_embedding": (
        "1.5 in the template, 1.5 in the preset, 1.5 in the headline since 2026-08-16 — and "
        "still 0.0 in the DATACLASS, which is what this entry now covers and all it covers. "
        "The measurement side is closed: two paired draws put 1.5 at +1.00/case over 0.0 "
        "[NR-38] and the headline was then re-measured there (+5.72 over 25 repositories, "
        "paired +4.16 against the agentic baseline). What remains is a genuine "
        "surface split, and it is deliberate. The dataclass default is what a user gets "
        "for a key they never wrote, including in the keyless out-of-the-box arm that "
        "scores −8.12; that arm was measured at 0.0, so raising the dataclass would put an "
        "unmeasured value under a published negative number. This field is also the only "
        "one whose behaviour depends on the INSTALL — without the `embeddings` extra it is "
        "inert, and the product says so at `cli.py` rather than failing, so a dataclass "
        "default of 1.5 would read as a promise the install may not keep"
    ),
    "triage.finescale.timeout": (
        "60 in the eval against 30 shipped, and bounded by a guard both sides run: "
        "enough_scored() refuses the whole stage below 50% success, so a timeout "
        "difference cannot quietly demote a band — it either changes nothing or is loud"
    ),
}


def coverage_gaps() -> list[Divergence]:
    """Config leaves that are neither compared nor excused.

    This is the check that makes the pass exhaustive rather than hand-scoped: a new config
    field fails the audit until somebody decides whether the benchmark measures it. C-14b
    is the argument — a guard that reads only where a bug was last found reports clean
    about everywhere it never looked.
    """
    classified = set(BENCHMARK_HEADLINE) | set(NOT_UNDER_TEST)
    out = []
    for key in sorted(set(config_leaves()) - classified):
        out.append(
            Divergence(
                key,
                repr(effective_shipped()[key]),
                "unclassified",
                "UNCLASSIFIED",
                "new config field: add it to BENCHMARK_HEADLINE or NOT_UNDER_TEST",
            )
        )
    for key in sorted(classified - set(config_leaves())):
        out.append(
            Divergence(
                key,
                "gone",
                "still classified",
                "UNCLASSIFIED",
                "classified field no longer exists in the config tree — stale entry",
            )
        )
    return out


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        try:
            return abs(float(a) - float(b)) < 1e-9
        except (TypeError, ValueError):
            return False
    return bool(a == b)


def surface_divergences() -> list[Divergence]:
    """Fields where `rr init`'s template and the dataclass default disagree.

    Not automatically wrong — a template is allowed to be opinionated — but it makes
    "the shipped default" ambiguous, and an ambiguous baseline is what let the config
    pass answer the wrong question for a month.
    """
    classes, template = config_leaves(), template_values()
    out = []
    for key, written in sorted(template.items()):
        if key not in classes or _same(classes[key], written):
            continue
        out.append(
            Divergence(
                key,
                repr(classes[key]),
                repr(written),
                "declared" if key in DECLARED else "DIVERGENT",
                DECLARED.get(key, "template overrides the dataclass with no reason recorded"),
            )
        )
    return out


def preset_divergences() -> list[Divergence]:
    """Fields where `rr init --measured` does NOT reproduce the measured configuration.

    Exact, with **no exemption mechanism at all** — unlike the default template, which is
    allowed to differ from the benchmark for reasons it declares. This one is not: its
    entire purpose is to be the configuration behind the published number, so any
    difference is a documentation defect by definition. Told a user "this gets you
    +5.42", we owe them the actual arms of that run.
    """
    effective = {**config_leaves(), **measured_preset()}
    out = []
    for key, measured in sorted(BENCHMARK_HEADLINE.items()):
        if key in effective and _same(effective[key], measured):
            continue
        out.append(
            Divergence(
                key,
                repr(effective.get(key, "<missing>")),
                repr(measured),
                "PRESET DRIFT",
                "rr init --measured no longer reproduces the configuration it cites",
            )
        )
    return out


def config_divergences() -> list[Divergence]:
    """Fields where what a user runs and what the benchmark measures disagree."""
    shipped = effective_shipped()
    out = []
    for key, measured in sorted(BENCHMARK_HEADLINE.items()):
        if key not in shipped or _same(shipped[key], measured):
            continue
        out.append(
            Divergence(
                key,
                repr(shipped[key]),
                repr(measured),
                "declared" if key in DECLARED else "DIVERGENT",
                DECLARED.get(key, "not declared — the benchmark is not measuring the default"),
            )
        )
    return out


def _print_table(title: str, divs: list[Divergence], left: str, right: str) -> None:
    print(f"\n  {title}")
    if not divs:
        print("     (none)")
        return
    print(f"     {'field':<34} {left:>14} {right:>14}  status")
    for d in divs:
        print(f"     {d.name:<34} {d.product:>14} {d.benchmark:>14}  {d.status}")


def pass_config() -> list[Divergence]:
    print("\n" + "=" * 78)
    print("2. CONFIGURATION — is the benchmark measuring what a user actually runs?")
    print("=" * 78)

    leaves, template = config_leaves(), template_values()
    print(
        f"\n  {len(leaves)} config leaves: {len(BENCHMARK_HEADLINE)} compared against the "
        f"benchmark, {len(NOT_UNDER_TEST)} excused."
    )
    print(f"  `rr init` writes {len(template)} of them; the rest fall back to the dataclass.")

    gaps = coverage_gaps()
    _print_table("a) coverage — every leaf classified?", gaps, "shipped", "bucket")

    surfaces = surface_divergences()
    _print_table("b) surfaces — template vs dataclass", surfaces, "dataclass", "rr init")

    divs = config_divergences()
    _print_table("c) measurement — effective vs benchmark", divs, "user runs", "measured")

    preset = preset_divergences()
    _print_table("d) `rr init --measured` — preset vs benchmark", preset, "preset", "measured")
    if not preset:
        print(
            f"        all {len(BENCHMARK_HEADLINE)} measured fields reproduced — the "
            "configuration we recommend is the one we measured."
        )

    findings = [d for d in gaps + surfaces + divs + preset if d.status != "declared"]
    # A field can be flagged by both (b) and (c) — `ranking.w_embedding` is — and the
    # reason is one reason, so print it once.
    seen: set[str] = set()
    for d in surfaces + divs:
        if d.status == "declared" and d.name not in seen:
            seen.add(d.name)
            print(f"\n  declared  {d.name}: {d.note}")
    return findings


# ── pass 3: blast radius ───────────────────────────────────────────────────

_MODERN = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")


def _split_v(arxiv_id: str) -> str:
    return arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id


# The commit that gave the HyDE candidate merge the shared id rule. Duplicates in a run
# recorded BEFORE this are the C-12 defect's history; duplicates after it are a live bug,
# and a reader cannot tell which without the date. Reporting a raw count conflates them.
HYDE_MERGE_FIX = "2026-08-14T03:35Z"  # cae8c88, `known = {dedup_id(...)}`
_RUN_STAMP = re.compile(r"(\d{8}T\d{6})Z")


def _run_stamp(fname: str) -> str:
    """The run's UTC timestamp, from its filename, as an ISO-ish string for comparison."""
    m = _RUN_STAMP.search(fname)
    if not m:
        return ""
    raw = m.group(1)
    return f"{raw[0:4]}-{raw[4:6]}-{raw[6:8]}T{raw[9:11]}:{raw[11:13]}Z"


def _frozen_pool_papers(blob: Any) -> tuple[list[dict[str, Any]], str]:
    """Papers out of a frozen-pool file, with the format it was read as.

    Two formats exist: v1 froze the pool AFTER ranking (`ranked`), v2 freezes it before
    (`candidates`, the change §8.8 argues for). A scanner that knows only the current key
    reports the older pools as *empty*, which reads identically to *clean* — void, not
    null, applied to this project's own artifacts. So the format is returned, an
    unrecognised one says so, and a shape that only PARTLY parses is reported as partial
    rather than as however many papers happened to come out. The first version of this
    function made exactly that mistake: v1 stores ``[paper, score]`` pairs, not papers, so
    it read 1,250 papers as 0 and printed "0 dup" about a pool it had not looked at.
    """
    if not isinstance(blob, dict):
        return [], "unrecognised"
    if isinstance(blob.get("candidates"), list):
        items = blob["candidates"]
        papers = [p for p in items if isinstance(p, dict) and p.get("arxiv_id")]
        return papers, "v2/candidates" if len(papers) == len(items) else "v2/PARTIAL"
    if isinstance(blob.get("ranked"), list):
        items = blob["ranked"]
        papers = []
        for entry in items:
            # v1 rows are (paper, score) pairs; unwrap before looking for an id.
            paper = entry[0] if isinstance(entry, list) and entry else entry
            if isinstance(paper, dict) and paper.get("arxiv_id"):
                papers.append(paper)
        return papers, "v1/ranked" if len(papers) == len(items) else "v1/PARTIAL"
    return [], "unrecognised"


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
    by_run: collections.Counter[str] = collections.Counter()
    for fname, _case, _base in dup_groups:
        by_run[fname] += 1
    for fname, n in by_run.most_common():
        stamp = _run_stamp(fname)
        era = "PRE-FIX" if stamp and stamp < HYDE_MERGE_FIX else "LIVE DEFECT"
        print(f"       {n:>3} in {fname[-34:]:<36} {stamp or '(no stamp)':<18} {era}")
    if by_run:
        print(f"       (the HyDE candidate merge got the shared id rule at {HYDE_MERGE_FIX})")

    print("\n  e) frozen pools — a duplicate here is reused by every arm that shares it:")
    for pool_dir in sorted((ROOT / "evals" / ".work").glob("pool-*")):
        n_papers = n_dups = 0
        formats: collections.Counter[str] = collections.Counter()
        for path in sorted(pool_dir.glob("*.json")):
            try:
                blob = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                formats["unreadable"] += 1
                continue
            papers, fmt = _frozen_pool_papers(blob)
            formats[fmt] += 1
            n_papers += len(papers)
            counted = collections.Counter(dedup_id(p["arxiv_id"]) for p in papers)
            n_dups += sum(1 for n in counted.values() if n > 1)
        shape = ", ".join(f"{k}x{v}" for k, v in sorted(formats.items()))
        flag = " !" if n_dups or "unrecognised" in formats or "unreadable" in formats else "  "
        print(f"     {flag} {pool_dir.name:<14} {n_papers:>6} papers  {n_dups} dup  [{shape}]")


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
    # Non-zero on a finding, so this can be a gate rather than a wall of text somebody
    # skims. The same reasoning as the pool-provenance and digest-width guards: a check
    # that reports a problem and then says everything is fine gets read as fine.
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
