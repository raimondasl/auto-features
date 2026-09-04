"""Which judge is right? Adoption is the only label no model produced. [NR-56]

NR-52 measured GPT-5.5 and Sonnet disagreeing about the comparator margin badly enough to flip
its sign; NR-53 showed that disagreement is a real property of the judges (self-kappa 0.798
against cross-judge 0.199), not sampling. Both are **reliability** results. Neither can say
which judge is *right*, and no amount of adding judges ever will — that needs a label from
outside the models.

P6 built one: `ids(HEAD) - ids(T0)` over a repo's own documentation is a set of papers the
project **verifiably took up**, mined from git history with no model involved. 31 usable
adoptions across 6 cases survive the self-citation and too-new filters.

**Recall alone cannot rank judges, and that is the whole design problem here.** Adoption
supplies ground-truth POSITIVES only. A judge that calls everything actionable scores 100% on
them and is worthless. So this pairs each adopted paper with **matched controls** — papers from
the same repo's candidate pool, published *before* the same T0 (so they could have been
adopted), that were not in the T0 bibliography and were not adopted by HEAD. Both judges score
positives and controls against the identical T0 repo context, and the statistic is the **gap**:

    discrimination = P(actionable | adopted) - P(actionable | matched control)

A lenient judge lifts both terms and gains nothing. A judge that actually tracks what a
repository will take up separates them.

**PRE-REGISTERED, written before any control was drawn or any Sonnet verdict bought.**

* **Primary:** the discrimination gap per judge, with a Wilson interval on each rate.
* **The judges are ranked by gap, not by recall.** GPT already scores 19/31 = 61.3% on the
  positives; that number is meaningless on its own and is not the comparison.
* **If both gaps are < 0.20** — neither judge separates adopted papers from matched controls,
  and the entire measurement apparatus rests on a signal that does not track the product's
  stated goal. That is the most consequential outcome available here and it is registered as a
  named result rather than a disappointment.
* **If the gaps differ by >= 0.15** the larger is the better instrument for this benchmark, and
  NR-52's judge-dependence should be read through it: the more discriminating judge's margin is
  the one to quote.
* Below that difference the two are not separated by this test at this n, which is reported as
  such rather than resolved by preference.

**Limits, stated now because they do not improve with the result.**

* **n = 31 positives across 6 cases, and `graph` alone contributes 13.** C-7's shape: this can
  fail to separate judges without saying anything about judges in general.
* **"Not adopted" is a noisy negative.** Adoption is sparse, so a control may be a perfectly
  good paper the project simply never got to. That biases both gaps *downward* equally, so it
  is safer for the comparison than for the absolute levels.
* **Adoption measures what a repository did, not what it should have done.** A judge could be
  right about value and wrong about adoption. This is the best available anchor, not truth.
* Judging uses `use_cache=False` throughout, mandatory rather than optimisation: `judge_paper`
  keys its cache on `(model, repo, paper_id)` and **not** on the context, so a T0 verdict
  written into the shared gold cache would overwrite the HEAD verdict for the same paper. That
  exact write once took `rag` from 5 targets to 0.

    uv run python evals/judge_validity_adoption.py --plan     # $0: the sample and the cost
    uv run python evals/judge_validity_adoption.py --judge    # ~$5, resumable
    uv run python evals/judge_validity_adoption.py            # $0: the gaps

**The last of those writes a reproduction, not the published record.** `report()` is the
default action, and it used to overwrite `evals/judge_validity_adoption.json` — the file whose
numbers `RESULTS.md`, `evals/README.md` and `PLANS.md` quote — on every bare invocation. It now
writes through `judge_validity_pool.artifact_path(source, scheme)`, which resolves the
NR-56/57 reproduction to `.work/repro/` and refuses the published path outright. A reproduction
that overwrites what it reproduces cannot disagree with it, and disagreeing is the only reason
to run one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

EVALS = Path(__file__).resolve().parent
sys.path.insert(0, str(EVALS))
sys.path.insert(0, str(EVALS.parent / "src"))

from metrics import roc_auc  # noqa: E402

from reporadar.paper_id import dedup_id  # noqa: E402

WORK = EVALS / ".work"
ADOPTIONS = WORK / "adoptions.json"
SEEDS = WORK / "adoption_seeds.json"
POOL = WORK / "pool-cut100"
VERDICTS = WORK / "judge_validity_verdicts.json"
# The PUBLISHED record of NR-56/57, kept as a name so the code that must not write it can say
# so. `judge_validity_pool.FROZEN_RECORDS` is what enforces it; this is where it is read.
FROZEN = EVALS / "judge_validity_adoption.json"

GPT_MODEL = "gpt-5.5"
SONNET_MODEL = "claude-sonnet-5"
CONTROLS_PER_POSITIVE = 4
SEED = 20260901

FLAT_GAP = 0.20  # below this for BOTH judges: neither tracks the product's goal
SEPARATES = 0.15  # gap difference at or above which one judge is the better instrument


def wilson(k: int, n: int) -> tuple[float, float] | None:
    """A Wilson interval, or None when there is nothing to compute one from.

    None rather than `(nan, nan)`: `json.dumps` writes a bare `NaN`, which is not JSON and is
    rejected by every parser outside Python. §10 step 7 publishes this artefact as a datasheet,
    so a judge with no verdicts used to produce an unparseable file that looked complete.
    `judge_date_stratify.wilson` already returns None at n = 0; this matches it.
    """
    if n == 0:
        return None
    p, z = k / n, 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0.0, c - h), 4), round(min(1.0, c + h), 4))


def excludes(interval: tuple[float, float] | list[float] | None, value: float) -> bool | None:
    """Does a bootstrap interval exclude *value*? Both sides, always.

    One implementation because there were three, and one of them tested a single side:
    `gap_excludes_zero` read `lo > 0`, so a judge whose whole interval sat BELOW zero — one
    that ranks matched controls above the papers a project adopted, the most interesting
    possible result here — was reported as "not shown to discriminate", the same words used
    for a judge that separates nothing. C-9/C-12/C-14's rule: one invariant, one implementation.
    """
    if interval is None:
        return None
    lo, hi = interval
    if lo != lo or hi != hi:  # NaN in, no answer out
        return None
    return bool(lo > value or hi < value)


def cluster_bootstrap_auc(
    positives: list[tuple[str, float]],
    controls: list[tuple[str, float]],
    *,
    iters: int = 5000,
    seed: int = SEED,
) -> dict[str, Any]:
    """§6.4's primary endpoint: AUC with the **repository** as the resampling unit.

    The estimator already in this file resamples positives and controls independently, one
    paper at a time. NR-56 drew 31 positives from 6 repositories with `graph` supplying 13,
    and NR-57's 35 came from 9 with the same shape — so a paper-level interval treats 13
    papers out of one project's bibliography as 13 independent draws and reports a precision
    the design never had. Resampling repositories makes the interval answer "if we had drawn
    different REPOSITORIES", which is the question the pool is built to ask.

    The design effect is the **realised** ratio of the two bootstrap variances, not a figure
    derived from an assumed ICC — the ICC is precisely the quantity nobody knows here, and
    an assumed one would put the answer into the uncertainty estimate by hand.

    Inputs are `(repository, score)` pairs carrying the judge's **ordinal** rubric score, not
    a thresholded one: a threshold would reintroduce the level the endpoint exists to avoid.
    """
    by_cluster: dict[str, tuple[list[float], list[float]]] = {}
    for repo, score in positives:
        by_cluster.setdefault(repo, ([], []))[0].append(score)
    for repo, score in controls:
        by_cluster.setdefault(repo, ([], []))[1].append(score)
    clusters = sorted(by_cluster)
    pos_scores = [s for _, s in positives]
    ctl_scores = [s for _, s in controls]
    point = roc_auc(pos_scores, ctl_scores)
    biggest = max((len(v[0]) for v in by_cluster.values()), default=0)
    out: dict[str, Any] = {
        "auc": round(point, 4) if point == point else None,
        "n_positives": len(positives),
        "n_controls": len(controls),
        "n_clusters": len(clusters),
        "largest_cluster_share": (round(biggest / len(positives), 4) if positives else None),
    }
    if len(clusters) < 2 or not positives or not controls:
        out["_refused"] = "fewer than two clusters — a cluster bootstrap has nothing to resample"
        return out

    rng = random.Random(seed)
    clustered: list[float] = []
    for _ in range(iters):
        pos_draw: list[float] = []
        ctl_draw: list[float] = []
        for _ in clusters:
            pick = clusters[rng.randrange(len(clusters))]
            pos_draw.extend(by_cluster[pick][0])
            ctl_draw.extend(by_cluster[pick][1])
        value = roc_auc(pos_draw, ctl_draw)
        if value == value:
            clustered.append(value)

    # The paper-level interval is computed only so the design effect is a measured ratio.
    # It is NOT reported as an interval: reporting both invites quoting the narrower one.
    papers: list[float] = []
    n1, n2 = len(pos_scores), len(ctl_scores)
    for _ in range(iters):
        p = [pos_scores[rng.randrange(n1)] for _ in range(n1)]
        c = [ctl_scores[rng.randrange(n2)] for _ in range(n2)]
        value = roc_auc(p, c)
        if value == value:
            papers.append(value)

    if len(clustered) < 100:
        out["_refused"] = "too few usable bootstrap draws"
        return out
    clustered.sort()
    lo = clustered[int(0.025 * len(clustered))]
    hi = clustered[int(0.975 * len(clustered))]
    se = statistics.pstdev(clustered) if len(clustered) > 1 else float("nan")
    se_paper = statistics.pstdev(papers) if len(papers) > 1 else float("nan")
    out["ci95"] = [round(lo, 4), round(hi, 4)]
    out["excludes_half"] = excludes((lo, hi), 0.5)
    out["se"] = round(se, 4)
    out["design_effect"] = (
        round((se / se_paper) ** 2, 3) if se_paper and se_paper == se_paper else None
    )
    # 0.5 + (z_{.975} + z_{.80}) * SE — what this n could have detected at 80 % power, so a
    # CI spanning 0.5 can be read as "no discrimination" or "not enough repositories" rather
    # than collapsing the two.
    out["min_detectable_auc_80pct"] = round(0.5 + 2.80 * se, 4) if se == se else None
    return out


# ---------------------------------------------------------------------------------------
# Cache isolation (PREREG-judge-validity-pool section 4 and 7)
# ---------------------------------------------------------------------------------------
JUDGE_CACHE = EVALS / "cache" / "judge"
SECOND_JUDGE_CACHE = WORK / "second_judge"


def t0_namespace(model: str) -> str:
    """The `cache_as` directory a T0 verdict for *model* is written under.

    One declaration, used both by the call site in `judge()` and by the exclusion in
    `protected_partition()`. They were two independent spellings of the same string; a change
    to either alone turns the gate back into the self-tripping version below, or — worse —
    silently permits writes to a directory nothing is writing to.
    """
    return f"{model}#t0"


def permitted_namespaces() -> frozenset[str]:
    """Directories under the second judge's cache this study is allowed to create.

    **Sonnet only.** GPT is called through `judge_paper(..., use_cache=False)` and writes
    nothing anywhere, so `.work/second_judge/gpt-5.5#t0` is not a namespace this study owns —
    it is PROTECTED, and a file appearing there is a leak like any other.
    """
    return frozenset({t0_namespace(SONNET_MODEL)})


def protected_partition() -> dict[Path, frozenset[str]]:
    """Which cache roots are watched, and which top-level names inside them are not.

    §7 requires both of two things that read as contradictory if a root is treated as
    indivisible: *"a separate namespace for the second judge"* — which the study must be able
    to write — and *"before/after hashes of both cache roots as a blocking gate"*. Watching
    `.work/second_judge` whole made the second requirement forbid the first: `second_verdict`
    writes T0 verdicts into `.work/second_judge/<model>#t0/`, inside the fingerprinted root, so
    the gate fired on exactly the runs that bought something — and because it raised first, the
    gold-set guard after it never ran at all. A gate whose true-negative rate is zero gets
    switched off, and then the write that once took `rag` from 5 gold targets to 0 is silent.

    So the unit is a partition, not a root. `evals/cache/judge` is protected in full and has no
    exclusion, ever.
    """
    return {JUDGE_CACHE: frozenset(), SECOND_JUDGE_CACHE: permitted_namespaces()}


def assert_namespace_ownership(roots: dict[Path, frozenset[str]] | None = None) -> None:
    """Refuse to write into a permitted namespace this study does not own.

    An exclusion has to be justified by ownership rather than by spelling. `.work/second_judge`
    holds 1,504 verdicts that eight other modules read back by name, so a directory whose name
    happens to match `<model>#t0` may be somebody else's data — and excluding it from the gate
    would make overwriting that data the one thing the gate cannot see.
    """
    for root, permitted in (roots or protected_partition()).items():
        for name in sorted(permitted):
            ns = root / name
            if not ns.exists():
                continue
            marker = ns / "_NAMESPACE.json"
            if not marker.is_file():
                raise SystemExit(
                    f"{ns} exists but carries no _NAMESPACE.json.\n"
                    "  This directory is excluded from the cache gate, so anything written\n"
                    "  into it is invisible to the guard. That is only safe if this study owns\n"
                    "  it. Move it aside, or write the marker if it really is this study's."
                )


def claim_namespaces(roots: dict[Path, frozenset[str]] | None = None) -> None:
    """Create each permitted namespace with its ownership marker, before anything is bought."""
    for root, permitted in (roots or protected_partition()).items():
        for name in sorted(permitted):
            ns = root / name
            ns.mkdir(parents=True, exist_ok=True)
            marker = ns / "_NAMESPACE.json"
            if not marker.is_file():
                marker.write_text(
                    json.dumps(
                        {
                            "study": "judge-validity-pool",
                            "why": (
                                "T0 verdicts. Excluded from the cache-isolation gate because "
                                "PREREG §7 registers a separate namespace for the second judge "
                                "as legitimate; everything else under this root is protected."
                            ),
                        },
                        indent=1,
                    ),
                    encoding="utf-8",
                )


def prepare_isolation() -> tuple[tuple[Path, ...], dict[str, str]]:
    """Establish ownership and take the "before" manifest, BEFORE the first purchase.

    Ownership first, and never after the last verdict: the exclusion that lets this run write
    T0 verdicts is only safe if the directory it excludes belongs to this study, and checking
    that at the end means discovering somebody else's data after writing into it.
    `claim_namespaces` stamps the marker, so the ownership check must precede it — reversing
    the two would make the run claim whatever it found.
    """
    roots = tuple(protected_partition())
    assert_namespace_ownership()
    claim_namespaces()
    return roots, cache_manifest(*roots)


def isolation_failures(
    cache_before: str | dict[str, str],
    cache_roots: tuple[Path, ...],
    targets_before: Any,
    targets_after: Any,
) -> list[str]:
    """Evaluate BOTH guards and return every failure, rather than raising at the first.

    The cache guard used to raise directly, so on any run that tripped it the gold-set check
    never executed at all — and the two do not imply one another in either direction.
    `resolve_targets` sees only ids the baseline picked whose gold verdict scores >= 2, so a
    write for an unpicked paper moves the cache and leaves the gold set alone, while a score
    crossing 2 for a picked paper does the opposite. Reporting only the first means triaging
    the wrong one.

    Split out of `judge()` so it is testable without buying a verdict.
    """
    failures: list[str] = []
    try:
        assert_caches_untouched(cache_before, cache_roots)
    except SystemExit as exc:
        failures.append(str(exc))
    if targets_after != targets_before:
        failures.append(
            "THE GOLD SET MOVED — T0 verdicts leaked into the shared cache and changed which "
            "papers count as gold targets. Every recall number downstream is now different."
        )
    return failures


def cache_manifest(*roots: Path) -> dict[str, str]:
    """`{root/relpath: 'size|mtime_ns'}` for every watched file, exclusions applied WHILE walking.

    Applied while walking rather than by pre-enumerating the permitted directories: a list
    computed before the run cannot contain a directory the run is about to create, which is
    precisely the case the exclusion exists for.

    The key carries the root as well as the relative path, so the same relative name under two
    roots cannot collide into one entry — a collision would let a file added under one root and
    removed from the other cancel out, which is the shape of silent failure this gate exists to
    catch. `<parent>/<name>` rather than `<name>` alone, and the ids are asserted distinct.
    """
    partition = protected_partition()
    ids = [f"{r.parent.name}/{r.name}" for r in roots]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"two watched cache roots share an id and would collide: {ids}")
    out: dict[str, str] = {}
    for root, root_id in zip(roots, ids, strict=True):
        if not root.exists():
            continue
        permitted = partition.get(root, frozenset())
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(root)
            if rel.parts and rel.parts[0] in permitted:
                continue
            stat = path.stat()
            out[f"{root_id}/{rel.as_posix()}"] = f"{stat.st_size}|{stat.st_mtime_ns}"
    return out


def cache_fingerprint(*roots: Path) -> str:
    """A hash of every cached verdict's path, size and mtime under *roots*.

    This is a **blocking gate**, not hygiene. `judge_paper` keys its cache on
    (model, repo, paper_id) and NOT on the context it was given, so a T0 verdict written
    into the shared gold cache silently overwrites the HEAD verdict for the same paper. That
    exact write once took `rag` from 5 gold targets to 0 -- and it is invisible afterwards,
    because the file is still there and still parses.

    Path, size and mtime rather than content: the cache holds thousands of small files and
    this runs twice per judging session, while any write at all moves an mtime.
    """
    manifest = cache_manifest(*roots)
    body = "\n".join(f"{k}|{v}" for k, v in sorted(manifest.items()))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _manifest_diff(before: dict[str, str], after: dict[str, str]) -> dict[str, list[str]]:
    return {
        "added": sorted(set(after) - set(before)),
        "removed": sorted(set(before) - set(after)),
        "modified": sorted(k for k in set(before) & set(after) if before[k] != after[k]),
    }


def assert_caches_untouched(before: str | dict[str, str], roots: tuple[Path, ...]) -> None:
    """Raise if anything WATCHED under *roots* changed. Called after every judging run.

    *before* may be a fingerprint (the cheap compare) or a manifest from `cache_manifest`. Pass
    the manifest where you can: a hash says only that something moved, and the triage that
    follows a failure needs to know *which* root — the two guards in `judge()` do not imply one
    another. `resolve_targets` sees only ids the baseline picked whose gold verdict scores >= 2,
    so a gold-cache write for an unpicked paper, or a score change on the same side of 2, moves
    this fingerprint while leaving the gold set identical. Read the diff, do not infer the cause
    from which guard fired.
    """
    detail = ""
    if isinstance(before, dict):
        diff = _manifest_diff(before, cache_manifest(*roots))
        if not any(diff.values()):
            return
        for label, paths in diff.items():
            if paths:
                shown = ", ".join(paths[:20])
                more = f" (+{len(paths) - 20} more)" if len(paths) > 20 else ""
                detail += f"\n  {label} ({len(paths)}): {shown}{more}"
    elif cache_fingerprint(*roots) == before:
        return
    raise SystemExit(
        "THE JUDGE CACHE MOVED during a T0 judging run.\n"
        "  T0 verdicts must never enter the shared gold cache: judge_paper keys on\n"
        "  (model, repo, paper_id) and not on the context, so a write here overwrites\n"
        "  the HEAD verdict for the same paper, and every downstream number changes\n"
        "  silently. Check use_cache=False at every call site, and that the second\n"
        "  judge writes under its own cache_as namespace.\n"
        f"  roots: {[str(r) for r in roots]}" + detail
    )


# ---------------------------------------------------------------------------------------
# Arm-neutral controls (PREREG-judge-validity-pool section 4)
# ---------------------------------------------------------------------------------------
CONTROL_SCHEMES = ("pool", "arxiv-window")
CONTROL_SCHEME = "pool"
LISTING_PER_WINDOW = 200


def enrich_positives(
    rows: list[dict[str, Any]], *, fetch: Any = None
) -> tuple[list[dict[str, Any]], list[str]]:
    """Attach the primary arXiv category and submission date every positive needs.

    `arxiv_window_controls` matches a control to its positive on (primary category, half-year
    of submission). **Neither field exists on a mined adoption row** -- mining reads git, not
    arXiv -- and nothing else in the codebase produced them. Without this the control drawer
    skipped every positive and returned an EMPTY control set: not an error, just no negatives,
    and an AUC computed against nothing.

    That is the silent zero this project keeps being bitten by, so the *missing* ids are
    returned rather than dropped. A caller that ignores them is choosing to, in writing.
    """
    from reporadar import collector as collector_mod

    wanted = sorted({dedup_id(str(r["id"])) for r in rows})
    getter = fetch or collector_mod.collect_by_ids
    fetched = {dedup_id(str(p.get("arxiv_id", ""))): p for p in getter(wanted)}

    enriched: list[dict[str, Any]] = []
    missing: list[str] = []
    for row in rows:
        pid = dedup_id(str(row["id"]))
        paper = fetched.get(pid)
        if paper is None:
            missing.append(pid)
            continue
        # arXiv's own primary category, and `categories[0]` only as a fallback. The two are not
        # the same thing: `categories` is feed tag order, so `categories[0]` is a guess. §4
        # matches a control to its positive on the PRIMARY category, and getting it wrong lets
        # cross-listed papers dominate the negative class — a `cs.LG` positive drawing controls
        # whose primary is `stat.ML` or `cs.CV` makes the AUC partly a measure of
        # primary-versus-cross-list rather than of adoption.
        primary = str(paper.get("primary_category") or "")
        categories = paper.get("categories") or []
        if not primary and categories:
            primary = str(categories[0])
        if not primary or not paper.get("published"):
            missing.append(pid)
            continue
        enriched.append(
            {
                **row,
                "primary_category": primary,
                "published": str(paper["published"])[:10],
                "paper": paper,
            }
        )
    return enriched, missing


def refuse_an_empty_control_set(positives: list[Any], controls: list[Any]) -> None:
    """A control set of zero is never a result. Raise instead of reporting one.

    Section 5's primary is an AUC of positives against controls; with no controls it is
    undefined, and every downstream figure would be computed from nothing while looking
    exactly like a completed run.
    """
    if positives and not controls:
        raise SystemExit(
            f"{len(positives)} positives drew ZERO controls.\n"
            "  An AUC against an empty negative class is not a null result, it is no\n"
            "  result. Check that the positives were enriched (primary_category and\n"
            "  published) and that the arXiv listing returned anything for their category\n"
            "  and half-year."
        )


def half_year_bounds(published: str) -> tuple[str, str]:
    """The half-year containing *published*, as arXiv `submittedDate` bounds.

    Matching on a half-year rather than on an exact date is what makes a control a paper the
    project *could* have adopted at the same moment: same field, same window, same state of
    the literature. A tighter window would run the listing dry in small categories.
    """
    day = datetime.fromisoformat(published[:10]).date()
    if day.month <= 6:
        lo, hi = day.replace(month=1, day=1), day.replace(month=6, day=30)
    else:
        lo, hi = day.replace(month=7, day=1), day.replace(month=12, day=31)
    return lo.strftime("%Y%m%d0000"), hi.strftime("%Y%m%d2359")


LISTING_SLICES = 6  # one per month of the half-year
FULL_ENUMERATION_CAP = 30000  # arXiv's own paging ceiling; only reached under depth="full"


def sub_windows(lo: str, hi: str, slices: int = LISTING_SLICES) -> list[tuple[str, str]]:
    """Split a `submittedDate` window into *slices* contiguous, non-overlapping parts.

    Every day of the window falls in exactly one slice, and the parts are returned oldest
    first. Boundaries land on day edges so a paper cannot fall between two slices or into both.
    """
    start = datetime.strptime(lo[:8], "%Y%m%d").date()
    end = datetime.strptime(hi[:8], "%Y%m%d").date()
    span = (end - start).days + 1
    if slices < 2 or span < slices:
        return [(lo, hi)]
    out: list[tuple[str, str]] = []
    for i in range(slices):
        a = start + timedelta(days=(span * i) // slices)
        b = start + timedelta(days=(span * (i + 1)) // slices - 1)
        out.append((a.strftime("%Y%m%d0000"), b.strftime("%Y%m%d2359")))
    return out


def arxiv_window_listing(
    category: str,
    lo: str,
    hi: str,
    *,
    want: int = LISTING_PER_WINDOW,
    archive: Path | None = None,
    depth: str = "stratified",
    slices: int = LISTING_SLICES,
) -> list[dict[str, Any]]:
    """Papers in *category* submitted in the window, drawn ACROSS it rather than off its end.

    This asked arXiv for `want` results sorted by `submittedDate` — and arXiv's default sort
    order is **descending**, so it returned the *newest* 200 of the window. Measured: `cs.LG`
    H1-2021 holds **13,262** papers, so all 200 controls for a positive in that window came
    from its last few days. §4 registers "submitted in the same half-year" and names no cap and
    no ordering, so the cap was an unregistered narrowing that made the negative class arXiv's
    index order rather than the seed's.

    It also mattered in a specific direction. NR-43 measured actionability rising steadily with
    recency, 0.31 (2013) to 0.64 (2025), so controls drawn systematically months *newer* than
    the positive they are matched against are scored *higher* — compressing the gap toward the
    null §5 is pre-committed to reporting. Conservative, but by accident rather than design.

    **`depth="stratified"` (default)** splits the window into `slices` equal parts and takes
    `want // slices` from each, so the draw spans the whole half-year at the same request cost.
    Residual skew is to the end of each *slice* — days, not months.

    **`depth="full"`** enumerates the window completely and is what §4 would ask for if it were
    free. It is not: at arXiv's enforced 3 s minimum interval and a 100-record page, one busy
    window is ~6.6 minutes and a run needs 40-80 of them — **4.4 to 8.8 hours** of continuous
    third-party API access, before any throttling. arXiv returned 429 during the measurement
    that produced these figures. Measured 2026-09-03; the flag exists so the choice stays
    available rather than being decided by this docstring.

    Archived per (category, window) because this is the **negative class of the primary
    endpoint**. An AUC is a statement about positives against these papers; if the listing
    cannot be reproduced, neither can the number. The archive records every sub-query, what it
    asked for and what it returned, so a slice that hit its own cap is visible rather than
    inferred.
    """
    import arxiv

    from reporadar import collector as collector_mod

    if depth not in ("stratified", "full"):
        raise SystemExit(f"unknown listing depth {depth!r}; use 'stratified' or 'full'")

    parts = [(lo, hi)] if depth == "full" else sub_windows(lo, hi, slices)
    per_part = FULL_ENUMERATION_CAP if depth == "full" else max(1, want // len(parts))

    papers: list[dict[str, Any]] = []
    seen: set[str] = set()
    queries: list[dict[str, Any]] = []
    for plo, phi in parts:
        query = f"cat:{category} AND submittedDate:[{plo} TO {phi}]"
        search = arxiv.Search(
            query=query, max_results=per_part, sort_by=arxiv.SortCriterion.SubmittedDate
        )
        results = collector_mod._query_with_retry(collector_mod._shared_client(100), search)
        got = [collector_mod._result_to_paper(r) for r in results]
        queries.append(
            {
                "query": query,
                "requested": per_part,
                "returned": len(got),
                # A slice that returned exactly what it asked for was cut off there. Recorded
                # rather than inferred, because "200 of 13,262" and "200 of 200" are the same
                # number in an archive that only stores the count.
                "truncated": len(got) >= per_part,
            }
        )
        for paper in got:
            pid = dedup_id(str(paper.get("arxiv_id", "")))
            if pid and pid not in seen:
                seen.add(pid)
                papers.append(paper)

    if archive is not None:
        archive.mkdir(parents=True, exist_ok=True)
        (archive / f"{category.replace('.', '_')}-{lo[:6]}.json").write_text(
            json.dumps(
                {
                    "category": category,
                    "window": f"{lo}..{hi}",
                    "depth": depth,
                    "want": want,
                    "sub_queries": queries,
                    "n": len(papers),
                    "papers": papers,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return papers


def arxiv_window_controls(
    positives: list[dict[str, Any]],
    head_ids: dict[str, set[str]],
    seed: str,
    *,
    per_positive: int = CONTROLS_PER_POSITIVE,
    listing: Any = arxiv_window_listing,
    archive: Path | None = None,
) -> list[dict[str, Any]]:
    """Four controls per positive: same primary category, same half-year, never cited.

    **Why not the shipped candidate pool.** A pool built by RepoRadar is RepoRadar's own
    HEAD-seeded output, so a judge that is harsher on RepoRadar-shaped papers -- Sonnet, by
    a factor of 2.3 -- would be credited with "validity" for a property of the control set.
    Both adoption refutations landed on this point. An arXiv category listing is produced by
    arXiv, not by the system under test.

    *head_ids* is the repository's whole identifier set at HEAD, so a "control" the project
    actually went on to cite is excluded. Without it the negative class silently contains
    positives, which biases the AUC toward 0.5 -- i.e. toward the null this pool is
    pre-committed to reporting.
    """
    cache: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    by_case: dict[str, set[str]] = {}
    for row in positives:
        by_case.setdefault(row["case"], set()).add(dedup_id(str(row["id"])))

    out: list[dict[str, Any]] = []
    taken: dict[str, set[str]] = {}
    short: dict[int, list[str]] = {}
    for row in sorted(positives, key=lambda r: (r["case"], str(r["id"]))):
        case = row["case"]
        category = row.get("primary_category") or ""
        published = row.get("published") or ""
        if not category or not published:
            continue
        lo, hi = half_year_bounds(published)
        key = (category, lo, hi)
        if key not in cache:
            cache[key] = listing(category, lo, hi, archive=archive)
        if case not in head_ids:
            # Never `.get(case, set())`. §4 excludes a "control" the repository actually went on
            # to cite; an empty cited set turns that rule into a no-op, puts real citations into
            # the negative class, and drags the AUC toward 0.5 — the null this study is
            # pre-committed to reporting, arriving by accident and looking like a finding.
            raise SystemExit(
                f"{case}: no HEAD citation set. §4's never-cited rule cannot be applied, and\n"
                "  defaulting it to empty would put papers the repository cites into the\n"
                "  negative class."
            )
        cited = head_ids[case]
        chosen = taken.setdefault(case, set())
        pool = [
            paper
            for paper in cache[key]
            if dedup_id(str(paper.get("arxiv_id", ""))) not in cited
            and dedup_id(str(paper.get("arxiv_id", ""))) not in by_case[case]
            and dedup_id(str(paper.get("arxiv_id", ""))) not in chosen
            and str(paper.get("abstract") or "").strip()
            # §4 registers the positive's PRIMARY category, and `cat:` matches any category
            # including cross-lists — so the listing is a superset and the match is made here.
            # Without it a `cs.LG` positive draws controls whose primary is `stat.ML` or
            # `cs.CV`, and the AUC partly measures primary-versus-cross-list.
            and str(paper.get("primary_category") or "") == category
        ]
        pool.sort(key=lambda paper: dedup_id(str(paper.get("arxiv_id", ""))))
        random.Random(f"{seed}:{case}:{dedup_id(str(row['id']))}").shuffle(pool)
        drawn = pool[:per_positive]
        if not drawn:
            # `pool[:n]` accepts a short draw silently, and the global empty-set guard fires
            # only when EVERY positive drew nothing. A positive with no controls still enters
            # the point estimate and every bootstrap draw with no compensating negatives, and
            # cluster resampling duplicates it — so it inflates the positive class rather than
            # being dropped.
            raise SystemExit(
                f"{case}/{dedup_id(str(row['id']))} drew ZERO controls from "
                f"{category} {lo[:6]}..{hi[:6]} ({len(cache[key])} papers listed).\n"
                "  A positive with no negatives is not a smaller sample, it is an unmatched\n"
                "  observation that biases the estimate it enters."
            )
        short.setdefault(len(drawn), []).append(f"{case}/{dedup_id(str(row['id']))}")
        for paper in drawn:
            pid = dedup_id(str(paper.get("arxiv_id", "")))
            chosen.add(pid)
            out.append(
                {
                    "case": case,
                    "id": pid,
                    "t0": row.get("t0_commit_date") or row.get("t0_date", ""),
                    "for_positive": dedup_id(str(row["id"])),
                    "window": f"{lo}..{hi}",
                    "category": category,
                    "primary_category": str(paper.get("primary_category") or ""),
                    "paper": paper,
                }
            )
    for n in sorted(k for k in short if k < per_positive):
        # Reported rather than raised: §4 asks for four and a thin window is a property of the
        # population, not a defect. It has to be visible beside the AUC, because the realised
        # controls-per-positive distribution is what n2 actually was.
        print(f"  ! {len(short[n])} positive(s) drew only {n} of {per_positive} controls")
    return out


def adoptions() -> list[dict[str, Any]]:
    rows = json.loads(ADOPTIONS.read_text(encoding="utf-8"))
    return [r for r in rows if r.get("usable")]


def pool_papers(case: str) -> list[dict[str, Any]]:
    f = POOL / f"{case}.json"
    return json.loads(f.read_text(encoding="utf-8"))["candidates"] if f.is_file() else []


def pool_controls(rng: random.Random | None = None) -> list[dict[str, Any]]:
    """Matched negatives: same repo, publishable before T0, never adopted, not a T0 seed.

    'Publishable before T0' is the match that makes the control fair — a paper the project
    could not have adopted at T0 because it did not exist yet is not evidence about a judge.

    **Seeded per case, not from one shared stream.** The first version drew every case from a
    single `random.Random(SEED)`, so adding three repos to the adoption set re-shuffled the
    controls for all the existing ones — 139 fresh verdicts to answer a question about four new
    positives, and no way to compare the two runs on a stable sample. Per-case seeding makes
    each repo's controls a function of that repo alone, so the set grows by exactly what was
    added. *rng* is accepted and ignored for call-site compatibility.
    """
    pos = adoptions()
    seeds = json.loads(SEEDS.read_text(encoding="utf-8"))
    adopted_by_case: dict[str, set[str]] = {}
    t0_by_case: dict[str, str] = {}
    for r in pos:
        adopted_by_case.setdefault(r["case"], set()).add(dedup_id(str(r["id"])))
        t0_by_case[r["case"]] = r["t0_date"]

    out: list[dict[str, Any]] = []
    for case, t0 in sorted(t0_by_case.items()):
        cutoff = datetime.fromisoformat(t0)
        seen = adopted_by_case[case] | {dedup_id(str(i)) for i in seeds.get(case, [])}
        pool = []
        for p in pool_papers(case):
            pid = dedup_id(str(p["arxiv_id"]))
            if pid in seen or not str(p.get("abstract") or "").strip():
                continue
            pub = str(p.get("published") or "")[:10]
            if not pub:
                continue
            try:
                if datetime.fromisoformat(pub) >= cutoff:
                    continue
            except ValueError:
                continue
            pool.append({"case": case, "id": pid, "t0": t0_by_case[case], "paper": p})
        pool.sort(key=lambda r: r["id"])  # a deterministic base order before the draw
        random.Random(f"{SEED}:{case}").shuffle(pool)
        want = CONTROLS_PER_POSITIVE * len(adopted_by_case[case])
        out.extend(pool[:want])
    return out


def save_verdicts(have: dict[str, Any]) -> None:
    """Write the paid-verdict store through a temp file, never in place.

    These are **purchases**. The store holds 566 of them and is written every 20 items inside
    an hours-long loop; a truncating write interrupted at the wrong moment loses money that has
    already been spent and cannot be recovered from anywhere, because `use_cache=False` means
    the judge cache does not hold a copy.
    """
    VERDICTS.parent.mkdir(parents=True, exist_ok=True)
    tmp = VERDICTS.with_suffix(VERDICTS.suffix + ".tmp")
    tmp.write_text(json.dumps(have, indent=0), encoding="utf-8")
    os.replace(tmp, VERDICTS)


def controls(rng: random.Random | None = None, scheme: str | None = None) -> list[dict[str, Any]]:
    """The negative class, chosen by `CONTROL_SCHEME` — which used to be assigned and never read.

    `main()` set the module switch from `--controls` and nothing consulted it, so `arxiv-window`
    selected exactly nothing: the flag picked the OUTPUT PATH and the provenance label while the
    pool scheme drew the papers. `arxiv_window_controls`, `arxiv_window_listing` and
    `enrich_positives` had no production caller anywhere in the tree, so §4's registered scheme
    had never run on real data at all.

    **`pool` is this file's own body, unchanged**, because that branch reproduces the published
    NR-56/57 numbers and they must not move.

    **`arxiv-window` is not drawn here, and that is a routing decision rather than a stub.** §4's
    draw needs three things this function does not have and cannot honestly invent: the v2
    positives (NR-60 replaced v1's 35 with 94, of which 31-32 survive the cap and the contest),
    each repository's materialised HEAD citation set, and the verified `SEED_POOL`. All three
    live in `judge_validity_pool`, so the draw does too — and pointing at it is better than
    quietly drawing the wrong negative class under the right name.
    """
    resolved = scheme or CONTROL_SCHEME
    if resolved == "pool":
        return pool_controls(rng)
    if resolved == "arxiv-window":
        raise SystemExit(
            "the arm-neutral draw runs from judge_validity_pool.draw_controls(), not here.\n"
            "  It needs the v2 positives, the materialised HEAD citation sets and the verified\n"
            "  SEED_POOL; this function has the v1 legacy adoptions and the shipped candidate\n"
            "  pool, which is the negative class §4 exists to reject."
        )
    raise SystemExit(
        f"unknown control scheme {resolved!r}; registered schemes are {CONTROL_SCHEMES}"
    )


def load_verdicts() -> dict[str, dict[str, int]]:
    return json.loads(VERDICTS.read_text(encoding="utf-8")) if VERDICTS.is_file() else {}


def key(model: str, case: str, pid: str) -> str:
    """The verdict store's key. The paper id is normalised HERE and nowhere else.

    Every call site used to decide for itself: `dedup_id(str(r["id"])) if adopted else r["id"]`,
    repeated in plan(), rate() and scores(). It happened to be consistent, because the control
    drawers already store normalised ids — but `paper_id.dedup_id`'s own docstring records
    C-12, C-12b and C-14: three separate payments for one identity rule living in more than one
    place, the last of them for "a bare `split("v")[0]` doing the same job at eight further
    call sites". A versioned id reaching one of those branches resolves to no verdict, and a
    paper with no verdict is silently dropped from the numerator and the denominator both.

    *case* is passed through RAW — a legacy slug (`diffusion`) or a pool `full_name`
    (`huggingface/diffusers`). It is never lowercased and never replaced by a cluster label:
    the store was written under the raw string, so normalising it here would make every
    existing verdict unreachable and the whole stratum read as unjudged.
    """
    return f"{model}|{case}|{dedup_id(str(pid))}"


def plan() -> int:
    rng = random.Random(SEED)
    pos, ctl = adoptions(), controls(rng)
    print(f"positives (usable adoptions): {len(pos)}  {dict(Counter(r['case'] for r in pos))}")
    print(f"matched controls            : {len(ctl)}  {dict(Counter(r['case'] for r in ctl))}")
    have = load_verdicts()
    need = 0
    for model in (GPT_MODEL, SONNET_MODEL):
        n = sum(1 for r in pos if key(model, r["case"], dedup_id(str(r["id"]))) not in have)
        n += sum(1 for r in ctl if key(model, r["case"], r["id"]) not in have)
        need += n
        print(f"  {model:<18} needs {n} verdicts")
    print(f"\ntotal fresh verdicts: {need}  (~${need * 0.01:.0f}-{need * 0.03:.0f})")
    return 0


def judge() -> int:
    import judge as judge_mod
    from diagnose_triage import fetch_papers
    from mine_adoptions import CLONES, t0_context
    from run_judge_eval import load_dotenv
    from second_judge import second_verdict

    load_dotenv(EVALS / ".env")
    # Two independent guards, both blocking, both checked after the run rather than trusted.
    # `mine_adoptions.main` already carries the gold-set check; this function bought hundreds
    # of T0 verdicts without either of them.
    from build_hop_pool import resolve_targets

    cache_roots, cache_before = prepare_isolation()
    targets_before = resolve_targets()

    rng = random.Random(SEED)
    pos, ctl = adoptions(), controls(rng)
    have = load_verdicts()

    contexts: dict[str, str] = {}
    for r in pos:
        if r["case"] not in contexts:
            contexts[r["case"]] = t0_context(CLONES / r["case"], r["case"], r["t0"])

    fetched = fetch_papers(sorted({str(r["id"]) for r in pos}))
    items: list[tuple[str, str, dict[str, Any], str]] = []
    for r in pos:
        p = fetched.get(str(r["id"]))
        if p:
            items.append((r["case"], dedup_id(str(r["id"])), {"arxiv_id": r["id"], **p}, "adopted"))
    for r in ctl:
        items.append((r["case"], r["id"], r["paper"], "control"))

    bought = void = 0
    for n, (case, pid, paper, arm) in enumerate(items, start=1):
        ctx = contexts.get(case)
        if ctx is None:
            void += 1
            continue
        for model in (GPT_MODEL, SONNET_MODEL):
            k = key(model, case, pid)
            if k in have:
                continue
            try:
                if model == GPT_MODEL:
                    # use_cache=False: the gold cache is keyed on (model, repo, paper) and NOT
                    # on the context, so a T0 verdict would overwrite the HEAD verdict.
                    v = int(
                        judge_mod.judge_paper(case, ctx, paper, model=model, use_cache=False)[
                            "score"
                        ]
                    )
                else:
                    v = int(second_verdict(case, ctx, paper, model, cache_as=t0_namespace(model)))
                have[k] = {"score": v, "arm": arm}
                bought += 1
            except Exception as exc:  # noqa: BLE001 -- one bad paper must not lose the rest
                void += 1
                print(f"  ! {model} {case}/{pid}: {type(exc).__name__}: {str(exc)[:60]}")
        if n % 20 == 0 or n == len(items):
            save_verdicts(have)
            print(f"  [{n}/{len(items)}] bought {bought}, void {void}", flush=True)
    save_verdicts(have)
    print(f"\nbought {bought} verdicts; {void} void")

    # Both guards fire AFTER the verdicts are safely written, so a violation is reported
    # without also losing the run that revealed it.
    failures = isolation_failures(cache_before, cache_roots, targets_before, resolve_targets())
    if failures:
        raise SystemExit("\n\n".join(failures))
    print("cache isolation: both roots and the gold set unchanged")
    return 0


def report() -> int:
    if CONTROL_SCHEME != "pool":
        # `--controls` is accepted by the parser and assigned to CONTROL_SCHEME, but `controls()`
        # does not read it yet: it always draws from `.work/pool-cut100/`, RepoRadar's own
        # HEAD-seeded candidate pool. Left unguarded, `--controls arxiv-window` would write
        # POOL-drawn numbers into a file named `-arxiv-window` and stamped `controls:
        # arxiv-window` — an artefact asserting it used the arm-neutral negative class when it
        # used the one §4 exists to reject. A flag that is read only by the label is worse than
        # a flag nobody reads.
        raise SystemExit(
            f"--controls {CONTROL_SCHEME} is not implemented in controls() yet.\n"
            "  §4's arm-neutral draw (arxiv_window_controls) has no production caller: the\n"
            "  scheme selects the OUTPUT PATH and the provenance label, and nothing else, so\n"
            "  this run would publish pool-scheme numbers under an arm-neutral name.\n"
            "  Run without --controls for the NR-56/57 reproduction."
        )
    rng = random.Random(SEED)
    pos, ctl = adoptions(), controls(rng)
    have = load_verdicts()

    out: dict[str, Any] = {
        "_comment": (
            "NR-56. Adoption is the only label in this benchmark no model produced: "
            "ids(HEAD) - ids(T0) over a repo's own docs. Recall on it cannot rank judges "
            "because it supplies POSITIVES only and a lenient judge scores 100%, so each "
            "adopted paper is paired with matched controls -- same repo, published before the "
            "same T0, never adopted, not a T0 seed -- and the statistic is the GAP. Derived by "
            "evals/judge_validity_adoption.py; pinned by tests/test_judge_validity_adoption.py."
        ),
        "pre_registered": {
            "primary": "discrimination gap = P(actionable|adopted) - P(actionable|control)",
            "judges_ranked_by_gap_not_recall": True,
            "flat_if_both_below": FLAT_GAP,
            "separated_if_difference_at_least": SEPARATES,
            "written_before_any_control_or_sonnet_verdict": True,
        },
        "n_positives": len(pos),
        "n_controls": len(ctl),
        "positives_by_case": dict(Counter(r["case"] for r in pos)),
        "judges": {},
    }

    def rate(model: str, rows: list[dict[str, Any]], adopted: bool) -> dict[str, Any]:
        got = []
        for r in rows:
            k = key(model, r["case"], r["id"])
            if k in have:
                got.append(have[k]["score"])
        n_act = sum(1 for s in got if s >= 2)
        return {
            "n": len(got),
            "actionable": n_act,
            "rate": round(n_act / len(got), 4) if got else None,
            "wilson95": wilson(n_act, len(got)),
        }

    for model in (GPT_MODEL, SONNET_MODEL):
        a = rate(model, pos, adopted=True)
        c = rate(model, ctl, adopted=False)
        gap = (a["rate"] - c["rate"]) if a["rate"] is not None and c["rate"] is not None else None
        # `if gap is not None`, not `if gap`: a gap of exactly 0.0 is falsy, and it is the most
        # consequential value this study can produce — a judge that separates adopted papers
        # from matched controls not at all. Recording it as null made "measured, and zero"
        # indistinguishable from "never asked", the distinction the printout below is written
        # to preserve ("VOID, not zero").
        out["judges"][model] = {
            "adopted": a,
            "control": c,
            "gap": round(gap, 4) if gap is not None else None,
        }

    # A judge with no verdicts is not a judge with a gap of zero. NR-56 hit 155 consecutive
    # 400s and every Sonnet verdict came back void; `or 0.0` would have scored that run as a
    # measured null result for Sonnet and compared it against GPT.
    g = {m: out["judges"][m]["gap"] for m in (GPT_MODEL, SONNET_MODEL)}
    both_measured = all(v is not None for v in g.values())
    diff = abs(g[GPT_MODEL] - g[SONNET_MODEL]) if both_measured else None

    # The two gaps are computed over the SAME papers, so an independent-samples SE would
    # overstate the uncertainty of their difference. Bootstrap the papers instead, which is
    # the project's house estimator and respects the pairing.
    def scored(model: str, rows: list[dict[str, Any]]) -> dict[tuple[str, str], int]:
        """The judge's thresholded verdicts, keyed by the PAPER rather than by position."""
        out_s: dict[tuple[str, str], int] = {}
        for r in rows:
            k = key(model, r["case"], r["id"])
            if k in have:
                out_s[(r["case"], dedup_id(str(r["id"])))] = 1 if have[k]["score"] >= 2 else 0
        return out_s

    scored_pos = {m: scored(m, pos) for m in (GPT_MODEL, SONNET_MODEL)}
    scored_ctl = {m: scored(m, ctl) for m in (GPT_MODEL, SONNET_MODEL)}

    # Paired on the papers BOTH judges scored, in row order. It used to be paired on the list
    # POSITION, with both judges' array lengths taken from GPT's: `scores()` skipped a paper
    # with no verdict, so index i meant a different paper for each judge as soon as one verdict
    # was missing, and the comment above claiming the estimator "respects the pairing" was
    # false in exactly the case that matters. Unequal coverage then either raised IndexError
    # (Sonnet short) or silently truncated Sonnet to GPT's length and reported the rate over a
    # prefix — measured on the frozen data as 1.0 where the true value was 0.333. NR-56's 155
    # consecutive void Sonnet verdicts are what that scenario looks like in this project.
    paired_pos = [k for k in scored_pos[GPT_MODEL] if k in scored_pos[SONNET_MODEL]]
    paired_ctl = [k for k in scored_ctl[GPT_MODEL] if k in scored_ctl[SONNET_MODEL]]
    for m in (GPT_MODEL, SONNET_MODEL):
        out["judges"][m]["n_scored"] = {
            "adopted": len(scored_pos[m]),
            "control": len(scored_ctl[m]),
        }
    out["n_paired"] = {"adopted": len(paired_pos), "control": len(paired_ctl)}

    pa = {m: [scored_pos[m][k] for k in paired_pos] for m in (GPT_MODEL, SONNET_MODEL)}
    pc = {m: [scored_ctl[m][k] for k in paired_ctl] for m in (GPT_MODEL, SONNET_MODEL)}
    boot = random.Random(SEED + 1)
    diffs = []
    if paired_pos and paired_ctl:
        na, nc = len(paired_pos), len(paired_ctl)
        for _ in range(5000):
            ia = [boot.randrange(na) for _ in range(na)]
            ic = [boot.randrange(nc) for _ in range(nc)]
            gaps = {}
            for m in (GPT_MODEL, SONNET_MODEL):
                gaps[m] = sum(pa[m][i] for i in ia) / na - sum(pc[m][i] for i in ic) / nc
            diffs.append(gaps[SONNET_MODEL] - gaps[GPT_MODEL])
        diffs.sort()
        lo, hi = diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs))]
    # Does each judge discriminate AT ALL? More decision-relevant than the comparison: a gap
    # whose interval spans zero is a judge that has not been shown to separate papers a
    # repository adopted from papers it did not.
    #
    # **Over that judge's OWN scored set, not the two-judge intersection.** The intersection is
    # the right unit for the DIFFERENCE above, which needs the pairing; it is the wrong unit
    # for a per-judge interval, which is a statement about that judge's own sample — and the
    # point estimate `gap` beside it comes from `rate()`, over that judge's own rows. Pairing
    # them mismatched published an interval for a sample the number next to it did not
    # describe: with half of Sonnet's verdicts missing, GPT's gap stays 0.1428 while GPT's
    # interval is recomputed over Sonnet's survivors, and it can cross zero — inverting this
    # study's headline finding about the primary judge without one GPT verdict changing.
    for m in (GPT_MODEL, SONNET_MODEL):
        own_pos = list(scored_pos[m].values())
        own_ctl = list(scored_ctl[m].values())
        if not own_pos or not own_ctl:
            continue
        mp, mc = len(own_pos), len(own_ctl)
        own = []
        for _ in range(5000):
            ia = [boot.randrange(mp) for _ in range(mp)]
            ic = [boot.randrange(mc) for _ in range(mc)]
            own.append(sum(own_pos[i] for i in ia) / mp - sum(own_ctl[i] for i in ic) / mc)
        own.sort()
        olo, ohi = own[int(0.025 * len(own))], own[int(0.975 * len(own))]
        out["judges"][m]["gap_ci95"] = [round(olo, 4), round(ohi, 4)]
        out["judges"][m]["gap_excludes_zero"] = excludes((olo, ohi), 0.0)
        # Two-sided `excludes` answers "is this judge distinguishable from no discrimination",
        # which is what the name says — but on its own it maps an interval entirely BELOW zero
        # onto the same `true` as a validated judge. A judge ranking matched controls above the
        # papers a project adopted is the most interesting outcome this design can produce, so
        # the direction is recorded beside the flag rather than left to be inferred from the
        # bounds. The old one-sided test collapsed the other pair instead, reporting an
        # anti-discriminating judge with the same words as one that separates nothing.
        out["judges"][m]["gap_direction"] = (
            "above zero" if olo > 0 else "BELOW ZERO" if ohi < 0 else "spans zero"
        )

    if diffs:
        out["gap_difference_bootstrap"] = {
            "_comment": (
                "Sonnet gap minus GPT gap, bootstrapped over the same papers both judges "
                "scored. Positive favours Sonnet. The registered separation bar is on the "
                "point estimate; this says how much the point estimate is worth."
            ),
            "point": round(sum(diffs) / len(diffs), 4),
            "ci95": [round(lo, 4), round(hi, 4)],
            "excludes_zero": excludes((lo, hi), 0.0),
        }
    better = max(g, key=lambda m: g[m]) if both_measured else None
    separated = bool(diff >= SEPARATES) if diff is not None else None
    out["verdict"] = {
        "gaps": g,
        "difference": round(diff, 4) if diff is not None else None,
        "both_flat": bool(max(g.values()) < FLAT_GAP) if both_measured else None,
        "separated": separated,
        "better_instrument": better if separated else None,
        "replicates_nr56": {
            "_comment": (
                "NR-56 ran on 31 positives across 6 cases with 124 controls drawn under a "
                "SHARED-rng scheme. This run has 35 positives across 9 cases and 140 controls "
                "drawn per-case, so the controls were fully redrawn -- it is an INDEPENDENT "
                "sample, not a superset. Both conclusions hold, slightly attenuated, which is "
                "what makes it a replication rather than an update."
            ),
            "nr56": {
                "n_pos": 31,
                "n_ctl": 124,
                "gpt_gap": 0.153,
                "sonnet_gap": 0.282,
                "difference": 0.129,
            },
            "this_run": {"n_pos": 35, "n_ctl": 140},
            "gpt_still_spans_zero": True,
            "sonnet_still_excludes_zero": True,
            "still_not_separated": True,
        },
        "what_would_settle_it": {
            "_comment": (
                "Precision here is governed almost entirely by the POSITIVES: the adopted "
                "variance term is 4-6x the control term because n_pos is a quarter of n_ctl. "
                "Mining every remaining benchmark case moved 31 -> 35, so the channel is "
                "exhausted at this scale and the shortfall is structural, not effort."
            ),
            "n_positives_needed_at_this_gap": 55,
            "n_positives_available": 35,
            "why_expansion_stalled": (
                "Of the 15 newly mined cases only 3 contributed. Several carry NO arXiv ids in "
                "their documentation at all (thin-kv, vectordb, webdev report 0 ids at HEAD); "
                "others have no history before the 24-month T0 cutoff (thin-gnn, thin-lang). "
                "Reaching 55 needs a longer window or cases selected for citation-rich docs -- "
                "a differently-constructed benchmark, not more of this one."
            ),
        },
        # None when the bootstrap never ran, not False. `.get(..., True)` defaulted a MISSING
        # interval to "excludes zero", so this — the single most consequential field in the
        # artefact, and the one `tests/test_judge_validity_adoption.py` pins as the study's
        # headline — published a positive discrimination claim computed from no data. It is the
        # one place absence still rendered as a conclusion after every neighbouring field was
        # made void-safe.
        "primary_judge_gap_spans_zero": (
            None
            if out["judges"][GPT_MODEL].get("gap_excludes_zero") is None
            else not out["judges"][GPT_MODEL]["gap_excludes_zero"]
        ),
        "headline": (
            "The primary judge -- gpt-5.5, the model every number in this project is scored "
            "against -- has NOT been shown to discriminate adoption: gap 0.153, CI [-0.040, "
            "+0.339], spanning zero. It calls 49.2% of matched controls actionable, papers "
            "from the same repo published before the same T0 that the project never took up. "
            "claude-sonnet-5's gap does exclude zero (0.282, CI [+0.097, +0.476]), but the "
            "DIFFERENCE between them (0.129, CI [-0.024, +0.274]) does not clear the "
            "registered 0.15 bar, so this does not name a better instrument. What it does say "
            "is that the project's primary judge lacks demonstrated validity against the only "
            "label here that no model produced. Absence of evidence, not evidence of error."
        ),
        "_prose_describes_nr56_not_this_run": (
            "`headline` and `caveats` are string literals NR-56 wrote and NR-57's re-run did "
            "not update, because they are hard-coded rather than derived. They quote gap "
            "0.153, CI [-0.040, +0.339], 49.2% of controls and n = 31 across 6 cases; the "
            "computed block in this same artefact says 0.1428, [-0.0429, 0.3214], 0.5143 and "
            "n = 35 across 9. The CONCLUSION is unchanged — the primary judge's interval still "
            "spans zero, and `caveats`' qualitative claims still hold — but the FIGURES inside "
            "the prose are from the earlier sample. Read the computed fields, not the "
            "sentences. The published record carries the same defect and is deliberately NOT "
            "edited here: it is cited by RESULTS.md, evals/README.md and PLANS.md, and "
            "silently correcting a published artefact is worse than labelling it. This is the "
            "incident that requires the pool wrapper to DERIVE its prose from computed values "
            "rather than hard-code it; that wrapper writes no prose yet."
        ),
        "caveats": (
            "n=31 positives across 6 cases with graph contributing 13 (C-7); 'not adopted' is a "
            "noisy negative that biases both gaps downward; adoption measures what a repository "
            "did, not what it should have done."
        ),
    }
    # NOT `FROZEN`. This function is the DEFAULT action of the script — no `--plan`, no
    # `--judge` falls through to here — and it used to overwrite the published NR-56/57 record
    # unconditionally. `judge_validity_adoption.json` is what RESULTS.md's two tables,
    # evals/README.md and PLANS.md quote; a run against a different positive set would have
    # replaced those numbers in place, silently, and the citations would still read as if
    # nothing had happened. §1: "the v1 record is immutable ... because it is the record v2 is
    # compared against."
    from judge_validity_pool import artifact_path, write_artifact

    out["_run"] = {"source": "legacy", "controls": CONTROL_SCHEME}
    written = write_artifact(artifact_path("legacy", CONTROL_SCHEME), out)

    print(
        f"positives {len(pos)} across {len(out['positives_by_case'])} cases, controls {len(ctl)}\n"
    )
    print(f"{'judge':<20}{'adopted':>18}{'control':>18}{'gap':>9}")
    for model in (GPT_MODEL, SONNET_MODEL):
        j = out["judges"][model]
        a, c = j["adopted"], j["control"]
        if a["rate"] is None or c["rate"] is None:
            # VOID, not zero: a judge with no verdicts has not scored badly, it has not been
            # asked. Printing 0.000 here would read as a measured floor.
            print(f"{model:<20}   NO VERDICTS (adopted {a['n']}, control {c['n']}) -- void")
            continue
        print(
            f"{model:<20}{a['actionable']:>4}/{a['n']:<4}{a['rate']:>8.3f}"
            f"{c['actionable']:>6}/{c['n']:<4}{c['rate']:>8.3f}{j['gap']:>9.3f}"
        )
    for model in (GPT_MODEL, SONNET_MODEL):
        j = out["judges"][model]
        if "gap_ci95" in j:
            mark = "excludes zero" if j["gap_excludes_zero"] else "SPANS ZERO"
            print(
                f"  {model:<20} gap {j['gap']:.3f}  "
                f"CI [{j['gap_ci95'][0]:+.3f}, {j['gap_ci95'][1]:+.3f}]  {mark}"
            )
    v = out["verdict"]
    if v["difference"] is None:
        # The same "VOID, not zero" rule one screen up: with a judge unscored there is no
        # difference to compare against the bar, and printing one would invent it.
        print("\ngap difference: VOID -- a judge has no verdicts, so there is nothing to compare.")
    else:
        print(f"\ngap difference: {v['difference']:.3f}  (separates at >= {SEPARATES})")
        if v["both_flat"]:
            print(f"BOTH GAPS < {FLAT_GAP}: neither judge separates adoption from controls.")
        elif v["separated"]:
            print(f"SEPARATED: {v['better_instrument']} is the better instrument here.")
        else:
            print("NOT SEPARATED at this n -- reported as such, not resolved by preference.")
    print(f"wrote {written}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--judge", action="store_true")
    ap.add_argument(
        "--controls",
        choices=CONTROL_SCHEMES,
        default="pool",
        help=(
            "'pool' reproduces NR-56/57 (RepoRadar's own candidate pool). 'arxiv-window' is "
            "the arm-neutral scheme the validity pool registers (§4): same primary category, "
            "same half-year, never cited by the repository at HEAD."
        ),
    )
    args = ap.parse_args()
    # A module-level switch rather than a threaded parameter: `controls()` is called from
    # plan(), judge() and report(), and a scheme that differed between planning and judging
    # would draw one control set and score another.
    global CONTROL_SCHEME
    CONTROL_SCHEME = args.controls
    if args.plan:
        return plan()
    if args.judge:
        return judge()
    return report()


if __name__ == "__main__":
    raise SystemExit(main())
