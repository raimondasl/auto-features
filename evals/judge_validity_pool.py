"""Section 5 of `PREREG-judge-validity-pool.md`, and the boundary around what it may write.

The pre-registration's own banner names the gap this module closes: *"the analysis wrapper that
assembles pool verdicts into section 5's tables does not exist... It must be written BEFORE any
verdict is bought, and it must implement section 5 as registered here rather than as convenient
later."* This file is that wrapper.

**What is here now is the boundary, not yet the endpoints.** The endpoints need the walk's
positives, which do not exist until the pulse at `2026-09-04T00:00:00Z`. The boundary does not
wait for them, because the thing it prevents can happen on the next bare invocation:

    uv run python evals/judge_validity_adoption.py

is the DEFAULT action of that script — no `--plan`, no `--judge`, falls through to `report()` —
and `report()` wrote `evals/judge_validity_adoption.json` unconditionally. That file is the
**published record of NR-56/57**: its computed numbers are quoted in `RESULTS.md` (the NR-57
table, gaps 0.143 and 0.243, and the NR-56 table, 0.153 and 0.282), in `evals/README.md`, and in
`PLANS.md`. Run the script against pool positives and the published record is silently replaced
by numbers from a different study, in a project whose §1 rule is that *"the v1 record is
immutable... because it is the record v2 is compared against"*.

So no analysis in this study writes a path it was not given, and the frozen records are checked
by hash on both sides of every write rather than trusted to a convention.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

EVALS = Path(__file__).resolve().parent
sys.path.insert(0, str(EVALS))
sys.path.insert(0, str(EVALS / "frame"))
sys.path.insert(0, str(EVALS.parent / "src"))

from reporadar.paper_id import dedup_id  # noqa: E402

WORK = EVALS / ".work"

# The walk's own output, read and never rewritten. `POOL` already means `.work/pool-cut100`
# in judge_validity_adoption, so these carry their own names rather than shadowing it.
POOL_FRAME = EVALS / "frame" / "pool"
SEED_FILE = POOL_FRAME / "SEED_POOL"
POOL_ADOPTIONS = POOL_FRAME / "adoptions-pool-v2.json"
POOL_WALK = POOL_FRAME / "validity_walk.csv"
POOL_SUMMARY = POOL_FRAME / "walk_summary.json"
POOL_CURVE = POOL_FRAME / "yield_curve.csv"
POOL_CONTEXTS = POOL_FRAME / "contexts"
POOL_HEAD_IDS = POOL_FRAME / "head_ids"
CANDIDATES = POOL_FRAME / "pool-universe-Dp.csv"

# NR-60's re-mine. §1: "The v1 record is immutable." Neither this nor `adoptions.json` is
# ever written by anything here.
LEGACY_ADOPTIONS = WORK / "adoptions-v2.json"

# Written by the purchase loop when it terminates cleanly. Its absence is what `refuse_to_peek`
# reads as "judging has not finished", so it is a completion record and not a log.
JUDGING_DONE = POOL_FRAME / "judging_complete.json"

# Published records. Not "files we would rather not touch" — files whose numbers are cited in
# prose elsewhere in the tree, so overwriting one makes the citation wrong without making it
# look wrong. `git log` on this path shows exactly two commits, [NR-56] and [NR-57].
FROZEN_RECORDS = frozenset({EVALS / "judge_validity_adoption.json"})

# Written into every artefact this study produces. `write_artifact` refuses to overwrite a file
# that lacks it, so a mistyped output path fails on the first run instead of on the run after
# somebody else's artefact has already gone.
ARTEFACT_MARKER = "judge-validity-pool"

SOURCES = ("pool", "legacy")
SCHEMES = ("pool", "arxiv-window")


def artifact_path(source: str, scheme: str) -> Path:
    """Where a (positive source, control scheme) run is allowed to write.

    Both arguments are required and neither has a default. The pairing decides the path because
    it decides what the numbers *are*: §4 registers the arXiv-window scheme for this pool and
    keeps the pool-control result "reported as NR-57", so the two are different results about
    different negative classes and must not land in one file.
    """
    if source not in SOURCES:
        raise SystemExit(f"unknown positive source {source!r}; registered sources are {SOURCES}")
    if scheme not in SCHEMES:
        # Not a formality. Everything that is not "pool" fell through to the arxiv-window
        # path, so a typo — or a caller passing "" or None — produced a file NAMED for the
        # arm-neutral scheme holding numbers from whatever was actually drawn.
        raise SystemExit(f"unknown control scheme {scheme!r}; registered schemes are {SCHEMES}")
    if source == "pool" and scheme == "pool":
        raise SystemExit(
            "REFUSED: pool positives with the pool control scheme.\n"
            "  §4 registers the arm-neutral scheme for this population and states the reason:\n"
            "  the shipped candidate pool is RepoRadar's own HEAD-seeded output, so a judge\n"
            "  harsher on RepoRadar-shaped papers — Sonnet, by a factor of 2.3 — would be\n"
            "  credited with 'validity' for a property of the control set.\n"
            "  There is also no data for it: pool_papers() reads .work/pool-cut100/<case>.json,\n"
            "  which exists only for the 37 legacy slugs, so every pool positive would draw\n"
            "  ZERO controls and the AUC would be computed against nothing."
        )
    if source == "pool":
        return EVALS / "judge_validity_pool.json"
    if scheme == "pool":
        # The NR-57 reproduction. It recomputes a published result, so it goes somewhere
        # untracked: a reproduction that overwrites the thing it reproduces cannot disagree
        # with it, and disagreeing is the only reason to run one.
        return WORK / "repro" / "judge_validity_adoption-legacy-pool.json"
    return EVALS / "judge_validity_adoption-arxiv-window.json"


def frozen_digests() -> dict[str, str]:
    """sha256 of every published record that currently exists."""
    out: dict[str, str] = {}
    for path in sorted(FROZEN_RECORDS):
        if path.is_file():
            out[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def assert_frozen_records_intact(before: dict[str, str]) -> None:
    """Raise if a published record moved. Checked on both sides of every write.

    A path guard alone would catch only the direct overwrite. This also catches the indirect
    one — a helper that resolves a relative path, a test fixture pointed at the real tree — for
    the cost of hashing one 6 KB file.
    """
    after = frozen_digests()
    moved = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
    if moved:
        raise SystemExit(
            "A PUBLISHED RECORD MOVED during this run:\n"
            + "".join(f"  {p}\n" for p in moved)
            + "  These files hold the numbers RESULTS.md, evals/README.md and PLANS.md quote.\n"
            "  Restore them from git before doing anything else: `git checkout -- <path>`."
        )


def write_artifact(path: Path, payload: dict[str, Any]) -> Path:
    """Serialise *payload* to *path*, or refuse and say which rule stopped it.

    Three refusals, each for a failure this project has already paid for once:

    * **a frozen record** — the overwrite this module exists to prevent;
    * **an existing file without the marker** — somebody else's artefact, reached by a typo;
    * **a non-finite number** — `json.dumps` writes a bare ``NaN``, which every JSON parser
      outside Python rejects, and §10 step 7 publishes this file as a datasheet. `wilson()`
      returned ``(nan, nan)`` at n = 0, so a judge with no verdicts produced an unparseable
      artefact that looked complete.

    The write is a temp file plus `os.replace`, so a failure halfway through leaves the previous
    artefact intact rather than a truncated one that still parses.
    """
    resolved = path.resolve()
    if resolved in {p.resolve() for p in FROZEN_RECORDS}:
        raise SystemExit(
            f"REFUSED: {resolved.name} is a published record and is never rewritten.\n"
            "  It holds NR-56/57 as published — the gaps 0.143 and 0.243 quoted in\n"
            "  RESULTS.md, evals/README.md and PLANS.md are read from this file.\n"
            "  Pass an output path from artifact_path(source, scheme) instead."
        )
    if resolved.is_file():
        try:
            existing = json.loads(resolved.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            existing = None
        if not (isinstance(existing, dict) and existing.get("_artefact") == ARTEFACT_MARKER):
            raise SystemExit(
                f"REFUSED: {resolved} already exists and was not written by this study.\n"
                f"  It carries no _artefact: {ARTEFACT_MARKER!r} marker, so overwriting it\n"
                "  would destroy an artefact this code knows nothing about."
            )

    body = {"_artefact": ARTEFACT_MARKER, **payload}
    # allow_nan=False so a non-finite number is a loud failure here rather than a silent one
    # in whatever reads the datasheet next.
    text = json.dumps(body, indent=1, allow_nan=False) + "\n"

    before = frozen_digests()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    tmp = resolved.with_suffix(resolved.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, resolved)
    assert_frozen_records_intact(before)
    return resolved


# ---------------------------------------------------------------------------------------
# The seed, and the bar against looking early (§2.4, §3.3)
# ---------------------------------------------------------------------------------------
def pool_seed(path: Path | None = None, *, verify: bool = True, verifier: Any = None) -> str:
    """`SEED_POOL`, checked against the beacon pulse §2.4 names before it selects anything.

    Verification is `walk_pool.verify_seed` — imported, not reimplemented, because the walk
    already ordered 17,888 candidates with it and a second implementation could disagree with
    the first about the same file. A typo, a truncated copy or a hand-edited value all walk a
    different order while still looking like a hex string.

    **This is the seed for everything that SELECTS data**: the legacy cap, the cross-repository
    contest, the order positives are judged in, and the control draw. It is deliberately not
    `judge_validity_adoption.SEED`, which stays bound to two things that select nothing — the
    NR-57 pool-scheme reproduction, and `cluster_bootstrap_auc`'s resampling RNG, a
    computational-reproducibility parameter pinned by its own tests.
    """
    import walk_pool

    src = path or SEED_FILE
    if not src.is_file():
        raise SystemExit(
            f"{src} does not exist.\n"
            f"  §2.4 names the pulse {walk_pool.REGISTERED_PULSE}; runbook step 4 writes its\n"
            "  outputValue here after that pulse, and nothing in this study may select or\n"
            "  order anything before it exists."
        )
    seed = src.read_text(encoding="utf-8").strip()
    if not seed:
        raise SystemExit(f"{src} is empty — SEED_POOL must be the beacon pulse outputValue")
    if verify:
        # Injectable, exactly as `walk_pool.verify_seed` takes its own `fetch`: the check hits
        # the live beacon, so a test that cannot reach the network must be able to supply one.
        (verifier or walk_pool.verify_seed)(seed, walk_pool.REGISTERED_PULSE)
    return seed


REPORTING_MINIMUM = 60  # §3.3: "60 is the reporting minimum, not a stopping point."


def walk_stop_reason(summary: dict[str, Any], n_candidates: int | None = None) -> str | None:
    """Which stop condition fired, or None if the walk must keep going.

    Three conditions, and the middle one is conditional in a way the section it comes from
    states plainly. §3.2 runs the walk "to B = 1,200 rows or until cumulative capped-usable new
    positives reach 100, whichever comes first" — but §3.4 then says that **if B is reached
    below 60 new positives the walk continues down the already-frozen seeded order, to the end
    of the list**. So reaching B is a stop only when the reporting minimum is already in hand;
    below it, B is a checkpoint and exhaustion is the real terminus. Treating B as an
    unconditional stop would let a short walk be analysed at n = 40 while §3.4 was still asking
    for more rows, and §9's power table is exactly where that shortfall does its damage.
    """
    import walk_pool

    walked = int(summary.get("walked") or 0)
    positives = int(summary.get("capped_positives") or 0)
    # The REGISTERED target, not the summary's. `walk_summary.json` is written from `--target`
    # and `--budget` on every run and is guarded by nothing, so reading the thresholds out of
    # it let the gate take its bar from the very artefact it is gating: a rehearsal run with
    # `--target 40` writes `{"target": 40, "capped_positives": 40}` and is blessed as having
    # met the target, at 71 positives instead of the registered 130. §9's own power table says
    # 90 may include 0.5 for a sampling reason. The seed is not trusted to its file either.
    # May GROW and only grow, exactly as the budget below may. A target BELOW the registered
    # one is the rehearsal-run hazard this check was built for and stays refused. A target
    # above it can only add positives, which narrows the primary interval and therefore makes
    # §5's pre-committed null HARDER to fire — the one direction a deviation cannot use to
    # manufacture the result the study would most like to report.
    #
    # §3.3's deviation of 2026-09-05 raises it from 100 to 150: the walk stopped at B on exactly
    # 60, which put the analysis set at §9's 90-positive row, the row whose entry reads "may
    # include 0.5". P5 is still scored against the REGISTERED 100 within B, and still fails.
    target = int(summary.get("target") or 0)
    if target < walk_pool.DEFAULT_TARGET:
        raise SystemExit(
            f"the walk ran with target={target}, below §3.3's registered "
            f"{walk_pool.DEFAULT_TARGET}.\n"
            "  A lowered target would bless a short walk as having met the stop rule."
        )
    if target > walk_pool.DEFAULT_TARGET:
        deviation = "DEVIATION, 2026-09-05" in (
            (EVALS / "PREREG-judge-validity-pool.md").read_text(encoding="utf-8")
            if (EVALS / "PREREG-judge-validity-pool.md").is_file()
            else ""
        )
        if not deviation:
            raise SystemExit(
                f"the walk ran with target={target}, above §3.3's registered "
                f"{walk_pool.DEFAULT_TARGET}, and no deviation is recorded in the "
                "pre-registration.\n  A stop rule changed after seeing the yield is a deviation "
                "whether or not it is written down; writing it down is what makes it one."
            )
    # The budget is allowed to GROW and only to grow: §3.4 executes by re-running past B, and
    # the only way `walk()` can do that is a larger `--budget`. A smaller one is refused,
    # because it would let a ten-row walk satisfy the budget branch.
    budget = int(summary.get("budget") or 0)
    if budget < walk_pool.DEFAULT_B:
        raise SystemExit(
            f"the walk ran with budget={budget}, below §3.2's B = {walk_pool.DEFAULT_B}.\n"
            "  §3.4 permits extending past B, never stopping short of it."
        )
    if n_candidates is not None and walked >= n_candidates > 0:
        # §3.4's recorded negative result: the list is spent, and whatever n exists is n.
        return "exhausted"
    if target > 0 and positives >= target:
        return "target"
    if budget > 0 and walked >= budget and positives >= REPORTING_MINIMUM:
        return "budget"
    return None


def refuse_to_peek(
    summary: dict[str, Any] | None = None,
    *,
    n_candidates: int | None = None,
    judging_done: Path | None = None,
) -> dict[str, Any]:
    """§3.3: "No endpoint is inspected before the stop rule fires." Enforced, not remembered.

    An endpoint computed mid-walk is not a preview of the final one — it is a result read at a
    moment chosen by whoever ran it, which is the discretion the frozen order and the
    unchoosable pulse exist to remove. The whole apparatus is defeated by one impatient run
    whose number is then in somebody's head.

    Returns the stop record on success. §5's shortfall block may print counts and the stop
    reason before this passes; nothing else may.
    """
    if summary is None:
        if not POOL_SUMMARY.is_file():
            raise SystemExit(f"{POOL_SUMMARY} does not exist — the walk has not run.")
        summary = json.loads(POOL_SUMMARY.read_text(encoding="utf-8"))
    if n_candidates is None and CANDIDATES.is_file():
        with CANDIDATES.open(encoding="utf-8", newline="") as fh:
            n_candidates = sum(1 for r in csv.DictReader(fh) if (r.get("full_name") or "").strip())

    reason = walk_stop_reason(summary, n_candidates)
    if reason is None:
        walked, budget = int(summary.get("walked") or 0), int(summary.get("budget") or 0)
        past_b = (
            " — B is reached, but §3.4 sends it on to the end of the list"
            if (budget and walked >= budget)
            else ""
        )
        raise SystemExit(
            "REFUSED: the walk has not stopped, so no endpoint may be inspected (§3.3).\n"
            f"  walked {walked} of budget {budget}"
            f"{f' and {n_candidates} candidates' if n_candidates else ''}; "
            f"capped positives {summary.get('capped_positives')} of target "
            f"{summary.get('target')} (reporting minimum {REPORTING_MINIMUM}){past_b}.\n"
            "  A number read now is a number read at a chosen moment."
        )

    done = judging_done or JUDGING_DONE
    if not done.is_file():
        raise SystemExit(
            f"REFUSED: the walk stopped on '{reason}', but judging has not finished.\n"
            f"  {done} is written by the purchase loop when it terminates cleanly; without it\n"
            "  an endpoint would be computed over however many verdicts happened to be bought\n"
            "  when someone looked, which is the same discretion under a different name."
        )
    return {
        "stop_reason": reason,
        "walked": summary.get("walked"),
        "stop_rule_capped_positives": summary.get("capped_positives"),
        # The thresholds that produced the reason, so a datasheet reading "stop_reason: target"
        # can be checked against WHICH target rather than taken on trust.
        "b0": summary.get("b0"),
        "budget": summary.get("budget"),
        "target": summary.get("target"),
        "reporting_minimum": REPORTING_MINIMUM,
        "n_candidates": n_candidates,
        "judging_complete": json.loads(done.read_text(encoding="utf-8")),
    }


# ---------------------------------------------------------------------------------------
# The analysis set (§3.3, §1's filter table)
# ---------------------------------------------------------------------------------------
POSITIVE_TERMS = ("usable", "genesis", "in_cap", "counted")


def _require_terms(
    rows: list[dict[str, Any]], where: str, terms: tuple[str, ...] = POSITIVE_TERMS
) -> None:
    """Every row must carry the terms it is about to be judged on. A missing key is never a value.

    `row.get("counted")` returns None on a partially written artefact, None is falsy, and the
    positive silently leaves the study. A walk interrupted between mining and assignment writes
    exactly that shape, so the failure is reachable rather than theoretical — and it shrinks n
    without shrinking anything visible.

    *terms* is narrower for the legacy artefact, which predates `in_cap` and `counted` and has
    them re-derived here. It still must carry `usable` and `genesis`: those are §1 filter
    results that only mining can produce, and `setdefault`ing either would let a truncated
    artefact through the legacy path that the pool path refuses — the same silent shrink,
    reached by the door left open for the fields that are genuinely absent by design.
    """
    for row in rows:
        missing = [t for t in terms if t not in row]
        if missing:
            raise SystemExit(
                f"{where}: row {row.get('case')}/{row.get('id')} lacks {missing}.\n"
                "  Every term of the positive definition must be present on every row;\n"
                "  an absent key would be read as False and drop the paper silently."
            )


def is_positive(row: dict[str, Any]) -> bool:
    """§3.3 and §1: usable, not genesis, inside the per-repository cap, and counted once.

    `genesis` is a term in its own right because §1's filter table lists it separately and
    `walk_row` does NOT fold it into `usable` — it sets `genesis: False` unconditionally, since
    PP2 >= 3 subsumes the doc-genesis guard. Reading it anyway costs nothing and means the
    definition here matches the registered table rather than the implementation's shortcut.
    """
    return (
        bool(row["usable"])
        and not bool(row["genesis"])
        and bool(row["in_cap"])
        and bool(row["counted"])
    )


def pool_positives(path: Path | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The pool stratum's positives, with a ledger accounting for every row that is not one."""
    src = path or POOL_ADOPTIONS
    if not src.is_file():
        raise SystemExit(f"{src} does not exist — the walk has not mined anything.")
    rows = json.loads(src.read_text(encoding="utf-8"))
    _require_terms(rows, str(src))
    for row in rows:
        row["stratum"] = "pool"
    return _select(rows, "pool")


def legacy_positives(
    seed: str, path: Path | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The legacy stratum's positives, capped and contested in the walk's own order.

    The legacy artefact carries none of `in_cap`, `counted` or `assigned_to` — it predates them
    — so they are re-derived here, **cap first and contest second**, byte-identical to what
    `walk_row` and `merge_adoptions` do for a pool repository. The order is not cosmetic:
    `walk_row` sets `in_cap` before any cross-repository comparison exists, and the assignment
    re-runs over the whole file afterwards. Contest-first would let the cap refill the slot a
    contested paper vacated and yield MORE positives, which is both the wrong order and the
    unconservative one.

    The contest is run with `legacy_ids=set()`: this IS the legacy stratum, so there is no
    outside cluster for it to lose a tie to.
    """
    import walk_pool

    src = path or LEGACY_ADOPTIONS
    if not src.is_file():
        raise SystemExit(f"{src} does not exist — NR-60's re-mine has not been run.")
    rows = json.loads(src.read_text(encoding="utf-8"))
    usable = [r for r in rows if r.get("usable")]

    _require_terms(rows, str(src), terms=("usable", "genesis"))
    capped = {id(r) for r in walk_pool.legacy_capped(usable, seed)}
    for row in rows:
        row["stratum"] = "legacy"
        row["in_cap"] = id(row) in capped

    # Over EVERY row, not just the capped ones — `merge_adoptions` passes the whole artefact,
    # and running a narrower contender set here would make the two strata's `counted` mean
    # different things while §5's transportability endpoint contrasts them. It matters: over
    # all rows a non-usable or over-cap row can WIN an identifier and knock out a capped one,
    # which is the `assigned_to_a_non_positive` case. Restricting the contenders would hide
    # exactly that loss, and would also be the unconservative choice.
    walk_pool.assign_across_repos(rows, seed, legacy_ids=set())
    for row in rows:
        row["counted"] = bool(row["counted"]) and row["in_cap"]
    return _select(rows, "legacy")


def _select(
    rows: list[dict[str, Any]], stratum: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Split rows into positives and a ledger that accounts for every one that is not.

    §3.3 says a shared identifier is "counted once". Counted **zero** times is a different
    thing, and it is invisible unless the rows that reach it are named — so the ledger carries
    ids, not totals.
    """
    positives = [r for r in rows if is_positive(r)]
    usable = [r for r in rows if r.get("usable")]
    in_cap = [r for r in usable if r.get("in_cap")]
    by_case: dict[str, str] = {}
    for row in positives:
        by_case[str(row["case"])] = stratum

    # A paper whose contest WINNER is itself not a positive leaves the study entirely: the loser
    # is uncounted by rule, and the winner is dropped by a filter. Counted zero times.
    #
    # `assigned_to == "legacy"` is excluded, and that exclusion is the difference between a
    # ledger and a false alarm. §3.3 registers the legacy tie — `assign_across_repos` sets
    # `assigned_to = "legacy"` and `counted = False` for every contender — so a pool row losing
    # to the legacy cluster is the rule WORKING. Counting those as orphans would fill the ledger
    # with the registered behaviour and bury the one thing it exists to surface. Those rows are
    # reported separately, as the de-duplication they are.
    winners = {(str(r.get("assigned_to")), str(r["id"])) for r in positives}
    lost_to_legacy = [
        {"case": str(r["case"]), "id": str(r["id"])}
        for r in in_cap
        if not r.get("counted") and str(r.get("assigned_to")) == "legacy"
    ]
    orphaned = [
        {"case": str(r["case"]), "id": str(r["id"]), "assigned_to": r.get("assigned_to")}
        for r in in_cap
        if not r.get("counted")
        and str(r.get("assigned_to")) != "legacy"
        and (str(r.get("assigned_to")), str(r["id"])) not in winners
    ]
    ledger = {
        "stratum": stratum,
        "n_rows": len(rows),
        "n_usable": len(usable),
        "n_genesis_excluded": sum(1 for r in usable if r.get("genesis")),
        "n_usable_over_cap": len(usable) - len(in_cap),
        "n_usable_in_cap": len(in_cap),
        "n_contested_lost": sum(1 for r in in_cap if not r.get("counted")),
        "n_positives": len(positives),
        "n_clusters": len(by_case),
        # Ids, because a count cannot be checked against anything. §3.3's "counted once" is
        # violated by these rows specifically, not by a number.
        "assigned_to_a_non_positive": sorted(orphaned, key=lambda r: (r["case"], r["id"])),
        # §3.3's registered de-duplication working as written, kept visible rather than silent:
        # these papers ARE counted, once, in the legacy stratum.
        "deduped_to_legacy": sorted(lost_to_legacy, key=lambda r: (r["case"], r["id"])),
    }
    return positives, ledger


# ---------------------------------------------------------------------------------------
# T0 contexts and head_ids (§3.1, §4)
# ---------------------------------------------------------------------------------------
CONTEXT_SUFFIX = re.compile(r"\.[0-9a-f]{12}\.txt$")
README_SECTION = "## README (excerpt)"
# Exactly the headings `mine_adoptions.t0_context` emits after the README, and nothing else.
# Anything matching a bare "## " may be the README's own markdown.
KNOWN_SECTIONS = (
    "\n## requirements.txt",
    "\n## pyproject.toml",
    "\n## package.json",
    "\n## setup.py",
    "\n## Source files (sample)",
)


def case_key(case: str) -> str:
    """The walk's filename mangling: `'/' -> '__'`. Never inverted.

    It is not injective — `a/b` and `a__b` collide — so every lookup here is built FORWARD from
    a case already in hand, and `assert_keys_are_injective` refuses a run where two cases would
    share a file rather than letting one silently read the other's T0 context.
    """
    return case.replace("/", "__")


def assert_keys_are_injective(cases: list[str]) -> None:
    seen: dict[str, str] = {}
    for case in sorted(set(cases)):
        key = case_key(case)
        if key in seen:
            raise SystemExit(
                f"two cases share one context filename: {seen[key]!r} and {case!r} both mangle "
                f"to {key!r}. One would be judged against the other's repository."
            )
        seen[key] = case


def assert_contexts_are_one_to_one(cases: list[str], contexts: Path | None = None) -> None:
    """Every persisted context belongs to exactly one case in the analysis set.

    Checked once and globally rather than by globbing `f'{key}.*.txt'` per case: repository
    names contain dots, so `owner__repo.*.txt` also matches `owner__repo.js.<digest>.txt`, and
    a perfectly healthy pair of repositories would abort the run inside the machinery meant to
    make failure loud.
    """
    root = contexts or POOL_CONTEXTS
    if not root.is_dir():
        return
    walked = {case_key(c) for c in cases}
    seen: dict[str, Path] = {}
    for path in sorted(root.glob("*.txt")):
        if not CONTEXT_SUFFIX.search(path.name):
            raise SystemExit(f"{path.name} does not match '<key>.<12 hex>.txt'")
        key = CONTEXT_SUFFIX.sub("", path.name)
        if key in seen:
            raise SystemExit(f"two contexts for one case: {seen[key].name} and {path.name}")
        seen[key] = path
    # Deliberately NOT the converse. `walk_row` writes the context inside `if usable:` — before
    # any cross-repository contest exists — and nothing prunes the directory afterwards, so a
    # repository whose every positive later lost the §3.3 tie keeps a context and is absent
    # from the analysis set. That is a correct walk, and refusing it would abort the run on
    # exactly the data the contest is supposed to produce.
    missing = sorted(k for k in walked if k not in seen)
    if missing:
        raise SystemExit(
            f"{len(missing)} case(s) in the analysis set have no persisted T0 context: "
            f"{missing[:5]}.\n  §3.1 persists one for every repository with a usable row, so a\n"
            "  positive without one means the directory is not the walk's own output."
        )


def assert_context_is_judgeable(case: str, text: str) -> None:
    """A context both judges can actually score, rather than a truthy string.

    The check is **substantive, not schematic**: it demands content, not a section layout.
    Requiring a README plus a manifest-or-listing looked stricter and was simply wrong for the
    population the walk admits. `eligibility.LANGUAGES` includes Julia, R and Fortran, whose
    source extensions are absent from `t0_context`'s `exts`, so those repositories emit no
    "Source files" section at all; and `eligibility.README_NAMES` accepts `README` and
    `Readme.md`, which `t0_context` never reads, so a repository can pass the English-README
    screen and still emit no README section. Either would have aborted a healthy run and
    blamed a missing clone.

    What must abort is the header-only context — the missing-clone signature. `t0_context` runs
    git through `subprocess.run` with no `check`, so against an unfetched clone the listing
    comes back empty, no section is emitted, and it returns a truthy one-line string naming the
    repository. Both judges would score every paper against six words, `void` would stay 0, the
    cache gate would pass, and the run would look finished. §6.6 makes the T0 context the thing
    that removes the *outcome* from the judge's view; six words remove the repository too.

    A header-only context is ambiguous in cause — an unfetched clone and an exotic-but-eligible
    repository produce the identical string — and deliberately not disambiguated: it is
    unjudgeable either way, so the refusal does not depend on knowing which happened.
    """
    header = f"Repository: {case}"
    rest = text.split(header, 1)[-1] if header in text else text
    if not rest.strip():
        # The missing-clone signature, and the only thing that must abort. `t0_context` runs git
        # with no `check`, so against an unfetched clone the listing is empty, no section is
        # emitted, and it returns a TRUTHY one-line string naming the repository. Both judges
        # would score every paper against six words, `void` would stay 0, the cache gate would
        # pass, and the run would look finished.
        raise SystemExit(
            f"{case}: T0 context is the header and nothing else ({len(text)} chars).\n"
            "  Six words are not a repository. This is what a missing or unfetched clone\n"
            "  produces, and it cannot be judged whatever its cause."
        )
    if README_SECTION in text:
        after = text.split(README_SECTION, 1)[1]
        # Split on the sections `t0_context` actually emits, never on a bare "## ". A README
        # body carries its own markdown headings — `graph`'s real T0 context contains
        # "## Library Highlights" inside the README excerpt — and a generic scan cannot tell
        # one from a section boundary. It failed both ways: it accepted a context with no file
        # listing whenever the README happened to contain a heading, and it truncated the body
        # at that heading, so a README opening with an H2 read as EMPTY and aborted a good run.
        starts = [after.index(m) for m in KNOWN_SECTIONS if m in after]
        if not (after[: min(starts)] if starts else after).strip():
            raise SystemExit(f"{case}: T0 context has an EMPTY README section")


def context_digests(walk_csv: Path | None = None) -> dict[str, str]:
    """`{case: digest}` read from each qualifying row's `note` cell in the walk ledger.

    The digest is the walk's own record of what it wrote. Reading it, rather than searching the
    directory for something that looks right, is what makes a replaced or truncated context a
    hash failure instead of a successful load of the wrong bytes.
    """
    src = walk_csv or POOL_WALK
    if not src.is_file():
        return {}
    out: dict[str, str] = {}
    with src.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            note = (row.get("note") or "").strip()
            if note.startswith("context "):
                out[str(row["full_name"])] = note.split("context ", 1)[1].strip()
    return out


def t0_context_for(
    case: str, digest: str, contexts: Path | None = None, *, validate: bool = True
) -> str:
    """The persisted T0 context for *case*, at the exact digest the walk recorded.

    §3.1 persists it "so judging never re-clones". Re-mining it here would also change what the
    judge is shown: the clone is deleted after the walk, and a fresh one is a different
    repository state from the one the label was mined against.
    """
    import walk_pool

    root = contexts or POOL_CONTEXTS
    path = root / f"{case_key(case)}.{digest}.txt"
    if not path.is_file():
        raise SystemExit(
            f"{case}: no persisted T0 context at {path.name}.\n"
            "  Judging must never re-clone (§3.1), and must never build a context from a\n"
            "  missing clone, which returns a truthy one-line string rather than failing."
        )
    text = path.read_text(encoding="utf-8")
    actual = walk_pool.context_hash("t0", text)
    if actual != digest:
        raise SystemExit(
            f"{case}: the T0 context does not match the digest the walk recorded.\n"
            f"  recorded {digest}, file hashes to {actual} — the judged prompt would not be\n"
            "  the prompt the label was mined against."
        )
    if validate:
        assert_context_is_judgeable(case, text)
    return text


def head_ids_for(cases: list[str], head_ids: Path | None = None) -> dict[str, set[str]]:
    """`{case: every identifier that case cites at HEAD}`, built forward from the positives.

    §4 excludes a "control" the repository actually went on to cite. Without this the negative
    class silently contains positives, which biases the AUC toward 0.5 — the study's own
    pre-committed null, arriving by accident and looking exactly like a finding. So a missing or
    empty file is a refusal here, never a `.get(case, set())` default.

    **A stated limitation rather than an enforcement claim:** this is the extractor-v2
    identifier set over the documentation globs, so §4's "not cited anywhere in the repository
    at HEAD" is honoured for the arXiv and Hugging Face documentation channel only. A paper
    cited by DOI, by PMID, in source comments or inside a notebook is not excluded.
    """
    root = head_ids or POOL_HEAD_IDS
    assert_keys_are_injective(cases)
    out: dict[str, set[str]] = {}
    for case in sorted(set(cases)):
        path = root / f"{case_key(case)}.json"
        if not path.is_file():
            raise SystemExit(
                f"{case}: no head_ids at {path.name}. §4 needs the repository's HEAD citation\n"
                "  set to keep a paper it actually cited out of the negative class; an empty\n"
                "  default turns that rule into a no-op and biases the AUC toward 0.5."
            )
        ids = json.loads(path.read_text(encoding="utf-8"))
        if not ids:
            raise SystemExit(f"{case}: head_ids is empty — the file is stale or truncated.")
        out[case] = {dedup_id(str(i)) for i in ids}
    return out


def assert_positives_are_cited_at_head(
    positives: list[dict[str, Any]], head_ids: dict[str, set[str]]
) -> None:
    """Every positive was mined FROM its repository's HEAD, so it must appear in that set.

    An absence does not mean the paper is wrong; it means the head_ids file is stale or
    truncated. A truncated one under-excludes, putting genuine citations into the negative
    class — the one direction §4 cannot afford to be wrong in.
    """
    for row in positives:
        case, pid = str(row["case"]), dedup_id(str(row["id"]))
        if pid not in head_ids.get(case, set()):
            raise SystemExit(
                f"{case}: positive {pid} is absent from its own head_ids set.\n"
                "  It was mined from that HEAD, so this cannot be the file the walk wrote."
            )


def analysis_set(
    seed: str,
    *,
    pool: Path | None = None,
    legacy: Path | None = None,
    require_pool: bool = True,
) -> dict[str, Any]:
    """Both strata, their ledgers, and the invariants that must hold across them.

    The two strata are kept labelled rather than merged into one list, because §5's
    transportability endpoint is a contrast BETWEEN them and §9's power table budgets them
    separately ("60 new + 30 legacy"). A merged set cannot answer either.
    """
    legacy_rows, legacy_ledger = legacy_positives(seed, legacy)
    if require_pool or (pool or POOL_ADOPTIONS).is_file():
        pool_rows, pool_ledger = pool_positives(pool)
    else:
        pool_rows, pool_ledger = (
            [],
            {
                "stratum": "pool",
                "n_rows": 0,
                "n_positives": 0,
                "n_clusters": 0,
                "_absent": "the walk has not mined anything yet",
            },
        )

    # §3.3: legacy wins ties, so no identifier may appear in both strata. Compared on the RAW
    # id string because that is the key `assign_across_repos` and `legacy_ids` both group on;
    # a stricter comparison here would report a violation the contest cannot actually produce.
    overlap = {str(r["id"]) for r in legacy_rows} & {str(r["id"]) for r in pool_rows}
    if overlap:
        raise SystemExit(
            f"{len(overlap)} identifier(s) are positives in BOTH strata: {sorted(overlap)[:5]}.\n"
            "  §3.3 gives legacy the tie, so the contest should have removed every one of\n"
            "  these from the pool side. §5's legacy-versus-pool contrast would otherwise\n"
            "  compare the two clusters over an overlapping set of papers."
        )

    # Paper-level dedup is part of §5's primary endpoint definition, so a cluster holding one
    # paper twice is a defect in the endpoint rather than untidiness in the data.
    for rows, name in ((legacy_rows, "legacy"), (pool_rows, "pool")):
        seen: dict[tuple[str, str], str] = {}
        for row in rows:
            k = (str(row["case"]), dedup_id(str(row["id"])))
            if k in seen:
                raise SystemExit(f"{name}: {k[0]} holds {k[1]} twice after the contest")
            seen[k] = str(row["id"])

    for ledger in (legacy_ledger, pool_ledger):
        orphans = ledger.get("assigned_to_a_non_positive") or []
        if orphans:
            # Not fatal, and deliberately not silent: §3.3 says a shared identifier is "counted
            # once", and these are counted ZERO times — the loser is uncounted by rule and the
            # winner is dropped by a filter. Reported with ids so the loss can be checked.
            print(
                f"  ! {ledger['stratum']}: {len(orphans)} identifier(s) assigned to a row that "
                f"is not a positive, so they are counted zero times: "
                f"{[o['id'] for o in orphans[:5]]}"
            )

    positives = legacy_rows + pool_rows
    return {
        "seed": seed,
        "positives": positives,
        "by_stratum": {"legacy": legacy_rows, "pool": pool_rows},
        "ledgers": {"legacy": legacy_ledger, "pool": pool_ledger},
        # §9's power table is stated in these terms, and the two are never conflated with the
        # walk's own `capped_positives`: that is the quantity the STOP RULE counted, before the
        # cross-repository contest removed anything.
        "analysis_set_positives": len(positives),
        "n_clusters": len({(str(r["stratum"]), str(r["case"])) for r in positives}),
        "largest_cluster_share": (
            round(
                max(
                    sum(1 for r in positives if str(r["case"]) == c)
                    for c in {str(r["case"]) for r in positives}
                )
                / len(positives),
                4,
            )
            if positives
            else None
        ),
    }


# ---------------------------------------------------------------------------------------
# Arm-neutral prompts (§4)
# ---------------------------------------------------------------------------------------
def normalise_text(value: Any) -> str:
    """Collapse every run of whitespace to one space. Applied identically to both arms.

    Not cosmetic. `diagnose_triage.fetch_papers` stores `" ".join(text.split())` while
    `collector._result_to_paper` stores `result.summary` verbatim, and the two arms were built
    by those two different paths — so **82 of 674** control-shaped abstracts in
    `.work/pool-cut100/ann.json` carry embedded newlines, and a mined positive never can.
    """
    return " ".join(str(value or "").split())


def judgeable_item(paper: dict[str, Any], case: str, arm: str) -> dict[str, Any]:
    """One paper, shaped so nothing about it says which arm it came from.

    §4 requires "byte-identical rubric, a T0 context ... **no arm**". The context was arm-neutral
    and the paper was not:

    * **The identifier.** `collector._result_to_paper` stores `result.get_short_id()`, which is
      VERSIONED — measured, **674 of 674** candidates in `.work/pool-cut100/ann.json` — while
      every mined positive id is unversioned by construction, because the extractor regexes
      capture `\\d{4}\\.\\d{4,5}` with no version group (**0 of 120** legacy rows are versioned).
      `judge._build_user_prompt` prints the id verbatim, so in NR-56/57 every control prompt
      read `arXiv: 2409.11629v1` and every positive `arXiv: 2409.11629`. That is a perfect,
      deterministic arm marker in the one place the design says must carry none — and whether
      either judge keyed on it is unmeasurable after the fact.
    * **The abstract.** See `normalise_text`.

    So both arms are assembled here, through one function, and every field the prompt prints is
    normalised the same way. `assert_arm_neutral` checks the result rather than trusting it.

    **Consequence, stated plainly: the 496 control verdicts already in
    `.work/judge_validity_verdicts.json` were bought under the old prompt shape and are not
    reusable for this pool.** They remain NR-56/57's record and are not rewritten.
    """
    pid = dedup_id(str(paper.get("arxiv_id") or ""))
    return {
        "case": case,
        "arm": arm,
        "arxiv_id": pid,
        "title": normalise_text(paper.get("title")),
        "abstract": normalise_text(paper.get("abstract")),
        "primary_category": str(paper.get("primary_category") or ""),
    }


def assert_arm_neutral(items: list[dict[str, Any]]) -> None:
    """Refuse to buy a verdict on a set whose two arms are distinguishable by shape.

    Checked over the assembled items rather than asserted of the code that made them: the two
    arms reach this point through different fetch paths, and the whole failure being guarded
    against is that one of those paths quietly reintroduces a marker.
    """
    for item in items:
        pid = str(item.get("arxiv_id") or "")
        if not pid:
            raise SystemExit(f"{item.get('case')}: an item has no arXiv id and cannot be judged")
        if dedup_id(pid) != pid:
            raise SystemExit(
                f"{item['case']}/{pid}: identifier is not in its canonical form.\n"
                "  A version suffix on one arm and not the other is a deterministic arm marker\n"
                "  in a prompt §4 requires to carry none."
            )
        for field in ("title", "abstract"):
            if str(item.get(field) or "") != normalise_text(item.get(field)):
                raise SystemExit(f"{item['case']}/{pid}: {field} is not normalised")
    arms = {item["arm"] for item in items}
    if items and len(arms) < 2:
        raise SystemExit(f"only one arm present ({arms}); there is nothing to discriminate")


def judgeable_items(
    positives: list[dict[str, Any]],
    controls: list[dict[str, Any]],
    *,
    fetch: Any = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Both arms, fetched by ONE path and shaped by one function.

    Returns the items and the ids that could not be fetched. A dropped paper is returned rather
    than silently skipped: a positive that never reaches the judge shrinks n without shrinking
    anything visible, which is this project's recurring failure and the reason
    `enrich_positives` hands back its own `missing` list too.
    """
    if fetch is None:
        from diagnose_triage import fetch_papers as fetch

    wanted = sorted({dedup_id(str(r["id"])) for r in positives})
    fetched = fetch(wanted) if wanted else {}

    items: list[dict[str, Any]] = []
    missing: list[str] = []
    for row in positives:
        pid = dedup_id(str(row["id"]))
        paper = fetched.get(pid) or fetched.get(str(row["id"]))
        if not paper:
            missing.append(pid)
            continue
        items.append(judgeable_item({**paper, "arxiv_id": pid}, str(row["case"]), "adopted"))
    for row in controls:
        # Controls arrive with their paper attached from the listing, so they are not re-fetched
        # — but they go through the identical shaping, which is the part that matters.
        paper = row.get("paper") or {}
        pid = dedup_id(str(paper.get("arxiv_id") or row.get("id") or ""))
        if not pid:
            missing.append(str(row.get("id") or "?"))
            continue
        items.append(judgeable_item({**paper, "arxiv_id": pid}, str(row["case"]), "control"))
    assert_arm_neutral(items)
    return items, missing


# ---------------------------------------------------------------------------------------
# The arm-neutral control draw (§4)
# ---------------------------------------------------------------------------------------
LISTINGS = POOL_FRAME / "listings"  # committed, URL-free manifests
LISTINGS_RAW = WORK / "listings-raw"  # untracked payloads
CONTROLS_ROWS = POOL_FRAME / "controls-arxiv-window.json"  # committed, URL-free
CONTROLS_PAYLOAD = WORK / "controls-arxiv-window-payload.json"  # untracked


def _listing_name(category: str, lo: str) -> str:
    return f"{category.replace('.', '_')}-{lo[:6]}.json"


def archived_listing(
    category: str, lo: str, hi: str, *, want: int, live: Any, raw: Path, manifest: Path
) -> list[dict[str, Any]]:
    """One (category, half-year) listing, fetched at most once and replayed ever after.

    §4: "The per-(category, half-year) listing is archived, because it is the negative class of
    the primary endpoint and must be reproducible." Two artefacts, following §2.1's
    `trim_archive` precedent:

    * the **raw** payload stays UNTRACKED, because arXiv abstracts routinely contain
      `github.com/<owner>/<repo>` strings and §11's prior-exposure grep is precisely what this
      pool must not contaminate;
    * a **committed, URL-free manifest** carries the query, the counts and the ordered
      identifiers with their primary categories — enough to check that the draw used the
      listing it says it did.

    **Write-once, and a disagreement is a blocking failure.** arXiv's `cat:` membership drifts
    as papers are cross-listed and withdrawn, so a second fetch of the same window can return a
    different set. Silently preferring either one would make the negative class depend on when
    somebody happened to run it.
    """
    raw_file = raw / _listing_name(category, lo)
    man_file = manifest / _listing_name(category, lo)
    if raw_file.is_file():
        stored = json.loads(raw_file.read_text(encoding="utf-8"))
        papers: list[dict[str, Any]] = stored["papers"]
        if man_file.is_file():
            recorded = json.loads(man_file.read_text(encoding="utf-8"))
            got = [dedup_id(str(p.get("arxiv_id") or "")) for p in papers]
            if got != [str(i) for i in recorded["ids"]]:
                raise SystemExit(
                    f"{man_file.name}: the archived listing and its manifest disagree.\n"
                    "  One of the two has been edited; the negative class cannot be trusted."
                )
        return papers

    papers = live(category, lo, hi, want=want, archive=None)
    raw_file.parent.mkdir(parents=True, exist_ok=True)
    raw_file.write_text(
        json.dumps({"query": f"cat:{category} [{lo} TO {hi}]", "papers": papers}, indent=1),
        encoding="utf-8",
    )
    man_file.parent.mkdir(parents=True, exist_ok=True)
    write_artifact(
        man_file,
        {
            "category": category,
            "window": f"{lo}..{hi}",
            "requested": want,
            "returned": len(papers),
            # `returned == requested` is the only signal available that a slice hit its own cap,
            # and "200 of 13,262" and "200 of 200" are the same number without it.
            "truncated": len(papers) >= want,
            "ids": [dedup_id(str(p.get("arxiv_id") or "")) for p in papers],
            "primary_categories": [str(p.get("primary_category") or "") for p in papers],
            "has_abstract": [bool(str(p.get("abstract") or "").strip()) for p in papers],
            "raw_sha256": hashlib.sha256(raw_file.read_bytes()).hexdigest(),
        },
    )
    return papers


def draw_controls(
    positives: list[dict[str, Any]],
    *,
    seed: str,
    head_ids: dict[str, set[str]],
    rows_out: Path | None = None,
    payload_out: Path | None = None,
    raw: Path | None = None,
    manifest: Path | None = None,
    listing: Any = None,
    enrich: Any = None,
) -> list[dict[str, Any]]:
    """§4's arm-neutral negative class, drawn once under SEED_POOL and reused thereafter.

    The draw is an artefact, not a computation repeated at each entry point: `plan`, the
    purchase loop and the analysis must all see the same negatives, and a redraw between them
    would change n2 after verdicts had been bought. Written when absent, replayed when present,
    and never overwritten — a second draw is a new artefact under a new name.

    *seed* is the verified `SEED_POOL`, never `judge_validity_adoption.SEED`. §4 says controls
    are "drawn per positive under SEED_POOL", and the module constant seeds the NR-57
    reproduction, which is a different study over a different negative class.
    """
    import judge_validity_adoption as jva

    rows_file = rows_out or CONTROLS_ROWS
    payload_file = payload_out or CONTROLS_PAYLOAD
    if rows_file.is_file() and payload_file.is_file():
        rows = json.loads(rows_file.read_text(encoding="utf-8"))["controls"]
        payload = json.loads(payload_file.read_text(encoding="utf-8"))
        if payload.get("seed") != seed:
            raise SystemExit(
                f"{rows_file.name} was drawn under a different seed.\n"
                "  §4 draws under SEED_POOL; a redraw would change the negative class after\n"
                "  verdicts had been bought against the old one."
            )
        for row in rows:
            row["paper"] = payload["papers"][row["id"]]
        jva.refuse_an_empty_control_set(positives, rows)
        return rows

    enriched, missing = (enrich or jva.enrich_positives)(positives)
    if missing:
        # Returned rather than dropped, and acted on rather than logged: a positive with no
        # category and no date draws no controls, so ignoring the list would shrink the
        # negative class silently.
        print(f"  ! {len(missing)} positive(s) could not be enriched: {missing[:5]}")
    if not enriched:
        raise SystemExit(
            "no positive could be enriched with a primary category and a submission date, so\n"
            "  §4's matched window cannot be resolved for any of them."
        )

    live = listing or jva.arxiv_window_listing
    raw_dir, man_dir = raw or LISTINGS_RAW, manifest or LISTINGS

    def archived(category: str, lo: str, hi: str, *, archive: Path | None = None) -> Any:
        return archived_listing(
            category,
            lo,
            hi,
            want=jva.LISTING_PER_WINDOW,
            live=live,
            raw=raw_dir,
            manifest=man_dir,
        )

    rows = jva.arxiv_window_controls(enriched, head_ids, seed, listing=archived)
    jva.refuse_an_empty_control_set(enriched, rows)

    papers = {row["id"]: row["paper"] for row in rows}
    payload_file.parent.mkdir(parents=True, exist_ok=True)
    payload_file.write_text(
        json.dumps({"seed": seed, "papers": papers}, indent=1), encoding="utf-8"
    )
    # The committed half carries no paper text and therefore no URLs (§2.1).
    per_positive = Counter(row["for_positive"] for row in rows)
    write_artifact(
        rows_file,
        {
            "seed": seed,
            "scheme": "arxiv-window",
            "stream_key_format": "{seed}:{case}:{dedup_id(positive_id)}",
            "n_positives": len(enriched),
            "n_control_rows": len(rows),
            "n_distinct_control_papers": len({row["id"] for row in rows}),
            # Cross-cluster sharing is a correlation the repository-cluster bootstrap does not
            # capture, so the interval is very slightly too narrow. Reported rather than fixed:
            # dropping the loser of a contest after the draw would strip a positive of controls
            # and change n2, which would be an instrument change.
            "n_control_papers_in_more_than_one_cluster": len(
                {
                    pid
                    for pid in {row["id"] for row in rows}
                    if len({row["case"] for row in rows if row["id"] == pid}) > 1
                }
            ),
            "controls_per_positive": dict(sorted(Counter(per_positive.values()).items())),
            "not_enriched": missing,
            "controls": [{k: v for k, v in row.items() if k != "paper"} for row in rows],
        },
    )
    return rows


# ---------------------------------------------------------------------------------------
# The purchase loop (§3.1, §4, §7, §10 step 7)
# ---------------------------------------------------------------------------------------
POOL_VERDICTS = WORK / "judge_validity_pool_verdicts.json"
PURCHASE_LOCK = WORK / "judge_validity_pool.lock"
LEGACY_VERDICTS = WORK / "judge_validity_verdicts.json"
CHECKPOINT = 20


def verdict_key(model: str, case: str, pid: str) -> str:
    """One key shape, with the paper id normalised here and nowhere else."""
    return f"{model}|{case}|{dedup_id(str(pid))}"


def _save_store(store: Path, have: dict[str, Any]) -> None:
    if store.resolve() == LEGACY_VERDICTS.resolve():
        raise SystemExit(
            "REFUSED: that is the NR-56/57 verdict store.\n"
            "  566 paid records that §4's pool-scheme result is reported from. §1's rule for\n"
            "  adoptions.json applies to it for the same reason: it is the record this study\n"
            "  is compared against."
        )
    store.parent.mkdir(parents=True, exist_ok=True)
    tmp = store.with_suffix(store.suffix + ".tmp")
    tmp.write_text(json.dumps(have, indent=0), encoding="utf-8")
    os.replace(tmp, store)


def import_legacy_verdicts(
    have: dict[str, Any],
    digests: dict[str, str],
    *,
    source: Path | None = None,
    recompute: Any = None,
) -> dict[str, int]:
    """Reuse an NR-56/57 verdict only when it answered THIS prompt, decided by recomputation.

    Every one of the 566 stored records is exactly `{score, arm}` — no context digest, because
    nothing recorded one. So the test cannot be "refuse a record without a digest": that would
    make this path dead code. It is "rebuild the T0 context at the revision that verdict was
    bought against and compare the digest to the one this run will use".

    **Two separate facts, both measured 2026-09-03, and they point opposite ways.** §7 says the
    legacy re-mine pins T0 "so that `t0` reproduces to the SHA" — that is **false for 2 of 9
    cases**: `peft` moved `4c3a76fa68` → `e8ba7de573` and `llminfer` `d565bb2fd5` →
    `8b3befc0e2`. But the T0 **context** at those two pairs of revisions is byte-identical
    (`peft` 597f0159179d, `llminfer` d912ce5d4a6a both ways): the commits differ in code, not in
    the documentation the prompt is built from. A SHA gate would have discarded both cases'
    verdicts for a difference the judge never saw. The digest is what the judge was shown, so
    the digest is what decides.

    Reuse is small either way, and that is stated rather than discovered later: 33 of v1's 35
    usable positives survive into v2 (both losses are `rl`, which contributes no usable v2
    rows), the cap of 8 bounds it further, and **all 496 control verdicts are pool-scheme draws
    under the old prompt shape** — versioned identifiers, unnormalised abstracts — so none of
    them is reusable here at all.
    """
    src = source or LEGACY_VERDICTS
    tally = {"imported": 0, "digest_mismatch": 0, "no_context": 0, "not_a_positive": 0}
    if not src.is_file():
        return tally
    legacy = json.loads(src.read_text(encoding="utf-8"))
    for key, record in legacy.items():
        model, case, pid = key.split("|", 2)
        if str(record.get("arm")) != "adopted":
            # Controls were drawn under the pool scheme and printed with a version suffix.
            tally["not_a_positive"] += 1
            continue
        want = digests.get(case)
        if not want:
            tally["no_context"] += 1
            continue
        was = (recompute or (lambda c: None))(case)
        if was != want:
            tally["digest_mismatch"] += 1
            continue
        have.setdefault(
            verdict_key(model, case, pid),
            {
                "score": int(record["score"]),
                "arm": "adopted",
                "model": model,
                "case": case,
                "id": dedup_id(str(pid)),
                "context_digest": want,
                "source": "nr56-57",
            },
        )
        tally["imported"] += 1
    return tally


def buy_verdicts(
    items: list[dict[str, Any]],
    contexts: dict[str, tuple[str, str]],
    *,
    judges: dict[str, Any],
    store: Path | None = None,
    scheme: str = "arxiv-window",
    checkpoint: int = CHECKPOINT,
    gate: Any = None,
    lock: Path | None = None,
    done_out: Path | None = None,
) -> dict[str, Any]:
    """Buy both judges' verdicts for every item, carrying the cache gate the whole way.

    Factored out of `judge()` so the gate cannot be left behind by a future entry point, and so
    a whole run is testable with fakes and no network. §7's cache clause binds every path that
    buys a T0 verdict, not only the one that happened to exist when it was written.

    **Every item exits with an outcome**, persisted as a list rather than the integer `void`
    the old loop counted and printed once. §3.1 states the principle one stage earlier: "A
    timeout is a recorded outcome, never a silent skip", and §10 step 7 publishes the void and
    timeout lists in the datasheet. A judge whose verdicts are 60 % complete produces an AUC
    over a different sample from the other judge's, and §5 compares the two directly — so the
    run ends by checking that every expected verdict is either present or explained, and raises
    on any that is neither.

    The gate runs at pre-flight, at every checkpoint and at the end, and the store is written
    **before** any raise: a mid-run trip must abort without also losing what was already paid
    for.
    """
    dest = store or POOL_VERDICTS
    lock_file = lock or PURCHASE_LOCK
    if lock_file.is_file():
        raise SystemExit(
            f"{lock_file} exists: {lock_file.read_text(encoding='utf-8').strip()}\n"
            "  Another purchase run holds it. Two loops against one store would interleave\n"
            "  writes and each would overwrite the other's verdicts."
        )

    import judge_validity_adoption as jva

    check = gate if gate is not None else jva.isolation_failures
    cache_roots, cache_before = ((), {}) if gate is not None else jva.prepare_isolation()

    have: dict[str, Any] = {}
    if dest.is_file():
        have = json.loads(dest.read_text(encoding="utf-8"))

    lock_file.parent.mkdir(parents=True, exist_ok=True)
    lock_file.write_text(f"pid {os.getpid()} started {_now()}", encoding="utf-8")
    outcomes: list[dict[str, Any]] = []
    bought = 0

    def fire_gate() -> None:
        _save_store(dest, have)  # money first, guard second
        failures = check(cache_before, cache_roots, None, None) if gate is None else check()
        if failures:
            raise SystemExit("\n\n".join(failures))

    try:
        for n, item in enumerate(items, start=1):
            case, pid, arm = str(item["case"]), str(item["arxiv_id"]), str(item["arm"])
            pair = contexts.get(case)
            if pair is None:
                outcomes.append({"case": case, "id": pid, "arm": arm, "outcome": "no_context"})
                continue
            context, digest = pair
            for model, ask in judges.items():
                key = verdict_key(model, case, pid)
                prior = have.get(key)
                if prior is not None:
                    if str(prior.get("arm")) != arm:
                        # A stored flag nobody reads catches nothing — which is what the
                        # assigned-but-unread control scheme already demonstrated.
                        raise SystemExit(
                            f"{key}: stored arm {prior.get('arm')!r} but this run says {arm!r}.\n"
                            "  The same paper cannot be a positive and a control in one study."
                        )
                    if str(prior.get("context_digest")) == digest:
                        continue
                    # Bought against a different prompt. Re-bought, never reused: the context is
                    # what makes a T0 verdict a T0 verdict.
                try:
                    score = int(ask(case, context, item, model))
                    have[key] = {
                        "score": score,
                        "arm": arm,
                        "model": model,
                        "case": case,
                        "id": pid,
                        "context_digest": digest,
                        "scheme": scheme,
                        "source": "pool",
                    }
                    bought += 1
                    outcomes.append(
                        {"case": case, "id": pid, "arm": arm, "outcome": "judged", "model": model}
                    )
                except TimeoutError:
                    outcomes.append(
                        {"case": case, "id": pid, "arm": arm, "outcome": "timeout", "model": model}
                    )
                except Exception as exc:  # noqa: BLE001 — one bad paper must not lose the rest
                    outcomes.append(
                        {
                            "case": case,
                            "id": pid,
                            "arm": arm,
                            "outcome": "judge_error",
                            "model": model,
                            "error": type(exc).__name__,
                        }
                    )
            if n % checkpoint == 0:
                fire_gate()
                print(f"  [{n}/{len(items)}] bought {bought}", flush=True)
        fire_gate()
    finally:
        lock_file.unlink(missing_ok=True)

    coverage = verdict_coverage(items, have, outcomes, judges=list(judges))
    record = {
        "scheme": scheme,
        "n_items": len(items),
        "bought": bought,
        "coverage": coverage,
        # Lists, not a count. §10 step 7 publishes them, and "17 void" cannot be checked
        # against anything.
        "outcomes": [o for o in outcomes if o["outcome"] != "judged"],
    }
    write_artifact(done_out or JUDGING_DONE, record)
    return record


def verdict_coverage(
    items: list[dict[str, Any]],
    have: dict[str, Any],
    outcomes: list[dict[str, Any]],
    *,
    judges: list[str],
) -> dict[str, Any]:
    """Per judge: expected, present, explained-away, and unexplained. Unexplained raises.

    An absence with no recorded reason is the failure this whole loop is arranged around: one
    arXiv 503 drops a hundred ids with a single printed line, and a judge missing those papers
    produces an AUC over a different sample from the other judge's — which §5 then compares
    directly, as though the two numbers were about the same thing.
    """
    out: dict[str, Any] = {}
    explained = {
        (o["model"], o["case"], o["id"])
        for o in outcomes
        if o["outcome"] != "judged" and "model" in o
    }
    no_context = {(o["case"], o["id"]) for o in outcomes if o["outcome"] == "no_context"}
    for model in judges:
        expected = [i for i in items if (str(i["case"]), str(i["arxiv_id"])) not in no_context]
        present = [
            i for i in expected if verdict_key(model, str(i["case"]), str(i["arxiv_id"])) in have
        ]
        void = [i for i in expected if (model, str(i["case"]), str(i["arxiv_id"])) in explained]
        missing = len(expected) - len(present) - len(void)
        out[model] = {
            "n_expected": len(expected),
            "n_present": len(present),
            "n_void_recorded": len(void),
            "n_missing_unexplained": missing,
            "coverage": round(len(present) / len(expected), 4) if expected else None,
        }
        if missing > 0:
            raise SystemExit(
                f"{model}: {missing} expected verdict(s) are neither present nor explained.\n"
                "  An absence with no recorded reason makes this judge's AUC a statement about\n"
                "  a different sample from the other judge's, which §5 compares directly."
            )
    return out


def _now() -> str:
    return datetime.now(tz=UTC).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------------------
# Section 5's endpoints
# ---------------------------------------------------------------------------------------
BOOTSTRAP_ITERS = 5000
ORDINAL_LEVELS = (0, 1, 2, 3)

# §9's two sizing rows, quoted verbatim so the discrepancy below is checkable rather than
# asserted. (capped positives, SE low, SE high, CI at AUC 0.60, MDA as printed)
PREREG_S9_ROWS = (
    {"positives": 90, "se": [0.047, 0.057], "ci_at_060": [0.489, 0.711], "mda_printed": 0.62},
    {"positives": 130, "se": [0.036, 0.044], "ci_at_060": [0.514, 0.686], "mda_printed": 0.578},
)


def cluster_key(row: dict[str, Any]) -> str:
    """The resampling unit: a repository, qualified by its stratum.

    Qualified because §5's transportability endpoint contrasts the two strata, and an
    unqualified slug that existed in both would silently merge them into one cluster.
    """
    return f"{row.get('stratum', 'pool')}:{row['case']}"


def _pairs(
    rows: list[dict[str, Any]],
    verdicts: dict[str, Any],
    model: str,
    strata: dict[str, str],
    *,
    threshold: int | None = None,
) -> list[tuple[str, float]]:
    """`(cluster, score)` for every row this judge scored.

    *threshold* None gives the RAW ordinal score, which is what the primary takes. Passing a
    threshold gives the 0/1 form, which is what the secondary takes — and the two must never be
    swapped: an AUC over a 2-valued array is exactly `0.5 + (p_adopted - p_control) / 2`, so the
    primary would become a monotone restatement of the secondary and would carry back in the
    level §5 makes it level-free to remove.
    """
    out: list[tuple[str, float]] = []
    for row in rows:
        case, pid = str(row["case"]), dedup_id(str(row["id"]))
        record = verdicts.get(verdict_key(model, case, pid))
        if record is None:
            continue
        score = float(record["score"])
        key = f"{strata.get(case, 'pool')}:{case}"
        out.append(
            (
                key,
                1.0
                if threshold is not None and score >= threshold
                else 0.0
                if threshold is not None
                else score,
            )
        )
    return out


def _histogram(pairs: list[tuple[str, float]]) -> dict[str, int]:
    counts = Counter(int(s) for _, s in pairs)
    return {str(level): counts.get(level, 0) for level in ORDINAL_LEVELS}


def tie_fraction(
    positives: list[tuple[str, float]], controls: list[tuple[str, float]]
) -> float | None:
    """The share of positive-control pairs that tie, and therefore contribute exactly 0.5.

    `roc_auc` is the Mann-Whitney form over average ranks, so a tie is not an error — but with
    four rubric levels spread over hundreds of papers a large fraction of the comparisons are
    decided by nothing at all, and an AUC of 0.58 built from 70 % ties is a different claim from
    one built from 5 %. Published beside every AUC rather than left to be inferred.
    """
    if not positives or not controls:
        return None
    p, c = Counter(int(s) for _, s in positives), Counter(int(s) for _, s in controls)
    shared = sum(p[k] * c[k] for k in set(p) & set(c))
    return round(shared / (len(positives) * len(controls)), 4)


def prereg_s9_reference() -> dict[str, Any]:
    """§9's sizing table beside the constant this code actually uses, and their disagreement.

    Measured, not asserted. §9's **interval** column is exactly `0.60 ± 1.96 × SE_upper` and
    reproduces to the printed digits. Its **minimum-detectable-AUC** column does not come from
    the committed `0.5 + 2.80 × SE`: that formula gives 0.660 at 90 positives and 0.623 at 130,
    against §9's ≈ 0.62 and ≈ 0.578, and the implied multipliers are 2.11 and 1.77 — nearer
    `1.96 × SE`, which is a 50 %-power quantity, than the 2.80 = 1.96 + 0.84 that 80 % power
    requires.

    Neither number is changed here. The constant is frozen by the registration banner and §9 is
    registered text; this records that the two disagree and by how much, so a reader comparing
    the artefact against the table is not left to wonder which is wrong. Any claim about whether
    a particular AUC was detectable is made from the **realised** `min_detectable_auc_80pct`
    after the bootstrap, never from this table.
    """
    rows = []
    for row in PREREG_S9_ROWS:
        se_hi = row["se"][1]
        rows.append(
            {
                **row,
                "ci_recomputed_at_1_96_se_upper": [
                    round(0.60 - 1.96 * se_hi, 3),
                    round(0.60 + 1.96 * se_hi, 3),
                ],
                "mda_from_committed_formula": [
                    round(0.5 + 2.80 * row["se"][0], 3),
                    round(0.5 + 2.80 * se_hi, 3),
                ],
                "implied_multiplier_of_printed_mda": round((row["mda_printed"] - 0.5) / se_hi, 2),
            }
        )
    return {
        "_comment": (
            "§9's interval column reproduces exactly as 0.60 +/- 1.96*SE_upper. Its "
            "minimum-detectable-AUC column does not come from the committed 0.5 + 2.80*SE; the "
            "implied multipliers are 2.11 and 1.77, nearer 1.96*SE (50% power) than 2.80 "
            "(80%). Recorded, not reconciled: the constant is frozen and §9 is registered "
            "text. Detectability claims are made from the realised value, never from this table."
        ),
        "committed_formula": "0.5 + 2.80 * se",
        "rows": rows,
    }


def primary_auc(
    model: str,
    positives: list[dict[str, Any]],
    controls: list[dict[str, Any]],
    verdicts: dict[str, Any],
    *,
    iters: int = BOOTSTRAP_ITERS,
) -> dict[str, Any]:
    """§5's primary: repository-clustered AUC of the judge's ORDINAL score, adopted vs control.

    Over the **pooled** analysis set — §9's power table reads "90 (60 new + 30 legacy)" and
    "130 (100 new + 30 legacy)", so the primary is legacy and pool together and the two strata
    are the transportability arms, not two primaries.

    The estimator is `judge_validity_adoption.cluster_bootstrap_auc`, called and not
    reimplemented: it is pinned by its own tests and frozen by the registration banner, and its
    paper-level interval — computed internally only so the design effect is a measured ratio —
    is deliberately not surfaced, for the reason its own comment gives.
    """
    import judge_validity_adoption as jva

    strata = {str(r["case"]): str(r.get("stratum", "pool")) for r in positives}
    pos = _pairs(positives, verdicts, model, strata)
    ctl = _pairs(controls, verdicts, model, strata)

    # The primary must never receive a THRESHOLDED array: an AUC over 0/1 is exactly
    # 0.5 + (p_adopted - p_control)/2, a monotone restatement of the secondary with the level
    # §5 removes carried back in. BOTH levels must be present to call it thresholded — an
    # all-ones array is degenerate, not thresholded, and refusing it would abort a run over a
    # judge that scored every paper the same, which is a finding rather than a defect.
    levels = {s for _, s in pos} | {s for _, s in ctl}
    if levels == {0.0, 1.0}:
        raise SystemExit(
            f"{model}: the primary was handed a 2-valued score array {sorted(levels)}.\n"
            "  An AUC over a thresholded score is 0.5 + (p_adopted - p_control)/2 — the\n"
            "  secondary wearing the primary's name, with the level §5 removes put back."
        )

    # A control cluster that carries no positive is a phantom the bootstrap can draw, diluting
    # every resample and inflating the cluster count.
    orphan = {k for k, _ in ctl} - {k for k, _ in pos}
    if orphan:
        raise SystemExit(
            f"{model}: {len(orphan)} control cluster(s) carry no positive: {sorted(orphan)[:5]}.\n"
            "  Every bootstrap draw could pick them, diluting the resample and inflating\n"
            "  n_clusters with repositories that contribute nothing to the comparison."
        )

    out = dict(jva.cluster_bootstrap_auc(pos, ctl, iters=iters, seed=jva.SEED))
    out.update(
        model=model,
        iters=iters,
        seed=jva.SEED,
        score_histogram={"adopted": _histogram(pos), "control": _histogram(ctl)},
        tie_fraction=tie_fraction(pos, ctl),
        prereg_s9_reference=prereg_s9_reference(),
    )
    return out


def _by_cluster(
    pos: list[tuple[str, float]], ctl: list[tuple[str, float]]
) -> dict[str, tuple[list[float], list[float]]]:
    out: dict[str, tuple[list[float], list[float]]] = {}
    for key, score in pos:
        out.setdefault(key, ([], []))[0].append(score)
    for key, score in ctl:
        out.setdefault(key, ([], []))[1].append(score)
    return out


def _cluster_bootstrap_gap(
    pos: list[tuple[str, float]],
    ctl: list[tuple[str, float]],
    *,
    iters: int,
    seed: int,
) -> dict[str, Any]:
    """A cluster bootstrap on the GAP, drawing the same way the AUC estimator draws.

    Same unit, same replacement rule, same seed — otherwise the two intervals beside each other
    would answer different design questions while looking like a pair.

    A rate is a plain division, so a draw whose pooled positive or control arm is empty has no
    gap at all; `roc_auc` returns NaN there and is filtered, and the equivalent here must be a
    SKIP rather than a guarded 0.0, which would read as "no actionable controls". With the
    legacy shape — one cluster carrying 46 of 94 usable rows before the cap — draws that pick
    only tiny clusters are not rare.
    """
    clusters = sorted(_by_cluster(pos, ctl))
    if len(clusters) < 2 or not pos or not ctl:
        return {"_refused": "fewer than two clusters — a cluster bootstrap has nothing to resample"}
    by = _by_cluster(pos, ctl)
    rng = random.Random(seed)
    draws: list[float] = []
    skipped = 0
    for _ in range(iters):
        p: list[float] = []
        c: list[float] = []
        for _ in clusters:
            pick = clusters[rng.randrange(len(clusters))]
            p.extend(by[pick][0])
            c.extend(by[pick][1])
        if not p or not c:
            skipped += 1
            continue
        draws.append(sum(p) / len(p) - sum(c) / len(c))
    if len(draws) < 100:
        return {"_refused": "too few usable bootstrap draws", "skipped_draws": skipped}
    draws.sort()
    lo, hi = draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws))]
    return {"ci95": [round(lo, 4), round(hi, 4)], "skipped_draws": skipped, "n_draws": len(draws)}


def secondary_gap(
    model: str,
    positives: list[dict[str, Any]],
    controls: list[dict[str, Any]],
    verdicts: dict[str, Any],
    *,
    iters: int = BOOTSTRAP_ITERS,
) -> dict[str, Any]:
    """§5's secondary: `P(actionable | adopted) − P(actionable | control)` at the shipped bar.

    "Each judge's own threshold" is the shipped `>= 2` cut applied to that judge's own score
    distribution — which is how NR-59 measured base rates of 0.874 and 0.494. Nothing in this
    repository registers a per-judge tuned threshold, and fitting one on this data would be
    exactly the level-fitting §5 forbids. The constant is IMPORTED from `metrics`, which already
    has three other copies scattered through the tree.

    Two intervals, and they are labelled because they answer different questions: the Wilson
    intervals are **paper-level** and do not account for repository clustering, while the
    bootstrap draws repositories. Neither substitutes for the other on a refusal.
    """
    import judge_validity_adoption as jva
    from metrics import RELEVANT_THRESHOLD

    strata = {str(r["case"]): str(r.get("stratum", "pool")) for r in positives}
    pos = _pairs(positives, verdicts, model, strata, threshold=RELEVANT_THRESHOLD)
    ctl = _pairs(controls, verdicts, model, strata, threshold=RELEVANT_THRESHOLD)
    if not pos or not ctl:
        return {
            "model": model,
            "threshold": RELEVANT_THRESHOLD,
            "_refused": f"empty arm: {len(pos)} adopted, {len(ctl)} control verdicts",
        }

    k_pos, k_ctl = int(sum(s for _, s in pos)), int(sum(s for _, s in ctl))
    p_adopted, p_control = k_pos / len(pos), k_ctl / len(ctl)
    boot = _cluster_bootstrap_gap(pos, ctl, iters=iters, seed=jva.SEED)
    return {
        "model": model,
        "threshold": RELEVANT_THRESHOLD,
        "adopted": {
            "n": len(pos),
            "actionable": k_pos,
            "rate": round(p_adopted, 4),
            "wilson95_paper_level": jva.wilson(k_pos, len(pos)),
        },
        "control": {
            "n": len(ctl),
            "actionable": k_ctl,
            "rate": round(p_control, 4),
            "wilson95_paper_level": jva.wilson(k_ctl, len(ctl)),
        },
        "gap": round(p_adopted - p_control, 4),
        "gap_cluster_bootstrap": boot,
        "gap_excludes_zero": jva.excludes(boot.get("ci95"), 0.0),
        "_note": (
            "Youden's J at this judge's shipped operating point. §5: no outcome here switches "
            "the primary label — a judge at a high positive rate sits in the top-right of its "
            "ROC and is bounded low by geometry. The Wilson intervals are PAPER-LEVEL and do "
            "not account for repository clustering; the bootstrap does."
        ),
    }


def control_base_rate(secondary: dict[str, Any]) -> dict[str, Any]:
    """`P(actionable | control)`: the level-sensitive descriptive, with no consequence of its own.

    §5 names it separately from the gap because it is what NR-59's 0.874-against-0.494
    disagreement actually is. It is reported beside both AUCs wherever the base-rate
    disagreement is named, and it decides nothing.
    """
    if "_refused" in secondary:
        return {"model": secondary["model"], "_refused": secondary["_refused"]}
    return {
        "model": secondary["model"],
        "rate": secondary["control"]["rate"],
        "wilson95_paper_level": secondary["control"]["wilson95_paper_level"],
        "_note": (
            "Level-sensitive descriptive. No pre-committed consequence attaches to it; it is "
            "the quantity the two judges disagree about (NR-59: 0.874 against 0.494)."
        ),
    }


RETIRED_RULE = (
    "PREREG-rung1's 0.15 separation bar is RETIRED here, before the data (§5). The gap at a "
    "judge's own threshold is Youden's J at that operating point, so a judge at an 87% positive "
    "rate sits in the top-right of its ROC and is bounded low by geometry: 'the larger gap' "
    "restates the base rates and identifies neither judge as correct."
)


def judge_difference(
    models: tuple[str, str],
    positives: list[dict[str, Any]],
    controls: list[dict[str, Any]],
    verdicts: dict[str, Any],
    *,
    iters: int = BOOTSTRAP_ITERS,
) -> dict[str, Any]:
    """Δ AUC between the two judges, PAIRED at the cluster level and on the intersection only.

    P6 predicts a difference of AUCs, and nothing computed one: the old code differenced the
    *gaps* and read the result against a bar §5 retires. Pairing is required because the two
    judges scored the same papers, and it is done at the cluster level because that is the unit
    both primaries resample — one drawn set of clusters, both AUCs computed on it, then the
    difference.

    **On the intersection only.** Voids are per (model, case, paper), so a paper can carry one
    judge's verdict and not the other's and still be fully explained by the void ledger. A
    "paired" difference over two different samples is precisely the error pairing exists to
    avoid. The per-judge primaries above are each computed on that judge's own coverage, so
    whenever coverage differs this difference is **not** the subtraction of the two numbers
    printed above it — which is stated rather than left to be noticed.
    """
    import judge_validity_adoption as jva
    from metrics import roc_auc

    a, b = models
    strata = {str(r["case"]): str(r.get("stratum", "pool")) for r in positives}

    def paired(rows: list[dict[str, Any]]) -> list[tuple[str, float, float]]:
        out: list[tuple[str, float, float]] = []
        for row in rows:
            case, pid = str(row["case"]), dedup_id(str(row["id"]))
            ra = verdicts.get(verdict_key(a, case, pid))
            rb = verdicts.get(verdict_key(b, case, pid))
            if ra is None or rb is None:
                continue
            out.append(
                (f"{strata.get(case, 'pool')}:{case}", float(ra["score"]), float(rb["score"]))
            )
        return out

    pos, ctl = paired(positives), paired(controls)
    shape = {
        "models": [a, b],
        "n_paired_positives": len(pos),
        "n_paired_controls": len(ctl),
        "n_clusters_paired": len({k for k, _, _ in pos}),
        "_note": (
            "Computed on the rows BOTH judges scored. Each per-judge primary is computed on "
            "that judge's own coverage, so where coverage differs this is not the subtraction "
            "of the two AUCs printed above it."
        ),
        "rung1_0_15_rule": RETIRED_RULE,
    }
    clusters = sorted({k for k, _, _ in pos} | {k for k, _, _ in ctl})
    if len(clusters) < 2 or not pos or not ctl:
        return {**shape, "_refused": "fewer than two clusters carrying both judges' verdicts"}

    by: dict[str, tuple[list[tuple[float, float]], list[tuple[float, float]]]] = {}
    for key, sa, sb in pos:
        by.setdefault(key, ([], []))[0].append((sa, sb))
    for key, sa, sb in ctl:
        by.setdefault(key, ([], []))[1].append((sa, sb))

    point = roc_auc([s for _, s, _ in pos], [s for _, s, _ in ctl]) - roc_auc(
        [s for _, _, s in pos], [s for _, _, s in ctl]
    )
    rng = random.Random(jva.SEED)
    draws: list[float] = []
    for _ in range(iters):
        pa: list[float] = []
        pb: list[float] = []
        ca: list[float] = []
        cb: list[float] = []
        for _ in clusters:
            pick = clusters[rng.randrange(len(clusters))]
            for sa, sb in by[pick][0]:
                pa.append(sa)
                pb.append(sb)
            for sa, sb in by[pick][1]:
                ca.append(sa)
                cb.append(sb)
        if not pa or not ca:
            continue
        delta = roc_auc(pa, ca) - roc_auc(pb, cb)
        if delta == delta:
            draws.append(delta)
    if len(draws) < 100:
        return {**shape, "_refused": "too few usable bootstrap draws"}
    draws.sort()
    lo, hi = draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws))]
    return {
        **shape,
        "delta_auc": round(point, 4),
        "ci95": [round(lo, 4), round(hi, 4)],
        "excludes_zero": jva.excludes((lo, hi), 0.0),
        "n_draws": len(draws),
    }


LOWER_BOUND_ARGUMENT = (
    "§4 makes the measured AUC a LOWER BOUND: a matched control may be a better paper for the "
    "repository than the one it actually adopted, and a genuinely useful paper sitting in the "
    "negative class can only pull the two classes together, never apart. So this is conservative "
    "evidence that the judge tracks something real — and it is equally not an upper limit on the "
    "judge's quality."
)

THREE_WAY_AMBIGUITY = (
    "An interval including 0.5 cannot separate three readings: (1) this judge does not "
    "discriminate; (2) the controls were genuinely good papers this project never got to, which "
    "§4 says compresses the classes together; (3) too few repositories — read against the "
    "realised minimum detectable AUC of {mda} over {clusters} clusters. A null here is "
    "three-ways ambiguous and is not to be reported as a clean negative."
)


def consequences(
    primaries: dict[str, dict[str, Any]], base_rates: dict[str, Any]
) -> dict[str, Any]:
    """§5's pre-committed branches, evaluated per judge from that judge's own primary interval.

    Four outcomes where §5 registers three, and the fourth is the reason the extra one exists:
    a **refusal** is not a null. §3.3 spent $8-12 of extra judging precisely so "the primary
    interval can include 0.5 for a sampling reason, which would fire section 5's pre-committed
    null branch on an artefact" — folding an arithmetic refusal into the null branch would fire
    it on an artefact anyway, by a different route. A judge with no verdicts at all is a fifth,
    distinct again: not asked and not measurable are different facts.

    No headline is hard-coded. Every sentence emitted here is built from values computed in this
    run, because the published NR-56/57 artefact carries prose quoting a *previous* run's figures
    beside its own computed block, and that is the failure this rule exists to prevent.
    """
    out: dict[str, Any] = {}
    for model, primary in primaries.items():
        common = {
            # Unconditional in every branch, §5: "No outcome switches the primary label."
            "primary_label_unchanged": True,
            "primary_label_reason": (
                "Adoption is a lower bound on actionability and cannot calibrate a level; the "
                "gap at a judge's own threshold is Youden's J at that operating point."
            ),
            "rung1_0_15_rule": RETIRED_RULE,
        }
        if primary.get("n_positives", 0) == 0 and primary.get("n_controls", 0) == 0:
            out[model] = {
                **common,
                "outcome": "void",
                "reason": "this judge has no verdicts; it was not asked",
            }
            continue
        if "_refused" in primary or primary.get("excludes_half") is None:
            out[model] = {
                **common,
                "outcome": "no_interval",
                "reason": primary.get("_refused", "no interval was computed"),
                "n_clusters": primary.get("n_clusters"),
                "not_a_null": (
                    "This is an arithmetic refusal, NOT §5's null branch. Reporting it as no "
                    "demonstrated discrimination would fire the null on an artefact, which is "
                    "the outcome §3.3's target of 100 positives exists to avoid."
                ),
            }
            continue
        if primary["excludes_half"]:
            out[model] = {
                **common,
                "outcome": "excludes_0.5",
                "demonstrated_adoption_discrimination": True,
                "lower_bound_argument": LOWER_BOUND_ARGUMENT,
            }
            continue
        out[model] = {
            **common,
            "outcome": "includes_0.5",
            "no_demonstrated_discrimination": True,
            "clean_negative": False,
            "carry_beside_headline": THREE_WAY_AMBIGUITY.format(
                mda=primary.get("min_detectable_auc_80pct"), clusters=primary.get("n_clusters")
            ),
        }

    decided = [v for v in out.values() if v["outcome"] in ("excludes_0.5", "includes_0.5")]
    both = (
        len(decided) == len(out)
        and out
        and all(v["outcome"] == "excludes_0.5" for v in out.values())
    )
    return {
        "per_judge": out,
        "both_exclude_half": bool(both),
        "both_exclude_statement": (
            "Both judges order papers meaningfully and the base-rate disagreement remains "
            f"unresolved: P(actionable|control) is "
            f"{ {m: r.get('rate') for m, r in base_rates.items()} }."
        )
        if both
        else None,
    }


def shortfall(
    analysis: dict[str, Any],
    stop: dict[str, Any] | None,
    primaries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Emitted unconditionally, whatever n turned out to be.

    §3.4: if the list is exhausted below 60 positives "that is the recorded negative result —
    the analysis runs at whatever *n* exists, the shortfall is reported against section 9's
    minimum detectable AUC, and section 5's null branch fires". So this never refuses on n; it
    states what n was and what that costs.
    """
    by_stratum = analysis.get("by_stratum", {})
    return {
        "analysis_set_positives": analysis.get("analysis_set_positives"),
        "n_positives_new": len(by_stratum.get("pool", [])),
        "n_positives_legacy": len(by_stratum.get("legacy", [])),
        "n_clusters": analysis.get("n_clusters"),
        "largest_cluster_share": analysis.get("largest_cluster_share"),
        # The quantity the STOP RULE counted, before the cross-repository contest removed
        # anything — never conflated with the analysis set.
        "stop_rule_capped_positives": (stop or {}).get("stop_rule_capped_positives"),
        "stop_reason": (stop or {}).get("stop_reason"),
        "target": (stop or {}).get("target"),
        "reporting_minimum": REPORTING_MINIMUM,
        "below_reporting_minimum": (
            (analysis.get("analysis_set_positives") or 0) < REPORTING_MINIMUM
        ),
        "realised_min_detectable_auc_80pct": {
            m: p.get("min_detectable_auc_80pct") for m, p in primaries.items()
        },
        "prereg_s9_reference": prereg_s9_reference(),
        "_note": (
            "§3.4: below the reporting minimum the analysis RUNS at whatever n exists and the "
            "shortfall is reported against §9's sizing. It is never a reason to refuse."
        ),
    }


# ---------------------------------------------------------------------------------------
# Transportability (§5) and the pre-declared sensitivity (§2.3)
# ---------------------------------------------------------------------------------------
DP = "2026-09-02"  # the day the candidate list was taken; there is no historical star index


def star_bands(candidates: Path | None = None) -> dict[str, int]:
    """`{full_name: stars at Dp}`, cross-checked against the enumeration's own star grid.

    The star count is on no walk column and no adoption row — it lives only in the committed
    candidate list, which is a **snapshot**. `enumerate_pool.refuse_a_backdated_snapshot`
    establishes there is no historical index, so this is a property at Dp and not at T0, and
    recomputing it from the live API would be a fresh measurement taken after the positives
    were visible.

    The band boundary is `enumerate_pool.STAR_SLICES`, imported rather than retyped, and each
    row's `slice` column carries the lower bound the enumeration actually queried under. They
    are cross-checked and a disagreement raises: a band edge chosen at analysis time is exactly
    the discretion an unchoosable pulse exists to remove.
    """
    import enumerate_pool as ep

    src = candidates or CANDIDATES
    out: dict[str, int] = {}
    with src.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            name = (row.get("full_name") or "").strip()
            if not name:
                continue
            stars = int(row.get("stars") or 0)
            declared = (row.get("slice") or "").split("|")
            if len(declared) >= 2 and declared[1].strip().isdigit():
                lo = int(declared[1])
                hi = next((h for low, h in ep.STAR_SLICES if low == lo), None)
                if stars < lo or (hi is not None and stars > hi):
                    raise SystemExit(
                        f"{name}: {stars} stars is outside the slice it was enumerated in "
                        f"({lo}..{hi}). The candidate list disagrees with itself."
                    )
            out[name] = stars
    return out


def star_band(stars: int | None) -> str:
    """The registered slice a star count falls in, or `unknown` when there is no count.

    `unknown` is a band, not a hole. Three of the eight contributing legacy repositories do not
    resolve to a row of the candidate list, and imputing a count for them would invent the very
    covariate the contrast is cut on; dropping them would quietly shrink the legacy stratum.
    """
    import enumerate_pool as ep

    if stars is None:
        return "unknown"
    for lo, hi in ep.STAR_SLICES:
        if stars >= lo and (hi is None or stars <= hi):
            return f"{lo}-{hi}" if hi is not None else f"{lo}+"
    return "unknown"


def legacy_star_counts(bench: Path | None = None, candidates: Path | None = None) -> dict[str, int]:
    """`{legacy case: stars at Dp}` for the legacy cases that appear in the candidate list."""
    stars = star_bands(candidates)
    lowered = {k.lower(): v for k, v in stars.items()}
    path = bench or (EVALS / "benchmark.yaml")
    if not path.is_file():
        return {}
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    entries = data["cases"] if isinstance(data, dict) else data
    out: dict[str, int] = {}
    for case in entries:
        if not (isinstance(case, dict) and case.get("live_repo") and case.get("name")):
            continue
        slug = case["live_repo"].rstrip("/").split("github.com/")[-1].lower()
        if slug in lowered:
            out[str(case["name"])] = lowered[slug]
    return out


def _subset_auc(
    model: str,
    positives: list[dict[str, Any]],
    controls: list[dict[str, Any]],
    verdicts: dict[str, Any],
    iters: int,
) -> dict[str, Any]:
    """The full estimator over a subgroup, or a refusal. Never a fallback to a narrower one."""
    import judge_validity_adoption as jva

    strata = {str(r["case"]): str(r.get("stratum", "pool")) for r in positives}
    pos = _pairs(positives, verdicts, model, strata)
    ctl = _pairs(controls, verdicts, model, strata)
    out = dict(jva.cluster_bootstrap_auc(pos, ctl, iters=iters, seed=jva.SEED))
    out["model"] = model
    return out


def transportability(
    models: tuple[str, ...],
    analysis: dict[str, Any],
    controls: list[dict[str, Any]],
    verdicts: dict[str, Any],
    *,
    iters: int = BOOTSTRAP_ITERS,
    candidates: Path | None = None,
) -> dict[str, Any]:
    """§5's transportability: legacy against pool, and across star bands.

    **No pre-committed consequence attaches to any of it.** §5 registers these as descriptive
    heterogeneity; §6 item 4 and §9's power table together mean every subgroup here is
    underpowered by construction, and the pooled primary is never re-weighted by any of them.
    """
    by_stratum = analysis.get("by_stratum", {})
    controls_by_case: dict[str, list[dict[str, Any]]] = {}
    for row in controls:
        controls_by_case.setdefault(str(row["case"]), []).append(row)

    def controls_for(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        cases = {str(r["case"]) for r in rows}
        return [c for case in cases for c in controls_by_case.get(case, [])]

    strata: dict[str, Any] = {}
    for name in ("legacy", "pool"):
        rows = by_stratum.get(name, [])
        strata[name] = (
            {m: _subset_auc(m, rows, controls_for(rows), verdicts, iters) for m in models}
            if rows
            else {"_refused": f"the {name} stratum has no positives"}
        )

    # Independent rather than paired: §2.2 excludes the 37 legacy repositories from the pool's
    # population, so the two strata share no repository and there is nothing to pair on.
    delta: dict[str, Any] = {}
    for m in models:
        a = strata["pool"].get(m, {}) if isinstance(strata["pool"], dict) else {}
        b = strata["legacy"].get(m, {}) if isinstance(strata["legacy"], dict) else {}
        if (
            not (isinstance(a, dict) and isinstance(b, dict))
            or a.get("auc") is None
            or (b.get("auc") is None)
        ):
            delta[m] = {"_refused": "one stratum produced no AUC"}
        else:
            delta[m] = {"delta_auc_pool_minus_legacy": round(a["auc"] - b["auc"], 4)}

    stars = star_bands(candidates)
    legacy_stars = legacy_star_counts(candidates=candidates)
    bands: dict[str, list[dict[str, Any]]] = {}
    unresolved: list[str] = []
    for row in analysis.get("positives", []):
        case = str(row["case"])
        if str(row.get("stratum")) == "legacy":
            count = legacy_stars.get(case)
            if count is None:
                unresolved.append(case)
        else:
            if case not in stars:
                # Every walked candidate came from this file; a miss means the analysis is
                # reading a different candidate list than the walk used.
                name = (candidates or CANDIDATES).name
                raise SystemExit(f"{case} is a pool positive but is absent from {name}.")
            count = stars[case]
        bands.setdefault(star_band(count), []).append(row)

    band_out: dict[str, Any] = {}
    for band, rows in sorted(bands.items()):
        entry: dict[str, Any] = {
            "n_positives": len(rows),
            "n_clusters": len({(str(r.get("stratum")), str(r["case"])) for r in rows}),
        }
        subset_ctl = controls_for(rows)
        entry["n_controls"] = len(subset_ctl)
        if entry["n_clusters"] >= 2 and subset_ctl:
            entry["by_judge"] = {
                m: _subset_auc(m, rows, subset_ctl, verdicts, iters) for m in models
            }
        else:
            entry["_refused"] = "fewer than two clusters, or no controls, in this band"
        band_out[band] = entry

    return {
        "strata": strata,
        "delta_by_judge": delta,
        "star_bands": band_out,
        "stars_measured_on": DP,
        "stars_note": (
            "Star counts are a property at Dp, the day the candidate list was taken. There is "
            "no historical index, so they are not counts at T0 — and recomputing them from the "
            "live API would be a fresh measurement taken after the positives were visible."
        ),
        "legacy_cases_without_a_star_count": sorted(set(unresolved)),
        "underpowered": True,
        "pre_committed_consequence": None,
        "_note": (
            "Descriptive heterogeneity. §5 attaches no consequence to any of it, every subgroup "
            "here is underpowered by construction, and the pooled primary is never re-weighted."
        ),
    }


def sensitivity_seeds_ge_10(
    models: tuple[str, ...],
    analysis: dict[str, Any],
    controls: list[dict[str, Any]],
    verdicts: dict[str, Any],
    *,
    iters: int = BOOTSTRAP_ITERS,
) -> dict[str, Any]:
    """§2.3's pre-declared sensitivity: the same endpoints over `ids_v2(T0) >= 10` only.

    "Retained as a pre-declared sensitivity. It is a strict subset of the judged set, so it
    costs nothing to report." `seeds_at_t0` is on every row of both strata, so nothing is
    re-mined and nothing is re-judged.
    """
    keep = [r for r in analysis.get("positives", []) if int(r.get("seeds_at_t0") or 0) >= 10]
    cases = {str(r["case"]) for r in keep}
    subset_ctl = [c for c in controls if str(c["case"]) in cases]
    out: dict[str, Any] = {
        "threshold": 10,
        "n_positives": len(keep),
        "n_clusters": len({(str(r.get("stratum")), str(r["case"])) for r in keep}),
        "n_controls": len(subset_ctl),
        "pre_committed_consequence": None,
    }
    if out["n_clusters"] >= 2 and subset_ctl:
        out["by_judge"] = {m: _subset_auc(m, keep, subset_ctl, verdicts, iters) for m in models}
    else:
        out["_refused"] = "fewer than two clusters, or no controls, above the threshold"
    return out


# ---------------------------------------------------------------------------------------
# Contamination sensitivity (§5), which is VOID for this pool (§5's 2026-09-03 decision)
# ---------------------------------------------------------------------------------------
CONTAMINATION_ENDPOINT = (
    "Contamination sensitivity: positives split by adoption commit date relative to each "
    "judge's published training cutoff; AUC on the post-cutoff subset."
)


def training_cutoffs(prereg: Path | None = None) -> dict[str, str | None]:
    """Each judge's published training cutoff, parsed from §5, or None where it is still blank.

    Parsed rather than hard-coded, because §7 requires them "recorded at registration, not
    after the positives are visible" — so the registered file is the only place they may come
    from, and reading them from code would let them be filled in afterwards without a visible
    commit to the pre-registration.
    """
    src = prereg or (EVALS / "PREREG-judge-validity-pool.md")
    out: dict[str, str | None] = {"gpt-5.5": None, "claude-sonnet-5": None}
    if not src.is_file():
        return out
    for line in src.read_text(encoding="utf-8").splitlines():
        if "Contamination sensitivity" not in line:
            continue
        for model, label in (("gpt-5.5", "GPT-5.5"), ("claude-sonnet-5", "Sonnet 5")):
            m = re.search(rf"{re.escape(label)}\s*`([^`]*)`", line)
            if m and m.group(1).strip("_ ") and re.match(r"\d{4}-\d{2}", m.group(1).strip()):
                out[model] = m.group(1).strip()
        break
    return out


def contamination_split(
    models: tuple[str, ...],
    analysis: dict[str, Any],
    controls: list[dict[str, Any]],
    verdicts: dict[str, Any],
    *,
    iters: int = BOOTSTRAP_ITERS,
    cutoffs: dict[str, str | None] | None = None,
) -> dict[str, Any]:
    """§5's contamination split — reported as an explicit VOID while the cutoffs are unfilled.

    Two independent reasons, both recorded in the registered file, and neither improves with
    time.

    **No published cutoff exists for either judge**, and none will be supplied: the maintainer's
    decision of 2026-09-03. Guessing was refused once already on the record — `judge_date_
    stratify.py` states its design "does not need to know the cutoff, and deliberately does not
    guess it", because testing for a discontinuity anywhere beats testing at an assumed date.

    **And the dates are 44 of 94 on the legacy stratum, missing where repositories are largest.**
    Even with two sourced cutoffs the legacy half of the split would run on a 47 % subset that
    is not missing at random.

    So this emits `status: not_computed` with the reason and the command that would produce it —
    values null, never a date, never 0, never an empty dict, and **the key is never absent**.
    `mcp_arm_report` already established the shape: "Void, not zero. An unrun arm reported as 0
    reads as a measurement, and this is its absence." §6 item 6 names this split as the only
    instrument against recognition bias, so its absence is a real weakening of what the pool can
    claim, and it is carried beside the primary rather than buried.
    """
    resolved = cutoffs if cutoffs is not None else training_cutoffs()
    dated = [r for r in analysis.get("positives", []) if r.get("adoption_date")]
    # Keyed on the DATE, never on the commit: `adoption_commit` has a branch returning
    # (sha, None) when the timestamp does not parse, so keying on the sha would file such a row
    # under a dated bucket holding a None date.
    undated_legacy = [
        r
        for r in analysis.get("positives", [])
        if not r.get("adoption_date") and str(r.get("stratum")) == "legacy"
    ]
    undated_pool = [
        r
        for r in analysis.get("positives", [])
        if not r.get("adoption_date") and str(r.get("stratum")) != "legacy"
    ]
    coverage = {
        "n_dated": len(dated),
        # Kept apart so ~30 structurally undated legacy rows do not read as 30 failed searches.
        "n_undated_legacy": len(undated_legacy),
        "n_undated_pool_miss": len(undated_pool),
    }

    if not all(resolved.get(m) for m in models):
        return {
            "status": "not_computed",
            "endpoint": CONTAMINATION_ENDPOINT,
            "cutoffs": {m: resolved.get(m) for m in models},
            "why": (
                "No published training cutoff is recorded for either judge and none will be "
                "supplied (maintainer decision, 2026-09-03, before any positive existed). §7 "
                "requires them recorded at registration rather than after the positives are "
                "visible, and this project has already refused once on the record to guess one "
                "(judge_date_stratify.py: the design 'does not need to know the cutoff, and "
                "deliberately does not guess it'). Independently, adoption dates are recovered "
                "for 44 of 94 legacy positives and are missing where repositories are largest, "
                "so even two sourced cutoffs would split a 47% subset that is not missing at "
                "random."
            ),
            "consequence": (
                "§6 item 6 names this split as the ONLY instrument available against "
                "recognition bias, so recognition remains an unmitigated confound in this pool. "
                "That is carried beside the primary, not buried."
            ),
            "how": (
                "Fill both cutoffs in §5 of PREREG-judge-validity-pool.md with a dated, sourced "
                "value and re-run the analysis; adoption_date is already recorded on every "
                "capped positive, so nothing is re-mined."
            ),
            "adoption_date_coverage": coverage,
        }

    # Reachable only once §5 carries two sourced cutoffs. Each judge gets its OWN split.
    by_positive: dict[str, list[dict[str, Any]]] = {}
    for row in controls:
        anchor = row.get("for_positive")
        if not anchor:
            raise SystemExit(
                "a control row carries no `for_positive`, so the post-cutoff subset cannot be "
                "joined to its positives. The pool-scheme drawer does not write it; §4's "
                "arm-neutral scheme does."
            )
        by_positive.setdefault(str(anchor), []).append(row)

    out: dict[str, Any] = {
        "status": "computed",
        "cutoffs": {m: resolved[m] for m in models},
        "adoption_date_coverage": coverage,
        "by_judge": {},
        "_note": (
            "Each judge is split at ITS OWN cutoff, so these are two different subsets. The two "
            "subset AUCs must never be differenced or compared with each other."
        ),
    }
    for model in models:
        cut = str(resolved[model])
        post = [r for r in dated if str(r["adoption_date"]) > cut]
        subset_ctl = [c for r in post for c in by_positive.get(dedup_id(str(r["id"])), [])]
        missing = [r for r in post if not by_positive.get(dedup_id(str(r["id"])))]
        if missing:
            raise SystemExit(
                f"{model}: {len(missing)} post-cutoff positive(s) have no matched controls."
            )
        entry = {
            "cutoff": cut,
            "n_post_cutoff": len(post),
            "n_pre_or_on_cutoff": len(dated) - len(post),
            "controls_per_positive": round(len(subset_ctl) / len(post), 2) if post else None,
            "pool_only": not any(str(r.get("stratum")) == "legacy" for r in post),
        }
        entry.update(_subset_auc(model, post, subset_ctl, verdicts, iters) if post else {})
        out["by_judge"][model] = entry
    return out


# ---------------------------------------------------------------------------------------
# Scoring the registered predictions (§8) and publishing the datasheet (§10 step 7)
# ---------------------------------------------------------------------------------------
def positives_within_b(curve: Path | None = None, b: int | None = None) -> dict[str, Any]:
    """Capped positives at the last curve point on or before *b* rows.

    Read from the yield curve rather than re-summed from the ledger, because the ledger's
    `capped` column is a per-row snapshot at mining time and the contest can un-count a paper
    later when a different repository wins it — summing it to rank 1,200 over-counts. The curve
    records `counted_positives(adoptions)` as it stood when the walk passed that row, which is
    what "within B rows" is asking about.

    Returns `capped_positives: None` when no point exists at or before *b*, so a caller cannot
    read a missing measurement as a zero.
    """
    import walk_pool

    src = curve or POOL_CURVE
    limit = walk_pool.DEFAULT_B if b is None else b
    if not src.is_file():
        return {"at": None, "capped_positives": None, "why": f"{src.name} does not exist"}
    best: dict[str, Any] = {"at": None, "capped_positives": None}
    with src.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                at = int(row.get("at") or 0)
            except ValueError:
                continue
            if at <= limit and (best["at"] is None or at > int(best["at"])):
                best = {"at": at, "capped_positives": int(row.get("capped_positives") or 0)}
    if best["at"] is None:
        best["why"] = f"no curve point at or before {limit} rows"
    return best


# §8's P6 brackets, per judge, exactly as registered. Held here rather than parsed out of the
# prediction sentence, so the numbers scored are the numbers written down.
P6_BRACKETS = {"gpt-5.5": (0.60, 0.70), "claude-sonnet-5": (0.62, 0.72)}


def score_predictions(
    summary: dict[str, Any],
    analysis: dict[str, Any],
    primaries: dict[str, dict[str, Any]],
    secondaries: dict[str, dict[str, Any]],
    difference: dict[str, Any],
    *,
    pool_adoptions: Path | None = None,
    curve: Path | None = None,
) -> dict[str, Any]:
    """§8's predictions, each scored from its own registered input and no other.

    Two rules do most of the work here.

    **A prediction is scored from the quantity it names.** P3 and P4 are defined over the
    unconditional B₀ prefix, so they read `q_over_b0` and `y_over_b0` and never recompute over
    the whole walk — a prefix whose length was chosen by the yield it produced is the inverse
    sampling §3.2 exists to avoid. P5 is about the STOP RULE reaching 100 within B, so it reads
    the stop count, not the analysis set.

    **A prediction registered at an n is not scored below it.** P6 brackets both AUCs "at ≥ 130
    capped positives"; scoring it at 70 would turn a power shortfall into a failed prediction.
    Below that it records `not_evaluable_at_this_n`, and the AUC itself still runs at whatever
    n exists — §3.4 requires exactly that.
    """
    import walk_pool  # P5 scores against the REGISTERED B, never the summary's budget

    out: dict[str, Any] = {}
    n_analysis = int(analysis.get("analysis_set_positives") or 0)

    # P1 and P2 were scored on 2026-09-02 against the legacy re-mine and are recorded in §8.
    out["P1"] = {"status": "scored_before_the_walk", "where": "§8, 2026-09-02 [NR-60]"}
    out["P2"] = {"status": "scored_before_the_walk", "where": "§8, 2026-09-02 [NR-60]"}

    q, y = summary.get("q_over_b0"), summary.get("y_over_b0")
    out["P3"] = (
        {
            "prediction": "q in [0.08, 0.30], point 0.15",
            "observed": q,
            "in_bracket": bool(0.08 <= q <= 0.30),
            "n_prefix_decided": summary.get("n_prefix_decided"),
        }
        if q is not None
        else {"status": "not_evaluable", "why": "the unconditional prefix produced no rate"}
    )
    out["P4"] = (
        {
            "prediction": "y in [0.8, 2.5], point 1.5",
            "observed": y,
            "in_bracket": bool(0.8 <= y <= 2.5),
        }
        if y is not None
        else {"status": "not_evaluable", "why": "no qualifying repository in the prefix"}
    )

    stop_count = int(summary.get("capped_positives") or 0)
    walked = int(summary.get("walked") or 0)
    at_b = positives_within_b(curve)
    reached = at_b.get("capped_positives")
    out["P5"] = {
        "prediction": "100 new positives within B = 1,200 rows",
        # The REGISTERED B, never `summary["budget"]`. This scored `walked <= budget` against
        # the budget the run happened to use, and §3.4 lets that budget GROW: raised to 6,000
        # for the extension, it made `3689 <= 6000` true and the prediction "met" on 157
        # positives that took 3,689 rows to reach. `walk_stop_reason` was hardened against
        # exactly this — "reading the thresholds out of it let the gate take its bar from the
        # very artefact it is gating" — and this scorer was left reading the same unguarded
        # field, so an operational decision about budget silently moved a registered bar.
        "registered_b": walk_pool.DEFAULT_B,
        "capped_positives_within_b": reached,
        "curve_point_used": at_b.get("at"),
        "stop_rule_capped_positives": stop_count,
        "walked": walked,
        "met": None if reached is None else bool(reached >= 100),
        "_note": (
            "Both clauses, scored against the registered B. The final count is reported beside "
            "it because reaching it is a real fact about the walk — it is simply not what P5 "
            "predicted."
        ),
    }

    if n_analysis >= 130:
        # Every clause scored, and `met` rendered. This branch used to compute the parts and
        # stop: the brackets lived in the prediction SENTENCE and were never compared to the
        # AUCs, and `difference_excludes_zero: True` — which is the prediction failing — was
        # emitted with nothing saying so. The `else` branch below carefully explains why P6 is
        # not evaluable at low n while the branch that IS evaluable returned no verdict.
        clauses: dict[str, Any] = {}
        for model, p in primaries.items():
            auc, bracket = p.get("auc"), P6_BRACKETS.get(model)
            clauses[f"auc_{model}_in_bracket"] = (
                None if auc is None or bracket is None else bool(bracket[0] <= auc <= bracket[1])
            )
        clauses["both_exclude_half"] = bool(all(p.get("excludes_half") for p in primaries.values()))
        excludes = difference.get("excludes_zero")
        clauses["difference_does_not_exclude_zero"] = (
            None if excludes is None else bool(not excludes)
        )
        out["P6"] = {
            "prediction": "AUC(gpt) 0.60-0.70, AUC(sonnet) 0.62-0.72; both exclude 0.5; "
            "their difference does not exclude 0",
            "auc": {m: p.get("auc") for m, p in primaries.items()},
            "brackets": {m: list(b) for m, b in P6_BRACKETS.items()},
            "both_exclude_half": clauses["both_exclude_half"],
            "difference_excludes_zero": excludes,
            "clauses": clauses,
            # A conjunction, and unknown is not pass: an unscorable clause makes the whole
            # prediction unscored rather than quietly dropping out of the `all()`.
            "met": None if any(v is None for v in clauses.values()) else all(clauses.values()),
        }
    else:
        out["P6"] = {
            "status": "not_evaluable_at_this_n",
            "why": (
                f"P6 registers its brackets at >= 130 capped positives; the analysis set holds "
                f"{n_analysis}. Scoring it here would convert a power shortfall into a failed "
                "prediction. The AUC itself still runs at whatever n exists (§3.4)."
            ),
        }

    rates = {m: s.get("control", {}).get("rate") for m, s in secondaries.items()}
    out["P7"] = (
        {
            "prediction": "P(actionable|control): gpt >= 0.70, sonnet <= 0.55",
            "observed": rates,
            "met": bool(
                (rates.get("gpt-5.5") or 0) >= 0.70 and (rates.get("claude-sonnet-5") or 1) <= 0.55
            ),
        }
        if all(v is not None for v in rates.values())
        else {"status": "not_evaluable", "why": "a judge has no control verdicts"}
    )

    deffs = {m: p.get("design_effect") for m, p in primaries.items()}
    out["P8"] = (
        {
            "prediction": "the realised design effect is >= 1.5",
            "observed": deffs,
            # Both, because one judge clearing it does not establish that the paper-level
            # interval would have been materially too narrow. Both values are reported either way.
            "met": all((v or 0) >= 1.5 for v in deffs.values()),
        }
        if all(v is not None for v in deffs.values())
        else {"status": "not_evaluable", "why": "a design effect was not computed"}
    )

    src = pool_adoptions or POOL_ADOPTIONS
    if src.is_file():
        rows = json.loads(src.read_text(encoding="utf-8"))
        fired = sum(1 for r in rows if r.get("reverse_cited") or r.get("genesis"))
        out["P9"] = {
            "prediction": "the reverse-citation and doc-genesis filters remove >= 5% of gross "
            "adoptions on the enumerated population",
            "n_gross_adoptions": len(rows),
            "n_removed": fired,
            "share": round(fired / len(rows), 4) if rows else None,
            "met": bool(rows and fired / len(rows) >= 0.05),
            "_note": (
                "`genesis` is hard-coded False by the walk because PP2 >= 3 subsumes the "
                "doc-genesis guard, so its half contributes structurally zero and this share is "
                "the reverse-citation filter alone."
            ),
        }
    else:
        out["P9"] = {"status": "not_evaluable", "why": "no pool adoptions artefact"}
    return out


DATASHEET_COMPONENTS = (
    "candidate_list",
    "seed_and_pulse",
    "walk_ledger",
    "positives_and_controls",
    "raw_ordinal_scores",
    "doi_pmid_covariates",
    "void_and_timeout_lists",
)


def _digest(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def datasheet(
    seed: str,
    analysis: dict[str, Any],
    controls: list[dict[str, Any]],
    verdicts: dict[str, Any],
    outcomes: list[dict[str, Any]],
    *,
    models: tuple[str, ...],
) -> dict[str, Any]:
    """§10 step 7's datasheet: the seven components it names, each present by name.

    "the datasheet — candidate list, seed and pulse timestamp, walk ledger, positives and
    controls, both judges' raw ordinal scores, DOI/PMID covariates, void and timeout lists —
    published with it." Each is emitted under its own key whether or not it has content, so a
    missing component is visible as an empty one rather than as an absent key nobody notices.

    The RAW ordinal scores matter and were never stored: the legacy artefact keeps thresholded
    counts only, so the four-level distribution the primary is actually computed over could not
    be recovered from it.
    """
    import walk_pool

    by_case_covariates: dict[str, Any] = {}
    for row in _read_walk_rows():
        if row.get("qualifies") == "True":
            by_case_covariates[row["full_name"]] = {
                k: int(row.get(k) or 0) for k in ("dois_head", "dois_t0", "pmids_head", "pmids_t0")
            }
    legacy_cov = {}
    if LEGACY_SIDECAR.is_file():
        legacy_cov = {
            case: {k: c.get(k) for k in ("dois_head", "dois_t0", "pmids_head", "pmids_t0")}
            for case, c in legacy_sidecar()["cases"].items()
        }

    scores: dict[str, list[dict[str, Any]]] = {}
    for model in models:
        rows = []
        for record in verdicts.values():
            if record.get("model") == model:
                rows.append(
                    {
                        "case": record.get("case"),
                        "id": record.get("id"),
                        "arm": record.get("arm"),
                        "score": record.get("score"),
                    }
                )
        scores[model] = sorted(rows, key=lambda r: (str(r["case"]), str(r["id"])))

    return {
        "candidate_list": {
            "path": str(CANDIDATES.relative_to(EVALS.parent)) if CANDIDATES.is_file() else None,
            "sha256": _digest(CANDIDATES),
            "dp": DP,
        },
        "seed_and_pulse": {
            "pulse": walk_pool.REGISTERED_PULSE,
            "seed_sha256": hashlib.sha256(seed.encode()).hexdigest(),
            # The seed value itself is committed at evals/frame/pool/SEED_POOL; the digest here
            # ties this artefact to that file without duplicating it.
            "seed_file": str(SEED_FILE.relative_to(EVALS.parent)),
        },
        "walk_ledger": {
            "path": str(POOL_WALK.relative_to(EVALS.parent)) if POOL_WALK.is_file() else None,
            "sha256": _digest(POOL_WALK),
            "n_rows": len(_read_walk_rows()),
        },
        "positives_and_controls": {
            "n_positives": analysis.get("analysis_set_positives"),
            "by_stratum": {k: len(v) for k, v in analysis.get("by_stratum", {}).items()},
            "n_controls": len(controls),
            "controls_artefact": str(CONTROLS_ROWS.relative_to(EVALS.parent)),
            "controls_sha256": _digest(CONTROLS_ROWS),
        },
        # Raw, not thresholded: the primary is computed over the four rubric levels, and the
        # legacy artefact stores only counts above the bar.
        "raw_ordinal_scores": scores,
        "doi_pmid_covariates": {
            "pool": by_case_covariates,
            "legacy": legacy_cov,
            "_note": (
                "§6.2 SIZES the life-science blind spot with these rather than closing it: 0 of "
                "6 bio-* and 0 of 6 mat-* legacy cases clear ids_v2(HEAD) >= 10. They are "
                "covariates and gate nothing."
            ),
        },
        # Two lists, not one sum. §3.1: "A timeout is a recorded outcome, never a silent skip."
        "void_and_timeout_lists": {
            "judging": [o for o in outcomes if o.get("outcome") != "judged"],
            "walk_timeouts": [
                {"rank": r.get("rank"), "full_name": r.get("full_name"), "note": r.get("note")}
                for r in _read_walk_rows()
                if r.get("outcome") in ("timeout", "clone_timeout")
            ],
            "walk_failures": [
                {"rank": r.get("rank"), "full_name": r.get("full_name"), "note": r.get("note")}
                for r in _read_walk_rows()
                if r.get("outcome") in ("clone_failed", "error", "no_head")
            ],
        },
        "limitations": [
            "head_ids reaches the arXiv/HF documentation channel only, so §4's 'not cited "
            "anywhere at HEAD' is honoured for that channel and not for DOIs, PMIDs, source "
            "comments or notebooks.",
            "Control listings are drawn from six monthly slices of the half-year rather than "
            "the whole window; the per-slice cap is an unregistered narrowing, recorded in §4.",
            "Controls may be shared across clusters, a correlation the repository-cluster "
            "bootstrap does not capture, so the interval is very slightly too narrow.",
            "X5's language detector is absent, so the population is not filtered for English "
            "prose (§2.2).",
            "Adoption dates cover 44 of 94 legacy positives and are missing where repositories "
            "are largest (§5).",
        ],
    }


def _read_walk_rows(path: Path | None = None) -> list[dict[str, str]]:
    """The walk ledger, typed the way the walk itself types it.

    `csv.DictReader` returns the STRING "False", which is truthy — `walk_pool` is careful about
    this in its own tallies and an analysis that forgot would report every walked candidate as
    a qualifier.
    """
    src = path or POOL_WALK
    if not src.is_file():
        return []
    with src.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------------------------------
# The legacy materialisation pass (§4, §5, §7)
# ---------------------------------------------------------------------------------------
LEGACY_SIDECAR = POOL_FRAME / "legacy_sidecar.json"
LEGACY_CLONES = WORK / "fullclone"


def _git(repo: Path, *args: str, timeout: float = 300.0) -> str:
    """Run git and RAISE on failure, which is the whole point of doing it here.

    `mine_adoptions` runs git through `subprocess.run` with no `check`, so a failure comes back
    as an empty string and reads downstream as "this repository has nothing". These clones are
    `--filter=blob:none` promisors: every blob read is a lazy fetch that can fail on a network
    blip, and a silent empty result would be written into a persisted artefact and judged.
    """
    out = subprocess.run(  # noqa: S603
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    if out.returncode != 0:
        raise SystemExit(
            f"git {' '.join(args[:2])} failed in {repo.name}: {(out.stderr or '').strip()[:200]}"
        )
    return out.stdout


def materialise_legacy(
    *,
    source: Path | None = None,
    clones: Path | None = None,
    contexts: Path | None = None,
    head_ids: Path | None = None,
    out: Path | None = None,
    timeout: float = 300.0,
    pickaxe_timeout: float = 20.0,
    dater: Any = None,
) -> dict[str, Any]:
    """Give the legacy stratum everything the pool stratum gets from the walk.

    §4 says "The legacy 35 are re-run under this control scheme", and that is only a re-run if
    the legacy arm is shown the same kind of prompt through the same code. It never has been:
    `adoptions-v2.json` carries no T0 context, no HEAD citation set, no adoption date and no
    realised T0 commit date, so without this pass §5's legacy-versus-pool transportability
    contrast would compare **two code paths** rather than two populations, and the primary would
    be pool-only — roughly 100 positives where §9's power table budgets 130, which is the
    difference between a minimum detectable AUC of 0.578 and 0.62.

    **Run once, while the clones are alive.** `.work/fullclone/*` are `--filter=blob:none`
    promisor clones, so every document read is a lazy fetch from the origin. This cannot be
    done at analysis time: it is mining, and mining after the verdicts are visible is the thing
    the whole design is arranged to prevent.

    Everything is pinned to the SHAs already recorded on each row (§7), so `t0` reproduces
    exactly and a stored T0 verdict stays a verdict about the same prompt. Nothing is written
    back into `adoptions-v2.json`, which §1 declares immutable; the derived fields land in a
    sidecar.

    Adoption dates are computed for **every** usable legacy positive rather than only the
    capped ones. The cap is a function of `SEED_POOL`, which does not exist until the pulse —
    computing the superset now removes the dependency and lets this run before it.
    """
    import mine_adoptions as ma
    import walk_pool

    src = source or LEGACY_ADOPTIONS
    clone_root = clones or LEGACY_CLONES
    ctx_dir = contexts or POOL_CONTEXTS
    ids_dir = head_ids or POOL_HEAD_IDS
    find_adoption = dater or walk_pool.adoption_commit
    rows = [r for r in json.loads(src.read_text(encoding="utf-8")) if r.get("usable")]

    by_case: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_case.setdefault(str(row["case"]), []).append(row)

    ctx_dir.mkdir(parents=True, exist_ok=True)
    ids_dir.mkdir(parents=True, exist_ok=True)
    cases: dict[str, Any] = {}

    for case, entries in sorted(by_case.items()):
        repo = clone_root / case
        if not (repo / ".git").is_dir() and not (repo / "HEAD").is_file():
            raise SystemExit(
                f"{case}: no clone at {repo}. This pass mines from the live clones and cannot\n"
                "  be deferred — they are promisor clones and the pass needs their origins."
            )
        head, t0 = str(entries[0]["head"]), str(entries[0]["t0"])
        # Every row of a case must agree about the pin, or the context, the citation set and the
        # adoption dates would describe different revisions of the same repository.
        for row in entries:
            if str(row["head"]) != head or str(row["t0"]) != t0:
                raise SystemExit(f"{case}: rows disagree about head/t0; the pin is not one pin")
        for rev in (head, t0):
            _git(repo, "rev-parse", "--verify", f"{rev}^{{commit}}", timeout=timeout)

        # §3.1: "the nominal cutoff is not the realised one" — `rev-list --before` can land
        # years earlier across a history gap, and §4 matches controls on the realised date.
        # Legacy rows carry only `t0_date`, the nominal one.
        t0_commit_date = _git(repo, "log", "-1", "--format=%cI", t0, timeout=timeout).strip()[:10]
        head_date = _git(repo, "log", "-1", "--format=%cI", head, timeout=timeout).strip()[:10]

        context = ma.t0_context(repo, case, t0)
        assert_context_is_judgeable(case, context)
        digest = walk_pool.context_hash("t0", context)
        (ctx_dir / f"{case_key(case)}.{digest}.txt").write_text(context, encoding="utf-8")

        cited = sorted(ma.ids_with_paths(repo, head, "v2", timeout))
        if not cited:
            raise SystemExit(
                f"{case}: zero identifiers at HEAD, yet it contributed usable adoptions.\n"
                "  A lazy fetch returned nothing rather than failing; §4's never-cited rule\n"
                "  would become a no-op and put positives into the negative class."
            )
        (ids_dir / f"{case_key(case)}.json").write_text(
            json.dumps(cited, indent=0), encoding="utf-8"
        )

        adoptions: dict[str, dict[str, Any]] = {}
        for row in entries:
            pid = str(row["id"])
            try:
                commit, when = find_adoption(repo, pid, t0, head, pickaxe_timeout)
                note = None if commit else "no commit introduced this identifier between t0/head"
            except subprocess.TimeoutExpired:
                # `git log -S` diffs every commit's documents across the whole window, and every
                # blob is a lazy fetch on a promisor clone. Measured on `diffusion`
                # (huggingface/diffusers): ONE identifier exceeds 300 s, so its 46 would run for
                # hours. Bounded and recorded rather than left running — the only consumer is
                # §5's contamination split, which is VOID for this pool because no published
                # training cutoff exists for either judge. An unavailable date therefore costs
                # nothing today, and the partial set is still here if two sourced cutoffs ever
                # arrive. Coverage is reported per case rather than left to be counted later.
                commit, when = None, None
                note = f"pickaxe exceeded {pickaxe_timeout:.0f}s on a promisor clone"
            # A miss is recorded as null with its reason, never as a blank that reads as a date.
            adoptions[pid] = {"adoption_commit": commit, "adoption_date": when, "note": note}

        cases[case] = {
            "head": head,
            "t0": t0,
            "t0_date": str(entries[0].get("t0_date") or ""),
            "t0_commit_date": t0_commit_date,
            "head_date": head_date,
            "window_days": (
                (datetime.fromisoformat(head_date) - datetime.fromisoformat(t0_commit_date)).days
            ),
            "context_digest": digest,
            "n_head_ids": len(cited),
            "adoptions": adoptions,
            "n_adoption_dates": sum(1 for a in adoptions.values() if a["adoption_date"]),
            # §6.2 sizes the life-science blind spot with these, so they are covariates rather
            # than decoration and the legacy arm must carry them too.
            "dois_head": _count_at(repo, head, walk_pool.DOI_GREP, walk_pool.DOI, timeout),
            "dois_t0": _count_at(repo, t0, walk_pool.DOI_GREP, walk_pool.DOI, timeout),
            "pmids_head": _count_at(repo, head, walk_pool.PMID_GREP, walk_pool.PMID, timeout),
            "pmids_t0": _count_at(repo, t0, walk_pool.PMID_GREP, walk_pool.PMID, timeout),
        }
        print(
            f"  {case:16} t0 {t0_commit_date} head {head_date}  "
            f"{len(cited):4} ids at HEAD  {len(entries):3} usable  "
            f"{sum(1 for a in adoptions.values() if a['adoption_date']):3} dated  ctx {digest}",
            flush=True,
        )

    payload = {
        "_comment": (
            "Derived from the legacy clones at each row's recorded head/t0 SHAs (PREREG §7). "
            "adoptions-v2.json is immutable (§1) and is not modified; these are the fields the "
            "walk records for a pool repository and the legacy artefact predates."
        ),
        "source": str(src),
        "n_cases": len(cases),
        "n_usable_rows": len(rows),
        "cases": cases,
    }
    write_artifact(out or LEGACY_SIDECAR, payload)
    return payload


def _count_at(repo: Path, rev: str, grep: str, pattern: Any, timeout: float) -> int:
    import mine_adoptions as ma

    return len(ma._matches_with_paths(repo, rev, grep, pattern, timeout))


def legacy_sidecar(path: Path | None = None) -> dict[str, Any]:
    """The materialisation pass's output, or a refusal naming the command that produces it."""
    src = path or LEGACY_SIDECAR
    if not src.is_file():
        raise SystemExit(
            f"{src} does not exist — the legacy stratum has not been materialised.\n"
            "  Without it the legacy arm has no T0 context, no HEAD citation set and no\n"
            "  adoption dates, so §5's transportability contrast would compare two code paths\n"
            "  rather than two populations and the primary would be pool-only.\n"
            "  Run: uv run python evals/judge_validity_pool.py --materialise-legacy"
        )
    return json.loads(src.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--materialise-legacy",
        action="store_true",
        help="mine T0 contexts, HEAD citation sets, adoption dates and covariates from the "
        "legacy clones at their recorded SHAs. Run once, while the clones are alive.",
    )
    args = ap.parse_args()
    if args.materialise_legacy:
        out = materialise_legacy()
        print(f"\nmaterialised {out['n_cases']} legacy cases, {out['n_usable_rows']} usable rows")
        print(f"wrote {LEGACY_SIDECAR}")
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
