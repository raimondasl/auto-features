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

import csv
import hashlib
import json
import os
import re
import sys
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
    target = walk_pool.DEFAULT_TARGET
    if int(summary.get("target") or 0) != target:
        raise SystemExit(
            f"the walk ran with target={summary.get('target')}, and §3.3 registers {target}.\n"
            "  The stop rule is frozen; a walk stopped on a different one is a different study."
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
