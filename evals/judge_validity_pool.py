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

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

EVALS = Path(__file__).resolve().parent
sys.path.insert(0, str(EVALS))

WORK = EVALS / ".work"

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
