"""The gold set is derived from an ungitignored cache. This pins it.

Every published recall denominator in this project -- hop 21/56, HyDE 34/56, union 43/56 --
counts against the gold set that `diagnose_pool.actionable_baseline_ids` *derives* at call
time from `evals/cache/baseline/cli/`. Two things make that a hazard rather than a detail:

* `evals/cache/` is gitignored, so those 25 baseline answers exist only on the machine that
  produced them. Nothing in version control held the gold set before `evals/gold_targets.json`.
* `run_baseline` re-parses the cached `raw` on every hit, so the ids are re-derived rather
  than read. Nine of the 56 cannot be re-derived: three cases carry a restoration note where
  their transcript used to be, and `rag` stores two ids its own `raw` never contained.

The 2026-08-09 incident is the precedent: a baseline re-run at 30 turns changed the cache
discriminator, invalidated every case rather than the two that needed it, and moved `graph`
from 3 targets to 4 -- which would have shifted every published denominator for a reason
unrelated to any research question. It was caught by `tests/test_eval_hop_pool.py` pinning a
frozen literal. This does the same job for the derived set as a whole.

A failure here is not necessarily a bug. It means a denominator moved, and the question is
whether that was intended. If it was, re-freeze deliberately:

    uv run python evals/freeze_gold_targets.py

**The set is now two cohorts, and only one of them is published.** On 2026-08-25 the
`bio-*`/`mat-*` cases got the `cli` comparator the other 25 had always been measured
against, and brought 17 gold targets of their own. Those are real targets and they belong
in the artifact -- but no published figure divides by them, so the artifact carries a
per-cohort split and `benchmark25` is pinned at **56** below. Anyone computing recall over
all 73 and comparing it to 43/56 would be committing C-17 with our own data.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EVALS = ROOT / "evals"
sys.path.insert(0, str(EVALS))

FROZEN = EVALS / "gold_targets.json"


@pytest.fixture(scope="module")
def frozen() -> dict:
    if not FROZEN.is_file():
        pytest.skip(f"no frozen gold set at {FROZEN}")
    return json.loads(FROZEN.read_text(encoding="utf-8"))


class TestTheArtifactIsSelfConsistent:
    """These run anywhere -- they read only the committed file."""

    def test_counts_match_the_payload(self, frozen):
        targets = frozen["targets"]
        assert frozen["n_cases"] == len(targets)
        assert frozen["n_targets"] == sum(len(v) for v in targets.values())

    def test_every_case_has_at_least_one_target(self, frozen):
        empty = [c for c, v in frozen["targets"].items() if not v]
        assert empty == [], f"cases frozen with no targets: {empty}"

    def test_no_duplicate_ids_within_a_case(self, frozen):
        for case, ids in frozen["targets"].items():
            assert len(ids) == len(set(ids)), f"{case} has duplicate target ids"

    def test_the_published_denominator_is_still_56(self, frozen):
        """The one number in this file that published figures actually divide by.

        21/56, 34/56 and 43/56 are over the **25-case** benchmark. The `bio-*`/`mat-*`
        cohort was added later and carries its own targets; adding them must never move
        this. If it does, every recall figure in the paper and RESULTS.md is wrong by
        exactly the C-17 mechanism -- a number measured under one set of coordinates
        quoted against another.
        """
        assert frozen["cohorts"]["benchmark25"] == {"n_targets": 56, "n_cases": 20}

    def test_the_cohorts_partition_the_set(self, frozen):
        from freeze_gold_targets import cohort_of

        counted = {}
        for case, ids in frozen["targets"].items():
            bucket = counted.setdefault(cohort_of(case), {"n_targets": 0, "n_cases": 0})
            bucket["n_targets"] += len(ids)
            bucket["n_cases"] += 1
        assert counted == frozen["cohorts"]
        assert sum(c["n_targets"] for c in counted.values()) == frozen["n_targets"]
        assert sum(c["n_cases"] for c in counted.values()) == frozen["n_cases"]

    def test_unfinished_cases_are_named_not_hidden(self, frozen):
        """A cached pick nobody judged contributes nothing — exactly like a rejected one.

        `mat-phonon` reached that state when arXiv throttled (HTTP 429) mid-run and one of
        its three picks could not be resolved, so it was never scored. Without this field
        the case reads as "the judge found one paper unactionable", which is a different
        claim and one nobody measured. Void, not null, in the gold set itself.
        """
        pending = frozen["incomplete"]
        assert frozen["n_incomplete_picks"] == sum(len(v) for v in pending.values())
        assert all(v for v in pending.values()), "an empty entry says nothing; drop the key"
        for case in pending:
            assert case not in frozen["orphans"], (
                f"{case} cannot be both unjudged and ids-only — the orphan label means the "
                "pick WAS judged, from a cache whose raw no longer reproduces it"
            )

    def test_the_scientific_cohort_is_all_bio_or_mat(self, frozen):
        """A miscategorised case would quietly move the published denominator."""
        from freeze_gold_targets import cohort_of

        sci = {c for c in frozen["targets"] if cohort_of(c) == "scisoft"}
        assert sci and all(c.startswith(("bio-", "mat-")) for c in sci)
        assert not any(c.startswith(("bio-", "mat-")) for c in frozen["targets"] if c not in sci)

    def test_provenance_covers_every_target(self, frozen):
        for case, ids in frozen["targets"].items():
            prov = frozen["provenance"][case]
            assert set(prov) == set(ids), f"{case}: provenance and targets disagree"
            assert set(prov.values()) <= {"raw", "ids-only"}

    def test_the_orphans_are_recorded_not_hidden(self, frozen):
        """The 9 ids-only targets are the ones a prompt edit would destroy.

        They are kept deliberately -- dropping them would move every published denominator
        -- so the artifact must name them rather than let them look like ordinary picks.
        """
        declared = {i for ids in frozen["orphans"].values() for i in ids}
        from_prov = {
            i
            for case, prov in frozen["provenance"].items()
            for i, p in prov.items()
            if p == "ids-only"
        }
        assert declared == from_prov
        assert frozen["n_ids_only"] == len(declared)


class TestTheDerivationStillReproducesIt:
    """Needs the local baseline + judge caches; skipped where they are absent."""

    def test_live_derivation_matches_the_frozen_set(self, frozen):
        if not (EVALS / "cache" / "baseline" / "cli").is_dir():
            pytest.skip("no local baseline cache")
        from build_hop_pool import resolve_targets

        live = {c: sorted(v) for c, v in resolve_targets().items() if v}
        expected = {c: sorted(v) for c, v in frozen["targets"].items()}
        assert live == expected, (
            "the derived gold set no longer matches evals/gold_targets.json -- a published "
            "denominator moved. If that was intended, re-freeze with "
            "`uv run python evals/freeze_gold_targets.py`."
        )
