"""Every flag that decides a run is written into the run's own record.

The gap this closes was found the expensive way. Reproducing the 2026-08-21 bio arm needed
its exact retrieval settings; the frozen-pool fingerprint guard correctly refused two
attempts, and the right command turned out to be recoverable only from a **prose section** of
`RESEARCH-scientific-software.md`. An audit then showed **7 of the 12 POOL_FLAGS were absent
from every run record** — `rr_all_time`, `rr_prose_chars`, `rr_hyde`, `rr_hyde_index`,
`rr_hyde_hypotheses`, `rr_hyde_top_k`, `rr_triage_model`. No artifact could answer "was HyDE
on?", for any run this project has ever made.

`digest_window` is recorded for exactly this reason, and said so: *"an arm cannot be reported
under a window its own run file contradicts."* A flag that decides the candidate pool has at
least as strong a claim.

The fix is `flag_values(args, POOL_FLAGS)` rather than a hand-listed block, because a constant
and a hand-maintained mirror of it drift silently — the same argument that makes
`KEYWORD_SOURCES` drive its fetcher loop. These tests hold that construction in place: adding
a flag to either group must make it appear in the record automatically, and if someone
reverts to a literal dict the parity assertions below fail.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))

import run_judge_eval as R  # noqa: E402


def _args(**over) -> argparse.Namespace:
    ns = argparse.Namespace(**{f: None for f in (*R.POOL_FLAGS, *R.RANKING_FLAGS)})
    for k, v in over.items():
        setattr(ns, k, v)
    return ns


class TestEveryDecidingFlagIsRecorded:
    def test_pool_config_covers_pool_flags_exactly(self):
        """Exactly, not merely 'at least': a superset would mean the record claims to
        describe a dimension the fingerprint does not cover, which is the opposite error."""
        assert set(R.flag_values(_args(), R.POOL_FLAGS)) == set(R.POOL_FLAGS)

    def test_ranking_config_covers_ranking_flags_exactly(self):
        assert set(R.flag_values(_args(), R.RANKING_FLAGS)) == set(R.RANKING_FLAGS)

    def test_the_seven_that_were_missing_are_present(self):
        """Named individually. A regression that dropped one would still satisfy the parity
        test above if it also dropped it from the constant — these are the flags whose
        absence actually cost an investigation."""
        rec = R.flag_values(_args(rr_hyde=True, rr_all_time=True), R.POOL_FLAGS)
        for flag in (
            "rr_all_time",
            "rr_prose_chars",
            "rr_hyde",
            "rr_hyde_index",
            "rr_hyde_hypotheses",
            "rr_hyde_top_k",
            "rr_triage_model",
        ):
            assert flag in rec, flag
        assert rec["rr_hyde"] is True and rec["rr_all_time"] is True

    def test_a_new_flag_records_itself(self):
        """The construction, not the current contents. If `flag_values` is ever replaced by
        a literal dict, this is what notices."""
        extended = (*R.POOL_FLAGS, "rr_invented_axis")
        rec = R.flag_values(_args(rr_invented_axis="on"), extended)
        assert rec["rr_invented_axis"] == "on"

    def test_the_record_is_json_serialisable(self):
        """`rr_hyde_index` is a Path on the argparse namespace, and a Path is not JSON. The
        record is written with `json.dumps`, so a non-scalar here fails the run at the very
        end — after every judge call has been paid for."""
        rec = R.flag_values(_args(rr_hyde_index=Path("evals/.work/hyde_index")), R.POOL_FLAGS)
        assert isinstance(rec["rr_hyde_index"], str)
        json.dumps(rec)

    def test_a_missing_attribute_reads_as_absent_not_as_a_crash(self):
        """Namespaces differ between entry points; a flag one runner does not define must not
        abort a run at write time."""
        assert R.flag_values(argparse.Namespace(), ("rr_nonexistent",)) == {"rr_nonexistent": None}


class TestTheFrozenPoolDescribesItself:
    def test_a_written_pool_carries_its_flag_values(self, tmp_path):
        R.save_frozen_pool(
            tmp_path,
            "case-x",
            "deadbeef",
            [{"arxiv_id": "2401.00001"}],
            pool_config=R.flag_values(_args(rr_hyde=True, sources=["arxiv"]), R.POOL_FLAGS),
        )
        stored = json.loads(R.frozen_pool_path(tmp_path, "case-x").read_text(encoding="utf-8"))
        assert stored["pool_flags"] == list(R.POOL_FLAGS), "the NAMES, for mismatch diagnosis"
        assert stored["pool_config"]["rr_hyde"] is True, "and the VALUES, so it self-describes"

    def test_pools_written_before_this_still_read(self, tmp_path):
        """Additive on purpose. Re-seeding the frozen pools is expensive — `pool-cohort3`
        alone is 21k papers — so a format change that stranded them would cost far more than
        the gap it closes."""
        R.save_frozen_pool(tmp_path, "case-y", "deadbeef", [{"arxiv_id": "2401.00002"}])
        stored = json.loads(R.frozen_pool_path(tmp_path, "case-y").read_text(encoding="utf-8"))
        assert stored["pool_config"] == {}
        assert stored["version"] == R.FROZEN_POOL_VERSION, "no version bump: nothing is stale"

    def test_an_empty_pool_is_still_never_written(self, tmp_path):
        """Pre-existing invariant, re-checked because this change touched the writer: an
        empty candidate list and a failed collection are the same bytes on disk, and a frozen
        empty scores as a legitimate 0.0 on every run that reuses it."""
        R.save_frozen_pool(tmp_path, "case-z", "deadbeef", [], pool_config={"a": 1})
        assert not R.frozen_pool_path(tmp_path, "case-z").exists()


class TestTheGroupsStayDisjoint:
    def test_a_flag_is_pool_affecting_or_ranking_never_both(self):
        """The distinction is what licenses reusing a frozen pool across an arm: ranking
        flags may vary against one, pool flags may not. A flag in both groups would make the
        guard's refusal message incoherent and the reuse rule unsound."""
        assert not (set(R.POOL_FLAGS) & set(R.RANKING_FLAGS))

    @pytest.mark.parametrize("flag", ["rr_window", "rr_w_embedding"])
    def test_the_axes_varied_against_a_frozen_pool_are_ranking_flags(self, flag):
        """P21 varied both against `pool-epmc-treat`. If either were pool-affecting, that
        comparison measured one arm's settings over the other arm's candidates."""
        assert flag in R.RANKING_FLAGS and flag not in R.POOL_FLAGS
