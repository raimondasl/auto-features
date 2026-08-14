"""Tests for the phrase-query arm (`--rr-bigrams`) and its report.

The report exists to stop two specific mistakes this project has already made once each,
so those are what these tests pin:

* **A void arm reported as a null.** The first IACR measurement scored both arms
  identically because zero IACR papers ever reached a top-10. `divergence` is what
  catches that shape, and it must return 0 changed cases when the arms are identical —
  otherwise a flag that did nothing reads as a flag that did nothing *useful*, which is a
  completely different claim.
* **An arm reported under a name its own run file contradicts.** Three run files that
  differ only by one flag are easy to pass in the wrong order on the command line.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))

from ablation_report import digest_width  # noqa: E402
from bigram_report import (  # noqa: E402
    check_labels,
    divergence,
    mre_for,
    paired_bootstrap,
    top10_ids,
)


def _case(name: str, ids: list[str], net: float = 0.0, mode: str = "adjacent") -> dict[str, Any]:
    return {
        "case": name,
        "bigram_mode": mode,
        "reporadar_toppicks": {"net_value@2": net},
        "returned": {"reporadar_top10": [{"arxiv_id": i} for i in ids]},
    }


class TestArmValidity:
    def test_identical_arms_report_zero_changed_cases(self) -> None:
        """The IACR shape: the flag reached nothing, so there is no effect to report."""
        control = {"a": _case("a", ["1", "2"]), "b": _case("b", ["3"])}
        arm = {"a": _case("a", ["1", "2"]), "b": _case("b", ["3"])}
        d = divergence(control, arm, ["a", "b"])
        assert d["changed_cases"] == 0
        assert d["mean_jaccard"] == 1.0

    def test_differing_arms_are_counted(self) -> None:
        control = {"a": _case("a", ["1", "2"]), "b": _case("b", ["3"])}
        arm = {"a": _case("a", ["1", "9"]), "b": _case("b", ["3"])}
        d = divergence(control, arm, ["a", "b"])
        assert d["changed_cases"] == 1
        # a: |{1}| / |{1,2,9}| = 1/3; b: identical = 1.0
        assert d["mean_jaccard"] == pytest.approx((1 / 3 + 1.0) / 2)

    def test_two_abstaining_cases_are_not_counted_as_divergent(self) -> None:
        """Both returned nothing — that is agreement, not a division by zero."""
        control = {"a": _case("a", [])}
        arm = {"a": _case("a", [])}
        d = divergence(control, arm, ["a"])
        assert d["changed_cases"] == 0
        assert d["mean_jaccard"] == 1.0

    def test_top10_ids_ignores_records_without_an_id(self) -> None:
        record = {"returned": {"reporadar_top10": [{"arxiv_id": "1"}, {"title": "no id"}]}}
        assert top10_ids(record) == {"1"}

    def test_missing_returned_block_is_empty_not_an_error(self) -> None:
        assert top10_ids({"case": "x"}) == set()


class TestLabelCheck:
    def test_mislabelled_arm_is_refused(self) -> None:
        arm = {"a": _case("a", ["1"], mode="none")}
        with pytest.raises(SystemExit, match="refusing to report an arm"):
            check_labels("verified", arm)

    def test_matching_label_passes(self) -> None:
        check_labels("none", {"a": _case("a", ["1"], mode="none")})

    def test_run_predating_the_flag_is_trusted_with_a_warning(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        arm = {"a": {"case": "a", "reporadar_toppicks": {"net_value@2": 0.0}}}
        check_labels("adjacent", arm)
        assert "no `bigram_mode` recorded" in capsys.readouterr().out


class TestTheFloorTracksTheWidthToo:
    """The floor is a property of the whole configuration, not one axis of it.

    `mre_for` keyed on pool provenance alone until 2026-08-15, when re-measuring at the new
    returned-set cut gave **0.74 against 0.48** on the same frozen pool — more papers per
    case means more chances for temperature-0 jitter to move one across the display
    threshold, and each is worth +1 or −2. A guard that is precise about one dimension and
    silent about another reads as authority on both, and this one was returning a floor 35%
    too tight for every future window-15 experiment.
    """

    def test_the_wider_window_has_the_wider_floor(self) -> None:
        assert mre_for("frozen", "15")[0] > mre_for("frozen", "10")[0]

    def test_the_measured_values(self) -> None:
        assert mre_for("frozen", "10")[0] == 0.48
        assert mre_for("frozen", "15")[0] == 0.74

    def test_an_unmeasured_width_falls_back_to_the_widest_known(self) -> None:
        """Not the nearest. Under-reporting the floor turns noise into a finding, and that
        is the direction that costs a published claim."""
        floor, why = mre_for("frozen", "25")
        assert floor == 0.74
        assert "UNMEASURED" in why

    def test_width_defaults_to_the_pre_flag_value(self) -> None:
        """Every run before the flag was cut at 10, so the default cannot silently widen
        the floor for the corpus it was measured on."""
        assert mre_for("frozen") == mre_for("frozen", "10")

    def test_live_arms_are_unaffected_by_width(self) -> None:
        """The live floor was measured at window 10 and no live window-15 draw exists;
        claiming otherwise would invent a number."""
        assert mre_for("live", "15")[0] == 1.04


class TestTheFloorIsDerivedNotChosen:
    """Which MRE applies is a property of how the arms were collected.

    A frozen-pool comparison read against the live floor would call a real effect
    unresolvable — and a flag defaulting to the live value would do exactly that by
    omission, which is the shape of every silently-wrong default this project has paid for.
    """

    def test_live_arms_get_the_live_floor(self) -> None:
        assert mre_for("live")[0] == 1.04

    def test_unlabelled_runs_are_treated_as_live(self) -> None:
        """Runs predating `--rr-frozen-pool` carry no provenance and were all live."""
        assert mre_for("unlabelled")[0] == 1.04

    @pytest.mark.parametrize("mode", ["frozen", "frozen:abc123", "frozen-seeded"])
    def test_frozen_arms_get_the_frozen_floor(self, mode: str) -> None:
        assert mre_for(mode)[0] == 0.48

    def test_the_frozen_floor_is_the_tighter_one(self) -> None:
        """Freezing removes the dominant variance term; if this inverts, something is wrong."""
        assert mre_for("frozen")[0] < mre_for("live")[0]


class TestLabelFieldIsConfigurable:
    def test_a_different_field_identifies_a_different_experiment(self) -> None:
        arm = {"a": {"case": "a", "absent_category": "zero", "bigram_mode": "verified"}}
        check_labels("zero", arm, "absent_category")
        with pytest.raises(SystemExit, match="refusing to report an arm"):
            check_labels("impute", arm, "absent_category")

    def test_the_default_field_is_still_bigram_mode(self) -> None:
        arm = {"a": {"case": "a", "bigram_mode": "none"}}
        check_labels("none", arm)

    def test_a_numeric_arm_matches_its_command_line_label(self) -> None:
        """`gate_depth` is an int and a CLI label is a string.

        Compared raw, the guard would fire on every arm of the depth experiment — and a
        check that always fails gets deleted, which is worse than one that never fires.
        """
        arm = {"a": {"case": "a", "gate_depth": 50}}
        check_labels("50", arm, "gate_depth")

    def test_a_numeric_arm_still_catches_a_swap(self) -> None:
        arm = {"a": {"case": "a", "gate_depth": 15}}
        with pytest.raises(SystemExit, match="refusing to report an arm"):
            check_labels("50", arm, "gate_depth")

    def test_the_digest_window_is_checkable_too(self) -> None:
        """`--rr-window` costs judge calls, so an arm reported under the wrong window
        would attribute one run's verdicts to a digest size that never produced them."""
        arm = {"a": {"case": "a", "digest_window": 15}}
        check_labels("15", arm, "digest_window")
        with pytest.raises(SystemExit, match="refusing to report an arm"):
            check_labels("10", arm, "digest_window")


class TestTheDigestWidthGuard:
    """A width mismatch is worth more than any treatment this project has published.

    The returned-set cut moved 10 -> 15 on 2026-08-15 at **+1.24 net@2/case**, and 91 of the
    92 runs on disk at that moment were the narrower one. Comparing across it measures the
    width and reports it under whatever the arms were named — the same shape as comparing a
    frozen arm against a live one, which the provenance guard already refuses.
    """

    def test_runs_predating_the_flag_read_as_ten(self) -> None:
        """Unlike pool provenance, the pre-flag value is KNOWN: it was a literal in the
        source, not a default anyone could have passed. So these read '10', not
        'unlabelled' — there is nothing to assume."""
        assert digest_width({"a": {"case": "a"}}) == "10"

    def test_a_recorded_width_is_used(self) -> None:
        assert digest_width({"a": {"case": "a", "digest_window": 15}}) == "15"

    def test_a_run_with_two_widths_is_mixed(self) -> None:
        """One run cannot have two windows unless something rewrote it; say so rather than
        silently picking one."""
        arm = {"a": {"case": "a", "digest_window": 10}, "b": {"case": "b", "digest_window": 15}}
        assert digest_width(arm) == "mixed"

    def test_none_is_treated_as_the_pre_flag_default(self) -> None:
        assert digest_width({"a": {"case": "a", "digest_window": None}}) == "10"


class TestPairedBootstrap:
    def test_is_deterministic(self) -> None:
        """A seeded CI, so re-running the report cannot quietly change a published number."""
        deltas = [1.0, -2.0, 0.5, 3.0, -1.0]
        assert paired_bootstrap(deltas) == paired_bootstrap(deltas)

    def test_brackets_the_observed_mean(self) -> None:
        deltas = [1.0, 2.0, 1.5, 0.5, 1.2]
        lo, hi = paired_bootstrap(deltas)
        assert lo <= sum(deltas) / len(deltas) <= hi

    def test_zero_variance_gives_a_degenerate_interval(self) -> None:
        lo, hi = paired_bootstrap([2.0] * 8)
        assert lo == hi == 2.0

    def test_empty_is_not_a_crash(self) -> None:
        assert paired_bootstrap([]) == (0.0, 0.0)
