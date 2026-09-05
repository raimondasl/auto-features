"""Section 5's endpoints, and the branches each result is allowed to fire. [PREREG §5, §9]

The primary is a repository-clustered AUC of the judge's RAW ordinal score. Level-free by
construction, which is the whole point given NR-59: the two judges order papers almost
identically (AUCs 0.027 apart) and disagree about level by nearly a factor of two, so any
statistic evaluated at a judge's own threshold restates the level disagreement instead of
measuring discrimination.

Two failures are guarded here because both would produce a plausible number rather than an
error. Feeding the primary a THRESHOLDED array turns it into `0.5 + (p_adopted - p_control)/2`
— the secondary wearing the primary's name, with the level put back. And folding an arithmetic
REFUSAL into §5's null branch fires "no demonstrated discrimination" on an artefact, which is
exactly what §3.3 spent $8-12 of extra judging to avoid.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / "evals", ROOT / "evals" / "frame", ROOT / "src"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import judge_validity_pool as jvp  # noqa: E402

ITERS = 400


def _set(n_clusters: int = 8, per: int = 5, *, adopted, control, models=("m",)):  # noqa: ANN001
    """A synthetic analysis set with a controllable score distribution per arm."""
    rng = random.Random(11)
    pos: list[dict[str, object]] = []
    ctl: list[dict[str, object]] = []
    verdicts: dict[str, object] = {}
    for c in range(n_clusters):
        case = f"o/r{c}"
        for i in range(per):
            pid = f"2401.{c:02d}{i:03d}"
            pos.append({"case": case, "id": pid, "stratum": "pool"})
            for m in models:
                verdicts[jvp.verdict_key(m, case, pid)] = {"score": rng.choice(adopted)}
            for j in range(4):
                cid = f"2402.{c:02d}{i}{j:02d}"
                ctl.append({"case": case, "id": cid})
                for m in models:
                    verdicts[jvp.verdict_key(m, case, cid)] = {"score": rng.choice(control)}
    return pos, ctl, verdicts


class TestThePrimaryIsLevelFree:
    def test_a_discriminating_judge_excludes_one_half(self) -> None:
        pos, ctl, v = _set(adopted=[2, 3, 3, 3], control=[0, 0, 1, 1])
        out = jvp.primary_auc("m", pos, ctl, v, iters=ITERS)
        assert out["auc"] > 0.8 and out["excludes_half"] is True

    def test_a_judge_that_orders_at_random_includes_it(self) -> None:
        pos, ctl, v = _set(adopted=[0, 1, 2, 3], control=[0, 1, 2, 3])
        out = jvp.primary_auc("m", pos, ctl, v, iters=ITERS)
        assert out["excludes_half"] is False

    def test_adding_a_constant_to_every_score_changes_nothing(self) -> None:
        """Level-free by construction — that is why §5 makes it primary."""
        pos, ctl, v = _set(adopted=[1, 2, 2], control=[0, 0, 1])
        shifted = {k: {"score": r["score"] + 1} for k, r in v.items()}
        assert (
            jvp.primary_auc("m", pos, ctl, v, iters=ITERS)["auc"]
            == jvp.primary_auc("m", pos, ctl, shifted, iters=ITERS)["auc"]
        )

    def test_a_judge_that_scored_everything_the_same_is_not_refused(self) -> None:
        """Degenerate is not thresholded. Refusing it would abort a run over a judge that gave
        every paper the same score, which is a finding rather than a defect."""
        pos, ctl, v = _set(adopted=[1], control=[1])
        assert jvp.primary_auc("m", pos, ctl, v, iters=ITERS)["tie_fraction"] == 1.0

    def test_a_thresholded_array_is_refused(self) -> None:
        """An AUC over 0/1 is exactly 0.5 + (p_adopted - p_control)/2 — a monotone restatement
        of the secondary, with the level §5 exists to remove carried back in."""
        pos, ctl, v = _set(adopted=[1], control=[0])
        with pytest.raises(SystemExit) as exc:
            jvp.primary_auc("m", pos, ctl, v, iters=ITERS)
        assert "2-valued score array" in str(exc.value)

    def test_a_control_cluster_with_no_positive_is_refused(self) -> None:
        """It becomes a phantom every bootstrap draw can pick, diluting the resample and
        inflating the cluster count with repositories that contribute nothing."""
        pos, ctl, v = _set(n_clusters=3, adopted=[3], control=[0])
        stray = {"case": "ghost/x", "id": "2402.99999"}
        v[jvp.verdict_key("m", "ghost/x", "2402.99999")] = {"score": 0}
        with pytest.raises(SystemExit) as exc:
            jvp.primary_auc("m", pos, [*ctl, stray], v, iters=ITERS)
        assert "carry no positive" in str(exc.value)


class TestTheCoReportsAreAllPresent:
    def test_every_registered_co_report_is_emitted(self) -> None:
        pos, ctl, v = _set(adopted=[2, 3], control=[0, 1])
        out = jvp.primary_auc("m", pos, ctl, v, iters=ITERS)
        assert set(out) >= {
            "auc",
            "ci95",
            "excludes_half",
            "se",
            "n_positives",
            "n_controls",
            "n_clusters",
            "largest_cluster_share",
            "design_effect",
            "min_detectable_auc_80pct",
            "iters",
            "seed",
        }

    def test_the_design_effect_is_measured_not_assumed(self) -> None:
        """A ratio of two realised bootstrap variances — the ICC is the quantity nobody knows,
        and assuming one would put the answer into the uncertainty estimate by hand. The arms
        overlap here on purpose: perfectly separated classes have zero paper-level variance, so
        the ratio is undefined and correctly comes back null."""
        pos, ctl, v = _set(adopted=[1, 2, 3], control=[0, 1, 2])
        assert jvp.primary_auc("m", pos, ctl, v, iters=ITERS)["design_effect"] > 0

    def test_perfect_separation_leaves_the_design_effect_undefined(self) -> None:
        pos, ctl, v = _set(adopted=[3], control=[0])
        assert jvp.primary_auc("m", pos, ctl, v, iters=ITERS)["design_effect"] is None

    def test_the_tie_fraction_is_published_beside_the_auc(self) -> None:
        """`roc_auc` is the Mann-Whitney form over average ranks, so with four levels over
        hundreds of papers many comparisons are decided by nothing — an AUC of 0.58 from 70%
        ties is a different claim from one built from 5%."""
        pos, ctl, v = _set(adopted=[1], control=[1])
        assert jvp.primary_auc("m", pos, ctl, v, iters=ITERS)["tie_fraction"] == 1.0
        pos, ctl, v = _set(adopted=[3], control=[0])
        assert jvp.primary_auc("m", pos, ctl, v, iters=ITERS)["tie_fraction"] == 0.0

    def test_the_score_histogram_covers_all_four_levels(self) -> None:
        pos, ctl, v = _set(adopted=[3], control=[0])
        hist = jvp.primary_auc("m", pos, ctl, v, iters=ITERS)["score_histogram"]
        assert set(hist["adopted"]) == {"0", "1", "2", "3"}
        assert hist["adopted"]["3"] == 40 and hist["control"]["0"] == 160

    def test_the_paper_level_interval_is_not_surfaced(self) -> None:
        """Reporting both invites quoting the narrower one; it is computed only so the design
        effect is a measured ratio."""
        pos, ctl, v = _set(adopted=[2, 3], control=[0, 1])
        out = jvp.primary_auc("m", pos, ctl, v, iters=ITERS)
        assert not any("paper" in k and "ci" in k for k in out)


class TestThePreregsOwnSizingTableIsCheckedNotQuoted:
    def test_the_interval_column_reproduces(self) -> None:
        """§9's CI column is exactly 0.60 +/- 1.96*SE_upper."""
        for row in jvp.prereg_s9_reference()["rows"]:
            lo, hi = row["ci_recomputed_at_1_96_se_upper"]
            assert abs(lo - row["ci_at_060"][0]) <= 0.0015  # §9 rounds to 3dp
            assert abs(hi - row["ci_at_060"][1]) <= 0.0015

    def test_the_mda_column_does_not_come_from_the_committed_formula(self) -> None:
        """0.5 + 2.80*SE gives 0.660 at 90 positives and 0.623 at 130, against §9's 0.62 and
        0.578. The implied multipliers are 2.11 and 1.77 — nearer 1.96*SE, a 50%-power
        quantity, than the 2.80 = 1.96 + 0.84 that 80% power needs. Recorded, not reconciled:
        the constant is frozen and §9 is registered text."""
        rows = jvp.prereg_s9_reference()["rows"]
        assert [r["implied_multiplier_of_printed_mda"] for r in rows] == [2.11, 1.77]
        assert rows[0]["mda_from_committed_formula"][1] == 0.66
        assert rows[1]["mda_from_committed_formula"][1] == 0.623

    def test_detectability_is_claimed_from_the_realised_value(self) -> None:
        pos, ctl, v = _set(adopted=[2, 3], control=[0, 1])
        out = jvp.primary_auc("m", pos, ctl, v, iters=ITERS)
        assert out["min_detectable_auc_80pct"] == round(0.5 + 2.80 * out["se"], 4)


class TestTheSecondaryUsesTheShippedThresholdAndSaysSo:
    def test_the_threshold_is_imported_not_retyped(self) -> None:
        """Four copies of that constant already exist in this tree."""
        from metrics import RELEVANT_THRESHOLD

        pos, ctl, v = _set(adopted=[3], control=[0])
        assert jvp.secondary_gap("m", pos, ctl, v, iters=ITERS)["threshold"] == RELEVANT_THRESHOLD

    def test_the_gap_and_both_intervals_are_emitted(self) -> None:
        pos, ctl, v = _set(adopted=[2, 3], control=[0, 1])
        out = jvp.secondary_gap("m", pos, ctl, v, iters=ITERS)
        assert out["gap"] == pytest.approx(1.0)
        assert out["adopted"]["wilson95_paper_level"] is not None
        assert out["gap_cluster_bootstrap"]["ci95"] is not None
        assert out["gap_excludes_zero"] is True

    def test_the_wilson_interval_is_labelled_paper_level(self) -> None:
        """It does not account for repository clustering, and the bootstrap beside it does —
        so the two must not read as a pair of equivalents."""
        pos, ctl, v = _set(adopted=[3], control=[0])
        out = jvp.secondary_gap("m", pos, ctl, v, iters=ITERS)
        assert "wilson95_paper_level" in out["adopted"]
        assert "PAPER-LEVEL" in out["_note"]

    def test_an_empty_arm_refuses_rather_than_dividing(self) -> None:
        pos, ctl, v = _set(adopted=[3], control=[0])
        assert "_refused" in jvp.secondary_gap("m", pos, [], v, iters=ITERS)

    def test_the_control_base_rate_carries_no_consequence(self) -> None:
        pos, ctl, v = _set(adopted=[3], control=[0, 3])
        rate = jvp.control_base_rate(jvp.secondary_gap("m", pos, ctl, v, iters=ITERS))
        assert 0.0 <= rate["rate"] <= 1.0
        assert "No pre-committed consequence" in rate["_note"]


class TestTheJudgeDifferenceIsPairedAtTheClusterLevel:
    def test_it_is_computed_on_the_intersection_only(self) -> None:
        """A paper can carry one judge's verdict and not the other's and still be fully
        explained by the void ledger; a "paired" difference over two samples is the error
        pairing exists to avoid."""
        pos, ctl, v = _set(adopted=[2, 3], control=[0, 1], models=("a", "b"))
        for row in pos[:5]:
            del v[jvp.verdict_key("b", str(row["case"]), str(row["id"]))]
        out = jvp.judge_difference(("a", "b"), pos, ctl, v, iters=ITERS)
        assert out["n_paired_positives"] == len(pos) - 5

    def test_a_difference_of_aucs_not_of_gaps(self) -> None:
        pos, ctl, v = _set(adopted=[3], control=[0], models=("a", "b"))
        for row in pos:
            v[jvp.verdict_key("b", str(row["case"]), str(row["id"]))] = {"score": 0}
        for row in ctl:
            v[jvp.verdict_key("b", str(row["case"]), str(row["id"]))] = {"score": 3}
        out = jvp.judge_difference(("a", "b"), pos, ctl, v, iters=ITERS)
        assert out["delta_auc"] == pytest.approx(1.0)
        assert out["excludes_zero"] is True

    def test_two_identical_judges_differ_by_nothing(self) -> None:
        pos, ctl, v = _set(adopted=[2, 3], control=[0, 1], models=("a",))
        for k in list(v):
            v[k.replace("a|", "b|", 1)] = v[k]
        out = jvp.judge_difference(("a", "b"), pos, ctl, v, iters=ITERS)
        assert out["delta_auc"] == 0.0
        assert out["excludes_zero"] is False

    def test_the_retired_bar_is_recorded_as_retired(self) -> None:
        pos, ctl, v = _set(adopted=[3], control=[0], models=("a", "b"))
        out = jvp.judge_difference(("a", "b"), pos, ctl, v, iters=ITERS)
        assert "RETIRED" in out["rung1_0_15_rule"]

    def test_the_wrapper_never_names_a_better_instrument(self) -> None:
        """§5 retires that decision rule before the data, so the code path must not be able to
        reach it."""
        import inspect

        for fn in (jvp.judge_difference, jvp.consequences, jvp.primary_auc, jvp.secondary_gap):
            src = inspect.getsource(fn)
            assert "better_instrument" not in src
            assert "SEPARATES" not in src and "FLAT_GAP" not in src


class TestARefusalIsNotANull:
    def _primary(self, **over):  # noqa: ANN202
        return {"m": {"n_positives": 4, "n_controls": 8, **over}}

    def test_an_interval_excluding_one_half_is_conservative_evidence(self) -> None:
        out = jvp.consequences(self._primary(excludes_half=True), {})["per_judge"]["m"]
        assert out["outcome"] == "excludes_0.5"
        assert "LOWER BOUND" in out["lower_bound_argument"]

    def test_an_interval_including_it_carries_the_three_way_ambiguity(self) -> None:
        out = jvp.consequences(
            self._primary(excludes_half=False, min_detectable_auc_80pct=0.62, n_clusters=9), {}
        )["per_judge"]["m"]
        assert out["outcome"] == "includes_0.5" and out["clean_negative"] is False
        carried = out["carry_beside_headline"]
        assert "0.62" in carried and "9 clusters" in carried
        assert "never got to" in carried

    def test_an_arithmetic_refusal_is_its_own_outcome(self) -> None:
        """§3.3 spent $8-12 of extra judging so the interval would not include 0.5 for a
        SAMPLING reason and fire the null branch on an artefact. Folding a refusal into that
        branch fires it on an artefact by another route."""
        out = jvp.consequences(self._primary(_refused="fewer than two clusters", n_clusters=1), {})[
            "per_judge"
        ]["m"]
        assert out["outcome"] == "no_interval"
        assert "NOT §5's null branch" in out["not_a_null"]

    def test_a_judge_never_asked_is_distinct_from_one_that_could_not_be_measured(self) -> None:
        out = jvp.consequences({"m": {"n_positives": 0, "n_controls": 0}}, {})["per_judge"]["m"]
        assert out["outcome"] == "void"

    def test_every_branch_states_the_primary_label_is_unchanged(self) -> None:
        for over in (
            {"excludes_half": True},
            {"excludes_half": False, "min_detectable_auc_80pct": 0.6, "n_clusters": 5},
            {"_refused": "x", "n_clusters": 1},
        ):
            out = jvp.consequences(self._primary(**over), {})["per_judge"]["m"]
            assert out["primary_label_unchanged"] is True
            assert "RETIRED" in out["rung1_0_15_rule"]

    def test_both_excluding_leaves_the_base_rates_unresolved(self) -> None:
        both = {m: {"n_positives": 4, "n_controls": 8, "excludes_half": True} for m in ("a", "b")}
        out = jvp.consequences(both, {"a": {"rate": 0.87}, "b": {"rate": 0.49}})
        assert out["both_exclude_half"] is True
        assert "0.87" in out["both_exclude_statement"]

    def test_one_refusal_stops_both_from_excluding(self) -> None:
        mixed = {
            "a": {"n_positives": 4, "n_controls": 8, "excludes_half": True},
            "b": {"n_positives": 4, "n_controls": 8, "_refused": "one cluster"},
        }
        assert jvp.consequences(mixed, {})["both_exclude_half"] is False


class TestTheShortfallBlockNeverRefusesOnN:
    def test_it_reports_below_the_minimum_and_still_runs(self) -> None:
        """§3.4: below 60 the analysis RUNS at whatever n exists; the shortfall is reported
        against §9's sizing rather than being a reason to stop."""
        analysis = {
            "analysis_set_positives": 31,
            "n_clusters": 8,
            "by_stratum": {"pool": [], "legacy": [{}] * 31},
        }
        out = jvp.shortfall(analysis, {"stop_reason": "exhausted", "target": 100}, {})
        assert out["below_reporting_minimum"] is True
        assert out["n_positives_legacy"] == 31 and out["n_positives_new"] == 0
        assert out["stop_reason"] == "exhausted"

    def test_it_never_conflates_the_stop_count_with_the_analysis_set(self) -> None:
        """The stop rule counted before the cross-repository contest removed anything."""
        analysis = {"analysis_set_positives": 96, "by_stratum": {"pool": [{}] * 96, "legacy": []}}
        out = jvp.shortfall(analysis, {"stop_rule_capped_positives": 100}, {})
        assert out["analysis_set_positives"] == 96
        assert out["stop_rule_capped_positives"] == 100

    def test_it_carries_the_realised_detectability_per_judge(self) -> None:
        out = jvp.shortfall({"by_stratum": {}}, None, {"m": {"min_detectable_auc_80pct": 0.61}})
        assert out["realised_min_detectable_auc_80pct"] == {"m": 0.61}
        assert out["prereg_s9_reference"]["rows"]
