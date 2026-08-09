"""Tests for the fine-scale calibration audit (evals/calibrate_finescale.py).

The audit decides whether the two frozen constants in `reporadar.finescale` are still
where they were fitted, so its arithmetic is load-bearing on a published claim. Two
properties get the most attention here, because both are ways this analysis could
silently agree with itself:

* the **leave-one-repo-out split** must never train on the repository it scores — and
  the failure mode is subtle, since two repositories can legitimately hold the same
  paper and a value-equality split would then leak the held-out row back in;
* **"not scored" must never read as "scored low"** — in the policy, in the cache, and in
  the run artifact. The whole campaign's discipline about failed LLM calls collapses if
  an omission becomes a zero anywhere along that path.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))

from calibrate_finescale import (  # noqa: E402
    _brier,
    _rate,
    analyse,
    auc,
    ece,
    fit_logistic,
    net2,
    reliability,
    shown_by_policy,
    sign_test,
)
from run_judge_eval import returned_records  # noqa: E402

from reporadar.finescale import INTERCEPT, SHOW_THRESHOLD, SLOPE, probability  # noqa: E402


def row(judge: int, gate: int | None = 2, expectation: float | None = None) -> dict[str, Any]:
    """One audit row. `expectation=None` models a fine-scale call that failed."""
    r: dict[str, Any] = {"arxiv_id": f"2401.{judge}{gate}", "judge_score": judge, "llm_score": gate}
    if expectation is not None:
        r["finescale"] = expectation
        r["finescale_p"] = probability(expectation)
    return r


# The expectation either side of where the frozen map crosses 2/3.
CROSSOVER = (math.log(2.0) - INTERCEPT) / SLOPE  # P(x) == 2/3 exactly here
CLEARS = CROSSOVER + 0.5
FAILS = CROSSOVER - 0.5


class TestPolicy:
    def test_the_crossover_is_where_the_derived_threshold_says_it_is(self) -> None:
        assert probability(CROSSOVER) == round(SHOW_THRESHOLD, 10) or math.isclose(
            probability(CROSSOVER), SHOW_THRESHOLD, abs_tol=1e-9
        )

    def test_a_band_paper_above_the_threshold_is_shown(self) -> None:
        assert shown_by_policy(row(3, gate=2, expectation=CLEARS), SLOPE, INTERCEPT)

    def test_a_band_paper_below_the_threshold_is_not(self) -> None:
        assert not shown_by_policy(row(3, gate=2, expectation=FAILS), SLOPE, INTERCEPT)

    def test_above_the_band_the_gate_is_trusted_without_a_rescore(self) -> None:
        # A gate-3 paper is shown even though it carries no fine-scale score at all.
        assert shown_by_policy({"judge_score": 3, "llm_score": 3}, SLOPE, INTERCEPT)

    def test_below_the_band_is_never_shown(self) -> None:
        assert not shown_by_policy(row(3, gate=1, expectation=CLEARS), SLOPE, INTERCEPT)

    def test_an_unscored_band_paper_is_not_shown(self) -> None:
        """The rule the product follows: could-not-score abstains, it does not admit."""
        assert not shown_by_policy(row(3, gate=2, expectation=None), SLOPE, INTERCEPT)

    def test_a_paper_the_gate_failed_on_is_not_shown(self) -> None:
        assert not shown_by_policy({"judge_score": 3, "llm_score": None}, SLOPE, INTERCEPT)

    def test_moving_the_intercept_moves_the_decision(self) -> None:
        """A refit has to be able to change the outcome, or the counterfactual is inert."""
        paper = row(3, gate=2, expectation=FAILS)
        assert not shown_by_policy(paper, SLOPE, INTERCEPT)
        assert shown_by_policy(paper, SLOPE, INTERCEPT + 2.0)


class TestNet2:
    def test_actionable_pays_one_and_a_dud_costs_two(self) -> None:
        assert net2([row(2), row(3), row(1)]) == 1.0 + 1.0 - 2.0

    def test_an_empty_digest_scores_zero_not_negative(self) -> None:
        assert net2([]) == 0.0


class TestCalibrationMetrics:
    def test_a_perfectly_calibrated_set_has_near_zero_ece(self) -> None:
        # 10 papers at P≈0.9 of which 9 are actionable, 10 at P≈0.1 of which 1 is.
        rows = [{"finescale_p": 0.9, "judge_score": 2 if i < 9 else 0} for i in range(10)]
        rows += [{"finescale_p": 0.1, "judge_score": 2 if i < 1 else 0} for i in range(10)]
        assert ece(rows, bins=5) < 0.02

    def test_a_systematically_overconfident_set_is_caught(self) -> None:
        rows = [{"finescale_p": 0.9, "judge_score": 2 if i < 3 else 0} for i in range(10)]
        assert ece(rows, bins=5) > 0.5

    def test_reliability_bins_report_the_gap_in_both_directions(self) -> None:
        rows = [{"finescale_p": 0.9, "judge_score": 0} for _ in range(4)]
        rows += [{"finescale_p": 0.1, "judge_score": 2} for _ in range(4)]
        table = reliability(rows, bins=5)
        assert [t["n"] for t in table] == [4, 4]
        assert table[0]["empirical"] == 1.0  # under-confident bin
        assert table[-1]["empirical"] == 0.0  # over-confident bin

    def test_auc_is_one_on_perfect_separation(self) -> None:
        rows = [row(3, expectation=8.0), row(0, expectation=1.0)]
        assert auc(rows) == 1.0

    def test_auc_is_none_when_the_set_is_one_class(self) -> None:
        """Undefined must stay undefined; 0.5 would read as 'no signal'."""
        assert auc([row(3, expectation=8.0), row(2, expectation=7.0)]) is None

    def test_brier_and_rate_are_none_on_an_empty_set_not_zero(self) -> None:
        assert _brier([]) is None
        assert _rate([]) is None

    def test_ties_count_half_in_auc(self) -> None:
        rows = [row(3, expectation=5.0), row(0, expectation=5.0)]
        assert auc(rows) == 0.5


class TestFitLogistic:
    def test_it_recovers_a_planted_map(self) -> None:
        true_slope, true_intercept = 1.4, -7.0
        xs, ys = [], []
        for i in range(1000):
            x = (i % 100) / 10.0
            p = 1.0 / (1.0 + math.exp(-(true_slope * x + true_intercept)))
            xs.append(x)
            ys.append(1 if (i / 1000.0) < p else 0)
        slope, intercept = fit_logistic(xs, ys, steps=8000, lr=0.2)
        assert abs(slope - true_slope) < 0.5
        assert abs(intercept - true_intercept) < 2.5
        # The decision boundary is what the audit reads; it must land close.
        assert abs(-intercept / slope - (-true_intercept / true_slope)) < 0.8

    def test_no_data_returns_the_frozen_map_unchanged(self) -> None:
        assert fit_logistic([], []) == (SLOPE, INTERCEPT)


class TestSignTest:
    def test_all_wins_is_significant(self) -> None:
        pos, neg, ties, p = sign_test([1.0] * 10)
        assert (pos, neg, ties) == (10, 0, 0)
        assert p < 0.01

    def test_ties_are_dropped_from_n_and_reported(self) -> None:
        pos, neg, ties, p = sign_test([1.0, -1.0, 0.0, 0.0])
        assert (pos, neg, ties) == (1, 1, 2)
        assert p == 1.0

    def test_all_ties_cannot_manufacture_significance(self) -> None:
        assert sign_test([0.0] * 20)[3] == 1.0


class TestLoroSplit:
    def _data(self) -> tuple[dict[str, list[dict[str, Any]]], dict[str, set[str]]]:
        # Two repos hold the *same* paper id with the same values — the shape that a
        # value-equality split silently leaks. `held` is actionable in both.
        held = {
            "arxiv_id": "2401.00001",
            "judge_score": 3,
            "llm_score": 2,
            "finescale": CLEARS,
            "finescale_p": probability(CLEARS),
        }
        data = {
            "alpha": [dict(held), row(0, gate=2, expectation=FAILS)],
            "beta": [dict(held), row(0, gate=2, expectation=FAILS)],
        }
        recorded = {c: {"2401.00001"} for c in data}
        return data, recorded

    def test_a_held_out_repo_never_appears_in_its_own_training_set(self) -> None:
        data, recorded = self._data()
        out = analyse(data, recorded)
        assert out["n_band"] == 4  # 2 per repo
        # Every fold trains on the whole band minus exactly this repo's share. Verified
        # against the mutation: a split by value equality trains on **0** rows here, not
        # 2 — every row of the other repo matches one of the held-out repo's and is
        # dropped from the training side too. The fit then silently returns its starting
        # values, which reads as "the refit changes nothing", the most plausible-looking
        # wrong answer this audit could produce.
        for entry in out["refit"]["loro"]:
            assert entry["n_train"] == 2, f"{entry['case']} trained on {entry['n_train']} rows"

    def test_the_refit_moves_off_the_frozen_map_when_the_data_disagrees(self) -> None:
        """Guards the inverse failure: a fit that silently returns its initial values."""
        data, recorded = self._data()
        out = analyse(data, recorded)
        assert out["refit"]["global"]["intercept"] != INTERCEPT

    def test_reproduction_is_reported_per_case_and_in_aggregate(self) -> None:
        data, recorded = self._data()
        out = analyse(data, recorded)
        rep = out["reproduction"]
        assert rep["live"] == 2
        assert rep["agree"] == 2  # the clearing paper is rebuilt as shown in both repos

    def test_a_policy_that_disagrees_is_flagged_not_hidden(self) -> None:
        data, _ = self._data()
        # Claim the live run showed a paper the rebuilt policy declines.
        out = analyse(data, {"alpha": {"2401.00001", "2401.02"}, "beta": {"2401.00001"}})
        alpha = next(r for r in out["reproduction"]["cases"] if r["case"] == "alpha")
        assert alpha["only_live"] == ["2401.02"]
        assert out["reproduction"]["agree"] < out["reproduction"]["live"]


class TestReturnedRecords:
    """The run artifact's contract, which the audit had to work around once already."""

    VERDICTS = {"2401.00001": {"score": 3, "justification": "yes"}}

    def test_stage_scores_travel_with_the_verdict(self) -> None:
        papers = [
            {"arxiv_id": "2401.00001v2", "llm_score": 2, "finescale": 6.1, "finescale_p": 0.7}
        ]
        (rec,) = returned_records(papers, self.VERDICTS)
        assert rec["judge_score"] == 3
        assert (rec["llm_score"], rec["finescale"], rec["finescale_p"]) == (2, 6.1, 0.7)

    def test_a_stage_that_did_not_run_leaves_no_key(self) -> None:
        """Absent, not null: a run without triage must not look like a failed triage."""
        (rec,) = returned_records([{"arxiv_id": "2401.00001"}], self.VERDICTS)
        assert "llm_score" not in rec
        assert "finescale_p" not in rec

    def test_a_stage_that_ran_and_failed_records_null(self) -> None:
        (rec,) = returned_records([{"arxiv_id": "2401.00001", "llm_score": None}], self.VERDICTS)
        assert "llm_score" in rec
        assert rec["llm_score"] is None

    def test_a_versioned_id_still_finds_its_verdict(self) -> None:
        (rec,) = returned_records([{"arxiv_id": "2401.00001v7"}], self.VERDICTS)
        assert rec["judge_score"] == 3

    def test_an_unjudged_paper_scores_none_never_zero(self) -> None:
        (rec,) = returned_records([{"arxiv_id": "2999.99999"}], self.VERDICTS)
        assert rec["judge_score"] is None
