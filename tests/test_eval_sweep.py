"""Tests for the Tier B gate-precision sweep (evals/run_judge_eval.py).

The sweep re-gates the same triaged Top-10 at each ``min_actionable`` threshold
(free — triage scores are computed once) to show the precision/recall trade of
raising the Top-Pick bar, and the cross-case rollup shows which threshold kills
false positives (e.g. the webdev negative control leaking a Top Pick).
"""

from __future__ import annotations

import math

from run_judge_eval import SWEEP_THRESHOLDS, aggregate_sweep, sweep_top_picks


def _gains_of(papers: list[dict]) -> list[int]:
    # Test double for the judge: each paper carries its own judge gain.
    return [p["_gain"] for p in papers]


class TestSweepTopPicks:
    def test_gates_and_scores_each_threshold(self) -> None:
        # b is a triage false positive: haiku says 2, judge says 0.
        ranked = [
            {"arxiv_id": "a", "llm_score": 3, "_gain": 3},
            {"arxiv_id": "b", "llm_score": 2, "_gain": 0},
            {"arxiv_id": "c", "llm_score": 1, "_gain": 2},
        ]
        pool = [3, 0, 2]
        sw = sweep_top_picks(ranked, _gains_of, pool)

        # >=1 returns a,b,c -> gains [3,0,2]: 2 actionable, 1 not -> net 2-2 = 0
        assert sw[1]["n_returned"] == 3
        assert sw[1]["net_value@2"] == 0.0
        # >=2 returns a,b -> gains [3,0]: 1 actionable, 1 not -> net 1-2 = -1
        assert sw[2]["n_returned"] == 2
        assert sw[2]["net_value@2"] == -1.0
        # >=3 returns a -> gains [3]: 1 actionable, 0 not -> net +1, precision 1.0
        assert sw[3]["n_returned"] == 1
        assert sw[3]["net_value@2"] == 1.0
        assert sw[3]["precision"] == 1.0

    def test_covers_all_thresholds(self) -> None:
        sw = sweep_top_picks([], _gains_of, [])
        assert set(sw) == set(SWEEP_THRESHOLDS)
        assert all(sw[t]["abstained"] for t in SWEEP_THRESHOLDS)  # empty -> all abstain


class TestAggregateSweep:
    def _m(self, returned: int, actionable: int, net: float, prec: float) -> dict:
        return {
            "n_returned": returned,
            "n_actionable": actionable,
            "net_value@2": net,
            "precision": prec,
        }

    def test_rollup_counts_false_positives_and_abstentions(self) -> None:
        # case1: a healthy repo; case2: a webdev-like negative control.
        per_case = [
            {2: self._m(3, 2, 0.0, 0.67), 3: self._m(1, 1, 1.0, 1.0)},
            {2: self._m(1, 0, -2.0, 0.0), 3: self._m(0, 0, 0.0, float("nan"))},
        ]
        agg = aggregate_sweep(per_case, thresholds=(2, 3))

        # At min>=2 the negative control leaks a false positive and drags the mean down.
        assert agg[2]["mean_net@2"] == -1.0  # (0.0 + -2.0) / 2
        assert agg[2]["n_false_positive"] == 1
        assert agg[2]["n_abstained"] == 0

        # At min>=3 the false positive is gone (it abstains) and the mean rises.
        assert agg[3]["mean_net@2"] == 0.5  # (1.0 + 0.0) / 2
        assert agg[3]["n_false_positive"] == 0
        assert agg[3]["n_abstained"] == 1

    def test_mean_precision_ignores_abstained_cases(self) -> None:
        # Only returned>0 cases count toward mean precision; an abstention is not 0.
        per_case = [
            {2: self._m(2, 2, 2.0, 1.0)},
            {2: self._m(0, 0, 0.0, float("nan"))},
        ]
        agg = aggregate_sweep(per_case, thresholds=(2,))
        assert agg[2]["mean_precision"] == 1.0
        assert not math.isnan(agg[2]["mean_precision"])
