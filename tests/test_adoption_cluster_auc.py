"""§6.4's primary endpoint, which did not exist in code. [frame §6.4]

The frame registers the pool's primary as *AUC of the judge's ordinal rubric score against
adopted/control, with a repo-cluster bootstrap*. `judge_validity_adoption.py` had neither
half: no AUC anywhere, and an interval that resamples positives and controls independently,
one paper at a time.

That matters here more than it usually would. NR-56 drew 31 positives from 6 repositories
with `graph` supplying 13; NR-57's 35 came from 9 with the same shape. A paper-level
bootstrap treats 13 papers out of one project's bibliography as 13 independent draws, so it
answers "if we had drawn different PAPERS from these repositories" when the design question
is "if we had drawn different REPOSITORIES". The last test below is the one that matters:
under concentration the clustered interval is materially wider, which is the defect being
fixed, stated as a measurement rather than as an assertion.

Nothing here touches the real verdicts. P13 registers AUC at ≥ 90 capped positives *including
the legacy cluster*, and §6.4 registers legacy-versus-pool heterogeneity — so computing the
legacy AUC before the pool runs would answer a registered prediction from the data it is
registered against. These fixtures are synthetic for that reason.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

EVALS = Path(__file__).resolve().parent.parent / "evals"


def _load(name: str):  # type: ignore[no-untyped-def]
    for path in (EVALS, EVALS.parent / "src"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    spec = importlib.util.spec_from_file_location(name, EVALS / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


jva = _load("judge_validity_adoption")
metrics = _load("metrics")


class TestTheAucItself:
    def test_perfect_separation_is_one_and_reversal_is_zero(self) -> None:
        assert metrics.roc_auc([3, 3, 2], [0, 0, 1]) == 1.0
        assert metrics.roc_auc([0, 0, 1], [3, 3, 2]) == 0.0

    def test_identical_distributions_are_a_half(self) -> None:
        assert metrics.roc_auc([1, 2, 3], [1, 2, 3]) == 0.5

    def test_ties_score_half_rather_than_being_broken(self) -> None:
        """A judge's rubric has four levels for hundreds of papers, so ties are the common
        case, not the edge case. Breaking them by list position would let the AUC depend on
        the order rows happen to arrive in."""
        assert metrics.roc_auc([2, 2], [2, 2]) == 0.5
        assert metrics.roc_auc([2], [1, 2, 3]) == pytest.approx(0.5)

    def test_it_is_invariant_to_the_judges_level(self) -> None:
        """The whole reason §6.4 makes it primary. NR-59 measured the judges ordering alike
        (AUCs 0.027 apart) and disagreeing about level by 0.380; a statistic taken at each
        judge's own threshold restates that level difference instead of measuring
        discrimination."""
        lenient = metrics.roc_auc([3, 3, 2, 2], [2, 2, 1, 1])
        harsh = metrics.roc_auc([2, 2, 1, 1], [1, 1, 0, 0])
        assert lenient == harsh

    def test_an_empty_side_is_not_a_number_rather_than_a_confident_half(self) -> None:
        assert metrics.roc_auc([], [1, 2]) != metrics.roc_auc([1, 2], [1, 2])
        assert metrics.roc_auc([], [1, 2]) != metrics.roc_auc([], [1, 2])  # nan != nan


class TestTheResamplingUnitIsTheRepository:
    @staticmethod
    def _concentrated() -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
        """One repository supplies most positives and is the only one that separates — the
        shape NR-56 actually had, with `graph` at 13 of 31."""
        pos: list[tuple[str, float]] = [("graph", 3.0)] * 13
        ctl: list[tuple[str, float]] = [("graph", 0.0)] * 13
        for repo in ("peft", "rag", "rl", "diffusion"):
            pos += [(repo, 1.0)] * 2
            ctl += [(repo, 1.0)] * 2
        return pos, ctl

    def test_it_reports_the_concentration_rather_than_hiding_it(self) -> None:
        pos, ctl = self._concentrated()
        out = jva.cluster_bootstrap_auc(pos, ctl, iters=800)
        assert out["n_clusters"] == 5
        assert out["largest_cluster_share"] == pytest.approx(13 / 21, abs=0.01)

    def test_concentration_widens_the_interval_and_the_design_effect_shows_it(self) -> None:
        """The defect, as a measurement. If one repository carries the signal, dropping it
        from a resample must move the estimate — and a paper-level bootstrap almost never
        drops it, because it resamples the 13 papers rather than the repository."""
        pos, ctl = self._concentrated()
        out = jva.cluster_bootstrap_auc(pos, ctl, iters=2000)
        assert out["design_effect"] > 1.5
        assert out["ci95"][1] - out["ci95"][0] > 0.2

    def test_a_single_cluster_refuses_instead_of_inventing_an_interval(self) -> None:
        """With one repository there is nothing to resample, and a bootstrap over one
        cluster reports zero width — a confident interval built from no evidence."""
        out = jva.cluster_bootstrap_auc([("graph", 3.0)] * 8, [("graph", 0.0)] * 8)
        assert "_refused" in out
        assert "ci95" not in out
        assert out["auc"] == 1.0

    def test_the_point_estimate_is_the_plain_auc(self) -> None:
        pos, ctl = self._concentrated()
        out = jva.cluster_bootstrap_auc(pos, ctl, iters=500)
        assert out["auc"] == pytest.approx(
            metrics.roc_auc([s for _, s in pos], [s for _, s in ctl]), abs=5e-5
        )

    def test_it_is_deterministic_under_a_fixed_seed(self) -> None:
        pos, ctl = self._concentrated()
        a = jva.cluster_bootstrap_auc(pos, ctl, iters=500, seed=7)
        b = jva.cluster_bootstrap_auc(pos, ctl, iters=500, seed=7)
        assert a == b

    def test_it_says_what_this_n_could_have_detected(self) -> None:
        """So that a CI spanning 0.5 can be read as "no discrimination" or as "not enough
        repositories", instead of the two collapsing into one sentence."""
        pos, ctl = self._concentrated()
        out = jva.cluster_bootstrap_auc(pos, ctl, iters=800)
        assert out["min_detectable_auc_80pct"] > 0.5


class TestTheOneInvariantHasOneImplementation:
    def test_both_analyses_share_the_rank_function(self) -> None:
        """C-9/C-12/C-14: two implementations of one invariant is this project's most
        repeated correction. The thin-docs Spearman and the adoption AUC need the identical
        tie rule, and a disagreement between them would be invisible in both."""
        import thin_docs_detector as det

        assert det.average_ranks is metrics.average_ranks
