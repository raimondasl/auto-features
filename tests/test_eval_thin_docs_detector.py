"""The thin-docs detector, and the provenance it depends on.

The detector asks whether documentation corpus size predicts that the system is about to
fail. It does not (NR-37): r = +0.14 across 25 repositories, and the thinnest quintile
scores *better* than the rest. What these tests pin is not that conclusion — a number in
a results file cannot be unit-tested — but the two things that would let it be quietly
wrong later.

* **The corpus must be the profiler's own.** A detector reading a different corpus from
  the profiler describes a repository nobody profiles. That is the C-9/C-12/C-14 shape,
  which this project has corrected three times.
* **The ablation arms must stay identified.** The four grid runs do not record which
  documentation budget they were: `rr_ablate_docs` was the last `POOL_FLAG` missing from
  the recorded fields, so on 2026-08-16 they could only be told apart by matching their
  means against a derived summary. That mapping is now code, and this checks it still
  resolves — otherwise the detector silently analyses four indistinguishable files.
"""

from __future__ import annotations

import ast
import json
import statistics as st
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))

RESULTS = ROOT / "evals" / "results"
# The means that identified each arm, from .work/ablation.json's summaries.
EXPECTED_MEANS = {"control": 5.1667, "1500": 3.0000, "300": 3.1667, "0": -0.5000}


def _net_by_case(path: Path) -> dict[str, float]:
    from metrics import net_actionable_value

    out = {}
    for case in json.loads(path.read_text(encoding="utf-8")):
        picks = (case.get("returned") or {}).get("reporadar_toppicks")
        if picks is None:
            continue
        scores = [p["judge_score"] for p in picks if p.get("judge_score") is not None]
        out[case["case"]] = net_actionable_value(scores, 2.0)
    return out


class TestTheCorpusIsTheProfilersOwn:
    def test_it_calls_the_shipped_collector(self) -> None:
        import thin_docs_detector as det

        from reporadar.profiler import _collect_text_corpus

        assert det._collect_text_corpus is _collect_text_corpus

    def test_no_second_corpus_implementation(self) -> None:
        """An AST check rather than a name check: re-deriving the corpus inline is the
        way this would drift, and it would not show up as a different import."""
        src = (ROOT / "evals" / "thin_docs_detector.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        walkers = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Attribute) and n.attr in ("rglob", "glob", "iterdir")
        ]
        assert walkers == [], "the detector walks the tree itself instead of asking the profiler"


class TestTheAblationArmsStayIdentified:
    """`rr_ablate_docs` is recorded from 2026-08-16; these four predate that."""

    @pytest.mark.parametrize("budget", sorted(EXPECTED_MEANS))
    def test_each_named_arm_still_has_the_mean_that_identified_it(self, budget: str) -> None:
        import thin_docs_detector as det

        path = RESULTS / det.ABLATION_ARMS[budget]
        if not path.exists():
            pytest.skip(f"{path.name} not present in this checkout")
        nets = _net_by_case(path)
        assert nets, f"{path.name} carries no scored cases"
        assert st.mean(nets.values()) == pytest.approx(EXPECTED_MEANS[budget], abs=0.01), (
            f"the file mapped to budget {budget} no longer has the mean that identified it — "
            "the mapping is stale and the detector would analyse the wrong arm"
        )

    def test_the_arms_are_four_distinct_files(self) -> None:
        import thin_docs_detector as det

        assert len(set(det.ABLATION_ARMS.values())) == 4

    def test_every_arm_covers_the_same_six_repositories(self) -> None:
        """A grid missing a repo at one budget is not a dose-response curve."""
        import thin_docs_detector as det

        seen = []
        for name in det.ABLATION_ARMS.values():
            path = RESULTS / name
            if not path.exists():
                pytest.skip(f"{name} not present")
            seen.append(frozenset(_net_by_case(path)))
        assert len(set(seen)) == 1, "the four arms do not cover an identical repository set"
        assert len(seen[0]) == 6


class TestTheBudgetIsRecordedGoingForward:
    def test_new_runs_record_the_ablation_budget(self) -> None:
        """The fix for the provenance gap above. Adding an output field changes no past
        number and leaves the pool fingerprint alone — `rr_ablate_docs` was always in
        POOL_FLAGS — so this is safe to assert without re-running anything."""
        import inspect

        import run_judge_eval

        src = inspect.getsource(run_judge_eval.run)
        assert '"ablate_docs": args.rr_ablate_docs,' in src

    def test_the_flag_is_still_in_the_pool_fingerprint(self) -> None:
        """Recording it must not be confused with fingerprinting it; it was always both
        a pool flag and, until now, unrecorded."""
        import run_judge_eval

        assert "rr_ablate_docs" in run_judge_eval.POOL_FLAGS


class TestTheCorrelationTheScriptNeverComputed:
    """NR-37's substantive finding is the correlation, and until now the script did not
    compute it — the r = +0.14 / rho = +0.20 in RESULTS.md were derived by hand from the
    per-case table. The frame's P0.4 asks for the same statistic over 37 cases, so it is
    code now, and these pin the two properties a hand-rolled rank correlation gets wrong.
    """

    def test_ties_get_the_average_rank(self) -> None:
        """net@2 is heavily tied — many cases sit at exactly 0.0. Breaking those ties by
        list position would invent an ordering the data does not contain and bias rho
        toward whatever order the cases happen to be in."""
        import thin_docs_detector as det

        assert det._ranks([5.0, 1.0, 1.0, 1.0, 9.0]) == [4.0, 2.0, 2.0, 2.0, 5.0]
        assert det._ranks([2.0, 2.0]) == [1.5, 1.5]

    def test_a_monotone_relationship_is_plus_one_either_way_round(self) -> None:
        import thin_docs_detector as det

        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert det._spearman(xs, [10.0, 20.0, 30.0, 40.0, 50.0]) == pytest.approx(1.0)
        assert det._spearman(xs, [50.0, 40.0, 30.0, 20.0, 10.0]) == pytest.approx(-1.0)

    def test_spearman_survives_a_monotone_transform_and_pearson_does_not(self) -> None:
        """The reason the frame asks for rho rather than r: corpus size spans four orders
        of magnitude, so a single large repository can carry a Pearson coefficient."""
        import thin_docs_detector as det

        xs = [1.0, 2.0, 3.0, 4.0, 100.0]
        ys = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert det._spearman(xs, ys) == pytest.approx(1.0)
        assert det._pearson(xs, ys) < 0.95

    def test_too_few_points_is_not_a_number_rather_than_a_confident_zero(self) -> None:
        import math

        import thin_docs_detector as det

        assert math.isnan(det._pearson([1.0, 2.0], [1.0, 2.0]))

    def test_a_flat_series_has_no_correlation_rather_than_a_crash(self) -> None:
        import math

        import thin_docs_detector as det

        assert math.isnan(det._pearson([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]))

    def test_the_run_file_is_selectable_so_p04_can_ask_for_37(self) -> None:
        """P0.4 re-runs NR-37 over all 37 cases. The script was pinned to a 25-case file,
        which is why the frame's blank could not be filled by running it."""
        import inspect

        import thin_docs_detector as det

        src = inspect.getsource(det.main)
        assert '"--run"' in src
        assert "default=SHIPPED_RUN" in src
