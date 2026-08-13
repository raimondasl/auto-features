"""Tests for the source A/B report.

Two properties carry the report, and both have cost this project a finding when absent:

* **Arm validity.** A treatment that returned nothing new is VOID, not neutral. Finding 3
  measured an S2 arm that contributed zero papers and explained the resulting noise with a
  mechanism that required papers.
* **The negative controls are read from the benchmark, not hardcoded.** They are reported
  apart from the mean because `gold_n: 0` encodes "no gold *arXiv* papers" — a coverage
  claim — and Tier B never sees the label, so the judge's scores on exactly the papers the
  treatment added are what decides whether those repos should abstain at all.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))

from source_ab_report import (  # noqa: E402
    added_papers,
    check_arms,
    negative_controls,
    score_histogram,
    source_marked_ids,
)


def _case(ids_and_scores: list[tuple[str, int | None]], net: float = 0.0) -> dict[str, Any]:
    return {
        "reporadar_toppicks": {"net_value@2": net},
        "returned": {
            "reporadar_top10": [
                {"arxiv_id": i, "judge_score": s, "title": f"paper {i}"} for i, s in ids_and_scores
            ]
        },
    }


class TestAddedPapers:
    def test_identifies_only_what_the_control_never_returned(self) -> None:
        control = _case([("a", 2), ("b", 1)])
        treat = _case([("a", 2), ("c", 3), ("d", 0)])
        assert {p["arxiv_id"] for p in added_papers(control, treat)} == {"c", "d"}

    def test_identical_arms_add_nothing(self) -> None:
        """The VOID shape: a treatment whose output matches the control did not run."""
        control = _case([("a", 2), ("b", 1)])
        assert added_papers(control, _case([("a", 2), ("b", 1)])) == []

    def test_a_control_that_returned_nothing_makes_everything_new(self) -> None:
        """Abstention in the control is the interesting case for a negative control."""
        assert len(added_papers(_case([]), _case([("a", 0), ("b", 1)]))) == 2

    def test_records_without_an_id_are_skipped(self) -> None:
        treat = {"returned": {"reporadar_top10": [{"judge_score": 3, "title": "no id"}]}}
        assert added_papers(_case([]), treat) == []


class TestScoreHistogram:
    def test_counts_by_judge_score(self) -> None:
        papers = [{"judge_score": 2}, {"judge_score": 2}, {"judge_score": 0}]
        assert score_histogram(papers) == {2: 2, 0: 1}

    def test_unscored_papers_are_counted_as_none_not_zero(self) -> None:
        """A paper the judge never scored is not a paper the judge rejected.

        Collapsing them would turn "we did not ask" into evidence against the source.
        """
        hist = score_histogram([{"judge_score": None}, {"judge_score": 0}])
        assert hist[None] == 1
        assert hist[0] == 1


class TestArmsAreVerifiedFromContent:
    """Run files do not record their `--sources`, so a command-line label is a claim.

    The papers are not a claim. A source enabled only in the treatment stamps its own id
    prefix on everything it contributes, so a swap is detectable even though nothing was
    recorded — a stronger guard than `bigram_report`'s label check, which can only catch a
    mismatch someone already wrote down.
    """

    def test_a_swap_is_refused(self) -> None:
        control = {"rag": _case([("ss:abc", 2)])}
        treat = {"rag": _case([("2401.1", 2)])}
        with pytest.raises(SystemExit, match="arms are"):
            check_arms(control, treat, "ss:")

    def test_the_expected_orientation_passes(self) -> None:
        control = {"rag": _case([("2401.1", 2)])}
        treat = {"rag": _case([("2401.1", 2), ("ss:abc", 3)])}
        check_arms(control, treat, "ss:")

    def test_a_treatment_with_no_marked_papers_warns_but_continues(self, capsys: Any) -> None:
        """Not fatal: S2 keeps a paper's arXiv id when it has one, so a real contribution
        can arrive carrying no `ss:` prefix at all. The added-paper count decides."""
        control = {"rag": _case([("2401.1", 2)])}
        treat = {"rag": _case([("2401.9", 2)])}
        check_arms(control, treat, "ss:")
        assert "no 'ss:' papers" in capsys.readouterr().out

    def test_counts_only_the_given_prefix(self) -> None:
        arm = {"a": _case([("ss:1", 2), ("iacr:2026/1", 3), ("2401.1", 1)])}
        assert source_marked_ids(arm, "ss:") == 1
        assert source_marked_ids(arm, "iacr:") == 1


class TestNegativeControls:
    def test_read_from_the_benchmark(self) -> None:
        """Hardcoding them means a control added later is silently pooled into the mean."""
        controls = negative_controls()
        assert {"webdev", "cli", "http"} <= controls

    def test_does_not_include_ordinary_cases(self) -> None:
        controls = negative_controls()
        for case in ("rag", "db", "speech", "linter"):
            assert case not in controls, f"{case} is not marked negative_control in benchmark.yaml"


class TestReportRefusesBadInputs:
    def test_mismatched_case_sets_are_refused(self, tmp_path: Path, capsys: Any) -> None:
        """Two arms over different repos do not have a paired delta."""
        import json
        import subprocess

        a = tmp_path / "a.json"
        b = tmp_path / "b.json"
        a.write_text(json.dumps([{"case": "rag", **_case([("x", 2)])}]), encoding="utf-8")
        b.write_text(json.dumps([{"case": "db", **_case([("x", 2)])}]), encoding="utf-8")
        script = Path(__file__).resolve().parents[1] / "evals" / "source_ab_report.py"
        proc = subprocess.run(
            [sys.executable, str(script), f"arxiv={a}", f"+s2={b}"],
            capture_output=True,
            text=True,
        )
        assert proc.returncode != 0
        assert "different cases" in (proc.stdout + proc.stderr)


class TestVerdictWording:
    """Two questions the report must not conflate.

    The MRE asks whether an effect of a given SIZE is detectable in principle; the
    interval asks whether THIS draw showed one. The first version conflated them and
    printed "RESOLVED" for a mean past the floor whose CI still spanned zero — and its
    containment check, written `(lo > 0) == (hi > 0)`, called `[-2.14, +0.00]` an interval
    that excludes zero, because neither bound is positive. That is a sign-agreement test.
    """

    @staticmethod
    def excludes_zero(lo: float, hi: float) -> bool:
        return lo > 0 or hi < 0

    @pytest.mark.parametrize(
        "lo,hi,expected",
        [
            (-2.14, 0.00, False),  # the real S2 interval: bound AT zero contains it
            (-2.00, -0.14, True),
            (0.10, 1.20, True),
            (-1.00, 1.00, False),
            (0.00, 1.00, False),  # lower bound at zero, likewise
        ],
    )
    def test_a_bound_at_zero_does_not_exclude_zero(self, lo: float, hi: float, expected: bool):
        assert self.excludes_zero(lo, hi) is expected


@pytest.mark.parametrize("scores,expected_actionable", [([2, 3], 2), ([0, 1], 0), ([0, 2, 3], 2)])
def test_actionable_split_matches_the_gate_threshold(
    scores: list[int], expected_actionable: int
) -> None:
    """2 is the actionable floor everywhere in this project; net@2 is named for it."""
    assert sum(1 for s in scores if s >= 2) == expected_actionable
