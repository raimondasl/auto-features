"""The restated runs, pinned [C-25].

`evals/restate_c25.py` recomputes every paired run that read the three damaged `cli` caches
-- four of them, the headline included -- with the picks those caches forfeited. The paper,
the README and `evals/RESULTS.md` all now quote the corrected column, and the run files it
derives from live under a gitignored directory, so without a committed artifact those
documents would cite numbers nothing in the repository can reproduce. That is the failure
`evals/gold_targets.json` was written for, one level up.

Two kinds of check here, deliberately:

* **Always** -- the artifact's own arithmetic, on the committed JSON, needing no caches.
  `baseline + paired == reporadar` in both columns; the correction moves the comparator UP
  and the margin DOWN by the same amount; the forfeited rows sum to that amount; the gain is
  the same in all four draws (a constant defect, not a draw-dependent one); and the single
  draw that stops separating from the baseline is named, because the paper says it does.
  A hand-edited figure fails these on any machine.
* **When the caches are present** -- the live derivation, against the artifact. This is the
  one that catches a re-run baseline or a changed verdict silently moving a published
  number, and it skips (rather than passes) where `evals/cache/` does not exist.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))

FROZEN = ROOT / "evals" / "restated_runs.json"


@pytest.fixture(scope="module")
def artifact() -> dict:
    return json.loads(FROZEN.read_text(encoding="utf-8"))


class TestTheCommittedArtifact:
    """No caches, no run file -- just whether the published numbers cohere."""

    def test_the_measured_column_is_the_published_headline(self, artifact):
        a = artifact["as_measured"]
        assert (a["reporadar"], a["baseline"], a["paired"]) == (5.72, 1.56, 4.16)
        assert a["ci"] == [2.44, 6.0] and a["wins"] == 18 and a["losses"] == 2
        assert (a["baseline_shown"], a["baseline_actionable"]) == (51, 47)
        assert a["baseline_precision"] == 0.922

    def test_the_corrected_column_is_what_the_paper_now_quotes(self, artifact):
        a = artifact["corrected"]
        assert (a["reporadar"], a["baseline"], a["paired"]) == (5.72, 1.84, 3.88)
        assert a["ci"] == [2.24, 5.6]
        assert (a["wins"], a["losses"], a["ties"]) == (17, 2, 6)
        assert a["sign_p"] < 0.01, "the correction must not cost significance silently"
        assert (a["baseline_shown"], a["baseline_actionable"]) == (58, 54)
        assert a["baseline_precision"] == 0.931

    def test_reporadars_own_composition_is_the_published_one(self, artifact):
        assert (artifact["reporadar_shown"], artifact["reporadar_actionable"]) == (212, 189)

    def test_all_four_affected_draws_are_restated(self, artifact):
        """The damage has a date, so the blast radius is exact rather than a judgment."""
        assert len(artifact["runs"]) == 4
        assert sum(1 for r in artifact["runs"] if r["is_headline"]) == 1
        assert artifact["damaged_caches"] == ["compiler", "graph", "storage"]
        assert all(
            {c["case"] for c in r["forfeited"]} == {"compiler", "graph", "storage"}
            for r in artifact["runs"]
        ), "all three caches were damaged at once, so every affected draw forfeits all three"

    def test_the_comparator_gains_the_same_amount_in_every_draw(self, artifact):
        """A constant defect, not a draw-dependent one -- the reassuring half."""
        gains = [
            r["corrected"]["baseline"] - r["as_measured"]["baseline"] for r in artifact["runs"]
        ]
        assert all(0.27 <= g <= 0.31 for g in gains), gains

    def test_the_draw_that_loses_significance_is_recorded(self, artifact):
        """08-10 goes p = 0.0414 -> 0.1153. The paper says so; this is what makes it check."""
        lost = [
            r
            for r in artifact["runs"]
            if r["as_measured"]["sign_p"] < 0.05 <= r["corrected"]["sign_p"]
        ]
        assert len(lost) == 1, "exactly one published draw stops separating under the correction"
        assert lost[0]["run_file"].startswith("judge-gpt-5.5-20260810")
        assert lost[0]["as_measured"]["sign_p"] == 0.041389
        assert lost[0]["corrected"]["sign_p"] == 0.115318

    def test_the_precision_caveat_widens(self, artifact):
        """SS8.7 quotes 0.888 against 0.938; corrected the agent reads 0.945."""
        draw = next(r for r in artifact["runs"] if r["run_file"].endswith("20260814T175835Z.json"))
        assert draw["reporadar_actionable"] / draw["reporadar_shown"] == pytest.approx(
            0.888, abs=5e-4
        )
        assert draw["as_measured"]["baseline_precision"] == 0.938
        assert draw["corrected"]["baseline_precision"] == 0.945

    def test_partial_draws_report_the_cases_their_baseline_failed_on(self, artifact):
        """`thin-lang` failed in three draws and was excluded from their published means."""
        for run in artifact["runs"]:
            assert run["n_cases"] + len(run["baseline_failed"]) == run["n_cases_in_run"]
        assert {r["n_cases"] for r in artifact["runs"]} == {23, 24, 25}

    @pytest.mark.parametrize("column", ["as_measured", "corrected"])
    def test_the_columns_are_internally_consistent(self, artifact, column):
        a = artifact[column]
        assert a["baseline"] + a["paired"] == pytest.approx(a["reporadar"], abs=0.005)
        assert a["wins"] + a["losses"] + a["ties"] == artifact["n_cases"]

    def test_the_correction_is_a_pure_transfer(self, artifact):
        """RepoRadar does not move: net@2 reads a system's OWN returned papers."""
        was, now = artifact["as_measured"], artifact["corrected"]
        assert was["reporadar"] == now["reporadar"]
        gain = now["baseline"] - was["baseline"]
        assert gain == pytest.approx(was["paired"] - now["paired"], abs=0.005)
        assert gain > 0, "recovering forfeited picks can only help the comparator"

    def test_the_forfeited_rows_account_for_the_whole_move(self, artifact):
        rows = artifact["forfeited"]
        assert {r["case"] for r in rows} == {"compiler", "graph", "storage"}
        assert all(r["baseline_was"] == 0.0 for r in rows), "only abstentions can forfeit"
        recovered = sum(r["baseline_now"] - r["baseline_was"] for r in rows)
        moved = (artifact["corrected"]["baseline"] - artifact["as_measured"]["baseline"]) * (
            artifact["n_cases"]
        )
        assert recovered == pytest.approx(moved, abs=0.005)
        assert recovered == 7.0, "seven picks, all judged actionable"

    def test_the_composition_moves_by_the_same_seven(self, artifact):
        """A comparator credited with seven more picks was also SHOWN seven more papers."""
        was, now = artifact["as_measured"], artifact["corrected"]
        assert now["baseline_shown"] - was["baseline_shown"] == 7
        assert now["baseline_actionable"] - was["baseline_actionable"] == 7


class TestTheLiveDerivation:
    """Against the caches, where this machine has them."""

    @pytest.fixture(scope="class")
    def live(self) -> dict:
        import restate_c25

        if not (restate_c25.RESULTS / restate_c25.HEADLINE_RUN).is_file():
            pytest.skip("no local run file (evals/results/ is gitignored)")
        if not any(restate_c25.BASELINE.glob("*.json")):
            pytest.skip("no local baseline cache")
        return restate_c25.build()

    def test_every_restated_run_still_derives(self, artifact, live):
        assert live["runs"] == artifact["runs"]

    def test_only_damaged_caches_are_recovered(self, live):
        """`webdev` answers with an explicit `[]`; it must never appear here."""
        for run in live["runs"]:
            assert "webdev" not in {r["case"] for r in run["forfeited"]}

    def test_pre_damage_abstentions_are_left_alone(self, live):
        """`storage` genuinely abstained on 2026-08-07 and `graph` on 2026-07-12."""
        import restate_c25

        stamps = [restate_c25._run_stamp(Path(r["run_file"]).stem) for r in live["runs"]]
        assert all(s >= live["damage_date"] for s in stamps), stamps
