"""The probe's write must not be destructive, and its statistics must skip absent rows.

Two defects found by running `evals/turn_budget_probe.py`, both of a class this project has
paid for before, both pinned here so the next edit cannot reintroduce them.

**The write.** A `--case` subset run overwrote the full-set artifact, because the patch that
added `--out` reached the `--report` read path and left `OUT.write_text` alone. That is
lesson 4 of the methodology section — "partial runs overwriting whole-set artifacts", three
scripts before merge-by-key became the standard write pattern — reproduced in a brand-new
script. The repair is not "remember to pass --out": it is that `merge_into` makes the write
non-destructive by construction.

**The statistics.** For a case with no cached baseline, `J(cached, control)` computes to 0.0
— an empty stored set against a non-empty fresh one — which reads as *total disagreement*
when it is an *absent measurement*. Merging four cacheless cases in dragged the headline
"what re-running alone costs" from 0.41 to 0.29 and flipped the verdict line. Void scored as
null, inside the statistic written to price that exact class of error, so the guard belongs
in a test rather than in a comment.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))

import turn_budget_probe as probe  # noqa: E402

ARTIFACT = ROOT / "evals" / "turn_budget_probe.json"


def _row(case: str, *, cached_status: str = "ok", cached=(), control=(), treat=()) -> dict:
    return {
        "case": case,
        "cohort": "scisoft" if case.startswith(("bio-", "mat-")) else "benchmark25",
        "cached": {"status": cached_status, "ids": list(cached), "num_turns": None},
        "control": {"status": "ok", "ids": list(control), "num_turns": 12},
        "treat": {"status": "ok", "ids": list(treat), "num_turns": 30},
        "j_cached_control": probe._jaccard(set(cached), set(control)),
        "j_control_treat": probe._jaccard(set(control), set(treat)),
    }


class TestTheWriteIsNotDestructive:
    def test_a_subset_run_keeps_the_cases_it_did_not_run(self, tmp_path):
        path = tmp_path / "probe.json"
        full = {"control_turns": 12, "cases": [_row("rag"), _row("linter"), _row("http")]}
        path.write_text(json.dumps(full), encoding="utf-8")

        subset = {"control_turns": 12, "cases": [_row("bio-scvi")]}
        merged = probe.merge_into(path, subset)

        assert [r["case"] for r in merged["cases"]] == ["bio-scvi", "http", "linter", "rag"]

    def test_a_rerun_replaces_that_case_rather_than_duplicating_it(self, tmp_path):
        path = tmp_path / "probe.json"
        path.write_text(json.dumps({"cases": [_row("rag", control=["old"])]}), encoding="utf-8")

        merged = probe.merge_into(path, {"cases": [_row("rag", control=["new"])]})

        assert len(merged["cases"]) == 1
        assert merged["cases"][0]["control"]["ids"] == ["new"]

    def test_a_first_run_needs_no_existing_artifact(self, tmp_path):
        fresh = {"cases": [_row("rag")]}
        assert probe.merge_into(tmp_path / "absent.json", fresh) == fresh

    def test_main_actually_routes_its_write_through_the_merge(self):
        """A correct helper nothing calls is C-9b, and it is how this bug happened.

        The original defect was not a wrong `merge_into` — there wasn't one. It was a write
        that bypassed the safe path entirely, which a test of the helper alone cannot see.
        """
        import inspect

        src = inspect.getsource(probe.main)
        assert "merge_into(out_path" in src, (
            "the probe's write no longer goes through merge_into — a --case subset run will "
            "overwrite the full-set artifact again"
        )
        assert "OUT.write_text" not in src, "writing to OUT directly ignores --out"


class TestAbsentIsNotDisagreement:
    def test_a_case_with_no_cache_scores_no_similarity(self):
        """0.0 would say "the answers differ"; there is no stored answer to differ from."""
        row = _row("mat-mlip", cached_status="missing", cached=(), control=["a", "b"])
        assert row["j_cached_control"] == 0.0, "this is the trap the report must not fall into"
        assert row["cached"]["status"] != "ok", "and this is the field that reveals it"

    def test_two_abstentions_are_not_scored_as_agreement(self):
        assert probe._jaccard(set(), set()) is None

    def test_the_report_excludes_cacheless_rows_from_the_noise_figure(self, capsys):
        """The regression that mattered: 0.41 -> 0.29 by adding rows with nothing to compare."""
        rows = [
            _row("rag", cached=["a", "b"], control=["a", "b"], treat=["a"]),
            _row("linter", cached=["c"], control=["c"], treat=["c"]),
            _row("mat-mlip", cached_status="missing", cached=(), control=["z"], treat=["z"]),
        ]
        probe.report({"control_turns": 12, "treatment_turns": 30, "cases": rows})
        out = capsys.readouterr().out
        assert "mean J(cached, control) = 1.00" in out, out
        assert "no successful cache (1)" in out


@pytest.mark.skipif(not ARTIFACT.is_file(), reason="no committed probe artifact")
class TestTheCommittedArtifact:
    @pytest.fixture(scope="class")
    def data(self) -> dict:
        return json.loads(ARTIFACT.read_text(encoding="utf-8"))

    def test_both_arms_are_present(self, data):
        """The six succeeding cases AND the four rescue cases, after the overwrite."""
        cases = {r["case"] for r in data["cases"]}
        assert {"rag", "linter", "http", "mat-descriptors", "bio-align", "bio-singlecell"} <= cases
        assert {"bio-scvi", "mat-mlip", "mat-toolkit", "mat-phonon"} <= cases

    def test_the_two_rescued_cases_succeeded_at_the_unchanged_cap(self, data):
        """C-28: their P14 failures were draws, not properties of the repositories."""
        by_case = {r["case"]: r for r in data["cases"]}
        for case in ("mat-mlip", "mat-phonon"):
            assert by_case[case]["control"]["status"] == "ok"
            assert by_case[case]["cached"]["status"] != "ok", "no cache: P14 could not run it"
