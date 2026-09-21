"""Pin C-37's shipped-band re-measurement of NR-64. The legacy/scientific split these numbers
show is one scorer-judge reading, gpt-4o-mini against GPT-5.5, not a property of the cohort
(C-38).

`evals/finescale_current_gate.json` is the tracked artifact behind the NR-64 entry in
RESULTS.md. These tests keep the file and the entry from drifting apart, in the direction
that matters: the numbers quoted in prose are recomputed here from the per-paper rows
rather than read back out of the summary the same script wrote.

C-38 withdraws the split as a property of the cohort; these tests pin the numbers, not that
reading. Overall band AUC drops from Testbed A's 0.841 to 0.726, and the drop concentrates in
the scientific cohort: 0.785 on the 25 legacy ML/CS cases against 0.587 on the 12 biology and
materials-science ones.

These are the HAIKU numbers, the shipped gate. C-37 records that the first version of this
experiment scored a Luna-gated band by mistake, so the gate assertion below is not decoration:
it is the guard against repeating it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "evals" / "finescale_current_gate.json"

SCIENTIFIC = {
    "bio-align",
    "bio-kmer",
    "bio-mdsim",
    "bio-mdtraj",
    "bio-scvi",
    "bio-singlecell",
    "mat-chgpot",
    "mat-descriptors",
    "mat-featurize",
    "mat-mlip",
    "mat-phonon",
    "mat-toolkit",
}


@pytest.fixture(scope="module")
def artifact() -> dict:
    # Asserted rather than skipped. The artifact is tracked, so its absence is a broken
    # repository, and a skip here would be this project's own void-not-null failure: the
    # suite would stay green while the entry it guards went unchecked.
    assert ARTIFACT.is_file(), f"{ARTIFACT} is tracked and must be present"
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def _auc(rows: list[dict]) -> float:
    """Recompute with the same estimator the experiment used, not a second implementation."""
    sys.path.insert(0, str(ROOT / "evals"))
    import band_testbeds as tb

    scored = [r for r in rows if r.get("exp09") is not None]
    return tb.auc([r["exp09"] for r in scored], [r["judge"] >= tb.ACTIONABLE for r in scored])


class TestTheArtifactIsComplete:
    def test_every_band_paper_scored(self, artifact: dict) -> None:
        s = artifact["summary"]
        assert s["band_papers"] == 328
        assert s["scored"] == 328, "an unscored paper is a void row, not a null one"
        assert s["unscored"] == 0

    def test_rows_match_the_declared_count(self, artifact: dict) -> None:
        assert len(artifact["rows"]) == artifact["summary"]["band_papers"]


class TestTheQuotedNumbersAreRecomputable:
    """RESULTS.md quotes these. Recompute them from the rows so prose cannot drift."""

    def test_overall(self, artifact: dict) -> None:
        assert _auc(artifact["rows"]) == pytest.approx(0.726, abs=0.001)

    def test_legacy_cohort_replicates(self, artifact: dict) -> None:
        rows = [r for r in artifact["rows"] if r["case"] not in SCIENTIFIC]
        assert len(rows) == 224
        assert _auc(rows) == pytest.approx(0.785, abs=0.001)

    def test_scientific_cohort(self, artifact: dict) -> None:
        rows = [r for r in artifact["rows"] if r["case"] in SCIENTIFIC]
        assert len(rows) == 104
        assert _auc(rows) == pytest.approx(0.587, abs=0.001)

    def test_the_split_is_not_a_class_balance_artifact(self, artifact: dict) -> None:
        """Both cohorts sit near a 0.83 base rate, so the gap is not imbalance."""
        for key in ("legacy", "scientific"):
            assert artifact["summary"][key]["base_rate"] == pytest.approx(0.83, abs=0.01)


class TestTheComparisonIsHonest:
    def test_the_testbed_a_reference_is_carried(self, artifact: dict) -> None:
        """The 0.841 must travel with the number that supersedes it, or the drop is invisible."""
        ref = artifact["summary"]["reference"]
        assert ref["testbed_a_auc"] == pytest.approx(0.8412, abs=0.0001)
        assert "different run" in ref["note"]

    def test_the_scorer_is_the_original(self, artifact: dict) -> None:
        """A different model would make the comparison to 0.841 meaningless."""
        assert artifact["summary"]["model"] == "gpt-4o-mini"

    def test_the_band_came_from_the_shipped_gate(self, artifact: dict) -> None:
        """C-37. The first run of this experiment scored a Luna-gated band while believing it
        was Haiku, because it read pool_config instead of ranking_config. Both files say
        claude-haiku-4-5 in pool_config, so the field that was checked could not tell the arms
        apart. This asserts the gate the artifact actually records."""
        gate = artifact["summary"]["gate"]
        assert gate["provider"] == "claude", f"band came from a {gate['provider']} gate"
        assert gate["model"] == "claude-haiku-4-5", f"band came from {gate['model']}"
