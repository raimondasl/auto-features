"""The gate is near-deterministic now, and the 37/37 prediction was wrong. [NR-55]

Verification of the `temperature=0` change, run in NR-54's design so the numbers are directly
comparable: two fresh runs of the shipped config against the **same frozen pool**.

| | pre-fix (NR-54) | post-fix |
|---|---|---|
| byte-identical cases | 10 / 37 | **31 / 37** |
| per-case net@2 sd | **1.44** | **0.37** |
| median digest Jaccard | 0.857 | **1.000** |

**The prediction stated before the runs finished was 37/37. It was 31/37.** Two of the six
remaining differences are ordering only (`columnar`, `rag` — same papers, same scores), which
net@2 cannot see. Four differ in composition: `db` and `numerics` swap one paper at identical
net@2, `thin-kv` and `thin-lang` each drop one. So **33 of 37 are deterministic in everything
net@2 reads**, at a **3.9× reduction** in per-case sd.

**The residual is most likely below the API.** Greedy decoding fixes the sampling rule; it does
not make a served model bit-reproducible, since batching and floating-point non-associativity
can move logits between requests. That is offered as the likeliest reading of a four-case
residual, **not as a finding this probe tests** — the alternative worth checking is a second
stochastic element downstream of the gate.

**NR-54's projected dividend is essentially fully realised.** It computed that a perfectly
deterministic gate would take the paired half-width from 0.78 to 0.63. Folding the measured
0.37 residual back in gives **0.64** — a gap to ideal of 0.01 net@2.

And it changes nothing about the plan, exactly as NR-54 predicted: ladder rungs run +0.20 to
+0.45 and stay under 0.64, so the bundle-only rule stands.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "evals" / "temperature_zero_check.json"


@pytest.fixture(scope="module")
def artifact() -> dict:
    return json.loads(FROZEN.read_text(encoding="utf-8"))


class TestThePredictionIsRecordedAsWrong:
    def test_it_was_37_of_37_and_it_was_not(self, artifact):
        """Stated in the PR before the runs finished. Kept because a probe whose prediction is
        quietly dropped teaches nothing about how good the predictions are."""
        p = artifact["prediction_was"]
        assert p["byte_identical"] == 37
        assert p["held"] is False
        assert artifact["byte_identical"] == 31
        assert artifact["verdict"]["prediction_held"] is False

    def test_the_miss_is_small_and_characterised(self, artifact):
        """Six cases differ; two only in ordering, which net@2 cannot see. Being wrong by four
        composition changes out of 37 is a different thing from being wrong about the fix."""
        assert sorted(artifact["order_only"]) == ["columnar", "rag"]
        assert sorted(artifact["composition_differs"]) == ["db", "numerics", "thin-kv", "thin-lang"]
        assert artifact["net2_relevant_determinism"] == 33


class TestTheFixWorked:
    def test_variance_fell_by_almost_four_times(self, artifact):
        v = artifact["vs_pre_fix"]
        assert v["pre_sd"] == 1.44
        assert v["post_sd"] == pytest.approx(0.37, abs=0.01)
        assert v["reduction_factor"] >= 3.5

    def test_byte_identical_cases_tripled(self, artifact):
        assert artifact["vs_pre_fix"]["pre_byte_identical"] == 10
        assert artifact["byte_identical"] > 3 * artifact["vs_pre_fix"]["pre_byte_identical"]

    def test_the_digest_is_now_typically_the_same_set(self, artifact):
        """Median Jaccard 1.000 against 0.857 before: the typical case now returns exactly the
        same papers, where before roughly one in seven moved."""
        assert artifact["median_digest_jaccard"] == 1.0
        assert artifact["verdict"]["gate_is_near_deterministic"] is True

    def test_the_two_runs_agree_on_the_headline(self, artifact):
        a, b = artifact["mean_net2"]
        assert abs(a - b) <= 0.05


class TestTheDividendAndItsLimits:
    def test_the_projection_is_essentially_realised(self, artifact):
        """0.78 -> 0.64 achieved against 0.63 projected for a perfect gate. The residual costs
        0.01 net@2, so nothing is left on the table by not chasing it."""
        d = artifact["dividend"]
        assert d["half_width_before"] == 0.78
        assert d["nr54_projected_ideal"] == 0.63
        assert d["half_width_achieved"] == pytest.approx(0.64, abs=0.01)
        assert abs(d["gap_to_ideal"]) <= 0.05
        assert artifact["verdict"]["dividend_essentially_realised"] is True

    def test_the_residual_explanation_is_labelled_as_a_hypothesis(self, artifact):
        """It is the likeliest reading of four cases, not something measured here. Saying so is
        the difference between a note and an unearned finding."""
        c = artifact["residual"]["_comment"]
        assert "NOT as a finding" in c
        assert "alternative worth checking" in c

    def test_the_plan_is_unchanged(self, artifact):
        """NR-54 said this would be hygiene rather than a lead, and it is. Rungs at +0.20 to
        +0.45 stay under 0.64."""
        assert artifact["dividend"]["ladder_rungs_still_unresolvable"] is True
        assert artifact["verdict"]["changes_the_plan"] is False
