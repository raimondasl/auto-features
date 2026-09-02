"""The judges order alike and level differently — so combining them cannot help. [NR-59]

The fine-scale probability map was fitted on GPT-labelled papers, so `finescale_p` estimates
*P(GPT calls this actionable)*, not P(actionable). This ran `calibrate_finescale`'s own
prescribed counterfactual — *"a leave-one-repo-out refit, which never sees the repo it is
scored on"* — on the **judge** dimension instead of the repo one.

**The registered test did not run.** Its blocking reproduction check failed at 0.799 against
0.90, so the 92.9% flip rate is recorded and not read. The diagnosis is that the shipped map
was fitted on a wider population (all gate scores) while this band is its *application*
population, where GPT's base rate is 0.874 — and a logistic fitted to a 0.874 slice is nearly
a constant. Refit and product are not the same operation.

**What the data says without any fitting is the result**: the two judges' AUCs against the
fine-scale score differ by **0.027** while their base rates differ by **0.380**. They agree
about which papers are better and disagree about how many are good. Every threshold in the
system is therefore a bet on a base rate — and no combination of judges measures one, it only
picks one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "evals" / "judge_refit.json"


@pytest.fixture(scope="module")
def art() -> dict:
    return json.loads(FROZEN.read_text(encoding="utf-8"))


class TestTheBlockingCheckWasHonoured:
    def test_the_reproduction_check_failed(self, art) -> None:
        r = art["reproduction"]
        assert r["gpt_refit_agrees_with_shipped"] == pytest.approx(0.799, abs=0.005)
        assert r["passes"] is False
        assert art["pre_registered"]["reproduction_bar"] == 0.90

    def test_the_flip_rate_is_recorded_and_not_read(self, art) -> None:
        """92.9% would have cleared the registered "substantially an artifact" bar of 25% by
        a mile. Moving a blocking check after seeing that is the failure NR-49 documented,
        so the number is kept and its scope is stated beside it."""
        v = art["verdict"]
        assert v["flip_rate"] == pytest.approx(0.929, abs=0.005)
        assert v["flip_rate_is_licensed"] is False
        assert "NOT licensed" in v["_flip_rate_scope"]

    def test_the_diagnosis_is_on_the_artifact_not_left_to_a_reader(self, art) -> None:
        """Why it failed: the map was fitted on a wider population than the band it is
        applied to, and a logistic fitted to an 0.874 slice is nearly a constant."""
        scope = art["verdict"]["_flip_rate_scope"]
        assert "WIDER population" in scope
        assert "0.874" in scope
        # The refit's slope collapses against the shipped one — the fingerprint of it.
        c = art["coefficients"]
        assert c["gpt_refit_all"]["slope"] < 0.6 < c["shipped"]["slope"]

    def test_the_claim_it_was_built_to_test_is_not_claimed(self, art) -> None:
        assert "NOT established here" in art["verdict"]["_flip_rate_scope"]


class TestOrderingAgreesAndLevelDoesNot:
    """The part that needs no model fitting, and therefore no reproduction check."""

    def test_the_two_judges_rank_the_band_alike(self, art) -> None:
        o = art["ordering_vs_level"]
        assert o["auc_finescale_vs_gpt"] == pytest.approx(0.729, abs=0.005)
        assert o["auc_finescale_vs_sonnet"] == pytest.approx(0.702, abs=0.005)
        assert o["auc_difference"] < 0.05

    def test_and_disagree_about_how_many_are_good_by_a_factor_of_nearly_two(self, art) -> None:
        o = art["ordering_vs_level"]
        assert o["base_rate_gpt"] == pytest.approx(0.874, abs=0.005)
        assert o["base_rate_sonnet"] == pytest.approx(0.494, abs=0.005)
        assert o["base_rate_difference"] > 10 * o["auc_difference"]

    def test_both_sides_of_the_threshold_are_present(self, art) -> None:
        """244 shown and 80 withheld. The range restriction that inverts a shown-only panel
        (NR-58, C-36) is structurally absent, which is why this population can carry the
        claim and NR-42's could not."""
        p = art["population"]
        assert p["n"] == 324
        assert p["n_shown_recorded"] == 244
        assert p["n_withheld_recorded"] == 80


class TestTheProductConsequence:
    def test_the_finescale_stage_has_the_opposite_sign_under_each_judge(self, art) -> None:
        """The decomposition arriving as a shipped consequence. The stage's job is
        abstention; how much abstention is worth is a function of the base rate; the base
        rate is the one thing the judges disagree about."""
        sv = art["stage_value_by_judge"]
        assert sv["sci"]["GPT-5.5"]["per_case"] < 0 < sv["sci"]["Sonnet"]["per_case"]
        assert sv["legacy"]["GPT-5.5"]["per_case"] < 0 < sv["legacy"]["Sonnet"]["per_case"]
        assert sv["sci"]["Sonnet"]["per_case"] == pytest.approx(3.75, abs=0.01)

    def test_it_says_why_a_third_judge_does_not_fix_it(self, art) -> None:
        """Consensus, majority-of-three and a tiebreaker all *pick* a base rate by
        construction; none *measures* one. Adoption is the only channel here that estimates
        an actionable rate with no model in the loop."""
        w = art["verdict"]["what_is_established"]
        assert "combining them picks a level by construction" in w
        assert "Adoption is the only channel" in w
