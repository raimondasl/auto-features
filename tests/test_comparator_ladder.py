"""The headline against every comparator we have, and both source arms. [P26]

The paper reports one number against one rival: **+3.88** against Claude Opus 4.8 at the v1
prompt and a 12-turn cap. That number is correct and is not touched here. What this pins is
how far it moves when the rival changes:

| RepoRadar | vs Opus 4.8 v1@12 | vs Opus 4.8 v2@30 | vs Opus 5 v2@30 |
|---|---|---|---|
| published headline +5.72 | **+3.88** p=0.0007 | +3.56 p=0.019 | **+1.52** 12w/13l p=1.00 |
| arXiv control +5.84 | +4.00 p<0.001 | +3.68 p=0.027 | +1.64 14w/11l p=0.69 |
| arXiv+EPMC +6.16 | +4.32 p<0.001 | +4.00 p=0.007 | +1.96 14w/10l p=0.54 |

Two things a single number cannot say. **The margin survives a harness upgrade and does not
survive a model upgrade** -- and against Opus 5 the sign test is p = 1.00 on the published
arm, because the win count is 12 to 13. The mean is ahead; the case count is not.

**The comparator was not under-resourced.** Holding the model and giving it the v2 prompt and
30 turns buys it +0.32. Swapping Opus 4.8 for Opus 5 buys +2.04. That is the answer to the
obvious reviewer question, and it exonerates the published figure as a fair instantiation of
the model it names.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "evals" / "comparator_ladder.json"
RESTATED = ROOT / "evals" / "restated_runs.json"


@pytest.fixture(scope="module")
def artifact() -> dict:
    return json.loads(FROZEN.read_text(encoding="utf-8"))


class TestItReproducesWhatIsPublished:
    def test_the_published_cell_is_the_published_figure(self, artifact):
        """The load-bearing assertion. If this cell stops matching `restated_runs.json`, the
        ladder is computing net@2 or its interval differently from the paper and every other
        cell in the table is suspect -- so the check is against the OTHER artifact, derived
        by other code, rather than against a constant typed here."""
        r = artifact["reproduces_published"]
        assert r["matches"], r
        assert r["got_paired"] == 3.88
        assert r["got_reporadar"] == 5.72
        assert r["got_baseline"] == 1.84

    def test_it_agrees_with_the_c25_restatement_case_by_case(self, artifact):
        """`restated_runs.json` is the authority on what the comparator forfeited. This
        artifact restores those picks rather than hard-coding +1.84, so the two must agree."""
        restated = json.loads(RESTATED.read_text(encoding="utf-8"))
        head = next(r for r in restated["runs"] if r.get("is_headline"))
        assert head["corrected"]["baseline"] == artifact["comparators"]["opus48_v1_12"]["mean_net2"]
        assert head["corrected"]["paired"] == 3.88
        assert head["as_measured"]["baseline"] == 1.56, "the uncorrected value, still recorded"

    def test_the_fresh_control_reproduces_the_published_arm(self, artifact):
        """Twelve days apart, +5.72 against +5.84. Inside the noise floor, which is what
        licenses reading the other rows as comparator effects rather than redraws."""
        pub = artifact["reporadar_arms"]["published_headline"]["mean_net2"]
        ctrl = artifact["reporadar_arms"]["arxiv"]["mean_net2"]
        assert pub == 5.72 and ctrl == 5.84
        assert abs(ctrl - pub) < 0.5


class TestTheLadder:
    def test_the_margin_survives_the_harness_and_not_the_model(self, artifact):
        """Every cell against both Opus 4.8 configurations clears p < 0.05; not one cell
        against Opus 5 does. The result is not fragile to how the rival is prompted -- it is
        fragile to which model the rival is."""
        for arm in artifact["paired"]:
            for comp in ("opus48_v1_12", "opus48_v2_30"):
                assert artifact["paired"][arm][comp]["significant_at_05"], f"{arm}/{comp}"
            assert not artifact["paired"][arm]["opus5_v2_30"]["significant_at_05"], arm

    def test_against_opus5_the_published_arm_is_behind_on_cases_won(self, artifact):
        """+1.52 on the mean and **12 wins to 13 losses** -- a sign test of exactly 1.00. The
        mean and the case count point different ways, which is the shape a few large wins
        produce, and reporting only the mean would hide it."""
        cell = artifact["paired"]["published_headline"]["opus5_v2_30"]
        assert cell["paired_delta"] == 1.52
        assert (cell["wins"], cell["losses"]) == (12, 13)
        assert cell["sign_p"] == 1.0

    def test_every_opus5_interval_crosses_zero(self, artifact):
        for arm, cells in artifact["paired"].items():
            lo, hi = cells["opus5_v2_30"]["ci95"]
            assert lo < 0 < hi, arm

    def test_the_ordering_is_monotone_in_comparator_strength(self, artifact):
        """A consistency check that would catch a mislabelled column: a stronger rival must
        shrink the margin on every arm, not on some."""
        for arm, cells in artifact["paired"].items():
            assert (
                cells["opus48_v1_12"]["paired_delta"]
                > cells["opus48_v2_30"]["paired_delta"]
                > cells["opus5_v2_30"]["paired_delta"]
            ), arm


class TestTheComparatorWasNotUnderResourced:
    def test_the_harness_step_is_small_and_the_model_step_is_not(self, artifact):
        """The reviewer question this answers: "did you cripple your baseline?" No -- the
        prompt and turn budget are worth +0.32 on the same model. +2.04 of the +2.36 is
        Opus 5 being a better model than Opus 4.8."""
        d = artifact["comparator_decomposition"]
        assert d["published"] == 1.84
        assert d["harness_v1_12_to_v2_30"] == 0.32
        assert d["model_opus48_to_opus5"] == 2.04
        assert d["model_opus48_to_opus5"] > 6 * d["harness_v1_12_to_v2_30"]
        assert d["harness_v1_12_to_v2_30"] + d["model_opus48_to_opus5"] == pytest.approx(
            d["total"], abs=0.01
        )

    def test_the_stronger_comparators_answer_far_more(self, artifact):
        """And their precision falls as they do, all three staying above the 2/3 breakeven --
        so the comparator's gain is volume bought at a rate that still pays, the same
        mechanism §8.3 measured on our own digest width."""
        c = artifact["comparators"]
        assert c["opus48_v1_12"]["papers_per_case"] < c["opus48_v2_30"]["papers_per_case"]
        assert c["opus48_v2_30"]["papers_per_case"] < c["opus5_v2_30"]["papers_per_case"]
        assert c["opus48_v1_12"]["precision"] > c["opus5_v2_30"]["precision"] > 2 / 3

    def test_the_published_comparator_abstains_most(self, artifact):
        """Which is the other half of the same story: at 12 turns and the v1 prompt it
        declined on repositories the later configurations answer."""
        c = artifact["comparators"]
        assert c["opus48_v1_12"]["abstentions"] > c["opus5_v2_30"]["abstentions"]


class TestBothSourceArmsAreReported:
    def test_the_shipped_arm_is_identified_as_such(self, artifact):
        """arXiv+EPMC scores higher on every cell and is NOT what ships. Recording which arm
        is the shipped one inside the artifact stops a reader taking the better number as the
        product's."""
        arms = artifact["reporadar_arms"]
        assert arms["arxiv"]["is_shipped_configuration"]
        assert arms["published_headline"]["is_shipped_configuration"]
        assert not arms["arxiv_epmc"]["is_shipped_configuration"]
        assert arms["arxiv_epmc"]["sources"] == ["arxiv", "europepmc"]

    def test_the_unshipped_arm_wins_every_cell_and_changes_no_verdict(self, artifact):
        """+0.32 on every comparator, and not one significance verdict flips with it. The
        source question is real but it is not what decides the headline."""
        for comp in ("opus48_v1_12", "opus48_v2_30", "opus5_v2_30"):
            a = artifact["paired"]["arxiv"][comp]
            e = artifact["paired"]["arxiv_epmc"][comp]
            assert e["paired_delta"] > a["paired_delta"], comp
            assert e["significant_at_05"] == a["significant_at_05"], comp

    def test_the_37_case_figures_are_held_apart(self, artifact):
        """No published denominator covers the scientific twelve, so the 37-case numbers are
        recorded in their own block rather than blended into the benchmark25 table."""
        assert artifact["cohort"] == "benchmark25"
        assert artifact["n_cases"] == 25
        allc = artifact["all37_opus5"]
        assert allc["arxiv"]["n_cases"] == allc["arxiv_epmc"]["n_cases"] == 37
        assert allc["arxiv"]["paired_delta"] == 0.54
        assert allc["arxiv_epmc"]["paired_delta"] == 1.08
        assert not allc["arxiv"]["significant_at_05"]
        assert not allc["arxiv_epmc"]["significant_at_05"]

    def test_the_shipped_arm_is_a_dead_heat_over_37_cases(self, artifact):
        """18 wins, 18 losses, one tie. The single most compact statement of where the
        comparator question stands: on the configuration that ships, over every repository
        measured, against a current frontier model."""
        a = artifact["all37_opus5"]["arxiv"]
        assert (a["wins"], a["losses"], a["ties"]) == (18, 18, 1)
        assert a["sign_p"] == 1.0
