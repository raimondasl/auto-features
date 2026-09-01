"""The three-arm scoreboard, and what it must refuse to say. [P27]

`evals/mcp_arm.json` compares RepoRadar alone (A), RepoRadar as the product actually
displays it (A′), Opus 5 alone (B), and — once it is run — Opus 5 with RepoRadar's MCP
server attached (C).

Two properties matter more than any figure in it:

* **It reproduces P26 from an independent code path.** A − B comes out +1.08, CI
  [−0.97, +3.22], which is `evals/opus5_arm.json`'s published number computed by a second
  implementation. A scoreboard that could not reproduce the arm it extends would be a new
  and unexplained result rather than an addition to one.
* **An unrun arm is void, not zero.** C scored as 0 net@2 would read as "the agent
  recommended nothing" — a measurement — where the truth is that nobody has asked.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "evals" / "mcp_arm.json"


@pytest.fixture(scope="module")
def art() -> dict:
    return json.loads(FROZEN.read_text(encoding="utf-8"))


class TestItReproducesTheArmItExtends:
    def test_a_minus_b_matches_the_published_p26_figure(self, art) -> None:
        """+1.08, CI [−0.97, +3.22], 20W/15L over 37 — `evals/opus5_arm.json`'s headline,
        recomputed here from the same two sources by different code."""
        d = art["cohorts"]["all37"]["A_minus_B"]
        assert d["mean"] == pytest.approx(1.08, abs=0.01)
        assert d["ci95"] == [-0.97, 3.22]
        assert (d["wins"], d["losses"]) == (20, 15)
        assert d["excludes_zero"] is False

    def test_the_cohort_levels_match_too(self, art) -> None:
        assert art["cohorts"]["all37"]["A"]["mean_net2"] == pytest.approx(6.27, abs=0.01)
        assert art["cohorts"]["all37"]["B"]["mean_net2"] == pytest.approx(5.19, abs=0.01)
        assert art["cohorts"]["matsci6"]["B"]["mean_net2"] == pytest.approx(8.67, abs=0.01)


class TestTheProductArmIsSeparatedFromThePublishedOne:
    def test_the_already_cited_mute_is_named_and_priced(self, art) -> None:
        """C's tool serves A′, so A′ has to exist as its own column. Comparing C against A
        would charge the augmented arm for papers it was never handed."""
        ap = art["arms"]["A_prime"]
        assert ap["n_muted"] == 11
        assert ap["n_picks"] == 325
        assert set(ap["muted_by_case"]) == {
            "ann",
            "bio-align",
            "graph",
            "mat-chgpot",
            "mat-descriptors",
            "mat-phonon",
            "peft",
            "rag",
        }

    def test_it_costs_arm_a_almost_nothing_and_that_was_not_the_first_guess(self, art) -> None:
        """8 of the 11 muted picks are judged actionable and 3 are not, so the `+1`s and
        the `−2`s nearly cancel: +0.05/case, not the +0.22 that counting only the
        actionable side gives. The estimate was wrong by a factor of four and the
        measurement is what caught it."""
        d = art["cohorts"]["all37"]["A_minus_A_prime"]
        assert d["mean"] == pytest.approx(0.05, abs=0.01)
        assert d["ci95"][0] < 0 < d["ci95"][1]

    def test_the_correction_flips_no_verdict(self, art) -> None:
        """A′ − B is +1.03 against A − B's +1.08. Both intervals still cross zero, so
        nothing that was reported as unresolved becomes resolved."""
        a = art["cohorts"]["all37"]["A_minus_B"]
        ap = art["cohorts"]["all37"]["A_prime_minus_B"]
        assert ap["mean"] == pytest.approx(1.03, abs=0.01)
        assert a["excludes_zero"] is False and ap["excludes_zero"] is False


class TestTheTwoScientificCohortsDisagree:
    """The arm ran on the 12 scientific cases. **Neither cohort separates, and they point
    in opposite directions** — which is the result, not a step toward one."""

    def test_matsci_goes_against_the_augmented_arm(self, art) -> None:
        e = art["cohorts"]["matsci6"]
        assert e["C"]["mean_net2"] == pytest.approx(4.50, abs=0.01)
        assert e["B_on_c_cases"]["mean_net2"] == pytest.approx(8.67, abs=0.01)
        d = e["C_minus_B"]
        assert d["mean"] == pytest.approx(-4.17, abs=0.01)
        assert d["ci95"][0] < 0 < d["ci95"][1]  # past the -1.50 bar, interval still crosses
        assert d["excludes_zero"] is False

    def test_bio_goes_the_other_way(self, art) -> None:
        e = art["cohorts"]["bio6"]
        assert e["C"]["mean_net2"] == pytest.approx(7.17, abs=0.01)
        assert e["C_minus_B"]["mean"] == pytest.approx(1.33, abs=0.01)
        assert e["C_minus_B"]["excludes_zero"] is False

    def test_pooled_they_cancel_and_nothing_is_established(self, art) -> None:
        """5W/5L/2T, sign p = 1.00. The registered rule's answer is `not separated`."""
        d = art["cohorts"]["scientific12"]["C_minus_B"]
        assert d["mean"] == pytest.approx(-1.42, abs=0.01)
        assert d["ci95"][0] < 0 < d["ci95"][1]
        assert (d["wins"], d["losses"], d["ties"]) == (5, 5, 2)
        assert d["sign_test_p"] == pytest.approx(1.0, abs=0.001)

    def test_the_difference_is_volume_at_equal_or_better_precision(self, art) -> None:
        """C is MORE precise and returns a QUARTER fewer papers. net@2 sums over what is
        returned, so at p ~ 0.9 each paper forgone costs ~0.7 -- 2.7 fewer papers per case
        is about -1.9, and the precision gain buys back ~0.4. That is the -1.42."""
        e = art["cohorts"]["scientific12"]
        assert e["C"]["shown_per_case"] == pytest.approx(8.1, abs=0.1)
        assert e["B_on_c_cases"]["shown_per_case"] == pytest.approx(10.8, abs=0.1)
        assert e["C"]["precision"] > e["B_on_c_cases"]["precision"]

    def test_the_arm_never_abstains_and_the_agent_alone_did(self, art) -> None:
        """`bio-mdtraj` is the mechanism in one case: Opus 5 alone returned NOTHING and
        scored 0; with the shortlist it returned 6 papers at 0.83 and scored +3. The
        shortlist rescues an abstention -- and caps a wide answer, which is the same
        behaviour costing 13 points on `mat-chgpot`."""
        e = art["cohorts"]["scientific12"]
        assert e["C"]["abstained_on"] == 0
        assert e["B_on_c_cases"]["abstained_on"] == 1
        assert e["C_minus_B"]["per_case"]["bio-mdtraj"] == 3.0
        assert e["C_minus_B"]["per_case"]["mat-chgpot"] == -13.0


class TestTheUnrunPartIsVoidNotZero:
    def test_the_unrun_cohort_carries_no_c_column_at_all(self, art) -> None:
        """core25 has not been run. It must have no C figures rather than zeros — an arm
        scored 0 reads as "the agent recommended nothing"."""
        assert "C" not in art["cohorts"]["core25"]
        assert "C_minus_B" not in art["cohorts"]["core25"]

    def test_a_partial_sweep_does_not_relabel_the_other_columns(self, art) -> None:
        """The bug this caught: intersecting the case set with a partially-run C shrank
        A, A' and B to C's 6 matsci cases and printed matsci's levels under `all37`."""
        assert art["cohorts"]["all37"]["n_cases"] == 37
        assert art["cohorts"]["all37"]["n_cases_c"] == 12
        assert art["cohorts"]["all37"]["c_complete"] is False
        assert art["cohorts"]["all37"]["A"]["mean_net2"] == pytest.approx(6.27, abs=0.01)

    def test_c_figures_are_compared_against_b_on_the_same_cases(self, art) -> None:
        """`B_on_c_cases`, not `B`. Comparing a 12-case C against a 37-case B is the same
        mistake wearing different clothes."""
        e = art["cohorts"]["all37"]
        assert e["B"]["mean_net2"] != e["B_on_c_cases"]["mean_net2"]
        assert len(e["cases_c"]) == e["n_cases_c"]

    def test_the_prereg_is_cited_from_the_artifact(self, art) -> None:
        """A decision rule a reader has to go looking for is one that can be quietly
        replaced. The artifact names the file it was registered in."""
        assert "PREREG-mcp-arm.md" in art["_comment"]
        assert (ROOT / "evals" / "PREREG-mcp-arm.md").is_file()


class TestTheKillConditionDidNotFire:
    def test_every_row_used_the_server(self, art) -> None:
        """ "The tool did not help" and "the agent never found the tool" are opposite
        findings, and only the call log separates them. **Zero rows of 12 made no call**,
        on a prompt that never mentions the server — so the −1.42 is a statement about
        RepoRadar's shortlist, not about whether an agent can find it."""
        c = art["arms"]["C"]
        assert c["rows_with_zero_mcp_calls"] == 0
        assert c["treatment_present"] is True
        assert c["mcp_calls_total"] == 87

    def test_the_agent_reached_past_the_shortlist_more_than_at_it(self, art) -> None:
        """48 of 87 calls are `search_papers` against a store holding only that case's
        digest picks (3–19 papers), where the product's store holds everything RepoRadar
        ever fetched. **Arm C's search tool is materially narrower than the product's** —
        the price of seeding exactly arm A's output, and the reason C is a floor on what a
        fully-populated store would give. Recorded so the next arm is obvious."""
        calls = art["arms"]["C"]["mcp_calls"]
        assert calls["search_papers"] == 48
        assert calls["get_ranked_papers"] == 21
