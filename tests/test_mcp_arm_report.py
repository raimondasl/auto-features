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


class TestTheUnrunArmIsVoidNotZero:
    def test_c_says_not_run_rather_than_scoring_nothing(self, art) -> None:
        c = art["arms"]["C"]
        if c.get("status") == "not_run":
            assert "mean_net2" not in c
            assert "C" not in art["cohorts"]["all37"]
            assert "C_minus_B" not in art["cohorts"]["all37"]
            assert "gold_spread.py --tools web+rr" in c["how"]
        else:
            # Once it runs, the comparison the pre-registration names must be present.
            assert "C_minus_B" in art["cohorts"]["all37"]

    def test_the_prereg_is_cited_from_the_artifact(self, art) -> None:
        """A decision rule a reader has to go looking for is one that can be quietly
        replaced. The artifact names the file it was registered in."""
        assert "PREREG-mcp-arm.md" in art["_comment"]
        assert (ROOT / "evals" / "PREREG-mcp-arm.md").is_file()


class TestTheKillConditionCanBeCheckedFromTheArtifact:
    def test_tool_use_is_recorded_when_the_augmented_arm_has_run(self, art) -> None:
        """ "The tool did not help" and "the agent never found the tool" are opposite
        findings, and only the call log separates them."""
        c = art["arms"]["C"]
        if c.get("status") == "not_run":
            pytest.skip("arm C has not been run")
        assert "mcp_calls_total" in c
        assert "rows_with_zero_mcp_calls" in c
        assert c["treatment_present"] == (c["rows_with_zero_mcp_calls"] <= 3)
