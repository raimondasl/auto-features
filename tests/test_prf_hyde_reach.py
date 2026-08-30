"""Item 12 stage 1: the second round clears the bar and the bar was wrong. [NR-49]

PRF-HyDE was the last open evidence-led lead. Stage 1 asked the only question a free probe
can: does a feedback-seeded second round of hypotheses **aim better at the same candidate
budget**? Budget-matched by construction, because NR-47 and NR-48 had just spent both
volume levers and an unmatched test would have re-run them under a new name.

| arm | reach | |
|---|---|---|
| round 1 @ 50 | 0.1962 | |
| **round 1 @ 100 — baseline** | **0.2231** | the pre-registered bar |
| **round 1 @ 50 ∪ round 2 @ 50** | **0.2288** | **+0.0057 — clears it** |
| round 1 @ 100 ∪ round 2 @ 100 | 0.2712 | reported, never the criterion |

**The pass is refused, and that is the finding.** +0.0057 is **three witnesses of 520**, at
McNemar p = 0.68 on 13 gained against 10 lost. NR-46 measured a plain hypothesis *redraw* —
same cut, same method, different draw — at **+0.0577**. The effect is one tenth of the noise
floor of the procedure it modifies.

The bar named a threshold and no minimum effect size, so a null cleared it. That is a defect
in the pre-registration, recorded as one rather than quietly repaired, and the reason a
stage-1 gate exists at all is to license spending — this does not.

**Two things the reach number hides, both kept.**

*The losses are mechanical.* All ten sit at round-1 ranks 51–95: papers dropped by halving
the cut, nothing to do with round 2's aim. The gains have median round-1 rank **296**.

*The gains are not noise.* Five witnesses sit past rank 1000 in round 1 — up to **4187** —
and land in round 2's top 100. **NR-47's widest measured cut was 1000**, so width cannot buy
these at any cut this project has run; the mechanism differs from the one NR-47 and NR-48
spent, which added ranks 100–1000 of the *same* query. Of the 13, seven have ever been
judged and **all seven are actionable**; six are void, not null.

That observation is **post hoc** and does not rescue the null. It is recorded as a
hypothesis owing its own pre-registered test.

**And PRF has a structural blind spot in the worst place.** Four cases — `cli`, `http`,
`linter`, `webdev` — produced no round 2, because the shipped arm showed nothing to feed on.
The abstention discipline that makes RepoRadar competitive with Opus 5 is what starves
feedback. Those cases hold **34 witnesses, none reached by any route**. They were skipped,
never imputed: a fresh draw would have been a round-1 redraw wearing PRF's name, at ten times
the effect under test.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "evals" / "prf_hyde_reach.json"


@pytest.fixture(scope="module")
def artifact() -> dict:
    return json.loads(FROZEN.read_text(encoding="utf-8"))


class TestTheComparisonIsBudgetMatched:
    def test_the_criterion_holds_candidate_count_fixed(self, artifact):
        """The design decision the whole probe rests on. Round 1 at 100 against round 1 at 50
        unioned with round 2 at 50 — same slots, different aim. Without this the arm measures
        volume, which NR-47 and NR-48 had already closed."""
        assert artifact["pre_registered"]["criterion"] == ("budget_matched union must beat the bar")
        assert artifact["pre_registered"]["bar"] == 0.2231
        assert artifact["pre_registered"]["written_before_round2_existed"] is True

    def test_the_unequal_arm_is_reported_and_is_not_the_criterion(self, artifact):
        """It is the larger number (0.2712) and it is not the answer. Reported so nobody has
        to wonder what was left out."""
        r = artifact["reach"]
        assert (
            r["unequal_100_plus_100_not_the_criterion"]["p"] > r["budget_matched_50_plus_50"]["p"]
        )
        assert "not_the_criterion" in "".join(r)


class TestThePassWasRefused:
    def test_it_clears_the_bar_as_written(self, artifact):
        v = artifact["verdict"]
        assert v["clears_the_bar_as_written"] is True
        assert v["budget_matched_reach"] == 0.2288

    def test_and_is_nonetheless_a_null(self, artifact):
        """Three witnesses of 520. The artifact must not be quotable as a pass."""
        v = artifact["verdict"]
        assert v["result_is_null"] is True
        assert v["licenses_paid_arm"] is False
        assert artifact["discordant"]["net"] == 3

    def test_the_effect_is_a_tenth_of_the_redraw_noise_floor(self, artifact):
        """The number that decides it. NR-46 measured redrawing hypotheses at the same cut at
        +0.0577 reach; this is +0.0057. An effect an order of magnitude below the draw-to-draw
        variance of the same procedure is not evidence about the procedure."""
        v = artifact["verdict"]
        assert v["redraw_noise_floor"] == 0.0577
        assert v["delta_vs_noise_floor"] <= 0.15

    def test_mcnemar_does_not_resolve(self, artifact):
        d = artifact["discordant"]
        assert d["gained"] == 13
        assert d["lost"] == 10
        assert d["mcnemar_exact_p"] > 0.05
        assert d["resolves"] is False

    def test_the_underspecified_bar_is_recorded_as_a_defect(self, artifact):
        """Not repaired after the fact. A threshold with no minimum effect size can be cleared
        by noise, which is exactly what happened, and the next pre-registration should say so
        before it is written rather than after."""
        why = artifact["verdict"]["the_bar_was_underspecified"]
        assert "minimum effect size" in why
        assert "refused" in why


class TestWhatTheReachNumberHides:
    def test_every_loss_is_mechanical(self, artifact):
        """All ten are round-1 ranks in [50, 100) — dropped by halving the cut, not by round
        2 aiming wrong. Without this the null reads as 'the second round is bad'."""
        ranks = artifact["discordant"]["lost_round1_ranks"]
        assert len(ranks) == 10
        assert all(50 <= r < 100 for r in ranks)

    def test_the_gains_come_from_much_deeper(self, artifact):
        assert artifact["discordant"]["gained_round1_rank_median"] > 150

    def test_five_sit_past_the_widest_cut_ever_measured(self, artifact):
        """The part worth keeping. NR-47 ran the cut out to 1000; these sit at 1986-4187 in
        round 1 and inside round 2's top 100. Width cannot buy them at any cut this project
        has run, which makes the mechanism different from the two that are spent."""
        w = artifact["reaches_what_width_cannot"]
        assert w["n"] == 5
        assert all(e["round1"] > 1000 for e in w["examples"])
        assert max(e["round1"] for e in w["examples"]) > 4000

    def test_everything_judged_among_them_is_actionable(self, artifact):
        """Seven of thirteen were ever scored; all seven are actionable. The other six are
        **void, not null** — never judged is not the same as judged badly (C-4, C-30)."""
        j = artifact["reaches_what_width_cannot"]["prf_unique_ever_judged"]
        assert j["actionable"] == j["judged"] == 7
        assert j["never_scored_void_not_null"] == 6

    def test_the_observation_is_labelled_post_hoc_and_does_not_rescue_the_null(self, artifact):
        """The discipline that keeps the null a null. This was found by looking at the result,
        so it earns a new pre-registered test, not a re-reading of this one."""
        c = artifact["reaches_what_width_cannot"]["_comment"]
        assert "POST HOC" in c
        assert "does NOT rescue" in c
        assert artifact["verdict"]["result_is_null"] is True


class TestTheBlindSpotIsRecordedRatherThanPapedOver:
    def test_four_cases_could_not_produce_a_second_round(self, artifact):
        """PRF needs something to feed on, and the shipped arm showed nothing in these four.
        The abstention discipline that makes RepoRadar competitive with Opus 5 is the same
        thing that starves feedback — a structural limit, not a run artifact."""
        b = artifact["prf_blind_spot"]
        assert sorted(b["cases"]) == ["cli", "http", "linter", "webdev"]
        assert artifact["cases_with_round2"] == 33

    def test_they_hold_the_hardest_witnesses(self, artifact):
        """34 of 520, and **none** is in the pool by any route. PRF is blind precisely where
        retrieval most visibly failed, which caps what a pass here could have meant."""
        b = artifact["prf_blind_spot"]
        assert b["witnesses_in_them"] == 34
        assert b["of_those_already_reached"] == 0

    def test_the_skipped_cases_were_never_imputed(self, artifact):
        """Filling them with a fresh draw would have made round 2 a round-1 redraw wearing
        PRF's name — at +0.0577, ten times the effect under test."""
        assert "never imputed" in artifact["prf_blind_spot"]["_comment"]
