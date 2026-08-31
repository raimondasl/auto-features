"""Rung 1: the margin passes the bar, and the bar barely tested it. [NR-52]

The validity gate the ladder in `RESEARCH-net2-directions.md` put before every dollar: is the
margin over Opus 5 a property of RepoRadar or of GPT-5.5? Pre-registered in
`evals/PREREG-rung1.md` and **committed before any margin was computed** (`7ce7a35`), with
three labels named in advance and all three reported.

| label | RepoRadar | Opus 5 | margin | CI95 | w/l/t |
|---|---|---|---|---|---|
| GPT | +5.51 | +5.19 | **+0.32** | [−1.78, +2.51] | 17/17/3 |
| consensus | +5.35 | +4.78 | **+0.57** | [−1.73, +2.92] | 18/16/3 |
| **Sonnet-only** | **−2.03** | **+1.38** | **−3.41** | [−7.00, +0.54] | 13/23/1 |

Both arms fully covered — 306/306 and 357/357 — all 37 cases clearing the prompt-hash drift
check, nothing void.

**The bar passes: |0.57 − 0.32| = 0.25 ≤ 0.5, sign preserved, 6/6 big science losses persist.**

**And the bar was close to unfalsifiable.** `Sonnet ≥ 1` demotes **2 of 272** GPT-actionable
shipped papers (0.7%) and **5 of 302** Opus 5 papers (1.7%), because Sonnet scores 0 on almost
nothing. The consensus label is very nearly GPT itself, so the ±0.5 test would have passed
whatever the truth was. That is the same defect NR-49 recorded, mirrored: there an
under-specified bar was cleared by a null, here it is cleared by a near-tautology. Recorded,
not repaired — moving a bar after seeing data is the failure this project keeps documenting.

**The informative reading flips.** The pre-registration named the Sonnet-only *sign* as the
part that carries information, and it reverses: **our shown papers run 58.5% actionable under
Sonnet, below net@2's 2/3 break-even, while Opus 5's run 71.4%, above it.** Under Sonnet we
destroy value and the comparator creates it. The prediction written in advance — that a harsher
judge would push *both* arms negative and penalise the arm showing more — is wrong in both
halves, and wrong in our disfavour.

**The GPT margin is not draw-stable either.** This control (`20260830T034455Z`, mean +5.51)
gives +0.32; `opus5_arm.json`'s control (`20260827T213701Z`, mean +5.73) gives +0.54 against an
identical Opus 5 arm. Same config, different draw, 0.22/case apart. C-7 applies to our own side.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "evals" / "rung1_second_judge.json"
PREREG = ROOT / "evals" / "PREREG-rung1.md"


@pytest.fixture(scope="module")
def artifact() -> dict:
    return json.loads(FROZEN.read_text(encoding="utf-8"))


class TestTheMeasurementIsSound:
    def test_both_arms_are_fully_covered(self, artifact):
        """A partial second judge would bias the labels, not merely widen them: void papers
        drop out of a label's arithmetic rather than scoring −2, so uneven coverage silently
        reweights the arms. This waited for 100% on both sides."""
        for name in ("gpt", "consensus", "sonnet_only"):
            v = artifact["labels"][name]
            assert v["rr_papers_scored"] == 306
            assert v["opus5_papers_scored"] == 357

    def test_no_case_was_lost_to_prompt_drift(self, artifact):
        """`verify_contexts` excludes any case whose repo clone moved since its GPT verdicts
        were cached — there the stored label answers a question we can no longer rebuild.
        All 37 cleared."""
        assert artifact["excluded_context_drift"] == []
        assert artifact["n_cases"] == 37

    def test_the_prereg_was_committed_before_any_margin(self, artifact):
        assert artifact["pre_registered"]["committed_before_any_margin"] is True
        assert PREREG.is_file()


class TestTheBarPasses:
    def test_the_consensus_margin_tracks_the_gpt_one(self, artifact):
        v = artifact["verdict"]
        assert v["gpt_margin"] == 0.32
        assert v["consensus_margin"] == 0.57
        assert v["abs_shift"] == 0.25
        assert v["within_bar"] is True
        assert v["sign_preserved"] is True

    def test_every_big_science_loss_persists(self, artifact):
        assert artifact["verdict"]["big_losses_persisting"] == 6

    def test_and_so_the_verdict_is_pass(self, artifact):
        assert artifact["verdict"]["passes"] is True


class TestTheBarWasNearlyUnfalsifiable:
    """The finding that matters more than the pass. A bar nobody checked the power of."""

    def test_the_consensus_label_hardly_binds(self, artifact):
        b = artifact["consensus_label_binding"]
        assert b["shipped"]["demoted_by_consensus"] == 2
        assert b["shipped"]["gpt_actionable"] == 272
        assert b["opus5"]["demoted_by_consensus"] == 5
        assert b["opus5"]["gpt_actionable"] == 302
        assert b["shipped"]["demotion_rate"] < 0.02
        assert b["opus5"]["demotion_rate"] < 0.02

    def test_the_weakness_is_recorded_not_repaired(self, artifact):
        """NR-49's discipline, applied to a pass instead of a null: the bar stands as written
        and its weakness is measured beside it, rather than the bar being rewritten."""
        why = artifact["verdict"]["but_the_bar_was_weak"]
        assert "near" in why.lower() or "little information" in why
        assert "NR-49" in why
        assert artifact["verdict"]["passes"] is True


class TestTheInformativeReadingFlips:
    def test_the_sonnet_only_margin_reverses_the_sign(self, artifact):
        v = artifact["verdict"]
        assert v["sonnet_only_margin"] == -3.41
        assert v["sonnet_only_sign_flips"] is True
        assert v["gpt_margin"] > 0 > v["sonnet_only_margin"]

    def test_we_fall_below_break_even_and_the_comparator_does_not(self, artifact):
        """The mechanism, and why the flip is not merely severity. A harsher judge lowers both
        arms; it does not have to move them across the 2/3 line in opposite directions. Ours
        lands below, Opus 5's above."""
        b = artifact["consensus_label_binding"]
        assert b["shipped"]["sonnet_precision"] < b["break_even_precision"]
        assert b["opus5"]["sonnet_precision"] > b["break_even_precision"]
        assert b["shipped"]["sonnet_precision"] == pytest.approx(0.585, abs=0.002)
        assert b["opus5"]["sonnet_precision"] == pytest.approx(0.714, abs=0.002)

    def test_the_advance_prediction_was_wrong_in_both_halves(self, artifact):
        """Written before the run: a harsher judge would push BOTH arms negative, and the arm
        showing MORE would lose more. Opus 5 stays positive and shows more. Recorded so the
        result cannot be re-read as having been anticipated."""
        r = artifact["verdict"]["sonnet_only_reading"]
        assert "wrong in both" in r
        assert artifact["labels"]["sonnet_only"]["opus5_mean_net2"] > 0
        assert artifact["labels"]["sonnet_only"]["rr_mean_net2"] < 0

    def test_it_carries_no_kill_condition_by_design(self, artifact):
        """Registered in advance: a bar on the Sonnet-only LEVEL would measure the judge's
        severity. Its sign is what was named as informative, and the sign is what moved."""
        assert "severity" in artifact["pre_registered"]["sonnet_only_has_no_bar"]


class TestTheGptMarginIsNotDrawStable:
    def test_two_of_our_own_draws_give_different_margins(self, artifact):
        """+0.32 here against +0.54 in opus5_arm.json, same config, same Opus 5 arm. 0.22/case
        of the headline is our own draw noise — C-7 was filed about the comparator's draws and
        applies identically to ours."""
        note = artifact["verdict"]["gpt_margin_is_draw_dependent"]
        assert "0.32" in note and "0.54" in note
        assert "C-7" in note
