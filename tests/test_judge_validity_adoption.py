"""The primary judge has not been shown to discriminate adoption. [NR-56]

NR-52 and NR-53 established that the judges disagree and that the disagreement is real rather
than sampling. Both are **reliability**. This is the first **validity** test: `ids(HEAD) −
ids(T0)` over a repo's own documentation is a set of papers the project verifiably took up,
mined from git history with no model in the loop.

Recall on that cannot rank judges — adoption gives positives only, and a judge that says yes to
everything scores 100%. So each of the 35 adopted papers is paired with matched controls (same
repo, published *before* the same T0, never adopted, not a T0 seed), both judges score
everything against the identical T0 context, and the statistic is the **gap**.

| judge | adopted | control | gap | 95% CI |
|---|---|---|---|---|
| **gpt-5.5** (primary) | 23/35 = 0.657 | **72/140 = 0.514** | **0.143** | **spans zero** |
| claude-sonnet-5 | 17/35 = 0.486 | 34/140 = 0.243 | 0.243 | excludes zero |

GPT's gap interval is [−0.043, +0.321]; Sonnet's is [+0.064, +0.421].

**The judge every number in this project rests on calls 51.4% of matched controls actionable** —
papers from the same repository, publishable before the same cutoff, that the project never took
up. Its discrimination interval includes zero.

**And this does not name a better instrument.** The gap difference is 0.100, short of the
pre-registered 0.15 separation bar. Reported as not separated rather than
resolved toward the judge that happens to look better.

**Absence of evidence, not evidence of error.** n = 35 across 9 cases with `graph` contributing
13; "not adopted" is a noisy negative that biases both gaps downward; and adoption measures what
a repository *did*, not what it *should* have done. What the result establishes is that the
primary judge's validity against the only model-free label available here is **unestablished**,
which is a different and more precise claim than "the judge is wrong".
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "evals" / "judge_validity_adoption.json"


@pytest.fixture(scope="module")
def artifact() -> dict:
    return json.loads(FROZEN.read_text(encoding="utf-8"))


class TestTheDesignCanRankJudgesAtAll:
    def test_it_uses_matched_controls_not_recall(self, artifact):
        """The design problem the probe exists to solve: adoption gives positives only, so
        recall rewards leniency. Controls are what make the number mean something."""
        assert artifact["n_positives"] == 35
        assert artifact["n_controls"] == 140
        assert "gap" in artifact["pre_registered"]["primary"]
        assert artifact["pre_registered"]["judges_ranked_by_gap_not_recall"] is True

    def test_the_bars_predate_the_controls_and_the_verdicts(self, artifact):
        assert artifact["pre_registered"]["written_before_any_control_or_sonnet_verdict"] is True
        assert artifact["pre_registered"]["flat_if_both_below"] == 0.20
        assert artifact["pre_registered"]["separated_if_difference_at_least"] == 0.15

    def test_the_positives_are_concentrated_and_it_is_recorded(self, artifact):
        """graph supplies 13 of 31. C-7's shape, stated in the artifact rather than left for a
        reader to derive from the per-case counts."""
        by_case = artifact["positives_by_case"]
        assert by_case["graph"] == 13
        assert sum(by_case.values()) == 35
        assert "C-7" in artifact["verdict"]["caveats"]


class TestThePrimaryJudgeIsNotShownToDiscriminate:
    def test_its_gap_interval_spans_zero(self, artifact):
        g = artifact["judges"]["gpt-5.5"]
        assert g["gap"] == pytest.approx(0.143, abs=0.005)
        assert g["gap_ci95"][0] < 0 < g["gap_ci95"][1]
        assert g["gap_excludes_zero"] is False
        assert artifact["verdict"]["primary_judge_gap_spans_zero"] is True

    def test_it_calls_half_the_matched_controls_actionable(self, artifact):
        """0.492 on papers the repository never took up. This is the lenient-judge failure the
        control arm was built to expose, and the primary judge exhibits it."""
        c = artifact["judges"]["gpt-5.5"]["control"]
        assert c["n"] == 140
        assert c["rate"] == pytest.approx(0.514, abs=0.005)

    def test_the_second_judge_does_exclude_zero(self, artifact):
        s = artifact["judges"]["claude-sonnet-5"]
        assert s["gap"] == pytest.approx(0.243, abs=0.005)
        assert s["gap_ci95"][0] > 0
        assert s["gap_excludes_zero"] is True
        assert s["control"]["rate"] < artifact["judges"]["gpt-5.5"]["control"]["rate"]


class TestItReplicatesNR56OnAnIndependentSample:
    """NR-56 ran 31 positives / 124 controls under a shared-rng control draw. This is 35 / 140
    with controls drawn PER CASE, so they were fully redrawn — an independent sample, not a
    superset. Both conclusions survive, slightly attenuated."""

    def test_the_prior_sample_is_recorded_not_overwritten(self, artifact):
        r = artifact["verdict"]["replicates_nr56"]
        assert r["nr56"]["n_pos"] == 31
        assert r["nr56"]["gpt_gap"] == 0.153
        assert r["nr56"]["sonnet_gap"] == 0.282
        assert r["this_run"]["n_pos"] == 35

    def test_both_conclusions_hold(self, artifact):
        r = artifact["verdict"]["replicates_nr56"]
        assert r["gpt_still_spans_zero"] is True
        assert r["sonnet_still_excludes_zero"] is True
        assert r["still_not_separated"] is True

    def test_the_controls_were_redrawn_and_that_is_stated(self, artifact):
        """Changing the sampling scheme is what made the samples independent. Saying so is the
        difference between a replication and an accidental one."""
        assert (
            "INDEPENDENT sample, not a superset"
            in artifact["verdict"]["replicates_nr56"]["_comment"]
        )


class TestTheChannelIsExhausted:
    def test_mining_every_case_moved_31_to_35(self, artifact):
        """15 newly mined cases contributed 4 adoptions. The shortfall is structural."""
        w = artifact["verdict"]["what_would_settle_it"]
        assert w["n_positives_available"] == 35
        assert w["n_positives_needed_at_this_gap"] == 55

    def test_the_reason_is_recorded_so_nobody_retries_it(self, artifact):
        """Several cases carry no arXiv ids in their docs at all; others have no history before
        the T0 cutoff. Reaching 55 needs a different benchmark, not more effort on this one."""
        why = artifact["verdict"]["what_would_settle_it"]["why_expansion_stalled"]
        assert "0 ids at HEAD" in why
        assert "differently-constructed benchmark" in why

    def test_precision_is_governed_by_the_positives(self, artifact):
        """The adopted variance term is 4-6x the control term because n_pos is a quarter of
        n_ctl — which is why adding controls would not help and adding adoptions would."""
        assert artifact["n_controls"] == 4 * artifact["n_positives"]
        assert (
            "governed almost entirely by the POSITIVES"
            in artifact["verdict"]["what_would_settle_it"]["_comment"]
        )


class TestItDoesNotNameAWinner:
    def test_the_difference_misses_the_registered_bar(self, artifact):
        v = artifact["verdict"]
        assert v["difference"] == pytest.approx(0.100, abs=0.005)
        assert v["separated"] is False
        assert v["better_instrument"] is None

    def test_the_difference_interval_includes_zero(self, artifact):
        b = artifact["gap_difference_bootstrap"]
        assert b["ci95"][0] < 0 < b["ci95"][1]
        assert b["excludes_zero"] is False

    def test_the_registered_both_flat_outcome_did_not_fire(self, artifact):
        """Sonnet clears 0.20, so the "neither judge tracks the goal" branch is not the result.
        The finding is narrower and specifically about the primary judge."""
        assert artifact["verdict"]["both_flat"] is False

    def test_the_headline_claims_absence_of_evidence_not_error(self, artifact):
        """The precise claim. A gap spanning zero at n=31 with noisy negatives does not show
        the judge is wrong; it shows its validity here is unestablished."""
        h = artifact["verdict"]["headline"]
        assert "Absence of evidence, not evidence of error" in h
        assert "does not name a better instrument" in h.lower() or "does not name" in h
