"""The primary judge has not been shown to discriminate adoption. [NR-56]

NR-52 and NR-53 established that the judges disagree and that the disagreement is real rather
than sampling. Both are **reliability**. This is the first **validity** test: `ids(HEAD) −
ids(T0)` over a repo's own documentation is a set of papers the project verifiably took up,
mined from git history with no model in the loop.

Recall on that cannot rank judges — adoption gives positives only, and a judge that says yes to
everything scores 100%. So each of the 31 adopted papers is paired with matched controls (same
repo, published *before* the same T0, never adopted, not a T0 seed), both judges score
everything against the identical T0 context, and the statistic is the **gap**.

| judge | adopted | control | gap | 95% CI |
|---|---|---|---|---|
| **gpt-5.5** (primary) | 20/31 = 0.645 | **61/124 = 0.492** | **0.153** | **spans zero** |
| claude-sonnet-5 | 15/31 = 0.484 | 25/124 = 0.202 | 0.282 | excludes zero |

GPT's gap interval is [−0.040, +0.339]; Sonnet's is [+0.097, +0.476].

**The judge every number in this project rests on calls 49.2% of matched controls actionable** —
papers from the same repository, publishable before the same cutoff, that the project never took
up. Its discrimination interval includes zero.

**And this does not name a better instrument.** The gap difference is 0.129, CI [−0.024,
+0.274], short of the pre-registered 0.15 separation bar. Reported as not separated rather than
resolved toward the judge that happens to look better.

**Absence of evidence, not evidence of error.** n = 31 across 6 cases with `graph` contributing
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
        assert artifact["n_positives"] == 31
        assert artifact["n_controls"] == 124
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
        assert sum(by_case.values()) == 31
        assert "C-7" in artifact["verdict"]["caveats"]


class TestThePrimaryJudgeIsNotShownToDiscriminate:
    def test_its_gap_interval_spans_zero(self, artifact):
        g = artifact["judges"]["gpt-5.5"]
        assert g["gap"] == pytest.approx(0.153, abs=0.005)
        assert g["gap_ci95"][0] < 0 < g["gap_ci95"][1]
        assert g["gap_excludes_zero"] is False
        assert artifact["verdict"]["primary_judge_gap_spans_zero"] is True

    def test_it_calls_half_the_matched_controls_actionable(self, artifact):
        """0.492 on papers the repository never took up. This is the lenient-judge failure the
        control arm was built to expose, and the primary judge exhibits it."""
        c = artifact["judges"]["gpt-5.5"]["control"]
        assert c["n"] == 124
        assert c["rate"] == pytest.approx(0.492, abs=0.005)

    def test_the_second_judge_does_exclude_zero(self, artifact):
        s = artifact["judges"]["claude-sonnet-5"]
        assert s["gap"] == pytest.approx(0.282, abs=0.005)
        assert s["gap_ci95"][0] > 0
        assert s["gap_excludes_zero"] is True
        assert s["control"]["rate"] < artifact["judges"]["gpt-5.5"]["control"]["rate"]


class TestItDoesNotNameAWinner:
    def test_the_difference_misses_the_registered_bar(self, artifact):
        v = artifact["verdict"]
        assert v["difference"] == pytest.approx(0.129, abs=0.005)
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
