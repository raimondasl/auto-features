"""P21's bio figures, pinned — because their run files are not in version control. [NR-40]

`evals/results/` is gitignored, so the numbers PLANS.md and RESULTS.md quote for the matched
bio comparison would otherwise cite something nothing in the repository can reproduce. That is
the failure `gold_targets.json` and `restated_runs.json` were built against, and the remedy is
the same: derive once (`evals/freeze_bio_arm.py`), commit the derivation, pin it here.

What the comparison establishes, and why the arm had to be *run* rather than derived:

* Opus 5 uses the v2 prompt, so **68% of its bio targets are non-arXiv**. Grading that against
  an arXiv-only RepoRadar was not a like-for-like contest — it was measuring a restriction.
* Three axes separated the two pre-existing bio runs at once. Two of them (window, source) can
  be isolated from data already in hand; `w_embedding` cannot, and it is the one that surprised.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "evals" / "bio_matched_arm.json"
BIO = {"bio-align", "bio-kmer", "bio-mdsim", "bio-mdtraj", "bio-scvi", "bio-singlecell"}


@pytest.fixture(scope="module")
def artifact() -> dict:
    return json.loads(FROZEN.read_text(encoding="utf-8"))


class TestTheArmsAreWhatTheyClaim:
    def test_each_arm_records_the_configuration_that_produced_it(self, artifact):
        """Three arms differing on three axes. Recorded per arm so a reader cannot mistake
        one for another — the confusion that produced a 5.5-point discrepancy in the first
        place."""
        a = artifact["arms"]
        assert (a["w15_epmc_wemb15"]["digest_window"], a["w15_epmc_wemb15"]["w_embedding"]) == (
            15,
            1.5,
        )
        assert a["w15_epmc_wemb15"]["sources"] == ["arxiv", "europepmc"]
        assert (a["w30_epmc_wemb0"]["digest_window"], a["w30_epmc_wemb0"]["w_embedding"]) == (
            30,
            0.0,
        )
        assert a["w15_arxiv_wemb0"]["sources"] == ["arxiv"]

    def test_every_arm_covers_the_same_six_cases(self, artifact):
        for label, arm in artifact["arms"].items():
            assert set(arm["per_case"]) == BIO, label
        assert set(artifact["opus5"]["per_case"]) == BIO

    def test_net2_is_recomputable_from_its_own_parts(self, artifact):
        """net@2 = #actionable - 2 x #non-actionable, so it is bounded by the count returned
        and has the parity of it."""
        for label, arm in artifact["arms"].items():
            for case, row in arm["per_case"].items():
                assert -2 * row["n"] <= row["net2"] <= row["n"], f"{label}/{case}"


class TestTheMatchedComparison:
    def test_reporadar_leads_opus5_at_a_matched_configuration(self, artifact):
        """+8.17 against +5.83. The arXiv-only arm said -0.33; the reversal is the source."""
        assert artifact["arms"]["w15_epmc_wemb15"]["mean_net2"] == 8.17
        assert artifact["opus5"]["mean_net2"] == 5.83
        m = artifact["matched_comparison"]
        assert (m["paired_delta"], m["wins"], m["losses"]) == (2.33, 4, 1)

    def test_the_arxiv_only_arm_reversed_the_sign(self, artifact):
        """Kept as a pin because it is the number that was reported first, and the reason it
        was wrong is a property of the comparison rather than of either system."""
        arxiv_only = artifact["arms"]["w15_arxiv_wemb0"]["mean_net2"]
        assert arxiv_only == 5.50
        assert arxiv_only < artifact["opus5"]["mean_net2"], "the arXiv-only arm trails Opus 5"


class TestTheDecomposition:
    def test_the_three_axes_account_for_the_gap(self, artifact):
        """window -1.50, Europe PMC +4.00, w_embedding -1.33 — and they must sum to the
        distance between the two pre-existing arms, or one of them is mis-attributed."""
        d = artifact["decomposition"]
        assert d["window_30_to_15"] == -1.5
        assert d["add_europepmc_at_wemb0"] == 4.0
        assert d["wemb_0_to_1p5_at_w15"] == -1.33
        a = artifact["arms"]
        total = a["w15_epmc_wemb15"]["mean_net2"] - a["w15_arxiv_wemb0"]["mean_net2"]
        parts = d["add_europepmc_at_wemb0"] + d["wemb_0_to_1p5_at_w15"]
        assert abs(total - parts) < 0.02, "the axes do not add up to the arms"

    def test_nr40_w_embedding_is_negative_on_this_cohort(self, artifact):
        """The shipped value, tuned on the arXiv-only core 25, costs 1.33 on the scientific
        cohort. One 6-case run — not conclusive, and pinned so it is not quoted as if it were.
        It was visible only because the axis was RUN; truncating the window-30 arm gives
        w_embedding 0.0 and answers a different question."""
        assert artifact["decomposition"]["wemb_0_to_1p5_at_w15"] < 0

    def test_the_source_outweighs_the_window(self, artifact):
        """The headline of the decomposition: the missing source did nearly three times the
        work the window did, and the window was the axis everyone looked at first."""
        d = artifact["decomposition"]
        assert d["add_europepmc_at_wemb0"] > 2 * abs(d["window_30_to_15"])


class TestWindowTruncationIsSound:
    def test_truncating_thirty_to_fifteen_never_raises_the_score(self, artifact):
        """The digest is a final cut on a list already ordered by `llm_score`, so dropping
        ranks 16-30 can only remove papers. A truncation that scored HIGHER would mean the
        stored picks are not in digest order and the derivation is invalid."""
        arm = artifact["arms"]["w30_epmc_wemb0"]
        for case, row in arm["per_case"].items():
            assert row["net2_truncated_15"] <= row["net2"], case
        assert arm["mean_net2_truncated_15"] <= arm["mean_net2"]

    def test_truncation_cannot_reach_the_w_embedding_axis(self, artifact):
        """Why the arm cost a run instead of a script: the truncated arm is w_embedding 0.0,
        and the shipped value is 1.5. They are different configurations and differ in fact."""
        a = artifact["arms"]
        assert a["w30_epmc_wemb0"]["w_embedding"] == 0.0
        assert a["w15_epmc_wemb15"]["w_embedding"] == 1.5
        assert a["w30_epmc_wemb0"]["mean_net2_truncated_15"] != a["w15_epmc_wemb15"]["mean_net2"]
