"""Which stage of our pipeline loses the papers Opus 5 finds. [NR-45]

P26 left the comparator at +1.52 on the core 25 and −3.17 on materials science. This is the
operational follow-up, $0, over artifacts already on disk: walk each of Opus 5's **302
judged-actionable picks** through the shipped pipeline and record where it dies.

| stage | all 37 | core 25 | bio 6 | matsci 6 |
|---|---|---|---|---|
| non-arXiv — out of the shipped config's reach | 30.5% | 29.9% | **68.1%** | 5.9% |
| not in our 3.1M index | **0.3%** | 0.5% | 0 | 0 |
| **in the index, never reaches our pool** | **54.3%** | 57.8% | 29.8% | **61.8%** |
| pooled but not shown | 10.6% | 8.0% | 0 | 25.0% |
| we showed it too | 4.3% | 3.7% | 2.1% | 7.4% |

**P12 holds against a frontier model.** One paper in 302 is outside the index, so more corpus
buys nothing — the verdict P12 reached against the gold set, now confirmed against Opus 5's
picks. On materials science, the one cohort we lose outright, non-arXiv is 5.9%: P24/P25's "no
more sources" survives exactly where reopening it would have been most tempting. Bio is the
mirror image at 68.1%, which is why Europe PMC's +4.17 source term lives there and nowhere else.

**The rank probe settles the fork.** `hyde.top_k = 100` per hypothesis is the cut those papers
failed. Their ranks under our own hypotheses: **median 1,087**, p25 323, p75 3,562. Only 11.5%
are reachable at the shipped cut; 49% at 1,000, 78.8% at 5,000. **The union is too narrow — the
hypotheses are not in the wrong register**, which is the cheaper of the two branches and makes
the follow-up a config integer rather than a new mechanism.

**Reach is not net@2, and the tests below say so.** NR-11 recorded a wider pool meeting a
near-binary gate and making the headline *worse*; §8.2's composition finding is that pool
expansion was a wash until the rescore ranked what the gate admitted. So this opens a
measurement, not a patch.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "evals" / "opus5_funnel.json"


@pytest.fixture(scope="module")
def artifact() -> dict:
    return json.loads(FROZEN.read_text(encoding="utf-8"))


class TestTheLeverIsPoolAssembly:
    def test_most_of_what_we_lose_is_in_our_own_index(self, artifact):
        """54.3% of Opus 5's actionable picks sit in the 3.1M vectors we already hold and
        never reach our candidate pool. That is one subsystem, and it is the queries."""
        f = artifact["funnel"]["all37"]
        assert f["actionable_picks"] == 302
        assert f["2_index_but_not_pool"] == 164
        assert f["2_index_but_not_pool_share"] > 0.5
        assert f["2_index_but_not_pool"] > 3 * f["3_pool_but_not_shown"]

    def test_a_wider_corpus_would_buy_essentially_nothing(self, artifact):
        """**One paper in 302.** P12 reached this verdict against the gold set — all 56
        targets already in-index, unreached ones at ranks up to 223,245 — and it survives
        being pointed at a frontier model's picks instead."""
        assert artifact["funnel"]["all37"]["1_not_in_index"] == 1
        assert artifact["funnel"]["all37"]["1_not_in_index_share"] < 0.01

    def test_the_cohort_we_lose_has_the_same_shape(self, artifact):
        """18 cases lose to Opus 5 on the shipped arm, and their funnel is the pooled one:
        51.8% stuck in the index. Whatever is wrong is not specific to where we lose."""
        f = artifact["funnel"]["cases_we_lose"]
        assert f["n_cases"] == 18
        assert f["2_index_but_not_pool_share"] > 0.5
        assert f["1_not_in_index"] <= 1


class TestTheCohortsDisagreeAboutSources:
    def test_matsci_cannot_be_fixed_with_more_sources(self, artifact):
        """5.9% non-arXiv on the one cohort we lose outright. P24 and P25 closed the source
        question on net@2 grounds; this closes it again on reach grounds, in the cohort where
        reopening it would have been most tempting."""
        assert artifact["funnel"]["mat"]["0_non_arxiv_share"] < 0.10
        assert artifact["funnel"]["mat"]["2_index_but_not_pool_share"] > 0.6

    def test_bio_is_the_mirror_image_and_explains_the_epmc_result(self, artifact):
        """68.1% of Opus 5's bio picks are not arXiv at all — which is why Europe PMC's own
        papers are worth +4.17/case there and +0.08 on the core 25."""
        assert artifact["funnel"]["bio"]["0_non_arxiv_share"] > 0.6
        assert (
            artifact["funnel"]["bio"]["0_non_arxiv_share"]
            > 2 * artifact["funnel"]["core"]["0_non_arxiv_share"]
        )

    def test_matsci_also_leaks_downstream_which_core_does_not(self, artifact):
        """A second, smaller loss unique to materials science: a quarter of its papers reach
        the pool and are not shown, against 8% on the core 25. Consistent with P26 finding
        Opus 5 out-picking us there from arXiv material we already hold."""
        assert artifact["funnel"]["mat"]["3_pool_but_not_shown_share"] > 0.2
        assert artifact["funnel"]["core"]["3_pool_but_not_shown_share"] < 0.1


class TestTheUnionIsTooNarrowNotTheRegisterWrong:
    def test_the_missing_papers_sit_just_below_the_cut(self, artifact):
        """The fork this probe existed to settle. A median rank of 1,087 against a cut of 100
        means our own hypotheses *do* find these papers — we throw them away. Tens of
        thousands would have meant the hypotheses were in the wrong register, which is a
        mechanism problem rather than a parameter one."""
        rp = artifact["rank_probe"]
        assert rp["median"] < 2000
        assert rp["p25"] < 500
        assert rp["verdict"]["union_too_narrow"] is True
        assert rp["verdict"]["hypotheses_in_wrong_register"] is False

    def test_the_shipped_cut_reaches_almost_none_of_them(self, artifact):
        rp = artifact["rank_probe"]
        assert artifact["shipped_hyde_top_k"] == 100
        assert rp["share_at_cut"]["100"] < 0.15
        assert rp["share_at_cut"]["1000"] > 0.45
        assert rp["share_at_cut"]["5000"] > 0.75

    def test_the_recovery_curve_is_monotone(self, artifact):
        """Trivially true of a correct implementation, and the cheapest check that the rank
        bookkeeping is not scrambled."""
        rp = artifact["rank_probe"]
        cuts = sorted(int(c) for c in rp["recovered_at_cut"])
        got = [rp["recovered_at_cut"][str(c)] for c in cuts]
        assert got == sorted(got)
        assert got[-1] <= rp["n_papers"]

    def test_the_uncovered_cases_are_absent_rather_than_assumed(self, artifact):
        """The probe covers the cases whose hypotheses the replication froze — 104 of the
        164, and none of the six materials cases. Generating the rest costs LLM calls, so
        they are reported missing rather than imputed. A rank probe that quietly filled them
        in would be answering for a cohort it never measured."""
        rp = artifact["rank_probe"]
        assert rp["n_papers"] == 104
        assert rp["n_papers"] < artifact["funnel"]["all37"]["2_index_but_not_pool"]
        assert "materials" in rp["_comment"]


class TestItOpensAMeasurementNotAPatch:
    def test_the_artifact_says_reach_is_not_net2(self, artifact):
        """The caveat has to travel with the number. NR-11 measured a wider pool meeting a
        near-binary gate and making the headline worse; anyone reading "78.8% recoverable at
        cut 5,000" needs that in the same breath."""
        c = artifact["_comment"]
        assert "Reach is not net@2" in c
        assert "NR-11" in c
        assert "measurement, not a patch" in c
