"""Item 5: the judge-contamination hypothesis, tested against stored verdicts and refuted.

The worry, from LitLLMs: an LLM judge has *seen* pre-cutoff papers in training, rewards that
familiarity, and so labels older work more kindly. If true, every recall and net@2 figure in
this project inherits the bias — which is why it is worth an afternoon of re-analysis and no
new spend.

**It is not supported.** The check that settles it is a second model. Claude Sonnet 5 judged
837 of the same papers under a byte-identical rubric, and it does not have GPT-5.5's cutoff:

| period | n | GPT-5.5 | Sonnet | they agree |
|---|---|---|---|---|
| 2024 and earlier | 500 | 0.720 | 0.410 | 0.662 |
| 2025 | 111 | 0.739 | 0.523 | 0.748 |
| 2026 to June | 89 | 0.685 | 0.461 | 0.775 |
| **2026-07** | 38 | **0.237** | **0.105** | **0.868** |

Both judges collapse on the newest month, and they agree *most* there. One model's training
cutoff cannot produce that.

**What is in the data instead is a retrieval symptom.** July papers draw a score of 0 — "no
relation to this repository" — **52% of the time against 10% for 2024–2025, a 5.1× jump**,
at the highest judged volume of any month. Unfamiliarity does not have that shape: an
unfamiliar paper draws a hedged 1 or 2, not a flat rejection. The freshest slice is where
off-topic material enters the pool, and RepoRadar is a freshness product.

**The residual is recorded, not buried.** If *both* judges' cutoffs fall in mid-2026, both
would be unfamiliar and both would mark July down — two models, one shared blind spot, the
same prediction. The single-judge story is refuted; the shared-cutoff story is not, and the
score-0 shape argues against it without excluding it. `shared_cutoff_excluded` is `false` in
the artifact for exactly that reason.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "evals" / "judge_date_stratify.json"


@pytest.fixture(scope="module")
def artifact() -> dict:
    return json.loads(FROZEN.read_text(encoding="utf-8"))


class TestTheEvidenceBase:
    def test_it_re_reads_stored_verdicts_rather_than_buying_new_ones(self, artifact):
        """The whole item is `$0 hygiene`: 4,019 GPT-5.5 verdicts already cached and the 837
        Sonnet verdicts P7 paid for. A re-analysis that needed new calls would be a different,
        more expensive item than the one the plan authorised."""
        v = artifact["verdicts"]
        assert v["primary_total"] == 4019
        assert v["second_total"] == v["paired"] == 837

    def test_undated_verdicts_are_counted_and_excluded(self, artifact):
        """447 verdicts are on DOIs, Semantic Scholar hashes or pre-2007 ids whose date this
        cannot read. They are reported rather than silently dropped — a denominator that
        quietly shrinks is how a stratification starts answering a different question."""
        v = artifact["verdicts"]
        assert v["primary_undated"] == 447
        assert v["primary_dated"] + v["primary_undated"] == v["primary_total"]


class TestTheTrendIsSmoothAndTheCollapseIsNot:
    def test_actionability_rises_with_recency_across_the_years(self, artifact):
        """The expected trend, and the reason a raw correlation would prove nothing: a recent
        paper is likelier to help a current codebase, and `recency` is one of the ranker's own
        scoring components. Contamination would be a STEP on top of this, not the slope."""
        y = artifact["by_year"]
        assert y["2013"]["actionable_rate"] < 0.4
        assert y["2025"]["actionable_rate"] > 0.6
        assert y["2019"]["actionable_rate"] < y["2025"]["actionable_rate"]

    def test_the_newest_month_falls_off_that_trend_entirely(self, artifact):
        h = artifact["headline"]
        assert h["newest_month"] == "2026-07"
        assert h["newest_month_n"] == 159
        assert h["newest_month_rate"] == 0.2327
        lo, hi = h["newest_month_ci95"]
        assert hi < 0.4, "the interval does not reach any other month of 2026"

    def test_it_is_one_month_rather_than_a_decline(self, artifact):
        """Every other month of 2026 sits between 0.46 and 0.68. A gradual loss of judge
        familiarity would taper; this does not taper."""
        months = artifact["by_month_recent"]
        others = [
            v["actionable_rate"]
            for k, v in months.items()
            if k.startswith("2026") and k != "2026-07"
        ]
        assert min(others) > 0.4
        assert months["2026-07"]["actionable_rate"] < min(others) / 1.5


class TestTheSecondJudgeIsWhatSettlesIt:
    def test_a_different_model_shows_the_same_collapse(self, artifact):
        """**The decisive test.** Sonnet does not share GPT-5.5's cutoff, and falls further
        (0.105 against 0.237). A bias belonging to one model's training data cannot appear in
        a second model that was trained differently."""
        t = artifact["two_judges"]
        assert t["newest_month"]["gpt_rate"] < 0.3
        assert t["newest_month"]["sonnet_rate"] < 0.2
        assert t["newest_month"]["gpt_rate"] < t["2025"]["gpt_rate"] / 2
        assert t["newest_month"]["sonnet_rate"] < t["2025"]["sonnet_rate"] / 2

    def test_the_judges_agree_MORE_on_the_newest_month_not_less(self, artifact):
        """The shape that kills the hypothesis. If the newest papers were the ones the judge
        could not assess, its verdicts there would be noisier and agreement would fall. It
        rises to 0.868, the highest of any period — the two models are not confused about
        these papers, they concur about them."""
        t = artifact["two_judges"]
        agreements = {k: v["agreement"] for k, v in t.items() if v}
        assert max(agreements, key=lambda k: agreements[k]) == "newest_month"
        assert t["newest_month"]["agreement"] > t["2024_and_earlier"]["agreement"]

    def test_the_verdict_is_recorded_with_its_limit(self, artifact):
        """Refuting the single-judge story is not the same as excluding a shared one, and the
        artifact says so in a field rather than in prose nobody reads."""
        h = artifact["headline"]
        assert h["single_judge_contamination_supported"] is False
        assert h["shared_cutoff_excluded"] is False


class TestTheConfoundsWereCheckedNotArguedAway:
    def test_it_is_not_case_mix(self, artifact):
        """The newest month over-represents `webdev`, `http` and `cli` — three of the four
        repositories RepoRadar abstains on, which run low in every period. So the comparison
        is made INSIDE each repository, and 10 of 11 still fall."""
        w = artifact["within_case"]
        assert w["n_cases"] == 11
        assert w["n_falling"] == 10
        assert w["mean_delta"] == -0.221

    def test_it_is_not_the_dense_index_running_out(self, artifact):
        """The index's newest paper is in the same month, so July is split between what the
        index holds and what only the live keyword channel could reach — and keyword-only
        retrieval is the configuration this project measures at −8.12. That would have been a
        tidy explanation. It fails: the IN-index half still falls to 0.333 against June's
        0.510."""
        b = artifact["index_boundary"]
        assert artifact["index_newest_month"] == [2026, 7]
        assert b["not_in_index_rate"] < b["in_index_rate"], "the channel does matter somewhat"
        assert b["in_index_rate"] < b["previous_month_rate"] * 0.75, "but it does not explain it"


class TestWhatTheDataActuallyHolds:
    def test_the_newest_papers_are_rejected_outright_not_hedged(self, artifact):
        """The measurement that reframes the finding. A 0 means "no relation to this
        repository"; a judge that could not assess a paper would hedge at 1 or 2. Half the
        newest month is a flat rejection, five times the base rate — that is a statement about
        what retrieval put in the pool, not about what the judge could recognise."""
        d = artifact["score_distribution"]
        assert d["newest_month"]["score_0"] == 0.522
        assert d["2024_2025"]["score_0"] == 0.103
        assert d["zero_rate_ratio"] > 5.0

    def test_the_hedging_scores_do_not_rise(self, artifact):
        """The clincher for that reading. Unfamiliarity would inflate the middle of the
        distribution; instead scores 1, 2 and 3 all FALL and only 0 grows."""
        new = artifact["score_distribution"]["newest_month"]
        old = artifact["score_distribution"]["2024_2025"]
        for k in ("score_1", "score_2", "score_3"):
            assert new[k] < old[k], k
        assert new["score_0"] > old["score_0"]
