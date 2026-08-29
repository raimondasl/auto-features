"""Item 9 closed negative, and NR-43's retrieval claim retracted. [C-35]

NR-43 saw that the newest month of the judge cache scores 52% "no relation" and wrote: *the
freshest slice is where off-topic material enters the pool, and RepoRadar is a freshness
product whose freshest slice is its worst.* It filed a mechanism with it — `recency` weighting
promoting fresh papers — and a $0 next step to confirm it.

**Both died in that $0 step, which is what it was for.**

*The mechanism does not exist.* `w_recency` is **0.0** — shipped default, `evals/harness.py`,
and every benchmark run since 2026-07-06, because `--rr-all-time` *is* `w_recency 0`. One
minute in `config.py` would have caught it before the claim was written.

*The effect is not in the product.* Promotion is flat in age (146.1 per 10k pooled for
pre-2026 against 183.2 for the newest month — about one paper's difference), and of 159 judged
July papers **only 11 were ever shown, 6.9% against 36–44% for every other period, and all 11
are actionable**. The 148 the gate declined score 0.176.

**So the gate is doing its job, visibly, exactly where the pool is hardest.**

**The error worth keeping is methodological.** The judge cache is not a sample of what the
product returns — it is the union of every experiment run here, including `diagnose_ranker.py`
judging ranks 151+ *by design*. Stratifying it by date measured the sampling. The tell sat in
the same artifact: July was the **most-judged** month and the **least-shown** one, and a slice
that is simultaneously over-judged and under-shown is a sampling artifact, not a finding.

NR-43's contamination result is untouched. That one rests on two judges agreeing, not on which
papers were sampled.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "evals" / "fresh_slice_probe.json"


@pytest.fixture(scope="module")
def artifact() -> dict:
    return json.loads(FROZEN.read_text(encoding="utf-8"))


class TestTheMechanismNeverExisted:
    def test_recency_is_not_weighted_in_the_measured_configuration(self, artifact):
        """The claim was "a paper from last month scores ~1.0 on recency regardless of
        topical fit". True of the component, irrelevant to the score: its weight is zero, so
        it contributes nothing. `--rr-all-time`, which every headline since 2026-07-06 uses,
        IS `w_recency 0`."""
        assert artifact["recency_weight_in_measured_config"] == 0.0
        assert artifact["verdict"]["recency_weight_is_zero"] is True
        assert artifact["verdict"]["recency_over_promotion"] is False

    def test_promotion_does_not_rise_with_freshness(self, artifact):
        """The measurement that would have shown over-promotion if it existed. Fresh papers
        are shown at 183.2 per 10,000 pooled against 146.1 for pre-2026 — on 273 pooled
        papers and 5 shown, so the gap is about one paper, and it points the harmless way."""
        p = artifact["promotion_by_age"]
        assert p["pre_2026"]["per_10k"] == 146.1
        assert p["newest_month"]["per_10k"] == 183.2
        assert p["newest_month"]["shown"] <= 6, "the whole difference is a handful of papers"
        assert p["newest_month"]["per_10k"] < 2 * p["pre_2026"]["per_10k"]


class TestTheProductIsFineWhereTheClaimSaidItWasWorst:
    def test_almost_nothing_from_the_newest_month_is_ever_shown(self, artifact):
        """6.9% against 36–44%. The collapse NR-43 found lives in papers the gate declined,
        which is the gate working rather than failing."""
        j = artifact["judged_versus_shown"]
        assert j["newest_month"]["shown_share"] == 0.0692
        for other in ("2026_h1", "2025", "2024_and_older"):
            assert j[other]["shown_share"] > 0.35, other

    def test_and_what_is_shown_from_it_is_perfect(self, artifact):
        """11 of 11 actionable — the highest of any period, against 0.755–0.822 elsewhere. A
        product defect cannot look like this."""
        j = artifact["judged_versus_shown"]
        assert j["newest_month"]["actionable_shown"] == 1.0
        assert j["newest_month"]["actionable_shown"] > j["2025"]["actionable_shown"]
        assert artifact["verdict"]["freshest_slice_defect_in_product"] is False

    def test_the_pool_really_is_worse_there_which_is_why_the_gate_matters(self, artifact):
        """The half of NR-43 that survives. Among papers never shown, the newest month scores
        0.176 against 0.38–0.50 for older periods — the freshest slice IS the hardest slice.
        What the gate does with it is the point: it declines almost all of it."""
        j = artifact["judged_versus_shown"]
        assert j["newest_month"]["actionable_never_shown"] < 0.2
        for other in ("2026_h1", "2025", "2024_and_older"):
            assert j[other]["actionable_never_shown"] > 0.35, other
        assert artifact["verdict"]["gate_declines_the_weak_fresh_papers"] is True


class TestWhatWentWrongIsRecordedAsWellAsWhatIsTrue:
    def test_the_retraction_is_a_field_not_a_footnote(self, artifact):
        """A correction that lives only in prose gets quoted around. The artifact carries the
        verdict so anyone reading the number reads the retraction with it."""
        v = artifact["verdict"]
        assert v["nr43_retrieval_claim_retracted"] is True
        assert v["nr43_contamination_result_unaffected"] is True

    def test_the_judge_cache_is_much_larger_than_what_was_ever_shown(self, artifact):
        """The root of the error, as a number. The cache spans 117 run files' worth of
        experiments — rank-stratified pool draws, source arms, ablations — so most of what it
        holds was never a digest entry and never could be. Any date-stratified reading of it
        describes the sampling."""
        assert artifact["run_files_scanned"] > 100
        j = artifact["judged_versus_shown"]
        judged = sum(v["judged"] for k, v in j.items() if k != "_comment")
        shown = sum(v["ever_shown"] for k, v in j.items() if k != "_comment")
        assert shown < judged / 2, "most judged papers were never shown by anything"

    def test_the_tell_was_in_the_same_artifact(self, artifact):
        """The newest month is the MOST judged and the LEAST shown. Over-judged and
        under-shown at once is the signature of a sampling artifact, and it was visible in
        NR-43's own tables before the retrieval claim was written on top of them."""
        j = artifact["judged_versus_shown"]
        assert j["newest_month"]["judged"] == 159
        assert j["newest_month"]["ever_shown"] == 11
        assert j["newest_month"]["shown_share"] == min(
            v["shown_share"] for k, v in j.items() if k != "_comment"
        )
