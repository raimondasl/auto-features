"""NR-42: the relevance filter, priced and closed. [P26 follow-up]

The "relevance condition on non-arXiv results" survived three entries: proposed in P22/P23,
retired in P24, reopened by C-33 in a narrower form, then aimed at the wrong term by C-34.
This is the $0 probe that closes it, and it closes it four ways:

* **It filters a term that is already positive** — +0.73/case for Europe PMC, +0.46 for
  OpenAlex. Filtering can only shrink that unless the filter is near-perfect.
* **The only filter we could build today is net negative.** Restricting non-arXiv papers to
  gate-3 costs 21 actionable papers to remove 7 non-actionable ones.
* **Neither instrument separates.** The gate-3 rate is **0.588 among actionable non-arXiv
  papers and 0.588 among non-actionable**; the fine-scale rescore's mean P is 0.842 against
  0.850, the wrong way round. These are the two stages that solved this exact problem for
  arXiv papers.
* **The cost is not in that term anyway.** 64% of OpenAlex's −1.22 is arXiv papers losing
  their place, and not in the digest — only 3–5 of 37 cases reach the 15-paper window. They
  lose it in the gate's shared 50-paper input.

What the probe *did* find is a different defect with a different name. A quarter of OpenAlex
candidates arrive with **no abstract**, the gate and rescore both read `paper["abstract"]`
with no guard, and among shown papers 4 of 17 non-actionable have none against 1 of 51
actionable. That is not a relevance problem — it is **void read as signal**, and this project
has a ledger of it (C-4, C-30, the 21% that measured nothing). The remedy is an
evidence-sufficiency guard, and it is deliberately NOT proposed as a way to make a source pay:
on Europe PMC, the only currently net-positive source, it is a complete no-op.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "evals" / "nonarxiv_evidence.json"
MRE = 1.04  # the benchmark's minimum resolvable effect, net@2/case (paper §3.5)


@pytest.fixture(scope="module")
def artifact() -> dict:
    return json.loads(FROZEN.read_text(encoding="utf-8"))


class TestTheCostIsNotWhereTheFilterAims:
    def test_the_digest_window_is_not_what_binds(self, artifact):
        """The premise correction. "Displacement" sounds like papers competing for the 15
        digest slots, and they are not: 3 to 5 of 37 cases reach the cap and the mean digest
        is 8.4. The competition is upstream, in the gate's `gate_depth: 50` input, which is
        shared across sources — so adding a source spends arXiv's gate slots."""
        assert artifact["config"]["digest_window"] == 15
        assert artifact["config"]["gate_depth"] == 50
        for arm, v in artifact["window_is_not_binding"].items():
            assert v["cases_at_the_cap"] <= 5, arm
            assert v["mean_digest"] < 10, arm

    def test_most_of_the_openalex_loss_is_slots_not_quality(self, artifact):
        """−1.22 = −0.78 slots + −0.44 quality. No filter on non-arXiv relevance touches
        the first term, which is the larger one."""
        oa = artifact["displacement_split"]["openalex"]
        assert oa["slots_term"] == -0.78
        assert oa["quality_term"] == -0.44
        assert oa["slots_term"] < oa["quality_term"] < 0
        assert oa["arxiv_dropped"] > oa["arxiv_added"]

    def test_europe_pmc_pays_the_same_term_and_covers_it(self, artifact):
        """Its swap is −0.19, almost all slots (−0.28) with quality neutral (+0.09). The
        difference from OpenAlex is not that it avoids displacement; it is that its own
        papers are worth more than the displacement costs."""
        e = artifact["displacement_split"]["epmc"]
        assert e["slots_term"] < 0
        assert e["quality_term"] > 0, "the arXiv papers it swaps in are no worse"
        assert artifact["counterfactuals"]["epmc"]["source_term"] > -e["swap_total"]


class TestNeitherInstrumentCanDiscriminate:
    def test_the_gate_score_shows_no_separation_at_all(self, artifact):
        """0.588 and 0.588 — not close, identical. The gate is the system's actionability
        judgment, and on non-arXiv papers its confidence tier carries nothing."""
        g = artifact["instruments"]["openalex"]["gate"]
        assert g["actionable_gate3_rate"] == g["non_actionable_gate3_rate"] == 0.588
        lo_a, hi_a = g["actionable_ci95"]
        lo_n, hi_n = g["non_actionable_ci95"]
        assert lo_n < hi_a and lo_a < hi_n, "the intervals overlap heavily"

    def test_the_europe_pmc_side_cannot_answer_this_question(self, artifact):
        """n = 1 non-actionable paper. Its gate-3 rate is 1.000 and that is not evidence of
        anything — recorded so nobody quotes it. C-33 was generalising from one source; this
        is the same shape at the level of one paper."""
        g = artifact["instruments"]["epmc"]["gate"]
        assert g["non_actionable_n"] == 1
        assert g["non_actionable_ci95"] == [0.207, 1.0], "an interval spanning most of [0,1]"

    def test_the_rescore_reaches_every_non_arxiv_band_paper(self, artifact):
        """The question worth asking before blaming the source: `score_papers` keys on
        `paper["arxiv_id"]`, which non-arXiv papers fill with their DOI, so nothing excludes
        them structurally — and nothing excludes them in fact. 28 of 28 and 13 of 13."""
        for arm in ("epmc", "openalex"):
            r = artifact["instruments"][arm]["rescore"]
            assert r["non_arxiv_band_scored"] == r["non_arxiv_band"], arm
            assert r["arxiv_band_scored"] == r["arxiv_band"], arm

    def test_and_it_does_not_rank_the_bad_ones_lower(self, artifact):
        """0.842 on the actionable, 0.850 on the non-actionable. The stage that solved the
        score-2 band for arXiv papers orders the non-arXiv band the wrong way round.

        Scoped honestly: these are papers the rescore ADMITTED, so the distribution is
        truncated at its own threshold. This bounds how well it orders within the admitted
        set — it is not a measurement of the signal it carries over papers it rejected, which
        this artifact cannot see.
        """
        r = artifact["instruments"]["openalex"]["rescore"]
        assert r["mean_p_non_actionable"] >= r["mean_p_actionable"]
        assert r["_caveat"].startswith("These are papers the rescore ADMITTED")


class TestEveryAvailableFilterLosesMoreThanItRemoves:
    def test_the_gate3_filter_is_net_negative_on_both_sources(self, artifact):
        """The only filter buildable today from a signal we already compute, and it makes
        both sources worse. On OpenAlex it discards 21 actionable papers to remove 7."""
        for arm in ("epmc", "openalex"):
            c = artifact["counterfactuals"][arm]
            assert c["gate3_only_filter"]["source_term"] < c["source_term"], arm
            assert (
                c["gate3_only_filter"]["dropped_actionable"]
                > c["gate3_only_filter"]["dropped_non_actionable"]
            ), arm

    def test_an_evidence_threshold_is_a_no_op_on_europe_pmc(self, artifact):
        """The reason this is filed as a defect fix and not as a source strategy. Europe PMC
        has 100% abstract coverage, so the guard cannot help the one source that currently
        pays — at every threshold up to 400 characters it changes nothing at all."""
        sweep = artifact["counterfactuals"]["epmc"]["evidence_threshold_sweep_chars"]
        base = artifact["counterfactuals"]["epmc"]["source_term"]
        for thr in ("1", "400"):
            assert sweep[thr]["source_term"] == base
            assert sweep[thr]["dropped_actionable"] == 0
            assert sweep[thr]["dropped_non_actionable"] == 0

    def test_and_it_does_not_rescue_openalex_either(self, artifact):
        """+0.46 → +0.65 on the source term, but the arm only goes −0.76 → −0.57: a
        display-time cut leaves the displacement term untouched. Even at 1000 characters,
        which discards 8 actionable papers, the arm stays negative."""
        c = artifact["counterfactuals"]["openalex"]
        sweep = c["evidence_threshold_sweep_chars"]
        assert sweep["1"]["source_term"] > c["source_term"]
        assert sweep["1"]["arm_if_slots_unchanged"] < 0
        assert sweep["1000"]["arm_if_slots_unchanged"] < 0


class TestTheDefectThatIsReal:
    def test_a_quarter_of_openalex_candidates_have_no_abstract(self, artifact):
        """26.5% of 10,501, against Europe PMC's 0 of 17,511. Both the gate and the rescore
        read `paper["abstract"]` with no guard, so those papers are scored on their titles."""
        oa = artifact["abstract_coverage_in_pool"]["openalex"]
        ep = artifact["abstract_coverage_in_pool"]["epmc"]
        assert oa["non_arxiv_with_abstract"] == 0.735
        assert ep["non_arxiv_with_abstract"] == 1.0
        assert oa["non_arxiv_mean_chars"] < ep["non_arxiv_mean_chars"]

    def test_the_papers_admitted_without_one_are_the_bad_ones(self, artifact):
        """4 of 17 non-actionable against 1 of 51 actionable, and half the abstract length
        among those that have one. The intervals separate — but only just, at n = 17, so this
        is stated as a defect in the scoring path rather than as a calibrated effect. C-33 and
        C-34 were both cases of reading more from a small margin than it held."""
        b = artifact["abstract_coverage_when_shown"]["openalex"]
        assert b["non_actionable"]["no_abstract"] == 4
        assert b["non_actionable"]["n"] == 17
        assert b["actionable"]["no_abstract"] == 1
        assert b["actionable"]["n"] == 51
        assert b["non_actionable"]["no_abstract_rate"] > b["actionable"]["no_abstract_rate"]
        assert b["actionable"]["ci95"][1] < b["non_actionable"]["ci95"][0] + 0.01, (
            "barely disjoint; do not quote this as a settled effect size"
        )
        assert b["non_actionable"]["mean_chars"] < b["actionable"]["mean_chars"] / 1.5


class TestWhatWouldReopenIt:
    def test_the_oracle_ceiling_clears_the_noise_floor(self, artifact):
        """The prize is real even though the instrument is not. A perfect discriminator with
        zero displacement takes OpenAlex to +7.11 against the control's +5.73 — headroom of
        +1.38, above the benchmark's MRE of 1.04. So the item is closed on the absence of an
        instrument, not on the absence of value, and a genuinely better discriminator would
        reopen it."""
        oa = artifact["oracle_ceiling"]["openalex"]
        assert oa["headroom_over_control"] == 1.38
        assert oa["headroom_over_control"] > MRE

    def test_but_every_realistic_fix_falls_below_it(self, artifact):
        """Europe PMC's ceiling is +0.78 and the best measured OpenAlex counterfactual leaves
        its arm negative. Fixing the slot term alone would be worth +0.28 on Europe PMC. None
        of these clears 1.04, so they are unmeasurable on this benchmark even if built —
        which is the selection rule refusing them before a run is paid for."""
        assert artifact["oracle_ceiling"]["epmc"]["headroom_over_control"] < MRE
        epmc_slots = -artifact["displacement_split"]["epmc"]["slots_term"]
        assert epmc_slots < MRE
        best = max(
            v["arm_if_slots_unchanged"]
            for v in artifact["counterfactuals"]["openalex"][
                "evidence_threshold_sweep_chars"
            ].values()
        )
        assert best < 0, "no evidence threshold makes the OpenAlex arm positive"
