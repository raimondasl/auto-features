"""The embedding does not discriminate actionability — and NR-42's evidence was conditioned.

[NR-58]

NR-42 closed the non-arXiv relevance-filter item on "no instrument discriminates" and named
the reopening condition: a genuinely better discriminator, against an oracle ceiling of
**+1.38** on OpenAlex that clears the benchmark's MRE of 1.04. It tested two instruments,
both LLM stages. This tests the third — the ranker's **dense embedding**, the one scoring
component non-arXiv papers do not escape.

**Registered answer: no.** AUC **0.578**, CI [0.415, 0.673], against a registered bar of 0.65
with an interval excluding 0.5. The item stays closed, now on two instruments rather than
one, and the second is better powered: 100 non-actionable papers against NR-42's 17, drawn
from the judge cache rather than from what the pipeline chose to show.

**The finding is the diagnostic, not the headline.** The same signal reads **0.096** on
NR-42's shown-only panel and **0.612** on papers the pipeline passed over, with disjoint
intervals. An instrument evaluated on the set it helped select looks *worse than it is* — a
paper admitted despite a low score on that instrument got in on something else, and that
something else correlates with being actionable. NR-42's conclusion survives this probe's
wider evidence; the argument it published for it does not.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "evals" / "embedding_discriminator.json"


@pytest.fixture(scope="module")
def art() -> dict:
    return json.loads(FROZEN.read_text(encoding="utf-8"))


class TestTheRegisteredAnswerIsNo:
    def test_the_bar_predates_the_number(self, art) -> None:
        pre = art["pre_registered"]
        assert pre["written_before_any_auc_was_computed"] is True
        assert pre["reopens_filter_item_if_auc_at_least"] == 0.65
        assert pre["and_ci_excludes"] == 0.5

    def test_the_wide_panel_misses_the_bar_and_covers_a_half(self, art) -> None:
        p = art["panels"]["non_arxiv_wide"]
        assert p["auc"] == pytest.approx(0.578, abs=0.005)
        assert p["auc"] < art["pre_registered"]["reopens_filter_item_if_auc_at_least"]
        assert p["ci95"][0] < 0.5 < p["ci95"][1]
        assert art["verdict"]["reopens_filter_item"] is False

    def test_it_is_better_powered_than_the_probe_it_extends(self, art) -> None:
        """NR-42's OpenAlex panel had 17 non-actionable papers, every one of which had
        already survived ranking and the gate. This has 100, from the verdict cache."""
        wide = art["panels"]["non_arxiv_wide"]
        shown = art["panels"]["non_arxiv_nr42_shown"]
        assert wide["n_non_actionable"] == 100
        assert shown["n_non_actionable"] == 17
        assert wide["n_pairs"] > 9 * shown["n_pairs"]

    def test_the_prediction_was_right(self, art) -> None:
        """Registered: "near 0.5-0.6 covering 0.5, arXiv control above it". 0.578 covering
        0.5, control 0.586. Recorded because a registered prediction that is never checked
        against the outcome is decoration."""
        assert "0.5-0.6" in art["pre_registered"]["prediction"]
        wide = art["panels"]["non_arxiv_wide"]["auc"]
        control = art["panels"]["arxiv_control"]["auc"]
        assert 0.5 <= wide <= 0.6
        assert control > wide


class TestTheControlMakesTheNullReadable:
    def test_the_instrument_is_weak_everywhere_not_off_arxiv(self, art) -> None:
        """The distinction the control exists to draw. An embedding that ordered arXiv well
        and failed off it would be a domain finding and an argument for a non-arXiv-specific
        remedy. **0.586 against 0.578** is not that: it is a weak actionability signal
        wherever it is pointed — which is not a defect, because it is weighted 1.5 in the
        shipped ranker for RELEVANCE, and relevance is not what is being asked of it here."""
        arxiv = art["panels"]["arxiv_control"]
        wide = art["panels"]["non_arxiv_wide"]
        assert abs(arxiv["auc"] - wide["auc"]) < 0.05
        assert arxiv["n_pairs"] > 29000  # and this one IS resolved: CI [0.531, 0.643]
        assert arxiv["ci95"][0] > 0.5

    def test_a_weak_signal_is_not_no_signal(self, art) -> None:
        """0.586 excluding 0.5 on 29k pairs says the embedding carries *some* actionability
        information. Reporting it as "no signal" would be the void-not-null error pointed the
        other way — and it is still far under the bar a filter would need."""
        assert art["panels"]["arxiv_control"]["excludes_half"] is True
        assert art["panels"]["arxiv_control"]["auc"] < 0.65


class TestNR42sEvidenceWasSelectionConditioned:
    """The reason this probe is worth more than its own null."""

    def test_the_same_signal_inverts_on_the_panel_nr42_used(self, art) -> None:
        shown = art["panels"]["non_arxiv_nr42_shown"]
        assert shown["auc"] == pytest.approx(0.096, abs=0.005)
        assert shown["ci95"][1] < 0.5  # decisively inverted, not merely flat

    def test_and_recovers_on_the_papers_the_pipeline_passed_over(self, art) -> None:
        """0.096 -> 0.612 with **disjoint intervals**. Same instrument, same cases, same
        computation; the only difference is whether the pipeline chose the paper."""
        c = art["verdict"]["conditioning"]
        assert c["non_arxiv_shown_auc"] < c["non_arxiv_not_shown_auc"]
        assert c["non_arxiv_intervals_disjoint"] is True

    def test_the_arxiv_panel_agrees_in_direction_and_is_not_claimed_as_more(self, art) -> None:
        """0.485 shown vs 0.554 not-shown on 20k pairs — the same direction at far higher n,
        but the intervals OVERLAP. Recorded as directional, because claiming it as
        established would be the exact overreach this probe caught NR-42 in."""
        c = art["verdict"]["conditioning"]
        assert c["arxiv_shown_auc"] < c["arxiv_not_shown_auc"]
        assert c["arxiv_intervals_disjoint"] is False

    def test_what_it_does_and_does_not_overturn(self, art) -> None:
        """NR-42's *conclusion* — no filter is buildable — survives, because this probe's
        wider panel independently says the same thing. Its *argument* does not: "the rescore
        scores them the wrong way round" was read off a panel where a genuinely
        discriminating instrument would also read low, and NR-42's own artifact carried the
        truncation caveat it then read past."""
        c = art["verdict"]["conditioning"]["_comment"]
        assert "CONCLUSION survives" in c
        assert "ARGUMENT does" in c
        assert art["verdict"]["reopens_filter_item"] is False


class TestTheLabelledSetIsBuiltHonestly:
    def test_ids_are_matched_forwards_never_parsed_out_of_a_filename(self, art) -> None:
        """`cs/0412098v3` is stored as `cs_0412098v3`, and `is_arxiv_id` says False for that
        stem — so classifying the verdict cache BY FILENAME counts old-style arXiv ids as
        non-arXiv. It inflated this labelled set from 380 to 471 before the direction was
        fixed. One rule, delegated: C-14, C-31."""
        source = (ROOT / "evals" / "embedding_discriminator.py").read_text(encoding="utf-8")
        assert "verdicts[case].get(_judge_stem(pid))" in source
        assert "is_arxiv_id(pid)" in source

    def test_unlocatable_verdicts_are_absent_and_counted(self, art) -> None:
        """1,925 of 4,391 cached verdicts name a paper no frozen pool holds. They are out of
        the labelled set, not scored — and the count is on the artifact so a reader sees the
        selection rather than inferring it from a total."""
        p = art["provenance"]
        assert p["n_verdicts_cached"] == 4391
        assert p["n_located_with_text"] == 2466
        assert p["n_unlocatable"] == p["n_verdicts_cached"] - p["n_located_with_text"]
        assert len(p["located_from"]) >= 5

    def test_the_interval_resamples_cases_not_papers(self, art) -> None:
        """Papers inside one repository share a pool, a profile and a repo embedding, and
        the 37 cases contribute counts spanning two orders of magnitude. A paper-level
        bootstrap would report an interval for a study nobody ran (C-7)."""
        source = (ROOT / "evals" / "embedding_discriminator.py").read_text(encoding="utf-8")
        assert "cluster bootstrap over cases" in art["pre_registered"]["interval"]
        assert "usable[rng.randrange(len(usable))]" in source
