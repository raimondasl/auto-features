"""The evidence-sufficiency guard: neither LLM stage scores a paper it cannot read. [NR-42]

Both stages describe a candidate to the model as a title plus ``abstract[:1500]``, and both
read that field with no guard. A paper whose abstract never arrived was still sent, still
scored 0-3, and still admitted to a digest on the strength of its title. That is void read as
signal, and it is measurable: **26.5% of OpenAlex candidates carry no abstract** against 0 of
17,511 from Europe PMC, and among papers that reached digests 4 of 17 non-actionable ones had
none against 1 of 51 actionable.

The change is deliberately narrow, and the tests below pin each boundary:

* **Absence, not brevity.** A short abstract is evidence, merely less of it. A character
  threshold would be tuning the gate against net@2 through a back door, which is exactly what
  NR-42 declined to do when it closed the relevance-filter item.
* **No backfill.** The freed gate budget is not spent further down the ranking. That would
  change which papers the gate sees — a separate decision needing its own measurement — and
  this change stays a pure removal so its effect is readable.
* **`enough_scored` counts attempts, not the band.** A paper that was never attempted must
  not read as a failed call, or the guard could abandon the whole rescore over papers it
  correctly declined — the void-as-null error one line below the guard against it.
* **Not configurable.** Every other stage's failure policy is an invariant. A flag whose
  off-position restores "score papers you cannot read" is a footgun, not a choice.

The expected benchmark effect is ~nothing, and that is the point: on Europe PMC, the only
currently net-positive source, abstract coverage is 100% and the guard is a complete no-op.
It is a correctness fix and is judged as one.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from reporadar.evidence import has_abstract, partition_by_evidence
from reporadar.finescale import enough_scored, score_papers
from reporadar.triage import triage_papers

_PROFILE = SimpleNamespace(
    keywords=[("retrieval", 0.5)],
    anchors=["faiss"],
    domains=["information retrieval"],
)
_OK = {"arxiv_id": "2401.00001", "title": "A Method", "abstract": "We improve ANN search."}


class TestWhatCountsAsMissing:
    def test_the_four_ways_an_abstract_can_be_absent_are_one_answer(self) -> None:
        """A missing key, an explicit None, an empty string and whitespace all reach the
        prompt as `Abstract: ` followed by nothing. Which of the four it was is a fact about
        the adapter, not about the paper, so the guard must not distinguish them."""
        assert not has_abstract({"arxiv_id": "x"})
        assert not has_abstract({"arxiv_id": "x", "abstract": None})
        assert not has_abstract({"arxiv_id": "x", "abstract": ""})
        assert not has_abstract({"arxiv_id": "x", "abstract": "   \n\t "})

    def test_real_text_passes(self) -> None:
        assert has_abstract(_OK)

    def test_a_short_abstract_is_still_evidence(self) -> None:
        """The guard is about ABSENCE, not brevity. NR-42 measured that non-actionable
        non-arXiv papers had roughly half the abstract length of actionable ones (729 vs 1393
        characters) — and deliberately did not turn that into a threshold, because picking
        one would be tuning the gate against net@2 rather than fixing a defect."""
        assert has_abstract({"arxiv_id": "x", "abstract": "Short."})
        assert has_abstract({"arxiv_id": "x", "abstract": "a"})

    def test_the_partition_is_complete_and_order_preserving(self) -> None:
        papers = [
            {"arxiv_id": "1", "abstract": "text"},
            {"arxiv_id": "2", "abstract": ""},
            {"arxiv_id": "3", "abstract": "more text"},
            {"arxiv_id": "4"},
        ]
        keep, skip = partition_by_evidence(papers)
        assert [p["arxiv_id"] for p in keep] == ["1", "3"]
        assert [p["arxiv_id"] for p in skip] == ["2", "4"]
        assert len(keep) + len(skip) == len(papers)


class TestTheGateDoesNotScoreWhatItCannotRead:
    def test_an_abstract_less_paper_is_never_sent_to_the_model(self) -> None:
        """Not "sent and then discarded" — never sent. The call costs money and its answer
        would be about a title, so the skip has to happen before the transport."""
        papers = [
            {"arxiv_id": "2401.00001", "title": "Has one", "abstract": "Real text here."},
            {"arxiv_id": "2401.00002", "title": "Has none", "abstract": ""},
        ]
        seen = []

        def record(prompt, cfg, **kw):
            seen.append(prompt)
            return '{"score": 3, "reason": "ok"}'

        with patch("reporadar.triage.complete", side_effect=record):
            out = triage_papers(papers, _PROFILE, SimpleNamespace(), top_k=10)

        assert set(out) == {"2401.00001"}
        assert len(seen) == 1, "one paper, one call"
        assert "Has none" not in seen[0]

    def test_it_is_omitted_rather_than_scored_zero(self) -> None:
        """The distinction the whole module exists for. A 0 is a confident rejection and
        would rank the paper below genuinely-judged bad ones; an omission means "not a
        confident Top Pick", which is what we actually know."""
        papers = [{"arxiv_id": "2401.00002", "title": "No abstract", "abstract": None}]
        with patch("reporadar.triage.complete", return_value='{"score": 0, "reason": "no"}'):
            out = triage_papers(papers, _PROFILE, SimpleNamespace(), top_k=10)
        assert out == {}, "absent from the result, not present with a 0"

    def test_the_freed_budget_is_not_backfilled(self) -> None:
        """`top_k` is applied BEFORE the guard, so skipping a paper shortens the batch
        rather than pulling the next-ranked paper up into it. Backfilling would change which
        papers the gate sees, which is a separate decision with its own measurement; keeping
        this a pure removal is what makes its effect readable."""
        papers = [
            {"arxiv_id": "1", "title": "a", "abstract": "text"},
            {"arxiv_id": "2", "title": "b", "abstract": ""},
            {"arxiv_id": "3", "title": "c", "abstract": "text"},
        ]
        with patch("reporadar.triage.complete", return_value='{"score": 2, "reason": "x"}'):
            out = triage_papers(papers, _PROFILE, SimpleNamespace(), top_k=2)
        assert set(out) == {"1"}, "paper 3 is NOT pulled up to replace the skipped paper 2"


class TestTheRescoreCarriesTheSameGuard:
    def test_it_skips_abstract_less_papers(self) -> None:
        """It should almost never fire here — the band has already been through a triage that
        drops them. It stays because `score_papers` is called directly by the eval harness and
        by tests, and a guard that only exists upstream is one refactor from being gone."""
        papers = [
            {"arxiv_id": "1", "title": "a", "abstract": "text"},
            {"arxiv_id": "2", "title": "b", "abstract": "  "},
        ]
        with patch("reporadar.finescale.top_logprobs", return_value=[("8", 1.0)]):
            out = score_papers(papers, _PROFILE, SimpleNamespace())
        assert set(out) == {"1"}


class TestASkippedPaperIsNotAFailedCall:
    def test_enough_scored_measures_attempts_not_the_band(self) -> None:
        """The trap this guard could have walked into. `enough_scored` exists to notice that
        a whole stage broke — a bad key, a network outage — and skip the fine-scale gate
        rather than abstain by accident. Its denominator is what was ATTEMPTED.

        Counting deliberately-skipped papers as attempts would let a band that is merely
        abstract-poor look like a broken run, abandoning the rescore for the papers that
        *were* readable. The pipeline therefore partitions the band first and passes the
        readable count. This test pins the arithmetic that makes that matter: a 10-paper band
        with 6 unreadable scores 4/4 as attempted, and 4/10 as the band.
        """
        assert enough_scored(4, 4, 0.5) is True, "4 attempted, 4 scored"
        assert enough_scored(4, 10, 0.5) is False, "the same run, judged against the band"

    def test_a_wholly_unreadable_band_does_not_look_like_an_outage(self) -> None:
        """Zero attempts is not a failure rate; `enough_scored` returns False on an empty
        denominator, and the pipeline never reaches it because `if band:` is false. The run
        skips the gate the same way it does when there is no band at all."""
        assert enough_scored(0, 0, 0.5) is False
        keep, skip = partition_by_evidence([{"arxiv_id": "1", "abstract": ""}, {"arxiv_id": "2"}])
        assert keep == [] and len(skip) == 2


class TestItIsAnInvariantNotASetting:
    def test_no_configuration_switches_it_off(self) -> None:
        """A flag whose off-position restores "score papers you cannot read" would be a
        footgun rather than a choice, and §8.4 is a whole section about the cost of a default
        that under-delivers. If a knob for this ever appears, this test should be the thing
        that argues with it."""
        from pathlib import Path

        import reporadar.config as config

        text = Path(config.__file__).read_text(encoding="utf-8")
        for name in ("require_abstract", "min_abstract", "skip_no_abstract", "abstract_guard"):
            assert name not in text, f"{name} suggests the guard became optional"
