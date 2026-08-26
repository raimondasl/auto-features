"""Resolution past arXiv, and the four outcomes it has to keep apart.

`evals/verify.py` resolved arXiv ids only, and `BASELINE_PROMPT` demands an `arxiv_id`, so
the two agreed with each other and the benchmark could not see past arXiv. Widening it is the
prerequisite for the v2 prompt: a baseline allowed to recommend a *Nature* paper, resolved by
an arXiv-only verifier, produces a pick that cannot be verified, cannot be judged, and
vanishes — while the run looks like it searched.

The tests here are almost all about **classification**, because that is where the damage is.
Four outcomes, and only one of them is the model's fault:

* `resolved` — a title and an abstract, enough to judge.
* `hallucinated` — proven not to exist. Counts against the recommender.
* `lookup_failed` — our infrastructure could not answer. Transient, retryable, never the
  model's fault (C-4 was paid for scoring an arXiv throttle as an honest zero).
* `unjudgeable` — the DOI resolves, so the paper is real, but nobody we ask has an abstract.
  Permanent and NOT retryable, which is why folding it into `lookup_failed` would strand a
  case forever (C-30, one layer down).

Everything is stubbed. These must run with no network, because a test that needs Semantic
Scholar to be up is a test that reports the weather.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))

import verify  # noqa: E402


class _FakeClient:
    """Stands in for `arxiv.Client`; never touches the network."""

    def __init__(self, behaviour="found"):
        self.behaviour = behaviour

    def results(self, search, *_a, **_kw):
        if self.behaviour == "error":
            raise RuntimeError("network is down")
        if self.behaviour == "empty":
            return iter(())
        raise AssertionError("unused")


@pytest.fixture
def stub(monkeypatch):
    """Control every tier independently: arXiv, doi.org, S2, Europe PMC."""

    def apply(
        *,
        arxiv_paper=None,
        arxiv_raises=False,
        exists=None,
        s2=None,
        s2_raises=False,
        epmc=None,
        epmc_raises=False,
    ):
        def _by_id(client, ref):
            if arxiv_raises:
                raise verify.SourceUnavailable("arxiv down")
            return arxiv_paper

        def _s2(doi, api_key=None):
            if s2_raises:
                raise verify.SourceUnavailable("s2 refused")
            return s2

        def _epmc(doi):
            if epmc_raises:
                raise verify.SourceUnavailable("epmc refused")
            return epmc

        monkeypatch.setattr(verify, "resolve_by_id", _by_id)
        monkeypatch.setattr(verify, "doi_exists", lambda doi: exists)
        monkeypatch.setattr(verify, "resolve_by_doi_s2", _s2)
        monkeypatch.setattr(verify, "resolve_by_doi_epmc", _epmc)

    return apply


PAPER = {"arxiv_id": "doi:10.1/x", "title": "T", "abstract": "A", "categories": [], "url": "u"}
ARXIV_PAPER = {
    "arxiv_id": "1706.03762",
    "title": "T",
    "abstract": "A",
    "categories": [],
    "url": "u",
}


class TestClassification:
    def test_arxiv_id_resolves_through_the_arxiv_tier(self, stub):
        stub(arxiv_paper=ARXIV_PAPER)
        assert verify.resolve_reference("1706.03762", _FakeClient())[1] == "resolved"

    def test_arxiv_id_with_no_match_is_a_hallucination(self, stub):
        stub(arxiv_paper=None)
        assert verify.resolve_reference("2999.99999", _FakeClient())[1] == "hallucinated"

    def test_arxiv_outage_is_not_a_hallucination(self, stub):
        stub(arxiv_raises=True)
        assert verify.resolve_reference("1706.03762", _FakeClient())[1] == "lookup_failed"

    def test_doi_resolved_by_s2(self, stub):
        stub(exists=True, s2=PAPER)
        paper, outcome = verify.resolve_reference("10.1038/x", _FakeClient())
        assert outcome == "resolved" and paper == PAPER

    def test_doi_falls_through_to_europepmc(self, stub):
        """The tier that exists because S2 returns null for bioRxiv preprints."""
        stub(exists=True, s2=None, epmc=PAPER)
        assert verify.resolve_reference("10.1101/x", _FakeClient())[1] == "resolved"

    def test_a_doi_that_does_not_resolve_is_a_hallucination(self, stub):
        stub(exists=False)
        assert verify.resolve_reference("10.9999/nope", _FakeClient())[1] == "hallucinated"

    def test_a_real_doi_nobody_has_an_abstract_for_is_unjudgeable(self, stub):
        stub(exists=True, s2=None, epmc=None)
        assert verify.resolve_reference("10.1/real", _FakeClient())[1] == "unjudgeable"

    def test_an_unreachable_registry_is_a_lookup_failure(self, stub):
        stub(exists=None)
        assert verify.resolve_reference("10.1/x", _FakeClient())[1] == "lookup_failed"

    @pytest.mark.parametrize("which", ["s2", "epmc"])
    def test_a_metadata_source_refusing_is_retryable_not_a_verdict(self, stub, which):
        """A 429 must never harden into a claim about the paper."""
        stub(exists=True, s2_raises=(which == "s2"), epmc_raises=(which == "epmc"), s2=None)
        assert verify.resolve_reference("10.1/x", _FakeClient())[1] == "lookup_failed"

    def test_something_that_is_neither_an_id_nor_a_doi(self, stub):
        """`publication/2256929` — the C-25 id. Our gap, so not charged to the model."""
        stub()
        assert verify.resolve_reference("publication/2256929", _FakeClient())[1] == "unjudgeable"


class TestTheSharedArxivPredicate:
    """C-14, found again by widening this module.

    `paper_id.is_arxiv_id` used `[a-z-]+/\\d{7}`, so it answered True for
    `publication/2256929` while `verify.extract_arxiv_ids` — hardened for exactly that id in
    C-25 — answered False. Two rules for one question, in the module that exists to hold one.
    The archive list now lives in `paper_id` and this module imports it.
    """

    def test_the_bogus_id_is_rejected_by_both(self):
        from reporadar.paper_id import is_arxiv_id

        assert not is_arxiv_id("publication/2256929")
        assert verify.extract_arxiv_ids("see publication/2256929") == []

    def test_real_old_style_ids_still_pass_both(self):
        from reporadar.paper_id import dedup_id, is_arxiv_id

        for pid in ("hep-th/9901001", "cond-mat/0403023", "cs.LG/0501001", "astro-ph/0605086"):
            assert is_arxiv_id(pid), pid
            assert verify.extract_arxiv_ids(f"see {pid}") == [pid], pid
        # and versions are still stripped on the old era (the C-14 half of the fix)
        assert dedup_id("cond-mat/0403023v2") == "cond-mat/0403023"

    def test_there_is_only_one_archive_list(self):
        from reporadar.paper_id import ARXIV_ARCHIVES

        assert verify._ARCHIVES is ARXIV_ARCHIVES


class TestCounters:
    def test_each_outcome_lands_in_its_own_counter(self, stub, monkeypatch):
        calls = iter(
            [
                (ARXIV_PAPER, "resolved"),
                (None, "hallucinated"),
                (None, "lookup_failed"),
                (None, "unjudgeable"),
            ]
        )
        monkeypatch.setattr(verify, "resolve_reference", lambda *_a, **_kw: next(calls))
        papers, hall, failed, unjudge = verify.resolve_references(["a", "b", "c", "d"], [])
        assert (len(papers), hall, failed, unjudge) == (1, 1, 1, 1)

    def test_titles_stay_arxiv_only(self, stub, monkeypatch):
        """A free-text title has no registry to check against; cross-source would guess."""
        seen: list[str] = []
        monkeypatch.setattr(
            verify,
            "resolve_by_title",
            lambda c, t: seen.append(t) or ARXIV_PAPER,  # noqa: E731
        )
        monkeypatch.setattr(
            verify,
            "resolve_reference",
            lambda *_a, **_kw: pytest.fail("titles must not go through DOI resolution"),
        )
        papers, *_ = verify.resolve_references([], ["Attention Is All You Need"])
        assert seen == ["Attention Is All You Need"] and len(papers) == 1

    def test_duplicates_collapse_on_the_shared_id_rule(self, stub, monkeypatch):
        monkeypatch.setattr(
            verify, "resolve_reference", lambda ref, *_a, **_kw: (dict(ARXIV_PAPER), "resolved")
        )
        papers, *_ = verify.resolve_references(["1706.03762", "1706.03762v3"], [])
        assert len(papers) == 1
