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
        openalex=None,
        openalex_raises=False,
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

        def _oa(doi):
            if openalex_raises:
                raise verify.SourceUnavailable("openalex refused")
            return openalex

        monkeypatch.setattr(verify, "resolve_by_doi_epmc", _epmc)
        # Every tier, every time. A fixture that stops covering one turns the tests that use
        # it into live network calls -- silently, and only for the paths that reach the new
        # tier, which are exactly the interesting ones. Adding `openalex` to `TIER_SET`
        # without adding it here broke two tests into real HTTP requests on the first run.
        monkeypatch.setattr(verify, "resolve_by_doi_openalex", _oa)

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


class TestRefusalIsNotTheSameAsRejection:
    """C-32: `_s2_batch_post` returned None for two different states, and the DOI tier read
    both as refusal.

    A 429 means *S2 would not talk to us* — transient, and hardening it into a verdict about
    the paper is C-4 exactly. A 400 means *S2 rejected this id* — an answer, equivalent to
    the empty-record case, and the right move is to fall through to the next tier.

    Reading a 400 as refusal produced a specific, self-perpetuating wrong answer: a real
    paper with no abstract anywhere is classified `lookup_failed` instead of `unjudgeable`,
    which marks the row retryable, so every future invocation re-asks a question that can
    never come back differently. The first v2 baseline sweep hit it on ACM DOIs — POPL,
    PLDI, CACM — which is precisely the literature the v2 prompt exists to reach.
    """

    def _s2(self, monkeypatch, *, code=None, data=None):
        def fake(_ids, _fields, _key, _retries, _delay, status=None):
            if code is not None and status is not None:
                status["http"] = code
            return data

        monkeypatch.setattr(verify, "_s2_batch_post", fake)

    def test_a_400_is_an_answer_not_a_refusal(self, monkeypatch):
        self._s2(monkeypatch, code=400)
        assert verify.resolve_by_doi_s2("10.1145/3236774") is None

    def test_a_429_that_exhausted_its_retries_still_raises(self, monkeypatch):
        """No status recorded — the retry loop fell out the bottom, so S2 never answered."""
        self._s2(monkeypatch)
        with pytest.raises(verify.SourceUnavailable):
            verify.resolve_by_doi_s2("10.1145/3236774")

    def test_an_explicit_429_still_raises(self, monkeypatch):
        self._s2(monkeypatch, code=429)
        with pytest.raises(verify.SourceUnavailable):
            verify.resolve_by_doi_s2("10.1145/3236774")

    def test_the_acm_case_end_to_end_is_unjudgeable(self, monkeypatch):
        """The outcome that decides whether the row can ever be finished."""
        self._s2(monkeypatch, code=400)
        monkeypatch.setattr(verify, "doi_exists", lambda _d: True)
        monkeypatch.setattr(verify, "resolve_by_doi_epmc", lambda _d: None)
        monkeypatch.setattr(verify, "resolve_by_doi_openalex", lambda _d: None)
        assert verify.resolve_reference("10.1145/3236774", _FakeClient())[1] == "unjudgeable"

    def test_a_throttle_end_to_end_is_still_retryable(self, monkeypatch):
        """The other half — this must NOT become a verdict about the paper."""
        self._s2(monkeypatch, code=429)
        monkeypatch.setattr(verify, "doi_exists", lambda _d: True)
        assert verify.resolve_reference("10.1145/3236774", _FakeClient())[1] == "lookup_failed"


class TestTheStatusOutParameter:
    """The product's `None means skip` contract is unchanged; the code is reported beside it."""

    def _post(self, monkeypatch, exc):
        import urllib.request

        from reporadar import citations

        monkeypatch.setattr(citations.s2_rate, "wait_turn", lambda: None)
        monkeypatch.setattr(citations.s2_rate, "note_throttled", lambda: None)
        monkeypatch.setattr(citations.time, "sleep", lambda _s: None)

        def boom(*_a, **_kw):
            raise exc

        monkeypatch.setattr(urllib.request, "urlopen", boom)
        status: dict[str, int] = {}
        return citations._s2_batch_post(["DOI:x"], "title", None, 2, 0.0, status=status), status

    def test_a_rejection_records_its_code(self, monkeypatch):
        import urllib.error

        err = urllib.error.HTTPError("u", 400, "Bad Request", {}, None)  # type: ignore[arg-type]
        data, status = self._post(monkeypatch, err)
        assert data is None and status == {"http": 400}

    def test_an_exhausted_throttle_records_nothing(self, monkeypatch):
        """The 429 path `continue`s, so the loop falls out the bottom having recorded no
        answer — which is what tells the caller it never got one."""
        import urllib.error

        err = urllib.error.HTTPError("u", 429, "Too Many", {}, None)  # type: ignore[arg-type]
        data, status = self._post(monkeypatch, err)
        assert data is None and status == {}


class TestTheOpenAlexTier:
    """Tier 3, added 2026-08-26 for the ACM proceedings S2 and Europe PMC do not carry."""

    def _oa(self, monkeypatch, *, work=None, code=None, raises=False):
        def fake(_doi, status=None):
            if raises:
                raise RuntimeError("socket exploded")
            if code is not None and status is not None:
                status["http"] = code
            return work

        monkeypatch.setattr(verify, "fetch_work_by_doi", fake)

    WORK = {
        "title": "Build Systems à la Carte",
        "abstract_inverted_index": {"Build": [0], "systems": [1], "matter": [2]},
        "publication_date": "2018-07-30",
        "primary_topic": {"display_name": "Programming Languages"},
        "doi": "https://doi.org/10.1145/3236774",
    }

    def test_an_inverted_index_becomes_an_abstract(self, monkeypatch):
        self._oa(monkeypatch, work=self.WORK)
        paper = verify.resolve_by_doi_openalex("10.1145/3236774")
        assert paper is not None
        assert paper["abstract"] == "Build systems matter"
        assert paper["arxiv_id"] == "doi:10.1145/3236774"

    def test_a_record_with_no_abstract_is_not_a_resolution(self, monkeypatch):
        """A titled stub in the pool is worse than an honest `unjudgeable` — there would be
        nothing for the judge to score, and the paper would count as covered."""
        self._oa(monkeypatch, work={"title": "T", "abstract_inverted_index": None})
        assert verify.resolve_by_doi_openalex("10.1145/x") is None

    def test_a_404_is_an_answer(self, monkeypatch):
        self._oa(monkeypatch, code=404)
        assert verify.resolve_by_doi_openalex("10.5555/nope") is None

    def test_a_refusal_with_no_code_still_raises(self, monkeypatch):
        """C-32 again, one source over: retries spent without an answer must not harden."""
        self._oa(monkeypatch)
        with pytest.raises(verify.SourceUnavailable):
            verify.resolve_by_doi_openalex("10.1145/x")

    def test_an_adapter_exception_is_a_refusal_not_an_absence(self, monkeypatch):
        self._oa(monkeypatch, raises=True)
        with pytest.raises(verify.SourceUnavailable):
            verify.resolve_by_doi_openalex("10.1145/x")

    def test_it_rescues_what_the_earlier_tiers_miss(self, stub):
        """The whole point: S2 rejects it, Europe PMC is biomedical, OpenAlex has it.

        Driven through the `stub` fixture rather than by patching `fetch_work_by_doi`: the
        fixture replaces `resolve_by_doi_openalex` wholesale, so patching what that function
        calls has no effect and the test would pass or fail for a reason unrelated to its
        name. It did — this assertion failed the first time for exactly that.
        """
        resolved = {"arxiv_id": "doi:10.1145/3236774", "title": "Build Systems", "abstract": "A"}
        stub(exists=True, s2=None, epmc=None, openalex=resolved)
        paper, outcome = verify.resolve_reference("10.1145/3236774", _FakeClient())
        assert outcome == "resolved" and paper["title"] == "Build Systems"

    def test_it_does_not_rescue_what_nobody_has(self, stub, monkeypatch):
        """The residual stays honest: 5 Springer chapters and an Elsevier paper on the
        2026-08-26 sweep had no abstract in any of the four sources."""
        stub(exists=True, s2=None, epmc=None, openalex=None)
        assert verify.resolve_reference("10.1007/978-3-031-07085-3_29", _FakeClient())[1] == (
            "unjudgeable"
        )

    def test_arxiv_never_reaches_the_doi_tiers(self, stub):
        # every DOI tier set to raise: reaching any of them fails the test loudly
        stub(arxiv_paper=ARXIV_PAPER, openalex_raises=True, s2_raises=True, epmc_raises=True)
        assert verify.resolve_reference("1706.03762", _FakeClient())[1] == "resolved"


class TestTheTierSetIsRecordedBecauseUnjudgeableIsRelative:
    """`unjudgeable` says "none of the sources we asked had an abstract".

    That is permanent given a fixed tier list and obsolete the moment the list grows. The
    verdict therefore has to carry the list that produced it, or it outlives its evidence —
    the same lesson as `prompt_version`, one module over.
    """

    def test_the_tier_set_names_the_tiers_that_exist(self):
        assert verify.TIER_SET == ("arxiv", "doi.org", "s2", "europepmc", "openalex")
        # The names are labels for the reader; the resolvers they stand for are checked by
        # hand rather than by string-building, because `europepmc` is reached through
        # `resolve_by_doi_epmc` and a convention that quietly did not hold would make this
        # assertion pass for the wrong reason.
        for fn in ("resolve_by_doi_s2", "resolve_by_doi_epmc", "resolve_by_doi_openalex"):
            assert callable(getattr(verify, fn)), fn

    def test_growth_reopens_the_question(self):
        assert verify.tiers_grew(verify.LEGACY_TIER_SET)
        assert verify.tiers_grew(["arxiv"])

    def test_the_same_set_does_not(self):
        """The clause has to terminate: a row re-asked under the current set records it, and
        must not come back on the next invocation. Otherwise it is C-30 through the door
        built to prevent C-30."""
        assert not verify.tiers_grew(list(verify.TIER_SET))

    def test_reordering_is_not_growth(self):
        assert not verify.tiers_grew(list(reversed(verify.TIER_SET)))

    def test_losing_a_tier_is_not_growth(self):
        """Fewer places to look cannot overturn a not-found; re-asking would spend calls to
        confirm what we already know."""
        assert not verify.tiers_grew([*verify.TIER_SET, "crossref"])

    def test_an_unrecorded_set_means_the_one_before_openalex(self):
        """Rows written by the 2026-08-26 v2 sweep carry no field; they ran on four tiers."""
        assert verify.tiers_grew(None)
        assert set(verify.LEGACY_TIER_SET) < set(verify.TIER_SET)


class TestTheOpenAlexStatusOutParameter:
    def _post(self, monkeypatch, exc):
        import urllib.request

        from reporadar.sources import openalex

        monkeypatch.setattr(openalex.time, "sleep", lambda _s: None)

        def boom(*_a, **_kw):
            raise exc

        monkeypatch.setattr(urllib.request, "urlopen", boom)
        status: dict[str, int] = {}
        return openalex._request_json("https://x", max_retries=2, status=status), status

    def test_a_404_records_its_code(self, monkeypatch):
        import urllib.error

        err = urllib.error.HTTPError("u", 404, "Not Found", {}, None)  # type: ignore[arg-type]
        data, status = self._post(monkeypatch, err)
        assert data is None and status == {"http": 404}

    def test_an_exhausted_throttle_records_nothing(self, monkeypatch):
        import urllib.error

        err = urllib.error.HTTPError("u", 429, "Too Many", {}, None)  # type: ignore[arg-type]
        data, status = self._post(monkeypatch, err)
        assert data is None and status == {}
