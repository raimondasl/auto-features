"""Tests for reporadar.collector."""

from __future__ import annotations

import ast
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import arxiv
import pytest

from reporadar.collector import (
    CollectionError,
    _category_filter,
    _generate_bigram_queries,
    _query_with_retry,
    _result_to_paper,
    build_queries,
    collect_papers,
    to_plain_keywords,
)
from reporadar.config import ArxivConfig, QueriesConfig
from reporadar.profiler import RepoProfile


def _make_profile(**overrides) -> RepoProfile:
    defaults = {
        "keywords": [("transformers", 0.8), ("retrieval", 0.6), ("generation", 0.5)],
        "anchors": ["torch", "transformers"],
        "domains": ["deep learning", "NLP"],
    }
    defaults.update(overrides)
    return RepoProfile(**defaults)


def _make_arxiv_result(
    entry_id: str = "http://arxiv.org/abs/2401.12345v1",
    title: str = "Test Paper",
    summary: str = "A test abstract.",
    authors: list[str] | None = None,
    categories: list[str] | None = None,
    published: datetime | None = None,
) -> arxiv.Result:
    """Create a mock-like arxiv.Result for testing."""
    if authors is None:
        authors = ["Alice Smith"]
    if categories is None:
        categories = ["cs.CL"]
    if published is None:
        published = datetime.now(UTC)

    result = arxiv.Result(
        entry_id=entry_id,
        title=title,
        summary=summary,
        published=published,
        updated=published,
        categories=categories,
    )
    # Set authors via Author objects
    result.authors = [arxiv.Result.Author(name) for name in authors]
    return result


class TestCategoryFilter:
    def test_single_category(self) -> None:
        assert _category_filter(["cs.LG"]) == "cat:cs.LG"

    def test_multiple_categories(self) -> None:
        result = _category_filter(["cs.LG", "cs.CL"])
        assert result == "cat:cs.LG OR cat:cs.CL"

    def test_empty(self) -> None:
        assert _category_filter([]) == ""


class TestBuildQueries:
    def test_seed_queries_included(self) -> None:
        profile = _make_profile()
        queries_cfg = QueriesConfig(seed=["retrieval augmented generation"])
        arxiv_cfg = ArxivConfig(categories=["cs.CL"])

        queries = build_queries(profile, queries_cfg, arxiv_cfg)

        assert any('"retrieval augmented generation"' in q for q in queries)

    def test_auto_queries_from_keywords(self) -> None:
        profile = _make_profile()
        queries_cfg = QueriesConfig()
        arxiv_cfg = ArxivConfig(categories=["cs.LG"])

        queries = build_queries(profile, queries_cfg, arxiv_cfg)

        assert len(queries) > 0
        assert any("transformers" in q for q in queries)

    def test_category_filter_applied(self) -> None:
        profile = _make_profile()
        queries_cfg = QueriesConfig()
        arxiv_cfg = ArxivConfig(categories=["cs.LG"])

        queries = build_queries(profile, queries_cfg, arxiv_cfg)

        for q in queries:
            assert "cat:cs.LG" in q

    def test_no_categories(self) -> None:
        profile = _make_profile()
        queries_cfg = QueriesConfig()
        arxiv_cfg = ArxivConfig(categories=[])

        queries = build_queries(profile, queries_cfg, arxiv_cfg)

        assert len(queries) > 0
        for q in queries:
            assert "cat:" not in q

    def test_empty_profile_with_seeds(self) -> None:
        profile = _make_profile(keywords=[], anchors=[], domains=[])
        queries_cfg = QueriesConfig(seed=["test query"])
        arxiv_cfg = ArxivConfig(categories=["cs.AI"])

        queries = build_queries(profile, queries_cfg, arxiv_cfg)

        assert len(queries) == 1
        assert '"test query"' in queries[0]

    def test_empty_everything_fallback(self) -> None:
        profile = _make_profile(keywords=[], anchors=[], domains=[])
        queries_cfg = QueriesConfig()
        arxiv_cfg = ArxivConfig(categories=["cs.LG"])

        queries = build_queries(profile, queries_cfg, arxiv_cfg)

        # Should fall back to category-only query
        assert len(queries) == 1
        assert "cat:cs.LG" in queries[0]

    def test_max_auto_queries(self) -> None:
        keywords = [(f"term{i}", 0.9 - i * 0.1) for i in range(10)]
        profile = _make_profile(keywords=keywords)
        queries_cfg = QueriesConfig()
        arxiv_cfg = ArxivConfig(categories=[])

        queries = build_queries(profile, queries_cfg, arxiv_cfg, max_auto_queries=3)

        # Count only single-keyword queries (exclude bigram phrase queries)
        keyword_queries = [q for q in queries if '"' not in q]
        assert len(keyword_queries) == 3


class TestResultToPaper:
    def test_converts_fields(self) -> None:
        result = _make_arxiv_result(
            entry_id="http://arxiv.org/abs/2401.99999v1",
            title="My Paper",
            summary="Abstract text.",
            authors=["Alice", "Bob"],
            categories=["cs.CL", "cs.AI"],
        )

        paper = _result_to_paper(result)

        assert paper["arxiv_id"] == "2401.99999v1"
        assert paper["title"] == "My Paper"
        assert paper["abstract"] == "Abstract text."
        assert paper["authors"] == ["Alice", "Bob"]
        assert paper["categories"] == ["cs.CL", "cs.AI"]
        assert "url" in paper
        assert paper["published"] is not None


class TestCollectPapers:
    @patch("reporadar.collector.arxiv.Client")
    def test_collects_and_deduplicates(self, MockClient: MagicMock) -> None:
        now = datetime.now(UTC)
        results_q1 = [
            _make_arxiv_result("http://arxiv.org/abs/2401.00001v1", "Paper A", published=now),
            _make_arxiv_result("http://arxiv.org/abs/2401.00002v1", "Paper B", published=now),
        ]
        results_q2 = [
            _make_arxiv_result(  # duplicate
                "http://arxiv.org/abs/2401.00002v1",
                "Paper B",
                published=now,
            ),
            _make_arxiv_result("http://arxiv.org/abs/2401.00003v1", "Paper C", published=now),
        ]

        mock_client = MockClient.return_value
        mock_client.results.side_effect = [iter(results_q1), iter(results_q2)]

        cfg = ArxivConfig(max_results_per_query=50, lookback_days=30)
        papers = collect_papers(["q1", "q2"], cfg)

        assert len(papers) == 3
        ids = {p["arxiv_id"] for p in papers}
        assert ids == {"2401.00001v1", "2401.00002v1", "2401.00003v1"}

    @patch("reporadar.collector.arxiv.Client")
    def test_filters_old_papers(self, MockClient: MagicMock) -> None:
        now = datetime.now(UTC)
        old = datetime(2020, 1, 1, tzinfo=UTC)

        results = [
            _make_arxiv_result("http://arxiv.org/abs/2401.00001v1", "New Paper", published=now),
            _make_arxiv_result("http://arxiv.org/abs/2001.00001v1", "Old Paper", published=old),
        ]

        mock_client = MockClient.return_value
        mock_client.results.return_value = iter(results)

        cfg = ArxivConfig(max_results_per_query=50, lookback_days=14)
        papers = collect_papers(["q1"], cfg)

        assert len(papers) == 1
        assert papers[0]["title"] == "New Paper"

    @patch("reporadar.collector.arxiv.Search")
    @patch("reporadar.collector.arxiv.Client")
    def test_sort_by_defaults_to_relevance(
        self, MockClient: MagicMock, MockSearch: MagicMock
    ) -> None:
        """The default flipped to relevance when the shipped config became all-time: a
        newest-first sort over a 100-year window returns the last fortnight and nothing else,
        which is the old default wearing a new lookback."""
        MockClient.return_value.results.return_value = iter([])
        collect_papers(["q1"], ArxivConfig(max_results_per_query=50))
        assert MockSearch.call_args.kwargs["sort_by"] == arxiv.SortCriterion.Relevance

    @patch("reporadar.collector.arxiv.Search")
    @patch("reporadar.collector.arxiv.Client")
    def test_an_explicit_submitted_sort_is_still_honoured(
        self, MockClient: MagicMock, MockSearch: MagicMock
    ) -> None:
        """The recency digest is still supported, just no longer the default."""
        MockClient.return_value.results.return_value = iter([])
        collect_papers(
            ["q1"], ArxivConfig(max_results_per_query=50, lookback_days=14, sort_by="submitted")
        )
        assert MockSearch.call_args.kwargs["sort_by"] == arxiv.SortCriterion.SubmittedDate

    @patch("reporadar.collector.arxiv.Search")
    @patch("reporadar.collector.arxiv.Client")
    def test_sort_by_relevance_uses_relevance_criterion(
        self, MockClient: MagicMock, MockSearch: MagicMock
    ) -> None:
        # all-time discovery mode: relevance-sorted so seminal older papers surface.
        MockClient.return_value.results.return_value = iter([])
        cfg = ArxivConfig(max_results_per_query=50, lookback_days=36500, sort_by="relevance")
        collect_papers(["q1"], cfg)
        assert MockSearch.call_args.kwargs["sort_by"] == arxiv.SortCriterion.Relevance

    @patch("reporadar.collector.arxiv.Client")
    def test_matched_query_recorded(self, MockClient: MagicMock) -> None:
        now = datetime.now(UTC)
        results = [
            _make_arxiv_result("http://arxiv.org/abs/2401.00001v1", published=now),
        ]

        mock_client = MockClient.return_value
        mock_client.results.return_value = iter(results)

        cfg = ArxivConfig(max_results_per_query=50, lookback_days=30)
        papers = collect_papers(["all:transformers"], cfg)

        assert papers[0]["matched_query"] == "all:transformers"

    @patch("reporadar.collector.arxiv.Client")
    def test_empty_results(self, MockClient: MagicMock) -> None:
        mock_client = MockClient.return_value
        mock_client.results.return_value = iter([])

        cfg = ArxivConfig(max_results_per_query=50, lookback_days=14)
        papers = collect_papers(["q1"], cfg)

        assert papers == []

    @patch("reporadar.collector.arxiv.Client")
    def test_on_query_start_callback(self, MockClient: MagicMock) -> None:
        now = datetime.now(UTC)
        results = [
            _make_arxiv_result("http://arxiv.org/abs/2401.00001v1", published=now),
        ]

        mock_client = MockClient.return_value
        mock_client.results.side_effect = [iter(results), iter([])]

        calls: list[tuple[int, int, str]] = []

        def callback(idx: int, total: int, query: str) -> None:
            calls.append((idx, total, query))

        cfg = ArxivConfig(max_results_per_query=50, lookback_days=30)
        collect_papers(["q1", "q2"], cfg, on_query_start=callback)

        assert len(calls) == 2
        assert calls[0] == (0, 2, "q1")
        assert calls[1] == (1, 2, "q2")

    @patch("reporadar.collector.arxiv.Client")
    def test_no_callback_by_default(self, MockClient: MagicMock) -> None:
        mock_client = MockClient.return_value
        mock_client.results.return_value = iter([])

        cfg = ArxivConfig(max_results_per_query=50, lookback_days=14)
        # Should not raise when callback is None (default)
        papers = collect_papers(["q1"], cfg)
        assert papers == []


class TestQueryWithRetry:
    @patch("reporadar.collector.time.sleep")
    def test_succeeds_after_transient_failure(self, mock_sleep: MagicMock) -> None:
        now = datetime.now(UTC)
        good_result = _make_arxiv_result(
            "http://arxiv.org/abs/2401.00001v1",
            published=now,
        )

        mock_client = MagicMock()
        mock_client.results.side_effect = [
            ConnectionError("network down"),
            [good_result],
        ]

        search = MagicMock()
        results = _query_with_retry(mock_client, search, max_retries=3, base_delay=1.0)

        assert len(results) == 1
        assert mock_sleep.call_count == 1
        # First retry delay should be base_delay * 2^0 = 1.0
        mock_sleep.assert_called_with(1.0)

    @patch("reporadar.collector.time.sleep")
    def test_exhausted_raises_collection_error(self, mock_sleep: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client.results.side_effect = ConnectionError("always fails")

        search = MagicMock()
        with pytest.raises(CollectionError, match="3 attempts"):
            _query_with_retry(mock_client, search, max_retries=3, base_delay=1.0)

        assert mock_sleep.call_count == 2  # retries = max_retries - 1

    @patch("reporadar.collector.time.sleep")
    def test_backoff_delay_doubles(self, mock_sleep: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client.results.side_effect = [
            TimeoutError("timeout"),
            OSError("network"),
            ConnectionError("fail"),
        ]

        search = MagicMock()
        with pytest.raises(CollectionError):
            _query_with_retry(mock_client, search, max_retries=3, base_delay=2.0)

        # Should have slept twice with exponential backoff
        assert mock_sleep.call_count == 2
        calls = [c.args[0] for c in mock_sleep.call_args_list]
        assert calls[0] == 2.0  # base_delay * 2^0
        assert calls[1] == 4.0  # base_delay * 2^1


class TestBigramQueries:
    """The pairing mechanics, under the `adjacent` policy that shipped until 2026-08-12.

    Every test here passes `mode="adjacent"` explicitly. That is not ceremony: the default
    is now `verified`, under which a synthetic profile carrying no `corpus_phrases` emits
    nothing at all — so these tests would still pass while asserting nothing. A vacuous
    test that reads as a passing one is the failure mode this file has already been bitten
    by (see TestPlainKeywordTranslation).
    """

    def test_bigrams_generated_from_top_keywords(self) -> None:
        profile = _make_profile(
            keywords=[("retrieval", 0.9), ("augmented", 0.8), ("generation", 0.7)]
        )
        bigrams = _generate_bigram_queries(profile, mode="adjacent")
        assert len(bigrams) >= 1
        assert '"retrieval augmented"' in bigrams

    def test_bigrams_quoted_in_query(self) -> None:
        profile = _make_profile(
            keywords=[("retrieval", 0.9), ("augmented", 0.8), ("generation", 0.7)]
        )
        bigrams = _generate_bigram_queries(profile, mode="adjacent")
        assert bigrams
        for b in bigrams:
            assert b.startswith('"') and b.endswith('"')

    def test_no_bigrams_from_single_keyword(self) -> None:
        profile = _make_profile(keywords=[("retrieval", 0.9)])
        bigrams = _generate_bigram_queries(profile, mode="adjacent")
        assert bigrams == []

    def test_bigrams_added_to_build_queries(self) -> None:
        profile = _make_profile(
            keywords=[("retrieval", 0.9), ("augmented", 0.8), ("generation", 0.7)]
        )
        queries_cfg = QueriesConfig(bigrams="adjacent")
        arxiv_cfg = ArxivConfig(categories=["cs.CL"])

        queries = build_queries(profile, queries_cfg, arxiv_cfg)

        # Should have bigram queries (quoted phrases) in the query list
        has_bigram = any('"retrieval augmented"' in q for q in queries)
        assert has_bigram

    def test_shipped_default_needs_the_phrase_to_exist(self) -> None:
        """The same profile under the shipped policy: no corpus evidence, no phrase query.

        Pins the behaviour change itself, so the new default cannot be reverted silently.
        """
        profile = _make_profile(
            keywords=[("retrieval", 0.9), ("augmented", 0.8), ("generation", 0.7)]
        )
        queries = build_queries(profile, QueriesConfig(), ArxivConfig(categories=["cs.CL"]))
        assert not any('"retrieval augmented"' in q for q in queries), queries
        # ...and with the evidence present, it comes back.
        seen = _make_profile(
            keywords=[("retrieval", 0.9), ("augmented", 0.8), ("generation", 0.7)],
            corpus_phrases=["retrieval augmented"],
        )
        queries = build_queries(seen, QueriesConfig(), ArxivConfig(categories=["cs.CL"]))
        assert any('"retrieval augmented"' in q for q in queries), queries

    def test_short_words_filtered(self) -> None:
        profile = _make_profile(keywords=[("an", 0.9), ("to", 0.8), ("transformers", 0.7)])
        bigrams = _generate_bigram_queries(profile, mode="adjacent")
        # "an to" should be filtered (both < 4 chars)
        assert '"an to"' not in bigrams

    def test_max_bigrams_respected(self) -> None:
        keywords = [(f"word{i:02d}", 0.9 - i * 0.05) for i in range(10)]
        profile = _make_profile(keywords=keywords)
        bigrams = _generate_bigram_queries(profile, max_bigrams=2, mode="adjacent")
        assert len(bigrams) <= 2


class TestMultiWordTermsAreQuoted:
    """An unquoted space after an arXiv field prefix is OR, not AND.

    The profiler runs TF-IDF with ``ngram_range=(1, 2)``, so bigrams reach the query
    builder. Measured against the live API: ``all:speech recognition`` matches 246,802
    papers — more than ``all:recognition`` alone (224,631) and 7x more than the narrower
    term ``all:speech`` (34,239), i.e. the union. ``all:"speech recognition"`` matches
    6,845. Emitting these unquoted turned the *most specific* terms the profiler produces
    into the *broadest* queries it sends, and only the first 50 results were kept.
    """

    def _profile(self, keywords: list[str]) -> RepoProfile:
        return RepoProfile(
            keywords=[(k, 1.0) for k in keywords], anchors=[], domains=[], source_signals={}
        )

    def test_a_multi_word_keyword_is_sent_as_a_phrase(self) -> None:
        queries = build_queries(
            self._profile(["speech recognition"]),
            QueriesConfig(),
            ArxivConfig(categories=[]),
        )
        assert 'all:"speech recognition"' in queries
        assert "all:speech recognition" not in queries

    def test_a_single_word_keyword_is_not_quoted(self) -> None:
        # Quoting a single token is harmless but noisy; keep the emitted syntax minimal
        # so a query is readable in `rr audit` and in the arXiv access logs.
        queries = build_queries(
            self._profile(["whisper"]), QueriesConfig(), ArxivConfig(categories=[])
        )
        assert "all:whisper" in queries

    def test_every_emitted_query_has_balanced_quotes(self) -> None:
        # A stray quote silently changes what arXiv matches rather than erroring.
        queries = build_queries(
            self._profile(["speech recognition", "whisper", "audio", "large model"]),
            QueriesConfig(seed=["end to end asr"]),
            ArxivConfig(categories=["cs.CL", "eess.AS"]),
        )
        assert queries
        for q in queries:
            assert q.count('"') % 2 == 0, q


class TestBigramQueriesAreRealPhrases:
    def test_multi_word_terms_are_not_concatenated(self) -> None:
        """Joining a TF-IDF bigram to its neighbour built phrases no paper contains.

        The whisper repo's keywords include both ``speech`` and ``speech recognition``,
        and pairing adjacent terms produced ``"speech speech recognition"`` and
        ``"speech recognition recognition"`` — two of its three phrase queries, each
        matching nothing. A multi-word term is already a phrase and reaches arXiv as one
        through the keyword path.
        """
        profile = RepoProfile(
            keywords=[(k, 1.0) for k in ("speech", "speech recognition", "recognition", "audio")],
            anchors=[],
            domains=[],
            source_signals={},
        )
        # `adjacent` explicitly: this class is about the PAIRING, and under the shipped
        # `verified` policy a fixture with no corpus_phrases emits nothing, which would
        # satisfy the loop below without exercising anything.
        emitted = _generate_bigram_queries(profile, mode="adjacent")
        assert emitted
        for phrase in emitted:
            words = phrase.strip('"').split()
            assert len(words) == 2, f"{phrase} is not a two-word phrase"
            assert len(set(words)) == 2, f"{phrase} repeats a word"

    def test_bigrams_are_still_generated_from_single_word_terms(self) -> None:
        # The filter must not disable the feature outright.
        profile = RepoProfile(
            keywords=[(k, 1.0) for k in ("retrieval", "ranking", "index")],
            anchors=[],
            domains=[],
            source_signals={},
        )
        assert _generate_bigram_queries(profile, mode="adjacent")


class TestArxivThrottlingIsRetriedNotRaised:
    """`arxiv.ArxivError` is not an `OSError`, so a 429 escaped the retry handler.

    Every call site catches only `CollectionError` — `cli.py` and, critically,
    `watcher.py`, the scheduled path — so one throttle response ended an `rr watch` loop
    with a traceback. arXiv throttles for real: sustained polling from this project's own
    machine earned a roughly 70-minute IP block.
    """

    def test_http_error_is_retried_and_wrapped(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        import arxiv

        from reporadar import collector

        assert not issubclass(arxiv.ArxivError, OSError), (
            "arxiv errors are now OSErrors; this test no longer proves anything"
        )

        calls = {"n": 0}

        class Boom:
            def results(self, search: object) -> list[object]:
                calls["n"] += 1
                raise arxiv.HTTPError("https://export.arxiv.org/api/query", 1, 429)

        monkeypatch.setattr(collector.time, "sleep", lambda _s: None)
        with pytest.raises(collector.CollectionError):
            collector._query_with_retry(Boom(), object(), max_retries=3)
        # Used to assert exactly 3. A throttle is now retried against a TIME budget rather
        # than an attempt count, because 3 attempts (30s + 60s) gave up after 90 seconds
        # while a real arXiv throttle ran for ~15 minutes — and two benchmark repos were
        # left with an empty pool that scored as a legitimate zero.
        assert calls["n"] > 3, "a throttled request must be waited out, not given up on"

    def test_a_transient_error_still_yields_results_on_retry(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        import arxiv

        from reporadar import collector

        state = {"n": 0}

        class Flaky:
            def results(self, search: object) -> list[object]:
                state["n"] += 1
                if state["n"] == 1:
                    raise arxiv.HTTPError("https://export.arxiv.org/api/query", 1, 503)
                return ["paper"]

        monkeypatch.setattr(collector.time, "sleep", lambda _s: None)
        assert collector._query_with_retry(Flaky(), object(), max_retries=3) == ["paper"]


class TestPlainKeywordTranslation:
    """arXiv's boolean grammar is not a keyword query, and every non-arXiv source needs one.

    This drifted silently once. Callers bridged the gap with
    ``q.replace("all:", "").strip('"')``, written for an older query shape; when
    `build_queries` began wrapping queries as ``(all:"x") AND (cat:y)`` the transform
    stopped removing anything meaningful, and DBLP, bioRxiv, OpenAlex and Semantic Scholar
    were all sent arXiv syntax as a search string. IACR ePrint returns **zero** results for
    it, which is how it was finally noticed.

    So these tests are written against the **real output of build_queries**, not against
    hand-written strings. A hand-written fixture is exactly what would have kept passing.
    """

    def _real_queries(self) -> list[str]:
        profile = RepoProfile(
            keywords=[("key cryptography", 0.9), ("openssl", 0.8), ("bytes", 0.7)],
            anchors=["cryptography"],
            domains=["security"],
        )
        return build_queries(
            profile,
            QueriesConfig(seed=["side channel"]),
            ArxivConfig(categories=["cs.CR", "cs.LG"]),
        )

    def test_no_arxiv_syntax_survives_translation(self) -> None:
        for query in self._real_queries():
            plain = to_plain_keywords(query)
            assert "cat:" not in plain, plain
            assert "all:" not in plain, plain
            assert " AND " not in plain and " OR " not in plain, plain
            assert "(" not in plain and '"' not in plain, plain

    def test_the_actual_search_terms_survive(self) -> None:
        """Stripping syntax must not strip the words — an empty query finds nothing."""
        plains = [to_plain_keywords(q) for q in self._real_queries()]
        assert all(p.strip() for p in plains), plains
        assert any("cryptography" in p for p in plains)
        assert any("side channel" in p for p in plains)

    def test_category_only_queries_do_not_become_empty_searches(self) -> None:
        """A profile with no keywords falls back to a bare category filter; that carries
        no search terms at all, and sending '' to a keyword API is not a query."""
        assert to_plain_keywords("cat:cs.CR OR cat:cs.LG") == ""

    def test_the_old_one_liner_would_fail_these(self) -> None:
        """Pins the regression itself, so the fix cannot be quietly reverted."""
        query = '(all:"key cryptography") AND (cat:cs.CR)'
        old = query.replace("all:", "").strip('"')
        assert "cat:" in old and " AND " in old, "the old transform left syntax behind"
        assert to_plain_keywords(query) == "key cryptography"


class TestBigramModes:
    """The phrase-query policy, and what each mode is allowed to emit.

    `adjacent` pairs each keyword with its TF-IDF neighbour whether or not the two words
    belong together. Measured 2026-08-12 it built `"use page"` and `"page refer"` for
    duckdb, `"data cd"` for redis, `"server code"` for ruff. Asked those, DBLP returned a
    content analysis of social-media posts and a chemical-compound database; asked the
    benchmark's hand-written queries it returned *Incremental Fusion: Unifying Compiled and
    Vectorized Query Execution*. The source was answering exactly what it was asked.
    """

    def _profile(self, **overrides) -> RepoProfile:
        defaults = {
            "keywords": [("duckdb", 0.9), ("sql", 0.8), ("use", 0.7), ("page", 0.6)],
            "anchors": ["duckdb"],
            "domains": ["databases"],
            # "duckdb sql" occurs; "sql use" and "use page" do not.
            "corpus_phrases": ["duckdb sql", "parquet files"],
        }
        defaults.update(overrides)
        return RepoProfile(**defaults)

    def test_adjacent_emits_pairs_the_repo_never_contains(self) -> None:
        """Pins the defect itself, so a later 'cleanup' cannot quietly call it fixed."""
        got = _generate_bigram_queries(self._profile(), mode="adjacent")
        assert '"use page"' in got, got

    def test_verified_drops_them(self) -> None:
        got = _generate_bigram_queries(self._profile(), mode="verified")
        assert '"duckdb sql"' in got, got
        assert '"use page"' not in got, got

    def test_none_emits_nothing(self) -> None:
        assert _generate_bigram_queries(self._profile(), mode="none") == []

    def test_verified_without_corpus_phrases_emits_nothing(self) -> None:
        """A profile built before this field existed must not silently behave as `adjacent`.

        Absent evidence is not evidence of occurrence — the same rule the ranker uses for
        a missing signal. Erring the other way would resurrect the bug for every cached or
        hand-constructed profile.
        """
        profile = self._profile(corpus_phrases=[])
        assert _generate_bigram_queries(profile, mode="verified") == []

    def test_unknown_mode_is_an_error_not_a_default(self) -> None:
        with pytest.raises(ValueError, match="Unknown bigram mode"):
            _generate_bigram_queries(self._profile(), mode="adjacant")

    @pytest.mark.parametrize("mode", ["adjacent", "verified", "none"])
    def test_build_queries_threads_the_mode(self, mode: str) -> None:
        """The knob must reach the query strings, not just the helper."""
        profile = self._profile()
        queries = build_queries(profile, QueriesConfig(bigrams=mode), ArxivConfig(categories=[]))
        phrases = {q for q in queries if q.startswith('all:"')}
        expected = {f"all:{p}" for p in _generate_bigram_queries(profile, mode=mode)}
        assert expected <= phrases, (mode, queries)
        if mode != "adjacent":
            assert 'all:"use page"' not in queries, queries

    def test_keywords_survive_every_mode(self) -> None:
        """Dropping phrases must not drop retrieval: an empty query list finds nothing."""
        for mode in ("adjacent", "verified", "none"):
            queries = build_queries(
                self._profile(), QueriesConfig(bigrams=mode), ArxivConfig(categories=[])
            )
            assert any("duckdb" in q for q in queries), (mode, queries)


# Every module that bridges arXiv's query grammar to a keyword source. Adding a source
# outside this list is fine; bridging queries in a module outside it is what this guard
# is for, so a new entry here is the deliberate way to say "this file does that too".
_BRIDGING_MODULES = (
    # `pipeline.py` since 2026-08-16; `cli.py` bridged queries until the orchestrator
    # moved out of it, and `rr workspace update` -- the one collector still in `cli.py` --
    # is arXiv-only, so it never translates.
    ("src", "reporadar", "pipeline.py"),
    ("evals", "harness.py"),
    ("evals", "run_eval.py"),
)


class TestEveryBridgeUsesTheSharedTranslator:
    """The bug was never one bad transform — it was five call sites each free to invent one.

    Fixing `to_plain_keywords` fixed nothing on its own: the first attempt routed only
    IACR and DBLP through it and left Semantic Scholar, OpenAlex, bioRxiv (in `cli.py`)
    and the Tier A/S runner (`evals/run_eval.py`) still hand-rolling the broken one-liner.
    A test of the translator passes happily in that state, because the translator was
    correct and unused.

    So this asserts the *wiring*, by reading the source: any comprehension that maps over
    `queries` to feed a keyword API must call `to_plain_keywords`. It is the only check
    here that would have failed on the half-finished fix.
    """

    def _list_comps_over_queries(self, path: Path) -> list[ast.ListComp]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        found = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.ListComp) or len(node.generators) != 1:
                continue
            source = node.generators[0].iter
            # `queries[:5]` is a Subscript wrapping the Name; `queries` is the Name itself.
            if isinstance(source, ast.Subscript):
                source = source.value
            if isinstance(source, ast.Name) and source.id == "queries":
                found.append(node)
        return found

    @pytest.mark.parametrize("parts", _BRIDGING_MODULES, ids=lambda p: p[-1])
    def test_no_hand_rolled_query_translation(self, parts: tuple[str, ...]) -> None:
        path = Path(__file__).resolve().parents[1].joinpath(*parts)
        comps = self._list_comps_over_queries(path)
        assert comps, f"{path.name} no longer maps over `queries` — has the bridge moved?"
        for comp in comps:
            call = comp.elt
            rendered = ast.unparse(comp)
            assert isinstance(call, ast.Call), (
                f"{path.name}:{comp.lineno} maps over queries without calling anything: {rendered}"
            )
            name = call.func.id if isinstance(call.func, ast.Name) else None
            assert name == "to_plain_keywords", (
                f"{path.name}:{comp.lineno} hand-rolls query translation instead of using "
                f"the shared translator: {rendered}"
            )

    @pytest.mark.parametrize("parts", _BRIDGING_MODULES, ids=lambda p: p[-1])
    def test_the_broken_one_liner_is_gone(self, parts: tuple[str, ...]) -> None:
        """Belt to the AST braces: the exact defective expression, in any spelling."""
        path = Path(__file__).resolve().parents[1].joinpath(*parts)
        text = path.read_text(encoding="utf-8")
        for line_no, line in enumerate(text.splitlines(), start=1):
            if line.lstrip().startswith("#"):
                continue  # a comment may quote the bug; only live code is the failure
            assert 'replace("all:"' not in line, f"{path.name}:{line_no}: {line.strip()}"
