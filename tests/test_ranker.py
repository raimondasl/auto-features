"""Tests for reporadar.ranker."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from reporadar.config import QueriesConfig, RankingConfig
from reporadar.profiler import RepoProfile
from reporadar.ranker import (
    compute_exclude_penalty,
    format_score_explanation,
    rank_papers,
    score_category_match,
    score_distribution,
    score_keyword_overlap,
    score_paper,
    score_recency,
)


def _make_paper(**overrides) -> dict:
    now = datetime.now(UTC)
    base = {
        "arxiv_id": "2401.12345v1",
        "title": "Retrieval Augmented Generation with Long Context Transformers",
        "authors": ["Alice Smith"],
        "abstract": "We propose a novel retrieval augmented generation framework "
        "that leverages long context transformers for improved question answering.",
        "categories": ["cs.CL", "cs.LG"],
        "published": now.isoformat(),
        "url": "http://arxiv.org/abs/2401.12345v1",
    }
    base.update(overrides)
    return base


def _make_profile(**overrides) -> RepoProfile:
    defaults = {
        "keywords": [
            ("retrieval", 0.8),
            ("transformers", 0.7),
            ("generation", 0.6),
            ("embeddings", 0.4),
        ],
        "anchors": ["torch", "transformers"],
        "domains": ["deep learning", "NLP"],
    }
    defaults.update(overrides)
    return RepoProfile(**defaults)


class TestScoreKeywordOverlap:
    def test_full_overlap(self) -> None:
        paper = _make_paper()
        profile = _make_profile(keywords=[("retrieval", 0.5), ("transformers", 0.5)])
        score = score_keyword_overlap(paper, profile)
        assert score == pytest.approx(1.0)

    def test_partial_overlap(self) -> None:
        paper = _make_paper()
        profile = _make_profile(keywords=[("retrieval", 0.5), ("quantum", 0.5)])
        score = score_keyword_overlap(paper, profile)
        assert 0.4 < score < 0.6

    def test_no_overlap(self) -> None:
        paper = _make_paper(
            title="Quantum Computing Advances",
            abstract="A new quantum error correction code.",
        )
        profile = _make_profile(keywords=[("retrieval", 0.5), ("transformers", 0.5)])
        score = score_keyword_overlap(paper, profile)
        assert score == 0.0

    def test_empty_profile(self) -> None:
        paper = _make_paper()
        profile = _make_profile(keywords=[])
        assert score_keyword_overlap(paper, profile) == 0.0

    def test_bigram_match(self) -> None:
        # Terms like "long context" should match if individual tokens match
        paper = _make_paper(title="Long Context Window Models")
        profile = _make_profile(keywords=[("long", 0.5), ("context", 0.5)])
        score = score_keyword_overlap(paper, profile)
        assert score == pytest.approx(1.0)


class TestScoreCategoryMatch:
    def test_full_match(self) -> None:
        paper = _make_paper(categories=["cs.CL", "cs.LG"])
        score = score_category_match(paper, ["cs.CL", "cs.LG"])
        assert score == pytest.approx(1.0)

    def test_partial_match(self) -> None:
        paper = _make_paper(categories=["cs.CL"])
        score = score_category_match(paper, ["cs.CL", "cs.LG"])
        assert score == pytest.approx(0.5)

    def test_no_match(self) -> None:
        paper = _make_paper(categories=["cs.CV"])
        score = score_category_match(paper, ["cs.CL", "cs.LG"])
        assert score == 0.0

    def test_empty_targets(self) -> None:
        paper = _make_paper(categories=["cs.CL"])
        assert score_category_match(paper, []) == 0.0

    def test_empty_paper_categories(self) -> None:
        paper = _make_paper(categories=[])
        assert score_category_match(paper, ["cs.CL"]) == 0.0


class TestScoreRecency:
    def test_today_is_1(self) -> None:
        now = datetime.now(UTC)
        paper = _make_paper(published=now.isoformat())
        score = score_recency(paper, lookback_days=14)
        assert score > 0.95

    def test_old_paper_is_0(self) -> None:
        old = datetime(2020, 1, 1, tzinfo=UTC)
        paper = _make_paper(published=old.isoformat())
        score = score_recency(paper, lookback_days=14)
        assert score == 0.0

    def test_midpoint(self) -> None:
        mid = datetime.now(UTC) - timedelta(days=7)
        paper = _make_paper(published=mid.isoformat())
        score = score_recency(paper, lookback_days=14)
        assert 0.4 < score < 0.6

    def test_decays_linearly(self) -> None:
        now = datetime.now(UTC)
        scores = []
        for days_ago in [1, 5, 10, 13]:
            pub = now - timedelta(days=days_ago)
            paper = _make_paper(published=pub.isoformat())
            scores.append(score_recency(paper, lookback_days=14))
        # Should be monotonically decreasing
        assert scores == sorted(scores, reverse=True)


class TestExcludePenalty:
    def test_no_exclude(self) -> None:
        paper = _make_paper()
        assert compute_exclude_penalty(paper, []) == 1.0

    def test_single_match(self) -> None:
        paper = _make_paper(title="A Survey of RAG Methods")
        penalty = compute_exclude_penalty(paper, ["survey"])
        assert penalty == pytest.approx(0.5)

    def test_multiple_matches(self) -> None:
        paper = _make_paper(
            title="A Survey and Benchmark of RAG Methods",
            abstract="We benchmark various survey approaches.",
        )
        penalty = compute_exclude_penalty(paper, ["survey", "benchmark"])
        assert penalty == pytest.approx(0.25)

    def test_no_match(self) -> None:
        paper = _make_paper()
        penalty = compute_exclude_penalty(paper, ["quantum", "biology"])
        assert penalty == 1.0


class TestCachedEmbeddingsEquivalence:
    def test_cached_dict_matches_uncached_compute(self) -> None:
        from unittest.mock import patch

        import numpy as np

        p1 = _make_paper(arxiv_id="2401.1v1")
        p2 = _make_paper(arxiv_id="2401.2v1")
        profile = _make_profile()
        repo_emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        vecs = {
            "2401.1v1": np.array([1.0, 0.5, 0.0], dtype=np.float32),
            "2401.2v1": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        }
        cfg = RankingConfig(w_keyword=1.0, w_category=0.5, w_recency=0.3, w_embedding=1.0)

        # Uncached path: rank_papers calls compute_paper_embedding per paper.
        with patch(
            "reporadar.embeddings.compute_paper_embedding",
            side_effect=lambda p: vecs[p["arxiv_id"]],
        ):
            uncached = rank_papers(
                [p1, p2], profile, cfg, QueriesConfig(), ["cs.CL"], repo_embedding=repo_emb
            )
        # Cached path: the same vectors supplied as a dict (no encoding).
        cached = rank_papers(
            [p1, p2],
            profile,
            cfg,
            QueriesConfig(),
            ["cs.CL"],
            repo_embedding=repo_emb,
            paper_embeddings=vecs,
        )

        u = {s["arxiv_id"]: s["embedding_score"] for s in uncached}
        c = {s["arxiv_id"]: s["embedding_score"] for s in cached}
        assert u == c
        assert u["2401.1v1"] is not None and u["2401.1v1"] > 0


class TestMissingCategories:
    def test_absent_categories_do_not_dilute_the_score(self) -> None:
        # Non-arXiv sources (S2 recommendations, DBLP, bioRxiv) carry no
        # categories. Scoring that *missing* signal as a zero would silently
        # handicap them against arXiv papers, so the weight is dropped instead.
        profile = _make_profile()
        cfg = RankingConfig(w_keyword=1.0, w_category=0.5, w_recency=0.3)
        no_cats = score_paper(_make_paper(categories=[]), profile, cfg, QueriesConfig(), ["cs.CL"])
        # Same paper, same keyword/recency signal, but category present and matching.
        with_cats = score_paper(
            _make_paper(categories=["cs.CL"]), profile, cfg, QueriesConfig(), ["cs.CL"]
        )
        assert no_cats["category_score"] == 0.0  # still reported as 0
        # A full category match should score at least as well, and the
        # category-less paper must not be dragged below its own keyword+recency mix.
        assert no_cats["score_total"] > 0.0
        assert with_cats["score_total"] >= no_cats["score_total"]

    def test_category_still_counts_when_present_but_unmatched(self) -> None:
        profile = _make_profile()
        cfg = RankingConfig(w_keyword=1.0, w_category=0.5, w_recency=0.3)
        mismatch = score_paper(
            _make_paper(categories=["q-bio.NC"]), profile, cfg, QueriesConfig(), ["cs.CL"]
        )
        absent = score_paper(_make_paper(categories=[]), profile, cfg, QueriesConfig(), ["cs.CL"])
        # A real non-match is penalized; an absent signal is not. Under `omit` — the
        # shipped default — that IS the bias: see TestAbsentCategoryPolicy below, which
        # measures how large it is rather than only noting its direction.
        assert mismatch["score_total"] < absent["score_total"]


class TestAbsentCategoryPolicy:
    """`ranking.absent_category` — what happens to `w_category` with no categories.

    `omit` was chosen so a missing signal would not handicap non-arXiv papers, and it does
    the reverse: an arXiv paper is averaged over keyword AND category, an uncategorised one
    over keyword alone, so at equal keyword relevance the uncategorised paper wins. The
    absent-signal rule is right when missingness is random and wrong here, because having
    categories is perfectly correlated with being an arXiv paper.

    The arithmetic is the whole claim, so it is asserted exactly rather than by direction.
    """

    def _cfg(self, mode: str) -> RankingConfig:
        # w_recency 0 to isolate the category axis: with recency in the mix the numbers
        # below would still order the same way but would stop being checkable by hand.
        return RankingConfig(w_keyword=1.0, w_category=0.5, w_recency=0.0, absent_category=mode)

    def _score(self, mode: str, categories: list[str]) -> float:
        paper = _make_paper(categories=categories)
        profile = _make_profile()
        out = score_paper(
            paper, profile, self._cfg(mode), QueriesConfig(), ["cs.CL"], absent_category_score=0.5
        )
        return float(out["score_total"])

    def test_the_advantage_appears_exactly_when_the_category_match_is_weak(self) -> None:
        """The precise rule, which is narrower than "uncategorised papers win".

        With weights w_kw and w_cat the two totals are (kw + w·cat)/(1 + w) and kw, so the
        uncategorised paper wins **iff kw > cat**. A paper in exactly the right category
        still beats it; a paper in a merely adjacent one does not. Partial and zero
        category matches are the common case in a real pool, which is why this shows up as
        systematic displacement rather than as an occasional upset.
        """
        absent = self._score("omit", [])
        assert absent > self._score("omit", ["cs.LG"])  # no match: cat 0.0 < kw
        assert absent < self._score("omit", ["cs.CL"])  # full match: cat 1.0 > kw

    def test_zero_removes_the_advantage(self) -> None:
        """Even against a non-matching arXiv paper, absence no longer wins."""
        assert self._score("zero", []) <= self._score("zero", ["cs.LG"])

    def test_zero_is_the_harshest_treatment(self) -> None:
        """It scores absence as a real non-match, which is the handicap `omit` avoided."""
        assert self._score("zero", []) < self._score("omit", [])

    def test_impute_sits_between_zero_and_omit(self) -> None:
        """An uncategorised paper is treated as an AVERAGE paper on this axis.

        Not the best (omit, which is what a perfect category match would earn) and not the
        worst (zero, a non-match) — the only one of the three that does not encode a claim
        about non-arXiv papers being systematically better or worse than arXiv ones.
        """
        assert self._score("zero", []) <= self._score("impute", []) <= self._score("omit", [])

    def test_a_categorised_paper_is_unaffected_by_the_policy(self) -> None:
        """The knob must only touch papers with nothing to score."""
        scores = {mode: self._score(mode, ["cs.CL"]) for mode in ("omit", "zero", "impute")}
        assert len(set(scores.values())) == 1, scores

    def test_impute_falls_back_to_zero_when_nothing_in_the_pool_has_categories(self) -> None:
        """A mean over an empty set is not a number, and imputing 1.0 there would hand
        every paper in an all-non-arXiv pool a perfect category score."""
        from reporadar.ranker import rank_papers

        papers = [_make_paper(arxiv_id=f"ss:{i}", categories=[]) for i in range(3)]
        ranked = rank_papers(
            papers, _make_profile(), self._cfg("impute"), QueriesConfig(), ["cs.CL"]
        )
        zeroed = rank_papers(papers, _make_profile(), self._cfg("zero"), QueriesConfig(), ["cs.CL"])
        assert [r["score_total"] for r in ranked] == [r["score_total"] for r in zeroed]


class TestSpecterComponent:
    def test_boost_and_gate(self) -> None:
        paper = _make_paper(
            title="An unrelated topic entirely",
            abstract="nothing here matches the profile",
            categories=["cs.CV"],
            published=datetime(2020, 1, 1, tzinfo=UTC).isoformat(),
        )
        profile = _make_profile()
        on = RankingConfig(w_specter=1.0)
        base = score_paper(paper, profile, on, QueriesConfig(), ["cs.CL"])
        boosted = score_paper(paper, profile, on, QueriesConfig(), ["cs.CL"], specter_score=1.0)
        assert boosted["score_total"] > base["score_total"]
        assert boosted["specter_score"] == 1.0

        # w_specter = 0 disables the component entirely.
        off = score_paper(
            paper, profile, RankingConfig(), QueriesConfig(), ["cs.CL"], specter_score=1.0
        )
        assert off["score_total"] == base["score_total"]

    def test_rank_papers_maps_scores_by_id(self) -> None:
        p1, p2 = _make_paper(arxiv_id="2401.1v1"), _make_paper(arxiv_id="2401.2v1")
        scores = rank_papers(
            [p1, p2],
            _make_profile(),
            RankingConfig(w_specter=5.0),
            QueriesConfig(),
            ["cs.CL"],
            specter={"2401.1v1": 1.0, "2401.2v1": 0.0},
        )
        by_id = {s["arxiv_id"]: s for s in scores}
        assert by_id["2401.1v1"]["specter_score"] == 1.0
        assert by_id["2401.2v1"]["specter_score"] == 0.0
        assert by_id["2401.1v1"]["score_total"] > by_id["2401.2v1"]["score_total"]

    def test_explanation_includes_specter(self) -> None:
        cfg = RankingConfig(w_specter=2.0)
        score = score_paper(
            _make_paper(), _make_profile(), cfg, QueriesConfig(), ["cs.CL"], specter_score=0.5
        )
        assert "specter" in format_score_explanation(score, cfg)


class TestCommunityComponent:
    def test_boost_and_gate(self) -> None:
        paper = _make_paper(
            title="An unrelated topic entirely",
            abstract="nothing here matches the profile",
            categories=["cs.CV"],
            published=datetime(2020, 1, 1, tzinfo=UTC).isoformat(),
        )
        profile = _make_profile()
        on = RankingConfig(w_community=1.0)
        base = score_paper(paper, profile, on, QueriesConfig(), ["cs.CL"])
        boosted = score_paper(paper, profile, on, QueriesConfig(), ["cs.CL"], community_score=1.0)
        assert boosted["score_total"] > base["score_total"]
        assert boosted["community_score"] == 1.0

        # w_community = 0 disables the component entirely.
        off = score_paper(
            paper, profile, RankingConfig(), QueriesConfig(), ["cs.CL"], community_score=1.0
        )
        assert off["score_total"] == base["score_total"]

    def test_absent_signal_is_not_a_zero(self) -> None:
        # A paper HF has never seen must not be ranked below an identical paper
        # that merely has *few* upvotes — absent is not the same as unpopular.
        paper = _make_paper()
        profile = _make_profile()
        cfg = RankingConfig(w_community=1.0)
        absent = score_paper(paper, profile, cfg, QueriesConfig(), ["cs.CL"])
        low = score_paper(paper, profile, cfg, QueriesConfig(), ["cs.CL"], community_score=0.1)
        assert absent["community_score"] is None
        assert absent["score_total"] > low["score_total"]

    def test_rank_papers_maps_scores_by_id(self) -> None:
        p1, p2 = _make_paper(arxiv_id="2401.1v1"), _make_paper(arxiv_id="2401.2v1")
        scores = rank_papers(
            [p1, p2],
            _make_profile(),
            RankingConfig(w_community=5.0),
            QueriesConfig(),
            ["cs.CL"],
            community={"2401.1v1": 1.0},
        )
        by_id = {s["arxiv_id"]: s for s in scores}
        assert by_id["2401.1v1"]["community_score"] == 1.0
        # Not in the mapping → no signal, not a zero.
        assert by_id["2401.2v1"]["community_score"] is None
        assert by_id["2401.1v1"]["score_total"] > by_id["2401.2v1"]["score_total"]

    def test_explanation_includes_community(self) -> None:
        cfg = RankingConfig(w_community=2.0)
        score = score_paper(
            _make_paper(), _make_profile(), cfg, QueriesConfig(), ["cs.CL"], community_score=0.5
        )
        assert "community" in format_score_explanation(score, cfg)


class TestAttentionComponent:
    def test_boost_and_gate(self) -> None:
        paper = _make_paper()
        profile = _make_profile()
        on = RankingConfig(w_attention=1.0)
        base = score_paper(paper, profile, on, QueriesConfig(), ["cs.CL"])
        boosted = score_paper(paper, profile, on, QueriesConfig(), ["cs.CL"], attention_score=1.0)
        assert boosted["score_total"] > base["score_total"]
        assert boosted["attention_score"] == 1.0

        off = score_paper(
            paper, profile, RankingConfig(), QueriesConfig(), ["cs.CL"], attention_score=1.0
        )
        assert off["score_total"] == base["score_total"]

    def test_never_discussed_is_absent_not_zero(self) -> None:
        # HN discusses a handful of papers a week, so ~every paper lacks this signal.
        # If absence read as 0.0, enabling w_attention would penalize the whole corpus.
        paper, profile = _make_paper(), _make_profile()
        cfg = RankingConfig(w_attention=2.0)
        absent = score_paper(paper, profile, cfg, QueriesConfig(), ["cs.CL"])
        low = score_paper(paper, profile, cfg, QueriesConfig(), ["cs.CL"], attention_score=0.2)
        assert absent["attention_score"] is None
        assert absent["score_total"] > low["score_total"]

    def test_rank_papers_maps_scores_by_id(self) -> None:
        p1, p2 = _make_paper(arxiv_id="2401.1v1"), _make_paper(arxiv_id="2401.2v1")
        scores = rank_papers(
            [p1, p2],
            _make_profile(),
            RankingConfig(w_attention=5.0),
            QueriesConfig(),
            ["cs.CL"],
            attention={"2401.1v1": 1.0},
        )
        by_id = {s["arxiv_id"]: s for s in scores}
        assert by_id["2401.1v1"]["attention_score"] == 1.0
        assert by_id["2401.2v1"]["attention_score"] is None

    def test_explanation_includes_attention(self) -> None:
        cfg = RankingConfig(w_attention=2.0)
        score = score_paper(
            _make_paper(), _make_profile(), cfg, QueriesConfig(), ["cs.CL"], attention_score=0.5
        )
        assert "attention" in format_score_explanation(score, cfg)


class TestWithdrawnPenalty:
    def test_withdrawn_paper_is_demoted(self) -> None:
        paper, profile = _make_paper(), _make_profile()
        cfg = RankingConfig()
        clean = score_paper(paper, profile, cfg, QueriesConfig(), ["cs.CL"])
        flagged = score_paper(paper, profile, cfg, QueriesConfig(), ["cs.CL"], withdrawn=True)
        assert flagged["score_total"] < clean["score_total"]

    def test_penalty_is_multiplicative_so_strength_elsewhere_cannot_escape_it(self) -> None:
        """A withdrawn paper must not reach Top Picks by scoring well on every signal.

        This is why withdrawal is a multiplier rather than one more weighted
        component: as a component it would be outvoted by keyword + category +
        recency + attention all firing at once.
        """
        perfect = _make_paper()  # matches the profile, current, right category
        cfg = RankingConfig(w_attention=1.0, w_specter=1.0, w_community=1.0)
        kwargs: dict[str, float] = {
            "attention_score": 1.0,
            "specter_score": 1.0,
            "community_score": 1.0,
        }
        clean = score_paper(perfect, _make_profile(), cfg, QueriesConfig(), ["cs.CL"], **kwargs)
        flagged = score_paper(
            perfect, _make_profile(), cfg, QueriesConfig(), ["cs.CL"], withdrawn=True, **kwargs
        )
        assert clean["score_total"] > 0.5  # would be a Top Pick
        assert flagged["score_total"] < 0.2  # below MAYBE_THRESHOLD -> Muted

    def test_penalty_leaves_the_paper_visible(self) -> None:
        # Not zero: the reader is better served by "this was withdrawn" than by a
        # paper silently vanishing from a digest they may have seen elsewhere.
        flagged = score_paper(
            _make_paper(),
            _make_profile(),
            RankingConfig(),
            QueriesConfig(),
            ["cs.CL"],
            withdrawn=True,
        )
        assert flagged["score_total"] > 0.0

    def test_configurable_and_disableable(self) -> None:
        paper, profile = _make_paper(), _make_profile()
        clean = score_paper(paper, profile, RankingConfig(), QueriesConfig(), ["cs.CL"])
        disabled = score_paper(
            paper,
            profile,
            RankingConfig(withdrawn_penalty=1.0),
            QueriesConfig(),
            ["cs.CL"],
            withdrawn=True,
        )
        assert disabled["score_total"] == clean["score_total"]

    def test_rank_papers_applies_it_by_id(self) -> None:
        p1, p2 = _make_paper(arxiv_id="2401.1v1"), _make_paper(arxiv_id="2401.2v1")
        scores = rank_papers(
            [p1, p2],
            _make_profile(),
            RankingConfig(),
            QueriesConfig(),
            ["cs.CL"],
            withdrawn={"2401.1v1"},
        )
        by_id = {s["arxiv_id"]: s for s in scores}
        assert by_id["2401.1v1"]["score_total"] < by_id["2401.2v1"]["score_total"]


class TestCitationProximity:
    def _low_scoring_paper(self) -> dict:
        # No keyword/category overlap and old → a low base score, so a proximity
        # boost is unambiguously visible.
        return _make_paper(
            title="An unrelated topic entirely",
            abstract="nothing here matches the repo profile at all",
            categories=["cs.CV"],
            published=datetime(2020, 1, 1, tzinfo=UTC).isoformat(),
        )

    def test_boost_raises_score(self) -> None:
        paper = self._low_scoring_paper()
        profile = _make_profile()
        cfg = RankingConfig(w_citation_proximity=1.0)
        base = score_paper(paper, profile, cfg, QueriesConfig(), ["cs.CL"])
        boosted = score_paper(
            paper, profile, cfg, QueriesConfig(), ["cs.CL"], citation_proximity_score=1.0
        )
        assert boosted["score_total"] > base["score_total"]

    def test_no_boost_when_weight_zero(self) -> None:
        paper = self._low_scoring_paper()
        profile = _make_profile()
        with_score = score_paper(
            paper,
            profile,
            RankingConfig(w_citation_proximity=0.0),
            QueriesConfig(),
            ["cs.CL"],
            citation_proximity_score=1.0,
        )
        without = score_paper(paper, profile, RankingConfig(), QueriesConfig(), ["cs.CL"])
        assert with_score["score_total"] == without["score_total"]


class TestScorePaper:
    def test_returns_expected_keys(self) -> None:
        paper = _make_paper()
        profile = _make_profile()
        result = score_paper(
            paper,
            profile,
            RankingConfig(),
            QueriesConfig(),
            ["cs.CL", "cs.LG"],
        )
        assert "arxiv_id" in result
        assert "score_total" in result
        assert "keyword_score" in result
        assert "category_score" in result
        assert "recency_score" in result

    def test_weights_affect_total(self) -> None:
        paper = _make_paper()
        profile = _make_profile()

        # Heavy keyword weight
        kw_heavy = score_paper(
            paper,
            profile,
            RankingConfig(w_keyword=10.0, w_category=0.0, w_recency=0.0),
            QueriesConfig(),
            ["cs.CL"],
        )
        # Heavy recency weight
        rec_heavy = score_paper(
            paper,
            profile,
            RankingConfig(w_keyword=0.0, w_category=0.0, w_recency=10.0),
            QueriesConfig(),
            ["cs.CL"],
        )

        # With different weights, totals should differ
        assert kw_heavy["score_total"] != rec_heavy["score_total"]

    def test_exclude_reduces_total(self) -> None:
        paper = _make_paper(title="A Survey of RAG")
        profile = _make_profile()

        without_exclude = score_paper(
            paper,
            profile,
            RankingConfig(),
            QueriesConfig(),
            ["cs.CL"],
        )
        with_exclude = score_paper(
            paper,
            profile,
            RankingConfig(),
            QueriesConfig(exclude=["survey"]),
            ["cs.CL"],
        )

        assert with_exclude["score_total"] < without_exclude["score_total"]


class TestEdgeCases:
    def test_recency_with_invalid_date(self) -> None:
        paper = _make_paper(published="not-a-date")
        score = score_recency(paper, lookback_days=14)
        assert score == 0.0

    def test_recency_with_missing_published(self) -> None:
        paper = _make_paper()
        del paper["published"]
        score = score_recency(paper, lookback_days=14)
        assert score == 0.0

    def test_recency_future_date(self) -> None:
        future = datetime.now(UTC) + timedelta(days=5)
        paper = _make_paper(published=future.isoformat())
        score = score_recency(paper, lookback_days=14)
        assert score == 1.0

    def test_keyword_overlap_empty_abstract(self) -> None:
        paper = _make_paper(title="", abstract="")
        profile = _make_profile()
        score = score_keyword_overlap(paper, profile)
        assert score == 0.0

    def test_all_weights_zero(self) -> None:
        paper = _make_paper()
        profile = _make_profile()
        result = score_paper(
            paper,
            profile,
            RankingConfig(w_keyword=0.0, w_category=0.0, w_recency=0.0),
            QueriesConfig(),
            ["cs.CL"],
        )
        assert result["score_total"] == 0.0

    def test_category_superset(self) -> None:
        # Paper has more categories than target — should still cap at 1.0
        paper = _make_paper(categories=["cs.CL", "cs.LG", "cs.AI"])
        score = score_category_match(paper, ["cs.CL"])
        assert score == pytest.approx(1.0)


class TestScoreNormalization:
    def test_default_weights_normalized(self) -> None:
        paper = _make_paper()
        profile = _make_profile()
        result = score_paper(
            paper,
            profile,
            RankingConfig(),
            QueriesConfig(),
            ["cs.CL", "cs.LG"],
        )
        assert result["score_total"] <= 1.0

    def test_all_max_scores_equals_one(self) -> None:
        # Paper matching all keywords, all categories, published now → 1.0
        now = datetime.now(UTC)
        paper = _make_paper(
            title="retrieval transformers generation embeddings",
            abstract="retrieval transformers generation embeddings",
            categories=["cs.CL", "cs.LG"],
            published=now.isoformat(),
        )
        profile = _make_profile()
        result = score_paper(
            paper,
            profile,
            RankingConfig(),
            QueriesConfig(),
            ["cs.CL", "cs.LG"],
        )
        assert result["score_total"] == pytest.approx(1.0, abs=0.02)

    def test_custom_weights_still_normalized(self) -> None:
        paper = _make_paper()
        profile = _make_profile()
        result = score_paper(
            paper,
            profile,
            RankingConfig(w_keyword=5.0, w_category=3.0, w_recency=2.0),
            QueriesConfig(),
            ["cs.CL", "cs.LG"],
        )
        assert result["score_total"] <= 1.0

    def test_zero_weights_returns_zero(self) -> None:
        paper = _make_paper()
        profile = _make_profile()
        result = score_paper(
            paper,
            profile,
            RankingConfig(w_keyword=0.0, w_category=0.0, w_recency=0.0),
            QueriesConfig(),
            ["cs.CL"],
        )
        assert result["score_total"] == 0.0


class TestRankPapers:
    def test_returns_sorted_by_score(self) -> None:
        profile = _make_profile()
        papers = [
            _make_paper(
                arxiv_id="2401.00001v1",
                title="Quantum Computing Basics",
                abstract="No overlap with profile.",
            ),
            _make_paper(
                arxiv_id="2401.00002v1",
                title="Retrieval Augmented Generation",
                abstract="Transformers for retrieval and generation.",
            ),
        ]

        scores = rank_papers(
            papers,
            profile,
            RankingConfig(),
            QueriesConfig(),
            ["cs.CL"],
        )

        assert len(scores) == 2
        assert scores[0]["score_total"] >= scores[1]["score_total"]
        # The RAG paper should rank higher
        assert scores[0]["arxiv_id"] == "2401.00002v1"

    def test_empty_papers(self) -> None:
        profile = _make_profile()
        scores = rank_papers(
            [],
            profile,
            RankingConfig(),
            QueriesConfig(),
            ["cs.CL"],
        )
        assert scores == []


class TestFormatScoreExplanation:
    def test_contains_component_names(self) -> None:
        score_dict = {
            "arxiv_id": "2401.00001v1",
            "score_total": 0.75,
            "keyword_score": 0.8,
            "category_score": 0.6,
            "recency_score": 0.9,
        }
        result = format_score_explanation(score_dict, RankingConfig())
        assert "keyword" in result
        assert "category" in result
        assert "recency" in result
        assert "2401.00001v1" in result

    def test_contains_weight_values(self) -> None:
        score_dict = {
            "arxiv_id": "2401.00001v1",
            "score_total": 0.75,
            "keyword_score": 0.8,
            "category_score": 0.6,
            "recency_score": 0.9,
        }
        result = format_score_explanation(score_dict, RankingConfig(w_keyword=2.0))
        assert "2.00" in result

    def test_contains_total(self) -> None:
        score_dict = {
            "arxiv_id": "2401.00001v1",
            "score_total": 0.75,
            "keyword_score": 0.8,
            "category_score": 0.6,
            "recency_score": 0.9,
        }
        result = format_score_explanation(score_dict, RankingConfig())
        assert "total" in result
        assert "0.7500" in result


class TestScoreDistribution:
    def test_known_inputs(self) -> None:
        scores = [
            {"score_total": 0.2},
            {"score_total": 0.4},
            {"score_total": 0.6},
            {"score_total": 0.8},
        ]
        dist = score_distribution(scores)
        assert dist["mean"] == pytest.approx(0.5)
        assert dist["median"] == pytest.approx(0.5)
        assert dist["min"] == pytest.approx(0.2)
        assert dist["max"] == pytest.approx(0.8)
        assert dist["count"] == 4

    def test_empty_list(self) -> None:
        dist = score_distribution([])
        assert dist["mean"] == 0.0
        assert dist["median"] == 0.0
        assert dist["min"] == 0.0
        assert dist["max"] == 0.0
        assert dist["count"] == 0

    def test_single_item(self) -> None:
        dist = score_distribution([{"score_total": 0.5}])
        assert dist["mean"] == pytest.approx(0.5)
        assert dist["median"] == pytest.approx(0.5)
        assert dist["count"] == 1


class TestEmbeddingScoreIntegration:
    def test_embedding_score_included_when_provided(self) -> None:
        paper = _make_paper()
        profile = _make_profile()
        result = score_paper(
            paper,
            profile,
            RankingConfig(w_keyword=1.0, w_category=0.5, w_recency=0.3, w_embedding=1.5),
            QueriesConfig(),
            ["cs.CL", "cs.LG"],
            embedding_score=0.8,
        )
        assert result["embedding_score"] == pytest.approx(0.8)
        assert result["score_total"] > 0

    def test_embedding_score_ignored_when_weight_zero(self) -> None:
        paper = _make_paper()
        profile = _make_profile()
        without = score_paper(
            paper,
            profile,
            RankingConfig(w_embedding=0.0),
            QueriesConfig(),
            ["cs.CL"],
        )
        with_emb = score_paper(
            paper,
            profile,
            RankingConfig(w_embedding=0.0),
            QueriesConfig(),
            ["cs.CL"],
            embedding_score=0.9,
        )
        assert without["score_total"] == with_emb["score_total"]

    def test_embedding_score_none_ignored(self) -> None:
        paper = _make_paper()
        profile = _make_profile()
        result = score_paper(
            paper,
            profile,
            RankingConfig(w_embedding=1.5),
            QueriesConfig(),
            ["cs.CL"],
            embedding_score=None,
        )
        assert result["embedding_score"] is None
        assert result["score_total"] <= 1.0

    def test_rank_papers_with_repo_embedding(self) -> None:
        from unittest.mock import patch

        import numpy as np

        papers = [
            _make_paper(arxiv_id="2401.00001v1", title="Paper A"),
            _make_paper(arxiv_id="2401.00002v1", title="Paper B"),
        ]
        profile = _make_profile()
        repo_emb = np.array([1.0, 0.0, 0.0])

        with (
            patch("reporadar.embeddings.compute_paper_embedding") as mock_emb,
            patch("reporadar.embeddings.cosine_similarity") as mock_cos,
        ):
            mock_emb.side_effect = [np.array([0.9, 0.1, 0.0]), np.array([0.0, 0.0, 1.0])]
            mock_cos.side_effect = [0.99, 0.0]

            scores = rank_papers(
                papers,
                profile,
                RankingConfig(w_embedding=1.5),
                QueriesConfig(),
                ["cs.CL"],
                repo_embedding=repo_emb,
            )

        assert len(scores) == 2
        # Paper A should have embedding_score
        paper_a = next(s for s in scores if s["arxiv_id"] == "2401.00001v1")
        assert paper_a["embedding_score"] is not None


class TestPerCategoryWeights:
    def test_weighted_category_higher_score(self) -> None:
        paper = _make_paper(categories=["cs.CL"])
        # cs.CL weighted 2.0, cs.LG weighted 1.0
        score_weighted = score_category_match(
            paper, ["cs.CL", "cs.LG"], category_weights={"cs.CL": 2.0, "cs.LG": 1.0}
        )
        score_default = score_category_match(paper, ["cs.CL", "cs.LG"])
        # With cs.CL weighted 2x, matching just cs.CL should give 2/3 vs 1/2
        assert score_weighted > score_default

    def test_equal_weights_matches_default(self) -> None:
        paper = _make_paper(categories=["cs.CL"])
        score_weighted = score_category_match(
            paper, ["cs.CL", "cs.LG"], category_weights={"cs.CL": 1.0, "cs.LG": 1.0}
        )
        score_default = score_category_match(paper, ["cs.CL", "cs.LG"])
        assert score_weighted == pytest.approx(score_default)

    def test_unweighted_category_uses_default(self) -> None:
        paper = _make_paper(categories=["cs.AI"])
        # cs.AI not in weights dict → defaults to 1.0
        score = score_category_match(paper, ["cs.CL", "cs.AI"], category_weights={"cs.CL": 2.0})
        # cs.AI matches with weight 1.0, cs.CL doesn't match: 1.0 / (2.0 + 1.0) = 0.333
        assert score == pytest.approx(1.0 / 3.0, abs=0.01)
