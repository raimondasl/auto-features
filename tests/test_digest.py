"""Tests for reporadar.digest."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from reporadar.digest import (
    categorize_papers,
    filter_since,
    generate_digest,
    generate_digest_csv,
    generate_digest_html,
    generate_digest_json,
    generate_digest_rss,
    mark_new_papers,
    markdown_to_html,
    write_digest,
)
from reporadar.store import PaperStore


def _days_ago(n: int) -> str:
    return (datetime.now(UTC) - timedelta(days=n)).strftime("%Y-%m-%dT00:00:00+00:00")


def _make_paper(arxiv_id: str = "2401.12345v1", **overrides) -> dict:
    base = {
        "arxiv_id": arxiv_id,
        "title": f"Test Paper {arxiv_id}",
        "authors": ["Alice Smith", "Bob Jones"],
        "abstract": "We propose a novel approach to retrieval augmented generation.",
        "categories": ["cs.CL", "cs.LG"],
        "published": "2024-01-20T00:00:00+00:00",
        "updated": "2024-01-21T00:00:00+00:00",
        "url": f"http://arxiv.org/abs/{arxiv_id}",
        "pdf_url": f"http://arxiv.org/pdf/{arxiv_id}",
    }
    base.update(overrides)
    return base


def _make_score(arxiv_id: str, score_total: float, **overrides) -> dict:
    base = {
        "arxiv_id": arxiv_id,
        "score_total": score_total,
        "keyword_score": score_total * 0.5,
        "category_score": score_total * 0.3,
        "recency_score": score_total * 0.2,
        "matched_query": "all:test",
    }
    base.update(overrides)
    return base


def _seed_store(store: PaperStore) -> int:
    """Insert sample papers, record a run, and save scores. Returns run_id."""
    papers = [
        _make_paper("2401.00001v1", title="High Relevance RAG Paper"),
        _make_paper("2401.00002v1", title="Medium Relevance Transformers Paper"),
        _make_paper("2401.00003v1", title="Low Relevance Quantum Paper"),
        _make_paper("2401.00004v1", title="Barely Relevant Misc Paper"),
    ]
    store.upsert_papers(papers)
    run_id = store.record_run(["all:test", "all:retrieval"], papers_new=4, papers_seen=0)

    scores = [
        _make_score("2401.00001v1", 0.85),
        _make_score("2401.00002v1", 0.45),
        _make_score("2401.00003v1", 0.15),
        _make_score("2401.00004v1", 0.05),
    ]
    store.save_scores(run_id, scores)
    return run_id


class TestFilterSince:
    def test_none_returns_all(self) -> None:
        papers = [{"published": _days_ago(100)}, {"published": _days_ago(1)}]
        assert filter_since(papers, None) == papers
        assert filter_since(papers, 0) == papers

    def test_drops_old_papers(self) -> None:
        papers = [
            {"arxiv_id": "old", "published": _days_ago(30)},
            {"arxiv_id": "new", "published": _days_ago(2)},
        ]
        kept = filter_since(papers, 7)
        ids = [p["arxiv_id"] for p in kept]
        assert ids == ["new"]

    def test_keeps_papers_with_missing_date(self) -> None:
        papers = [{"arxiv_id": "x", "published": ""}, {"arxiv_id": "y"}]
        kept = filter_since(papers, 7)
        assert len(kept) == 2

    def test_generate_digest_applies_since(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            store.upsert_papers(
                [
                    _make_paper("2401.00001v1", published=_days_ago(2)),
                    _make_paper("2401.00002v1", published=_days_ago(40)),
                ]
            )
            run_id = store.record_run(["q1"], 2, 0)
            store.save_scores(
                run_id,
                [_make_score("2401.00001v1", 0.9), _make_score("2401.00002v1", 0.8)],
            )
            md = generate_digest(store, run_id, since_days=7)
            assert "2401.00001v1" in md
            assert "2401.00002v1" not in md


class TestCategorizePapers:
    def test_splits_into_tiers(self) -> None:
        scored = [
            {"score_total": 0.9, "arxiv_id": "a"},
            {"score_total": 0.6, "arxiv_id": "b"},
            {"score_total": 0.3, "arxiv_id": "c"},
            {"score_total": 0.1, "arxiv_id": "d"},
        ]
        top, maybe, muted = categorize_papers(scored)

        assert len(top) == 2  # 0.9, 0.6 >= 0.5
        assert len(maybe) == 1  # 0.3 >= 0.2
        assert len(muted) == 1  # 0.1 < 0.2

    def test_respects_top_n(self) -> None:
        scored = [{"score_total": 0.9, "arxiv_id": f"p{i}"} for i in range(20)]
        top, maybe, muted = categorize_papers(scored, top_n=5)

        total = len(top) + len(maybe) + len(muted)
        assert total == 5

    def test_triage_threshold_gates_top_picks(self) -> None:
        # All heuristically Top Pick (0.9), but tiered by llm_score when set.
        scored = [
            {"arxiv_id": "a", "score_total": 0.9, "llm_score": 2},  # actionable -> top
            {"arxiv_id": "b", "score_total": 0.9, "llm_score": 1},  # related -> maybe
            {"arxiv_id": "c", "score_total": 0.9, "llm_score": 0},  # unrelated -> muted
            {"arxiv_id": "d", "score_total": 0.9},  # gate ran, no verdict -> maybe
        ]
        top, maybe, muted = categorize_papers(scored, triage_threshold=2)
        assert [p["arxiv_id"] for p in top] == ["a"]
        assert [p["arxiv_id"] for p in maybe] == ["b", "d"]
        assert [p["arxiv_id"] for p in muted] == ["c"]

    def test_an_ungated_paper_does_not_reach_top_picks(self) -> None:
        """`d` above, stated as its own claim, because the expectation used to be the
        opposite and the change is a real one to the shipped digest.

        When the gate ran, a paper it did not score used to fall back to the heuristic
        0.5 threshold — the threshold Feature 6 replaced after it measured net@2 −11 on
        the user-facing output. That fall-back fires whenever `output.top_n` exceeds
        `triage.top_k` (papers past the gate's window fill the digest ungated) or a gate
        call fails. The benchmark has always scored the strict rule; see
        `evals/audit_product_divergence.py`, which is what found the two disagreeing.
        """
        ungated = [{"arxiv_id": "x", "score_total": 0.99}]
        top, maybe, muted = categorize_papers(ungated, triage_threshold=2)
        assert top == []
        assert [p["arxiv_id"] for p in maybe] == ["x"]
        assert muted == []

    def test_no_triage_threshold_ignores_llm_score(self) -> None:
        scored = [{"arxiv_id": "a", "score_total": 0.9, "llm_score": 0}]
        top, _, _ = categorize_papers(scored)  # no threshold -> heuristic wins
        assert [p["arxiv_id"] for p in top] == ["a"]

    def test_with_the_gate_off_the_heuristic_is_still_the_only_rule(self) -> None:
        """The strict rule must not leak into runs that never gated anything.

        `triage_threshold=None` means the gate did not run, so every paper is "ungated" —
        demoting them all would empty Top Picks for every user with triage disabled.
        """
        scored = [{"arxiv_id": "a", "score_total": 0.9}, {"arxiv_id": "b", "score_total": 0.05}]
        top, _, muted = categorize_papers(scored, triage_threshold=None)
        assert [p["arxiv_id"] for p in top] == ["a"]
        assert [p["arxiv_id"] for p in muted] == ["b"]

    def test_a_paper_beyond_the_gate_window_is_not_promoted(self) -> None:
        """The configuration that makes this bite: a digest wider than the gate.

        With `output.top_n = 4` and a gate that scored 2 papers, the two unscored ones used
        to enter Top Picks on `score_total` alone, indistinguishable in the rendered digest
        from papers an LLM had actually endorsed.
        """
        scored = [
            {"arxiv_id": "a", "score_total": 0.9, "llm_score": 2},
            {"arxiv_id": "b", "score_total": 0.9, "llm_score": 3},
            {"arxiv_id": "c", "score_total": 0.8},
            {"arxiv_id": "d", "score_total": 0.7},
        ]
        top, maybe, _ = categorize_papers(scored, top_n=4, triage_threshold=2)
        assert [p["arxiv_id"] for p in top] == ["a", "b"]
        assert [p["arxiv_id"] for p in maybe] == ["c", "d"]

    def test_rerank_surfaces_buried_actionable_paper(self) -> None:
        # In score_total order the actionable paper (c) is beyond top_n=2, so
        # without rerank it never enters the window. Rerank floats it into Top Picks.
        scored = [
            {"arxiv_id": "a", "score_total": 0.9, "llm_score": 0},
            {"arxiv_id": "b", "score_total": 0.8, "llm_score": 1},
            {"arxiv_id": "c", "score_total": 0.3, "llm_score": 3},  # buried but actionable
        ]
        top_no, _, _ = categorize_papers(scored, top_n=2, triage_threshold=2, rerank=False)
        assert [p["arxiv_id"] for p in top_no] == []  # c is cut before the window

        top_yes, maybe_yes, _ = categorize_papers(scored, top_n=2, triage_threshold=2, rerank=True)
        assert [p["arxiv_id"] for p in top_yes] == ["c"]  # reranked into the window + gate
        assert [p["arxiv_id"] for p in maybe_yes] == ["b"]

    def test_rerank_without_triage_threshold_is_noop(self) -> None:
        # Rerank only fires with triage scoring; otherwise heuristic order stands.
        scored = [
            {"arxiv_id": "a", "score_total": 0.9, "llm_score": 0},
            {"arxiv_id": "b", "score_total": 0.1, "llm_score": 3},
        ]
        top, _, _ = categorize_papers(scored, top_n=1, rerank=True)  # no triage_threshold
        assert [p["arxiv_id"] for p in top] == ["a"]

    def test_empty_input(self) -> None:
        top, maybe, muted = categorize_papers([])
        assert top == []
        assert maybe == []
        assert muted == []

    def test_all_top_picks(self) -> None:
        scored = [{"score_total": 0.8, "arxiv_id": f"p{i}"} for i in range(3)]
        top, maybe, muted = categorize_papers(scored)
        assert len(top) == 3
        assert len(maybe) == 0
        assert len(muted) == 0

    def test_all_muted(self) -> None:
        scored = [{"score_total": 0.05, "arxiv_id": f"p{i}"} for i in range(3)]
        top, maybe, muted = categorize_papers(scored)
        assert len(top) == 0
        assert len(maybe) == 0
        assert len(muted) == 3

    def test_custom_thresholds(self) -> None:
        scored = [
            {"score_total": 0.9, "arxiv_id": "a"},
            {"score_total": 0.5, "arxiv_id": "b"},
        ]
        top, maybe, muted = categorize_papers(scored, top_threshold=0.8, maybe_threshold=0.4)
        assert len(top) == 1
        assert len(maybe) == 1


class TestGenerateDigest:
    def test_renders_markdown(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            run_id = _seed_store(store)
            content = generate_digest(store, run_id)

        assert "# RepoRadar Digest" in content
        assert "Top Picks" in content
        assert "High Relevance RAG Paper" in content

    def test_triage_gates_and_badges(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            store.upsert_papers(
                [
                    _make_paper("2401.00001v1", title="Actionable Paper"),
                    _make_paper("2401.00002v1", title="Vague Paper"),
                ]
            )
            run_id = store.record_run(["q1"], 2, 0)
            # Both heuristically high (0.9), so without triage both are Top Picks.
            store.save_scores(
                run_id, [_make_score("2401.00001v1", 0.9), _make_score("2401.00002v1", 0.9)]
            )
            store.save_llm_scores(
                run_id,
                {
                    "2401.00001v1": {"llm_score": 3, "llm_reason": "direct fit"},
                    "2401.00002v1": {"llm_score": 0, "llm_reason": "unrelated"},
                },
            )
            md = generate_digest(store, run_id, triage_threshold=2)

        # Actionable paper carries the badge + justification and is a Top Pick.
        assert "[ACTIONABLE 3/3]" in md
        assert "direct fit" in md
        # The vague paper (llm 0) is gated out of Top Picks — no actionable badge.
        assert "[ACTIONABLE 0/3]" not in md

    def test_includes_maybe_section(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            run_id = _seed_store(store)
            content = generate_digest(store, run_id)

        assert "Maybe Relevant" in content
        assert "Medium Relevance Transformers Paper" in content

    def test_includes_muted_section(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            run_id = _seed_store(store)
            content = generate_digest(store, run_id)

        assert "Muted" in content

    def test_includes_score_breakdown(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            run_id = _seed_store(store)
            content = generate_digest(store, run_id)

        assert "keyword:" in content
        assert "category:" in content
        assert "recency:" in content

    def test_includes_queries(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            run_id = _seed_store(store)
            content = generate_digest(store, run_id)

        assert "all:test" in content
        assert "all:retrieval" in content

    def test_includes_arxiv_links(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            run_id = _seed_store(store)
            content = generate_digest(store, run_id)

        assert "http://arxiv.org/abs/2401.00001v1" in content

    def test_empty_run(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            run_id = store.record_run(["q1"], 0, 0)
            content = generate_digest(store, run_id)

        assert "No scored papers found" in content

    def test_top_n_limits_output(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            run_id = _seed_store(store)
            content = generate_digest(store, run_id, top_n=1)

        # Only the highest-scoring paper should appear
        assert "High Relevance RAG Paper" in content
        assert "Medium Relevance" not in content
        assert "Low Relevance" not in content


class TestMarkdownToHtml:
    def test_wraps_in_html(self) -> None:
        html = markdown_to_html("# Hello World")
        assert "<!DOCTYPE html>" in html
        assert "Hello World" in html
        assert "<title>RepoRadar Digest</title>" in html

    def test_empty_content(self) -> None:
        html = markdown_to_html("")
        assert "<!DOCTYPE html>" in html

    def test_preserves_markdown_content(self) -> None:
        md = "## Section\n- item 1\n- item 2"
        html = markdown_to_html(md)
        assert "Section" in html
        assert "item 1" in html


class TestGenerateDigestHtml:
    def test_renders_real_html_not_pre_wrapped_markdown(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            run_id = _seed_store(store)
            html = generate_digest_html(store, run_id)

        assert "<!DOCTYPE html>" in html
        assert "<title>RepoRadar Digest</title>" in html
        # A real rendered page, not the legacy markdown-in-<pre> wrapper.
        assert "<pre>" not in html
        assert "High Relevance RAG Paper" in html
        assert 'class="total"' in html  # score styling → structured markup

    def test_autoescapes_paper_text(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            store.upsert_paper(_make_paper("2401.55555v1", title="Attention <script> & Beyond"))
            run_id = store.record_run(["all:test"], papers_new=1, papers_seen=0)
            store.save_scores(run_id, [_make_score("2401.55555v1", 0.9)])
            html = generate_digest_html(store, run_id)

        # The angle brackets and ampersand from the title must be escaped, so no
        # raw <script> tag ends up in the published page.
        assert "<script>" not in html
        assert "&lt;script&gt;" in html
        assert "Attention" in html

    def test_drops_non_http_url_scheme(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            store.upsert_paper(_make_paper("2401.77777v1", url="javascript:alert(1)"))
            run_id = store.record_run(["all:test"], papers_new=1, papers_seen=0)
            store.save_scores(run_id, [_make_score("2401.77777v1", 0.9)])
            html = generate_digest_html(store, run_id)

        # A javascript: URL from an upstream source must not become a live link.
        assert "javascript:alert" not in html
        assert 'href="#"' in html

    def test_keeps_https_url(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            store.upsert_paper(_make_paper("2401.88888v1", url="https://arxiv.org/abs/2401.88888"))
            run_id = store.record_run(["all:test"], papers_new=1, papers_seen=0)
            store.save_scores(run_id, [_make_score("2401.88888v1", 0.9)])
            html = generate_digest_html(store, run_id)

        assert "https://arxiv.org/abs/2401.88888" in html


class TestRecommendedSource:
    def test_prefers_api_recommendations_in_score_order(self, tmp_path: Path) -> None:
        # Papers fetched via the S2 recommender carry matched_query="recommendation";
        # the digest shows them (already re-ranked locally) over the keyword recommender.
        with PaperStore(tmp_path / "papers.db") as store:
            store.upsert_paper(_make_paper("2402.00001v1", title="Rec High"))
            store.upsert_paper(_make_paper("2402.00002v1", title="Rec Low"))
            store.upsert_paper(_make_paper("2402.00003v1", title="Not A Rec"))
            run_id = store.record_run(["q"], 3, 0)
            store.save_scores(
                run_id,
                [
                    _make_score("2402.00001v1", 0.9, matched_query="recommendation"),
                    _make_score("2402.00002v1", 0.4, matched_query="recommendation"),
                    _make_score("2402.00003v1", 0.8, matched_query="all:test"),
                ],
            )
            md = generate_digest(store, run_id)

        assert "Recommended for You" in md
        # Best-scoring recommendation first; the non-recommendation isn't in the section.
        rec_section = md.split("## Recommended for You")[1].split("\n## ")[0]
        assert "Rec High" in rec_section
        assert "Not A Rec" not in rec_section
        assert rec_section.index("Rec High") < rec_section.index("Rec Low")

    def test_drops_low_scoring_recommendations(self, tmp_path: Path) -> None:
        # The S2 recommender is repo-agnostic; anything the local ranker scored
        # below the "maybe" tier must not be shown as a recommendation.
        with PaperStore(tmp_path / "papers.db") as store:
            store.upsert_paper(_make_paper("2402.00009v1", title="Off Topic Rec"))
            run_id = store.record_run(["q"], 1, 0)
            store.save_scores(
                run_id, [_make_score("2402.00009v1", 0.05, matched_query="recommendation")]
            )
            md = generate_digest(store, run_id)
        # It may still appear in the Muted tier, but never as a recommendation.
        assert "Recommended for You" not in md

    def test_recommendations_survive_since_filter(self, tmp_path: Path) -> None:
        # Recommendations are a user-seeded feed, not a publication window, so
        # `--since` must not silently empty the section.
        with PaperStore(tmp_path / "papers.db") as store:
            store.upsert_paper(
                _make_paper("2402.00007v1", title="Older Rec", published=_days_ago(90))
            )
            run_id = store.record_run(["q"], 1, 0)
            store.save_scores(
                run_id, [_make_score("2402.00007v1", 0.9, matched_query="recommendation")]
            )
            md = generate_digest(store, run_id, since_days=7)
        assert "Recommended for You" in md
        assert "Older Rec" in md

    def test_falls_back_to_local_recommender(self, tmp_path: Path) -> None:
        # With no API recommendations in the run, the keyword recommender still runs.
        with PaperStore(tmp_path / "papers.db") as store:
            run_id = _seed_store(store)
            md = generate_digest(store, run_id)
        assert "Recommended for You" not in md  # no ratings seeded → nothing to recommend


class TestExtendsStarred:
    def test_section_and_badge_when_paper_cites_starred(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            store.upsert_paper(_make_paper("2402.00001v1", title="New Extending Paper"))
            store.upsert_paper(_make_paper("2401.00099v1", title="Starred Seed"))
            run_id = store.record_run(["q"], 1, 0)
            store.save_scores(run_id, [_make_score("2402.00001v1", 0.9)])
            store.star_paper("2401.00099v1")
            store.save_citations([("2402.00001v1", "2401.00099")])

            md = generate_digest(store, run_id)
            html = generate_digest_html(store, run_id)

        assert "Extends work you starred" in md
        assert "[EXTENDS STARRED]" in md
        assert "New Extending Paper" in md
        assert "Extends work you starred" in html and "Extends starred" in html

    def test_no_section_when_cited_paper_not_starred(self, tmp_path: Path) -> None:
        # An edge exists but the cited paper is not (any longer) a seed → no section.
        with PaperStore(tmp_path / "papers.db") as store:
            store.upsert_paper(_make_paper("2402.00001v1"))
            run_id = store.record_run(["q"], 1, 0)
            store.save_scores(run_id, [_make_score("2402.00001v1", 0.9)])
            store.save_citations([("2402.00001v1", "2401.00099")])
            md = generate_digest(store, run_id)
        assert "Extends work you starred" not in md

    def test_summary_count_matches_the_rendered_section(self, tmp_path: Path) -> None:
        # The notification count and the digest section come from the same helper,
        # so a citation alert can't be pushed without appearing in the digest.
        with PaperStore(tmp_path / "papers.db") as store:
            store.upsert_paper(_make_paper("2402.00001v1", title="Extends A"))
            store.upsert_paper(_make_paper("2402.00002v1", title="Extends B"))
            store.upsert_paper(_make_paper("2402.00003v1", title="Cites Nothing Starred"))
            store.upsert_paper(_make_paper("2401.00099v1", title="Starred Seed"))
            run_id = store.record_run(["q"], 3, 0)
            store.save_scores(
                run_id,
                [
                    _make_score("2402.00001v1", 0.9),
                    _make_score("2402.00002v1", 0.8),
                    _make_score("2402.00003v1", 0.7),
                ],
            )
            store.star_paper("2401.00099v1")
            store.save_citations(
                [
                    ("2402.00001v1", "2401.00099"),
                    ("2402.00002v1", "2401.00099"),
                    ("2402.00003v1", "2401.00500"),  # not a seed
                ]
            )
            _, summary = write_digest(store, run_id, tmp_path / "digest.md")
        assert summary is not None
        assert summary.extends_starred_count == 2

    def test_summary_count_is_zero_without_seeds(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            store.upsert_paper(_make_paper("2402.00001v1"))
            run_id = store.record_run(["q"], 1, 0)
            store.save_scores(run_id, [_make_score("2402.00001v1", 0.9)])
            store.save_citations([("2402.00001v1", "2401.00099")])
            _, summary = write_digest(store, run_id, tmp_path / "digest.md")
        assert summary is not None
        assert summary.extends_starred_count == 0


class TestSignalRendering:
    def _seed(self, store: PaperStore) -> int:
        store.upsert_paper(_make_paper("2607.00001v1", title="A Withdrawn Result"))
        store.upsert_paper(_make_paper("2607.00002v1", title="A Much Discussed Paper"))
        run_id = store.record_run(["q"], 2, 0)
        store.save_scores(
            run_id,
            [
                _make_score("2607.00002v1", 0.9),
                # Already penalized by the ranker, so it lands in Muted.
                _make_score("2607.00001v1", 0.08),
            ],
        )
        store.save_signals(
            [
                ("2607.00001v1", "withdrawn", "comment", None),
                ("2607.00002v1", "hn", "1351", "https://news.ycombinator.com/item?id=42823568"),
            ]
        )
        return run_id

    def test_withdrawal_survives_demotion_to_muted(self, tmp_path: Path) -> None:
        """The warning must not depend on the paper's tier.

        The ranking penalty deliberately pushes a withdrawn paper into Muted, which
        renders no per-card badges — so a card-only warning would disappear in
        exactly the case it exists for. Hence a dedicated section.
        """
        with PaperStore(tmp_path / "papers.db") as store:
            run_id = self._seed(store)
            md = generate_digest(store, run_id)
            html = generate_digest_html(store, run_id)
        assert "## Withdrawn by their authors" in md
        assert "A Withdrawn Result" in md.split("## Withdrawn by their authors")[1]
        assert "Withdrawn by their authors" in html

    def test_hn_badge_and_link(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            run_id = self._seed(store)
            md = generate_digest(store, run_id)
            html = generate_digest_html(store, run_id)
        assert "[HN 1351]" in md
        assert "**Hacker News:** 1351 points" in md
        assert "item?id=42823568" in md
        assert 'badge">HN 1351' in html

    def test_each_metadata_line_stays_on_its_own_line(self, tmp_path: Path) -> None:
        # trim_blocks eats the newline after a line-ending {% endif %}, which used to
        # concatenate every optional bullet into one unreadable line.
        with PaperStore(tmp_path / "papers.db") as store:
            run_id = self._seed(store)
            store.save_enrichments(
                {
                    "2607.00002v1": {
                        "arxiv_id": "2607.00002v1",
                        "has_code": True,
                        "code_urls": ["https://github.com/a/b"],
                        "models": ["m/1"],
                        "datasets": ["d/1"],
                        "tasks": ["t"],
                        "upvotes": 7,
                    }
                }
            )
            md = generate_digest(store, run_id)
        for line in md.splitlines():
            assert line.count("- **") <= 1, f"bullets ran together: {line!r}"

    def test_withdrawn_paper_cannot_reach_top_picks_via_triage(self, tmp_path: Path) -> None:
        """The LLM triage branch must not route around the withdrawal penalty.

        categorize_papers tiers by llm_score and returns before reading score_total,
        so a withdrawn paper with a high actionability score landed in Top Picks with
        the 0.1x multiplier silently ignored.
        """
        with PaperStore(tmp_path / "papers.db") as store:
            store.upsert_paper(_make_paper("2607.00001v1", title="A Withdrawn Result"))
            run_id = store.record_run(["q"], 1, 0)
            store.save_scores(run_id, [_make_score("2607.00001v1", 0.08)])
            store.save_llm_scores(run_id, {"2607.00001v1": {"llm_score": 3, "llm_reason": "r"}})
            store.save_signals([("2607.00001v1", "withdrawn", "comment", None)])
            md = generate_digest(store, run_id, triage_threshold=2)
        top = md.split("## Top Picks")[1] if "## Top Picks" in md else ""
        assert "A Withdrawn Result" not in top
        assert "## Withdrawn by their authors" in md

    def test_a_checked_clean_paper_is_not_flagged(self, tmp_path: Path) -> None:
        # Clean results are stored too (so the next run can skip them), as a NULL
        # value — which must not read as "withdrawn".
        with PaperStore(tmp_path / "papers.db") as store:
            store.upsert_paper(_make_paper("2607.00001v1", title="A Clean Paper"))
            run_id = store.record_run(["q"], 1, 0)
            store.save_scores(run_id, [_make_score("2607.00001v1", 0.9)])
            store.save_signals([("2607.00001v1", "withdrawn", None, None)])
            md = generate_digest(store, run_id)
        assert "Withdrawn by their authors" not in md
        assert "[WITHDRAWN]" not in md

    def test_no_sections_without_signals(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            run_id = _seed_store(store)
            md = generate_digest(store, run_id)
        assert "Withdrawn by their authors" not in md
        assert "[HN " not in md


class TestDigestRunMetadata:
    def test_header_uses_requested_run_not_latest(self, tmp_path: Path) -> None:
        # A digest for an older run must show THAT run's stats, not the newest run's.
        with PaperStore(tmp_path / "papers.db") as store:
            store.upsert_paper(_make_paper("2401.00001v1"))
            run1 = store.record_run(["all:oldquery"], papers_new=7, papers_seen=1)
            store.save_scores(run1, [_make_score("2401.00001v1", 0.9)])
            # A newer run with unrelated metadata.
            store.record_run(["all:newquery"], papers_new=99, papers_seen=99)

            md = generate_digest(store, run1)

        assert "all:oldquery" in md
        assert "all:newquery" not in md
        assert "7 new" in md
        assert "99" not in md


class TestGenerateDigestSuggestions:
    def test_top_picks_have_suggestions_key(self, tmp_path: Path) -> None:
        """Top pick papers with matching abstract patterns should get suggestions."""
        with PaperStore(tmp_path / "papers.db") as store:
            # Insert a paper whose abstract triggers suggestion patterns
            paper = _make_paper(
                "2401.99999v1",
                title="Benchmark Paper",
                abstract="We evaluate on GLUE benchmark and outperforms BERT-base.",
            )
            store.upsert_paper(paper)
            run_id = store.record_run(["q1"], 1, 0)
            store.save_scores(run_id, [_make_score("2401.99999v1", 0.9)])

            content = generate_digest(store, run_id)

        assert "Action ideas" in content

    def test_suggestions_labeled_as_ideas(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            paper = _make_paper(
                "2401.99999v1",
                abstract="Code is open-sourced at our repository.",
            )
            store.upsert_paper(paper)
            run_id = store.record_run(["q1"], 1, 0)
            store.save_scores(run_id, [_make_score("2401.99999v1", 0.9)])

            content = generate_digest(store, run_id)

        assert "auto-generated" in content

    def test_json_path_enriches_and_filters(self, tmp_path: Path) -> None:
        import json as json_mod

        class _Cfg:
            provider = "template"

        with PaperStore(tmp_path / "papers.db") as store:
            store.upsert_papers(
                [
                    _make_paper(
                        "2401.00001v1",
                        published=_days_ago(2),
                        abstract="We evaluate on GLUE benchmark and outperforms BERT.",
                    ),
                    _make_paper("2401.00002v1", published=_days_ago(40)),
                ]
            )
            run_id = store.record_run(["q1"], 2, 0)
            store.save_scores(
                run_id,
                [_make_score("2401.00001v1", 0.9), _make_score("2401.00002v1", 0.8)],
            )
            # suggestions_config + since_days now flow through the JSON path.
            out = generate_digest_json(store, run_id, suggestions_config=_Cfg(), since_days=7)
            data = json_mod.loads(out)

        ids = [p["arxiv_id"] for p in data["top_picks"]]
        assert "2401.00001v1" in ids
        assert "2401.00002v1" not in ids  # filtered out by since_days
        assert "suggestions" in data["top_picks"][0]


class TestEnrichmentBadges:
    def test_code_badge_appears(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            paper = _make_paper("2401.00001v1", title="Code Paper")
            store.upsert_paper(paper)
            store.save_enrichment(
                {
                    "arxiv_id": "2401.00001v1",
                    "pwc_id": "test",
                    "has_code": True,
                    "code_urls": ["https://github.com/foo/bar"],
                    "datasets": [],
                    "tasks": [],
                }
            )
            run_id = store.record_run(["q1"], 1, 0)
            store.save_scores(run_id, [_make_score("2401.00001v1", 0.9)])
            content = generate_digest(store, run_id)

        assert "[CODE]" in content

    def test_data_badge_appears(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            paper = _make_paper("2401.00001v1", title="Dataset Paper")
            store.upsert_paper(paper)
            store.save_enrichment(
                {
                    "arxiv_id": "2401.00001v1",
                    "pwc_id": "test",
                    "has_code": False,
                    "code_urls": [],
                    "datasets": ["ImageNet"],
                    "tasks": [],
                }
            )
            run_id = store.record_run(["q1"], 1, 0)
            store.save_scores(run_id, [_make_score("2401.00001v1", 0.9)])
            content = generate_digest(store, run_id)

        assert "[DATA]" in content
        assert "ImageNet" in content

    def test_no_badges_without_enrichment(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            paper = _make_paper("2401.00001v1", title="Plain Paper")
            store.upsert_paper(paper)
            run_id = store.record_run(["q1"], 1, 0)
            store.save_scores(run_id, [_make_score("2401.00001v1", 0.9)])
            content = generate_digest(store, run_id)

        assert "[CODE]" not in content
        assert "[DATA]" not in content


class TestDiffMode:
    def test_diff_with_two_runs(self, tmp_path: Path) -> None:
        """Papers in run 2 but not run 1 should be marked [NEW]."""
        with PaperStore(tmp_path / "papers.db") as store:
            # Shared paper
            store.upsert_paper(_make_paper("2401.00001v1", title="Old Paper"))
            # New paper in run 2
            store.upsert_paper(_make_paper("2401.00002v1", title="Brand New Paper"))

            r1 = store.record_run(["q1"], 1, 0)
            store.save_scores(r1, [_make_score("2401.00001v1", 0.8)])

            r2 = store.record_run(["q2"], 2, 0)
            store.save_scores(
                r2,
                [
                    _make_score("2401.00001v1", 0.8),
                    _make_score("2401.00002v1", 0.7),
                ],
            )

            content = generate_digest(store, r2, diff=True)

        assert "[NEW]" in content
        # "Old Paper" was in r1, should NOT have [NEW]
        # "Brand New Paper" was NOT in r1, should have [NEW]
        lines = content.split("\n")
        for line in lines:
            if "Brand New Paper" in line and "###" in line:
                assert "[NEW]" in line
            if "Old Paper" in line and "###" in line:
                assert "[NEW]" not in line

    def test_diff_no_previous_run_all_new(self, tmp_path: Path) -> None:
        """If there is no previous run, all papers are [NEW]."""
        with PaperStore(tmp_path / "papers.db") as store:
            store.upsert_paper(_make_paper("2401.00001v1", title="Only Paper"))
            r1 = store.record_run(["q1"], 1, 0)
            store.save_scores(r1, [_make_score("2401.00001v1", 0.8)])

            content = generate_digest(store, r1, diff=True)

        assert "[NEW]" in content

    def test_no_diff_no_badges(self, tmp_path: Path) -> None:
        """Without diff=True, no [NEW] badges should appear."""
        with PaperStore(tmp_path / "papers.db") as store:
            store.upsert_paper(_make_paper("2401.00001v1", title="Some Paper"))
            r1 = store.record_run(["q1"], 1, 0)
            store.save_scores(r1, [_make_score("2401.00001v1", 0.8)])

            content = generate_digest(store, r1, diff=False)

        assert "[NEW]" not in content

    def test_a_new_version_of_a_known_paper_is_not_new(self, tmp_path: Path) -> None:
        """ "Is this the same paper" has one answer, and the diff used to give another.

        arXiv hands back whatever the current version is, so a paper seen as ``v1`` last
        week arrives as ``v2`` this week. Comparing raw ids badged it [NEW] — a third rule
        for the invariant `dedup_id` exists to hold.
        """
        with PaperStore(tmp_path / "papers.db") as store:
            store.upsert_paper(_make_paper("2401.00001v1", title="Revised Paper"))
            store.upsert_paper(_make_paper("2401.00001v2", title="Revised Paper"))
            r1 = store.record_run(["q1"], 1, 0)
            store.save_scores(r1, [_make_score("2401.00001v1", 0.8)])
            r2 = store.record_run(["q2"], 1, 0)
            store.save_scores(r2, [_make_score("2401.00001v2", 0.8)])

            scored = mark_new_papers(store, store.get_scores_for_run(r2), r2)

        assert [p["is_new"] for p in scored] == [False]

    def test_a_genuinely_unseen_paper_is_still_new(self, tmp_path: Path) -> None:
        """The other half: normalising must not swallow the signal it exists to give."""
        with PaperStore(tmp_path / "papers.db") as store:
            store.upsert_paper(_make_paper("2401.00001v1"))
            store.upsert_paper(_make_paper("2401.00009v1"))
            r1 = store.record_run(["q1"], 1, 0)
            store.save_scores(r1, [_make_score("2401.00001v1", 0.8)])
            r2 = store.record_run(["q2"], 1, 0)
            store.save_scores(r2, [_make_score("2401.00009v1", 0.8)])

            scored = mark_new_papers(store, store.get_scores_for_run(r2), r2)

        assert [p["is_new"] for p in scored] == [True]


class TestWriteDigest:
    def test_writes_file(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            run_id = _seed_store(store)
            out, summary = write_digest(store, run_id, tmp_path / "output" / "digest.md")

        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "# RepoRadar Digest" in content

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            run_id = _seed_store(store)
            out, _ = write_digest(store, run_id, tmp_path / "deep" / "nested" / "digest.md")

        assert out.exists()

    def test_html_format(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            run_id = _seed_store(store)
            out, _ = write_digest(store, run_id, tmp_path / "digest.md", fmt="html")

        assert out.suffix == ".html"
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content
        assert "RepoRadar Digest" in content

    def test_html_format_explicit_extension(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            run_id = _seed_store(store)
            out, _ = write_digest(store, run_id, tmp_path / "output.html", fmt="html")

        assert out.suffix == ".html"
        assert out.exists()

    def test_write_digest_json(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            run_id = _seed_store(store)
            out, _ = write_digest(store, run_id, tmp_path / "digest.md", fmt="json")

        assert out.suffix == ".json"
        assert out.exists()

    def test_write_digest_csv(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            run_id = _seed_store(store)
            out, _ = write_digest(store, run_id, tmp_path / "digest.md", fmt="csv")

        assert out.suffix == ".csv"
        assert out.exists()

    def test_write_digest_rss(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            run_id = _seed_store(store)
            out, _ = write_digest(store, run_id, tmp_path / "digest.md", fmt="rss")

        assert out.suffix == ".xml"
        assert out.exists()


class TestWriteDigestSummary:
    def test_returns_tuple(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            run_id = _seed_store(store)
            result = write_digest(store, run_id, tmp_path / "digest.md")

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_summary_has_correct_stats(self, tmp_path: Path) -> None:
        from reporadar.notify import DigestSummary

        with PaperStore(tmp_path / "papers.db") as store:
            run_id = _seed_store(store)
            _, summary = write_digest(store, run_id, tmp_path / "digest.md")

        assert isinstance(summary, DigestSummary)
        assert summary.run_id == run_id
        assert summary.papers_new == 4
        assert summary.papers_seen == 0
        assert summary.total_scored == 4
        assert summary.top_picks_count == 1  # only 0.85 >= 0.5
        assert summary.fmt == "md"

    def test_summary_digest_path(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            run_id = _seed_store(store)
            out, summary = write_digest(store, run_id, tmp_path / "digest.md")

        assert summary.digest_path == str(out)


class TestJsonExport:
    def test_generates_valid_json(self, tmp_path: Path) -> None:
        import json

        with PaperStore(tmp_path / "papers.db") as store:
            run_id = _seed_store(store)
            content = generate_digest_json(store, run_id)

        data = json.loads(content)
        assert "generated_at" in data
        assert "run_id" in data
        assert "top_picks" in data
        assert "maybe_relevant" in data
        assert "muted" in data

    def test_json_includes_all_tiers(self, tmp_path: Path) -> None:
        import json

        with PaperStore(tmp_path / "papers.db") as store:
            run_id = _seed_store(store)
            content = generate_digest_json(store, run_id)

        data = json.loads(content)
        # _seed_store creates: 0.85 (top), 0.45 (maybe), 0.15 (muted), 0.05 (muted)
        assert len(data["top_picks"]) == 1
        assert len(data["maybe_relevant"]) == 1
        assert len(data["muted"]) == 2

    def test_json_includes_scores(self, tmp_path: Path) -> None:
        import json

        with PaperStore(tmp_path / "papers.db") as store:
            run_id = _seed_store(store)
            content = generate_digest_json(store, run_id)

        data = json.loads(content)
        top = data["top_picks"][0]
        assert "score_total" in top
        assert "keyword_score" in top
        assert "category_score" in top


class TestCsvExport:
    def test_generates_valid_csv(self, tmp_path: Path) -> None:
        import csv
        import io

        with PaperStore(tmp_path / "papers.db") as store:
            run_id = _seed_store(store)
            content = generate_digest_csv(store, run_id)

        reader = csv.DictReader(io.StringIO(content))
        rows = list(reader)
        assert len(rows) > 0
        assert "arxiv_id" in reader.fieldnames
        assert "title" in reader.fieldnames
        assert "score_total" in reader.fieldnames
        assert "tier" in reader.fieldnames

    def test_csv_includes_all_papers(self, tmp_path: Path) -> None:
        import csv
        import io

        with PaperStore(tmp_path / "papers.db") as store:
            run_id = _seed_store(store)
            content = generate_digest_csv(store, run_id)

        reader = csv.DictReader(io.StringIO(content))
        rows = list(reader)
        assert len(rows) == 4  # _seed_store creates 4 papers


class TestRssExport:
    def test_generates_valid_xml(self, tmp_path: Path) -> None:
        import xml.etree.ElementTree as ET

        with PaperStore(tmp_path / "papers.db") as store:
            run_id = _seed_store(store)
            content = generate_digest_rss(store, run_id)

        root = ET.fromstring(content)
        assert root.tag == "rss"
        channel = root.find("channel")
        assert channel is not None
        assert channel.find("title").text == "RepoRadar Digest"

    def test_rss_includes_papers(self, tmp_path: Path) -> None:
        import xml.etree.ElementTree as ET

        with PaperStore(tmp_path / "papers.db") as store:
            run_id = _seed_store(store)
            content = generate_digest_rss(store, run_id)

        root = ET.fromstring(content)
        items = root.findall(".//item")
        assert len(items) > 0


class TestWithdrawnSectionScope:
    """A retraction is worth reporting when it was competing for a slot, and not otherwise.

    The section used to list every withdrawn paper in the run, at any rank. On a repository
    whose thin profile produced generic queries (`all:model`, `all:use`) that pulled in
    hundreds of unrelated papers, three off-topic retractions -- hydraulic fracturing,
    opioid use disorders, music sight reading -- headed a materials-science digest whose
    Top Picks were sound.
    """

    def test_withdrawn_far_below_the_window_is_not_reported(self) -> None:
        from reporadar.digest import digest_window

        scored = [{"score_total": 0.9 - i / 100, "arxiv_id": f"p{i}"} for i in range(40)]
        scored[35]["withdrawn_in"] = "comment"
        window, withdrawn = digest_window(scored, top_n=5)

        assert len(window) == 5
        assert withdrawn == []

    def test_withdrawn_inside_the_window_is_reported_and_displaces(self) -> None:
        from reporadar.digest import digest_window

        scored = [{"score_total": 0.9 - i / 100, "arxiv_id": f"p{i}"} for i in range(40)]
        scored[2]["withdrawn_in"] = "comment"
        window, withdrawn = digest_window(scored, top_n=5)

        assert [p["arxiv_id"] for p in withdrawn] == ["p2"]
        # The withdrawn paper's slot goes to the next paper rather than being wasted.
        assert [p["arxiv_id"] for p in window] == ["p0", "p1", "p3", "p4", "p5"]

    def test_rendered_section_matches_the_window(self) -> None:
        """The template's list and `digest_window` are one rule, not two.

        `_build_digest_context` used to rebuild this list by scanning every scored paper,
        which is how the two came to disagree.
        """
        with PaperStore(Path(":memory:")) as store:
            papers = [_make_paper(f"24010000{i}v1") for i in range(4)]
            papers[3]["title"] = "Retracted And Buried"
            store.upsert_papers(papers)
            run_id = store.record_run(["q"], 4, 0)
            scores = [_make_score(f"24010000{i}v1", 0.9 - i / 10) for i in range(4)]
            scores[3]["withdrawn_in"] = "comment"
            store.save_scores(run_id, scores)

            md = generate_digest(store, run_id, top_n=2)
            assert "Withdrawn by their authors" not in md
            assert "Retracted And Buried" not in md


class TestAlreadyCitedPapers:
    """Papers the repository's own text cites are found, and not recommended back."""

    def _profile(self, *ids: str) -> object:
        class _P:
            cited_arxiv_ids = frozenset(ids)
            keywords: list[tuple[str, float]] = []
            anchors: list[str] = []
            domains: list[str] = []
            prose = ""

        return _P()

    def test_cited_paper_leaves_the_tiers(self) -> None:
        scored = [
            {"score_total": 0.9, "arxiv_id": "2303.14046v1", "llm_score": 3},
            {"score_total": 0.8, "arxiv_id": "2405.08137v1", "llm_score": 3},
        ]
        top, _, muted = categorize_papers(
            scored, triage_threshold=2, cited_ids={"2303.14046"}
        )

        assert [p["arxiv_id"] for p in top] == ["2405.08137v1"]
        assert [p["arxiv_id"] for p in muted] == ["2303.14046v1"]
        assert muted[0]["already_cited"] is True

    def test_version_suffixes_do_not_defeat_the_match(self) -> None:
        scored = [{"score_total": 0.9, "arxiv_id": "1708.01492v5"}]
        top, _, muted = categorize_papers(scored, cited_ids={"1708.01492"})

        assert top == []
        assert muted[0]["already_cited"] is True

    def test_no_cited_ids_changes_nothing(self) -> None:
        scored = [{"score_total": 0.9, "arxiv_id": "2303.14046v1"}]
        top, _, muted = categorize_papers(scored, cited_ids=None)

        assert [p["arxiv_id"] for p in top] == ["2303.14046v1"]
        assert muted == []

    def test_excluding_one_cannot_pull_an_unseen_paper_into_the_digest(self) -> None:
        """Cited papers are muted *inside* the window, never removed before the cut.

        Removing them first would promote a paper from outside the digest's own top_n --
        one nothing upstream (gate, rescore) ever scored.
        """
        scored = [{"score_total": 0.9 - i / 100, "arxiv_id": f"240{i}.00001"} for i in range(6)]
        top, maybe, muted = categorize_papers(scored, top_n=3, cited_ids={"2401.00001"})

        shown = [p["arxiv_id"] for p in top + maybe + muted]
        assert "2403.00001" not in shown
        assert len(shown) == 3

    def test_section_renders_and_the_paper_is_not_a_top_pick(self) -> None:
        with PaperStore(Path(":memory:")) as store:
            papers = [_make_paper("2303.14046v1"), _make_paper("2405.08137v1")]
            papers[0]["title"] = "Updates To The Library Itself"
            store.upsert_papers(papers)
            run_id = store.record_run(["q"], 2, 0)
            store.save_scores(
                run_id,
                [_make_score("2303.14046v1", 0.9), _make_score("2405.08137v1", 0.8)],
            )

            md = generate_digest(store, run_id, profile=self._profile("2303.14046"))
            assert "Already cited by this repository" in md
            assert "Updates To The Library Itself" in md
            top_section = md.split("## Top Picks")[1].split("## Already cited")[0]
            assert "Updates To The Library Itself" not in top_section

    def test_json_and_html_agree_with_markdown(self) -> None:
        with PaperStore(Path(":memory:")) as store:
            store.upsert_papers([_make_paper("2303.14046v1"), _make_paper("2405.08137v1")])
            run_id = store.record_run(["q"], 2, 0)
            store.save_scores(
                run_id,
                [_make_score("2303.14046v1", 0.9), _make_score("2405.08137v1", 0.8)],
            )
            profile = self._profile("2303.14046")

            import json as json_mod

            payload = json_mod.loads(
                generate_digest_json(store, run_id, profile=profile)
            )
            assert [p["arxiv_id"] for p in payload["top_picks"]] == ["2405.08137v1"]
            assert [p["arxiv_id"] for p in payload["already_cited"]] == ["2303.14046v1"]

            html = generate_digest_html(store, run_id, profile=profile)
            assert "Already cited by this repository" in html
