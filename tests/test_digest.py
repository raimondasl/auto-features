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
            {"arxiv_id": "d", "score_total": 0.9},  # untriaged -> heuristic fallback (top)
        ]
        top, maybe, muted = categorize_papers(scored, triage_threshold=2)
        assert [p["arxiv_id"] for p in top] == ["a", "d"]
        assert [p["arxiv_id"] for p in maybe] == ["b"]
        assert [p["arxiv_id"] for p in muted] == ["c"]

    def test_no_triage_threshold_ignores_llm_score(self) -> None:
        scored = [{"arxiv_id": "a", "score_total": 0.9, "llm_score": 0}]
        top, _, _ = categorize_papers(scored)  # no threshold -> heuristic wins
        assert [p["arxiv_id"] for p in top] == ["a"]

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
