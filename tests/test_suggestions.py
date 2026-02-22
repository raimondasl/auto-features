"""Tests for reporadar.suggestions."""

from __future__ import annotations

from reporadar.suggestions import (
    MAX_SUGGESTIONS,
    enrich_papers_with_suggestions,
    generate_suggestions,
)


def _make_paper(**overrides) -> dict:
    base = {
        "arxiv_id": "2401.12345v1",
        "title": "A Novel Approach",
        "abstract": "We propose a method for text generation.",
        "authors": ["Alice"],
        "categories": ["cs.CL"],
    }
    base.update(overrides)
    return base


class TestGenerateSuggestions:
    def test_benchmark_pattern(self) -> None:
        paper = _make_paper(abstract="We evaluate on GLUE benchmark and show improvements.")
        suggestions = generate_suggestions(paper)
        assert any("GLUE" in s for s in suggestions)

    def test_outperforms_pattern(self) -> None:
        paper = _make_paper(abstract="Our method outperforms BERT-base on all tasks.")
        suggestions = generate_suggestions(paper)
        assert any("BERT" in s for s in suggestions)

    def test_propose_pattern(self) -> None:
        paper = _make_paper(
            abstract="We propose a novel attention mechanism that improves performance."
        )
        suggestions = generate_suggestions(paper)
        assert any("attention mechanism" in s.lower() for s in suggestions)

    def test_open_source_pattern(self) -> None:
        paper = _make_paper(abstract="Code is open-sourced at our repository.")
        suggestions = generate_suggestions(paper)
        assert any("publicly available" in s for s in suggestions)

    def test_no_matches(self) -> None:
        paper = _make_paper(abstract="This is a simple abstract with no patterns.")
        suggestions = generate_suggestions(paper)
        assert suggestions == []

    def test_max_suggestions_limit(self) -> None:
        paper = _make_paper(
            abstract=(
                "We propose a novel retrieval system that outperforms BM25. "
                "Evaluated on MSMARCO benchmark. "
                "Code is open-sourced. "
                "We introduce a new loss function called InfoNCE. "
                "Achieves state-of-the-art on question answering."
            )
        )
        suggestions = generate_suggestions(paper)
        assert len(suggestions) <= MAX_SUGGESTIONS

    def test_no_duplicate_suggestion_types(self) -> None:
        paper = _make_paper(
            abstract="We outperforms BERT and also outperforms RoBERTa on benchmarks."
        )
        suggestions = generate_suggestions(paper)
        # Should not have two "Compare against X" suggestions
        compare_suggestions = [s for s in suggestions if "Compare" in s]
        assert len(compare_suggestions) <= 1

    def test_empty_abstract(self) -> None:
        paper = _make_paper(abstract="")
        suggestions = generate_suggestions(paper)
        assert suggestions == []


class TestEnrichPapers:
    def test_adds_suggestions_key(self) -> None:
        papers = [
            _make_paper(abstract="We evaluate on GLUE benchmark."),
            _make_paper(abstract="Simple abstract."),
        ]
        enriched = enrich_papers_with_suggestions(papers)

        assert all("suggestions" in p for p in enriched)
        assert len(enriched[0]["suggestions"]) > 0
        assert enriched[1]["suggestions"] == []

    def test_mutates_in_place(self) -> None:
        papers = [_make_paper()]
        result = enrich_papers_with_suggestions(papers)
        assert result is papers
