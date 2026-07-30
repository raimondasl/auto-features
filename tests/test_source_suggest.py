"""Tests for reporadar.sources.suggest."""

from __future__ import annotations

from reporadar.profiler import RepoProfile
from reporadar.sources.suggest import format_suggestion, suggest_sources


def _profile(
    keywords: list[str] | None = None,
    anchors: list[str] | None = None,
    domains: list[str] | None = None,
    signals: list[str] | None = None,
) -> RepoProfile:
    return RepoProfile(
        keywords=[(k, 0.5) for k in (keywords or [])],
        anchors=anchors or [],
        domains=domains or [],
        source_signals=signals or [],
    )


class TestSuggestSources:
    def test_bio_packages_suggest_biorxiv(self) -> None:
        profile = _profile(anchors=["scanpy", "anndata", "numpy"])
        suggestions = suggest_sources(profile, ["arxiv"])
        assert [s.source for s in suggestions] == ["biorxiv"]
        assert "scanpy" in suggestions[0].evidence

    def test_systems_packages_suggest_dblp(self) -> None:
        profile = _profile(anchors=["duckdb"], domains=["databases"])
        suggestions = suggest_sources(profile, ["arxiv"])
        assert [s.source for s in suggestions] == ["dblp"]

    def test_dblp_suggestion_carries_its_caveat(self) -> None:
        # DBLP is rate-limited and year-granular; suggesting it without saying so
        # would set the user up for slow runs and an empty recency window.
        profile = _profile(anchors=["rocksdb"])
        (suggestion,) = suggest_sources(profile, ["arxiv"])
        assert "year" in suggestion.caveat
        assert "rate-limited" in suggestion.caveat
        assert "year" in format_suggestion(suggestion)

    def test_biorxiv_suggestion_carries_its_caveat(self) -> None:
        # bioRxiv's cost is coverage, not rate limits: no keyword search, so a run
        # pages the whole window and a wide lookback can truncate.
        (suggestion,) = suggest_sources(_profile(anchors=["biopython"]), ["arxiv"])
        assert "no keyword search" in suggestion.caveat
        assert "Note:" in format_suggestion(suggestion)

    def test_quiet_for_a_realistic_serving_stack(self) -> None:
        # torch + fastapi + redis + grpcio + kubernetes is a model-serving repo whose
        # papers are on arXiv. Commodity infrastructure must not read as DBLP evidence.
        profile = _profile(
            keywords=["model serving latency", "inference runtime", "request scheduler"],
            anchors=["torch", "fastapi", "redis", "grpcio", "kubernetes", "sqlalchemy"],
            domains=["deep learning", "web APIs", "containers", "databases"],
        )
        assert suggest_sources(profile, ["arxiv"]) == []

    def test_deduplicates_evidence(self) -> None:
        # "duckdb" is both a detected package and a keyword token here.
        profile = _profile(keywords=["duckdb query optimizer"], anchors=["duckdb"])
        (suggestion,) = suggest_sources(profile, ["arxiv"])
        assert len(suggestion.evidence) == len(set(suggestion.evidence))

    def test_quiet_for_a_plain_ml_repo(self) -> None:
        profile = _profile(
            keywords=["retrieval augmented generation", "transformer attention"],
            anchors=["torch", "transformers"],
            domains=["deep learning", "NLP"],
        )
        assert suggest_sources(profile, ["arxiv"]) == []

    def test_ml_vocabulary_does_not_trigger_dblp(self) -> None:
        # "distributed" (training), "scheduler" (learning-rate) and "kernel" (CUDA)
        # are ML words too — a repo arXiv serves well must not be pushed to DBLP.
        profile = _profile(
            keywords=["distributed training", "learning rate scheduler", "cuda kernel fusion"],
            anchors=["torch"],
            domains=["deep learning"],
        )
        assert suggest_sources(profile, ["arxiv"]) == []

    def test_one_loose_keyword_is_not_enough(self) -> None:
        # A single passing mention of "protein" must not nag the user; anchors are
        # decisive, bare keywords need corroboration.
        profile = _profile(keywords=["protein folding demo"], anchors=["torch"])
        assert suggest_sources(profile, ["arxiv"]) == []

    def test_two_keywords_clear_the_bar(self) -> None:
        profile = _profile(keywords=["genomics pipeline", "rna sequencing"])
        assert [s.source for s in suggest_sources(profile, ["arxiv"])] == ["biorxiv"]

    def test_never_suggests_an_active_source(self) -> None:
        profile = _profile(anchors=["scanpy", "duckdb"])
        active = ["arxiv", "biorxiv", "dblp"]
        assert suggest_sources(profile, active) == []

    def test_source_signals_count_as_evidence(self) -> None:
        # Source scanning (profiler.scan_source) surfaces patterns the manifests miss.
        profile = _profile(signals=["genome assembly", "crispr screen"])
        assert [s.source for s in suggest_sources(profile, ["arxiv"])] == ["biorxiv"]

    def test_both_sources_ranked_by_evidence_strength(self) -> None:
        profile = _profile(
            keywords=["genomics", "rna", "dna"],
            anchors=["scanpy", "anndata", "duckdb"],
            domains=["databases"],
        )
        sources = [s.source for s in suggest_sources(profile, ["arxiv"])]
        assert sources == ["biorxiv", "dblp"]  # more bio evidence -> listed first

    def test_anchor_matching_ignores_separators_and_case(self) -> None:
        profile = _profile(anchors=["Scikit_Bio"])
        assert [s.source for s in suggest_sources(profile, ["arxiv"])] == ["biorxiv"]

    def test_empty_profile_is_silent(self) -> None:
        assert suggest_sources(_profile(), ["arxiv"]) == []


class TestFormatSuggestion:
    def test_includes_source_reason_and_evidence(self) -> None:
        (suggestion,) = suggest_sources(_profile(anchors=["biopython"]), ["arxiv"])
        line = format_suggestion(suggestion)
        assert "biorxiv" in line
        assert "bioRxiv/medRxiv" in line
        assert "biopython" in line
        assert "\n" not in line  # stays a single CLI line
