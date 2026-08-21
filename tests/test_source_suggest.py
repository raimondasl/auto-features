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
    def test_bio_packages_suggest_europepmc(self) -> None:
        profile = _profile(anchors=["scanpy", "anndata", "numpy"])
        suggestions = suggest_sources(profile, ["arxiv"])
        assert [s.source for s in suggestions] == ["europepmc"]
        assert "scanpy" in suggestions[0].evidence

    def test_bio_packages_never_suggest_the_broken_biorxiv_adapter(self) -> None:
        """The defect this correction is for, stated as an invariant.

        `biorxiv`'s details endpoint is a date-interval listing, so under the product's own
        default lookback it returns 2013-2016 postings rather than papers about the repo.
        Recommending it sent bio users — exactly the users this module exists for — to the one
        channel that could not answer them. §21 measured `europepmc` instead.
        """
        for anchors in (["scanpy"], ["biopython"], ["anndata", "scanpy"]):
            sources = [s.source for s in suggest_sources(_profile(anchors=anchors), ["arxiv"])]
            assert "biorxiv" not in sources, f"{anchors} still routed to the broken adapter"

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

    def test_europepmc_suggestion_carries_the_cost_that_was_measured(self) -> None:
        """Europe PMC's cost is not rate limits or coverage — it is DISPLACEMENT.

        §21.4: it supplied over half of every candidate pool and pushed 44% of the arXiv-only
        run's Top Picks out of the window. A user told only "it adds biology" would be
        surprised when papers they were watching disappear, so the caveat says so.
        """
        (suggestion,) = suggest_sources(_profile(anchors=["biopython"]), ["arxiv"])
        assert "competes" in suggestion.caveat
        assert "44%" in suggestion.caveat
        assert "Note:" in format_suggestion(suggestion)

    def test_the_europepmc_reason_cites_the_measurement_not_a_hope(self) -> None:
        """The old text said "coverage, not a measured improvement". It is measured now, and
        the claim is precision-matching rather than net@2 improvement, which is unresolved."""
        (suggestion,) = suggest_sources(_profile(anchors=["biopython"]), ["arxiv"])
        assert "measured" in suggestion.reason
        assert "precision" in suggestion.reason

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
        assert [s.source for s in suggest_sources(profile, ["arxiv"])] == ["europepmc"]

    def test_never_suggests_an_active_source(self) -> None:
        profile = _profile(anchors=["scanpy", "duckdb"])
        active = ["arxiv", "europepmc", "dblp"]
        assert suggest_sources(profile, active) == []

    def test_source_signals_count_as_evidence(self) -> None:
        # Source scanning (profiler.scan_source) surfaces patterns the manifests miss.
        profile = _profile(signals=["genome assembly", "crispr screen"])
        assert [s.source for s in suggest_sources(profile, ["arxiv"])] == ["europepmc"]

    def test_both_sources_ranked_by_evidence_strength(self) -> None:
        profile = _profile(
            keywords=["genomics", "rna", "dna"],
            anchors=["scanpy", "anndata", "duckdb"],
            domains=["databases"],
        )
        sources = [s.source for s in suggest_sources(profile, ["arxiv"])]
        assert sources == ["europepmc", "dblp"]  # more bio evidence -> listed first

    def test_anchor_matching_ignores_separators_and_case(self) -> None:
        profile = _profile(anchors=["Scikit_Bio"])
        assert [s.source for s in suggest_sources(profile, ["arxiv"])] == ["europepmc"]

    def test_empty_profile_is_silent(self) -> None:
        assert suggest_sources(_profile(), ["arxiv"]) == []


class TestFormatSuggestion:
    def test_includes_source_reason_and_evidence(self) -> None:
        (suggestion,) = suggest_sources(_profile(anchors=["biopython"]), ["arxiv"])
        line = format_suggestion(suggestion)
        assert "europepmc" in line
        assert "bioRxiv/medRxiv" in line
        assert "biopython" in line
        assert "\n" not in line  # stays a single CLI line
