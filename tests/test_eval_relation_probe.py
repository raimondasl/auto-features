"""The relation probe, and the void-not-null bug it shipped with for one run.

The probe answers whether roadmap item 16's relation claim can be *grounded* — whether a
repository's own vocabulary appears in the abstracts of papers judged actionable. Its
first run reported **0.0% keyword hits in all three strata**, which reads as the finding
"abstracts never mention these" and was in fact "the extractor returned nothing":
`profile.keywords` is a list of `(term, weight)` pairs, and a `len(term) >= 3` filter
applied to the pair measured the tuple's length of 2 and discarded every keyword.

That is the exact shape this project has corrected repeatedly — the pool scanner that read
1,250 papers as 0, the ablation guard that read a constant instead of the arm's setting —
and it happened inside a probe written to look for it. So the tests below pin the two
things that make it not happen twice: the pair-unpacking, and a guard that refuses to
report a term class extracted as empty everywhere.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))


class TestKeywordsArePairsNotStrings:
    """The bug, pinned at the boundary where it entered."""

    def test_the_profiler_still_returns_pairs(self) -> None:
        """If this ever becomes a list of strings, `_repo_terms` must be revisited — and
        this failing is how anyone would find out."""
        from reporadar.profiler import RepoProfile

        annotation = RepoProfile.__annotations__["keywords"]
        assert "tuple" in str(annotation), (
            f"keywords is annotated {annotation!r}; the probe unpacks (term, weight) pairs"
        )

    def test_terms_survive_the_length_filter(self, tmp_path: Path) -> None:
        """A regression test for the real defect: the filter must see the TERM's length,
        not the pair's. Every pair has length 2, so a pair-length filter at >= 3 empties
        the set silently."""
        import relation_probe as probe

        repo = tmp_path / "case"
        repo.mkdir()
        (repo / "README.md").write_text(
            "Vector similarity search over embeddings with faiss and product quantization.",
            encoding="utf-8",
        )
        (repo / "requirements.txt").write_text(
            "faiss-cpu\nsentence-transformers\n", encoding="utf-8"
        )
        monkey_work = probe.WORK
        try:
            probe.WORK = tmp_path
            anchors, keywords = probe._repo_terms("case")
        finally:
            probe.WORK = monkey_work
        assert keywords, "keywords came back empty — the pair-length bug is back"
        assert all(isinstance(k, str) for k in keywords)
        assert all(len(k) >= probe.MIN_ANCHOR_LEN for k in keywords)


class TestTheGuardAgainstReportingNothing:
    def test_it_fires_when_a_class_is_empty_everywhere(self) -> None:
        import relation_probe as probe

        per_case = [{"case": "a", "anchors": 0, "keywords": 0, "judged": 5}]
        rows = [{"relations": {"improves"}}]
        with pytest.raises(SystemExit, match="void read as null"):
            probe._no_empty_class(per_case, rows)

    def test_it_stays_quiet_when_extraction_worked(self) -> None:
        import relation_probe as probe

        per_case = [{"case": "a", "anchors": 2, "keywords": 9, "judged": 5}]
        probe._no_empty_class(per_case, [{"relations": {"improves"}}])

    def test_it_also_catches_a_dead_relation_matcher(self) -> None:
        import relation_probe as probe

        per_case = [{"case": "a", "anchors": 2, "keywords": 9, "judged": 5}]
        with pytest.raises(SystemExit, match="matcher is broken"):
            probe._no_empty_class(per_case, [{"relations": set()}])


class TestMatching:
    def test_single_words_match_as_whole_tokens(self) -> None:
        """`ann` must not match `annotation`, or every repo looks grounded."""
        import relation_probe as probe

        assert probe._mentions("an annotation pipeline", {"ann"}) == set()
        assert probe._mentions("we use ann search", {"ann"}) == {"ann"}

    def test_multi_word_terms_match_as_phrases(self) -> None:
        """The profiler emits bigrams; a token check can never match them, so they would
        silently contribute nothing — the same failure one level down."""
        import relation_probe as probe

        assert probe._mentions("fast similarity search at scale", {"similarity search"}) == {
            "similarity search"
        }
        assert probe._mentions("similarity of search terms", {"similarity search"}) == set()

    def test_empty_inputs_are_not_errors(self) -> None:
        import relation_probe as probe

        assert probe._mentions("", {"faiss"}) == set()
        assert probe._mentions("faiss", set()) == set()


class TestRelationVocabulary:
    def test_the_verbs_are_the_roadmap_ones(self) -> None:
        """Chosen before the data was seen. Tuning the vocabulary after looking is how a
        null becomes a result."""
        import relation_probe as probe

        assert set(probe.RELATION_VERBS) == {"improves", "replaces", "extends", "uses"}

    def test_it_detects_what_it_claims_to(self) -> None:
        import relation_probe as probe

        assert "replaces" in probe._relations("A drop-in alternative to IVF-PQ indexes")
        assert "improves" in probe._relations("Our method outperforms the baseline by 3x")
        assert probe._relations("A survey of the field") == set()


def test_it_runs_on_the_real_artifacts() -> None:
    """Skipped where the gitignored artifacts are absent; on the eval machine it must
    still parse today's pools, profiles and verdicts."""
    import relation_probe as probe

    if not probe.POOL.exists() or not probe.JUDGE.exists():
        pytest.skip("run artifacts are gitignored; present only on the eval machine")
    assert probe.main() == 0


def test_the_probe_costs_nothing() -> None:
    """A '$0 probe' that reaches the network is not a $0 probe. Static check, because the
    claim is in the docstring and the file name."""
    src = (ROOT / "evals" / "relation_probe.py").read_text(encoding="utf-8")
    body = src.split('"""', 2)[2]
    for banned in ("urlopen", "requests.", "judge_paper", "llm_client", "openai", "anthropic"):
        assert banned not in body, f"{banned} would make this probe cost money or time"
