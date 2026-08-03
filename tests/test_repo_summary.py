"""Tests for reporadar.repo_summary — the LLM reading of what a repo is and lacks."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from reporadar.repo_summary import (
    RepoSummary,
    _parse_summary,
    _verbatim_only,
    collect_doc_corpus,
    summarize_repo,
)

_CFG = SimpleNamespace(provider="claude", timeout=5)
_GOOD = (
    '{"purpose": "Late-interaction passage retrieval over BERT.",'
    ' "capabilities": ["MaxSim scoring", "quantized index"],'
    ' "improvement_areas": ["adaptive quantization"],'
    ' "key_sentences": ["ColBERT encodes each passage into a matrix of embeddings."]}'
)
_CORPUS = "ColBERT encodes each passage into a matrix of embeddings. Other text."


def _repo(tmp_path: Path, readme: str) -> Path:
    (tmp_path / "README.md").write_text(readme, encoding="utf-8")
    return tmp_path


class TestParseSummary:
    def test_parses_every_field(self) -> None:
        s = _parse_summary(_GOOD, _CORPUS, "h1")
        assert s.purpose.startswith("Late-interaction")
        assert s.capabilities == ["MaxSim scoring", "quantized index"]
        assert s.improvement_areas == ["adaptive quantization"]
        assert s.key_sentences == ["ColBERT encodes each passage into a matrix of embeddings."]
        assert s.source_hash == "h1"

    def test_a_summary_with_no_purpose_is_rejected(self) -> None:
        """Purpose is the one field this exists to produce.

        Accepting a purpose-less summary would put an empty description into every triage
        prompt for that repo while still looking like a success to the caller.
        """
        with pytest.raises(ValueError, match="no purpose"):
            _parse_summary('{"purpose": "  ", "capabilities": ["x"]}', _CORPUS, "h")

    def test_no_json_raises(self) -> None:
        with pytest.raises(ValueError, match="no JSON"):
            _parse_summary("I could not read that repository, sorry.", _CORPUS, "h")

    def test_a_non_list_field_raises_rather_than_being_coerced(self) -> None:
        raw = '{"purpose": "p", "capabilities": "MaxSim, quantized index"}'
        with pytest.raises(ValueError, match="not a list"):
            _parse_summary(raw, _CORPUS, "h")


class TestVerbatimGuard:
    """The excerpt path is only worth anything if the sentences are really the repo's."""

    def test_keeps_a_sentence_that_occurs_in_the_source(self) -> None:
        assert _verbatim_only(["ColBERT encodes each passage into a matrix"], _CORPUS)

    def test_drops_a_paraphrase(self) -> None:
        """A tidied 'quote' is a paraphrase wearing a quote's clothes.

        Downstream nothing can tell the difference, so it has to be caught here while the
        source text is still in hand — otherwise the whole point of extracting verbatim
        text (keeping the repo's own vocabulary) is silently lost.
        """
        assert _verbatim_only(["ColBERT encodes passages as embedding matrices"], _CORPUS) == []

    def test_ignores_whitespace_differences_only(self) -> None:
        assert _verbatim_only(["ColBERT   encodes each\npassage into a matrix"], _CORPUS)

    def test_drops_a_fragment_too_short_to_mean_anything(self) -> None:
        """A three-word 'sentence' matches almost any corpus and carries no signal."""
        assert _verbatim_only(["ColBERT"], _CORPUS) == []

    def test_parse_applies_the_guard(self) -> None:
        raw = _GOOD.replace(
            "ColBERT encodes each passage into a matrix of embeddings.",
            "ColBERT represents passages using embedding matrices.",
        )
        assert _parse_summary(raw, _CORPUS, "h").key_sentences == []


class TestPromptBlocks:
    def _summary(self, **kw: object) -> RepoSummary:
        base = {
            "purpose": "Late-interaction retrieval.",
            "capabilities": ["MaxSim"],
            "improvement_areas": ["adaptive quantization"],
            "key_sentences": ["ColBERT encodes each passage into a matrix."],
        }
        base.update(kw)
        return RepoSummary(**base)  # type: ignore[arg-type]

    def test_include_gaps_false_drops_only_the_gaps(self) -> None:
        block = self._summary().as_prompt_block(include_gaps=False)
        assert "adaptive quantization" not in block
        assert "Late-interaction retrieval." in block
        assert "MaxSim" in block

    def test_an_empty_summary_produces_no_block(self) -> None:
        """An empty labelled section reads as 'the project said nothing about itself'."""
        assert RepoSummary().as_prompt_block() == ""
        assert RepoSummary().as_excerpt_block() == ""

    def test_no_gaps_means_no_gaps_header(self) -> None:
        block = self._summary(improvement_areas=[]).as_prompt_block()
        assert "could improve" not in block

    def test_excerpt_block_is_only_the_repo_words(self) -> None:
        block = self._summary().as_excerpt_block()
        assert "ColBERT encodes each passage into a matrix." in block
        assert "MaxSim" not in block  # capabilities are the model's words, not the repo's
        assert "adaptive quantization" not in block


class TestCache:
    def test_a_hash_hit_makes_no_model_call(self, tmp_path: Path) -> None:
        repo = _repo(tmp_path, "# T\n\nColBERT encodes each passage into a matrix.\n")
        cache: dict = {}
        with patch("reporadar.repo_summary.complete", return_value=_GOOD) as m:
            first = summarize_repo(repo, _CFG, cache=cache)
            second = summarize_repo(repo, _CFG, cache=cache)
        assert m.call_count == 1
        assert first.purpose == second.purpose

    def test_changed_docs_invalidate_the_cache(self, tmp_path: Path) -> None:
        repo = _repo(tmp_path, "# T\n\nColBERT encodes each passage into a matrix.\n")
        cache: dict = {}
        with patch("reporadar.repo_summary.complete", return_value=_GOOD) as m:
            summarize_repo(repo, _CFG, cache=cache)
            (repo / "README.md").write_text("# T\n\nSomething else entirely.\n", encoding="utf-8")
            summarize_repo(repo, _CFG, cache=cache)
        assert m.call_count == 2

    def test_a_summariser_change_invalidates_the_cache(self, tmp_path: Path) -> None:
        """The key hashes the docs, so without a version the prompt could change silently.

        A cache keyed on inputs alone happily serves a summary produced by a prompt that
        no longer exists — the stalest possible result, and invisible.
        """
        repo = _repo(tmp_path, "# T\n\nColBERT encodes each passage into a matrix.\n")
        cache: dict = {}
        with patch("reporadar.repo_summary.complete", return_value=_GOOD) as m:
            summarize_repo(repo, _CFG, cache=cache)
            with patch("reporadar.repo_summary._PROMPT_VERSION", 99):
                summarize_repo(repo, _CFG, cache=cache)
        assert m.call_count == 2

    def test_a_repo_with_no_docs_costs_nothing(self, tmp_path: Path) -> None:
        with patch("reporadar.repo_summary.complete") as m:
            assert summarize_repo(tmp_path, _CFG) == RepoSummary()
        m.assert_not_called()


class TestCorpus:
    def test_readme_and_metadata_both_reach_the_corpus(self, tmp_path: Path) -> None:
        """Unlike the profiler's corpus, neither source displaces the other.

        `_collect_text_corpus(repo)[0]` returning the packaging one-liner instead of the
        README is what invalidated an earlier measurement; here they are labelled and both
        present, so there is no element-0 to guess wrong about.
        """
        repo = _repo(tmp_path, "# Thing\n\nDoes late interaction.\n")
        (repo / "pyproject.toml").write_text(
            '[project]\nname = "thing"\ndescription = "A tagline."\n', encoding="utf-8"
        )
        corpus = collect_doc_corpus(repo)
        assert "Does late interaction." in corpus
        assert "A tagline." in corpus

    def test_a_huge_readme_cannot_blow_the_input_budget(self, tmp_path: Path) -> None:
        from reporadar.repo_summary import MAX_INPUT_CHARS

        repo = _repo(tmp_path, "# T\n\n" + ("retrieval " * 50_000))
        assert len(collect_doc_corpus(repo)) <= MAX_INPUT_CHARS
