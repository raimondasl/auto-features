"""Tests for reporadar.llm_suggestions (mocked HTTP calls)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from reporadar.llm_suggestions import (
    _build_prompt,
    _parse_suggestions,
    generate_llm_suggestions,
)
from reporadar.profiler import RepoProfile


def _make_profile() -> RepoProfile:
    return RepoProfile(
        keywords=[("transformer", 0.8), ("attention", 0.6)],
        anchors=["torch", "transformers"],
        domains=["deep learning", "NLP"],
    )


def _make_paper(**overrides) -> dict:
    base = {
        "arxiv_id": "2401.12345v1",
        "title": "A Novel Approach",
        "abstract": "We propose a method for text generation using transformers.",
        "authors": ["Alice"],
        "categories": ["cs.CL"],
    }
    base.update(overrides)
    return base


@dataclass
class MockConfig:
    provider: str = "ollama"
    ollama_model: str = "llama3.2"
    ollama_url: str = "http://localhost:11434"
    claude_api_key: str = "test-key"
    claude_model: str = "claude-sonnet-4-20250514"
    max_suggestions: int = 3
    timeout: int = 30


class TestBuildPrompt:
    def test_includes_paper_info(self) -> None:
        paper = _make_paper()
        profile = _make_profile()
        prompt = _build_prompt(paper, profile)
        assert "A Novel Approach" in prompt
        assert "transformer" in prompt
        assert "torch" in prompt

    def test_includes_domains(self) -> None:
        profile = _make_profile()
        prompt = _build_prompt(_make_paper(), profile)
        assert "deep learning" in prompt
        assert "NLP" in prompt

    def test_truncates_long_abstract(self) -> None:
        paper = _make_paper(abstract="x" * 1000)
        prompt = _build_prompt(paper, _make_profile())
        # Abstract in prompt should be truncated to 500 chars
        assert len(prompt) < 2000


class TestParseSuggestions:
    def test_numbered_list(self) -> None:
        text = "1. First suggestion.\n2. Second suggestion.\n3. Third suggestion."
        result = _parse_suggestions(text)
        assert len(result) == 3
        assert result[0] == "First suggestion."

    def test_bullet_list(self) -> None:
        text = "- First idea.\n- Second idea."
        result = _parse_suggestions(text)
        assert len(result) == 2

    def test_respects_max(self) -> None:
        text = "1. A\n2. B\n3. C\n4. D\n5. E"
        result = _parse_suggestions(text, max_suggestions=2)
        assert len(result) == 2

    def test_empty_text(self) -> None:
        assert _parse_suggestions("") == []
        assert _parse_suggestions("\n\n") == []


class TestGenerateLlmSuggestions:
    def test_ollama_provider(self) -> None:
        config = MockConfig(provider="ollama")
        paper = _make_paper()
        profile = _make_profile()

        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "response": "1. Try fine-tuning.\n2. Use attention.\n3. Add training."
        }).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            result = generate_llm_suggestions(paper, profile, config)

        assert len(result) == 3
        assert "fine-tuning" in result[0].lower()

    def test_claude_provider(self) -> None:
        config = MockConfig(provider="claude")
        paper = _make_paper()
        profile = _make_profile()

        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "content": [
                {
                    "type": "text",
                    "text": "1. Integrate transformer.\n2. Add attention.\n3. Try training.",
                }
            ]
        }).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            result = generate_llm_suggestions(paper, profile, config)

        assert len(result) == 3

    def test_claude_no_api_key_raises(self) -> None:
        config = MockConfig(provider="claude", claude_api_key="")
        paper = _make_paper()
        profile = _make_profile()

        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="No Claude API key"):
                generate_llm_suggestions(paper, profile, config)

    def test_network_error_propagates(self) -> None:
        import urllib.error

        config = MockConfig(provider="ollama")
        paper = _make_paper()
        profile = _make_profile()

        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("Connection refused"),
        ):
            with pytest.raises(urllib.error.URLError):
                generate_llm_suggestions(paper, profile, config)


class TestEnrichWithLlmFallback:
    def test_falls_back_to_template_on_error(self) -> None:
        from reporadar.suggestions import enrich_papers_with_suggestions

        config = MockConfig(provider="ollama")
        papers = [_make_paper(abstract="We evaluate on GLUE benchmark.")]

        with patch(
            "reporadar.llm_suggestions.generate_llm_suggestions",
            side_effect=Exception("Connection refused"),
        ):
            result = enrich_papers_with_suggestions(papers, config=config, profile=_make_profile())

        # Should have fallen back to template-based suggestions
        assert "suggestions" in result[0]

    def test_template_when_no_config(self) -> None:
        from reporadar.suggestions import enrich_papers_with_suggestions

        papers = [_make_paper(abstract="We evaluate on GLUE benchmark.")]
        result = enrich_papers_with_suggestions(papers)
        assert "suggestions" in result[0]
