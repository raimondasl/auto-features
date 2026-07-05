"""Tests for reporadar.triage (LLM actionability scoring)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from reporadar.llm_client import LLMError
from reporadar.triage import (
    _parse_verdict,
    build_triage_prompt,
    score_actionability,
    triage_papers,
)

_PROFILE = SimpleNamespace(
    keywords=[("retrieval", 0.5), ("ranking", 0.3)],
    anchors=["faiss", "sentence-transformers"],
    domains=["information retrieval"],
)
_PAPER = {
    "arxiv_id": "2401.00001",
    "title": "A Retrieval Method",
    "abstract": "We improve ANN search.",
}


class TestParseVerdict:
    def test_valid(self) -> None:
        assert _parse_verdict('{"score": 2, "reason": "applies to ANN"}') == (2, "applies to ANN")

    def test_embedded_in_prose(self) -> None:
        raw = 'Here is my verdict:\n{"score": 3, "reason": "direct fit"}\nThanks.'
        assert _parse_verdict(raw) == (3, "direct fit")

    def test_missing_score_raises(self) -> None:
        with pytest.raises(ValueError, match="missing 'score'"):
            _parse_verdict('{"reason": "no score here"}')

    def test_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="out of range"):
            _parse_verdict('{"score": 5, "reason": "too high"}')

    def test_no_json_raises(self) -> None:
        with pytest.raises(ValueError, match="no JSON"):
            _parse_verdict("the paper is great, score 2")


class TestBuildPrompt:
    def test_includes_repo_and_paper(self) -> None:
        prompt = build_triage_prompt(_PAPER, _PROFILE)
        assert "faiss" in prompt
        assert "information retrieval" in prompt
        assert "A Retrieval Method" in prompt
        assert '{"score"' in prompt  # rubric asks for JSON


class TestScoreActionability:
    def test_returns_parsed_verdict(self) -> None:
        with patch("reporadar.triage.complete", return_value='{"score": 2, "reason": "ok"}'):
            assert score_actionability(_PAPER, _PROFILE, SimpleNamespace()) == (2, "ok")


class TestTriagePapers:
    def test_scores_and_skips_failures(self) -> None:
        p1 = {**_PAPER, "arxiv_id": "2401.00001"}
        p2 = {**_PAPER, "arxiv_id": "2401.00002"}
        p3 = {**_PAPER, "arxiv_id": "2401.00003"}

        # Call 1 succeeds; call 2 errors (LLM); call 3 returns malformed output.
        # Only the successful paper should appear — failures are skipped, never 0.
        calls = {"n": 0}

        def by_call(prompt, cfg, **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                return '{"score": 3, "reason": "great"}'
            if calls["n"] == 2:
                raise LLMError("network down")
            return "not json"

        with patch("reporadar.triage.complete", side_effect=by_call):
            out = triage_papers([p1, p2, p3], _PROFILE, SimpleNamespace(), top_k=10)

        assert set(out) == {"2401.00001"}
        assert out["2401.00001"] == {"llm_score": 3, "llm_reason": "great"}

    def test_respects_top_k(self) -> None:
        papers = [{**_PAPER, "arxiv_id": f"2401.{i:05d}"} for i in range(10)]
        with patch("reporadar.triage.complete", return_value='{"score": 2, "reason": "x"}'):
            out = triage_papers(papers, _PROFILE, SimpleNamespace(), top_k=3)
        assert len(out) == 3
