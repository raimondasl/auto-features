"""Tests for reporadar.llm_client (shared LLM transport)."""

from __future__ import annotations

import json
import urllib.error
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from reporadar.llm_client import LLMError, complete


def _resp(payload: dict) -> MagicMock:
    m = MagicMock()
    m.read.return_value = json.dumps(payload).encode()
    m.__enter__ = lambda s: s
    m.__exit__ = MagicMock(return_value=False)
    return m


class TestComplete:
    def test_ollama_returns_response(self) -> None:
        cfg = SimpleNamespace(provider="ollama", ollama_model="llama3.2", timeout=5)
        with patch("urllib.request.urlopen", return_value=_resp({"response": "hello"})):
            assert complete("prompt", cfg) == "hello"

    def test_claude_joins_text_blocks(self) -> None:
        cfg = SimpleNamespace(
            provider="claude", claude_api_key="k", claude_model="claude-haiku-4-5"
        )
        payload = {"content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}
        with patch("urllib.request.urlopen", return_value=_resp(payload)):
            assert complete("prompt", cfg, max_tokens=50) == "a\nb"

    def test_claude_no_key_raises_llmerror(self) -> None:
        cfg = SimpleNamespace(provider="claude", claude_api_key="")
        with (
            patch.dict("os.environ", {}, clear=True),
            pytest.raises(LLMError, match="No Claude API key"),
        ):
            complete("prompt", cfg)

    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(LLMError, match="Unknown LLM provider"):
            complete("prompt", SimpleNamespace(provider="gpt4"))

    def test_network_error_retries_then_llmerror(self) -> None:
        cfg = SimpleNamespace(provider="ollama", timeout=5)
        with (
            patch("reporadar.llm_client.time.sleep") as sleep,
            patch(
                "urllib.request.urlopen",
                side_effect=urllib.error.URLError("refused"),
            ) as urlopen,
            pytest.raises(LLMError, match="failed after"),
        ):
            complete("prompt", cfg, max_retries=2)
        assert urlopen.call_count == 3  # initial + 2 retries
        assert sleep.call_count == 2

    def test_client_http_error_not_retried(self) -> None:
        # A 400 is a permanent error — raise immediately, don't burn retries.
        cfg = SimpleNamespace(provider="claude", claude_api_key="k")
        err = urllib.error.HTTPError("u", 400, "bad", {}, None)  # type: ignore[arg-type]
        with (
            patch("urllib.request.urlopen", side_effect=err) as urlopen,
            pytest.raises(LLMError, match="HTTP 400"),
        ):
            complete("prompt", cfg)
        assert urlopen.call_count == 1
