"""Tests for reporadar.llm_client (shared LLM transport)."""

from __future__ import annotations

import io
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

    def test_claude_sends_temperature_zero(self) -> None:
        """A regression guard on an omission that cost two probes to find.

        Until 2026-09-01 this path sent no temperature, so the Anthropic default of 1.0 applied
        and every call was a sample — the gate, HyDE's hypotheses, the repo summary, typed
        anchors, the second judge. NR-53 measured the judge disagreeing with itself on 8.4% of
        label decisions; NR-54 measured net@2 moving sd 1.44 per case across a re-run on a
        byte-identical pool. The failure was invisible in every exit code, so it gets a test
        that reads the wire payload rather than a comment asking for care.
        """
        cfg = SimpleNamespace(
            provider="claude", claude_api_key="k", claude_model="claude-haiku-4-5"
        )
        seen: dict = {}

        def _capture(req, *a, **kw):
            seen["payload"] = json.loads(req.data.decode())
            return _resp({"content": [{"type": "text", "text": "x"}]})

        with patch("urllib.request.urlopen", side_effect=_capture):
            complete("prompt", cfg, max_tokens=50)
        assert seen["payload"]["temperature"] == 0

    def test_claude_retries_without_temperature_when_the_model_rejects_it(self) -> None:
        """The Claude 5 family answers 400 "temperature is deprecated for this model".

        The temperature=0 change shipped without this and broke every claude-sonnet-5 call —
        155 straight 400s in NR-56 — while looking verified, because NR-55's runs exercised the
        gate on claude-haiku-4-5, which accepts the parameter. Same shape as the OpenAI-side
        retry in evals/judge.py.
        """
        cfg = SimpleNamespace(provider="claude", claude_api_key="k", claude_model="claude-sonnet-5")
        sent: list[dict] = []

        def _capture(req, *a, **kw):
            body = json.loads(req.data.decode())
            sent.append(body)
            if "temperature" in body:
                raise urllib.error.HTTPError(
                    "u",
                    400,
                    "Bad Request",
                    {},
                    io.BytesIO(b'{"error":{"message":"`temperature` is deprecated"}}'),
                )
            return _resp({"content": [{"type": "text", "text": "ok"}]})

        with patch("urllib.request.urlopen", side_effect=_capture):
            assert complete("prompt", cfg, max_tokens=50) == "ok"
        assert len(sent) == 2
        assert "temperature" in sent[0] and "temperature" not in sent[1]

    def test_the_rejection_is_learned_once_not_retried_forever(self) -> None:
        """Blind retry would pay a wasted request on every call, and 400s still consume
        request-rate budget. Discovering it from the API and remembering costs one extra
        request per model per process; a hardcoded model list would go stale instead."""
        from reporadar import llm_client

        cfg = SimpleNamespace(provider="claude", claude_api_key="k", claude_model="claude-sonnet-5")
        sent: list[dict] = []

        def _capture(req, *a, **kw):
            body = json.loads(req.data.decode())
            sent.append(body)
            if "temperature" in body:
                raise urllib.error.HTTPError(
                    "u",
                    400,
                    "Bad Request",
                    {},
                    io.BytesIO(b'{"error":{"message":"`temperature` is deprecated"}}'),
                )
            return _resp({"content": [{"type": "text", "text": "ok"}]})

        llm_client._REJECTS_TEMPERATURE.discard("claude-sonnet-5")
        try:
            with patch("urllib.request.urlopen", side_effect=_capture):
                complete("a", cfg, max_tokens=10)
                complete("b", cfg, max_tokens=10)
                complete("c", cfg, max_tokens=10)
        finally:
            llm_client._REJECTS_TEMPERATURE.discard("claude-sonnet-5")
        # 2 for the first call (reject + retry), then 1 each: 4, not 6.
        assert len(sent) == 4
        assert sum(1 for b in sent if "temperature" in b) == 1

    def test_a_model_that_accepts_temperature_never_pays_the_retry(self) -> None:
        cfg = SimpleNamespace(
            provider="claude", claude_api_key="k", claude_model="claude-haiku-4-5"
        )
        sent: list[dict] = []

        def _ok(req, *a, **kw):
            sent.append(json.loads(req.data.decode()))
            return _resp({"content": [{"type": "text", "text": "ok"}]})

        with patch("urllib.request.urlopen", side_effect=_ok):
            complete("a", cfg, max_tokens=10)
            complete("b", cfg, max_tokens=10)
        assert len(sent) == 2
        assert all("temperature" in b and b["temperature"] == 0 for b in sent)

    def test_claude_other_400s_are_not_retried_away(self) -> None:
        """The retry is narrow on purpose. A blanket one would swallow quota and rate-limit
        errors as if they were parameter problems, which is how a broken run looks healthy."""
        cfg = SimpleNamespace(provider="claude", claude_api_key="k", claude_model="claude-sonnet-5")

        def _bad(req, *a, **kw):
            raise urllib.error.HTTPError(
                "u", 400, "Bad Request", {}, io.BytesIO(b'{"error":{"message":"credit balance"}}')
            )

        with patch("urllib.request.urlopen", side_effect=_bad), pytest.raises(LLMError):
            complete("prompt", cfg, max_tokens=50)

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
