"""Tests for the eval harness's reference parsing (evals/baseline.py, verify.py).

Regression guard for the prose-scraping bug: an honest baseline abstention (an
empty ```json [] block) was overridden by arXiv-looking IDs scraped from the
surrounding "sources reviewed" prose — including a ResearchGate URL path
(``publication/2256929``) that the old-style-ID regex wrongly matched. That
bogus ID then 400'd against arXiv and nuked the baseline's metrics as
``arxiv_unverified`` on every webdev run.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import baseline
from baseline import _parse_cli_payload, _parse_recommendations, _run_cli
from verify import extract_arxiv_ids


def _proc(returncode: int, stdout: str = "", stderr: str = "") -> SimpleNamespace:
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


def _ok_stdout(result: str, cost: float = 0.5) -> str:
    return json.dumps({"subtype": "success", "result": result, "total_cost_usd": cost})


class TestExtractArxivIds:
    def test_new_style(self) -> None:
        assert extract_arxiv_ids("arXiv:2410.14924 and 2301.01261") == [
            "2410.14924",
            "2301.01261",
        ]

    def test_old_style_real_archives(self) -> None:
        assert extract_arxiv_ids("see hep-th/9901001 and cs.LG/0501001") == [
            "hep-th/9901001",
            "cs.LG/0501001",
        ]

    def test_rejects_non_arxiv_url_path(self) -> None:
        # researchgate.net/publication/225692935_... must NOT parse as old-style ID.
        assert extract_arxiv_ids("researchgate.net/publication/225692935_Hardened") == []

    def test_dedupes(self) -> None:
        assert extract_arxiv_ids("2410.14924 ... again 2410.14924") == ["2410.14924"]


class TestParseRecommendations:
    def test_json_block_is_authoritative(self) -> None:
        # An ID mentioned only in prose is discussion, not a recommendation.
        text = (
            "I reviewed arXiv:2304.01982 but it's a replacement model, not an improvement.\n"
            '```json\n[{"arxiv_id": "2409.14683"}]\n```'
        )
        ids, titles = _parse_recommendations(text)
        assert ids == ["2409.14683"]
        assert titles == []

    def test_empty_json_block_is_an_abstention(self) -> None:
        # The webdev bug: an explicit [] plus prose "sources reviewed" must yield NO refs.
        text = (
            "Sources reviewed: https://arxiv.org/pdf/2410.14924 and "
            "researchgate.net/publication/225692935_Foo.\n"
            "My recommendation is to recommend nothing.\n```json\n[]\n```"
        )
        assert _parse_recommendations(text) == ([], [])

    def test_prose_fallback_when_no_json_block(self) -> None:
        # If the baseline ignored the protocol (no JSON block), fall back to prose.
        ids, titles = _parse_recommendations("I recommend arXiv:2409.14683 and 2404.02805.")
        assert ids == ["2409.14683", "2404.02805"]
        assert titles == []

    def test_title_only_recommendations(self) -> None:
        ids, titles = _parse_recommendations(
            '```json\n[{"title": "Some Paper Without An Arxiv Id"}]\n```'
        )
        assert ids == []
        assert titles == ["Some Paper Without An Arxiv Id"]


class TestParseCliPayload:
    def test_ok(self) -> None:
        ok, reason = _parse_cli_payload(_ok_stdout("```json\n[]\n```", cost=0.7))
        assert reason == ""
        assert ok is not None and ok["status"] == "ok" and ok["cost_usd"] == 0.7

    def test_non_json(self) -> None:
        ok, reason = _parse_cli_payload("not json at all")
        assert ok is None and "non-JSON" in reason

    def test_is_error_flag(self) -> None:
        ok, reason = _parse_cli_payload(json.dumps({"is_error": True, "result": "x"}))
        assert ok is None and "reported failure" in reason

    def test_empty_result(self) -> None:
        ok, reason = _parse_cli_payload(json.dumps({"subtype": "success", "result": "   "}))
        assert ok is None and "empty result" in reason


class TestRunCliRetry:
    def test_transient_failure_then_success(self) -> None:
        # First call fails (exit 1), retry succeeds — the eval should not lose the baseline.
        calls = [_proc(1, stderr="segfault"), _proc(0, stdout=_ok_stdout("```json\n[]\n```"))]
        with (
            patch("baseline.subprocess.run", side_effect=calls) as run,
            patch("baseline.time.sleep"),
        ):
            out = _run_cli(SimpleNamespace(), flags=[], timeout=10)
        assert out["status"] == "ok"
        assert run.call_count == 2

    def test_strips_api_key_on_auth_conflict(self) -> None:
        # The claude.ai-login-vs-ANTHROPIC_API_KEY conflict: retry drops the key
        # from the subprocess env so the CLI uses its own login.
        warn = (
            "connectors are disabled because ANTHROPIC_API_KEY takes precedence over "
            "your claude.ai login"
        )
        calls = [_proc(1, stderr=warn), _proc(0, stdout=_ok_stdout("```json\n[]\n```"))]
        seen_envs = []

        def record(cmd, **kw):
            seen_envs.append(kw.get("env") or {})
            return calls[len(seen_envs) - 1]

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test"}, clear=False),
            patch("baseline.subprocess.run", side_effect=record),
            patch("baseline.time.sleep"),
        ):
            out = _run_cli(SimpleNamespace(), flags=[], timeout=10)
        assert out["status"] == "ok"
        assert "ANTHROPIC_API_KEY" in seen_envs[0]  # first attempt keeps the key
        assert "ANTHROPIC_API_KEY" not in seen_envs[1]  # retry drops it

    def test_gives_up_after_max_retries(self) -> None:
        with (
            patch("baseline.subprocess.run", return_value=_proc(1, stderr="boom")) as run,
            patch("baseline.time.sleep"),
        ):
            out = _run_cli(SimpleNamespace(), flags=[], timeout=10)
        assert out["status"] == "error"
        assert run.call_count == baseline._CLI_MAX_RETRIES + 1
