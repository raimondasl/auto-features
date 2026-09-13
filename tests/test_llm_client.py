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


class TestThePromptCacheBreakpoint:
    """The context is ~86% of a judge prompt and repeats for every paper in a case. Splitting
    it off behind a cache breakpoint changes what the call is billed, never what it reads."""

    @staticmethod
    def _body_of(mock_urlopen: MagicMock) -> dict:
        return json.loads(mock_urlopen.call_args[0][0].data.decode("utf-8"))

    def _run(self, prompt: str, **kw: object) -> dict:
        cfg = SimpleNamespace(
            provider="claude", claude_api_key="k", claude_model="claude-sonnet-5", timeout=5
        )
        payload = {"content": [{"type": "text", "text": "ok"}]}
        with patch("urllib.request.urlopen", return_value=_resp(payload)) as m:
            complete(prompt, cfg, **kw)  # type: ignore[arg-type]
            return self._body_of(m)

    def test_without_the_marker_the_request_is_a_plain_string(self) -> None:
        """The default must be byte-identical to a request built before this existed —
        every other caller of `complete` in the product goes down this path."""
        long = "x" * 9000
        assert self._run(long)["messages"][0]["content"] == long

    def test_the_split_preserves_the_prompt_exactly(self) -> None:
        prompt = "# Repository context\n" + "y" * 9000 + "\n\n# Candidate paper\nTitle: T"
        blocks = self._run(prompt, cache_split_on="# Candidate paper")["messages"][0]["content"]
        assert isinstance(blocks, list) and len(blocks) == 2
        # The rendered text is the prompt, unchanged and in order. A breakpoint that dropped
        # or duplicated a byte would be a different prompt sent to a judge under a rubric
        # that requires the two arms be shown identical text.
        assert "".join(b["text"] for b in blocks) == prompt
        assert blocks[0]["cache_control"] == {"type": "ephemeral"}
        assert "cache_control" not in blocks[1], "the volatile half must not be cached"
        assert blocks[1]["text"].startswith("# Candidate paper")

    def test_a_prefix_too_short_to_cache_is_left_alone(self) -> None:
        """Anthropic silently declines to cache a prefix under its minimum — no error and no
        cache_creation_input_tokens. Sending two blocks for one would cost the same and read
        as if caching were working."""
        prompt = "# Repository context\nshort\n\n# Candidate paper\nTitle: T"
        assert (
            self._run(prompt, cache_split_on="# Candidate paper")["messages"][0]["content"]
            == prompt
        )

    def test_an_absent_marker_falls_back_rather_than_guessing(self) -> None:
        prompt = "# Repository context\n" + "z" * 9000
        assert self._run(prompt, cache_split_on="# NOT PRESENT")["messages"][0]["content"] == prompt

    def test_redaction_cannot_shift_the_split(self) -> None:
        """`complete` redacts before dispatch. A caller passing a character offset would have
        it land mid-prompt once redaction shortened the text — putting the paper inside the
        cached half. Splitting on a marker is immune, so this asserts the boundary holds."""
        cfg = SimpleNamespace(
            provider="claude",
            claude_api_key="k",
            claude_model="claude-sonnet-5",
            timeout=5,
            redact=["SECRET"],
        )
        prompt = "# Repository context\nSECRET " + "w" * 9000 + "\n\n# Candidate paper\nTitle: T"
        payload = {"content": [{"type": "text", "text": "ok"}]}
        with patch("urllib.request.urlopen", return_value=_resp(payload)) as m:
            complete(prompt, cfg, cache_split_on="# Candidate paper")
            blocks = self._body_of(m)["messages"][0]["content"]
        assert "SECRET" not in blocks[0]["text"], "redaction still applies"
        assert blocks[1]["text"].startswith("# Candidate paper"), "boundary survived redaction"


class TestTheOpenAIProvider:
    """The gate ran on Claude and the fine-scale rescore on OpenAI, so a measured-configuration
    install needed a key from two vendors for two calls that ask nearly the same question. That
    is where most people installing this would stop."""

    CFG = SimpleNamespace(
        provider="openai", openai_api_key="k", openai_model="gpt-4o-mini", timeout=5
    )

    @staticmethod
    def _ok(text: str = "hello") -> MagicMock:
        return _resp({"choices": [{"message": {"content": text}, "finish_reason": "stop"}]})

    def test_a_completion_comes_back_as_text(self) -> None:
        with patch("urllib.request.urlopen", return_value=self._ok()):
            assert complete("prompt", self.CFG) == "hello"

    def test_the_key_falls_back_to_the_environment(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        monkeypatch.setenv("OPENAI_API_KEY", "from-env")
        cfg = SimpleNamespace(provider="openai", openai_model="gpt-4o-mini", timeout=5)
        with patch("urllib.request.urlopen", return_value=self._ok()) as m:
            complete("prompt", cfg)
        assert m.call_args[0][0].headers["Authorization"] == "Bearer from-env"

    def test_no_key_anywhere_is_a_config_error_not_a_request(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        cfg = SimpleNamespace(provider="openai", openai_model="gpt-4o-mini", timeout=5)
        with pytest.raises(LLMError, match="No OpenAI API key"):
            complete("prompt", cfg)

    def test_an_empty_message_raises_rather_than_returning_it(self) -> None:
        """Callers parse JSON out of this. An empty string is a real answer to nothing and
        would surface as a parse error somewhere far from the cause."""
        empty = _resp({"choices": [{"message": {"content": "  "}, "finish_reason": "length"}]})
        with (
            patch("urllib.request.urlopen", return_value=empty),
            pytest.raises(LLMError, match="empty message"),
        ):
            complete("prompt", self.CFG)

    def test_a_model_that_rejects_temperature_is_retried_without_it(self) -> None:
        """The reasoning models refuse the parameter; the Claude path already learned this."""
        import urllib.error

        from reporadar import llm_client

        llm_client._REJECTS_TEMPERATURE.discard("gpt-4o-mini")
        err = urllib.error.HTTPError(
            "u", 400, "bad", {}, io.BytesIO(b'{"error":{"message":"temperature unsupported"}}')
        )
        with patch("urllib.request.urlopen", side_effect=[err, self._ok("second")]) as m:
            assert complete("prompt", self.CFG) == "second"
        first = json.loads(m.call_args_list[0][0][0].data)
        second = json.loads(m.call_args_list[1][0][0].data)
        assert "temperature" in first and "temperature" not in second
        assert "gpt-4o-mini" in llm_client._REJECTS_TEMPERATURE, "remembered for the process"
        llm_client._REJECTS_TEMPERATURE.discard("gpt-4o-mini")

    def test_the_cache_marker_is_accepted_and_changes_nothing(self) -> None:
        """OpenAI matches the prefix server-side, so there is no annotation to add. Rejecting
        the argument would make one provider's caching a caller's problem."""
        with patch("urllib.request.urlopen", return_value=self._ok()) as m:
            complete("a" * 9000, self.CFG, cache_split_on="# Candidate paper")
        body = json.loads(m.call_args[0][0].data)
        assert isinstance(body["messages"][0]["content"], str), "no content blocks"

    def test_openai_is_a_known_provider_for_suggestions_and_triage(self) -> None:
        """Both validators refused it, so the gate could not run on OpenAI at all — the
        transport existed for the rescore and nothing could reach it for the gate."""
        from reporadar.config import RepoRadarConfig, validate_config

        cfg = RepoRadarConfig()
        cfg.repo_path = "."
        cfg.suggestions.provider = "openai"
        cfg.triage.enabled = True
        assert not [w for w in validate_config(cfg) if "provider" in w.lower()]


class TestTheOpenAIRequestShapeIsLearnedNotGuessed:
    """Measured 2026-09-07 against gpt-5.6-luna: `max_tokens` is refused outright ("use
    'max_completion_tokens' instead") and `temperature: 0` is refused separately ("only the
    default (1) value is supported"). The first version of this retried once and popped
    temperature, so it failed on the other and raised — it would never have reached the model
    it was written for."""

    CFG = SimpleNamespace(
        provider="openai", openai_api_key="k", openai_model="m-reasoning", timeout=5
    )

    @staticmethod
    def _400(msg: str):  # noqa: ANN205
        import urllib.error

        return urllib.error.HTTPError(
            "u", 400, "bad", {}, io.BytesIO(json.dumps({"error": {"message": msg}}).encode())
        )

    @staticmethod
    def _ok() -> MagicMock:
        return _resp({"choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}]})

    def _reset(self) -> None:
        from reporadar import llm_client

        llm_client._TOKEN_CAP.pop("m-reasoning", None)
        llm_client._REJECTS_TEMPERATURE.discard("m-reasoning")

    def test_both_rejections_are_adapted_to_in_one_call(self) -> None:
        self._reset()
        side = [
            self._400("Unsupported parameter: 'max_tokens' is not supported with this model."),
            self._400("Unsupported value: 'temperature' does not support 0 with this model."),
            self._ok(),
        ]
        with patch("urllib.request.urlopen", side_effect=side) as m:
            assert complete("p", self.CFG) == "hi"
        final = json.loads(m.call_args_list[-1][0][0].data)
        assert "max_completion_tokens" in final and "max_tokens" not in final
        assert "temperature" not in final
        self._reset()

    def test_the_rename_is_remembered_so_the_next_call_costs_no_400(self) -> None:
        """A 400 still consumes request-rate budget, which this project has repeatedly hit."""
        self._reset()
        first = [
            self._400("Unsupported parameter: 'max_tokens' is not supported with this model."),
            self._ok(),
        ]
        with patch("urllib.request.urlopen", side_effect=first):
            complete("p", self.CFG)
        with patch("urllib.request.urlopen", side_effect=[self._ok()]) as m:
            complete("p", self.CFG)
        assert "max_completion_tokens" in json.loads(m.call_args_list[0][0][0].data)
        self._reset()

    def test_an_unrelated_400_is_not_retried_into_oblivion(self) -> None:
        """A blanket retry would swallow quota and rate-limit errors as parameter problems."""
        self._reset()
        with (
            patch("urllib.request.urlopen", side_effect=[self._400("insufficient_quota")]),
            pytest.raises(LLMError, match="insufficient_quota"),
        ):
            complete("p", self.CFG)
        self._reset()

    def test_reasoning_effort_is_sent_only_when_configured(self) -> None:
        with patch("urllib.request.urlopen", return_value=self._ok()) as m:
            complete("p", self.CFG)
        assert "reasoning_effort" not in json.loads(m.call_args[0][0].data)
        cfg = SimpleNamespace(**{**vars(self.CFG), "openai_reasoning_effort": "none"})
        with patch("urllib.request.urlopen", return_value=self._ok()) as m:
            complete("p", cfg)
        assert json.loads(m.call_args[0][0].data)["reasoning_effort"] == "none"


# ── Keyless Azure OpenAI (PLANS item 17) ────────────────────────────────────────────────────
#
# Error texts below are VERBATIM from the live Azure probe of 2026-09-13, except where marked:
# the adaptation rules match substrings, so a paraphrased message would test a rule against text
# no service sends.
_AZ_MAX_TOKENS = (
    "Unsupported parameter: 'max_tokens' is not supported with this model. "
    "Use 'max_completion_tokens' instead."
)
_AZ_TEMPERATURE = (
    "Unsupported value: 'temperature' does not support 0 with this model. "
    "Only the default (1) value is supported."
)
_AZ_TOP_LOGPROBS = "Invalid value for 'top_logprobs': must be less than or equal to 5."


def _http(
    code: int, body: dict | None = None, headers: dict | None = None
) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "u", code, "err", headers or {}, io.BytesIO(json.dumps(body or {}).encode())
    )


def _400(message: str, code: str | None = None) -> urllib.error.HTTPError:
    return _http(400, {"error": {"message": message, "code": code}})


def _forget(*keys: str) -> None:
    from reporadar import llm_client

    for key in keys:
        llm_client._TOKEN_CAP.pop(key, None)
        llm_client._REJECTS_TEMPERATURE.discard(key)
        llm_client._REJECTS_EFFORT.discard(key)
        llm_client._TOP_LOGPROBS_CAP.pop(key, None)


_AZ_KEY = "azure:myres.openai.azure.com:gpt-5.6-luna"


@pytest.fixture
def entra(monkeypatch: pytest.MonkeyPatch):  # noqa: ANN201
    """A token from `az`, without running it. Records which tenant asked."""
    from reporadar import azure_auth

    asked: list[str] = []
    monkeypatch.setattr(azure_auth, "get_token", lambda tenant="": asked.append(tenant) or "entra")
    _forget(_AZ_KEY, "gpt-5.6-luna")
    yield asked
    _forget(_AZ_KEY, "gpt-5.6-luna")


def _azure_cfg(**overrides: object) -> SimpleNamespace:
    base = {
        "provider": "azure_openai",
        "azure_endpoint": "https://myres.openai.azure.com",
        "azure_tenant": "",
        "azure_deployment": "gpt-5.6-luna",
        "openai_reasoning_effort": "none",
        "timeout": 5,
    }
    return SimpleNamespace(**{**base, **overrides})


def _ok(text: str = "hi", finish: str = "stop") -> MagicMock:
    return _resp({"choices": [{"message": {"content": text}, "finish_reason": finish}]})


class TestTheAzureOpenAIProvider:
    def test_the_request_is_the_openai_body_at_the_v1_url_with_an_entra_token(
        self, entra: list[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-must-not-be-sent-to-azure")
        with patch("urllib.request.urlopen", return_value=_ok()) as m:
            assert complete("prompt", _azure_cfg()) == "hi"
        req = m.call_args[0][0]
        assert req.full_url == "https://myres.openai.azure.com/openai/v1/chat/completions"
        assert req.headers["Authorization"] == "Bearer entra", "the token, never an OpenAI key"
        body = json.loads(req.data)
        assert body["model"] == "gpt-5.6-luna", "the DEPLOYMENT name goes in `model`"
        assert body["reasoning_effort"] == "none"

    def test_the_tenant_travels_to_the_token_request(self, entra: list[str]) -> None:
        with patch("urllib.request.urlopen", return_value=_ok()):
            complete("p", _azure_cfg(azure_tenant="contoso.onmicrosoft.com"))
        assert entra == ["contoso.onmicrosoft.com"]

    def test_a_refused_endpoint_costs_no_token_and_sends_nothing(self, entra: list[str]) -> None:
        """The committed-config attack: a repository that points your token somewhere else."""
        with (
            patch("urllib.request.urlopen") as m,
            pytest.raises(LLMError, match="Refusing Azure OpenAI endpoint"),
        ):
            complete("p", _azure_cfg(azure_endpoint="https://attacker.example.com"))
        m.assert_not_called()
        assert entra == []

    def test_a_missing_deployment_is_a_config_error_not_a_request(self, entra: list[str]) -> None:
        with patch("urllib.request.urlopen") as m, pytest.raises(LLMError, match="deployment"):
            complete("p", _azure_cfg(azure_deployment=""))
        m.assert_not_called()

    def test_a_token_failure_is_reported_with_its_fix_and_not_retried(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from reporadar import azure_auth

        def refuse(tenant: str = "") -> str:
            raise azure_auth.AzureAuthError("Not signed in to Azure. Run `az login` in a terminal.")

        monkeypatch.setattr(azure_auth, "get_token", refuse)
        with patch("urllib.request.urlopen") as m, pytest.raises(LLMError, match="az login"):
            complete("p", _azure_cfg())
        m.assert_not_called()

    @pytest.mark.parametrize(
        ("code", "expected"),
        [
            (403, "Cognitive Services OpenAI User"),
            (401, "azure_openai.tenant"),
            (404, "not the model's name"),
        ],
    )
    def test_the_refusals_a_keyless_setup_hits_name_their_fix(
        self, entra: list[str], code: int, expected: str
    ) -> None:
        """403 is the one that actually happened: an Owner with no data-plane role."""
        with (
            patch("urllib.request.urlopen", side_effect=[_http(code)]) as m,
            pytest.raises(LLMError, match=expected),
        ):
            complete("p", _azure_cfg())
        assert m.call_count == 1, "a refusal is not transient; retrying it only burns quota"

    def test_a_blocked_prompt_is_reported_as_a_content_filter_not_retried(
        self, entra: list[str]
    ) -> None:
        blocked = _http(
            400,
            {
                "error": {
                    "code": "content_filter",
                    "message": "The response was filtered",
                    "innererror": {"code": "ResponsibleAIPolicyViolation"},
                }
            },
        )
        with (
            patch("urllib.request.urlopen", side_effect=[blocked]) as m,
            pytest.raises(LLMError, match="content filter"),
        ):
            complete("p", _azure_cfg())
        assert m.call_count == 1

    def test_a_withheld_completion_is_reported_as_a_content_filter(self, entra: list[str]) -> None:
        withheld = _resp(
            {"choices": [{"message": {"content": None}, "finish_reason": "content_filter"}]}
        )
        with (
            patch("urllib.request.urlopen", return_value=withheld),
            pytest.raises(LLMError, match="content filter"),
        ):
            complete("p", _azure_cfg())

    def test_azures_own_error_text_drives_the_existing_adaptation(self, entra: list[str]) -> None:
        with patch("urllib.request.urlopen", side_effect=[_400(_AZ_MAX_TOKENS), _ok()]) as m:
            assert complete("p", _azure_cfg()) == "hi"
        final = json.loads(m.call_args_list[-1][0][0].data)
        assert "max_completion_tokens" in final and "max_tokens" not in final

    def test_what_azure_teaches_about_a_name_does_not_leak_into_openai(
        self, entra: list[str]
    ) -> None:
        """gpt-5.6-luna refuses `temperature: 0` on OpenAI and accepted it on Azure: the same
        name, two behaviours, so one endpoint's lesson must not rewrite the other's request."""
        from reporadar import llm_client

        with patch("urllib.request.urlopen", side_effect=[_400(_AZ_TEMPERATURE), _ok()]):
            complete("p", _azure_cfg())
        assert _AZ_KEY in llm_client._REJECTS_TEMPERATURE
        assert "gpt-5.6-luna" not in llm_client._REJECTS_TEMPERATURE


class TestTheFineScaleRequestAdaptsToo:
    """It used to send `max_tokens`, `temperature: 0` and `top_logprobs: 20` with no retry, so it
    could not run on a reasoning model at all — on OpenAI's own API as much as on Azure. Live on
    Azure, gpt-5.6-luna needed both max_completion_tokens and top_logprobs <= 5."""

    @staticmethod
    def _logprobs() -> MagicMock:
        return _resp(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "logprobs": {
                            "content": [
                                {"token": "6", "top_logprobs": [{"token": "6", "logprob": -0.03}]}
                            ]
                        },
                    }
                ]
            }
        )

    def test_every_rejection_luna_made_is_adapted_to_in_one_call(self, entra: list[str]) -> None:
        from reporadar.llm_client import top_logprobs

        side = [
            _400(_AZ_MAX_TOKENS),
            _400(_AZ_TEMPERATURE),
            _400(_AZ_TOP_LOGPROBS),
            self._logprobs(),
        ]
        cfg = _azure_cfg(reasoning_effort="none")
        with patch("urllib.request.urlopen", side_effect=side) as m:
            got = top_logprobs("p", cfg)
        assert got[0][0] == "6"
        final = json.loads(m.call_args_list[-1][0][0].data)
        assert final["top_logprobs"] == 5
        assert "max_completion_tokens" in final and "max_tokens" not in final
        assert "temperature" not in final
        assert final["reasoning_effort"] == "none"

    def test_the_cap_is_remembered_so_the_next_paper_costs_no_400(self, entra: list[str]) -> None:
        from reporadar.llm_client import top_logprobs

        first = [_400(_AZ_TOP_LOGPROBS), self._logprobs()]
        with patch("urllib.request.urlopen", side_effect=first):
            top_logprobs("p", _azure_cfg())
        with patch("urllib.request.urlopen", side_effect=[self._logprobs()]) as m:
            top_logprobs("p", _azure_cfg())
        assert json.loads(m.call_args_list[0][0][0].data)["top_logprobs"] == 5

    def test_a_model_that_refuses_reasoning_effort_has_it_dropped(self, entra: list[str]) -> None:
        """Seen live: gpt-4.1-mini refused the "none" the Azure template sets for this stage.
        The message text here is paraphrased; the rule matches only the parameter's name."""
        from reporadar.llm_client import top_logprobs

        key = "azure:myres.openai.azure.com:gpt-4.1-mini"
        refused = _400("Unrecognized request argument supplied: reasoning_effort")
        cfg = _azure_cfg(azure_deployment="gpt-4.1-mini", reasoning_effort="none")
        _forget(key)
        with patch("urllib.request.urlopen", side_effect=[refused, self._logprobs()]) as m:
            top_logprobs("p", cfg)
        assert "reasoning_effort" not in json.loads(m.call_args_list[-1][0][0].data)
        _forget(key)

    def test_without_a_provider_the_rescore_is_still_openai(self) -> None:
        """Every config written before Azure existed, and the eval harness, pass no provider."""
        from reporadar.llm_client import top_logprobs

        with patch("urllib.request.urlopen", return_value=self._logprobs()) as m:
            top_logprobs("p", SimpleNamespace(openai_api_key="k", timeout=5))
        assert m.call_args[0][0].full_url == "https://api.openai.com/v1/chat/completions"


class TestA429IsGivenTheWaitItAsksFor:
    """Azure enforces requests-per-minute over 1-10 s windows and says how long in
    `retry-after-ms`. The fixed half-second backoff spent every retry inside the same window."""

    CFG = SimpleNamespace(provider="openai", openai_api_key="k", openai_model="m", timeout=5)

    def _sleeps(self, headers: dict | None) -> list[float]:
        with (
            patch("urllib.request.urlopen", side_effect=[_http(429, headers=headers), _ok()]),
            patch("reporadar.llm_client.time.sleep") as sleep,
        ):
            complete("p", self.CFG)
        return [c.args[0] for c in sleep.call_args_list]

    def test_retry_after_ms_is_obeyed(self) -> None:
        assert self._sleeps({"retry-after-ms": "2500"}) == [2.5]

    def test_retry_after_seconds_is_obeyed(self) -> None:
        assert self._sleeps({"retry-after": "3"}) == [3.0]

    def test_an_absurd_wait_is_capped(self) -> None:
        from reporadar.llm_client import RETRY_AFTER_CAP_SECONDS

        assert self._sleeps({"retry-after": "86400"}) == [RETRY_AFTER_CAP_SECONDS]

    def test_without_a_header_the_backoff_is_unchanged(self) -> None:
        assert self._sleeps(None) == [0.5]
