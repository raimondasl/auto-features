"""Tests for anonymous.azure_auth — keyless Azure OpenAI tokens, and where they may go.

The security properties are the load-bearing ones. `.anonymous.yml` is committed to repositories,
so the endpoint and tenant are attacker-controllable the moment someone runs Anonymous in a
repository they did not write: the endpoint must never send a token outside Azure, and the tenant
must never smuggle anything into the `az` command line, which on Windows runs through cmd.exe.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from typing import Any

import pytest

from anonymous import azure_auth
from anonymous.azure_auth import AzureAuthError, chat_completions_url, validate_tenant

_REAL_FETCH = azure_auth._fetch  # captured before conftest's guard replaces it per test


class TestOnlyAzureHostsCanReceiveATokenRequest:
    @pytest.mark.parametrize(
        "endpoint",
        [
            "https://myres.openai.azure.com",
            "https://myres.openai.azure.com/",
            "https://myres.openai.azure.com/openai/v1",
            "https://myres.openai.azure.com/openai/v1/",
            "myres.openai.azure.com",
            "  https://MyRes.OpenAI.Azure.com  ",
        ],
    )
    def test_what_users_paste_is_normalised_to_the_v1_url(self, endpoint: str) -> None:
        assert chat_completions_url(endpoint) == (
            "https://myres.openai.azure.com/openai/v1/chat/completions"
        )

    @pytest.mark.parametrize(
        "host", ["myres.services.ai.azure.com", "myres.cognitiveservices.azure.com"]
    )
    def test_the_other_two_host_forms_are_accepted(self, host: str) -> None:
        """Both answered the live probe; `cognitiveservices` is what the CLI prints."""
        assert (
            chat_completions_url(f"https://{host}/") == f"https://{host}/openai/v1/chat/completions"
        )

    @pytest.mark.parametrize(
        "endpoint",
        [
            "http://myres.openai.azure.com",  # not https
            "https://attacker.example.com",  # not Azure
            "https://myres.openai.azure.com.attacker.net",  # suffix lookalike
            "https://attacker.net/myres.openai.azure.com",  # host in the path
            "https://myres.openai.azure.com@attacker.net",  # userinfo trick
            "https://user:pw@myres.openai.azure.com",  # credentials in the URL
            "https://a.b.openai.azure.com",  # not a single resource label
            "https://openai.azure.com",  # the bare suffix
            "https://myres.openai.azure.com:8443",  # a port
            "https://myres.openai.azure.com/api/projects/p",  # a path this does not support
            "https://myres.openai.azure.com/?next=https://attacker.net",  # query
            "https://eastus.api.cognitive.microsoft.com",  # regional: rejects Entra anyway
            "https://myres.openai.azure.com#frag",
            "https://[myres.openai.azure.com",  # urlparse raises rather than reports this
        ],
    )
    def test_anything_else_is_refused_before_it_can_become_a_request(self, endpoint: str) -> None:
        with pytest.raises(AzureAuthError, match="Refusing Azure OpenAI endpoint"):
            chat_completions_url(endpoint)

    def test_an_empty_endpoint_says_what_to_set(self) -> None:
        with pytest.raises(AzureAuthError, match="azure_openai.endpoint"):
            chat_completions_url("")

    def test_the_url_is_rebuilt_from_the_host_alone(self) -> None:
        """Nothing but the validated host reaches the request, whatever else was pasted."""
        url = chat_completions_url("https://myres.openai.azure.com/openai/v1/")
        assert url.count("/openai/v1") == 1


class TestTheTenantCannotReachTheCommandLineAsAnythingButATenant:
    @pytest.mark.parametrize(
        "tenant", ["", "12345678-1234-1234-1234-123456789abc", "contoso.onmicrosoft.com"]
    )
    def test_ids_and_domains_pass(self, tenant: str) -> None:
        assert validate_tenant(f" {tenant} ") == tenant

    @pytest.mark.parametrize(
        "tenant",
        [
            "x & calc",
            "x|whoami",
            'x" --query accessToken',
            "--output tsv",
            "x%PATH%",
            "x^y",
            "tenant with spaces",
            "-leading-dash",
            "x\ny",
        ],
    )
    def test_metacharacters_and_flags_are_refused(self, tenant: str) -> None:
        with pytest.raises(AzureAuthError, match="not a tenant id or domain"):
            validate_tenant(tenant)


@pytest.fixture
def real_fetch(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """The real `_fetch`; each test replaces `_run`, so no process is started."""
    monkeypatch.setattr(azure_auth, "_fetch", _REAL_FETCH)
    monkeypatch.setattr(azure_auth, "az_executable", lambda: "/usr/bin/az")
    calls: list[dict[str, Any]] = []
    return calls


def _completed(stdout: str = "", stderr: str = "", code: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=code, stdout=stdout, stderr=stderr)


def _token_json(token: str = "tok", expires_in: float = 3600) -> str:
    return json.dumps(
        {"accessToken": token, "expires_on": int(time.time() + expires_in), "tokenType": "Bearer"}
    )


class TestGettingAToken:
    def test_az_is_asked_for_the_cognitive_services_audience(
        self, real_fetch: list, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: dict[str, Any] = {}

        def run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
            seen["cmd"], seen["kwargs"] = cmd, kwargs
            return _completed(_token_json())

        monkeypatch.setattr(azure_auth, "_run", run)
        assert azure_auth.get_token() == "tok"
        assert seen["cmd"][1:] == [
            "account",
            "get-access-token",
            "--resource",
            azure_auth.AUDIENCE,
            "--output",
            "json",
        ]

    def test_az_is_given_a_generous_timeout(
        self, real_fetch: list, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: dict[str, Any] = {}
        monkeypatch.setattr(
            azure_auth,
            "_run",
            lambda cmd, **kw: seen.update(kw) or _completed(_token_json()),
        )
        azure_auth.get_token()
        assert seen["timeout"] >= 30, "az can take 10-15 s; a short timeout drops the gate"

    def test_a_tenant_is_passed_as_its_own_argument(
        self, real_fetch: list, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: dict[str, Any] = {}
        monkeypatch.setattr(
            azure_auth,
            "_run",
            lambda cmd, **kw: seen.update(cmd=cmd) or _completed(_token_json()),
        )
        azure_auth.get_token("contoso.onmicrosoft.com")
        assert seen["cmd"][-2:] == ["--tenant", "contoso.onmicrosoft.com"]

    def test_an_invalid_tenant_never_reaches_az(
        self, real_fetch: list, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ran: list[Any] = []
        monkeypatch.setattr(azure_auth, "_run", lambda *a, **k: ran.append(a))
        with pytest.raises(AzureAuthError):
            azure_auth.get_token("x & calc")
        assert not ran

    def test_one_fetch_serves_calls_until_near_expiry(
        self, real_fetch: list, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The gate makes up to 50 calls, and `az` can take 15 s each time."""
        runs: list[int] = []
        monkeypatch.setattr(
            azure_auth,
            "_run",
            lambda *a, **k: runs.append(1) or _completed(_token_json(f"t{len(runs)}")),
        )
        assert [azure_auth.get_token() for _ in range(5)] == ["t1"] * 5
        assert len(runs) == 1

    def test_a_token_close_to_expiry_is_refreshed_not_handed_out(
        self, real_fetch: list, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        expiring = [azure_auth.REFRESH_MARGIN_SECONDS - 10, 3600]
        monkeypatch.setattr(
            azure_auth,
            "_run",
            lambda *a, **k: _completed(_token_json(f"t{len(expiring)}", expiring.pop(0))),
        )
        assert azure_auth.get_token() == "t2"
        assert azure_auth.get_token() == "t1", "the near-expired token was reused"

    def test_tokens_are_cached_per_tenant(
        self, real_fetch: list, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            azure_auth,
            "_run",
            lambda cmd, **k: _completed(_token_json("tenant-b" if "--tenant" in cmd else "home")),
        )
        assert azure_auth.get_token() == "home"
        assert azure_auth.get_token("b.onmicrosoft.com") == "tenant-b"

    def test_a_response_without_expires_on_is_cached_only_briefly(
        self, real_fetch: list, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Older CLIs return only `expiresOn`, a naive LOCAL time; trusting it is a tz bug."""
        body = json.dumps({"accessToken": "tok", "expiresOn": "2099-01-01 00:00:00.000000"})
        monkeypatch.setattr(azure_auth, "_run", lambda *a, **k: _completed(body))
        azure_auth.get_token()
        _, expires_at = azure_auth._cache[""]
        assert expires_at - time.time() < 3600


class TestFailuresSayWhatToDo:
    def test_no_az_on_path_mentions_restarting_the_editor(
        self, real_fetch: list, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The failure that actually happened: `az` installed, editor still running with the old
        PATH, and the server it launched unable to see it."""
        monkeypatch.setattr(azure_auth, "az_executable", lambda: None)
        with pytest.raises(AzureAuthError, match="restart the editor"):
            azure_auth.get_token()

    @pytest.mark.parametrize(
        ("stderr", "tenant", "expected"),
        [
            ("ERROR: Please run 'az login' to setup account.", "", "Run `az login`"),
            (
                "AADSTS50076: Due to a configuration change ... you must use multi-factor",
                "t.onmicrosoft.com",
                "az login --tenant t.onmicrosoft.com",
            ),
            ("AADSTS700082: The refresh token has expired", "", "has expired"),
            ("something unexpected", "", "something unexpected"),
        ],
    )
    def test_az_errors_are_translated(
        self,
        real_fetch: list,
        monkeypatch: pytest.MonkeyPatch,
        stderr: str,
        tenant: str,
        expected: str,
    ) -> None:
        monkeypatch.setattr(azure_auth, "_run", lambda *a, **k: _completed(stderr=stderr, code=1))
        with pytest.raises(AzureAuthError, match=expected.replace("`", ".")):
            azure_auth.get_token(tenant)

    def test_a_hung_az_is_a_clear_error(
        self, real_fetch: list, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def run(*a: Any, **k: Any) -> Any:
            raise subprocess.TimeoutExpired(cmd="az", timeout=60)

        monkeypatch.setattr(azure_auth, "_run", run)
        with pytest.raises(AzureAuthError, match="did not answer"):
            azure_auth.get_token()

    def test_unreadable_output_is_an_error_not_an_empty_token(
        self, real_fetch: list, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(azure_auth, "_run", lambda *a, **k: _completed("not json"))
        with pytest.raises(AzureAuthError, match="could not read"):
            azure_auth.get_token()


class TestFailuresAndRefusalsAreNotRepeated:
    def test_a_failed_fetch_answers_the_callers_right_behind_it(
        self, real_fetch: list, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A lapsed `az login` used to start one `az` per gate paper -- fifty CLI start-ups, each
        failing with the same message."""
        runs: list[int] = []
        monkeypatch.setattr(
            azure_auth,
            "_run",
            lambda *a, **k: runs.append(1) or _completed(stderr="Please run 'az login'", code=1),
        )
        for _ in range(5):
            with pytest.raises(AzureAuthError, match="az login"):
                azure_auth.get_token()
        assert len(runs) == 1

    def test_a_remembered_failure_expires_so_signing_in_takes_effect(
        self, real_fetch: list, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        answers = [_completed(stderr="Please run 'az login'", code=1), _completed(_token_json())]
        monkeypatch.setattr(azure_auth, "_run", lambda *a, **k: answers.pop(0))
        monkeypatch.setattr(azure_auth, "FAILURE_TTL_SECONDS", 0.0)
        with pytest.raises(AzureAuthError):
            azure_auth.get_token()
        assert azure_auth.get_token() == "tok"
        assert azure_auth.FAILURE_TTL_SECONDS < 60, "longer than a user takes to run az login"

    def test_forget_makes_the_next_call_ask_az_again(
        self, real_fetch: list, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """For a token Azure refused: after `az login` as the right account, a long-lived server
        must not keep sending the old one for an hour."""
        tokens = ["old", "new"]
        monkeypatch.setattr(
            azure_auth, "_run", lambda *a, **k: _completed(_token_json(tokens.pop(0)))
        )
        assert azure_auth.get_token("t.onmicrosoft.com") == "old"
        azure_auth.forget("t.onmicrosoft.com")
        assert azure_auth.get_token("t.onmicrosoft.com") == "new"


class TestTheCliIsNeverTakenFromTheRepository:
    def test_a_planted_az_in_the_working_directory_is_not_found(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The review's reproduction: `shutil.which` on Windows searched the working directory
        first, returned `.\\az.cmd.COM`, and CreateProcess ran the repository's binary."""
        for name in ("az", "az.exe", "az.cmd", "az.cmd.com", "az.cmd.exe", "az.bat"):
            planted = tmp_path / name
            planted.write_text("", encoding="utf-8")
            planted.chmod(0o755)
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("NoDefaultCurrentDirectoryInExePath", raising=False)
        monkeypatch.setenv("PATH", os.pathsep.join(["", ".", "bin"]))
        assert azure_auth.az_executable() is None

    def test_what_it_finds_is_an_absolute_path(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bindir = tmp_path / "cli"
        bindir.mkdir()
        real = bindir / ("az.cmd" if sys.platform == "win32" else "az")
        real.write_text("", encoding="utf-8")
        real.chmod(0o755)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("PATH", os.pathsep.join([".", str(bindir)]))
        found = azure_auth.az_executable()
        assert found is not None and os.path.isabs(found)
        assert os.path.samefile(found, real)


class TestRunningAz:
    """`_run` against real, harmless child processes -- never the Azure CLI."""

    def test_the_child_never_shares_the_callers_stdin(self) -> None:
        """Under the MCP server stdin is the editor's JSON-RPC pipe, and a child that inherits
        it deadlocks on Windows — a hang this module must not reintroduce."""
        done = azure_auth._run(
            [sys.executable, "-c", "import sys; print(repr(sys.stdin.read()))"], timeout=30
        )
        assert done.returncode == 0
        assert done.stdout.strip() == "''"

    def test_a_timeout_kills_the_grandchild_that_holds_the_pipes(self) -> None:
        """`az.cmd` runs python through cmd.exe. `subprocess.run` killed only the wrapper and then
        waited, with no timeout, on pipes the surviving python still held."""
        grandchild = "import time; time.sleep(60)"
        wrapper = f"import subprocess, sys; subprocess.run([sys.executable, '-c', {grandchild!r}])"
        started = time.monotonic()
        with pytest.raises(subprocess.TimeoutExpired):
            azure_auth._run([sys.executable, "-c", wrapper], timeout=2)
        assert time.monotonic() - started < 30


def test_the_suite_cannot_reach_the_real_azure_cli() -> None:
    """If this fails, the conftest guard is gone and tests may be using a real identity."""
    assert azure_auth._fetch is not _REAL_FETCH
