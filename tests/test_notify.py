"""Tests for reporadar.notify."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

from reporadar.config import EmailHookConfig, HooksConfig
from reporadar.notify import (
    DigestSummary,
    dispatch_notification,
    run_shell_hook,
    send_discord_webhook,
    send_email,
    send_slack_webhook,
    summary_to_env,
)


def _make_summary(**overrides) -> DigestSummary:
    base = {
        "digest_path": "/tmp/digest.md",
        "run_id": 42,
        "papers_new": 10,
        "papers_seen": 5,
        "top_picks_count": 3,
        "total_scored": 15,
        "fmt": "md",
    }
    base.update(overrides)
    return DigestSummary(**base)


class TestSummaryToEnv:
    def test_all_keys_present(self) -> None:
        env = summary_to_env(_make_summary())
        assert env["RR_DIGEST_PATH"] == "/tmp/digest.md"
        assert env["RR_RUN_ID"] == "42"
        assert env["RR_PAPERS_NEW"] == "10"
        assert env["RR_PAPERS_SEEN"] == "5"
        assert env["RR_TOP_PICKS_COUNT"] == "3"
        assert env["RR_TOTAL_SCORED"] == "15"
        assert env["RR_FORMAT"] == "md"

    def test_all_values_are_strings(self) -> None:
        env = summary_to_env(_make_summary())
        for v in env.values():
            assert isinstance(v, str)

    def test_custom_values(self) -> None:
        env = summary_to_env(_make_summary(run_id=99, papers_new=0))
        assert env["RR_RUN_ID"] == "99"
        assert env["RR_PAPERS_NEW"] == "0"


class TestRunShellHook:
    @patch("reporadar.notify.subprocess.run")
    def test_success(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(args="echo ok", returncode=0)
        result = run_shell_hook("echo ok", _make_summary())
        assert result is True
        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args
        assert call_kwargs.kwargs["shell"] is True
        assert call_kwargs.kwargs["timeout"] == 60

    @patch("reporadar.notify.subprocess.run")
    def test_nonzero_exit(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args="false", returncode=1, stderr="error"
        )
        result = run_shell_hook("false", _make_summary())
        assert result is False

    @patch("reporadar.notify.subprocess.run")
    def test_timeout(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="sleep 100", timeout=60)
        result = run_shell_hook("sleep 100", _make_summary())
        assert result is False

    @patch("reporadar.notify.subprocess.run")
    def test_os_error(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = OSError("command not found")
        result = run_shell_hook("bad_cmd", _make_summary())
        assert result is False

    @patch("reporadar.notify.subprocess.run")
    def test_env_vars_passed(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(args="echo", returncode=0)
        run_shell_hook("echo $RR_RUN_ID", _make_summary(run_id=7))
        env = mock_run.call_args.kwargs["env"]
        assert env["RR_RUN_ID"] == "7"


class TestSendSlackWebhook:
    @patch("reporadar.notify.urllib.request.urlopen")
    def test_success(self, mock_urlopen: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = send_slack_webhook("https://hooks.slack.com/test", _make_summary())
        assert result is True

    @patch("reporadar.notify.urllib.request.urlopen")
    def test_network_error(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = Exception("connection refused")
        result = send_slack_webhook("https://hooks.slack.com/test", _make_summary())
        assert result is False

    @patch("reporadar.notify.urllib.request.urlopen")
    def test_posts_json_with_text(self, mock_urlopen: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        send_slack_webhook("https://hooks.slack.com/test", _make_summary())
        req = mock_urlopen.call_args[0][0]
        assert req.get_header("Content-type") == "application/json"

    @patch("reporadar.notify.urllib.request.urlopen")
    def test_server_error(self, mock_urlopen: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.status = 500
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = send_slack_webhook("https://hooks.slack.com/test", _make_summary())
        assert result is False


class TestSendDiscordWebhook:
    @patch("reporadar.notify.urllib.request.urlopen")
    def test_success(self, mock_urlopen: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.status = 204
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = send_discord_webhook("https://discord.com/api/webhooks/test", _make_summary())
        assert result is True

    @patch("reporadar.notify.urllib.request.urlopen")
    def test_network_error(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = Exception("connection refused")
        result = send_discord_webhook("https://discord.com/api/webhooks/test", _make_summary())
        assert result is False

    @patch("reporadar.notify.urllib.request.urlopen")
    def test_posts_json_with_content(self, mock_urlopen: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.status = 204
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        send_discord_webhook("https://discord.com/api/webhooks/test", _make_summary())
        req = mock_urlopen.call_args[0][0]
        assert req.get_header("Content-type") == "application/json"


class TestSendEmail:
    @patch("reporadar.notify.smtplib.SMTP")
    def test_success(self, mock_smtp_class: MagicMock) -> None:
        mock_server = MagicMock()
        mock_smtp_class.return_value.__enter__ = MagicMock(return_value=mock_server)
        mock_smtp_class.return_value.__exit__ = MagicMock(return_value=False)

        result = send_email(
            smtp_host="smtp.example.com",
            port=587,
            from_addr="from@test.com",
            to="to@test.com",
            summary=_make_summary(),
            username="user",
            password="pass",
        )
        assert result is True
        mock_server.starttls.assert_called_once()
        mock_server.login.assert_called_once_with("user", "pass")
        mock_server.sendmail.assert_called_once()

    @patch("reporadar.notify.smtplib.SMTP")
    def test_no_tls(self, mock_smtp_class: MagicMock) -> None:
        mock_server = MagicMock()
        mock_smtp_class.return_value.__enter__ = MagicMock(return_value=mock_server)
        mock_smtp_class.return_value.__exit__ = MagicMock(return_value=False)

        result = send_email(
            smtp_host="smtp.example.com",
            port=25,
            from_addr="from@test.com",
            to="to@test.com",
            summary=_make_summary(),
            use_tls=False,
        )
        assert result is True
        mock_server.starttls.assert_not_called()

    @patch("reporadar.notify.smtplib.SMTP")
    def test_no_auth(self, mock_smtp_class: MagicMock) -> None:
        mock_server = MagicMock()
        mock_smtp_class.return_value.__enter__ = MagicMock(return_value=mock_server)
        mock_smtp_class.return_value.__exit__ = MagicMock(return_value=False)

        send_email(
            smtp_host="smtp.example.com",
            port=587,
            from_addr="from@test.com",
            to="to@test.com",
            summary=_make_summary(),
            username="",
            password="",
        )
        mock_server.login.assert_not_called()

    @patch("reporadar.notify.smtplib.SMTP")
    def test_connection_error(self, mock_smtp_class: MagicMock) -> None:
        mock_smtp_class.side_effect = ConnectionRefusedError("refused")
        result = send_email(
            smtp_host="bad.host",
            port=587,
            from_addr="from@test.com",
            to="to@test.com",
            summary=_make_summary(),
        )
        assert result is False


class TestDispatchNotification:
    def test_shell_dispatch(self) -> None:
        hooks = HooksConfig(on_digest="echo done")
        with patch("reporadar.notify.run_shell_hook", return_value=True) as mock:
            result = dispatch_notification("shell", hooks, _make_summary())
        assert result is True
        mock.assert_called_once()

    def test_shell_no_command(self) -> None:
        hooks = HooksConfig(on_digest="")
        result = dispatch_notification("shell", hooks, _make_summary())
        assert result is False

    def test_slack_dispatch(self) -> None:
        hooks = HooksConfig(slack_webhook_url="https://hooks.slack.com/test")
        with patch("reporadar.notify.send_slack_webhook", return_value=True) as mock:
            result = dispatch_notification("slack", hooks, _make_summary())
        assert result is True
        mock.assert_called_once()

    def test_slack_no_url(self) -> None:
        hooks = HooksConfig(slack_webhook_url="")
        result = dispatch_notification("slack", hooks, _make_summary())
        assert result is False

    def test_discord_dispatch(self) -> None:
        hooks = HooksConfig(discord_webhook_url="https://discord.com/api/webhooks/test")
        with patch("reporadar.notify.send_discord_webhook", return_value=True) as mock:
            result = dispatch_notification("discord", hooks, _make_summary())
        assert result is True
        mock.assert_called_once()

    def test_discord_no_url(self) -> None:
        hooks = HooksConfig(discord_webhook_url="")
        result = dispatch_notification("discord", hooks, _make_summary())
        assert result is False

    def test_email_dispatch(self) -> None:
        email = EmailHookConfig(smtp_host="smtp.test.com", to="user@test.com")
        hooks = HooksConfig(email=email)
        with patch("reporadar.notify.send_email", return_value=True) as mock:
            result = dispatch_notification("email", hooks, _make_summary())
        assert result is True
        mock.assert_called_once()

    def test_email_not_configured(self) -> None:
        hooks = HooksConfig()
        result = dispatch_notification("email", hooks, _make_summary())
        assert result is False

    def test_unknown_channel(self) -> None:
        hooks = HooksConfig()
        result = dispatch_notification("carrier_pigeon", hooks, _make_summary())
        assert result is False
