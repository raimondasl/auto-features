"""Tests for reporadar.watcher."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from reporadar.watcher import (
    parse_interval,
    run_update_cycle,
    send_desktop_notification,
    watch_loop,
)


class TestParseInterval:
    def test_hours(self) -> None:
        assert parse_interval("6h") == 21600
        assert parse_interval("1h") == 3600

    def test_minutes(self) -> None:
        assert parse_interval("30m") == 1800
        assert parse_interval("5m") == 300

    def test_days(self) -> None:
        assert parse_interval("1d") == 86400
        assert parse_interval("7d") == 604800

    def test_seconds(self) -> None:
        assert parse_interval("3600s") == 3600
        assert parse_interval("60s") == 60

    def test_invalid_empty(self) -> None:
        with pytest.raises(ValueError, match="Empty"):
            parse_interval("")

    def test_invalid_suffix(self) -> None:
        with pytest.raises(ValueError, match="suffix"):
            parse_interval("5x")

    def test_invalid_number(self) -> None:
        with pytest.raises(ValueError, match="Invalid"):
            parse_interval("abch")

    def test_negative_value(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            parse_interval("-1h")

    def test_whitespace(self) -> None:
        assert parse_interval("  6h  ") == 21600


class TestSendDesktopNotification:
    @patch("reporadar.watcher.platform.system", return_value="Linux")
    @patch("reporadar.watcher.subprocess.run")
    def test_linux(self, mock_run: MagicMock, mock_sys: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(args="notify-send", returncode=0)
        assert send_desktop_notification("Title", "Msg") is True
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert args[0] == "notify-send"

    @patch("reporadar.watcher.platform.system", return_value="Darwin")
    @patch("reporadar.watcher.subprocess.run")
    def test_macos(self, mock_run: MagicMock, mock_sys: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(args="osascript", returncode=0)
        assert send_desktop_notification("Title", "Msg") is True
        args = mock_run.call_args[0][0]
        assert args[0] == "osascript"

    @patch("reporadar.watcher.platform.system", return_value="Windows")
    @patch("reporadar.watcher.subprocess.run")
    def test_windows(self, mock_run: MagicMock, mock_sys: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(args="powershell", returncode=0)
        assert send_desktop_notification("Title", "Msg") is True
        args = mock_run.call_args[0][0]
        assert args[0] == "powershell"

    @patch("reporadar.watcher.platform.system", return_value="Linux")
    @patch("reporadar.watcher.subprocess.run")
    def test_command_not_found(self, mock_run: MagicMock, mock_sys: MagicMock) -> None:
        mock_run.side_effect = FileNotFoundError("not found")
        assert send_desktop_notification("Title", "Msg") is False

    @patch("reporadar.watcher.platform.system", return_value="Linux")
    @patch("reporadar.watcher.subprocess.run")
    def test_command_failure(self, mock_run: MagicMock, mock_sys: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(args="notify-send", returncode=1)
        assert send_desktop_notification("Title", "Msg") is False

    @patch("reporadar.watcher.platform.system", return_value="FreeBSD")
    def test_unsupported_platform(self, mock_sys: MagicMock) -> None:
        assert send_desktop_notification("Title", "Msg") is False


class TestRunUpdateCycle:
    @patch("reporadar.store.PaperStore")
    @patch("reporadar.collector.collect_papers")
    @patch("reporadar.collector.build_queries", return_value=["all:test"])
    @patch("reporadar.profiler.profile_repo")
    @patch("reporadar.ranker.rank_papers")
    @patch("reporadar.digest.write_digest")
    @patch("reporadar.config.load_config")
    @patch("reporadar.config.validate_config", return_value=[])
    def test_successful_cycle(
        self,
        mock_validate: MagicMock,
        mock_load: MagicMock,
        mock_write: MagicMock,
        mock_rank: MagicMock,
        mock_profile: MagicMock,
        mock_build: MagicMock,
        mock_collect: MagicMock,
        mock_store_cls: MagicMock,
    ) -> None:
        cfg = MagicMock()
        cfg.repo_path = "/tmp/repo"
        cfg.output.digest_path = "/tmp/digest.md"
        cfg.output.top_n = 15
        cfg.ranking = MagicMock()
        cfg.queries = MagicMock()
        cfg.arxiv.categories = ["cs.CL"]
        cfg.arxiv.lookback_days = 14
        mock_load.return_value = cfg
        mock_collect.return_value = [{"arxiv_id": "123", "title": "Test"}]
        mock_rank.return_value = [{"arxiv_id": "123", "score_total": 0.8}]

        mock_store = MagicMock()
        mock_store.upsert_papers.return_value = (1, 0)
        mock_store.record_run.return_value = 1
        mock_store.__enter__ = MagicMock(return_value=mock_store)
        mock_store.__exit__ = MagicMock(return_value=False)
        mock_store_cls.return_value = mock_store

        from pathlib import Path

        mock_write.return_value = (Path("/tmp/digest.md"), MagicMock())

        result = run_update_cycle("/tmp/config.yml")
        assert result["success"] is True
        assert result["papers_new"] == 1

    @patch("reporadar.config.load_config")
    def test_config_not_found(self, mock_load: MagicMock) -> None:
        mock_load.side_effect = FileNotFoundError("not found")
        result = run_update_cycle("/nonexistent/config.yml")
        assert result["success"] is False
        assert "not found" in result.get("error", "").lower()

    @patch("reporadar.collector.collect_papers")
    @patch("reporadar.collector.build_queries", return_value=["q"])
    @patch("reporadar.profiler.profile_repo")
    @patch("reporadar.config.load_config")
    @patch("reporadar.config.validate_config", return_value=[])
    def test_collection_error(
        self,
        mock_validate: MagicMock,
        mock_load: MagicMock,
        mock_profile: MagicMock,
        mock_build: MagicMock,
        mock_collect: MagicMock,
    ) -> None:
        from reporadar.collector import CollectionError

        cfg = MagicMock()
        cfg.repo_path = "/tmp/repo"
        mock_load.return_value = cfg
        mock_collect.side_effect = CollectionError("fail")

        result = run_update_cycle("/tmp/config.yml")
        assert result["success"] is False

    @patch("reporadar.collector.collect_papers")
    @patch("reporadar.collector.build_queries", return_value=[])
    @patch("reporadar.profiler.profile_repo")
    @patch("reporadar.config.load_config")
    @patch("reporadar.config.validate_config", return_value=[])
    def test_no_queries(
        self,
        mock_validate: MagicMock,
        mock_load: MagicMock,
        mock_profile: MagicMock,
        mock_build: MagicMock,
        mock_collect: MagicMock,
    ) -> None:
        cfg = MagicMock()
        cfg.repo_path = "/tmp/repo"
        mock_load.return_value = cfg

        result = run_update_cycle("/tmp/config.yml")
        assert result["success"] is True
        assert result["papers_new"] == 0


class TestWatchLoop:
    @patch("reporadar.watcher.time.sleep")
    @patch("reporadar.watcher.run_update_cycle")
    def test_runs_n_cycles(self, mock_cycle: MagicMock, mock_sleep: MagicMock) -> None:
        mock_cycle.return_value = {"success": True, "papers_new": 0, "top_picks_count": 0}
        watch_loop("/tmp/config.yml", interval_seconds=60, max_cycles=3, notify=False)
        assert mock_cycle.call_count == 3

    @patch("reporadar.watcher.time.sleep")
    @patch("reporadar.watcher.run_update_cycle")
    def test_respects_interval(self, mock_cycle: MagicMock, mock_sleep: MagicMock) -> None:
        mock_cycle.return_value = {"success": True, "papers_new": 0, "top_picks_count": 0}
        watch_loop("/tmp/config.yml", interval_seconds=120, max_cycles=2, notify=False)
        # Sleep should be called once (between cycles, not after last)
        assert mock_sleep.call_count == 1
        mock_sleep.assert_called_with(120)

    @patch("reporadar.watcher.time.sleep")
    @patch("reporadar.watcher.run_update_cycle")
    def test_continues_on_failure(self, mock_cycle: MagicMock, mock_sleep: MagicMock) -> None:
        mock_cycle.side_effect = [
            {"success": False, "error": "fail"},
            {"success": True, "papers_new": 1, "top_picks_count": 1},
        ]
        watch_loop("/tmp/config.yml", interval_seconds=10, max_cycles=2, notify=False)
        assert mock_cycle.call_count == 2

    @patch("reporadar.watcher.send_desktop_notification")
    @patch("reporadar.watcher.time.sleep")
    @patch("reporadar.watcher.run_update_cycle")
    def test_sends_notification_on_new_papers(
        self, mock_cycle: MagicMock, mock_sleep: MagicMock, mock_notify: MagicMock
    ) -> None:
        mock_cycle.return_value = {"success": True, "papers_new": 5, "top_picks_count": 2}
        mock_notify.return_value = True
        watch_loop("/tmp/config.yml", interval_seconds=10, max_cycles=1, notify=True)
        mock_notify.assert_called_once()

    @patch("reporadar.watcher.send_desktop_notification")
    @patch("reporadar.watcher.time.sleep")
    @patch("reporadar.watcher.run_update_cycle")
    def test_no_notification_when_disabled(
        self, mock_cycle: MagicMock, mock_sleep: MagicMock, mock_notify: MagicMock
    ) -> None:
        mock_cycle.return_value = {"success": True, "papers_new": 5, "top_picks_count": 2}
        watch_loop("/tmp/config.yml", interval_seconds=10, max_cycles=1, notify=False)
        mock_notify.assert_not_called()
