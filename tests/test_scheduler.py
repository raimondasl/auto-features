"""Tests for reporadar.scheduler."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from reporadar.scheduler import (
    CRON_MARKER,
    _build_command,
    _cron_to_schtasks_args,
    add_cron_job,
    add_schedule,
    add_schtask,
    list_cron_jobs,
    list_schedules,
    list_schtasks,
    remove_cron_job,
    remove_schedule,
    remove_schtask,
)


class TestCronToSchtasksArgs:
    def test_daily(self) -> None:
        args = _cron_to_schtasks_args("0 9 * * *")
        assert args == ["/SC", "DAILY", "/ST", "09:00"]

    def test_daily_with_minutes(self) -> None:
        args = _cron_to_schtasks_args("30 14 * * *")
        assert args == ["/SC", "DAILY", "/ST", "14:30"]

    def test_weekly_monday(self) -> None:
        args = _cron_to_schtasks_args("0 9 * * 1")
        assert args == ["/SC", "WEEKLY", "/D", "MON", "/ST", "09:00"]

    def test_weekly_friday(self) -> None:
        args = _cron_to_schtasks_args("0 17 * * 5")
        assert args == ["/SC", "WEEKLY", "/D", "FRI", "/ST", "17:00"]

    def test_hourly(self) -> None:
        args = _cron_to_schtasks_args("0 * * * *")
        assert args == ["/SC", "HOURLY"]

    def test_invalid_not_enough_fields(self) -> None:
        with pytest.raises(ValueError, match="Expected 5 cron fields"):
            _cron_to_schtasks_args("0 9 *")

    def test_unsupported_complex_pattern(self) -> None:
        with pytest.raises(ValueError, match="Unsupported"):
            _cron_to_schtasks_args("0 9 1 * *")  # specific day-of-month

    def test_invalid_day_of_week(self) -> None:
        with pytest.raises(ValueError, match="Invalid day"):
            _cron_to_schtasks_args("0 9 * * 8")


class TestBuildCommand:
    def test_builds_command(self) -> None:
        cmd = _build_command("/path/to/config.yml")
        assert "update --config /path/to/config.yml" in cmd
        assert "digest --config /path/to/config.yml" in cmd
        assert "&&" in cmd


class TestCronJobs:
    @patch("reporadar.scheduler._set_crontab", return_value=True)
    @patch("reporadar.scheduler._get_current_crontab", return_value="")
    def test_add_to_empty_crontab(self, mock_get: MagicMock, mock_set: MagicMock) -> None:
        result = add_cron_job("0 9 * * 1", "/path/config.yml")
        assert result is True
        written = mock_set.call_args[0][0]
        assert CRON_MARKER in written
        assert "0 9 * * 1" in written

    @patch("reporadar.scheduler._set_crontab", return_value=True)
    @patch(
        "reporadar.scheduler._get_current_crontab",
        return_value=f"0 8 * * * old_cmd {CRON_MARKER}\n",
    )
    def test_replace_existing(self, mock_get: MagicMock, mock_set: MagicMock) -> None:
        result = add_cron_job("0 9 * * 1", "/path/config.yml")
        assert result is True
        written = mock_set.call_args[0][0]
        assert "old_cmd" not in written
        assert "0 9 * * 1" in written

    @patch(
        "reporadar.scheduler._get_current_crontab",
        return_value=f"0 9 * * 1 rr update {CRON_MARKER}\n",
    )
    def test_list(self, mock_get: MagicMock) -> None:
        tasks = list_cron_jobs()
        assert len(tasks) == 1
        assert tasks[0].cron_expr == "0 9 * * 1"
        assert tasks[0].platform == "unix"

    @patch("reporadar.scheduler._get_current_crontab", return_value="")
    def test_list_empty(self, mock_get: MagicMock) -> None:
        tasks = list_cron_jobs()
        assert tasks == []

    @patch("reporadar.scheduler._set_crontab", return_value=True)
    @patch(
        "reporadar.scheduler._get_current_crontab",
        return_value=f"0 9 * * 1 rr cmd {CRON_MARKER}\nother_job\n",
    )
    def test_remove(self, mock_get: MagicMock, mock_set: MagicMock) -> None:
        result = remove_cron_job()
        assert result is True
        written = mock_set.call_args[0][0]
        assert CRON_MARKER not in written
        assert "other_job" in written


class TestSchtasks:
    @patch("reporadar.scheduler.subprocess.run")
    def test_add(self, mock_run: MagicMock) -> None:
        # First call: delete existing (success), second call: create
        mock_run.return_value = subprocess.CompletedProcess(args="schtasks", returncode=0)
        result = add_schtask("0 9 * * *", "/path/config.yml")
        assert result is True
        assert mock_run.call_count == 2  # delete + create

    @patch("reporadar.scheduler.subprocess.run")
    def test_add_create_failure(self, mock_run: MagicMock) -> None:
        # Delete succeeds, create fails
        mock_run.side_effect = [
            subprocess.CompletedProcess(args="del", returncode=0),
            subprocess.CompletedProcess(args="create", returncode=1, stderr="fail"),
        ]
        result = add_schtask("0 9 * * *", "/path/config.yml")
        assert result is False

    @patch("reporadar.scheduler.subprocess.run")
    def test_add_unsupported_cron(self, mock_run: MagicMock) -> None:
        result = add_schtask("0 9 1 * *", "/path/config.yml")
        assert result is False
        mock_run.assert_not_called()

    @patch("reporadar.scheduler.subprocess.run")
    def test_list(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args="query",
            returncode=0,
            stdout="Task To Run: rr update\nSchedule Type: Daily\n",
        )
        tasks = list_schtasks()
        assert len(tasks) == 1
        assert tasks[0].command == "rr update"
        assert tasks[0].platform == "windows"

    @patch("reporadar.scheduler.subprocess.run")
    def test_list_not_found(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(args="query", returncode=1, stdout="")
        tasks = list_schtasks()
        assert tasks == []

    @patch("reporadar.scheduler.subprocess.run")
    def test_remove_success(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(args="del", returncode=0)
        assert remove_schtask() is True

    @patch("reporadar.scheduler.subprocess.run")
    def test_remove_not_found(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(args="del", returncode=1)
        assert remove_schtask() is False


class TestPublicInterface:
    @patch("reporadar.scheduler._detect_platform", return_value="unix")
    @patch("reporadar.scheduler.add_cron_job", return_value=True)
    def test_add_schedule_unix(self, mock_add: MagicMock, mock_plat: MagicMock) -> None:
        result = add_schedule("0 9 * * 1", "/path/config.yml")
        assert result is True
        mock_add.assert_called_once()

    @patch("reporadar.scheduler._detect_platform", return_value="windows")
    @patch("reporadar.scheduler.add_schtask", return_value=True)
    def test_add_schedule_windows(self, mock_add: MagicMock, mock_plat: MagicMock) -> None:
        result = add_schedule("0 9 * * *", "/path/config.yml")
        assert result is True
        mock_add.assert_called_once()

    @patch("reporadar.scheduler._detect_platform", return_value="unix")
    @patch("reporadar.scheduler.list_cron_jobs", return_value=[])
    def test_list_schedules_unix(self, mock_list: MagicMock, mock_plat: MagicMock) -> None:
        result = list_schedules()
        assert result == []
        mock_list.assert_called_once()

    @patch("reporadar.scheduler._detect_platform", return_value="windows")
    @patch("reporadar.scheduler.list_schtasks", return_value=[])
    def test_list_schedules_windows(self, mock_list: MagicMock, mock_plat: MagicMock) -> None:
        result = list_schedules()
        assert result == []
        mock_list.assert_called_once()

    @patch("reporadar.scheduler._detect_platform", return_value="unix")
    @patch("reporadar.scheduler.remove_cron_job", return_value=True)
    def test_remove_schedule_unix(self, mock_rm: MagicMock, mock_plat: MagicMock) -> None:
        assert remove_schedule() is True
        mock_rm.assert_called_once()

    @patch("reporadar.scheduler._detect_platform", return_value="windows")
    @patch("reporadar.scheduler.remove_schtask", return_value=True)
    def test_remove_schedule_windows(self, mock_rm: MagicMock, mock_plat: MagicMock) -> None:
        assert remove_schedule() is True
        mock_rm.assert_called_once()
