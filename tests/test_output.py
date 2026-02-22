"""Tests for reporadar.output."""

from __future__ import annotations

from unittest.mock import patch

import click
from click.testing import CliRunner

from reporadar.output import error, info, muted, setup_verbose_logging, success, warn


def _capture(func, msg):
    """Run an output function inside a Click context and capture output."""

    @click.command()
    def cmd():
        func(msg)

    runner = CliRunner()
    return runner.invoke(cmd)


class TestOutputHelpers:
    def test_success_produces_output(self) -> None:
        result = _capture(success, "All good")
        assert result.exit_code == 0
        assert "All good" in result.output

    def test_warn_produces_output(self) -> None:
        result = _capture(warn, "Be careful")
        assert result.exit_code == 0
        assert "Be careful" in result.output

    def test_error_goes_to_stderr(self) -> None:
        # error() writes to stderr via click.echo(..., err=True)
        with patch("reporadar.output.click.echo") as mock_echo:
            error("Something broke")
            mock_echo.assert_called_once()
            # Verify err=True was passed
            _, kwargs = mock_echo.call_args
            assert kwargs.get("err") is True

    def test_info_produces_output(self) -> None:
        result = _capture(info, "Just FYI")
        assert result.exit_code == 0
        assert "Just FYI" in result.output

    def test_muted_produces_output(self) -> None:
        result = _capture(muted, "Quiet message")
        assert result.exit_code == 0
        assert "Quiet message" in result.output

    def test_setup_verbose_logging(self) -> None:
        # Should not raise
        setup_verbose_logging()
