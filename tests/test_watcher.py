"""Tests for reporadar.watcher."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from reporadar.config import RepoRadarConfig
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
    @patch("reporadar.pipeline.PaperStore")
    @patch("reporadar.pipeline.collect_papers")
    @patch("reporadar.pipeline.build_queries", return_value=["all:test"])
    @patch("reporadar.pipeline.profile_repo")
    @patch("reporadar.pipeline.rank_papers")
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
        # A REAL config, not a MagicMock: the cycle now asks it which stages it enables,
        # and a mock answers every numeric comparison with a TypeError. Using the real
        # dataclass also means a renamed config field breaks this test rather than
        # silently disabling the disclosure.
        cfg = RepoRadarConfig(repo_path="/tmp/repo")
        cfg.output.digest_path = "/tmp/digest.md"
        cfg.output.top_n = 15
        cfg.arxiv.categories = ["cs.CL"]
        cfg.arxiv.lookback_days = 14
        # This test covers cycle mechanics; the integrity check (on by default) would
        # otherwise reach arXiv. TestWatcherIntegrity below covers it directly.
        cfg.signals.integrity = False
        # The watch cycle now runs the FULL pipeline, which includes stage 9
        # enrichment. Off here for the same reason integrity is: this test covers
        # cycle mechanics, and the `_no_network` fixture fails the run otherwise.
        cfg.enrichment.provider = "off"
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
        # The Tier 0 fix, visible at the cycle level: a watch cycle skips NOTHING its
        # config enables, because it runs the same `run_pipeline` as `rr update`. The key
        # still travels in the result (not only in a log line) so the GitHub Action and
        # anything else wrapping the loop reads data rather than stderr -- and so a future
        # re-fork of the pipeline shows up here as a non-empty list.
        assert result["skipped_stages"] == []

    @patch("reporadar.pipeline.PaperStore")
    @patch("reporadar.pipeline.collect_papers")
    @patch("reporadar.pipeline.build_queries", return_value=["all:test"])
    @patch("reporadar.pipeline.profile_repo")
    @patch("reporadar.pipeline.rank_papers")
    @patch("reporadar.digest.write_digest")
    @patch("reporadar.config.load_config")
    @patch("reporadar.config.validate_config", return_value=[])
    def test_a_broken_disclosure_does_not_stop_the_cycle(
        self,
        mock_validate: MagicMock,
        mock_load: MagicMock,
        mock_write: MagicMock,
        mock_rank: MagicMock,
        mock_profile: MagicMock,
        mock_build: MagicMock,
        mock_collect: MagicMock,
        mock_store_cls: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The disclosure is advisory and must never take down an unattended loop — but it
        must also not report "nothing skipped", which would be a silent false all-clear.

        The failure is injected into `unrun_stages` itself rather than through a malformed
        config value. An earlier version of this test set `w_embedding` to a string, which
        stopped working once the watcher shared the real pipeline: that value now breaks
        the RUN, not just the warning. (It breaks `rr update` the same way and always has —
        `validate_config` does not type-check numeric fields. Pre-existing, not this
        change's to fix, but the reason this test could not stay as written.)"""
        cfg = RepoRadarConfig(repo_path="/tmp/repo")
        cfg.output.digest_path = "/tmp/digest.md"
        cfg.arxiv.categories = ["cs.CL"]
        cfg.signals.integrity = False
        # The watch cycle now runs the FULL pipeline, which includes stage 9
        # enrichment. Off here for the same reason integrity is: this test covers
        # cycle mechanics, and the `_no_network` fixture fails the run otherwise.
        cfg.enrichment.provider = "off"
        mock_load.return_value = cfg
        mock_collect.return_value = [{"arxiv_id": "123", "title": "Test"}]
        mock_rank.return_value = [{"arxiv_id": "123", "score_total": 0.8}]

        mock_store = MagicMock()
        mock_store.upsert_papers.return_value = (1, 0)
        mock_store.record_run.return_value = 1
        mock_store.__enter__ = MagicMock(return_value=mock_store)
        mock_store.__exit__ = MagicMock(return_value=False)
        mock_store_cls.return_value = mock_store
        mock_write.return_value = (Path("/tmp/digest.md"), MagicMock())

        with patch("reporadar.stages.unrun_stages", side_effect=RuntimeError("registry exploded")):
            result = run_update_cycle("/tmp/config.yml")
        assert result["success"] is True, "an advisory warning must not fail the cycle"
        assert result["skipped_stages"] is None, "void, not an empty all-clear"
        assert "could not determine" in caplog.text

    @patch("reporadar.config.load_config")
    def test_config_not_found(self, mock_load: MagicMock) -> None:
        mock_load.side_effect = FileNotFoundError("not found")
        result = run_update_cycle("/nonexistent/config.yml")
        assert result["success"] is False
        assert "not found" in result.get("error", "").lower()

    @patch("reporadar.pipeline.collect_papers")
    @patch("reporadar.pipeline.build_queries", return_value=["q"])
    @patch("reporadar.pipeline.profile_repo")
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

        cfg = RepoRadarConfig(repo_path="/tmp/repo")
        mock_load.return_value = cfg
        mock_collect.side_effect = CollectionError("fail")

        result = run_update_cycle("/tmp/config.yml")
        assert result["success"] is False

    @patch("reporadar.pipeline.collect_papers")
    @patch("reporadar.pipeline.build_queries", return_value=[])
    @patch("reporadar.pipeline.profile_repo")
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
        cfg = RepoRadarConfig(repo_path="/tmp/repo")
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


class TestWatcherIntegrity:
    """`rr watch` is the unattended path, so a withdrawn paper here reaches a digest
    nobody is watching for it. The check was originally absent from the watch loop, which
    meant the headline safety property held for `rr update` and silently did not hold
    here; it was then added back by hand while the rest of the pipeline stayed missing.
    Since 2026-08-16 the loop shares `run_pipeline`, so there is one implementation to be
    right about -- but these tests stay end-to-end, because "the shared pipeline is wired
    into the watcher" is exactly the thing a unit test of the pipeline cannot show."""

    def _run(self, tmp_path, comments: dict[str, str]):
        from pathlib import Path

        from reporadar.config import RepoRadarConfig, SignalsConfig
        from reporadar.store import PaperStore

        repo = Path(tmp_path)
        repo.mkdir(parents=True, exist_ok=True)
        (repo / "README.md").write_text("retrieval augmented generation", encoding="utf-8")
        cfg = RepoRadarConfig(repo_path=str(repo), signals=SignalsConfig(integrity=True))
        cfg.queries.seed = ["retrieval augmented generation"]
        # These run a real end-to-end cycle, and since 2026-08-16 that cycle is the FULL
        # pipeline -- including stage 9 enrichment, which reaches Hugging Face. The subject
        # here is the integrity check, so the network stage is off; `_no_network` fails the
        # run at teardown otherwise, which is how this was noticed rather than guessed.
        cfg.enrichment.provider = "off"
        paper = {
            "arxiv_id": "2607.00001v1",
            "title": "A Paper",
            "authors": ["A"],
            "abstract": "retrieval augmented generation",
            "categories": ["cs.CL"],
            "published": "2026-07-25T00:00:00+00:00",
            "updated": None,
            "url": "http://arxiv.org/abs/2607.00001v1",
            "pdf_url": None,
        }
        with (
            patch("reporadar.config.load_config", return_value=cfg),
            patch("reporadar.config.validate_config", return_value=[]),
            patch("reporadar.pipeline.collect_papers", return_value=[paper]),
            patch("reporadar.signals.integrity.fetch_comments", return_value=comments),
            patch("reporadar.digest.write_digest", return_value=(repo / "d.md", None)),
        ):
            result = run_update_cycle("cfg.yml")
        with PaperStore(repo / ".reporadar" / "papers.db") as store:
            run_id = store.get_last_run()["run_id"]
            scores = store.get_scores_for_run(run_id)
        return result, scores

    def test_withdrawn_paper_is_flagged_and_kept_out_of_top_picks(self, tmp_path) -> None:
        result, scores = self._run(tmp_path, {"2607.00001v1": "Withdrawn by the authors"})
        assert result["success"] is True
        assert scores[0]["withdrawn_in"] == "comment"
        assert result["top_picks_count"] == 0

    def test_clean_paper_is_unaffected(self, tmp_path) -> None:
        result, scores = self._run(tmp_path, {"2607.00001v1": "10 pages, 3 figures"})
        assert scores[0]["withdrawn_in"] is None
        assert result["top_picks_count"] == 1

    def test_the_stored_score_is_actually_penalized(self, tmp_path) -> None:
        """The flag and the mute both come from the store join, so they pass even if
        `rank_papers(withdrawn=...)` is never wired up. Only the score proves the
        ranker itself applied the multiplier — verified by removing the wiring and
        watching this fail while the two tests above stayed green."""
        _, withdrawn = self._run(tmp_path / "a", {"2607.00001v1": "Withdrawn"})
        _, clean = self._run(tmp_path / "b", {"2607.00001v1": "10 pages"})
        assert withdrawn[0]["score_total"] < clean[0]["score_total"]
