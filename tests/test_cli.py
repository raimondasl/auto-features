"""Tests for reporadar.cli — integration tests for all CLI commands."""

from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import click
import pytest
from click.testing import CliRunner

from reporadar.cli import _format_size, _parse_since, cli
from reporadar.store import PaperStore

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _setup_repo(tmp_path: Path) -> Path:
    """Create a minimal repo with config and README."""
    shutil.copy(FIXTURES_DIR / "sample_readme.md", tmp_path / "README.md")
    config_file = tmp_path / ".reporadar.yml"
    config_file.write_text(
        f"repo_path: {tmp_path}\n"
        "arxiv:\n"
        "  categories: [cs.CL]\n"
        "  max_results_per_query: 10\n"
        "  lookback_days: 14\n"
        "queries:\n"
        "  seed: []\n"
        "  exclude: []\n"
        "ranking:\n"
        "  w_keyword: 1.0\n"
        "  w_category: 0.5\n"
        "  w_recency: 0.3\n"
        "output:\n"
        f"  digest_path: {tmp_path / 'digest.md'}\n"
        "  top_n: 15\n",
        encoding="utf-8",
    )
    return tmp_path


def _seed_db(tmp_path: Path) -> None:
    """Create a populated DB with papers, a run, and scores."""
    db_path = tmp_path / ".reporadar" / "papers.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with PaperStore(db_path) as store:
        papers = [
            {
                "arxiv_id": "2401.00001v1",
                "title": "Test Paper on RAG",
                "authors": ["Alice"],
                "abstract": "Retrieval augmented generation with transformers.",
                "categories": ["cs.CL"],
                "published": datetime.now(UTC).isoformat(),
                "updated": None,
                "url": "http://arxiv.org/abs/2401.00001v1",
                "pdf_url": "http://arxiv.org/pdf/2401.00001v1",
            },
            {
                "arxiv_id": "2401.00002v1",
                "title": "Low Score Paper",
                "authors": ["Bob"],
                "abstract": "Something unrelated.",
                "categories": ["cs.CV"],
                "published": "2020-01-01T00:00:00+00:00",
                "updated": None,
                "url": "http://arxiv.org/abs/2401.00002v1",
                "pdf_url": None,
            },
        ]
        store.upsert_papers(papers)
        run_id = store.record_run(["all:test"], papers_new=2, papers_seen=0)
        store.save_scores(
            run_id,
            [
                {
                    "arxiv_id": "2401.00001v1",
                    "score_total": 0.85,
                    "keyword_score": 0.5,
                    "category_score": 0.2,
                    "recency_score": 0.15,
                    "matched_query": "all:test",
                },
                {
                    "arxiv_id": "2401.00002v1",
                    "score_total": 0.1,
                    "keyword_score": 0.05,
                    "category_score": 0.0,
                    "recency_score": 0.05,
                    "matched_query": "all:test",
                },
            ],
        )


class TestParseSince:
    def test_valid_days(self) -> None:
        assert _parse_since("7d") == 7
        assert _parse_since("14d") == 14
        assert _parse_since("1d") == 1
        assert _parse_since("  30d  ") == 30

    def test_invalid_format(self) -> None:
        with pytest.raises(click.BadParameter):
            _parse_since("7")
        with pytest.raises(click.BadParameter):
            _parse_since("abc")
        with pytest.raises(click.BadParameter):
            _parse_since("7w")
        with pytest.raises(click.BadParameter):
            _parse_since("xd")


class TestInitCommand:
    def test_creates_config_and_dir(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["init", "--path", str(tmp_path)])

        assert result.exit_code == 0
        assert (tmp_path / ".reporadar.yml").exists()
        assert (tmp_path / ".reporadar").is_dir()
        assert "RepoRadar initialized" in result.output

    def test_idempotent(self, tmp_path: Path) -> None:
        runner = CliRunner()
        runner.invoke(cli, ["init", "--path", str(tmp_path)])
        result = runner.invoke(cli, ["init", "--path", str(tmp_path)])

        assert result.exit_code == 0
        assert "Config already exists" in result.output
        assert "Storage directory already exists" in result.output

    def test_config_is_valid_yaml(self, tmp_path: Path) -> None:
        runner = CliRunner()
        runner.invoke(cli, ["init", "--path", str(tmp_path)])

        import yaml

        content = (tmp_path / ".reporadar.yml").read_text(encoding="utf-8")
        data = yaml.safe_load(content)
        assert data["repo_path"] == "."
        assert "arxiv" in data


class TestProfileCommand:
    def test_prints_keywords(self, tmp_path: Path) -> None:
        repo = _setup_repo(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ["profile", "--config", str(repo / ".reporadar.yml")])

        assert result.exit_code == 0
        assert "Keywords (TF-IDF):" in result.output
        assert "Anchors (packages):" in result.output
        assert "Inferred domains:" in result.output

    def test_empty_repo(self, tmp_path: Path) -> None:
        config_file = tmp_path / ".reporadar.yml"
        config_file.write_text(f"repo_path: {tmp_path}\n", encoding="utf-8")

        runner = CliRunner()
        result = runner.invoke(cli, ["profile", "--config", str(config_file)])

        assert result.exit_code == 0
        assert "(none found)" in result.output


class TestUpdateCommand:
    @patch("reporadar.cli.collect_papers")
    def test_full_pipeline(self, mock_collect: MagicMock, tmp_path: Path) -> None:
        repo = _setup_repo(tmp_path)
        now = datetime.now(UTC).isoformat()
        mock_collect.return_value = [
            {
                "arxiv_id": "2401.99999v1",
                "title": "Mock Paper",
                "authors": ["Test Author"],
                "abstract": "A test abstract about retrieval.",
                "categories": ["cs.CL"],
                "published": now,
                "updated": None,
                "url": "http://arxiv.org/abs/2401.99999v1",
                "pdf_url": None,
                "matched_query": "all:test",
            },
        ]

        runner = CliRunner()
        result = runner.invoke(cli, ["update", "--config", str(repo / ".reporadar.yml")])

        assert result.exit_code == 0
        assert "Profiling repo:" in result.output
        assert "Fetching papers" in result.output
        assert "Scoring papers" in result.output
        assert "Done!" in result.output
        assert (repo / ".reporadar" / "papers.db").exists()

    @patch("reporadar.cli.collect_papers")
    def test_no_papers_found(self, mock_collect: MagicMock, tmp_path: Path) -> None:
        repo = _setup_repo(tmp_path)
        mock_collect.return_value = []

        runner = CliRunner()
        result = runner.invoke(cli, ["update", "--config", str(repo / ".reporadar.yml")])

        assert result.exit_code == 0
        assert "No new papers found" in result.output

    @patch("reporadar.cli.collect_papers")
    def test_collection_error(self, mock_collect: MagicMock, tmp_path: Path) -> None:
        from reporadar.collector import CollectionError

        repo = _setup_repo(tmp_path)
        mock_collect.side_effect = CollectionError("network down")

        runner = CliRunner()
        result = runner.invoke(cli, ["update", "--config", str(repo / ".reporadar.yml")])

        assert result.exit_code == 1
        assert "Failed to fetch papers" in result.output

    @patch("reporadar.cli.collect_papers")
    def test_no_queries(self, mock_collect: MagicMock, tmp_path: Path) -> None:
        # Empty repo with no README — profiler finds no keywords
        config_file = tmp_path / ".reporadar.yml"
        config_file.write_text(
            f"repo_path: {tmp_path}\narxiv:\n  categories: []\nqueries:\n  seed: []\n",
            encoding="utf-8",
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["update", "--config", str(config_file)])

        assert result.exit_code == 0
        assert "No queries to run" in result.output


class TestDigestCommand:
    def test_generates_digest(self, tmp_path: Path) -> None:
        repo = _setup_repo(tmp_path)
        _seed_db(repo)

        runner = CliRunner()
        result = runner.invoke(cli, ["digest", "--config", str(repo / ".reporadar.yml")])

        assert result.exit_code == 0
        assert "Digest written to" in result.output
        assert (repo / "digest.md").exists()

        content = (repo / "digest.md").read_text(encoding="utf-8")
        assert "RepoRadar Digest" in content
        assert "Test Paper on RAG" in content

    def test_html_format(self, tmp_path: Path) -> None:
        repo = _setup_repo(tmp_path)
        _seed_db(repo)

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "digest",
                "--config",
                str(repo / ".reporadar.yml"),
                "--format",
                "html",
            ],
        )

        assert result.exit_code == 0
        assert (repo / "digest.html").exists()

    def test_custom_output(self, tmp_path: Path) -> None:
        repo = _setup_repo(tmp_path)
        _seed_db(repo)
        out = tmp_path / "custom" / "output.md"

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "digest",
                "--config",
                str(repo / ".reporadar.yml"),
                "-o",
                str(out),
            ],
        )

        assert result.exit_code == 0
        assert out.exists()

    def test_diff_flag(self, tmp_path: Path) -> None:
        repo = _setup_repo(tmp_path)
        _seed_db(repo)

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "digest",
                "--config",
                str(repo / ".reporadar.yml"),
                "--diff",
            ],
        )

        assert result.exit_code == 0
        assert "Digest written to" in result.output
        content = (repo / "digest.md").read_text(encoding="utf-8")
        # All papers should be [NEW] since there's only one run
        assert "[NEW]" in content

    def test_no_database(self, tmp_path: Path) -> None:
        repo = _setup_repo(tmp_path)

        runner = CliRunner()
        result = runner.invoke(cli, ["digest", "--config", str(repo / ".reporadar.yml")])

        assert result.exit_code == 1
        assert "No database found" in result.output

    def test_no_runs(self, tmp_path: Path) -> None:
        repo = _setup_repo(tmp_path)
        # Create empty DB with no runs
        db_path = repo / ".reporadar" / "papers.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with PaperStore(db_path):
            pass

        runner = CliRunner()
        result = runner.invoke(cli, ["digest", "--config", str(repo / ".reporadar.yml")])

        assert result.exit_code == 1
        assert "No runs found" in result.output


class TestOpenCommand:
    @patch("reporadar.cli.webbrowser.open")
    def test_opens_papers(self, mock_open: MagicMock, tmp_path: Path) -> None:
        repo = _setup_repo(tmp_path)
        _seed_db(repo)

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "open",
                "--config",
                str(repo / ".reporadar.yml"),
                "-n",
                "1",
            ],
        )

        assert result.exit_code == 0
        assert "Opening:" in result.output
        assert "Opened 1 papers" in result.output
        mock_open.assert_called_once()

    @patch("reporadar.cli.webbrowser.open")
    def test_default_top_5(self, mock_open: MagicMock, tmp_path: Path) -> None:
        repo = _setup_repo(tmp_path)
        _seed_db(repo)

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "open",
                "--config",
                str(repo / ".reporadar.yml"),
            ],
        )

        assert result.exit_code == 0
        assert mock_open.call_count == 2  # only 2 papers in seeded DB

    def test_no_database(self, tmp_path: Path) -> None:
        repo = _setup_repo(tmp_path)

        runner = CliRunner()
        result = runner.invoke(cli, ["open", "--config", str(repo / ".reporadar.yml")])

        assert result.exit_code == 1
        assert "No database found" in result.output

    def test_no_runs(self, tmp_path: Path) -> None:
        repo = _setup_repo(tmp_path)
        db_path = repo / ".reporadar" / "papers.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with PaperStore(db_path):
            pass

        runner = CliRunner()
        result = runner.invoke(cli, ["open", "--config", str(repo / ".reporadar.yml")])

        assert result.exit_code == 1
        assert "No runs found" in result.output


class TestStatusCommand:
    def test_status_with_db_and_runs(self, tmp_path: Path) -> None:
        repo = _setup_repo(tmp_path)
        _seed_db(repo)

        runner = CliRunner()
        result = runner.invoke(cli, ["status", "--config", str(repo / ".reporadar.yml")])

        assert result.exit_code == 0
        assert "Repo path:" in result.output
        assert "Categories:" in result.output
        assert "DB size:" in result.output
        assert "Papers:" in result.output
        assert "Last run:" in result.output
        assert "New/seen:" in result.output

    def test_status_no_db(self, tmp_path: Path) -> None:
        repo = _setup_repo(tmp_path)

        runner = CliRunner()
        result = runner.invoke(cli, ["status", "--config", str(repo / ".reporadar.yml")])

        assert result.exit_code == 0
        assert "No database found" in result.output

    def test_status_db_no_runs(self, tmp_path: Path) -> None:
        repo = _setup_repo(tmp_path)
        db_path = repo / ".reporadar" / "papers.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with PaperStore(db_path):
            pass

        runner = CliRunner()
        result = runner.invoke(cli, ["status", "--config", str(repo / ".reporadar.yml")])

        assert result.exit_code == 0
        assert "No runs yet" in result.output


class TestHistoryCommand:
    def test_history_with_runs(self, tmp_path: Path) -> None:
        repo = _setup_repo(tmp_path)
        _seed_db(repo)

        runner = CliRunner()
        result = runner.invoke(cli, ["history", "--config", str(repo / ".reporadar.yml")])

        assert result.exit_code == 0
        assert "Run" in result.output
        assert "New" in result.output

    def test_history_no_db(self, tmp_path: Path) -> None:
        repo = _setup_repo(tmp_path)

        runner = CliRunner()
        result = runner.invoke(cli, ["history", "--config", str(repo / ".reporadar.yml")])

        assert result.exit_code == 1
        assert "No database found" in result.output

    def test_history_no_runs(self, tmp_path: Path) -> None:
        repo = _setup_repo(tmp_path)
        db_path = repo / ".reporadar" / "papers.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with PaperStore(db_path):
            pass

        runner = CliRunner()
        result = runner.invoke(cli, ["history", "--config", str(repo / ".reporadar.yml")])

        assert result.exit_code == 0
        assert "No runs found" in result.output

    def test_history_limit_flag(self, tmp_path: Path) -> None:
        repo = _setup_repo(tmp_path)
        db_path = repo / ".reporadar" / "papers.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with PaperStore(db_path) as store:
            for i in range(5):
                store.record_run([f"q{i}"], i, 0)

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "history",
                "--config",
                str(repo / ".reporadar.yml"),
                "--limit",
                "2",
            ],
        )

        assert result.exit_code == 0
        # Should show header + separator + 2 data rows
        lines = [line for line in result.output.strip().split("\n") if line.strip()]
        # header + separator + 2 runs = 4 lines
        assert len(lines) == 4


class TestFormatSize:
    def test_bytes(self) -> None:
        assert _format_size(500) == "500 B"

    def test_kilobytes(self) -> None:
        result = _format_size(2048)
        assert "KB" in result

    def test_megabytes(self) -> None:
        result = _format_size(5 * 1024 * 1024)
        assert "MB" in result
