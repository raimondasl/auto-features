"""Tests for reporadar.gh_issues."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from reporadar.gh_issues import (
    check_gh_available,
    create_issue,
    create_issues,
    format_issue,
)


def _make_paper(**overrides) -> dict:
    base = {
        "arxiv_id": "2401.12345v1",
        "title": "Test Paper on RAG",
        "authors": ["Alice Smith", "Bob Jones"],
        "abstract": "We propose a novel approach to retrieval augmented generation.",
        "categories": ["cs.CL", "cs.LG"],
        "published": "2024-01-20T00:00:00+00:00",
        "url": "http://arxiv.org/abs/2401.12345v1",
        "score_total": 0.85,
        "suggestions": ["Try integrating RAG pipeline"],
    }
    base.update(overrides)
    return base


class TestCheckGhAvailable:
    @patch("reporadar.gh_issues.subprocess.run")
    def test_returns_true_when_available(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=0)
        assert check_gh_available() is True

    @patch("reporadar.gh_issues.subprocess.run")
    def test_returns_false_when_not_found(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = FileNotFoundError("gh not found")
        assert check_gh_available() is False

    @patch("reporadar.gh_issues.subprocess.run")
    def test_returns_false_on_nonzero_exit(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=1)
        assert check_gh_available() is False


class TestFormatIssue:
    def test_title_format(self) -> None:
        paper = _make_paper()
        issue = format_issue(paper)
        assert issue["title"] == "[RepoRadar] Test Paper on RAG"

    def test_body_contains_paper_link(self) -> None:
        paper = _make_paper()
        issue = format_issue(paper)
        assert "http://arxiv.org/abs/2401.12345v1" in issue["body"]

    def test_body_contains_abstract(self) -> None:
        paper = _make_paper()
        issue = format_issue(paper)
        assert "retrieval augmented generation" in issue["body"]

    def test_body_contains_score(self) -> None:
        paper = _make_paper()
        issue = format_issue(paper)
        assert "0.85" in issue["body"]

    def test_body_contains_suggestions(self) -> None:
        paper = _make_paper()
        issue = format_issue(paper)
        assert "Try integrating RAG pipeline" in issue["body"]


class TestFormatIssueWithEnrichment:
    def test_code_links_in_body(self) -> None:
        paper = _make_paper()
        enrichment = {
            "has_code": True,
            "code_urls": ["https://github.com/foo/bar"],
            "datasets": [],
            "tasks": [],
        }
        issue = format_issue(paper, enrichment)
        assert "https://github.com/foo/bar" in issue["body"]

    def test_datasets_in_body(self) -> None:
        paper = _make_paper()
        enrichment = {
            "has_code": False,
            "code_urls": [],
            "datasets": ["ImageNet", "CIFAR-10"],
            "tasks": [],
        }
        issue = format_issue(paper, enrichment)
        assert "ImageNet" in issue["body"]
        assert "CIFAR-10" in issue["body"]

    def test_tasks_in_body(self) -> None:
        paper = _make_paper()
        enrichment = {
            "has_code": False,
            "code_urls": [],
            "datasets": [],
            "tasks": ["Image Classification"],
        }
        issue = format_issue(paper, enrichment)
        assert "Image Classification" in issue["body"]


class TestCreateIssue:
    @patch("reporadar.gh_issues.subprocess.run")
    def test_calls_gh_with_correct_args(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="https://github.com/foo/bar/issues/1\n",
        )
        issue = {"title": "Test", "body": "Body"}
        url = create_issue(issue, labels=["reporadar"])

        assert url == "https://github.com/foo/bar/issues/1"
        call_args = mock_run.call_args[0][0]
        assert "gh" in call_args
        assert "issue" in call_args
        assert "create" in call_args
        assert "--label" in call_args

    @patch("reporadar.gh_issues.subprocess.run")
    def test_returns_none_on_failure(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=1, stderr="error")
        issue = {"title": "Test", "body": "Body"}
        assert create_issue(issue) is None


class TestCreateIssues:
    @patch("reporadar.gh_issues.create_issue")
    def test_batch_creation(self, mock_create: MagicMock) -> None:
        mock_create.return_value = "https://github.com/foo/bar/issues/1"
        papers = [_make_paper(), _make_paper(arxiv_id="2401.00002v1")]

        results = create_issues(papers)

        assert len(results) == 2
        assert all(r["status"] == "created" for r in results)

    @patch("reporadar.gh_issues.create_issue")
    def test_dry_run_mode(self, mock_create: MagicMock) -> None:
        papers = [_make_paper()]
        results = create_issues(papers, dry_run=True)

        assert len(results) == 1
        assert results[0]["status"] == "dry_run"
        mock_create.assert_not_called()

    @patch("reporadar.gh_issues.create_issue")
    def test_handles_creation_failure(self, mock_create: MagicMock) -> None:
        mock_create.return_value = None
        papers = [_make_paper()]

        results = create_issues(papers)

        assert len(results) == 1
        assert results[0]["status"] == "skipped"
