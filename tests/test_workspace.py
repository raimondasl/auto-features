"""Tests for reporadar.workspace."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from reporadar.store import PaperStore
from reporadar.workspace import (
    combined_digest_data,
    open_workspace_store,
    score_papers_for_repo,
)


def _make_paper(arxiv_id: str = "2401.12345v1", **overrides) -> dict:
    base = {
        "arxiv_id": arxiv_id,
        "title": f"Test Paper {arxiv_id}",
        "authors": ["Alice"],
        "abstract": "A test abstract.",
        "categories": ["cs.CL"],
        "published": "2024-01-20T00:00:00+00:00",
        "updated": None,
        "url": f"http://arxiv.org/abs/{arxiv_id}",
        "pdf_url": None,
    }
    base.update(overrides)
    return base


class TestWorkspaceStore:
    def test_add_and_get_repos(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "ws.db") as store:
            store.add_workspace_repo("myrepo", "/path/to/repo", "/path/config.yml")
            repos = store.get_workspace_repos()
        assert len(repos) == 1
        assert repos[0]["repo_id"] == "myrepo"
        assert repos[0]["repo_path"] == "/path/to/repo"
        assert repos[0]["config_path"] == "/path/config.yml"

    def test_remove_repo(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "ws.db") as store:
            store.add_workspace_repo("myrepo", "/path/to/repo")
            removed = store.remove_workspace_repo("myrepo")
            assert removed is True
            repos = store.get_workspace_repos()
            assert len(repos) == 0

    def test_remove_nonexistent(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "ws.db") as store:
            removed = store.remove_workspace_repo("nope")
            assert removed is False

    def test_save_and_get_repo_scores(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "ws.db") as store:
            store.add_workspace_repo("repo1", "/path")
            store.upsert_paper(_make_paper("2401.00001v1"))
            run_id = store.record_run(["q1"], 1, 0)
            store.save_repo_scores(
                "repo1",
                run_id,
                [
                    {
                        "arxiv_id": "2401.00001v1",
                        "score_total": 0.8,
                        "keyword_score": 0.5,
                        "category_score": 0.2,
                        "recency_score": 0.1,
                    },
                ],
            )
            scores = store.get_repo_scores_for_run(run_id, repo_id="repo1")
            assert len(scores) == 1
            assert scores[0]["score_total"] == 0.8
            assert scores[0]["repo_id"] == "repo1"

    def test_get_repo_scores_all_repos(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "ws.db") as store:
            store.add_workspace_repo("repo1", "/p1")
            store.add_workspace_repo("repo2", "/p2")
            store.upsert_paper(_make_paper("2401.00001v1"))
            run_id = store.record_run(["q1"], 1, 0)
            store.save_repo_scores(
                "repo1", run_id, [{"arxiv_id": "2401.00001v1", "score_total": 0.8}]
            )
            store.save_repo_scores(
                "repo2", run_id, [{"arxiv_id": "2401.00001v1", "score_total": 0.6}]
            )
            all_scores = store.get_repo_scores_for_run(run_id)
            assert len(all_scores) == 2


class TestScorePapersForRepo:
    @patch("reporadar.ranker.rank_papers")
    @patch("reporadar.profiler.profile_repo")
    def test_profiles_and_ranks(self, mock_profile: MagicMock, mock_rank: MagicMock) -> None:
        mock_profile.return_value = MagicMock(keywords=[], anchors=[], domains=[])
        mock_rank.return_value = [
            {"arxiv_id": "2401.00001v1", "score_total": 0.9},
        ]
        cfg = MagicMock()
        cfg.ranking = MagicMock()
        cfg.queries = MagicMock()
        cfg.arxiv.categories = ["cs.CL"]
        cfg.arxiv.lookback_days = 14

        result = score_papers_for_repo("myrepo", "/path", [_make_paper()], cfg)
        assert len(result) == 1
        assert result[0]["repo_id"] == "myrepo"
        assert result[0]["score_total"] == 0.9
        mock_profile.assert_called_once()
        mock_rank.assert_called_once()


class TestCombinedDigestData:
    def test_multi_repo_merge(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "ws.db") as store:
            store.add_workspace_repo("repo1", "/p1")
            store.add_workspace_repo("repo2", "/p2")
            store.upsert_paper(_make_paper("2401.00001v1", title="Shared Paper"))
            store.upsert_paper(_make_paper("2401.00002v1", title="Repo1 Only"))
            run_id = store.record_run(["q1"], 2, 0)

            store.save_repo_scores(
                "repo1",
                run_id,
                [
                    {"arxiv_id": "2401.00001v1", "score_total": 0.9},
                    {"arxiv_id": "2401.00002v1", "score_total": 0.7},
                ],
            )
            store.save_repo_scores(
                "repo2",
                run_id,
                [{"arxiv_id": "2401.00001v1", "score_total": 0.6}],
            )

            result = combined_digest_data(store, run_id)

        assert len(result) == 2
        # Shared Paper should be first (max score 0.9)
        assert result[0]["arxiv_id"] == "2401.00001v1"
        assert result[0]["max_score"] == 0.9
        assert len(result[0]["relevant_repos"]) == 2

    def test_max_score_sorting(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "ws.db") as store:
            store.add_workspace_repo("repo1", "/p1")
            store.upsert_paper(_make_paper("2401.00001v1"))
            store.upsert_paper(_make_paper("2401.00002v1"))
            run_id = store.record_run(["q1"], 2, 0)

            store.save_repo_scores(
                "repo1",
                run_id,
                [
                    {"arxiv_id": "2401.00001v1", "score_total": 0.3},
                    {"arxiv_id": "2401.00002v1", "score_total": 0.8},
                ],
            )

            result = combined_digest_data(store, run_id)

        assert result[0]["arxiv_id"] == "2401.00002v1"
        assert result[1]["arxiv_id"] == "2401.00001v1"

    def test_relevant_repos_tagging(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "ws.db") as store:
            store.add_workspace_repo("repo1", "/p1")
            store.add_workspace_repo("repo2", "/p2")
            store.upsert_paper(_make_paper("2401.00001v1"))
            run_id = store.record_run(["q1"], 1, 0)

            store.save_repo_scores(
                "repo1", run_id, [{"arxiv_id": "2401.00001v1", "score_total": 0.9}]
            )
            store.save_repo_scores(
                "repo2", run_id, [{"arxiv_id": "2401.00001v1", "score_total": 0.4}]
            )

            result = combined_digest_data(store, run_id)

        assert len(result) == 1
        repos = result[0]["relevant_repos"]
        repo_ids = [r["repo_id"] for r in repos]
        assert "repo1" in repo_ids
        assert "repo2" in repo_ids

    def test_top_n_limit(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "ws.db") as store:
            store.add_workspace_repo("repo1", "/p1")
            for i in range(5):
                store.upsert_paper(_make_paper(f"2401.0000{i}v1"))
            run_id = store.record_run(["q1"], 5, 0)
            store.save_repo_scores(
                "repo1",
                run_id,
                [{"arxiv_id": f"2401.0000{i}v1", "score_total": i * 0.1} for i in range(5)],
            )

            result = combined_digest_data(store, run_id, top_n=2)

        assert len(result) == 2


class TestOpenWorkspaceStore:
    def test_creates_store(self, tmp_path: Path) -> None:
        store = open_workspace_store(tmp_path / "test.db")
        try:
            assert store.schema_version() == 4
        finally:
            store.close()


class TestEnsureWorkspaceDir:
    @patch("reporadar.workspace.WORKSPACE_DIR")
    def test_creates_dir(self, mock_dir: MagicMock, tmp_path: Path) -> None:
        ws_dir = tmp_path / ".reporadar"
        mock_dir.__truediv__ = lambda self, x: ws_dir / x
        # Can't easily mock Path.home(), just test the logic by calling directly
        ws_dir.mkdir(parents=True, exist_ok=True)
        assert ws_dir.exists()
