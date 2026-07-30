"""Tests for reporadar.cli — integration tests for all CLI commands."""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import click
import pytest
from click.testing import CliRunner

from reporadar.cli import (
    _FOUNDATIONAL_LOOKBACK,
    _apply_foundational,
    _format_size,
    _parse_since,
    cli,
)
from reporadar.config import ArxivConfig, RankingConfig
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
        # Enrichment defaults to the live HF Papers API, and `rr update` runs it as
        # stage 9 — so without this every update test made real HTTP calls (and paid
        # three retries with backoff each once the network guard blocked them).
        "enrichment:\n"
        "  provider: 'off'\n"
        # Likewise the integrity check, which defaults ON and queries arXiv for
        # withdrawal comments. Tests that exercise it patch fetch_comments instead.
        "signals:\n"
        "  integrity: false\n"
        "output:\n"
        f"  digest_path: {tmp_path / 'digest.md'}\n"
        "  top_n: 15\n",
        encoding="utf-8",
    )
    return tmp_path


def _community_paper(arxiv_id: str) -> dict:
    """A minimal arXiv-shaped paper for the community-signal tests."""
    return {
        "arxiv_id": arxiv_id,
        "title": f"Paper {arxiv_id}",
        "authors": ["A"],
        "abstract": "language model evaluation",
        "categories": ["cs.CL"],
        "published": datetime.now(UTC).isoformat(),
        "updated": None,
        "url": f"http://arxiv.org/abs/{arxiv_id}",
        "pdf_url": None,
    }


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


class TestSearchCommand:
    def test_text_output_finds_matching_paper(self, tmp_path: Path) -> None:
        repo = _setup_repo(tmp_path)
        _seed_db(tmp_path)
        result = CliRunner().invoke(
            cli,
            ["search", "retrieval augmented generation", "--config", str(repo / ".reporadar.yml")],
        )
        assert result.exit_code == 0
        assert "Test Paper on RAG" in result.output
        assert "2401.00001v1" in result.output

    def test_json_output(self, tmp_path: Path) -> None:
        repo = _setup_repo(tmp_path)
        _seed_db(tmp_path)
        result = CliRunner().invoke(
            cli,
            ["search", "retrieval", "--config", str(repo / ".reporadar.yml"), "--format", "json"],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert any(p["arxiv_id"] == "2401.00001v1" for p in data)
        assert all({"arxiv_id", "title", "search_score"} <= set(p) for p in data)

    def test_no_match_message(self, tmp_path: Path) -> None:
        repo = _setup_repo(tmp_path)
        _seed_db(tmp_path)
        result = CliRunner().invoke(
            cli, ["search", "zzzznomatchxyz", "--config", str(repo / ".reporadar.yml")]
        )
        assert result.exit_code == 0
        assert "No matches" in result.output

    def test_missing_db_errors(self, tmp_path: Path) -> None:
        repo = _setup_repo(tmp_path)  # no _seed_db → no papers.db
        result = CliRunner().invoke(cli, ["search", "x", "--config", str(repo / ".reporadar.yml")])
        assert result.exit_code == 1

    def test_rejects_non_positive_limit(self, tmp_path: Path) -> None:
        repo = _setup_repo(tmp_path)
        _seed_db(tmp_path)
        result = CliRunner().invoke(
            cli, ["search", "retrieval", "--config", str(repo / ".reporadar.yml"), "-n", "0"]
        )
        assert result.exit_code == 2  # click IntRange rejects 0 as a usage error

    def test_semantic_without_embeddings_errors(self, tmp_path: Path) -> None:
        repo = _setup_repo(tmp_path)
        _seed_db(tmp_path)
        with patch("reporadar.embeddings.EMBEDDINGS_AVAILABLE", False):
            result = CliRunner().invoke(
                cli, ["search", "x", "--config", str(repo / ".reporadar.yml"), "--semantic"]
            )
        assert result.exit_code == 1
        assert "embeddings extra" in result.output


class TestDedupId:
    def test_version_strips_only_arxiv_ids(self) -> None:
        from reporadar.cli import _dedup_id

        assert _dedup_id("2401.12345v3") == "2401.12345"
        assert _dedup_id("2401.12345") == "2401.12345"
        # a 'v' inside a DBLP key must NOT be treated as a version suffix
        assert _dedup_id("dblp:conf/vldb/Smith25") == "dblp:conf/vldb/Smith25"


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


class TestFoundational:
    def test_sets_relevance_alltime_and_drops_recency(self) -> None:
        arxiv, ranking = _apply_foundational(
            ArxivConfig(sort_by="submitted", lookback_days=14),
            RankingConfig(w_recency=0.3, w_keyword=1.0, w_category=0.5),
        )
        assert arxiv.sort_by == "relevance"
        assert arxiv.lookback_days == _FOUNDATIONAL_LOOKBACK
        assert ranking.w_recency == 0.0
        # other ranking weights are preserved
        assert ranking.w_keyword == 1.0
        assert ranking.w_category == 0.5

    def test_does_not_mutate_inputs(self) -> None:
        arxiv_in = ArxivConfig(sort_by="submitted", lookback_days=14)
        ranking_in = RankingConfig(w_recency=0.3)
        _apply_foundational(arxiv_in, ranking_in)
        assert arxiv_in.sort_by == "submitted" and arxiv_in.lookback_days == 14
        assert ranking_in.w_recency == 0.3


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

    def test_suggests_a_domain_source_for_a_bio_repo(self, tmp_path: Path) -> None:
        (tmp_path / "README.md").write_text(
            "# scRNA tooling\n\nSingle-cell RNA sequencing and genome analysis.\n",
            encoding="utf-8",
        )
        (tmp_path / "requirements.txt").write_text("scanpy\nanndata\n", encoding="utf-8")
        config_file = tmp_path / ".reporadar.yml"
        config_file.write_text(f"repo_path: {tmp_path}\nsources: [arxiv]\n", encoding="utf-8")

        result = CliRunner().invoke(cli, ["profile", "--config", str(config_file)])

        assert result.exit_code == 0
        assert "Consider adding 'biorxiv'" in result.output
        # A suggestion, not an activation: the config is untouched.
        assert "biorxiv" not in config_file.read_text(encoding="utf-8")

    def test_no_suggestion_when_the_source_is_already_enabled(self, tmp_path: Path) -> None:
        (tmp_path / "README.md").write_text(
            "# scRNA tooling\n\nSingle-cell RNA sequencing and genome analysis.\n",
            encoding="utf-8",
        )
        (tmp_path / "requirements.txt").write_text("scanpy\nanndata\n", encoding="utf-8")
        config_file = tmp_path / ".reporadar.yml"
        config_file.write_text(
            f"repo_path: {tmp_path}\nsources: [arxiv, biorxiv]\n", encoding="utf-8"
        )

        result = CliRunner().invoke(cli, ["profile", "--config", str(config_file)])

        assert result.exit_code == 0
        assert "Consider adding" not in result.output


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
    @patch("reporadar.citations.fetch_references")
    def test_citation_proximity_wiring(
        self, mock_refs: MagicMock, mock_collect: MagicMock, tmp_path: Path
    ) -> None:
        # Pins the versioned(citing)-vs-base(cited) id contract end-to-end through
        # the update -> fetch_references -> find_citation_links -> save_citations chain.
        repo = _setup_repo(tmp_path)
        cfg_path = repo / ".reporadar.yml"
        cfg_path.write_text(
            cfg_path.read_text(encoding="utf-8").replace(
                "  w_recency: 0.3\n", "  w_recency: 0.3\n  w_citation_proximity: 5.0\n"
            ),
            encoding="utf-8",
        )
        db = repo / ".reporadar" / "papers.db"
        db.parent.mkdir(parents=True, exist_ok=True)
        with PaperStore(db) as store:
            store.star_paper("2401.00099v1")  # the seed

        mock_collect.return_value = [
            {
                "arxiv_id": "2402.00001v1",
                "title": "Extends It",
                "authors": ["A"],
                "abstract": "builds on prior work",
                "categories": ["cs.CL"],
                "published": datetime.now(UTC).isoformat(),
                "updated": None,
                "url": "http://arxiv.org/abs/2402.00001v1",
                "pdf_url": None,
                "matched_query": "all:test",
            }
        ]
        mock_refs.return_value = {"2402.00001v1": ["2401.00099"]}

        result = CliRunner().invoke(cli, ["update", "--config", str(cfg_path)])

        assert result.exit_code == 0
        assert "cite work you starred" in result.output
        with PaperStore(db) as store:
            assert store.get_citations_for(["2402.00001v1"]) == {"2402.00001v1": ["2401.00099"]}

    @patch("reporadar.cli.collect_papers")
    @patch("reporadar.sources.s2_recommendations.fetch_recommendations")
    def test_recommendations_wiring(
        self, mock_recs: MagicMock, mock_collect: MagicMock, tmp_path: Path
    ) -> None:
        # Pins the seed contract: stars/high ratings are positives, low ratings are
        # negatives, and a low rating beats an implicit star for the same paper.
        repo = _setup_repo(tmp_path)
        cfg_path = repo / ".reporadar.yml"
        cfg_path.write_text(
            cfg_path.read_text(encoding="utf-8") + "recommendations:\n  enabled: true\n",
            encoding="utf-8",
        )
        db = repo / ".reporadar" / "papers.db"
        db.parent.mkdir(parents=True, exist_ok=True)
        with PaperStore(db) as store:
            store.save_rating("2401.00010v1", 5)  # positive
            store.save_rating("2401.00011v1", 1)  # negative
            store.star_paper("2401.00012v1")  # positive (starred)
            store.star_paper("2401.00011v1")  # starred AND disliked -> negative wins

        mock_collect.return_value = [
            {
                "arxiv_id": "2402.00001v1",
                "title": "Fetched",
                "authors": ["A"],
                "abstract": "retrieval",
                "categories": ["cs.CL"],
                "published": datetime.now(UTC).isoformat(),
                "updated": None,
                "url": "http://arxiv.org/abs/2402.00001v1",
                "pdf_url": None,
                "matched_query": "all:test",
            }
        ]
        mock_recs.return_value = [
            {
                "arxiv_id": "2403.00002v1",
                "title": "Recommended",
                "authors": ["B"],
                "abstract": "retrieval augmented generation",
                "categories": [],
                "published": datetime.now(UTC).isoformat(),
                "updated": None,
                "url": "http://arxiv.org/abs/2403.00002v1",
                "pdf_url": None,
                "matched_query": "recommendation",
            }
        ]

        result = CliRunner().invoke(cli, ["update", "--config", str(cfg_path)])

        assert result.exit_code == 0
        mock_recs.assert_called_once()
        positives, negatives = mock_recs.call_args[0][0], mock_recs.call_args[0][1]
        assert "2401.00010v1" in positives and "2401.00012v1" in positives
        assert "2401.00011v1" in negatives
        assert "2401.00011v1" not in positives  # low rating beats the star
        # The recommended paper is stored and scored alongside the fetched one.
        with PaperStore(db) as store:
            assert store.get_paper("2403.00002v1") is not None

    @patch("reporadar.cli.collect_papers")
    @patch("reporadar.specter.fetch_specter_vectors")
    def test_specter_wiring(
        self, mock_vectors: MagicMock, mock_collect: MagicMock, tmp_path: Path
    ) -> None:
        import numpy as np

        from reporadar.specter import SPECTER_DIM

        repo = _setup_repo(tmp_path)
        cfg_path = repo / ".reporadar.yml"
        cfg_path.write_text(
            cfg_path.read_text(encoding="utf-8").replace(
                "  w_recency: 0.3\n", "  w_recency: 0.3\n  w_specter: 5.0\n"
            ),
            encoding="utf-8",
        )
        db = repo / ".reporadar" / "papers.db"
        db.parent.mkdir(parents=True, exist_ok=True)
        seed_paper = {
            "arxiv_id": "2401.00050v1",
            "title": "Seed",
            "authors": ["A"],
            "abstract": "seed",
            "categories": ["cs.CL"],
            "published": datetime.now(UTC).isoformat(),
            "updated": None,
            "url": "http://arxiv.org/abs/2401.00050v1",
            "pdf_url": None,
        }
        with PaperStore(db) as store:
            store.upsert_paper(seed_paper)
            store.star_paper("2401.00050v1")  # the liked paper -> query centroid

        def _vec(first: float, second: float) -> np.ndarray:
            out = np.zeros(SPECTER_DIM, dtype=np.float32)
            out[0], out[1] = first, second
            return out

        # At least _MIN_POOL papers with a real spread — below that the component
        # is intentionally dropped rather than amplifying noise.
        near = {**seed_paper, "arxiv_id": "2402.00001v1", "title": "Near"}
        mid = {**seed_paper, "arxiv_id": "2402.00003v1", "title": "Mid"}
        far = {**seed_paper, "arxiv_id": "2402.00002v1", "title": "Far"}
        mock_collect.return_value = [near, mid, far]
        shapes = {
            "2402.00001v1": _vec(1.0, 0.0),
            "2402.00003v1": _vec(1.0, 0.6),
            "2402.00002v1": _vec(0.0, 1.0),
        }
        mock_vectors.side_effect = lambda ids, api_key=None: {
            aid: shapes.get(aid, _vec(1.0, 0.0)) for aid in ids
        }

        result = CliRunner().invoke(cli, ["update", "--config", str(cfg_path)])

        assert result.exit_code == 0
        assert "SPECTER2" in result.output
        with PaperStore(db) as store:
            run_id = store.get_last_run()["run_id"]
            by_id = {s["arxiv_id"]: s for s in store.get_scores_for_run(run_id)}
            # The paper aligned with the starred seed scores 1.0; the other 0.0.
            assert by_id["2402.00001v1"]["specter_score"] == 1.0
            assert by_id["2402.00002v1"]["specter_score"] == 0.0

    @patch("reporadar.cli.collect_papers")
    def test_community_wiring_uses_cached_upvotes(
        self, mock_collect: MagicMock, tmp_path: Path
    ) -> None:
        repo = _setup_repo(tmp_path)
        cfg_path = repo / ".reporadar.yml"
        # _setup_repo already sets enrichment.provider to "off", which is the point:
        # ranking reads upvotes cached by an *earlier* run, since enrichment (stage 9)
        # happens after ranking (stage 8).
        cfg_path.write_text(
            cfg_path.read_text(encoding="utf-8").replace(
                "  w_recency: 0.3\n", "  w_recency: 0.3\n  w_community: 5.0\n"
            ),
            encoding="utf-8",
        )
        db = repo / ".reporadar" / "papers.db"
        db.parent.mkdir(parents=True, exist_ok=True)
        popular, quiet, unseen = (
            _community_paper("2402.00001v1"),
            _community_paper("2402.00002v1"),
            _community_paper("2402.00003v1"),
        )
        with PaperStore(db) as store:
            for paper in (popular, quiet):
                store.upsert_paper(paper)
            store.save_enrichments(
                {
                    "2402.00001v1": {"arxiv_id": "2402.00001v1", "upvotes": 200},
                    "2402.00002v1": {"arxiv_id": "2402.00002v1", "upvotes": 2},
                }
            )
        mock_collect.return_value = [popular, quiet, unseen]

        result = CliRunner().invoke(cli, ["update", "--config", str(cfg_path)])

        assert result.exit_code == 0
        assert "Community signal" in result.output
        with PaperStore(db) as store:
            run_id = store.get_last_run()["run_id"]
            by_id = {s["arxiv_id"]: s for s in store.get_scores_for_run(run_id)}
        assert by_id["2402.00001v1"]["community_score"] == 1.0  # the run's leader
        assert 0.0 < by_id["2402.00002v1"]["community_score"] < 1.0
        # Never enriched -> absent signal, not a zero, so it outranks the paper
        # HF *did* see but that barely anyone upvoted.
        assert by_id["2402.00003v1"]["community_score"] is None
        assert by_id["2402.00003v1"]["score_total"] > by_id["2402.00002v1"]["score_total"]

    @patch("reporadar.cli.collect_papers")
    def test_feedback_tuning_preserves_non_learned_weights(
        self, mock_collect: MagicMock, tmp_path: Path
    ) -> None:
        """Feedback tuning must not silently disable the optional components.

        The weight rebuild used to enumerate fields by hand, so each newly added
        component was dropped exactly when ratings existed to tune on.

        Asserts on *score ordering*, not on the recorded ``community_score``: that
        value is stored whenever the signal exists, regardless of the weight, so an
        assertion on it would pass with the bug reinstated (verified — reverting to
        the hand-written rebuild leaves such an assertion green).
        """
        repo = _setup_repo(tmp_path)
        cfg_path = repo / ".reporadar.yml"
        cfg_path.write_text(
            cfg_path.read_text(encoding="utf-8").replace(
                "  w_recency: 0.3\n",
                "  w_recency: 0.3\n  w_community: 5.0\n"
                "feedback:\n  enabled: true\n  min_ratings: 2\n",
            ),
            encoding="utf-8",
        )
        db = repo / ".reporadar" / "papers.db"
        db.parent.mkdir(parents=True, exist_ok=True)
        upvoted, quiet = _community_paper("2402.00001v1"), _community_paper("2402.00002v1")
        with PaperStore(db) as store:
            store.upsert_paper(upvoted)
            store.upsert_paper(quiet)
            # Two enriched papers with different counts: the weight has to be in the
            # weighted sum for their totals to differ at all.
            store.save_enrichments(
                {
                    "2402.00001v1": {"arxiv_id": "2402.00001v1", "upvotes": 500},
                    "2402.00002v1": {"arxiv_id": "2402.00002v1", "upvotes": 1},
                }
            )
            # Enough ratings to trip feedback.min_ratings and rebuild the config.
            for i, rating in enumerate((5, 1, 4), start=10):
                aid = f"2401.000{i}v1"
                store.upsert_paper(_community_paper(aid))
                prev_run = store.record_run(["q"], 1, 0)
                store.save_scores(
                    prev_run, [{"arxiv_id": aid, "score_total": 0.5, "keyword_score": 0.5}]
                )
                store.save_rating(aid, rating)
        mock_collect.return_value = [upvoted, quiet]

        result = CliRunner().invoke(cli, ["update", "--config", str(cfg_path), "-v"])

        assert result.exit_code == 0
        # -v proves the rebuild actually happened; without it this test would pass
        # vacuously whenever feedback tuning silently bailed out.
        assert "adjusted ranking weights" in result.output
        with PaperStore(db) as store:
            run_id = store.get_last_run()["run_id"]
            by_id = {s["arxiv_id"]: s for s in store.get_scores_for_run(run_id)}
        # Same title/abstract/date, so every other component ties: the totals can
        # only differ if w_community survived the feedback rebuild.
        assert by_id["2402.00001v1"]["community_score"] == 1.0
        assert by_id["2402.00002v1"]["community_score"] < 1.0
        assert by_id["2402.00001v1"]["score_total"] > by_id["2402.00002v1"]["score_total"]

    @patch("reporadar.signals.integrity.fetch_comments")
    @patch("reporadar.cli.collect_papers")
    def test_integrity_demotes_and_records_a_withdrawn_paper(
        self, mock_collect: MagicMock, mock_comments: MagicMock, tmp_path: Path
    ) -> None:
        repo = _setup_repo(tmp_path)
        cfg_path = repo / ".reporadar.yml"
        cfg_path.write_text(
            cfg_path.read_text(encoding="utf-8").replace(
                "signals:\n  integrity: false\n", "signals:\n  integrity: true\n"
            ),
            encoding="utf-8",
        )
        db = repo / ".reporadar" / "papers.db"
        db.parent.mkdir(parents=True, exist_ok=True)
        withdrawn, clean = _community_paper("2607.00001v1"), _community_paper("2607.00002v1")
        mock_collect.return_value = [withdrawn, clean]
        mock_comments.return_value = {
            "2607.00001v1": "Withdrawn by the authors due to an error in Theorem 2",
            "2607.00002v1": "10 pages, 3 figures",
        }

        result = CliRunner().invoke(cli, ["update", "--config", str(cfg_path)])

        assert result.exit_code == 0
        assert "withdrawn by their authors" in result.output.lower()
        with PaperStore(db) as store:
            run_id = store.get_last_run()["run_id"]
            by_id = {s["arxiv_id"]: s for s in store.get_scores_for_run(run_id)}
            signals = store.get_signals(["2607.00001v1", "2607.00002v1"], "withdrawn")
        # Same title/abstract/date, so only the penalty can separate them.
        assert by_id["2607.00001v1"]["score_total"] < by_id["2607.00002v1"]["score_total"]
        assert signals["2607.00001v1"]["value"] == "comment"
        # The clean paper is recorded as checked-with-no-notice (a NULL value), so the
        # next run can skip it — that is not the same as being flagged.
        assert signals["2607.00002v1"]["value"] is None

    @patch("reporadar.signals.integrity.fetch_comments")
    @patch("reporadar.cli.collect_papers")
    def test_integrity_failure_does_not_break_the_run(
        self, mock_collect: MagicMock, mock_comments: MagicMock, tmp_path: Path
    ) -> None:
        repo = _setup_repo(tmp_path)
        cfg_path = repo / ".reporadar.yml"
        cfg_path.write_text(
            cfg_path.read_text(encoding="utf-8").replace(
                "signals:\n  integrity: false\n", "signals:\n  integrity: true\n"
            ),
            encoding="utf-8",
        )
        mock_collect.return_value = [_community_paper("2607.00001v1")]
        mock_comments.side_effect = RuntimeError("arXiv unreachable")

        result = CliRunner().invoke(cli, ["update", "--config", str(cfg_path)])

        assert result.exit_code == 0
        assert "Integrity check failed" in result.output

    @patch("reporadar.signals.integrity.fetch_comments")
    @patch("reporadar.cli.collect_papers")
    def test_an_arxiv_outage_is_reported_not_hidden(
        self, mock_collect: MagicMock, mock_comments: MagicMock, tmp_path: Path
    ) -> None:
        """fetch_comments swallows per-batch failures and returns what it got.

        So an outage returns {} without raising, and the run would otherwise print a
        clean "no withdrawn papers" while having checked nothing.
        """
        repo = _setup_repo(tmp_path)
        cfg_path = repo / ".reporadar.yml"
        cfg_path.write_text(
            cfg_path.read_text(encoding="utf-8").replace(
                "signals:\n  integrity: false\n", "signals:\n  integrity: true\n"
            ),
            encoding="utf-8",
        )
        mock_collect.return_value = [_community_paper("2607.00001v1")]
        mock_comments.return_value = {}  # what an outage looks like from here

        result = CliRunner().invoke(cli, ["update", "--config", str(cfg_path)])

        assert result.exit_code == 0
        assert "could not reach arXiv" in result.output

    @patch("reporadar.signals.integrity.fetch_comments")
    @patch("reporadar.cli.collect_papers")
    def test_local_fallback_flags_a_paper_arxiv_cannot_resolve(
        self, mock_collect: MagicMock, mock_comments: MagicMock, tmp_path: Path
    ) -> None:
        # A notice in the stored abstract needs no network, so it must cover papers
        # from non-arXiv sources too — whose synthetic ids the API cannot resolve.
        repo = _setup_repo(tmp_path)
        cfg_path = repo / ".reporadar.yml"
        cfg_path.write_text(
            cfg_path.read_text(encoding="utf-8").replace(
                "signals:\n  integrity: false\n", "signals:\n  integrity: true\n"
            ),
            encoding="utf-8",
        )
        paper = _community_paper("ss:987654")
        paper["abstract"] = "This paper has been withdrawn by the authors."
        mock_collect.return_value = [paper]
        mock_comments.return_value = {}

        result = CliRunner().invoke(cli, ["update", "--config", str(cfg_path)])

        assert result.exit_code == 0
        assert "withdrawn by their authors" in result.output.lower()
        # A synthetic id must not be queued for a lookup it can never satisfy.
        mock_comments.assert_not_called()

    @patch("reporadar.signals.integrity.fetch_comments")
    @patch("reporadar.cli.collect_papers")
    def test_integrity_skips_recently_checked_papers(
        self, mock_collect: MagicMock, mock_comments: MagicMock, tmp_path: Path
    ) -> None:
        """The recheck must be bounded by checked_at, not re-run over everything.

        arXiv wants 3s between requests, so an unbounded pass over a --foundational
        store would spend minutes per run on a signal that fires for under 1% of
        papers. A stored (clean) result from today means no lookup at all.
        """
        repo = _setup_repo(tmp_path)
        cfg_path = repo / ".reporadar.yml"
        cfg_path.write_text(
            cfg_path.read_text(encoding="utf-8").replace(
                "signals:\n  integrity: false\n", "signals:\n  integrity: true\n"
            ),
            encoding="utf-8",
        )
        db = repo / ".reporadar" / "papers.db"
        db.parent.mkdir(parents=True, exist_ok=True)
        paper = _community_paper("2607.00001v1")
        with PaperStore(db) as store:
            store.upsert_paper(paper)
            store.save_signals([("2607.00001v1", "withdrawn", None, None)])  # checked, clean
        mock_collect.return_value = [paper]

        result = CliRunner().invoke(cli, ["update", "--config", str(cfg_path)])

        assert result.exit_code == 0
        mock_comments.assert_not_called()
        assert "0 looked up" in result.output

    @patch("reporadar.signals.integrity.fetch_comments")
    @patch("reporadar.cli.collect_papers")
    def test_a_previously_flagged_paper_stays_demoted_without_a_refetch(
        self, mock_collect: MagicMock, mock_comments: MagicMock, tmp_path: Path
    ) -> None:
        repo = _setup_repo(tmp_path)
        cfg_path = repo / ".reporadar.yml"
        cfg_path.write_text(
            cfg_path.read_text(encoding="utf-8").replace(
                "signals:\n  integrity: false\n", "signals:\n  integrity: true\n"
            ),
            encoding="utf-8",
        )
        db = repo / ".reporadar" / "papers.db"
        db.parent.mkdir(parents=True, exist_ok=True)
        flagged, clean = _community_paper("2607.00001v1"), _community_paper("2607.00002v1")
        with PaperStore(db) as store:
            store.upsert_paper(flagged)
            store.upsert_paper(clean)
            store.save_signals(
                [
                    ("2607.00001v1", "withdrawn", "comment", None),
                    ("2607.00002v1", "withdrawn", None, None),
                ]
            )
        mock_collect.return_value = [flagged, clean]

        result = CliRunner().invoke(cli, ["update", "--config", str(cfg_path)])

        assert result.exit_code == 0
        mock_comments.assert_not_called()
        with PaperStore(db) as store:
            run_id = store.get_last_run()["run_id"]
            by_id = {s["arxiv_id"]: s for s in store.get_scores_for_run(run_id)}
        assert by_id["2607.00001v1"]["score_total"] < by_id["2607.00002v1"]["score_total"]

    @patch("reporadar.signals.hn.fetch_attention")
    @patch("reporadar.cli.collect_papers")
    def test_hackernews_wiring(
        self, mock_collect: MagicMock, mock_attention: MagicMock, tmp_path: Path
    ) -> None:
        repo = _setup_repo(tmp_path)
        cfg_path = repo / ".reporadar.yml"
        cfg_path.write_text(
            cfg_path.read_text(encoding="utf-8")
            .replace("  w_recency: 0.3\n", "  w_recency: 0.3\n  w_attention: 5.0\n")
            .replace("  integrity: false\n", "  integrity: false\n  hackernews: true\n"),
            encoding="utf-8",
        )
        db = repo / ".reporadar" / "papers.db"
        db.parent.mkdir(parents=True, exist_ok=True)
        discussed, quiet = _community_paper("2607.00001v1"), _community_paper("2607.00002v1")
        mock_collect.return_value = [discussed, quiet]
        mock_attention.return_value = {
            "2607.00001v1": {
                "points": 1351,
                "comments": 1056,
                "story_url": "https://news.ycombinator.com/item?id=42823568",
                "title": "A Story",
                "submissions": 1,
            }
        }

        result = CliRunner().invoke(cli, ["update", "--config", str(cfg_path)])

        assert result.exit_code == 0
        assert "Hacker News" in result.output
        with PaperStore(db) as store:
            run_id = store.get_last_run()["run_id"]
            by_id = {s["arxiv_id"]: s for s in store.get_scores_for_run(run_id)}
            signals = store.get_signals(["2607.00001v1"], "hn")
        assert by_id["2607.00001v1"]["attention_score"] == 1.0
        # Never discussed -> absent signal, so it must not be scored 0 and sink.
        assert by_id["2607.00002v1"]["attention_score"] is None
        assert by_id["2607.00001v1"]["score_total"] > by_id["2607.00002v1"]["score_total"]
        assert signals["2607.00001v1"]["value"] == "1351"
        assert signals["2607.00001v1"]["detail"].endswith("42823568")

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
    def test_explain_flag(self, mock_collect: MagicMock, tmp_path: Path) -> None:
        repo = _setup_repo(tmp_path)
        now = datetime.now(UTC).isoformat()
        mock_collect.return_value = [
            {
                "arxiv_id": "2401.99999v1",
                "title": "Mock Paper on Retrieval",
                "authors": ["Test Author"],
                "abstract": "A test abstract about retrieval and transformers.",
                "categories": ["cs.CL"],
                "published": now,
                "updated": None,
                "url": "http://arxiv.org/abs/2401.99999v1",
                "pdf_url": None,
                "matched_query": "all:test",
            },
        ]

        runner = CliRunner()
        result = runner.invoke(
            cli, ["update", "--config", str(repo / ".reporadar.yml"), "--explain"]
        )

        assert result.exit_code == 0
        assert "Score explanations:" in result.output
        assert "keyword" in result.output
        assert "category" in result.output
        assert "recency" in result.output

    @patch("reporadar.cli.collect_papers")
    def test_explain_uses_the_feedback_tuned_weights(
        self, mock_collect: MagicMock, tmp_path: Path
    ) -> None:
        # With feedback on, the scores come from adjusted weights; explaining them
        # with the file's weights printed contributions that didn't sum to the total.
        repo = _setup_repo(tmp_path)
        cfg_path = repo / ".reporadar.yml"
        cfg_path.write_text(
            cfg_path.read_text(encoding="utf-8").replace(
                "  w_recency: 0.3\n",
                "  w_recency: 0.3\nfeedback:\n  enabled: true\n  min_ratings: 2\n",
            ),
            encoding="utf-8",
        )
        db = repo / ".reporadar" / "papers.db"
        db.parent.mkdir(parents=True, exist_ok=True)
        target = _community_paper("2402.00001v1")
        with PaperStore(db) as store:
            store.upsert_paper(target)
            for i, rating in enumerate((5, 1, 4), start=10):
                aid = f"2401.000{i}v1"
                store.upsert_paper(_community_paper(aid))
                prev_run = store.record_run(["q"], 1, 0)
                store.save_scores(
                    prev_run, [{"arxiv_id": aid, "score_total": 0.5, "keyword_score": 0.5}]
                )
                store.save_rating(aid, rating)
        mock_collect.return_value = [target]

        result = CliRunner().invoke(cli, ["update", "--config", str(cfg_path), "--explain", "-v"])

        assert result.exit_code == 0
        assert "adjusted ranking weights" in result.output  # tuning really ran
        # The untuned weights are 1.00/0.50/0.30; the tuned ones differ, so the
        # explanation must not be printing the config's defaults.
        explanation = result.output[result.output.index("Score explanations:") :]
        assert "1.00 *" not in explanation

    @patch("reporadar.cli.collect_papers")
    def test_score_distribution_shown(self, mock_collect: MagicMock, tmp_path: Path) -> None:
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
        assert "Score stats:" in result.output
        assert "mean=" in result.output
        assert "median=" in result.output

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


class TestQueriesCommand:
    def test_shows_queries(self, tmp_path: Path) -> None:
        repo = _setup_repo(tmp_path)

        runner = CliRunner()
        result = runner.invoke(cli, ["queries", "--config", str(repo / ".reporadar.yml")])

        assert result.exit_code == 0
        assert "queries" in result.output.lower()

    def test_no_queries_message(self, tmp_path: Path) -> None:
        config_file = tmp_path / ".reporadar.yml"
        config_file.write_text(
            f"repo_path: {tmp_path}\narxiv:\n  categories: []\nqueries:\n  seed: []\n",
            encoding="utf-8",
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["queries", "--config", str(config_file)])

        assert result.exit_code == 0
        assert "No queries generated" in result.output


class TestGhIssuesCommand:
    @patch("reporadar.gh_issues.check_gh_available", return_value=True)
    @patch("reporadar.gh_issues.create_issue")
    def test_dry_run_shows_preview(
        self, mock_create: MagicMock, mock_gh: MagicMock, tmp_path: Path
    ) -> None:
        repo = _setup_repo(tmp_path)
        _seed_db(repo)

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["gh-issues", "--config", str(repo / ".reporadar.yml"), "--dry-run"],
        )

        assert result.exit_code == 0
        assert "DRY RUN" in result.output
        mock_create.assert_not_called()

    @patch("reporadar.gh_issues.check_gh_available", return_value=True)
    @patch("reporadar.gh_issues.create_issue")
    def test_skips_already_exported(
        self, mock_create: MagicMock, mock_gh: MagicMock, tmp_path: Path
    ) -> None:
        repo = _setup_repo(tmp_path)
        _seed_db(repo)

        # Mark all papers as already exported
        db_path = repo / ".reporadar" / "papers.db"
        with PaperStore(db_path) as store:
            store.record_export("2401.00001v1", "github_issue", "url1")
            store.record_export("2401.00002v1", "github_issue", "url2")

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["gh-issues", "--config", str(repo / ".reporadar.yml")],
        )

        assert result.exit_code == 0
        assert "already been exported" in result.output
        mock_create.assert_not_called()

    @patch("reporadar.gh_issues.check_gh_available", return_value=False)
    def test_gh_not_available(self, mock_gh: MagicMock, tmp_path: Path) -> None:
        repo = _setup_repo(tmp_path)
        _seed_db(repo)

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["gh-issues", "--config", str(repo / ".reporadar.yml")],
        )

        assert result.exit_code == 1
        assert "gh" in result.output.lower()


class TestNotifyCommand:
    @patch("reporadar.notify.dispatch_notification", return_value=True)
    def test_success(self, mock_dispatch: MagicMock, tmp_path: Path) -> None:
        repo = _setup_repo(tmp_path)
        _seed_db(repo)

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["notify", "--config", str(repo / ".reporadar.yml"), "--channel", "shell"],
        )

        assert result.exit_code == 0
        assert "Notification sent" in result.output

    def test_no_database(self, tmp_path: Path) -> None:
        repo = _setup_repo(tmp_path)

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["notify", "--config", str(repo / ".reporadar.yml"), "--channel", "shell"],
        )

        assert result.exit_code == 1
        assert "No database found" in result.output

    def test_no_runs(self, tmp_path: Path) -> None:
        repo = _setup_repo(tmp_path)
        db_path = repo / ".reporadar" / "papers.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with PaperStore(db_path):
            pass

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["notify", "--config", str(repo / ".reporadar.yml"), "--channel", "shell"],
        )

        assert result.exit_code == 1
        assert "No runs found" in result.output

    @patch("reporadar.notify.dispatch_notification", return_value=False)
    def test_failure_exits_1(self, mock_dispatch: MagicMock, tmp_path: Path) -> None:
        repo = _setup_repo(tmp_path)
        _seed_db(repo)

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["notify", "--config", str(repo / ".reporadar.yml"), "--channel", "shell"],
        )

        assert result.exit_code == 1
        assert "failed" in result.output


class TestScheduleCommand:
    @patch("reporadar.scheduler.add_schedule", return_value=True)
    def test_add_success(self, mock_add: MagicMock, tmp_path: Path) -> None:
        repo = _setup_repo(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["schedule", "--config", str(repo / ".reporadar.yml"), "--cron", "0 9 * * 1"],
        )
        assert result.exit_code == 0
        assert "Schedule registered" in result.output

    @patch("reporadar.scheduler.add_schedule", return_value=False)
    def test_add_failure(self, mock_add: MagicMock, tmp_path: Path) -> None:
        repo = _setup_repo(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["schedule", "--config", str(repo / ".reporadar.yml"), "--cron", "0 9 * * 1"],
        )
        assert result.exit_code == 1
        assert "Failed" in result.output

    @patch("reporadar.scheduler.list_schedules", return_value=[])
    def test_list_empty(self, mock_list: MagicMock) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["schedule", "--list"])
        assert result.exit_code == 0
        assert "No schedules" in result.output

    @patch("reporadar.scheduler.list_schedules")
    def test_list_with_tasks(self, mock_list: MagicMock) -> None:
        from reporadar.scheduler import ScheduledTask

        mock_list.return_value = [
            ScheduledTask(cron_expr="0 9 * * 1", command="rr update", platform="unix")
        ]
        runner = CliRunner()
        result = runner.invoke(cli, ["schedule", "--list"])
        assert result.exit_code == 0
        assert "0 9 * * 1" in result.output
        assert "unix" in result.output

    @patch("reporadar.scheduler.remove_schedule", return_value=True)
    def test_remove_success(self, mock_rm: MagicMock) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["schedule", "--remove"])
        assert result.exit_code == 0
        assert "Schedule removed" in result.output

    @patch("reporadar.scheduler.remove_schedule", return_value=False)
    def test_remove_not_found(self, mock_rm: MagicMock) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["schedule", "--remove"])
        assert result.exit_code == 0
        assert "No schedule found" in result.output

    def test_no_option_error(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["schedule"])
        assert result.exit_code == 1

    def test_invalid_cron(self, tmp_path: Path) -> None:
        repo = _setup_repo(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["schedule", "--config", str(repo / ".reporadar.yml"), "--cron", "bad"],
        )
        assert result.exit_code == 1
        assert "Invalid cron" in result.output


class TestWorkspaceCommands:
    @patch("reporadar.workspace.WORKSPACE_DIR")
    @patch("reporadar.workspace.WORKSPACE_DB")
    def test_init(self, mock_db: MagicMock, mock_dir: MagicMock, tmp_path: Path) -> None:
        mock_dir.__truediv__ = lambda self, x: tmp_path / x
        mock_db.__fspath__ = lambda self: str(tmp_path / "workspace.db")

        # Directly test with a custom db_path to avoid home directory side effects
        with (
            patch(
                "reporadar.workspace.open_workspace_store",
                return_value=PaperStore(tmp_path / "workspace.db"),
            ),
            patch("reporadar.workspace.ensure_workspace_dir", return_value=tmp_path),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["workspace", "init"])

        assert result.exit_code == 0
        assert "Workspace initialized" in result.output

    def test_add_list_remove(self, tmp_path: Path) -> None:
        ws_db = tmp_path / "workspace.db"
        repo_dir = tmp_path / "myrepo"
        repo_dir.mkdir()

        with patch(
            "reporadar.workspace.open_workspace_store",
            return_value=PaperStore(ws_db),
        ):
            runner = CliRunner()

            # Add
            result = runner.invoke(
                cli,
                ["workspace", "add", "myrepo", "--path", str(repo_dir)],
            )
            assert result.exit_code == 0
            assert "Added repo" in result.output

        with patch(
            "reporadar.workspace.open_workspace_store",
            return_value=PaperStore(ws_db),
        ):
            # List
            result = runner.invoke(cli, ["workspace", "list"])
            assert result.exit_code == 0
            assert "myrepo" in result.output

        with patch(
            "reporadar.workspace.open_workspace_store",
            return_value=PaperStore(ws_db),
        ):
            # Remove
            result = runner.invoke(cli, ["workspace", "remove", "myrepo"])
            assert result.exit_code == 0
            assert "Removed" in result.output

    def test_list_empty(self, tmp_path: Path) -> None:
        ws_db = tmp_path / "workspace.db"
        with patch(
            "reporadar.workspace.open_workspace_store",
            return_value=PaperStore(ws_db),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["workspace", "list"])
        assert result.exit_code == 0
        assert "No repos registered" in result.output

    @patch("reporadar.cli.collect_papers")
    def test_update_pipeline(self, mock_collect: MagicMock, tmp_path: Path) -> None:
        ws_db = tmp_path / "workspace.db"
        repo_dir = _setup_repo(tmp_path)

        mock_collect.return_value = [
            {
                "arxiv_id": "2401.99999v1",
                "title": "Mock Paper",
                "authors": ["Test"],
                "abstract": "Test abstract.",
                "categories": ["cs.CL"],
                "published": datetime.now(UTC).isoformat(),
                "updated": None,
                "url": "http://arxiv.org/abs/2401.99999v1",
                "pdf_url": None,
                "matched_query": "all:test",
            },
        ]

        store = PaperStore(ws_db)
        store.add_workspace_repo("testrepo", str(repo_dir), str(repo_dir / ".reporadar.yml"))
        store.close()

        with patch(
            "reporadar.workspace.open_workspace_store",
            return_value=PaperStore(ws_db),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["workspace", "update"])

        assert result.exit_code == 0

    def test_digest_no_runs(self, tmp_path: Path) -> None:
        ws_db = tmp_path / "workspace.db"
        with patch(
            "reporadar.workspace.open_workspace_store",
            return_value=PaperStore(ws_db),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["workspace", "digest"])
        assert result.exit_code == 1
        assert "No runs found" in result.output


class TestWatchCommand:
    @patch("reporadar.watcher.watch_loop")
    def test_basic_invocation(self, mock_loop: MagicMock, tmp_path: Path) -> None:
        repo = _setup_repo(tmp_path)
        mock_loop.side_effect = KeyboardInterrupt()

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["watch", "--config", str(repo / ".reporadar.yml"), "--interval", "1m"],
        )
        assert result.exit_code == 0
        assert "Watch stopped" in result.output

    def test_invalid_interval(self, tmp_path: Path) -> None:
        repo = _setup_repo(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["watch", "--config", str(repo / ".reporadar.yml"), "--interval", "bad"],
        )
        assert result.exit_code == 1
        assert "Invalid interval" in result.output


class TestFormatSize:
    def test_bytes(self) -> None:
        assert _format_size(500) == "500 B"

    def test_kilobytes(self) -> None:
        result = _format_size(2048)
        assert "KB" in result

    def test_megabytes(self) -> None:
        result = _format_size(5 * 1024 * 1024)
        assert "MB" in result


class TestEvalCommand:
    def _repo_with_ratings(self, tmp_path: Path, n: int = 24) -> Path:
        repo = _setup_repo(tmp_path)
        db = repo / ".reporadar" / "papers.db"
        db.parent.mkdir(parents=True, exist_ok=True)
        on_topic = "retrieval augmented generation with transformers"
        off_topic = "combinatorial scheduling heuristics on graphs"
        with PaperStore(db) as store:
            for i in range(n):
                arxiv_id = f"2607.{i:05d}v1"
                on = i % 2 == 0
                store.upsert_paper(
                    {
                        "arxiv_id": arxiv_id,
                        "title": f"Paper {i}",
                        "authors": ["A"],
                        "abstract": on_topic if on else off_topic,
                        "categories": ["cs.CL"],
                        "published": datetime.now(UTC).isoformat(),
                        "updated": None,
                        "url": f"http://arxiv.org/abs/{arxiv_id}",
                        "pdf_url": None,
                    }
                )
                store.save_rating(arxiv_id, 5 if on else 1)
        return repo

    def test_reports_metrics(self, tmp_path: Path) -> None:
        repo = self._repo_with_ratings(tmp_path)
        result = CliRunner().invoke(cli, ["eval", "--config", str(repo / ".reporadar.yml")])
        assert result.exit_code == 0
        assert "nDCG@10" in result.output
        assert "Judged papers: 24" in result.output

    def test_json_output_is_machine_readable(self, tmp_path: Path) -> None:
        repo = self._repo_with_ratings(tmp_path)
        result = CliRunner().invoke(
            cli, ["eval", "--config", str(repo / ".reporadar.yml"), "--format", "json"]
        )
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert {"ndcg@k", "precision@k", "recall@k", "mrr", "n_judged"} <= set(payload)
        # Internal keys must not leak into the machine-readable contract.
        assert not any(key.startswith("_") for key in payload)

    def test_no_ratings_explains_what_to_do(self, tmp_path: Path) -> None:
        repo = _setup_repo(tmp_path)
        _seed_db(tmp_path)  # papers and scores, but no ratings or stars
        result = CliRunner().invoke(cli, ["eval", "--config", str(repo / ".reporadar.yml")])
        assert result.exit_code == 1
        assert "rr rate" in result.output

    def test_missing_db_errors(self, tmp_path: Path) -> None:
        repo = _setup_repo(tmp_path)
        result = CliRunner().invoke(cli, ["eval", "--config", str(repo / ".reporadar.yml")])
        assert result.exit_code == 1
        assert "rr update" in result.output

    def test_compare_detects_a_regression(self, tmp_path: Path) -> None:
        repo = self._repo_with_ratings(tmp_path, n=30)
        good = repo / ".reporadar.yml"
        bad = repo / "bad.yml"
        bad.write_text(
            good.read_text(encoding="utf-8").replace("  w_keyword: 1.0\n", "  w_keyword: 0.0\n"),
            encoding="utf-8",
        )
        result = CliRunner().invoke(
            cli, ["eval", "--config", str(good), "--compare", str(good), str(bad)]
        )
        assert result.exit_code == 0
        assert "interval excludes zero" in result.output
        assert "A is better" in result.output

    def test_baseline_is_recorded_and_listed(self, tmp_path: Path) -> None:
        repo = self._repo_with_ratings(tmp_path)
        cfg_path = str(repo / ".reporadar.yml")
        recorded = CliRunner().invoke(
            cli, ["eval", "--config", cfg_path, "--baseline", "--label", "before"]
        )
        assert recorded.exit_code == 0
        assert "Recorded baseline #1 (before)" in recorded.output

        listed = CliRunner().invoke(cli, ["eval", "--config", cfg_path, "--history"])
        assert listed.exit_code == 0
        assert "before" in listed.output

    def test_history_without_snapshots_is_not_an_error(self, tmp_path: Path) -> None:
        repo = self._repo_with_ratings(tmp_path)
        result = CliRunner().invoke(
            cli, ["eval", "--config", str(repo / ".reporadar.yml"), "--history"]
        )
        assert result.exit_code == 0
        assert "No baselines recorded yet" in result.output

    def test_baseline_stores_the_weights_that_produced_it(self, tmp_path: Path) -> None:
        # A snapshot without its config cannot be compared to anything later.
        repo = self._repo_with_ratings(tmp_path)
        CliRunner().invoke(cli, ["eval", "--config", str(repo / ".reporadar.yml"), "--baseline"])
        with PaperStore(repo / ".reporadar" / "papers.db") as store:
            snapshot = store.get_metric_snapshots()[0]
        assert snapshot["config"]["w_keyword"] == 1.0
        assert snapshot["metrics"]["ndcg@k"] > 0


class TestEvalRegressionGate:
    """`--against` is what makes --baseline a CI gate rather than a diary."""

    def _repo(self, tmp_path: Path, n: int = 30) -> Path:
        repo = _setup_repo(tmp_path)
        db = repo / ".reporadar" / "papers.db"
        db.parent.mkdir(parents=True, exist_ok=True)
        on_topic = "retrieval augmented generation with transformers"
        off_topic = "combinatorial scheduling heuristics on graphs"
        with PaperStore(db) as store:
            for i in range(n):
                arxiv_id = f"2607.{i:05d}v1"
                on = i % 2 == 0
                store.upsert_paper(
                    {
                        "arxiv_id": arxiv_id,
                        "title": f"Paper {i}",
                        "authors": ["A"],
                        "abstract": on_topic if on else off_topic,
                        "categories": ["cs.CL"],
                        "published": datetime.now(UTC).isoformat(),
                        "updated": None,
                        "url": f"http://arxiv.org/abs/{arxiv_id}",
                        "pdf_url": None,
                    }
                )
                store.save_rating(arxiv_id, 5 if on else 1)
        return repo

    def test_a_regression_exits_nonzero(self, tmp_path: Path) -> None:
        # Exit status is the whole point: a CI job gates the build on it.
        repo = self._repo(tmp_path)
        good = repo / ".reporadar.yml"
        bad = repo / "bad.yml"
        bad.write_text(
            good.read_text(encoding="utf-8").replace("  w_keyword: 1.0\n", "  w_keyword: 0.0\n"),
            encoding="utf-8",
        )
        runner = CliRunner()
        runner.invoke(cli, ["eval", "--config", str(good), "--baseline", "--label", "good"])
        result = runner.invoke(cli, ["eval", "--config", str(bad), "--against", "latest"])
        assert result.exit_code == 1
        assert "REGRESSION" in result.output

    def test_an_unchanged_config_passes(self, tmp_path: Path) -> None:
        repo = self._repo(tmp_path)
        cfg_path = str(repo / ".reporadar.yml")
        runner = CliRunner()
        runner.invoke(cli, ["eval", "--config", cfg_path, "--baseline"])
        result = runner.invoke(cli, ["eval", "--config", cfg_path, "--against", "latest"])
        assert result.exit_code == 0
        assert "no regression" in result.output

    def test_against_a_specific_snapshot_id(self, tmp_path: Path) -> None:
        repo = self._repo(tmp_path)
        cfg_path = str(repo / ".reporadar.yml")
        runner = CliRunner()
        runner.invoke(cli, ["eval", "--config", cfg_path, "--baseline", "--label", "first"])
        result = runner.invoke(cli, ["eval", "--config", cfg_path, "--against", "1"])
        assert result.exit_code == 0
        assert "first" in result.output

    def test_missing_baseline_is_an_error_not_a_silent_pass(self, tmp_path: Path) -> None:
        # A CI job gating on exit 0 must not be told "fine" when nothing was compared.
        repo = self._repo(tmp_path)
        result = CliRunner().invoke(
            cli, ["eval", "--config", str(repo / ".reporadar.yml"), "--against", "latest"]
        )
        assert result.exit_code == 1
        assert "No recorded baseline" in result.output

    def test_compare_plus_against_warns_rather_than_ignoring(self, tmp_path: Path) -> None:
        repo = self._repo(tmp_path)
        cfg_path = str(repo / ".reporadar.yml")
        result = CliRunner().invoke(
            cli,
            ["eval", "--config", cfg_path, "--compare", cfg_path, cfg_path, "--against", "latest"],
        )
        assert result.exit_code == 0
        assert "do nothing with --compare" in result.output

    def test_history_shows_the_cutoff_each_snapshot_used(self, tmp_path: Path) -> None:
        # Snapshots taken at different k are not comparable; hiding k hides that.
        repo = self._repo(tmp_path)
        cfg_path = str(repo / ".reporadar.yml")
        runner = CliRunner()
        runner.invoke(cli, ["eval", "--config", cfg_path, "-k", "5", "--baseline"])
        result = runner.invoke(cli, ["eval", "--config", cfg_path, "--history"])
        assert result.exit_code == 0
        assert " k " in result.output or "k " in result.output.splitlines()[0]
