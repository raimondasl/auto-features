"""Tests for reporadar.config."""

from __future__ import annotations

from pathlib import Path

import pytest

from reporadar.config import (
    ArxivConfig,
    OutputConfig,
    QueriesConfig,
    RankingConfig,
    RepoRadarConfig,
    default_config_yaml,
    load_config,
    validate_config,
)


class TestLoadConfig:
    def test_load_full_config(self, tmp_path: Path) -> None:
        config_file = tmp_path / ".reporadar.yml"
        config_file.write_text(
            """\
repo_path: /some/repo

arxiv:
  categories: [cs.AI]
  max_results_per_query: 100
  lookback_days: 7

queries:
  seed:
    - "neural architecture search"
  exclude:
    - "survey"

ranking:
  w_keyword: 2.0
  w_category: 1.0
  w_recency: 0.5

output:
  digest_path: ./output/digest.md
  top_n: 10
""",
            encoding="utf-8",
        )

        cfg = load_config(config_file)

        assert cfg.repo_path == "/some/repo"
        assert cfg.arxiv.categories == ["cs.AI"]
        assert cfg.arxiv.max_results_per_query == 100
        assert cfg.arxiv.lookback_days == 7
        assert cfg.queries.seed == ["neural architecture search"]
        assert cfg.queries.exclude == ["survey"]
        assert cfg.ranking.w_keyword == 2.0
        assert cfg.output.digest_path == "./output/digest.md"
        assert cfg.output.top_n == 10

    def test_load_minimal_config(self, tmp_path: Path) -> None:
        config_file = tmp_path / ".reporadar.yml"
        config_file.write_text("repo_path: .\n", encoding="utf-8")

        cfg = load_config(config_file)

        assert cfg.repo_path == "."
        # All sub-configs should have defaults
        assert cfg.arxiv.categories == ["cs.LG", "cs.CL"]
        assert cfg.arxiv.max_results_per_query == 50
        assert cfg.queries.seed == []
        assert cfg.ranking.w_keyword == 1.0
        assert cfg.output.top_n == 15

    def test_load_empty_yaml(self, tmp_path: Path) -> None:
        config_file = tmp_path / ".reporadar.yml"
        config_file.write_text("", encoding="utf-8")

        cfg = load_config(config_file)

        # Should return all defaults
        assert isinstance(cfg, RepoRadarConfig)
        assert cfg.repo_path == "."

    def test_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent.yml")


class TestDefaultConfigYaml:
    def test_generates_valid_yaml(self) -> None:
        import yaml

        text = default_config_yaml()
        data = yaml.safe_load(text)

        assert data["repo_path"] == "."
        assert "arxiv" in data
        assert "queries" in data
        assert "ranking" in data
        assert "output" in data

    def test_round_trips_through_load(self, tmp_path: Path) -> None:
        config_file = tmp_path / ".reporadar.yml"
        config_file.write_text(default_config_yaml(), encoding="utf-8")

        cfg = load_config(config_file)

        assert cfg.arxiv.categories == ["cs.LG", "cs.CL"]
        assert cfg.output.top_n == 15


class TestValidateConfig:
    def test_valid_config_no_warnings(self) -> None:
        cfg = RepoRadarConfig()
        warnings = validate_config(cfg)
        assert warnings == []

    def test_unknown_category_prefix(self) -> None:
        cfg = RepoRadarConfig(arxiv=ArxivConfig(categories=["xx.YY", "cs.LG"]))
        warnings = validate_config(cfg)
        assert len(warnings) == 1
        assert "Unknown arXiv category prefix" in warnings[0]
        assert "xx" in warnings[0]

    def test_max_results_out_of_range(self) -> None:
        cfg = RepoRadarConfig(arxiv=ArxivConfig(max_results_per_query=0))
        warnings = validate_config(cfg)
        assert any("max_results_per_query" in w for w in warnings)

        cfg2 = RepoRadarConfig(arxiv=ArxivConfig(max_results_per_query=501))
        warnings2 = validate_config(cfg2)
        assert any("max_results_per_query" in w for w in warnings2)

    def test_lookback_days_too_low(self) -> None:
        cfg = RepoRadarConfig(arxiv=ArxivConfig(lookback_days=0))
        warnings = validate_config(cfg)
        assert any("lookback_days" in w for w in warnings)

    def test_negative_ranking_weights(self) -> None:
        cfg = RepoRadarConfig(ranking=RankingConfig(w_keyword=-1.0, w_category=0.5, w_recency=0.3))
        warnings = validate_config(cfg)
        assert any("Negative ranking weight" in w for w in warnings)
        assert any("w_keyword" in w for w in warnings)

    def test_negative_embedding_weight(self) -> None:
        cfg = RepoRadarConfig(ranking=RankingConfig(w_embedding=-0.5))
        warnings = validate_config(cfg)
        assert any("w_embedding" in w for w in warnings)

    def test_negative_category_weight(self) -> None:
        cfg = RepoRadarConfig(ranking=RankingConfig(category_weights={"cs.CL": -1.0}))
        warnings = validate_config(cfg)
        assert any("Negative category weight" in w for w in warnings)
        assert any("cs.CL" in w for w in warnings)

    def test_negative_citations_weight(self) -> None:
        cfg = RepoRadarConfig(ranking=RankingConfig(w_citations=-0.5))
        warnings = validate_config(cfg)
        assert any("w_citations" in w for w in warnings)

    def test_top_n_too_low(self) -> None:
        cfg = RepoRadarConfig(output=OutputConfig(top_n=0))
        warnings = validate_config(cfg)
        assert any("top_n" in w for w in warnings)


class TestDataclassDefaults:
    def test_arxiv_defaults(self) -> None:
        cfg = ArxivConfig()
        assert cfg.categories == ["cs.LG", "cs.CL"]
        assert cfg.max_results_per_query == 50
        assert cfg.lookback_days == 14

    def test_queries_defaults(self) -> None:
        cfg = QueriesConfig()
        assert cfg.seed == []
        assert cfg.exclude == []

    def test_ranking_defaults(self) -> None:
        cfg = RankingConfig()
        assert cfg.w_keyword == 1.0
        assert cfg.w_category == 0.5
        assert cfg.w_recency == 0.3
        assert cfg.w_embedding == 0.0
        assert cfg.w_citations == 0.0

    def test_output_defaults(self) -> None:
        cfg = OutputConfig()
        assert cfg.digest_path == "./reporadar_digest.md"
        assert cfg.top_n == 15
