"""Tests for reporadar.config."""

from __future__ import annotations

from pathlib import Path

import pytest

from reporadar.config import (
    ArxivConfig,
    EmailHookConfig,
    HooksConfig,
    OpenAlexConfig,
    OutputConfig,
    QueriesConfig,
    RankingConfig,
    RepoRadarConfig,
    SuggestionsConfig,
    TriageConfig,
    default_config_yaml,
    load_config,
    validate_config,
)


class TestEnvExpansion:
    def test_expands_braced_env_var(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("RR_TEST_SLACK", "https://hooks.example/abc")
        cfg_file = tmp_path / ".reporadar.yml"
        cfg_file.write_text("hooks:\n  slack_webhook_url: ${RR_TEST_SLACK}\n", encoding="utf-8")
        cfg = load_config(cfg_file)
        assert cfg.hooks.slack_webhook_url == "https://hooks.example/abc"

    def test_unset_var_becomes_empty(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.delenv("RR_TEST_MISSING", raising=False)
        cfg_file = tmp_path / ".reporadar.yml"
        cfg_file.write_text("openalex:\n  api_key: ${RR_TEST_MISSING}\n", encoding="utf-8")
        cfg = load_config(cfg_file)
        assert cfg.openalex.api_key == ""

    def test_leaves_bare_dollar_and_plain_text_untouched(self, tmp_path: Path, monkeypatch) -> None:
        # A bare $HOME (no braces) and a lone $ must be left exactly as written.
        monkeypatch.setenv("HOME", "/wherever")
        cfg_file = tmp_path / ".reporadar.yml"
        cfg_file.write_text('queries:\n  seed:\n    - "cost is $5 for $HOME"\n', encoding="utf-8")
        cfg = load_config(cfg_file)
        assert cfg.queries.seed == ["cost is $5 for $HOME"]

    def test_expands_inside_list_items(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("RR_TERM", "diffusion")
        cfg_file = tmp_path / ".reporadar.yml"
        cfg_file.write_text('queries:\n  seed:\n    - "${RR_TERM} models"\n', encoding="utf-8")
        cfg = load_config(cfg_file)
        assert cfg.queries.seed == ["diffusion models"]

    def test_non_string_leaves_round_trip(self, tmp_path: Path, monkeypatch) -> None:
        # Ints/bools must pass through untouched while a sibling ${VAR} still expands.
        monkeypatch.setenv("RR_CAT", "cs.AI")
        cfg_file = tmp_path / ".reporadar.yml"
        cfg_file.write_text(
            'arxiv:\n  categories: ["${RR_CAT}"]\n'
            "  max_results_per_query: 42\n  lookback_days: 7\n",
            encoding="utf-8",
        )
        cfg = load_config(cfg_file)
        assert cfg.arxiv.categories == ["cs.AI"]
        assert cfg.arxiv.max_results_per_query == 42
        assert cfg.arxiv.lookback_days == 7


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

    def test_triage_enabled_without_llm_provider_warns(self) -> None:
        # Default suggestions.provider is "template" — triage needs an LLM.
        cfg = RepoRadarConfig(triage=TriageConfig(enabled=True))
        warnings = validate_config(cfg)
        assert any("triage.enabled" in w and "LLM provider" in w for w in warnings)

    def test_triage_enabled_with_llm_provider_no_warning(self) -> None:
        cfg = RepoRadarConfig(
            triage=TriageConfig(enabled=True),
            suggestions=SuggestionsConfig(provider="claude", claude_api_key="k"),
        )
        warnings = validate_config(cfg)
        assert not any("triage.enabled" in w for w in warnings)

    def test_openalex_enabled_without_api_key_warns(self) -> None:
        cfg = RepoRadarConfig(sources=["arxiv", "openalex"])
        warnings = validate_config(cfg)
        assert any("openalex" in w and "api_key" in w for w in warnings)

    def test_openalex_with_api_key_no_warning(self) -> None:
        cfg = RepoRadarConfig(
            sources=["arxiv", "openalex"],
            openalex=OpenAlexConfig(api_key="k"),
        )
        warnings = validate_config(cfg)
        assert not any("openalex" in w and "api_key" in w for w in warnings)

    def test_openalex_not_enabled_no_key_warning(self) -> None:
        # Default sources is [arxiv] only; no OpenAlex key warning expected.
        cfg = RepoRadarConfig()
        warnings = validate_config(cfg)
        assert not any("api_key" in w for w in warnings)

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

    def test_sources_defaults(self) -> None:
        cfg = RepoRadarConfig()
        assert cfg.sources == ["arxiv"]

    def test_openalex_defaults(self) -> None:
        cfg = OpenAlexConfig()
        assert cfg.email == ""


class TestSourcesConfig:
    def test_load_sources(self, tmp_path: Path) -> None:
        config_file = tmp_path / ".reporadar.yml"
        config_file.write_text(
            "repo_path: .\nsources: [arxiv, semantic_scholar, openalex]\n",
            encoding="utf-8",
        )
        cfg = load_config(config_file)
        assert cfg.sources == ["arxiv", "semantic_scholar", "openalex"]

    def test_unknown_source_warning(self) -> None:
        cfg = RepoRadarConfig(sources=["arxiv", "unknown_source"])
        warnings = validate_config(cfg)
        assert any("Unknown source" in w for w in warnings)

    def test_valid_sources_no_warning(self) -> None:
        cfg = RepoRadarConfig(sources=["arxiv", "semantic_scholar", "openalex"])
        warnings = validate_config(cfg)
        assert not any("Unknown source" in w for w in warnings)

    def test_load_openalex_config(self, tmp_path: Path) -> None:
        config_file = tmp_path / ".reporadar.yml"
        config_file.write_text(
            "repo_path: .\nopenalex:\n  email: user@example.com\n",
            encoding="utf-8",
        )
        cfg = load_config(config_file)
        assert cfg.openalex.email == "user@example.com"


class TestHooksConfig:
    def test_hooks_defaults(self) -> None:
        cfg = HooksConfig()
        assert cfg.on_digest == ""
        assert cfg.slack_webhook_url == ""
        assert cfg.discord_webhook_url == ""
        assert isinstance(cfg.email, EmailHookConfig)
        assert cfg.email.smtp_port == 587
        assert cfg.email.use_tls is True

    def test_email_hook_defaults(self) -> None:
        cfg = EmailHookConfig()
        assert cfg.smtp_host == ""
        assert cfg.smtp_port == 587
        assert cfg.from_addr == ""
        assert cfg.to == ""
        assert cfg.username == ""
        assert cfg.password == ""
        assert cfg.use_tls is True

    def test_reporadar_config_has_hooks(self) -> None:
        cfg = RepoRadarConfig()
        assert isinstance(cfg.hooks, HooksConfig)

    def test_load_hooks_from_yaml(self, tmp_path: Path) -> None:
        config_file = tmp_path / ".reporadar.yml"
        config_file.write_text(
            "repo_path: .\n"
            "hooks:\n"
            "  on_digest: echo done\n"
            "  slack_webhook_url: https://hooks.slack.com/test\n"
            "  discord_webhook_url: https://discord.com/api/webhooks/test\n"
            "  email:\n"
            "    smtp_host: smtp.example.com\n"
            "    smtp_port: 465\n"
            "    from_addr: bot@example.com\n"
            "    to: user@example.com\n"
            "    username: myuser\n"
            "    password: mypass\n"
            "    use_tls: false\n",
            encoding="utf-8",
        )
        cfg = load_config(config_file)
        assert cfg.hooks.on_digest == "echo done"
        assert cfg.hooks.slack_webhook_url == "https://hooks.slack.com/test"
        assert cfg.hooks.discord_webhook_url == "https://discord.com/api/webhooks/test"
        assert cfg.hooks.email.smtp_host == "smtp.example.com"
        assert cfg.hooks.email.smtp_port == 465
        assert cfg.hooks.email.from_addr == "bot@example.com"
        assert cfg.hooks.email.to == "user@example.com"
        assert cfg.hooks.email.username == "myuser"
        assert cfg.hooks.email.password == "mypass"
        assert cfg.hooks.email.use_tls is False

    def test_hooks_missing_in_yaml_uses_defaults(self, tmp_path: Path) -> None:
        config_file = tmp_path / ".reporadar.yml"
        config_file.write_text("repo_path: .\n", encoding="utf-8")
        cfg = load_config(config_file)
        assert cfg.hooks.on_digest == ""
        assert cfg.hooks.email.smtp_host == ""

    def test_invalid_smtp_port_warning(self) -> None:
        hooks = HooksConfig(email=EmailHookConfig(smtp_port=0))
        cfg = RepoRadarConfig(hooks=hooks)
        warnings = validate_config(cfg)
        assert any("smtp_port" in w for w in warnings)
