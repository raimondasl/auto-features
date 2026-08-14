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
    RecommendationsConfig,
    RepoRadarConfig,
    SignalsConfig,
    SuggestionsConfig,
    TriageConfig,
    default_config_yaml,
    load_config,
    measured_config_yaml,
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


class TestSignalsConfig:
    def test_defaults(self) -> None:
        cfg = RepoRadarConfig()
        # Integrity defaults ON: recommending retracted work is the worst ranking
        # failure, and the check costs a couple of throttled requests.
        assert cfg.signals.integrity is True
        # HN defaults OFF: measured 0/340 coverage for papers from the last 2 weeks.
        assert cfg.signals.hackernews is False
        assert cfg.ranking.w_attention == 0.0
        assert cfg.ranking.withdrawn_penalty == 0.1

    def test_parsed_from_yaml(self, tmp_path: Path) -> None:
        config_file = tmp_path / ".reporadar.yml"
        config_file.write_text(
            "repo_path: .\nsignals:\n  integrity: false\n  hackernews: true\n",
            encoding="utf-8",
        )
        cfg = load_config(config_file)
        assert cfg.signals.integrity is False
        assert cfg.signals.hackernews is True

    def test_attention_weight_without_the_source_warns(self) -> None:
        # Otherwise the weight is silently inert: nothing ever fetches the data.
        cfg = RepoRadarConfig(ranking=RankingConfig(w_attention=1.0))
        assert any("w_attention" in w and "hackernews" in w for w in validate_config(cfg))

    def test_attention_weight_with_the_source_is_clean(self) -> None:
        cfg = RepoRadarConfig(
            ranking=RankingConfig(w_attention=1.0), signals=SignalsConfig(hackernews=True)
        )
        assert validate_config(cfg) == []

    def test_out_of_range_withdrawn_penalty_warns(self) -> None:
        # > 1 would *promote* withdrawn papers — an inverted feature, not a tweak.
        for bad in (2.0, -0.5):
            cfg = RepoRadarConfig(ranking=RankingConfig(withdrawn_penalty=bad))
            assert any("withdrawn_penalty" in w for w in validate_config(cfg))

    def test_penalty_of_one_is_a_legitimate_opt_out(self) -> None:
        cfg = RepoRadarConfig(ranking=RankingConfig(withdrawn_penalty=1.0))
        assert validate_config(cfg) == []

    def test_negative_attention_weight_warns(self) -> None:
        cfg = RepoRadarConfig(
            ranking=RankingConfig(w_attention=-1.0), signals=SignalsConfig(hackernews=True)
        )
        assert any("Negative ranking weight: w_attention" in w for w in validate_config(cfg))


class TestEnrichmentOffSwitch:
    """``provider: off`` must disable enrichment even unquoted.

    PyYAML is YAML 1.1, so a bare ``off`` parses as the boolean ``False`` — which
    made the documented off-switch a no-op (``False != "off"``) unless the user
    happened to quote it, and produced the nonsense warning "Unknown enrichment
    provider: False".
    """

    def _load(self, tmp_path: Path, raw: str) -> RepoRadarConfig:
        config_file = tmp_path / ".reporadar.yml"
        config_file.write_text(f"repo_path: .\nenrichment:\n  provider: {raw}\n", encoding="utf-8")
        return load_config(config_file)

    def test_unquoted_off_disables_enrichment(self, tmp_path: Path) -> None:
        cfg = self._load(tmp_path, "off")
        assert cfg.enrichment.provider == "off"
        assert validate_config(cfg) == []

    def test_quoted_off_still_works(self, tmp_path: Path) -> None:
        assert self._load(tmp_path, "'off'").enrichment.provider == "off"

    def test_yaml_falsey_synonyms_mean_off(self, tmp_path: Path) -> None:
        for raw in ("no", "false", "No", "FALSE"):
            assert self._load(tmp_path, raw).enrichment.provider == "off"

    def test_yaml_truthy_means_the_default_provider(self, tmp_path: Path) -> None:
        # `provider: on` reads as "enrichment on", so honor that rather than
        # leaving a bare True to fall through as an unknown provider.
        assert self._load(tmp_path, "on").enrichment.provider == "huggingface"

    def test_real_provider_is_untouched(self, tmp_path: Path) -> None:
        assert self._load(tmp_path, "huggingface").enrichment.provider == "huggingface"

    def test_unknown_provider_still_warns(self, tmp_path: Path) -> None:
        cfg = self._load(tmp_path, "bogus")
        assert any("Unknown enrichment provider" in w for w in validate_config(cfg))


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

    def test_recommendations_defaults_and_bounds(self) -> None:
        cfg = RepoRadarConfig()
        assert cfg.recommendations.enabled is False  # opt-in
        assert validate_config(cfg) == []

        bad = RepoRadarConfig(recommendations=RecommendationsConfig(limit=0, max_seeds=0))
        warnings = validate_config(bad)
        assert any("recommendations.limit" in w for w in warnings)
        assert any("recommendations.max_seeds" in w for w in warnings)

    def test_recommendations_loaded_from_yaml(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / ".reporadar.yml"
        cfg_file.write_text(
            "recommendations:\n  enabled: true\n  limit: 5\n  max_seeds: 3\n", encoding="utf-8"
        )
        cfg = load_config(cfg_file)
        assert cfg.recommendations.enabled is True
        assert cfg.recommendations.limit == 5
        assert cfg.recommendations.max_seeds == 3

    def test_biorxiv_dblp_are_known_sources(self) -> None:
        cfg = RepoRadarConfig(sources=["arxiv", "biorxiv", "dblp"])
        assert not any("Unknown source" in w for w in validate_config(cfg))

    def test_unknown_source_warns(self) -> None:
        cfg = RepoRadarConfig(sources=["arxiv", "bogus"])
        warnings = validate_config(cfg)
        assert any("Unknown source" in w and "bogus" in w for w in warnings)

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
        assert cfg.lookback_days == 36500
        assert cfg.sort_by == "relevance"

    def test_queries_defaults(self) -> None:
        cfg = QueriesConfig()
        assert cfg.seed == []
        assert cfg.exclude == []

    def test_ranking_defaults(self) -> None:
        cfg = RankingConfig()
        assert cfg.w_keyword == 1.0
        assert cfg.w_category == 0.5
        assert cfg.w_recency == 0.0
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


class TestShippedDefaultsMatchTheMeasuredConfiguration:
    """The default used to be a 14-day submitted-first window, and nothing measured it.

    Every headline Tier B number since 2026-07-06 was produced under `--rr-all-time`, which
    is exactly these three values. The defaults and the validated configuration had drifted
    apart for a month, so they are pinned here: changing one of them silently would make
    every published number describe a configuration users do not get.

    All 48 benchmark targets are >= 11 months old. A fortnight's window cannot reach any of
    them, which is why this is a correctness question and not a taste one.
    """

    def test_discovery_is_all_time_and_relevance_first(self) -> None:
        cfg = ArxivConfig()
        assert cfg.lookback_days == 36500
        assert cfg.sort_by == "relevance"

    def test_recency_carries_no_ranking_weight_by_default(self) -> None:
        assert RankingConfig().w_recency == 0.0

    def test_the_generated_template_agrees_with_the_dataclass_defaults(self) -> None:
        """`rr init` writes the template; a template that disagrees with the code ships a
        different product than the tests exercise.

        This assertion used to name three fields, and `ranking.w_embedding` — the one
        field where the two surfaces actually disagree — was not one of them. So it read
        as a general guarantee while checking a hand-picked corner of it. It now walks
        every key the template sets, and a disagreement must be listed with its reason.
        """
        import yaml

        # Fields where the template deliberately overrides the dataclass. The reason lives
        # in the template itself, next to the value; this list only pins THAT the
        # disagreement is intended, so an accidental one fails.
        INTENDED_OVERRIDES = {"ranking.w_embedding"}

        cfg = RepoRadarConfig(repo_path=".")
        parsed = yaml.safe_load(default_config_yaml())
        mismatches = []
        for section, body in parsed.items():
            if not isinstance(body, dict):
                continue
            for key, written in body.items():
                default = getattr(getattr(cfg, section), key)
                if written != default and f"{section}.{key}" not in INTENDED_OVERRIDES:
                    mismatches.append(f"{section}.{key}: template={written!r} code={default!r}")
        assert mismatches == [], "template disagrees with the dataclass: " + "; ".join(mismatches)

    def test_the_default_template_says_how_weak_it_is_and_where_to_go(self) -> None:
        """The shipped default enables neither the gate, the rescore, nor HyDE — each for a
        good reason (a credential, a second vendor, 1.1 GB of index). A template that says
        nothing about them hands a new user the ungated digest with no hint that the rest
        exists, which is how the product came to differ from every number published about it.

        It points at `rr init --measured` rather than inlining a commented copy of that
        config, because a commented copy is a second implementation of the recommendation
        and would drift from the first — the defect this project keeps finding.
        """
        text = default_config_yaml()
        # Both halves of the measured pair, from the 2026-08-16 run of each configuration
        # on all 25 cases at width 15. This asserted "-11" and "+5.42" until C-17: the
        # first had been measured on four repositories in July and the second is the
        # 24-case vs-baseline figure, so the pair was not comparable in either direction.
        assert "-8.12" in text and "+5.12" in text
        assert "rr init --measured" in text

    def test_the_measured_template_names_every_stage_it_turns_on(self) -> None:
        """The paths it sets must be real. A recommended config naming a renamed field is
        worse than none: it looks authoritative and silently does nothing."""
        import yaml

        data = yaml.safe_load(measured_config_yaml())
        cfg = RepoRadarConfig(repo_path=".")
        for section, field in (
            ("triage", "enabled"),
            ("hyde", "enabled"),
            ("suggestions", "provider"),
            ("ranking", "hybrid"),
        ):
            assert field in data[section], f"measured config lost {section}.{field}"
            assert hasattr(getattr(cfg, section), field)
        assert "enabled" in data["triage"]["finescale"]
        assert hasattr(cfg.triage.finescale, "enabled")

    def test_the_measured_template_states_its_dependencies_and_price(self) -> None:
        """Four paid/heavy stages recommended without naming the bill is worse than
        silence, so the file itself must carry the keys, the download and the cost."""
        text = measured_config_yaml()
        for needed in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "rr sync-index", "1.1 GB"):
            assert needed in text, f"the measured config never mentions {needed}"
        assert "$0.01" in text
