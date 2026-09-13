"""Keyless Azure OpenAI across the product: config, the stage registry, privacy, setup, doctor.

The transport and the token have their own files (test_llm_client.py, test_azure_auth.py). This
one holds the wiring, because the failure this project keeps paying for is a provider one site
knows about and another does not: the OpenAI gate that `rr doctor` certified while the pipeline
skipped it, and the stage registry that still did not count an OpenAI gate as a gate when this
work began.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from reporadar import azure_auth, stages
from reporadar.cli import cli
from reporadar.config import LLM_PROVIDERS, load_config, validate_config
from reporadar.mcp_server import setup_repo_action
from reporadar.privacy import DESTINATIONS

SRC = Path(__file__).resolve().parents[1] / "src" / "reporadar"
ENDPOINT = "https://myres.openai.azure.com"


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / ".reporadar.yml"
    path.write_text(body, encoding="utf-8")
    return path


AZURE_YAML = f"""
repo_path: .
azure_openai:
  endpoint: {ENDPOINT}
  tenant: contoso.onmicrosoft.com
suggestions:
  provider: azure_openai
  azure_deployment: gpt-5.6-luna
triage:
  enabled: true
  finescale:
    enabled: true
    azure_deployment: gpt-4.1-mini
"""


class TestTheConfigReachesBothStages:
    def test_one_endpoint_and_tenant_travel_to_the_gate_and_the_rescore(
        self, tmp_path: Path
    ) -> None:
        cfg = load_config(_write(tmp_path, AZURE_YAML))
        for stage_cfg in (cfg.suggestions, cfg.triage.finescale):
            assert stage_cfg.azure_endpoint == ENDPOINT
            assert stage_cfg.azure_tenant == "contoso.onmicrosoft.com"
        assert cfg.suggestions.azure_deployment == "gpt-5.6-luna"
        assert cfg.triage.finescale.azure_deployment == "gpt-4.1-mini"

    def test_an_unset_rescore_provider_follows_an_azure_gate(self, tmp_path: Path) -> None:
        """So an Azure-only setup does not quietly demand an OpenAI key for one stage."""
        cfg = load_config(_write(tmp_path, AZURE_YAML))
        assert cfg.triage.finescale.provider == "azure_openai"

    def test_it_follows_nothing_else(self, tmp_path: Path) -> None:
        """Every config written before Azure existed keeps its OpenAI rescore."""
        cfg = load_config(_write(tmp_path, "suggestions:\n  provider: claude\n"))
        assert cfg.triage.finescale.provider == "openai"

    def test_an_explicit_rescore_provider_wins(self, tmp_path: Path) -> None:
        body = AZURE_YAML.replace(
            "    enabled: true\n", "    enabled: true\n    provider: openai\n"
        )
        assert load_config(_write(tmp_path, body)).triage.finescale.provider == "openai"

    def test_a_complete_azure_config_validates_clean(self, tmp_path: Path) -> None:
        assert validate_config(load_config(_write(tmp_path, AZURE_YAML))) == []


class TestValidationNamesWhatWouldStopIt:
    def _warnings(self, tmp_path: Path, body: str) -> str:
        return "\n".join(validate_config(load_config(_write(tmp_path, body))))

    def test_a_missing_endpoint(self, tmp_path: Path) -> None:
        body = AZURE_YAML.replace(f"  endpoint: {ENDPOINT}\n", "")
        assert "azure_openai.endpoint" in self._warnings(tmp_path, body)

    def test_an_endpoint_that_is_not_azure(self, tmp_path: Path) -> None:
        body = AZURE_YAML.replace(ENDPOINT, "https://attacker.example.com")
        assert "Refusing Azure OpenAI endpoint" in self._warnings(tmp_path, body)

    def test_a_missing_gate_deployment(self, tmp_path: Path) -> None:
        body = AZURE_YAML.replace("  azure_deployment: gpt-5.6-luna\n", "")
        assert "suggestions.azure_deployment" in self._warnings(tmp_path, body)

    def test_a_rescore_on_azure_with_no_deployment(self, tmp_path: Path) -> None:
        body = AZURE_YAML.replace("    azure_deployment: gpt-4.1-mini\n", "")
        assert "triage.finescale.azure_deployment" in self._warnings(tmp_path, body)

    def test_a_tenant_that_could_be_a_command(self, tmp_path: Path) -> None:
        body = AZURE_YAML.replace("contoso.onmicrosoft.com", '"x & calc"')
        assert "not a tenant id or domain" in self._warnings(tmp_path, body)

    def test_an_unknown_rescore_provider(self, tmp_path: Path) -> None:
        body = "triage:\n  finescale:\n    provider: claude\n"
        assert "triage.finescale.provider" in self._warnings(tmp_path, body)

    def test_azure_openai_can_run_the_gate(self, tmp_path: Path) -> None:
        assert "triage needs an LLM provider" not in self._warnings(tmp_path, AZURE_YAML)


class TestEveryGateDecisionUsesTheOneList:
    """The guard for the drift this change found on its way in."""

    @pytest.mark.parametrize("provider", LLM_PROVIDERS)
    def test_the_stage_registry_counts_every_provider_as_a_gate(self, provider: str) -> None:
        """It counted only ollama and claude, so an OpenAI-gated config was reported by
        `rr workspace` as having no gate at all."""
        from reporadar.config import RepoRadarConfig

        cfg = RepoRadarConfig()
        cfg.triage.enabled = True
        cfg.suggestions.provider = provider
        assert stages._gate_on(cfg)

    @pytest.mark.parametrize(
        "module", ["pipeline.py", "stages.py", "config.py", "suggestions.py", "cli.py"]
    )
    def test_no_gate_decision_keeps_its_own_provider_list(self, module: str) -> None:
        source = "\n".join(
            line
            for line in (SRC / module).read_text(encoding="utf-8").splitlines()
            if not line.startswith("LLM_PROVIDERS")  # the one place the list is written
        )
        own_list = re.findall(r"""\(\s*["']ollama["']\s*,\s*["']claude["']""", source)
        assert not own_list, f"{module} spells out the gate providers instead of LLM_PROVIDERS"
        assert "LLM_PROVIDERS" in source


def _cfg(tmp_path: Path, body: str) -> Any:
    return load_config(_write(tmp_path, body))


def _active(cfg: Any, service: str) -> bool:
    return any(d.active(cfg) for d in DESTINATIONS if d.service.startswith(service))


class TestTheAuditReportsWherePromptsGo:
    def test_an_openai_gate_is_declared_at_last(self, tmp_path: Path) -> None:
        """OpenAI was absent from the registry, so `rr audit` never mentioned it."""
        cfg = _cfg(tmp_path, "suggestions:\n  provider: openai\n")
        assert _active(cfg, "OpenAI")
        assert not _active(cfg, "Azure OpenAI")

    def test_an_openai_rescore_behind_a_claude_gate_is_declared(self, tmp_path: Path) -> None:
        body = "suggestions:\n  provider: claude\ntriage:\n  finescale:\n    enabled: true\n"
        assert _active(_cfg(tmp_path, body), "OpenAI")

    def test_an_azure_setup_declares_the_resource_and_the_token_request(
        self, tmp_path: Path
    ) -> None:
        cfg = _cfg(tmp_path, AZURE_YAML)
        assert _active(cfg, "Azure OpenAI")
        assert _active(cfg, "Microsoft Entra ID")
        assert not _active(cfg, "OpenAI"), "nothing goes to api.openai.com"

    def test_the_token_request_is_declared_as_carrying_nothing_from_the_repo(self) -> None:
        entra = next(d for d in DESTINATIONS if d.service.startswith("Microsoft Entra ID"))
        assert entra.sensitivity == "none"


@pytest.fixture
def token_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(azure_auth, "get_token", lambda tenant="": "entra")


def _repo(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\n", encoding="utf-8")
    return tmp_path


def _setup(tmp_path: Path, **azure: Any) -> dict[str, Any]:
    repo = _repo(tmp_path)
    return setup_repo_action(
        repo, repo / ".reporadar.yml", categories=["cs.SE"], provider="azure_openai", **azure
    )


class TestSetupAsksForWhatItCannotInfer:
    @pytest.mark.usefixtures("token_ok")
    def test_without_endpoint_or_deployment_it_asks_and_writes_nothing(
        self, tmp_path: Path
    ) -> None:
        result = _setup(tmp_path)
        assert result["status"] == "needs_input"
        assert {"azure_endpoint", "azure_deployment"} <= set(result["missing"])
        assert not (tmp_path / ".reporadar.yml").exists()

    @pytest.mark.usefixtures("token_ok")
    def test_a_non_azure_endpoint_is_refused_with_the_reason(self, tmp_path: Path) -> None:
        result = _setup(
            tmp_path, azure_endpoint="https://attacker.example.com", azure_deployment="d"
        )
        assert result["status"] == "needs_input"
        assert "Refusing Azure OpenAI endpoint" in result["why"]
        assert not (tmp_path / ".reporadar.yml").exists()

    @pytest.mark.usefixtures("token_ok")
    def test_a_deployment_name_cannot_inject_yaml(self, tmp_path: Path) -> None:
        """It arrives as a tool argument and lands in a committed file."""
        result = _setup(tmp_path, azure_endpoint=ENDPOINT, azure_deployment="x\n  provider: claude")
        assert result["status"] == "needs_input"
        assert not (tmp_path / ".reporadar.yml").exists()


class TestSetupRetriesKeepWhatTheCallerSaid:
    """An agent follows `retry.with` literally. The categories retry used to carry only
    categories, so following it wrote an OpenAI gate on a machine with no OpenAI key, and
    setup_repo then refused to change the config it had just written."""

    @pytest.mark.usefixtures("token_ok")
    def test_following_the_categories_retry_still_configures_azure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import reporadar.mcp_server as mcp_server

        monkeypatch.setattr(mcp_server, "profile_payload", lambda *a, **k: {"keywords": []})
        repo = _repo(tmp_path)
        first = setup_repo_action(
            repo,
            repo / ".reporadar.yml",
            provider="azure_openai",
            azure_endpoint=ENDPOINT,
            azure_deployment="gate",
        )
        assert first["status"] == "needs_input"
        assert first["missing"] == ["categories"]
        assert "repo_profile" in first
        retry = dict(first["retry"]["with"])
        assert retry["provider"] == "azure_openai"
        assert retry["azure_endpoint"] == ENDPOINT
        assert retry["azure_deployment"] == "gate"
        retry["categories"] = ["cs.SE"]
        retry.pop("azure_finescale_deployment")  # the optional placeholder, left unset
        second = setup_repo_action(repo, repo / ".reporadar.yml", **retry)
        assert second["status"] == "ok"
        assert load_config(repo / ".reporadar.yml").suggestions.provider == "azure_openai"

    def test_missing_azure_values_and_categories_are_asked_for_together(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import reporadar.mcp_server as mcp_server

        monkeypatch.setattr(mcp_server, "profile_payload", lambda *a, **k: {"keywords": []})
        repo = _repo(tmp_path)
        result = setup_repo_action(
            repo, repo / ".reporadar.yml", provider="azure_openai", azure_endpoint=ENDPOINT
        )
        assert set(result["missing"]) == {"azure_deployment", "categories"}
        assert "repo_profile" in result
        assert result["retry"]["with"]["azure_endpoint"] == ENDPOINT
        assert "categories" in result["retry"]["with"]

    def test_an_unmeasured_azure_setup_is_refused_not_silently_ignored(
        self, tmp_path: Path
    ) -> None:
        """measured=False writes a config with no gate; it used to drop every Azure argument and
        then tell the user to run `rr auth --provider azure_openai`, which does not exist."""
        repo = _repo(tmp_path)
        result = setup_repo_action(
            repo,
            repo / ".reporadar.yml",
            categories=["cs.SE"],
            measured=False,
            provider="azure_openai",
            azure_endpoint=ENDPOINT,
            azure_deployment="gate",
        )
        assert result["status"] == "error"
        assert not (repo / ".reporadar.yml").exists()

    @pytest.mark.usefixtures("token_ok")
    @pytest.mark.parametrize("name", ["123", "true", "null", "0x1F", "1_000"])
    def test_a_deployment_name_yaml_would_retype_still_loads_as_that_name(
        self, tmp_path: Path, name: str
    ) -> None:
        result = _setup(
            tmp_path,
            azure_endpoint=ENDPOINT,
            azure_deployment=name,
            azure_finescale_deployment=name,
        )
        assert result["status"] == "ok"
        cfg = load_config(tmp_path / ".reporadar.yml")
        assert cfg.suggestions.azure_deployment == name
        assert cfg.triage.finescale.azure_deployment == name


class TestSetupWritesAWorkingKeylessConfig:
    @pytest.mark.usefixtures("token_ok")
    def test_the_written_config_loads_as_an_azure_gate_and_validates(self, tmp_path: Path) -> None:
        result = _setup(tmp_path, azure_endpoint=ENDPOINT + "/", azure_deployment="gpt-5.6-luna")
        assert result["status"] == "ok"
        cfg = load_config(tmp_path / ".reporadar.yml")
        assert cfg.suggestions.provider == "azure_openai"
        assert cfg.suggestions.azure_deployment == "gpt-5.6-luna"
        assert cfg.azure_openai.endpoint == ENDPOINT
        assert validate_config(cfg) == []

    @pytest.mark.usefixtures("token_ok")
    def test_the_rescore_is_on_only_when_a_deployment_was_named(self, tmp_path: Path) -> None:
        off = _setup(tmp_path / "a", azure_endpoint=ENDPOINT, azure_deployment="g")
        assert off["finescale_enabled"] is False
        assert load_config(tmp_path / "a" / ".reporadar.yml").triage.finescale.enabled is False

        on = _setup(
            tmp_path / "b",
            azure_endpoint=ENDPOINT,
            azure_deployment="g",
            azure_finescale_deployment="gpt-4.1-mini",
        )
        cfg = load_config(tmp_path / "b" / ".reporadar.yml")
        assert on["finescale_enabled"] is True
        assert (cfg.triage.finescale.enabled, cfg.triage.finescale.provider) == (
            True,
            "azure_openai",
        )

    @pytest.mark.usefixtures("token_ok")
    def test_a_keyless_config_never_tells_the_user_they_need_a_key(self, tmp_path: Path) -> None:
        _setup(tmp_path, azure_endpoint=ENDPOINT, azure_deployment="g")
        text = (tmp_path / ".reporadar.yml").read_text(encoding="utf-8")
        assert "API_KEY" not in text
        assert "Cognitive Services OpenAI User" in text

    def test_no_token_yet_says_to_sign_in_never_to_paste_anything(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def refuse(tenant: str = "") -> str:
            raise azure_auth.AzureAuthError("Not signed in to Azure. Run `az login` in a terminal.")

        monkeypatch.setattr(azure_auth, "get_token", refuse)
        result = _setup(tmp_path, azure_endpoint=ENDPOINT, azure_deployment="g")
        assert result["status"] == "ok", "the config is still worth writing"
        assert result["gate_key_present"] is False
        assert "az login" in result["next"] and "never ask" in result["next"]

    @pytest.mark.usefixtures("token_ok")
    def test_the_outcome_states_the_role_and_the_calibration_cost(self, tmp_path: Path) -> None:
        result = _setup(tmp_path, azure_endpoint=ENDPOINT, azure_deployment="g")
        notes = " ".join(result["notes"])
        assert "Cognitive Services OpenAI User" in notes
        assert "uncalibrated" in notes


class TestDoctorOnAKeylessConfig:
    def _doctor(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
        monkeypatch.setattr(azure_auth, "get_token", lambda tenant="": "entra")
        _setup(
            tmp_path,
            azure_endpoint=ENDPOINT,
            azure_deployment="gpt-5.6-luna",
            azure_finescale_deployment="gpt-4.1-mini",
        )
        with (
            patch("reporadar.hyde.index_shards", return_value=[1]),
            patch("reporadar.embeddings.EMBEDDINGS_AVAILABLE", True),
        ):
            return CliRunner().invoke(cli, ["doctor", "--config", str(tmp_path / ".reporadar.yml")])

    def test_both_stages_pass_without_any_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        out = self._doctor(tmp_path, monkeypatch).output
        assert "gate: azure_openai — deployment 'gpt-5.6-luna'" in out
        assert "fine-scale rescore: azure_openai — deployment 'gpt-4.1-mini'" in out
        assert "no OpenAI key" not in out

    def test_it_states_what_it_cannot_check_once(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        out = self._doctor(tmp_path, monkeypatch).output
        assert out.count("Cognitive Services OpenAI User") == 1
        assert "uncalibrated" in out

    def test_no_token_is_a_gap_that_fails_the_command(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(azure_auth, "get_token", lambda tenant="": "entra")
        _setup(tmp_path, azure_endpoint=ENDPOINT, azure_deployment="gpt-5.6-luna")

        def refuse(tenant: str = "") -> str:
            raise azure_auth.AzureAuthError("Not signed in to Azure. Run `az login` in a terminal.")

        monkeypatch.setattr(azure_auth, "get_token", refuse)
        result = CliRunner().invoke(cli, ["doctor", "--config", str(tmp_path / ".reporadar.yml")])
        assert result.exit_code == 1
        assert "no Entra token" in result.output and "az login" in result.output
