"""Load and validate .reporadar.yml configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_CONFIG_NAME = ".reporadar.yml"


@dataclass
class ArxivConfig:
    categories: list[str] = field(default_factory=lambda: ["cs.LG", "cs.CL"])
    max_results_per_query: int = 50
    lookback_days: int = 14


@dataclass
class QueriesConfig:
    seed: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)


@dataclass
class RankingConfig:
    w_keyword: float = 1.0
    w_category: float = 0.5
    w_recency: float = 0.3


@dataclass
class OutputConfig:
    digest_path: str = "./reporadar_digest.md"
    top_n: int = 15


@dataclass
class RepoRadarConfig:
    repo_path: str = "."
    arxiv: ArxivConfig = field(default_factory=ArxivConfig)
    queries: QueriesConfig = field(default_factory=QueriesConfig)
    ranking: RankingConfig = field(default_factory=RankingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def _dict_to_config(data: dict) -> RepoRadarConfig:
    """Build a RepoRadarConfig from a raw dict (parsed YAML)."""
    arxiv = ArxivConfig(**data["arxiv"]) if "arxiv" in data else ArxivConfig()
    queries = QueriesConfig(**data["queries"]) if "queries" in data else QueriesConfig()
    ranking = RankingConfig(**data["ranking"]) if "ranking" in data else RankingConfig()
    output = OutputConfig(**data["output"]) if "output" in data else OutputConfig()

    return RepoRadarConfig(
        repo_path=data.get("repo_path", "."),
        arxiv=arxiv,
        queries=queries,
        ranking=ranking,
        output=output,
    )


def load_config(config_path: str | Path | None = None) -> RepoRadarConfig:
    """Load config from a YAML file.

    If *config_path* is None, looks for .reporadar.yml in the current directory.
    Raises FileNotFoundError if the file doesn't exist.
    """
    if config_path is None:
        config_path = Path(os.getcwd()) / DEFAULT_CONFIG_NAME

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return _dict_to_config(data)


KNOWN_ARXIV_PREFIXES = frozenset(
    {
        "astro-ph",
        "cond-mat",
        "cs",
        "econ",
        "eess",
        "gr-qc",
        "hep-ex",
        "hep-lat",
        "hep-ph",
        "hep-th",
        "math",
        "math-ph",
        "nlin",
        "nucl-ex",
        "nucl-th",
        "physics",
        "q-bio",
        "q-fin",
        "quant-ph",
        "stat",
    }
)


def validate_config(cfg: RepoRadarConfig) -> list[str]:
    """Return a list of warning messages for suspicious config values."""
    warnings: list[str] = []

    # Check arXiv category prefixes
    for cat in cfg.arxiv.categories:
        prefix = cat.split(".")[0]
        if prefix not in KNOWN_ARXIV_PREFIXES:
            warnings.append(f"Unknown arXiv category prefix: {prefix!r} (in {cat!r})")

    # Numeric bounds
    if cfg.arxiv.max_results_per_query < 1 or cfg.arxiv.max_results_per_query > 500:
        warnings.append(
            f"max_results_per_query={cfg.arxiv.max_results_per_query} is outside range [1, 500]"
        )

    if cfg.arxiv.lookback_days < 1:
        warnings.append(f"lookback_days={cfg.arxiv.lookback_days} should be >= 1")

    # Negative ranking weights
    for name in ("w_keyword", "w_category", "w_recency"):
        val = getattr(cfg.ranking, name)
        if val < 0:
            warnings.append(f"Negative ranking weight: {name}={val}")

    # top_n
    if cfg.output.top_n < 1:
        warnings.append(f"top_n={cfg.output.top_n} should be >= 1")

    return warnings


def default_config_yaml() -> str:
    """Return the default .reporadar.yml content as a string."""
    return """\
repo_path: .

arxiv:
  categories: [cs.LG, cs.CL]
  max_results_per_query: 50
  lookback_days: 14

queries:
  seed: []
  exclude: []

ranking:
  w_keyword: 1.0
  w_category: 0.5
  w_recency: 0.3

output:
  digest_path: ./reporadar_digest.md
  top_n: 15
"""
