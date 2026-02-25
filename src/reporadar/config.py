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
    w_embedding: float = 0.0
    w_citations: float = 0.0
    category_weights: dict[str, float] = field(default_factory=dict)


@dataclass
class SemanticScholarConfig:
    api_key: str = ""


@dataclass
class OutputConfig:
    digest_path: str = "./reporadar_digest.md"
    top_n: int = 15


@dataclass
class OpenAlexConfig:
    email: str = ""


@dataclass
class EmailHookConfig:
    smtp_host: str = ""
    smtp_port: int = 587
    from_addr: str = ""
    to: str = ""
    username: str = ""
    password: str = ""
    use_tls: bool = True


@dataclass
class HooksConfig:
    on_digest: str = ""
    slack_webhook_url: str = ""
    discord_webhook_url: str = ""
    email: EmailHookConfig = field(default_factory=EmailHookConfig)


@dataclass
class ProfilerConfig:
    scan_source: bool = False
    max_files: int = 100
    source_extensions: list[str] = field(
        default_factory=lambda: [".py", ".js", ".ts", ".tsx", ".jsx"]
    )


@dataclass
class SuggestionsConfig:
    provider: str = "template"  # "template" | "ollama" | "claude"
    ollama_model: str = "llama3.2"
    ollama_url: str = "http://localhost:11434"
    claude_api_key: str = ""
    claude_model: str = "claude-sonnet-4-20250514"
    max_suggestions: int = 3
    timeout: int = 30


@dataclass
class FeedbackConfig:
    enabled: bool = False
    min_ratings: int = 10
    learning_rate: float = 0.1


@dataclass
class RepoRadarConfig:
    repo_path: str = "."
    sources: list[str] = field(default_factory=lambda: ["arxiv"])
    arxiv: ArxivConfig = field(default_factory=ArxivConfig)
    queries: QueriesConfig = field(default_factory=QueriesConfig)
    ranking: RankingConfig = field(default_factory=RankingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    semantic_scholar: SemanticScholarConfig = field(default_factory=SemanticScholarConfig)
    openalex: OpenAlexConfig = field(default_factory=OpenAlexConfig)
    hooks: HooksConfig = field(default_factory=HooksConfig)
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)
    suggestions: SuggestionsConfig = field(default_factory=SuggestionsConfig)
    feedback: FeedbackConfig = field(default_factory=FeedbackConfig)


def _dict_to_config(data: dict) -> RepoRadarConfig:
    """Build a RepoRadarConfig from a raw dict (parsed YAML)."""
    arxiv = ArxivConfig(**data["arxiv"]) if "arxiv" in data else ArxivConfig()
    queries = QueriesConfig(**data["queries"]) if "queries" in data else QueriesConfig()
    if "ranking" in data:
        ranking_data = dict(data["ranking"])
        cat_weights = ranking_data.pop("category_weights", {})
        ranking = RankingConfig(**ranking_data, category_weights=cat_weights or {})
    else:
        ranking = RankingConfig()
    output = OutputConfig(**data["output"]) if "output" in data else OutputConfig()
    semantic_scholar = (
        SemanticScholarConfig(**data["semantic_scholar"])
        if "semantic_scholar" in data
        else SemanticScholarConfig()
    )
    openalex = OpenAlexConfig(**data["openalex"]) if "openalex" in data else OpenAlexConfig()
    sources = data.get("sources", ["arxiv"])

    if "hooks" in data:
        hooks_data = dict(data["hooks"])
        email_data = hooks_data.pop("email", {})
        email_hook = EmailHookConfig(**email_data) if email_data else EmailHookConfig()
        hooks = HooksConfig(**hooks_data, email=email_hook)
    else:
        hooks = HooksConfig()

    profiler = ProfilerConfig(**data["profiler"]) if "profiler" in data else ProfilerConfig()
    suggestions = (
        SuggestionsConfig(**data["suggestions"]) if "suggestions" in data else SuggestionsConfig()
    )
    feedback = FeedbackConfig(**data["feedback"]) if "feedback" in data else FeedbackConfig()

    return RepoRadarConfig(
        repo_path=data.get("repo_path", "."),
        sources=sources,
        arxiv=arxiv,
        queries=queries,
        ranking=ranking,
        output=output,
        semantic_scholar=semantic_scholar,
        openalex=openalex,
        hooks=hooks,
        profiler=profiler,
        suggestions=suggestions,
        feedback=feedback,
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

    # Check sources
    known_sources = {"arxiv", "semantic_scholar", "openalex"}
    for src in cfg.sources:
        if src not in known_sources:
            known = ", ".join(sorted(known_sources))
            warnings.append(f"Unknown source: {src!r}. Known sources: {known}")

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
    for name in ("w_keyword", "w_category", "w_recency", "w_embedding", "w_citations"):
        val = getattr(cfg.ranking, name)
        if val < 0:
            warnings.append(f"Negative ranking weight: {name}={val}")

    # Category weights
    for cat, weight in cfg.ranking.category_weights.items():
        if weight < 0:
            warnings.append(f"Negative category weight: {cat}={weight}")

    # top_n
    if cfg.output.top_n < 1:
        warnings.append(f"top_n={cfg.output.top_n} should be >= 1")

    # Hooks
    if cfg.hooks.email.smtp_port < 1 or cfg.hooks.email.smtp_port > 65535:
        warnings.append(
            f"hooks.email.smtp_port={cfg.hooks.email.smtp_port} is outside range [1, 65535]"
        )

    # Profiler
    if cfg.profiler.max_files < 1:
        warnings.append(f"profiler.max_files={cfg.profiler.max_files} should be >= 1")

    # Suggestions
    known_providers = {"template", "ollama", "claude"}
    if cfg.suggestions.provider not in known_providers:
        warnings.append(
            f"Unknown suggestions provider: {cfg.suggestions.provider!r}. "
            f"Known: {', '.join(sorted(known_providers))}"
        )
    if cfg.suggestions.max_suggestions < 1:
        warnings.append(
            f"suggestions.max_suggestions={cfg.suggestions.max_suggestions} should be >= 1"
        )
    if cfg.suggestions.timeout < 1:
        warnings.append(f"suggestions.timeout={cfg.suggestions.timeout} should be >= 1")

    # Feedback
    if cfg.feedback.min_ratings < 1:
        warnings.append(f"feedback.min_ratings={cfg.feedback.min_ratings} should be >= 1")
    if not (0 < cfg.feedback.learning_rate <= 1.0):
        warnings.append(
            f"feedback.learning_rate={cfg.feedback.learning_rate} should be in (0, 1.0]"
        )

    return warnings


def default_config_yaml() -> str:
    """Return the default .reporadar.yml content as a string."""
    return """\
repo_path: .

sources: [arxiv]

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
  w_embedding: 1.5

output:
  digest_path: ./reporadar_digest.md
  top_n: 15
"""
