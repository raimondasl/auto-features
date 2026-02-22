"""Click CLI entry points for RepoRadar."""

from __future__ import annotations

from pathlib import Path

import click

from reporadar.config import DEFAULT_CONFIG_NAME, default_config_yaml, load_config
from reporadar.profiler import profile_repo


@click.group()
@click.version_option(package_name="reporadar")
def cli() -> None:
    """RepoRadar — arXiv paper discovery for your repo."""


@cli.command()
@click.option(
    "--path",
    default=".",
    type=click.Path(exists=True, file_okay=False),
    help="Repo directory to initialize in.",
)
def init(path: str) -> None:
    """Initialize RepoRadar in a repository.

    Creates .reporadar.yml and the .reporadar/ storage directory.
    """
    repo = Path(path).resolve()
    config_file = repo / DEFAULT_CONFIG_NAME
    storage_dir = repo / ".reporadar"

    if config_file.exists():
        click.echo(f"Config already exists: {config_file}")
    else:
        config_file.write_text(default_config_yaml(), encoding="utf-8")
        click.echo(f"Created {config_file}")

    if storage_dir.exists():
        click.echo(f"Storage directory already exists: {storage_dir}")
    else:
        storage_dir.mkdir(parents=True)
        click.echo(f"Created {storage_dir}/")

    click.echo("RepoRadar initialized. Edit .reporadar.yml to customize.")


@cli.command()
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to .reporadar.yml (default: .reporadar.yml in current dir).",
)
def profile(config_path: str | None) -> None:
    """Print the inferred topic profile for this repository."""
    cfg = load_config(config_path)
    repo_path = Path(cfg.repo_path).resolve()

    click.echo(f"Profiling repo: {repo_path}\n")

    result = profile_repo(repo_path)

    # Keywords
    click.echo("Keywords (TF-IDF):")
    if result.keywords:
        for term, weight in result.keywords:
            bar = "#" * int(weight * 40)
            click.echo(f"  {weight:.4f}  {bar:20s}  {term}")
    else:
        click.echo("  (none found)")

    # Anchors
    click.echo(f"\nAnchors (packages): {', '.join(result.anchors) if result.anchors else '(none)'}")

    # Domains
    click.echo(f"Inferred domains:   {', '.join(result.domains) if result.domains else '(none)'}")
