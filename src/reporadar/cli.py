"""Click CLI entry points for RepoRadar."""

from __future__ import annotations

import logging
from pathlib import Path

import click

from reporadar.collector import build_queries, collect_papers
from reporadar.config import DEFAULT_CONFIG_NAME, default_config_yaml, load_config
from reporadar.profiler import profile_repo
from reporadar.store import PaperStore


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


@cli.command()
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to .reporadar.yml (default: .reporadar.yml in current dir).",
)
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging.")
def update(config_path: str | None, verbose: bool) -> None:
    """Fetch new papers from arXiv and store them.

    Profiles the repo, builds queries, fetches papers, and stores
    them in the local SQLite database.
    """
    if verbose:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    cfg = load_config(config_path)
    repo_path = Path(cfg.repo_path).resolve()
    db_path = repo_path / ".reporadar" / "papers.db"

    # Ensure storage dir exists
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Profile
    click.echo(f"Profiling repo: {repo_path}")
    repo_profile = profile_repo(repo_path)
    click.echo(f"  Found {len(repo_profile.keywords)} keywords, {len(repo_profile.anchors)} anchors")

    # 2. Build queries
    queries = build_queries(repo_profile, cfg.queries, cfg.arxiv)
    click.echo(f"  Built {len(queries)} queries")

    if not queries:
        click.echo("No queries to run. Add seed queries to .reporadar.yml or ensure the repo has a README.")
        return

    # 3. Collect
    click.echo("Fetching papers from arXiv...")
    papers = collect_papers(queries, cfg.arxiv)
    click.echo(f"  Fetched {len(papers)} unique papers")

    if not papers:
        click.echo("No new papers found.")
        return

    # 4. Store
    with PaperStore(db_path) as store:
        new_count, seen_count = store.upsert_papers(papers)
        run_id = store.record_run(
            queries_used=queries,
            papers_new=new_count,
            papers_seen=seen_count,
        )

    click.echo(f"\nDone! Run #{run_id}: {new_count} new, {seen_count} already seen.")
    click.echo(f"Total papers in DB: {new_count + seen_count}")
