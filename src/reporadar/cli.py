"""Click CLI entry points for RepoRadar."""

from __future__ import annotations

import contextlib
import webbrowser
from collections.abc import Iterator
from pathlib import Path

import click

from reporadar.collector import CollectionError, build_queries, collect_papers
from reporadar.config import (
    DEFAULT_CONFIG_NAME,
    RepoRadarConfig,
    default_config_yaml,
    load_config,
    validate_config,
)
from reporadar.digest import write_digest
from reporadar.output import error, info, muted, setup_verbose_logging, success, warn
from reporadar.profiler import profile_repo
from reporadar.ranker import rank_papers
from reporadar.store import PaperStore, StoreError


@contextlib.contextmanager
def _open_store(db_path: Path) -> Iterator[PaperStore]:
    """Open a PaperStore, catching StoreError with a user-friendly message."""
    try:
        store = PaperStore(db_path)
    except StoreError as exc:
        error(str(exc))
        raise SystemExit(1) from exc
    try:
        yield store
    finally:
        store.close()


def _load_and_validate(config_path: str | Path | None) -> RepoRadarConfig:
    """Load config and print any validation warnings."""
    cfg = load_config(config_path)
    for warning in validate_config(cfg):
        warn(warning)
    return cfg


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
        warn(f"Config already exists: {config_file}")
    else:
        config_file.write_text(default_config_yaml(), encoding="utf-8")
        success(f"Created {config_file}")

    if storage_dir.exists():
        warn(f"Storage directory already exists: {storage_dir}")
    else:
        storage_dir.mkdir(parents=True)
        success(f"Created {storage_dir}/")

    success("RepoRadar initialized. Edit .reporadar.yml to customize.")


@cli.command()
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to .reporadar.yml (default: .reporadar.yml in current dir).",
)
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging.")
def profile(config_path: str | None, verbose: bool) -> None:
    """Print the inferred topic profile for this repository."""
    if verbose:
        setup_verbose_logging()

    cfg = _load_and_validate(config_path)
    repo_path = Path(cfg.repo_path).resolve()

    info(f"Profiling repo: {repo_path}\n")

    result = profile_repo(repo_path)

    # Keywords
    info("Keywords (TF-IDF):")
    if result.keywords:
        for term, weight in result.keywords:
            bar = "#" * int(weight * 40)
            info(f"  {weight:.4f}  {bar:20s}  {term}")
    else:
        muted("  (none found)")

    # Anchors
    info(f"\nAnchors (packages): {', '.join(result.anchors) if result.anchors else '(none)'}")

    # Domains
    info(f"Inferred domains:   {', '.join(result.domains) if result.domains else '(none)'}")


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
        setup_verbose_logging()

    cfg = _load_and_validate(config_path)
    repo_path = Path(cfg.repo_path).resolve()
    db_path = repo_path / ".reporadar" / "papers.db"

    # Ensure storage dir exists
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Profile
    info(f"Profiling repo: {repo_path}")
    repo_profile = profile_repo(repo_path)
    info(f"  Found {len(repo_profile.keywords)} keywords, {len(repo_profile.anchors)} anchors")

    # 2. Build queries
    queries = build_queries(repo_profile, cfg.queries, cfg.arxiv)
    info(f"  Built {len(queries)} queries")

    if not queries:
        warn(
            "No queries to run. Add seed queries to .reporadar.yml or ensure the repo has a README."
        )
        return

    # 3. Collect
    info("Fetching papers from arXiv...")
    try:
        papers = collect_papers(queries, cfg.arxiv)
    except CollectionError as exc:
        error(f"Failed to fetch papers: {exc}")
        error("Check your network connection and try again.")
        raise SystemExit(1) from exc
    info(f"  Fetched {len(papers)} unique papers")

    if not papers:
        warn("No new papers found.")
        return

    # 4. Store
    with _open_store(db_path) as store:
        new_count, seen_count = store.upsert_papers(papers)
        run_id = store.record_run(
            queries_used=queries,
            papers_new=new_count,
            papers_seen=seen_count,
        )

        # 5. Rank
        info("Scoring papers...")
        scores = rank_papers(
            papers,
            repo_profile,
            cfg.ranking,
            cfg.queries,
            cfg.arxiv.categories,
            cfg.arxiv.lookback_days,
        )
        store.save_scores(run_id, scores)

        top = scores[:5]
        if top:
            info("\nTop papers:")
            for i, s in enumerate(top, 1):
                # Find paper title from the collected papers
                title = next(
                    (p["title"] for p in papers if p["arxiv_id"] == s["arxiv_id"]),
                    s["arxiv_id"],
                )
                info(f"  {i}. [{s['score_total']:.2f}] {title}")

    success(f"\nDone! Run #{run_id}: {new_count} new, {seen_count} already seen.")
    info(f"Total papers in DB: {new_count + seen_count}")


def _parse_since(since: str) -> int:
    """Parse a human-friendly duration like '7d' or '14d' into days."""
    since = since.strip().lower()
    if since.endswith("d"):
        try:
            return int(since[:-1])
        except ValueError:
            pass
    raise click.BadParameter(f"Invalid duration: {since!r}. Use format like '7d' or '14d'.")


@cli.command()
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to .reporadar.yml (default: .reporadar.yml in current dir).",
)
@click.option(
    "--since",
    default="7d",
    help="Only include papers from the last N days (e.g. 7d, 14d).",
)
@click.option(
    "--run-id",
    default=None,
    type=int,
    help="Use scores from a specific run ID (default: latest run).",
)
@click.option(
    "-o",
    "--output",
    "output_path",
    default=None,
    help="Output file path (default: from config digest_path).",
)
@click.option(
    "--format",
    "fmt",
    default="md",
    type=click.Choice(["md", "html"], case_sensitive=False),
    help="Output format: md (default) or html.",
)
@click.option("--diff", is_flag=True, help="Mark papers as [NEW] vs. carried over.")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging.")
def digest(
    config_path: str | None,
    since: str,
    run_id: int | None,
    output_path: str | None,
    fmt: str,
    diff: bool,
    verbose: bool,
) -> None:
    """Generate a Markdown digest of scored papers.

    By default uses the latest run's scores and writes to the
    path configured in .reporadar.yml.
    """
    if verbose:
        setup_verbose_logging()

    cfg = _load_and_validate(config_path)
    repo_path = Path(cfg.repo_path).resolve()
    db_path = repo_path / ".reporadar" / "papers.db"

    if not db_path.exists():
        error("No database found. Run `rr update` first.")
        raise SystemExit(1)

    with _open_store(db_path) as store:
        if run_id is None:
            last_run = store.get_last_run()
            if last_run is None:
                error("No runs found. Run `rr update` first.")
                raise SystemExit(1)
            run_id = last_run["run_id"]

        dest = output_path or cfg.output.digest_path
        out = write_digest(store, run_id, dest, top_n=cfg.output.top_n, fmt=fmt, diff=diff)

    success(f"Digest written to {out}")


@cli.command(name="open")
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to .reporadar.yml (default: .reporadar.yml in current dir).",
)
@click.option(
    "-n",
    "--top",
    "top_n",
    default=5,
    type=int,
    help="Number of top papers to open (default: 5).",
)
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging.")
def open_top(config_path: str | None, top_n: int, verbose: bool) -> None:
    """Open top-scored papers in the default browser."""
    if verbose:
        setup_verbose_logging()

    cfg = _load_and_validate(config_path)
    repo_path = Path(cfg.repo_path).resolve()
    db_path = repo_path / ".reporadar" / "papers.db"

    if not db_path.exists():
        error("No database found. Run `rr update` first.")
        raise SystemExit(1)

    with _open_store(db_path) as store:
        last_run = store.get_last_run()
        if last_run is None:
            error("No runs found. Run `rr update` first.")
            raise SystemExit(1)

        scores = store.get_scores_for_run(last_run["run_id"])

    if not scores:
        warn("No scored papers found.")
        return

    for s in scores[:top_n]:
        url = s["url"]
        info(f"Opening: {s['title']}")
        webbrowser.open(url)

    success(f"\nOpened {min(top_n, len(scores))} papers in browser.")


def _format_size(size_bytes: int) -> str:
    """Format a byte count as a human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}" if unit != "B" else f"{size_bytes} {unit}"
        size_bytes /= 1024  # type: ignore[assignment]
    return f"{size_bytes:.1f} TB"


@cli.command()
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to .reporadar.yml (default: .reporadar.yml in current dir).",
)
def status(config_path: str | None) -> None:
    """Show the current RepoRadar status."""
    cfg = _load_and_validate(config_path)
    repo_path = Path(cfg.repo_path).resolve()
    db_path = repo_path / ".reporadar" / "papers.db"

    info(f"Repo path:    {repo_path}")
    info(f"Categories:   {', '.join(cfg.arxiv.categories)}")
    info(f"DB path:      {db_path}")

    if not db_path.exists():
        warn("No database found. Run `rr update` first.")
        return

    db_size = db_path.stat().st_size
    info(f"DB size:      {_format_size(db_size)}")

    with _open_store(db_path) as store:
        count = store.paper_count()
        info(f"Papers:       {count}")

        last_run = store.get_last_run()
        if last_run is None:
            warn("No runs yet.")
        else:
            info(f"Last run:     #{last_run['run_id']} at {last_run['run_time']}")
            info(f"  New/seen:   {last_run['papers_new']}/{last_run['papers_seen']}")
            info(f"  Queries:    {len(last_run['queries_used'])}")


@cli.command()
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to .reporadar.yml (default: .reporadar.yml in current dir).",
)
@click.option(
    "--limit",
    default=10,
    type=int,
    help="Maximum number of runs to show (default: 10).",
)
def history(config_path: str | None, limit: int) -> None:
    """Show past collection runs."""
    cfg = _load_and_validate(config_path)
    repo_path = Path(cfg.repo_path).resolve()
    db_path = repo_path / ".reporadar" / "papers.db"

    if not db_path.exists():
        error("No database found. Run `rr update` first.")
        raise SystemExit(1)

    with _open_store(db_path) as store:
        runs = store.get_runs(limit=limit)

    if not runs:
        warn("No runs found.")
        return

    info(f"{'Run':>5}  {'Time':25s}  {'New':>4}  {'Seen':>4}  {'Queries':>7}")
    info(f"{'---':>5}  {'----':25s}  {'---':>4}  {'----':>4}  {'-------':>7}")
    for run in runs:
        info(
            f"#{run['run_id']:>4}  {run['run_time'][:25]:25s}"
            f"  {run['papers_new']:>4}  {run['papers_seen']:>4}"
            f"  {len(run['queries_used']):>7}"
        )
