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
from reporadar.ranker import format_score_explanation, rank_papers, score_distribution
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
@click.option("--explain", is_flag=True, help="Show detailed score breakdown for top papers.")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging.")
def update(config_path: str | None, explain: bool, verbose: bool) -> None:
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

    # 3b. Semantic Scholar source
    if "semantic_scholar" in cfg.sources:
        try:
            from reporadar.sources.semantic_scholar import (
                collect_papers as ss_collect,
            )

            info("Fetching papers from Semantic Scholar...")
            ss_queries = [q.replace("all:", "").strip('"') for q in queries[:5]]
            api_key = cfg.semantic_scholar.api_key or None
            ss_papers = ss_collect(
                ss_queries,
                api_key=api_key,
                lookback_days=cfg.arxiv.lookback_days,
            )
            # Merge: arXiv results take priority
            existing_ids = {p["arxiv_id"] for p in papers}
            new_from_ss = [p for p in ss_papers if p["arxiv_id"] not in existing_ids]
            papers.extend(new_from_ss)
            info(f"  {len(new_from_ss)} additional papers from Semantic Scholar")
        except Exception as exc:
            info(f"  Semantic Scholar collection failed: {exc}")

    # 3c. OpenAlex source
    if "openalex" in cfg.sources:
        try:
            from reporadar.sources.openalex import collect_papers as oa_collect

            info("Fetching papers from OpenAlex...")
            oa_queries = [q.replace("all:", "").strip('"') for q in queries[:5]]
            oa_papers = oa_collect(
                oa_queries,
                email=cfg.openalex.email or None,
                lookback_days=cfg.arxiv.lookback_days,
            )
            existing_ids = {p["arxiv_id"] for p in papers}
            new_from_oa = [p for p in oa_papers if p["arxiv_id"] not in existing_ids]
            papers.extend(new_from_oa)
            info(f"  {len(new_from_oa)} additional papers from OpenAlex")
        except Exception as exc:
            info(f"  OpenAlex collection failed: {exc}")

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

        # 5. Compute embeddings (optional)
        repo_embedding = None
        if cfg.ranking.w_embedding > 0:
            try:
                from reporadar.embeddings import EMBEDDINGS_AVAILABLE, compute_repo_embedding

                if EMBEDDINGS_AVAILABLE:
                    info("Computing embedding similarity...")
                    repo_embedding = compute_repo_embedding(repo_path)
                    if repo_embedding is None:
                        info("  No repo text found for embedding.")
                    else:
                        info("  Embedding similarity enabled.")
                else:
                    info("  Embedding similarity not available (install sentence-transformers).")
            except ImportError:
                info("  Embedding similarity not available (install sentence-transformers).")

        # 6. Citation counts (optional)
        citation_scores = None
        if cfg.ranking.w_citations > 0:
            info("Fetching citation counts...")
            try:
                from reporadar.citations import fetch_citation_counts, normalize_citations

                arxiv_ids = [p["arxiv_id"] for p in papers]
                api_key = cfg.semantic_scholar.api_key or None
                raw_counts = fetch_citation_counts(arxiv_ids, api_key=api_key)
                if raw_counts:
                    citation_scores = normalize_citations(raw_counts)
                    info(f"  Citation data for {len(citation_scores)} papers.")
                else:
                    info("  No citation data available.")
            except Exception as exc:
                info(f"  Citation lookup failed: {exc}")

        # 7. Rank
        info("Scoring papers...")
        scores = rank_papers(
            papers,
            repo_profile,
            cfg.ranking,
            cfg.queries,
            cfg.arxiv.categories,
            cfg.arxiv.lookback_days,
            repo_embedding=repo_embedding,
            citation_scores=citation_scores,
        )
        store.save_scores(run_id, scores)

        # 8. PwC enrichments for top papers
        try:
            from reporadar.paperswithcode import fetch_enrichments_batch

            top_ids = [s["arxiv_id"] for s in scores[: cfg.output.top_n]]
            if top_ids:
                info("Enriching top papers with Papers With Code data...")
                enrichments = fetch_enrichments_batch(top_ids, rate_limit=1.0)
                if enrichments:
                    store.save_enrichments(enrichments)
                    info(f"  Enrichment data for {len(enrichments)} papers.")
                else:
                    info("  No enrichment data found.")
        except Exception as exc:
            info(f"  PwC enrichment failed: {exc}")

        # Distribution stats
        dist = score_distribution(scores)
        info(
            f"  Score stats: mean={dist['mean']:.2f}, median={dist['median']:.2f}, "
            f"min={dist['min']:.2f}, max={dist['max']:.2f}, count={dist['count']}"
        )

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

        if explain and top:
            info("\nScore explanations:")
            for s in top:
                info(format_score_explanation(s, cfg.ranking))

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
    type=click.Choice(["md", "html", "json", "csv", "rss"], case_sensitive=False),
    help="Output format: md (default), html, json, csv, or rss.",
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
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging.")
def queries(config_path: str | None, verbose: bool) -> None:
    """Show the auto-generated queries that would be used by `update`."""
    if verbose:
        setup_verbose_logging()

    cfg = _load_and_validate(config_path)
    repo_path = Path(cfg.repo_path).resolve()

    repo_profile = profile_repo(repo_path)
    query_list = build_queries(repo_profile, cfg.queries, cfg.arxiv)

    if not query_list:
        warn("No queries generated. Add seed queries or ensure the repo has a README.")
        return

    # Categorize queries for display
    seed_queries: list[str] = []
    bigram_queries: list[str] = []
    keyword_queries: list[str] = []

    seed_set = set()
    for seed in cfg.queries.seed:
        seed_set.add(f'"{seed}"')

    for q in query_list:
        if any(s in q for s in seed_set):
            seed_queries.append(q)
        elif '" ' in q and 'all:"' in q:
            bigram_queries.append(q)
        else:
            keyword_queries.append(q)

    idx = 1
    if seed_queries:
        info("Seed queries:")
        for q in seed_queries:
            info(f"  {idx}. {q}")
            idx += 1

    if bigram_queries:
        info("Bigram queries:")
        for q in bigram_queries:
            info(f"  {idx}. {q}")
            idx += 1

    if keyword_queries:
        info("Keyword queries:")
        for q in keyword_queries:
            info(f"  {idx}. {q}")
            idx += 1

    info(f"\nTotal: {len(query_list)} queries")


@cli.command(name="gh-issues")
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to .reporadar.yml.",
)
@click.option(
    "--top",
    "top_n",
    default=5,
    type=int,
    help="Number of top papers to create issues for (default: 5).",
)
@click.option(
    "--run-id",
    default=None,
    type=int,
    help="Which run's scores to use (default: latest).",
)
@click.option("--dry-run", is_flag=True, help="Preview issues without creating them.")
@click.option(
    "--labels",
    default="reporadar",
    help="Comma-separated labels to add (default: reporadar).",
)
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging.")
def gh_issues(
    config_path: str | None,
    top_n: int,
    run_id: int | None,
    dry_run: bool,
    labels: str,
    verbose: bool,
) -> None:
    """Export top papers as GitHub Issues.

    Requires the `gh` CLI to be installed and authenticated.
    """
    if verbose:
        setup_verbose_logging()

    from reporadar.gh_issues import check_gh_available, create_issues
    from reporadar.suggestions import enrich_papers_with_suggestions

    if not dry_run and not check_gh_available():
        error("GitHub CLI (gh) not found or not authenticated.")
        error("Install it from https://cli.github.com/ and run `gh auth login`.")
        raise SystemExit(1)

    cfg = _load_and_validate(config_path)
    repo_path = Path(cfg.repo_path).resolve()
    db_path = repo_path / ".reporadar" / "papers.db"

    if not db_path.exists():
        error("No database found. Run `rr update` first.")
        raise SystemExit(1)

    label_list = [lbl.strip() for lbl in labels.split(",") if lbl.strip()]

    with _open_store(db_path) as store:
        if run_id is None:
            last_run = store.get_last_run()
            if last_run is None:
                error("No runs found. Run `rr update` first.")
                raise SystemExit(1)
            run_id = last_run["run_id"]

        scores = store.get_scores_for_run(run_id)
        if not scores:
            warn("No scored papers found.")
            return

        # Filter out already-exported papers
        exported = store.get_exported_ids("github_issue")
        candidates = [s for s in scores if s["arxiv_id"] not in exported][:top_n]

        if not candidates:
            info("All top papers have already been exported as issues.")
            return

        # Enrich with suggestions
        enrich_papers_with_suggestions(candidates)

        # Get enrichments
        arxiv_ids = [p["arxiv_id"] for p in candidates]
        enrichments = store.get_enrichments(arxiv_ids)

        info(f"{'[DRY RUN] ' if dry_run else ''}Creating issues for {len(candidates)} papers...")
        results = create_issues(
            candidates,
            enrichments=enrichments,
            labels=label_list,
            dry_run=dry_run,
        )

        for r in results:
            if r["status"] == "dry_run":
                info(f"  [DRY RUN] {r.get('title', r['arxiv_id'])}")
            elif r["status"] == "created":
                store.record_export(r["arxiv_id"], "github_issue", r["issue_url"])
                success(f"  Created: {r['issue_url']}")
            else:
                warn(f"  Skipped: {r['arxiv_id']}")

    created = sum(1 for r in results if r["status"] == "created")
    if dry_run:
        info(f"\nDry run complete. {len(results)} issues would be created.")
    elif created:
        success(f"\nCreated {created} GitHub issues.")


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
