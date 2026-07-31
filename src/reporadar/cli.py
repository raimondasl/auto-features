"""Click CLI entry points for RepoRadar."""

from __future__ import annotations

import contextlib
import re
import webbrowser
from collections.abc import Iterator
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import click

from reporadar.collector import CollectionError, build_queries, collect_papers
from reporadar.config import (
    DEFAULT_CONFIG_NAME,
    ArxivConfig,
    RankingConfig,
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

# Foundational/seed discovery: reach back over all of arXiv, relevance-first, with
# recency weight dropped — surfaces seminal work instead of only the recent fetch
# window (the baseline's edge on the Tier B benchmark). ~100 years ≈ no cutoff.
_FOUNDATIONAL_LOOKBACK = 36500

# Modern arXiv id with a version suffix, for version-insensitive cross-source dedup.
_ARXIV_VER_RE = re.compile(r"^(\d{4}\.\d{4,5})v\d+$")

# The only ranking weights feedback.compute_adjusted_weights learns; every other
# RankingConfig field is passed through untouched (see stage 7 in `update`).
_LEARNED_WEIGHTS = ("w_keyword", "w_category", "w_recency", "w_embedding", "w_citations")

# How many baselines `rr eval --history` lists before it says there are more.
_HISTORY_LIMIT = 10


def _dedup_id(arxiv_id: str) -> str:
    """Version-strip a modern arXiv id for cross-source dedup; leave others as-is."""
    match = _ARXIV_VER_RE.match(arxiv_id)
    return match.group(1) if match else arxiv_id


def _apply_foundational(
    arxiv_cfg: ArxivConfig, ranking_cfg: RankingConfig
) -> tuple[ArxivConfig, RankingConfig]:
    """Return (arxiv, ranking) configs adjusted for foundational discovery."""
    return (
        replace(arxiv_cfg, sort_by="relevance", lookback_days=_FOUNDATIONAL_LOOKBACK),
        replace(ranking_cfg, w_recency=0.0),
    )


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


def _load_and_validate(
    config_path: str | Path | None, warnings_to_stderr: bool = False
) -> RepoRadarConfig:
    """Load config and print any validation warnings.

    *warnings_to_stderr* is for commands whose stdout is machine-readable: a validation
    warning printed above a JSON document makes that document unparseable, so callers
    emitting JSON route the prose to stderr where it is still seen but not swallowed.
    """
    cfg = load_config(config_path)
    for warning in validate_config(cfg):
        if warnings_to_stderr:
            click.echo(click.style(warning, fg="yellow"), err=True)
        else:
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

    result = profile_repo(repo_path, profiler_cfg=cfg.profiler)

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

    # Source signals (only shown when source scanning is active)
    if result.source_signals:
        info(f"Source signals:     {', '.join(result.source_signals)}")

    # Paper sources this repo's literature suggests but the config doesn't enable.
    # Advisory only — enabling a source has real costs, so it stays the user's call.
    from reporadar.sources.suggest import format_suggestion, suggest_sources

    suggestions = suggest_sources(result, cfg.sources)
    if suggestions:
        info("")
        for suggestion in suggestions:
            warn(format_suggestion(suggestion))


@cli.command()
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to .reporadar.yml (default: .reporadar.yml in current dir).",
)
@click.option("--explain", is_flag=True, help="Show detailed score breakdown for top papers.")
@click.option(
    "--foundational",
    is_flag=True,
    help="Seed-corpus mode: fetch all-time, relevance-first (no recency window), so seminal "
    "foundational papers surface. Use for a one-time deep sweep; the default is the recent digest.",
)
@click.option(
    "--rebuild-embeddings",
    is_flag=True,
    help="Clear and recompute the cached paper embeddings for this run's model.",
)
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging.")
def update(
    config_path: str | None,
    explain: bool,
    foundational: bool,
    rebuild_embeddings: bool,
    verbose: bool,
) -> None:
    """Fetch new papers from arXiv and store them.

    Profiles the repo, builds queries, fetches papers, and stores
    them in the local SQLite database.
    """
    if verbose:
        setup_verbose_logging()

    cfg = _load_and_validate(config_path)
    if foundational:
        cfg.arxiv, cfg.ranking = _apply_foundational(cfg.arxiv, cfg.ranking)
        info("Foundational mode: all-time relevance discovery (recency weight dropped).")
    repo_path = Path(cfg.repo_path).resolve()
    db_path = repo_path / ".reporadar" / "papers.db"

    # Ensure storage dir exists
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Profile
    info(f"Profiling repo: {repo_path}")
    repo_profile = profile_repo(repo_path, profiler_cfg=cfg.profiler)
    info(f"  Found {len(repo_profile.keywords)} keywords, {len(repo_profile.anchors)} anchors")

    # Nudge (never auto-enable) domain sources this repo's literature needs but
    # the config leaves off — an arXiv-only run silently misses those papers.
    from reporadar.sources.suggest import format_suggestion, suggest_sources

    for suggestion in suggest_sources(repo_profile, cfg.sources):
        warn(f"  {format_suggestion(suggestion)}")

    # 2. Build queries
    queries = build_queries(repo_profile, cfg.queries, cfg.arxiv)
    info(f"  Built {len(queries)} queries")

    if not queries:
        if cfg.queries.redact:
            # Redaction can empty the query set outright, and blaming a missing README
            # for it would send the user looking in the wrong place entirely.
            warn(
                "No queries to run: `privacy.redact` removed every search term. "
                "Run `rr audit` to see what is left, or narrow the redaction list."
            )
        else:
            warn(
                "No queries to run. Add seed queries to .reporadar.yml or ensure "
                "the repo has a README."
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
            # Merge: arXiv results take priority. Version-strip both sides — arXiv hands
            # back `2401.12345` where another source may say `2401.12345v2`, and matching
            # raw ids lets the same paper through twice. DBLP and recommendations already
            # went through `_dedup_id`; these three merges had been left on raw equality.
            existing_ids = {_dedup_id(p["arxiv_id"]) for p in papers}
            new_from_ss = [p for p in ss_papers if _dedup_id(p["arxiv_id"]) not in existing_ids]
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
                api_key=cfg.openalex.api_key or None,
            )
            existing_ids = {_dedup_id(p["arxiv_id"]) for p in papers}
            new_from_oa = [p for p in oa_papers if _dedup_id(p["arxiv_id"]) not in existing_ids]
            papers.extend(new_from_oa)
            info(f"  {len(new_from_oa)} additional papers from OpenAlex")
        except Exception as exc:
            info(f"  OpenAlex collection failed: {exc}")

    # 3d. bioRxiv/medRxiv source (biology repos)
    if "biorxiv" in cfg.sources:
        try:
            from reporadar.sources.biorxiv import collect_papers as bx_collect

            info("Fetching papers from bioRxiv...")
            bx_queries = [q.replace("all:", "").strip('"') for q in queries[:5]]
            bx_papers = bx_collect(bx_queries, lookback_days=cfg.arxiv.lookback_days)
            existing_ids = {_dedup_id(p["arxiv_id"]) for p in papers}
            new_from_bx = [p for p in bx_papers if _dedup_id(p["arxiv_id"]) not in existing_ids]
            papers.extend(new_from_bx)
            info(f"  {len(new_from_bx)} additional papers from bioRxiv")
        except Exception as exc:
            info(f"  bioRxiv collection failed: {exc}")

    # 3e. DBLP source (systems / PL / DB / theory repos)
    if "dblp" in cfg.sources:
        try:
            from reporadar.sources.dblp import collect_papers as dblp_collect

            info("Fetching papers from DBLP...")
            dblp_queries = [q.replace("all:", "").strip('"') for q in queries[:5]]
            dblp_papers = dblp_collect(dblp_queries, lookback_days=cfg.arxiv.lookback_days)
            # Version-insensitive dedup so a DBLP CoRR paper collapses onto its
            # arXiv copy (which carries a version suffix).
            existing_ids = {_dedup_id(p["arxiv_id"]) for p in papers}
            new_from_dblp = [p for p in dblp_papers if _dedup_id(p["arxiv_id"]) not in existing_ids]
            papers.extend(new_from_dblp)
            info(f"  {len(new_from_dblp)} additional papers from DBLP")
        except Exception as exc:
            info(f"  DBLP collection failed: {exc}")

    # 3f. Learned recommendations from your ratings/stars (Feature 5, optional).
    #     Merged into the candidate pool so the local ranker re-filters them — the
    #     API is repo-agnostic and can return off-topic results for a niche seed.
    if cfg.recommendations.enabled:
        try:
            from reporadar.sources.s2_recommendations import fetch_recommendations

            with _open_store(db_path) as rec_store:
                ratings = rec_store.get_all_ratings()
                negatives = [aid for aid, r in ratings.items() if r <= 2]
                disliked = set(negatives)
                # Stars first (newest-first, and the most explicit signal) so they
                # survive the max_seeds cap; then highly-rated papers. An explicit
                # low rating beats an implicit star from `rr open`, which stars
                # every paper it opens — otherwise the same id lands in both lists.
                positives = [a for a in rec_store.get_starred_papers() if a not in disliked]
                positives += [
                    aid
                    for aid, r in ratings.items()
                    if r >= 4 and aid not in disliked and aid not in positives
                ]

            if positives:
                info("Fetching learned recommendations from your ratings...")
                recs = fetch_recommendations(
                    positives,
                    negatives,
                    limit=cfg.recommendations.limit,
                    max_seeds=cfg.recommendations.max_seeds,
                    api_key=cfg.semantic_scholar.api_key or None,
                )
                if recs is None:
                    warn("  Recommendations unavailable (Semantic Scholar error) — skipping.")
                else:
                    existing_ids = {_dedup_id(p["arxiv_id"]) for p in papers}
                    new_recs = [p for p in recs if _dedup_id(p["arxiv_id"]) not in existing_ids]
                    papers.extend(new_recs)
                    info(f"  {len(new_recs)} recommended papers (re-ranked locally)")
            else:
                info("  No rated/starred papers yet — skipping recommendations.")
        except Exception as exc:
            info(f"  Recommendations failed: {exc}")

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

        # 5b. Per-paper embeddings, cached across runs (compute once, not per run).
        paper_embeddings = None
        if repo_embedding is not None:
            try:
                from reporadar.embedding_cache import default_model, embed_papers_cached

                if rebuild_embeddings:
                    # Clear every cached vector (MiniLM + SPECTER2 + miss markers)
                    # so unresolvable papers get retried too.
                    cleared = store.clear_embeddings()
                    info(f"  Rebuilding embedding cache ({cleared} cleared).")
                before = store.embedding_count(default_model())
                paper_embeddings = embed_papers_cached(store, papers)
                newly = store.embedding_count(default_model()) - before
                info(f"  Embedding cache: {newly} encoded, {len(paper_embeddings) - newly} reused.")
            except Exception as exc:
                if verbose:
                    info(f"  Embedding cache skipped: {exc}")
                paper_embeddings = None

        # 6. Citation counts (optional)
        citation_scores = None
        if cfg.ranking.w_citations > 0:
            info("Fetching citation counts...")
            try:
                from reporadar.citations import fetch_citation_counts, normalize_citations

                # Only real arXiv ids resolve at S2; synthetic ones (ss:/dblp:/
                # biorxiv:/oa:) would just burn slots in the 500-id batch.
                arxiv_ids = [p["arxiv_id"] for p in papers if ":" not in p["arxiv_id"]]
                api_key = cfg.semantic_scholar.api_key or None
                raw_counts = fetch_citation_counts(arxiv_ids, api_key=api_key)
                if raw_counts:
                    citation_scores = normalize_citations(raw_counts)
                    info(f"  Citation data for {len(citation_scores)} papers.")
                else:
                    info("  No citation data available.")
            except Exception as exc:
                info(f"  Citation lookup failed: {exc}")

        # 6b. Citation proximity — "extends work you starred" (Feature 8, optional)
        citation_proximity = None
        if cfg.ranking.w_citation_proximity > 0:
            try:
                from reporadar.citation_graph import build_seed_set, find_citation_links
                from reporadar.citations import fetch_references

                seeds = build_seed_set(store)
                if seeds:
                    info("Checking which papers extend your starred/rated work...")
                    refs = fetch_references(
                        [p["arxiv_id"] for p in papers if ":" not in p["arxiv_id"]],
                        api_key=cfg.semantic_scholar.api_key or None,
                    )
                    links = find_citation_links(refs, seeds)
                    if links:
                        store.save_citations(
                            [(citing, cited) for citing, cs in links.items() for cited in cs]
                        )
                        citation_proximity = {citing: 1.0 for citing in links}
                        info(f"  {len(links)} paper(s) cite work you starred.")
                    else:
                        info("  None of this run's papers cite your starred work.")
            except Exception as exc:
                info(f"  Citation-proximity check failed: {exc}")

        # 6c. SPECTER2 similarity to the work you liked (Feature 7, optional).
        #     Citation-trained scientific embeddings, served free by Semantic
        #     Scholar — no local model. Needs at least one starred/highly-rated
        #     paper to form the query centroid.
        specter = None
        if cfg.ranking.w_specter > 0:
            try:
                from reporadar.specter import SPECTER_MODEL
                from reporadar.specter import score_papers as specter_score_papers

                liked = set(store.get_starred_papers()) | {
                    a for a, r in store.get_all_ratings().items() if r >= 4
                }
                resolvable = [a for a in liked if ":" not in a]
                if not liked:
                    info("  No starred or highly-rated papers yet — skipping SPECTER2.")
                elif not resolvable:
                    info("  Your liked papers are all non-arXiv — SPECTER2 has nothing to seed on.")
                else:
                    info("Scoring SPECTER2 similarity to your starred/rated papers...")
                    specter = specter_score_papers(
                        store, papers, api_key=cfg.semantic_scholar.api_key or None
                    )
                    if specter:
                        cached = store.embedding_count(SPECTER_MODEL)
                        info(f"  SPECTER2 scored {len(specter)} papers ({cached} vectors cached).")
                    else:
                        info("  No usable SPECTER2 signal for this run's papers.")
            except Exception as exc:
                info(f"  SPECTER2 scoring failed: {exc}")
                specter = None

        # 6d. Community attention from HF upvotes (Feature 1, optional).
        #     Enrichment runs *after* ranking (stage 9) because it only fetches for
        #     the top papers — so this reads the enrichments cached by *previous*
        #     runs. A brand-new paper therefore has no community signal on its first
        #     run; it is simply not scored on that component rather than penalised.
        community: dict[str, float] | None = None
        if cfg.ranking.w_community > 0:
            try:
                from reporadar.sources.hf_papers import normalize_upvotes

                cached_enrichments = store.get_enrichments([p["arxiv_id"] for p in papers])
                community = normalize_upvotes(
                    {aid: int(e.get("upvotes") or 0) for aid, e in cached_enrichments.items()}
                )
                if community:
                    info(f"  Community signal (HF upvotes) for {len(community)} papers.")
                else:
                    info("  No cached HF upvotes yet — community signal starts next run.")
            except Exception as exc:
                info(f"  Community signal failed: {exc}")
                community = None

        # 6e. Integrity: is any candidate a withdrawn paper (Feature 9)? On by
        #     default — recommending retracted work is the worst thing the ranker can
        #     do. Re-checked every run, not just at collect time: a paper can be
        #     withdrawn days after it was ingested (observed in a real corpus).
        withdrawn: set[str] = set()
        if cfg.signals.integrity:
            try:
                from reporadar.signals.integrity import (
                    fetch_comments,
                    find_withdrawn,
                    stale_ids,
                )

                # The free half first: a notice the authors put in the title or the
                # abstract needs no network at all, so it covers every candidate —
                # including papers from non-arXiv sources that the API cannot resolve.
                flags = find_withdrawn(papers, {})

                # Then the comment lookup, which is where most notices actually live.
                # Bounded: skip papers checked within the recheck window and cap how
                # many are looked up per run. arXiv wants 3s between requests, so an
                # unbounded pass over a --foundational (all-time) store would spend
                # minutes on a signal that fires for well under 1% of papers. Only ids
                # arXiv can resolve enter the queue, or synthetic ss:/oa:/dblp: ids
                # would occupy the cap forever and starve the real re-checks.
                previous = store.get_signals([p["arxiv_id"] for p in papers], "withdrawn")
                resolvable = [p["arxiv_id"] for p in papers if ":" not in p["arxiv_id"]]
                to_check = stale_ids(
                    resolvable, {aid: row["checked_at"] for aid, row in previous.items()}
                )
                comments = fetch_comments(to_check) if to_check else {}
                flags.update(find_withdrawn(papers, comments))

                if to_check and comments:
                    # Record a row for every id asked about, not just the answered
                    # ones: arXiv silently drops ids it does not know, and those would
                    # otherwise stay "never checked" forever and consume the whole
                    # per-run budget on every future run, starving the real re-checks.
                    store.save_signals(
                        [(aid, "withdrawn", flags.get(aid), None) for aid in to_check]
                    )
                # Trust a stored flag for anything not re-checked this run.
                for aid, row in previous.items():
                    if row["value"] and aid not in comments:
                        flags.setdefault(aid, row["value"])
                withdrawn = set(flags)

                if to_check and not comments:
                    # fetch_comments swallows per-batch failures, so an arXiv outage
                    # would otherwise print a clean "0 withdrawn" and hide the gap.
                    warn(
                        f"  Integrity check could not reach arXiv for {len(to_check)} "
                        "paper(s); only their stored title/abstract was checked."
                    )
                if withdrawn:
                    warn(f"  {len(withdrawn)} paper(s) withdrawn by their authors - demoted.")
                else:
                    info(f"  Integrity check: no withdrawn papers ({len(comments)} looked up).")
            except Exception as exc:
                info(f"  Integrity check failed: {exc}")
                withdrawn = set()

        # 6f. Hacker News attention (Feature 9, optional). Off by default: measured
        #     coverage is ~0% for papers published in the last two weeks.
        attention: dict[str, float] | None = None
        if cfg.signals.hackernews:
            try:
                from reporadar.signals.hn import fetch_attention, normalize_points

                stories = fetch_attention([p["arxiv_id"] for p in papers])
                if stories:
                    attention = normalize_points({k: v["points"] for k, v in stories.items()})
                    store.save_signals(
                        [
                            (aid, "hn", str(s["points"]), s["story_url"])
                            for aid, s in stories.items()
                        ]
                    )
                    info(f"  Discussed on Hacker News: {len(stories)} paper(s).")
                else:
                    info("  No Hacker News discussion found for this run's papers.")
            except Exception as exc:
                info(f"  Hacker News lookup failed: {exc}")
                attention = None

        # 7. Apply feedback-adjusted weights if enabled
        ranking_cfg = cfg.ranking
        if cfg.feedback.enabled:
            try:
                from reporadar.feedback import compute_adjusted_weights

                rated_scores = store.get_rated_paper_scores()
                if len(rated_scores) >= cfg.feedback.min_ratings:
                    current_weights = {k: getattr(ranking_cfg, k) for k in _LEARNED_WEIGHTS}
                    new_weights = compute_adjusted_weights(
                        rated_scores, current_weights, cfg.feedback.learning_rate
                    )
                    # replace() overrides only the learned weights, so every other
                    # knob (proximity, specter, community, category_weights, hybrid)
                    # survives feedback tuning automatically. Rebuilding the config
                    # field-by-field instead silently dropped each new component
                    # until someone remembered to add it here.
                    ranking_cfg = replace(
                        ranking_cfg,
                        w_keyword=new_weights["w_keyword"],
                        w_category=new_weights["w_category"],
                        w_recency=new_weights["w_recency"],
                        w_embedding=new_weights["w_embedding"],
                        w_citations=new_weights["w_citations"],
                    )
                    if verbose:
                        info("  Feedback: adjusted ranking weights from user ratings.")
                        for k, v in new_weights.items():
                            info(f"    {k}: {v:.4f}")
            except Exception as exc:
                if verbose:
                    info(f"  Feedback weight adjustment skipped: {exc}")

        # 8. Rank
        info("Scoring papers...")
        scores = rank_papers(
            papers,
            repo_profile,
            ranking_cfg,
            cfg.queries,
            cfg.arxiv.categories,
            cfg.arxiv.lookback_days,
            repo_embedding=repo_embedding,
            citation_scores=citation_scores,
            paper_embeddings=paper_embeddings,
            citation_proximity=citation_proximity,
            specter=specter,
            community=community,
            attention=attention,
            withdrawn=withdrawn,
        )
        # 8b. Hybrid retrieval (roadmap #4): fuse the heuristic order with a BM25
        #     lexical order via RRF, so a paper buried on vocabulary mismatch can
        #     surface. Sets rrf_score (persisted); the digest orders by it.
        if cfg.ranking.hybrid:
            from reporadar.retrieval import hybrid_reorder

            scores = hybrid_reorder(scores, papers, repo_profile)
            info("  Hybrid retrieval: fused heuristic + BM25 ranking (RRF).")
        store.save_scores(run_id, scores)

        # 8c. LLM actionability triage (Feature 6) — score the top papers for
        # whether they could genuinely improve THIS repo, so the digest can gate
        # its Top Picks on applicability instead of the raw heuristic score.
        if cfg.triage.enabled and cfg.suggestions.provider in ("ollama", "claude"):
            info(f"Triaging top {cfg.triage.top_k} papers for actionable relevance...")
            try:
                from reporadar.triage import triage_papers

                papers_by_id = {p["arxiv_id"]: p for p in papers}
                top_papers = [
                    papers_by_id[s["arxiv_id"]]
                    for s in scores[: cfg.triage.top_k]
                    if s["arxiv_id"] in papers_by_id
                ]
                llm_scores = triage_papers(
                    top_papers, repo_profile, cfg.suggestions, top_k=cfg.triage.top_k
                )
                if llm_scores:
                    store.save_llm_scores(run_id, llm_scores)
                    n_act = sum(
                        1
                        for v in llm_scores.values()
                        if v["llm_score"] >= cfg.triage.min_actionable
                    )
                    info(
                        f"  Triaged {len(llm_scores)} papers; {n_act} actionable "
                        f"(score >= {cfg.triage.min_actionable})."
                    )
                else:
                    info("  Triage produced no scores (all calls failed).")
            except Exception as exc:
                info(f"  Triage failed: {exc}")
        elif cfg.triage.enabled:
            info("  Triage enabled but suggestions.provider is not an LLM — skipping.")

        # 8. Save keyword frequencies for trend detection
        try:
            from reporadar.trends import compute_keyword_frequencies

            kw_freqs = compute_keyword_frequencies(papers, repo_profile)
            if kw_freqs:
                store.save_keyword_frequencies(run_id, kw_freqs)
        except Exception:
            pass  # Non-critical

        # 9. Hugging Face Papers enrichment for top papers
        if cfg.enrichment.provider != "off":
            try:
                from reporadar.sources.hf_papers import fetch_enrichments_batch

                # Only real arXiv IDs resolve on HF; skip synthetic ss:/oa: IDs.
                top_ids = [
                    s["arxiv_id"] for s in scores[: cfg.output.top_n] if ":" not in s["arxiv_id"]
                ]
                if top_ids:
                    info("Enriching top papers with Hugging Face Papers data...")
                    token = cfg.enrichment.hf_token or None
                    enrichments = fetch_enrichments_batch(top_ids, token=token)
                    if enrichments:
                        store.save_enrichments(enrichments)
                        info(f"  Enrichment data for {len(enrichments)} papers.")
                    else:
                        info("  No enrichment data found.")
            except Exception as exc:
                info(f"  Enrichment failed: {exc}")

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
                # ranking_cfg, not cfg.ranking: with feedback tuning on, the weights
                # that produced these scores are the adjusted ones, so explaining
                # them with the file's weights printed contributions that didn't
                # sum to the printed total.
                info(format_score_explanation(s, ranking_cfg))

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
    default=None,
    help="Only include papers published in the last N days (e.g. 7d, 14d). "
    "Default: include all scored papers in the run.",
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
    since: str | None,
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

        # LLM-powered suggestions need the repo profile; only pay to compute it
        # when an LLM provider is actually configured (templates don't need it).
        repo_profile = None
        if cfg.suggestions.provider in ("ollama", "claude"):
            info(f"Using LLM suggestions provider: {cfg.suggestions.provider}")
            repo_profile = profile_repo(repo_path, profiler_cfg=cfg.profiler)

        since_days = _parse_since(since) if since else None
        out, summary = write_digest(
            store,
            run_id,
            dest,
            top_n=cfg.output.top_n,
            fmt=fmt,
            diff=diff,
            since_days=since_days,
            suggestions_config=cfg.suggestions,
            profile=repo_profile,
            triage_threshold=(cfg.triage.min_actionable if cfg.triage.enabled else None),
            rerank=(cfg.triage.rerank if cfg.triage.enabled else False),
        )

    success(f"Digest written to {out}")

    # Fire on_digest hook if configured
    if summary and cfg.hooks.on_digest:
        from reporadar.notify import run_shell_hook

        info("Running on_digest hook...")
        if run_shell_hook(cfg.hooks.on_digest, summary):
            success("Hook completed.")
        else:
            warn("Hook failed.")


@cli.command()
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to .reporadar.yml (default: .reporadar.yml in current dir).",
)
@click.option(
    "--archive-dir",
    "archive_dir",
    default="digests",
    help="Directory for the published digest archive (default: digests).",
)
@click.option(
    "--run-id",
    default=None,
    type=int,
    help="Archive a specific run ID (default: latest run).",
)
@click.option(
    "--date",
    "date_str",
    default=None,
    help="Date label for this edition, YYYY-MM-DD (default: today, UTC).",
)
@click.option(
    "--since",
    default=None,
    help="Only include papers published in the last N days (e.g. 7d).",
)
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging.")
def archive(
    config_path: str | None,
    archive_dir: str,
    run_id: int | None,
    date_str: str | None,
    since: str | None,
    verbose: bool,
) -> None:
    """Publish the latest digest into a dated HTML archive (for GitHub Pages).

    Writes ``<archive-dir>/<date>.html`` and regenerates ``index.html`` listing
    every archived edition. Re-running on the same date replaces that edition.
    """
    if verbose:
        setup_verbose_logging()

    cfg = _load_and_validate(config_path)
    repo_path = Path(cfg.repo_path).resolve()
    db_path = repo_path / ".reporadar" / "papers.db"

    if not db_path.exists():
        error("No database found. Run `rr update` first.")
        raise SystemExit(1)

    from reporadar.archive import archive_digest

    with _open_store(db_path) as store:
        if run_id is None:
            last_run = store.get_last_run()
            if last_run is None:
                error("No runs found. Run `rr update` first.")
                raise SystemExit(1)
            run_id = last_run["run_id"]

        repo_profile = None
        if cfg.suggestions.provider in ("ollama", "claude"):
            repo_profile = profile_repo(repo_path, profiler_cfg=cfg.profiler)

        since_days = _parse_since(since) if since else None
        entry_path, index_path = archive_digest(
            store,
            run_id,
            archive_dir,
            date_str=date_str,
            top_n=cfg.output.top_n,
            suggestions_config=cfg.suggestions,
            profile=repo_profile,
            since_days=since_days,
            triage_threshold=(cfg.triage.min_actionable if cfg.triage.enabled else None),
            rerank=(cfg.triage.rerank if cfg.triage.enabled else False),
        )

    success(f"Archived digest -> {entry_path}")
    info(f"Index: {index_path}")


@cli.command()
@click.argument("query")
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to .reporadar.yml (default: .reporadar.yml in current dir).",
)
@click.option(
    "-n",
    "--limit",
    default=20,
    type=click.IntRange(min=1),
    help="Max results to show (default: 20).",
)
@click.option(
    "--since",
    default=None,
    help="Only search papers published in the last N days (e.g. 7d).",
)
@click.option(
    "--semantic",
    is_flag=True,
    help="Rank by embedding similarity instead of BM25 (needs the 'embeddings' extra).",
)
@click.option(
    "--hybrid",
    is_flag=True,
    help="Fuse semantic + BM25 rankings via RRF (implies --semantic).",
)
@click.option(
    "--format",
    "fmt",
    default="text",
    type=click.Choice(["text", "json"], case_sensitive=False),
    help="Output format: text (default) or json.",
)
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging.")
def search(
    query: str,
    config_path: str | None,
    limit: int,
    since: str | None,
    semantic: bool,
    hybrid: bool,
    fmt: str,
    verbose: bool,
) -> None:
    """Search every stored paper by free-text QUERY (over the whole local corpus).

    The store accumulates every paper `rr update` has ever fetched. Default search
    is offline BM25; `--semantic` ranks by embedding similarity and `--hybrid`
    fuses both (both need the `embeddings` extra and encode uncached papers once).
    """
    if verbose:
        setup_verbose_logging()

    cfg = _load_and_validate(config_path)
    repo_path = Path(cfg.repo_path).resolve()
    db_path = repo_path / ".reporadar" / "papers.db"

    if not db_path.exists():
        error("No database found. Run `rr update` first.")
        raise SystemExit(1)

    use_semantic = semantic or hybrid
    if use_semantic:
        from reporadar.embeddings import EMBEDDINGS_AVAILABLE

        if not EMBEDDINGS_AVAILABLE:
            error("Semantic search needs the embeddings extra: pip install 'reporadar[embeddings]'")
            raise SystemExit(1)

    from reporadar.digest import filter_since

    since_days = _parse_since(since) if since else None

    with _open_store(db_path) as store:
        papers = filter_since(store.get_all_papers(), since_days)
        if use_semantic:
            from reporadar.embedding_cache import default_model
            from reporadar.semantic import semantic_search

            if store.embedding_count(default_model()) < len(papers):
                info("Encoding uncached papers (first run may be slow)...")
            results = semantic_search(store, query, limit=limit, hybrid=hybrid, papers=papers)
        else:
            from reporadar.search import search_corpus

            results = search_corpus(papers, query, limit=limit)

    if fmt == "json":
        import json

        payload = [
            {
                "arxiv_id": p["arxiv_id"],
                "title": p["title"],
                "search_score": p["search_score"],
                "url": p.get("url"),
                "published": (p.get("published") or "")[:10],
                "authors": p.get("authors", []),
                "categories": p.get("categories", []),
            }
            for p in results
        ]
        click.echo(json.dumps(payload, indent=2, default=str))
        return

    if not results:
        warn(f"No matches for {query!r} across {len(papers)} stored papers.")
        return

    info(f"{len(results)} result(s) for {query!r} (of {len(papers)} stored papers):\n")
    for i, p in enumerate(results, 1):
        published = (p.get("published") or "")[:10]
        click.echo(f"{i:>2}. [{p['search_score']:.3f}] {p['title']}")
        click.echo(f"    {p['arxiv_id']}  {published}  {p.get('url', '')}")
        abstract = (p.get("abstract") or "").strip().replace("\n", " ")
        if abstract:
            snippet = abstract[:160] + ("..." if len(abstract) > 160 else "")
            click.echo(f"    {snippet}")
        click.echo("")


@cli.command()
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to .reporadar.yml.",
)
@click.option(
    "--channel",
    required=True,
    type=click.Choice(["shell", "slack", "discord", "email"], case_sensitive=False),
    help="Notification channel to use.",
)
@click.option(
    "--run-id",
    default=None,
    type=int,
    help="Which run to report on (default: latest).",
)
def notify(config_path: str | None, channel: str, run_id: int | None) -> None:
    """Send a notification about a digest run."""
    from reporadar.digest import categorize_papers, find_extends_starred
    from reporadar.notify import DigestSummary, dispatch_notification

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
            run: dict[str, Any] | None = last_run
        else:
            run = store.get_last_run()

        scored = store.get_scores_for_run(run_id)
        top_picks, _, _ = categorize_papers(
            scored,
            top_n=cfg.output.top_n,
            triage_threshold=(cfg.triage.min_actionable if cfg.triage.enabled else None),
            rerank=(cfg.triage.rerank if cfg.triage.enabled else False),
        )

        summary = DigestSummary(
            digest_path=cfg.output.digest_path,
            run_id=run_id,
            papers_new=run["papers_new"] if run else 0,
            papers_seen=run["papers_seen"] if run else 0,
            top_picks_count=len(top_picks),
            total_scored=len(scored),
            fmt="md",
            extends_starred_count=len(find_extends_starred(store, scored)),
        )

    if dispatch_notification(channel, cfg.hooks, summary):
        success(f"Notification sent via {channel}.")
    else:
        error(f"Notification via {channel} failed.")
        raise SystemExit(1)


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
            store.star_paper(s["arxiv_id"])

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

    # Same reason as `rr audit`: this command claims to show the queries `update` would
    # run, and omitting profiler_cfg silently makes that untrue whenever scan_source is on.
    repo_profile = profile_repo(repo_path, profiler_cfg=cfg.profiler)
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


@cli.command()
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to .reporadar.yml (default: .reporadar.yml in current dir).",
)
@click.option("--json", "as_json", is_flag=True, help="Emit the audit as JSON.")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging.")
def audit(config_path: str | None, as_json: bool, verbose: bool) -> None:
    """Show every network destination and query string this profile would send.

    Nothing here contacts the network: the report is derived from your config and a
    local repo profile, and the query strings come from the same `build_queries` call
    `update` uses, so they are the strings that would genuinely be transmitted.
    """
    if verbose:
        setup_verbose_logging()

    import json

    from reporadar.privacy import REPO_AND_CONTENT, audit_plan

    cfg = _load_and_validate(config_path, warnings_to_stderr=as_json)
    repo_path = Path(cfg.repo_path).resolve()
    # `profiler_cfg` is not optional here. Without it `scan_source` is off regardless of
    # config, so with `profiler.scan_source: true` the audit would report a docs-only
    # query set while `update` transmitted a different one built from source identifiers
    # — the report would omit the very strings a proprietary codebase cares about, while
    # printing "scan_source is on" underneath. Under-reporting is the one failure this
    # command must never have.
    repo_profile = profile_repo(repo_path, profiler_cfg=cfg.profiler)

    query_list = build_queries(repo_profile, cfg.queries, cfg.arxiv)
    # The same build, with redaction switched off — the only honest way to show what
    # the filter removed now that redaction happens inside build_queries itself.
    unredacted = build_queries(repo_profile, replace(cfg.queries, redact=[]), cfg.arxiv)
    plan = audit_plan(cfg, repo_profile, query_list, queries_unredacted=unredacted)

    if as_json:

        def _dest(d: Any) -> dict[str, Any]:
            # `active` is a predicate lambda, not data — drop it rather than let
            # json fall back to str() and emit "<function <lambda> at 0x...>".
            return {
                "module": d.module,
                "service": d.service,
                "endpoint": d.endpoint,
                "sends": d.sends,
                "sensitivity": d.sensitivity,
                "enabled_by": d.enabled_by,
            }

        payload = {
            **{k: v for k, v in plan.items() if k not in ("destinations", "on_demand")},
            "destinations": [_dest(d) for d in plan["destinations"]],
            "on_demand": [_dest(d) for d in plan["on_demand"]],
        }
        click.echo(json.dumps(payload, indent=2))
        return

    info(f"Repo: {repo_path}\n")

    dests = plan["destinations"]
    if dests:
        info(f"Reached by every `rr update` ({len(dests)}):")
        for d in dests:
            info(f"  [{d.sensitivity}] {d.service} - {d.endpoint}")
            muted(f"      sends: {d.sends}")
            muted(f"      enabled by: {d.enabled_by}")
    else:
        info("Reached by every `rr update`: nothing — no network source is enabled.")

    on_demand = plan["on_demand"]
    if on_demand:
        info(f"\nReached only by an explicit command ({len(on_demand)}):")
        for d in on_demand:
            info(f"  [{d.sensitivity}] {d.service} - {d.endpoint}")
            muted(f"      sends: {d.sends}")
            muted(f"      enabled by: {d.enabled_by}")

    info(f"\nQuery strings that would be transmitted ({len(plan['queries'])}):")
    for i, q in enumerate(plan["queries"], 1):
        info(f"  {i}. {q}")

    n_patterns = plan["n_patterns"]
    if not n_patterns:
        info("\nRedaction: not configured (`privacy.redact` is empty).")
    elif plan["redaction_changed_anything"]:
        before, after = len(plan["queries_before_redaction"]), len(plan["queries"])
        success(f"\nRedaction: {n_patterns} pattern(s) active - queries above are filtered.")
        if before != after:
            muted(f"      {before - after} quer(y/ies) dropped entirely by redaction.")
    else:
        warn(
            f"\nRedaction: {n_patterns} pattern(s) configured but nothing matched. "
            "Check the terms: literal unless prefixed with `re:`."
        )

    # The honest part. A redaction list is a filter on literal strings; it does not
    # make the rest of the payload anonymous, and saying so is the point of `audit`.
    info("\nWhat leaves regardless of redaction:")
    kw = plan["keywords"][:8]
    if kw:
        muted(f"  Profile keywords, which encode your domain: {', '.join(kw)}")
    if plan["anchors"]:
        muted(f"  Anchor terms: {', '.join(plan['anchors'][:8])}")
    if plan["scans_source"]:
        muted("  profiler.scan_source is on - keywords are drawn from source, not just docs.")
    # Name the destinations rather than inferring "LLM" from the sensitivity tier: the
    # shell hook is rated at that tier too, and calling it an LLM would be a plain lie
    # in the one section of this report whose whole job is not to overstate.
    worst = [d.service for d in dests + on_demand if d.sensitivity == REPO_AND_CONTENT]
    if worst:
        muted(f"  Full paper abstracts and your profile, to: {', '.join(worst)}")
    muted("  Paper IDs you rate, star or open, wherever a destination takes them.")


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
        enrich_papers_with_suggestions(candidates, config=cfg.suggestions)

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


@cli.command(name="eval")
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to .reporadar.yml.",
)
@click.option("-k", "top_k", default=10, type=click.IntRange(min=1), help="Cut-off for @k metrics.")
@click.option(
    "--compare",
    nargs=2,
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Two config files to A/B on identical data, with a bootstrap interval.",
)
@click.option(
    "--baseline",
    is_flag=True,
    help="Record this measurement so a later run (or CI) can detect a regression.",
)
@click.option("--label", default=None, help="Name for the recorded baseline.")
@click.option(
    "--against",
    default=None,
    help="Compare this run to a recorded baseline ('latest' or a snapshot id); "
    "exits 1 on a regression, so CI can gate on it.",
)
@click.option("--history", is_flag=True, help="Show previously recorded baselines and exit.")
@click.option("--format", "fmt", type=click.Choice(["text", "json"]), default="text")
def eval_cmd(
    config_path: str | None,
    top_k: int,
    compare: tuple[str, str] | None,
    baseline: bool,
    label: str | None,
    against: str | None,
    history: bool,
    fmt: str,
) -> None:
    """Score the ranker against the ratings you have already given.

    Treats your 4-5 star ratings (and stars) as relevant and 1-2 as not, then measures
    how well the current ranking config orders them. Use `--compare a.yml b.yml` to
    A/B two configs on identical data before trusting a ranking change.

    These metrics answer "does it order what I judged well", not "does it find
    everything good": your ratings only cover papers an earlier ranking surfaced.
    """
    import json as json_mod

    from reporadar.evaluation import (
        compare_configs,
        compare_to_baseline,
        evaluate,
        format_baseline_check,
        format_comparison,
        format_report,
        load_judgments,
    )

    cfg = _load_and_validate(config_path)
    repo_path = Path(cfg.repo_path).resolve()
    db_path = repo_path / ".reporadar" / "papers.db"
    if not db_path.exists():
        error("No database found. Run `rr update` first.")
        raise SystemExit(1)

    with _open_store(db_path) as store:
        if history:
            snapshots = store.get_metric_snapshots(limit=_HISTORY_LIMIT)
            if not snapshots:
                info("No baselines recorded yet. Run `rr eval --baseline` to record one.")
                return
            if fmt == "json":
                click.echo(json_mod.dumps(snapshots, indent=2, default=str))
                return
            # `k` is shown because snapshots taken at different cut-offs are not
            # comparable, and a column of bare nDCG values hides that.
            info(f"{'id':>4}  {'when':<20} {'label':<18} {'k':>3} {'nDCG':>7} {'judged':>7}")
            for snap in snapshots:
                nd = snap["metrics"].get("ndcg@k", 0.0)
                info(
                    f"{snap['snapshot_id']:>4}  {snap['taken_at'][:19]:<20} "
                    f"{(snap['label'] or '-'):<18} {snap['k']:>3} {nd:>7.3f} "
                    f"{snap['n_judged']:>7}"
                )
            if len(snapshots) == _HISTORY_LIMIT:
                info(f"  (showing the most recent {_HISTORY_LIMIT}; older baselines exist)")
            return

        judgments = load_judgments(store)
        if not judgments.labels:
            error("No ratings or stars yet - nothing to evaluate.")
            info("Rate papers with `rr rate <arxiv_id> <1-5>`, then run this again.")
            raise SystemExit(1)

        profile = profile_repo(repo_path, profiler_cfg=cfg.profiler)

        if compare:
            cfg_a, cfg_b = (_load_and_validate(path) for path in compare)
            # Only the `ranking:` block is applied. A user who also changed
            # `queries.exclude` or `arxiv.categories` between the two files would
            # otherwise believe those were compared too, and read the verdict as
            # covering a change that was never made.
            ignored = [
                name
                for name, a, b in (
                    ("queries", cfg_a.queries, cfg_b.queries),
                    ("arxiv.categories", cfg_a.arxiv.categories, cfg_b.arxiv.categories),
                    ("arxiv.lookback_days", cfg_a.arxiv.lookback_days, cfg_b.arxiv.lookback_days),
                )
                if a != b
            ]
            if ignored:
                warn(
                    "  Only the `ranking:` block is compared; these also differ and are "
                    f"NOT reflected below: {', '.join(ignored)}"
                )
            comparison = compare_configs(
                store, judgments, profile, cfg, cfg_a.ranking, cfg_b.ranking, k=top_k
            )
            if fmt == "json":
                click.echo(json_mod.dumps(comparison, indent=2, default=str))
            else:
                info(f"A = {compare[0]}")
                info(f"B = {compare[1]}\n")
                info(format_comparison(comparison, judgments))
            if baseline or against:
                # Say so rather than returning past them: a CI job that passed
                # --against and got exit 0 would read that as "no regression".
                warn("  --baseline/--against do nothing with --compare; nothing recorded.")
            return

        result = evaluate(store, judgments, profile, cfg, k=top_k)
        public = {m: v for m, v in result.items() if not m.startswith("_")}
        if fmt == "json":
            click.echo(json_mod.dumps(public, indent=2, default=str))
        else:
            info(format_report(result, judgments))

        if against:
            snapshots = store.get_metric_snapshots(limit=100)
            if against == "latest":
                snapshot = snapshots[0] if snapshots else None
            else:
                snapshot = next((s for s in snapshots if str(s["snapshot_id"]) == against), None)
            if snapshot is None:
                error(f"No recorded baseline matching {against!r}. Run `rr eval --baseline` first.")
                raise SystemExit(1)
            check = compare_to_baseline(public, snapshot)
            info("")
            info(format_baseline_check(check))
            if check.get("regressed"):
                # Non-zero exit is the whole point: this is what a CI job gates on.
                raise SystemExit(1)

        if baseline:
            snapshot_id = store.save_metric_snapshot(
                public,
                k=top_k,
                n_judged=judgments.n_judged,
                n_relevant=judgments.n_relevant,
                label=label,
                config=asdict(cfg.ranking),
            )
            info("")
            success(f"Recorded baseline #{snapshot_id}" + (f" ({label})" if label else ""))


@cli.command()
@click.argument("arxiv_id")
@click.argument("rating", type=click.IntRange(1, 5))
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to .reporadar.yml.",
)
def rate(arxiv_id: str, rating: int, config_path: str | None) -> None:
    """Rate a paper from 1 (not useful) to 5 (very useful).

    These ratings are used to learn your preferences and improve
    future paper recommendations when feedback.enabled is true.
    """
    cfg = _load_and_validate(config_path)
    repo_path = Path(cfg.repo_path).resolve()
    db_path = repo_path / ".reporadar" / "papers.db"

    if not db_path.exists():
        error("No database found. Run `rr update` first.")
        raise SystemExit(1)

    with _open_store(db_path) as store:
        paper = store.get_paper(arxiv_id)
        if paper is None:
            error(f"Paper {arxiv_id!r} not found in database.")
            raise SystemExit(1)

        store.save_rating(arxiv_id, rating)
        all_ratings = store.get_all_ratings()

    success(f"Rated {arxiv_id} = {rating}/5")
    info(f"  Paper: {paper['title']}")
    info(f"  Total ratings: {len(all_ratings)}")

    if cfg.feedback.enabled:
        needed = cfg.feedback.min_ratings - len(all_ratings)
        if needed > 0:
            info(f"  {needed} more ratings needed to enable weight adjustment.")
        else:
            info("  Weight adjustment active.")
    else:
        info("  Tip: set feedback.enabled: true in config to use ratings for ranking.")


@cli.command()
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to .reporadar.yml.",
)
def mcp(config_path: str | None) -> None:
    """Run RepoRadar as an MCP server (stdio) for coding agents.

    Exposes repo-aware tools — get_repo_profile, get_ranked_papers,
    explain_relevance, rate_paper, search_papers — to Claude Code / Cursor / VS Code
    / Windsurf.
    Requires the optional MCP extra:  uv pip install -e ".[mcp]"
    """
    cfg = _load_and_validate(config_path)
    repo_path = Path(cfg.repo_path).resolve()
    db_path = repo_path / ".reporadar" / "papers.db"

    from reporadar.mcp_server import run_stdio

    try:
        run_stdio(repo_path, db_path, profiler_cfg=cfg.profiler, ranking_cfg=cfg.ranking)
    except ImportError:
        error('MCP support not installed. Run: uv pip install -e ".[mcp]"')
        raise SystemExit(1) from None


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


@cli.command()
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to .reporadar.yml.",
)
@click.option("--cron", default=None, help='Cron expression to register (e.g. "0 9 * * 1").')
@click.option("--list", "list_flag", is_flag=True, help="Show registered schedules.")
@click.option("--remove", is_flag=True, help="Remove the registered schedule.")
def schedule(config_path: str | None, cron: str | None, list_flag: bool, remove: bool) -> None:
    """Manage scheduled runs (via crontab or schtasks)."""
    from reporadar.scheduler import add_schedule, list_schedules, remove_schedule

    if not cron and not list_flag and not remove:
        error("Specify --cron EXPR, --list, or --remove.")
        raise SystemExit(1)

    if list_flag:
        tasks = list_schedules()
        if not tasks:
            info("No schedules registered.")
        else:
            for t in tasks:
                info(f"  [{t.platform}] {t.cron_expr}  {t.command}")
        return

    if remove:
        if remove_schedule():
            success("Schedule removed.")
        else:
            warn("No schedule found to remove.")
        return

    # --cron: register schedule
    assert cron is not None
    fields = cron.strip().split()
    if len(fields) != 5:
        error(f"Invalid cron expression: expected 5 fields, got {len(fields)}.")
        raise SystemExit(1)

    cfg = _load_and_validate(config_path)
    config_file = config_path or str(Path(cfg.repo_path).resolve() / DEFAULT_CONFIG_NAME)

    if add_schedule(cron, config_file):
        success(f"Schedule registered: {cron}")
    else:
        error("Failed to register schedule.")
        raise SystemExit(1)


@cli.group()
def workspace() -> None:
    """Manage multi-repo workspaces."""


@workspace.command(name="init")
def workspace_init() -> None:
    """Initialize the workspace directory and database."""
    from reporadar.workspace import ensure_workspace_dir, open_workspace_store

    ws_dir = ensure_workspace_dir()
    store = open_workspace_store()
    store.close()
    success(f"Workspace initialized at {ws_dir}")


@workspace.command(name="add")
@click.argument("name")
@click.option(
    "--path",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Path to the repository.",
)
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to .reporadar.yml for this repo.",
)
def workspace_add(name: str, path: str, config_path: str | None) -> None:
    """Register a repository in the workspace."""
    from reporadar.workspace import open_workspace_store

    resolved = str(Path(path).resolve())
    store = open_workspace_store()
    try:
        store.add_workspace_repo(name, resolved, config_path)
    finally:
        store.close()
    success(f"Added repo '{name}' at {resolved}")


@workspace.command(name="list")
def workspace_list() -> None:
    """List registered repos in the workspace."""
    from reporadar.workspace import open_workspace_store

    store = open_workspace_store()
    try:
        repos = store.get_workspace_repos()
    finally:
        store.close()

    if not repos:
        info("No repos registered. Use `rr workspace add` to add one.")
        return

    for r in repos:
        cfg_note = f" (config: {r['config_path']})" if r.get("config_path") else ""
        info(f"  {r['repo_id']}: {r['repo_path']}{cfg_note}")


@workspace.command(name="remove")
@click.argument("name")
def workspace_remove(name: str) -> None:
    """Unregister a repository from the workspace."""
    from reporadar.workspace import open_workspace_store

    store = open_workspace_store()
    try:
        if store.remove_workspace_repo(name):
            success(f"Removed repo '{name}'.")
        else:
            warn(f"Repo '{name}' not found.")
    finally:
        store.close()


@workspace.command(name="update")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging.")
def workspace_update(verbose: bool) -> None:
    """Collect and score papers across all workspace repos."""
    if verbose:
        setup_verbose_logging()

    from reporadar.workspace import open_workspace_store, score_papers_for_repo

    store = open_workspace_store()
    try:
        repos = store.get_workspace_repos()
        if not repos:
            warn("No repos registered.")
            return

        # Gather all papers from each repo's config
        all_papers: list[dict[str, Any]] = []
        seen_ids: set[str] = set()

        for repo in repos:
            cfg_path = repo.get("config_path")
            if not cfg_path:
                info(f"  Skipping {repo['repo_id']} (no config)")
                continue

            cfg = _load_and_validate(cfg_path)
            repo_path = Path(repo["repo_path"]).resolve()

            info(f"Profiling {repo['repo_id']}...")
            # Each member is profiled under its own `profiler:` block. Dropping it here
            # made `scan_source: true` a no-op, so a workspace run collected against a
            # different query set than `rr update` would for the same repo.
            repo_profile = profile_repo(repo_path, profiler_cfg=cfg.profiler)
            queries = build_queries(repo_profile, cfg.queries, cfg.arxiv)

            if queries:
                try:
                    papers = collect_papers(queries, cfg.arxiv)
                    for p in papers:
                        if p["arxiv_id"] not in seen_ids:
                            all_papers.append(p)
                            seen_ids.add(p["arxiv_id"])
                except CollectionError as exc:
                    warn(f"  Collection failed for {repo['repo_id']}: {exc}")

        if not all_papers:
            warn("No papers collected.")
            return

        # Store papers and record run
        new_count, seen_count = store.upsert_papers(all_papers)
        run_id = store.record_run(
            queries_used=[],
            papers_new=new_count,
            papers_seen=seen_count,
        )

        # Score per repo
        for repo in repos:
            cfg_path = repo.get("config_path")
            if not cfg_path:
                continue
            cfg = _load_and_validate(cfg_path)
            info(f"Scoring for {repo['repo_id']}...")
            scores = score_papers_for_repo(repo["repo_id"], repo["repo_path"], all_papers, cfg)
            store.save_repo_scores(repo["repo_id"], run_id, scores)

        success(f"Run #{run_id}: {new_count} new, {seen_count} seen across {len(repos)} repos.")
    finally:
        store.close()


@workspace.command(name="digest")
@click.option("--run-id", default=None, type=int, help="Run ID (default: latest).")
@click.option("-o", "--output", "output_path", default=None, help="Output file path.")
@click.option(
    "--format",
    "fmt",
    default="md",
    type=click.Choice(["md"], case_sensitive=False),
    help="Output format (md).",
)
def workspace_digest(run_id: int | None, output_path: str | None, fmt: str) -> None:
    """Generate a combined workspace digest."""
    from datetime import UTC, datetime

    from jinja2 import Environment, PackageLoader

    from reporadar.workspace import combined_digest_data, open_workspace_store

    store = open_workspace_store()
    try:
        if run_id is None:
            last_run = store.get_last_run()
            if last_run is None:
                error("No runs found. Run `rr workspace update` first.")
                raise SystemExit(1)
            run_id = last_run["run_id"]

        repos = store.get_workspace_repos()
        papers = combined_digest_data(store, run_id)

        env = Environment(
            loader=PackageLoader("reporadar", "templates"),
            keep_trailing_newline=True,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        template = env.get_template("workspace_digest.md.j2")
        content = template.render(
            generated_at=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
            run_id=run_id,
            total_papers=len(papers),
            total_repos=len(repos),
            papers=papers,
        )

        dest = Path(output_path or "workspace_digest.md")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")
        success(f"Workspace digest written to {dest}")
    finally:
        store.close()


@cli.command()
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to .reporadar.yml.",
)
@click.option(
    "--interval",
    default="6h",
    help="Update interval (e.g. 6h, 30m, 1d).",
)
@click.option(
    "--no-notify",
    is_flag=True,
    help="Disable desktop notifications.",
)
def watch(config_path: str | None, interval: str, no_notify: bool) -> None:
    """Continuously monitor for new papers.

    Runs update+digest cycles at the specified interval.
    Press Ctrl+C to stop.
    """
    from reporadar.watcher import parse_interval as _parse_interval
    from reporadar.watcher import watch_loop

    try:
        seconds = _parse_interval(interval)
    except ValueError:
        error(f"Invalid interval: {interval!r}. Use format like '6h', '30m', or '1d'.")
        raise SystemExit(1) from None

    cfg = _load_and_validate(config_path)
    cfg_path = config_path or str(Path(cfg.repo_path).resolve() / DEFAULT_CONFIG_NAME)

    info(f"Watching every {interval} (Ctrl+C to stop)...")
    try:
        watch_loop(cfg_path, seconds, notify=not no_notify)
    except KeyboardInterrupt:
        info("\nWatch stopped.")
