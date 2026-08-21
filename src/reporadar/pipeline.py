"""The collect -> enrich -> rank pipeline, in one place.

This is the pipeline every published number in this project describes. It lived inline in
`cli.update` as a ~700-line orchestrator while `watcher.py` and `workspace.py`
re-implemented its front half and stopped before most of it — no gate, no rescore, no
HyDE, no fusion, no embeddings, and arXiv as the only source whatever `sources:` said.
That is ROADMAP's Tier 0 "pipeline drift", and its cost was not duplication: a user whose
config read `triage.enabled: true` got an *ungated* digest from `rr watch`, which is the
configuration the benchmark scores at **-8.12** rather than the **+5.72** the config
promised. `stages.py` disclosed that gap. This closes it for `rr watch`.

**Output goes through a `Reporter`, not through `click`.** `rr update` talks to a terminal
and `rr watch` talks to a log file, and the previous split existed partly because the
orchestrator called `click`-flavoured helpers directly. Threading a reporter keeps the CLI
output byte-identical while letting the watcher receive the same messages as log records.

**What is deliberately still separate.** `rr workspace update` collects one shared pool
across many member repos under a single run id, then scores each member against it. That
is a genuinely different shape from one-repo-one-run, not the same code duplicated, so it
keeps the `stages.py` disclosure rather than a rushed unification — see ROADMAP.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol

from reporadar.collector import (
    CollectionError,
    build_queries,
    collect_by_ids,
    collect_papers,
    to_plain_keywords,
)
from reporadar.llm_client import LLMError
from reporadar.paper_id import dedup_id as _dedup_id
from reporadar.profiler import profile_repo
from reporadar.ranker import rank_papers
from reporadar.store import PaperStore, StoreError

# The only ranking weights feedback.compute_adjusted_weights learns; every other
# RankingConfig field is passed through untouched (stage 7).
_LEARNED_WEIGHTS = ("w_keyword", "w_category", "w_recency", "w_embedding", "w_citations")

# How many of the built queries reach each non-arXiv source. It was 5, an arbitrary number
# that withheld **50 of the 175 queries (28.6%)** the 25 benchmark repositories produce; 8 is
# the most any of them builds, so every one of their queries now goes out. Raised rather than
# removed because the cap is a real bound: it is one HTTP request per query per enabled
# source, and a keyless Semantic Scholar caller is already the first thing to see 429s.
#
# A seed-heavy config still truncates, and the truncation is a prefix — `build_queries` emits
# seeds, then bigram phrases, then single keywords, so the queries dropped first are the plain
# ones. That ordering is worth revisiting (`queries.bigrams` documents that phrase queries are
# safe on arXiv *because* a category clause catches a meaningless phrase, and that keyword
# sources have no such fallback) but changing the ORDER changes retrieval, and this project
# does not do that without a measurement. Recorded in the research doc instead.
KEYWORD_SOURCE_QUERIES = 8
# Every non-arXiv source this pipeline can fetch, named once and load-bearing below. The eval
# harness has its own dispatch and asserts against this list, because it silently lacked a
# `europepmc` branch for the two days after §13 shipped that source — so `--sources
# arxiv,europepmc` could not have measured the channel it was built to measure.
KEYWORD_SOURCES = ("semantic_scholar", "openalex", "biorxiv", "europepmc", "iacr", "dblp")


class Reporter(Protocol):
    """Where a stage's progress goes. `rr update` renders it, `rr watch` logs it."""

    def info(self, message: str) -> None: ...
    def warn(self, message: str) -> None: ...


@dataclass
class LogReporter:
    """Reporter for unattended runs. `info` is debug-level; `warn` stays visible."""

    logger: Any

    def info(self, message: str) -> None:
        self.logger.debug("%s", message.strip())

    def warn(self, message: str) -> None:
        self.logger.warning("%s", message.strip())


@dataclass
class PipelineResult:
    """Everything a caller needs to render, digest or report on the run.

    *stopped* is the reason the pipeline returned early, or None. It is a named reason
    rather than an empty result, because "no queries" and "no papers" call for different
    messages and a caller cannot tell them apart from `papers == []`.
    """

    stopped: str | None = None
    run_id: int | None = None
    queries: list[str] = field(default_factory=list)
    papers: list[dict[str, Any]] = field(default_factory=list)
    scores: list[dict[str, Any]] = field(default_factory=list)
    repo_profile: Any = None
    # The weights that actually produced `scores`. With feedback tuning on these are NOT
    # `cfg.ranking`, and explaining a score with the file's weights printed contributions
    # that did not sum to the printed total.
    ranking_cfg: Any = None
    new_count: int = 0
    seen_count: int = 0


@contextlib.contextmanager
def open_store(db_path: Path, report: Reporter) -> Iterator[PaperStore]:
    """Open a PaperStore, turning StoreError into a reported failure."""
    try:
        store = PaperStore(db_path)
    except StoreError as exc:
        report.warn(str(exc))
        raise
    try:
        yield store
    finally:
        store.close()


def run_pipeline(
    cfg: Any,
    *,
    repo_path: Path,
    db_path: Path,
    report: Reporter,
    verbose: bool = False,
    rebuild_embeddings: bool = False,
) -> PipelineResult:
    """Profile, collect, enrich, rank and gate. Raises CollectionError if arXiv fails."""
    # 1. Profile
    report.info(f"Profiling repo: {repo_path}")
    repo_profile = profile_repo(repo_path, profiler_cfg=cfg.profiler)
    report.info(
        f"  Found {len(repo_profile.keywords)} keywords, {len(repo_profile.anchors)} anchors"
    )

    # Nudge (never auto-enable) domain sources this repo's literature needs but
    # the config leaves off — an arXiv-only run silently misses those papers.
    from reporadar.sources.suggest import format_suggestion, suggest_sources

    for suggestion in suggest_sources(repo_profile, cfg.sources):
        report.warn(f"  {format_suggestion(suggestion)}")

    # 2. Build queries
    queries = build_queries(repo_profile, cfg.queries, cfg.arxiv)
    report.info(f"  Built {len(queries)} queries")

    if not queries:
        if cfg.queries.redact:
            # Redaction can empty the query set outright, and blaming a missing README
            # for it would send the user looking in the wrong place entirely.
            report.warn(
                "No queries to run: `privacy.redact` removed every search term. "
                "Run `rr audit` to see what is left, or narrow the redaction list."
            )
        else:
            report.warn(
                "No queries to run. Add seed queries to .reporadar.yml or ensure "
                "the repo has a README."
            )
        return PipelineResult(stopped="no_queries", repo_profile=repo_profile)

    # 3. Collect
    report.info("Fetching papers from arXiv...")
    papers = collect_papers(queries, cfg.arxiv)
    report.info(f"  Fetched {len(papers)} unique papers")

    papers = _collect_extra_sources(
        cfg, papers, queries, repo_profile, db_path=db_path, report=report
    )

    if not papers:
        report.warn("No new papers found.")
        return PipelineResult(stopped="no_papers", queries=queries, repo_profile=repo_profile)

    # 4. Store
    with open_store(db_path, report) as store:
        new_count, seen_count = store.upsert_papers(papers)
        run_id = store.record_run(
            queries_used=queries,
            papers_new=new_count,
            papers_seen=seen_count,
        )

        signals = _enrich(
            cfg,
            papers,
            store=store,
            repo_path=repo_path,
            report=report,
            verbose=verbose,
            rebuild_embeddings=rebuild_embeddings,
        )

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
                        report.info("  Feedback: adjusted ranking weights from user ratings.")
                        for k, v in new_weights.items():
                            report.info(f"    {k}: {v:.4f}")
            except Exception as exc:
                if verbose:
                    report.info(f"  Feedback weight adjustment skipped: {exc}")

        # 8. Rank
        report.info("Scoring papers...")
        scores = rank_papers(
            papers,
            repo_profile,
            ranking_cfg,
            cfg.queries,
            cfg.arxiv.categories,
            cfg.arxiv.lookback_days,
            **signals,
        )
        # 8b. Hybrid retrieval (roadmap #4): fuse the heuristic order with a BM25
        #     lexical order via RRF, so a paper buried on vocabulary mismatch can
        #     surface. Sets rrf_score (persisted); the digest orders by it.
        if cfg.ranking.hybrid:
            from reporadar.retrieval import hybrid_reorder

            scores = hybrid_reorder(scores, papers, repo_profile)
            report.info("  Hybrid retrieval: fused heuristic + BM25 ranking (RRF).")
        store.save_scores(run_id, scores)

        _triage(
            cfg,
            papers,
            scores,
            store=store,
            run_id=run_id,
            repo_profile=repo_profile,
            report=report,
        )

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
                    report.info("Enriching top papers with Hugging Face Papers data...")
                    token = cfg.enrichment.hf_token or None
                    enrichments = fetch_enrichments_batch(top_ids, token=token)
                    if enrichments:
                        store.save_enrichments(enrichments)
                        report.info(f"  Enrichment data for {len(enrichments)} papers.")
                    else:
                        report.info("  No enrichment data found.")
            except Exception as exc:
                report.info(f"  Enrichment failed: {exc}")

    return PipelineResult(
        run_id=run_id,
        queries=queries,
        papers=papers,
        scores=scores,
        repo_profile=repo_profile,
        ranking_cfg=ranking_cfg,
        new_count=new_count,
        seen_count=seen_count,
    )


def _collect_extra_sources(
    cfg: Any,
    papers: list[dict[str, Any]],
    queries: list[str],
    repo_profile: Any,
    *,
    db_path: Path,
    report: Reporter,
) -> list[dict[str, Any]]:
    """Stages 3a-3f: HyDE, the four non-arXiv sources, and learned recommendations."""
    # 3a. HyDE dense discovery — the channel keyword search structurally cannot be.
    #     Queries built from a repository describe what it HAS; the useful paper describes
    #     what it should ADOPT, and across nine benchmark repos the keyword path reached 0
    #     of 24 known-good papers. Searching by a hypothesised abstract sidesteps that: the
    #     query is written in the literature's register by construction. Measured 27/48,
    #     with 15 targets no other channel reaches. Offline once the index is synced.
    if cfg.hyde.enabled:
        from reporadar import hyde

        index_dir = Path(cfg.hyde.index_dir).expanduser()
        try:
            age = hyde.index_age_days(index_dir)
            if age is not None and age > cfg.hyde.stale_after_days:
                report.warn(
                    f"  HyDE index is {age:.0f} days old — it is a periodically republished "
                    f"mirror, so recent papers may be missing. Refresh: rr sync-index --refresh"
                )
            report.info("Discovering papers by hypothesis (HyDE)...")
            hyde_ids = hyde.discover(
                repo_profile,
                cfg.suggestions,
                index_dir,
                n_hypotheses=cfg.hyde.n_hypotheses,
                top_k=cfg.hyde.top_k,
                model_name=cfg.hyde.model,
                verify=cfg.hyde.verify_encoder,
                on_progress=lambda msg: report.info(f"  {msg}"),
            )
            # Same normaliser as every other merge in this function. It used to be a bare
            # `split("v")[0]` here and `_dedup_id` five lines down, which is one invariant
            # with two implementations — the shape that produced C-9 and C-12.
            known = {_dedup_id(p["arxiv_id"]) for p in papers}
            fresh = [pid for pid in hyde_ids if pid not in known]
            hyde_papers = collect_by_ids(fresh)
            papers.extend(p for p in hyde_papers if _dedup_id(p["arxiv_id"]) not in known)
            report.info(
                f"  HyDE: {len(hyde_ids)} candidates, {len(fresh)} new, "
                f"{len(hyde_papers)} resolved."
            )
        except hyde.HydeError as exc:
            # Loud, and never silently degraded to the keyword-only path: that path is the
            # one measured at 0/24, so a user who enabled HyDE and got it must be told.
            report.warn(f"  HyDE discovery unavailable: {exc}")
        except (CollectionError, LLMError) as exc:
            report.warn(f"  HyDE discovery failed: {exc}")

    # 3b-3e. The non-arXiv sources.
    #
    # One merge rule, five call sites — `_merge_source` below. Not a table of module
    # *strings* driving `importlib`: that was the first version of this refactor, and it
    # would have blinded `tests/test_stages.py`, which reads the import graph to prove the
    # drift warning tells the truth. A guard that cannot see the import cannot check it.
    # So each fetcher keeps a real, lazy `from ... import` inside its own function.
    plain = [to_plain_keywords(q) for q in queries[:KEYWORD_SOURCE_QUERIES]]
    lookback = cfg.arxiv.lookback_days

    def _semantic_scholar() -> list[dict[str, Any]]:
        from reporadar.sources.semantic_scholar import collect_papers as ss_collect

        return ss_collect(
            plain, api_key=cfg.semantic_scholar.api_key or None, lookback_days=lookback
        )

    def _openalex() -> list[dict[str, Any]]:
        from reporadar.sources.openalex import collect_papers as oa_collect

        return oa_collect(
            plain,
            email=cfg.openalex.email or None,
            lookback_days=lookback,
            api_key=cfg.openalex.api_key or None,
        )

    def _biorxiv() -> list[dict[str, Any]]:
        from reporadar.sources.biorxiv import collect_papers as bx_collect

        return bx_collect(plain, lookback_days=lookback)

    def _europepmc() -> list[dict[str, Any]]:
        from reporadar.sources.europepmc import collect_papers as epmc_collect

        # No `email=`. Europe PMC accepts one as politeness and works without it, and
        # `openalex.email` was given to this project for OpenAlex's polite pool — forwarding
        # it to a second service is a data flow the user did not agree to and the privacy
        # registry would have to declare. Nothing is gained that is worth that.
        return epmc_collect(plain, lookback_days=lookback)

    def _iacr() -> list[dict[str, Any]]:
        from reporadar.sources.iacr import collect_papers as iacr_collect

        return iacr_collect(plain, lookback_days=lookback)

    def _dblp() -> list[dict[str, Any]]:
        from reporadar.sources.dblp import collect_papers as dblp_collect

        return dblp_collect(plain, lookback_days=lookback)

    fetchers = {
        "semantic_scholar": ("Semantic Scholar", _semantic_scholar),
        "openalex": ("OpenAlex", _openalex),
        "biorxiv": ("bioRxiv", _biorxiv),
        "europepmc": ("Europe PMC (bioRxiv/medRxiv)", _europepmc),
        "iacr": ("IACR ePrint", _iacr),
        "dblp": ("DBLP", _dblp),
    }
    # Driven by KEYWORD_SOURCES so the constant cannot drift from what actually runs: a key
    # listed there with no fetcher raises here rather than being quietly skipped.
    for key in KEYWORD_SOURCES:
        label, fetch = fetchers[key]
        if key in cfg.sources:
            _merge_source(papers, label, fetch, report=report)

    # 3f. Learned recommendations from your ratings/stars (Feature 5, optional).
    #     Merged into the candidate pool so the local ranker re-filters them — the
    #     API is repo-agnostic and can return off-topic results for a niche seed.
    if cfg.recommendations.enabled:
        try:
            from reporadar.sources.s2_recommendations import fetch_recommendations

            with open_store(db_path, report) as rec_store:
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
                report.info("Fetching learned recommendations from your ratings...")
                recs = fetch_recommendations(
                    positives,
                    negatives,
                    limit=cfg.recommendations.limit,
                    max_seeds=cfg.recommendations.max_seeds,
                    api_key=cfg.semantic_scholar.api_key or None,
                )
                if recs is None:
                    report.warn(
                        "  Recommendations unavailable (Semantic Scholar error) — skipping."
                    )
                else:
                    existing_ids = {_dedup_id(p["arxiv_id"]) for p in papers}
                    new_recs = [p for p in recs if _dedup_id(p["arxiv_id"]) not in existing_ids]
                    papers.extend(new_recs)
                    report.info(f"  {len(new_recs)} recommended papers (re-ranked locally)")
            else:
                report.info("  No rated/starred papers yet — skipping recommendations.")
        except Exception as exc:
            report.info(f"  Recommendations failed: {exc}")

    return papers


def _merge_source(
    papers: list[dict[str, Any]],
    label: str,
    fetch: Any,
    *,
    report: Reporter,
) -> None:
    """Fetch from one non-arXiv source and merge in place, version-insensitively.

    arXiv hands back `2401.12345` where another source may say `2401.12345v2`, and
    matching raw ids lets the same paper through twice. That defect (C-12) reached three
    separate call sites because each merge had its own copy of the rule; this is the one
    copy. A source that fails is reported and skipped — never fatal, since arXiv has
    already produced a pool by this point.
    """
    try:
        report.info(f"Fetching papers from {label}...")
        fetched = fetch()
        existing_ids = {_dedup_id(p["arxiv_id"]) for p in papers}
        fresh = [p for p in fetched if _dedup_id(p["arxiv_id"]) not in existing_ids]
        papers.extend(fresh)
        report.info(f"  {len(fresh)} additional papers from {label}")
    except Exception as exc:
        report.info(f"  {label} collection failed: {exc}")


def _enrich(
    cfg: Any,
    papers: list[dict[str, Any]],
    *,
    store: PaperStore,
    repo_path: Path,
    report: Reporter,
    verbose: bool,
    rebuild_embeddings: bool,
) -> dict[str, Any]:
    """Stages 5-6f. Returns the keyword arguments `rank_papers` takes for each signal."""
    # 5. Compute embeddings (optional)
    repo_embedding = None
    if cfg.ranking.w_embedding > 0:
        try:
            from reporadar.embeddings import EMBEDDINGS_AVAILABLE, compute_repo_embedding

            if EMBEDDINGS_AVAILABLE:
                report.info("Computing embedding similarity...")
                repo_embedding = compute_repo_embedding(repo_path)
                if repo_embedding is None:
                    report.info("  No repo text found for embedding.")
                else:
                    report.info("  Embedding similarity enabled.")
            else:
                report.info("  Embedding similarity not available (install sentence-transformers).")
        except ImportError:
            report.info("  Embedding similarity not available (install sentence-transformers).")

    # 5b. Per-paper embeddings, cached across runs (compute once, not per run).
    paper_embeddings = None
    if repo_embedding is not None:
        try:
            from reporadar.embedding_cache import default_model, embed_papers_cached

            if rebuild_embeddings:
                # Clear every cached vector (MiniLM + SPECTER2 + miss markers)
                # so unresolvable papers get retried too.
                cleared = store.clear_embeddings()
                report.info(f"  Rebuilding embedding cache ({cleared} cleared).")
            before = store.embedding_count(default_model())
            paper_embeddings = embed_papers_cached(store, papers)
            newly = store.embedding_count(default_model()) - before
            report.info(
                f"  Embedding cache: {newly} encoded, {len(paper_embeddings) - newly} reused."
            )
        except Exception as exc:
            if verbose:
                report.info(f"  Embedding cache skipped: {exc}")
            paper_embeddings = None

    # 6. Citation counts (optional)
    citation_scores = None
    if cfg.ranking.w_citations > 0:
        report.info("Fetching citation counts...")
        try:
            from reporadar.citations import fetch_citation_counts, normalize_citations

            # Only real arXiv ids resolve at S2; synthetic ones (ss:/dblp:/
            # biorxiv:/oa:) would just burn slots in the 500-id batch.
            arxiv_ids = [p["arxiv_id"] for p in papers if ":" not in p["arxiv_id"]]
            api_key = cfg.semantic_scholar.api_key or None
            raw_counts = fetch_citation_counts(arxiv_ids, api_key=api_key)
            if raw_counts:
                citation_scores = normalize_citations(raw_counts)
                report.info(f"  Citation data for {len(citation_scores)} papers.")
            else:
                report.info("  No citation data available.")
        except Exception as exc:
            report.info(f"  Citation lookup failed: {exc}")

    # 6b. Citation proximity — "extends work you starred" (Feature 8, optional)
    citation_proximity = None
    if cfg.ranking.w_citation_proximity > 0:
        try:
            from reporadar.citation_graph import build_seed_set, find_citation_links
            from reporadar.citations import fetch_references

            seeds = build_seed_set(store)
            if seeds:
                report.info("Checking which papers extend your starred/rated work...")
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
                    report.info(f"  {len(links)} paper(s) cite work you starred.")
                else:
                    report.info("  None of this run's papers cite your starred work.")
        except Exception as exc:
            report.info(f"  Citation-proximity check failed: {exc}")

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
                report.info("  No starred or highly-rated papers yet — skipping SPECTER2.")
            elif not resolvable:
                report.info(
                    "  Your liked papers are all non-arXiv — SPECTER2 has nothing to seed on."
                )
            else:
                report.info("Scoring SPECTER2 similarity to your starred/rated papers...")
                specter = specter_score_papers(
                    store, papers, api_key=cfg.semantic_scholar.api_key or None
                )
                if specter:
                    cached = store.embedding_count(SPECTER_MODEL)
                    report.info(
                        f"  SPECTER2 scored {len(specter)} papers ({cached} vectors cached)."
                    )
                else:
                    report.info("  No usable SPECTER2 signal for this run's papers.")
        except Exception as exc:
            report.info(f"  SPECTER2 scoring failed: {exc}")
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
                report.info(f"  Community signal (HF upvotes) for {len(community)} papers.")
            else:
                report.info("  No cached HF upvotes yet — community signal starts next run.")
        except Exception as exc:
            report.info(f"  Community signal failed: {exc}")
            community = None

    withdrawn = _integrity(cfg, papers, store=store, report=report)

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
                    [(aid, "hn", str(s["points"]), s["story_url"]) for aid, s in stories.items()]
                )
                report.info(f"  Discussed on Hacker News: {len(stories)} paper(s).")
            else:
                report.info("  No Hacker News discussion found for this run's papers.")
        except Exception as exc:
            report.info(f"  Hacker News lookup failed: {exc}")
            attention = None

    return {
        "repo_embedding": repo_embedding,
        "citation_scores": citation_scores,
        "paper_embeddings": paper_embeddings,
        "citation_proximity": citation_proximity,
        "specter": specter,
        "community": community,
        "attention": attention,
        "withdrawn": withdrawn,
    }


def _integrity(
    cfg: Any, papers: list[dict[str, Any]], *, store: PaperStore, report: Reporter
) -> set[str]:
    """Stage 6e. Is any candidate a withdrawn paper (Feature 9)? On by default —
    recommending retracted work is the worst thing the ranker can do. Re-checked every
    run, not just at collect time: a paper can be withdrawn days after it was ingested
    (observed in a real corpus)."""
    withdrawn: set[str] = set()
    if not cfg.signals.integrity:
        return withdrawn
    try:
        from reporadar.signals.integrity import fetch_comments, find_withdrawn, stale_ids

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
        to_check = stale_ids(resolvable, {aid: row["checked_at"] for aid, row in previous.items()})
        comments = fetch_comments(to_check) if to_check else {}
        flags.update(find_withdrawn(papers, comments))

        if to_check and comments:
            # Record a row for every id asked about, not just the answered
            # ones: arXiv silently drops ids it does not know, and those would
            # otherwise stay "never checked" forever and consume the whole
            # per-run budget on every future run, starving the real re-checks.
            store.save_signals([(aid, "withdrawn", flags.get(aid), None) for aid in to_check])
        # Trust a stored flag for anything not re-checked this run.
        for aid, row in previous.items():
            if row["value"] and aid not in comments:
                flags.setdefault(aid, row["value"])
        withdrawn = set(flags)

        if to_check and not comments:
            # fetch_comments swallows per-batch failures, so an arXiv outage
            # would otherwise print a clean "0 withdrawn" and hide the gap.
            report.warn(
                f"  Integrity check could not reach arXiv for {len(to_check)} "
                "paper(s); only their stored title/abstract was checked."
            )
        if withdrawn:
            report.warn(f"  {len(withdrawn)} paper(s) withdrawn by their authors - demoted.")
        else:
            report.info(f"  Integrity check: no withdrawn papers ({len(comments)} looked up).")
    except Exception as exc:
        report.info(f"  Integrity check failed: {exc}")
        withdrawn = set()
    return withdrawn


def _triage(
    cfg: Any,
    papers: list[dict[str, Any]],
    scores: list[dict[str, Any]],
    *,
    store: PaperStore,
    run_id: int,
    repo_profile: Any,
    report: Reporter,
) -> None:
    """Stages 8c-8d. The gate, and the fine-scale rescore of the band that sits on it."""
    # 8c. LLM actionability triage (Feature 6) — score the top papers for
    # whether they could genuinely improve THIS repo, so the digest can gate
    # its Top Picks on applicability instead of the raw heuristic score.
    if cfg.triage.enabled and cfg.suggestions.provider in ("ollama", "claude"):
        report.info(f"Triaging top {cfg.triage.top_k} papers for actionable relevance...")
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
                    1 for v in llm_scores.values() if v["llm_score"] >= cfg.triage.min_actionable
                )
                report.info(
                    f"  Triaged {len(llm_scores)} papers; {n_act} actionable "
                    f"(score >= {cfg.triage.min_actionable})."
                )
            else:
                report.info("  Triage produced no scores (all calls failed).")

            # 8d. Fine-scale rescore of the band that sits exactly at the gate's
            #     threshold. The 0-3 gate is near-binary, and within its score-2
            #     band the share of genuinely actionable papers ran 0%-100% by repo
            #     — the single biggest remaining error source in the digest. Asking
            #     the same question on a 0-9 scale and reading the answer's token
            #     distribution orders that band (AUC 0.84) and lifts mean net@2 from
            #     +1.91 to +2.91 on the 22-repo benchmark.
            if llm_scores and cfg.triage.finescale.enabled:
                from reporadar.digest import digest_window
                from reporadar.finescale import enough_scored, score_papers

                # Scope the band to papers the digest can still SHOW. Everything
                # outside `digest_window` is dropped before tiering, so rescoring it
                # buys nothing — and the waste tripled on 2026-08-14 when the measured
                # depth experiment moved `triage.top_k` from 15 to 50.
                #
                # The window comes from `store.get_scores_for_run`, which is the exact
                # list `rr digest` reads: same ordering (`COALESCE(rrf_score,
                # score_total) DESC`), same llm_score join, same withdrawn flag. Read
                # back rather than rebuilt from `scores` in memory, because rebuilding
                # it would be a second implementation of the digest's input — and this
                # project has paid for that shape four times (C-9, C-12, C-14).
                #
                # Known residual: `rr update` has no `--since`, so a later
                # `rr digest --since` can promote a band paper past this window, and it
                # will have no `finescale_p` and reach Maybe rather than Top Picks.
                # Conservative, and the same rule ungated papers already follow — see
                # tests/test_finescale_window.py, which pins it.
                showable = {
                    r["arxiv_id"]
                    for r in digest_window(
                        store.get_scores_for_run(run_id),
                        cfg.output.top_n,
                        triage_threshold=cfg.triage.min_actionable,
                        rerank=cfg.triage.rerank,
                    )[0]
                }
                band = [
                    papers_by_id[pid]
                    for pid, v in llm_scores.items()
                    if v["llm_score"] == cfg.triage.min_actionable
                    and pid in papers_by_id
                    and pid in showable
                ]
                if band:
                    report.info(f"  Rescoring {len(band)} band papers on the fine scale...")
                    fine = score_papers(band, repo_profile, cfg.triage.finescale)
                    if enough_scored(
                        len(fine), len(band), cfg.triage.finescale.min_success_fraction
                    ):
                        # Persist ONLY when the gate will apply, so that a stored
                        # `finescale_p` means "this run was fine-scale gated" for
                        # every later reader (archive, notify, watcher). Half-written
                        # scores from a failed run would make those readers apply a
                        # gate `rr update` itself declined to apply.
                        store.save_finescale_scores(run_id, fine)
                        n_pass = sum(
                            1
                            for v in fine.values()
                            if v["finescale_p"] >= cfg.triage.finescale.threshold
                        )
                        report.info(
                            f"  Fine-scale: {n_pass}/{len(band)} band papers clear "
                            f"P >= {cfg.triage.finescale.threshold:.2f}."
                        )
                    else:
                        # Loudly, not silently: applying the gate here would demote
                        # every band paper and produce an abstention that looks
                        # deliberate. A broken key must not read as "nothing good".
                        report.warn(
                            f"  Fine-scale scored only {len(fine)}/{len(band)} papers "
                            f"— skipping the gate for this run (check OPENAI_API_KEY)."
                        )
        except Exception as exc:
            report.info(f"  Triage failed: {exc}")
    elif cfg.triage.enabled:
        # The two-field trap: `triage.enabled: true` alone gates nothing, and the
        # config gives no hint of the second field. Name it rather than say "skipping".
        report.warn("  Triage is enabled but `suggestions.provider` is 'template', so NO")
        report.warn("  actionability gate ran. Set `suggestions.provider: claude` (or ollama)")
        report.warn("  Enabling the gate takes BOTH fields.")
    else:
        # Said once per run, because this is the difference between the configuration
        # the benchmark scores at -8.12 and the one it scores at +5.72, and a user who
        # never opens the config file would otherwise never learn which one they have.
        report.info(
            "  Ranking by keyword overlap only (no actionability gate). "
            "`rr init --measured` writes the configuration behind the published "
            "+5.72; this one measures -8.12 (precision 0.379 against 0.892)."
        )
