"""Digest generation — render scored papers into a Markdown file."""

from __future__ import annotations

import csv
import io
import json as json_mod
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from jinja2 import Environment, PackageLoader

from reporadar.notify import DigestSummary
from reporadar.store import PaperStore
from reporadar.suggestions import enrich_papers_with_suggestions

TEMPLATES_DIR = Path(__file__).parent / "templates"

# Score thresholds for categorizing papers into tiers.
# Papers with score >= TOP_THRESHOLD go into "Top Picks",
# >= MAYBE_THRESHOLD into "Maybe Relevant", rest into "Muted".
TOP_THRESHOLD = 0.5
MAYBE_THRESHOLD = 0.2


def _load_template(template_name: str = "digest.md.j2") -> Any:
    """Load a Jinja2 template from the templates directory (no autoescaping).

    Used for the text formats (Markdown, RSS) and the legacy markdown-in-<pre>
    HTML wrapper, where autoescaping would corrupt the intended output.
    """
    env = Environment(
        loader=PackageLoader("reporadar", "templates"),
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    return env.get_template(template_name)


def _safe_url(url: Any) -> str:
    """Return *url* only if it is an ``http(s)`` link, else ``#``.

    Paper URLs come from external source APIs; autoescaping neutralizes markup
    but not a ``javascript:``/``data:`` scheme (no ``<>&"`` to escape). Since the
    digest is published to GitHub Pages, allow-list the scheme so a hostile URL
    can't become an executable link.
    """
    s = str(url or "")
    return s if s.startswith(("http://", "https://")) else "#"


def _load_html_template(template_name: str) -> Any:
    """Load an HTML template with autoescaping on.

    The rendered-HTML digest and the Pages archive index interpolate live paper
    text (titles, abstracts, author names) into markup, so autoescaping is
    required to stay correct and safe on ``<``/``&`` in that content.
    """
    env = Environment(
        loader=PackageLoader("reporadar", "templates"),
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
        autoescape=True,
    )
    env.filters["safe_url"] = _safe_url
    return env.get_template(template_name)


def filter_since(
    scored_papers: list[dict[str, Any]], since_days: int | None
) -> list[dict[str, Any]]:
    """Keep only papers published within the last *since_days* days.

    A falsy or non-positive *since_days* means no filtering (returns the input
    unchanged). Papers with a missing/unparseable ``published`` date are kept.
    """
    if not since_days or since_days <= 0:
        return scored_papers
    cutoff = (datetime.now(UTC) - timedelta(days=since_days)).strftime("%Y-%m-%d")
    kept: list[dict[str, Any]] = []
    for p in scored_papers:
        pub = (p.get("published") or "")[:10]
        # Keep papers with an unknown date rather than silently hiding them.
        if not pub or pub >= cutoff:
            kept.append(p)
    return kept


def categorize_papers(
    scored_papers: list[dict[str, Any]],
    top_n: int = 15,
    top_threshold: float = TOP_THRESHOLD,
    maybe_threshold: float = MAYBE_THRESHOLD,
    triage_threshold: int | None = None,
    rerank: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Split scored papers into three tiers: top_picks, maybe_relevant, muted.

    Papers are assumed to be sorted by score descending. Only the first
    *top_n* papers are included across all tiers.

    When *triage_threshold* is set, papers that carry an LLM actionability score
    (``llm_score``) are tiered by it — ``>= triage_threshold`` = Top Pick, ``>= 1``
    = Maybe, ``0`` = Muted — so the digest abstains from "Top Picks" unless a
    paper is judged genuinely applicable. Papers without an ``llm_score`` fall
    back to the heuristic ``score_total`` thresholds.

    When *rerank* is set (and triage scores are present), papers are reordered by
    ``llm_score`` *before* the *top_n* cut, so an actionable paper the heuristic
    ranker buried below the window isn't dropped before it can be a Top Pick.
    """
    if rerank and triage_threshold is not None:
        from reporadar.triage import rerank_by_actionability

        scored_papers = rerank_by_actionability(scored_papers)
    limited = scored_papers[:top_n]

    top_picks = []
    maybe_relevant = []
    muted = []

    for paper in limited:
        llm = paper.get("llm_score")
        if triage_threshold is not None and llm is not None:
            if llm >= triage_threshold:
                top_picks.append(paper)
            elif llm >= 1:
                maybe_relevant.append(paper)
            else:
                muted.append(paper)
            continue
        score = paper["score_total"]
        if score >= top_threshold:
            top_picks.append(paper)
        elif score >= maybe_threshold:
            maybe_relevant.append(paper)
        else:
            muted.append(paper)

    return top_picks, maybe_relevant, muted


def _build_digest_context(
    store: PaperStore,
    run_id: int,
    top_n: int = 15,
    diff: bool = False,
    suggestions_config: Any | None = None,
    profile: Any | None = None,
    since_days: int | None = None,
    triage_threshold: int | None = None,
    rerank: bool = False,
) -> dict[str, Any]:
    """Build the shared template context for a run (Markdown and HTML render it)."""
    scored = filter_since(store.get_scores_for_run(run_id), since_days)
    # Resolve the run being rendered (not necessarily the latest — `--run-id N`),
    # so the header stats match the papers below them.
    run = store.get_run(run_id)

    # Diff mode: determine which papers are new vs. carried over
    diff_mode = False
    if diff:
        diff_mode = True
        prev_id = store.get_previous_run_id(run_id)
        prev_ids = store.get_scored_paper_ids_for_run(prev_id) if prev_id is not None else set()
        for paper in scored:
            paper["is_new"] = paper["arxiv_id"] not in prev_ids

    top_picks, maybe_relevant, muted = categorize_papers(
        scored, top_n=top_n, triage_threshold=triage_threshold, rerank=rerank
    )
    enrich_papers_with_suggestions(top_picks, config=suggestions_config, profile=profile)

    has_embeddings = any(p.get("embedding_score") is not None for p in scored)
    has_citations = any(p.get("citation_score") is not None for p in scored)
    has_enrichments = any(p.get("has_code") or p.get("datasets") for p in scored)

    # Trend detection
    trends: list[dict[str, Any]] = []
    try:
        from reporadar.trends import detect_trends

        trends = detect_trends(store, run_id)
    except Exception:
        pass

    # Recommended papers (from feedback loop)
    recommended: list[dict[str, Any]] = []
    try:
        from reporadar.feedback import find_similar_to_highly_rated

        ratings = store.get_all_ratings()
        if ratings:
            rated_papers = {}
            for arxiv_id, rating in ratings.items():
                p = store.get_paper(arxiv_id)
                if p:
                    rated_papers[arxiv_id] = {**p, "rating": rating}
            if rated_papers:
                recommended = find_similar_to_highly_rated(scored, rated_papers)
    except Exception:
        pass

    # "Extends work you starred" (Feature 8): papers in this run that cite a
    # starred/highly-rated paper (re-checked against the CURRENT seed set).
    extends_starred: list[dict[str, Any]] = []
    try:
        from reporadar.citation_graph import build_seed_set

        seeds = build_seed_set(store)
        if seeds:
            edges = store.get_citations_for([p["arxiv_id"] for p in scored])
            for paper in scored:  # scored is sorted by score, best-first
                cites = sorted(c for c in edges.get(paper["arxiv_id"], []) if c in seeds)
                if cites:
                    paper["cites_starred"] = cites
                    extends_starred.append(
                        {"title": paper["title"], "url": paper["url"], "cites": cites}
                    )
    except Exception:
        pass

    return {
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "run_id": run_id,
        "papers_new": run["papers_new"] if run else 0,
        "papers_seen": run["papers_seen"] if run else 0,
        "total_scored": len(scored),
        "queries_used": run["queries_used"] if run else [],
        "top_picks": top_picks,
        "maybe_relevant": maybe_relevant,
        "muted": muted,
        "diff_mode": diff_mode,
        "has_embeddings": has_embeddings,
        "has_citations": has_citations,
        "has_enrichments": has_enrichments,
        "trends": trends,
        "recommended": recommended,
        "extends_starred": extends_starred,
    }


def generate_digest(
    store: PaperStore,
    run_id: int,
    top_n: int = 15,
    diff: bool = False,
    suggestions_config: Any | None = None,
    profile: Any | None = None,
    since_days: int | None = None,
    triage_threshold: int | None = None,
    rerank: bool = False,
) -> str:
    """Generate digest Markdown content for a given run.

    If *diff* is True, marks each paper as new or carried over from the
    previous run by setting ``is_new`` on each paper dict. If *since_days*
    is given, only papers published within that window are included.

    Returns the rendered Markdown string.
    """
    context = _build_digest_context(
        store,
        run_id,
        top_n=top_n,
        diff=diff,
        suggestions_config=suggestions_config,
        profile=profile,
        since_days=since_days,
        triage_threshold=triage_threshold,
        rerank=rerank,
    )
    rendered: str = _load_template().render(**context)
    return rendered


def generate_digest_html(
    store: PaperStore,
    run_id: int,
    top_n: int = 15,
    diff: bool = False,
    suggestions_config: Any | None = None,
    profile: Any | None = None,
    since_days: int | None = None,
    triage_threshold: int | None = None,
    rerank: bool = False,
) -> str:
    """Generate a fully rendered HTML digest page (not markdown-in-<pre>).

    Renders the same context as :func:`generate_digest` through an autoescaped
    HTML template, producing a browser-ready page suitable for GitHub Pages.
    """
    context = _build_digest_context(
        store,
        run_id,
        top_n=top_n,
        diff=diff,
        suggestions_config=suggestions_config,
        profile=profile,
        since_days=since_days,
        triage_threshold=triage_threshold,
        rerank=rerank,
    )
    rendered: str = _load_html_template("digest_page.html.j2").render(**context)
    return rendered


def markdown_to_html(md_content: str) -> str:
    """Wrap rendered Markdown in a basic HTML page.

    Uses a simple CSS stylesheet for readability. This avoids adding a
    Markdown-to-HTML library dependency — the output is raw Markdown
    inside <pre> tags with some basic styling. For a richer rendering,
    users can pipe the .md through any Markdown renderer.
    """
    template = _load_template("digest.html.j2")
    rendered: str = template.render(markdown_content=md_content)
    return rendered


def generate_digest_json(
    store: PaperStore,
    run_id: int,
    top_n: int = 15,
    diff: bool = False,
    suggestions_config: Any | None = None,
    profile: Any | None = None,
    since_days: int | None = None,
    triage_threshold: int | None = None,
    rerank: bool = False,
) -> str:
    """Generate digest as a JSON string.

    Returns a JSON string with top_picks, maybe_relevant, and muted tiers.
    """
    scored = filter_since(store.get_scores_for_run(run_id), since_days)
    run = store.get_run(run_id)

    if diff:
        prev_id = store.get_previous_run_id(run_id)
        prev_ids = store.get_scored_paper_ids_for_run(prev_id) if prev_id is not None else set()
        for paper in scored:
            paper["is_new"] = paper["arxiv_id"] not in prev_ids

    top_picks, maybe_relevant, muted = categorize_papers(
        scored, top_n=top_n, triage_threshold=triage_threshold, rerank=rerank
    )
    enrich_papers_with_suggestions(top_picks, config=suggestions_config, profile=profile)

    payload: str = json_mod.dumps(
        {
            "generated_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
            "run_id": run_id,
            "papers_new": run["papers_new"] if run else 0,
            "papers_seen": run["papers_seen"] if run else 0,
            "top_picks": top_picks,
            "maybe_relevant": maybe_relevant,
            "muted": muted,
        },
        indent=2,
        default=str,
    )
    return payload


_CSV_FIELDS = [
    "arxiv_id",
    "title",
    "score_total",
    "keyword_score",
    "category_score",
    "recency_score",
    "embedding_score",
    "citation_score",
    "tier",
    "authors",
    "categories",
    "published",
    "url",
    "has_code",
    "datasets",
    "models",
    "upvotes",
]


def generate_digest_csv(
    store: PaperStore,
    run_id: int,
    top_n: int = 15,
    diff: bool = False,
    since_days: int | None = None,
    triage_threshold: int | None = None,
    rerank: bool = False,
) -> str:
    """Generate digest as a CSV string."""
    scored = filter_since(store.get_scores_for_run(run_id), since_days)
    top_picks, maybe_relevant, muted = categorize_papers(
        scored, top_n=top_n, triage_threshold=triage_threshold, rerank=rerank
    )

    # Tag each paper with its tier
    for p in top_picks:
        p["tier"] = "top_pick"
    for p in maybe_relevant:
        p["tier"] = "maybe_relevant"
    for p in muted:
        p["tier"] = "muted"

    all_papers = top_picks + maybe_relevant + muted

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=_CSV_FIELDS, extrasaction="ignore")
    writer.writeheader()
    for p in all_papers:
        row = dict(p)
        # Flatten list fields
        row["authors"] = "; ".join(p.get("authors", []))
        row["categories"] = "; ".join(p.get("categories", []))
        row["datasets"] = "; ".join(p.get("datasets", []))
        row["models"] = "; ".join(p.get("models", []))
        writer.writerow(row)

    return output.getvalue()


def generate_digest_rss(
    store: PaperStore,
    run_id: int,
    top_n: int = 15,
    diff: bool = False,
    since_days: int | None = None,
    triage_threshold: int | None = None,
    rerank: bool = False,
) -> str:
    """Generate digest as an RSS 2.0 XML string."""
    scored = filter_since(store.get_scores_for_run(run_id), since_days)
    top_picks, maybe_relevant, muted = categorize_papers(
        scored, top_n=top_n, triage_threshold=triage_threshold, rerank=rerank
    )
    all_papers = top_picks + maybe_relevant + muted

    template = _load_template("digest.rss.xml.j2")
    rendered: str = template.render(
        generated_at=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        run_id=run_id,
        papers=all_papers,
    )
    return rendered


def write_digest(
    store: PaperStore,
    run_id: int,
    output_path: str | Path,
    top_n: int = 15,
    fmt: str = "md",
    diff: bool = False,
    suggestions_config: Any | None = None,
    profile: Any | None = None,
    since_days: int | None = None,
    triage_threshold: int | None = None,
    rerank: bool = False,
) -> tuple[Path, DigestSummary | None]:
    """Generate and write the digest to a file.

    *fmt* can be ``"md"``, ``"html"``, ``"json"``, ``"csv"``, or ``"rss"``.
    *since_days*, when given, limits output to papers published in that window.
    *triage_threshold*, when given, tiers Top Picks by LLM actionability score.
    Returns a tuple of (output_path, DigestSummary).
    """
    output_path = Path(output_path)

    if fmt == "json":
        content = generate_digest_json(
            store,
            run_id,
            top_n=top_n,
            diff=diff,
            suggestions_config=suggestions_config,
            profile=profile,
            since_days=since_days,
            triage_threshold=triage_threshold,
            rerank=rerank,
        )
        if output_path.suffix in (".md", ".html"):
            output_path = output_path.with_suffix(".json")
    elif fmt == "csv":
        content = generate_digest_csv(
            store,
            run_id,
            top_n=top_n,
            diff=diff,
            since_days=since_days,
            triage_threshold=triage_threshold,
            rerank=rerank,
        )
        if output_path.suffix in (".md", ".html"):
            output_path = output_path.with_suffix(".csv")
    elif fmt == "rss":
        content = generate_digest_rss(
            store,
            run_id,
            top_n=top_n,
            diff=diff,
            since_days=since_days,
            triage_threshold=triage_threshold,
            rerank=rerank,
        )
        if output_path.suffix in (".md", ".html"):
            output_path = output_path.with_suffix(".xml")
    elif fmt == "html":
        content = generate_digest_html(
            store,
            run_id,
            top_n=top_n,
            diff=diff,
            suggestions_config=suggestions_config,
            profile=profile,
            since_days=since_days,
            triage_threshold=triage_threshold,
            rerank=rerank,
        )
        if output_path.suffix == ".md":
            output_path = output_path.with_suffix(".html")
    else:
        content = generate_digest(
            store,
            run_id,
            top_n=top_n,
            diff=diff,
            suggestions_config=suggestions_config,
            profile=profile,
            since_days=since_days,
            triage_threshold=triage_threshold,
            rerank=rerank,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")

    # Build digest summary (reflects the same since_days filtering as the output)
    scored = filter_since(store.get_scores_for_run(run_id), since_days)
    run = store.get_last_run()
    top_picks, _, _ = categorize_papers(
        scored, top_n=top_n, triage_threshold=triage_threshold, rerank=rerank
    )
    summary = DigestSummary(
        digest_path=str(output_path),
        run_id=run_id,
        papers_new=run["papers_new"] if run else 0,
        papers_seen=run["papers_seen"] if run else 0,
        top_picks_count=len(top_picks),
        total_scored=len(scored),
        fmt=fmt,
    )

    return output_path, summary
