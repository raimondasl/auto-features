"""Digest generation — render scored papers into a Markdown file."""

from __future__ import annotations

import csv
import io
import json as json_mod
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jinja2 import Environment, PackageLoader

from reporadar.store import PaperStore
from reporadar.suggestions import enrich_papers_with_suggestions

TEMPLATES_DIR = Path(__file__).parent / "templates"

# Score thresholds for categorizing papers into tiers.
# Papers with score >= TOP_THRESHOLD go into "Top Picks",
# >= MAYBE_THRESHOLD into "Maybe Relevant", rest into "Muted".
TOP_THRESHOLD = 0.5
MAYBE_THRESHOLD = 0.2


def _load_template(template_name: str = "digest.md.j2") -> Any:
    """Load a Jinja2 template from the templates directory."""
    env = Environment(
        loader=PackageLoader("reporadar", "templates"),
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    return env.get_template(template_name)


def categorize_papers(
    scored_papers: list[dict[str, Any]],
    top_n: int = 15,
    top_threshold: float = TOP_THRESHOLD,
    maybe_threshold: float = MAYBE_THRESHOLD,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Split scored papers into three tiers: top_picks, maybe_relevant, muted.

    Papers are assumed to be sorted by score descending. Only the first
    *top_n* papers are included across all tiers.
    """
    limited = scored_papers[:top_n]

    top_picks = []
    maybe_relevant = []
    muted = []

    for paper in limited:
        score = paper["score_total"]
        if score >= top_threshold:
            top_picks.append(paper)
        elif score >= maybe_threshold:
            maybe_relevant.append(paper)
        else:
            muted.append(paper)

    return top_picks, maybe_relevant, muted


def generate_digest(
    store: PaperStore,
    run_id: int,
    top_n: int = 15,
    diff: bool = False,
) -> str:
    """Generate digest Markdown content for a given run.

    If *diff* is True, marks each paper as new or carried over from the
    previous run by setting ``is_new`` on each paper dict.

    Returns the rendered Markdown string.
    """
    scored = store.get_scores_for_run(run_id)
    run = store.get_last_run()

    # Diff mode: determine which papers are new vs. carried over
    diff_mode = False
    if diff:
        diff_mode = True
        prev_id = store.get_previous_run_id(run_id)
        prev_ids = store.get_scored_paper_ids_for_run(prev_id) if prev_id is not None else set()
        for paper in scored:
            paper["is_new"] = paper["arxiv_id"] not in prev_ids

    top_picks, maybe_relevant, muted = categorize_papers(scored, top_n=top_n)
    enrich_papers_with_suggestions(top_picks)

    has_embeddings = any(p.get("embedding_score") is not None for p in scored)
    has_citations = any(p.get("citation_score") is not None for p in scored)
    has_enrichments = any(p.get("has_code") or p.get("datasets") for p in scored)

    template = _load_template()
    return template.render(
        generated_at=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        run_id=run_id,
        papers_new=run["papers_new"] if run else 0,
        papers_seen=run["papers_seen"] if run else 0,
        total_scored=len(scored),
        queries_used=run["queries_used"] if run else [],
        top_picks=top_picks,
        maybe_relevant=maybe_relevant,
        muted=muted,
        diff_mode=diff_mode,
        has_embeddings=has_embeddings,
        has_citations=has_citations,
        has_enrichments=has_enrichments,
    )


def markdown_to_html(md_content: str) -> str:
    """Wrap rendered Markdown in a basic HTML page.

    Uses a simple CSS stylesheet for readability. This avoids adding a
    Markdown-to-HTML library dependency — the output is raw Markdown
    inside <pre> tags with some basic styling. For a richer rendering,
    users can pipe the .md through any Markdown renderer.
    """
    template = _load_template("digest.html.j2")
    return template.render(markdown_content=md_content)


def generate_digest_json(
    store: PaperStore,
    run_id: int,
    top_n: int = 15,
    diff: bool = False,
) -> str:
    """Generate digest as a JSON string.

    Returns a JSON string with top_picks, maybe_relevant, and muted tiers.
    """
    scored = store.get_scores_for_run(run_id)
    run = store.get_last_run()

    if diff:
        prev_id = store.get_previous_run_id(run_id)
        prev_ids = store.get_scored_paper_ids_for_run(prev_id) if prev_id is not None else set()
        for paper in scored:
            paper["is_new"] = paper["arxiv_id"] not in prev_ids

    top_picks, maybe_relevant, muted = categorize_papers(scored, top_n=top_n)
    enrich_papers_with_suggestions(top_picks)

    return json_mod.dumps(
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
]


def generate_digest_csv(
    store: PaperStore,
    run_id: int,
    top_n: int = 15,
    diff: bool = False,
) -> str:
    """Generate digest as a CSV string."""
    scored = store.get_scores_for_run(run_id)
    top_picks, maybe_relevant, muted = categorize_papers(scored, top_n=top_n)

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
        writer.writerow(row)

    return output.getvalue()


def generate_digest_rss(
    store: PaperStore,
    run_id: int,
    top_n: int = 15,
    diff: bool = False,
) -> str:
    """Generate digest as an RSS 2.0 XML string."""
    scored = store.get_scores_for_run(run_id)
    top_picks, maybe_relevant, muted = categorize_papers(scored, top_n=top_n)
    all_papers = top_picks + maybe_relevant + muted

    template = _load_template("digest.rss.xml.j2")
    return template.render(
        generated_at=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        run_id=run_id,
        papers=all_papers,
    )


def write_digest(
    store: PaperStore,
    run_id: int,
    output_path: str | Path,
    top_n: int = 15,
    fmt: str = "md",
    diff: bool = False,
) -> Path:
    """Generate and write the digest to a file.

    *fmt* can be ``"md"``, ``"html"``, ``"json"``, ``"csv"``, or ``"rss"``.
    Returns the output path.
    """
    output_path = Path(output_path)

    if fmt == "json":
        content = generate_digest_json(store, run_id, top_n=top_n, diff=diff)
        if output_path.suffix in (".md", ".html"):
            output_path = output_path.with_suffix(".json")
    elif fmt == "csv":
        content = generate_digest_csv(store, run_id, top_n=top_n, diff=diff)
        if output_path.suffix in (".md", ".html"):
            output_path = output_path.with_suffix(".csv")
    elif fmt == "rss":
        content = generate_digest_rss(store, run_id, top_n=top_n, diff=diff)
        if output_path.suffix in (".md", ".html"):
            output_path = output_path.with_suffix(".xml")
    elif fmt == "html":
        content = generate_digest(store, run_id, top_n=top_n, diff=diff)
        content = markdown_to_html(content)
        if output_path.suffix == ".md":
            output_path = output_path.with_suffix(".html")
    else:
        content = generate_digest(store, run_id, top_n=top_n, diff=diff)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    return output_path
