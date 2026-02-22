"""Digest generation — render scored papers into a Markdown file."""

from __future__ import annotations

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


def write_digest(
    store: PaperStore,
    run_id: int,
    output_path: str | Path,
    top_n: int = 15,
    fmt: str = "md",
    diff: bool = False,
) -> Path:
    """Generate and write the digest to a file.

    *fmt* can be ``"md"`` (default) or ``"html"``.
    Returns the output path.
    """
    content = generate_digest(store, run_id, top_n=top_n, diff=diff)

    output_path = Path(output_path)
    if fmt == "html":
        content = markdown_to_html(content)
        if output_path.suffix == ".md":
            output_path = output_path.with_suffix(".html")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    return output_path
