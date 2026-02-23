"""GitHub Issue export — format and create issues from scored papers."""

from __future__ import annotations

import logging
import subprocess
from typing import Any

logger = logging.getLogger(__name__)


def check_gh_available() -> bool:
    """Check if the GitHub CLI (gh) is installed and authenticated."""
    try:
        result = subprocess.run(
            ["gh", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


def format_issue(
    paper: dict[str, Any],
    enrichment: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Format a paper as a GitHub issue dict with title and body.

    Returns ``{"title": ..., "body": ...}``.
    """
    title = f"[RepoRadar] {paper['title']}"

    lines: list[str] = []
    lines.append(f"**Paper:** [{paper['arxiv_id']}]({paper['url']})")
    lines.append(f"**Score:** {paper.get('score_total', 0):.2f}")
    lines.append(f"**Authors:** {', '.join(paper.get('authors', []))}")
    lines.append(f"**Published:** {paper.get('published', '')[:10]}")
    lines.append("")

    # Abstract excerpt
    abstract = paper.get("abstract", "")
    if abstract:
        excerpt = abstract[:300] + ("..." if len(abstract) > 300 else "")
        lines.append(f"> {excerpt}")
        lines.append("")

    # Enrichment badges
    if enrichment:
        if enrichment.get("has_code"):
            code_urls = enrichment.get("code_urls", [])
            if code_urls:
                lines.append(f"**Code:** {', '.join(code_urls)}")
            else:
                lines.append("**Code:** Available (check paper)")
        if enrichment.get("datasets"):
            lines.append(f"**Datasets:** {', '.join(enrichment['datasets'])}")
        if enrichment.get("tasks"):
            lines.append(f"**Tasks:** {', '.join(enrichment['tasks'])}")
        lines.append("")

    # Suggestions
    suggestions = paper.get("suggestions", [])
    if suggestions:
        lines.append("### Action Ideas")
        for s in suggestions:
            lines.append(f"- {s}")
        lines.append("")

    lines.append("---")
    lines.append("*Created by [RepoRadar](https://github.com/raimondasl/auto-features)*")

    return {"title": title, "body": "\n".join(lines)}


def create_issue(
    issue: dict[str, str],
    labels: list[str] | None = None,
) -> str | None:
    """Create a GitHub issue using the gh CLI.

    Returns the issue URL on success, or None on failure.
    """
    cmd = ["gh", "issue", "create", "--title", issue["title"], "--body", issue["body"]]
    if labels:
        cmd.extend(["--label", ",".join(labels)])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            url = result.stdout.strip()
            return url if url else None
        logger.warning("gh issue create failed: %s", result.stderr.strip())
        return None
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        logger.warning("Failed to create issue: %s", exc)
        return None


def create_issues(
    papers: list[dict[str, Any]],
    enrichments: dict[str, dict[str, Any]] | None = None,
    labels: list[str] | None = None,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """Create GitHub issues for a batch of papers.

    Returns a list of ``{"arxiv_id": ..., "issue_url": ..., "status": ...}``
    where status is ``"created"``, ``"skipped"``, or ``"dry_run"``.
    """
    results: list[dict[str, Any]] = []
    enrichments = enrichments or {}

    for paper in papers:
        arxiv_id = paper["arxiv_id"]
        enrichment = enrichments.get(arxiv_id)
        issue = format_issue(paper, enrichment)

        if dry_run:
            results.append(
                {
                    "arxiv_id": arxiv_id,
                    "title": issue["title"],
                    "issue_url": None,
                    "status": "dry_run",
                }
            )
            continue

        url = create_issue(issue, labels=labels)
        if url:
            results.append(
                {
                    "arxiv_id": arxiv_id,
                    "issue_url": url,
                    "status": "created",
                }
            )
        else:
            results.append(
                {
                    "arxiv_id": arxiv_id,
                    "issue_url": None,
                    "status": "skipped",
                }
            )

    return results
