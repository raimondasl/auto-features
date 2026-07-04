"""The strong baseline: Opus 4.8 via Claude Code headless.

Runs the user's prompt — "fetch and summarize research papers that relate to the
code in the repository and propose methods to improve it" — as an agentic
Claude Code run with web tools, in the target repo's directory. Parses the
recommended papers out of the answer and caches the raw output per repo.

Flags vary by Claude Code version, so CLAUDE_FLAGS is a single editable list.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

from verify import extract_arxiv_ids

CACHE_DIR = Path(__file__).resolve().parent / "cache" / "baseline"

BASELINE_MODEL = "claude-opus-4-8"

# Flags for headless, web-enabled, non-interactive Opus. Adjust for your Claude
# Code version if needed (e.g. add --permission-mode dontAsk or
# --dangerously-skip-permissions, or --bare).
CLAUDE_FLAGS = [
    "--model",
    BASELINE_MODEL,
    "--output-format",
    "json",
    "--allowedTools",
    "WebSearch,WebFetch",
    "--max-turns",
    "12",
]

BASELINE_PROMPT = (
    "Please fetch and summarize research papers that relate to the code in this "
    "repository and propose methods to improve it. Focus on papers whose methods "
    "could actually be applied to improve this codebase; it is better to recommend "
    "nothing than to recommend papers that are not genuinely useful. "
    "End your response with a fenced ```json code block containing a JSON array of "
    'the papers you recommend, each as {"arxiv_id": "XXXX.XXXXX", "title": "..."}. '
    "Use an empty array [] if you recommend nothing."
)


def _parse_recommendations(text: str) -> tuple[list[str], list[str]]:
    """Extract (arxiv_ids, titles) from the baseline's answer.

    Prefers the final ```json block; unions in any arXiv IDs found in prose.
    """
    ids: list[str] = []
    titles: list[str] = []

    for block in re.findall(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL):
        try:
            items = json.loads(block)
        except json.JSONDecodeError:
            continue
        for it in items:
            if not isinstance(it, dict):
                continue
            if it.get("arxiv_id"):
                ids.append(str(it["arxiv_id"]))
            elif it.get("title"):
                titles.append(str(it["title"]))

    # Union with any IDs mentioned anywhere (belt and suspenders).
    for rid in extract_arxiv_ids(text):
        if rid not in ids:
            ids.append(rid)

    return ids, titles


def run_baseline(
    repo_dir: Path,
    *,
    repo_name: str,
    mock: bool = False,
    use_cache: bool = True,
    flags: list[str] | None = None,
    timeout: int = 900,
) -> dict[str, Any]:
    """Run the baseline for one repo. Returns {ids, titles, raw, cost_usd}."""
    cache_file = CACHE_DIR / f"{re.sub(r'[^A-Za-z0-9_.-]', '_', repo_name)}.json"
    if use_cache and cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))

    if mock:
        out = {"ids": [], "titles": [], "raw": "[mock baseline: no papers]", "cost_usd": 0.0}
    else:
        cmd = ["claude", "-p", *(flags or CLAUDE_FLAGS), BASELINE_PROMPT]
        try:
            proc = subprocess.run(
                cmd, cwd=str(repo_dir), capture_output=True, text=True, timeout=timeout
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "`claude` CLI not found. Install Claude Code and ensure it is on PATH, "
                "or run the baseline on a machine where it is."
            ) from exc
        except subprocess.TimeoutExpired:
            out = {"ids": [], "titles": [], "raw": "[baseline timed out]", "cost_usd": 0.0}
            _write_cache(cache_file, out, use_cache)
            return out

        raw_text = proc.stdout
        cost = 0.0
        try:
            payload = json.loads(proc.stdout)
            raw_text = payload.get("result", proc.stdout)
            cost = float(payload.get("total_cost_usd", 0.0) or 0.0)
        except json.JSONDecodeError:
            pass  # non-JSON output; treat stdout as the answer text

        ids, titles = _parse_recommendations(raw_text)
        out = {"ids": ids, "titles": titles, "raw": raw_text, "cost_usd": cost}

    _write_cache(cache_file, out, use_cache)
    return out


def _write_cache(path: Path, data: dict[str, Any], use_cache: bool) -> None:
    if use_cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
