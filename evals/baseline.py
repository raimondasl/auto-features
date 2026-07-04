"""The strong baseline: Opus 4.8, in two modes.

- mode="cli"  : Opus 4.8 via Claude Code headless (`claude -p`) with web tools,
                run in the target repo's directory (the agent reads the repo).
- mode="api"  : Opus 4.8 via the Anthropic Messages API with the server-side
                web_search tool. Works without the Claude Code CLI — the repo
                context is passed in the prompt instead of a filesystem mount.

Both parse the recommended papers out of the answer. Every run records a
`status` ("ok" | "missing_cli" | "error" | "timeout" | "no_key") and its raw
output, and **failures are never cached** — so a broken run retries next time
instead of silently poisoning the cache. Caches are keyed by mode, so a --mock
run can never be served to a real run.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from verify import extract_arxiv_ids

CACHE_DIR = Path(__file__).resolve().parent / "cache" / "baseline"

BASELINE_MODEL = "claude-opus-4-8"
# Opus 4.8 pricing ($/1M tokens) for a rough cost estimate (excludes web-search fees).
_PRICE_IN, _PRICE_OUT = 5.0, 25.0

# Headless CLI flags. Configurable because they vary by Claude Code version;
# override the whole list with RR_EVAL_CLAUDE_FLAGS (space-separated) if needed.
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

    for rid in extract_arxiv_ids(text):
        if rid not in ids:
            ids.append(rid)

    return ids, titles


def _cache_path(mode: str, repo_name: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", repo_name)
    return CACHE_DIR / mode / f"{safe}.json"


def _discriminator(mode: str, repo_context: str, flags: list[str] | None) -> str:
    """Hash the inputs that change the baseline's answer, so a model/prompt/flag
    (or, for api mode, repo-context) change invalidates a stale cached run."""
    parts = [BASELINE_MODEL, BASELINE_PROMPT, mode]
    if mode == "api":
        parts.append(repo_context)
    elif mode == "cli":
        parts.append(" ".join(flags or CLAUDE_FLAGS))
        parts.append(os.environ.get("RR_EVAL_CLAUDE_FLAGS", ""))
    return hashlib.sha256("\0".join(parts).encode()).hexdigest()[:12]


def _empty(status: str, raw: str) -> dict[str, Any]:
    return {"ids": [], "titles": [], "raw": raw, "cost_usd": 0.0, "status": status}


# ── mode: mock ─────────────────────────────────────────────────────────────


def _run_mock() -> dict[str, Any]:
    return {
        "ids": [],
        "titles": [],
        "raw": "[mock baseline: no papers]",
        "cost_usd": 0.0,
        "status": "ok",
    }


# ── mode: cli (Claude Code headless) ───────────────────────────────────────


def _run_cli(repo_dir: Path, *, flags: list[str] | None, timeout: int) -> dict[str, Any]:
    claude_bin = os.environ.get("RR_EVAL_CLAUDE_BIN", "claude")
    env_flags = os.environ.get("RR_EVAL_CLAUDE_FLAGS")
    flag_list = flags or (env_flags.split() if env_flags else CLAUDE_FLAGS)
    cmd = [claude_bin, "-p", *flag_list, BASELINE_PROMPT]

    try:
        proc = subprocess.run(
            cmd, cwd=str(repo_dir), capture_output=True, text=True, timeout=timeout
        )
    except FileNotFoundError:
        return _empty(
            "missing_cli",
            f"`{claude_bin}` not found on PATH. Install Claude Code, set RR_EVAL_CLAUDE_BIN "
            "to its full path, or use --baseline api.",
        )
    except subprocess.TimeoutExpired:
        return _empty("timeout", f"claude timed out after {timeout}s")

    if proc.returncode != 0:
        return _empty(
            "error",
            f"claude exited {proc.returncode}. stderr: {(proc.stderr or '').strip()[:500]}",
        )

    # `claude -p --output-format json` reports internal failures (turn-limit,
    # execution errors) via is_error / subtype while still exiting 0. A non-JSON
    # stdout means it printed a raw error instead of the expected result. Treat
    # all of these as failures rather than parsing them for "recommendations".
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return _empty("error", f"non-JSON output from claude: {proc.stdout.strip()[:400]}")

    if payload.get("is_error") or payload.get("subtype") not in (None, "success"):
        return _empty(
            "error",
            f"claude reported failure (subtype={payload.get('subtype')}, "
            f"is_error={payload.get('is_error')})",
        )
    raw_text = payload.get("result", "")
    if not raw_text.strip():
        return _empty("error", "claude returned an empty result")
    cost = float(payload.get("total_cost_usd", 0.0) or 0.0)

    ids, titles = _parse_recommendations(raw_text)
    return {"ids": ids, "titles": titles, "raw": raw_text, "cost_usd": cost, "status": "ok"}


# ── mode: api (Anthropic Messages API + web_search) ────────────────────────


def _estimate_cost(usage: Any) -> float:
    tin = getattr(usage, "input_tokens", 0) or 0
    tout = getattr(usage, "output_tokens", 0) or 0
    return (tin / 1_000_000) * _PRICE_IN + (tout / 1_000_000) * _PRICE_OUT


def _run_api(repo_context: str, *, max_uses: int = 8) -> dict[str, Any]:
    try:
        from anthropic import Anthropic
    except ImportError:
        return _empty(
            "error",
            'anthropic package not installed. Run: uv pip install -e ".[evals]"',
        )
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return _empty("no_key", "ANTHROPIC_API_KEY not set (needed for --baseline api).")

    client = Anthropic()
    tools = [{"type": "web_search_20260209", "name": "web_search", "max_uses": max_uses}]
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": f"{BASELINE_PROMPT}\n\n# Repository context\n{repo_context}",
        }
    ]

    text_parts: list[str] = []
    cost = 0.0
    final_stop: str | None = None
    try:
        for _ in range(6):  # continue across server-tool pause_turn boundaries
            resp = client.messages.create(
                model=BASELINE_MODEL, max_tokens=8000, tools=tools, messages=messages
            )
            cost += _estimate_cost(resp.usage)
            text_parts += [b.text for b in resp.content if b.type == "text"]
            final_stop = resp.stop_reason
            if resp.stop_reason == "pause_turn":
                messages.append({"role": "assistant", "content": resp.content})
                continue
            break
    except Exception as exc:  # noqa: BLE001 — surface any API error as a failed run
        return _empty("error", f"Anthropic API error: {exc}")

    raw = "\n".join(text_parts)
    ids, titles = _parse_recommendations(raw)
    # Only a clean completion is trustworthy. max_tokens truncation, a refusal,
    # or still being paused after the loop cap means the recommendations block
    # was likely never emitted — mark it failed (and don't cache) rather than
    # reporting a false abstention.
    if final_stop != "end_turn":
        return {
            "ids": ids,
            "titles": titles,
            "raw": f"[incomplete: stop_reason={final_stop}]\n{raw}",
            "cost_usd": cost,
            "status": "error",
        }
    return {"ids": ids, "titles": titles, "raw": raw, "cost_usd": cost, "status": "ok"}


# ── dispatcher ─────────────────────────────────────────────────────────────


def run_baseline(
    repo_dir: Path,
    *,
    repo_name: str,
    repo_context: str = "",
    mode: str = "cli",
    mock: bool = False,
    use_cache: bool = True,
    flags: list[str] | None = None,
    timeout: int = 900,
) -> dict[str, Any]:
    """Run the baseline for one repo. Returns {ids, titles, raw, cost_usd, status}.

    Only successful runs (status == "ok") are cached, and caches are per-mode, so
    a mock run can never be served to a real run and a failure always retries.
    """
    effective_mode = "mock" if mock else mode
    cache_file = _cache_path(effective_mode, repo_name)
    disc = _discriminator(effective_mode, repo_context, flags)

    if use_cache and cache_file.exists():
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        if cached.get("_disc") == disc:
            return cached
        # else: model/prompt/flags/context changed -> stale, re-run

    if mock:
        out = _run_mock()
    elif mode == "api":
        out = _run_api(repo_context)
    else:
        out = _run_cli(repo_dir, flags=flags, timeout=timeout)

    out["_disc"] = disc
    # Only cache clean runs — a failure must retry next time, never be served.
    if use_cache and out.get("status") == "ok":
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out
