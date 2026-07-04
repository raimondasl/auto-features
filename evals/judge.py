"""Neutral LLM judge: does a paper propose a genuine improvement to THIS repo?

Uses an OpenAI model (default GPT-5.5) so it is neutral toward the
Anthropic-produced baseline. Scores each paper 0-3 against a rubric, grounded in
the repository context. Verdicts are cached on disk keyed by
(rubric_version, model, repo, paper_id) so re-runs are cheap and stable.

A --mock path (no key, no spend) scores by keyword overlap so the whole
pipeline can be wired and tested offline.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

RUBRIC_VERSION = "v1"  # bump to invalidate cached verdicts
DEFAULT_JUDGE_MODEL = "gpt-5.5"

CACHE_DIR = Path(__file__).resolve().parent / "cache" / "judge"

RUBRIC = """\
You are a neutral senior research engineer judging whether a research paper can
be used to genuinely IMPROVE a specific software repository — not merely whether
it is on a related topic.

Score the paper for THIS repository on a 0-3 scale:
  0 = unrelated or not applicable to this repository.
  1 = same general topic, but offers no concrete, actionable improvement to this
      codebase (a "vaguely related" paper).
  2 = proposes a method/technique that could plausibly be integrated to improve
      this repository, with a concrete implementation path.
  3 = directly addresses a known limitation or core capability of this
      repository with a strong, specific, implementable improvement.

Be strict. Reserve 2 and 3 for papers whose METHOD could actually be applied to
this code. Ground your justification only in the provided abstract — do not
invent findings. Returning a low score is correct when the paper is not useful.

Respond with a JSON object with exactly these keys:
  "score": one of 0, 1, 2, 3
  "justification": one sentence
  "proposed_change": one concrete change to this repo (empty string if score < 2)
"""


def _cache_path(model: str, repo: str, paper_id: str) -> Path:
    safe_repo = re.sub(r"[^A-Za-z0-9_.-]", "_", repo)
    safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", paper_id)
    return (
        CACHE_DIR
        / RUBRIC_VERSION
        / re.sub(r"[^A-Za-z0-9_.-]", "_", model)
        / safe_repo
        / f"{safe_id}.json"
    )


def _build_user_prompt(repo_context: str, paper: dict[str, Any]) -> str:
    return (
        f"# Repository context\n{repo_context}\n\n"
        f"# Candidate paper\n"
        f"Title: {paper.get('title', '')}\n"
        f"arXiv: {paper.get('arxiv_id', '')}\n"
        f"Abstract: {paper.get('abstract', '')[:1800]}\n\n"
        f"Score this paper for the repository above using the rubric."
    )


def _mock_score(repo_context: str, paper: dict[str, Any]) -> dict[str, Any]:
    """Deterministic pseudo-score by keyword overlap — for offline wiring only."""

    def tok(s: str) -> set[str]:
        return set(re.findall(r"[a-z][a-z0-9]{3,}", s.lower()))

    repo_tokens = tok(repo_context)
    paper_tokens = tok(f"{paper.get('title', '')} {paper.get('abstract', '')}")
    overlap = len(repo_tokens & paper_tokens) / len(paper_tokens) if paper_tokens else 0.0
    score = 3 if overlap > 0.18 else 2 if overlap > 0.12 else 1 if overlap > 0.06 else 0
    return {
        "score": score,
        "justification": f"[mock] token overlap {overlap:.2f}",
        "proposed_change": "" if score < 2 else "[mock] apply the paper's method",
    }


def _call_openai(model: str, repo_context: str, paper: dict[str, Any]) -> dict[str, Any]:
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "The openai package is required for real judging. Install it with:\n"
            '    uv pip install -e ".[evals]"'
        ) from exc

    client = OpenAI()  # reads OPENAI_API_KEY
    messages = [
        {"role": "system", "content": RUBRIC},
        {"role": "user", "content": _build_user_prompt(repo_context, paper)},
    ]
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "response_format": {"type": "json_object"},
    }
    try:
        resp = client.chat.completions.create(temperature=0, **kwargs)
    except Exception:
        # Some reasoning models reject non-default temperature — retry without.
        resp = client.chat.completions.create(**kwargs)

    raw = resp.choices[0].message.content or "{}"
    data = json.loads(raw)
    return {
        "score": int(data.get("score", 0)),
        "justification": str(data.get("justification", "")),
        "proposed_change": str(data.get("proposed_change", "")),
    }


def judge_paper(
    repo: str,
    repo_context: str,
    paper: dict[str, Any],
    *,
    model: str = DEFAULT_JUDGE_MODEL,
    mock: bool = False,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Return {score, justification, proposed_change} for one paper, cached."""
    paper_id = paper.get("arxiv_id", paper.get("title", "unknown"))
    model_tag = "mock" if mock else model
    cache_file = _cache_path(model_tag, repo, paper_id)

    if use_cache and cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))

    verdict = _mock_score(repo_context, paper) if mock else _call_openai(model, repo_context, paper)

    if use_cache:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    return verdict
