"""Neutral LLM judge: does a paper propose a genuine improvement to THIS repo?

Uses an OpenAI model (default GPT-5.5) so it is neutral toward the
Anthropic-produced baseline. Scores each paper 0-3 against a rubric, grounded in
the repository context. Verdicts are cached on disk keyed by
(rubric_version, model, repo, paper_id) so re-runs are cheap and stable.

A --mock path (no key, no spend) scores by keyword overlap so the whole
pipeline can be wired and tested offline.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

# This module is imported by eight entry scripts that each insert `src` themselves. Doing it
# here as well costs nothing, is idempotent, and means `import judge` also works standalone —
# the alternative is an import that depends on which script got there first.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from reporadar.paper_id import is_arxiv_id  # noqa: E402

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


# Identifier labels by scheme. The pipeline's shared paper shape carries either a real arXiv
# id or a synthetic "<source>:<id>" (see sources/__init__.py), so one hardcoded "arXiv:" label
# states something false about every non-arXiv paper the judge scores.
_ID_LABELS = {
    "doi": "DOI",
    "biorxiv": "bioRxiv",
    "medrxiv": "medRxiv",
    "oa": "OpenAlex",
    "ss": "Semantic Scholar",
    "dblp": "DBLP",
    "iacr": "IACR ePrint",
}


def _identifier_line(paper: dict[str, Any]) -> str:
    """The paper's id, labelled by the scheme it actually belongs to.

    **The arXiv branch is byte-exact on purpose, and that is the whole design.** A verdict is
    cached under `sha256(RUBRIC \\0 repo_context)` — the paper is NOT in that hash — so
    rewording this line for a paper already in the cache leaves the key identical while the
    question changes. That is the silent staleness `second_judge.verify_contexts` exists to
    catch one level up, and it would be undetectable here.

    Measured before writing this, not assumed: 3178 of the 3239 cached verdicts are arXiv
    papers and keep byte-identical prompts. The other 61 are `ss:` and `iacr:` papers spread
    over 19 live benchmark cases; they were deleted with this change rather than left holding
    answers to a prompt that is no longer sent.
    """
    ident = str(paper.get("arxiv_id", "") or "")
    if not ident or is_arxiv_id(ident):
        return f"arXiv: {ident}"
    scheme, sep, rest = ident.partition(":")
    if not sep or not rest:
        # A non-arXiv id with no scheme. Naming it neutrally beats inventing a provenance.
        return f"Identifier: {ident}"
    return f"{_ID_LABELS.get(scheme, scheme)}: {rest}"


def _id_line_matches(cached: dict[str, Any], ident_line: str) -> bool:
    """Whether a cached verdict was produced with today's identifier line.

    `_prompt_hash` covers the rubric and the repo context — **not the paper** — so a change to
    how a paper's id is rendered is invisible to it: the cache key stays identical while the
    question changes. That is the silent staleness this project has already paid for twice, and
    it is what §20.4's fix would otherwise have caused.

    Only non-arXiv papers are checked, and that asymmetry is the point. arXiv papers render
    byte-identically to how they always have, so demanding a marker they were never written
    with would invalidate ~3200 sound verdicts in order to catch 61 unsound ones. Entries
    written before this marker existed simply lack it, so exactly those 61 re-judge — and it
    happens on every machine, which deleting local cache files would not.
    """
    if ident_line.startswith("arXiv: "):
        return True
    return cached.get("_id_line") == ident_line


def _build_user_prompt(repo_context: str, paper: dict[str, Any]) -> str:
    return (
        f"# Repository context\n{repo_context}\n\n"
        f"# Candidate paper\n"
        f"Title: {paper.get('title', '')}\n"
        f"{_identifier_line(paper)}\n"
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

    # Do NOT coerce a degenerate response into a valid score-0 verdict — that
    # would silently penalize whichever system proposed the paper and get cached
    # forever. Raise instead, so the caller can skip the paper (never fabricate).
    choice = resp.choices[0]
    content = choice.message.content
    if not content:
        raise RuntimeError(
            f"empty judge response (finish_reason={getattr(choice, 'finish_reason', '?')})"
        )
    data = json.loads(content)
    if "score" not in data:
        raise RuntimeError(f"judge response missing 'score': {content[:200]}")
    score = int(data["score"])
    if score not in (0, 1, 2, 3):
        raise RuntimeError(f"judge score out of range 0-3: {score}")
    return {
        "score": score,
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
    """Return {score, justification, proposed_change} for one paper, cached.

    Raises on a judge failure (empty/malformed/out-of-range response) so the
    caller can skip the paper — a failure is never cached or scored as 0. A
    cache hit is only honored when the rubric + repo-context hash still matches,
    so a rubric edit or a changed repo re-judges instead of serving stale.
    """
    paper_id = paper.get("arxiv_id", paper.get("title", "unknown"))
    model_tag = "mock" if mock else model
    cache_file = _cache_path(model_tag, repo, paper_id)
    prompt_hash = hashlib.sha256(f"{RUBRIC}\0{repo_context}".encode()).hexdigest()[:12]

    ident_line = _identifier_line(paper)

    if use_cache and cache_file.exists():
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        if cached.get("_prompt_hash") == prompt_hash and _id_line_matches(cached, ident_line):
            return cached
        # else: rubric, repo context or the id line changed -> stale, re-judge

    verdict = _mock_score(repo_context, paper) if mock else _call_openai(model, repo_context, paper)
    verdict["_prompt_hash"] = prompt_hash
    # Written only for non-arXiv papers, so an arXiv verdict's file is byte-for-byte what it
    # was before this change and `second_judge.verify_contexts` still finds its hash.
    if not ident_line.startswith("arXiv: "):
        verdict["_id_line"] = ident_line

    if use_cache:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    return verdict
