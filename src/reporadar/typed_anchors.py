"""Typed entity spans from a README, as an additional source of profile anchors.

An *anchor* is a named thing the repository depends on. Until now the only way to get one
was to parse a dependency manifest, so a C, C++, Rust or Go project had none at all --
12 of the 25 benchmark repositories, measured in NR-39. This module reads the README
instead and asks an LLM for the named entities in it, which is the mechanism NERdME
(arXiv:2603.05750) proposes for exactly that gap.

**Spans are verbatim or they are discarded.** A span that is not literally present in the
README is a hallucination, not an anchor, and is dropped rather than trusted. This is not
defensive coding for its own sake: NR-12 measured that verbatim README text beats LLM
paraphrase by +21 net@2 as gate context, and that full paraphrase scores *below* sending
nothing. Whatever this channel is worth, it is worth it as quotation.

**Failure is loud.** If extraction cannot run, this raises rather than returning an empty
list. An opt-in stage that silently produces nothing turns a treatment arm into the control
arm and reports the difference as a null -- the C-9 shape, where a malformed query broke
every non-arXiv source from day one and the arms were compared anyway.

Off by default (``profiler.typed_anchors``). Measured in P9/P10; see evals/RESULTS.md for
what is and is not established about it.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from reporadar.llm_client import LLMError, complete

# The five types are chosen for the ANCHOR role -- things a paper's abstract might name
# when it improves on them -- not to replicate NERdME's ten, which its abstract does not
# enumerate.
ENTITY_TYPES = ("library", "method", "format", "model", "dataset")

# The profiler's README search order. Kept identical to `_repo_prose`'s so this reads the
# document the rest of the profile describes.
README_NAMES = ("README.md", "README.rst", "README.txt", "README")

# An unbounded read would put a 41k-character README into a prompt. The profiler budgets
# gate prose at 300 chars for a different reason (disclosure); this is a cost bound.
README_BUDGET = 12_000

MAX_ENTITIES = 40

PROMPT = """\
You are extracting named entities from a software repository's README, to answer one \
question: which named things does THIS repository actually depend on, implement, or \
build upon?

Return only entities that are CENTRAL to what the repository itself is or does.

Do NOT return:
- systems the README merely compares against or benchmarks rivals to
- documentation, CI, packaging, linting or test tooling
- generic topic words ("machine learning", "database", "performance")
- anything not written literally in the text below

Entity types:
- library:  a named software library, framework or engine it uses or builds on
- method:   a named algorithm or technique it implements (e.g. "IVF-PQ", "MVCC")
- format:   a named data format, protocol or wire encoding
- model:    a named model or architecture
- dataset:  a named dataset

Each span must be copied VERBATIM from the README -- the exact substring, not a \
paraphrase, expansion or correction of it.

Return JSON only, no prose:
{{"entities": [{{"span": "<verbatim>", "type": "<one of the five>"}}]}}

At most {max_entities} entities. If the README names none, return {{"entities": []}}.

README of the repository "{name}":
---
{readme}
---
"""


def read_readme(repo_path: Path, budget: int = README_BUDGET) -> str:
    """The README the profiler would describe, truncated to *budget*."""
    for name in README_NAMES:
        path = repo_path / name
        if path.is_file():
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if text.strip():
                return text[:budget]
    return ""


def parse_entities(raw: str) -> list[dict[str, str]]:
    """Pull the entity list out of a model reply, tolerating fences and stray prose."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return []
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []
    out: list[dict[str, str]] = []
    for entity in data.get("entities", []) or []:
        if not isinstance(entity, dict):
            continue
        span = str(entity.get("span", "")).strip()
        etype = str(entity.get("type", "")).strip().lower()
        if span and etype in ENTITY_TYPES:
            out.append({"span": span, "type": etype})
    return out


def _cache_key(readme: str) -> str:
    """Keyed on the README, so an edited README re-extracts and an unchanged one does not."""
    return hashlib.sha256(readme.encode("utf-8")).hexdigest()[:16]


def extract_typed_anchors(
    repo_path: str | Path,
    *,
    llm_cfg: Any,
    cache_path: str | Path | None = None,
    budget: int = README_BUDGET,
) -> list[str]:
    """Named entities from the repository's README, lowercased, verbatim-checked.

    *llm_cfg* is anything ``llm_client.complete`` accepts -- a SuggestionsConfig or the
    ProfilerConfig the loader mirrors those fields onto.

    Raises LLMError if extraction cannot run. A repository with no README returns an empty
    list without calling anything: that is an absent document, not a failed extraction.
    """
    repo_path = Path(repo_path)
    readme = read_readme(repo_path, budget)
    if not readme.strip():
        return []

    key = _cache_key(readme)
    cache_file = Path(cache_path) if cache_path else repo_path / ".reporadar" / "typed_anchors.json"
    if cache_file.is_file():
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            if cached.get("key") == key:
                return list(cached.get("anchors", []))
        except (json.JSONDecodeError, OSError):
            pass  # a corrupt cache is a re-extract, not a failure

    prompt = PROMPT.format(name=repo_path.name, readme=readme, max_entities=MAX_ENTITIES)
    raw = complete(prompt, llm_cfg, max_tokens=2000, max_retries=2)
    entities = parse_entities(raw)
    if not entities and raw.strip() and "entities" not in raw:
        raise LLMError(f"typed-anchor extraction returned no parseable entities: {raw[:200]!r}")

    low = readme.lower()
    anchors: list[str] = []
    seen: set[str] = set()
    for entity in entities:
        span = entity["span"].lower().strip()
        if span and span in low and span not in seen:
            seen.add(span)
            anchors.append(span)

    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(
            json.dumps({"key": key, "anchors": anchors}, indent=1), encoding="utf-8"
        )
    except OSError:
        pass  # an unwritable cache costs a re-extraction, never a wrong answer

    return anchors
