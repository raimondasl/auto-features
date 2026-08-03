"""One LLM call that reads the repo's docs and says what it does — and where it could improve.

The triage gate needs to know what a project is FOR. Two earlier attempts to supply that
both failed for the same reason: they took a *positional slice* of a document and hoped the
useful sentences were inside it.

  - `_collect_text_corpus(repo)[0]` — the packaging one-liner on 11 of 12 benchmark repos.
  - `README[:300]` — measured best of four budgets, but the arms were all prefixes of the
    same document, so "300 beats 2000" confounded quantity with content. Reading the text
    settles it: chars 300-2000 hold ColBERT's late-interaction explanation and Detectron2's
    capability list, while `graph`'s first 300 characters are link badges and its actual
    description begins after the cut. A prefix is a lottery on how the author laid the file
    out, and the differences between budgets were never significant (300 vs 2000: +9,
    P = 0.108).

A summariser removes the lottery: it reads a generous slice of README, packaging metadata
and `docs/`, and returns the same two things for every repo no matter where in the file they
were stated. The second half — candidate improvement areas — is the part a prefix can never
supply at all, because projects rarely write down what they are missing.

Cost is one small-model call per repo, cached on a hash of the input, so it is paid once and
then only when the docs actually change.

**STATUS: measured, and it does not beat the prefix it was built to replace.** On 602
labelled papers (evals/RESULTS.md -> "four ways to tell the gate what a repo is"):

    keywords (no description)   net@2 +73
    prose 300 (prefix)                +95
    as_excerpt_block (verbatim)       +91   tied with the prefix: -4, P = 0.778
    as_prompt_block (paraphrase)      +70   WORSE than no description at all

So this module is **not wired into `rr update`** and no config turns it on — doing that
would add a config surface, a cache migration and a much larger LLM disclosure for a
change that measured as a tie. Its consumers are the eval harness and any future
experiment. Two things here are worth keeping regardless of that verdict:

* `as_excerpt_block` vs `as_prompt_block` is a clean +21 for **verbatim text over
  paraphrase**, which is the one solid finding of the exercise and a warning for anything
  that plans to feed model-written repo descriptions to another model.
* `improvement_areas` has never been tested where it most plausibly belongs — *retrieval*.
  Phrases like "adaptive quantization strategies" or "pseudo-relevance feedback" are
  search queries, and retrieval (not gating) is the measured bottleneck at 18/24 reach.

If it is ever wired into `rr update`, it must be opt-in and declared in `privacy.py`: it
sends up to MAX_INPUT_CHARS of your README and docs to an LLM, far more than the keyword
profile does.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from reporadar.llm_client import complete
from reporadar.profiler import _clean_document, _packaging_metadata_text, _read_text_file

logger = logging.getLogger(__name__)

# Generous, because the summariser's whole point is not to bet on where the useful
# sentences sit. It is one call per repo, not one per paper, so the input can be large.
MAX_INPUT_CHARS = 12_000
MAX_README = 8_000
MAX_DOC_FILE = 1_500
MAX_DOC_FILES = 4

# Bumped whenever _PROMPT or the RepoSummary shape changes. Folded into the cache key so a
# summariser change invalidates old entries: the key is otherwise a hash of the repo's docs
# alone, which would happily serve a summary produced by a prompt that no longer exists.
_PROMPT_VERSION = 2

_PROMPT = """\
You are reading a software repository's own documentation. Report what the project IS
and where it could plausibly be IMPROVED by new research.

Be concrete and technical. Name the actual methods, data structures and subsystems the
project uses. Do not describe it in generic terms ("a Python library", "a tool for
developers") — those are useless for matching research papers to this codebase.

For improvement areas, prefer limitations the docs themselves admit (roadmap items, "not
yet supported", known issues, TODOs). Where the docs say nothing, infer from what the
system does — the techniques it would benefit from that it clearly does not implement.
Do not invent limitations you have no evidence for; an empty list is acceptable.

Respond with ONLY a JSON object:
{"purpose": "<2-3 sentences: what this project does, technically>",
 "capabilities": ["<specific method/subsystem>", ...],
 "improvement_areas": ["<concrete technical area where research could help>", ...],
 "key_sentences": ["<sentence copied EXACTLY from the documentation above>", ...]}
Use at most 8 capabilities and at most 6 improvement areas.

key_sentences: 3-6 sentences COPIED VERBATIM, character for character, from the
documentation above — the ones that best identify what this project does technically.
Do not rewrite, shorten or join them. They are checked against the source and silently
dropped if they do not appear in it, so paraphrasing loses the sentence entirely."""


@dataclass
class RepoSummary:
    """What the repo does and where it might improve, as extracted by the summariser."""

    purpose: str = ""
    capabilities: list[str] = field(default_factory=list)
    improvement_areas: list[str] = field(default_factory=list)
    # Sentences copied verbatim from the docs. A summary paraphrases, and paraphrase
    # measured WORSE than raw README text at matched length (+70 vs +86 net@2 at ~1.85k
    # chars) — the plausible reason being that a README states techniques in the same
    # vocabulary the relevant papers use, which rewriting discards. These keep that
    # vocabulary while still letting a model choose WHICH sentences matter, instead of
    # betting on a prefix.
    key_sentences: list[str] = field(default_factory=list)
    # Hash of the documentation this was derived from. The cache key, and the reason a
    # summary cannot silently outlive the README that produced it.
    source_hash: str = ""

    def as_prompt_block(self, *, include_gaps: bool = True) -> str:
        """The summary as it appears inside a triage prompt. Empty when there is nothing.

        Sections are omitted individually: a repo whose docs admit no limitations should
        get no `Known gaps` header rather than an empty one, which would read to the model
        as "this project has no gaps" instead of "nobody said".

        *include_gaps* exists because the gap list is the risky half. Listing six concrete
        limitations invites a gate to treat them as the *only* ways the repo could improve
        and reject everything else — which is what the first measurement showed, recall
        falling 0.68 -> 0.52 and hardest on the repos with the longest gap lists. The
        wording below hedges against that; the flag is how the hedge gets tested.
        """
        if not self.purpose and not self.capabilities:
            return ""
        parts = []
        if self.purpose:
            parts.append(f"What this project is:\n{self.purpose}")
        if self.capabilities:
            parts.append("Implements: " + ", ".join(self.capabilities))
        if include_gaps and self.improvement_areas:
            parts.append(
                "Some areas it could improve (NOT an exhaustive list — a paper addressing "
                "anything else this project does is equally valid): "
                + ", ".join(self.improvement_areas)
            )
        return "\n".join(parts)

    def as_excerpt_block(self) -> str:
        """Only the verbatim sentences — the repo's own words, semantically selected.

        The counterpart to ``profile.prose``: both give the gate the repository's actual
        wording, but this one lets a model choose WHICH sentences rather than taking
        whatever the first N characters happen to be. On `graph` that is the difference
        between four link badges and "All Graph Neural Network layers are implemented via
        the nn.MessagePassing interface."
        """
        if not self.key_sentences:
            return ""
        return "What this project is, in its own words:\n" + "\n".join(
            f"- {s}" for s in self.key_sentences
        )


def collect_doc_corpus(repo_path: Path) -> str:
    """README + packaging metadata + a sample of docs/, newest-relevant first.

    Ordered by how likely each source is to state the project's purpose, and capped per
    source so one enormous file cannot crowd out the README.
    """
    parts: list[str] = []
    metadata = _packaging_metadata_text(repo_path).strip()
    if metadata:
        parts.append(f"# Package metadata\n{metadata}")

    for name in ("README.md", "README.rst", "README.txt", "README"):
        text = _read_text_file(repo_path / name)
        if text:
            cleaned = _clean_document(text).strip()
            if cleaned:
                parts.append(f"# README\n{cleaned[:MAX_README]}")
            break

    docs_dir = repo_path / "docs"
    if docs_dir.is_dir():
        seen = 0
        for ext in ("*.md", "*.rst"):
            for doc_file in sorted(docs_dir.rglob(ext)):
                if seen >= MAX_DOC_FILES:
                    break
                text = _read_text_file(doc_file)
                if not text:
                    continue
                cleaned = _clean_document(text).strip()
                if cleaned:
                    parts.append(f"# docs/{doc_file.name}\n{cleaned[:MAX_DOC_FILE]}")
                    seen += 1
    return "\n\n".join(parts)[:MAX_INPUT_CHARS]


def corpus_hash(corpus: str) -> str:
    return hashlib.sha256(corpus.encode("utf-8", "replace")).hexdigest()[:16]


def _verbatim_only(candidates: list[str], corpus: str) -> list[str]:
    """Keep only sentences that really occur in the docs, ignoring whitespace runs.

    The whole value of an extracted sentence is that it is the repo's own wording. A
    model asked for verbatim text will still occasionally tidy it, and a tidied sentence
    is a paraphrase wearing a quote's clothes — indistinguishable downstream, so it is
    dropped here where the source is still in hand.
    """
    flat = " ".join(corpus.split())
    kept = []
    for c in candidates:
        needle = " ".join(c.split())
        if len(needle) >= 20 and needle in flat:
            kept.append(needle)
        else:
            logger.debug("dropped non-verbatim key sentence: %s", needle[:80])
    return kept


def _parse_summary(raw: str, corpus: str, source_hash: str) -> RepoSummary:
    """Parse the summariser's JSON. Raises on anything malformed — never half-fills.

    A partially-parsed summary would be worse than none: the triage prompt would claim to
    describe the project and then describe part of it, with no way for a reader to tell.
    """
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object in summary response: {raw[:150]}")
    data = json.loads(match.group(0))
    purpose = str(data.get("purpose", "")).strip()
    if not purpose:
        raise ValueError("summary has no purpose, which is the one field it exists for")

    def _strings(key: str, limit: int) -> list[str]:
        raw_list = data.get(key) or []
        if not isinstance(raw_list, list):
            raise ValueError(f"summary field {key!r} is not a list")
        return [str(x).strip() for x in raw_list if str(x).strip()][:limit]

    return RepoSummary(
        purpose=purpose,
        capabilities=_strings("capabilities", 8),
        improvement_areas=_strings("improvement_areas", 6),
        key_sentences=_verbatim_only(_strings("key_sentences", 6), corpus),
        source_hash=source_hash,
    )


def summarize_repo(
    repo_path: Path, llm_cfg: Any, *, cache: dict[str, Any] | None = None
) -> RepoSummary:
    """Summarise the repo, reusing *cache* when the docs have not changed.

    *cache* is a plain dict (persist it however you like — the store, a JSON file). On a
    hash hit no model call is made. Raises LLMError/ValueError on failure; callers decide
    whether to degrade to the keyword profile, because a fabricated summary would poison
    every triage verdict for that repo.
    """
    corpus = collect_doc_corpus(repo_path)
    if not corpus.strip():
        return RepoSummary()
    digest = corpus_hash(f"v{_PROMPT_VERSION}\n{corpus}")
    if cache is not None and cache.get("source_hash") == digest:
        cached = {k: v for k, v in cache.items() if k in RepoSummary.__dataclass_fields__}
        return RepoSummary(**cached)
    summary = _parse_summary(
        complete(f"{_PROMPT}\n\n# Repository documentation\n{corpus}", llm_cfg, max_tokens=900),
        corpus,
        digest,
    )
    if cache is not None:
        cache.clear()
        cache.update(asdict(summary))
    return summary
