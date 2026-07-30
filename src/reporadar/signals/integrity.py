"""Withdrawal detection for arXiv papers (Feature 9, integrity half).

A withdrawn paper is the one case where RepoRadar's ranking can be actively
harmful: it reads as a fresh, on-topic result and the reader may spend an hour on
work its own authors have retracted. So withdrawal is a **hard multiplicative
penalty**, not a weighted component — a withdrawn paper must not be able to reach
Top Picks by scoring well everywhere else.

**arXiv exposes withdrawal only as free text.** There is no status field: a
withdrawn paper's Atom entry keeps its original title, and its ``<summary>`` is
often the complete original abstract. The notice lives in ``<arxiv:comment>``,
written by hand, so the matcher *is* the feature.

The asymmetry that makes it work: a comment is short curated metadata, so a bare
"withdrawn" in it is unambiguous, while an abstract is prose in which a paper may
legitimately discuss a drug withdrawn from the market or a withdrawn auction bid.
The comment therefore gets a liberal matcher and prose an anchored one.

Measured, and worth reading before changing the regexes — **how the sample is drawn
matters more than the sample size here**:

* sampled by the bare token ``withdrawn``: the phrasing arXiv's own help pages
  suggest catches **29%**, the matcher below **100%** (100/100, and 300/300 on a
  larger draw). The single most common real comment is just ``"Withdrawn"``.
* that sample **cannot contain** a paper whose notice only ever says "retracted" or
  "the authors withdrew this" — and those stayed invisible through two rounds of
  validation because of it. Sampled by those verbs instead: **83%** (30/36,
  ``withdrew``) and **85%** (63/74, ``retracted``). Most remaining misses are correct
  refusals, where the notice belongs to a *cited* paper rather than to this one.
* precision over 600 ordinary papers across six fields: two flags, **both of which
  turned out to be genuine withdrawals** — no confirmed false positive.

Known limitation, measured and accepted: "withdrawn from the conference proceedings"
flags a paper that is still a live preprint. An exhaustive phrase search found two
such papers in all of arXiv, and the cost is one paper demoted to Muted *with its
reason shown* — cheaper than a heuristic that might suppress real withdrawals.

Matching on title + abstract alone — the fields the store already holds, i.e. no
network at all — is **not** a reliable substitute. Two independent samples put its
recall at 46% and 0%, the spread again coming from how each was drawn; the honest
summary is that some authors replace the title/abstract with the notice and many
only add a comment. That fallback still runs, because it is free, covers papers whose
ids arXiv cannot resolve, and is occasionally right — but the comment fetch is what
makes the check trustworthy.
"""

from __future__ import annotations

import logging
import re
import time
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

# arXiv asks for 3s between requests. ``arxiv.Client``'s own delay is *instance*
# scoped (it guards on ``self._last_request_dt``) and collector.py builds its own
# client, so two clients would mean two independent clocks and double the real
# request rate. Hence a module-level interval, as in sources/dblp.py.
_MIN_REQUEST_INTERVAL_S = 3.0
_last_request_at = 0.0

# Verified: 200 ids in a single id_list request. 100 keeps the URL well clear of
# any server-side length limit.
ID_BATCH = 100

# Liberal, for the short metadata comment field only. Covers "withdrew" and
# "retracted" as well as "withdrawn": a sample drawn by searching for the token
# ``withdrawn`` cannot, by construction, contain the papers that only ever say
# "retracted" or "the authors withdrew this" — and measuring on such a sample is how
# those phrasings stayed invisible through two rounds of validation.
_COMMENT_RE = re.compile(
    r"\b(?:withdraw(?:n|al|ing|s)?|withdrew|retract(?:ed|ing|ion|ions|s)?)\b", re.IGNORECASE
)

# A comment that *opens* with the notice is about this paper, whatever it cites
# afterwards: "Withdrawn; superseded by arXiv:2504.01234" is a withdrawal, but the
# cross-reference guard below would otherwise discard it.
_LEAD_NOTICE_RE = re.compile(
    r"^\s*(?:v\d+\s*[:.\-]?\s*)?"
    r"(?:this\s+|the\s+)?(?:paper|article|manuscript|submission|preprint|version)?\s*"
    r"(?:has\s+been\s+|is\s+|was\s+)?"
    r"(?:withdraw(?:n|al)?|withdrew|retract(?:ed|ion)?)\b",
    re.IGNORECASE,
)

# "The *previous* version was withdrawn" describes a superseded version, not this
# paper — a live, corrected paper. Only the anchored prose matcher may override this.
_PRIOR_VERSION_RE = re.compile(
    r"\b(?:previous|previously|earlier|prior|old|first|initial|v\d+)\s+"
    r"(?:version|draft|submission|revision)\b",
    re.IGNORECASE,
)

# Anchored, for prose (title, abstract) where "withdrawn" may be the subject matter.
_PROSE_RE = re.compile(
    r"\b(?:this|the)\s+"
    r"(?:paper|article|manuscript|submission|preprint|report|note|work|version|entry)"
    r"\s+(?:ha[sd]\s+been|have\s+been|is|was|as\s+been|been)\s+(?:withdrawn|retracted)"
    r"|\b(?:withdrawn|retracted|removed)\s+by\s+(?:the\s+)?(?:author|arxiv|submitter|admin)"
    r"|\bauthors?\s+(?:have\s+)?withdrew\b"
    r"|\barxiv\s+admin(?:istrator)?s?\s+note:[^.]*(?:withdraw|retract|remov)",
    re.IGNORECASE,
)

# A title that *is* the notice, e.g. "Withdrawn" or "Some Result (withdrawn)".
_TITLE_RE = re.compile(r"(?:^\s*\[?\s*withdrawn\b)|(?:\(\s*withdrawn\s*\)\s*$)", re.IGNORECASE)

# A comment citing another arXiv paper — "incorporates arXiv:2011.10199 ..., which
# has been withdrawn" describes *that* paper, not this one, and was the only false
# positive in the 500-paper control sample. Such a comment falls back to the
# anchored prose matcher, which requires "this/the paper has been withdrawn" as
# adjacent words and so ignores a withdrawal attached to a cited title.
_XREF_RE = re.compile(r"\barxiv[:\s]*\d{4}\.\d{4,5}", re.IGNORECASE)


# How long a withdrawal check stays good, and how many papers one run may look up.
# Withdrawal is rare and not urgent — a paper withdrawn today can be caught next week
# — but the cost is real: arXiv wants 3s between requests, so an unbounded pass over
# an all-time store would spend minutes per run on a signal that fires for <1% of
# papers. Newly-seen papers are always checked; the cap only limits re-checks.
RECHECK_AFTER_DAYS = 7
MAX_CHECKS_PER_RUN = 300


def stale_ids(
    arxiv_ids: list[str],
    checked_at: dict[str, str],
    recheck_after_days: int = RECHECK_AFTER_DAYS,
    limit: int = MAX_CHECKS_PER_RUN,
) -> list[str]:
    """Pick which ids to look up: never-checked first, then the most stale.

    *checked_at* maps an id to its stored ISO timestamp. Papers checked more recently
    than *recheck_after_days* are skipped entirely; the rest are ordered oldest-first
    so a capped run still makes progress through a large backlog instead of
    re-checking the same head every time.
    """
    cutoff = datetime.now(UTC) - timedelta(days=recheck_after_days)
    unchecked: list[str] = []
    stale: list[tuple[str, str]] = []
    for arxiv_id in arxiv_ids:
        stamp = checked_at.get(arxiv_id)
        if not stamp:
            unchecked.append(arxiv_id)
            continue
        try:
            when = datetime.fromisoformat(stamp)
        except ValueError:
            unchecked.append(arxiv_id)
            continue
        if when.tzinfo is None:
            when = when.replace(tzinfo=UTC)
        if when < cutoff:
            stale.append((arxiv_id, stamp))
    stale.sort(key=lambda pair: pair[1])
    return (unchecked + [aid for aid, _ in stale])[:limit]


def _throttle() -> None:
    """Block until at least ``_MIN_REQUEST_INTERVAL_S`` since the last request."""
    global _last_request_at
    wait = _MIN_REQUEST_INTERVAL_S - (time.monotonic() - _last_request_at)
    if wait > 0:
        time.sleep(wait)
    _last_request_at = time.monotonic()


def detect_withdrawal(
    title: str = "", abstract: str = "", comment: str | None = None
) -> str | None:
    """Return the field a withdrawal notice was found in, or ``None``.

    Returns the field name (``"title"``, ``"comment"``, ``"abstract"``) rather than
    a bool so a digest can say *why* a paper was flagged and a maintainer can tell
    a high-confidence hit from a marginal one.
    """
    if _TITLE_RE.search(title) or _PROSE_RE.search(title):
        return "title"

    if comment:
        anchored = bool(_PROSE_RE.search(comment))
        if _LEAD_NOTICE_RE.search(comment):
            # Opens with the notice, so it is about *this* paper no matter what it
            # cites afterwards ("Withdrawn; superseded by arXiv:2504.01234").
            return "comment"
        if _PRIOR_VERSION_RE.search(comment) and not anchored:
            # "The previous version was withdrawn" is a live, corrected paper.
            return None
        if _XREF_RE.search(comment):
            # Cites another paper: demand the anchored phrasing, so a withdrawal
            # attached to the *cited* title doesn't flag this paper.
            if anchored:
                return "comment"
        elif anchored or _COMMENT_RE.search(comment):
            # The anchored pattern also covers phrasings the liberal one must not:
            # "removed by arXiv admin" is a withdrawal, but a bare "removed" is one of
            # the commonest words in an ordinary comment ("removed the appendix").
            return "comment"

    if _PROSE_RE.search(abstract):
        return "abstract"

    return None


@lru_cache(maxsize=1)
def _client() -> Any:
    """One shared ``arxiv.Client``, so its instance-scoped delay actually applies."""
    import arxiv

    return arxiv.Client(page_size=ID_BATCH, delay_seconds=_MIN_REQUEST_INTERVAL_S, num_retries=3)


def _base_id(arxiv_id: str) -> str:
    """Strip a version suffix: ``1407.6496v2`` -> ``1407.6496``."""
    return re.sub(r"v\d+$", "", arxiv_id)


def fetch_comments(arxiv_ids: list[str]) -> dict[str, str]:
    """Fetch the ``comment`` field for *arxiv_ids*, keyed by the id passed in.

    Returns only ids the API actually answered for. **arXiv silently drops unknown
    ids from ``id_list``** rather than erroring, so a short result means "unknown",
    never "clean" — callers must not infer absence of withdrawal from a missing key.
    Non-arXiv ids (synthetic ``ss:``/``oa:``/``biorxiv:`` ids from other sources)
    are skipped, since ``id_list`` rejects the whole request on a malformed id.
    """
    resolvable = [a for a in arxiv_ids if ":" not in a]
    if not resolvable:
        return {}

    try:
        import arxiv
    except ImportError:  # pragma: no cover - arxiv is a hard dependency
        logger.warning("arxiv package unavailable; skipping withdrawal check")
        return {}

    by_base = {_base_id(a): a for a in resolvable}
    comments: dict[str, str] = {}

    bases = list(by_base)
    for start in range(0, len(bases), ID_BATCH):
        chunk = bases[start : start + ID_BATCH]
        _throttle()
        try:
            search = arxiv.Search(id_list=chunk, max_results=len(chunk))
            for result in _client().results(search):
                # entry_id looks like http://arxiv.org/abs/1407.6496v2
                base = _base_id(result.entry_id.rsplit("/", 1)[-1])
                original = by_base.get(base)
                if original is not None:
                    comments[original] = result.comment or ""
        except Exception as exc:
            logger.warning(
                "arXiv comment fetch failed for %d ids (%s); those stay unknown",
                len(chunk),
                exc,
            )
            continue

    return comments


def find_withdrawn(papers: list[dict[str, Any]], comments: dict[str, str]) -> dict[str, str]:
    """Return ``{arxiv_id: field}`` for every paper that looks withdrawn.

    *comments* comes from :func:`fetch_comments`; papers missing from it are still
    checked against their stored title/abstract, which catches roughly half of real
    withdrawals on its own and costs nothing.
    """
    flagged: dict[str, str] = {}
    for paper in papers:
        arxiv_id = paper.get("arxiv_id")
        if not arxiv_id:
            continue
        field = detect_withdrawal(
            title=paper.get("title") or "",
            abstract=paper.get("abstract") or "",
            comment=comments.get(arxiv_id),
        )
        if field:
            flagged[arxiv_id] = field
    return flagged
