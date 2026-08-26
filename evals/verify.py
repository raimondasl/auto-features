"""Hallucination guard: resolve proposed papers against arXiv, then off it.

LLM baselines (and sometimes RepoRadar's non-arXiv sources) can reference papers
that don't exist or misquote IDs. Before judging, every proposed paper is
resolved to real metadata; anything that can't be resolved is counted as a
hallucination and scored 0, so the comparison stays honest.

**Why this is not arXiv-only any more.** The baseline prompt demands an
``arxiv_id`` and this module resolved nothing else, so the two agreed with each
other and the benchmark could not see past arXiv. That is an *upper bound on
what any measurement here can find*, not a property of the literature: [P12]
found 79 judged-actionable non-arXiv papers, and [P16] measured the shipped pool
reaching 1 of 19 papers repositories demonstrably adopted. A baseline allowed to
recommend a Nature paper, resolved by an arXiv-only verifier, produces a pick
that cannot be verified, cannot be judged, and vanishes — while the run looks
like it searched. Widening the verifier is therefore a prerequisite for the v2
prompt, not a companion to it.

**Four outcomes, because three of them are not the model's fault.** The counters
are kept apart on purpose; conflating any two of them has cost this project a
finding before:

* **resolved** — metadata good enough to judge (a title *and* an abstract).
* **hallucinated** — a successful lookup that proved the reference is not real.
  Only this one counts against the recommender.
* **lookup_failed** — our infrastructure could not answer (network, HTTP, rate
  limit). Transient: retry later, and never blame the baseline (C-4 was paid for
  scoring an arXiv throttle as an honest zero).
* **unjudgeable** — the DOI *resolves* at doi.org, so the paper exists, but no
  metadata source we query has an abstract for it. Permanent, not the model's
  fault, and *not* retryable — which is why it cannot be folded into
  ``lookup_failed``: a caller that retries on that counter would mark such a case
  incomplete forever (the C-30 stranding, one layer down).

Resolution is tiered, cheapest and most authoritative first: arXiv for arXiv ids;
then Semantic Scholar's batch endpoint by DOI; then Europe PMC, which carries the
bioRxiv/medRxiv preprints S2 returns null for; then the DOI Handle API, which
answers only *does this exist* and is used to separate `hallucinated` from
`unjudgeable`.
"""

from __future__ import annotations

import json as json_mod
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

import arxiv

from reporadar.citations import _s2_batch_post
from reporadar.paper_id import ARXIV_ARCHIVES, dedup_id, doi_key, is_arxiv_id
from reporadar.sources.europepmc import EPMC_SEARCH
from reporadar.sources.europepmc import _normalize as _epmc_normalize
from reporadar.sources.europepmc import _request_json as _epmc_request


class SourceUnavailable(Exception):
    """Raised when a metadata lookup *errors* (network/HTTP/refusal), as opposed
    to succeeding with no match. A lookup error must NOT be counted as a
    hallucination — otherwise an outage silently turns every real baseline paper
    into an invented one and corrupts the comparison.

    Now raised by the Semantic Scholar and Europe PMC tiers too, not only arXiv;
    `ArxivUnavailable` remains as an alias so existing handlers keep working."""


#: Backwards-compatible alias. The condition was never arXiv-specific — only the
#: sources were — and renaming without an alias would break `except` clauses that
#: are correct.
ArxivUnavailable = SourceUnavailable


def _norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()


# arXiv IDs: new style 2401.12345 (optionally vN), old style hep-th/9901001.
# The old-style alternative is restricted to real arXiv archive prefixes so a stray URL path
# (e.g. researchgate.net/publication/2256929...) is not mistaken for an old-style ID.
#
# The list is IMPORTED, not repeated. It lived only here until 2026-08-26, while
# `paper_id.is_arxiv_id` used `[a-z-]+/\d{7}` — so this module rejected
# `publication/2256929` and the shared predicate accepted it, which is the C-14 shape in the
# module written to prevent it. Found by widening this file to resolve DOIs: the bogus id
# took the arXiv branch and came back `lookup_failed` instead of "not an id at all".
_ARCHIVES = ARXIV_ARCHIVES
_ID_RE = re.compile(
    rf"(?:arxiv[:\s/]*)?(\d{{4}}\.\d{{4,5}}(?:v\d+)?)|((?:{_ARCHIVES})(?:\.[A-Z]{{2}})?/\d{{7}})",
    re.IGNORECASE,
)


def extract_arxiv_ids(text: str) -> list[str]:
    """Pull arXiv IDs out of free text (e.g. a baseline's answer)."""
    ids: list[str] = []
    for m in _ID_RE.finditer(text):
        rid = m.group(1) or m.group(2)
        if rid and rid not in ids:
            ids.append(rid)
    return ids


def _normalize(result: arxiv.Result) -> dict[str, Any]:
    return {
        "arxiv_id": result.get_short_id(),
        "title": result.title.strip().replace("\n", " "),
        "abstract": result.summary.strip().replace("\n", " "),
        "categories": list(result.categories),
        "published": result.published.isoformat() if result.published else "",
        "url": result.entry_id,
    }


def resolve_by_id(client: arxiv.Client, arxiv_id: str) -> dict[str, Any] | None:
    base = dedup_id(arxiv_id)
    try:
        result = next(client.results(arxiv.Search(id_list=[base])), None)
    except arxiv.UnexpectedEmptyPageError:
        return None  # genuinely no such paper (empty feed)
    except Exception as exc:  # noqa: BLE001 — network/HTTP: lookup failed, NOT "not found"
        raise ArxivUnavailable(f"id lookup failed for {base}: {exc}") from exc
    return _normalize(result) if result is not None else None


def resolve_by_title(client: arxiv.Client, title: str) -> dict[str, Any] | None:
    title = title.strip()
    if len(title) < 8:
        return None
    try:
        # Fetch a few and require a near-exact normalized title match, so a
        # superstring hit ("<title>: A Survey") is not accepted as the paper.
        results = []
        for r in client.results(arxiv.Search(query=f'ti:"{title}"', max_results=5)):
            results.append(r)
            if len(results) >= 5:
                break
    except arxiv.UnexpectedEmptyPageError:
        return None
    except Exception as exc:  # noqa: BLE001 — network/HTTP: lookup failed, NOT "not found"
        raise ArxivUnavailable(f"title lookup failed: {exc}") from exc

    want = _norm_title(title)
    for r in results:
        if _norm_title(r.title) == want:
            return _normalize(r)
    return None


DOI_HANDLE_URL = "https://doi.org/api/handles/"
_HANDLE_FOUND = 1  # responseCode 1 = handle exists; 100 = HANDLE NOT FOUND


def doi_exists(doi: str) -> bool | None:
    """Does this DOI resolve? True / False / None when the question could not be asked.

    The Handle System is the registry a DOI *is*, so this is the authoritative existence
    test and it is deliberately the only thing asked of it — it returns a URL, never an
    abstract. Its whole job is to separate "the model invented this" from "the model found
    something real that our metadata sources do not carry", which are different failures
    with different remedies and only one of which is the model's fault.
    """
    key = doi_key(doi).removeprefix("doi:")
    if not key:
        return None
    try:
        req = urllib.request.Request(
            DOI_HANDLE_URL + urllib.parse.quote(key, safe="/"),
            headers={"Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            payload = json_mod.loads(resp.read())
    except urllib.error.HTTPError as exc:
        # 404 is a real answer from this API ("not found"), anything else is our problem.
        return False if exc.code == 404 else None
    except Exception:  # noqa: BLE001 — network/TLS/timeout: unanswerable, not "absent"
        return None
    code = payload.get("responseCode")
    if code == _HANDLE_FOUND:
        return True
    return False if code is not None else None


def resolve_by_doi_s2(doi: str, api_key: str | None = None) -> dict[str, Any] | None:
    """Semantic Scholar by DOI. None when S2 has no usable record (or refused).

    The key defaults from the environment because keyless S2 is rate-limited hard enough to
    turn a resolvable DOI into a 429 and then into a *hallucination* — a wrong answer about
    the model, produced by our own throttling. C-4's rule, one layer out.
    """
    key = doi_key(doi).removeprefix("doi:")
    if not key:
        return None
    api_key = api_key or os.environ.get("SEMANTIC_SCHOLAR_API_KEY") or None
    data = _s2_batch_post([f"DOI:{key}"], "title,abstract,externalIds,year", api_key, 3, 2.0)
    if data is None:
        # `_s2_batch_post` returns None when S2 REFUSED and a list (with per-id nulls) when
        # it answered. Collapsing the two would let a 429 fall through to the existence check
        # and be recorded as a permanent `unjudgeable` — a transient failure frozen into a
        # verdict about the paper. Refusal is raised so the caller counts it as retryable.
        raise ArxivUnavailable(f"Semantic Scholar refused the lookup for {key}")
    if not data:
        return None
    rec = data[0]
    if not rec or not (rec.get("title") or "").strip():
        return None
    # No abstract means nothing to judge on. Reporting it as resolved would put an
    # unscoreable paper into the pool, which is worse than reporting it as unjudgeable.
    if not (rec.get("abstract") or "").strip():
        return None
    year = rec.get("year")
    return {
        "arxiv_id": doi_key(key),
        "title": rec["title"].strip(),
        "abstract": rec["abstract"].strip(),
        "categories": [],
        "published": f"{year}-01-01T00:00:00+00:00" if year else "",
        "url": f"https://doi.org/{key}",
    }


def resolve_by_doi_epmc(doi: str) -> dict[str, Any] | None:
    """Europe PMC by DOI — the bioRxiv/medRxiv preprints S2 returns null for."""
    key = doi_key(doi).removeprefix("doi:")
    if not key:
        return None
    query = urllib.parse.urlencode(
        {"query": f'DOI:"{key}"', "format": "json", "resultType": "core", "pageSize": 1}
    )
    try:
        payload = _epmc_request(f"{EPMC_SEARCH}?{query}")
    except Exception:  # noqa: BLE001 — the adapter raises on refusal; that is not "absent"
        raise ArxivUnavailable(f"Europe PMC lookup failed for {key}") from None
    results = ((payload or {}).get("resultList") or {}).get("result") or []
    if not results:
        return None
    paper = _epmc_normalize(results[0])
    # `_normalize` accepts an empty abstract; this caller cannot, for the reason above.
    return paper if paper and (paper.get("abstract") or "").strip() else None


def resolve_reference(
    ref: str,
    client: arxiv.Client,
    *,
    s2_api_key: str | None = None,
) -> tuple[dict[str, Any] | None, str]:
    """Resolve one reference. Returns (paper, outcome).

    ``outcome`` is one of "resolved", "hallucinated", "lookup_failed", "unjudgeable".
    Classification uses the shared predicates in `reporadar.paper_id` rather than a local
    rule, because "is this an arXiv id" already had three implementations once (C-14).
    """
    ref = ref.strip()
    if not ref:
        return None, "hallucinated"

    if is_arxiv_id(ref):
        try:
            paper = resolve_by_id(client, ref)
        except ArxivUnavailable:
            return None, "lookup_failed"
        return (paper, "resolved") if paper else (None, "hallucinated")

    if not doi_key(ref):
        # Neither an arXiv id nor a DOI — nothing here can look it up, and calling that a
        # hallucination would blame the model for our own missing adapter.
        return None, "unjudgeable"

    # Existence FIRST, metadata second. The registry is cheap, authoritative, and answers the
    # only question that can count against the model; asking it first also sidesteps a real
    # ambiguity downstream, because `_s2_batch_post` returns None both when S2 refuses (429)
    # and when it rejects an id (400). Ordered the other way, an invented DOI came back
    # `lookup_failed` — S2's 400 read as a refusal — which is a wrong answer that would have
    # let a fabricated reference sit in the retry queue forever instead of counting against
    # the recommender.
    exists = doi_exists(ref)
    if exists is None:
        return None, "lookup_failed"
    if not exists:
        return None, "hallucinated"

    for lookup in (lambda: resolve_by_doi_s2(ref, s2_api_key), lambda: resolve_by_doi_epmc(ref)):
        try:
            paper = lookup()
        except SourceUnavailable:
            return None, "lookup_failed"
        if paper:
            return paper, "resolved"

    # The DOI resolves but nobody we ask carries an abstract: real, and unscoreable.
    return None, "unjudgeable"


def resolve_references(
    ids: list[str],
    titles: list[str],
    client: arxiv.Client | None = None,
    *,
    s2_api_key: str | None = None,
) -> tuple[list[dict[str, Any]], int, int, int]:
    """Resolve proposed papers to real metadata, arXiv or otherwise.

    Returns ``(resolved_papers, n_hallucinated, n_lookup_failed, n_unjudgeable)``; see the
    module docstring for why the last three are separate counters and why the fourth cannot
    be folded into the third. ``ids`` may mix arXiv ids and DOIs in any form doi_key accepts.
    Deduplicates on the shared id rule.
    """
    client = client or arxiv.Client(page_size=25, delay_seconds=3.0, num_retries=2)
    resolved: dict[str, dict[str, Any]] = {}
    counts = {"hallucinated": 0, "lookup_failed": 0, "unjudgeable": 0}

    for ref in ids:
        paper, outcome = resolve_reference(ref, client, s2_api_key=s2_api_key)
        if paper is not None:
            resolved.setdefault(dedup_id(paper["arxiv_id"]), paper)
        else:
            counts[outcome] += 1

    for title in titles:
        # Title lookup stays arXiv-only: a free-text title has no authoritative registry to
        # check against, so a cross-source title search would trade hallucinations for
        # near-miss matches. `resolve_by_title` already requires an exact normalised match.
        try:
            paper = resolve_by_title(client, title)
        except ArxivUnavailable:
            counts["lookup_failed"] += 1
            continue
        if paper is None:
            counts["hallucinated"] += 1
        else:
            resolved.setdefault(dedup_id(paper["arxiv_id"]), paper)

    return (
        list(resolved.values()),
        counts["hallucinated"],
        counts["lookup_failed"],
        counts["unjudgeable"],
    )
