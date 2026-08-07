"""arXiv paper collection — query building and API calls."""

from __future__ import annotations

import logging
import time
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any

import arxiv

from reporadar import arxiv_rate
from reporadar.config import ArxivConfig, QueriesConfig
from reporadar.profiler import RepoProfile

logger = logging.getLogger(__name__)


class CollectionError(Exception):
    """Raised when arXiv collection fails after exhausting retries."""


def _generate_bigram_queries(
    profile: RepoProfile,
    max_bigrams: int = 3,
) -> list[str]:
    """Generate bigram phrase queries from adjacent top keywords.

    Takes the top keywords and generates adjacent pairs as quoted phrases.
    Filters out bigrams where both words are short (< 4 chars).

    Only pairs *single-word* terms. The profiler's TF-IDF already emits bigrams of its
    own, and concatenating those with a neighbour produced three-word phrases that no
    paper contains — `"speech speech recognition"` and `"speech recognition recognition"`
    were two of the three phrase queries built for the whisper repo. A multi-word term is
    already a phrase, and reaches arXiv as one via the keyword path below.
    """
    if not profile.keywords or len(profile.keywords) < 2:
        return []

    terms = [term for term, _weight in profile.keywords if " " not in term]
    bigrams: list[str] = []

    for i in range(len(terms) - 1):
        a, b = terms[i], terms[i + 1]
        # Skip if both words are short/common
        if len(a) < 4 and len(b) < 4:
            continue
        bigrams.append(f'"{a} {b}"')
        if len(bigrams) >= max_bigrams:
            break

    return bigrams


def build_queries(
    profile: RepoProfile,
    queries_cfg: QueriesConfig,
    arxiv_cfg: ArxivConfig,
    max_auto_queries: int = 5,
) -> list[str]:
    """Build arXiv query strings from the repo profile and config.

    Strategy:
    1. Use all user-provided seed queries (scoped to configured categories).
    2. Insert bigram phrase queries from adjacent top keywords.
    3. Auto-generate queries from the top profile keywords.

    Returns a list of arXiv query strings ready for the API.
    """
    if queries_cfg.redact:
        # Redact the *terms*, before they are assembled into query syntax. Doing it
        # to the finished string instead would leave the scaffolding behind — a
        # keyword query would degrade to `(all: ) AND (cat:cs.IR)`, which is not a
        # redacted search but a broken one. A term that redacts away is dropped.
        from reporadar.privacy import compile_patterns, redact, redact_all

        patterns = compile_patterns(queries_cfg.redact)
        queries_cfg = replace(queries_cfg, seed=redact_all(queries_cfg.seed, patterns))
        profile = replace(
            profile,
            keywords=[
                (t, w) for t, w in ((redact(t, patterns), w) for t, w in profile.keywords) if t
            ],
        )

    cat_filter = _category_filter(arxiv_cfg.categories)
    queries: list[str] = []

    # Seed queries from config
    for seed in queries_cfg.seed:
        q = f'all:"{seed}"'
        if cat_filter:
            q = f"({q}) AND ({cat_filter})"
        queries.append(q)

    # Bigram phrase queries (higher priority than single keywords)
    for phrase in _generate_bigram_queries(profile):
        q = f"all:{phrase}"
        if cat_filter:
            q = f"({q}) AND ({cat_filter})"
        if q not in queries:
            queries.append(q)

    # Auto-generated queries from top keywords
    if profile.keywords:
        top_terms = [term for term, _weight in profile.keywords[:max_auto_queries]]
        for term in top_terms:
            # Quote anything with a space. The profiler runs TF-IDF with
            # ngram_range=(1, 2), so bigrams like "speech recognition" reach this line —
            # and on the arXiv API an unquoted space after a field prefix is **OR**, not
            # AND. `all:speech recognition` matched 246,802 papers (essentially "anything
            # about recognition"); `all:"speech recognition"` matches 6,845. Emitting it
            # unquoted made the most specific terms the profiler produces into the
            # broadest queries it sends, and we then kept only the first 50 results.
            q = f'all:"{term}"' if " " in term else f"all:{term}"
            if cat_filter:
                q = f"({q}) AND ({cat_filter})"
            # Skip if it duplicates a seed query
            if q not in queries:
                queries.append(q)

    # Fallback: if no queries at all, search by category only
    if not queries and cat_filter:
        queries.append(cat_filter)

    return queries


def _category_filter(categories: list[str]) -> str:
    """Build an OR-joined category filter string."""
    if not categories:
        return ""
    if len(categories) == 1:
        return f"cat:{categories[0]}"
    parts = [f"cat:{c}" for c in categories]
    return " OR ".join(parts)


def _result_to_paper(result: arxiv.Result) -> dict[str, Any]:
    """Convert an arxiv.Result to our internal paper dict."""
    return {
        "arxiv_id": result.get_short_id(),
        "title": result.title,
        "authors": [a.name for a in result.authors],
        "abstract": result.summary,
        "categories": result.categories,
        "published": result.published.isoformat(),
        "updated": result.updated.isoformat() if result.updated else None,
        "url": result.entry_id,
        "pdf_url": result.pdf_url,
    }


_CLIENTS: dict[tuple[int, int], arxiv.Client] = {}


def _shared_client(page_size: int) -> arxiv.Client:
    """One client per page size, reused for the life of the process.

    `arxiv.Client` spaces its requests using a timestamp stored **on the instance**, so a
    fresh client per call reset the clock and fired immediately. That is how a 22-repo
    collection issued its first request per repo with no spacing and earned HTTP 429 on
    everything after the fifteenth. Reusing the instance makes the library's own
    `delay_seconds` carry across calls.

    Keyed on `id(arxiv.Client)` as well as page size so a test that patches the class gets
    its own entry instead of a cached real client — or, worse, a later test inheriting an
    earlier test's mock.
    """
    key = (id(arxiv.Client), page_size)
    client = _CLIENTS.get(key)
    if client is None:
        if len(_CLIENTS) > 64:  # only reachable under repeated patching; bound the dict
            _CLIENTS.clear()
        client = arxiv.Client(
            page_size=page_size,
            delay_seconds=arxiv_rate.min_interval(),
            num_retries=3,
        )
        arxiv_rate.identify(getattr(client, "_session", None))
        _CLIENTS[key] = client
    return client


def _query_with_retry(
    client: arxiv.Client,
    search: arxiv.Search,
    max_retries: int = 3,
    base_delay: float = 2.0,
) -> list[arxiv.Result]:
    """Execute an arXiv query with exponential backoff on transient errors.

    Raises CollectionError if all retries are exhausted.
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        # Process-wide, shared with signals/integrity.py. Two independent 3-second
        # limiters permit two requests per three seconds, which is not a limit.
        arxiv_rate.wait_turn()
        try:
            return list(client.results(search))
        # `arxiv.ArxivError` is NOT an OSError — it subclasses Exception directly — so
        # `arxiv.HTTPError` (a 429 or 503 from export.arxiv.org) escaped this handler
        # entirely and surfaced as a traceback. Every call site catches only
        # `CollectionError`, including `watcher.py`, so a single throttle response ended
        # a scheduled `rr watch` loop. arXiv throttles for real: sustained polling earned
        # this project's own machine a ~70-minute IP block.
        except (ConnectionError, TimeoutError, OSError, arxiv.ArxivError) as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                # A throttle response is not a flaky socket. Retrying a 429 after 2 s is
                # impolite exactly when the server has said "slow down", and it is how a
                # throttle turns into an IP block.
                if getattr(exc, "status", None) in (429, 503):
                    delay = arxiv_rate.THROTTLED_BACKOFF * (2**attempt)
                else:
                    delay = max(base_delay * (2**attempt), arxiv_rate.min_interval())
                logger.warning(
                    "arXiv query failed (attempt %d/%d): %s. Retrying in %.1fs...",
                    attempt + 1,
                    max_retries,
                    exc,
                    delay,
                )
                time.sleep(delay)
    raise CollectionError(f"arXiv query failed after {max_retries} attempts: {last_exc}")


def collect_papers(
    queries: list[str],
    arxiv_cfg: ArxivConfig,
    on_query_start: Any | None = None,
) -> list[dict[str, Any]]:
    """Execute arXiv queries and return deduplicated paper dicts.

    Deduplication is by arxiv_id (first result wins).

    *on_query_start*, if provided, is called at the start of each query with
    ``(query_index, total_queries, query_string)``.
    """
    client = _shared_client(arxiv_cfg.max_results_per_query)

    cutoff = datetime.now(UTC) - timedelta(days=arxiv_cfg.lookback_days)
    seen_ids: set[str] = set()
    papers: list[dict[str, Any]] = []
    total = len(queries)

    for idx, query_str in enumerate(queries):
        if on_query_start is not None:
            on_query_start(idx, total, query_str)

        logger.info("Querying arXiv: %s", query_str)
        sort_criterion = (
            arxiv.SortCriterion.Relevance
            if arxiv_cfg.sort_by == "relevance"
            else arxiv.SortCriterion.SubmittedDate
        )
        search = arxiv.Search(
            query=query_str,
            max_results=arxiv_cfg.max_results_per_query,
            sort_by=sort_criterion,
            sort_order=arxiv.SortOrder.Descending,
        )

        results = _query_with_retry(client, search)
        for result in results:
            # Skip papers older than lookback window
            if result.published.replace(tzinfo=UTC) < cutoff:
                continue

            paper = _result_to_paper(result)
            if paper["arxiv_id"] not in seen_ids:
                seen_ids.add(paper["arxiv_id"])
                paper["matched_query"] = query_str
                papers.append(paper)

    logger.info("Collected %d unique papers from %d queries", len(papers), len(queries))
    return papers
