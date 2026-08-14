"""arXiv paper collection — query building and API calls."""

from __future__ import annotations

import logging
import re
import time
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any

import arxiv

from reporadar import arxiv_cache, arxiv_rate
from reporadar.config import BIGRAM_MODES, ArxivConfig, QueriesConfig
from reporadar.profiler import RepoProfile

logger = logging.getLogger(__name__)


class CollectionError(Exception):
    """Raised when arXiv collection fails after exhausting retries."""


def _generate_bigram_queries(
    profile: RepoProfile,
    max_bigrams: int = 3,
    # Kept in step with QueriesConfig.bigrams deliberately. A private helper whose default
    # differs from the shipped config is a trap: `_generate_bigram_queries(profile)` in a
    # future test or diagnostic would silently measure a policy the product does not use.
    mode: str = "verified",
) -> list[str]:
    """Bigram phrase queries from the top keywords, under one of three policies.

    ``adjacent`` (the original behaviour) pairs each keyword with its neighbour in the
    TF-IDF ranking. **Nothing in that pairing requires the two words to belong together** —
    they merely scored next to each other — so it emits phrases the repository never
    contains. Measured 2026-08-12, it built ``"use page"`` and ``"page refer"`` for duckdb,
    ``"server code"`` for ruff, ``"data cd"`` for redis. Asked those, DBLP returned a
    content analysis of social-media posts, serverless computing, and a chemical-compound
    database; asked the benchmark's hand-written queries instead, it returned *Incremental
    Fusion: Unifying Compiled and Vectorized Query Execution*. The source was answering
    exactly what it was asked.

    arXiv hides this: a category clause (``AND (cat:cs.DB)``) keeps results in the right
    field however meaningless the phrase. Keyword sources have no category to fall back
    on — :func:`to_plain_keywords` correctly strips it — so they get the bare phrase. That
    asymmetry is why the 25-case benchmark, which runs on arXiv, cannot resolve a defect
    this visible: **all three arms tie there** (see below).

    ``verified`` (the default since 2026-08-12) keeps the same candidates but emits one only
    if it occurs, literally and adjacently, in the profiled text
    (:attr:`RepoProfile.corpus_phrases`).

    ``none`` emits nothing, leaving single keywords and anchors to carry retrieval. It was
    **measured worse** — −0.48 net@2/case and precision 0.914 → 0.880 across 25 cases — so
    the obvious "just delete the junk phrases" fix is refuted rather than untested. A
    meaningless phrase is still a query, and dropping three of five shrinks the pool.

    Only *single-word* terms are paired, under every mode. The profiler's TF-IDF emits
    bigrams of its own, and concatenating those with a neighbour produced three-word
    phrases that no paper contains — ``"speech speech recognition"`` and ``"speech
    recognition recognition"`` were two of the three phrase queries built for the whisper
    repo. A multi-word term is already a phrase and reaches arXiv as one via the keyword
    path in :func:`build_queries`.
    """
    if mode not in BIGRAM_MODES:
        raise ValueError(f"Unknown bigram mode {mode!r}; expected one of {BIGRAM_MODES}")
    if mode == "none":
        return []
    if not profile.keywords or len(profile.keywords) < 2:
        return []

    terms = [term for term, _weight in profile.keywords if " " not in term]
    # A set, not a list scan: `corpus_phrases` is small but this runs per candidate pair.
    observed = set(profile.corpus_phrases)
    bigrams: list[str] = []

    for i in range(len(terms) - 1):
        a, b = terms[i], terms[i + 1]
        # Skip if both words are short/common
        if len(a) < 4 and len(b) < 4:
            continue
        if mode == "verified" and f"{a} {b}" not in observed:
            continue
        bigrams.append(f'"{a} {b}"')
        if len(bigrams) >= max_bigrams:
            break

    return bigrams


_CAT_TERM_RE = re.compile(r"\bcat:\S+")
_FIELD_PREFIX_RE = re.compile(r"\b(?:all|ti|abs|au|co|jr|rn|id):")
_BOOLEAN_RE = re.compile(r"\b(?:AND NOT|ANDNOT|AND|OR|NOT)\b")


def to_plain_keywords(query: str) -> str:
    """Strip arXiv query syntax down to the words a keyword search can use.

    Every non-arXiv source — DBLP, bioRxiv, OpenAlex, Semantic Scholar, IACR — takes a
    plain keyword string, while :func:`build_queries` emits arXiv's boolean grammar.
    Callers used to bridge that with ``q.replace("all:", "").strip('"')``, which leaves
    the current form,

        (all:"key cryptography") AND (cat:cs.CR)

    as ``("key cryptography") AND (cat:cs.CR)`` — sent verbatim as a search string.

    **It was never right, and the first telling of this was wrong.** That transform was
    described here as "written for an older query shape" and broken later by drift. Git
    says otherwise: ``build_queries`` emitted the parenthesised ``(...) AND (...)`` form in
    `29ecffa` (2026-02-22), and the one-liner was written the *next day* in `18dfe51`
    against a builder that already produced it. There was no working era to regress from,
    so the blast radius is every non-arXiv fetch this project has ever made, not a window.

    Measured 2026-08-12 against real ``build_queries`` output, the two failure modes are
    opposite and the quiet one is worse:

    - **DBLP and IACR return nothing.** All-time lookback, so no date filter is involved:
      0 hits on every malformed query; 1 and 4 hits on the repaired ones.
    - **bioRxiv returns everything.** Its filter keeps a paper if any query word of more
      than two characters appears in the title or abstract, and the surviving word is the
      boolean operator ``AND``. Every real term matched 0/90 papers; ``and`` matched 90/90.
      Enabling bioRxiv did not add biology papers, it disabled the topical filter.

    The fix lives here, beside the function whose output it consumes, so the two cannot
    drift apart again — and `tests/test_collector.py` pins it against real `build_queries`
    output rather than hand-written strings, which is what let this go unnoticed. It also
    asserts every bridging call site *uses* this function: routing two of five sources
    through it and leaving three on the one-liner is the state that shipped once already.
    """
    # Category terms are dropped whole — `cat:cs.CR` carries no keyword, and stripping only
    # the prefix would turn the no-keyword fallback query `cat:cs.CR OR cat:cs.LG` into a
    # search for the literal words "cs.CR cs.LG".
    text = _CAT_TERM_RE.sub(" ", query)
    text = _BOOLEAN_RE.sub(" ", text)
    text = _FIELD_PREFIX_RE.sub(" ", text)
    text = re.sub(r'[()"]', " ", text)
    return " ".join(text.split())


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
    for phrase in _generate_bigram_queries(profile, mode=queries_cfg.bigrams):
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


def _published_before(published: str | None, cutoff: datetime) -> bool:
    """Is this paper older than the window? Unparseable or missing dates are KEPT.

    The stored `published` is whatever `_result_to_paper` wrote, and the arxiv library has
    handed back both naive and aware datetimes over its life — so a naive string is
    assumed UTC rather than allowed to raise mid-collection. A date we cannot read is not
    evidence the paper is old, and dropping it would quietly shrink a pool.
    """
    if not published:
        return False
    try:
        when = datetime.fromisoformat(published)
    except ValueError:
        return False
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return when < cutoff


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
    attempt = 0
    throttle_waited = 0.0
    while attempt < max_retries or throttle_waited < arxiv_rate.THROTTLE_BUDGET_S:
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
            throttled = getattr(exc, "status", None) in (429, 503)
            # A throttle is not a flaky socket, and it outlasts three retries. Ordinary
            # errors get `max_retries` attempts; a throttle gets a TIME budget, because
            # giving up after 90 seconds is how two benchmark repos ended up with an empty
            # pool that then scored as a legitimate zero.
            if throttled:
                if throttle_waited >= arxiv_rate.THROTTLE_BUDGET_S:
                    break
                delay = min(
                    arxiv_rate.THROTTLED_BACKOFF * (2**attempt),
                    arxiv_rate.THROTTLE_MAX_SLEEP_S,
                )
                throttle_waited += delay
                logger.warning(
                    "arXiv throttled us (HTTP %s). Waiting %.0fs; %.0fs of %.0fs budget used.",
                    getattr(exc, "status", "?"),
                    delay,
                    throttle_waited,
                    arxiv_rate.THROTTLE_BUDGET_S,
                )
            elif attempt < max_retries - 1:
                delay = max(base_delay * (2**attempt), arxiv_rate.min_interval())
                logger.warning(
                    "arXiv query failed (attempt %d/%d): %s. Retrying in %.1fs...",
                    attempt + 1,
                    max_retries,
                    exc,
                    delay,
                )
            else:
                break
            time.sleep(delay)
            attempt += 1
    raise CollectionError(
        f"arXiv query failed after {attempt + 1} attempts "
        f"({throttle_waited:.0f}s waiting out throttles): {last_exc}"
    )


def collect_by_ids(ids: list[str], batch_size: int = 100) -> list[dict[str, Any]]:
    """Fetch paper metadata for specific arXiv ids, in batches, through the shared gate.

    The counterpart to :func:`collect_papers` for channels that discover *identifiers*
    rather than run queries — the HyDE dense search and the citation hop both do. It
    deliberately applies **no date cutoff**: those channels exist to reach the seminal
    older work a recency window structurally cannot see, and re-filtering their output by
    date here would undo the only thing they are for.

    Missing ids are dropped silently by the API, so the result may be shorter than the
    input; a batch that fails outright raises, rather than returning a short list that a
    caller would read as "these papers do not exist".
    """
    if not ids:
        return []
    client = _shared_client(batch_size)
    papers: list[dict[str, Any]] = []
    seen: set[str] = set()
    for start in range(0, len(ids), batch_size):
        chunk = ids[start : start + batch_size]
        # Sorted, so the same set of ids in a different order is one cache entry rather
        # than a new fetch — HyDE's id lists come back ranked and the order varies.
        cache_fields = {"kind": "ids", "ids": sorted(chunk)}
        cached = arxiv_cache.get(cache_fields)
        if cached is None:
            search = arxiv.Search(id_list=chunk, max_results=len(chunk))
            cached = [_result_to_paper(r) for r in _query_with_retry(client, search)]
            arxiv_cache.put(cache_fields, cached, empty_is_real=True)
        for paper in cached:
            if paper["arxiv_id"] not in seen:
                seen.add(paper["arxiv_id"])
                papers.append({**paper, "matched_query": "hyde"})
    logger.info("Fetched %d of %d requested arXiv ids", len(papers), len(ids))
    return papers


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
        # Everything that can change the response. `lookback_days` is deliberately NOT
        # here: it filters results below rather than changing the request, so a 90-day run
        # can reuse an all-time fetch, and including it would split the cache pointlessly.
        cache_fields = {
            "kind": "search",
            "query": query_str,
            "max_results": arxiv_cfg.max_results_per_query,
            "sort_by": arxiv_cfg.sort_by,
        }
        cached = arxiv_cache.get(cache_fields)
        if cached is not None:
            fresh_papers = cached
        else:
            search = arxiv.Search(
                query=query_str,
                max_results=arxiv_cfg.max_results_per_query,
                sort_by=sort_criterion,
                sort_order=arxiv.SortOrder.Descending,
            )
            # `_query_with_retry` raises rather than returning [] when it gives up, so
            # reaching this line means arXiv answered — an empty answer included.
            fresh_papers = [_result_to_paper(r) for r in _query_with_retry(client, search)]
            arxiv_cache.put(cache_fields, fresh_papers, empty_is_real=True)

        for paper in fresh_papers:
            # Skip papers older than lookback window. Applied AFTER the cache so one stored
            # response serves any window.
            if _published_before(paper.get("published"), cutoff):
                continue
            if paper["arxiv_id"] not in seen_ids:
                seen_ids.add(paper["arxiv_id"])
                paper = {**paper, "matched_query": query_str}
                papers.append(paper)

    logger.info("Collected %d unique papers from %d queries", len(papers), len(queries))
    return papers
