"""An on-disk cache for arXiv query responses, so we stop asking the same thing repeatedly.

`arxiv_rate` makes this project polite about **how fast** it asks arXiv. It says nothing
about **how often it asks the same question**, and that turned out to be the binding
constraint. On 2026-08-12 this machine issued roughly 760 arXiv requests in a day — three
phrase-query arms (~597) plus an S2 yield probe (~162) — and the last two benchmark cases
were refused after 930 s of waiting out throttles. The rate limiter was working correctly
the whole time: the failures arrived at cases 24 and 25, not at case 2, which is the
signature of a volume ceiling rather than a rate violation.

**The waste was total.** A 25-case sweep issues 174 queries, and those queries are
byte-identical between runs — same repos, same profiles, same `build_queries` output. The
three arms fetched the same pool three times and the probe fetched it a fourth.

So: cache the response, keyed on everything that could change it. A repeat sweep then costs
zero requests, experiments stop competing with diagnostics for one shared budget, and a
long run cannot lose its last cases to a throttle it spent on data it already had.

**Off unless asked.** `rr update` wants fresh papers, and serving a 6-hour-old answer to a
daily digest is a behaviour change nobody measured. So the cache does nothing until
:func:`configure` is called with a directory — the eval harness and the diagnostics call it,
the product does not. This is the same reasoning as `--rr-frozen-pool`: reuse is a
deliberate, labelled act, never a silent default.

Entries record what they were keyed on, so a cache directory can be audited by reading it
rather than by trusting this module.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Long by design. The point is reproducibility and politeness across a session or a day of
# experiments, not freshness — anything that needs fresh results should not be caching.
DEFAULT_TTL_S = 7 * 24 * 3600

_directory: Path | None = None
_ttl_s: float = DEFAULT_TTL_S
_stats = {"hits": 0, "misses": 0, "writes": 0, "expired": 0}


def configure(directory: Path | str | None, ttl_s: float = DEFAULT_TTL_S) -> None:
    """Enable caching into *directory*, or disable it with ``None``."""
    global _directory, _ttl_s
    _directory = Path(directory) if directory is not None else None
    _ttl_s = max(0.0, float(ttl_s))
    if _directory is not None:
        _directory.mkdir(parents=True, exist_ok=True)
        logger.info("arXiv cache enabled at %s (ttl %.0fs)", _directory, _ttl_s)


def enabled() -> bool:
    return _directory is not None and _ttl_s > 0


def stats() -> dict[str, int]:
    """Hit/miss counts, so a run can report how many requests the cache actually saved."""
    return dict(_stats)


def reset_stats() -> None:
    for k in _stats:
        _stats[k] = 0


def _key(fields: dict[str, Any]) -> str:
    """A digest of everything that determines the response.

    Sorted and JSON-encoded rather than str()-formatted so that dict ordering cannot
    produce two keys for one query — the kind of drift that made a *verdict* cache
    keyed without its prompt return answers to a question nobody asked.
    """
    canonical = json.dumps(fields, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:24]


def get(fields: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Cached papers for this query, or None on a miss, an expiry, or a corrupt entry."""
    if not enabled():
        return None
    assert _directory is not None
    path = _directory / f"{_key(fields)}.json"
    if not path.is_file():
        _stats["misses"] += 1
        return None
    try:
        entry = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        # A damaged entry is a miss, never an empty result: returning [] here would put a
        # silent zero into a pool, which is the failure this project has paid for twice.
        logger.warning("arXiv cache entry unreadable (%s); refetching: %s", path.name, exc)
        _stats["misses"] += 1
        return None
    if time.time() - entry.get("cached_at", 0) > _ttl_s:
        _stats["expired"] += 1
        _stats["misses"] += 1
        return None
    papers = entry.get("papers")
    if not isinstance(papers, list):
        _stats["misses"] += 1
        return None
    _stats["hits"] += 1
    return papers


def put(
    fields: dict[str, Any],
    papers: list[dict[str, Any]],
    *,
    empty_is_real: bool = False,
) -> None:
    """Store a response.

    An empty list and a failed fetch are the same bytes on disk, and this project has
    already scored seven pools of "no papers" that were really an arXiv 429 storm — so an
    empty result is dropped unless the caller states that it *observed* one.

    ``empty_is_real=True`` is that statement, and it belongs at call sites that can prove
    it. :func:`collector._query_with_retry` **raises** ``CollectionError`` when retries are
    exhausted rather than returning ``[]``, so any list it returns is an answer arXiv
    actually gave. Measured on `rag`, 2 of 5 queries genuinely match nothing; refusing to
    cache those spent a request on every run forever to guard against a failure mode that
    cannot reach this function from there.

    The flag defaults to False so a future caller without that guarantee gets the safe
    behaviour by omission rather than by remembering.
    """
    if not enabled():
        return
    if not papers and not empty_is_real:
        return
    assert _directory is not None
    path = _directory / f"{_key(fields)}.json"
    payload = {
        "cached_at": time.time(),
        # Stored so a cache directory can be audited by reading it, and so a key collision
        # would be visible rather than silently serving the wrong query's papers.
        "keyed_on": fields,
        "papers": papers,
    }
    tmp = path.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(path)  # atomic, so a killed run cannot leave a half-written entry
        _stats["writes"] += 1
    except OSError as exc:
        logger.warning("could not write arXiv cache entry: %s", exc)
        tmp.unlink(missing_ok=True)
