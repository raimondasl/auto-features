"""One rate limiter for every Semantic Scholar request this process makes.

Semantic Scholar's documented limit for an API key is **1 request per second on all
endpoints** — per key, across endpoints, not per endpoint. Four modules here talk to
``api.semanticscholar.org``:

* ``sources/semantic_scholar.py`` — ``/graph/v1/paper/search``
* ``sources/s2_recommendations.py`` — ``/recommendations/v1/papers``
* ``specter.py`` — ``/graph/v1/paper/batch``
* ``citations.py`` — ``/graph/v1/paper/batch``

Before this module, **three of the four had no rate limiting whatsoever** — only retry
backoff, which fires *after* the server has already been hit too fast — and the fourth
slept *between* queries inside a single ``collect_papers`` call, skipping the last one and
resetting for the next caller. So one ``rr update`` with recommendations, SPECTER2 and the
citation hop enabled issued four independent bursts against a shared 1 RPS budget, and an
eval sweeping 25 repos in one process started each repo with no spacing at all.

That is the exact failure :mod:`reporadar.arxiv_rate` was written for, in its own words:
"Two independent 3-second limiters permit two requests per three seconds whenever both
run." Four independent limiters permit four requests per second. So this follows the same
shape deliberately — one module-level clock, one lock, one interval — rather than inventing
a second pattern for the same problem.

The lock is held *while sleeping*, so concurrent callers queue behind it instead of all
reading the same stale timestamp and waking together.
"""

from __future__ import annotations

import logging
import threading
import time

logger = logging.getLogger(__name__)

# S2's stated ceiling for a key is 1 RPS. The extra 10% is margin, not caution theatre:
# `wait_turn` spaces request *starts* using a monotonic clock whose granularity is not
# guaranteed, and a server counting in fixed one-second windows will see two requests in
# one window if we aim at exactly the boundary. Being 10% slow costs a sweep a few seconds.
DEFAULT_MIN_REQUEST_INTERVAL = 1.1

# What a 429 earns. Retrying a throttle response on the ordinary backoff is impolite
# exactly when the server has said "slow down", and it is how a throttle becomes a block —
# the lesson arXiv taught this project with a ~70-minute IP ban.
THROTTLED_BACKOFF = 30.0

_lock = threading.Lock()
_min_interval = DEFAULT_MIN_REQUEST_INTERVAL
# The monotonic time at which the next request may be issued. A *deadline* rather than a
# "last request at" timestamp, because `time.sleep(d)` is documented to sleep *at least* d
# but is not guaranteed to — measured on Windows it returns up to ~7 ms early, which turned
# a 1.1 s interval into 1.093 s gaps. Sleeping toward a deadline in a loop cannot undershoot.
_next_allowed_at = 0.0


def min_interval() -> float:
    """The interval currently enforced between Semantic Scholar requests, in seconds."""
    return _min_interval


def set_min_interval(seconds: float) -> float:
    """Set the interval and return the previous one.

    Exists for the test suite, which sets it to 0 so mocked requests do not sleep, and for
    a caller who needs to be *slower*. Nothing should use it to go faster than
    :data:`DEFAULT_MIN_REQUEST_INTERVAL` against the live API.

    Setting the interval also re-bases any outstanding deadline down to ``now + seconds``.
    Without that, ``set_min_interval(0)`` does not actually disable the gate while a
    :func:`note_throttled` hold is pending — the suite would inherit a 30-second wait from
    whichever earlier test mocked a 429, which is a cross-test leak *and* a knob that does
    not do what it says.
    """
    global _min_interval, _next_allowed_at
    with _lock:
        previous, _min_interval = _min_interval, max(0.0, float(seconds))
        _next_allowed_at = min(_next_allowed_at, time.monotonic() + _min_interval)
    return previous


def wait_turn() -> float:
    """Block until this process may issue another S2 request. Returns seconds slept."""
    global _next_allowed_at
    with _lock:
        start = time.monotonic()
        # Loop, don't sleep once: a short sleep leaves the deadline unmet and would issue
        # the request early. This is the difference between honouring the interval and
        # honouring it on average.
        while True:
            remaining = _next_allowed_at - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(remaining)
        _next_allowed_at = time.monotonic() + _min_interval
        return max(0.0, time.monotonic() - start)


def note_throttled() -> None:
    """Record that the server said 429, pushing the next turn a full backoff away.

    Without this the caller's own backoff sleeps, then the next caller — a different
    module, sharing this clock — takes its turn immediately because the clock only knows
    about *successful* turns. A 429 is information about the budget, not about one request.

    A zero interval means rate limiting is switched off (the test suite), so the hold is
    switched off with it. Otherwise a mocked 429 would park the shared clock 30 seconds
    into the future and every later test touching S2 would really sleep.
    """
    global _next_allowed_at
    with _lock:
        if _min_interval <= 0:
            return
        _next_allowed_at = time.monotonic() + THROTTLED_BACKOFF
    logger.warning(
        "Semantic Scholar returned 429; holding all S2 requests for %.0fs", THROTTLED_BACKOFF
    )
