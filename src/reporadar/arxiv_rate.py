"""One rate limiter for every arXiv request this process makes.

arXiv asks API callers to make **no more than one request every three seconds** and to
identify themselves. This module is the single place that promise is kept, because keeping it
in two places is the same as not keeping it:

* ``collector.py`` built a fresh ``arxiv.Client`` per ``collect_papers`` call. The library
  spaces requests using a timestamp stored **on the client instance**, so every new call
  started with a clean slate and fired immediately. A script that collected for 22 repos back
  to back issued its first request per repo with no spacing at all — and arXiv answered with
  HTTP 429 for every query after roughly the fifteenth repo.
* ``signals/integrity.py`` had its own private ``_throttle()`` against its own
  ``_last_request_at``. Two independent 3-second limiters permit **two requests per three
  seconds** whenever both run, which is exactly what one ``rr update`` does.

So: one module-level clock, one lock, one interval. Every path that talks to
``export.arxiv.org`` calls :func:`wait_turn` first.

The lock is held *while sleeping* on purpose. Concurrent callers queue behind it rather than
all observing the same stale timestamp and waking together — the failure a naive
check-then-sleep produces the moment anything is parallelised.
"""

from __future__ import annotations

import contextlib
import threading
import time

from . import __version__

# arXiv's stated ceiling. Not a tuning knob: going faster is a promise broken, and this
# project has already earned its own machine a ~70-minute IP block by ignoring it.
DEFAULT_MIN_REQUEST_INTERVAL = 3.0

# What a 429 or 503 earns, as opposed to the couple of seconds a flaky socket earns.
# Retrying a throttle response after 2 s is impolite precisely when the server has said
# "slow down", and it is how a throttle becomes a block.
THROTTLED_BACKOFF = 30.0

# arXiv asks callers to identify themselves so they can be contacted rather than banned.
USER_AGENT = f"RepoRadar/{__version__} (+https://github.com/raimondasl/auto-features)"

_lock = threading.Lock()
_min_interval = DEFAULT_MIN_REQUEST_INTERVAL
_last_request_at = 0.0


def min_interval() -> float:
    """The interval currently enforced between arXiv requests, in seconds."""
    return _min_interval


def set_min_interval(seconds: float) -> float:
    """Set the interval and return the previous one.

    Exists for two callers: the test suite, which sets it to 0 so a thousand mocked
    requests do not sleep for an hour, and a caller who needs to be *slower* than the
    default. Nothing should use it to go faster than
    :data:`DEFAULT_MIN_REQUEST_INTERVAL` against the live API.
    """
    global _min_interval
    previous, _min_interval = _min_interval, max(0.0, float(seconds))
    return previous


def wait_turn() -> float:
    """Block until this process may issue another arXiv request. Returns seconds slept."""
    global _last_request_at
    with _lock:
        wait = _min_interval - (time.monotonic() - _last_request_at)
        if wait > 0:
            time.sleep(wait)
        else:
            wait = 0.0
        _last_request_at = time.monotonic()
        return wait


def note_request() -> None:
    """Record that a request happened without waiting for a turn.

    For the library-internal pagination we do not intercept: ``arxiv.Client`` issues its
    own follow-up page requests, and the clock should know about them even though we could
    not gate them.
    """
    global _last_request_at
    with _lock:
        _last_request_at = time.monotonic()


def identify(session: object) -> None:
    """Attach RepoRadar's User-Agent to a requests session, if it has headers.

    Defensive because it reaches into ``arxiv.Client._session``, which is private. A
    missing or differently-shaped session is not worth failing a run over — the request
    still goes out, just as ``python-requests/x.y``.
    """
    headers = getattr(session, "headers", None)
    if headers is None:
        return
    # Politeness is best-effort, never fatal: a differently-shaped session is not worth
    # failing a run over. The request still goes out, just as `python-requests/x.y`.
    with contextlib.suppress(Exception):
        headers["User-Agent"] = USER_AGENT
