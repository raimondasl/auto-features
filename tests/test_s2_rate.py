"""Tests for the process-wide Semantic Scholar rate limiter.

Semantic Scholar's key limit is **1 request per second across all endpoints**, per key.
Four modules here talk to it, and before `s2_rate` three of them had no rate limiting at
all — only retry backoff, which fires after the server has already been hit too fast.

The property that matters is not "each module sleeps" but "**all modules share one
clock**". Two correct-looking 1 RPS limiters permit 2 RPS whenever both run, which is the
exact bug `arxiv_rate` was written for after arXiv IP-banned this machine. So the central
test here drives the real request functions of different modules and asserts they queue
behind each other.
"""

from __future__ import annotations

import json
import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from reporadar import citations, s2_rate
from reporadar.sources import semantic_scholar


@pytest.fixture(autouse=True)
def _reset_clock() -> Any:
    """Each test starts from a known interval and leaves the default behind it."""
    previous = s2_rate.set_min_interval(s2_rate.DEFAULT_MIN_REQUEST_INTERVAL)
    yield
    s2_rate.set_min_interval(previous)


class TestTheGate:
    def test_default_is_at_least_one_second(self) -> None:
        """S2 documents 1 RPS for a key. Faster is a promise broken, not a tuning choice."""
        assert s2_rate.DEFAULT_MIN_REQUEST_INTERVAL >= 1.0

    def test_first_call_does_not_wait(self) -> None:
        """Exactly 0.0, not "about zero".

        `wait_turn` reports accumulated *sleep*, so a turn that did not sleep returns
        exactly zero on every platform. Reporting elapsed wall time instead made this
        0.0 on Windows (coarse clock) and ~7e-07 on Linux — the version that passed
        locally and failed CI three ways.
        """
        s2_rate.set_min_interval(0.05)
        s2_rate._next_allowed_at = 0.0
        assert s2_rate.wait_turn() == 0.0

    def test_second_call_waits(self) -> None:
        s2_rate.set_min_interval(0.05)
        s2_rate.wait_turn()
        assert s2_rate.wait_turn() > 0.0

    def test_the_interval_is_never_undershot(self) -> None:
        """A single `time.sleep(d)` can return early; the gate must not.

        Measured against the live API before this was a loop: a 1.1s interval produced
        1.093s gaps on Windows. Under S2's real 1.0s limit that is still compliant, but a
        limiter that misses its own target by 7ms would miss a tighter one by the same
        margin, and the whole point of the module is that the promise is kept.
        """
        s2_rate.set_min_interval(0.05)
        s2_rate._next_allowed_at = 0.0
        s2_rate.wait_turn()
        for _ in range(10):
            before = time.monotonic()
            s2_rate.wait_turn()
            assert time.monotonic() - before >= 0.05

    def test_set_min_interval_returns_the_previous_value(self) -> None:
        previous = s2_rate.set_min_interval(2.5)
        assert s2_rate.min_interval() == 2.5
        assert s2_rate.set_min_interval(previous) == 2.5

    def test_negative_interval_clamps_to_zero(self) -> None:
        s2_rate.set_min_interval(-5)
        assert s2_rate.min_interval() == 0.0

    def test_throttle_note_pushes_the_next_turn_out(self) -> None:
        """A 429 is information about the shared budget, not about one request.

        Without this, the caller that got the 429 backs off while a *different* module
        sharing the clock takes its turn immediately — hitting a server that just asked
        everyone to slow down.

        Asserted on the deadline rather than on a patched `time.sleep`: `wait_turn` sleeps
        toward a deadline in a loop, so a no-op sleep turns it into a 30-second busy-wait
        instead of something to assert on.
        """
        s2_rate.set_min_interval(0.01)
        s2_rate.wait_turn()
        before = s2_rate._next_allowed_at
        s2_rate.note_throttled()
        held_for = s2_rate._next_allowed_at - time.monotonic()
        assert held_for > s2_rate.THROTTLED_BACKOFF - 1.0, held_for
        assert s2_rate._next_allowed_at > before

    def test_throttle_note_is_inert_when_limiting_is_disabled(self) -> None:
        """`set_min_interval(0)` means no limiting; a 429 hold would defeat that.

        This is also cross-test isolation: the suite disables the gate for speed, and
        without it whichever test mocked a 429 first would leave a 30-second hold behind
        for every later test that touches S2.

        The suite disables the gate for speed. Without this, one mocked 429 would park the
        shared clock 30 s ahead and every later test touching S2 would really sleep.
        """
        s2_rate.set_min_interval(0.0)
        s2_rate.note_throttled()
        with patch("reporadar.s2_rate.time.sleep") as sleep:
            assert s2_rate.wait_turn() == 0.0
        assert not sleep.called


class TestEveryS2ModuleSharesTheClock:
    """The point of the module: one budget, not one per caller."""

    def _json_response(self, payload: Any) -> MagicMock:
        resp = MagicMock()
        resp.read.return_value = json.dumps(payload).encode()
        resp.__enter__ = lambda s: s
        resp.__exit__ = lambda *a: False
        return resp

    def test_search_and_batch_queue_behind_each_other(self) -> None:
        """A search then a /paper/batch must be spaced, though they live in different modules.

        Driven through the real request functions rather than by asserting each calls
        `wait_turn` — a test that only checks the call would pass if both modules kept
        private clocks, which is precisely the bug.
        """
        s2_rate.set_min_interval(0.02)
        s2_rate._next_allowed_at = 0.0
        turns: list[float] = []
        real_wait = s2_rate.wait_turn

        def recording_wait() -> float:
            slept = real_wait()
            turns.append(slept)
            return slept

        with (
            patch("reporadar.s2_rate.wait_turn", recording_wait),
            patch("urllib.request.urlopen", return_value=self._json_response({"data": []})),
        ):
            semantic_scholar.search_papers("q", api_key="k")
            citations._s2_batch_post(
                ["arXiv:2401.00001"], "citationCount", "k", max_retries=1, base_delay=0.0
            )

        assert len(turns) == 2, turns
        # The first is free; the second must have waited, which can only happen if the
        # second module read the same clock the first one set.
        assert turns[0] == 0.0
        assert turns[1] > 0.0

    def test_recommendations_uses_the_gate(self) -> None:
        s2_rate.set_min_interval(0.0)
        from reporadar.sources import s2_recommendations

        with (
            patch("reporadar.s2_rate.wait_turn", return_value=0.0) as gate,
            patch(
                "urllib.request.urlopen",
                return_value=self._json_response({"recommendedPapers": []}),
            ),
        ):
            s2_recommendations.fetch_recommendations(["2401.00001"], api_key="k")
        assert gate.called, "the recommendations endpoint bypassed the shared 1 RPS gate"

    def test_specter_goes_through_citations_batch(self) -> None:
        """specter.py imports `_s2_batch_post`, so gating that one covers both."""
        from reporadar import specter

        assert specter._s2_batch_post is citations._s2_batch_post
