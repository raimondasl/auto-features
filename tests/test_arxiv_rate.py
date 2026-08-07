"""Tests for the single process-wide arXiv rate limiter.

The bug this replaces: `collector.py` built a fresh `arxiv.Client` per `collect_papers`
call, and the library spaces requests using a timestamp on the *instance*, so every call
started with a clean clock and fired immediately. Collecting for 22 repos back to back
earned HTTP 429 on everything after roughly the fifteenth. Separately,
`signals/integrity.py` kept its own 3-second clock — and two independent 3-second limiters
permit two requests per three seconds, which is not a limit.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from reporadar import arxiv_rate, collector
from reporadar.signals import integrity


@pytest.fixture(autouse=True)
def _restore() -> object:
    previous = arxiv_rate.min_interval()
    yield
    arxiv_rate.set_min_interval(previous)


class TestTheShippedIntervalIsArxivsStatedCeiling:
    def test_the_default_is_three_seconds(self) -> None:
        """The suite runs with the interval at 0 (see conftest). This asserts what ships,
        so the fixture cannot quietly become the product's behaviour."""
        assert arxiv_rate.DEFAULT_MIN_REQUEST_INTERVAL == 3.0

    def test_a_throttle_response_backs_off_far_longer_than_a_flaky_socket(self) -> None:
        assert arxiv_rate.THROTTLED_BACKOFF >= 10 * arxiv_rate.DEFAULT_MIN_REQUEST_INTERVAL

    def test_the_user_agent_identifies_the_project(self) -> None:
        assert "RepoRadar" in arxiv_rate.USER_AGENT
        assert "github.com" in arxiv_rate.USER_AGENT

    def test_the_interval_cannot_be_set_negative(self) -> None:
        arxiv_rate.set_min_interval(-5)
        assert arxiv_rate.min_interval() == 0.0


class TestWaitTurnSpacesRequests:
    def test_the_second_caller_waits(self) -> None:
        arxiv_rate.set_min_interval(0.25)
        arxiv_rate.wait_turn()
        t0 = time.monotonic()
        arxiv_rate.wait_turn()
        assert time.monotonic() - t0 >= 0.2

    def test_concurrent_callers_queue_rather_than_waking_together(self) -> None:
        """A naive check-then-sleep lets N threads read the same stale timestamp and fire
        at once — the exact burst a rate limiter exists to prevent."""
        arxiv_rate.set_min_interval(0.1)
        arxiv_rate.wait_turn()
        stamps: list[float] = []
        lock = threading.Lock()

        def go() -> None:
            arxiv_rate.wait_turn()
            with lock:
                stamps.append(time.monotonic())

        threads = [threading.Thread(target=go) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        stamps.sort()
        gaps = [b - a for a, b in zip(stamps, stamps[1:], strict=False)]
        assert all(g >= 0.08 for g in gaps), f"requests bunched: {gaps}"


class TestOneClockNotTwo:
    def test_the_collector_and_the_integrity_check_share_the_gate(self) -> None:
        """Two modules with two 3-second clocks issue two requests per three seconds."""
        calls: list[str] = []
        with patch.object(arxiv_rate, "wait_turn", side_effect=lambda: calls.append("x") or 0.0):
            integrity._throttle()
        assert calls == ["x"], "integrity no longer takes a turn at the shared gate"


class TestClientReuse:
    def test_the_same_page_size_reuses_one_client(self) -> None:
        """A fresh client per call is what reset the spacing clock and caused the 429s."""
        collector._CLIENTS.clear()
        with patch.object(collector.arxiv, "Client", MagicMock(return_value=MagicMock())) as cls:
            first = collector._shared_client(50)
            second = collector._shared_client(50)
        assert first is second
        assert cls.call_count == 1

    def test_a_different_page_size_gets_its_own_client(self) -> None:
        collector._CLIENTS.clear()
        with patch.object(
            collector.arxiv, "Client", MagicMock(side_effect=lambda **k: MagicMock())
        ):
            assert collector._shared_client(50) is not collector._shared_client(100)

    def test_the_client_is_built_with_the_shared_interval(self) -> None:
        collector._CLIENTS.clear()
        arxiv_rate.set_min_interval(7.0)
        with patch.object(collector.arxiv, "Client", MagicMock(return_value=MagicMock())) as cls:
            collector._shared_client(50)
        assert cls.call_args.kwargs["delay_seconds"] == 7.0


class TestThrottleResponsesBackOffProperly:
    def _error(self, status: int) -> Exception:
        exc = collector.arxiv.HTTPError("http://x", 1, status)
        return exc

    def test_a_429_waits_far_longer_than_a_timeout(self) -> None:
        slept: list[float] = []
        client = MagicMock()
        client.results.side_effect = self._error(429)
        with (
            patch.object(arxiv_rate, "wait_turn", return_value=0.0),
            patch.object(collector.time, "sleep", side_effect=slept.append),
            pytest.raises(collector.CollectionError),
        ):
            collector._query_with_retry(client, MagicMock(), max_retries=3, base_delay=2.0)
        assert slept and min(slept) >= arxiv_rate.THROTTLED_BACKOFF

    def test_an_ordinary_failure_still_respects_the_polite_floor(self) -> None:
        """Even a plain timeout must not retry faster than arXiv's stated interval.

        `wait_turn` is patched out rather than left live: with the interval at 3 s it would
        do three real sleeps inside one test, which adds nine seconds to the suite and makes
        the assertion depend on wall-clock timing it is not trying to measure.
        """
        arxiv_rate.set_min_interval(3.0)
        slept: list[float] = []
        client = MagicMock()
        client.results.side_effect = TimeoutError("boom")
        with (
            patch.object(arxiv_rate, "wait_turn", return_value=0.0),
            patch.object(collector.time, "sleep", side_effect=slept.append),
            pytest.raises(collector.CollectionError),
        ):
            collector._query_with_retry(client, MagicMock(), max_retries=3, base_delay=2.0)
        assert slept and min(slept) >= 3.0

    def test_every_attempt_takes_a_turn_at_the_gate(self) -> None:
        turns: list[int] = []
        client = MagicMock()
        client.results.side_effect = TimeoutError("boom")
        with (
            patch.object(arxiv_rate, "wait_turn", side_effect=lambda: turns.append(1) or 0.0),
            patch.object(collector.time, "sleep"),
            pytest.raises(collector.CollectionError),
        ):
            collector._query_with_retry(client, MagicMock(), max_retries=3)
        assert len(turns) == 3, "a retry that skips the gate is a burst"
