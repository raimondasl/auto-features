"""Trend detection — track keyword frequency changes across runs."""

from __future__ import annotations

from typing import Any

from reporadar.profiler import RepoProfile
from reporadar.store import PaperStore


def compute_keyword_frequencies(
    papers: list[dict[str, Any]],
    profile: RepoProfile,
) -> dict[str, int]:
    """Count how many papers match each profile keyword.

    For each keyword in the profile, counts papers whose title+abstract
    contains at least one token from that keyword.
    """
    import re

    def _tokenize(text: str) -> set[str]:
        return set(re.findall(r"[a-z][a-z0-9_-]+", text.lower()))

    frequencies: dict[str, int] = {}
    for keyword, _weight in profile.keywords:
        keyword_tokens = _tokenize(keyword)
        if not keyword_tokens:
            continue
        count = 0
        for paper in papers:
            paper_tokens = _tokenize(
                paper.get("title", "") + " " + paper.get("abstract", "")
            )
            if keyword_tokens & paper_tokens:
                count += 1
        frequencies[keyword] = count
    return frequencies


def detect_trends(
    store: PaperStore,
    run_id: int,
    lookback_runs: int = 5,
) -> list[dict[str, Any]]:
    """Compare current keyword frequencies to recent averages.

    Returns keywords with >50% frequency increase, sorted by increase
    magnitude. Each entry: {"keyword": str, "current": int, "average": float,
    "increase_pct": float}.
    """
    current_freqs = store.get_keyword_frequencies(run_id)
    if not current_freqs:
        return []

    # Gather historical frequencies from previous runs
    runs = store.get_runs(limit=lookback_runs + 1)
    # Exclude current run and keep up to lookback_runs previous
    previous_run_ids = [r["run_id"] for r in runs if r["run_id"] != run_id][:lookback_runs]

    if not previous_run_ids:
        return []

    # Compute average frequency per keyword across previous runs
    historical: dict[str, list[int]] = {}
    for prev_id in previous_run_ids:
        prev_freqs = store.get_keyword_frequencies(prev_id)
        for keyword, freq in prev_freqs.items():
            historical.setdefault(keyword, []).append(freq)

    trends: list[dict[str, Any]] = []
    for keyword, current in current_freqs.items():
        prev_values = historical.get(keyword, [])
        if not prev_values:
            continue
        avg = sum(prev_values) / len(prev_values)
        if avg == 0:
            if current > 0:
                trends.append({
                    "keyword": keyword,
                    "current": current,
                    "average": 0.0,
                    "increase_pct": float("inf"),
                })
            continue
        increase_pct = (current - avg) / avg * 100
        if increase_pct > 50:
            trends.append({
                "keyword": keyword,
                "current": current,
                "average": round(avg, 1),
                "increase_pct": round(increase_pct, 1),
            })

    trends.sort(key=lambda t: t["increase_pct"], reverse=True)
    return trends
