"""Watch mode — continuous monitoring with desktop notifications."""

from __future__ import annotations

import logging
import platform
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def parse_interval(interval_str: str) -> int:
    """Parse a human-friendly interval string into seconds.

    Supported formats: ``"6h"`` (hours), ``"30m"`` (minutes),
    ``"1d"`` (days), ``"3600s"`` (seconds).

    Raises ValueError for invalid formats.
    """
    s = interval_str.strip().lower()
    if not s:
        raise ValueError("Empty interval string")

    suffix = s[-1]
    try:
        value = int(s[:-1])
    except ValueError as exc:
        raise ValueError(f"Invalid interval: {interval_str!r}") from exc

    if value <= 0:
        raise ValueError(f"Interval must be positive: {interval_str!r}")

    if suffix == "h":
        return value * 3600
    elif suffix == "m":
        return value * 60
    elif suffix == "d":
        return value * 86400
    elif suffix == "s":
        return value
    else:
        raise ValueError(f"Invalid interval suffix: {suffix!r}. Use h, m, d, or s.")


def send_desktop_notification(title: str, message: str) -> bool:
    """Send a desktop notification. Returns True on success."""
    system = platform.system()
    try:
        if system == "Linux":
            result = subprocess.run(
                ["notify-send", title, message],
                capture_output=True,
                timeout=10,
            )
            return result.returncode == 0
        elif system == "Darwin":
            script = f'display notification "{message}" with title "{title}"'
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                timeout=10,
            )
            return result.returncode == 0
        elif system == "Windows":
            ps_script = (
                "[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, "
                "ContentType = WindowsRuntime] | Out-Null; "
                "$template = [Windows.UI.Notifications.ToastNotificationManager]::"
                "GetTemplateContent([Windows.UI.Notifications.ToastTemplateType]::ToastText02); "
                "$textNodes = $template.GetElementsByTagName('text'); "
                f"$textNodes.Item(0).AppendChild($template.CreateTextNode('{title}')); "
                f"$textNodes.Item(1).AppendChild($template.CreateTextNode('{message}')); "
                "$toast = [Windows.UI.Notifications.ToastNotification]::new($template); "
                "[Windows.UI.Notifications.ToastNotificationManager]::"
                "CreateToastNotifier('RepoRadar').Show($toast)"
            )
            result = subprocess.run(
                ["powershell", "-Command", ps_script],
                capture_output=True,
                timeout=10,
            )
            return result.returncode == 0
        else:
            logger.warning("Desktop notifications not supported on %s", system)
            return False
    except FileNotFoundError:
        logger.warning("Notification command not found on %s", system)
        return False
    except subprocess.TimeoutExpired:
        logger.warning("Notification command timed out")
        return False
    except Exception as exc:
        logger.warning("Desktop notification failed: %s", exc)
        return False


def run_update_cycle(
    config_path: str,
    on_new_papers: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run a single update+digest cycle.

    Returns a dict with: ``success``, ``run_id``, ``papers_new``,
    ``top_picks_count``, ``digest_path``.
    """
    from reporadar.collector import CollectionError, build_queries, collect_papers
    from reporadar.config import load_config, validate_config
    from reporadar.digest import categorize_papers, write_digest
    from reporadar.profiler import profile_repo
    from reporadar.ranker import rank_papers
    from reporadar.store import PaperStore

    try:
        cfg = load_config(config_path)
        for w in validate_config(cfg):
            logger.warning("Config warning: %s", w)
    except FileNotFoundError:
        return {"success": False, "error": "Config not found"}

    repo_path = Path(cfg.repo_path).resolve()
    db_path = repo_path / ".reporadar" / "papers.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Profile + collect
    repo_profile = profile_repo(repo_path)
    queries = build_queries(repo_profile, cfg.queries, cfg.arxiv)
    if not queries:
        return {"success": True, "run_id": None, "papers_new": 0, "top_picks_count": 0}

    try:
        papers = collect_papers(queries, cfg.arxiv)
    except CollectionError as exc:
        return {"success": False, "error": str(exc)}

    if not papers:
        return {"success": True, "run_id": None, "papers_new": 0, "top_picks_count": 0}

    with PaperStore(db_path) as store:
        new_count, seen_count = store.upsert_papers(papers)
        run_id = store.record_run(queries, new_count, seen_count)

        scores = rank_papers(
            papers,
            repo_profile,
            cfg.ranking,
            cfg.queries,
            cfg.arxiv.categories,
            cfg.arxiv.lookback_days,
        )
        store.save_scores(run_id, scores)

        top_picks, _, _ = categorize_papers(scores, top_n=cfg.output.top_n)

        out, summary = write_digest(store, run_id, cfg.output.digest_path, top_n=cfg.output.top_n)

    result = {
        "success": True,
        "run_id": run_id,
        "papers_new": new_count,
        "top_picks_count": len(top_picks),
        "digest_path": str(out),
    }

    if on_new_papers and new_count > 0:
        on_new_papers(result)

    return result


def watch_loop(
    config_path: str,
    interval_seconds: int,
    notify: bool = True,
    max_cycles: int | None = None,
) -> None:
    """Run update cycles in a blocking loop.

    *max_cycles* limits the number of iterations (for testing).
    """
    cycle = 0
    while max_cycles is None or cycle < max_cycles:
        cycle += 1
        logger.info("Watch cycle %d starting...", cycle)

        result = run_update_cycle(config_path)

        if result["success"]:
            new = result.get("papers_new", 0)
            top = result.get("top_picks_count", 0)
            if new > 0 and notify:
                send_desktop_notification(
                    "RepoRadar",
                    f"{new} new papers found, {top} top picks",
                )
            logger.info(
                "Cycle %d complete: %d new papers, %d top picks",
                cycle,
                new,
                top,
            )
        else:
            logger.warning("Cycle %d failed: %s", cycle, result.get("error", "unknown"))

        if max_cycles is None or cycle < max_cycles:
            time.sleep(interval_seconds)
