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

    Returns a dict with: ``success``, ``run_id``, ``papers_new``, ``top_picks_count``,
    ``digest_path``, ``skipped_stages``.

    This runs THE pipeline -- the same `pipeline.run_pipeline` that `rr update` calls, so
    a watch cycle produces the configuration the config file describes. Until 2026-08-16
    it ran a shorter copy: no gate, no rescore, no HyDE, no fusion, no embeddings, arXiv
    only. A user whose config said `triage.enabled: true` got an ungated digest -- the
    -8.12 configuration under the name of the +5.72 one.
    """
    from reporadar.collector import CollectionError
    from reporadar.config import load_config, validate_config
    from reporadar.digest import categorize_papers, write_digest
    from reporadar.pipeline import LogReporter, open_store, run_pipeline
    from reporadar.profiler import cited_arxiv_ids_of
    from reporadar.stages import WATCH, unrun_stages
    from reporadar.store import StoreError

    try:
        cfg = load_config(config_path)
        for w in validate_config(cfg):
            logger.warning("Config warning: %s", w)
    except FileNotFoundError:
        return {"success": False, "error": "Config not found"}

    # Kept after the unification, not as a leftover. `unrun_stages` should now return []
    # for a watch cycle, and `tests/test_stages.py` proves the table matches the code --
    # but if a future stage lands in `rr update` and not here, this is what says so out
    # loud instead of letting the gap reopen in silence.
    #
    # Advisory: it must never take the cycle down. A disclosure that FAILED is not a
    # disclosure that found nothing, so the failure is loud and `skipped_stages` is None
    # rather than [] (void, not null).
    skipped: list[str] | None
    try:
        skipped = [s.key for s in unrun_stages(cfg, WATCH)]
    except Exception as exc:  # noqa: BLE001 -- a bad config value must not stop the watch
        logger.warning(
            "could not determine which stages this cycle skips (%s) -- treat the pipeline "
            "as reduced; `rr update` is the one every published number describes",
            exc,
        )
        skipped = None
    if skipped:
        logger.warning(
            "reduced pipeline: %d configured stage(s) will NOT run (%s). "
            "Published numbers describe `rr update`.",
            len(skipped),
            ", ".join(skipped),
        )

    repo_path = Path(cfg.repo_path).resolve()
    db_path = repo_path / ".reporadar" / "papers.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        result = run_pipeline(
            cfg,
            repo_path=repo_path,
            db_path=db_path,
            report=LogReporter(logger),
        )
    except (CollectionError, StoreError) as exc:
        return {"success": False, "error": str(exc)}

    if result.stopped is not None or result.run_id is None:
        # `stopped` and `run_id is None` are the same condition, but only the second
        # convinces the type checker -- and asserting instead would turn an ordinary
        # "nothing to do this cycle" into a crash in an unattended loop.
        return {
            "success": True,
            "run_id": None,
            "papers_new": 0,
            "top_picks_count": 0,
            "skipped_stages": skipped,
        }
    run_id = result.run_id

    with open_store(db_path, LogReporter(logger)) as store:
        # Re-read through the store so the withdrawal flag and the gate's llm_score are
        # joined in, exactly as every other consumer sees it.
        top_picks, _, _ = categorize_papers(
            store.get_scores_for_run(run_id),
            top_n=cfg.output.top_n,
            cited_ids=cited_arxiv_ids_of(Path(cfg.repo_path).resolve()),
        )
        out, summary = write_digest(
            store, result.run_id, cfg.output.digest_path, top_n=cfg.output.top_n
        )

    cycle_result = {
        "success": True,
        "run_id": run_id,
        "papers_new": result.new_count,
        "top_picks_count": len(top_picks),
        "digest_path": str(out),
        "skipped_stages": skipped,
    }

    if on_new_papers and result.new_count > 0:
        on_new_papers(cycle_result)

    return cycle_result


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
