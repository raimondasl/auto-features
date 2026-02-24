"""Platform-specific scheduling via crontab (Unix) and schtasks (Windows)."""

from __future__ import annotations

import logging
import platform
import shutil
import subprocess
from dataclasses import dataclass

logger = logging.getLogger(__name__)

TASK_NAME = "RepoRadar"
CRON_MARKER = "# RepoRadar scheduled run"


@dataclass
class ScheduledTask:
    cron_expr: str
    command: str
    platform: str


def _detect_platform() -> str:
    """Return 'windows' or 'unix'."""
    return "windows" if platform.system() == "Windows" else "unix"


def _build_command(config_path: str) -> str:
    """Construct the command string for a scheduled run."""
    rr = shutil.which("rr") or "rr"
    return f"{rr} update --config {config_path} && {rr} digest --config {config_path}"


# ── Unix (crontab) ──────────────────────────────────────────────────


def _get_current_crontab() -> str:
    """Read the current user crontab."""
    result = subprocess.run(
        ["crontab", "-l"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return ""
    return result.stdout


def _set_crontab(content: str) -> bool:
    """Write content as the user crontab."""
    result = subprocess.run(
        ["crontab", "-"],
        input=content,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def add_cron_job(cron_expr: str, config_path: str) -> bool:
    """Add or replace a RepoRadar cron job. Returns True on success."""
    command = _build_command(config_path)
    crontab = _get_current_crontab()

    # Remove existing RepoRadar lines
    lines = [line for line in crontab.splitlines() if CRON_MARKER not in line]

    # Add new entry
    lines.append(f"{cron_expr} {command} {CRON_MARKER}")
    new_content = "\n".join(lines) + "\n"

    return _set_crontab(new_content)


def list_cron_jobs() -> list[ScheduledTask]:
    """Parse RepoRadar entries from the current crontab."""
    crontab = _get_current_crontab()
    tasks = []
    for line in crontab.splitlines():
        if CRON_MARKER in line:
            # Strip the marker comment
            parts = line.replace(CRON_MARKER, "").strip()
            # First 5 fields are cron expression
            fields = parts.split()
            if len(fields) >= 6:
                cron_expr = " ".join(fields[:5])
                command = " ".join(fields[5:])
                tasks.append(ScheduledTask(cron_expr=cron_expr, command=command, platform="unix"))
    return tasks


def remove_cron_job() -> bool:
    """Remove all RepoRadar lines from crontab. Returns True on success."""
    crontab = _get_current_crontab()
    lines = [line for line in crontab.splitlines() if CRON_MARKER not in line]
    new_content = "\n".join(lines) + "\n" if lines else ""
    return _set_crontab(new_content)


# ── Windows (schtasks) ──────────────────────────────────────────────


def _cron_to_schtasks_args(cron_expr: str) -> list[str]:
    """Convert a subset of cron expressions to schtasks /SC arguments.

    Supports:
    - ``0 * * * *`` → HOURLY
    - ``0 9 * * *`` → DAILY at 09:00
    - ``0 9 * * 1`` → WEEKLY on Monday at 09:00

    Raises ValueError for unsupported patterns.
    """
    fields = cron_expr.strip().split()
    if len(fields) != 5:
        raise ValueError(f"Expected 5 cron fields, got {len(fields)}: {cron_expr!r}")

    minute, hour, dom, month, dow = fields

    # Hourly: "0 * * * *" or "N * * * *"
    if hour == "*" and dom == "*" and month == "*" and dow == "*":
        return ["/SC", "HOURLY"]

    # Daily: "M H * * *"
    if dom == "*" and month == "*" and dow == "*":
        try:
            h = int(hour)
            m = int(minute)
        except ValueError as exc:
            raise ValueError(f"Unsupported cron expression: {cron_expr!r}") from exc
        return ["/SC", "DAILY", "/ST", f"{h:02d}:{m:02d}"]

    # Weekly: "M H * * N"
    if dom == "*" and month == "*" and dow != "*":
        try:
            h = int(hour)
            m = int(minute)
            d = int(dow)
        except ValueError as exc:
            raise ValueError(f"Unsupported cron expression: {cron_expr!r}") from exc
        day_names = {0: "SUN", 1: "MON", 2: "TUE", 3: "WED", 4: "THU", 5: "FRI", 6: "SAT"}
        day_name = day_names.get(d)
        if day_name is None:
            raise ValueError(f"Invalid day of week: {d}")
        return ["/SC", "WEEKLY", "/D", day_name, "/ST", f"{h:02d}:{m:02d}"]

    raise ValueError(f"Unsupported cron expression: {cron_expr!r}")


def add_schtask(cron_expr: str, config_path: str) -> bool:
    """Register a Windows scheduled task. Returns True on success."""
    command = _build_command(config_path)
    try:
        sc_args = _cron_to_schtasks_args(cron_expr)
    except ValueError as exc:
        logger.warning("Cannot schedule: %s", exc)
        return False

    # Remove existing task first (ignore errors)
    subprocess.run(
        ["schtasks", "/Delete", "/TN", TASK_NAME, "/F"],
        capture_output=True,
    )

    result = subprocess.run(
        ["schtasks", "/Create", "/TN", TASK_NAME, "/TR", command, *sc_args, "/F"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.warning("schtasks /Create failed: %s", result.stderr)
        return False
    return True


def list_schtasks() -> list[ScheduledTask]:
    """List RepoRadar scheduled tasks on Windows."""
    result = subprocess.run(
        ["schtasks", "/Query", "/TN", TASK_NAME, "/FO", "LIST", "/V"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return []

    # Parse basic info from output
    command = ""
    schedule = ""
    for line in result.stdout.splitlines():
        if "Task To Run:" in line:
            command = line.split(":", 1)[1].strip()
        elif "Schedule Type:" in line:
            schedule = line.split(":", 1)[1].strip()

    if command:
        return [ScheduledTask(cron_expr=schedule, command=command, platform="windows")]
    return []


def remove_schtask() -> bool:
    """Remove the RepoRadar Windows scheduled task. Returns True on success."""
    result = subprocess.run(
        ["schtasks", "/Delete", "/TN", TASK_NAME, "/F"],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


# ── Public interface ─────────────────────────────────────────────────


def add_schedule(cron_expr: str, config_path: str) -> bool:
    """Add a schedule, auto-detecting the platform."""
    if _detect_platform() == "windows":
        return add_schtask(cron_expr, config_path)
    return add_cron_job(cron_expr, config_path)


def list_schedules() -> list[ScheduledTask]:
    """List registered schedules for the current platform."""
    if _detect_platform() == "windows":
        return list_schtasks()
    return list_cron_jobs()


def remove_schedule() -> bool:
    """Remove the schedule for the current platform."""
    if _detect_platform() == "windows":
        return remove_schtask()
    return remove_cron_job()
