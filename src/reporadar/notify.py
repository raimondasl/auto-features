"""Notification dispatch — shell hooks, Slack, Discord, and email."""

from __future__ import annotations

import json
import logging
import smtplib
import subprocess
import urllib.request
from dataclasses import dataclass
from email.mime.text import MIMEText
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reporadar.config import HooksConfig

logger = logging.getLogger(__name__)


@dataclass
class DigestSummary:
    """Summary of a digest run for notification payloads."""

    digest_path: str
    run_id: int
    papers_new: int
    papers_seen: int
    top_picks_count: int
    total_scored: int
    fmt: str


def summary_to_env(summary: DigestSummary) -> dict[str, str]:
    """Convert a DigestSummary to ``RR_``-prefixed environment variables."""
    return {
        "RR_DIGEST_PATH": str(summary.digest_path),
        "RR_RUN_ID": str(summary.run_id),
        "RR_PAPERS_NEW": str(summary.papers_new),
        "RR_PAPERS_SEEN": str(summary.papers_seen),
        "RR_TOP_PICKS_COUNT": str(summary.top_picks_count),
        "RR_TOTAL_SCORED": str(summary.total_scored),
        "RR_FORMAT": summary.fmt,
    }


def _format_message(summary: DigestSummary) -> str:
    """Build a human-readable notification message."""
    return (
        f"RepoRadar digest #{summary.run_id}: "
        f"{summary.papers_new} new papers, "
        f"{summary.top_picks_count} top picks "
        f"({summary.total_scored} scored)"
    )


def run_shell_hook(command: str, summary: DigestSummary) -> bool:
    """Run a shell command with ``RR_``-prefixed env vars. Returns True on success."""
    import os

    env = {**os.environ, **summary_to_env(summary)}
    try:
        result = subprocess.run(
            command,
            shell=True,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            logger.warning("Shell hook failed (exit %d): %s", result.returncode, result.stderr)
            return False
        return True
    except subprocess.TimeoutExpired:
        logger.warning("Shell hook timed out after 60s: %s", command)
        return False
    except OSError as exc:
        logger.warning("Shell hook error: %s", exc)
        return False


def send_slack_webhook(url: str, summary: DigestSummary) -> bool:
    """POST a Slack-formatted message to a webhook URL. Returns True on success."""
    payload = json.dumps({"text": _format_message(summary)}).encode()
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return 200 <= resp.status < 300
    except Exception as exc:
        logger.warning("Slack webhook failed: %s", exc)
        return False


def send_discord_webhook(url: str, summary: DigestSummary) -> bool:
    """POST a Discord-formatted message to a webhook URL. Returns True on success."""
    payload = json.dumps({"content": _format_message(summary)}).encode()
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return 200 <= resp.status < 300
    except Exception as exc:
        logger.warning("Discord webhook failed: %s", exc)
        return False


def send_email(
    smtp_host: str,
    port: int,
    from_addr: str,
    to: str,
    summary: DigestSummary,
    username: str = "",
    password: str = "",
    use_tls: bool = True,
) -> bool:
    """Send a digest summary email via SMTP. Returns True on success."""
    msg = MIMEText(_format_message(summary))
    msg["Subject"] = f"RepoRadar Digest #{summary.run_id}"
    msg["From"] = from_addr
    msg["To"] = to

    try:
        with smtplib.SMTP(smtp_host, port, timeout=30) as server:
            if use_tls:
                server.starttls()
            if username and password:
                server.login(username, password)
            server.sendmail(from_addr, [to], msg.as_string())
        return True
    except Exception as exc:
        logger.warning("Email send failed: %s", exc)
        return False


def dispatch_notification(
    channel: str,
    hooks_cfg: HooksConfig,
    summary: DigestSummary,
) -> bool:
    """Dispatch a notification to the specified channel. Returns True on success."""
    if channel == "shell":
        if not hooks_cfg.on_digest:
            logger.warning("No shell hook configured (hooks.on_digest is empty)")
            return False
        return run_shell_hook(hooks_cfg.on_digest, summary)
    elif channel == "slack":
        if not hooks_cfg.slack_webhook_url:
            logger.warning("No Slack webhook URL configured")
            return False
        return send_slack_webhook(hooks_cfg.slack_webhook_url, summary)
    elif channel == "discord":
        if not hooks_cfg.discord_webhook_url:
            logger.warning("No Discord webhook URL configured")
            return False
        return send_discord_webhook(hooks_cfg.discord_webhook_url, summary)
    elif channel == "email":
        email = hooks_cfg.email
        if not email.smtp_host or not email.to:
            logger.warning("Email not configured (missing smtp_host or to)")
            return False
        return send_email(
            smtp_host=email.smtp_host,
            port=email.smtp_port,
            from_addr=email.from_addr,
            to=email.to,
            summary=summary,
            username=email.username,
            password=email.password,
            use_tls=email.use_tls,
        )
    else:
        logger.warning("Unknown notification channel: %s", channel)
        return False
