"""Semantic output helpers with colored CLI output."""

from __future__ import annotations

import logging

import click


def success(msg: str) -> None:
    """Print a success message in green."""
    click.echo(click.style(msg, fg="green"))


def warn(msg: str) -> None:
    """Print a warning message in yellow."""
    click.echo(click.style(msg, fg="yellow"))


def error(msg: str) -> None:
    """Print an error message in red to stderr."""
    click.echo(click.style(msg, fg="red"), err=True)


def info(msg: str) -> None:
    """Print an informational message (plain)."""
    click.echo(msg)


def muted(msg: str) -> None:
    """Print a muted/dim message."""
    click.echo(click.style(msg, dim=True))


def setup_verbose_logging() -> None:
    """Configure logging for verbose output."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
