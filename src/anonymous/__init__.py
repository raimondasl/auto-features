"""Anonymous: a paper recommender for code repositories that can decline to recommend."""

from importlib import metadata as _metadata

# Read from the installed distribution rather than written here, so `pyproject.toml` is the
# only place the number is written.
#
# The distribution is `anonymous-papers`, not the import name.
# `anonymous.delegate.installed_version()` performs the same lookup but answers None on failure,
# because pinning a child process to an invented version is worse than not delegating at all; a
# User-Agent only needs to be honest, so this falls back to a version that says it is unknown.
try:
    __version__ = _metadata.version("anonymous-papers")
except _metadata.PackageNotFoundError:  # pragma: no cover - a source tree never installed
    __version__ = "0.0.0+unknown"
