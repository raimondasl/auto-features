"""RepoRadar — research papers your repo should act on, not just ones about its topic."""

from importlib import metadata as _metadata

# Read from the installed distribution rather than written here. This was a literal nobody
# bumped at release time, so from 1.0.1 through 1.0.6 RepoRadar introduced itself to arXiv, IACR
# and the HyDE index host as 1.0.0. `pyproject.toml` is now the only place the number is written.
#
# The distribution is `reporadar-papers`, not the import name: PyPI refuses `reporadar`.
# `reporadar.delegate.installed_version()` performs the same lookup but answers None on failure,
# because pinning a child process to an invented version is worse than not delegating at all; a
# User-Agent only needs to be honest, so this falls back to a version that says it is unknown.
try:
    __version__ = _metadata.version("reporadar-papers")
except _metadata.PackageNotFoundError:  # pragma: no cover - a source tree never installed
    __version__ = "0.0.0+unknown"
