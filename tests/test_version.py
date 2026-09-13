"""`reporadar.__version__` is the version pyproject declares, and nothing else writes one.

It used to be a literal in `__init__.py` that no release ever bumped: from 1.0.1 through 1.0.6,
RepoRadar introduced itself to arXiv, IACR and the HyDE index host as 1.0.0, and two other
adapters hardcoded `RepoRadar/1.0` on top of that. Now the number is written once, in
`pyproject.toml`, and read from installed metadata everywhere else.
"""

from __future__ import annotations

import re
import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pytest

import reporadar

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "reporadar"


def _declared() -> str:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "version"
    ]


def test_the_version_is_the_one_pyproject_declares() -> None:
    try:
        version("reporadar-papers")
    except PackageNotFoundError:
        pytest.skip("reporadar-papers is not installed, so there is no metadata to read")

    assert reporadar.__version__ == _declared(), (
        f"reporadar reports {reporadar.__version__} but pyproject.toml declares {_declared()}. "
        "The installed metadata is stale -- reinstall (`uv sync`, or `uv pip install -e .`). "
        "It matters beyond this test: the MCP server pins its dense-discovery child to this "
        "version, and a stale one spawns a release that does not match the code."
    )


def test_no_module_hardcodes_a_version_in_a_user_agent() -> None:
    """How two adapters came to say `RepoRadar/1.0` while the package said 1.0.0 and the
    release was 1.0.6: each wrote its own number. Build it from `__version__` instead."""
    offenders = [
        f"{path.relative_to(SRC)}:{number}"
        for path in sorted(SRC.rglob("*.py"))
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if re.search(r"RepoRadar/\d", line)
    ]
    assert not offenders, f"hardcoded version in a User-Agent: {offenders}"


def test_the_module_level_user_agents_carry_the_real_version() -> None:
    from reporadar import arxiv_rate, hyde
    from reporadar.sources import iacr

    for module in (arxiv_rate, hyde, iacr):
        assert f"RepoRadar/{reporadar.__version__} " in module.USER_AGENT, module.__name__
