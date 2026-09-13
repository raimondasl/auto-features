"""Tests for reporadar.executables — PATH lookups that cannot land in the repository.

RepoRadar's working directory is the repository it profiles, which may be one the user did not
write. `shutil.which` on Windows looks there before PATH, so every program RepoRadar starts by
name -- `az`, `uvx`, the `rr` a schedule runs -- is found with this instead.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from reporadar import delegate, scheduler
from reporadar.executables import find_on_path


def _plant(directory: Path, name: str) -> Path:
    path = directory / name
    path.write_text("", encoding="utf-8")
    path.chmod(0o755)
    return path


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A working directory holding every spelling of the programs RepoRadar starts."""
    here = tmp_path / "repo"
    here.mkdir()
    for stem in ("uvx", "rr", "az"):
        for ext in ("", ".exe", ".com", ".bat", ".cmd"):
            _plant(here, stem + ext)
    monkeypatch.chdir(here)
    monkeypatch.delenv("NoDefaultCurrentDirectoryInExePath", raising=False)
    return here


class TestTheWorkingDirectoryIsNeverSearched:
    def test_not_implicitly(self, repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PATH", str(repo.parent / "empty"))
        assert find_on_path("uvx") is None

    @pytest.mark.parametrize("entry", ["", ".", "bin", "./"])
    def test_not_through_a_relative_path_entry(
        self, repo: Path, monkeypatch: pytest.MonkeyPatch, entry: str
    ) -> None:
        (repo / "bin").mkdir(exist_ok=True)
        _plant(repo / "bin", "uvx.exe" if sys.platform == "win32" else "uvx")
        monkeypatch.setenv("PATH", entry)
        assert find_on_path("uvx") is None

    def test_the_delegated_collection_and_the_schedule_use_it(
        self, repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PATH", ".")
        assert delegate.uvx_executable() is None
        assert scheduler._build_command("c.yml").startswith("rr update")


class TestWhatIsFound:
    def test_an_absolute_path_from_the_first_directory_that_has_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        first, second = tmp_path / "first", tmp_path / "second"
        first.mkdir()
        second.mkdir()
        name = "tool.exe" if sys.platform == "win32" else "tool"
        _plant(second, name)
        expected = _plant(first, name)
        monkeypatch.setenv("PATH", os.pathsep.join([str(first), str(second)]))
        found = find_on_path("tool")
        assert found is not None and os.path.isabs(found)
        assert os.path.samefile(found, expected)

    @pytest.mark.skipif(sys.platform != "win32", reason="PATHEXT is a Windows lookup rule")
    def test_pathext_order_decides_between_spellings(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _plant(tmp_path, "tool.cmd")
        exe = _plant(tmp_path, "tool.exe")
        monkeypatch.setenv("PATH", str(tmp_path))
        monkeypatch.setenv("PATHEXT", ".EXE;.CMD")
        found = find_on_path("tool")
        assert found is not None and os.path.samefile(found, exe)
        cmd = find_on_path("tool.cmd")
        assert cmd is not None and cmd.lower().endswith("tool.cmd")

    def test_a_name_with_a_directory_is_not_looked_up(self) -> None:
        assert find_on_path(os.path.join(".", "uvx")) is None
