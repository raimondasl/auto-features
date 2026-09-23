"""Properties of what `uv build` produces, checked on the source it builds from.

Both were found by an independent check of the built artifacts, and neither shows up in a
run: a wheel built on Windows differed byte-for-byte from one built on Linux, and every type
annotation the package ships was being discarded by downstream type checkers.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "anonymous"


def test_the_package_declares_that_it_is_typed() -> None:
    """PEP 561: without this file, a type checker treats an installed `anonymous` as
    untyped and silently ignores the annotations, however strictly they were checked here."""
    assert (PACKAGE / "py.typed").is_file()


def test_py_typed_reaches_the_wheel() -> None:
    """Hatchling ships everything under the package directory, so this holds as long as the
    wheel target stays `packages = ["src/anonymous"]` with no include list that would drop
    a non-.py file."""
    lines = (ROOT / "pyproject.toml").read_text(encoding="utf-8").splitlines()
    # By whole line: a comment sixty lines earlier names this table too.
    start = lines.index("[tool.hatch.build.targets.wheel]") + 1
    settings = []
    for line in lines[start:]:
        if line.startswith("["):
            break
        if line.strip() and not line.lstrip().startswith("#"):
            settings.append(line.strip())
    assert settings == ['packages = ["src/anonymous"]'], (
        f"the wheel target is now {settings}; make sure py.typed is still shipped"
    )


def test_text_files_are_lf_everywhere_so_a_build_is_reproducible() -> None:
    """`.gitattributes` settles the WORKING TREE, which is what the build reads. Without
    `eol=lf`, a Windows checkout gives every file CRLF and the wheel built there differs
    from the Linux one for no reason anybody can see."""
    assert "* text=auto eol=lf" in (ROOT / ".gitattributes").read_text(encoding="utf-8")


@pytest.mark.parametrize("suffix", [".py", ".j2"])
def test_no_source_file_carries_crlf(suffix: str) -> None:
    crlf = [
        str(path.relative_to(ROOT))
        for path in PACKAGE.rglob(f"*{suffix}")
        if "__pycache__" not in path.parts and b"\r\n" in path.read_bytes()
    ]
    assert not crlf, f"CRLF in {crlf[:5]}; a wheel built here would not match a Linux build"
