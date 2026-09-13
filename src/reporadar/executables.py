"""Find a program on PATH without ever finding one the current repository put there.

``shutil.which`` on Windows searches the current directory *before* PATH unless
``NoDefaultCurrentDirectoryInExePath`` is set, which it normally is not, and returns what it found
there as a relative path (``.\\az.cmd.COM``). ``CreateProcess`` resolves a relative program path
against the parent's working directory whatever ``cwd=`` the child is given, so the repository's
file runs. RepoRadar's working directory is the repository being profiled -- ``rr update`` in a
clone, the delegated ``uvx`` child, an editor-launched server -- and the repository can be one the
user did not write, with a committed ``.reporadar.yml`` choosing which program gets looked up. So
this walks PATH itself: absolute directories only, never ``os.curdir``, and it returns an absolute
path or nothing.
"""

from __future__ import annotations

import os
import sys


def find_on_path(name: str) -> str | None:
    """The absolute path of *name* in the first PATH directory holding it, or None.

    On Windows a *name* without one of PATHEXT's extensions is tried with each of them, in
    PATHEXT's order, the way the shell does; a *name* that already carries one is tried as it is.
    """
    if not name or os.path.dirname(name):
        return None  # only bare names: a path is either trusted by the caller or not found here
    candidates = [name]
    if sys.platform == "win32":
        exts = [e for e in os.environ.get("PATHEXT", ".COM;.EXE;.BAT;.CMD").split(os.pathsep) if e]
        if not any(name.lower().endswith(e.lower()) for e in exts):
            candidates = [name + e for e in exts]
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        directory = directory.strip().strip('"')
        # A relative entry ("", ".", "bin") is a lookup in the working directory under another name.
        if not directory or not os.path.isabs(directory):
            continue
        for candidate in candidates:
            path = os.path.join(directory, candidate)
            if os.path.isfile(path) and (sys.platform == "win32" or os.access(path, os.X_OK)):
                return os.path.abspath(path)
    return None
