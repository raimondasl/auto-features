"""§2.2's remaining eligibility rules: language, software-project, source floor, README prose.

§2.2 admits repositories "with ... a primary language in the committed set" and subject to
"software-project and README-quality rules as committed". Those rules *are* committed — in
`PREREG-benchmark-expansion.md` §1(v) and §3, rules X4, X5 and X7 — and none of them was
implemented. Until now the walk asked only PP1 (history) and PP2 (identifiers at T0), so
nothing stopped a curated paper list, the artefact §2.2 names explicitly, from clearing
`ids_v2(T0) ≥ 3` on a bibliography of two hundred identifiers and dominating the positives.

The four rules, and what each is for:

* **Language** (§1(v)) — a primary language in a stated set. 1,408 of the 17,888 candidate
  rows (7.9 %) report no primary language at all. Checked from the candidate CSV, so a
  failing row costs no clone.
* **X4, software project** — the committed regex over `topics ∪ name ∪ README[:300]`. The
  frame's version also reads `description`; this one cannot, because §2.1 forbids the
  description from entering the tree (it carries repository URLs). Recorded as a narrowing.
* **X7, source floor** — ≥ 20 files carrying a source extension of the reported primary
  language at HEAD. A docs-only or data-only repository fails here.
* **X5, README prose in English** — fastText `lid.176` at p(en) ≥ 0.8, applied only when
  ≥ 300 characters of prose remain after code blocks, badges and URLs are stripped; shorter
  READMEs pass flagged `lid_na`, so the rule cannot cull thin repositories.

**X5 is scaffolded but NOT APPLIED, and that is a recorded deviation.** `lid.176` is a
126 MB model behind a package that is not a dependency of this project, and inventing a
substitute detector would be a different rule wearing X5's name — the failure mode being
that it culls a different set and nobody can tell. The stripping, the 300-character rule and
the flag are implemented exactly as registered; the detector is injectable and defaults to
absent, in which case every candidate passes flagged `lid_na_no_detector`. Supply a detector
and the rule applies as written.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

# §1(v) of PREREG-benchmark-expansion.md, verbatim.
LANGUAGES = frozenset(
    {"Python", "C", "C++", "Rust", "Go", "Julia", "JavaScript", "TypeScript", "R", "Fortran"}
)

# X7's "a source extension of the reported primary language".
SOURCE_EXTENSIONS: dict[str, frozenset[str]] = {
    "Python": frozenset({".py", ".pyx", ".pyi"}),
    "C": frozenset({".c", ".h"}),
    "C++": frozenset({".cpp", ".cc", ".cxx", ".hpp", ".hh", ".hxx", ".h"}),
    "Rust": frozenset({".rs"}),
    "Go": frozenset({".go"}),
    "Julia": frozenset({".jl"}),
    "JavaScript": frozenset({".js", ".mjs", ".cjs", ".jsx"}),
    "TypeScript": frozenset({".ts", ".tsx", ".mts", ".cts"}),
    "R": frozenset({".r", ".rmd"}),
    "Fortran": frozenset({".f", ".f90", ".f95", ".f03", ".for", ".ftn"}),
}
MIN_SOURCE_FILES = 20  # X7

# X4, the committed regex. Transcribed from the frame's table, where `\|` is a markdown
# escape rather than part of the pattern.
NOT_SOFTWARE = re.compile(
    r"(awesome|curated[- ]list|paper[- ]list|reading[- ]list|tutorial|course|homework"
    r"|lecture|book|cheat[- ]?sheet|interview|roadmap|template|boilerplate|starter"
    r"|dotfiles|dataset[- ]only|official (implementation|code)|code for (the|our) paper"
    r"|implementation of (the|our) paper)",
    re.I,
)
X4_README_CHARS = 300

X5_MIN_PROSE_CHARS = 300
X5_MIN_P_EN = 0.8

README_NAMES = ("README.md", "README.rst", "README.txt", "README", "readme.md", "Readme.md")

_FENCED = re.compile(r"```.*?```|~~~.*?~~~", re.S)
_INDENTED = re.compile(r"^(?: {4}|\t).*$", re.M)
_BADGE = re.compile(r"!?\[[^\]]*\]\([^)]*\)")
_URL = re.compile(r"https?://\S+")
_HTML = re.compile(r"<[^>]+>")


def language_ok(language: str) -> bool:
    """§1(v). A blank primary language fails: 7.9 % of the candidate rows report none, and
    "unknown" is not one of the committed languages."""
    return (language or "").strip() in LANGUAGES


def is_software_project(topics: str, full_name: str, readme: str) -> bool:
    """X4, over `topics ∪ name ∪ README[:300]`.

    The frame's X4 also reads the repository description. This cannot: §2.1 forbids the
    description from entering the tree because it routinely carries a repository URL, so the
    enumeration never recorded it. A narrowing of the rule's input, recorded rather than
    silently absorbed — it can only let *more* repositories through, never fewer.
    """
    haystack = " ".join(
        [(topics or "").replace("|", " "), full_name or "", (readme or "")[:X4_README_CHARS]]
    )
    return not NOT_SOFTWARE.search(haystack)


def readme_prose(text: str) -> str:
    """X5's preparation: strip code blocks, badges, links and URLs, leaving prose."""
    out = _FENCED.sub(" ", text or "")
    out = _INDENTED.sub(" ", out)
    out = _BADGE.sub(" ", out)
    out = _URL.sub(" ", out)
    out = _HTML.sub(" ", out)
    return " ".join(out.split())


def english_readme(
    text: str, detector: Callable[[str], float] | None = None
) -> tuple[bool, str, int]:
    """X5. Returns (passes, flag, prose characters).

    Registered behaviour, implemented exactly: strip, and apply the test **only** when ≥ 300
    characters of prose remain, so the rule cannot cull a thin repository. Shorter READMEs
    pass flagged `lid_na`.

    With no detector supplied every candidate passes flagged `lid_na_no_detector`. That is a
    recorded deviation, not a silent pass: see the module docstring on why a substitute
    detector would be a different rule wearing X5's name.
    """
    prose = readme_prose(text)
    if len(prose) < X5_MIN_PROSE_CHARS:
        return True, "lid_na", len(prose)
    if detector is None:
        return True, "lid_na_no_detector", len(prose)
    return detector(prose) >= X5_MIN_P_EN, "lid", len(prose)


def read_readme(repo: Path, rev: str, git: Any, timeout: float | None = None) -> str:
    """The README at *rev*, without checking anything out. Empty when there is none."""
    listing = git(repo, "ls-tree", "--name-only", rev, check=False, timeout=timeout).splitlines()
    present = {name.strip() for name in listing}
    for name in README_NAMES:
        if name in present:
            return git(repo, "show", f"{rev}:{name}", check=False, timeout=timeout)
    return ""


def source_file_count(
    repo: Path, rev: str, language: str, git: Any, timeout: float | None = None
) -> int:
    """X7. Files at *rev* carrying a source extension of the reported primary language."""
    extensions = SOURCE_EXTENSIONS.get(language, frozenset())
    if not extensions:
        return 0
    listing = git(repo, "ls-tree", "-r", "--name-only", rev, check=False, timeout=timeout)
    return sum(1 for path in listing.splitlines() if Path(path).suffix.lower() in extensions)
