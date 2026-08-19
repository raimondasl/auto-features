"""Repo topic profiling — extract keywords, anchors, and domains from a repo."""

from __future__ import annotations

import ast
import configparser
import json
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer


@dataclass
class RepoProfile:
    """Result of profiling a repository."""

    keywords: list[tuple[str, float]]  # (term, tfidf_weight) sorted desc
    anchors: list[str]  # library/package names found in manifests
    domains: list[str]  # inferred domain labels
    source_signals: list[str] = field(default_factory=list)  # ML/domain patterns from source
    # The project described in its own words, for prompts that need to know what the repo
    # is FOR rather than what it CONTAINS. Every other field here says what the repo has —
    # its libraries, its frequent terms — which is why a keyword profile of ColBERT reports
    # "web APIs" (it depends on flask) and never says retrieval. Empty when the repo has no
    # README and declares no packaging description. Budgeted at profile time, not at use
    # time, so a 41k-character README cannot reach a prompt by accident.
    prose: str = ""
    # Two-word phrases from the keyword vocabulary that ACTUALLY OCCUR, adjacently, in the
    # profiled text. Query building pairs adjacent keywords into quoted phrases, and nothing
    # ever required the two words to belong together — "use page" and "data cd" were sent to
    # real search APIs as phrase queries. Verifying is only possible here, because the corpus
    # does not survive profiling. Empty when nothing was read.
    corpus_phrases: list[str] = field(default_factory=list)
    # arXiv ids the repository's own text already cites (README, CITATION file, docs/).
    # A paper a project links to is not news to its maintainers -- it is at best something
    # they already adopted, which is exactly the signal `evals/mine_adoptions.py` used as
    # ground truth. The digest routes these out of Top Picks into their own section rather
    # than recommending them back. Version-stripped, so the comparison survives
    # `2303.14046` vs `2303.14046v1`.
    cited_arxiv_ids: frozenset[str] = frozenset()


# Mapping from common package names to domain labels.
PACKAGE_DOMAIN_MAP: dict[str, str] = {
    "torch": "deep learning",
    "pytorch": "deep learning",
    "tensorflow": "deep learning",
    "keras": "deep learning",
    "transformers": "NLP",
    "huggingface": "NLP",
    "spacy": "NLP",
    "nltk": "NLP",
    "langchain": "LLM applications",
    "openai": "LLM applications",
    "anthropic": "LLM applications",
    "fastapi": "web APIs",
    "flask": "web APIs",
    "django": "web development",
    "numpy": "scientific computing",
    "scipy": "scientific computing",
    "pandas": "data analysis",
    "scikit-learn": "machine learning",
    "sklearn": "machine learning",
    "matplotlib": "visualization",
    "plotly": "visualization",
    "opencv": "computer vision",
    "cv2": "computer vision",
    "pillow": "image processing",
    "sqlalchemy": "databases",
    "react": "frontend",
    "vue": "frontend",
    "angular": "frontend",
    "express": "web APIs",
    "next": "frontend",
    "tensorflow-hub": "deep learning",
    "jax": "deep learning",
    "ray": "distributed computing",
    "dask": "distributed computing",
    "airflow": "data pipelines",
    "prefect": "data pipelines",
    "docker": "containers",
    "kubernetes": "containers",
}


def _read_text_file(path: Path) -> str:
    """Read a text file, returning empty string if it doesn't exist."""
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _parse_requirements_txt(path: Path) -> list[str]:
    """Extract package names from requirements.txt."""
    text = _read_text_file(path)
    if not text:
        return []

    packages: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        # Strip version specifiers, extras and environment markers. Routed through
        # `_requirement_name` rather than repeating its character class, because the copy
        # that used to live here omitted `~`: matminer pins `requests~=2.31` and
        # `scikit-learn~=1.3` and yielded the anchors `requests~` and `scikit-learn~`. No
        # package has those names, and `scikit-learn~` misses PACKAGE_DOMAIN_MAP, so a
        # repository that depends on scikit-learn never inferred "machine learning".
        name = _requirement_name(line)
        if name:
            packages.append(name)
    return packages


def _parse_pyproject_toml(path: Path) -> list[str]:
    """Extract dependency names from pyproject.toml."""
    if not path.is_file():
        return []

    with open(path, "rb") as f:
        data = tomllib.load(f)

    packages: list[str] = []

    # [project.dependencies]
    for dep in data.get("project", {}).get("dependencies", []):
        name = _requirement_name(dep)
        if name:
            packages.append(name)

    # [project.optional-dependencies]
    for deps in data.get("project", {}).get("optional-dependencies", {}).values():
        for dep in deps:
            name = _requirement_name(dep)
            if name:
                packages.append(name)

    return packages


def _requirement_name(spec: str) -> str:
    """Reduce a requirement string like ``torch>=2.1`` to its bare package name."""
    return re.split(r"[>=<!~;\[\]]", spec)[0].strip().lower()


# Module-level variables in a setup.py that plausibly hold requirement strings. Real
# packaging files rarely pass a literal to `setup()`; they build a list (or a table of
# pinned versions) above it and reference it, so the literal has to be found by name.
_DEP_VAR_HINTS = ("dep", "require", "install")


def _parse_setup_py(path: Path) -> list[str]:
    """Extract dependency names from setup.py **without executing it**.

    Reading this file matters more than it looks: a HuggingFace-style project (diffusers,
    transformers, peft) ships no requirements.txt and a pyproject.toml holding only tool
    config, so setup.py is the *only* place its dependencies are declared. Skipping it
    left those repos with no anchors at all, no inferred domains, and a keyword list that
    fell through to README boilerplate — `license` and `https` became search queries.

    Parsed with :mod:`ast`, never ``exec``: a setup.py is arbitrary code, and profiling a
    repository must not run it. That rules out resolving computed values, so this reads
    literal lists of strings only, which is what the overwhelming majority of them use.
    """
    text = _read_text_file(path)
    if not text:
        return []
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []  # a setup.py we cannot parse is not worth failing a profile over

    packages: list[str] = []

    def _from_list(node: ast.AST) -> None:
        if not isinstance(node, ast.List):
            return
        for element in node.elts:
            if isinstance(element, ast.Constant) and isinstance(element.value, str):
                name = _requirement_name(element.value)
                if name:
                    packages.append(name)

    for node in ast.walk(tree):
        # `install_requires = [...]`, `_deps = [...]`, `REQUIRED = [...]`
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and any(
                    hint in target.id.lower() for hint in _DEP_VAR_HINTS
                ):
                    _from_list(node.value)
        # `setup(install_requires=[...], extras_require={...: [...]})`
        elif isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg in ("install_requires", "setup_requires"):
                    _from_list(kw.value)
                elif kw.arg == "extras_require" and isinstance(kw.value, ast.Dict):
                    for value in kw.value.values:
                        _from_list(value)

    return packages


def _packaging_metadata_text(repo_path: Path) -> str:
    """The project's own one-line description and keywords, if it declares any.

    This is the most concentrated topic signal a repo has — an author writing
    ``keywords="deep learning diffusion jax pytorch"`` is stating the subject directly,
    where a README says it diffusely between badges and install instructions. It is a
    handful of words against a README's thousands, so it enters TF-IDF as its own
    document rather than being appended to one.
    """
    parts: list[str] = []

    setup_py = repo_path / "setup.py"
    text = _read_text_file(setup_py)
    if text:
        try:
            tree = ast.parse(text)
        except SyntaxError:
            tree = None
        if tree is not None:
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                for kw in node.keywords:
                    if (
                        kw.arg in ("description", "keywords")
                        and isinstance(kw.value, ast.Constant)
                        and isinstance(kw.value.value, str)
                    ):
                        parts.append(kw.value.value)

    pyproject = repo_path / "pyproject.toml"
    if pyproject.is_file():
        try:
            with open(pyproject, "rb") as f:
                data = tomllib.load(f)
        except tomllib.TOMLDecodeError:
            data = {}
        project = data.get("project", {})
        if isinstance(project.get("description"), str):
            parts.append(project["description"])
        keywords = project.get("keywords")
        if isinstance(keywords, list):
            parts.extend(k for k in keywords if isinstance(k, str))

    return " ".join(parts)


def _parse_package_json(path: Path) -> list[str]:
    """Extract dependency names from package.json."""
    if not path.is_file():
        return []

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    packages: list[str] = []
    for key in ("dependencies", "devDependencies"):
        for name in data.get(key, {}):
            # Strip npm scope prefix for keyword purposes
            clean = name.lstrip("@").replace("/", "-").lower()
            packages.append(clean)
    return packages


def _parse_setup_cfg(path: Path) -> list[str]:
    """Extract dependency names from a setuptools ``setup.cfg``.

    The declarative half of setuptools, and the reason a real project can profile with no
    anchors at all. A repository that keeps its metadata here ships a ``pyproject.toml``
    holding only ``[build-system]``, so every other parser in this module finds nothing:
    MACE — a PyTorch interatomic-potential model, and one of the best-known codes in
    computational materials science — reached the gate as "Dependencies/libraries: none /
    Domains: general" while declaring torch, e3nn and matscipy nineteen lines into its
    setup.cfg.

    Only ``[options] install_requires`` is read. ``[options.extras_require]`` was measured
    and rejected: on MACE it adds pytest, pytest-benchmark, pytest-cov, pytest-split,
    pytest-xdist, black, isort, mypy, pre-commit and pylint, and `pytest` then becomes the
    repository's top keyword and an arXiv query — the development toolchain drowning the
    subject, which is the defect this file already fights in `_BOILERPLATE_STOPWORDS`.
    """
    text = _read_text_file(path)
    if not text:
        return []
    # `interpolation=None` because a setup.cfg is setuptools' config file, not
    # configparser's: a literal `%` in a description or a coverage `exclude_lines` block is
    # not a substitution and must not raise.
    parser = configparser.ConfigParser(interpolation=None)
    try:
        parser.read_string(text)
    except configparser.Error:
        return []  # an unparseable setup.cfg is not worth failing a whole profile over

    packages: list[str] = []
    for line in parser.get("options", "install_requires", fallback="").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name = _requirement_name(line)
        if name:
            packages.append(name)
    return packages


# Directories that hold the Python package when the repository root does not. Measured,
# not guessed: MDAnalysis ships its library from `package/` (the root holds CI files and a
# testsuite), and tblite and LAMMPS ship theirs from `python/` beside the C and Fortran
# sources. An allowlist of two, not a search -- see the fallback for why.
_NESTED_PACKAGE_DIRS = ("package", "python")


def _manifest_anchors(root: Path) -> list[str]:
    """Every manifest this module knows, read from one directory."""
    anchors: list[str] = []
    anchors.extend(_parse_requirements_txt(root / "requirements.txt"))
    anchors.extend(_parse_pyproject_toml(root / "pyproject.toml"))
    anchors.extend(_parse_setup_cfg(root / "setup.cfg"))
    anchors.extend(_parse_setup_py(root / "setup.py"))
    anchors.extend(_parse_package_json(root / "package.json"))
    return anchors


def _extract_anchors(repo_path: Path) -> list[str]:
    """Collect library/package names from all supported manifest files."""
    anchors = _manifest_anchors(repo_path)

    # Nested packaging, and ONLY when the root declared nothing. Firing solely on that
    # condition keeps every repository that profiles today byte-identical and confines the
    # change to the ones with the defect.
    #
    # A general sub-directory search was measured and is worse than nothing:
    # `lammps/doc/utils/requirements.txt` is sixteen Sphinx packages and the only
    # requirements.txt LAMMPS has, so a search would hand a molecular-dynamics engine a
    # documentation-toolchain profile, and `deepmd-kit/source/tests/pt/requirements.txt` is
    # a test matrix.
    if not anchors:
        for nested in _NESTED_PACKAGE_DIRS:
            found = _manifest_anchors(repo_path / nested)
            if found:
                anchors.extend(found)
                break
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for a in anchors:
        if a not in seen:
            seen.add(a)
            unique.append(a)
    return unique


def _infer_domains(anchors: list[str]) -> list[str]:
    """Map known package names to domain labels."""
    domains: set[str] = set()
    for pkg in anchors:
        # Normalize: some packages use hyphens
        normalized = pkg.replace("-", "").replace("_", "")
        for pattern, domain in PACKAGE_DOMAIN_MAP.items():
            if pattern.replace("-", "").replace("_", "") == normalized:
                domains.add(domain)
                break
    return sorted(domains)


# Markup that carries no topic signal but plenty of tokens. Badge rows and link targets
# are the worst offenders: a README's first screen is often four shields.io URLs, and
# every one of them contributes `https`, `com`, `svg`, `badge` to the term counts.
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")  # badges
_MD_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")  # keep the text, drop the target
_HTML_TAG_RE = re.compile(r"<[^>]+>")
# The optional `|name|` group is the whole fix: a substitution definition is
# `.. |docs| image:: URL`, so what follows `.. ` is not `[\w-]+::` and the pattern missed
# it. mdanalysis's README has 13 of them and `image` tied for its #1 keyword.
_RST_DIRECTIVE_RE = re.compile(r"^\s*\.\.\s+(?:\|[^|\n]+\|\s+)?[\w-]+::.*$", re.MULTILINE)
# ...and the in-body uses `|numfocus| |build| |cov|`, which are badge names, not words.
_RST_SUBREF_RE = re.compile(r"\|[\w-]+\|")
_RST_OPTION_RE = re.compile(r"^\s*:[\w-]+:.*$", re.MULTILINE)
# Markdown's *reference* forms, which the parenthesised patterns above never see. A badge
# row written `[![Stars][gh-stars-badge]][gh-stars-link]` survives both of them intact, and
# scvi-tools' first 300 README characters -- what the gate and HyDE are told the project IS
# -- were 100% exactly that. The definitions `[gh-stars-badge]: https://...` are their own
# line form and need their own pattern.
_MD_REF_IMAGE_RE = re.compile(r"!\[[^\]]*\]\[[^\]]*\]")
_MD_REF_LINK_RE = re.compile(r"\[([^\]]*)\]\[[^\]]*\]")  # keep the text, drop the label
_MD_LINK_DEF_RE = re.compile(r"^[ \t]*\[[^\]\n]+\]:[ \t]*\S+.*$", re.MULTILINE)
# Inline roles in both docs dialects: MyST's {func}`x` and reST's :mod:`x` / :py:class:`x`.
# `_RST_OPTION_RE` only anchors at line start, so a role used mid-sentence survives and its
# NAME becomes a term -- `mod` was mdanalysis's #4 keyword, `footcite` tblite's #6, and
# scanpy's `{smaller}` / `{pr}` / `{func}` were its top three. Only the name is removed: the
# target stays, because that is where the API names live (`:mod:`MDAnalysis.analysis`` is
# worth `mdanalysis` and `analysis`). The lookbehind keeps `https://` from matching.
_INLINE_ROLE_RE = re.compile(r"\{[a-zA-Z][\w+-]*\}(?=`)|(?<![\w`]):(?:[a-zA-Z][\w+-]*:)+(?=`)")

# Vocabulary that survives `stop_words="english"` but describes *packaging and docs
# tooling*, never a research topic. Measured across the 13 benchmark repos, these
# dominated the keyword list for several of them: the `cv` fixture's top six were
# `detectron2, md, undoc-members show-inheritance, show-inheritance, members, automodule`
# — five of six pure Sphinx. Library names are deliberately absent from this list; a repo
# whose subject really is `pytest` should still be able to say so.
_BOILERPLATE_STOPWORDS = frozenset(
    {
        # links and hosting
        "https",
        "http",
        "www",
        "com",
        "org",
        "io",
        "github",
        "gitlab",
        "badge",
        "badges",
        "svg",
        "png",
        "img",
        "shields",
        "href",
        "url",
        "link",
        # packaging and repo furniture
        "license",
        "licensed",
        "licence",
        "copyright",
        "install",
        "installation",
        "pip",
        "conda",
        "setup",
        "usage",
        "changelog",
        "contributing",
        "readme",
        "version",
        "versions",
        "release",
        "releases",
        "download",
        "downloads",
        # file extensions and formats
        "py",
        "yml",
        "yaml",
        "json",
        "md",
        "rst",
        "txt",
        "toml",
        "cfg",
        "ini",
        "sh",
        "csv",
        "html",
        "css",
        # Sphinx / docs directives
        "autodoc",
        "automodule",
        "autofunction",
        "autoclass",
        "autoattribute",
        "automethod",
        "currentmodule",
        "toctree",
        "maxdepth",
        "undoc",
        "members",
        "inheritance",
        "eval",
        "ref",
        "doc",
        "docs",
        "docstring",
        "api",
        "index",
        "genindex",
        "modindex",
        "versionadded",
        "versionchanged",
        "deprecated",
        "seealso",
        "rubric",
        "literalinclude",
        "codeblock",
        # generic prose scaffolding the english list misses
        "import",
        "imports",
        "example",
        "examples",
        "documentation",
        "reference",
        "note",
        "notes",
        "warning",
        "parameters",
        "returns",
        "args",
        "kwargs",
        # hyphenated Sphinx tokens, which survive as single terms under our token pattern
        "eval-rst",
        "undoc-members",
        "show-inheritance",
        "code-block",
        # code and template syntax quoted in docs — `def`, `self` and Jinja's `endfor`
        # are among the commonest terms in a docs/ tree and mean nothing as a topic
        "def",
        "cls",
        "self",
        "obj",
        "param",
        "true",
        "false",
        "none",
        "null",
        "endfor",
        "endif",
        "endblock",
        "endmacro",
        # build-system furniture, which dominates C/system repos with no Python manifest
        "make",
        "makefile",
        "cmake",
        "build",
        "src",
        "usr",
        "bin",
        "lib",
        "tar",
        "gz",
        "config",
        "configure",
    }
)


def _clean_document(text: str) -> str:
    """Strip markup that inflates term counts without carrying topic signal.

    The ORDER is load-bearing and each pair is a reference form ahead of its parenthesised
    twin: `![alt][ref]` has to go before `![alt](url)` or the outer brackets of a badge row
    survive as `[ ][link]`. Inline roles are stripped before `_RST_OPTION_RE`, which deletes
    whole lines beginning `:word:` — so a line that opens with ``:mod:`pkg.core` `` keeps its
    target text instead of being deleted entirely.
    """
    text = _MD_REF_IMAGE_RE.sub(" ", text)
    text = _MD_IMAGE_RE.sub(" ", text)
    text = _MD_REF_LINK_RE.sub(r"\1", text)
    text = _MD_LINK_RE.sub(r"\1", text)
    text = _MD_LINK_DEF_RE.sub(" ", text)
    text = _INLINE_ROLE_RE.sub(" ", text)
    text = _RST_DIRECTIVE_RE.sub(" ", text)
    text = _RST_SUBREF_RE.sub(" ", text)
    text = _RST_OPTION_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = _HTML_TAG_RE.sub(" ", text)
    return text


# `docs/` is a Python-web convention, and scientific codes mostly do not follow it. Of 19
# bio and materials repositories measured, five keep their manual in `doc/` (lammps, phonopy,
# deepmd-kit, sourmash, tblite), openmm uses `docs-source/`, and mdanalysis keeps it under
# `package/doc/`. Reading only `docs/` left seven of the nineteen profiled from their README
# alone: phonopy's top-20 was thirteen of its own dependency names.
_DOC_DIR_NAMES = ("docs", "doc", "docs-source")

# Documents that record a project's HISTORY rather than its subject. A changelog is a list of
# contributor names, version numbers and issue references. scanpy ships 69 of them under
# `docs/release-notes/` — 69 of its 113 corpus documents — and they are the single reason its
# top keywords were `smaller`, `pr`, `func` and two maintainer surnames. `_templates` holds
# Sphinx autosummary Jinja, whose loop variable reached its top eight.
#
# Only names that FIRE on that corpus are listed: `release-notes` (74 files), `_templates`
# (6), stem `changelog` (3), stem `changes` (1). The obvious near-synonyms — `releases`,
# `history`, `news`, `whatsnew`, `releasenotes` — matched nothing and are deliberately absent,
# because a list padded with plausible entries is indistinguishable from one that was
# measured, and this module's stoplists have been burned that way before.
_NON_TOPIC_DIRS = frozenset({"release-notes", "_templates"})
_NON_TOPIC_STEMS = frozenset({"changelog", "changes"})


def _doc_roots(repo_path: Path) -> list[Path]:
    """Directories holding the project's prose documentation.

    One level of nesting is searched because a package-in-subdirectory layout puts the docs
    under it (mdanalysis: `package/doc/`). Depth is capped at one on purpose: deeper walks
    start finding vendored third-party manuals, and on the 19 measured clones depth 1 already
    finds every doc tree that exists and nothing that is not one.

    Cost, recorded rather than capped: on lammps this admits 1,077 files and 7.3 MB, taking
    the profile from 17 ms to ~4 s. That is real, and it is noise beside the ~4 minutes an
    `rr update` spends on the network and the LLM stages. A file or byte cap would bound it,
    but every cap considered was itself unmeasured, and a silent truncation of the corpus is
    the failure mode this project keeps writing checks against.
    """
    roots: list[Path] = []
    for name in _DOC_DIR_NAMES:
        candidate = repo_path / name
        if candidate.is_dir():
            roots.append(candidate)
    for child in sorted(repo_path.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        for name in _DOC_DIR_NAMES:
            candidate = child / name
            if candidate.is_dir():
                roots.append(candidate)
    return roots


def _is_non_topic_doc(relative_path: Path) -> bool:
    """True for a doc file that records the project's history rather than its subject."""
    if any(part.lower() in _NON_TOPIC_DIRS for part in relative_path.parts):
        return True
    return relative_path.stem.lower() in _NON_TOPIC_STEMS


def _collect_text_corpus(repo_path: Path) -> list[str]:
    """Gather text documents from the repo for TF-IDF analysis.

    Returns a list of document strings. Each document is one logical source
    (README, each doc file, etc.).
    """
    documents: list[str] = []

    # The project's own description/keywords, if declared. Short and dense, so it goes
    # in first and stands as its own document rather than being merged into the README.
    metadata = _packaging_metadata_text(repo_path)
    if metadata.strip():
        documents.append(metadata)

    # README variants
    for name in ("README.md", "README.rst", "README.txt", "README"):
        text = _read_text_file(repo_path / name)
        if text:
            documents.append(_clean_document(text))

    # Documentation trees — read .md, .rst and .txt, skipping the project's own history
    for docs_dir in _doc_roots(repo_path):
        for ext in ("*.md", "*.rst", "*.txt"):
            for doc_file in docs_dir.rglob(ext):
                if _is_non_topic_doc(doc_file.relative_to(repo_path)):
                    continue
                text = _read_text_file(doc_file)
                if text:
                    documents.append(_clean_document(text))

    return documents


# arXiv ids in either era's format, as they appear in prose or in an abs/pdf URL. The month
# is VALIDATED (01-12), and that is the whole difference between a usable signal and noise:
# a bare `\d{4}\.\d{4,5}` also matches DOI fragments and version strings, and on the four
# repositories this was built against it matched `2019.10694` (a Comput. Phys. Commun. DOI in
# dscribe's README) and `1029.28096`, `1484.11876`, `2042.03300` in MACE's. All four are
# rejected here; every real id in the same files survives.
_CITED_ARXIV_NEW_RE = re.compile(r"\b((?:0[7-9]|[1-9]\d)(?:0[1-9]|1[0-2])\.\d{4,5})(?:v\d+)?\b")
_CITED_ARXIV_OLD_RE = re.compile(r"\b([a-z-]{2,}(?:\.[A-Za-z-]+)?/\d{7})(?:v\d+)?\b")


def cited_arxiv_ids_of(repo_path: Path) -> frozenset[str]:
    """arXiv ids the repository itself cites: README, CITATION file, and ``docs/``.

    Read **raw**, deliberately not through :func:`_clean_document`, which strips URLs --
    and most citations arrive as `arxiv.org/abs/<id>` links rather than as bare ids, so
    cleaning first would drop the majority of them.

    This walks the same files :func:`_collect_text_corpus` reads (plus ``CITATION*``) a
    second time rather than returning a second value from it: that function's
    list-of-strings contract has three callers outside this module, one of which joins it.
    """
    ids: set[str] = set()

    def scan(text: str) -> None:
        for pattern in (_CITED_ARXIV_NEW_RE, _CITED_ARXIV_OLD_RE):
            ids.update(match.group(1) for match in pattern.finditer(text))

    for name in ("README.md", "README.rst", "README.txt", "README"):
        scan(_read_text_file(repo_path / name))
    for citation in sorted(repo_path.glob("CITATION*")):
        if citation.is_file():
            scan(_read_text_file(citation))
    # The same roots as the topic corpus, but NOT its history filter: a release note saying
    # "implements arXiv:2303.14046" is exactly the evidence this function wants, and is
    # exactly the vocabulary the corpus excludes. Same files, different question.
    for docs_dir in _doc_roots(repo_path):
        # `.bib` joins the corpus extensions here and nowhere else: a bibliography is the
        # plainest statement a project makes about what it already cites, and it is the only
        # place some of them are recorded (dscribe cites the REMatch-kernel paper its own
        # tutorial teaches only in `docs/src/references.bib`). Built HTML is deliberately not
        # read: it is a generated copy of the same citations, and a committed docs build can
        # be thousands of files.
        for ext in ("*.md", "*.rst", "*.txt", "*.bib"):
            for doc_file in docs_dir.rglob(ext):
                scan(_read_text_file(doc_file))
    return frozenset(ids)


def _repo_prose(repo_path: Path, budget: int) -> str:
    """The repo's own description: its README, else its packaging description.

    Deliberately NOT ``_collect_text_corpus(repo)[0]``. That corpus puts the packaging
    metadata first whenever a project declares one, so taking element 0 yields a 23-230
    character tagline on almost every real repo — "Python HTTP for Humans." for `requests`.
    An earlier experiment did exactly that and was recorded as a README result; it was not
    one. See evals/RESULTS.md -> "what the README variant actually sent".

    The README is preferred because it is the document that says what the project is FOR.
    Metadata is the fallback, not the default: it is one sentence and usually already
    implied by the anchors.
    """
    if budget <= 0:
        return ""
    for name in ("README.md", "README.rst", "README.txt", "README"):
        text = _read_text_file(repo_path / name)
        if text:
            cleaned = _clean_document(text).strip()
            if cleaned:
                return cleaned[:budget]
    return _packaging_metadata_text(repo_path).strip()[:budget]


# Matches TfidfVectorizer's token_pattern below, so "occurs in the corpus" is judged on
# the same tokens the keywords were drawn from.
_TOKEN_RE = re.compile(r"(?u)\b[a-zA-Z][a-zA-Z0-9_-]{1,}\b")


def _observed_phrases(documents: list[str], keywords: list[tuple[str, float]]) -> list[str]:
    """Adjacent keyword pairs that literally occur in the text, as "a b" strings.

    Adjacency is **literal** — stop words are not skipped over. A quoted phrase query is
    matched literally by every search API it is sent to, so "use page" must really be
    followed by "page" in some document to be worth asking for. Skipping stop words here
    would re-admit exactly the phrases this is meant to reject: "use OF page" would
    license the query `"use page"`, which no paper title contains.

    Restricted to the keyword vocabulary, so the result is a few dozen pairs at most
    rather than every bigram in a 60k-character README.
    """
    vocab = {term for term, _weight in keywords if " " not in term}
    if len(vocab) < 2:
        return []
    seen: set[str] = set()
    ordered: list[str] = []
    for document in documents:
        tokens = _TOKEN_RE.findall(document.lower())
        for first, second in zip(tokens, tokens[1:], strict=False):
            if first in vocab and second in vocab:
                phrase = f"{first} {second}"
                if phrase not in seen:
                    seen.add(phrase)
                    ordered.append(phrase)
    return ordered


def _extract_keywords(
    documents: list[str],
    anchors: list[str],
    max_keywords: int = 20,
) -> list[tuple[str, float]]:
    """Run TF-IDF on the collected documents and return top keywords.

    If there are no documents, falls back to anchor names as keywords.
    """
    if not documents:
        # Fallback: use anchor names as keywords with uniform weight
        return [(a, 1.0) for a in anchors[:max_keywords]]

    # Combine all documents with anchors as an additional "document"
    # so anchor terms get some weight even if README doesn't mention them.
    all_docs = documents + [" ".join(anchors)] if anchors else documents

    # With very few documents, max_df=0.95 would filter out all terms
    # (every term appears in 100% of a single-doc corpus).
    effective_max_df = 1.0 if len(all_docs) < 3 else 0.95

    # `stop_words="english"` removes "the" and "and" but not `license`, `https` or
    # `automodule`, which is how README furniture reached the arXiv API as a query.
    # Stopping happens before n-grams are formed, so this also kills the bigrams built
    # out of two boilerplate tokens ("license https").
    stop_words = sorted(ENGLISH_STOP_WORDS | _BOILERPLATE_STOPWORDS)

    vectorizer = TfidfVectorizer(
        max_features=200,
        stop_words=stop_words,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9_-]{1,}\b",
        ngram_range=(1, 2),
        max_df=effective_max_df,
        min_df=1,
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(all_docs)
    except ValueError:
        # Edge case: all documents empty after tokenization
        return [(a, 1.0) for a in anchors[:max_keywords]]

    feature_names = vectorizer.get_feature_names_out()

    # Average TF-IDF across all documents
    avg_scores = tfidf_matrix.mean(axis=0).A1
    top_indices = avg_scores.argsort()[::-1][:max_keywords]

    keywords: list[tuple[str, float]] = []
    for idx in top_indices:
        score = float(avg_scores[idx])
        if score > 0:
            keywords.append((feature_names[idx], round(score, 4)))

    return keywords


def profile_repo(
    repo_path: str | Path,
    profiler_cfg: object | None = None,
) -> RepoProfile:
    """Build a topic profile for the repository at *repo_path*.

    If *profiler_cfg* is a ProfilerConfig with ``scan_source=True``,
    source code analysis is performed: extracted imports are merged into
    anchors, ML patterns into domains, and identifiers into the TF-IDF corpus.
    """
    repo_path = Path(repo_path).resolve()

    if not repo_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {repo_path}")

    anchors = _extract_anchors(repo_path)
    domains = _infer_domains(anchors)
    documents = _collect_text_corpus(repo_path)
    source_signals: list[str] = []

    # Typed README entities (optional). Runs BEFORE source scanning so that the two
    # optional anchor sources compose in a fixed order and a profile is reproducible
    # whichever of them is enabled.
    if getattr(profiler_cfg, "typed_anchors", False):
        from reporadar.typed_anchors import extract_typed_anchors

        existing = set(anchors)
        for span in extract_typed_anchors(repo_path, llm_cfg=profiler_cfg):
            if span not in existing:
                anchors.append(span)
                existing.add(span)
        domains = _infer_domains(anchors)

    # Source code analysis (optional)
    scan_source = getattr(profiler_cfg, "scan_source", False)
    if scan_source:
        from reporadar.source_analysis import (
            detect_ml_patterns,
            extract_identifiers,
            extract_imports,
        )

        max_files = getattr(profiler_cfg, "max_files", 100)
        extensions = getattr(profiler_cfg, "source_extensions", None)

        # Merge imports into anchors
        src_imports = extract_imports(repo_path, extensions=extensions, max_files=max_files)
        existing_anchors = set(anchors)
        for imp in src_imports:
            if imp not in existing_anchors:
                anchors.append(imp)
                existing_anchors.add(imp)

        # Re-infer domains with expanded anchors
        domains = _infer_domains(anchors)

        # Detect ML patterns and merge into domains
        ml_signals = detect_ml_patterns(repo_path, max_files=max_files)
        source_signals = ml_signals
        domain_set = set(domains)
        for sig in ml_signals:
            if sig not in domain_set:
                domains.append(sig)
                domain_set.add(sig)

        # Add identifiers as an additional TF-IDF document
        identifiers = extract_identifiers(repo_path, max_files=max_files)
        if identifiers:
            documents.append(" ".join(identifiers))

    keywords = _extract_keywords(documents, anchors)

    return RepoProfile(
        keywords=keywords,
        anchors=anchors,
        domains=domains,
        source_signals=source_signals,
        prose=_repo_prose(repo_path, getattr(profiler_cfg, "prose_chars", 300)),
        corpus_phrases=_observed_phrases(documents, keywords),
        cited_arxiv_ids=cited_arxiv_ids_of(repo_path),
    )
