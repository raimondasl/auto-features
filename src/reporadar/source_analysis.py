"""Source code analysis — extract imports, ML patterns, and identifiers."""

from __future__ import annotations

import ast
import re
from pathlib import Path

# Directories to skip when scanning source files.
_SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", ".tox", ".eggs", "dist"}

# Python stdlib top-level modules (subset — covers the most common ones).
_PYTHON_STDLIB = frozenset(
    {
        "abc", "argparse", "ast", "asyncio", "base64", "bisect", "builtins",
        "calendar", "cmath", "codecs", "collections", "colorsys", "concurrent",
        "configparser", "contextlib", "copy", "csv", "ctypes", "dataclasses",
        "datetime", "decimal", "difflib", "email", "enum", "errno",
        "filecmp", "fnmatch", "fractions", "ftplib", "functools", "gc",
        "getpass", "gettext", "glob", "gzip", "hashlib", "heapq", "hmac",
        "html", "http", "imaplib", "importlib", "inspect", "io", "ipaddress",
        "itertools", "json", "keyword", "locale", "logging", "lzma", "math",
        "mimetypes", "multiprocessing", "netrc", "numbers", "operator", "os",
        "pathlib", "pickle", "platform", "pprint", "profile", "queue",
        "random", "re", "secrets", "select", "shelve", "shlex", "shutil",
        "signal", "site", "smtplib", "socket", "sqlite3", "ssl",
        "stat", "statistics", "string", "struct", "subprocess", "sys",
        "sysconfig", "tempfile", "textwrap", "threading", "time", "timeit",
        "token", "tokenize", "tomllib", "traceback", "types", "typing",
        "unicodedata", "unittest", "urllib", "uuid", "venv", "warnings",
        "weakref", "webbrowser", "xml", "xmlrpc", "zipfile", "zipimport",
        "zlib", "_thread",
    }
)

# Regex patterns for Python imports.
_PY_IMPORT_RE = re.compile(
    r"^\s*(?:import|from)\s+([\w.]+)", re.MULTILINE
)

# Regex patterns for JS/TS imports.
_JS_IMPORT_RE = re.compile(
    r"""(?:import\s+.*?\s+from\s+['"]([^'"./][^'"]*?)['"]"""
    r"""|require\s*\(\s*['"]([^'"./][^'"]*?)['"]\s*\))""",
    re.MULTILINE,
)

# ML-related code patterns.
_ML_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\.backward\(\)"), "autograd/backprop"),
    (re.compile(r"\bnn\.Module\b"), "PyTorch neural networks"),
    (re.compile(r"\bDataLoader\b"), "data loading pipeline"),
    (re.compile(r"\boptimizer\.(?:step|zero_grad)\b"), "optimizer training loop"),
    (re.compile(r"\bmodel\.fit\b"), "Keras/sklearn model training"),
    (re.compile(r"\bmodel\.predict\b"), "model inference"),
    (re.compile(r"\btf\.(?:keras|data|function)\b"), "TensorFlow"),
    (re.compile(r"\bjax\.(?:grad|jit|vmap)\b"), "JAX transformations"),
    (re.compile(r"\b(?:CrossEntropyLoss|MSELoss|BCELoss)\b"), "loss functions"),
    (re.compile(r"\bfrom\s+transformers\s+import\b"), "HuggingFace transformers"),
    (re.compile(r"\bAutoTokenizer\b"), "tokenization pipeline"),
    (re.compile(r"\bwandb\.(?:init|log)\b"), "experiment tracking"),
    (re.compile(r"\bmlflow\.(?:start_run|log)\b"), "MLflow tracking"),
]


def _iter_source_files(
    repo_path: Path, extensions: list[str], max_files: int
) -> list[Path]:
    """Collect source files up to *max_files*, skipping ignored dirs."""
    files: list[Path] = []
    for ext in extensions:
        for path in repo_path.rglob(f"*{ext}"):
            if any(part in _SKIP_DIRS for part in path.relative_to(repo_path).parts):
                continue
            files.append(path)
            if len(files) >= max_files:
                return files
    return files


def extract_imports(
    repo_path: Path,
    extensions: list[str] | None = None,
    max_files: int = 100,
) -> list[str]:
    """Extract third-party import names from source files.

    Skips Python stdlib modules. Returns deduplicated, sorted package names.
    """
    if extensions is None:
        extensions = [".py", ".js", ".ts", ".tsx", ".jsx"]

    files = _iter_source_files(repo_path, extensions, max_files)
    imports: set[str] = set()

    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        suffix = path.suffix

        if suffix == ".py":
            for m in _PY_IMPORT_RE.finditer(text):
                top = m.group(1).split(".")[0]
                if top and top not in _PYTHON_STDLIB and not top.startswith("_"):
                    imports.add(top.lower())
        elif suffix in (".js", ".ts", ".tsx", ".jsx"):
            for m in _JS_IMPORT_RE.finditer(text):
                pkg = (m.group(1) or m.group(2) or "").strip()
                if pkg:
                    # Take the scope+name for scoped packages, or just the name
                    if pkg.startswith("@"):
                        parts = pkg.split("/")
                        name = "/".join(parts[:2]) if len(parts) >= 2 else pkg
                    else:
                        name = pkg.split("/")[0]
                    imports.add(name.lower())

    return sorted(imports)


def detect_ml_patterns(repo_path: Path, max_files: int = 100) -> list[str]:
    """Scan Python files for ML-related code patterns.

    Returns a deduplicated list of domain signal strings.
    """
    py_files = _iter_source_files(repo_path, [".py"], max_files)
    signals: set[str] = set()

    for path in py_files:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        for pattern, signal in _ML_PATTERNS:
            if pattern.search(text):
                signals.add(signal)

    return sorted(signals)


def extract_identifiers(repo_path: Path, max_files: int = 100) -> list[str]:
    """Extract public class and function names from Python files using AST.

    Returns a deduplicated, sorted list of identifier strings.
    """
    py_files = _iter_source_files(repo_path, [".py"], max_files)
    identifiers: set[str] = set()

    for path in py_files:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        try:
            tree = ast.parse(text, filename=str(path))
        except SyntaxError:
            continue

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                identifiers.add(node.name)
            elif isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                identifiers.add(node.name)

    return sorted(identifiers)
