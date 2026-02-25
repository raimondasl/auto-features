"""Repo topic profiling — extract keywords, anchors, and domains from a repo."""

from __future__ import annotations

import json
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class RepoProfile:
    """Result of profiling a repository."""

    keywords: list[tuple[str, float]]  # (term, tfidf_weight) sorted desc
    anchors: list[str]  # library/package names found in manifests
    domains: list[str]  # inferred domain labels
    source_signals: list[str] = field(default_factory=list)  # ML/domain patterns from source


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
        # Strip version specifiers, extras, etc.
        name = re.split(r"[>=<!;\[\]]", line)[0].strip()
        if name:
            packages.append(name.lower())
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
        name = re.split(r"[>=<!;\[\]]", dep)[0].strip()
        if name:
            packages.append(name.lower())

    # [project.optional-dependencies]
    for deps in data.get("project", {}).get("optional-dependencies", {}).values():
        for dep in deps:
            name = re.split(r"[>=<!;\[\]]", dep)[0].strip()
            if name:
                packages.append(name.lower())

    return packages


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


def _extract_anchors(repo_path: Path) -> list[str]:
    """Collect library/package names from all supported manifest files."""
    anchors: list[str] = []
    anchors.extend(_parse_requirements_txt(repo_path / "requirements.txt"))
    anchors.extend(_parse_pyproject_toml(repo_path / "pyproject.toml"))
    anchors.extend(_parse_package_json(repo_path / "package.json"))
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


def _collect_text_corpus(repo_path: Path) -> list[str]:
    """Gather text documents from the repo for TF-IDF analysis.

    Returns a list of document strings. Each document is one logical source
    (README, each doc file, etc.).
    """
    documents: list[str] = []

    # README variants
    for name in ("README.md", "README.rst", "README.txt", "README"):
        text = _read_text_file(repo_path / name)
        if text:
            documents.append(text)

    # docs/ directory — read .md and .rst files
    docs_dir = repo_path / "docs"
    if docs_dir.is_dir():
        for ext in ("*.md", "*.rst", "*.txt"):
            for doc_file in docs_dir.rglob(ext):
                text = _read_text_file(doc_file)
                if text:
                    documents.append(text)

    return documents


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

    vectorizer = TfidfVectorizer(
        max_features=200,
        stop_words="english",
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
    )
