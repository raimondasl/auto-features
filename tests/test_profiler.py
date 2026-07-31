"""Tests for reporadar.profiler."""

from __future__ import annotations

from pathlib import Path

import pytest

from reporadar.profiler import (
    RepoProfile,
    _extract_anchors,
    _extract_keywords,
    _infer_domains,
    _parse_package_json,
    _parse_pyproject_toml,
    _parse_requirements_txt,
    profile_repo,
)


class TestParseRequirementsTxt:
    def test_parses_packages(self, fixtures_dir: Path) -> None:
        packages = _parse_requirements_txt(fixtures_dir / "sample_requirements.txt")
        assert "torch" in packages
        assert "transformers" in packages
        assert "langchain" in packages
        assert "numpy" in packages
        assert "pytest" in packages

    def test_skips_comments_and_flags(self, fixtures_dir: Path) -> None:
        packages = _parse_requirements_txt(fixtures_dir / "sample_requirements.txt")
        for pkg in packages:
            assert not pkg.startswith("#")
            assert not pkg.startswith("-")

    def test_strips_version_specifiers(self, fixtures_dir: Path) -> None:
        packages = _parse_requirements_txt(fixtures_dir / "sample_requirements.txt")
        for pkg in packages:
            assert ">=" not in pkg
            assert "=" not in pkg

    def test_nonexistent_file(self, tmp_path: Path) -> None:
        result = _parse_requirements_txt(tmp_path / "missing.txt")
        assert result == []


class TestParsePyprojectToml:
    def test_parses_dependencies(self, fixtures_dir: Path) -> None:
        packages = _parse_pyproject_toml(fixtures_dir / "sample_pyproject.toml")
        assert "torch" in packages
        assert "transformers" in packages
        assert "fastapi" in packages
        assert "scikit-learn" in packages

    def test_parses_optional_dependencies(self, fixtures_dir: Path) -> None:
        packages = _parse_pyproject_toml(fixtures_dir / "sample_pyproject.toml")
        assert "pytest" in packages
        assert "mypy" in packages

    def test_nonexistent_file(self, tmp_path: Path) -> None:
        result = _parse_pyproject_toml(tmp_path / "missing.toml")
        assert result == []


class TestParsePackageJson:
    def test_parses_dependencies(self, fixtures_dir: Path) -> None:
        packages = _parse_package_json(fixtures_dir / "sample_package.json")
        assert "react" in packages
        assert "next" in packages

    def test_parses_dev_dependencies(self, fixtures_dir: Path) -> None:
        packages = _parse_package_json(fixtures_dir / "sample_package.json")
        assert "typescript" in packages
        assert "eslint" in packages

    def test_handles_scoped_packages(self, fixtures_dir: Path) -> None:
        packages = _parse_package_json(fixtures_dir / "sample_package.json")
        # @tanstack/react-query should be cleaned
        assert any("tanstack" in p for p in packages)

    def test_nonexistent_file(self, tmp_path: Path) -> None:
        result = _parse_package_json(tmp_path / "missing.json")
        assert result == []


class TestExtractAnchors:
    def test_collects_from_all_manifests(self, tmp_repo: Path) -> None:
        anchors = _extract_anchors(tmp_repo)
        # From requirements.txt
        assert "torch" in anchors
        # From pyproject.toml
        assert "fastapi" in anchors
        # From package.json
        assert "react" in anchors

    def test_deduplicates(self, tmp_repo: Path) -> None:
        anchors = _extract_anchors(tmp_repo)
        # torch appears in both requirements.txt and pyproject.toml
        assert anchors.count("torch") == 1

    def test_empty_repo(self, tmp_repo_empty: Path) -> None:
        anchors = _extract_anchors(tmp_repo_empty)
        assert anchors == []


class TestInferDomains:
    def test_maps_known_packages(self) -> None:
        domains = _infer_domains(["torch", "transformers", "fastapi"])
        assert "deep learning" in domains
        assert "NLP" in domains
        assert "web APIs" in domains

    def test_empty_anchors(self) -> None:
        domains = _infer_domains([])
        assert domains == []

    def test_unknown_packages_ignored(self) -> None:
        domains = _infer_domains(["some-obscure-lib", "another-one"])
        assert domains == []


class TestExtractKeywords:
    def test_returns_sorted_keywords(self) -> None:
        docs = ["retrieval augmented generation with long context transformers"]
        keywords = _extract_keywords(docs, [])
        assert len(keywords) > 0
        # Should be sorted by weight descending
        weights = [w for _, w in keywords]
        assert weights == sorted(weights, reverse=True)

    def test_fallback_to_anchors_when_no_docs(self) -> None:
        keywords = _extract_keywords([], ["torch", "transformers"])
        assert len(keywords) == 2
        assert keywords[0] == ("torch", 1.0)

    def test_respects_max_keywords(self) -> None:
        docs = ["word " * 100]  # Lots of text
        keywords = _extract_keywords(docs, [], max_keywords=5)
        assert len(keywords) <= 5


class TestProfileRepo:
    def test_full_profile(self, tmp_repo: Path) -> None:
        profile = profile_repo(tmp_repo)

        assert isinstance(profile, RepoProfile)
        assert len(profile.keywords) > 0
        assert len(profile.anchors) > 0
        assert len(profile.domains) > 0

    def test_minimal_repo(self, tmp_repo_minimal: Path) -> None:
        profile = profile_repo(tmp_repo_minimal)

        assert isinstance(profile, RepoProfile)
        assert len(profile.keywords) > 0
        # No manifest files → no anchors
        assert profile.anchors == []
        assert profile.domains == []

    def test_empty_repo(self, tmp_repo_empty: Path) -> None:
        profile = profile_repo(tmp_repo_empty)

        assert isinstance(profile, RepoProfile)
        # No docs, no anchors → empty keywords
        assert profile.keywords == []
        assert profile.anchors == []
        assert profile.domains == []

    def test_nonexistent_dir(self) -> None:
        with pytest.raises(NotADirectoryError):
            profile_repo("/nonexistent/path/12345")

    def test_keywords_have_positive_weights(self, tmp_repo: Path) -> None:
        profile = profile_repo(tmp_repo)
        for _term, weight in profile.keywords:
            assert weight > 0


# A repo shaped like diffusers/transformers/peft: no requirements.txt, a pyproject.toml
# holding only tool config, and dependencies declared in setup.py. This layout is
# standard across the HuggingFace ecosystem — exactly the ML repos RepoRadar targets.
_HF_STYLE_SETUP_PY = """\
import sys
from setuptools import setup, find_packages

_deps = [
    "Pillow",
    "torch>=2.1",
    "transformers>=4.40.0",
    "accelerate>=0.31.0",
    "numpy",
]
deps = {b: a for a, b in ((x, x.split(">")[0]) for x in _deps)}

setup(
    name="diffusers",
    version="0.40.0.dev0",
    description="State-of-the-art diffusion in PyTorch and JAX.",
    keywords="deep learning diffusion jax pytorch stable diffusion",
    license="Apache 2.0 License",
    url="https://github.com/huggingface/diffusers",
    packages=find_packages("src"),
    install_requires=list(deps.values()),
)
"""

# The first screen of a real ML README: badges, links, install instructions. Under the
# old profiler this text alone decided the search queries.
_BOILERPLATE_README = """\
# Diffusers

[![License](https://img.shields.io/github/license/huggingface/diffusers.svg)](https://github.com/huggingface/diffusers/blob/main/LICENSE)
[![GitHub release](https://img.shields.io/github/release/huggingface/diffusers.svg)](https://github.com/huggingface/diffusers/releases)

Install with pip:

```
pip install diffusers
```

See the documentation at https://huggingface.co/docs/diffusers for usage.
License: Apache 2.0. Report issues at https://github.com/huggingface/diffusers/issues
"""


def _hf_style_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "hfrepo"
    repo.mkdir()
    (repo / "setup.py").write_text(_HF_STYLE_SETUP_PY, encoding="utf-8")
    # A pyproject with tool config only — no [project] table, so no dependencies there.
    (repo / "pyproject.toml").write_text(
        '[tool.ruff]\nline-length = 119\n\n[tool.isort]\nprofile = "black"\n', encoding="utf-8"
    )
    (repo / "README.md").write_text(_BOILERPLATE_README, encoding="utf-8")
    return repo


class TestSetupPyAnchors:
    def test_dependencies_are_found_in_setup_py(self, tmp_path: Path) -> None:
        from reporadar.profiler import _parse_setup_py

        found = _parse_setup_py(_hf_style_repo(tmp_path) / "setup.py")
        assert "torch" in found
        assert "transformers" in found
        assert "accelerate" in found
        # Version specifiers are stripped, as they are for every other manifest.
        assert not any(">" in f for f in found)

    def test_a_hf_style_repo_is_no_longer_anchorless(self, tmp_path: Path) -> None:
        """The whole defect in one assertion.

        With no requirements.txt and no `[project]` table, `_extract_anchors` returned
        `[]`. Empty anchors meant `_infer_domains` had nothing to map, the anchor
        pseudo-document never entered TF-IDF, and README boilerplate decided the queries.
        """
        repo = _hf_style_repo(tmp_path)
        profile = profile_repo(repo)
        assert profile.anchors, "setup.py dependencies are still not being read"
        assert "torch" in profile.anchors
        # And the anchors have to reach domains, which is what they are for.
        assert "deep learning" in profile.domains

    def test_setup_py_is_never_executed(self, tmp_path: Path) -> None:
        """Profiling a repo must not run its code. Parsed with `ast`, never `exec`.

        A setup.py is arbitrary Python from an untrusted repository. If profiling
        executed it, `rr profile` on a cloned repo would be remote code execution.
        """
        from reporadar.profiler import _parse_setup_py

        repo = tmp_path / "hostile"
        repo.mkdir()
        canary = tmp_path / "canary.txt"
        # The side effect goes FIRST, before any import. Putting it after
        # `from setuptools import setup` made this test unable to fail: setuptools is
        # not in the venv, so an exec would raise before ever reaching the canary and
        # the assertion below would pass against a profiler that does execute the file.
        (repo / "setup.py").write_text(
            f"open({str(canary)!r}, 'w').write('executed')\n"
            f"from setuptools import setup\n"
            f"setup(name='x', install_requires=['torch'])\n",
            encoding="utf-8",
        )
        found = _parse_setup_py(repo / "setup.py")
        assert "torch" in found, "the fixture must still parse, or this proves nothing"
        assert not canary.exists(), "profiling EXECUTED the repo's setup.py"

    def test_an_unparseable_setup_py_does_not_break_profiling(self, tmp_path: Path) -> None:
        from reporadar.profiler import _parse_setup_py

        repo = tmp_path / "broken"
        repo.mkdir()
        (repo / "setup.py").write_text("this is (not python\n", encoding="utf-8")
        assert _parse_setup_py(repo / "setup.py") == []


class TestPackagingMetadata:
    def test_keywords_and_description_are_collected(self, tmp_path: Path) -> None:
        # `keywords="deep learning diffusion ..."` is the author stating the subject
        # outright — the densest topic signal in the repo.
        from reporadar.profiler import _packaging_metadata_text

        text = _packaging_metadata_text(_hf_style_repo(tmp_path)).lower()
        assert "diffusion" in text
        assert "state-of-the-art" in text

    def test_pyproject_description_is_collected(self, tmp_path: Path) -> None:
        from reporadar.profiler import _packaging_metadata_text

        repo = tmp_path / "modern"
        repo.mkdir()
        (repo / "pyproject.toml").write_text(
            '[project]\nname = "x"\ndescription = "A retrieval engine for sparse indexes"\n'
            'keywords = ["retrieval", "bm25"]\n',
            encoding="utf-8",
        )
        text = _packaging_metadata_text(repo).lower()
        assert "retrieval" in text and "bm25" in text

    def test_a_repo_without_metadata_yields_nothing(self, tmp_path: Path) -> None:
        from reporadar.profiler import _packaging_metadata_text

        repo = tmp_path / "bare"
        repo.mkdir()
        assert _packaging_metadata_text(repo) == ""


class TestBoilerplateIsNotATopic:
    def test_badges_and_urls_are_stripped(self) -> None:
        from reporadar.profiler import _clean_document

        cleaned = _clean_document(_BOILERPLATE_README)
        assert "shields.io" not in cleaned
        assert "https://" not in cleaned
        # Link *text* survives; only the target is dropped.
        assert "License" in cleaned

    def test_rst_directives_are_stripped(self) -> None:
        from reporadar.profiler import _clean_document

        cleaned = _clean_document(
            ".. automodule:: mypkg.core\n    :members:\n    :undoc-members:\n\nReal prose here.\n"
        )
        assert "automodule" not in cleaned
        assert "Real prose here." in cleaned

    def test_readme_furniture_never_becomes_a_keyword(self, tmp_path: Path) -> None:
        """`license` and `https` were live arXiv queries, not a hypothetical.

        A stored run for the `diffusion` benchmark repo transmitted `(all:license)`,
        `(all:https)` and `(all:"license https")` — three of its eight queries — because
        `stop_words="english"` removes "the" and "and" but not README furniture.
        """
        repo = _hf_style_repo(tmp_path)
        terms = {term for term, _ in profile_repo(repo).keywords}
        for junk in ("license", "https", "http", "github", "com", "install", "pip", "usage"):
            assert junk not in terms, f"{junk!r} is still a keyword, and so still a query"

    def test_the_real_topic_survives_the_filtering(self, tmp_path: Path) -> None:
        # The filter must not be so aggressive that it removes the signal too. This is
        # the assertion that stops the stoplist growing into everything.
        repo = _hf_style_repo(tmp_path)
        terms = {term for term, _ in profile_repo(repo).keywords}
        assert "diffusion" in terms
        assert terms & {"torch", "pytorch", "diffusers"}

    def test_sphinx_vocabulary_never_becomes_a_keyword(self, tmp_path: Path) -> None:
        # Measured on the `cv` benchmark repo, five of its top six keywords were Sphinx
        # directives: `md, undoc-members show-inheritance, show-inheritance, members,
        # automodule`. A docs/ tree can outweigh the README by an order of magnitude.
        repo = tmp_path / "sphinxy"
        repo.mkdir()
        (repo / "README.md").write_text("# Detector\n\nAn object detection library.\n", "utf-8")
        docs = repo / "docs"
        docs.mkdir()
        for i in range(6):
            (docs / f"api{i}.rst").write_text(
                f".. automodule:: pkg.mod{i}\n"
                "    :members:\n"
                "    :undoc-members:\n"
                "    :show-inheritance:\n\n"
                ".. currentmodule:: pkg\n\n"
                "Segmentation of objects in images.\n",
                encoding="utf-8",
            )
        terms = {term for term, _ in profile_repo(repo).keywords}
        for junk in ("automodule", "members", "undoc-members", "show-inheritance", "currentmodule"):
            assert junk not in terms, f"Sphinx directive {junk!r} is still a keyword"


class TestQueriesAreAboutTheRepo:
    def test_no_query_is_built_from_boilerplate(self, tmp_path: Path) -> None:
        """End-to-end: the profile feeds `build_queries`, and that is what is transmitted.

        Asserting on keywords alone would miss a regression in how they become queries,
        which is the stage that actually reaches the arXiv API.
        """
        from reporadar.collector import build_queries
        from reporadar.config import ArxivConfig, QueriesConfig

        repo = _hf_style_repo(tmp_path)
        queries = build_queries(
            profile_repo(repo), QueriesConfig(), ArxivConfig(categories=["cs.CV"])
        )
        assert queries
        joined = " ".join(queries).lower()
        for junk in ("license", "https", "github"):
            assert f"all:{junk}" not in joined, f"still searching arXiv for {junk!r}"
            assert f'all:"{junk}' not in joined
        assert any("diffusion" in q.lower() for q in queries), queries
