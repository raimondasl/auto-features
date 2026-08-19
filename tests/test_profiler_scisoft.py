"""Repository layouts and markup the profiler did not handle, found on scientific software.

Nineteen bioinformatics and materials-science repositories were profiled with the shipping
profiler (`evals/RESEARCH-scientific-software.md`). Seven of them read no documentation at
all because their manual is in `doc/` rather than `docs/`; MACE declared torch, e3nn and
matscipy in a `setup.cfg` nothing parsed and reached the gate as "Dependencies/libraries:
none"; scanpy's top keywords were `smaller`, `pr` and `func` — MyST role names from 69
release-note files; and scvi-tools' 300-character prose, which is what the gate and HyDE are
told the project *is*, was 100% badge markup.

Every case below is one of those repositories, named in the assertion, so a reader can check
the claim rather than take it. The paired guard for all of it is
`tests/test_profiler_golden.py`: these fixes must leave the four ML benchmark fixtures
byte-identical, and that is asserted rather than argued.
"""

from __future__ import annotations

from pathlib import Path

from reporadar.profiler import (
    _clean_document,
    _collect_text_corpus,
    _extract_anchors,
    _infer_domains,
    _parse_requirements_txt,
    _parse_setup_cfg,
    cited_arxiv_ids_of,
)


class TestManifestsThePythonWebConventionsMiss:
    def test_setup_cfg_install_requires_becomes_anchors(self, tmp_path: Path) -> None:
        """MACE declared torch, e3nn and matscipy and profiled with NO anchors at all.

        A project whose metadata lives in setup.cfg ships a pyproject.toml holding only
        `[build-system]`, so every other parser found nothing.
        """
        (tmp_path / "setup.cfg").write_text(
            "[metadata]\nname = mace\n\n[options]\ninstall_requires =\n"
            "    torch>=1.12\n    e3nn==0.4.4\n    # a comment\n    matscipy\n",
            encoding="utf-8",
        )
        assert _parse_setup_cfg(tmp_path / "setup.cfg") == ["torch", "e3nn", "matscipy"]

    def test_extras_require_is_deliberately_not_read(self, tmp_path: Path) -> None:
        """Measured and rejected: on MACE it adds pytest, black, isort, mypy and pre-commit,
        and `pytest` then becomes the repository's top keyword and an arXiv query."""
        (tmp_path / "setup.cfg").write_text(
            "[options]\ninstall_requires =\n    torch\n\n"
            "[options.extras_require]\ndev =\n    pytest\n    black\n",
            encoding="utf-8",
        )
        assert _parse_setup_cfg(tmp_path / "setup.cfg") == ["torch"]

    def test_a_percent_sign_does_not_cost_the_whole_profile(self, tmp_path: Path) -> None:
        """setup.cfg is setuptools' config file, not configparser's: a literal `%` in a
        description is not a substitution and must not raise."""
        (tmp_path / "setup.cfg").write_text(
            "[metadata]\ndescription = 100% coverage\n\n[options]\ninstall_requires =\n    torch\n",
            encoding="utf-8",
        )
        assert _parse_setup_cfg(tmp_path / "setup.cfg") == ["torch"]

    def test_compatible_release_specifiers_are_stripped(self, tmp_path: Path) -> None:
        """matminer pins `scikit-learn~=1.3` and yielded the anchor `scikit-learn~`, which
        misses PACKAGE_DOMAIN_MAP — so a scikit-learn project never inferred "machine
        learning". The `~` was absent from one character class and present in another."""
        (tmp_path / "requirements.txt").write_text(
            "requests~=2.31\nscikit-learn~=1.3\ntorch>=2.1\n", encoding="utf-8"
        )
        names = _parse_requirements_txt(tmp_path / "requirements.txt")
        assert names == ["requests", "scikit-learn", "torch"]
        assert "machine learning" in _infer_domains(names)

    def test_a_package_subdirectory_is_read_only_when_the_root_declares_nothing(
        self, tmp_path: Path
    ) -> None:
        """MDAnalysis ships its library from `package/`; its root holds CI and a testsuite.

        Firing only on an empty root is what keeps every repository that profiles today
        byte-identical, and confines the change to the ones with the defect.
        """
        (tmp_path / "package").mkdir()
        (tmp_path / "package" / "requirements.txt").write_text("numpy\nscipy\n", encoding="utf-8")
        assert _extract_anchors(tmp_path) == ["numpy", "scipy"]

        (tmp_path / "requirements.txt").write_text("torch\n", encoding="utf-8")
        assert _extract_anchors(tmp_path) == ["torch"], "a declaring root must win outright"


class TestDocumentationTreesAndHistory:
    def test_doc_and_docs_source_trees_are_read(self, tmp_path: Path) -> None:
        """Five of nineteen scientific repositories keep their manual in `doc/`, one in
        `docs-source/`. They profiled from the README alone — phonopy's top-20 was thirteen
        of its own dependency names."""
        (tmp_path / "README.md").write_text("A phonon code.", encoding="utf-8")
        (tmp_path / "doc").mkdir()
        (tmp_path / "doc" / "guide.rst").write_text("supercell displacements", encoding="utf-8")
        (tmp_path / "docs-source").mkdir()
        (tmp_path / "docs-source" / "api.md").write_text("force constants", encoding="utf-8")

        corpus = " ".join(_collect_text_corpus(tmp_path))
        assert "supercell" in corpus
        assert "force constants" in corpus

    def test_a_nested_doc_tree_is_found_one_level_down(self, tmp_path: Path) -> None:
        """mdanalysis keeps its 182 doc files under `package/doc/`."""
        (tmp_path / "README.md").write_text("Trajectory analysis.", encoding="utf-8")
        nested = tmp_path / "package" / "doc"
        nested.mkdir(parents=True)
        (nested / "topology.rst").write_text("universe selection language", encoding="utf-8")

        assert "universe selection language" in " ".join(_collect_text_corpus(tmp_path))

    def test_release_notes_are_not_topic_text(self, tmp_path: Path) -> None:
        """scanpy ships 69 release notes — 69 of its 113 documents — and they are the reason
        its top keywords were `smaller`, `pr`, `func` and two maintainer surnames."""
        (tmp_path / "README.md").write_text("Single-cell analysis.", encoding="utf-8")
        docs = tmp_path / "docs"
        (docs / "release-notes").mkdir(parents=True)
        (docs / "release-notes" / "1.9.0.md").write_text("angerer virshup", encoding="utf-8")
        (docs / "CHANGELOG.md").write_text("bumped versions", encoding="utf-8")
        (docs / "tutorial.md").write_text("clustering neighbors umap", encoding="utf-8")

        corpus = " ".join(_collect_text_corpus(tmp_path))
        assert "clustering neighbors umap" in corpus
        assert "angerer" not in corpus
        assert "bumped versions" not in corpus

    def test_a_release_note_still_counts_as_a_citation(self, tmp_path: Path) -> None:
        """The declared divergence between the two walks over the same roots.

        A release note saying "implements arXiv:2303.14046" is exactly the evidence
        `cited_arxiv_ids_of` wants, and exactly the vocabulary the topic corpus excludes.
        Same files, different question — asserted here so the divergence stays deliberate.
        """
        (tmp_path / "README.md").write_text("A descriptor library.", encoding="utf-8")
        notes = tmp_path / "docs" / "release-notes"
        notes.mkdir(parents=True)
        (notes / "2.0.md").write_text("angerer implemented arXiv:2303.14046", encoding="utf-8")

        assert cited_arxiv_ids_of(tmp_path) == {"2303.14046"}
        assert "angerer" not in " ".join(_collect_text_corpus(tmp_path))


class TestMarkupThatSurvivedTheCleaner:
    def test_reference_style_badges_do_not_become_the_prose(self) -> None:
        """scvi-tools' first 300 README characters were 100% this, and the parenthesised
        patterns never saw it: `![alt](url)` does not match `![alt][ref]`."""
        cleaned = _clean_document(
            "[![Stars][gh-stars-badge]][gh-stars-link]\n"
            "[![Docs][docs-badge]][docs-link]\n\n"
            "scvi-tools is a package for probabilistic modeling.\n\n"
            "[gh-stars-badge]: https://img.shields.io/badge/stars\n"
        )
        assert "probabilistic modeling" in cleaned
        assert "gh-stars-badge" not in cleaned
        assert "shields.io" not in cleaned

    def test_role_names_are_stripped_and_their_targets_survive(self) -> None:
        """`{smaller}`, `{pr}` and `{func}` were scanpy's top three keywords and `mod`
        mdanalysis's fourth. The target is where the API names live, so it stays."""
        cleaned = _clean_document(
            "See {func}`scanpy.pp.pca` and :mod:`MDAnalysis.analysis` for details.\n"
        )
        assert "func" not in cleaned.split()
        assert "mod" not in cleaned.split()
        assert "scanpy.pp.pca" in cleaned
        assert "MDAnalysis.analysis" in cleaned

    def test_a_role_at_line_start_keeps_its_target(self) -> None:
        """`_RST_OPTION_RE` deletes whole lines beginning `:word:`. Stripping the role name
        first is what leaves the API name behind instead of deleting the line, and it is why
        the substitution order in `_clean_document` is load-bearing."""
        assert "pkg.core" in _clean_document(":mod:`pkg.core` is the entry point.\n")

    def test_rst_substitution_definitions_and_references_are_stripped(self) -> None:
        """mdanalysis has 13 `.. |docs| image::` lines and `image` tied for keyword #1."""
        cleaned = _clean_document(
            ".. |docs| image:: https://img.shields.io/badge/docs\n"
            "|docs| |build| MDAnalysis analyses trajectories.\n"
        )
        assert "image" not in cleaned
        assert "MDAnalysis analyses trajectories." in cleaned

    def test_the_cleaner_has_not_grown_into_everything(self) -> None:
        """The paired guard. A cleaner that removes the signal along with the markup passes
        every test above and is useless."""
        cleaned = _clean_document("Install from https://example.org/pkg then run it.\n")
        assert "example.org" not in cleaned
        assert "Install from" in cleaned
        assert "then run it." in cleaned
