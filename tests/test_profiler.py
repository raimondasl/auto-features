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


class TestCorpusPhrases:
    """`RepoProfile.corpus_phrases` — the phrases the repo actually contains.

    Query building pairs adjacent keywords into quoted phrases, and nothing ever checked
    that the two words belong together. Only the profiler can check: the corpus does not
    survive profiling. So the field is computed here, and these tests pin the property
    that makes it worth having — a pair that occurs is kept, a pair that does not is not.
    """

    def _repo(self, tmp_path: Path, readme: str) -> Path:
        repo = tmp_path / "proj"
        repo.mkdir()
        (repo / "README.md").write_text(readme, encoding="utf-8")
        return repo

    def test_records_pairs_that_occur_and_omits_pairs_that_do_not(self, tmp_path: Path) -> None:
        repo = self._repo(
            tmp_path,
            "# duckdb\n\nduckdb sql engine for analytics. duckdb sql runs fast.\n"
            "Vectorized execution over columnar storage, repeatedly: vectorized execution.\n",
        )
        phrases = set(profile_repo(repo).corpus_phrases)
        assert "duckdb sql" in phrases, phrases
        # "sql vectorized" never appears adjacently anywhere in that README.
        assert "sql vectorized" not in phrases, phrases

    def test_adjacency_is_literal_not_stopword_skipping(self, tmp_path: Path) -> None:
        """ "use of page" must NOT license the phrase query `"use page"`.

        Every search API this reaches matches a quoted phrase literally, so skipping over
        stop words would re-admit exactly the queries this exists to reject.
        """
        repo = self._repo(
            tmp_path,
            "# tool\n\nMake use of page tables for caching. use of page tables again.\n"
            "page tables are the unit. use it wisely, page it out.\n",
        )
        phrases = set(profile_repo(repo).corpus_phrases)
        assert "use page" not in phrases, phrases

    def test_empty_repo_yields_no_phrases_rather_than_failing(self, tmp_path: Path) -> None:
        repo = tmp_path / "bare"
        repo.mkdir()
        assert profile_repo(repo).corpus_phrases == []


class TestRepoProse:
    """`RepoProfile.prose` — the repo described in its own words.

    Guarded closely because getting this subtly wrong is not hypothetical: an earlier
    experiment read `_collect_text_corpus(repo)[0]`, believed it was the README, and
    recorded a result under that name. On 11 of 12 benchmark repos element 0 is the
    packaging one-liner (23-230 chars), so the experiment tested something else entirely.
    """

    def _repo(self, tmp_path: Path, readme: str | None, pyproject: str | None) -> Path:
        repo = tmp_path / "r"
        repo.mkdir(exist_ok=True)
        if readme is not None:
            (repo / "README.md").write_text(readme, encoding="utf-8")
        if pyproject is not None:
            (repo / "pyproject.toml").write_text(pyproject, encoding="utf-8")
        return repo

    _PYPROJECT = '[project]\nname = "thing"\ndescription = "A one line tagline."\n'

    def test_prefers_the_readme_over_the_packaging_description(self, tmp_path: Path) -> None:
        """The exact bug that invalidated the earlier measurement.

        Both documents exist and the README is the one that says what the project is for;
        taking the metadata instead yields a tagline and looks like it worked.
        """
        repo = self._repo(
            tmp_path,
            "# Thing\n\nThing performs late-interaction retrieval over BERT embeddings.\n",
            self._PYPROJECT,
        )
        prose = profile_repo(repo).prose
        assert "late-interaction retrieval" in prose
        assert "A one line tagline." not in prose

    def test_falls_back_to_the_packaging_description_with_no_readme(self, tmp_path: Path) -> None:
        repo = self._repo(tmp_path, None, self._PYPROJECT)
        assert "A one line tagline." in profile_repo(repo).prose

    def test_is_empty_when_the_repo_describes_itself_nowhere(self, tmp_path: Path) -> None:
        assert profile_repo(self._repo(tmp_path, None, None)).prose == ""

    def test_a_whitespace_only_readme_does_not_shadow_the_metadata(self, tmp_path: Path) -> None:
        """`_read_text_file` returns the file, so a blank README would otherwise win."""
        repo = self._repo(tmp_path, "\n\n   \n", self._PYPROJECT)
        assert "A one line tagline." in profile_repo(repo).prose

    def test_respects_the_configured_budget(self, tmp_path: Path) -> None:
        """A 41k-character README must not reach a prompt because nobody truncated it."""
        from reporadar.config import ProfilerConfig

        repo = self._repo(tmp_path, "# T\n\n" + ("retrieval " * 5000), None)
        prose = profile_repo(repo, profiler_cfg=ProfilerConfig(prose_chars=120)).prose
        assert len(prose) == 120

    @pytest.mark.parametrize("budget", [0, -1])
    def test_a_non_positive_budget_sends_no_prose_at_all(self, tmp_path: Path, budget: int) -> None:
        """The privacy opt-out: on a proprietary codebase the README is the disclosure.

        `rr audit` tells users `profiler.prose_chars: 0` withholds it, so this is the
        assertion that keeps that sentence true.

        `-1` is not a hypothetical. Slicing is what enforces the budget, and `text[:0]`
        is empty while `text[:-1]` is the *whole README bar one character* — so a typo'd
        or negative setting would silently invert the opt-out into a full disclosure.
        That inversion is the only thing the explicit guard buys, which is why it is
        tested here rather than left to the slice.
        """
        from reporadar.config import ProfilerConfig
        from reporadar.triage import build_triage_prompt

        repo = self._repo(tmp_path, "# T\n\nSecret internal system.\n", self._PYPROJECT)
        profile = profile_repo(repo, profiler_cfg=ProfilerConfig(prose_chars=budget))
        assert profile.prose == ""
        assert "Secret internal system" not in build_triage_prompt(
            {"title": "t", "abstract": "a"}, profile
        )

    def test_the_prose_reaches_the_triage_prompt(self, tmp_path: Path) -> None:
        """End-to-end, because the field existing is not the point — being sent is."""
        from reporadar.triage import build_triage_prompt

        repo = self._repo(tmp_path, "# Thing\n\nThing does late-interaction retrieval.\n", None)
        prompt = build_triage_prompt({"title": "P", "abstract": "a"}, profile_repo(repo))
        assert "late-interaction retrieval" in prompt

    def test_the_default_budget_is_the_one_that_was_measured(self) -> None:
        """300 is the best-measured budget, NOT a demonstrated optimum.

        On 602 labelled papers: +95 net@2 against +73 for no prose (+22, CI [+4, +41]),
        with 2000 at +86 and 6000 at +89. Only "some prose beats none" is established —
        300 vs 2000 is +9 at P = 0.108, so this is the argmax of noisy arms. The test
        pins the number so a change is deliberate and re-measured
        (`evals/diagnose_triage.py --repo-context prose --prose-chars N`), not so anyone
        believes 300 is optimal.
        """
        from reporadar.config import ProfilerConfig

        assert ProfilerConfig().prose_chars == 300

    def test_the_undeclared_fallback_matches_the_config_default(self, tmp_path: Path) -> None:
        """`profile_repo` accepts any object as profiler_cfg and falls back via getattr.

        A fallback that drifts from the dataclass default means callers passing a bare
        namespace silently get a different budget from callers passing a real config —
        the same class of split-brain that put an undeclared 2000 in two places.
        """
        from types import SimpleNamespace

        from reporadar.config import ProfilerConfig

        repo = self._repo(tmp_path, "# T\n\n" + ("x " * 4000), None)
        bare = profile_repo(repo, profiler_cfg=SimpleNamespace(scan_source=False))
        typed = profile_repo(repo, profiler_cfg=ProfilerConfig())
        assert len(bare.prose) == len(typed.prose) == ProfilerConfig().prose_chars


class TestCitedArxivIds:
    """What the repository itself cites -- the signal that keeps its own paper out of Top Picks.

    Measured on six scientific repositories: five of them had the repo's OWN publication as a
    gate-score-3 Top Pick, and a neutral judge scored every one of them 1 ("the method this
    repository already implements").
    """

    def test_reads_readme_citation_and_docs(self, tmp_path: Path) -> None:
        from reporadar.profiler import cited_arxiv_ids_of

        (tmp_path / "README.md").write_text(
            "See our paper: https://arxiv.org/abs/1708.01492v5", encoding="utf-8"
        )
        (tmp_path / "CITATION.cff").write_text("identifiers: arXiv:2302.14231", encoding="utf-8")
        docs = tmp_path / "docs" / "src"
        docs.mkdir(parents=True)
        (docs / "references.bib").write_text(
            "@article{de2016, eprint = {1601.04077}}", encoding="utf-8"
        )
        (tmp_path / "docs" / "guide.rst").write_text("cf. arXiv:2303.14046", encoding="utf-8")

        assert cited_arxiv_ids_of(tmp_path) == {
            "1708.01492",
            "2302.14231",
            "1601.04077",
            "2303.14046",
        }

    def test_rejects_dois_and_version_strings(self, tmp_path: Path) -> None:
        """The month is validated, and that is what separates a citation from a DOI.

        Every string below was found in a real repository's README by the unvalidated
        pattern: `2019.10694` is a Comput. Phys. Commun. DOI fragment in dscribe's, and the
        other three are MACE's. A month of 19, 29, 84 or 42 is not an arXiv id.
        """
        from reporadar.profiler import cited_arxiv_ids_of

        (tmp_path / "README.md").write_text(
            "doi 10.1016/j.cpc.2019.10694, refs 1029.28096 1484.11876 2042.03300 7022.2013",
            encoding="utf-8",
        )
        assert cited_arxiv_ids_of(tmp_path) == frozenset()

    def test_old_style_ids_survive(self, tmp_path: Path) -> None:
        from reporadar.profiler import cited_arxiv_ids_of

        (tmp_path / "README").write_text("cond-mat/0501001 and math.GT/0309136v2", encoding="utf-8")
        assert cited_arxiv_ids_of(tmp_path) == {"cond-mat/0501001", "math.GT/0309136"}

    def test_profile_repo_exposes_them(self, tmp_path: Path) -> None:
        from reporadar.profiler import profile_repo

        (tmp_path / "README.md").write_text(
            "A tool for sequence alignment. Paper: arXiv:1708.01492", encoding="utf-8"
        )
        assert profile_repo(tmp_path).cited_arxiv_ids == {"1708.01492"}

    def test_no_citations_is_an_empty_set_not_a_crash(self, tmp_path: Path) -> None:
        from reporadar.profiler import cited_arxiv_ids_of, profile_repo

        (tmp_path / "README.md").write_text("No papers here.", encoding="utf-8")
        assert cited_arxiv_ids_of(tmp_path) == frozenset()
        assert profile_repo(tmp_path).cited_arxiv_ids == frozenset()


class TestNonPythonManifests:
    """R, Julia and Rust dependency manifests, read only when nothing else declared anything.

    §10.2 considered these and dropped them on a measurement: "Zero of the 19 clones has any
    of them at the repository root." The arithmetic was right and the *scope* was the problem
    — the 19 are Python and ML repositories. §33 profiled the population §14.2 had excluded
    and found 8 of 9 with zero anchors, the only exception being the only one shipping a
    Python manifest.
    """

    def test_r_description_yields_its_analysis_stack(self, tmp_path: Path) -> None:
        (tmp_path / "DESCRIPTION").write_text(
            "Package: Seurat\n"
            "Depends: R (>= 4.1.0), methods, SeuratObject (>= 5.0.2)\n"
            "Imports: cluster, cowplot,\n"
            "    Matrix (>= 1.5-0)\n"
            "Suggests: DESeq2, SingleCellExperiment\n"
            "License: MIT\n",
            encoding="utf-8",
        )
        anchors = _extract_anchors(tmp_path)
        assert "seuratobject" in anchors
        assert "matrix" in anchors, "a continuation line is part of the field above it"
        assert "deseq2" in anchors, "R's Suggests is the analysis stack, not a test toolchain"
        assert "methods" in anchors

    def test_r_description_drops_the_language_itself(self, tmp_path: Path) -> None:
        """Every R package depends on R, so it is the language and not a topic — and a bare
        `r` would reach PACKAGE_DOMAIN_MAP and the query builder as one ambiguous letter."""
        (tmp_path / "DESCRIPTION").write_text(
            "Package: x\nDepends: R (>= 4.1.0), cluster\n", encoding="utf-8"
        )
        assert "r" not in _extract_anchors(tmp_path)

    def test_julia_project_toml(self, tmp_path: Path) -> None:
        (tmp_path / "Project.toml").write_text(
            'name = "DifferentialEquations"\n\n'
            "[deps]\n"
            'OrdinaryDiffEq = "1dea7af3-3e70-54e6-95c3-0bf5283fa5ed"\n'
            'StochasticDiffEq = "789caeaf-c7a9-5a7d-9973-96adeb23e2a0"\n',
            encoding="utf-8",
        )
        anchors = _extract_anchors(tmp_path)
        assert "ordinarydiffeq" in anchors
        assert "stochasticdiffeq" in anchors
        assert "name" not in anchors, "only [deps] keys, not top-level metadata"

    def test_rust_cargo_toml(self, tmp_path: Path) -> None:
        (tmp_path / "Cargo.toml").write_text(
            '[dependencies]\nmemchr = "2.7"\nbytes = { version = "1" }\n'
            '[dev-dependencies]\ncriterion = "0.5"\n',
            encoding="utf-8",
        )
        anchors = _extract_anchors(tmp_path)
        assert {"memchr", "bytes", "criterion"} <= set(anchors)

    def test_rust_workspace_root_declares_no_dependencies_of_its_own(self, tmp_path: Path) -> None:
        """noodles — nine bioinformatics format crates — yielded zero anchors from a
        Cargo.toml listing its whole stack under `[workspace.dependencies]`."""
        (tmp_path / "Cargo.toml").write_text(
            '[workspace]\nmembers = ["noodles-bam", "noodles-sam"]\n\n'
            "[workspace.dependencies]\n"
            'noodles-bgzf = "0.30"\nlibdeflater = "1.20"\n',
            encoding="utf-8",
        )
        anchors = _extract_anchors(tmp_path)
        assert {"noodles-bgzf", "libdeflater"} <= set(anchors)

    def test_an_unparseable_manifest_does_not_fail_the_profile(self, tmp_path: Path) -> None:
        (tmp_path / "Cargo.toml").write_text("this is not [ valid toml", encoding="utf-8")
        assert _extract_anchors(tmp_path) == []

    def test_they_do_not_fire_when_a_python_manifest_declared_something(
        self, tmp_path: Path
    ) -> None:
        """The gate that bounds the blast radius, and the whole reason this is shippable.

        Measured before it was added: reading these unconditionally moves five benchmark
        cases, two of them hard (`vectordb` 0 -> 128 anchors, `linter` 0 -> 172) and one into
        a new anchor-bigram keyword. §10.4's debt — a profile change invalidates every
        published pool — is not worth re-opening for repositories that already describe
        themselves. Same condition and same reasoning as `_NESTED_PACKAGE_DIRS`.
        """
        (tmp_path / "requirements.txt").write_text("torch>=2.1\n", encoding="utf-8")
        (tmp_path / "Cargo.toml").write_text('[dependencies]\nmemchr = "2.7"\n', encoding="utf-8")
        anchors = _extract_anchors(tmp_path)
        assert anchors == ["torch"], "a repo that already profiles must be byte-identical"


class TestProseAnchor:
    """§42: where the 300-char window starts, and the condition that keeps it small.

    `bio-align` scores exactly +0.0 net@2 because its window lands on a phishing warning and a
    build recipe while "Minimap2 is a versatile sequence alignment program" sits at line 59
    (§41). `prose_anchor="self_description"` re-anchors on that sentence — but only when it
    falls outside the window, because re-anchoring unconditionally moves 17 of 37 benchmark
    repositories, most by a handful of characters, for nothing.
    """

    def _repo(self, tmp_path: Path, readme: str, name: str = "minimap2") -> Path:
        repo = tmp_path / name
        repo.mkdir()
        (repo / "README.md").write_text(readme, encoding="utf-8")
        return repo

    def _prose(self, repo: Path, anchor: str) -> str:
        from reporadar.config import ProfilerConfig

        return profile_repo(
            repo, profiler_cfg=ProfilerConfig(prose_chars=300, prose_anchor=anchor)
        ).prose

    def test_the_shipped_default_is_the_prefix(self, tmp_path: Path) -> None:
        repo = self._repo(tmp_path, "Getting started. " + "x " * 200 + "minimap2 is a mapper.")
        assert self._prose(repo, "start").startswith("Getting started.")

    def test_it_re_anchors_when_the_description_is_out_of_view(self, tmp_path: Path) -> None:
        """The bio-align shape: junk first, the real sentence far past the budget."""
        repo = self._repo(
            tmp_path,
            "ALERT: phishing site. Do not use it. " + "filler " * 100 + "minimap2 is a versatile "
            "sequence alignment program that aligns DNA sequences.",
        )
        assert self._prose(repo, "self_description").startswith("minimap2 is a versatile")

    def test_it_leaves_a_description_that_is_already_in_view_alone(self, tmp_path: Path) -> None:
        """The condition that keeps this a 3-case change rather than a 20-case one."""
        readme = "# minimap2\n\nminimap2 is a versatile mapper. " + "detail " * 200
        repo = self._repo(tmp_path, readme)
        assert self._prose(repo, "self_description") == self._prose(repo, "start")

    def test_no_such_sentence_leaves_the_prose_untouched(self, tmp_path: Path) -> None:
        """17 of 37 benchmark repositories are in this case and must not move."""
        readme = "Quickly search and compare data sets. " + "detail " * 200
        repo = self._repo(tmp_path, readme)
        assert self._prose(repo, "self_description") == self._prose(repo, "start")

    def test_the_name_must_be_the_subject(self, tmp_path: Path) -> None:
        """The ruff false positive (§42.1).

        "...getting started with ruff, the default rule set IS A great place to start" matched a
        pattern that allowed 80 characters of slack, and would have replaced ruff's real
        description with a note from its configuration section.
        """
        readme = (
            "An extremely fast Python linter, written in Rust. "
            + "detail " * 100
            + "If you're just getting started with ruff, the default rule set is a great place "
            "to start."
        )
        repo = self._repo(tmp_path, readme, name="ruff")
        assert self._prose(repo, "self_description") == self._prose(repo, "start")

    def test_a_parenthetical_gloss_is_allowed(self, tmp_path: Path) -> None:
        """scvi-tools introduces itself that way, and it is a real self-description."""
        readme = (
            "Badges and links. "
            + "filler " * 100
            + (
                "scvi-tools (single-cell variational inference tools) is a package for "
                "probabilistic modeling."
            )
        )
        repo = self._repo(tmp_path, readme, name="scvi-tools")
        assert self._prose(repo, "self_description").startswith("scvi-tools (single-cell")

    def test_the_name_comes_from_git_config_when_the_directory_disagrees(
        self, tmp_path: Path
    ) -> None:
        """The benchmark names its clones for the CASE, not the project.

        `evals/.work/bio-align` holds minimap2. A directory-only rule would never fire on the
        population that measures it — §21.0's shape, where a flag existed and the dispatch
        behind it did not.
        """
        repo = self._repo(
            tmp_path,
            "ALERT: phishing. " + "filler " * 100 + "minimap2 is a versatile mapper for DNA.",
            name="bio-align",
        )
        (repo / ".git").mkdir()
        (repo / ".git" / "config").write_text(
            '[remote "origin"]\n\turl = https://github.com/lh3/minimap2.git\n', encoding="utf-8"
        )
        assert self._prose(repo, "self_description").startswith("minimap2 is a versatile")

    def test_an_unreadable_git_config_falls_back_rather_than_raising(self, tmp_path: Path) -> None:
        readme = "ALERT. " + "filler " * 100 + "minimap2 is a versatile mapper for DNA."
        repo = self._repo(tmp_path, readme)
        (repo / ".git").mkdir()
        (repo / ".git" / "config").write_text("not an ini file at all", encoding="utf-8")
        assert self._prose(repo, "self_description").startswith("minimap2 is a versatile")

    def test_prose_chars_zero_still_withholds_everything(self, tmp_path: Path) -> None:
        """The privacy contract outranks the anchor: 0 means send no prose at all."""
        from reporadar.config import ProfilerConfig

        repo = self._repo(tmp_path, "filler " * 100 + "minimap2 is a versatile mapper.")
        assert (
            profile_repo(
                repo,
                profiler_cfg=ProfilerConfig(prose_chars=0, prose_anchor="self_description"),
            ).prose
            == ""
        )
