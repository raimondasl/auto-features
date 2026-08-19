"""A non-arXiv paper should be ranked on what it says, not on which adapter found it.

Three defects found by the scientific-software audit (`evals/RESEARCH-scientific-software.md`,
F2/F4/F15) share one shape: every part of the pipeline downstream of collection was written
when arXiv was the only source, and each treats a non-arXiv record as a malformed arXiv one.

* **F4** — the category axis asked "does this paper have categories", and three adapters fill
  that field from a taxonomy that cannot intersect `arxiv.categories`. Same paper, keyword 0.6:
  arXiv 0.733, Semantic Scholar 0.600, bioRxiv/OpenAlex **0.400**.
* **F15** — each adapter minted an id from its own API's handle, so one preprint entered the
  pool as `oa:`, `ss:` and `biorxiv:` and survived every dedup this project has.
* **F2** — the OpenAlex source, whose purpose is literature arXiv does not carry, filtered
  preprints out.

The paired guards matter as much as the fixes: arXiv papers must be untouched, a record with
no DOI must keep its synthetic id, and the ranker must not have grown a rule that calls
everything comparable.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from reporadar.config import QueriesConfig, RankingConfig
from reporadar.paper_id import dedup_id, doi_key, is_arxiv_id
from reporadar.pipeline import KEYWORD_SOURCE_QUERIES
from reporadar.profiler import RepoProfile
from reporadar.ranker import has_comparable_categories, rank_papers, score_paper
from reporadar.sources.biorxiv import _normalize as biorxiv_normalize
from reporadar.sources.openalex import _normalize_paper as openalex_normalize
from reporadar.sources.semantic_scholar import _normalize_paper as s2_normalize

ROOT = Path(__file__).resolve().parents[1]
PREPRINT_DOI = "10.1101/2024.03.15.585278"
TITLE = "A deep learning model for splice site prediction"


def _profile() -> RepoProfile:
    """Weighted so keyword overlap is exactly 0.6 — the audit's worked example."""
    return RepoProfile(
        keywords=[("splice", 0.6), ("absent", 0.4)],
        anchors=[],
        domains=[],
        prose="",
        corpus_phrases=[],
    )


def _paper(**overrides) -> dict:
    base = {
        "arxiv_id": "2401.12345",
        "title": TITLE,
        "abstract": "splice",
        "categories": [],
        "published": "2024-03-15T00:00:00+00:00",
    }
    base.update(overrides)
    return base


def _score(paper: dict) -> float:
    return score_paper(
        paper,
        _profile(),
        RankingConfig(),
        QueriesConfig(),
        ["q-bio.GN"],
        lookback_days=14,
    )["score_total"]


class TestTheCategoryAxisAsksAboutVocabularyNotPresence:
    def test_a_foreign_taxonomy_no_longer_costs_a_third_of_the_score(self) -> None:
        """The measured defect, at the shipped weights and the audit's own numbers.

        `absent_category` exists to decide how a paper with no comparable category signal is
        treated, and three adapters routed around it by filling the field from another
        taxonomy — so which source found a paper moved it by 0.2 for no stated reason.
        """
        on_arxiv = _score(_paper(categories=["q-bio.GN"]))
        from_s2 = _score(_paper(categories=[]))
        from_biorxiv = _score(_paper(categories=["Bioinformatics "]))
        from_openalex = _score(_paper(categories=["Machine Learning"]))

        assert on_arxiv == pytest.approx(0.733, abs=0.005)
        assert from_s2 == pytest.approx(0.600, abs=0.005)
        assert from_biorxiv == pytest.approx(from_s2), "was 0.400 — a taxonomy, not a judgement"
        assert from_openalex == pytest.approx(from_s2)

    @pytest.mark.parametrize(
        ("categories", "comparable"),
        [
            (["cs.LG"], True),
            (["cond-mat.mtrl-sci"], True),
            (["q-bio.QM"], True),
            (["physics.optics"], True),
            (["hep-th"], True),  # an era with no dot at all
            (["cs.CR"], True),  # IACR hardcodes this one, correctly
            (["Machine Learning"], False),  # OpenAlex primary_topic
            (["Bioinformatics "], False),  # bioRxiv subject, trailing space and all
            (["conf/vldb"], False),  # DBLP venue key
            ([], False),
        ],
    )
    def test_the_vocabularies(self, categories: list[str], comparable: bool) -> None:
        assert has_comparable_categories({"categories": categories}) is comparable

    def test_an_arxiv_paper_that_misses_its_target_still_scores_zero(self) -> None:
        """The paired guard, and the reason this is a vocabulary test and not a source test.

        `cs.CL` is a real arXiv category that genuinely does not match a `q-bio.GN` target.
        That is evidence, and it must keep costing the paper — a rule that quietly forgave
        every non-match would pass the test above and destroy the component.
        """
        assert _score(_paper(categories=["cs.CL"])) == pytest.approx(0.400, abs=0.005)

    def test_a_trailing_space_does_not_lose_a_real_match(self) -> None:
        assert _score(_paper(categories=["q-bio.GN "])) == pytest.approx(0.733, abs=0.005)

    def test_the_imputed_mean_is_taken_over_comparable_papers_only(self) -> None:
        """`impute` averages the pool's category scores. Foreign-vocabulary papers score a
        guaranteed 0, so counting them dragged the mean toward zero — the same defect one
        level up, which is why both sites call the same predicate."""
        pool = [
            _paper(arxiv_id="2401.1", categories=["q-bio.GN"]),
            _paper(arxiv_id="2401.2", categories=["q-bio.GN"]),
            _paper(arxiv_id="oa:W1", categories=["Machine Learning"]),
            _paper(arxiv_id="ss:abc", categories=[]),
        ]
        scores = rank_papers(
            pool,
            _profile(),
            RankingConfig(absent_category="impute"),
            QueriesConfig(),
            ["q-bio.GN"],
            lookback_days=14,
        )
        by_id = {s["arxiv_id"]: s["score_total"] for s in scores}
        # Mean over the two comparable papers is 1.0, so both uncategorised papers are
        # imputed 1.0 and land on the arXiv papers' score rather than below them.
        assert by_id["oa:W1"] == pytest.approx(by_id["2401.1"])
        assert by_id["ss:abc"] == pytest.approx(by_id["oa:W1"])


class TestOnePreprintGetsOneId:
    def test_three_adapters_agree_on_the_same_preprint(self) -> None:
        openalex = openalex_normalize(
            {
                "id": "https://openalex.org/W4392847362",
                "doi": f"https://doi.org/{PREPRINT_DOI}",
                "title": TITLE,
                "publication_date": "2024-03-16",
                "primary_topic": {"display_name": "Machine Learning"},
            }
        )
        semantic_scholar = s2_normalize(
            {
                "paperId": "649def34f8be52c8b66281af98ae884c09aef38b",
                # Upper-cased on purpose: DOI names are case-insensitive and the sources
                # disagree in practice, which is the trap `_extract_arxiv_id` already fell
                # into once for arXiv DOIs.
                "externalIds": {"DOI": PREPRINT_DOI.upper()},
                "title": TITLE,
                "year": 2024,
            }
        )
        biorxiv = biorxiv_normalize(
            {
                "doi": PREPRINT_DOI,
                "title": TITLE,
                "authors": "Doe, J.",
                "date": "2024-03-15",
                "abstract": "...",
            },
            "biorxiv",
        )
        ids = {
            openalex["arxiv_id"],
            semantic_scholar["arxiv_id"],
            biorxiv["arxiv_id"],
        }
        assert ids == {f"doi:{PREPRINT_DOI}"}
        assert len({dedup_id(i) for i in ids}) == 1

    def test_an_arxiv_paper_keeps_its_arxiv_id(self) -> None:
        """The whole project is keyed on arXiv ids — the HyDE index, the judge cache, every
        stored score. A DOI id for an arXiv paper would orphan all of it."""
        paper = openalex_normalize(
            {
                "id": "https://openalex.org/W1",
                "doi": "https://doi.org/10.48550/arxiv.2401.12345",
                "title": TITLE,
                "publication_date": "2024-01-01",
            }
        )
        assert paper["arxiv_id"] == "2401.12345"

    def test_a_record_with_no_doi_keeps_its_synthetic_id(self) -> None:
        """The paired guard: DOI-first must not mean DOI-only."""
        paper = openalex_normalize(
            {"id": "https://openalex.org/W99", "title": TITLE, "publication_date": "2024-01-01"}
        )
        assert paper["arxiv_id"] == "oa:W99"

        s2 = s2_normalize({"paperId": "deadbeef", "externalIds": {}, "title": TITLE, "year": 2024})
        assert s2["arxiv_id"] == "ss:deadbeef"

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("10.1101/2024.03.15.585278", "doi:10.1101/2024.03.15.585278"),
            ("https://doi.org/10.1101/ABC", "doi:10.1101/abc"),
            ("http://dx.doi.org/10.1234/x", "doi:10.1234/x"),
            ("doi:10.1234/x", "doi:10.1234/x"),
            ("  10.1234/x  ", "doi:10.1234/x"),
            ("not-a-doi", ""),
            ("10.1234", ""),  # a prefix with no suffix is not a DOI
            ("", ""),
            (None, ""),
        ],
    )
    def test_doi_normalisation(self, raw: str | None, expected: str) -> None:
        assert doi_key(raw) == expected

    def test_dedup_id_leaves_a_doi_id_alone(self) -> None:
        """`dedup_id` version-strips arXiv ids. A DOI can end in anything, including
        something version-shaped, and must pass through untouched."""
        assert dedup_id("doi:10.1234/paper.v2") == "doi:10.1234/paper.v2"


class TestTheKeywordSourcesSeeTheQueriesTheRepositoryProduces:
    def test_the_cap_covers_every_benchmark_repository(self) -> None:
        """It was 5, which withheld 50 of the 175 queries the 25 benchmark repositories
        build (28.6%); 8 is the most any of them produces."""
        assert KEYWORD_SOURCE_QUERIES >= 8

    @pytest.mark.parametrize("rel", ["evals/openalex_yield.py", "evals/s2_yield.py"])
    def test_the_stage_one_probes_slice_by_the_same_constant(self, rel: str) -> None:
        """Both probes measure "can this channel reach a top-10", and both carried their own
        `queries[:5]`. A probe that sends fewer queries than the product understates the
        channel it exists to judge — the C-9/C-12 shape, in the measuring instrument.

        Importing the constant is the guard: there is no second number to drift.
        """
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert "queries[:5]" not in text, f"{rel} kept its own copy of the cap"
        assert "KEYWORD_SOURCE_QUERIES" in text


class TestArxivnessIsTestedPositively:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("2401.12345", True),
            ("2401.12345v2", True),
            ("cond-mat.supr-con/9501001", True),
            ("solv-int/9801001v1", True),
            ("doi:10.1101/x", False),
            ("ss:abc", False),
            ("oa:W1", False),
            ("dblp:conf/vldb/X", False),
            ("", False),
        ],
    )
    def test_is_arxiv_id(self, value: str, expected: bool) -> None:
        assert is_arxiv_id(value) is expected

    def test_semantic_scholar_does_not_fabricate_an_arxiv_link_for_a_doi(self) -> None:
        """The bug the DOI id would have introduced. That adapter built an abstract URL for
        any id not starting with `ss:` — correct only while `ss:` was the sole alternative,
        and it would have produced `arxiv.org/abs/doi:10.1101/...`."""
        paper = s2_normalize(
            {
                "paperId": "abc",
                "externalIds": {"DOI": PREPRINT_DOI},
                "title": TITLE,
                "year": 2024,
                "url": "",
            }
        )
        assert paper["arxiv_id"].startswith("doi:")
        assert "arxiv.org" not in paper["url"]
