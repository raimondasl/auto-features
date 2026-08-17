"""Tests for evals/nerdme_probe.py (P9).

The load-bearing part of this probe is not the extraction -- it is the set of estimators
that replaced the pooled rate its first version reported. C-20 records what a pooled rate
did here: `peft` supplies 23% of all score-3 papers at a 100% hit rate, so the headline was
close to a report about one repository. These tests pin the estimators against synthetic
cells where the right answer is known by construction, so a future edit cannot quietly
restore the pooled reading.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

EVALS = Path(__file__).resolve().parent.parent / "evals"
sys.path.insert(0, str(EVALS))

nerdme = pytest.importorskip("nerdme_probe")


class TestParseEntities:
    def test_plain_json(self):
        got = nerdme._parse_entities('{"entities":[{"span":"FAISS","type":"library"}]}')
        assert got == [{"span": "FAISS", "type": "library"}]

    def test_fenced_json(self):
        raw = '```json\n{"entities":[{"span":"IVF-PQ","type":"method"}]}\n```'
        assert nerdme._parse_entities(raw) == [{"span": "IVF-PQ", "type": "method"}]

    def test_unknown_type_is_dropped(self):
        raw = '{"entities":[{"span":"x","type":"vibe"},{"span":"y","type":"model"}]}'
        assert nerdme._parse_entities(raw) == [{"span": "y", "type": "model"}]

    def test_garbage_returns_empty_rather_than_raising(self):
        assert nerdme._parse_entities("I could not find any entities.") == []

    def test_blank_span_is_dropped(self):
        assert nerdme._parse_entities('{"entities":[{"span":"  ","type":"library"}]}') == []


class TestSpansAsAnchors:
    """The treatment must obey the rules the baseline obeys, or the comparison is rigged."""

    def test_stopword_anchors_are_excluded(self):
        entry = {"entities": [{"span": "pytest", "type": "library"}]}
        assert nerdme._spans_as_anchors(entry) == set()

    def test_short_spans_are_excluded(self):
        entry = {"entities": [{"span": "ab", "type": "library"}]}
        assert nerdme._spans_as_anchors(entry) == set()

    def test_type_filter_selects_one_class(self):
        entry = {
            "entities": [
                {"span": "rocksdb", "type": "library"},
                {"span": "ivf-pq", "type": "method"},
            ]
        }
        assert nerdme._spans_as_anchors(entry, ("method",)) == {"ivf-pq"}


# (hits among score-3, n score-3, hits among below-2, n below-2)
BIG = (10, 10, 0, 100)  # a large case with a perfect gap
SMALL = (1, 1, 0, 1)  # a one-paper case with the same gap and no evidence


class TestEstimators:
    def test_usable_drops_cases_missing_a_stratum(self):
        cells = {"a": (1, 2, 1, 2), "empty3": (0, 0, 1, 5), "emptybelow": (1, 3, 0, 0)}
        assert set(nerdme._usable(cells)) == {"a"}

    def test_case_gap_is_a_percentage_point_difference(self):
        assert nerdme._case_gaps({"a": (1, 2, 1, 4)})["a"] == pytest.approx(25.0)

    def test_macro_gives_every_case_one_vote(self):
        # SMALL rests on a single paper; unweighted macro cannot tell it from BIG.
        assert nerdme._macro({"big": BIG, "small": SMALL}) == pytest.approx(100.0)

    def test_min_n_drops_the_underpowered_case(self):
        # This is the guard that keeps `columnar` (+84.2pt off ONE abstract) from voting
        # as loudly as `storage` (twenty-one papers).
        cells = {"big": BIG, "small": (0, 1, 1, 1)}
        assert nerdme._macro(cells) == pytest.approx(0.0)
        assert nerdme._macro(cells, min_n=5) == pytest.approx(100.0)

    def test_mh_downweights_the_underpowered_case(self):
        # MH weight is n1*n0/(n1+n0): BIG earns ~9.1, SMALL earns 0.5.
        cells = {"big": BIG, "small": (0, 1, 1, 1)}
        assert nerdme._mh(cells) > nerdme._macro(cells)

    def test_pooling_can_invert_what_every_case_shows(self):
        """The C-20/C-21 shape, in miniature.

        Both cases show a NEGATIVE within-case gap, yet one case supplies nearly all the
        score-3 papers at a high hit rate and the other supplies the below-2 papers at a
        low one -- so the pooled rate comes out positive. This is why the probe never
        prints a pooled figure without the per-case columns beside it.
        """
        cells = {"dominant": (90, 100, 10, 10), "other": (0, 1, 5, 100)}
        pooled = 100 * (90 + 0) / 101 - 100 * (10 + 5) / 110
        assert pooled > 0
        assert nerdme._macro(cells) < 0
        assert nerdme._mh(cells) < 0

    def test_sign_counts_cases_not_papers(self):
        assert nerdme._sign({"a": (1, 1, 0, 1), "b": (0, 1, 1, 1), "c": (1, 2, 1, 2)}) == "1/1/1"

    def test_n_positive_reports_breadth(self):
        assert nerdme._n_positive({"a": BIG, "b": (0, 1, 1, 1)}) == (1, 2)

    def test_loo_names_the_load_bearing_case(self):
        cells = {"carrier": BIG, "flat1": (1, 2, 1, 2), "flat2": (1, 2, 1, 2)}
        case, delta = nerdme._loo_worst(cells)
        assert case == "carrier"
        assert delta < 0

    def test_bootstrap_is_deterministic_and_brackets_the_macro(self):
        cells = {"a": (8, 10, 2, 10), "b": (7, 10, 3, 10), "c": (9, 10, 1, 10)}
        first = nerdme._bootstrap_ci(cells)
        assert first == nerdme._bootstrap_ci(cells)
        lo, hi = first
        assert lo <= nerdme._macro(cells) <= hi

    def test_empty_input_does_not_divide_by_zero(self):
        assert nerdme._macro({}) == 0.0
        assert nerdme._mh({}) == 0.0
        assert nerdme._bootstrap_ci({}) == (0.0, 0.0)


class TestReadmeDiscoveryMatchesTheProfiler:
    """The docstring promises this list is asserted rather than left to drift."""

    def test_names_match_the_shipped_profiler_source(self):
        src = (
            Path(__file__).resolve().parent.parent / "src" / "reporadar" / "profiler.py"
        ).read_text(encoding="utf-8")
        for name in nerdme.README_NAMES:
            assert f'"{name}"' in src, (
                f"{name} is in nerdme_probe.README_NAMES but not in profiler.py -- the "
                "probe would read a document the profiler never sees"
            )

    def test_order_matches_the_profilers_preference(self):
        assert nerdme.README_NAMES[0] == "README.md"
        assert "README.rst" in nerdme.README_NAMES, "3 benchmark repos are .rst only"


class TestEmptyExtractionIsRefusedNotReported:
    """The relation_probe lesson: a channel extracted as empty everywhere scores 0% in
    every stratum, which reads as a finding rather than a broken extractor."""

    def test_analyse_refuses_when_no_case_yielded_a_span(self, monkeypatch, tmp_path):
        monkeypatch.setattr(nerdme, "POOL", tmp_path)
        (tmp_path / "fake.json").write_text('{"candidates":[]}', encoding="utf-8")
        monkeypatch.setattr(nerdme, "_repo_terms", lambda case: (set(), set()))
        monkeypatch.setattr(nerdme, "_pool_abstracts", lambda case: {})
        monkeypatch.setattr(nerdme, "_verdicts", lambda case: {})
        with pytest.raises(SystemExit, match="extractor is broken"):
            nerdme.analyse({"fake": {"entities": []}})
