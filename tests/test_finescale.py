"""Tests for the fine-scale actionability rescore.

The stage is a calibrated probability bolted onto a near-binary gate, so most of what
can go wrong is silent: a prompt that drifts from the one the map was fitted against, a
threshold that stops being applied in one of six output formats, a failed API call that
reads as a low score. Each of those gets a test here, because none of them would show up
as a crash.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from reporadar import finescale
from reporadar.digest import categorize_papers
from reporadar.llm_client import LLMError, top_logprobs
from reporadar.store import PaperStore
from reporadar.triage import build_triage_prompt, repo_context_block


class _Profile:
    anchors = ["torch", "numpy"]
    domains = ["deep learning"]
    keywords = [("vector search", 1.0), ("hnsw", 0.5)]
    prose = "A vector database."


class TestTheProbabilityMap:
    def test_it_is_monotone(self) -> None:
        ps = [finescale.probability(x) for x in range(10)]
        assert ps == sorted(ps)

    def test_it_crosses_the_show_threshold_inside_the_top_band(self) -> None:
        """The map must put the 2/3 breakeven where the rubric's "plausibly integrable"
        band ends. A map crossing at 3 or at 9 would be calibrated to a different
        question than the one the prompt asks."""
        crossing = (math.log(2.0) - finescale.INTERCEPT) / finescale.SLOPE
        assert 6.0 < crossing < 7.5
        assert finescale.probability(crossing) == pytest.approx(finescale.SHOW_THRESHOLD)

    def test_the_threshold_is_the_metrics_breakeven_not_a_tuned_number(self) -> None:
        """net@2 pays 3p-2 per shown paper, so showing is worth it exactly above 2/3."""
        assert pytest.approx(2.0 / 3.0) == finescale.SHOW_THRESHOLD

    def test_the_frozen_coefficients_are_the_fitted_ones(self) -> None:
        """Pinned so a later 'cleanup' cannot round them into a different calibration."""
        assert pytest.approx(0.9674915191842209) == finescale.SLOPE
        assert pytest.approx(-5.809648743203093) == finescale.INTERCEPT


class TestDigitExpectation:
    def test_it_renormalizes_over_digit_tokens_only(self) -> None:
        got = finescale.digit_expectation([("7", 0.6), ("8", 0.3), (" the", 0.1)])
        assert got == pytest.approx((7 * 0.6 + 8 * 0.3) / 0.9)

    def test_a_stray_token_does_not_drag_the_score_down(self) -> None:
        """Counting non-digits in the denominator would bias every score toward 0 —
        which on this scale means "unrelated", the exact opposite of "unparseable"."""
        clean = finescale.digit_expectation([("8", 0.9), ("7", 0.1)])
        noisy = finescale.digit_expectation([("8", 0.45), ("7", 0.05), ("Score", 0.5)])
        assert clean == pytest.approx(noisy)

    def test_whitespace_padded_digits_count(self) -> None:
        assert finescale.digit_expectation([(" 5", 1.0)]) == pytest.approx(5.0)

    def test_no_digit_is_none_never_zero(self) -> None:
        """None means "unknown". 0.0 would mean "definitely useless" and would sail
        through the pipeline as a confident rejection."""
        assert finescale.digit_expectation([("Sorry", 0.9), ("I", 0.1)]) is None
        assert finescale.digit_expectation([]) is None

    def test_two_digit_tokens_of_the_same_value_are_summed(self) -> None:
        assert finescale.digit_expectation([("4", 0.5), (" 4", 0.5)]) == pytest.approx(4.0)


class TestThePromptIsTheOneTheMapWasFittedAgainst:
    """The frozen logistic is only valid for the prompt it was fitted with. These pin the
    two halves of that prompt; changing either without refitting silently decalibrates
    the threshold, and nothing else in the system would notice."""

    def test_the_rubric_bands_span_the_full_scale(self) -> None:
        for band in ("0-1 =", "2-3 =", "4-6 =", "7-9 ="):
            assert band in finescale.RUBRIC

    def test_the_prompt_asks_for_a_bare_digit(self) -> None:
        prompt = finescale.build_prompt({"title": "T", "abstract": "A"}, _Profile())
        assert prompt.rstrip().endswith("Respond with ONLY a single digit 0-9.")

    def test_it_matches_the_benchmark_prompt_byte_for_byte(self) -> None:
        """`evals/exp_finescale.py` produced the 219 scores the map was fitted on."""
        evals = Path(__file__).resolve().parent.parent / "evals"
        if str(evals) not in sys.path:
            sys.path.insert(0, str(evals))
        spec = importlib.util.spec_from_file_location("exp_finescale", evals / "exp_finescale.py")
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        sys.modules["exp_finescale"] = mod
        spec.loader.exec_module(mod)
        expected = mod.SCALE_PROMPT.format(
            repo=repo_context_block(_Profile()), title="T", abstract="A"
        )
        assert finescale.build_prompt({"title": "T", "abstract": "A"}, _Profile()) == expected

    def test_the_abstract_is_truncated_the_same_way_as_the_gate(self) -> None:
        long = {"title": "T", "abstract": "x" * 5000}
        assert finescale.build_prompt(long, _Profile()).count("x") == 1500


class TestTheSharedRepoBlock:
    def test_the_gate_prompt_is_unchanged_by_the_extraction(self) -> None:
        """`repo_context_block` was factored out of `build_triage_prompt`; the 0-3 gate's
        prompt is what every shipped benchmark number was measured with, so it must come
        out identical."""
        prompt = build_triage_prompt({"title": "T", "abstract": "A"}, _Profile())
        assert "Dependencies/libraries: torch, numpy\n" in prompt
        assert "Domains: deep learning\n" in prompt
        assert "Key topics: vector search, hnsw\n" in prompt
        assert "What this project is, in its own words:\nA vector database.\n" in prompt

    def test_a_repo_with_no_prose_omits_the_section_entirely(self) -> None:
        class Bare:
            anchors: list[str] = []
            domains: list[str] = []
            keywords: list[tuple[str, float]] = []
            prose = ""

        block = repo_context_block(Bare())
        assert "in its own words" not in block
        assert "Dependencies/libraries: none" in block
        assert "Key topics: n/a" in block

    def test_a_summary_supersedes_the_prose(self) -> None:
        summary = SimpleNamespace(as_prompt_block=lambda: "Summarised description.")
        block = repo_context_block(_Profile(), summary)
        assert "Summarised description." in block
        assert "A vector database." not in block


def _row() -> dict:
    """The minimum a paper row needs to reach `get_scores_for_run`."""
    return {
        "arxiv_id": "a",
        "title": "T",
        "authors": [],
        "abstract": "x",
        "categories": [],
        "published": "2026-01-01T00:00:00+00:00",
        "url": "https://arxiv.org/abs/a",
    }


def _paper(pid: str, llm: int, p: float | None = None, score: float = 0.9) -> dict:
    row = {"arxiv_id": pid, "title": pid, "llm_score": llm, "score_total": score}
    if p is not None:
        row["finescale_p"] = p
    return row


class TestTheDigestGate:
    def test_a_band_paper_below_the_bar_drops_to_maybe(self) -> None:
        top, maybe, _ = categorize_papers(
            [_paper("low", 2, 0.20)], triage_threshold=2, finescale_threshold=2 / 3
        )
        assert top == []
        assert [p["arxiv_id"] for p in maybe] == ["low"]

    def test_a_band_paper_above_the_bar_stays_a_top_pick(self) -> None:
        top, _, _ = categorize_papers(
            [_paper("high", 2, 0.90)], triage_threshold=2, finescale_threshold=2 / 3
        )
        assert [p["arxiv_id"] for p in top] == ["high"]

    def test_a_paper_above_the_band_is_never_re_gated(self) -> None:
        """Score-3 papers were reliable in the benchmark; only the score-2 band was not.
        Re-gating them would throw away the +10 cases to fix the -5 ones."""
        top, _, _ = categorize_papers(
            [_paper("three", 3, 0.01)], triage_threshold=2, finescale_threshold=2 / 3
        )
        assert [p["arxiv_id"] for p in top] == ["three"]

    def test_an_unscored_band_paper_does_not_make_top_picks(self) -> None:
        """The measured policy: unproven is not shown. The caller is responsible for not
        passing a threshold when the whole stage failed — see enough_scored."""
        top, maybe, _ = categorize_papers(
            [_paper("unscored", 2)], triage_threshold=2, finescale_threshold=2 / 3
        )
        assert top == []
        assert [p["arxiv_id"] for p in maybe] == ["unscored"]

    def test_without_a_threshold_nothing_changes(self) -> None:
        papers = [_paper("a", 2, 0.01), _paper("b", 2), _paper("c", 3)]
        top, _, _ = categorize_papers(papers, triage_threshold=2)
        assert [p["arxiv_id"] for p in top] == ["a", "b", "c"]

    def test_the_bar_is_inclusive_at_exactly_the_threshold(self) -> None:
        top, _, _ = categorize_papers(
            [_paper("edge", 2, 2 / 3)], triage_threshold=2, finescale_threshold=2 / 3
        )
        assert len(top) == 1

    def test_papers_below_the_band_are_untouched(self) -> None:
        _, maybe, muted = categorize_papers(
            [_paper("one", 1, 0.99), _paper("zero", 0, 0.99)],
            triage_threshold=2,
            finescale_threshold=2 / 3,
        )
        assert [p["arxiv_id"] for p in maybe] == ["one"]
        assert [p["arxiv_id"] for p in muted] == ["zero"]


class TestEveryOutputFormatHonoursTheGate:
    """The parameter has to be threaded through five generators, `write_digest` and the
    archive. This repo has already shipped a bug of exactly that shape (a config object
    omitted at three of four call sites), and the failure is silent: one format quietly
    tiers differently from the rest. So assert it on all of them at once, through the
    real store, rather than trusting the plumbing.
    """

    @pytest.fixture()
    def store(self, tmp_path: Path) -> PaperStore:
        s = PaperStore(tmp_path / "db.sqlite")
        run_id = s.record_run([], 2, 0)
        papers = [
            {
                "arxiv_id": pid,
                "title": f"Paper {pid}",
                "authors": ["A"],
                "abstract": "x",
                "categories": ["cs.LG"],
                "published": "2026-01-01T00:00:00+00:00",
                "url": f"https://arxiv.org/abs/{pid}",
            }
            for pid in ("keep", "drop")
        ]
        s.upsert_papers(papers)
        s.save_scores(
            run_id,
            [
                {
                    "arxiv_id": pid,
                    "score_total": total,
                    # The templates format these with `%`, so a None reaches Jinja as a
                    # TypeError rather than a blank — populate what a real run would.
                    "keyword_score": total,
                    "category_score": 0.5,
                    "recency_score": 0.1,
                    "matched_query": "q",
                }
                for pid, total in (("keep", 0.9), ("drop", 0.8))
            ],
        )
        s.save_llm_scores(
            run_id,
            {
                "keep": {"llm_score": 2, "llm_reason": ""},
                "drop": {"llm_score": 2, "llm_reason": ""},
            },
        )
        s.save_finescale_scores(
            run_id,
            {
                "keep": {"finescale": 8.0, "finescale_p": 0.95},
                "drop": {"finescale": 3.0, "finescale_p": 0.05},
            },
        )
        s.run_id = run_id  # type: ignore[attr-defined]
        yield s
        s.close()

    def _render(self, store: PaperStore, tmp_path: Path, fmt: str, threshold: float | None) -> str:
        from reporadar.digest import write_digest

        out, _ = write_digest(
            store,
            store.run_id,  # type: ignore[attr-defined]
            tmp_path / f"d_{threshold}.{fmt}",
            fmt=fmt,
            triage_threshold=2,
            finescale_threshold=threshold,
        )
        return out.read_text(encoding="utf-8")

    @pytest.mark.parametrize("fmt", ["md", "html", "json", "csv"])
    def test_the_low_probability_paper_leaves_top_picks(
        self, store: PaperStore, tmp_path: Path, fmt: str
    ) -> None:
        ungated = self._render(store, tmp_path, fmt, None)
        gated = self._render(store, tmp_path, fmt, 2 / 3)
        assert "keep" in ungated and "drop" in ungated
        assert gated != ungated, f"{fmt} ignores finescale_threshold"

    def test_rss_is_deliberately_unaffected(self, store: PaperStore, tmp_path: Path) -> None:
        """RSS emits ``top_picks + maybe_relevant + muted`` as one flat feed, so it
        carries no tier distinction for the gate to change. Asserted rather than skipped
        so that if RSS ever grows tiers, this test fails and gets revisited."""
        gated = self._render(store, tmp_path, "rss", 2 / 3)
        assert "keep" in gated and "drop" in gated

    def test_the_archive_manifest_count_reflects_the_gate(
        self, store: PaperStore, tmp_path: Path
    ) -> None:
        from reporadar.archive import archive_digest

        archive_digest(
            store,
            store.run_id,  # type: ignore[attr-defined]
            tmp_path / "arch",
            triage_threshold=2,
            finescale_threshold=2 / 3,
        )
        entries = json.loads((tmp_path / "arch" / "manifest.json").read_text(encoding="utf-8"))
        assert entries[0]["top_picks"] == 1, f"archive counted {entries[0]}, expected 1 top pick"


class TestRunLevelHelpers:
    def test_enough_scored_gates_on_the_fraction(self) -> None:
        assert finescale.enough_scored(5, 10) is True
        assert finescale.enough_scored(4, 10) is False
        assert finescale.enough_scored(10, 10, 1.0) is True

    def test_nothing_attempted_is_not_success(self) -> None:
        assert finescale.enough_scored(0, 0) is False

    def test_threshold_for_run_reads_the_data_not_the_config(self) -> None:
        """Post-hoc readers (archive, notify) must not have to remember whether the
        stage ran — scores are persisted only when the gate applies."""
        assert finescale.threshold_for_run([_paper("a", 2)]) is None
        assert finescale.threshold_for_run([_paper("a", 2, 0.9)]) == pytest.approx(2 / 3)
        assert finescale.threshold_for_run([_paper("a", 2, 0.9)], 0.8) == pytest.approx(0.8)


class TestScoringNeverFabricates:
    def _cfg(self) -> SimpleNamespace:
        return SimpleNamespace(openai_api_key="k", openai_model="m", timeout=5)

    def test_a_failed_call_omits_the_paper_rather_than_scoring_it_zero(self) -> None:
        with patch.object(finescale, "top_logprobs", side_effect=LLMError("boom")):
            out = finescale.score_papers(
                [{"arxiv_id": "a", "title": "T", "abstract": "A"}], _Profile(), self._cfg()
            )
        assert out == {}

    def test_a_digitless_response_omits_the_paper(self) -> None:
        with patch.object(finescale, "top_logprobs", return_value=[("Sorry", 1.0)]):
            out = finescale.score_papers(
                [{"arxiv_id": "a", "title": "T", "abstract": "A"}], _Profile(), self._cfg()
            )
        assert out == {}

    def test_a_good_call_yields_both_the_score_and_its_probability(self) -> None:
        with patch.object(finescale, "top_logprobs", return_value=[("8", 1.0)]):
            out = finescale.score_papers(
                [{"arxiv_id": "a", "title": "T", "abstract": "A"}], _Profile(), self._cfg()
            )
        assert out["a"]["finescale"] == pytest.approx(8.0)
        assert out["a"]["finescale_p"] == pytest.approx(finescale.probability(8.0))

    def test_a_paper_without_an_id_is_skipped_not_keyed_on_none(self) -> None:
        with patch.object(finescale, "top_logprobs", return_value=[("8", 1.0)]):
            assert finescale.score_papers([{"title": "T"}], _Profile(), self._cfg()) == {}


class TestTheOpenAITransport:
    def test_a_missing_key_raises_rather_than_returning_empty(self, monkeypatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(LLMError, match="No OpenAI API key"):
            top_logprobs("p", SimpleNamespace(openai_api_key="", timeout=5))

    def test_it_parses_the_first_token_alternatives(self) -> None:
        body = json.dumps(
            {
                "choices": [
                    {
                        "logprobs": {
                            "content": [
                                {
                                    "token": "7",
                                    "top_logprobs": [
                                        {"token": "7", "logprob": math.log(0.7)},
                                        {"token": "8", "logprob": math.log(0.3)},
                                    ],
                                }
                            ]
                        }
                    }
                ]
            }
        ).encode()
        resp = MagicMock()
        resp.read.return_value = body
        resp.__enter__ = lambda s: s
        resp.__exit__ = lambda *a: None
        with patch("urllib.request.urlopen", return_value=resp):
            got = top_logprobs("p", SimpleNamespace(openai_api_key="k", timeout=5))
        assert [t for t, _ in got] == ["7", "8"]
        assert got[0][1] == pytest.approx(0.7)

    def test_a_response_without_logprobs_raises(self) -> None:
        resp = MagicMock()
        resp.read.return_value = json.dumps({"choices": [{"logprobs": {}}]}).encode()
        resp.__enter__ = lambda s: s
        resp.__exit__ = lambda *a: None
        with (
            patch("urllib.request.urlopen", return_value=resp),
            pytest.raises(LLMError, match="no logprobs"),
        ):
            top_logprobs("p", SimpleNamespace(openai_api_key="k", timeout=5))

    def test_redaction_applies_before_the_prompt_leaves(self) -> None:
        seen: list[str] = []
        with patch.object(
            sys.modules["reporadar.llm_client"],
            "_call_openai_top_logprobs",
            side_effect=lambda p, *a: seen.append(p) or [("5", 1.0)],
        ):
            top_logprobs(
                "secret-project is great",
                SimpleNamespace(openai_api_key="k", timeout=5, redact=["secret-project"]),
            )
        assert "secret-project" not in seen[0]


class TestTheStore:
    def test_finescale_columns_round_trip(self, tmp_path: Path) -> None:
        store = PaperStore(tmp_path / "db.sqlite")
        run_id = store.record_run([], 1, 0)
        store.upsert_papers([_row()])
        store.save_scores(run_id, [{"arxiv_id": "a", "score_total": 0.5}])
        store.save_llm_scores(run_id, {"a": {"llm_score": 2, "llm_reason": "r"}})
        store.save_finescale_scores(run_id, {"a": {"finescale": 7.25, "finescale_p": 0.81}})
        row = next(r for r in store.get_scores_for_run(run_id) if r["arxiv_id"] == "a")
        assert row["llm_score"] == 2
        assert row["finescale"] == pytest.approx(7.25)
        assert row["finescale_p"] == pytest.approx(0.81)
        store.close()

    def test_an_ungated_run_reads_back_as_null_not_zero(self, tmp_path: Path) -> None:
        store = PaperStore(tmp_path / "db.sqlite")
        run_id = store.record_run([], 1, 0)
        store.upsert_papers([_row()])
        store.save_scores(run_id, [{"arxiv_id": "a", "score_total": 0.5}])
        store.save_llm_scores(run_id, {"a": {"llm_score": 2, "llm_reason": "r"}})
        row = next(r for r in store.get_scores_for_run(run_id) if r["arxiv_id"] == "a")
        assert row["finescale_p"] is None
        store.close()

    def test_it_updates_and_never_invents_a_gate_score(self, tmp_path: Path) -> None:
        """An INSERT here would fabricate an llm_score the gate never gave."""
        store = PaperStore(tmp_path / "db.sqlite")
        run_id = store.record_run([], 1, 0)
        store.upsert_papers([_row()])
        store.save_scores(run_id, [{"arxiv_id": "a", "score_total": 0.5}])
        store.save_finescale_scores(run_id, {"a": {"finescale": 7.0, "finescale_p": 0.7}})
        row = next(r for r in store.get_scores_for_run(run_id) if r["arxiv_id"] == "a")
        assert row["llm_score"] is None
        store.close()
