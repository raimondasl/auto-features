"""Tests for P7's second-judge agreement measurement.

P7 exists to put an error bar under every labelled-set conclusion in the project, so its own
arithmetic has to be right: a kappa that reports agreement where there is none, or a sample
that is really one case, would replace an unbounded number with a wrong one — which is worse.

The guard that matters most is the last class here. A second judge's verdict written into
`evals/cache/judge/` would be indistinguishable from the first judge's, and this project has
already lost a gold set to exactly that write once.
"""

from __future__ import annotations

import importlib.util
import json
import random
import sys
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import pytest

EVALS = Path(__file__).resolve().parent.parent / "evals"


def _load(name: str):  # type: ignore[no-untyped-def]
    if str(EVALS) not in sys.path:
        sys.path.insert(0, str(EVALS))
    spec = importlib.util.spec_from_file_location(name, EVALS / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


sj = _load("second_judge")


class TestCohensKappa:
    def test_perfect_agreement_is_one(self) -> None:
        assert sj.cohens_kappa([0, 1, 1, 0, 1], [0, 1, 1, 0, 1]) == pytest.approx(1.0)

    def test_chance_agreement_is_zero(self) -> None:
        """50% raw agreement on a balanced two-class problem is exactly chance."""
        assert sj.cohens_kappa([1, 1, 0, 0], [1, 0, 1, 0]) == pytest.approx(0.0)

    def test_systematic_disagreement_is_negative(self) -> None:
        assert sj.cohens_kappa([1, 1, 0, 0], [0, 0, 1, 1]) == pytest.approx(-1.0)

    def test_both_judges_constant_does_not_divide_by_zero(self) -> None:
        """If neither judge ever says "actionable", pe is 1 and the formula is 0/0."""
        assert sj.cohens_kappa([0] * 10, [0] * 10) == 1.0

    def test_high_raw_agreement_on_a_skewed_split_is_not_high_kappa(self) -> None:
        """The reason kappa is the pre-registered statistic and raw agreement is not.

        19 of 20 agree, but both judges say 0 almost always, so most of that is chance.
        """
        a = [0] * 18 + [1, 1]
        b = [0] * 19 + [1]
        assert sum(1 for x, y in zip(a, b, strict=True) if x == y) / 20 == pytest.approx(0.95)
        assert sj.cohens_kappa(a, b) < 0.7


class TestQuadraticKappa:
    def test_perfect_agreement_is_one(self) -> None:
        assert sj.quadratic_kappa([0, 1, 2, 3, 2], [0, 1, 2, 3, 2]) == pytest.approx(1.0)

    def test_adjacent_disagreement_beats_distant_disagreement(self) -> None:
        """A 2-vs-3 is nearly agreement on an ordinal rubric; a 0-vs-3 is not."""
        base = [0, 1, 2, 3] * 5
        near = [0, 1, 3, 3] * 5
        far = [3, 1, 2, 0] * 5
        assert sj.quadratic_kappa(base, near) > sj.quadratic_kappa(base, far)


class TestThresholdShiftSeparatesOffsetFromDisagreement:
    """Measured: kappa 0.51 at the shipped cut, 0.71 with the second judge's cut moved to 1.

    A single kappa cannot tell "these judges rank papers differently" from "these judges rank
    papers the same and one is stricter". Those imply opposite remedies — adjudicate labels
    versus recalibrate a threshold — so the report has to distinguish them.
    """

    def test_a_pure_one_notch_offset_is_recovered_by_moving_the_cut(self) -> None:
        rows = [{"gpt": s, "sonnet": max(0, s - 1)} for s in (0, 1, 2, 3) for _ in range(10)]
        by_cut = {cut: k for cut, k, _ in sj.threshold_shift(rows)}
        assert by_cut[2] < 0.6, "at the shipped cut the offset looks like disagreement"
        assert by_cut[1] == pytest.approx(1.0), "shifting the cut recovers perfect agreement"

    def test_genuine_disagreement_is_not_rescued_by_any_cut(self) -> None:
        rows = [{"gpt": 3, "sonnet": 0}] * 10 + [{"gpt": 0, "sonnet": 3}] * 10
        assert all(k <= 0.0 for _, k, _ in sj.threshold_shift(rows))


class TestNetAtTwo:
    LABELS = {("cv", "a"): 3, ("cv", "b"): 0, ("cv", "c"): 2}

    def test_only_admitted_papers_are_scored(self) -> None:
        rows = [
            {"case": "cv", "id": "a", "triage": 2},
            {"case": "cv", "id": "b", "triage": 1},  # rejected: contributes nothing
        ]
        assert sj.net_at_2(rows, self.LABELS) == 1

    def test_a_false_positive_costs_two(self) -> None:
        rows = [{"case": "cv", "id": "b", "triage": 3}]
        assert sj.net_at_2(rows, self.LABELS) == -2

    def test_an_unlabelled_paper_is_skipped_not_counted_as_wrong(self) -> None:
        rows = [{"case": "cv", "id": "zzz", "triage": 3}]
        assert sj.net_at_2(rows, self.LABELS) == 0


class TestStratifiedSample:
    def _labels(self) -> dict[tuple[str, str], int]:
        out = {}
        for i in range(500):
            out[("big", f"b{i}")] = i % 4
        for i in range(12):
            out[("small", f"s{i}")] = i % 4
        return out

    def test_one_large_case_does_not_take_the_whole_sample(self) -> None:
        picked = sj.sample(self._labels(), {"big", "small"}, 40, random.Random(1))
        counts = Counter(c for c, _ in picked)
        assert len(picked) == 40
        assert counts["small"] == 12, "the small case should be exhausted, not skipped"
        assert counts["big"] == 28

    def test_every_verdict_level_is_represented(self) -> None:
        labels = self._labels()
        picked = sj.sample(labels, {"big", "small"}, 40, random.Random(1))
        scores = Counter(labels[k] for k in picked)
        assert set(scores) == {0, 1, 2, 3}
        assert min(scores.values()) >= 8, f"a verdict level was starved: {scores}"

    def test_the_same_seed_reproduces_the_sample(self) -> None:
        labels = self._labels()
        a = sj.sample(labels, {"big", "small"}, 30, random.Random(sj.SEED))
        b = sj.sample(labels, {"big", "small"}, 30, random.Random(sj.SEED))
        assert a == b

    def test_a_case_outside_the_allowed_set_is_never_drawn(self) -> None:
        """Cases whose prompt hash did not reproduce must not enter the sample."""
        picked = sj.sample(self._labels(), {"small"}, 40, random.Random(1))
        assert {c for c, _ in picked} == {"small"}


class TestPromptHashGuard:
    def test_a_case_whose_context_changed_is_excluded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A drifted clone means the stored label answers a question we cannot rebuild."""
        import hashlib

        work = tmp_path / "work"
        gold = tmp_path / "gold"
        for case in ("good", "drifted"):
            (work / case).mkdir(parents=True)
            (work / case / "README.md").write_text(f"# {case}\n", encoding="utf-8")
            (gold / case).mkdir(parents=True)
        monkeypatch.setattr(sj, "WORK_DIR", work)
        monkeypatch.setattr(sj, "GOLD", gold)

        ctx = sj.assemble_repo_context(work / "good")
        good_hash = hashlib.sha256(f"{sj.judge_mod.RUBRIC}\0{ctx}".encode()).hexdigest()[:12]
        (gold / "good" / "1.json").write_text(
            json.dumps({"score": 2, "_prompt_hash": good_hash}), encoding="utf-8"
        )
        (gold / "drifted" / "1.json").write_text(
            json.dumps({"score": 2, "_prompt_hash": "deadbeefcafe"}), encoding="utf-8"
        )

        contexts, drifted = sj.verify_contexts(["good", "drifted"])
        assert set(contexts) == {"good"}
        assert drifted == ["drifted"]


class TestSecondVerdictNeverTouchesTheGoldCache:
    """The failure that already happened once, in a different script, in the same week.

    `judge_paper` keys its cache on (model, repo, paper_id) and a score written there is
    indistinguishable from the first judge's. A second judge MUST write elsewhere.
    """

    def test_the_verdict_lands_outside_evals_cache(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sj, "CACHE", tmp_path / "second_judge")
        with patch.object(sj, "complete", return_value='{"score": 2, "justification": "x"}'):
            score = sj.second_verdict(
                "cv", "ctx", {"arxiv_id": "2106.09685", "title": "t", "abstract": "a"}, "m"
            )
        assert score == 2
        written = list((tmp_path / "second_judge").rglob("*.json"))
        assert len(written) == 1
        assert sj.GOLD not in written[0].parents
        assert "cache" not in written[0].relative_to(tmp_path).parts

    def test_a_cached_verdict_is_reused_without_calling_the_model(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sj, "CACHE", tmp_path / "second_judge")
        path = tmp_path / "second_judge" / "m" / "cv" / "2106.09685.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"score": 3}), encoding="utf-8")
        with patch.object(sj, "complete", side_effect=AssertionError("model was called")):
            assert (
                sj.second_verdict(
                    "cv", "ctx", {"arxiv_id": "2106.09685", "title": "t", "abstract": "a"}, "m"
                )
                == 3
            )

    def test_an_out_of_range_score_is_rejected_not_cached(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Never fabricate a verdict: a malformed response must raise, not become a 0."""
        monkeypatch.setattr(sj, "CACHE", tmp_path / "second_judge")
        with (
            patch.object(sj, "complete", return_value='{"score": 7}'),
            pytest.raises(ValueError),
        ):
            sj.second_verdict("cv", "ctx", {"arxiv_id": "1.1", "title": "t", "abstract": "a"}, "m")
        assert not list((tmp_path / "second_judge").rglob("*.json"))
