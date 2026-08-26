"""The k=3 redraw of the gold set, pinned [P17].

`evals/gold_spread.json` holds 75 independent baseline draws over the benchmark25 cohort,
judged. It is what licenses restating every published recall figure as an estimate with an
interval rather than an exact fraction, so the numbers the documents quote are pinned here —
its inputs (baseline caches, judge verdicts, frozen pools) are all gitignored.

The load-bearing checks, in order of what they protect:

* **`partial` rows must never be counted.** A row whose picks could not all be resolved or
  judged has a target set that is a FLOOR. Counting it would bias every fresh draw downward
  and manufacture instability out of an arXiv throttle — which is exactly what the 429s of
  2026-08-26 would have produced. The analysis reads `ok` rows only, and a test enforces it.
* **Failure is not "found nothing".** A case that never produced an `ok` baseline in a draw
  is absent from that draw, not empty in it. `void, not null`, once more.
* **The headline figures**, so a hand-edit fails on any machine.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))

FROZEN = ROOT / "evals" / "gold_spread.json"
USABLE = ("ok", "partial")


@pytest.fixture(scope="module")
def artifact() -> dict:
    if not FROZEN.is_file():
        pytest.skip("no gold_spread artifact")
    return json.loads(FROZEN.read_text(encoding="utf-8"))


def _rows(artifact: dict, draw: int) -> dict[str, dict]:
    return {
        key.split("/", 1)[1]: row
        for key, row in artifact["results"].items()
        if int(key.split("/", 1)[0]) == draw
    }


class TestTheArtifact:
    def test_shape(self, artifact):
        assert artifact["cohort"] == "benchmark25"
        assert artifact["draws"] == 3
        assert len(artifact["results"]) == 75, "25 cases x 3 draws"
        for key in artifact["results"]:
            draw, case = key.split("/", 1)
            assert draw in {"1", "2", "3"} and case

    def test_every_usable_row_is_internally_consistent(self, artifact):
        for key, row in artifact["results"].items():
            if row["status"] not in USABLE:
                assert "raw" in row or row["status"] == "clone_failed", key
                continue
            targets, scores = row["targets"], row["scores"]
            assert set(targets) <= set(scores), key
            assert all(scores[t] >= 2 for t in targets), key
            assert sorted(targets) == targets, f"{key}: targets must be sorted"
            # A target is a judged pick, so it cannot exceed the picks the draw made.
            assert set(targets) <= set(row["picks"]), key

    def test_partial_rows_are_floors_and_say_so(self, artifact):
        """`partial` means something was unresolved or unjudged — never a clean count."""
        for key, row in artifact["results"].items():
            if row["status"] == "partial":
                assert row["n_lookup_failed"] or row["n_judge_failed"], key
            elif row["status"] == "ok":
                assert not row["n_lookup_failed"] and not row["n_judge_failed"], key


class TestTheAnalysisExcludesWhatItMustExclude:
    def test_only_ok_rows_feed_the_target_arithmetic(self, artifact):
        """The regression that would have turned an arXiv 429 into a moving gold set."""
        import gold_spread

        for draw in (1, 2, 3):
            counted = gold_spread._draw_targets(artifact, draw) or {}
            rows = _rows(artifact, draw)
            for case in counted:
                assert rows[case]["status"] == "ok", f"draw {draw}/{case} is not ok"
            for case, row in rows.items():
                if row["status"] != "ok":
                    assert case not in counted, f"draw {draw}/{case} ({row['status']}) counted"

    def test_a_failed_case_is_absent_not_empty(self, artifact):
        import gold_spread

        for draw in (1, 2, 3):
            counted = gold_spread._draw_targets(artifact, draw) or {}
            failed = [c for c, r in _rows(artifact, draw).items() if r["status"] not in USABLE]
            assert failed, f"draw {draw} had no failures; update this test's premise"
            for case in failed:
                assert case not in counted


class TestTheHeadlineFigures:
    """What RESULTS.md and PLANS.md quote."""

    def test_failure_counts_per_draw(self, artifact):
        counts = {}
        for draw in (1, 2, 3):
            rows = _rows(artifact, draw)
            counts[draw] = sum(1 for r in rows.values() if r["status"] not in USABLE)
        assert counts == {1: 6, 2: 5, 3: 3}
        assert sum(counts.values()) / 75 == pytest.approx(14 / 75, abs=1e-9)

    def test_the_cases_that_failed_every_draw(self, artifact):
        """They all have a cached success — the selection effect, made concrete."""
        always = {
            case
            for case in {k.split("/", 1)[1] for k in artifact["results"]}
            if all(_rows(artifact, d).get(case, {}).get("status") not in USABLE for d in (1, 2, 3))
        }
        assert always == {"thin-lang", "vectordb"}

    def test_target_counts_per_draw(self, artifact):
        """Two scopes, both pinned, because they differ and the write-up quotes one.

        The report compares each draw against the frozen set on the cases they SHARE, so a
        draw that succeeded on a case the frozen set has no targets for contributes to the
        draw's own total but not to the comparison. Draw 1 is 39 all-in and 38 shared —
        conflating them is a small C-17 waiting to happen.
        """
        import gold_spread

        gold = json.loads((ROOT / "evals" / "gold_targets.json").read_text(encoding="utf-8"))
        frozen = {
            c: set(ids) for c, ids in gold["targets"].items() if not c.startswith(("bio-", "mat-"))
        }
        all_in, shared_only = {}, {}
        for d in (1, 2, 3):
            targets = gold_spread._draw_targets(artifact, d) or {}
            all_in[d] = sum(len(v) for v in targets.values())
            shared = set(targets) & set(frozen)
            shared_only[d] = sum(len(targets[c]) for c in shared)
        assert all_in == {1: 39, 2: 39, 3: 46}
        assert shared_only == {1: 38, 2: 39, 3: 46}, "the figures RESULTS.md quotes"

    def test_reproducibility_is_not_above_the_pick_level(self, artifact):
        """The pre-registered prediction FAILED: the judge absorbs none of the churn.

        Pinned as a property rather than a number so it keeps meaning something: target-level
        agreement (~0.39) must stay at or below the pick-level 0.41 that P15 measured. If a
        later change pushes it above, the prediction has become true and the write-up is wrong.
        """
        import gold_spread

        gold = json.loads((ROOT / "evals" / "gold_targets.json").read_text(encoding="utf-8"))
        frozen = {
            c: set(ids) for c, ids in gold["targets"].items() if not c.startswith(("bio-", "mat-"))
        }
        for draw in (1, 2, 3):
            targets = gold_spread._draw_targets(artifact, draw) or {}
            shared = set(targets) & set(frozen)
            hit = sum(len(frozen[c] & targets[c]) for c in shared)
            tot = sum(len(frozen[c]) for c in shared)
            assert 0.30 <= hit / tot <= 0.45, f"draw {draw}: {hit}/{tot}"
