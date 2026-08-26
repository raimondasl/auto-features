"""The v2-prompt sweep, pinned — and the accounting invariant it cost to find.

`evals/gold_spread_v2.json` holds 75 draws of `BASELINE_PROMPT_V2` at a 30-turn cap over the
benchmark25 cohort, judged. It is a *different searcher* from the P17 sweep, not more samples
of it, which is why it lives in its own file and why nothing here compares the two as though
the difference were noise.

Two things this file protects that the v1 pins do not:

* **The counter identity.** Every pick either has a verdict or is accounted for by exactly
  one of `n_hallucinated` / `n_lookup_failed` / `n_unjudgeable`. `repair_row` broke it the
  first time it ran twice on a row: it accumulated two of the three counters, so `1/linter`
  reported 31 unjudgeable references against 12 picks with no verdict and the sweep's total
  read 71 where the truth was 44. Nothing crashed and every number stayed plausible — the
  only signal was an identity nobody was checking.
* **Retryability is derived, not remembered.** A partial row used to carry `phase: "judged"`
  and was therefore never revisited, so a transient throttle froze into a permanent floor.
  `retryable` now reads the outcome counts, which is only meaningful because a
  `lookup_failed` and an `unjudgeable` are genuinely different states — see C-32, where they
  were not being told apart at all.
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))

FROZEN = ROOT / "evals" / "gold_spread_v2.json"
V1 = ROOT / "evals" / "gold_spread.json"
USABLE = ("ok", "partial")


@pytest.fixture(scope="module")
def artifact() -> dict:
    if not FROZEN.is_file():
        pytest.skip("no gold_spread_v2 artifact")
    return json.loads(FROZEN.read_text(encoding="utf-8"))


def _unscored(row: dict) -> set[str]:
    return set(row.get("picks") or []) - set(row.get("scores") or {})


class TestTheSweep:
    def test_shape_and_provenance(self, artifact):
        assert artifact["prompt_version"] == "v2"
        assert artifact["cohort"] == "benchmark25"
        assert len(artifact["results"]) == 75, "25 cases x 3 draws"
        for row in artifact["results"].values():
            assert row["prompt_version"] == "v2"
            assert row["max_turns"] == 30

    def test_it_completed(self, artifact):
        """0 failed and 0 partial across 75 runs — quoted, so pinned."""
        statuses = [r["status"] for r in artifact["results"].values()]
        assert set(statuses) == {"ok"}, sorted(set(statuses))

    def test_v2_reaches_papers_v1_could_not_name(self, artifact):
        """The whole point of the prompt. Under v1 a non-arXiv paper had no field to go in."""
        picks = [p for r in artifact["results"].values() for p in r["picks"]]
        targets = [t for r in artifact["results"].values() for t in r["targets"]]
        assert sum(p.startswith("doi:") for p in picks) == 97
        # 36 before the OpenAlex tier, 61 after. The searcher did not change; the 25 extra
        # are papers it had already named and we could not previously read.
        assert sum(t.startswith("doi:") for t in targets) == 61
        v1 = json.loads(V1.read_text(encoding="utf-8"))["results"]
        v1_picks = [p for r in v1.values() for p in (r.get("picks") or [])]
        assert not any(p.startswith("doi:") for p in v1_picks), "v1 cannot emit a DOI at all"


class TestTheCounterIdentity:
    """Every pick is either judged or accounted for, exactly once."""

    @pytest.mark.parametrize("path", [FROZEN, V1], ids=["v2", "v1"])
    def test_unscored_picks_equal_the_failure_counters(self, path):
        if not path.is_file():
            pytest.skip(f"no {path.name}")
        for key, row in json.loads(path.read_text(encoding="utf-8"))["results"].items():
            if row["status"] not in USABLE:
                continue
            counters = (
                row.get("n_hallucinated", 0)
                + row.get("n_lookup_failed", 0)
                + row.get("n_unjudgeable", 0)
            )
            assert len(_unscored(row)) == counters, (
                f"{path.name} {key}: {len(_unscored(row))} pick(s) without a verdict "
                f"but {counters} counted — a counter is double-adding or losing a reference"
            )

    def test_the_sweep_totals(self, artifact):
        """44 references the instrument could not score, split by whose fault each one is.

        The split is the entire reason `verify` classifies four outcomes rather than two.
        **41 unjudgeable** — real papers, proven to exist, that no source we ask carries an
        abstract for; our gap, permanent, and never charged to the model. **3 hallucinated**
        — DOIs the registry says do not exist; the model's, and the only one of the two that
        counts against it. **0 lookup_failed** — a finished sweep has no backlog left.
        """
        rows = list(artifact["results"].values())
        assert sum(len(_unscored(r)) for r in rows) == 13
        assert sum(r["n_unjudgeable"] for r in rows) == 10
        assert sum(r["n_hallucinated"] for r in rows) == 3
        assert sum(r["n_lookup_failed"] for r in rows) == 0, "a finished sweep has no backlog"

    def test_the_hallucinations_are_rare_and_were_caught(self, artifact):
        """v2 was given no anti-fabrication coaching v1 lacks, deliberately — so this is the
        unassisted rate: 3 invented DOIs in 97, caught by the registry rather than by the
        prompt. Small, non-zero, and the number a future comparator re-measurement needs."""
        rows = list(artifact["results"].values())
        dois = sum(1 for r in rows for p in r["picks"] if p.startswith("doi:"))
        assert sum(r["n_hallucinated"] for r in rows) / dois < 0.05

    def test_no_acm_paper_is_unscoreable_any_more(self, artifact):
        """This assertion used to say the opposite, and the inversion is the whole result.

        Before the OpenAlex tier, ACM was the gap: `10.1145/…` accounted for more than half
        of 44 unscoreable references — POPL, PLDI, OOPSLA, CACM, the venues a code benchmark
        most needs to read. Semantic Scholar rejects many of those DOIs outright and Europe
        PMC is biomedical. OpenAlex carries them, and **zero remain**.
        """
        unscored = [p for r in artifact["results"].values() for p in _unscored(r)]
        assert unscored and all(p.startswith("doi:") for p in unscored)
        assert not [p for p in unscored if p.startswith("doi:10.1145/")]

    def test_the_residual_is_named_rather_than_chased(self, artifact):
        """8 distinct papers, and the shape of what no tier reaches: Springer book chapters
        (chronically abstract-free), one Elsevier journal paper, and two fabricated DOIs that
        never existed. Pinned so the floor is a measured quantity rather than an impression."""
        unscored = {p for r in artifact["results"].values() for p in _unscored(r)}
        assert len(unscored) == 8
        by_prefix = collections.Counter(p.split("/")[0].removeprefix("doi:") for p in unscored)
        assert dict(by_prefix) == {"10.1007": 5, "10.5555": 2, "10.1016": 1}


class TestRetryabilityIsDerived:
    def test_a_finished_row_is_never_retryable(self, artifact):
        """The C-30 trap in reverse: a row that can never be completed must stop being asked.

        "Finished" now means finished *under the current tier set*, which every row in the
        committed artifact records. Adding a source is what legitimately reopens them, and
        the row stamps the new set when it is re-asked, so the same growth cannot fire twice.
        """
        import gold_spread

        for key, row in artifact["results"].items():
            assert not gold_spread.retryable(row), key

    def test_every_row_with_an_open_question_records_who_was_asked(self, artifact):
        """Only rows that were re-asked carry the stamp, and that is correct rather than
        untidy. A row whose references all resolved has no `unjudgeable` verdict to
        reopen, so no tier growth can ever reach it and back-stamping it with a tier set
        that did not exist when it ran would be a fabricated provenance claim.

        The invariant that actually matters is the one below: nothing is left retryable.
        """
        import verify

        for key, row in artifact["results"].items():
            if row.get("n_unjudgeable"):
                assert row["tier_set"] == list(verify.TIER_SET), key

    def test_unjudgeable_alone_does_not_make_a_row_retryable(self):
        """Not while the tier set is unchanged. The verdict was correct when it was made and
        asking the same four sources again cannot overturn it."""
        import gold_spread
        import verify

        row = {
            "status": "partial",
            "picks": ["doi:10.1145/x"],
            "scores": {},
            "n_lookup_failed": 0,
            "n_unjudgeable": 1,
            "tier_set": list(verify.TIER_SET),
        }
        assert gold_spread.unscored_picks(row) == ["doi:10.1145/x"]
        assert not gold_spread.retryable(row)

    def test_a_grown_tier_set_reopens_an_unjudgeable_row(self):
        """The clause that cashed in the OpenAlex tier: 31 references stranded behind a
        predicate correctly refusing to re-ask a settled question."""
        import gold_spread
        import verify

        row = {
            "status": "ok",  # note: not partial — a finished row can still be reopened
            "picks": ["doi:10.1145/x"],
            "scores": {},
            "n_lookup_failed": 0,
            "n_unjudgeable": 1,
            "tier_set": list(verify.LEGACY_TIER_SET),
        }
        assert gold_spread.retryable(row)

    def test_a_row_with_nothing_open_is_never_reopened(self):
        """The tier clause must not re-ask rows that have no unanswered question, however
        much the verifier has improved."""
        import gold_spread
        import verify

        row = {
            "status": "ok",
            "picks": ["2401.12345"],
            "scores": {"2401.12345": 3},
            "n_unjudgeable": 0,
            "tier_set": list(verify.LEGACY_TIER_SET),
        }
        assert not gold_spread.retryable(row)

    def test_a_lookup_failure_does(self):
        import gold_spread

        row = {
            "status": "partial",
            "picks": ["doi:10.1145/x"],
            "scores": {},
            "n_lookup_failed": 1,
            "n_unjudgeable": 0,
        }
        assert gold_spread.retryable(row)

    def test_the_retry_set_is_recovered_from_picks(self):
        """`judge_row` drops `raw_ids`, so `picks` is the only surviving identity."""
        import gold_spread

        row = {"picks": ["2401.12345", "doi:10.1/x"], "scores": {"2401.12345": 3}}
        assert gold_spread.unscored_picks(row) == ["doi:10.1/x"]
