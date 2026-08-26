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
        assert sum(t.startswith("doi:") for t in targets) == 36
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
        assert sum(len(_unscored(r)) for r in rows) == 44
        assert sum(r["n_unjudgeable"] for r in rows) == 41
        assert sum(r["n_hallucinated"] for r in rows) == 3
        assert sum(r["n_lookup_failed"] for r in rows) == 0, "a finished sweep has no backlog"

    def test_the_hallucinations_are_rare_and_were_caught(self, artifact):
        """v2 was given no anti-fabrication coaching v1 lacks, deliberately — so this is the
        unassisted rate: 3 invented DOIs in 97, caught by the registry rather than by the
        prompt. Small, non-zero, and the number a future comparator re-measurement needs."""
        rows = list(artifact["results"].values())
        dois = sum(1 for r in rows for p in r["picks"] if p.startswith("doi:"))
        assert sum(r["n_hallucinated"] for r in rows) / dois < 0.05

    def test_everything_unscored_is_a_doi(self, artifact):
        """The gap is entirely non-arXiv, and mostly ACM: real papers with no abstract in
        Semantic Scholar or Europe PMC. It is the cost of v2's reach, and it is permanent —
        which is why these rows are `ok` rather than queued for a retry that cannot help."""
        unscored = [p for r in artifact["results"].values() for p in _unscored(r)]
        assert unscored and all(p.startswith("doi:") for p in unscored)
        acm = sum(1 for p in unscored if p.startswith("doi:10.1145/"))
        assert acm >= len(unscored) // 2, f"only {acm}/{len(unscored)} are ACM"


class TestRetryabilityIsDerived:
    def test_a_finished_row_is_never_retryable(self, artifact):
        """The C-30 trap in reverse: a row that can never be completed must stop being asked."""
        import gold_spread

        for key, row in artifact["results"].items():
            assert not gold_spread.retryable(row), key

    def test_unjudgeable_alone_does_not_make_a_row_retryable(self):
        import gold_spread

        row = {
            "status": "partial",
            "picks": ["doi:10.1145/x"],
            "scores": {},
            "n_lookup_failed": 0,
            "n_unjudgeable": 1,
        }
        assert gold_spread.unscored_picks(row) == ["doi:10.1145/x"]
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
