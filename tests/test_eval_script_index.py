"""The paper's reproducibility claim, made checkable.

`paper/DRAFT.md` says every quantitative result maps to a script in `evals/`. That claim
was in the paper for weeks against an index that **did not exist** — it promised scripts
"named per section in the artifact" and no such naming was anywhere in the repository. An
unverifiable assertion about our own artifact is the shape §10 spends its length
cataloguing, so the index now exists (`evals/README.md`) and these tests keep it true.

Both directions matter, and the second is the one the C-9b lesson is about:

* every script the index **names** must exist — otherwise a reader follows a dead pointer;
* every script in `evals/` must be **either indexed or declared** non-paper — otherwise the
  index silently covers whatever subset somebody remembered, which is precisely how the
  config audit came to compare twelve fields out of seventy-nine.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVALS = ROOT / "evals"
INDEX_HEADING = "## Which script re-derives which section of the paper"

# Scripts that are deliberately not rows in the index, with the reason. Infrastructure and
# one-shot data preparation are not "the script that re-derives section N" — but saying so
# here is a decision, where leaving them out of the index silently would be an oversight
# nobody could tell apart from one.
NOT_A_PAPER_SCRIPT: dict[str, str] = {
    "harness.py": "shared library imported by the runners; indexed via the runners",
    "metrics.py": "the metric implementations themselves (indexed under 4.1)",
    "judge.py": "the judge client, used by every Tier B row",
    "baseline.py": "the Opus baseline client, used by every Tier B row",
    "verify.py": "hallucination guard applied inside the runners",
    "build_fixtures.py": "builds the offline fixtures the cases are made of",
    "fill_pool_metadata.py": "one-shot metadata backfill for the hop pool",
    "band_testbeds.py": "shared testbed construction for E1-E5",
    "seeded.py": "Tier S library behind run_seeded_eval.py",
    "compare_finescale_baseline.py": "reporting view over a run the index already names",
    "diagnose_ranker.py": "diagnostic behind 6.5, listed there",
    "make_goals.py": "generates inputs for the stated-intent arm, listed there",
    "fetch_wants.py": "generates inputs for the gate-context arm, listed there",
}


def _index_text() -> str:
    body = (EVALS / "README.md").read_text(encoding="utf-8")
    assert INDEX_HEADING in body, "the script index heading is gone from evals/README.md"
    return body.split(INDEX_HEADING, 1)[1].split("\n---\n", 1)[0]


def _indexed_scripts() -> set[str]:
    return set(re.findall(r"`([a-z0-9_]+\.py)", _index_text()))


def _actual_scripts() -> set[str]:
    return {p.name for p in EVALS.glob("*.py")}


class TestTheIndexPointsAtRealScripts:
    def test_every_named_script_exists(self) -> None:
        missing = sorted(_indexed_scripts() - _actual_scripts())
        assert missing == [], f"the index names scripts that do not exist: {missing}"

    def test_the_index_is_not_empty(self) -> None:
        """A checker that passes on an empty index is not a checker."""
        assert len(_indexed_scripts()) > 25


class TestEveryScriptIsIndexedOrDeclared:
    def test_no_script_is_silently_uncovered(self) -> None:
        """The C-9b direction. An index covering the subset somebody remembered reads
        exactly like one covering everything."""
        uncovered = sorted(_actual_scripts() - _indexed_scripts() - set(NOT_A_PAPER_SCRIPT))
        assert uncovered == [], (
            "these eval scripts are neither in the paper index nor declared "
            f"non-paper in NOT_A_PAPER_SCRIPT: {uncovered}"
        )

    def test_every_declaration_names_a_real_script(self) -> None:
        """A declaration for a deleted script is an exemption that can never expire."""
        stale = sorted(set(NOT_A_PAPER_SCRIPT) - _actual_scripts())
        assert stale == [], f"NOT_A_PAPER_SCRIPT lists scripts that are gone: {stale}"

    def test_every_declaration_carries_a_reason(self) -> None:
        assert all(reason.strip() for reason in NOT_A_PAPER_SCRIPT.values())


class TestThePaperPointsAtTheIndex:
    def test_the_reproducibility_claim_is_not_a_promise_about_nothing(self) -> None:
        """It used to say scripts were "named per section in the artifact" while no such
        naming existed anywhere. Whatever it says now must point somewhere real."""
        paper = (ROOT / "paper" / "DRAFT.md").read_text(encoding="utf-8")
        repro = paper.split("## Reproducibility", 1)[1].split("\n## ", 1)[0]
        assert "evals/README.md" in repro, (
            "the reproducibility section no longer points at the script index"
        )
        # The old wording may still appear, but only in quotation marks — the section
        # recounts the correction. Asserted as a claim (unquoted) it would be the same
        # promise about nothing that C-16's family is made of.
        stale = "named per section in the artifact"
        for match in re.finditer(re.escape(stale), repro):
            quoted = repro[max(0, match.start() - 1)] in "“\"'"
            assert quoted, "the reproducibility section asserts the old promise again"
