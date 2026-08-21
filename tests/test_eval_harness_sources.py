"""The eval harness must be able to fetch every source the product can.

`evals/harness.collect_live_papers` has its own source dispatch, separate from
`reporadar.pipeline`. When §13 shipped the Europe PMC adapter it was wired into the product and
**not** into the harness, so `--sources arxiv,europepmc` could not have measured the channel it
was built to measure — the run would have died on the unknown-source guard after cloning six
repositories. Found while about to run the §20 arm, which is two days later than it should have
been.

The guard below is the cheap version of that discovery: a source the product knows and the
harness does not is a benchmark that cannot measure a shipped feature.

The harness's own `unknown` check is the runtime half and it is deliberately loud — its comment
records that `--sources arxiv,dblp` was once run as a silent no-op and reported as a
measurement. This file is the half that fires before anyone spends money.
"""

from __future__ import annotations

import sys
from pathlib import Path

EVALS = Path(__file__).resolve().parent.parent / "evals"
if str(EVALS) not in sys.path:
    sys.path.insert(0, str(EVALS))

import harness  # noqa: E402

from reporadar.pipeline import KEYWORD_SOURCE_QUERIES, KEYWORD_SOURCES  # noqa: E402


def _harness_source_branches() -> set[str]:
    """The source names `collect_live_papers` actually branches on.

    Read from the source text rather than by calling it, because calling it needs a repo, a
    profile and the network. A string search is enough: each branch is written as
    `if "<name>" in sources:` and nothing else in the function has that shape.
    """
    text = Path(harness.__file__).read_text(encoding="utf-8")
    body = text.split("def collect_live_papers", 1)[1].split("\ndef ", 1)[0]
    return {
        line.split('"')[1]
        for line in body.splitlines()
        if line.strip().startswith('if "') and line.strip().endswith('" in sources:')
    }


class TestTheHarnessKnowsEverySourceTheProductKnows:
    def test_every_product_source_has_a_harness_branch(self) -> None:
        missing = set(KEYWORD_SOURCES) - _harness_source_branches()
        assert missing == set(), (
            f"the product can fetch {sorted(missing)} and the eval harness cannot, so no "
            "benchmark arm can measure them. Add a branch in harness.collect_live_papers."
        )

    def test_the_harness_invents_no_source_the_product_lacks(self) -> None:
        """The other direction: a harness-only source would benchmark something unshippable."""
        extra = _harness_source_branches() - set(KEYWORD_SOURCES) - {"arxiv"}
        assert extra == set(), f"harness fetches {sorted(extra)}, which the product does not"

    def test_europepmc_specifically(self) -> None:
        """Named because it is the one that was missing, and the §20 arm depends on it."""
        assert "europepmc" in _harness_source_branches()


class TestOneKeywordQueryCap:
    def test_the_harness_uses_the_products_cap(self) -> None:
        """The harness hardcoded 5 while the product used 8, so every non-arXiv source was
        benchmarked on 5/8 of its shipped queries — B4 (§12.2) raised it in one place only."""
        text = Path(harness.__file__).read_text(encoding="utf-8")
        body = text.split("def collect_live_papers", 1)[1].split("\ndef ", 1)[0]
        assert "queries[:KEYWORD_SOURCE_QUERIES]" in body
        assert "queries[:5]" not in body

    def test_the_cap_is_what_b4_set_it_to(self) -> None:
        assert KEYWORD_SOURCE_QUERIES == 8
