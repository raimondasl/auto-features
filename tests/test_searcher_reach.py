"""Pin Table 3's channels measured against other searchers' picks (evals/searcher_reach.py).

NR-71, post hoc and descriptive. On the same 20 repositories, with the same hypotheses, encoder,
index, hop pools and tie rule as Table 3:

* The 56 reference papers, ranked afresh: hypothetical-document search 35, the hop 21, the two
  together 43 (33 and 42 with ties broken against them). Table 3's stored ranks give 34, 21 and
  43; re-encoding moved 13 of the 56 ranks and one paper across the cut, as NR-46's run did.
* All other searchers' arXiv picks, judged >= 2 by GPT-5.5 (244 papers): the two channels together
  reach 0.68, inside Table 3's [0.64, 0.86], and 0.08 below the 56, paired interval
  [-0.16, +0.01].
* A redraw of the same searcher (91 papers) is reached at 0.73, no better than Opus 5 (0.72) or a
  second prompt version (0.72).

The pure functions are tested on hand-built rows. The artifact is rebuilt from the cached ranks
when they are present, which they are in the development tree and not in CI.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import searcher_reach as sr

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "evals" / "searcher_reach.json"


def row(case: str, rank: int, in_hop: bool = False, pess: int | None = None) -> dict[str, object]:
    return {
        "case": case,
        "id": f"{case}-{rank}",
        "rank_opt": rank,
        "rank_pess": rank if pess is None else pess,
        "in_hop": in_hop,
    }


def test_wilson_reproduces_table_3s_union_interval() -> None:
    assert sr.wilson(43, 56) == [0.6423, 0.859]
    assert sr.wilson(0, 0) is None


def test_reached_by_channel() -> None:
    near = row("a", 1000)
    far = row("a", 1001, in_hop=True)
    tie = row("a", 990, pess=1010)
    assert sr.reached(near, "hyde") and not sr.reached(far, "hyde")
    assert sr.reached(far, "hop") and sr.reached(far, "union")
    assert sr.reached(tie, "hyde") and not sr.reached(tie, "hyde_pess")
    assert not sr.reached(tie, "union_pess")
    with pytest.raises(ValueError):
        sr.reached(near, "nonsense")


def test_share_counts_over_rows() -> None:
    rows = [row("a", 5), row("a", 5000), row("b", 5000, in_hop=True)]
    assert sr.share(rows, "hyde") == (1, 3)
    assert sr.share(rows, "union") == (2, 3)


def test_paired_bootstrap_of_a_set_against_itself_is_zero() -> None:
    rows = [row("a", 5), row("a", 5000), row("b", 50), row("c", 9999, in_hop=True)]
    assert sr.bootstrap(rows, ["a", "b", "c"], "union", rows) == [0.0, 0.0]


def test_bootstrap_is_deterministic() -> None:
    rows = [row("a", 5), row("a", 5000), row("b", 50), row("c", 9999)]
    first = sr.bootstrap(rows, ["a", "b", "c"], "hyde")
    assert first == sr.bootstrap(rows, ["a", "b", "c"], "hyde")


def test_summarise_against_a_reference() -> None:
    ref = [row("a", 5), row("b", 5)]
    rows = [row("a", 5), row("b", 5000)]
    s = sr.summarise(rows, ["a", "b"], ref)
    assert s["papers"] == 2 and s["repositories"] == 2
    assert s["union"]["reached"] == 1 and s["union"]["minus_reference"] == -0.5


def test_artifact_pins_the_record() -> None:
    d = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    a = d["anchors"]
    assert a["table3_stored_counts"] == {"hyde": 34, "hop": 21, "union": 43}
    assert a["fresh_counts_for_the_56"] == sr.EXPECTED_FRESH
    assert a["the_56_ranks_moved_since_table3"] == 13
    assert a["ranks_not_in_nr46"] == 0
    ref = d["reference"]
    assert (ref["papers"], ref["repositories"]) == (56, 20)
    assert ref["union"]["wilson95"] == [0.6423, 0.859]

    pooled = d["sets"]["other searchers, pooled"]
    assert pooled["papers"] == 244
    assert pooled["union"]["share"] == 0.6844
    assert pooled["union"]["minus_reference"] == -0.0834
    assert pooled["union"]["minus_reference_paired95"] == [-0.1629, 0.0135]
    shares = {k: v["union"]["share"] for k, v in d["sets"].items()}
    assert shares["cli-redraw"] == 0.7253
    assert shares["cli-v2-opus5@30"] == 0.7177
    assert shares["cli-v2@30"] == 0.7196
    for label in ("cli-redraw", "cli-v2-opus5@30", "cli-v2@30", "other searchers, pooled"):
        lo, hi = d["sets"][label]["union"]["minus_reference_paired95"]
        assert lo < 0 < hi, label  # no comparison set differs from the 56 beyond noise

    reading = d["reading"]
    assert reading["pooled_union_inside_table3_interval"] is True
    assert reading["same_searcher_redraw_union"] == {"cli-redraw": 0.7253}
    assert "cli-redraw@30" not in reading["same_searcher_redraw_union"]  # 9 papers, not read


@pytest.mark.skipif(
    not (sr.RANKS.is_file() and sr.REPLICATION.is_file() and sr.NR46_RANKS.is_file()),
    reason="needs the cached ranks under evals/.work, which are gitignored",
)
def test_rebuild_matches_the_artifact() -> None:
    assert sr.build() == json.loads(ARTIFACT.read_text(encoding="utf-8"))
