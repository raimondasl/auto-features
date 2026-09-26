"""Pin the gate's descriptive figures on all 37 repositories (evals/gate_on_37.py).

The paper's Section 5 quotes these, so a change to them is a change to the paper. They are post
hoc and descriptive. The pinned figures, all under the primary judge on the shipped Haiku gate's
run, with "admitted" meaning admitted within the 15-paper digest:

* 76 of 404 admitted papers score 3.
* Over the 34 repositories with any admitted paper, a repository's share of 3s correlates with
  its precision at r = +0.25, 95% CI [-0.00, +0.52].
* Showing only the 3s shows nothing in 13 of 37 repositories, and its precision, 0.84, is no
  higher than that of every admitted paper, 0.83. Mean net@2 falls from +5.49 to +1.08.
* Within the score-2 band, the heuristic ranker puts 55.1 percent of the 412 informative pairs in
  the judge's order, [0.46, 0.62].

The pure functions are tested on hand-built rows. The artifact is rebuilt from the run file when
the run file is present, which it is in the development tree and in the data bundle, not in CI.
"""

from __future__ import annotations

import json
from pathlib import Path

import finescale_model_transfer as fmt
import gate_on_37 as g
import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "evals" / "gate_on_37.json"
RUN = ROOT / "evals" / "results" / fmt.BANDS["H"].run


def paper(gate: int, judge: int) -> dict[str, int]:
    return {"llm_score": gate, "judge_score": judge}


def entry(case: str, papers: list[dict[str, int]]) -> dict[str, object]:
    return {"case": case, "digest_window": 15, "returned": {"anonymous_top10": papers}}


class TestPureFunctions:
    def test_net_counts_plus_one_and_minus_two(self) -> None:
        assert g.net([paper(2, 2), paper(2, 3), paper(3, 1)]) == 0

    def test_case_rows_count_admitted_threes_and_ordered_pairs(self) -> None:
        # listed order is the ranker's: an accepted score-2 paper above a rejected one is ordered
        rows = g.case_rows(
            [entry("x", [paper(3, 2), paper(2, 2), paper(2, 0), paper(2, 3), paper(1, 3)])]
        )
        r = rows[0]
        assert (r["admitted"], r["threes"], r["actionable"], r["actionable_threes"]) == (4, 1, 3, 1)
        # score-2 papers at positions 1 (yes), 2 (no), 3 (yes): pairs (1,2) ordered, (3,2) not
        assert (r["band_pairs"], r["band_pairs_ordered"]) == (2, 1)
        assert (r["net_all"], r["net_threes"]) == (1, 1)

    def test_a_paper_outside_the_gate_is_not_admitted(self) -> None:
        r = g.case_rows([entry("x", [paper(1, 3), paper(0, 3)])])[0]
        assert (r["admitted"], r["band_pairs"]) == (0, 0)

    def test_refuses_an_unjudged_admitted_paper(self) -> None:
        with pytest.raises(SystemExit, match="no primary-judge score"):
            g.case_rows([entry("x", [{"llm_score": 2, "judge_score": None}])])

    def test_refuses_a_different_window(self) -> None:
        bad = entry("x", [paper(2, 2)])
        bad["digest_window"] = 10
        with pytest.raises(SystemExit, match="digest_window"):
            g.case_rows([bad])


class TestArtifact:
    @pytest.fixture(scope="class")
    def art(self) -> dict:
        return json.loads(ARTIFACT.read_text(encoding="utf-8"))["summary"]

    def test_share_of_threes(self, art: dict) -> None:
        assert art["share_of_threes"] == {"threes": 76, "admitted": 404, "share": 0.1881}

    def test_share_against_precision(self, art: dict) -> None:
        s = art["share_vs_precision"]
        assert (s["repositories"], s["pearson_r"]) == (34, 0.2478)
        assert s["ci95"] == [-0.0026, 0.5157]

    def test_only_threes_buys_no_precision(self, art: dict) -> None:
        p = art["policies"]
        assert p["repositories"] == 37
        assert p["show_only_threes"]["repositories_shown_nothing"] == 13
        assert (p["show_only_threes"]["precision"], p["show_admitted"]["precision"]) == (
            0.8421,
            0.8342,
        )
        assert (p["show_only_threes"]["mean_net2"], p["show_admitted"]["mean_net2"]) == (
            1.0811,
            5.4865,
        )

    def test_ranker_within_band(self, art: dict) -> None:
        k = art["ranker_within_band"]
        assert (k["repositories"], k["pairs"], k["auc"]) == (26, 412, 0.551)
        assert k["ci95"] == [0.4581, 0.6248]

    def test_names_its_run(self, art: dict) -> None:
        assert art["run"] == fmt.BANDS["H"].run

    @pytest.mark.skipif(not RUN.exists(), reason="the run file is in the data bundle, not in git")
    def test_rebuilds_from_the_run(self, art: dict) -> None:
        rows = g.case_rows(json.loads(RUN.read_text(encoding="utf-8")))
        stored = json.loads(ARTIFACT.read_text(encoding="utf-8"))["rows"]
        assert rows == stored
        nonempty = [r for r in rows if r["admitted"]]
        assert round(g.share_precision_r(nonempty), 4) == art["share_vs_precision"]["pearson_r"]
