"""The pooled witness set (v2), pinned [P16].

`evals/witness_set.json` is the committed record of every judged-actionable paper any
measured source has surfaced, with provenance, plus the derived measures (reach, regret,
capture). Most of its inputs are gitignored -- baseline caches, judge verdicts, frozen
pools, the headline run -- so without these pins the documents quoting it would cite
numbers nothing in the repository can reproduce, the failure `gold_targets.json` and
`restated_runs.json` were built against.

Two layers, as in the other artifact tests:

* **Always** -- internal consistency of the committed JSON: counts add up, every witness is
  judged >= 2 with at least one known source, the arithmetic of regret holds per case, and
  -- the load-bearing one -- the `cli` source is EXACTLY the gold set. v2 must not quietly
  fork v1: a witness in one and not the other means two derivations of "what did the
  baseline pick" have drifted, which is the C-12/C-14 shape (and the first draft of the
  builder had exactly that bug -- it re-implemented the pick replay and dropped `rag`'s two
  ids-only orphans).
* **When the local inputs exist** -- the live derivation against the artifact, and one
  invariant that doubles as a self-test of the pool matching: every `reporadar`-sourced
  witness sits in its own run's frozen pool by construction, so a single miss there means
  the id normalisation is broken, not that the pool is small.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))

FROZEN = ROOT / "evals" / "witness_set.json"
GOLD = ROOT / "evals" / "gold_targets.json"

KNOWN_SOURCES = {"cli", "api", "reporadar", "adoption"}
NON_SELF = {"cli", "api", "adoption"}


@pytest.fixture(scope="module")
def artifact() -> dict:
    return json.loads(FROZEN.read_text(encoding="utf-8"))


class TestTheCommittedArtifact:
    def test_counts_add_up(self, artifact):
        witnesses = artifact["witnesses"]
        assert artifact["n_cases"] == len(witnesses)
        assert artifact["n_witnesses"] == sum(len(p) for p in witnesses.values())
        by_source: dict[str, int] = {}
        non_self = 0
        for papers in witnesses.values():
            for meta in papers.values():
                for s in meta["sources"]:
                    by_source[s] = by_source.get(s, 0) + 1
                non_self += any(s in NON_SELF for s in meta["sources"])
        assert by_source == artifact["by_source"]
        assert non_self == artifact["n_non_self"]

    def test_every_witness_is_certified(self, artifact):
        """A witness is a certificate: judged actionable, by a source we can name."""
        for case, papers in artifact["witnesses"].items():
            for pid, meta in papers.items():
                assert meta["judge"] >= 2, f"{case}/{pid} is not actionable"
                assert meta["sources"], f"{case}/{pid} has no source"
                assert set(meta["sources"]) <= KNOWN_SOURCES, f"{case}/{pid}: {meta['sources']}"
                assert len(meta["sources"]) == len(set(meta["sources"]))

    def test_the_cli_source_is_exactly_the_gold_set(self, artifact):
        """v2 must not fork v1 -- one derivation of "what did the baseline pick"."""
        gold = json.loads(GOLD.read_text(encoding="utf-8"))["targets"]
        cli = {
            case: sorted(pid for pid, m in papers.items() if "cli" in m["sources"])
            for case, papers in artifact["witnesses"].items()
        }
        cli = {c: ids for c, ids in cli.items() if ids}
        assert cli == {c: sorted(v) for c, v in gold.items()}

    def test_reach_rows_are_probabilities_with_sane_intervals(self, artifact):
        for r in artifact["reach"]:
            source_ns = []
            for label in (*NON_SELF, "pooled_non_self"):
                row = r[label]
                assert 0 <= row["reached"] <= row["n"]
                if row["n"]:
                    assert row["p"] == pytest.approx(row["reached"] / row["n"], abs=5e-4)
                    lo, hi = row["ci"]
                    assert 0.0 <= lo <= row["p"] <= hi <= 1.0
                if label != "pooled_non_self":
                    source_ns.append(row["n"])
            # pooled counts DISTINCT witnesses; sources overlap, so it is bounded by both.
            pooled = r["pooled_non_self"]["n"]
            assert max(source_ns) <= pooled <= sum(source_ns)

    def test_regret_arithmetic_holds_per_case(self, artifact):
        reg = artifact["regret"]
        assert reg["window"] == 15
        for case, row in reg["per_case"].items():
            assert row["regret"] == row["fills"] + 3 * row["swaps"], case
            assert row["fills"] + row["swaps"] <= row["witnesses_available"], case
        n = len(reg["per_case"])
        assert reg["mean_regret"] == pytest.approx(
            sum(r["regret"] for r in reg["per_case"].values()) / n, abs=5e-3
        )

    def test_capture_is_a_lower_bound_and_the_histogram_is_complete(self, artifact):
        cap = artifact["capture"]
        c = cap["cli_draws"]
        assert c["chao1_lower_bound"] >= c["s_obs"]
        assert c["f1"] + c["f2"] <= c["s_obs"]
        assert "pick" in c["unit"], "draw-level captures are picks, and must say so"
        assert sum(cap["source_overlap"].values()) == artifact["n_witnesses"]

    def test_the_headline_regret_figures(self, artifact):
        """The numbers the write-up quotes; a hand-edit fails here on any machine."""
        reg = artifact["regret"]
        assert reg["mean_actual_net2"] == 5.72
        assert reg["mean_regret"] == 3.48

    def test_the_headline_reach_figures(self, artifact):
        wemb = next(r for r in artifact["reach"] if r["pool"] == "pool-wemb")
        assert (wemb["cli"]["reached"], wemb["cli"]["n"]) == (8, 56)
        assert (wemb["adoption"]["reached"], wemb["adoption"]["n"]) == (1, 19)
        assert wemb["pooled_non_self"]["p"] == 0.14


class TestTheLiveDerivation:
    """Against the local caches and pools, where this machine has them."""

    @pytest.fixture(scope="class")
    def live(self) -> dict:
        import witness_set

        if not any((ROOT / "evals" / "cache" / "baseline" / "cli").glob("*.json")):
            pytest.skip("no local baseline cache")
        return witness_set.gather_witnesses()

    def test_the_witnesses_still_derive(self, artifact, live):
        rebuilt = {
            case: {pid: live[case][pid] for pid in sorted(live[case])} for case in sorted(live)
        }
        assert rebuilt == artifact["witnesses"]

    def test_reporadar_witnesses_are_in_their_own_pool(self, artifact):
        """By construction -- so a miss here means the id matching broke, nothing else."""
        import witness_set

        from reporadar.paper_id import dedup_id  # noqa: F401  (parity with builder)

        pools = witness_set._pool_ids("pool-wemb")
        if not pools:
            pytest.skip("no local pool-wemb")
        for case, papers in artifact["witnesses"].items():
            if case not in pools:
                continue
            for pid, meta in papers.items():
                if meta["sources"] == ["reporadar"]:
                    assert pid in pools[case], f"{case}/{pid}: id normalisation is broken"
