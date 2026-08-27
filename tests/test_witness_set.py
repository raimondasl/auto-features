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

# Written out rather than imported, and that is the point: `witness_set` derives its labels
# from whatever draws exist, so a new configuration appears in the artifact on its own. This
# set is the review gate on that — a label nobody declared here fails the suite, and whoever
# adds it has to decide whether it is a self source (excluded from every reach denominator)
# or a grading one. `SELF` is the only asymmetry in the design; getting a new source's side
# wrong is invisible in every number the artifact prints.
KNOWN_SOURCES = {
    "cli",
    "cli-redraw",
    "cli-redraw@30",
    "cli-v2@30",  # the v2-prompt sweep, 2026-08-26: grading, not self
    "cli-v2-opus5@30",  # Opus 5, draw 1 only (quota-truncated): grading, not self
    "api",
    "reporadar",
    "adoption",
}
SELF = {"reporadar"}
NON_SELF = KNOWN_SOURCES - SELF


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

    def test_the_source_list_matches_the_witnesses(self, artifact):
        """`sources` is derived, so it can drift from the set it claims to describe."""
        found = {
            s
            for papers in artifact["witnesses"].values()
            for m in papers.values()
            for s in m["sources"]
        }
        assert set(artifact["sources"]) == found
        assert found <= KNOWN_SOURCES, f"undeclared source label(s): {found - KNOWN_SOURCES}"

    def test_every_reach_table_grades_by_every_non_self_source_present(self, artifact):
        """The failure this guards is a source that is in the set but in no denominator.

        It would be invisible: every printed number stays well-formed and simply describes
        fewer witnesses than the artifact holds. A source that grades nothing is not a source
        scoring zero — void, not null — so the two are separated here by construction.
        """
        present = {s for s in artifact["sources"] if s not in SELF}
        for r in artifact["reach"]:
            assert set(r["graded_by"]) == present, r["pool"]

    def test_reach_rows_are_probabilities_with_sane_intervals(self, artifact):
        for r in artifact["reach"]:
            source_ns = []
            for label in (*r["graded_by"], "pooled_non_self"):
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

    def test_the_two_capture_estimators_never_merge(self, artifact):
        """One counts picks, the other counts witnesses. A reader who averages them is
        wrong, so each block states its own unit and they live under different keys."""
        cap = artifact["capture"]
        assert "pick" in cap["cli_draws"]["unit"]
        for label, c in cap["redraws"].items():
            assert "witness" in c["unit"], label
            assert c["occasions"] >= 2, f"{label}: one occasion estimates nothing"
            assert c["chao1_lower_bound"] >= c["s_obs"], label
            assert c["f1"] + c["f2"] <= c["s_obs"], label

    def test_the_headline_regret_figures(self, artifact):
        """The numbers the write-up quotes; a hand-edit fails here on any machine.

        Regret is a function of the witness set's SIZE, by design — it counts unshown
        witnesses that would fill or displace, so a larger set reveals more headroom (never
        inflates it: the digest window bounds it). The series, each figure correct at its own
        set size and none of them overwriting the last (the C-17 rule): **+3.48 over 319**
        witnesses from four sources, **+4.80 over 385** with the P17 v1 redraws pooled in,
        **+5.56 over 462** with the v2-prompt sweep as well, **+6.24 over 482** once the
        OpenAlex tier made 31 previously-unscoreable references judgeable, and **+7.28 over
        572** with Opus 5's (incomplete, 21-run) draw pooled in.

        The last step is worth separating from the others: it added no new *search*. Those
        papers had already been found and named by a searcher already in the pool — the set
        grew because the instrument could finally read them.
        """
        reg = artifact["regret"]
        assert reg["mean_actual_net2"] == 5.72, "net@2 reads the system's own returns; fixed"
        assert reg["mean_regret"] == 7.28
        assert artifact["n_witnesses"] == 572

    def test_the_headline_reach_figures(self, artifact):
        """`cli` at 8/56 is the load-bearing line: pooling 237 further witnesses in must not
        move the frozen gold-set source by one paper. Nor did it move `cli-redraw` or
        `adoption` — a source's reach is a property of that source, and the only figure a new
        source may change is the pooled one.

        And it did: pooled reach FELL from 0.174 to 0.149 when `cli-v2@30` joined, and again
        to 0.138 when the OpenAlex tier let 31 more of its references be judged. `cli-v2@30`
        itself sits at 0.123, the lowest of the family and *falling as it grows* — the
        non-arXiv papers the tier unlocked are ones the shipped pool holds even less often
        than the arXiv ones. A larger witness set lowering pooled reach is the measure
        working: it means the new witnesses are papers we do not fetch, which is the answer
        the question was asked for.
        """
        wemb = next(r for r in artifact["reach"] if r["pool"] == "pool-wemb")
        assert (wemb["cli"]["reached"], wemb["cli"]["n"]) == (8, 56)
        assert (wemb["adoption"]["reached"], wemb["adoption"]["n"]) == (1, 19)
        assert (wemb["cli-redraw"]["reached"], wemb["cli-redraw"]["n"]) == (19, 92)
        assert (wemb["cli-v2@30"]["reached"], wemb["cli-v2@30"]["n"]) == (19, 155)
        assert wemb["pooled_non_self"]["p"] == 0.133

    def test_opus5_is_a_separate_source_from_opus48(self, artifact):
        """A model is a searcher. Pooling the two under one label would hide the quantity
        that decides whether running a second model is worth anything — and it nearly did:
        `draw_source_label` handled the model axis while `_draw_rows` still enumerated prompt
        versions at the default model, so the Opus 5 artifact was never opened at all.
        `--check` passed, because the derivation and the artifact agreed about a file neither
        had read."""
        labels = set(artifact["sources"])
        assert {"cli-v2@30", "cli-v2-opus5@30"} <= labels
        wemb = next(r for r in artifact["reach"] if r["pool"] == "pool-wemb")
        assert wemb["cli-v2-opus5@30"]["n"] == 158, "21 runs, and more witnesses than 75 of 4.8's"

    def test_the_redraws_agree_with_cli_on_reach(self, artifact):
        """A consistency check that costs nothing and would catch a mislabelled source.

        `cli` and `cli-redraw` are the same prompt, model and cap — different draws of one
        searcher. Their reach into the pool is estimating the same quantity, so the intervals
        must overlap. If they ever stop, either a draw was filed under the wrong label or the
        pool is not the fixed object it is assumed to be.
        """
        wemb = next(r for r in artifact["reach"] if r["pool"] == "pool-wemb")
        lo_a, hi_a = wemb["cli"]["ci"]
        lo_b, hi_b = wemb["cli-redraw"]["ci"]
        assert lo_a <= hi_b and lo_b <= hi_a, "same searcher, disjoint reach intervals"


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
