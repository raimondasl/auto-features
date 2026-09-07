"""The cross-repository negative class (PREREG-judge-crossrepo-controls.md).

Written before the seed that will drive the draw exists, which is the point: the estimator and
the eligibility rule are fixed while no data can inform them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / "evals", ROOT / "evals" / "frame", ROOT / "src"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import judge_validity_pool as jvp  # noqa: E402

SEED = "XREPO-PULSE-FIXTURE"
W = ("cs.LG", "202401010000")


def _paper(pid: str) -> dict[str, object]:
    return {"arxiv_id": pid, "title": f"T{pid}", "abstract": "a" * 400, "primary_category": "cs.LG"}


class TestTheEligibilityRuleIsTheRegisteredOne:
    def test_a_control_must_be_cited_by_another_repository(self) -> None:
        pos = [{"case": "A", "id": "1"}]
        # `9` is cited by nobody; `3` is cited by C. Only `3` is eligible.
        out = jvp.crossrepo_eligible(pos, {"A": {"1"}, "C": {"3"}}, {"1": W, "3": W, "9": W})
        assert out[("A", "1")] == ["3"]

    def test_a_paper_the_focal_repository_cites_is_never_a_control(self) -> None:
        """Clause 3. Without it a paper A actually references sits in A's negative class, which
        drags the AUC toward 0.5 — the null this study is pre-committed to reporting, arriving
        by accident and looking like a finding."""
        pos = [{"case": "A", "id": "1"}]
        head = {"A": {"1", "9"}, "B": {"9", "3"}}
        out = jvp.crossrepo_eligible(pos, head, {"1": W, "3": W, "9": W})
        assert out[("A", "1")] == ["3"], "9 is cited by both A and B; A's copy disqualifies it"

    def test_the_positive_is_never_its_own_control(self) -> None:
        pos = [{"case": "A", "id": "1"}]
        out = jvp.crossrepo_eligible(pos, {"A": {"1"}, "B": {"1", "3"}}, {"1": W, "3": W})
        assert out[("A", "1")] == ["3"]

    def test_the_window_must_match_exactly(self) -> None:
        """Clause 2 keeps the arm marker shut: an arXiv id encodes its month, so a control from
        a neighbouring window is distinguishable from a positive by its id alone."""
        pos = [{"case": "A", "id": "1"}]
        head = {"A": {"1"}, "B": {"3", "4"}}
        windows = {"1": W, "3": ("cs.LG", "202407010000"), "4": ("cs.CV", "202401010000")}
        assert jvp.crossrepo_eligible(pos, head, windows)[("A", "1")] == []

    def test_a_positive_with_no_window_yields_an_empty_class_not_a_crash(self) -> None:
        out = jvp.crossrepo_eligible([{"case": "A", "id": "1"}], {"B": {"3"}}, {"3": W})
        assert out[("A", "1")] == []


class TestTheDrawIsAFunctionOfTheSeedAndNothingElse:
    def _draw(self, tmp_path: Path, seed: str = SEED, per: int = 2) -> list[dict[str, object]]:
        pos = [{"case": "A", "id": "1"}]
        head = {"A": {"1"}, "B": {"3", "4"}, "C": {"5", "6"}}
        windows = {k: W for k in ("1", "3", "4", "5", "6")}
        papers = {k: _paper(k) for k in ("3", "4", "5", "6")}
        return jvp.crossrepo_controls(
            pos,
            seed=seed,
            head_ids=head,
            windows=windows,
            papers=papers,
            per_positive=per,
            rows_out=tmp_path / f"rows-{seed}.json",
            payload_out=tmp_path / f"payload-{seed}.json",
        )

    def test_the_same_seed_gives_the_same_controls(self, tmp_path: Path) -> None:
        a = self._draw(tmp_path / "a")
        b = self._draw(tmp_path / "b")
        assert [r["id"] for r in a] == [r["id"] for r in b]

    def test_a_different_seed_gives_a_different_draw(self, tmp_path: Path) -> None:
        a = {r["id"] for r in self._draw(tmp_path / "a")}
        b = {r["id"] for r in self._draw(tmp_path / "b", seed="OTHER-PULSE")}
        assert a != b, "the draw does not depend on the seed"

    def test_a_stored_draw_under_another_seed_is_refused_not_regenerated(
        self, tmp_path: Path
    ) -> None:
        """A redraw after verdicts were bought changes n2 — an instrument change wearing the
        clothes of a rerun."""
        d = tmp_path / "same"
        self._draw(d)
        with pytest.raises(SystemExit) as exc:
            jvp.crossrepo_controls(
                [{"case": "A", "id": "1"}],
                seed="A-DIFFERENT-PULSE",
                head_ids={"A": {"1"}, "B": {"3"}},
                windows={"1": W, "3": W},
                papers={"3": _paper("3")},
                rows_out=d / f"rows-{SEED}.json",
                payload_out=d / f"payload-{SEED}.json",
            )
        assert "different seed" in str(exc.value)

    def test_the_committed_half_carries_no_paper_text(self, tmp_path: Path) -> None:
        """§2.1 keeps abstracts and URLs out of the tree; the payload beside it is untracked."""
        import json

        self._draw(tmp_path)
        stored = json.loads((tmp_path / f"rows-{SEED}.json").read_text(encoding="utf-8"))
        assert stored["controls"] and all("paper" not in r for r in stored["controls"])
        assert stored["scheme"] == "crossrepo"

    def test_a_positive_with_an_empty_class_is_counted_not_hidden(self, tmp_path: Path) -> None:
        import json

        pos = [{"case": "A", "id": "1"}, {"case": "Z", "id": "8"}]
        jvp.crossrepo_controls(
            pos,
            seed=SEED,
            head_ids={"A": {"1"}, "B": {"3"}, "Z": {"8"}},
            windows={"1": W, "3": W, "8": ("cs.CV", "202401010000")},
            papers={"3": _paper("3")},
            rows_out=tmp_path / "r.json",
            payload_out=tmp_path / "p.json",
        )
        stored = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
        assert stored["n_positives_offered"] == 2
        assert stored["n_positives_with_a_control"] == 1
        assert stored["n_positives_excluded_empty_class"] == 1


class TestTheSchemeContrastIsPairedOnIdenticalPositives:
    """`judge_difference` pairs two judges over one negative class; this pairs two negative
    classes over one judge, and the positive rows are literally the same verdicts."""

    MODEL = "m"

    def _world(self, n_cases: int = 6) -> tuple[list, list, list, dict]:
        pos, ca, cb, verdicts = [], [], [], {}
        for c in range(n_cases):
            case = f"case{c}"
            pid = f"p{c}"
            pos.append({"case": case, "id": pid, "stratum": "pool"})
            verdicts[jvp.verdict_key(self.MODEL, case, pid)] = {"score": 3, "arm": "adopted"}
            for j in range(2):
                easy, hard = f"e{c}{j}", f"h{c}{j}"
                ca.append({"case": case, "id": easy, "for_positive": pid})
                cb.append({"case": case, "id": hard, "for_positive": pid})
                verdicts[jvp.verdict_key(self.MODEL, case, easy)] = {"score": 0, "arm": "control"}
                # One hard control ties the positive and one sits below it, so the harder class
                # overlaps rather than merely sitting lower — a class that is uniformly one
                # notch down still separates perfectly and would not exercise the contrast.
                verdicts[jvp.verdict_key(self.MODEL, case, hard)] = {
                    "score": 3 if j == 0 else 2,
                    "arm": "control",
                }
        return pos, ca, cb, verdicts

    def test_a_harder_negative_class_shows_up_as_a_positive_delta(self) -> None:
        pos, ca, cb, v = self._world()
        out = jvp.scheme_difference(self.MODEL, pos, ca, cb, v, iters=300)
        assert out["auc"]["arxiv-window"] == 1.0, "score 3 vs 0 separates perfectly"
        assert out["auc"]["crossrepo"] < out["auc"]["arxiv-window"]
        assert out["delta_auc"] > 0 and out["excludes_zero"] is True

    def test_positives_absent_from_either_scheme_are_dropped_from_both(self) -> None:
        """§2 registers the analysis set as the positives whose cross-repo class is non-empty,
        and §4 requires the category-matched arm to be recomputed over that same subset — so
        the published full-set AUC is not the left operand of this difference."""
        pos, ca, cb, v = self._world()
        pos.append({"case": "lonely", "id": "px", "stratum": "pool"})
        v[jvp.verdict_key(self.MODEL, "lonely", "px")] = {"score": 3, "arm": "adopted"}
        ca.append({"case": "lonely", "id": "ex", "for_positive": "px"})
        v[jvp.verdict_key(self.MODEL, "lonely", "ex")] = {"score": 0, "arm": "control"}
        out = jvp.scheme_difference(self.MODEL, pos, ca, cb, v, iters=300)
        assert out["n_positives_offered"] == 7
        assert out["n_positives_shared"] == 6, "the positive with only one scheme is dropped"
        assert out["n_clusters"] == 6

    def test_one_cluster_draw_serves_both_arms(self) -> None:
        """Resampling the two arms independently would add variance that is not in the
        contrast, and let them disagree about which repositories exist in a draw. With
        identical control sets the delta is then exactly zero in every draw."""
        pos, ca, _, v = self._world()
        out = jvp.scheme_difference(self.MODEL, pos, ca, list(ca), v, iters=300)
        assert out["delta_auc"] == 0.0
        assert out["ci95"] == [0.0, 0.0] and out["excludes_zero"] is False

    def test_too_few_clusters_refuses_rather_than_returning_a_number(self) -> None:
        pos, ca, cb, v = self._world(n_cases=1)
        out = jvp.scheme_difference(self.MODEL, pos, ca, cb, v, iters=300)
        assert "_refused" in out and "delta_auc" not in out


class TestTheSeedIsCheckedAgainstItsOwnPulse:
    def test_a_missing_seed_file_names_the_pulse_and_refuses(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit) as exc:
            jvp.xrepo_seed(tmp_path / "absent")
        assert jvp.XREPO_PULSE in str(exc.value)

    def test_an_empty_seed_file_is_not_a_seed(self, tmp_path: Path) -> None:
        f = tmp_path / "SEED_XREPO"
        f.write_text("   \n", encoding="utf-8")
        with pytest.raises(SystemExit) as exc:
            jvp.xrepo_seed(f)
        assert "empty" in str(exc.value)

    def test_the_refusal_names_SEED_XREPO_not_SEED_POOL(self, tmp_path: Path) -> None:
        """The check is shared with the pool study deliberately — a second implementation could
        disagree with the first about the same file. What must not be shared is the message: a
        refusal naming SEED_POOL and section 2.4 sends the operator to the wrong file and the
        wrong document."""
        import walk_pool

        with pytest.raises(SystemExit) as exc:
            walk_pool.verify_seed(
                "DEADBEEF",
                jvp.XREPO_PULSE,
                lambda _p: "CAFE",
                name="SEED_XREPO",
                section="§3 of PREREG-judge-crossrepo-controls",
            )
        msg = str(exc.value)
        assert "SEED_XREPO" in msg and "SEED_POOL" not in msg
        assert "PREREG-judge-crossrepo-controls" in msg

    def test_the_pool_seed_refusal_is_unchanged(self) -> None:
        """The default arguments must keep the existing message byte-for-byte in spirit —
        every runbook and every prior incident note points at that wording."""
        import walk_pool

        with pytest.raises(SystemExit) as exc:
            walk_pool.verify_seed("AAA", "2026-09-04T00:00:00Z", lambda _p: "BBB")
        msg = str(exc.value)
        assert "SEED_POOL" in msg and "section 2.4" in msg

    def test_a_verified_seed_is_returned(self, tmp_path: Path) -> None:
        f = tmp_path / "SEED_XREPO"
        f.write_text("ABC123\n", encoding="utf-8")
        seen: dict[str, object] = {}

        def fake(seed: str, pulse: str, *a: object, **kw: object) -> None:
            seen.update({"seed": seed, "pulse": pulse, **kw})

        assert jvp.xrepo_seed(f, verifier=fake) == "ABC123"
        assert seen["pulse"] == jvp.XREPO_PULSE and seen["name"] == "SEED_XREPO"
