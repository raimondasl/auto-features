"""Buying verdicts, with the cache gate travelling alongside. [PREREG §3.1, §4, §7, §10 step 7]

The purchase is factored out of `judge()` for two reasons. §7's cache clause binds every path
that buys a T0 verdict, not only the one that happened to exist when it was written — so the
gate has to live with the loop rather than at one call site a future entry point can forget.
And a whole run has to be drivable with fakes and no network, or none of the failure paths that
matter is ever exercised.

Every item exits with a recorded outcome, persisted as a LIST. The old loop kept an integer
`void` and printed it once, which cannot be checked against anything — and §10 step 7 publishes
the void and timeout lists in the datasheet. A judge whose verdicts are 60% complete produces
an AUC over a different sample from the other judge's, and §5 compares the two directly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / "evals", ROOT / "evals" / "frame", ROOT / "src"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import judge_validity_pool as jvp  # noqa: E402

DIGEST = "abc123456789"
CONTEXTS = {"g": ("Repository: g\n\n## README (excerpt)\nprose\n\n## Source files\na.py\n", DIGEST)}


def _items(n_pos: int = 2, n_ctl: int = 4) -> list[dict[str, object]]:
    pos = [
        {"case": "g", "arm": "adopted", "arxiv_id": f"2401.{i:05d}", "title": "t", "abstract": "a"}
        for i in range(n_pos)
    ]
    ctl = [
        {"case": "g", "arm": "control", "arxiv_id": f"2402.{i:05d}", "title": "t", "abstract": "a"}
        for i in range(n_ctl)
    ]
    return pos + ctl


def _run(tmp_path: Path, items, judges, contexts=None, gate=None, **over):  # noqa: ANN001, ANN202
    return jvp.buy_verdicts(
        items,
        contexts if contexts is not None else CONTEXTS,
        judges=judges,
        store=tmp_path / "verdicts.json",
        lock=tmp_path / "lock",
        done_out=tmp_path / "done.json",
        gate=gate if gate is not None else (lambda: []),
        **over,
    )


def _ok(score: int = 3):  # noqa: ANN202
    return lambda case, ctx, item, model: score


class TestTheGateTravelsWithThePurchase:
    def test_a_leak_stops_the_run(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit) as exc:
            _run(tmp_path, _items(), {"m": _ok()}, gate=lambda: ["THE JUDGE CACHE MOVED"])
        assert "THE JUDGE CACHE MOVED" in str(exc.value)

    def test_the_store_is_written_before_the_raise(self, tmp_path: Path) -> None:
        """A mid-run trip must abort without also losing what was already paid for. The gold
        cache is gitignored with no version history, so the gate's whole value is in firing
        early — and firing early is worthless if it costs the run's purchases."""
        with pytest.raises(SystemExit):
            _run(tmp_path, _items(), {"m": _ok()}, gate=lambda: ["leak"])
        store = json.loads((tmp_path / "verdicts.json").read_text(encoding="utf-8"))
        assert len(store) == 6, "verdicts bought before the trip must survive it"

    def test_the_gate_runs_at_every_checkpoint_not_only_at_the_end(self, tmp_path: Path) -> None:
        calls: list[int] = []
        with pytest.raises(SystemExit):
            _run(
                tmp_path,
                _items(n_pos=5, n_ctl=5),
                {"m": _ok()},
                gate=lambda: calls.append(1) or (["leak"] if len(calls) >= 2 else []),
                checkpoint=2,
            )
        assert len(calls) >= 2

    def test_a_clean_run_reports_full_coverage(self, tmp_path: Path) -> None:
        rec = _run(tmp_path, _items(), {"m": _ok()})
        assert rec["coverage"]["m"]["coverage"] == 1.0
        assert rec["outcomes"] == []

    def test_a_second_loop_against_one_store_is_refused(self, tmp_path: Path) -> None:
        """Two loops interleaving writes would each overwrite the other's verdicts."""
        (tmp_path / "lock").write_text("pid 1 started now", encoding="utf-8")
        with pytest.raises(SystemExit) as exc:
            _run(tmp_path, _items(), {"m": _ok()})
        assert "Another purchase run holds it" in str(exc.value)

    def test_the_lock_is_released_even_when_the_run_fails(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit):
            _run(tmp_path, _items(), {"m": _ok()}, gate=lambda: ["leak"])
        assert not (tmp_path / "lock").exists()


class TestTheLegacyStoreIsNeverRewritten:
    def test_writing_to_the_nr56_57_store_is_refused_by_name(self, tmp_path: Path) -> None:
        """566 paid records that §4's pool-scheme result is reported from. §1's rule for
        adoptions.json applies for the same reason: it is the record this study is compared
        against."""
        with pytest.raises(SystemExit) as exc:
            jvp._save_store(jvp.LEGACY_VERDICTS, {"k": {"score": 1}})
        assert "NR-56/57 verdict store" in str(exc.value)

    def test_a_pool_run_leaves_it_byte_identical(self, tmp_path: Path) -> None:
        if not jvp.LEGACY_VERDICTS.is_file():
            pytest.skip("evals/.work/ is gitignored; local only")
        before = jvp.LEGACY_VERDICTS.read_bytes()
        _run(tmp_path, _items(), {"m": _ok()})
        assert jvp.LEGACY_VERDICTS.read_bytes() == before


class TestAVerdictKnowsWhichPromptItAnswered:
    def test_a_matching_digest_is_reused(self, tmp_path: Path) -> None:
        _run(tmp_path, _items(), {"m": _ok()})

        def explode(*_a: object) -> int:
            raise AssertionError("re-bought a verdict that already answered this prompt")

        assert _run(tmp_path, _items(), {"m": explode})["bought"] == 0

    def test_a_digest_mismatch_re_buys_instead_of_reusing(self, tmp_path: Path) -> None:
        """The context is what makes a T0 verdict a T0 verdict."""
        _run(tmp_path, _items(), {"m": _ok(3)})
        moved = {"g": (CONTEXTS["g"][0], "ffffffffffff")}
        rec = _run(tmp_path, _items(), {"m": _ok(1)}, contexts=moved)
        assert rec["bought"] == 6
        store = json.loads((tmp_path / "verdicts.json").read_text(encoding="utf-8"))
        assert store["m|g|2401.00000"]["score"] == 1

    def test_every_record_carries_its_provenance(self, tmp_path: Path) -> None:
        _run(tmp_path, _items(), {"m": _ok()})
        store = json.loads((tmp_path / "verdicts.json").read_text(encoding="utf-8"))
        record = store["m|g|2401.00000"]
        assert set(record) >= {"score", "arm", "model", "case", "id", "context_digest", "scheme"}
        assert record["context_digest"] == DIGEST

    def test_an_arm_mismatch_raises_and_names_the_key(self, tmp_path: Path) -> None:
        """A stored flag nobody reads catches nothing — which is what the assigned-but-unread
        control scheme already demonstrated."""
        _run(tmp_path, _items(), {"m": _ok()})
        flipped = [{**i, "arm": "control"} for i in _items(n_pos=1, n_ctl=0)]
        with pytest.raises(SystemExit) as exc:
            _run(tmp_path, flipped, {"m": _ok()})
        assert "m|g|2401.00000" in str(exc.value)


class TestImportingAnOldVerdictIsDecidedByTheDigest:
    def test_a_recomputed_match_imports(self, tmp_path: Path) -> None:
        src = tmp_path / "legacy.json"
        src.write_text(
            json.dumps({"gpt-5.5|graph|2401.00001": {"score": 3, "arm": "adopted"}}),
            encoding="utf-8",
        )
        have: dict[str, object] = {}
        tally = jvp.import_legacy_verdicts(
            have, {"graph": DIGEST}, source=src, recompute=lambda case: DIGEST
        )
        assert tally["imported"] == 1
        assert have["gpt-5.5|graph|2401.00001"]["context_digest"] == DIGEST

    def test_a_recomputed_mismatch_is_not_imported(self, tmp_path: Path) -> None:
        src = tmp_path / "legacy.json"
        src.write_text(
            json.dumps({"gpt-5.5|graph|2401.00001": {"score": 3, "arm": "adopted"}}),
            encoding="utf-8",
        )
        have: dict[str, object] = {}
        tally = jvp.import_legacy_verdicts(
            have, {"graph": DIGEST}, source=src, recompute=lambda case: "different1234"
        )
        assert tally == {"imported": 0, "digest_mismatch": 1, "no_context": 0, "not_a_positive": 0}
        assert have == {}

    def test_no_control_verdict_is_ever_imported(self, tmp_path: Path) -> None:
        """All 496 are pool-scheme draws under the old prompt shape — versioned identifiers and
        unnormalised abstracts — so none is a verdict about a prompt this study would send."""
        src = tmp_path / "legacy.json"
        src.write_text(
            json.dumps({"gpt-5.5|graph|2402.00002": {"score": 0, "arm": "control"}}),
            encoding="utf-8",
        )
        have: dict[str, object] = {}
        tally = jvp.import_legacy_verdicts(
            have, {"graph": DIGEST}, source=src, recompute=lambda case: DIGEST
        )
        assert tally["not_a_positive"] == 1 and have == {}

    def test_the_gate_is_the_digest_and_not_the_sha(self) -> None:
        """§7 says the re-mine pins T0 so it "reproduces to the SHA". Measured 2026-09-03, that
        is FALSE for 2 of 9 cases — peft 4c3a76fa68 -> e8ba7de573 and llminfer d565bb2fd5 ->
        8b3befc0e2 — while the T0 CONTEXT at both pairs is byte-identical. Those commits differ
        in code, not in the documentation the prompt is built from, so a SHA gate would have
        discarded both cases for a difference the judge never saw."""
        doc = jvp.import_legacy_verdicts.__doc__ or ""
        assert "reproduces to the SHA" in doc and "byte-identical" in doc


class TestEveryItemExitsWithAnOutcome:
    def test_a_case_with_no_context_is_recorded_not_skipped(self, tmp_path: Path) -> None:
        rec = _run(tmp_path, _items(), {"m": _ok()}, contexts={})
        assert {o["outcome"] for o in rec["outcomes"]} == {"no_context"}
        assert len(rec["outcomes"]) == 6

    def test_a_judge_error_is_recorded_with_its_type(self, tmp_path: Path) -> None:
        def flaky(case, ctx, item, model):  # noqa: ANN001, ANN202
            if item["arxiv_id"] == "2401.00001":
                raise ValueError("bad request")
            return 3

        rec = _run(tmp_path, _items(), {"m": flaky})
        errors = [o for o in rec["outcomes"] if o["outcome"] == "judge_error"]
        assert [e["error"] for e in errors] == ["ValueError"]

    def test_a_timeout_is_its_own_outcome(self, tmp_path: Path) -> None:
        """§3.1 states the principle one stage earlier: "A timeout is a recorded outcome, never
        a silent skip"."""

        def slow(case, ctx, item, model):  # noqa: ANN001, ANN202
            raise TimeoutError

        rec = _run(tmp_path, _items(n_pos=1, n_ctl=0), {"m": slow})
        assert [o["outcome"] for o in rec["outcomes"]] == ["timeout"]

    def test_the_lists_are_persisted_not_summed(self, tmp_path: Path) -> None:
        """ "17 void" cannot be checked against anything, and §10 step 7 publishes the lists."""

        def flaky(case, ctx, item, model):  # noqa: ANN001, ANN202
            raise ValueError("nope")

        _run(tmp_path, _items(n_pos=1, n_ctl=1), {"m": flaky})
        done = json.loads((tmp_path / "done.json").read_text(encoding="utf-8"))
        assert [o["id"] for o in done["outcomes"]] == ["2401.00000", "2402.00000"]

    def test_coverage_is_reported_per_judge(self, tmp_path: Path) -> None:
        def only_gpt(case, ctx, item, model):  # noqa: ANN001, ANN202
            if model == "sonnet" and item["arm"] == "control":
                raise ValueError("400")
            return 3

        rec = _run(tmp_path, _items(), {"gpt": only_gpt, "sonnet": only_gpt})
        assert rec["coverage"]["gpt"]["coverage"] == 1.0
        assert rec["coverage"]["sonnet"]["n_void_recorded"] == 4
        assert rec["coverage"]["sonnet"]["coverage"] == pytest.approx(2 / 6, abs=1e-4)

    def test_an_unexplained_absence_raises(self) -> None:
        """One arXiv 503 drops a hundred ids with a single printed line, and a judge missing
        those papers produces an AUC over a different sample from the other judge's."""
        items = _items(n_pos=1, n_ctl=0)
        with pytest.raises(SystemExit) as exc:
            jvp.verdict_coverage(items, {}, [], judges=["m"])
        assert "neither present nor explained" in str(exc.value)

    def test_the_completion_record_is_what_unlocks_the_analysis(self, tmp_path: Path) -> None:
        """`refuse_to_peek` reads its absence as "judging has not finished"."""
        _run(tmp_path, _items(), {"m": _ok()})
        done = json.loads((tmp_path / "done.json").read_text(encoding="utf-8"))
        assert done["_artefact"] == jvp.ARTEFACT_MARKER
        assert done["n_items"] == 6 and done["bought"] == 6
