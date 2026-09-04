"""The wrapper reads the walk's output, and refuses to read it early. [PREREG §2.4, §3.1-3.3, §4]

Everything here is between the walk and the first verdict. The walk writes six artefacts and
then deletes its clones; from that point the study exists only as those files, so every way of
misreading them is a way of computing a plausible number from the wrong data — and the two that
matter most are silent. A missing `counted` key reads as False and shrinks n. An empty
`head_ids` file turns §4's "not cited anywhere at HEAD" into a no-op, puts positives into the
negative class, and biases the AUC toward 0.5 — which is the outcome §5 pre-commits to
reporting, arriving by accident and looking exactly like a finding.
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
import walk_pool as wp  # noqa: E402

SEED = "PULSE-FIXTURE"

# The legacy artefact lives under `evals/.work/`, which is gitignored — so every assertion
# about it would be skipped in CI, and the cap and contest logic would be pinned only on this
# machine. The fixture carries the REAL shape (120 rows, 94 usable, the same per-case counts,
# and `2107.03006` usable in both `diffusion` and `llminfer`) with everything the plumbing does
# not read stripped out. `test_the_fixture_still_matches_the_real_artefact` is what keeps the
# two from drifting; it runs wherever the real file exists.
LEGACY_SHAPE = ROOT / "tests" / "fixtures" / "adoptions-v2-shape.json"


def _adoption(case: str, pid: str, **over: object) -> dict[str, object]:
    row = {
        "case": case,
        "id": pid,
        "usable": True,
        "genesis": False,
        "in_cap": True,
        "counted": True,
        "assigned_to": case,
    }
    row.update(over)
    return row


class TestTheSeedIsVerifiedBeforeItSelectsAnything:
    def test_an_absent_seed_file_names_the_pulse_and_the_runbook_step(self, tmp_path) -> None:
        with pytest.raises(SystemExit) as exc:
            jvp.pool_seed(tmp_path / "SEED_POOL", verify=False)
        assert wp.REGISTERED_PULSE in str(exc.value)
        assert "step 4" in str(exc.value)

    def test_an_empty_seed_file_is_refused(self, tmp_path) -> None:
        path = tmp_path / "SEED_POOL"
        path.write_text("   \n", encoding="utf-8")
        with pytest.raises(SystemExit) as exc:
            jvp.pool_seed(path, verify=False)
        assert "empty" in str(exc.value)

    def test_it_verifies_against_the_registered_pulse(self, tmp_path) -> None:
        """`walk_pool.verify_seed` is used rather than reimplemented: the walk ordered 17,888
        candidates with it, and a second implementation could disagree with the first about the
        same file. The verifier is injected rather than monkeypatched onto the module, because
        `tests/test_pool_walk.py` re-registers its own `walk_pool` in `sys.modules` and a patch
        would land on whichever object happened to be there."""
        path = tmp_path / "SEED_POOL"
        path.write_text("ABCDEF", encoding="utf-8")
        seen: list[tuple[str, str]] = []
        assert jvp.pool_seed(path, verifier=lambda s, p: seen.append((s, p))) == "ABCDEF"
        assert seen == [("ABCDEF", wp.REGISTERED_PULSE)]

    def test_a_seed_that_fails_verification_never_returns(self, tmp_path) -> None:
        path = tmp_path / "SEED_POOL"
        path.write_text("TRUNCATED", encoding="utf-8")

        def boom(*_a: object) -> None:
            raise SystemExit("does not match the beacon pulse")

        with pytest.raises(SystemExit):
            jvp.pool_seed(path, verifier=boom)

    def test_the_default_verifier_is_the_walks_own(self) -> None:
        assert jvp.pool_seed.__doc__ and "verify_seed" in jvp.pool_seed.__doc__
        import inspect

        assert "walk_pool.verify_seed" in inspect.getsource(jvp.pool_seed)


class TestNoEndpointIsInspectedBeforeTheStopRuleFires:
    """§3.3, last line. An endpoint computed mid-walk is not a preview of the final one — it is
    a result read at a moment chosen by whoever ran it, which is the discretion the frozen order
    and the unchoosable pulse exist to remove."""

    MID = {"walked": 300, "budget": 1200, "target": 100, "capped_positives": 40}

    def test_a_walk_in_progress_has_no_stop_reason(self) -> None:
        assert jvp.walk_stop_reason(self.MID, n_candidates=17888) is None

    def test_the_target_stops_it(self) -> None:
        s = {**self.MID, "capped_positives": 100}
        assert jvp.walk_stop_reason(s, n_candidates=17888) == "target"

    def test_the_budget_stops_it_once_the_reporting_minimum_is_in_hand(self) -> None:
        s = {**self.MID, "walked": 1200, "capped_positives": 70}
        assert jvp.walk_stop_reason(s, n_candidates=17888) == "budget"

    def test_reaching_the_budget_below_sixty_is_a_checkpoint_not_a_stop(self) -> None:
        """§3.2 runs the walk to B "or until ... 100, whichever comes first", but §3.4 then
        says that if B is reached below 60 the walk **continues down the already-frozen seeded
        order, to the end of the list**. Treating B as an unconditional stop would let a short
        walk be analysed at n = 40 while §3.4 was still asking for rows — and §9's power table
        is exactly where that shortfall does its damage."""
        s = {**self.MID, "walked": 1200, "capped_positives": 40}
        assert jvp.walk_stop_reason(s, n_candidates=17888) is None
        assert jvp.walk_stop_reason({**s, "capped_positives": 60}) == "budget"

    def test_exhausting_the_list_stops_it_at_whatever_n_exists(self) -> None:
        """§3.4's recorded negative result: the list is spent, so n is n."""
        s = {**self.MID, "walked": 17888, "capped_positives": 12}
        assert jvp.walk_stop_reason(s, n_candidates=17888) == "exhausted"

    def test_the_refusal_says_when_the_budget_is_spent_but_the_walk_goes_on(self, tmp_path) -> None:
        s = {**self.MID, "walked": 1200, "capped_positives": 40}
        with pytest.raises(SystemExit) as exc:
            jvp.refuse_to_peek(s, n_candidates=17888, judging_done=tmp_path / "none.json")
        assert "§3.4" in str(exc.value)

    def test_the_target_comes_from_the_registration_not_the_summary(self, tmp_path) -> None:
        """`walk_summary.json` is written from `--target` on every run and guarded by nothing,
        so reading the bar out of it let the gate take its threshold from the artefact it is
        gating: a rehearsal run with `--target 40` writes {"target": 40, "capped_positives": 40}
        and was blessed as having met the target — at ~71 positives instead of the registered
        130, where §9's own power table says the interval may include 0.5 for a sampling
        reason. The seed is not trusted to its file either."""
        rehearsal = {**self.MID, "target": 40, "capped_positives": 40}
        with pytest.raises(SystemExit) as exc:
            jvp.walk_stop_reason(rehearsal, n_candidates=17888)
        assert "40" in str(exc.value) and str(wp.DEFAULT_TARGET) in str(exc.value)

    def test_a_budget_below_the_registered_b_is_refused(self) -> None:
        """A ten-row walk would otherwise satisfy the budget branch outright."""
        with pytest.raises(SystemExit) as exc:
            jvp.walk_stop_reason({**self.MID, "walked": 10, "budget": 10, "capped_positives": 60})
        assert "below" in str(exc.value)

    def test_a_budget_extended_past_b_is_allowed(self) -> None:
        """§3.4 executes by re-running past B, and the only way `walk()` can do that is a
        larger `--budget`. The bar may grow and only grow."""
        s = {**self.MID, "walked": 2400, "budget": 2400, "capped_positives": 70}
        assert jvp.walk_stop_reason(s, n_candidates=17888) == "budget"

    def test_the_stop_record_carries_the_thresholds_that_produced_it(self, tmp_path) -> None:
        """So a datasheet reading "stop_reason: target" can be checked against WHICH target."""
        done = tmp_path / "done.json"
        done.write_text(json.dumps({"bought": 1300}), encoding="utf-8")
        s = {**self.MID, "b0": 300, "capped_positives": 100}
        rec = jvp.refuse_to_peek(s, n_candidates=17888, judging_done=done)
        assert rec["target"] == 100 and rec["budget"] == 1200 and rec["b0"] == 300
        assert rec["reporting_minimum"] == jvp.REPORTING_MINIMUM

    def test_a_running_walk_is_refused_and_told_where_it_is(self, tmp_path) -> None:
        with pytest.raises(SystemExit) as exc:
            jvp.refuse_to_peek(self.MID, n_candidates=17888, judging_done=tmp_path / "none.json")
        message = str(exc.value)
        assert "300" in message and "1200" in message and "40" in message
        assert "§3.3" in message

    def test_a_stopped_walk_with_judging_unfinished_is_still_refused(self, tmp_path) -> None:
        """Half the gate is no gate: an endpoint over however many verdicts happened to be
        bought when someone looked is the same discretion under a different name."""
        s = {**self.MID, "capped_positives": 100}
        with pytest.raises(SystemExit) as exc:
            jvp.refuse_to_peek(s, n_candidates=17888, judging_done=tmp_path / "absent.json")
        assert "judging has not finished" in str(exc.value)

    def test_both_conditions_met_returns_the_stop_record(self, tmp_path) -> None:
        done = tmp_path / "judging_complete.json"
        done.write_text(json.dumps({"bought": 1300, "void": 2}), encoding="utf-8")
        s = {**self.MID, "capped_positives": 100}
        rec = jvp.refuse_to_peek(s, n_candidates=17888, judging_done=done)
        assert rec["stop_reason"] == "target"
        assert rec["stop_rule_capped_positives"] == 100
        assert rec["judging_complete"]["bought"] == 1300


class TestAPositiveIsAllFourTerms:
    def test_all_four_must_be_present_on_every_row(self, tmp_path) -> None:
        """`row.get("counted")` returns None on a partially written artefact, None is falsy, and
        the positive silently leaves the study. A walk interrupted between mining and assignment
        writes exactly that shape."""
        src = tmp_path / "adoptions-pool-v2.json"
        row = _adoption("owner/repo", "2401.00001")
        del row["counted"]
        src.write_text(json.dumps([row]), encoding="utf-8")
        with pytest.raises(SystemExit) as exc:
            jvp.pool_positives(src)
        assert "counted" in str(exc.value) and "owner/repo" in str(exc.value)

    @pytest.mark.parametrize(
        ("field", "value"),
        [("usable", False), ("genesis", True), ("in_cap", False), ("counted", False)],
    )
    def test_each_term_alone_disqualifies(self, field: str, value: object) -> None:
        assert jvp.is_positive(_adoption("o/r", "2401.00001")) is True
        assert jvp.is_positive(_adoption("o/r", "2401.00001", **{field: value})) is False

    def test_genesis_is_read_even_though_the_walk_hard_codes_it(self) -> None:
        """§1's filter table lists it separately and `walk_row` does not fold it into `usable`,
        so the definition here matches the registered table rather than the shortcut."""
        assert "genesis" in jvp.POSITIVE_TERMS


class TestTheLegacyStratumIsCappedThenContested:
    """Cap first, contest second — byte-identical to the order `walk_row` and
    `merge_adoptions` apply for a pool repository. `walk_row` sets `in_cap` before any
    cross-repository comparison exists, and the assignment re-runs over the whole file
    afterwards. Contest-first would let the cap refill the slot a contested paper vacated and
    yield MORE positives: both the wrong order and the unconservative one."""

    def test_the_real_legacy_artefact_reduces_as_measured(self) -> None:
        rows, ledger = jvp.legacy_positives(SEED, LEGACY_SHAPE)
        assert ledger["n_rows"] == 120
        assert ledger["n_usable"] == 94
        assert ledger["n_usable_in_cap"] == 32
        assert ledger["n_usable_over_cap"] == 62
        # 31 or 32, and which one is a property of the pulse rather than of this code:
        # 2107.03006 is usable in both `diffusion` and `llminfer`, so it costs a positive only
        # when the seeded cap keeps it in both. The invariant is what is asserted.
        assert ledger["n_positives"] == len(rows)
        assert ledger["n_positives"] == ledger["n_usable_in_cap"] - ledger["n_contested_lost"]
        assert ledger["n_positives"] in (31, 32)

    def test_no_case_contributes_more_than_the_cap(self) -> None:
        rows, _ = jvp.legacy_positives(SEED, LEGACY_SHAPE)
        counts: dict[str, int] = {}
        for row in rows:
            counts[row["case"]] = counts.get(row["case"], 0) + 1
        assert max(counts.values()) <= wp.PER_REPO_CAP

    def test_the_cap_is_what_keeps_one_project_from_dominating(self) -> None:
        """Uncapped, `diffusion` supplies 46 of 94 — a largest-cluster share of 0.489, and
        every registered co-report (cluster count, largest share, design effect, minimum
        detectable AUC) would be computed on a set §9 does not describe."""
        rows, _ = jvp.legacy_positives(SEED, LEGACY_SHAPE)
        share = max(
            sum(1 for r in rows if r["case"] == c) for c in {r["case"] for r in rows}
        ) / len(rows)
        assert share < 0.30

    def test_it_is_deterministic_under_one_seed(self) -> None:
        a, _ = jvp.legacy_positives(SEED, LEGACY_SHAPE)
        b, _ = jvp.legacy_positives(SEED, LEGACY_SHAPE)
        assert [r["id"] for r in a] == [r["id"] for r in b]

    def test_a_different_pulse_selects_a_different_set(self) -> None:
        a, _ = jvp.legacy_positives("PULSE-A", LEGACY_SHAPE)
        b, _ = jvp.legacy_positives("PULSE-B", LEGACY_SHAPE)
        assert {r["id"] for r in a} != {r["id"] for r in b}

    def test_the_contest_sees_every_row_as_the_walk_does(self, tmp_path) -> None:
        """`merge_adoptions` passes the WHOLE artefact to `assign_across_repos`, so a
        non-usable or over-cap row can win an identifier and knock out a capped one. Running a
        narrower contender set here would make `counted` mean different things in the two
        strata while §5's transportability endpoint contrasts them — and would hide exactly
        that loss."""
        src = tmp_path / "adoptions-v2.json"
        rows = [
            {"case": "aaa", "id": "2401.00001", "usable": False, "genesis": False},
            {"case": "zzz", "id": "2401.00001", "usable": True, "genesis": False},
        ]
        src.write_text(json.dumps(rows), encoding="utf-8")
        seeds = ["S1", "S2", "S3", "S4", "S5"]
        outcomes = {len(jvp.legacy_positives(s, src)[0]) for s in seeds}
        # Under a seed where the non-usable `aaa` row wins, the paper is counted zero times.
        assert 0 in outcomes, "the non-usable row must be able to win the contest"

    def test_a_legacy_artefact_missing_a_mined_field_is_refused(self, tmp_path) -> None:
        """`usable` and `genesis` are §1 filter results only mining can produce. `in_cap` and
        `counted` are absent by design and re-derived; `setdefault`ing the mined ones too would
        let a truncated artefact through the legacy path that the pool path refuses."""
        src = tmp_path / "adoptions-v2.json"
        src.write_text(json.dumps([{"case": "graph", "id": "2401.00001"}]), encoding="utf-8")
        with pytest.raises(SystemExit) as exc:
            jvp.legacy_positives(SEED, src)
        assert "usable" in str(exc.value) and "genesis" in str(exc.value)

    def test_the_source_artefact_is_never_written(self) -> None:
        """§1: "The v1 record is immutable." `assign_across_repos` mutates the rows it is
        handed, so the only thing keeping that true is that nothing writes them back."""
        before = LEGACY_SHAPE.read_bytes()
        jvp.legacy_positives(SEED, LEGACY_SHAPE)
        assert LEGACY_SHAPE.read_bytes() == before

    @pytest.mark.skipif(
        not jvp.LEGACY_ADOPTIONS.is_file(), reason="evals/.work/ is gitignored; local only"
    )
    def test_the_fixture_still_matches_the_real_artefact(self) -> None:
        """The fixture stands in for a gitignored file, so it must be checked against it
        wherever that file exists — otherwise CI pins a shape that has quietly stopped being
        the real one."""
        real = json.loads(jvp.LEGACY_ADOPTIONS.read_text(encoding="utf-8"))
        shape = json.loads(LEGACY_SHAPE.read_text(encoding="utf-8"))
        assert [(r["case"], r["id"], bool(r.get("usable"))) for r in real] == [
            (r["case"], r["id"], r["usable"]) for r in shape
        ]


class TestTheLedgerAccountsForEveryDroppedRow:
    def test_a_paper_whose_winner_is_not_a_positive_is_named(self, tmp_path) -> None:
        """§3.3 says a shared identifier is "counted once". Counted ZERO times is a different
        thing: the loser is uncounted by rule, and the winner is dropped by a filter."""
        src = tmp_path / "adoptions-pool-v2.json"
        src.write_text(
            json.dumps(
                [
                    _adoption("a/one", "2401.00001", usable=False, assigned_to="a/one"),
                    _adoption("b/two", "2401.00001", counted=False, assigned_to="a/one"),
                ]
            ),
            encoding="utf-8",
        )
        positives, ledger = jvp.pool_positives(src)
        assert positives == []
        assert [o["id"] for o in ledger["assigned_to_a_non_positive"]] == ["2401.00001"]

    def test_a_legacy_tie_loss_is_a_dedup_not_an_orphan(self, tmp_path) -> None:
        """§3.3 REGISTERS the legacy tie: `assign_across_repos` sets `assigned_to = "legacy"`
        and `counted = False` for every contender. Counting those as orphans would fill the
        ledger with the rule working and bury the one thing it exists to surface — and every
        pool paper that legitimately deduplicates against legacy would read as lost."""
        src = tmp_path / "adoptions-pool-v2.json"
        src.write_text(
            json.dumps([_adoption("o/r", "2401.00001", counted=False, assigned_to="legacy")]),
            encoding="utf-8",
        )
        positives, ledger = jvp.pool_positives(src)
        assert positives == []
        assert ledger["assigned_to_a_non_positive"] == []
        assert [d["id"] for d in ledger["deduped_to_legacy"]] == ["2401.00001"]

    def test_a_legitimate_contest_loss_is_not_reported_as_orphaned(self, tmp_path) -> None:
        src = tmp_path / "adoptions-pool-v2.json"
        src.write_text(
            json.dumps(
                [
                    _adoption("a/one", "2401.00001", assigned_to="a/one"),
                    _adoption("b/two", "2401.00001", counted=False, assigned_to="a/one"),
                ]
            ),
            encoding="utf-8",
        )
        positives, ledger = jvp.pool_positives(src)
        assert len(positives) == 1
        assert ledger["assigned_to_a_non_positive"] == []
        assert ledger["n_contested_lost"] == 1


class TestTheTwoStrataStaySeparateAndDisjoint:
    def test_an_identifier_positive_in_both_strata_raises(self, tmp_path, monkeypatch) -> None:
        """§3.3 gives legacy the tie, so the contest should have removed every one of these."""
        shape = json.loads(LEGACY_SHAPE.read_text(encoding="utf-8"))
        first = next(r["id"] for r in shape if r["usable"])
        src = tmp_path / "adoptions-pool-v2.json"
        src.write_text(json.dumps([_adoption("o/r", first)]), encoding="utf-8")
        monkeypatch.setattr(
            jvp,
            "legacy_positives",
            lambda s, p=None: ([_adoption("graph", first)], {"stratum": "legacy"}),
        )
        with pytest.raises(SystemExit) as exc:
            jvp.analysis_set(SEED, pool=src, legacy=LEGACY_SHAPE)
        assert "BOTH strata" in str(exc.value)

    def test_a_cluster_holding_one_paper_twice_raises(self, tmp_path) -> None:
        """Paper-level dedup is part of §5's primary endpoint definition, so this is a defect
        in the endpoint rather than untidiness in the data."""
        src = tmp_path / "adoptions-pool-v2.json"
        src.write_text(
            json.dumps([_adoption("o/r", "2401.00001"), _adoption("o/r", "2401.00001v2")]),
            encoding="utf-8",
        )
        with pytest.raises(SystemExit) as exc:
            jvp.analysis_set(SEED, pool=src, legacy=LEGACY_SHAPE)
        assert "twice" in str(exc.value)

    def test_the_strata_stay_labelled_rather_than_merged(self, tmp_path) -> None:
        """§5's transportability endpoint is a contrast BETWEEN them and §9's power table
        budgets them separately. A merged set cannot answer either."""
        src = tmp_path / "adoptions-pool-v2.json"
        src.write_text(json.dumps([_adoption("o/r", "2999.00001")]), encoding="utf-8")
        out = jvp.analysis_set(SEED, pool=src, legacy=LEGACY_SHAPE)
        assert set(out["by_stratum"]) == {"legacy", "pool"}
        assert len(out["by_stratum"]["pool"]) == 1
        assert out["analysis_set_positives"] == len(out["by_stratum"]["legacy"]) + 1

    def test_an_unwalked_pool_is_allowed_only_when_asked_for(self, tmp_path) -> None:
        out = jvp.analysis_set(
            SEED, pool=tmp_path / "absent.json", legacy=LEGACY_SHAPE, require_pool=False
        )
        assert out["ledgers"]["pool"]["_absent"]
        with pytest.raises(SystemExit):
            jvp.analysis_set(SEED, pool=tmp_path / "absent.json", legacy=LEGACY_SHAPE)


class TestTheContextIsTheOneTheWalkWrote:
    GOOD = (
        "Repository: owner/repo\n\n## README (excerpt)\nreal prose\n\n"
        "## Source files (sample)\na.py\n"
    )

    def _write(self, root: Path, case: str, text: str) -> str:
        digest = wp.context_hash("t0", text)
        root.mkdir(parents=True, exist_ok=True)
        (root / f"{jvp.case_key(case)}.{digest}.txt").write_text(text, encoding="utf-8")
        return digest

    def test_it_loads_at_the_recorded_digest(self, tmp_path) -> None:
        digest = self._write(tmp_path, "owner/repo", self.GOOD)
        assert jvp.t0_context_for("owner/repo", digest, tmp_path) == self.GOOD

    def test_an_edited_context_fails_the_hash(self, tmp_path) -> None:
        digest = self._write(tmp_path, "owner/repo", self.GOOD)
        (tmp_path / f"owner__repo.{digest}.txt").write_text(self.GOOD + "x", encoding="utf-8")
        with pytest.raises(SystemExit) as exc:
            jvp.t0_context_for("owner/repo", digest, tmp_path)
        assert "would not be" in str(exc.value)

    def test_a_missing_context_refuses_rather_than_re_cloning(self, tmp_path) -> None:
        """§3.1 persists it so judging never re-clones. Re-mining would also change what the
        judge is shown: the clone is deleted after the walk."""
        with pytest.raises(SystemExit) as exc:
            jvp.t0_context_for("owner/repo", "0" * 12, tmp_path)
        assert "never re-clone" in str(exc.value)

    def test_the_six_word_context_from_a_missing_clone_is_refused(self, tmp_path) -> None:
        """`t0_context` runs git with no `check`, so against a missing clone it returns a
        TRUTHY one-line string. Both judges would score every paper against six words, void
        would stay 0, the cache gate would pass, and the run would look finished."""
        text = "Repository: owner/repo\n"
        digest = self._write(tmp_path, "owner/repo", text)
        with pytest.raises(SystemExit) as exc:
            jvp.t0_context_for("owner/repo", digest, tmp_path)
        assert "header and nothing else" in str(exc.value)

    def test_an_empty_readme_body_is_refused(self, tmp_path) -> None:
        text = "Repository: owner/repo\n\n## README (excerpt)\n\n\n## Source files (sample)\na.py\n"
        digest = self._write(tmp_path, "owner/repo", text)
        with pytest.raises(SystemExit) as exc:
            jvp.t0_context_for("owner/repo", digest, tmp_path)
        assert "EMPTY README" in str(exc.value)

    def test_a_readme_with_no_listing_is_accepted(self, tmp_path) -> None:
        """The check is substantive, not schematic. Demanding a manifest-or-listing looked
        stricter and was wrong for the population the walk admits: `eligibility.LANGUAGES`
        includes Julia, R and Fortran, whose source extensions are absent from `t0_context`'s
        `exts`, so those repositories emit no "Source files" section at all."""
        text = "Repository: owner/repo\n\n## README (excerpt)\nreal prose about the project\n"
        digest = self._write(tmp_path, "owner/repo", text)
        assert jvp.t0_context_for("owner/repo", digest, tmp_path) == text

    def test_a_context_with_no_readme_section_is_accepted(self) -> None:
        """`eligibility.README_NAMES` accepts `README` and `Readme.md`, which `t0_context`
        never reads — so a repository can pass the English-README screen and still emit no
        README section. Refusing it would abort a healthy run and blame a missing clone."""
        text = "Repository: o/r\n\n## pyproject.toml\n[project]\n\n## Source files (sample)\na.jl\n"
        jvp.assert_context_is_judgeable("o/r", text)

    def test_a_readmes_own_heading_does_not_truncate_the_body(self) -> None:
        """`graph`'s real T0 context contains "## Library Highlights" inside the README
        excerpt. Splitting the body at a bare "## " made a README opening with a heading read
        as EMPTY and abort a healthy run, so sections are matched by NAME."""
        opens = (
            "Repository: o/r\n\n## README (excerpt)\n## Overview\nprose\n\n"
            "## Source files (sample)\na.py\n"
        )
        jvp.assert_context_is_judgeable("o/r", opens)

    def test_a_readme_opening_with_a_heading_is_not_read_as_empty(self) -> None:
        """The other direction of the same bug: splitting the body at the first "## " made a
        README that opens with an H2 read as an EMPTY body, aborting a healthy run."""
        opens = (
            "Repository: o/r\n\n## README (excerpt)\n## Overview\nreal prose\n\n"
            "## Source files (sample)\na.py\n"
        )
        jvp.assert_context_is_judgeable("o/r", opens)

    def test_every_section_the_walk_can_emit_is_recognised(self) -> None:
        """Pinned against `mine_adoptions.t0_context`'s own manifest list: if that function
        learns a new manifest, this list must learn it too or a context carrying only that
        manifest reads as having no sections at all."""
        import mine_adoptions as ma

        source = ma.t0_context.__code__.co_consts
        manifests = next(c for c in source if isinstance(c, tuple) and "pyproject.toml" in c)
        for manifest in manifests:
            assert f"\n## {manifest}" in jvp.KNOWN_SECTIONS

    def test_a_dotted_repository_name_is_not_confused_with_another(self, tmp_path) -> None:
        """The reason the one-to-one check is global rather than a per-case glob: repository
        names contain dots, so `owner__repo.*.txt` also matches `owner__repo.js.<digest>.txt`
        and a healthy pair of repositories would abort the run."""
        self._write(tmp_path, "owner/repo", self.GOOD)
        self._write(tmp_path, "owner/repo.js", self.GOOD)
        jvp.assert_contexts_are_one_to_one(["owner/repo", "owner/repo.js"], tmp_path)

    def test_a_context_with_no_positive_left_is_not_an_error(self, tmp_path) -> None:
        """`walk_row` writes the context inside `if usable:` — before any cross-repository
        contest exists — and nothing prunes the directory afterwards. A repository whose every
        positive later lost the §3.3 tie keeps its context and is absent from the analysis set.
        That is a CORRECT walk; refusing it would abort the run on exactly the data the contest
        is supposed to produce."""
        self._write(tmp_path, "contested/repo", self.GOOD)
        self._write(tmp_path, "owner/repo", self.GOOD)
        jvp.assert_contexts_are_one_to_one(["owner/repo"], tmp_path)

    def test_a_positive_with_no_context_raises(self, tmp_path) -> None:
        """The converse, which is a real failure: §3.1 persists one for every repository with a
        usable row, so a positive without one means the directory is not the walk's output."""
        self._write(tmp_path, "owner/repo", self.GOOD)
        with pytest.raises(SystemExit) as exc:
            jvp.assert_contexts_are_one_to_one(["owner/repo", "missing/repo"], tmp_path)
        assert "no persisted T0 context" in str(exc.value)

    def test_the_digest_comes_from_the_walk_ledger(self, tmp_path) -> None:
        csv_path = tmp_path / "validity_walk.csv"
        csv_path.write_text(
            "full_name,note\nowner/repo,context abcdef123456\nother/x,\n", encoding="utf-8"
        )
        assert jvp.context_digests(csv_path) == {"owner/repo": "abcdef123456"}


class TestHeadIdsAreNeverDefaultedToEmpty:
    """§4 excludes a "control" the repository actually went on to cite. An empty set turns that
    rule into a no-op, puts positives into the negative class, and biases the AUC toward 0.5 —
    the study's own pre-committed null, arriving by accident and looking like a finding."""

    def test_it_loads_and_normalises(self, tmp_path) -> None:
        (tmp_path / "owner__repo.json").write_text(
            json.dumps(["2401.00001v3", "2402.00002"]), encoding="utf-8"
        )
        assert jvp.head_ids_for(["owner/repo"], tmp_path) == {
            "owner/repo": {"2401.00001", "2402.00002"}
        }

    def test_a_missing_file_refuses_rather_than_returning_an_empty_set(self, tmp_path) -> None:
        with pytest.raises(SystemExit) as exc:
            jvp.head_ids_for(["owner/repo"], tmp_path)
        assert "0.5" in str(exc.value)

    def test_an_empty_file_refuses(self, tmp_path) -> None:
        (tmp_path / "owner__repo.json").write_text("[]", encoding="utf-8")
        with pytest.raises(SystemExit) as exc:
            jvp.head_ids_for(["owner/repo"], tmp_path)
        assert "stale or truncated" in str(exc.value)

    def test_a_positive_absent_from_its_own_head_set_raises(self) -> None:
        """It was mined FROM that HEAD, so the file cannot be the one the walk wrote — and a
        truncated file under-excludes, in the one direction §4 cannot afford."""
        positives = [_adoption("owner/repo", "2401.00001")]
        with pytest.raises(SystemExit) as exc:
            jvp.assert_positives_are_cited_at_head(positives, {"owner/repo": {"2402.00002"}})
        assert "absent from its own head_ids" in str(exc.value)

    def test_a_consistent_pair_passes(self) -> None:
        positives = [_adoption("owner/repo", "2401.00001v2")]
        jvp.assert_positives_are_cited_at_head(positives, {"owner/repo": {"2401.00001"}})

    def test_colliding_case_names_are_refused_before_any_file_is_read(self, tmp_path) -> None:
        with pytest.raises(SystemExit) as exc:
            jvp.head_ids_for(["a/b", "a__b"], tmp_path)
        assert "mangle" in str(exc.value)
