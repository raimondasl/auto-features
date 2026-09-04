"""Arm-neutral controls for the validity pool. [PREREG-judge-validity-pool §4]

The control set decides what the primary endpoint means. NR-56/57 drew controls from
RepoRadar's own candidate pool, which is a problem the frame states plainly: a pool built by
the system under test is HEAD-seeded by that system, so a judge that happens to be harsher on
RepoRadar-shaped papers — Sonnet, by a factor of 2.3 — gets credited with "validity" for a
property of the *controls*. Both adoption refutations landed on exactly this. An arXiv
category listing is produced by arXiv.

Two properties do the work, and both fail silently if wrong:

* **Matched, not arbitrary.** Same primary category, same half-year of submission. A control
  the project could not have adopted at T0 — because it did not exist yet, or is from another
  field — is not evidence about a judge.
* **Never cited by the repository at HEAD.** A "control" the project actually went on to cite
  is a positive sitting in the negative class. That biases the AUC toward 0.5 — toward the
  null the pool is pre-committed to reporting — so it would look like a finding.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest import mock

import pytest

EVALS = Path(__file__).resolve().parent.parent / "evals"


def _load(name: str):  # type: ignore[no-untyped-def]
    for path in (EVALS, EVALS.parent / "src"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    spec = importlib.util.spec_from_file_location(name, EVALS / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


jva = _load("judge_validity_adoption")


def _paper(pid: str, primary: str = "cs.LG") -> dict:
    # `primary_category` is not decoration: §4 matches a control to its positive on it, and
    # `cat:` matches cross-lists too, so the listing is a superset the draw has to filter.
    return {
        "arxiv_id": pid,
        "title": f"paper {pid}",
        "abstract": "a real abstract",
        "primary_category": primary,
    }


def _listing_of(ids: list[str]):  # type: ignore[no-untyped-def]
    calls: list[tuple[str, str, str]] = []

    def listing(category: str, lo: str, hi: str, *, archive: Path | None = None) -> list[dict]:
        calls.append((category, lo, hi))
        return [_paper(pid) for pid in ids]

    listing.calls = calls  # type: ignore[attr-defined]
    return listing


CITED: dict[str, set[str]] = {"acme/rich": set()}

POSITIVE = {
    "case": "acme/rich",
    "id": "2103.00001",
    "primary_category": "cs.LG",
    "published": "2021-03-14",
    "t0_commit_date": "2022-01-01",
}


class TestTheWindowIsAHalfYear:
    def test_the_first_half(self) -> None:
        assert jva.half_year_bounds("2021-03-14") == ("202101010000", "202106302359")

    def test_the_second_half(self) -> None:
        assert jva.half_year_bounds("2021-09-30") == ("202107010000", "202112312359")

    def test_the_boundary_months_land_on_the_right_side(self) -> None:
        assert jva.half_year_bounds("2021-06-30")[0] == "202101010000"
        assert jva.half_year_bounds("2021-07-01")[0] == "202107010000"


class TestTheControlsAreMatchedAndClean:
    def test_four_controls_per_positive_from_the_matching_window(self) -> None:
        listing = _listing_of([f"2104.0000{i}" for i in range(10)])
        out = jva.arxiv_window_controls([POSITIVE], CITED, "SEED", listing=listing)
        assert len(out) == 4
        assert listing.calls == [("cs.LG", "202101010000", "202106302359")]
        assert all(c["case"] == "acme/rich" for c in out)
        assert all(c["for_positive"] == "2103.00001" for c in out)

    def test_a_paper_the_repo_cites_at_head_is_never_a_control(self) -> None:
        """The failure that would look like a finding: a positive sitting in the negative
        class drags the AUC toward 0.5, which is the outcome §5 pre-commits to reporting as
        'no demonstrated discrimination'."""
        ids = [f"2104.0000{i}" for i in range(6)]
        listing = _listing_of(ids)
        cited = {"2104.00000", "2104.00001"}
        out = jva.arxiv_window_controls([POSITIVE], {"acme/rich": cited}, "SEED", listing=listing)
        assert {c["id"] for c in out} & cited == set()

    def test_the_positive_itself_is_never_drawn_as_its_own_control(self) -> None:
        listing = _listing_of(["2103.00001", *[f"2104.0000{i}" for i in range(6)]])
        out = jva.arxiv_window_controls([POSITIVE], CITED, "SEED", listing=listing)
        assert "2103.00001" not in {c["id"] for c in out}

    def test_a_paper_with_no_abstract_is_skipped(self) -> None:
        """Both judges are shown the abstract; a control without one is not the same task."""

        def listing(category: str, lo: str, hi: str, *, archive: Path | None = None) -> list[dict]:
            return [
                {"arxiv_id": "2104.00001", "abstract": ""},
                *[_paper(f"2104.0001{i}") for i in range(5)],
            ]

        out = jva.arxiv_window_controls([POSITIVE], CITED, "SEED", listing=listing)
        assert "2104.00001" not in {c["id"] for c in out}

    def test_controls_are_not_reused_across_positives_in_one_repo(self) -> None:
        """Reuse would make the negative class smaller than it looks and correlate the
        per-positive comparisons inside a repository, which the cluster bootstrap then
        cannot see."""
        listing = _listing_of([f"2104.000{i:02d}" for i in range(20)])
        second = {**POSITIVE, "id": "2104.09999"}
        out = jva.arxiv_window_controls([POSITIVE, second], CITED, "SEED", listing=listing)
        assert len(out) == 8
        assert len({c["id"] for c in out}) == 8

    def test_the_listing_is_fetched_once_per_category_and_window(self) -> None:
        """Positives cluster inside a repository and a field; re-fetching per positive would
        multiply arXiv requests for an identical answer."""
        listing = _listing_of([f"2104.000{i:02d}" for i in range(20)])
        same_window = {**POSITIVE, "id": "2105.00002", "published": "2021-05-02"}
        jva.arxiv_window_controls([POSITIVE, same_window], CITED, "SEED", listing=listing)
        assert len(listing.calls) == 1

    def test_a_positive_missing_its_category_is_skipped_rather_than_guessed(self) -> None:
        listing = _listing_of([f"2104.0000{i}" for i in range(6)])
        bare = {"case": "acme/rich", "id": "2103.00002"}
        assert jva.arxiv_window_controls([bare], CITED, "SEED", listing=listing) == []


class TestTheDrawIsSeeded:
    def test_the_same_seed_gives_the_same_controls(self) -> None:
        listing = _listing_of([f"2104.000{i:02d}" for i in range(30)])
        a = jva.arxiv_window_controls([POSITIVE], CITED, "SEED-1", listing=listing)
        b = jva.arxiv_window_controls([POSITIVE], CITED, "SEED-1", listing=listing)
        assert [c["id"] for c in a] == [c["id"] for c in b]

    def test_a_different_seed_gives_different_controls(self) -> None:
        listing = _listing_of([f"2104.000{i:02d}" for i in range(30)])
        a = jva.arxiv_window_controls([POSITIVE], CITED, "SEED-1", listing=listing)
        b = jva.arxiv_window_controls([POSITIVE], CITED, "SEED-2", listing=listing)
        assert [c["id"] for c in a] != [c["id"] for c in b]

    def test_the_draw_does_not_depend_on_the_listing_order(self) -> None:
        """arXiv returns results in its own order; a control set that depended on it would
        not be reproducible from the archived listing."""
        ids = [f"2104.000{i:02d}" for i in range(30)]
        forward = jva.arxiv_window_controls([POSITIVE], CITED, "S", listing=_listing_of(ids))
        backward = jva.arxiv_window_controls(
            [POSITIVE], CITED, "S", listing=_listing_of(list(reversed(ids)))
        )
        assert {c["id"] for c in forward} == {c["id"] for c in backward}


class TestTheListingIsArchived:
    def test_it_writes_the_query_and_the_papers(self, tmp_path: Path) -> None:
        """This is the negative class of the primary endpoint. An AUC that cannot be
        recomputed from an archived control set is not a reproducible number."""

        def listing(category: str, lo: str, hi: str, *, archive: Path | None = None) -> list[dict]:
            assert archive is not None
            archive.mkdir(parents=True, exist_ok=True)
            (archive / f"{category.replace('.', '_')}-{lo[:6]}.json").write_text(
                json.dumps({"query": f"cat:{category}", "papers": [_paper("2104.00001")]}),
                encoding="utf-8",
            )
            return [_paper(f"2104.0000{i}") for i in range(6)]

        jva.arxiv_window_controls(
            [POSITIVE], CITED, "SEED", listing=listing, archive=tmp_path / "listings"
        )
        saved = list((tmp_path / "listings").glob("*.json"))
        assert [p.name for p in saved] == ["cs_LG-202101.json"]


class TestTheSchemeIsSelectable:
    def test_both_schemes_are_offered_and_pool_remains_the_default(self) -> None:
        """NR-56/57 are published under the pool scheme; changing the default would silently
        re-report them under a different negative class."""
        assert jva.CONTROL_SCHEMES == ("pool", "arxiv-window")
        assert jva.CONTROL_SCHEME == "pool"

    def test_the_flag_exists_and_is_wired_to_the_module_switch(self) -> None:
        import inspect

        src = inspect.getsource(jva.main)
        assert '"--controls"' in src
        assert "CONTROL_SCHEME = args.controls" in src


class TestWhyNotThePool:
    def test_the_reason_is_recorded_beside_the_code(self) -> None:
        """A judge harsher on RepoRadar-shaped papers would be credited with validity for a
        property of the control set. Both adoption refutations landed here, so the reason
        lives next to the function rather than only in the pre-registration."""
        doc = jva.arxiv_window_controls.__doc__ or ""
        assert "HEAD-seeded" in doc
        assert "2.3" in doc


@pytest.mark.parametrize("published", ["2021-01-01", "2021-12-31"])
def test_every_month_maps_to_a_valid_window(published: str) -> None:
    lo, hi = jva.half_year_bounds(published)
    assert len(lo) == len(hi) == 12
    assert lo < hi


class TestTheCacheIsolationGate:
    """§4/§7. `judge_paper` keys its cache on (model, repo, paper_id) and **not** on the
    context it was given, so a T0 verdict written into the shared gold cache silently
    overwrites the HEAD verdict for the same paper. That exact write once took `rag` from 5
    gold targets to 0 — and it is invisible afterwards, because the file is still there and
    still parses. `mine_adoptions.main` already checked the gold set; this function bought
    hundreds of T0 verdicts with no guard at all.
    """

    def test_a_new_file_changes_the_fingerprint(self, tmp_path: Path) -> None:
        root = tmp_path / "judge"
        root.mkdir()
        (root / "a.json").write_text("{}", encoding="utf-8")
        before = jva.cache_fingerprint(root)
        (root / "b.json").write_text("{}", encoding="utf-8")
        assert jva.cache_fingerprint(root) != before

    def test_rewriting_a_file_changes_the_fingerprint(self, tmp_path: Path) -> None:
        """The actual failure mode: same path, different verdict. A fingerprint over names
        alone would miss precisely the overwrite this exists to catch."""
        root = tmp_path / "judge"
        root.mkdir()
        target = root / "a.json"
        target.write_text('{"score": 3}', encoding="utf-8")
        before = jva.cache_fingerprint(root)
        target.write_text('{"score": 0, "longer": true}', encoding="utf-8")
        assert jva.cache_fingerprint(root) != before

    def test_an_untouched_tree_is_stable(self, tmp_path: Path) -> None:
        root = tmp_path / "judge"
        (root / "nested").mkdir(parents=True)
        (root / "nested" / "a.json").write_text("{}", encoding="utf-8")
        assert jva.cache_fingerprint(root) == jva.cache_fingerprint(root)

    def test_a_missing_root_is_not_an_error(self, tmp_path: Path) -> None:
        """The second judge's cache does not exist until it has written something."""
        assert jva.cache_fingerprint(tmp_path / "never") == jva.cache_fingerprint(tmp_path / "nope")

    def test_the_guard_raises_and_says_why(self, tmp_path: Path) -> None:
        root = tmp_path / "judge"
        root.mkdir()
        before = jva.cache_fingerprint(root)
        (root / "leaked.json").write_text("{}", encoding="utf-8")
        with pytest.raises(SystemExit) as exc:
            jva.assert_caches_untouched(before, (root,))
        assert "use_cache=False" in str(exc.value)

    def test_the_guard_is_silent_when_nothing_moved(self, tmp_path: Path) -> None:
        root = tmp_path / "judge"
        root.mkdir()
        jva.assert_caches_untouched(jva.cache_fingerprint(root), (root,))

    def test_both_roots_are_watched(self) -> None:
        """The second judge writes under `.work/second_judge/` with a `cache_as` namespace.
        Watching only the gold cache would miss a namespace collision there.

        This asserted the source literal `cache_roots = (JUDGE_CACHE, SECOND_JUDGE_CACHE)`,
        which **was the defect**: watching that root whole also watched the namespace
        `second_verdict(cache_as=...)` writes T0 verdicts into, so the gate fired on every run
        that bought anything. The requirement is that both roots are watched; the spelling is
        not the requirement, and pinning the spelling is what made the bug look tested.
        """
        assert set(jva.protected_partition()) == {jva.JUDGE_CACHE, jva.SECOND_JUDGE_CACHE}
        assert jva.protected_partition()[jva.JUDGE_CACHE] == frozenset()

    def test_a_write_into_the_permitted_t0_namespace_passes(self, tmp_path: Path) -> None:
        """The #11 regression. `second_verdict(cache_as=f"{model}#t0")` writes T0 verdicts under
        `.work/second_judge/<model>#t0/`, inside a fingerprinted root — so the gate fired on
        exactly the runs that bought something, and a gate whose true-negative rate is zero gets
        switched off. §7 registers that separate namespace as legitimate, so the hash cannot
        also forbid it."""
        second = tmp_path / "second_judge"
        (second / jva.t0_namespace(jva.SONNET_MODEL) / "graph").mkdir(parents=True)
        gold = tmp_path / "judge"
        gold.mkdir()
        partition = {gold: frozenset(), second: jva.permitted_namespaces()}
        with mock.patch.object(jva, "protected_partition", lambda: partition):
            before = jva.cache_manifest(gold, second)
            written = second / jva.t0_namespace(jva.SONNET_MODEL) / "graph" / "2401.00001.json"
            written.write_text('{"score": 2}', encoding="utf-8")
            jva.assert_caches_untouched(before, (gold, second))

    def test_a_write_into_the_second_judges_own_model_directory_raises(self, tmp_path) -> None:
        """The permitted namespace is one directory, not the root. `.work/second_judge` holds
        1,504 verdicts eight other modules read back by name."""
        second = tmp_path / "second_judge"
        (second / jva.SONNET_MODEL / "graph").mkdir(parents=True)
        with mock.patch.object(
            jva, "protected_partition", lambda: {second: jva.permitted_namespaces()}
        ):
            before = jva.cache_manifest(second)
            (second / jva.SONNET_MODEL / "graph" / "2401.00001.json").write_text("{}", "utf-8")
            with pytest.raises(SystemExit) as exc:
                jva.assert_caches_untouched(before, (second,))
        assert "2401.00001.json" in str(exc.value)

    def test_the_gold_cache_has_no_exclusion_ever(self, tmp_path: Path) -> None:
        gold = tmp_path / "judge"
        (gold / jva.t0_namespace(jva.SONNET_MODEL)).mkdir(parents=True)
        with mock.patch.object(jva, "protected_partition", lambda: {gold: frozenset()}):
            before = jva.cache_manifest(gold)
            (gold / jva.t0_namespace(jva.SONNET_MODEL) / "leak.json").write_text("{}", "utf-8")
            with pytest.raises(SystemExit):
                jva.assert_caches_untouched(before, (gold,))

    def test_the_gpt_t0_namespace_is_protected_not_permitted(self) -> None:
        """GPT is called with `use_cache=False` and writes nothing anywhere, so a file appearing
        under `gpt-5.5#t0` is a leak like any other rather than this study's own output."""
        permitted = jva.permitted_namespaces()
        assert jva.t0_namespace(jva.SONNET_MODEL) in permitted
        assert jva.t0_namespace(jva.GPT_MODEL) not in permitted

    def test_the_call_site_and_the_exclusion_share_one_declaration(self) -> None:
        """Two independent spellings of the same string is how the gate became self-tripping."""
        import inspect

        assert "cache_as=t0_namespace(model)" in inspect.getsource(jva.judge)

    def test_an_existing_namespace_without_the_ownership_marker_refuses(self, tmp_path) -> None:
        """An exclusion is justified by ownership, not by spelling: a directory whose name
        happens to match may be somebody else's data, and excluding it would make overwriting
        that data the one thing the gate cannot see."""
        second = tmp_path / "second_judge"
        (second / jva.t0_namespace(jva.SONNET_MODEL)).mkdir(parents=True)
        with pytest.raises(SystemExit) as exc:
            jva.assert_namespace_ownership({second: jva.permitted_namespaces()})
        assert "_NAMESPACE.json" in str(exc.value)

    def test_a_namespace_this_study_claimed_is_accepted(self, tmp_path: Path) -> None:
        second = tmp_path / "second_judge"
        partition = {second: jva.permitted_namespaces()}
        jva.claim_namespaces(partition)
        jva.assert_namespace_ownership(partition)

    def test_an_absent_namespace_is_fine(self, tmp_path: Path) -> None:
        """It does not exist until the second judge has written something."""
        jva.assert_namespace_ownership({tmp_path / "never": jva.permitted_namespaces()})

    def test_the_failure_message_carries_the_manifest_diff(self, tmp_path: Path) -> None:
        """A hash says only that something moved. Triage needs to know which root and what."""
        root = tmp_path / "judge"
        root.mkdir()
        (root / "kept.json").write_text("{}", encoding="utf-8")
        with mock.patch.object(jva, "protected_partition", lambda: {root: frozenset()}):
            before = jva.cache_manifest(root)
            (root / "added.json").write_text("{}", encoding="utf-8")
            with pytest.raises(SystemExit) as exc:
                jva.assert_caches_untouched(before, (root,))
        assert "added (1)" in str(exc.value)
        assert "/judge/added.json" in str(exc.value)

    def test_the_exclusion_is_applied_while_walking(self, tmp_path: Path) -> None:
        """Not by pre-enumerating directories: a list computed before the run cannot contain a
        directory the run is about to create, which is the case the exclusion exists for."""
        second = tmp_path / "second_judge"
        second.mkdir()
        with mock.patch.object(
            jva, "protected_partition", lambda: {second: jva.permitted_namespaces()}
        ):
            before = jva.cache_manifest(second)
            fresh = second / jva.t0_namespace(jva.SONNET_MODEL) / "graph"
            fresh.mkdir(parents=True)
            (fresh / "2401.00001.json").write_text("{}", encoding="utf-8")
            jva.assert_caches_untouched(before, (second,))

    def test_the_gold_set_is_checked_too(self) -> None:
        """Two independent guards: the fingerprint catches a write, `resolve_targets` catches
        the consequence. Either alone can pass while the other fails.

        The comparison itself moved into `isolation_failures`, so that the pair could be
        exercised without buying a verdict; `test_either_guard_can_fire_alone` is what now
        holds this requirement. What stays here is that `judge()` still SAMPLES the gold set
        before buying anything — a snapshot taken afterwards would compare the leak to itself.
        """
        import inspect

        src = inspect.getsource(jva.judge)
        assert "targets_before = resolve_targets()" in src
        assert src.index("targets_before = resolve_targets()") < src.index("for n, (case")
        assert "isolation_failures(cache_before, cache_roots, targets_before" in src

    def test_the_guards_run_after_the_verdicts_are_written(self) -> None:
        """A violation must be reported without also losing the expensive run that revealed
        it."""
        import inspect

        src = inspect.getsource(jva.judge)
        last_write = src.rindex("save_verdicts(have)")
        assert src.index("isolation_failures(cache_before") > last_write

    def test_both_guards_are_evaluated_not_short_circuited(self, tmp_path: Path) -> None:
        """The cache guard raised first, so on any run that tripped it the gold-set check never
        executed. The two do not imply one another: `resolve_targets` sees only ids the baseline
        picked whose gold verdict scores >= 2, so a write for an unpicked paper moves the
        fingerprint and leaves the gold set alone, and a score crossing 2 does the opposite.

        Behavioural, over the extracted helper: the previous version read the substring
        "failures" out of `judge()`'s source, which would pass on a body that still raised at
        the first guard.
        """
        root = tmp_path / "judge"
        root.mkdir()
        with mock.patch.object(jva, "protected_partition", lambda: {root: frozenset()}):
            before = jva.cache_manifest(root)
            (root / "leaked.json").write_text("{}", encoding="utf-8")
            failures = jva.isolation_failures(before, (root,), ["a"], ["b"])
        assert len(failures) == 2, "the cache guard must not hide the gold-set result"
        assert "THE JUDGE CACHE MOVED" in failures[0]
        assert "THE GOLD SET MOVED" in failures[1]

    def test_either_guard_can_fire_alone(self, tmp_path: Path) -> None:
        root = tmp_path / "judge"
        root.mkdir()
        with mock.patch.object(jva, "protected_partition", lambda: {root: frozenset()}):
            clean = jva.cache_manifest(root)
            gold_only = jva.isolation_failures(clean, (root,), ["a"], ["b"])
            (root / "leaked.json").write_text("{}", encoding="utf-8")
            cache_only = jva.isolation_failures(clean, (root,), ["a"], ["a"])
        assert [f[:20] for f in gold_only] == ["THE GOLD SET MOVED —"]
        assert [f[:21] for f in cache_only] == ["THE JUDGE CACHE MOVED"]

    def test_a_clean_run_reports_nothing(self, tmp_path: Path) -> None:
        root = tmp_path / "judge"
        root.mkdir()
        with mock.patch.object(jva, "protected_partition", lambda: {root: frozenset()}):
            assert jva.isolation_failures(jva.cache_manifest(root), (root,), ["a"], ["a"]) == []

    def test_ownership_is_established_before_the_before_manifest(self, tmp_path) -> None:
        """`claim_namespaces` stamps the marker, so the ownership check must precede it —
        reversing the two would make the run claim whatever it found. And both must precede the
        first purchase: checking at the end means discovering somebody else's data after
        writing into it."""
        second = tmp_path / "second_judge"
        (second / jva.t0_namespace(jva.SONNET_MODEL)).mkdir(parents=True)
        partition = {second: jva.permitted_namespaces()}
        with (
            mock.patch.object(jva, "protected_partition", lambda: partition),
            pytest.raises(SystemExit) as exc,
        ):
            jva.prepare_isolation()
        assert "_NAMESPACE.json" in str(exc.value)

    def test_prepare_isolation_claims_and_returns_the_watched_roots(self, tmp_path) -> None:
        second = tmp_path / "second_judge"
        with mock.patch.object(
            jva, "protected_partition", lambda: {second: jva.permitted_namespaces()}
        ):
            roots, before = jva.prepare_isolation()
        assert roots == (second,)
        marker = second / jva.t0_namespace(jva.SONNET_MODEL) / "_NAMESPACE.json"
        assert marker.is_file()
        # The marker lives inside the excluded namespace, so it is not in the manifest itself.
        assert before == {}

    def test_two_roots_sharing_an_id_are_refused_rather_than_merged(self, tmp_path) -> None:
        """A collision would let a file added under one root and removed from the other cancel
        out — the exact silent failure the gate exists to catch."""
        a = tmp_path / "x" / "judge"
        b = tmp_path / "x" / "judge2"
        for root in (a, b):
            root.mkdir(parents=True)
        with pytest.raises(SystemExit) as exc:
            jva.cache_manifest(a, a)
        assert "share an id" in str(exc.value)
        jva.cache_manifest(a, b)  # distinct ids are fine


class TestPositivesMustBeEnrichedBeforeControlsCanBeDrawn:
    """The gap that would have produced no controls at all.

    `arxiv_window_controls` matches a control to its positive on (primary category, half-year
    of submission). Neither field exists on a mined adoption row — mining reads git, not arXiv
    — and nothing in the codebase produced them. So on real data every positive was skipped
    and the control set came back EMPTY: not an error, just no negatives, and a primary
    endpoint computed against nothing.
    """

    ROWS = [{"case": "acme/rich", "id": "2103.00001"}, {"case": "acme/rich", "id": "2104.00002"}]

    def test_it_attaches_the_category_and_date_the_controls_match_on(self) -> None:
        def fetch(ids: list[str]) -> list[dict]:
            return [
                {
                    "arxiv_id": "2103.00001",
                    "categories": ["cs.LG", "stat.ML"],
                    "published": "2021-03-14T00:00:00+00:00",
                },
                {
                    "arxiv_id": "2104.00002",
                    "categories": ["cs.CV"],
                    "published": "2021-04-02T00:00:00+00:00",
                },
            ]

        enriched, missing = jva.enrich_positives(self.ROWS, fetch=fetch)
        assert missing == []
        assert [r["primary_category"] for r in enriched] == ["cs.LG", "cs.CV"]
        assert [r["published"] for r in enriched] == ["2021-03-14", "2021-04-02"]

    def test_an_unfetchable_positive_is_returned_as_missing_not_dropped(self) -> None:
        """Silently dropping it would shrink the positive set without saying so — the same
        shape as the empty control set this exists to prevent."""

        def fetch(ids: list[str]) -> list[dict]:
            return [{"arxiv_id": "2103.00001", "categories": ["cs.LG"], "published": "2021-03-14"}]

        enriched, missing = jva.enrich_positives(self.ROWS, fetch=fetch)
        assert [r["id"] for r in enriched] == ["2103.00001"]
        assert missing == ["2104.00002"]

    def test_a_paper_with_no_category_counts_as_missing(self) -> None:
        def fetch(ids: list[str]) -> list[dict]:
            return [
                {"arxiv_id": "2103.00001", "categories": [], "published": "2021-03-14"},
                {"arxiv_id": "2104.00002", "categories": ["cs.LG"], "published": ""},
            ]

        enriched, missing = jva.enrich_positives(self.ROWS, fetch=fetch)
        assert enriched == []
        assert sorted(missing) == ["2103.00001", "2104.00002"]

    def test_enriched_rows_actually_draw_controls(self) -> None:
        """The end-to-end point: unenriched rows draw nothing, enriched rows draw four each."""
        listing = _listing_of([f"2104.000{i:02d}" for i in range(20)])
        assert jva.arxiv_window_controls(self.ROWS, CITED, "SEED", listing=listing) == []

        def fetch(ids: list[str]) -> list[dict]:
            return [
                {"arxiv_id": "2103.00001", "categories": ["cs.LG"], "published": "2021-03-14"},
                {"arxiv_id": "2104.00002", "categories": ["cs.LG"], "published": "2021-04-02"},
            ]

        enriched, _ = jva.enrich_positives(self.ROWS, fetch=fetch)
        assert len(jva.arxiv_window_controls(enriched, CITED, "SEED", listing=listing)) == 8


class TestAnEmptyControlSetIsRefused:
    def test_positives_with_no_controls_raises(self) -> None:
        """An AUC against an empty negative class is not a null result, it is no result — and
        without this it would look exactly like a completed run."""
        with pytest.raises(SystemExit) as exc:
            jva.refuse_an_empty_control_set([{"id": "1"}], [])
        assert "ZERO controls" in str(exc.value)

    def test_a_normal_draw_passes_silently(self) -> None:
        jva.refuse_an_empty_control_set([{"id": "1"}], [{"id": "2"}])

    def test_no_positives_is_not_this_functions_problem(self) -> None:
        jva.refuse_an_empty_control_set([], [])


class TestTheDrawSpansTheWindowRatherThanItsEnd:
    """`arxiv.Search(sort_by=SubmittedDate)` defaults to DESCENDING order, so asking for 200
    returned the *newest* 200 of the half-year. Measured: `cs.LG` H1-2021 holds 13,262 papers,
    so every control for a positive in that window came from its last few days.

    §4 registers "submitted in the same half-year" and names no cap and no ordering, so the cap
    was an unregistered narrowing that made the negative class arXiv's index order rather than
    the seed's. It also ran in a direction: NR-43 measured actionability rising with recency
    (0.31 in 2013 to 0.64 in 2025), so controls months newer than their positive score higher
    and compress the gap toward the null §5 pre-commits to reporting.
    """

    def test_the_half_year_splits_into_contiguous_slices(self) -> None:
        lo, hi = jva.half_year_bounds("2021-03-14")
        parts = jva.sub_windows(lo, hi)
        assert len(parts) == jva.LISTING_SLICES
        assert parts[0][0] == lo and parts[-1][1] == hi
        for (_, end), (start, _) in zip(parts, parts[1:], strict=False):
            assert end[:8] < start[:8], "slices must not overlap"

    def test_every_day_of_the_window_lands_in_exactly_one_slice(self) -> None:
        from datetime import date, timedelta

        lo, hi = jva.half_year_bounds("2021-09-30")
        covered: set[date] = set()
        for start, end in jva.sub_windows(lo, hi):
            a = date(int(start[:4]), int(start[4:6]), int(start[6:8]))
            b = date(int(end[:4]), int(end[4:6]), int(end[6:8]))
            while a <= b:
                assert a not in covered, "a paper could fall into two slices"
                covered.add(a)
                a += timedelta(days=1)
        assert len(covered) == 184  # Jul 1 - Dec 31

    def test_the_listing_queries_every_slice_not_just_the_window(self) -> None:
        calls = self._listing_with(want=180)
        assert len(calls) == jva.LISTING_SLICES
        assert len(set(calls)) == jva.LISTING_SLICES, "each slice needs its own query"

    def _listing_with(self, *, want, depth="stratified", archive=None):  # noqa: ANN001
        queries: list[str] = []

        class FakeSearch:
            def __init__(self, query, max_results, sort_by):  # noqa: ANN001
                queries.append(query)
                self.max_results = max_results

        import arxiv

        from reporadar import collector as collector_mod

        lo, hi = jva.half_year_bounds("2021-03-14")
        with (
            mock.patch.object(arxiv, "Search", FakeSearch),
            mock.patch.object(collector_mod, "_query_with_retry", lambda c, s: []),
            mock.patch.object(collector_mod, "_shared_client", lambda n: None),
        ):
            jva.arxiv_window_listing("cs.LG", lo, hi, want=want, depth=depth, archive=archive)
        return queries

    def test_full_depth_asks_for_the_whole_window_in_one_query(self) -> None:
        calls = self._listing_with(want=200, depth="full")
        assert len(calls) == 1
        assert "20210101" in calls[0] and "20210630" in calls[0]

    def test_an_unknown_depth_is_refused(self) -> None:
        lo, hi = jva.half_year_bounds("2021-03-14")
        with pytest.raises(SystemExit) as exc:
            jva.arxiv_window_listing("cs.LG", lo, hi, depth="newest")
        assert "stratified" in str(exc.value)

    def test_the_archive_records_every_slice_and_whether_it_was_cut_off(
        self, tmp_path: Path
    ) -> None:
        """ "200 of 13,262" and "200 of 200" are the same number in an archive that stores only
        the count, so the truncation flag is recorded rather than inferred."""
        self._listing_with(want=180, archive=tmp_path)
        saved = json.loads((tmp_path / "cs_LG-202101.json").read_text(encoding="utf-8"))
        assert len(saved["sub_queries"]) == jva.LISTING_SLICES
        assert saved["depth"] == "stratified"
        for q in saved["sub_queries"]:
            assert {"query", "requested", "returned", "truncated"} <= set(q)
