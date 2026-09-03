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


def _paper(pid: str) -> dict:
    return {"arxiv_id": pid, "title": f"paper {pid}", "abstract": "a real abstract"}


def _listing_of(ids: list[str]):  # type: ignore[no-untyped-def]
    calls: list[tuple[str, str, str]] = []

    def listing(category: str, lo: str, hi: str, *, archive: Path | None = None) -> list[dict]:
        calls.append((category, lo, hi))
        return [_paper(pid) for pid in ids]

    listing.calls = calls  # type: ignore[attr-defined]
    return listing


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
        out = jva.arxiv_window_controls([POSITIVE], {}, "SEED", listing=listing)
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
        out = jva.arxiv_window_controls([POSITIVE], {}, "SEED", listing=listing)
        assert "2103.00001" not in {c["id"] for c in out}

    def test_a_paper_with_no_abstract_is_skipped(self) -> None:
        """Both judges are shown the abstract; a control without one is not the same task."""

        def listing(category: str, lo: str, hi: str, *, archive: Path | None = None) -> list[dict]:
            return [
                {"arxiv_id": "2104.00001", "abstract": ""},
                *[_paper(f"2104.0001{i}") for i in range(5)],
            ]

        out = jva.arxiv_window_controls([POSITIVE], {}, "SEED", listing=listing)
        assert "2104.00001" not in {c["id"] for c in out}

    def test_controls_are_not_reused_across_positives_in_one_repo(self) -> None:
        """Reuse would make the negative class smaller than it looks and correlate the
        per-positive comparisons inside a repository, which the cluster bootstrap then
        cannot see."""
        listing = _listing_of([f"2104.000{i:02d}" for i in range(20)])
        second = {**POSITIVE, "id": "2104.09999"}
        out = jva.arxiv_window_controls([POSITIVE, second], {}, "SEED", listing=listing)
        assert len(out) == 8
        assert len({c["id"] for c in out}) == 8

    def test_the_listing_is_fetched_once_per_category_and_window(self) -> None:
        """Positives cluster inside a repository and a field; re-fetching per positive would
        multiply arXiv requests for an identical answer."""
        listing = _listing_of([f"2104.000{i:02d}" for i in range(20)])
        same_window = {**POSITIVE, "id": "2105.00002", "published": "2021-05-02"}
        jva.arxiv_window_controls([POSITIVE, same_window], {}, "SEED", listing=listing)
        assert len(listing.calls) == 1

    def test_a_positive_missing_its_category_is_skipped_rather_than_guessed(self) -> None:
        listing = _listing_of([f"2104.0000{i}" for i in range(6)])
        bare = {"case": "acme/rich", "id": "2103.00002"}
        assert jva.arxiv_window_controls([bare], {}, "SEED", listing=listing) == []


class TestTheDrawIsSeeded:
    def test_the_same_seed_gives_the_same_controls(self) -> None:
        listing = _listing_of([f"2104.000{i:02d}" for i in range(30)])
        a = jva.arxiv_window_controls([POSITIVE], {}, "SEED-1", listing=listing)
        b = jva.arxiv_window_controls([POSITIVE], {}, "SEED-1", listing=listing)
        assert [c["id"] for c in a] == [c["id"] for c in b]

    def test_a_different_seed_gives_different_controls(self) -> None:
        listing = _listing_of([f"2104.000{i:02d}" for i in range(30)])
        a = jva.arxiv_window_controls([POSITIVE], {}, "SEED-1", listing=listing)
        b = jva.arxiv_window_controls([POSITIVE], {}, "SEED-2", listing=listing)
        assert [c["id"] for c in a] != [c["id"] for c in b]

    def test_the_draw_does_not_depend_on_the_listing_order(self) -> None:
        """arXiv returns results in its own order; a control set that depended on it would
        not be reproducible from the archived listing."""
        ids = [f"2104.000{i:02d}" for i in range(30)]
        forward = jva.arxiv_window_controls([POSITIVE], {}, "S", listing=_listing_of(ids))
        backward = jva.arxiv_window_controls(
            [POSITIVE], {}, "S", listing=_listing_of(list(reversed(ids)))
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
            [POSITIVE], {}, "SEED", listing=listing, archive=tmp_path / "listings"
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
        Watching only the gold cache would miss a namespace collision there."""
        import inspect

        src = inspect.getsource(jva.judge)
        assert "cache_roots = (JUDGE_CACHE, SECOND_JUDGE_CACHE)" in src
        assert "assert_caches_untouched(cache_before, cache_roots)" in src

    def test_the_gold_set_is_checked_too(self) -> None:
        """Two independent guards: the fingerprint catches a write, `resolve_targets` catches
        the consequence. Either alone can pass while the other fails."""
        import inspect

        src = inspect.getsource(jva.judge)
        assert "targets_before = resolve_targets()" in src
        assert "resolve_targets() != targets_before" in src

    def test_the_guards_run_after_the_verdicts_are_written(self) -> None:
        """A violation must be reported without also losing the expensive run that revealed
        it."""
        import inspect

        src = inspect.getsource(jva.judge)
        last_write = src.rindex("VERDICTS.write_text")
        assert src.index("assert_caches_untouched(cache_before") > last_write


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
        assert jva.arxiv_window_controls(self.ROWS, {}, "SEED", listing=listing) == []

        def fetch(ids: list[str]) -> list[dict]:
            return [
                {"arxiv_id": "2103.00001", "categories": ["cs.LG"], "published": "2021-03-14"},
                {"arxiv_id": "2104.00002", "categories": ["cs.LG"], "published": "2021-04-02"},
            ]

        enriched, _ = jva.enrich_positives(self.ROWS, fetch=fetch)
        assert len(jva.arxiv_window_controls(enriched, {}, "SEED", listing=listing)) == 8


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
