"""The registered control scheme actually runs, and both arms look the same. [PREREG §4]

Three things had to be true before a verdict could be bought and none of them was.

`CONTROL_SCHEME` was assigned from `--controls` and never read, so `arxiv-window` — the scheme
§4 registers — selected nothing: the flag picked the output path and the provenance label while
the pool scheme drew the papers. `arxiv_window_controls`, `arxiv_window_listing` and
`enrich_positives` had no production caller anywhere in the tree.

The draw matched on `categories[0]`, which is feed tag order rather than arXiv's own primary
category, against a `cat:` query that matches cross-lists too.

And the two arms were assembled by two different fetch paths with two different
normalisations. Measured: **674 of 674** control-shaped papers in `.work/pool-cut100/ann.json`
carry a versioned `arxiv_id` and **0 of 120** mined positives do, while 82 of those 674
abstracts contain newlines a positive cannot have. `judge._build_user_prompt` prints both
verbatim, so every control prompt read `arXiv: 2409.11629v1` and every positive
`arXiv: 2409.11629` — a deterministic arm marker in the one place §4 says must carry none.
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

import judge_validity_adoption as jva  # noqa: E402
import judge_validity_pool as jvp  # noqa: E402

SEED = "PULSE-FIXTURE"


def _positives(n: int = 2, case: str = "acme/rich") -> list[dict[str, object]]:
    return [{"case": case, "id": f"2103.{i:05d}", "t0_commit_date": "2022-01-01"} for i in range(n)]


def _enrich(rows, **_k):  # noqa: ANN001, ANN202
    return [
        {
            **r,
            "primary_category": "cs.LG",
            "published": "2021-03-01",
            "paper": {"arxiv_id": r["id"]},
        }
        for r in rows
    ], []


def _listing(n: int = 40, primary: str = "cs.LG"):  # noqa: ANN202
    def listing(category, lo, hi, *, want=200, archive=None, **_k):  # noqa: ANN001
        return [
            {
                "arxiv_id": f"2104.{i:05d}v1",
                "title": f"t{i}",
                "abstract": "an abstract",
                "primary_category": primary,
            }
            for i in range(n)
        ]

    return listing


class TestTheSchemeIsActuallyRead:
    def test_the_pool_branch_is_the_published_draw(self) -> None:
        """That branch reproduces NR-56/57's numbers, so it is this file's own body unchanged."""
        import inspect

        assert "return pool_controls(rng)" in inspect.getsource(jva.controls)

    def test_arxiv_window_routes_to_the_drawer_that_has_the_inputs(self) -> None:
        """Not a stub: §4's draw needs the v2 positives, the materialised HEAD citation sets and
        the verified SEED_POOL, none of which `controls()` has. Pointing at the tool that does
        beats quietly drawing the wrong negative class under the right name."""
        with pytest.raises(SystemExit) as exc:
            jva.controls(scheme="arxiv-window")
        assert "judge_validity_pool.draw_controls" in str(exc.value)

    def test_an_unknown_scheme_names_the_registered_ones(self) -> None:
        with pytest.raises(SystemExit) as exc:
            jva.controls(scheme="newest-200")
        assert "('pool', 'arxiv-window')" in str(exc.value)

    def test_the_flag_still_reaches_the_switch(self) -> None:
        import inspect

        assert "CONTROL_SCHEME = args.controls" in inspect.getsource(jva.main)


class TestTheMatchIsOnTheArxivPrimaryCategory:
    def test_a_cross_listed_paper_is_not_a_match(self) -> None:
        """`cat:cs.LG` matches any category including cross-lists, so the listing is a superset.
        Without the filter a `cs.LG` positive draws controls whose primary is `stat.ML`, and the
        AUC partly measures primary-versus-cross-list rather than adoption."""
        enriched, _ = _enrich(_positives(1))
        with pytest.raises(SystemExit) as exc:
            jva.arxiv_window_controls(
                enriched, {"acme/rich": set()}, SEED, listing=_listing(primary="stat.ML")
            )
        assert "drew ZERO controls" in str(exc.value)

    def test_a_matching_primary_is_drawn(self) -> None:
        enriched, _ = _enrich(_positives(1))
        out = jva.arxiv_window_controls(
            enriched, {"acme/rich": set()}, SEED, listing=_listing(primary="cs.LG")
        )
        assert len(out) == 4
        assert {c["primary_category"] for c in out} == {"cs.LG"}

    def test_the_collector_now_keeps_it(self) -> None:
        """`categories[0]` is feed tag order and is not a promise about which one is primary."""
        from reporadar.collector import _result_to_paper

        class R:
            entry_id = "http://arxiv.org/abs/2401.00001v1"
            title = "T"
            summary = "S"
            authors: list[object] = []
            categories = ["cs.CL", "cs.AI"]
            primary_category = "cs.AI"
            pdf_url = ""

            def get_short_id(self) -> str:
                return "2401.00001v1"

            class _P:
                @staticmethod
                def isoformat() -> str:
                    return "2024-01-01"

            published = _P()
            updated = None

        assert _result_to_paper(R())["primary_category"] == "cs.AI"


class TestEveryPositiveGetsControlsOrTheRunStops:
    def test_a_positive_that_drew_nothing_raises(self) -> None:
        """`pool[:n]` accepted a short draw silently and the global guard fires only when EVERY
        positive drew nothing. An unmatched positive still enters the point estimate and every
        bootstrap draw with no compensating negatives, and cluster resampling duplicates it."""
        enriched, _ = _enrich(_positives(1))
        with pytest.raises(SystemExit) as exc:
            jva.arxiv_window_controls(
                enriched, {"acme/rich": set()}, SEED, listing=lambda *a, **k: []
            )
        assert "ZERO controls" in str(exc.value)

    def test_a_short_draw_is_reported_not_raised(self, capsys) -> None:
        """A thin window is a property of the population, not a defect — but the realised
        controls-per-positive distribution is what n2 actually was, so it has to be visible."""
        enriched, _ = _enrich(_positives(1))
        out = jva.arxiv_window_controls(enriched, {"acme/rich": set()}, SEED, listing=_listing(n=2))
        assert len(out) == 2
        assert "only 2 of 4" in capsys.readouterr().out

    def test_a_missing_citation_set_is_never_defaulted_to_empty(self) -> None:
        enriched, _ = _enrich(_positives(1))
        with pytest.raises(SystemExit) as exc:
            jva.arxiv_window_controls(enriched, {}, SEED, listing=_listing())
        assert "no HEAD citation set" in str(exc.value)


class TestTheListingIsArchivedWriteOnce:
    def _draw(self, tmp_path: Path, listing, **over):  # noqa: ANN001, ANN202
        return jvp.draw_controls(
            _positives(1),
            seed=SEED,
            head_ids={"acme/rich": set()},
            rows_out=tmp_path / "rows.json",
            payload_out=tmp_path / "payload.json",
            raw=tmp_path / "raw",
            manifest=tmp_path / "man",
            listing=listing,
            enrich=_enrich,
            **over,
        )

    def test_the_committed_manifest_carries_no_paper_text(self, tmp_path: Path) -> None:
        """§2.1's no-URL rule and §11's prior-exposure coupling: arXiv abstracts routinely carry
        `github.com/<owner>/<repo>` strings, and this pool must not contaminate that grep."""
        self._draw(tmp_path, _listing())
        man = json.loads((tmp_path / "man" / "cs_LG-202101.json").read_text(encoding="utf-8"))
        assert set(man) >= {"ids", "primary_categories", "requested", "returned", "raw_sha256"}
        assert "papers" not in man  # ids and flags only; `has_abstract` is a flag, not text
        blob = json.dumps(man)
        assert "github.com" not in blob and "http" not in blob
        assert "an abstract" not in blob and "t0" not in man

    def test_the_raw_payload_is_kept_untracked_beside_it(self, tmp_path: Path) -> None:
        self._draw(tmp_path, _listing())
        raw = json.loads((tmp_path / "raw" / "cs_LG-202101.json").read_text(encoding="utf-8"))
        assert raw["papers"] and "abstract" in raw["papers"][0]

    def test_a_second_run_replays_rather_than_refetching(self, tmp_path: Path) -> None:
        """arXiv `cat:` membership drifts as papers are cross-listed and withdrawn, so a second
        fetch can return a different set. Preferring either would make the negative class depend
        on when somebody ran it."""
        first = self._draw(tmp_path, _listing())

        def explode(*_a: object, **_k: object) -> None:
            raise AssertionError("re-fetched an archived window")

        assert [c["id"] for c in self._draw(tmp_path, explode)] == [c["id"] for c in first]

    def test_an_edited_archive_is_a_blocking_failure(self, tmp_path: Path) -> None:
        self._draw(tmp_path, _listing())
        raw_file = tmp_path / "raw" / "cs_LG-202101.json"
        stored = json.loads(raw_file.read_text(encoding="utf-8"))
        stored["papers"][0]["arxiv_id"] = "9999.99999"
        raw_file.write_text(json.dumps(stored), encoding="utf-8")
        (tmp_path / "rows.json").unlink()
        with pytest.raises(SystemExit) as exc:
            self._draw(tmp_path, _listing())
        assert "disagree" in str(exc.value)


class TestTheDrawnSetIsAnArtefactNotARecomputation:
    def _draw(self, tmp_path: Path, seed: str = SEED, listing=None):  # noqa: ANN001, ANN202
        return jvp.draw_controls(
            _positives(2),
            seed=seed,
            head_ids={"acme/rich": set()},
            rows_out=tmp_path / "rows.json",
            payload_out=tmp_path / "payload.json",
            raw=tmp_path / "raw",
            manifest=tmp_path / "man",
            listing=listing or _listing(),
            enrich=_enrich,
        )

    def test_it_is_written_once_and_replayed(self, tmp_path: Path) -> None:
        """`plan`, the purchase loop and the analysis must see the same negatives; a redraw
        between them would change n2 after verdicts had been bought."""
        a = self._draw(tmp_path)
        b = self._draw(tmp_path)
        assert [c["id"] for c in a] == [c["id"] for c in b]

    def test_a_different_seed_against_an_existing_draw_is_refused(self, tmp_path: Path) -> None:
        self._draw(tmp_path)
        with pytest.raises(SystemExit) as exc:
            self._draw(tmp_path, seed="OTHER-PULSE")
        assert "different seed" in str(exc.value)

    def test_the_committed_rows_carry_no_paper_text(self, tmp_path: Path) -> None:
        self._draw(tmp_path)
        rows = json.loads((tmp_path / "rows.json").read_text(encoding="utf-8"))
        assert all("paper" not in row for row in rows["controls"])
        assert rows["seed"] == SEED and rows["scheme"] == "arxiv-window"

    def test_cross_cluster_sharing_is_reported_rather_than_deduplicated(
        self, tmp_path: Path
    ) -> None:
        """Dropping the loser of a contest after the draw would strip a positive of controls and
        change n2 — an instrument change. It is a correlation the repository-cluster bootstrap
        does not capture, so the interval is very slightly too narrow, and that is stated."""
        self._draw(tmp_path)
        rows = json.loads((tmp_path / "rows.json").read_text(encoding="utf-8"))
        assert "n_control_papers_in_more_than_one_cluster" in rows
        assert rows["controls_per_positive"] == {"4": 2}


class TestNeitherArmCanBeIdentifiedByItsShape:
    def test_a_versioned_control_id_is_canonicalised(self) -> None:
        """674 of 674 shipped candidates are versioned; 0 of 120 mined positives are."""
        items, _ = jvp.judgeable_items(
            [{"case": "g", "id": "2401.00001"}],
            [{"case": "g", "id": "2402.00002", "paper": {"arxiv_id": "2402.00002v3"}}],
            fetch=lambda ids: {"2401.00001": {"title": "P", "abstract": "x"}},
        )
        assert [i["arxiv_id"] for i in items] == ["2401.00001", "2402.00002"]

    def test_abstracts_are_normalised_the_same_way_for_both_arms(self) -> None:
        """`fetch_papers` collapses whitespace and `_result_to_paper` does not; 82 of 674
        control abstracts carry newlines a positive never can."""
        items, _ = jvp.judgeable_items(
            [{"case": "g", "id": "2401.00001"}],
            [
                {
                    "case": "g",
                    "id": "2402.00002",
                    "paper": {"arxiv_id": "2402.00002", "abstract": "a\n\nb"},
                }
            ],
            fetch=lambda ids: {"2401.00001": {"title": "P", "abstract": "x  \n y"}},
        )
        assert [i["abstract"] for i in items] == ["x y", "a b"]

    def test_a_marker_that_survived_assembly_is_refused_before_any_purchase(self) -> None:
        with pytest.raises(SystemExit) as exc:
            jvp.assert_arm_neutral(
                [
                    {
                        "case": "g",
                        "arm": "adopted",
                        "arxiv_id": "2401.00001",
                        "title": "",
                        "abstract": "",
                    },
                    {
                        "case": "g",
                        "arm": "control",
                        "arxiv_id": "2402.00002v1",
                        "title": "",
                        "abstract": "",
                    },
                ]
            )
        assert "arm marker" in str(exc.value)

    def test_an_unnormalised_abstract_is_refused(self) -> None:
        with pytest.raises(SystemExit) as exc:
            jvp.assert_arm_neutral(
                [
                    {
                        "case": "g",
                        "arm": "adopted",
                        "arxiv_id": "2401.00001",
                        "title": "",
                        "abstract": "a\nb",
                    },
                    {
                        "case": "g",
                        "arm": "control",
                        "arxiv_id": "2402.00002",
                        "title": "",
                        "abstract": "a b",
                    },
                ]
            )
        assert "not normalised" in str(exc.value)

    def test_one_arm_alone_is_refused(self) -> None:
        with pytest.raises(SystemExit) as exc:
            jvp.assert_arm_neutral(
                [
                    {
                        "case": "g",
                        "arm": "adopted",
                        "arxiv_id": "2401.00001",
                        "title": "",
                        "abstract": "",
                    }
                ]
            )
        assert "one arm" in str(exc.value)

    def test_a_paper_that_could_not_be_fetched_is_returned_not_dropped(self) -> None:
        """A positive that never reaches the judge shrinks n without shrinking anything
        visible."""
        items, missing = jvp.judgeable_items(
            [{"case": "g", "id": "2401.00001"}, {"case": "g", "id": "2401.00009"}],
            [{"case": "g", "id": "2402.00002", "paper": {"arxiv_id": "2402.00002"}}],
            fetch=lambda ids: {"2401.00001": {"title": "P", "abstract": "x"}},
        )
        assert missing == ["2401.00009"]
        assert len(items) == 2
