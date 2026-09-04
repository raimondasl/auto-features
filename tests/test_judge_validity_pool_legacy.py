"""The legacy stratum gets what the walk gives the pool stratum. [PREREG §4, §5, §7]

§4 says "The legacy 35 are re-run under this control scheme", and that is only a re-run if the
legacy arm is shown the same kind of prompt through the same code. `adoptions-v2.json` carries
no T0 context, no HEAD citation set, no adoption date and no realised T0 commit date, so
without this pass §5's transportability contrast would compare two code paths rather than two
populations — and the primary would be pool-only, around 100 positives where §9's power table
budgets 130.

It is a MINING pass, not an analysis one. It reads the live promisor clones at each row's
recorded SHAs, and cannot be deferred to analysis time: mining after the verdicts are visible
is the thing the whole design is arranged to prevent.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / "evals", ROOT / "evals" / "frame", ROOT / "src"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import judge_validity_pool as jvp  # noqa: E402
import walk_pool as wp  # noqa: E402

GOOD_CONTEXT = (
    "Repository: graph\n\n## README (excerpt)\nreal prose\n\n## Source files (sample)\na.py\n"
)


@pytest.fixture
def legacy_source(tmp_path: Path) -> Path:
    src = tmp_path / "adoptions-v2.json"
    src.write_text(
        json.dumps(
            [
                {"case": "graph", "id": "2401.00001", "usable": True, "head": "H", "t0": "T"},
                {"case": "graph", "id": "2401.00002", "usable": True, "head": "H", "t0": "T"},
                {"case": "graph", "id": "2401.00003", "usable": False, "head": "H", "t0": "T"},
            ]
        ),
        encoding="utf-8",
    )
    return src


@pytest.fixture
def fake_clone(tmp_path: Path) -> Path:
    root = tmp_path / "fullclone" / "graph"
    (root / ".git").mkdir(parents=True)
    return root.parent


def _patch(monkeypatch, *, context=GOOD_CONTEXT, cited=("2401.00001", "2401.00002"), commit=True):
    import mine_adoptions as ma

    monkeypatch.setattr(jvp, "_git", lambda repo, *a, **k: "2024-05-01T00:00:00+00:00\n")
    monkeypatch.setattr(ma, "t0_context", lambda repo, case, rev, **k: context)
    monkeypatch.setattr(
        ma, "ids_with_paths", lambda repo, rev, ex, t=None: dict.fromkeys(cited, [])
    )
    monkeypatch.setattr(jvp, "_count_at", lambda *a, **k: 7)


def _dater(commit=True):  # noqa: ANN001, ANN202
    """Injected rather than monkeypatched onto `walk_pool`: `tests/test_pool_walk.py`
    re-registers its own module under that name in `sys.modules`, so a patch would land on
    whichever object happened to be there when this test ran."""
    return (lambda *a, **k: ("abc123", "2024-06-01")) if commit else (lambda *a, **k: (None, None))


class TestItMinesFromThePinnedRevisions:
    def test_it_writes_a_context_at_the_walks_own_digest(
        self, tmp_path, monkeypatch, legacy_source, fake_clone
    ) -> None:
        """§7 pins the legacy re-mine to each row's recorded SHAs "so that t0 reproduces to the
        SHA and the stored T0 verdicts remain verdicts about the same prompt"."""
        _patch(monkeypatch)
        out = jvp.materialise_legacy(
            source=legacy_source,
            clones=fake_clone,
            contexts=tmp_path / "ctx",
            head_ids=tmp_path / "ids",
            out=tmp_path / "sidecar.json",
            dater=_dater(),
        )
        digest = out["cases"]["graph"]["context_digest"]
        assert digest == wp.context_hash("t0", GOOD_CONTEXT)
        written = tmp_path / "ctx" / f"graph.{digest}.txt"
        assert written.read_text(encoding="utf-8") == GOOD_CONTEXT
        # The loader must accept exactly what this pass wrote, through the same digest scheme.
        assert jvp.t0_context_for("graph", digest, tmp_path / "ctx") == GOOD_CONTEXT

    def test_the_head_citation_set_is_written_where_the_control_drawer_reads_it(
        self, tmp_path, monkeypatch, legacy_source, fake_clone
    ) -> None:
        _patch(monkeypatch)
        jvp.materialise_legacy(
            source=legacy_source,
            clones=fake_clone,
            contexts=tmp_path / "ctx",
            head_ids=tmp_path / "ids",
            out=tmp_path / "sidecar.json",
            dater=_dater(),
        )
        assert jvp.head_ids_for(["graph"], tmp_path / "ids") == {
            "graph": {"2401.00001", "2401.00002"}
        }

    def test_it_records_the_realised_t0_date_not_only_the_nominal_one(
        self, tmp_path, monkeypatch, legacy_source, fake_clone
    ) -> None:
        """§3.1: "the nominal cutoff is not the realised one" — `rev-list --before` can land
        years earlier across a history gap, and §4 matches controls on the realised date.
        Legacy rows carry only `t0_date`."""
        _patch(monkeypatch)
        out = jvp.materialise_legacy(
            source=legacy_source,
            clones=fake_clone,
            contexts=tmp_path / "ctx",
            head_ids=tmp_path / "ids",
            out=tmp_path / "sidecar.json",
            dater=_dater(),
        )
        assert out["cases"]["graph"]["t0_commit_date"] == "2024-05-01"

    def test_it_carries_the_doi_and_pmid_covariates(
        self, tmp_path, monkeypatch, legacy_source, fake_clone
    ) -> None:
        """§6.2 SIZES the life-science blind spot with these rather than closing it, so the
        legacy arm must carry them or the covariate table is pool-only."""
        _patch(monkeypatch)
        out = jvp.materialise_legacy(
            source=legacy_source,
            clones=fake_clone,
            contexts=tmp_path / "ctx",
            head_ids=tmp_path / "ids",
            out=tmp_path / "sidecar.json",
            dater=_dater(),
        )
        case = out["cases"]["graph"]
        assert case["dois_head"] == case["dois_t0"] == case["pmids_head"] == case["pmids_t0"] == 7

    def test_only_usable_rows_are_mined(
        self, tmp_path, monkeypatch, legacy_source, fake_clone
    ) -> None:
        _patch(monkeypatch)
        out = jvp.materialise_legacy(
            source=legacy_source,
            clones=fake_clone,
            contexts=tmp_path / "ctx",
            head_ids=tmp_path / "ids",
            out=tmp_path / "sidecar.json",
            dater=_dater(),
        )
        assert set(out["cases"]["graph"]["adoptions"]) == {"2401.00001", "2401.00002"}


class TestNothingIsWrittenBackIntoTheImmutableRecord:
    def test_the_source_artefact_is_untouched(
        self, tmp_path, monkeypatch, legacy_source, fake_clone
    ) -> None:
        """§1: "The v1 record is immutable ... because it is the record v2 is compared against"."""
        _patch(monkeypatch)
        before = legacy_source.read_bytes()
        jvp.materialise_legacy(
            source=legacy_source,
            clones=fake_clone,
            contexts=tmp_path / "ctx",
            head_ids=tmp_path / "ids",
            out=tmp_path / "sidecar.json",
            dater=_dater(),
        )
        assert legacy_source.read_bytes() == before

    def test_the_sidecar_goes_through_the_artefact_boundary(
        self, tmp_path, monkeypatch, legacy_source, fake_clone
    ) -> None:
        _patch(monkeypatch)
        jvp.materialise_legacy(
            source=legacy_source,
            clones=fake_clone,
            contexts=tmp_path / "ctx",
            head_ids=tmp_path / "ids",
            out=tmp_path / "sidecar.json",
            dater=_dater(),
        )
        written = json.loads((tmp_path / "sidecar.json").read_text(encoding="utf-8"))
        assert written["_artefact"] == jvp.ARTEFACT_MARKER


class TestASilentEmptyResultIsRefused:
    def test_a_missing_clone_is_named_rather_than_mined_around(
        self, tmp_path, legacy_source
    ) -> None:
        """The clones are promisors and this pass cannot be deferred — so an absent one is a
        refusal now, not an empty context later."""
        with pytest.raises(SystemExit) as exc:
            jvp.materialise_legacy(
                source=legacy_source,
                clones=tmp_path / "nowhere",
                contexts=tmp_path / "ctx",
                head_ids=tmp_path / "ids",
                out=tmp_path / "sidecar.json",
            )
        assert "no clone" in str(exc.value)

    def test_a_failing_git_call_raises_instead_of_returning_empty(self, tmp_path) -> None:
        """`mine_adoptions` runs git with no `check`, so a failure comes back as "" and reads
        downstream as "this repository has nothing". Every blob here is a lazy fetch that can
        fail on a network blip."""
        repo = tmp_path / "repo"
        repo.mkdir()
        with pytest.raises(SystemExit) as exc:
            jvp._git(repo, "rev-parse", "--verify", "deadbeef^{commit}")
        assert "failed in repo" in str(exc.value)

    def test_zero_head_identifiers_is_refused_for_a_contributing_case(
        self, tmp_path, monkeypatch, legacy_source, fake_clone
    ) -> None:
        """A case that contributed usable adoptions cannot cite nothing at HEAD. An empty set
        would make §4's never-cited rule a no-op and put positives in the negative class."""
        _patch(monkeypatch, cited=())
        with pytest.raises(SystemExit) as exc:
            jvp.materialise_legacy(
                source=legacy_source,
                clones=fake_clone,
                contexts=tmp_path / "ctx",
                head_ids=tmp_path / "ids",
                out=tmp_path / "sidecar.json",
            )
        assert "zero identifiers at HEAD" in str(exc.value)

    def test_a_header_only_context_is_refused(
        self, tmp_path, monkeypatch, legacy_source, fake_clone
    ) -> None:
        _patch(monkeypatch, context="Repository: graph\n")
        with pytest.raises(SystemExit) as exc:
            jvp.materialise_legacy(
                source=legacy_source,
                clones=fake_clone,
                contexts=tmp_path / "ctx",
                head_ids=tmp_path / "ids",
                out=tmp_path / "sidecar.json",
            )
        assert "header and nothing else" in str(exc.value)

    def test_rows_disagreeing_about_the_pin_are_refused(
        self, tmp_path, monkeypatch, fake_clone
    ) -> None:
        """Otherwise the context, the citation set and the adoption dates would describe
        different revisions of one repository."""
        src = tmp_path / "adoptions-v2.json"
        src.write_text(
            json.dumps(
                [
                    {"case": "graph", "id": "2401.00001", "usable": True, "head": "H", "t0": "T"},
                    {"case": "graph", "id": "2401.00002", "usable": True, "head": "H2", "t0": "T"},
                ]
            ),
            encoding="utf-8",
        )
        _patch(monkeypatch)
        with pytest.raises(SystemExit) as exc:
            jvp.materialise_legacy(
                source=src,
                clones=fake_clone,
                contexts=tmp_path / "ctx",
                head_ids=tmp_path / "ids",
                out=tmp_path / "sidecar.json",
            )
        assert "not one pin" in str(exc.value)


class TestAnUnavailableAdoptionDateIsRecordedNotBlank:
    def test_a_pickaxe_timeout_is_null_with_its_reason(
        self, tmp_path, monkeypatch, legacy_source, fake_clone
    ) -> None:
        """`git log -S` diffs every commit's documents across the window, and every blob is a
        lazy fetch. Measured on `diffusion`: ONE identifier exceeds 300 s, so its 46 would run
        for hours. Bounded and recorded — the only consumer is §5's contamination split, which
        is VOID for this pool because no published training cutoff exists for either judge."""
        _patch(monkeypatch)

        def slow(*_a: object, **_k: object) -> tuple[str, str]:
            raise subprocess.TimeoutExpired("git log -S", 20.0)

        out = jvp.materialise_legacy(
            source=legacy_source,
            clones=fake_clone,
            contexts=tmp_path / "ctx",
            head_ids=tmp_path / "ids",
            out=tmp_path / "sidecar.json",
            dater=slow,
        )
        entry = out["cases"]["graph"]["adoptions"]["2401.00001"]
        assert entry["adoption_date"] is None
        assert "pickaxe exceeded" in entry["note"]
        assert out["cases"]["graph"]["n_adoption_dates"] == 0

    def test_a_paper_with_no_introducing_commit_says_so(
        self, tmp_path, monkeypatch, legacy_source, fake_clone
    ) -> None:
        _patch(monkeypatch)
        out = jvp.materialise_legacy(
            source=legacy_source,
            clones=fake_clone,
            contexts=tmp_path / "ctx",
            head_ids=tmp_path / "ids",
            out=tmp_path / "sidecar.json",
            dater=_dater(commit=False),
        )
        entry = out["cases"]["graph"]["adoptions"]["2401.00001"]
        assert entry["adoption_date"] is None
        assert "no commit introduced" in entry["note"]

    def test_coverage_is_reported_per_case(
        self, tmp_path, monkeypatch, legacy_source, fake_clone
    ) -> None:
        _patch(monkeypatch)
        out = jvp.materialise_legacy(
            source=legacy_source,
            clones=fake_clone,
            contexts=tmp_path / "ctx",
            head_ids=tmp_path / "ids",
            out=tmp_path / "sidecar.json",
            dater=_dater(),
        )
        assert out["cases"]["graph"]["n_adoption_dates"] == 2


class TestTheAnalysisRefusesToProceedWithoutIt:
    def test_an_absent_sidecar_names_the_command_and_the_consequence(self, tmp_path) -> None:
        with pytest.raises(SystemExit) as exc:
            jvp.legacy_sidecar(tmp_path / "absent.json")
        message = str(exc.value)
        assert "--materialise-legacy" in message
        assert "two code paths" in message

    def test_a_present_sidecar_loads(
        self, tmp_path, monkeypatch, legacy_source, fake_clone
    ) -> None:
        _patch(monkeypatch)
        jvp.materialise_legacy(
            source=legacy_source,
            clones=fake_clone,
            contexts=tmp_path / "ctx",
            head_ids=tmp_path / "ids",
            out=tmp_path / "sidecar.json",
            dater=_dater(),
        )
        assert jvp.legacy_sidecar(tmp_path / "sidecar.json")["n_cases"] == 1
