"""The cached-`raw` replay, and the one case where it must not run.

`run_baseline` re-parses the cached model answer on every cache hit, so a parser fix
reaches already-cached runs. That is right, and it had one consequence nobody had looked
at: three `cli` caches (`compiler`, `graph`, `storage`) hold a 128-character restoration
note where their transcript used to be, after a 30-turn re-run displaced the 12-turn entry
on 2026-08-09 and only the ids survived, recovered from a run record. Replaying a note
yields nothing, so those three scored as **abstentions** in every run since — while
`diagnose_pool.actionable_baseline_ids`, reading the `ids` field of the same file, counted
their seven targets. One cache, two consumers, opposite answers, and the published
comparator forfeited +0.28 net@2/case of its own picks.

The fallback added for that is deliberately narrow, and these tests pin both halves:

* a cache whose `raw` holds **no recommendation block** falls back to stored ids;
* a cache whose `raw` holds an explicit ``[]`` keeps **zero**, because that is an answer.

The second is not hypothetical. `webdev` says *"My recommendation is to recommend nothing"*,
emits ``[]``, and still carries four ids an older parser scraped out of its prose —
including `publication/2256929`, a bare URL path. A fallback keyed on "no ids parsed" rather
than "no block present" would resurrect exactly the picks the authoritative-block rule was
written to discard.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))

import baseline as baseline_mod  # noqa: E402

RESTORED_NOTE = (
    "restored 2026-08-09 from results/judge-gpt-5.5-20260809T172421Z.json "
    "after a 30-turn baseline re-run displaced the 12-turn entry"
)
REAL_ANSWER = (
    'I reviewed the repository.\n\n```json\n[{"arxiv_id": "2401.12345", "title": "A Paper"}]\n```'
)
EXPLICIT_ABSTENTION = "I reviewed it and found nothing applicable.\n\n```json\n[]\n```"


class TestHasAnswerBlock:
    def test_a_restoration_note_is_not_an_answer(self):
        assert baseline_mod._has_answer_block(RESTORED_NOTE) is False

    def test_a_recommendation_is_an_answer(self):
        assert baseline_mod._has_answer_block(REAL_ANSWER) is True

    def test_an_empty_array_is_an_answer(self):
        """The whole point: `[]` is a decision, not an absence."""
        assert baseline_mod._has_answer_block(EXPLICIT_ABSTENTION) is True


def _cached(tmp_path, monkeypatch, raw: str, ids: list[str]) -> dict:
    monkeypatch.setattr(baseline_mod, "CACHE_DIR", tmp_path)
    disc = baseline_mod._discriminator("cli", "", None)
    path = tmp_path / "cli" / "case.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"status": "ok", "raw": raw, "ids": ids, "titles": [], "_disc": disc}),
        encoding="utf-8",
    )
    return baseline_mod.run_baseline(tmp_path, repo_name="case", mode="cli")


class TestReplayFallback:
    def test_restoration_note_recovers_stored_ids(self, tmp_path, monkeypatch):
        out = _cached(tmp_path, monkeypatch, RESTORED_NOTE, ["1601.05400", "2004.03082"])
        assert out["ids"] == ["1601.05400", "2004.03082"]

    def test_explicit_abstention_stays_empty(self, tmp_path, monkeypatch):
        """`webdev`'s shape: an answer of `[]` plus stale scraped ids."""
        out = _cached(
            tmp_path, monkeypatch, EXPLICIT_ABSTENTION, ["publication/2256929", "2410.14924"]
        )
        assert out["ids"] == [], "an explicit [] must not be overridden by stale ids"

    def test_a_real_answer_is_replayed_not_overridden(self, tmp_path, monkeypatch):
        """`rag`'s shape: raw parses to fewer ids than are stored. Raw wins."""
        out = _cached(tmp_path, monkeypatch, REAL_ANSWER, ["2401.12345", "9999.99999"])
        assert out["ids"] == ["2401.12345"]

    def test_no_stored_ids_and_no_block_is_still_empty(self, tmp_path, monkeypatch):
        out = _cached(tmp_path, monkeypatch, RESTORED_NOTE, [])
        assert out["ids"] == []


class TestTheThreeAffectedCachesInThisRepo:
    """Guards the actual artifacts, where they exist."""

    @pytest.mark.parametrize(("case", "n"), [("compiler", 2), ("graph", 3), ("storage", 2)])
    def test_restored_caches_report_their_picks(self, case, n):
        path = ROOT / "evals" / "cache" / "baseline" / "cli" / f"{case}.json"
        if not path.is_file():
            pytest.skip("no local baseline cache")
        data = json.loads(path.read_text(encoding="utf-8"))
        assert baseline_mod._has_answer_block(data.get("raw") or "") is False
        assert len(data.get("ids") or []) == n

    def test_webdev_still_abstains(self):
        path = ROOT / "evals" / "cache" / "baseline" / "cli" / "webdev.json"
        if not path.is_file():
            pytest.skip("no local baseline cache")
        data = json.loads(path.read_text(encoding="utf-8"))
        assert baseline_mod._has_answer_block(data["raw"]) is True
        assert baseline_mod._parse_recommendations(data["raw"])[0] == []
