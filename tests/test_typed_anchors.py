"""Tests for `reporadar.typed_anchors` (the P9/P10/P11 channel, shipped off by default).

The module's two invariants are the ones the measurement record cares about:

* **verbatim or discarded** — NR-12 measured LLM paraphrase as *worse than sending nothing*
  as gate context, so a span the README does not literally contain is not an anchor;
* **failure is loud** — an opt-in stage that silently yields nothing turns a treatment arm
  into the control arm and reports the difference as a null, which is the C-9 shape.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reporadar import typed_anchors as ta
from reporadar.llm_client import LLMError


class _Reply:
    """Stand-in for `llm_client.complete`, recording how often it was called."""

    def __init__(self, text: str) -> None:
        self.text, self.calls = text, 0

    def __call__(self, prompt: str, cfg: object, **kw: object) -> str:
        self.calls += 1
        return self.text


def _repo(tmp_path: Path, readme: str, name: str = "README.md") -> Path:
    (tmp_path / name).write_text(readme, encoding="utf-8")
    return tmp_path


class TestParseEntities:
    def test_plain_and_fenced(self):
        body = '{"entities":[{"span":"FAISS","type":"library"}]}'
        assert ta.parse_entities(body) == [{"span": "FAISS", "type": "library"}]
        assert ta.parse_entities(f"```json\n{body}\n```") == [{"span": "FAISS", "type": "library"}]

    def test_unknown_type_dropped(self):
        raw = '{"entities":[{"span":"x","type":"vibe"},{"span":"IVF-PQ","type":"method"}]}'
        assert ta.parse_entities(raw) == [{"span": "IVF-PQ", "type": "method"}]

    def test_malformed_returns_empty_rather_than_raising(self):
        assert ta.parse_entities("sorry, no entities here") == []
        assert ta.parse_entities('{"entities": [1, "two", null]}') == []


class TestReadReadme:
    def test_prefers_md_then_rst(self, tmp_path):
        assert ta.read_readme(_repo(tmp_path, "# md")) == "# md"
        (tmp_path / "README.md").unlink()
        (tmp_path / "README.rst").write_text("rst body", encoding="utf-8")
        assert ta.read_readme(tmp_path) == "rst body"

    def test_missing_readme_is_empty(self, tmp_path):
        assert ta.read_readme(tmp_path) == ""

    def test_search_order_matches_the_profiler(self):
        from reporadar import profiler

        src = Path(profiler.__file__).read_text(encoding="utf-8")
        for name in ta.README_NAMES:
            assert f'"{name}"' in src


class TestVerbatimDiscipline:
    def test_hallucinated_spans_are_dropped(self, tmp_path, monkeypatch):
        reply = _Reply(
            '{"entities":[{"span":"RocksDB","type":"library"},'
            '{"span":"TensorFlow","type":"library"}]}'
        )
        monkeypatch.setattr(ta, "complete", reply)
        got = ta.extract_typed_anchors(_repo(tmp_path, "We build on RocksDB."), llm_cfg=object())
        assert got == ["rocksdb"], "a span absent from the README is a hallucination"

    def test_duplicates_collapse_preserving_order(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            ta,
            "complete",
            _Reply(
                '{"entities":[{"span":"MVCC","type":"method"},'
                '{"span":"mvcc","type":"method"},{"span":"LSM","type":"method"}]}'
            ),
        )
        got = ta.extract_typed_anchors(_repo(tmp_path, "MVCC and LSM"), llm_cfg=object())
        assert got == ["mvcc", "lsm"]


class TestFailureIsLoud:
    def test_unparseable_reply_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ta, "complete", _Reply("I was unable to comply."))
        with pytest.raises(LLMError, match="no parseable entities"):
            ta.extract_typed_anchors(_repo(tmp_path, "uses RocksDB"), llm_cfg=object())

    def test_an_empty_but_wellformed_reply_is_a_real_answer(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ta, "complete", _Reply('{"entities": []}'))
        assert ta.extract_typed_anchors(_repo(tmp_path, "a prose readme"), llm_cfg=object()) == []

    def test_absent_readme_costs_nothing(self, tmp_path, monkeypatch):
        reply = _Reply('{"entities":[]}')
        monkeypatch.setattr(ta, "complete", reply)
        assert ta.extract_typed_anchors(tmp_path, llm_cfg=object()) == []
        assert reply.calls == 0, "no README is an absent document, not a failed extraction"


class TestCache:
    def test_second_call_does_not_re_extract(self, tmp_path, monkeypatch):
        reply = _Reply('{"entities":[{"span":"LSM","type":"method"}]}')
        monkeypatch.setattr(ta, "complete", reply)
        repo = _repo(tmp_path, "an LSM tree")
        assert ta.extract_typed_anchors(repo, llm_cfg=object()) == ["lsm"]
        assert ta.extract_typed_anchors(repo, llm_cfg=object()) == ["lsm"]
        assert reply.calls == 1

    def test_edited_readme_re_extracts(self, tmp_path, monkeypatch):
        reply = _Reply('{"entities":[{"span":"LSM","type":"method"}]}')
        monkeypatch.setattr(ta, "complete", reply)
        repo = _repo(tmp_path, "an LSM tree")
        ta.extract_typed_anchors(repo, llm_cfg=object())
        (repo / "README.md").write_text("an LSM tree, rewritten", encoding="utf-8")
        ta.extract_typed_anchors(repo, llm_cfg=object())
        assert reply.calls == 2, "the cache is keyed on the README, not on the path"

    def test_corrupt_cache_re_extracts_rather_than_failing(self, tmp_path, monkeypatch):
        reply = _Reply('{"entities":[{"span":"LSM","type":"method"}]}')
        monkeypatch.setattr(ta, "complete", reply)
        repo = _repo(tmp_path, "an LSM tree")
        cache = repo / ".reporadar" / "typed_anchors.json"
        cache.parent.mkdir(parents=True)
        cache.write_text("{not json", encoding="utf-8")
        assert ta.extract_typed_anchors(repo, llm_cfg=object()) == ["lsm"]
        assert json.loads(cache.read_text(encoding="utf-8"))["anchors"] == ["lsm"]


class TestProfilerIntegration:
    def test_off_by_default_costs_nothing(self, tmp_path, monkeypatch):
        from reporadar.config import ProfilerConfig
        from reporadar.profiler import profile_repo

        reply = _Reply('{"entities":[{"span":"LSM","type":"method"}]}')
        monkeypatch.setattr(ta, "complete", reply)
        _repo(tmp_path, "# proj\n\nan LSM tree storage engine\n")
        profile_repo(tmp_path, profiler_cfg=ProfilerConfig())
        assert reply.calls == 0, "typed_anchors defaults False; NR-36 left scan_source the same"

    def test_enabled_merges_spans_into_anchors(self, tmp_path, monkeypatch):
        from reporadar.config import ProfilerConfig
        from reporadar.profiler import profile_repo

        monkeypatch.setattr(ta, "complete", _Reply('{"entities":[{"span":"LSM","type":"method"}]}'))
        _repo(tmp_path, "# proj\n\nan LSM tree storage engine\n")
        base = profile_repo(tmp_path, profiler_cfg=ProfilerConfig())
        typed = profile_repo(tmp_path, profiler_cfg=ProfilerConfig(typed_anchors=True))
        assert "lsm" not in base.anchors
        assert "lsm" in typed.anchors
