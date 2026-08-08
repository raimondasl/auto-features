"""Tests for HyDE dense discovery.

The dangerous failures in this module are all silent. A mismatched encoder returns
confident nonsense that looks exactly like a working search; a half-written shard loads as
a truncated index; a channel that cannot run at all degrades to the keyword path measured
at 0/24 while the run reports success. Each gets a test, because none of them raises on
its own.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from reporadar import hyde


class _Profile:
    anchors = ["torch", "faiss"]
    domains = ["information retrieval"]
    keywords = [("vector search", 1.0), ("hnsw", 0.5)]
    prose = "A vector database."


def _write_shard(index_dir: Path, year: int, vectors: np.ndarray, ids: list[str]) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    np.save(index_dir / f"{year}.npy", vectors)
    (index_dir / f"{year}.ids").write_text("\n".join(ids), encoding="utf-8")


class TestTheIndexOnDisk:
    def test_a_shard_needs_both_files_to_count(self, tmp_path: Path) -> None:
        """Vectors are written before ids, so an interrupt between the two writes must
        re-fetch rather than half-load. A shard is present only when both exist."""
        np.save(tmp_path / "2020.npy", np.zeros((3, 128), dtype=np.uint8))
        assert hyde.index_shards(tmp_path) == []
        (tmp_path / "2020.ids").write_text("a\nb\nc", encoding="utf-8")
        assert [p.stem for p in hyde.index_shards(tmp_path)] == ["2020"]

    def test_shards_are_ordered_oldest_first(self, tmp_path: Path) -> None:
        for year in (2021, 1999, 2010):
            _write_shard(tmp_path, year, np.zeros((1, 128), dtype=np.uint8), ["x"])
        assert [p.stem for p in hyde.index_shards(tmp_path)] == ["1999", "2010", "2021"]

    def test_age_is_none_when_nothing_is_synced(self, tmp_path: Path) -> None:
        assert hyde.index_age_days(tmp_path) is None

    def test_age_is_measured_from_the_newest_shard(self, tmp_path: Path) -> None:
        _write_shard(tmp_path, 2020, np.zeros((1, 128), dtype=np.uint8), ["x"])
        age = hyde.index_age_days(tmp_path)
        assert age is not None and age < 1.0

    def test_shard_url_points_at_the_pinned_dataset(self) -> None:
        url = hyde.shard_url(2024)
        assert hyde.DATASET in url
        assert url.endswith("/data/2024.parquet")


class TestSearch:
    def _index(self, tmp_path: Path) -> Path:
        # Three papers; "a" is bit-identical to the query, "c" is its complement.
        query = np.zeros(128, dtype=np.uint8)
        vectors = np.stack([query, np.full(128, 0x0F, np.uint8), np.full(128, 0xFF, np.uint8)])
        _write_shard(tmp_path, 2020, vectors, ["2001.00001", "2001.00002", "2001.00003"])
        return tmp_path

    def test_nearest_first(self, tmp_path: Path) -> None:
        got = hyde.search_index(self._index(tmp_path), np.zeros((1, 128), dtype=np.uint8))
        assert got[0] == "2001.00001"
        assert got[-1] == "2001.00003"

    def test_a_missing_index_raises_rather_than_returning_nothing(self, tmp_path: Path) -> None:
        """Empty and never-looked must not be the same value: the caller would fall back to
        the keyword path, which is the one measured at 0 of 24."""
        with pytest.raises(hyde.HydeError, match="no HyDE index"):
            hyde.search_index(tmp_path, np.zeros((1, 128), dtype=np.uint8))

    def test_a_one_dimensional_query_is_accepted(self, tmp_path: Path) -> None:
        got = hyde.search_index(self._index(tmp_path), np.zeros(128, dtype=np.uint8))
        assert got[0] == "2001.00001"

    def test_top_k_bounds_each_query_list(self, tmp_path: Path) -> None:
        got = hyde.search_index(self._index(tmp_path), np.zeros((1, 128), dtype=np.uint8), top_k=2)
        assert len(got) == 2

    def test_queries_are_fused_by_best_rank_across_hypotheses(self, tmp_path: Path) -> None:
        """The measured arm is the union of the per-hypothesis lists — 27/48 against 23/48
        for a single guess at an equal budget. A paper any hypothesis ranks first leads."""
        index = self._index(tmp_path)
        far = np.full((1, 128), 0xFF, np.uint8)  # nearest to "...0003"
        near = np.zeros((1, 128), dtype=np.uint8)  # nearest to "...0001"
        fused = hyde.search_index(index, np.concatenate([far, near]), top_k=1)
        assert set(fused) == {"2001.00001", "2001.00003"}

    def test_a_corrupt_shard_is_skipped_not_misaligned(self, tmp_path: Path) -> None:
        """An id/vector length mismatch would silently attach the wrong id to every paper
        after the gap — worse than dropping the shard."""
        _write_shard(tmp_path, 2020, np.zeros((3, 128), dtype=np.uint8), ["a", "b", "c"])
        _write_shard(tmp_path, 2021, np.zeros((3, 128), dtype=np.uint8), ["x", "y"])
        got = hyde.search_index(tmp_path, np.zeros((1, 128), dtype=np.uint8))
        assert set(got) == {"a", "b", "c"}

    def test_it_searches_every_shard(self, tmp_path: Path) -> None:
        _write_shard(tmp_path, 2019, np.full((1, 128), 0x0F, np.uint8), ["old"])
        _write_shard(tmp_path, 2024, np.zeros((1, 128), dtype=np.uint8), ["new"])
        assert set(hyde.search_index(tmp_path, np.zeros((1, 128), dtype=np.uint8))) == {
            "old",
            "new",
        }


class TestTheEncoderGuard:
    """The fifth dependency: our vectors must be comparable to the index's. Nothing
    downstream can detect a mismatch, so this check is the only thing standing between a
    wrong model and a page of confident nonsense."""

    def _rows(self, stored: bytes) -> MagicMock:
        table = MagicMock()
        table.to_pylist.return_value = [{"vector": stored, "abstract": "an abstract"}]
        pf = MagicMock()
        pf.read_row_group.return_value = table
        return pf

    def _model(self, vec: np.ndarray) -> MagicMock:
        model = MagicMock()
        model.encode.return_value = vec
        return model

    def test_identical_vectors_verify(self) -> None:
        raw = np.zeros((1, 1024), dtype=np.float32) - 1.0  # every bit 0 after >0
        stored = np.packbits(raw[0] > 0).tobytes()
        with (
            patch.object(hyde, "RangeFile"),
            patch("pyarrow.parquet.ParquetFile", return_value=self._rows(stored)),
        ):
            ok, dists = hyde.verify_encoder(self._model(raw))
        assert ok and dists == [0]

    def test_a_different_space_is_caught(self) -> None:
        raw = np.ones((1, 1024), dtype=np.float32)  # every bit 1
        stored = bytes(128)  # every bit 0
        with (
            patch.object(hyde, "RangeFile"),
            patch("pyarrow.parquet.ParquetFile", return_value=self._rows(stored)),
        ):
            ok, dists = hyde.verify_encoder(self._model(raw))
        assert not ok
        assert dists == [1024]

    def test_discover_refuses_to_search_on_mismatch(self, tmp_path: Path) -> None:
        _write_shard(tmp_path, 2020, np.zeros((1, 128), dtype=np.uint8), ["2001.00001"])
        with (
            patch.object(hyde, "generate_hypotheses", return_value=["an abstract"]),
            patch.object(hyde, "load_encoder", return_value=MagicMock()),
            patch.object(hyde, "verify_encoder", return_value=(False, [512])),
            pytest.raises(hyde.HydeError, match="does not reproduce the index"),
        ):
            hyde.discover(_Profile(), SimpleNamespace(), tmp_path)

    def test_verification_can_be_skipped_but_is_on_by_default(self, tmp_path: Path) -> None:
        from reporadar.config import HydeConfig

        assert HydeConfig().verify_encoder is True
        _write_shard(tmp_path, 2020, np.zeros((1, 128), dtype=np.uint8), ["2001.00001"])
        with (
            patch.object(hyde, "generate_hypotheses", return_value=["an abstract"]),
            patch.object(hyde, "load_encoder", return_value=MagicMock()),
            patch.object(hyde, "encode_binary", return_value=np.zeros((1, 128), np.uint8)),
            patch.object(hyde, "verify_encoder") as guard,
        ):
            hyde.discover(_Profile(), SimpleNamespace(), tmp_path, verify=False)
        guard.assert_not_called()


class TestHypotheses:
    def test_it_parses_a_json_array(self) -> None:
        raw = '["first abstract", "second abstract"]'
        with patch.object(hyde, "complete", return_value=raw):
            got = hyde.generate_hypotheses(_Profile(), SimpleNamespace(), n=2)
        assert got == ["first abstract", "second abstract"]

    def test_prose_around_the_array_is_tolerated(self) -> None:
        with patch.object(hyde, "complete", return_value='Sure!\n["one"]\nHope that helps'):
            assert hyde.generate_hypotheses(_Profile(), SimpleNamespace(), n=1) == ["one"]

    def test_it_returns_at_most_n(self) -> None:
        with patch.object(hyde, "complete", return_value=json.dumps(["a", "b", "c", "d", "e"])):
            assert len(hyde.generate_hypotheses(_Profile(), SimpleNamespace(), n=3)) == 3

    def test_no_array_raises(self) -> None:
        with (
            patch.object(hyde, "complete", return_value="I cannot do that"),
            pytest.raises(hyde.HydeError, match="no JSON array"),
        ):
            hyde.generate_hypotheses(_Profile(), SimpleNamespace())

    def test_an_empty_array_raises_rather_than_searching_on_nothing(self) -> None:
        with (
            patch.object(hyde, "complete", return_value="[]"),
            pytest.raises(hyde.HydeError, match="no abstracts"),
        ):
            hyde.generate_hypotheses(_Profile(), SimpleNamespace())

    def test_the_prompt_carries_the_repo_and_forbids_naming_it(self) -> None:
        seen: list[str] = []

        def capture(prompt: str, *_a: object, **_k: object) -> str:
            seen.append(prompt)
            return '["x"]'

        with patch.object(hyde, "complete", side_effect=capture):
            hyde.generate_hypotheses(_Profile(), SimpleNamespace(), n=1)
        assert "vector search" in seen[0]  # the repo's own profile reached the prompt
        assert "Do not mention the repository" in seen[0]
        assert "arXiv abstract" in seen[0]


class TestDiscover:
    def test_non_arxiv_ids_are_dropped(self, tmp_path: Path) -> None:
        """The index is arXiv-wide but its id column has held odd rows; anything the arXiv
        metadata fetch cannot resolve is noise in the pool."""
        _write_shard(
            tmp_path,
            2020,
            np.zeros((3, 128), dtype=np.uint8),
            ["2001.00001", "not-an-id", "cs/0112017"],
        )
        with (
            patch.object(hyde, "generate_hypotheses", return_value=["a"]),
            patch.object(hyde, "load_encoder", return_value=MagicMock()),
            patch.object(hyde, "encode_binary", return_value=np.zeros((1, 128), np.uint8)),
        ):
            got = hyde.discover(_Profile(), SimpleNamespace(), tmp_path, verify=False)
        assert set(got) == {"2001.00001", "cs/0112017"}

    def test_a_missing_index_raises_before_spending_an_llm_call(self, tmp_path: Path) -> None:
        with (
            patch.object(hyde, "generate_hypotheses") as gen,
            pytest.raises(hyde.HydeError, match="rr sync-index"),
        ):
            hyde.discover(_Profile(), SimpleNamespace(), tmp_path)
        gen.assert_not_called()


class TestCollectByIds:
    def test_no_ids_makes_no_request(self) -> None:
        from reporadar import collector

        with patch.object(collector, "_shared_client") as client:
            assert collector.collect_by_ids([]) == []
        client.assert_not_called()

    def test_it_batches_and_dedups(self) -> None:
        from reporadar import collector

        def result(pid: str) -> MagicMock:
            r = MagicMock()
            r.get_short_id.return_value = pid
            r.title = "t"
            r.authors = []
            r.summary = "a"
            r.categories = ["cs.LG"]
            r.published.isoformat.return_value = "2020-01-01"
            r.updated = None
            r.entry_id = f"http://arxiv.org/abs/{pid}"
            r.pdf_url = ""
            return r

        with (
            patch.object(collector, "_shared_client", return_value=MagicMock()),
            patch.object(
                collector,
                "_query_with_retry",
                side_effect=[[result("1"), result("2")], [result("2"), result("3")]],
            ),
        ):
            got = collector.collect_by_ids(["1", "2", "3"], batch_size=2)
        assert [p["arxiv_id"] for p in got] == ["1", "2", "3"]

    def test_a_decades_old_paper_survives(self) -> None:
        """HyDE and the citation hop exist to reach seminal older work — every one of the
        48 benchmark targets is >= 11 months old, and six are pre-2015. Re-applying the
        recency window to their output would undo the only thing they are for, so this
        asserts the behaviour rather than the absence of a keyword."""
        from reporadar import collector

        ancient = MagicMock()
        ancient.get_short_id.return_value = "1409.0473"  # Bahdanau et al., 2014
        ancient.title = "Neural Machine Translation by Jointly Learning to Align and Translate"
        ancient.authors = []
        ancient.summary = "a"
        ancient.categories = ["cs.CL"]
        ancient.published.isoformat.return_value = "2014-09-01T00:00:00+00:00"
        ancient.updated = None
        ancient.entry_id = "http://arxiv.org/abs/1409.0473"
        ancient.pdf_url = ""

        with (
            patch.object(collector, "_shared_client", return_value=MagicMock()),
            patch.object(collector, "_query_with_retry", return_value=[ancient]),
        ):
            got = collector.collect_by_ids(["1409.0473"])
        assert [p["arxiv_id"] for p in got] == ["1409.0473"]
