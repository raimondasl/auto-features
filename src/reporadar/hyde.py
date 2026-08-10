"""HyDE discovery — search arXiv by the abstract of the paper this repo *wishes existed*.

Keyword retrieval from a repository reaches almost none of the papers that would improve
it. Measured: across nine benchmark repositories RepoRadar's own queries fetched 2,030
papers and reached **0** of the 24 an independent baseline recommended and a judge
confirmed actionable — not a ranking miss, a disjoint set. The cause is a **register
mismatch**: a codebase's vocabulary (and any text derived from reading it) describes what
the project *has*, while the useful paper describes what it should *adopt*. Those are
different registers, and every channel built on repository text inherits the gap. Asking an
LLM to name what the repo lacks does not fix it either — it aims at a plausible *different*
research agenda and, measured, ranks the true targets worse than random.

This module routes around the mismatch instead of trying to close it. The LLM writes the
**abstract of a paper that does not exist** but which, if it did, would most improve this
codebase. That text is in the literature's register by construction — because that is what
the model was asked to produce — so it can be embedded and matched against real abstracts
in a dense space, where "experience replay prioritization methods" and "prioritized
experience replay" are near neighbours rather than disjoint strings.

Measured on the 22-repository benchmark (`evals/hyde_replication.py`, blind protocol — the
generator never saw the targets): four hypotheses per repo reach **27 of 48** known-good
papers in the top 1,000, median rank 837, against 10/48 for embedding the README and 3/48
for the shipped keyword queries **in the same index**. Its value is not only the count:
**15 of those targets are unreachable by the citation hop**, including every repository
with no arXiv-indexed bibliography, so the two channels are additive (union 36/48 = 75%).

**The check that makes the rest trustworthy.** Four dependencies were verified before this
was built (the index exists under a permissive licence, columnar range-fetch is real, query
latency is ~1.2 s, the targets are present). A *fifth* was not on that list and matters
more than any of them: **a vector we compute must be comparable to the vectors in the
index.** If the publisher had embedded title+abstract, or normalised differently, every
query would have measured nothing while looking perfectly healthy — a failure this project
has shipped twice in other forms. So :func:`verify_encoder` reproduces stored vectors
bit-for-bit and the search path *refuses to run* if it cannot. Do not make that check
advisory.

Two properties worth keeping in mind:

* **Discovery becomes offline.** Once the index is synced, matching sends nothing — a net
  privacy improvement over transmitting repo-derived keywords to arXiv on every run. Only
  the hypothesis generation (an LLM call) and the metadata fetch for the winners leave the
  machine.
* **The bottleneck moves to the hypothesis.** Recall is capped by whether the guesses
  happen to include the gap that matters. Four diverse guesses measured far better than
  one (42/48 vs 23/48 reachable within an equal candidate budget), which is why the default
  is 4 and why the prompt insists they differ.
"""

from __future__ import annotations

import io
import json
import logging
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from reporadar import __version__
from reporadar.llm_client import complete
from reporadar.profiler import RepoProfile
from reporadar.triage import repo_context_block

logger = logging.getLogger(__name__)

# PaperMatch's published index: whole-arXiv abstract embeddings, binarised, apache-2.0.
# Verified 2026-08-06 at 3,106,925 rows. It is a third-party mirror republished
# periodically — see `stale_after_days` in the config for the freshness posture.
DATASET = "bluuebunny/arxiv_abstract_embedding_mxbai_large_v1_milvus_binary"
HUB = "https://huggingface.co"
MODEL_NAME = "mixedbread-ai/mxbai-embed-large-v1"

# The index covers arXiv from its first year to the snapshot. Shards are per year.
FIRST_YEAR = 1991
LAST_YEAR = 2027

VECTOR_BYTES = 128  # 1024 bits, packed — asserted on every shard we write
USER_AGENT = f"RepoRadar/{__version__} (+https://github.com/raimondasl/auto-features)"

HYPOTHESIS_PROMPT = """\
You are shown a software repository. Write the ABSTRACTS of {n} different research papers
that do not necessarily exist, but which — if they did — would most improve this specific
codebase.

Each abstract must read exactly like an arXiv abstract in its field: the literature's terms
of art, the usual "we propose / we show / on benchmark X we obtain" register, 120-200 words.
Do not mention the repository, its name, or its files. Do not write a title. Do not hedge or
describe what you are doing — write the abstract itself.

Make the {n} genuinely different from each other: different subproblems of this repository,
not four phrasings of one idea.

Repository:
{context}

Respond with ONLY a JSON array of {n} strings.
"""

# Appended AFTER the shared repository block, never inside it. See `generate_hypotheses`.
GOAL_BLOCK = """\

What the maintainers say they want to improve, in their own words:
{goal}

Weight the abstracts toward that stated goal. It describes what the project NEEDS, which
the repository's own documentation cannot: documentation describes what a project IS.
"""


class HydeError(Exception):
    """Raised when HyDE cannot run — a missing index, a missing dependency, a bad encoder."""


# ── The remote index ───────────────────────────────────────────────────────────────────


def _get(url: str, *, headers: dict[str, str] | None = None, retries: int = 4) -> bytes:
    head = {"User-Agent": USER_AGENT, **(headers or {})}
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(
                urllib.request.Request(url, headers=head), timeout=120
            ) as resp:
                data: bytes = resp.read()
                return data
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = exc
            time.sleep(2 * (attempt + 1))
    raise HydeError(f"fetch failed after {retries} attempts: {url} ({last})")


class RangeFile(io.RawIOBase):
    """A seekable read-only file over HTTP Range requests, counting what it transfers.

    This is what makes the index sync affordable. Parquet keeps its schema and per-column
    byte offsets in a footer, so a reader that can seek never has to touch columns it was
    not asked for. We want ``id`` and ``vector`` and none of the other seven, which turns a
    2,542 MB dataset into a **432 MB** fetch — measured, and the difference between a
    feasible one-time sync and a 6× one.
    """

    def __init__(self, url: str) -> None:
        super().__init__()
        self.url = url
        self._pos = 0
        self.bytes_fetched = 0
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT}, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as resp:
            # LFS objects report the real object size in x-linked-size; Content-Length on
            # the pointer redirect is not it.
            linked = resp.headers.get("x-linked-size")
            self.size = int(linked or resp.headers["Content-Length"])

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self._pos, io.SEEK_END: self.size}[whence]
        self._pos = max(0, min(self.size, base + offset))
        return self._pos

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = self.size - self._pos
        size = min(size, self.size - self._pos)
        if size <= 0:
            return b""
        data = _get(self.url, headers={"Range": f"bytes={self._pos}-{self._pos + size - 1}"})
        self.bytes_fetched += len(data)
        self._pos += len(data)
        return data

    def readinto(self, b: Any) -> int:
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def shard_url(year: int) -> str:
    return f"{HUB}/datasets/{DATASET}/resolve/main/data/{year}.parquet"


def sync_index(index_dir: Path, *, refresh: bool = False, on_shard: Any = None) -> dict[str, int]:
    """Fetch the ``id`` + ``vector`` columns of every year shard into *index_dir*.

    Resumable by construction: a shard already on disk is skipped unless *refresh*, so an
    interrupted sync costs only what it had not yet fetched. Returns
    ``{"shards": n, "rows": n, "bytes": n}``.
    """
    try:
        import numpy as np
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise HydeError(
            'HyDE needs pyarrow and numpy. Install with: uv pip install -e ".[hyde]"'
        ) from exc

    index_dir.mkdir(parents=True, exist_ok=True)
    stats = {"shards": 0, "rows": 0, "bytes": 0}
    for year in range(FIRST_YEAR, LAST_YEAR):
        vec_path = index_dir / f"{year}.npy"
        id_path = index_dir / f"{year}.ids"
        if vec_path.is_file() and id_path.is_file() and not refresh:
            continue
        try:
            fh = RangeFile(shard_url(year))
            table = pq.ParquetFile(fh).read(columns=["id", "vector"])
        except Exception as exc:  # noqa: BLE001 — a missing year is not fatal
            logger.info("HyDE index: no shard for %s (%s)", year, exc)
            continue
        ids = [str(x) for x in table.column("id").to_pylist()]
        vecs = table.column("vector").to_pylist()
        arr = np.frombuffer(b"".join(vecs), dtype=np.uint8).reshape(len(vecs), -1)
        if arr.shape[1] != VECTOR_BYTES:
            raise HydeError(f"{year}: vector width {arr.shape[1]}, expected {VECTOR_BYTES}")
        if len(ids) != arr.shape[0]:
            raise HydeError(f"{year}: {len(ids)} ids for {arr.shape[0]} vectors")
        # Write the vectors first: a shard counts as present only when BOTH files exist,
        # so an interrupt between the two writes re-fetches rather than half-loading.
        np.save(vec_path, arr)
        id_path.write_text("\n".join(ids), encoding="utf-8")
        stats["shards"] += 1
        stats["rows"] += len(ids)
        stats["bytes"] += fh.bytes_fetched
        if on_shard is not None:
            on_shard(year, len(ids), fh.bytes_fetched)
    return stats


def index_shards(index_dir: Path) -> list[Path]:
    """Year shards present on disk, oldest first. Empty when the index is not synced."""
    return sorted(p for p in index_dir.glob("*.npy") if (index_dir / f"{p.stem}.ids").is_file())


def index_age_days(index_dir: Path) -> float | None:
    """Days since the newest shard was written, or None when nothing is synced."""
    shards = index_shards(index_dir)
    if not shards:
        return None
    newest = max(p.stat().st_mtime for p in shards)
    return (time.time() - newest) / 86400.0


# ── Encoding, and the check that guards it ─────────────────────────────────────────────


def load_encoder(model_name: str = MODEL_NAME) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise HydeError(
            'HyDE needs sentence-transformers. Install with: uv pip install -e ".[hyde]"'
        ) from exc
    return SentenceTransformer(model_name)


def encode_binary(model: Any, texts: list[str]) -> Any:
    """Embed *texts* and binarise exactly the way the index was built: L2-normalise, >0,
    packbits. Any deviation here silently moves our queries into a different space."""
    import numpy as np

    vecs = model.encode(texts, normalize_embeddings=True)
    return np.packbits(vecs > 0, axis=-1).astype(np.uint8)


def verify_encoder(model: Any, *, samples: int = 5) -> tuple[bool, list[int]]:
    """Reproduce stored vectors bit-for-bit, or report that we cannot.

    The four pre-build checks established that the index exists, is fetchable, is fast and
    holds the targets. **None of them established that our vectors are comparable to its
    vectors** — and if they are not, every query returns confident nonsense. This reads a
    few abstracts out of the smallest shard, embeds them here, and compares packed bytes.
    Anything but Hamming 0 means the search path must not run.
    """
    import numpy as np
    import pyarrow.parquet as pq

    fh = RangeFile(shard_url(1992))
    table = pq.ParquetFile(fh).read_row_group(0, columns=["vector", "abstract"])
    rows = table.to_pylist()[:samples]
    if not rows:
        return False, []
    mine = encode_binary(model, [r["abstract"] for r in rows])
    dists = [
        int(np.bitwise_count(m ^ np.frombuffer(r["vector"], dtype=np.uint8)).sum())
        for r, m in zip(rows, mine, strict=True)
    ]
    return all(d == 0 for d in dists), dists


# ── Hypotheses ─────────────────────────────────────────────────────────────────────────


def generate_hypotheses(
    profile: RepoProfile, llm_cfg: Any, *, n: int = 4, goal: str | None = None
) -> list[str]:
    """Ask the LLM for *n* abstracts of papers that would most improve this repository.

    The repository is described with the same block the gate uses, so one description
    serves both stages and cannot drift between them.

    *goal* is a maintainer's own statement of what they want to improve. It is appended
    **after** :func:`~reporadar.triage.repo_context_block`, never merged into it, and two
    separate results make that placement non-negotiable:

    * a repo block that differed between stages would silently invalidate the fine-scale
      stage's frozen probability map, which is fitted to those exact bytes; and
    * feeding stated wants to the **gate** was measured at net@2 +57 against +95 for the
      shipped prompt — the worst arm in the campaign — because 15 named wants replace the
      gate's question ("would this improve the project?") with a checklist ("is this on
      the list?"). That experiment's own conclusion was that wants belong in the *query*,
      which is here, and nowhere else.

    So a goal changes what RepoRadar goes looking for and never what it accepts.
    """
    context = repo_context_block(profile)[:6000]
    if goal and goal.strip():
        context += GOAL_BLOCK.format(goal=goal.strip()[:600])
    prompt = HYPOTHESIS_PROMPT.format(n=n, context=context)
    raw = complete(prompt, llm_cfg, max_tokens=2500)
    start, end = raw.find("["), raw.rfind("]")
    if start < 0 or end < 0:
        raise HydeError(f"no JSON array in hypothesis response: {raw[:160]}")
    items = [str(x).strip() for x in json.loads(raw[start : end + 1]) if str(x).strip()]
    if not items:
        raise HydeError("hypothesis response contained no abstracts")
    return items[:n]


# ── Search ─────────────────────────────────────────────────────────────────────────────


def search_index(index_dir: Path, query_bits: Any, *, top_k: int = 100) -> list[str]:
    """Union of the nearest *top_k* arXiv ids across all query rows, best rank first.

    Shards are memory-mapped and scanned one at a time rather than concatenated: the full
    index is ~400 MB of uint8, and a recommender that needs 400 MB resident to produce ten
    papers is not one people leave enabled. Each query row keeps its own top-*k*; the union
    is what the replication measured (four diverse hypotheses beat one by a wide margin).
    """
    import numpy as np

    shards = index_shards(index_dir)
    if not shards:
        raise HydeError(f"no HyDE index in {index_dir} — run `rr sync-index` first")
    if query_bits.ndim == 1:
        query_bits = query_bits.reshape(1, -1)

    # Per query row: the running best (distance, id) heap as parallel arrays.
    best: list[list[tuple[int, str]]] = [[] for _ in range(query_bits.shape[0])]
    for shard in shards:
        vectors = np.load(shard, mmap_mode="r")
        ids = (index_dir / f"{shard.stem}.ids").read_text(encoding="utf-8").split("\n")
        if len(ids) != vectors.shape[0]:
            logger.warning(
                "HyDE index: %s has %d ids for %d vectors — skipping shard",
                shard.stem,
                len(ids),
                vectors.shape[0],
            )
            continue
        for row in range(query_bits.shape[0]):
            dists = _hamming(vectors, query_bits[row])
            take = min(top_k, dists.shape[0])
            local = np.argpartition(dists, take - 1)[:take]
            best[row].extend((int(dists[i]), ids[i]) for i in local)
            best[row].sort(key=lambda pair: pair[0])
            del best[row][top_k:]

    # Fuse the per-query lists by best rank achieved in any of them, which is the
    # "hyde4-union" arm — the one that measured 27/48.
    ranked: dict[str, int] = {}
    for per_query in best:
        for rank, (_dist, pid) in enumerate(per_query):
            if pid not in ranked or rank < ranked[pid]:
                ranked[pid] = rank
    return [pid for pid, _ in sorted(ranked.items(), key=lambda kv: kv[1])]


def _hamming(index: Any, query: Any, chunk: int = 250_000) -> Any:
    """Packed-bit Hamming distance from *query* to every row, in memory-bounded chunks."""
    import numpy as np

    out = np.empty(index.shape[0], dtype=np.uint16)
    for start in range(0, index.shape[0], chunk):
        block = np.asarray(index[start : start + chunk])
        out[start : start + len(block)] = np.bitwise_count(block ^ query).sum(
            axis=1, dtype=np.uint16
        )
    return out


# ── The whole channel ──────────────────────────────────────────────────────────────────

_ARXIV_ID = re.compile(r"^\d{4}\.\d{4,5}$|^[a-z-]+(\.[A-Z]{2})?/\d{7}$")


def discover(
    profile: RepoProfile,
    llm_cfg: Any,
    index_dir: Path,
    *,
    n_hypotheses: int = 4,
    top_k: int = 100,
    model_name: str = MODEL_NAME,
    verify: bool = True,
    on_progress: Any = None,
    goal: str | None = None,
) -> list[str]:
    """arXiv ids for this repository, discovered offline against the dense index.

    Raises :class:`HydeError` for every failure mode — no index, no dependency, an encoder
    that does not reproduce the index — rather than returning an empty list, because a
    caller cannot tell "found nothing" from "was never able to look" and would quietly
    degrade to the keyword-only path that reaches 0/24.
    """
    shards = index_shards(index_dir)
    if not shards:
        raise HydeError(f"no HyDE index in {index_dir} — run `rr sync-index` first")

    if on_progress:
        on_progress(f"writing {n_hypotheses} hypothesis abstracts...")
    hypotheses = generate_hypotheses(profile, llm_cfg, n=n_hypotheses, goal=goal)

    if on_progress:
        on_progress(f"loading {model_name}...")
    model = load_encoder(model_name)
    if verify:
        ok, dists = verify_encoder(model)
        if not ok:
            raise HydeError(
                f"encoder does not reproduce the index (Hamming {dists}, expected all 0) — "
                "refusing to search, because every result would be noise. The index model "
                f"is {MODEL_NAME}; check `hyde.model`."
            )

    if on_progress:
        on_progress(f"searching {len(shards)} shards...")
    ids = search_index(index_dir, encode_binary(model, hypotheses), top_k=top_k)
    return [pid for pid in ids if _ARXIV_ID.match(pid)]
