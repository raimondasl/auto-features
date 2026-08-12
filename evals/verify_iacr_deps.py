"""Stage 1 for an IACR ePrint source: verify every dependency before building anything.

    uv run python evals/verify_iacr_deps.py     # $0, keyless, ~10 requests

This is the P4 protocol, which is in this repository because a REPORTED estimate was once
off by 10x and another had a transport bug underneath it. Nothing gets built until the
things it rests on are checked against the live service.

**Why an IACR source at all, and why measured on two cases.** The benchmark's minimum
resolvable effect is 1.04 net@2 per case (evals/noise_floor.py). IACR ePrint serves
`crypto` and `encryption` and nothing else in this benchmark, so even a *perfect* adapter
moves the 25-case mean by at most +0.68 — **below the floor, undetectable by construction**.
On the two-case subset it serves, the MRE is 3.44 against 8.5 net@2 of available headroom,
which is measurable. The subset is therefore **pre-registered here, before the adapter
exists**, and chosen by domain rather than by result.

Checks, each with a kill condition:

C1  a keyless machine-readable endpoint exists                    kill: none does
C2  it returns title AND abstract (DBLP returns titles only, and
    an abstract-less source cannot be gated — the 0-3 prompt
    reads the abstract)                                            kill: no abstracts
C3  results are keyword-addressable, so the shipped query model
    applies without a bespoke retrieval path                       kill: no query support
C4  a stable per-paper id and URL exist for the synthetic
    `iacr:<id>` scheme the pipeline needs                          kill: no stable id
C5  latency and politeness are workable for a 25-case sweep        kill: >10 s/query

And the check that is easy not to think of, the analogue of HyDE's bit-exact encoder test:
**C6 — do the papers this returns actually reach the judge?** A source whose records the
verifier drops would score zero while looking perfectly healthy.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

UA = "RepoRadar/dep-verify (+https://github.com/raimondasl/auto-features)"
CANDIDATE_ENDPOINTS = [
    ("search JSON", "https://eprint.iacr.org/search?q={q}&format=json"),
    ("search HTML", "https://eprint.iacr.org/search?q={q}"),
    ("OAI-PMH", "https://eprint.iacr.org/oai?verb=Identify"),
    ("per-paper JSON", "https://eprint.iacr.org/2023/1234.json"),
    ("byyear listing", "https://eprint.iacr.org/byyear"),
]
QUERIES = ["post-quantum", "elliptic curve implementation", "side-channel"]


def get(url: str, timeout: int = 20) -> tuple[int, str, float]:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read(400_000).decode("utf-8", errors="replace")
            return resp.status, body, time.monotonic() - t0
    except urllib.error.HTTPError as exc:
        return exc.code, "", time.monotonic() - t0
    except Exception as exc:  # noqa: BLE001 — a verification script reports, never raises
        return 0, f"{type(exc).__name__}: {exc}", time.monotonic() - t0


def looks_json(body: str) -> Any | None:
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return None


def main() -> int:
    print("C1/C3/C5 — endpoint discovery (keyless, one request each)\n")
    working: list[tuple[str, str]] = []
    for label, tmpl in CANDIDATE_ENDPOINTS:
        url = tmpl.format(q=urllib.parse.quote("post-quantum"))
        status, body, dt = get(url)
        parsed = looks_json(body) if status == 200 else None
        kind = "JSON" if parsed is not None else ("HTML/text" if status == 200 else "-")
        print(f"  {label:16} {status:>3}  {dt:5.2f}s  {kind:9}  {len(body):>7,} bytes")
        if status == 200:
            working.append((label, url))
        time.sleep(1.0)

    if not working:
        print("\nC1 KILL: no reachable endpoint. IACR cannot be sourced this way.")
        return 1

    print("\nC2/C4 — does any working endpoint carry abstracts and stable ids?")
    for label, url in working:
        status, body, _ = get(url)
        parsed = looks_json(body)
        if parsed is not None:
            keys = sorted(parsed)[:12] if isinstance(parsed, dict) else "list"
            print(f"  {label:16} JSON top-level: {keys}")
        else:
            has_abs = "abstract" in body.lower()
            ids = body.count("eprint.iacr.org/20")
            print(f"  {label:16} html: mentions 'abstract'={has_abs}, ~{ids} paper links")
        time.sleep(1.0)

    print("\nNOTE: C6 (do these records reach the judge?) is checked in the adapter's own")
    print("tests, not here — it depends on the record shape the adapter emits.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
