"""Attention and integrity signals about papers (Feature 9).

Where ``sources/`` adapters *find* papers, ``signals/`` adapters say something
*about* a paper RepoRadar already has: is it withdrawn, was it discussed, was it
published somewhere real.

Every module here follows the same contract as ``sources/``:

* stdlib-only HTTP with retry/backoff and a **module-level** request throttle
  (per-instance throttles don't compose — see ``sources/dblp.py``);
* graceful, *visible* degradation: a failed lookup yields no signal and logs a
  warning, and callers must treat "no signal" as **absent**, never as a zero —
  scoring a missing signal as 0.0 silently handicaps the paper;
* pure detection logic separated from I/O, so it is testable offline.

Signals deliberately *not* shipped, each because probing disproved the premise
rather than because they were hard — see ROADMAP.md Feature 9 for the measurements.
"""
