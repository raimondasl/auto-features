"""The technical report is frozen, and the freeze is checked rather than announced.

`paper/DRAFT.md` carries a freeze notice: content as of 2026-08-28, no changes from
2026-09-19 onward. Findings recorded after that date live in `evals/RESULTS.md` and are
deliberately not folded back into the report.

A notice is a claim about our own artifact, and this repository keeps a catalogue of those
decaying quietly while nothing checks them -- the reproducibility promise that pointed at an
index which did not exist is the nearest relative (see `test_eval_script_index.py`). So the
freeze is pinned to a content hash instead of being asserted in prose: any edit to the report
fails here, which is what "no changes from this date onward" has to mean if it means anything.

To unfreeze deliberately -- a camera-ready de-anonymisation is the expected reason -- change
the notice and update `FROZEN_SHA256` in the same commit. The point is not that the report can
never change; it is that it cannot change by drift.
"""

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRAFT = ROOT / "paper" / "DRAFT.md"

FROZEN_SHA256 = "36aab141351f899edaff6071ebe540a57ddbe589c81dbc835e8a1d6aefeb1bd7"
FREEZE_NOTICE = "**Frozen 2026-09-19**"


def _normalised_bytes() -> bytes:
    """Hash line-ending-normalised text: the checkout may be CRLF on Windows and LF on CI,
    and a freeze that depends on which platform cloned the repository checks nothing."""
    return DRAFT.read_text(encoding="utf-8").replace("\r\n", "\n").encode("utf-8")


class TestTheReportIsFrozen:
    def test_the_notice_is_present(self) -> None:
        assert FREEZE_NOTICE in DRAFT.read_text(encoding="utf-8"), (
            "paper/DRAFT.md no longer states its freeze date"
        )

    def test_the_content_has_not_changed_since_the_freeze(self) -> None:
        actual = hashlib.sha256(_normalised_bytes()).hexdigest()
        assert actual == FROZEN_SHA256, (
            "paper/DRAFT.md changed after its declared freeze of 2026-09-19.\n"
            "Post-freeze findings belong in evals/RESULTS.md, not in the report.\n"
            "If this edit is deliberate, update FROZEN_SHA256 to:\n"
            f"    {actual}\n"
            "in the same commit that changes the freeze notice."
        )


class TestTheReportCarriesNoByline:
    def test_no_author_attribution(self) -> None:
        """Removed while a double-blind submission derived from this report is under review.

        The names are assembled rather than written, because a guard that spells out the
        string it forbids puts that string in a public repository, which is the thing the
        de-bylining was for. FROZEN_SHA256 is the stronger check anyway: it catches any
        edit at all. This one exists for the clearer failure message.
        """
        text = DRAFT.read_text(encoding="utf-8")
        for parts in (("Raim", "ondas"), ("raimo", "ndasl")):
            identifying = "".join(parts)
            assert identifying not in text, (
                "paper/DRAFT.md reintroduces an author-identifying string; it is de-bylined"
            )
