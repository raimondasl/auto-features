"""P22: what Europe PMC returns when a compiler asks it a question.

Europe PMC gave **+4.00 net@2** on the six bio repositories (P21), by coverage rather than
ranking — 54% of the shown digest came from it, at 0.97 precision. The obvious next move was a
multi-source default. This probe is why that would have been wrong.

Sending each core-25 repository's **own** queries to Europe PMC returns hits for **all 25,
never zero**, and **68% of them are indexed into MeSH** — catalogued as biomedical by NLM
cataloguers, not guessed at by us. The query `lint code` returns *"Occurrence of postoperative
pneumoencephalus in posterior fossa surgery"*; `arrow file` returns zebrafish telomerase.

That is the expensive outcome rather than the harmless one. A source that stays quiet outside
its domain costs nothing to enable everywhere; one that answers **confidently and off-domain**
feeds the candidate pool of every repository, where net@2 charges **2 per false positive** and
non-arXiv papers additionally escape the ranker's category component (a bias already measured
moving 18 of 32 such papers into or out of the top-10).

**A correction is pinned here alongside the result.** The first run of this probe reported
**21%** biomedical. `_looks_biomedical` was reading `journalTitle` and `pubType`, which are
`None` on essentially every Europe PMC record, so it substring-matched empty strings and the
number measured nothing. It was caught by reading the sampled titles — transparently
biological while the counter said 79% were not. `test_the_flag_reads_fields_that_exist` is the
regression test for that specific mistake.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "evals" / "europepmc_collision.json"
OA = ROOT / "evals" / "openalex_collision.json"


@pytest.fixture(scope="module")
def artifact() -> dict:
    return json.loads(FROZEN.read_text(encoding="utf-8"))


class TestTheProbeCoveredTheCohort:
    def test_all_twenty_five_core_repositories(self, artifact):
        cases = artifact["cases"]
        assert len(cases) == 25
        assert not [c for c in cases if c.startswith(("bio-", "mat-"))], "core only"

    def test_every_case_sent_real_queries(self, artifact):
        """The queries come from `harness.profile_case_repo` and `collector.build_queries` —
        the shared implementations. A probe that invented its own would measure a query set no
        run sends."""
        for case, row in artifact["cases"].items():
            assert row["queries"], case
            assert all(isinstance(q, str) and q.strip() for q in row["queries"]), case
            assert not any("all:" in q or "cat:" in q for q in row["queries"]), (
                f"{case}: arXiv boolean syntax reached a keyword source — that is C-9"
            )


class TestTheCollisionIsReal:
    def test_europe_pmc_answers_every_repository(self, artifact):
        """The finding that rules out 'harmless outside biology'. Not one of the 25 gets
        silence, so enabling the source everywhere is never free."""
        silent = [c for c, r in artifact["cases"].items() if r["n_hits"] == 0]
        assert silent == []
        assert sum(r["n_hits"] for r in artifact["cases"].values()) == 1721

    def test_most_of_what_comes_back_is_indexed_as_biomedical(self, artifact):
        total = sum(r["n_hits"] for r in artifact["cases"].values())
        bio = sum(r["n_biomedical"] for r in artifact["cases"].values())
        assert round(bio / total, 2) == 0.68

    def test_every_repository_is_above_half(self, artifact):
        """Not an average dragged up by a few — the collision is present everywhere, 57% to
        87%. A single clean repository would weaken the case for routing considerably."""
        shares = {c: r["biomedical_share"] for c, r in artifact["cases"].items()}
        assert min(shares.values()) >= 0.5, min(shares, key=shares.get)
        assert max(shares.values()) <= 1.0

    def test_the_share_is_consistent_with_its_own_counts(self, artifact):
        for case, r in artifact["cases"].items():
            assert r["biomedical_share"] == pytest.approx(
                r["n_biomedical"] / r["n_hits"], abs=1e-3
            ), case


class TestTheFlagIsSoundThisTime:
    def test_the_flag_reads_fields_that_exist(self):
        """The regression test for the 21% that measured nothing.

        `journalTitle` and `pubType` are `None` on Europe PMC records; the real fields are
        `meshHeadingList`, `subsetList` and the nested `journalInfo`. A flag reading the
        first pair silently returns False for everything, which is indistinguishable from a
        genuinely clean result — and reads as good news.
        """
        import sys

        sys.path.insert(0, str(ROOT / "evals"))
        import epmc_collision as P

        mesh = {"meshHeadingList": {"meshHeading": [{"descriptorName": "Neoplasms"}]}}
        assert P._looks_biomedical(mesh), "MeSH headings are the primary signal"
        assert P._looks_biomedical({"subsetList": {"subset": [{"code": "IM"}]}})
        assert P._looks_biomedical(
            {"journalInfo": {"journal": {"title": "Nucleic Acids Research"}}}
        )
        # The fields the broken version read, and nothing else: must NOT be enough.
        assert not P._looks_biomedical({"journalTitle": None, "pubType": None})
        assert not P._looks_biomedical({})

    def test_a_computer_science_record_is_not_flagged(self, artifact):
        """The probe has to be able to answer 'on-topic', or it only confirms its premise.
        Europe PMC does index CS work — `crypto` gets real post-quantum cryptography — and
        those records carry no MeSH."""
        import sys

        sys.path.insert(0, str(ROOT / "evals"))
        import epmc_collision as P

        assert not P._looks_biomedical(
            {"title": "Post-Quantum Cryptocode Constructions on Elliptic Curves"}
        )
        unflagged = [
            h for r in artifact["cases"].values() for h in r["sample"] if not h["biomedical"]
        ]
        assert unflagged, "if nothing is unflagged the flag is not discriminating"

    def test_the_evidence_travels_with_the_number(self, artifact):
        """Samples carry the query that produced them, so a reader can check the collision
        rather than take 68% on trust — which is exactly what the first run's reader could
        not do."""
        for case, r in artifact["cases"].items():
            for h in r["sample"]:
                assert h["query"] in r["queries"], case
                assert h["title"], case


class TestTheOpenAlexArm:
    """P23: the same probe, the other source, and a better instrument.

    OpenAlex labels every work with `primary_topic.field` — its own 26-field taxonomy. That
    needs no marker list, cannot silently read an empty field (the P22 failure), and reports
    WHICH discipline came back rather than a yes/no.
    """

    @pytest.fixture(scope="class")
    def oa(self) -> dict:
        return json.loads(OA.read_text(encoding="utf-8"))

    def test_openalex_is_cleaner_than_europe_pmc_but_not_clean(self, oa, artifact):
        """48% against 68%. The headline is the comparison, not either number alone — and
        neither supports switching a source on for every repository."""
        oa_off = sum(r["n_off_domain"] for r in oa["cases"].values())
        oa_tot = sum(r["n_hits"] for r in oa["cases"].values())
        epmc_bio = sum(r["n_biomedical"] for r in artifact["cases"].values())
        epmc_tot = sum(r["n_hits"] for r in artifact["cases"].values())
        assert round(oa_off / oa_tot, 2) == 0.48
        assert oa_off / oa_tot < epmc_bio / epmc_tot

    def test_it_still_answers_every_repository(self, oa):
        assert [c for c, r in oa["cases"].items() if r["n_hits"] == 0] == []

    def test_computer_science_is_the_largest_field_but_a_minority(self, oa):
        """34% — the single biggest field, and nowhere near a majority. OpenAlex CAN reach
        the ACM/IEEE/VLDB literature Europe PMC structurally cannot; it just brings a great
        deal else along."""
        agg: dict[str, int] = {}
        for r in oa["cases"].values():
            for f, n in r["fields"].items():
                agg[f] = agg.get(f, 0) + n
        top = max(agg, key=agg.get)
        assert top == "Computer Science"
        assert 0.30 < agg[top] / sum(agg.values()) < 0.40

    def test_the_spread_is_much_wider_than_europe_pmcs(self, oa, artifact):
        """24% (`speech`) to 84% (`webdev`), against Europe PMC's 57–87%. The width is the
        finding: distinctive technical vocabularies retrieve cleanly, generic English does
        not — so the axis is the QUERY, which is what rules out per-domain routing."""
        oa_shares = [r["off_domain_share"] for r in oa["cases"].values()]
        epmc_shares = [r["biomedical_share"] for r in artifact["cases"].values()]
        assert max(oa_shares) - min(oa_shares) > max(epmc_shares) - min(epmc_shares)
        assert min(oa_shares) < 0.30 and max(oa_shares) > 0.80

    def test_the_field_counts_reconcile_with_the_hits(self, oa):
        for case, r in oa["cases"].items():
            assert sum(r["fields"].values()) == r["n_hits"], case

    def test_both_arms_share_one_query_builder(self, oa, artifact):
        """The part that had to be shared: a probe measuring a query set no run sends would
        answer a question about itself. Same repositories, same queries, both sources."""
        for case in artifact["cases"]:
            assert oa["cases"][case]["queries"] == artifact["cases"][case]["queries"], case
