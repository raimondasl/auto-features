# Timeline of registrations

Every "registered before" claim in the paper rests on commit order in the development
repository, which this mirror does not carry. The times below are the commit times of each
registration and of the first committed result it governs, in UTC. The commit identifiers are
withheld during review and will be added with the de-anonymised version; they can then be checked
against the public history.

| registration | registered | first result committed |
|---|---|---|
| `evals/PREREG-rung1.md`, the three labels of the system comparison | 2026-08-31 15:16 | 2026-08-31 19:08 |
| `evals/PREREG-judge-validity-pool.md`, the adoption walk and its controls | 2026-09-02 16:35 | 2026-09-06 20:29 |
| `evals/PREREG-judge-crossrepo-controls.md`, the cross-repository controls | 2026-09-07 00:22 | 2026-09-07 17:17 |
| `evals/PREREG-finescale-current-gate.md`, the rescore on the shipped band | 2026-09-20 22:32 | 2026-09-20 22:54 |
| `evals/PREREG-finescale-model-transfer.md`, the second scorer | 2026-09-21 15:50 | 2026-09-21 17:33 |
| `evals/PREREG-sonnet-id-probe.md`, the identifier-formatting re-judging | 2026-09-22 06:22 | 2026-09-22 07:10 |
| `evals/PREREG-third-judge.md`, the third judge | 2026-09-22 19:12 | 2026-09-22 20:14 |
| `evals/registrations/E1-E5-registration.md`, the five band-ordering mechanisms | 2026-08-08 01:00 | 2026-08-08 03:47 |
| `evals/registrations/scientific-cohort-registration.md`, the scientific cohort | 2026-08-20 03:24 | 2026-08-20 06:19 |
| `evals/registrations/benchmark-expansion-rules-draft.md` | 2026-09-02 05:31, never registered | none |

Notes:

- The third judge's first verdict is timestamped 2026-09-22 19:21 UTC in the data bundle's
  verdict cache, nine minutes after its registration. The registration's own header says
  "Registered 2026-09-23". That is a typo in a file the run froze by hash, so it was not edited.
- The system comparison's registration fixed its control run before any margin on it existed. The
  "+0.54/case" it calls the published margin came from an earlier run of the same configuration;
  the paper reports the run the registration fixed.
- The E1 to E5 registration gives the only-3s reference policy as "+0.50, 16/22 abstain". On the
  registered testbed file it computes to +0.82 with 14 of 22 abstaining, the figure the paper uses.
  The registration mis-transcribed a figure from an earlier run.
- E5 ran five of the seven features its registration listed (age, S2 citations, S2 influential
  citations, HyDE rank, hop coupling). Has-code, stars and SPECTER2 cosine were dropped and raw
  citations added.
