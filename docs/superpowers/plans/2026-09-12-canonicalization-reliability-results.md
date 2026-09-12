# Canonicalization reliability: implementation results

The ten implementation tasks are delivered on `codex/canonicalization-reliability`.
Representative-corpus accuracy remains an external validation gate: no existing
corpus output or human labels were available in this checkout. No production
corpus was modified and no extraction/enrichment provider calls were made.

## Delivered changes

| Plan tasks | Implementation and regression evidence |
| --- | --- |
| 1–2: checkpoint integrity and source preservation | Schema-bearing empty CSVs; failed metadata blocks downstream work; prior run success summaries cleared; stable source registry; duplicate source bibliographies retained; valid graph endpoints. `test_reliability_pipeline.py`, `test_ids.py`, `test_graph.py`. |
| 3–4: enrichment provenance and attachment | Original-input cache fingerprints; identical fresh/cache behavior; canonical extracted author population in composed/staged runs; one-to-one provider attachment preserving blank positions; collision history invalidates unsafe legacy IDs. `test_enrich.py`, `test_reliability_pipeline.py`, `test_authors.py`. |
| 5: author conflict handling | Full-name, ORCID, corporate/person and same-work guards; occurrence-specific coauthor evidence; direct/transitive ambiguous identity bridges stay unresolved; attested name variants remain supported. `test_authors.py`. |
| 6–7: work matching | Unicode/list normalization; symmetric missing-author retrieval and multiple title routes; explicit match/review/reject evidence; negation, part-number and substantive-expansion guards. Small candidate oracle plus a 1,000-observation scoring-count regression. `test_dedup.py`, `test_reliability_dedup.py`. |
| 8: truthful audits | Actual work lineage, author occurrences/identifiers and decisions; effective configuration; input/output fingerprints including empty schemas; atomic JSON writes; missing/stale/invalid report states without recomputation. `test_io.py`, `test_resolution_audit.py`, `test_report.py`, `test_webui.py`. |
| 9: manual corrections | Occurrence merge/separate CSV; snapshot binding; contradictions, stale/cyclic aliases and invalid positions rejected before output mutation; correction reasons in audit. `test_author_overrides.py`, `test_reliability_pipeline.py`. |
| 10: evaluation and migration | Offline labeled fixtures, pair metrics, candidate recall, review workload, repeat/CSV/reversal checks; frozen unlabeled 150-work/150-author samples with observed lineage; copy-only migration guide. `test_canonicalization_evaluation.py`, `docs/USER_GUIDE.md`. |

## Verification on 2026-09-12

- Full suite: **464 passed**, no skips. Graph, plotting and loopback web-server
  tests executed. Runtime: project Python 3.11 virtual environment, with missing
  dev graph/plot packages installed in `/private/tmp/citegraph-test-deps`.
- Ruff and whitespace checks pass.
- Independent final review: no unresolved blockers in the reviewed fixes.
- Offline evaluator: 11 work cases and 11 author cases. Work pairs: TP 6,
  TN 5, FP 0, FN 0; 3 ambiguous pairs excluded. Author pairs: TP 3, TN 9,
  FP 0, FN 0; 2 ambiguous pairs excluded. Candidate recall is 1.0 on the
  six positive work pairs; author candidate recall is unavailable. These small
  synthetic counts describe regression coverage, not literature-wide accuracy.

Reproduction commands (with the project and dev extras installed):

```bash
python -m pytest -o addopts=''
ruff check .
python scripts/evaluate_canonicalization.py \
  --fixtures tests/fixtures/canonicalization \
  --out /private/tmp/citegraph-evaluation.json
```

## Remaining limits and adoption

Work clustering still selects the first compatible representative. The explicit
`representative_order_sensitivity` fixture demonstrates a changed partition when
reversed; this is reported rather than concealed. All author fixture partitions
remain equal under reversal, and conflicting direct-ID cases have permutation
regressions. This is not a universal order-invariance guarantee.

Keep `source_ids.json`, `work_id_collisions.json`, audits and correction bindings
with the corpus. IDs and rankings may change after corrected canonicalization.
Legacy collisions and changed metadata can require enrichment refresh; missing
reference caches can require paid extraction. Follow the preflight and copied
output workflow in `docs/USER_GUIDE.md` before adoption.

Next validation requires a representative corpus and human adjudication of the
frozen sample, including independently found duplicates to assess retrieval
omissions. Compare memberships and citation rankings in a copy before replacing
existing research outputs. The implementation branch has not been merged or
published.
