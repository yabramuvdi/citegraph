# Changelog

## 0.2.1 — 2026-11-01

- Describe the library as building and analyzing citation and author networks
  with LLM-assisted extraction.
- Make the current Gemini-only extraction support explicit in the README and
  user guide, with additional LLM providers planned. This release changes
  documentation and package metadata; it does not add another LLM provider.

## 0.2.0 — 2026-11-01

Public research beta. This release replaces the earlier papers/references model
with a unified works model. Review generated records before using them in research.

### Breaking changes and migration

- Source papers and cited references now share canonical `w-` work IDs, with
  discovery depth in `ring` and processed-PDF provenance in `source_file`.
- `sources.csv`, `citations_raw.csv`, `works.csv`, and `citation_graph.csv`
  replace the old papers/references artifacts. `PipelineResult` and
  `CitationGraph` expose the unified works model.
- Rebuild existing outputs in a copy using the
  [migration guide](docs/USER_GUIDE.md#rebuild-an-existing-corpus-in-a-copy).
  Preserve identity registries, correction files, and extraction caches. Corrected
  clusters can change IDs, citation counts, and rankings; invalidated or missing
  caches can require new provider calls.

### Added and improved

- Work and author resolution safeguards, saved decision audits, manual identity
  constraints, stable identity registries, and fingerprinted incremental caches.
- Enrichment over core and cited works, author-agreement checks, OpenAlex API-key
  support, transient-failure retries, and conservative journal normalization.
- Automatic OCR retries for image-only and glyph-corrupted conversions, extraction
  failure isolation, and atomic JSON and CSV checkpoints.
- Author co-citation networks and per-tie comparison tests, optional plotting,
  core-author Excel export, work annotations, and title-page JEL extraction.
- A read-only local monitor, self-contained quality-control reports, a staged user
  guide, and a finished demo corpus that can be queried without an API key.
- Optional PDF conversion dependencies; the PDF extras include EasyOCR support.

### Known limitations

- Human-reviewed, held-out corpus accuracy has not been established. Synthetic
  evaluation results measure regression coverage, not extraction or matching
  accuracy on arbitrary literature.
- Work clustering can depend on input order for ambiguous matches. Archive inputs,
  outputs, corrections, and the package version for reproducible analysis.
- Extraction requires Gemini and sends converted paper text to Google. Cost
  estimates are approximate; the `metadata` command and Python API do not prompt
  for cost approval.
- `CITEGRAPH_REQUEST_TIMEOUT_S` is defined but is not yet applied to Gemini calls.
- Overlapping author ties do not support a confirmatory combined p-value; the
  Fisher diagnostic assumes independent ties.
