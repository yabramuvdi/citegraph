# PyPI Release Hardening Design

**Date:** 2026-09-19

## Goal

Resolve four verified release blockers without expanding the library's scope:
make the advertised OCR install complete, remove an invalid aggregate statistic,
refuse unmatched co-citation nulls, and prevent curated workbook sheets from
overwriting generated sheets.

## Scope

This hardening pass changes four existing behaviors:

1. The `pdf` and `all` package extras must install the EasyOCR engine selected
   by the forced and automatic OCR paths.
2. The co-citation API must not report a binomial aggregate p-value for
   dependent ties.
3. A tie without a degree-matched substitute for either endpoint must be
   skipped rather than scored against substitutes for only one endpoint.
4. Extra workbook sheet names must not collide with generated or other extra
   sheets.

Release workflow changes, new statistical methods, general Excel-name
sanitization, and unrelated release polish are outside this design.

## 1. OCR Dependency Contract

`convert_pdf_to_markdown(..., ocr=True)` explicitly constructs Docling's
`EasyOcrOptions`. Plain `docling` does not install EasyOCR in current releases;
Docling exposes it through its `easyocr` extra and otherwise raises at runtime.

The `pdf` and `all` extras will therefore require `docling[easyocr]>=2.0`. The
base installation remains unchanged and lightweight. The development extra will
not gain Docling because conversion tests intentionally use fakes and installing
Docling would make every contributor and CI job download its large ML stack.

The packaging test will check the semantic requirement: the base dependencies
exclude Docling, while both `pdf` and `all` include `docling[easyocr]`.

## 2. Honest Co-citation Aggregates

`TieCocitationTest.binomial_p_value` applies a binomial tail to the number of
individually significant ties while claiming validity when ties share authors.
That conclusion does not follow from valid marginal p-values: the binomial tail
requires independent Bernoulli trials. Shared-advisor ties are dependent by
construction.

The invalid aggregate will be removed rather than replaced:

- remove `TieCocitationTest.binomial_p_value`;
- remove the public `binomial_tail_p` helper and its `__all__` entry;
- remove its unit tests and all guidance recommending it;
- retain `n_significant` and `expected_significant` as descriptive summaries;
- retain `fisher_p_value`, labeled as an assumption-bound diagnostic that is
  not confirmatory evidence for overlapping ties.

This is an unreleased API, so removal needs no compatibility shim or
deprecation cycle. A dependency-aware global test would be a new research
method and is explicitly deferred until it has its own design and validation.

## 3. Degree-matched Null Eligibility

Each endpoint of a tie must have at least one substitute in its own prominence
stratum. If either `source_pool` or `target_pool` is empty, the tie will be
recorded in `skipped` with a reason that identifies the missing degree-matched
comparison. The sampler will no longer fall back to drawing both authors from
the non-empty pool.

After the per-endpoint check, the existing distinct-pair guard remains: two
non-empty pools can still contain only the same single author, which cannot form
a valid null pair. Sampling continues to redraw collisions between otherwise
valid pools.

A regression fixture will reproduce the verified failure: one highly prominent
endpoint has no substitute, the other does, and the current code reports a very
small p-value with `pool_size == 0`. The corrected behavior must return no scored
tie and one explicit skipped result.

## 4. Workbook Sheet-name Validation

`build_core_author_workbook` is the shared boundary for CLI and library calls,
so it will validate every `ExtraSheet.name` before filtering data or opening the
destination workbook.

Names are compared case-insensitively because Excel worksheet names are
case-insensitive. The permanently reserved generated names are:

- `README`
- `Authors`
- `Works`
- `Review_Flags`

`Review_Flags` remains reserved even when that optional table is absent, keeping
the contract stable across corpora. Two extra sheets that differ only by case
are also rejected. Errors name the conflicting sheets and occur before
`pd.ExcelWriter` opens or truncates the destination.

The regression tests will cover a reserved-name collision and a case-insensitive
duplicate extra name. One test will pre-create a destination sentinel and assert
that failed validation leaves it unchanged.

General Excel constraints such as forbidden characters and the 31-character
limit remain openpyxl's responsibility; they do not cause the silent overwrite
being fixed here.

## Documentation

The implementation will update:

- `README.md` installation text only if necessary to keep the OCR extra behavior
  explicit;
- `CLAUDE.md` to remove the invalid binomial recommendation and describe skipped
  unmatched strata;
- `docs/superpowers/specs/2026-09-15-supervision-cocitation-figures-design.md`
  to remove the invalid aggregate from the approved research design.

## Verification

Each behavior is implemented test-first. Focused verification is followed by
the entire offline release gate:

```bash
pytest tests/test_packaging.py tests/test_cocitation_stats.py tests/test_author_export.py -v
pytest -o addopts=''
ruff check .
python -m build
python -m twine check --strict dist/*
```

The release claim requires all optional graph, plotting, and Excel tests to run
without skips. No Gemini, enrichment-provider, or paid network calls are needed.

## Acceptance Criteria

- Installing `citegraph[pdf]` or `citegraph[all]` declares EasyOCR through
  Docling's supported extra.
- The public co-citation API contains no binomial aggregate p-value or helper.
- A tie with an empty endpoint stratum is skipped and never receives a p-value.
- Reserved and duplicate extra sheet names fail before the output file changes.
- Documentation makes no claim that a combined p-value is valid for overlapping
  ties.
- Focused tests, the full suite, Ruff, build, and strict Twine checks pass.
