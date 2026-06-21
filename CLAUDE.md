# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common commands

```bash
pip install -e ".[dev]"            # editable install with test/lint deps
pip install -e ".[dev,crossref]"   # add CrossRef/OpenAlex enrichment

pytest                              # full test suite
pytest tests/test_dedup.py          # one file
pytest tests/test_dedup.py::test_name -v   # one test
pytest --cov=citegraph              # with coverage (pytest-cov is installed)

ruff check .                        # lint (CI runs this)
ruff check --fix .                  # autofix

# CLI smoke test (requires GOOGLE_API_KEY in env or .env)
citegraph run ./pdfs --out ./out

# Per-stage / progressive runs (each reads prior artifacts from --out):
citegraph convert ./pdfs --out ./out      # stage 1: PDFs -> markdown/
citegraph metadata --out ./out            # stage 2: -> papers.csv
citegraph references --out ./out          # stage 3: -> references_raw.csv
citegraph dedup --out ./out               # stage 4: -> references.csv, citation_graph.csv
citegraph enrich --out ./out              # stage 5 (optional): CrossRef/OpenAlex
citegraph authors --out ./out             # stage 4b: -> authors.csv, author_citations.csv
citegraph estimate --out ./out            # pre-flight token/cost estimate (no API calls)
citegraph status --out ./out              # report which artifacts exist
```

`citegraph run` runs `convert` first, prints a cost estimate, then prompts before any LLM call. Pass `--yes`/`-y` to skip the prompt (useful in CI/automation).

The library mirrors the CLI: every `Pipeline` stage method accepts its input
explicitly *or* loads it from `out_dir` if called with no arguments. Missing
upstream artifacts raise `StageNotReadyError` with a hint at the prior step.

`GOOGLE_API_KEY` is required for any path that calls Gemini. Tests avoid the network entirely (see "Testing without network" below).

## Architecture

`citegraph` turns a folder of academic PDFs into three CSVs (`papers.csv`, `references.csv`, `citation_graph.csv`). The whole flow is orchestrated by `Pipeline.run()` in [pipeline.py](src/citegraph/pipeline.py); everything else is a stage it composes.

### Stage pipeline (each stage is checkpointed on disk)

1. **PDFs → markdown** — [pdf_to_markdown.py](src/citegraph/pdf_to_markdown.py) calls `docling`. Output cached as `out_dir/markdown/<stem>.md`. Idempotent: re-runs skip files unless `overwrite_markdown=True`. Docling is imported lazily inside the function so `import citegraph` stays cheap. Pass `recursive=True` (Pipeline kwarg / `--recursive` CLI flag) to walk subdirectories of `pdf_dir`; in that mode cache stems carry the relative path via `cache_stem_for` (e.g. `journal_X/paper.pdf` → `journal_X__paper`) so same-named PDFs in different folders don't clobber each other. Hidden directories (names starting with `.`) are skipped, and a stem-collision pre-check raises a clear `ValueError` before docling is ever invoked. Pass `ocr="auto"` (Pipeline kwarg / `--ocr-auto` CLI flag) to convert normally first and re-run only image-only outputs with EasyOCR; pass `ocr=True` / `--ocr` to force full-page OCR for every PDF. After each conversion pass, `_check_conversion_quality` in [pipeline.py](src/citegraph/pipeline.py) checks every output markdown with `_is_image_only` (strips `<!-- image -->` tags and `#` headers; flags files with fewer than 200 substantive chars) and writes `conversion_warnings.json` when any are found.
2. **markdown → paper metadata** — [extract_metadata.py](src/citegraph/extract_metadata.py) sends each markdown to Gemini with `PaperMetadata` as the structured-output schema. Cached as `out_dir/metadata/<stem>.json`. The per-paper loop runs concurrently, capped by `Pipeline(llm_concurrency=...)` / `--llm-concurrency` / `CITEGRAPH_LLM_CONCURRENCY` (default `4`), while collecting results in input order and preserving per-paper failure isolation. After all papers are processed, `_detect_source_duplicates` in [pipeline.py](src/citegraph/pipeline.py) applies `compare_papers` (from [dedup.py](src/citegraph/dedup.py)) across the source-paper records to find differently-named PDFs that contain the same paper. Results are written to `source_duplicates.json` (absent = no duplicates). The references stage reads this file to emit an informative warning instead of a cryptic "no paper id" message when it encounters a duplicate markdown.
3. **markdown → reference list** — [extract_references.py](src/citegraph/extract_references.py) extracts `list[Reference]` per paper. Cached as `out_dir/references/<stem>.json`. Like metadata, the per-paper loop runs concurrently under `llm_concurrency` while preserving input-order CSV output and per-paper failure isolation. Three non-obvious behaviors wrap the LLM call: (a) before sending, `slice_to_references_section` cuts the markdown to its bibliography section using a strict `##`-style header regex (falls back to the full file when no header matches, so the paper is never dropped); (b) after the response arrives, `response_was_truncated` triggers a one-shot retry at 2× output cap when the bibliography hit `max_output_tokens`, hard-capped at 80k tokens; (c) `estimate_inline_citations` over the body warns at WARNING level when the LLM returned <50% of the references the body appears to cite (gated on ≥10 distinct citations and successful slicing — small bodies and unsliced papers skip the check to avoid noise). After the loop, `_write_no_references` writes `papers_no_references.json` for any paper that was successfully processed (not a failure, not a duplicate) but returned zero references — each entry has `paper_id`, `source_file`, and `title`. Common cause: image-only PDF that was not run with `--ocr-auto` / `--ocr`.
4. **fuzzy dedup → canonical references + edges** — [dedup.py](src/citegraph/dedup.py) clusters references with a weighted rapidfuzz score over title/authors/journal plus a year window. The `compare_papers` function fails *closed* on genuinely unparseable year data (e.g. a non-numeric string) — preserve that behavior. Missing-year sentinels (the schema's `Year=0`, `None`, or empty string) are treated as "year unknown" and do *not* reject the cluster; title+authors fuzzy carries the decision instead, so a reference cited with a year in one paper still merges with the same reference cited without a year in another. Output: `references.csv` (canonical, indexed by `id`) and `citation_graph.csv` (`citing_id`, `cited_id` edges).
5. **optional enrichment** — [enrich.py](src/citegraph/enrich.py) hits CrossRef then OpenAlex over the deduplicated references. Gated behind `enrich=True` *and* the `[crossref]` extra (httpx is imported via `_try_import_httpx` so the base install stays light). The `_normalize_record` step keeps OpenAlex `author.id` and ORCID alongside `display_name` in a parallel `OpenAlex_Authors` list (one dict per author, positionally aligned with `Authors_List`); the author-normalization stage reads this column from `enriched_references.csv` and treats those identifiers as ground truth for identity. Each enriched row also carries match diagnostics (`enrichment_status`, `enrichment_miss_reason`, `enrichment_title_score`, `enrichment_candidate_title`, `enrichment_year_match`), and the stage writes `enrichment_summary.json` plus `enrichment_misses.csv` for corpus-level review. Don't drop the column when refactoring enrichment — the author stage degrades gracefully without it but loses its single best disambiguation signal.
6. **author normalization → canonical authors + author-citation edges** — [authors.py](src/citegraph/authors.py) clusters every author across the corpus (both reference authors and source-paper authors) into canonical records. `parse_author` turns one raw string ("Cárdenas, J.-C.", "Adele Diamond", "García Márquez, Gabriel") into a structured `ParsedAuthor` (surname / given-names / initials / suffix) with diacritic-stripped `surname_norm` for blocking. `normalize_authors` blocks by `surname_norm`, then within each block applies a precision-first algorithm: full-first-name records form anchor clusters; initial-only records merge into an anchor only when exactly one matches the initial (or co-author overlap on the same record disambiguates). OpenAlex / ORCID ids from the enrichment stage override string evidence — same id ⇒ same cluster, different ids ⇒ different clusters even when names look identical. External-id clusters also act as anchors for no-id variants when the full name / initials match unambiguously, so enriched "Juan-Camilo Cardenas" can absorb no-id "Cardenas, J.C." rather than staying split. `merge_mode="loose"` collapses by `(surname, first_initial)` regardless. Hand-curated `cluster_id,canonical_id` overrides in `out_dir/author_aliases.csv` are applied last and let the user fix anything the algorithm gets wrong without rerunning. Output: `authors.csv` (canonical, indexed by `id`, `a-` prefix) and `author_citations.csv` (one row per occurrence with back-pointers to record id, position, and citing-paper id).

The cache layout is owned by `OutLayout` in [io.py](src/citegraph/io.py). If you add a new artifact, add it to `OutLayout` rather than hardcoding paths.

### `CitationGraph` query view

[graph.py](src/citegraph/graph.py) defines `CitationGraph`, a small read-only wrapper over the three output DataFrames (papers, references, edges) plus the optional author tables. It exists to deliver on the package name — without it, the library produces edges in a CSV but offers nothing to traverse. Two constructors: `CitationGraph.from_out_dir(out_dir)` (loads CSVs) and `CitationGraph.from_pipeline_result(result)`. Methods are deliberately scope-tight: `n_papers` / `n_references` / `n_edges`, `top_cited(n)`, `cited_by(paper_id)`, `citers_of(reference_id)`, `to_networkx()`. `networkx` is a *lazy* import so the base install stays light; the import-error message tells the user how to install. Note the asymmetry it preserves: `papers` keeps `id` as a column (matching `papers.csv`), `references` is indexed by `id` (matching `references.csv`) — methods do the right thing on each side, callers rarely need to think about it.

When `authors.csv` and `author_citations.csv` exist in `out_dir`, `from_out_dir` loads them too and `has_authors` becomes `True`, unlocking `top_cited_authors(n)`, `find_author(query)` (diacritic-insensitive substring match on surname or display name), `citations_of(author_id)` (every reference where the author was cited), `papers_citing_author(author_id)`, `citation_context_for_author(author_id)`, `citing_papers_by_author(author_id)`, and `source_journals_citing_author(author_id)`. The last three join through `citation_graph.csv` so source-paper journal rollups answer "which journals are doing the citing?" rather than "where were the cited works published?" These methods raise `RuntimeError` with an actionable hint when called on a graph whose author tables aren't loaded, rather than silently returning empty results.

### Per-paper failure isolation and warning files

Both per-paper extraction loops (`Pipeline.extract_paper_metadata` and `Pipeline.extract_paper_references`) wrap the per-paper body in `try/except Exception`. A single bad PDF — or one Gemini error after retries are exhausted — is recorded as a `PaperFailure` row and the loop continues with the rest. Failures are persisted as JSONL via `_write_failures` to `out_dir/metadata_failures.jsonl` / `references_failures.jsonl`; the file is *removed* when there are no failures so `path.exists()` <=> failures occurred. The cache is never written for a failed paper, so re-runs naturally retry it.

Most warning sidecar files follow the "absent = clean" convention. Stage 5
enrichment review files are written when enrichment runs so match quality can be
inspected:

| File | Stage | Created when |
| ---- | ----- | ------------ |
| `conversion_warnings.json` | 1 (convert) | any markdown is image-only after conversion |
| `source_duplicates.json` | 2 (metadata) | two PDFs contain the same paper |
| `papers_no_references.json` | 3 (references) | a successfully-processed paper returned 0 references |
| `metadata_failures.jsonl` | 2 (metadata) | at least one paper raised an exception |
| `references_failures.jsonl` | 3 (references) | at least one paper raised an exception |
| `enrichment_summary.json` | 5 (enrich) | enrichment ran; contains match/miss/source counts and config |
| `enrichment_misses.csv` | 5 (enrich) | enrichment ran and produced unmatched references for review |
| `author_review.json` | 6 (authors) | a cluster looks low-confidence (initial-only with several citations and no external id) |

Counts for the pipeline warning sidecars are surfaced in `run_summary.json` (`n_conversion_warnings`, `n_source_duplicates`, `n_papers_no_references`, `n_metadata_failures`, `n_references_failures`, `n_author_review_flags`). The `citegraph run`, `citegraph convert`, and `citegraph authors` CLIs emit a yellow warning line with the file path when any are non-empty.

### Stable IDs

[ids.py](src/citegraph/ids.py) builds slug-style IDs from `(first-author-surname, year, title)` — `p-` prefix for source papers, `r-` for references. IDs must be deterministic across runs (the dedup step assumes the *first* row in a cluster names the cluster), so don't introduce hash-based or row-index-based IDs.

Author cluster IDs follow the same principle: `_make_author_id` in [authors.py](src/citegraph/authors.py) slugs `surname_norm + given_token` to produce `a-…` ids, where `given_token` is the longest full first name observed in the cluster (or the canonical initials if no full name is present). When two genuinely distinct people slug to the same id (rare, e.g. two "John Smith" clusters split by OpenAlex), a numeric suffix `-2`, `-3`, … disambiguates. The hand-curated `author_aliases.csv` references these ids directly, so keep id generation deterministic across runs.

### Gemini wrapper

All provider-specific code lives in [llm.py](src/citegraph/llm.py). Other modules speak Pydantic models, not Gemini types. Three non-obvious behaviors:

- `tenacity` retry with exponential backoff wraps every call.
- `response_was_truncated(response)` inspects `candidates[*].finish_reason` for `MAX_TOKENS` (defensive against SDK shape variation — works whether `finish_reason` is an enum or a plain string). The references stage uses this to retry once with a 2× output cap *before* falling back to JSON repair; metadata calls don't bother (they're capped at 500 output tokens).
- `fix_incomplete_json_string` is the last-resort safety net used by `parse_structured_response` when the JSON itself fails to parse (e.g. response truncated despite the retry). If you change the JSON shape that references come back as, also update the repair function.

### Schemas double as DTOs *and* response schemas

`PaperMetadata` / `Reference` in [schemas.py](src/citegraph/schemas.py) are passed to Gemini as `response_schema` and also used as the in-memory record format and the JSON cache format. Renaming fields (`Title`, `Authors_List`, `Journal`, `Year`) breaks both the LLM contract and the on-disk cache simultaneously — bump cache compatibility deliberately. `Authors` is *not* part of the LLM contract: it's a derived `@property` (`", ".join(Authors_List)`) added back in via a `model_dump()` override so CSV/cache writers and downstream consumers (e.g. `dedup.py`) still see a single human-readable string. Old caches that contain an `Authors` key load fine because `extra="ignore"` drops it during validation; the property recomputes on access.

### Configuration

Runtime settings flow through `pydantic-settings` in [config.py](src/citegraph/config.py) (env vars + `.env`). User-facing knobs (dedup weights, threshold, year window, model, enrich, ocr, LLM concurrency, author merge mode) are exposed both on `Pipeline(...)` and on the relevant `citegraph` CLIs in [cli.py](src/citegraph/cli.py); keep those two surfaces in sync when adding a knob.

### Cost estimation

[cost_estimation.py](src/citegraph/cost_estimation.py) gives a pre-flight token + USD estimate for stages 2 and 3 *without making any API calls*. It walks `out_dir/markdown/`, skips files whose `metadata/<stem>.json` and `references/<stem>.json` caches already exist, and runs the same `slice_to_references_section` slicer to size the references-stage input. Surfaced as `Pipeline.estimate_extraction_cost()` and `citegraph estimate`; `citegraph run` calls it after stage 1 and prompts before any LLM call (`--yes` skips). The heuristic constants at the top of the module (`_CHARS_PER_TOKEN`, `_METADATA_OVERHEAD_CHARS`, `_TOKENS_PER_REFERENCE`, etc.) are deliberately rough — adjust them if benchmarks drift. The `_PRICING` table is dated (mid-2025 Gemini rates); update it from https://ai.google.dev/pricing rather than trusting it as ground truth. Unknown models return `cost_usd() == None` and the summary degrades gracefully rather than failing.

### `research/` is not part of the package

Scripts under [research/](research/) are kept for historical context, not installed by pip, excluded from CI, and lint-loosened in `pyproject.toml`. Don't import from `research/` in `src/`.

## Testing without network

Tests must never hit Gemini. The pattern is in [test_pipeline.py](tests/test_pipeline.py): subclass `GeminiClient` with a `_FakeClient` that overrides `generate_structured`, distinguishing metadata vs. references requests by sniffing the prompt text (`"references or bibliography section" in prompt`). Pass it via `Pipeline(client=_FakeClient())`. If you change the prompt strings in `extract_metadata.py` / `extract_references.py`, update the sniff in `_FakeClient` too.
