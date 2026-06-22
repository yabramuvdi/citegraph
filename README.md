# citegraph

Build a deduplicated citation graph from a folder of academic PDFs.

`citegraph` walks a directory of PDFs, converts each one to markdown with
[`docling`](https://github.com/DS4SD/docling), uses Google Gemini to extract
the paper's own bibliographic metadata and its full reference list, runs a
fuzzy deduplication pass on the references (rapidfuzz on title + authors +
year), and optionally enriches references with canonical metadata from
CrossRef / OpenAlex. The result is three CSVs you can analyse directly:

- `papers.csv` &mdash; one row per source paper.
- `references.csv` &mdash; one row per cited reference (deduplicated).
- `citation_graph.csv` &mdash; `(citing_id, cited_id)` edges.

## Install

```bash
pip install citegraph                  # base install (Gemini + fuzzy dedup + graph queries)
pip install "citegraph[pdf]"           # + PDF conversion via docling
pip install "citegraph[all]"           # + PDF conversion and CrossRef/OpenAlex enrichment
```

You'll need a Google Gemini API key:

```bash
export GOOGLE_API_KEY="your-key-here"
```

(or put it in a `.env` file in your working directory).

## Quickstart

```python
from citegraph import Pipeline

pipe = Pipeline(pdf_dir="./pdfs", out_dir="./out", enrich=True)
result = pipe.run()

result.papers       # DataFrame of source papers (with stable IDs)
result.references   # DataFrame of all extracted references
result.graph        # DataFrame of (citing_id, cited_id) edges
```

Or via the CLI:

```bash
citegraph run ./pdfs --out ./out --enrich
citegraph dedup ./out/references_raw.csv --out ./out
```

### Scanned PDFs

If some of your PDFs are scanned (each page is a bitmap with no selectable text),
use `ocr="auto"` to convert normally first and re-run only image-only outputs
with EasyOCR:

```python
pipe = Pipeline(pdf_dir="./pdfs", out_dir="./out", ocr="auto")
result = pipe.run()
```

Or with the CLI:

```bash
citegraph convert ./pdfs --out ./out --ocr-auto
citegraph run ./pdfs --out ./out --ocr-auto
```

Use `ocr=True` / `--ocr` when you want to force full-page OCR for every PDF.

After each conversion run `citegraph` writes a `conversion_warnings.json` sidecar
listing any markdown files that appear to be image-only (fewer than 200
substantive text characters after stripping image tags). If you see warnings
there, re-running with `--ocr-auto` is usually the fastest fix; `--ocr` is the
force-everything option.

Every step is checkpointed on disk: re-running skips work that's already done,
so an interrupted LLM pass doesn't waste API calls.

## Cost estimate

`citegraph run` converts your PDFs to markdown first, then prints an estimated
token usage and dollar cost for the LLM extraction stages and waits for
confirmation before making any API calls. Pass `--yes` / `-y` to skip the
prompt.

To inspect the estimate without running the pipeline:

```bash
citegraph convert ./pdfs --out ./out      # produce markdown
citegraph estimate --out ./out            # report tokens / cost; no API calls
```

Or from the library:

```python
estimate = pipe.estimate_extraction_cost()
print(estimate.format_summary())
```

Files whose `metadata/<stem>.json` and `references/<stem>.json` caches already
exist are excluded from the totals. Pricing numbers are approximate Gemini
rates; the table lives in
[`src/citegraph/cost_estimation.py`](src/citegraph/cost_estimation.py) and
should be checked against https://ai.google.dev/pricing before trusting the
output for budgeting.

## Real smoke test

For a new corpus, especially while tuning OCR, LLM concurrency, or enrichment,
prefer a staged smoke test over a single `run` command. This uses the same
checkpointed artifacts as the full pipeline, but gives you an inspection point
after every externally fragile step.

From a repository checkout, the maintained helper is:

```bash
ENRICH_CONTACT="you@example.com" scripts/smoke_test.sh /path/to/pdf-folder
```

Set `OUT_DIR`, `MODEL`, `LLM_CONCURRENCY`, `ENRICH_MAX_WORKERS`, or
`ENRICH_TIMEOUT` in the environment to override the defaults.

If you installed from PyPI, use the CLI directly. The expanded command sequence
is:

```bash
export PDF_DIR="/path/to/pdf-folder"
export OUT_DIR="/tmp/citegraph_smoke_out"

citegraph convert "$PDF_DIR" --out "$OUT_DIR" --recursive --ocr-auto --verbose
citegraph estimate --out "$OUT_DIR" --model gemini-3.1-flash-lite

citegraph metadata --out "$OUT_DIR" --model gemini-3.1-flash-lite --llm-concurrency 4
citegraph references --out "$OUT_DIR" --model gemini-3.1-flash-lite --llm-concurrency 4 --yes

citegraph dedup --out "$OUT_DIR"
citegraph enrich --out "$OUT_DIR" \
  --enrich-contact "you@example.com" \
  --enrich-max-workers 2 \
  --enrich-timeout 15
citegraph authors --out "$OUT_DIR"

citegraph status --out "$OUT_DIR"
```

Two details matter in practice:

- Keep `--ocr-auto` on the first pass unless you already know every PDF has
  selectable text. If conversion leaves `conversion_warnings.json`, inspect it
  before paying for LLM extraction.
- Use a real `--enrich-contact` and a modest `--enrich-max-workers` value when
  querying CrossRef/OpenAlex. This is slower, but friendlier to provider rate
  limits; re-runs reuse the per-reference cache in `OUT_DIR/enrichment/`.

After the smoke test, load the graph directly from disk:

```python
from citegraph import CitationGraph

g = CitationGraph.from_out_dir("/tmp/citegraph_smoke_out")
print(g)
print(g.top_cited(10)[["Title", "Year", "citation_count"]])

if g.has_authors:
    print(g.top_cited_authors(10)[["display_name", "n_reference_citations"]])
```

## Pipeline

```
PDFs ──docling──▶ markdown ──Gemini──▶ metadata + references
                                          │
                                          ▼
                              fuzzy dedup (rapidfuzz)
                                          │
                                  optional CrossRef/OpenAlex
                                          │
                                          ▼
                                  citation_graph.csv
```

## Output schema

After a run, `out_dir/` contains the four CSVs below plus a few sidecar
files described at the end of this section. The per-stage caches live
under `markdown/`, `metadata/<paper-id>.json`, and
`references/<paper-id>.json` &mdash; safe to inspect, safe to delete to
force a re-extraction.

### `papers.csv` &mdash; one row per source paper

| Column | Type | Notes |
| ------ | ---- | ----- |
| `id` | string | `p-<surname>-<year>-<title-slug>`, deterministic across runs |
| `Title` | string | as extracted by the LLM |
| `Authors_List` | string | a Python list serialized via `repr`, e.g. `"['Jane Doe', 'John Smith']"` &mdash; round-trip with `ast.literal_eval` |
| `Authors` | string | derived `", ".join(Authors_List)`, ready for display |
| `Journal` | string | venue / publication |
| `Year` | int | publication year (`0` is the missing-value sentinel) |
| `source_file` | string | original PDF filename |

### `references_raw.csv` &mdash; one row per *citation event* (before dedup)

Same columns as `papers.csv` minus `source_file` and `id`, plus:

| Column | Type | Notes |
| ------ | ---- | ----- |
| `citing_id` | string | the `p-…` id of the paper that contained this citation |

A reference cited by N papers appears N times here, with N different `citing_id`s.

### `references.csv` &mdash; deduplicated references, indexed by `id`

| Column | Type | Notes |
| ------ | ---- | ----- |
| `id` (index) | string | `r-<surname>-<year>-<title-slug>` |
| `Title`, `Authors`, `Journal`, `Year` | | from the first member of the cluster |
| `doi` | string | only after `--enrich`; `None` for unmatched rows |
| `enrichment_source` | string | only after `--enrich`; `"crossref"` or `"openalex"` |
| `enrichment_status` | string | only after `--enrich`; `"matched"` or `"miss"` |
| `enrichment_miss_reason` | string | only after `--enrich`; reason for unmatched rows, e.g. `"no_openalex_candidates"`, `"below_title_threshold"`, `"year_mismatch"`, or `"http_error"` |
| `enrichment_title_score` | float | only after `--enrich`; raw fuzzy title score for the best accepted or rejected candidate |
| `enrichment_adjusted_score` | float | only after `--enrich`; title score after applying the year-mismatch penalty |
| `enrichment_candidate_title` | string | only after `--enrich`; external title that was accepted or explains the miss |
| `enrichment_year_match` | bool | only after `--enrich`; whether the input and candidate years agreed when both were known |
| `enrichment_year_delta` | int | only after `--enrich`; absolute year difference when both years were known |

`Authors_List` is preserved through the dedup stage so author normalization can use structured author strings instead of comma-splitting display text.

### `citation_graph.csv` &mdash; the actual graph

| Column | Type | Notes |
| ------ | ---- | ----- |
| `citing_id` | string | `p-…` |
| `cited_id` | string | `r-…` |

### Querying the graph

For ad-hoc analysis there's a small `CitationGraph` class. Construct it
either directly from a finished run's `out_dir` (the most common shape:
"I ran extraction yesterday, today I want to query") or from a freshly
returned `PipelineResult`:

```python
from citegraph import CitationGraph

g = CitationGraph.from_out_dir("./out")
# or: g = CitationGraph.from_pipeline_result(pipe.run())

g                                  # <CitationGraph: 25 papers, 432 references, 678 edges>
g.n_papers, g.n_references, g.n_edges

g.top_cited(n=10)                  # most-cited references in this corpus
g.cited_by("p-doe-2020-some-paper")
g.citers_of("r-hardin-1968-tragedy-of-the-commons")
```

When `authors.csv` and `author_citations.csv` exist, author-level queries
are available too. These join through `citation_graph.csv`, so source-paper
journal counts answer "which journals are doing the citing?":

```python
cardenas_id = g.find_author("cardenas").index[0]

g.citation_context_for_author(cardenas_id)   # one source-paper -> cited-reference edge per row
g.citing_papers_by_author(cardenas_id)       # distinct source papers citing that author
g.source_journals_citing_author(cardenas_id) # source-paper journal rollup
```

For graph analysis, export to [`networkx`](https://networkx.org/) (not a
default dependency &mdash; install separately):

```python
nx_graph = g.to_networkx()         # DiGraph with edges citing -> cited
```

### Sidecar files

Most warning sidecar files follow the same convention: **absent means clean**.
They are created only when the condition they describe actually occurred, so
`path.exists()` is a sufficient check. Enrichment review files are different:
they are written when stage 5 runs so you can inspect match quality.

- `run_summary.json` &mdash; counts for the whole run: `n_papers`, `n_references_raw`, `n_references_dedup`, `n_edges`, `n_metadata_failures`, `n_references_failures`, `n_source_duplicates`, `n_papers_no_references`, `n_conversion_warnings`, plus the model id and dedup configuration.
- `metadata_failures.jsonl` and `references_failures.jsonl` &mdash; one JSON line per failed paper: `{source_file, stage, error_class, error_message}`. The pipeline keeps going past per-paper failures; re-running will retry them (their caches were not written).
- `source_duplicates.json` &mdash; written when two differently-named PDFs contain the same paper (detected by the same fuzzy-match logic used for reference deduplication). Each entry names the canonical source file, the duplicate file(s), and the paper title. The duplicate PDFs are silently skipped in the references stage; remove them from `pdf_dir` and re-run to clean up.
- `papers_no_references.json` &mdash; written when a paper is successfully processed but the LLM returned an empty reference list. Each entry has `paper_id`, `source_file`, and `title`. Common causes: image-only PDFs (see `--ocr-auto` / `--ocr`), papers with no bibliography section, or Gemini returning an empty list despite the content being present.
- `conversion_warnings.json` &mdash; written after stage 1 when any output markdown appears to be image-only (scanned PDF converted without OCR). Each entry has `source_file` and a human-readable `reason`. Re-run stage 1 with `--ocr-auto` to regenerate only the flagged files with OCR, or `--ocr` to force OCR for every PDF.
- `enrichment_summary.json` &mdash; written by stage 5 when enrichment runs. Contains match/miss counts, source counts, match rate, and the `EnrichConfig` values used.
- `enrichment_misses.csv` &mdash; written by stage 5 when enrichment runs. Lists unmatched references with their miss reason so they can be inspected or curated later.

## Evaluation

A small gold standard lives in [`evaluation/gold/`](evaluation/gold/). Run

```bash
python evaluation/evaluate.py --gold evaluation/gold --out ./out
```

to get metadata field-level accuracy, reference recall/precision, and
deduplication F1. Numbers from the reference run are reported in
[`evaluation/RESULTS.md`](evaluation/RESULTS.md).

## Configuration

`Pipeline(...)` arguments:

| Argument | Default | Meaning |
| -------- | ------- | ------- |
| `pdf_dir` | required | folder of input PDFs (`None` is allowed when resuming from a previous run's cache) |
| `out_dir` | required | where outputs and checkpoints are written |
| `model` | `gemini-3.1-flash-lite` | Gemini model id (or set `CITEGRAPH_MODEL`) |
| `enrich` | `False` | run a CrossRef/OpenAlex pass over the deduplicated references (requires the `[crossref]` or `[all]` extra) |
| `enrich_config` | `EnrichConfig()` | tune the enrichment pass &mdash; see below |
| `dedup_config` | `DedupConfig()` | tune the fuzzy dedup &mdash; see below |
| `llm_concurrency` | `4` | maximum concurrent Gemini extraction calls for metadata/references (or set `CITEGRAPH_LLM_CONCURRENCY`). CLI flag on LLM stages: `--llm-concurrency` |
| `overwrite_markdown` | `False` | re-run docling even when a cached `.md` exists (PDF conversion requires the `[pdf]` or `[all]` extra) |
| `recursive` | `False` | walk subdirectories of `pdf_dir`. Cache filenames are disambiguated by relative path (e.g. `journal_X/foo.pdf` → `journal_X__foo.md`); a clear error is raised if two PDFs would still collide. CLI flag: `--recursive` / `-r` |
| `ocr` | `False` | OCR mode: `False` disables OCR, `"auto"` retries image-only outputs with OCR, and `True` forces full-page OCR for every PDF. CLI flags: `--ocr-auto` / `--ocr` |

Dedup is tuned through `DedupConfig`:

```python
from citegraph import Pipeline
from citegraph.dedup import DedupConfig

pipe = Pipeline(
    pdf_dir="./pdfs",
    out_dir="./out",
    dedup_config=DedupConfig(
        threshold=85.0,        # similarity threshold (0-100)
        title_weight=0.7,
        authors_weight=0.3,
        journal_weight=0.0,
        year_window=1,         # max year difference for a match
    ),
)
```

Enrichment is tuned through `EnrichConfig`:

```python
from citegraph.enrich import EnrichConfig

pipe = Pipeline(
    pdf_dir="./pdfs",
    out_dir="./out",
    enrich=True,
    enrich_config=EnrichConfig(
        contact_email="you@example.com",  # CrossRef polite-pool User-Agent
        title_match_threshold=90.0,
        timeout_s=15.0,
        rows=3,                           # API result rows to consider
        max_workers=8,                    # concurrent enrichment workers
        year_mismatch_penalty=8.0,         # subtract when candidate year differs
        retry_attempts=3,                  # retry transient 429/503/timeouts
        retry_wait_s=0.25,                 # exponential-backoff base wait
    ),
)
```

The CLI exposes the same knobs as flags &mdash; see `citegraph run --help`.

When enrichment runs, every row gets explicit diagnostics in
`enriched_references.csv`: `enrichment_status`, `enrichment_miss_reason`,
`enrichment_title_score`, `enrichment_adjusted_score`,
`enrichment_candidate_title`, `enrichment_year_match`, and
`enrichment_year_delta`. The same run writes `enrichment_summary.json` and
`enrichment_misses.csv` under `out_dir/` for quick review. The enrichment CLI
flags mirror these knobs: `--enrich-year-penalty`,
`--enrich-retry-attempts`, `--enrich-retry-wait`, and
`--enrich-max-workers`.

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check .
```

CI runs the same lint + test commands on Python 3.10 / 3.11 / 3.12; see
[.github/workflows/ci.yml](.github/workflows/ci.yml).

## Releasing

The release workflow ([.github/workflows/publish.yml](.github/workflows/publish.yml))
uses PyPI's [trusted publishing](https://docs.pypi.org/trusted-publishers/),
so no API token is stored in the repo. See [docs/RELEASE.md](docs/RELEASE.md)
for the full checklist. To cut a release:

1. Bump the `version` in [`pyproject.toml`](pyproject.toml) and in
   `src/citegraph/__init__.py`.
2. Commit and push to `main`.
3. `git tag v0.1.0 && git push --tags` &mdash; the workflow then builds an
   sdist + wheel and publishes them to PyPI.

To dry-run against TestPyPI first, build locally and upload manually:

```bash
python -m build
python -m twine upload --repository testpypi dist/*
```

## License

MIT &mdash; see [LICENSE](LICENSE).
