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
pip install citegraph                  # base install (Gemini + fuzzy dedup)
pip install "citegraph[crossref]"      # + CrossRef / OpenAlex enrichment
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

`Authors_List` is *not* preserved through the dedup stage &mdash; only the derived `Authors` string survives.

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

For graph analysis, export to [`networkx`](https://networkx.org/) (not a
default dependency &mdash; install separately):

```python
nx_graph = g.to_networkx()         # DiGraph with edges citing -> cited
```

### Sidecar files

- `run_summary.json` &mdash; counts (`n_papers`, `n_references_raw`, `n_references_dedup`, `n_edges`, `n_metadata_failures`, `n_references_failures`), the model id, and the dedup configuration used.
- `metadata_failures.jsonl` and `references_failures.jsonl` &mdash; **only created when at least one paper failed**. One JSON object per line: `{source_file, stage, error_class, error_message}`. The pipeline keeps going past per-paper failures; re-running will retry them (their caches were not written).

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
| `enrich` | `False` | run a CrossRef/OpenAlex pass over the deduplicated references (requires the `[crossref]` extra) |
| `enrich_config` | `EnrichConfig()` | tune the enrichment pass &mdash; see below |
| `dedup_config` | `DedupConfig()` | tune the fuzzy dedup &mdash; see below |
| `overwrite_markdown` | `False` | re-run docling even when a cached `.md` exists |
| `recursive` | `False` | walk subdirectories of `pdf_dir`. Cache filenames are disambiguated by relative path (e.g. `journal_X/foo.pdf` → `journal_X__foo.md`); a clear error is raised if two PDFs would still collide. CLI flag: `--recursive` / `-r` |

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
    ),
)
```

The CLI exposes the same knobs as flags &mdash; see `citegraph run --help`.

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
so no API token is stored in the repo. To cut a release:

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
