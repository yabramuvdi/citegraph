# citegraph User Guide

This guide walks a researcher from a folder of PDFs to a checked, queryable
citation graph. It also explains which decisions deserve human review and what
data leaves the computer during a run.

## 1. Install

`citegraph` requires Python 3.11 or newer. Upgrade `pip` before installing the
PDF extra. Docling has a large dependency tree, and resolving a compatible set
can still take several minutes with a current installer.

```bash
python -m pip install --upgrade pip
python -m pip install "citegraph[pdf]"
```

Use the complete extra if you also want CrossRef and OpenAlex enrichment:

```bash
python -m pip install "citegraph[all]"
```

Set a Gemini API key in the environment or in a `.env` file in the directory
where you run the command:

```bash
export GOOGLE_API_KEY="your-key-here"
```

Keep `.env` out of version control. The repository's `.gitignore` already does
this for a source checkout.

## 2. Prepare a corpus

Put the source PDFs in one directory. A flat directory is simplest. If papers
are organized into subdirectories, pass `--recursive`; hidden directories are
ignored. When recursive mode is enabled, cache filenames include the relative
path so same-named PDFs in different directories do not overwrite one another.

Use a new output directory for each corpus or materially different experiment:

```bash
export PDF_DIR="/path/to/pdfs"
export OUT_DIR="/path/to/citegraph-output"
```

The output directory is a resumable workspace. Do not treat it as disposable
until the analysis has been reviewed and archived.

## 3. Run in stages

The staged workflow gives you a quality-control checkpoint before each
expensive or interpretive step.

### Convert PDFs locally

```bash
citegraph convert "$PDF_DIR" --out "$OUT_DIR" --recursive --ocr-auto
```

Docling performs conversion on the local machine. With `--ocr-auto`, citegraph
first converts normally and retries outputs that look image-only with OCR.

Before continuing:

1. Open a few files under `markdown/`, including the longest and shortest.
2. Check headings, author names, tables, and the bibliography boundary.
3. If `conversion_warnings.json` exists, inspect every listed paper.

Use `--ocr` instead of `--ocr-auto` only when every PDF should receive
full-page OCR.

### Estimate Gemini usage

```bash
citegraph estimate --out "$OUT_DIR"
```

This command makes no API calls. It estimates input tokens, output tokens, and
cost for uncached metadata and reference extraction. The estimate is useful for
approval and comparison, not billing reconciliation.

### Extract paper metadata

```bash
citegraph metadata --out "$OUT_DIR" --llm-concurrency 4
```

Review `papers.csv`. At minimum, spot-check titles, author order, journal, year,
and `source_file`. If present, also inspect:

- `metadata_failures.jsonl`: papers that raised an error and will be retried on
  the next run;
- `source_duplicates.json`: differently named PDFs that appear to contain the
  same source paper.

### Extract references

```bash
citegraph references --out "$OUT_DIR" --llm-concurrency 4
```

The command prints a fresh estimate and asks for confirmation before making
Gemini calls. Pass `--yes` only in automation where cost approval has already
happened.

Review `references_raw.csv` and these optional sidecars:

- `references_failures.jsonl`: papers that failed extraction;
- `papers_no_references.json`: successfully processed papers for which Gemini
  returned no references.

An absent warning sidecar means that check was clean.

### Deduplicate references and build edges

```bash
citegraph dedup --out "$OUT_DIR"
```

This produces:

- `references.csv`: canonical cited works;
- `citation_graph.csv`: source-paper-to-reference edges.

Deduplication is precision-oriented but still heuristic. Review works with very
similar titles, inconsistent years, or large citation counts before relying on
them for publication-quality statistics.

### Optionally enrich references

```bash
citegraph enrich --out "$OUT_DIR" \
  --enrich-contact "you@example.com" \
  --enrich-max-workers 2
```

Enrichment tries CrossRef first and OpenAlex second. It writes
`enriched_references.csv`, `enrichment_summary.json`, and—when misses occur—
`enrichment_misses.csv`. A match is evidence to review, not a guarantee: inspect
the candidate title, raw title score, adjusted score, and year diagnostics.

### Normalize authors

```bash
citegraph authors --out "$OUT_DIR"
```

This produces `authors.csv` and `author_citations.csv`. External OpenAlex and
ORCID identifiers are treated as stronger identity evidence than name strings.
If `author_review.json` exists, inspect the flagged clusters.

For a manual correction, create `author_aliases.csv` with
`cluster_id,canonical_id` columns, then rerun `citegraph authors`. Keep this file
with the corpus; it records a research decision.

## 4. Check status and preserve provenance

```bash
citegraph status --out "$OUT_DIR"
```

For a full `citegraph run`, `run_summary.json` records row and warning counts,
while `artifact_manifest.json` records the package version, stage, artifact
paths, and relevant configuration. Preserve both with the CSVs and source PDFs.

For reproducible analysis, archive:

- the exact input corpus or an immutable reference to it;
- the entire output directory, including caches and review sidecars;
- the citegraph version and Gemini model;
- any `author_aliases.csv` decisions;
- the analysis code or notebook used after extraction.

## 5. Explore results

```python
from citegraph import CitationGraph

graph = CitationGraph.from_out_dir("/path/to/citegraph-output")

print(graph)
print(graph.top_cited(10)[["Title", "Year", "citation_count"]])

paper_id = graph.papers.iloc[0]["id"]
print(graph.cited_by(paper_id))

if graph.has_authors:
    matches = graph.find_author("cardenas")
    if not matches.empty:
        author_id = matches.index[0]
        print(graph.citing_papers_by_author(author_id))
        print(graph.source_journals_citing_author(author_id))
```

`CitationGraph` is a read-only view. It does not modify the reviewed CSV
artifacts. `to_networkx()` is available when NetworkX is installed separately.

## 6. Interpret the outputs carefully

The graph represents citations within the supplied corpus—not global citation
counts. A reference with `citation_count == 5` was cited by five source papers
in this corpus.

Important distinctions:

- `papers.csv` contains the PDFs being analyzed.
- `references_raw.csv` contains citation occurrences before deduplication.
- `references.csv` contains canonical cited works after deduplication.
- `citation_graph.csv` connects source paper IDs (`p-…`) to reference IDs
  (`r-…`).
- `author_citations.csv` records author occurrences and back-pointers; it is not
  a coauthorship graph.

Automated extraction, fuzzy deduplication, external matching, and author
clustering can all introduce error. Report manual review rules alongside
quantitative findings derived from the graph.

## 7. Privacy and external services

The stages have different data boundaries:

| Stage | Service | Data sent externally |
| --- | --- | --- |
| PDF conversion | Docling, local | None by citegraph; Docling may download model files |
| Metadata extraction | Google Gemini | A bounded portion of converted paper text |
| Reference extraction | Google Gemini | The detected bibliography section, or full markdown when no section is found |
| Deduplication | Local | None |
| Enrichment | CrossRef and OpenAlex | Extracted titles, authors, and query parameters |
| Author normalization | Local | None |

Do not process confidential, embargoed, personally sensitive, or
license-restricted documents until their disclosure to the configured external
services has been approved. Review the providers' current retention and data
use terms for the account and region you use.

The API key is a secret. Do not place it in notebooks, CSV files, output
directories, screenshots, or committed configuration.

## 8. Troubleshooting

| Symptom | Likely cause | Action |
| --- | --- | --- |
| Installing `[pdf]` or `[all]` spends a long time resolving | Docling's large dependency graph and citegraph's CLI compatibility pins | Upgrade pip, allow several minutes, and retry in a new virtual environment if resolution was interrupted |
| `PDF conversion requires the 'pdf' extra` | Base package installed without Docling | Install `citegraph[pdf]` or `citegraph[all]` |
| Gemini authentication error | Missing or invalid `GOOGLE_API_KEY` | Set the environment variable or correct `.env` |
| `conversion_warnings.json` exists | PDF produced too little substantive text | Inspect markdown and rerun with `--ocr-auto` or `--ocr` |
| A paper has no extracted references | Image-only input, unusual bibliography heading, or empty model response | Inspect markdown and `papers_no_references.json`; retry after correcting conversion |
| One paper fails but the run continues | Per-paper failure isolation is working | Inspect the relevant failures JSONL and rerun; failed items are not cached |
| Reference count looks too low | Bibliography conversion or model output may be incomplete | Inspect markdown and logs for the inline-citation undercount warning |
| Many enrichment misses | External records differ or thresholds are strict | Inspect `enrichment_misses.csv`; tune thresholds only with manual validation |
| Author clusters are ambiguous | Initial-only names lack disambiguating evidence | Inspect `author_review.json` and record decisions in `author_aliases.csv` |
| `StageNotReadyError` | Required upstream artifact is missing | Run the stage named in the exception hint or point `--out` at the correct workspace |

Re-running a stage is normally safe: successful per-paper caches are reused and
failed items are retried.
