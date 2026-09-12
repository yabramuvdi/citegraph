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

Review `sources.csv`. At minimum, spot-check titles, author order, journal, year,
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

Review `citations_raw.csv` and these optional sidecars:

- `references_failures.jsonl`: papers that failed extraction;
- `papers_no_references.json`: successfully processed papers for which Gemini
  returned no references.

An absent warning sidecar means that check was clean.

### Canonicalize works and build edges

```bash
citegraph dedup --out "$OUT_DIR"
```

This clusters your source papers and every extracted citation into one
canonical **works** table and produces:

- `works.csv`: canonical works — your own papers at `ring` 0 (the *core*),
  works discovered in their bibliographies at `ring` 1; `source_file` is
  non-empty for processed PDFs;
- `citation_graph.csv`: work-to-work edges. A citation of one of your own
  papers resolves to that same work, so core-cites-core edges appear here.

Canonicalization is precision-oriented but still heuristic. Review works with
very similar titles, inconsistent years, or large citation counts before
relying on them for publication-quality statistics.

### Optionally enrich works

```bash
citegraph enrich --out "$OUT_DIR" \
  --enrich-contact "you@example.com" \
  --enrich-max-workers 2
```

Enrichment runs over every work — your core papers included, which is how
they gain DOIs and external author identifiers. It tries CrossRef first and
OpenAlex second, and writes
`enriched_works.csv`, `enrichment_summary.json`, and—when misses occur—
`enrichment_misses.csv`. A match is evidence to review, not a guarantee: inspect
the candidate title, raw title score, adjusted score, and year diagnostics.

### Normalize authors

```bash
citegraph authors --out "$OUT_DIR"
```

This produces `authors.csv` and `author_citations.csv`. External OpenAlex and
ORCID identifiers are treated as stronger identity evidence than name strings.
If `author_review.json` exists, inspect the flagged clusters.

By default clustering is precision-first (`--merge-mode strict`): an
initial-only name merges into a full-name cluster only when the match is
unambiguous. `--merge-mode loose` uses surname and first initial as positive
evidence, but still respects contradictory full names, identifiers, distinct
same-work positions, corporate/person separation, and manual separation rules.

For a manual correction, create `author_aliases.csv` with
`cluster_id,canonical_id` columns, then rerun `citegraph authors` (or pass an
external file via `--aliases`). Keep this file with the corpus; it records a
research decision.

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

- `works.csv` contains every canonical work; filter `ring == 0` for the PDFs
  being analyzed (your core corpus) — e.g. in R:
  `works %>% filter(ring == 0)`.
- `citations_raw.csv` contains citation occurrences before canonicalization.
- `citation_graph.csv` connects work IDs (`w-…`) to work IDs — including your
  own papers citing each other.
- `author_citations.csv` records author-work occurrences; it is not a
  coauthorship graph.

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

## Canonicalization reliability and migration

`sources.csv` now retains every successful PDF observation, including duplicate
PDFs. `source_ids.json` reserves provisional IDs by source filename and complete
metadata fingerprint. Keep it with the corpus: unchanged observations retain
allocations when files are added or inputs are reordered. Changed metadata may
allocate a new ID. A fresh registry after corpus growth can assign different
numeric suffixes from an incremental registry. Canonical works still prefer a
source representative; genuinely distinct works with colliding slugs receive
suffixes rather than disappearing.

`work_id_collisions.json` also retains historical collisions among canonical
works, including cited works. Keep this file with `source_ids.json`: removing
one collider later must not make an old enrichment identity trustworthy again.
Legacy enrichment without input fingerprints is ignored for these IDs.

Work matching normalizes accents and author lists before candidate lookup and
scoring. Negation changes, differing part numbers, and substantive title
expansions remain separate for review. An explicit omitted subtitle can still
match when there is no protected title conflict. Missing years remain unknown;
malformed nonempty years reject a match. Similar titles and initials are evidence,
not established identity. Work clustering still uses the first representative,
so changing input order can change a partition in borderline cases.

Author occurrences always come from extracted canonical works. Enrichment adds
one-to-one identity evidence; a provider-only extra author does not become an
extracted occurrence. Review includes unmatched or ambiguous enrichment names,
contradictory full names or identifiers, and duplicate same-work identity claims.
Initials cannot bridge conflicting full names. Same-name coauthors remain
separate without direct consistent identity evidence. Inspect `author_review.json`
and `author_resolution_audit.json` before interpreting author rankings.
An ORCID-free identity chain connected to conflicting ORCIDs stays unresolved
instead of selecting whichever anchor appears first. Corporate/person conflicts
also block automatic merging even when the provider assigned the same identifier.

`canonicalization_audit.json` and `author_resolution_audit.json` preserve actual
stage assignments, reasons, effective settings, and input/output fingerprints.
The report reads this saved evidence. Missing legacy audits are unavailable;
changed snapshots are stale. Neither status causes the report to reconstruct
historical decisions using current defaults. Preserve these files with the CSVs.

### Correct an author merge or split

For new corrections, put occurrence coordinates in `author_overrides.csv`:

```csv
action,left_record_id,left_position,right_record_id,right_position,reason
separate,w-1,0,w-2,0,Distinct people verified from the papers
merge,w-1,0,w-3,0,Same person verified from the papers
```

Positions are zero-based within the extracted `Authors_List`. Use actual work
IDs from your snapshot, then run `citegraph authors --out "$OUT_DIR"`. Forced
merge components are applied before automatic name joins; separation constraints
apply to all joins, including legacy aliases and external-ID joins. A forced
merge can override name evidence but cannot override conflicting ORCIDs or an
explicit separation. Contradictions, invalid positions, and stale legacy alias
targets or cycles fail with a diagnostic rather than silently changing membership.

The first successful validation binds corrections to the works snapshot in
`author_overrides_meta.json`. If that snapshot changes, review every referenced
occurrence against the new works table. Only after that review, remove the binding
file and rerun authors to bind the reviewed correction file. Keep the old binding
and corrections in your research archive. Existing `author_aliases.csv` files
remain supported, but cannot express a split and are subject to conflict checks.

### Rebuild an existing corpus in a copy

Use a new output directory. The following commands preserve the original and
reuse per-stem extraction caches. Replace the first two paths before running:

```bash
export CITEGRAPH_ORIGINAL_OUT="/absolute/path/to/existing-out"
export CITEGRAPH_MIGRATION_OUT="/private/tmp/citegraph-migration-review"
test -d "$CITEGRAPH_ORIGINAL_OUT" || exit 1
test ! -e "$CITEGRAPH_MIGRATION_OUT" || exit 1
cp -a "$CITEGRAPH_ORIGINAL_OUT" "$CITEGRAPH_MIGRATION_OUT"
python - <<'PY'
import json
import os
from pathlib import Path
from citegraph.schemas import PaperMetadata, Reference
out = Path(os.environ['CITEGRAPH_MIGRATION_OUT'])
problems = []
markdown = sorted((out / 'markdown').glob('*.md'))
if not markdown:
    problems.append('No cached markdown; restore it before rebuilding')
for md in markdown:
    for stage in ('metadata', 'references'):
        cache = out / stage / f'{md.stem}.json'
        try:
            value = json.loads(cache.read_text())
            if stage == 'metadata':
                PaperMetadata.model_validate(value)
            else:
                if not isinstance(value, list):
                    raise ValueError('expected a reference list')
                for item in value:
                    Reference.model_validate(item)
        except Exception as exc:
            problems.append(f'{cache}: {exc}')
print('\n'.join(problems) if problems else 'All per-stem extraction caches validate.')
print('Enrichment may need provider refresh after IDs or fingerprints change; '
      'the offline rebuild below does not run enrichment.')
if problems:
    raise SystemExit('Stop: missing/invalid caches can trigger paid extraction. '
                     'Review citegraph estimate before authorizing provider work.')
PY
```

Continue only if this preflight succeeds. A failed preflight requires restoring
caches or explicitly authorizing extraction after `citegraph estimate --out
"$CITEGRAPH_MIGRATION_OUT"`. Missing reference caches can trigger Gemini calls;
metadata itself does not prompt. A scripted reference run uses `--yes` only after
that work is authorized. When the validated caches are all present:

```bash
citegraph metadata --out "$CITEGRAPH_MIGRATION_OUT"
citegraph references --out "$CITEGRAPH_MIGRATION_OUT" --yes
citegraph dedup --out "$CITEGRAPH_MIGRATION_OUT"
```

Before author normalization, set aside the copied enrichment table for comparison
so the initial review is explicitly based on extracted names:

```bash
python - <<'PY'
import os
from pathlib import Path
out = Path(os.environ['CITEGRAPH_MIGRATION_OUT'])
old = out / 'enriched_works.csv'
backup = out / 'enriched_works.before-migration.csv'
if old.exists():
    if backup.exists():
        raise SystemExit(f'Preserve the existing backup before continuing: {backup}')
    old.rename(backup)
PY
citegraph authors --out "$CITEGRAPH_MIGRATION_OUT"
citegraph report --out "$CITEGRAPH_MIGRATION_OUT"
```

Existing corrections can stop this run because IDs or bindings changed. Resolve
those diagnostics by reviewing the copied correction files, not by editing
generated author tables. Preserve `source_ids.json`; do not delete it to clear an
error. Compare observation counts, graph endpoints, source membership, author
memberships, citation counts, and top rankings with the original before adoption.

For enrichment, changed input fingerprints invalidate cached provider responses;
collision-affected legacy entries also need refresh. Unaffected compatible caches
remain reusable. There is no promise that rebuilding enrichment is free: inspect
changed IDs and available caches, contact/rate-limit settings, and provider
budgets before authorizing `citegraph enrich`. After authorized enrichment, rerun
`authors` and `report` and compare rankings again. Keep the original corpus until
both the migration and any provider refresh have been reviewed.

### Offline evaluation and human review

```bash
python scripts/evaluate_canonicalization.py \
  --fixtures tests/fixtures/canonicalization \
  --out /private/tmp/citegraph-evaluation.json
python scripts/evaluate_canonicalization.py \
  --corpus-out "$CITEGRAPH_MIGRATION_OUT" \
  --sample-out /private/tmp/citegraph-human-review.json
```

The fixture evaluator reports TP/FP/FN/TN, precision, recall, excluded ambiguity,
work candidate recall, and review volume separately. Undefined ratios are null.
It also checks identical-input repeats, CSV roundtrips, and reversed-input
partitions, while preserving the assignments and work citation counts for
inspection. Author candidate recall is unavailable rather than inferred from
final merges. These hand-labeled synthetic regressions are not corpus accuracy.

Sampling never invokes providers or assigns labels. It targets 150 work and 150
author observation groups (all available when smaller), balancing review strata
such as common surnames, initials, compounds, identifier conflicts, similar
titles, and current or historical ID collisions. When the work audit matches the
current files, each work group includes its source and raw-citation members,
including variants absorbed by canonicalization. If `work_audit_status` is not
`current`, rebuild the dedup audit before assessing false merges. Each group
contains its raw context; provider
context is observational and may itself be stale. A deterministic fifth is marked
`held_out` before labeling, and the output refuses overwrite to protect the split.
Keep held-out labels out of tuning. Reviewers must locate independently known
duplicates outside candidate groups to measure blocking omissions, and record
`same`, `different`, or `ambiguous` pair labels using observation IDs. Neither
unlabeled samples nor candidate-only labels establish recall. Representative
corpus accuracy remains unknown until that human evaluation is complete.
