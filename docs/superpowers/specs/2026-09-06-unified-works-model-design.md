# Unified works model — design

**Date:** 2026-09-06
**Status:** approved design, pre-implementation
**Owner:** yabra

## Motivation

citegraph currently conflates two orthogonal properties of a bibliographic
record inside a single "papers vs references" split:

1. *Provenance* — did the user seed this record (their own PDFs, the "core"
   corpus) or was it discovered inside a bibliography?
2. *Full text* — do we have a processed PDF for it?

Consequences of the conflation:

- **Core papers are second-class.** Enrichment (CrossRef/OpenAlex) runs only
  over references, so source papers never get DOIs, canonical metadata, or
  external author ids — which also weakens author clustering exactly for the
  authors the user cares most about.
- **Corpus-internal citations are invisible.** When core paper A cites core
  paper B, B appears as an unlinked `r-` reference. In citation-dense corpora
  (e.g. the Cárdenas-school LFE literature) this hides a large fraction of the
  interesting structure: who among the core papers cites whom.
- **No natural place to ask core-only questions** ("most prominent authors on
  *my* 94 papers") without ad-hoc pandas over two tables.
- **Snowballing has no model.** Processing PDFs of important references later
  would create records that are simultaneously "papers" and "references".

## Goals

1. A single stored concept — the **work** — with provenance (`ring`) and full
   text (`source_file`) as independent attributes. Core = ring 0.
2. Core papers flow through the *same* quality machinery as everything else:
   dedup/canonicalization, enrichment, author normalization.
3. Citations of core papers resolve to the core work itself (core→core edges).
4. First-class query API for core-restricted analysis.
5. Snowballing becomes a data question (works gaining `source_file`, rings > 1),
   not a schema change.

## Non-goals

- No change to the LLM layer: `PaperMetadata` / `Reference` Pydantic schemas,
  prompts, `llm.py`, and the per-stem `metadata/<stem>.json` /
  `references/<stem>.json` caches stay byte-compatible.
- No backward-compatible output format. Pre-1.0 clean break was explicitly
  chosen; legacy `papers.csv`/`references.csv` are not emitted or aliased.
- No snowball *ingestion workflow* yet (no `--ring` flag, no reference-PDF
  matching UX). The schema supports it; building it is a future project.
- `research/` scripts remain untouched.

## Concepts

| Term | Meaning |
| ---- | ------- |
| **work** | Any canonical bibliographic record. The only entity. |
| **ring** | Discovery depth. 0 = seeded by the user (**core**); n = first discovered in the bibliography of a ring n−1 work. Stored per work, computed at canonicalization. |
| **source** | A processed PDF. A work with non-empty `source_file` has full text. Today: ring 0 ⇔ has source; after snowballing, ring ≥ 1 works may gain sources. |
| **core** | Shorthand for ring 0. User-facing vocabulary in API, CLI text, report. |

## On-disk artifacts (owned by `OutLayout`)

Unchanged: `markdown/`, `metadata/`, `references/`, `enrichment/` (cache
dirs); all warning sidecars keep their names and the "absent = clean"
convention; `author_aliases.csv`, `author_review.json`, `report.html`,
`artifact_manifest.json`.

| Old | New | Notes |
| --- | --- | ----- |
| `papers.csv` | `sources.csv` | Stage-2 checkpoint. One row per successfully processed markdown: `id` (`w-…`), `source_file`, `Title`, `Authors_List`, `Authors`, `Journal`, `Year`. Pre-canonical (duplicate PDFs may both appear; resolved at stage 4). |
| `references_raw.csv` | `citations_raw.csv` | Stage-3 checkpoint. One row per raw citation occurrence: the `Reference` fields + `citing_id` (a `w-` id from `sources.csv`). |
| `references.csv` | `works.csv` | Canonical output, indexed by `id`. Columns: `ring`, `source_file` (empty for stubs), `Title`, `Authors`, `Authors_List`, `Journal`, `Year`. |
| `citation_graph.csv` | `citation_graph.csv` | Same name and columns (`citing_id`, `cited_id`) — both ends are now work ids; core→core edges exist. |
| `enriched_references.csv` | `enriched_works.csv` | Same enrichment columns (doi, canonical metadata, `OpenAlex_Authors`, diagnostics), over all works. |
| `enrichment_misses.csv`, `enrichment_summary.json` | unchanged names | Summary gains a per-ring breakdown. |
| `authors.csv` | `authors.csv` | Metric columns change — see Authors stage. |
| `author_citations.csv` | `author_citations.csv` | `record_id` → work id; `record_kind` column dropped (derivable via `works.ring`); `citing_paper_id` column dropped (derivable via edges). |

`run_summary.json` counters become works-based: `n_works`, `n_core_works`,
`n_citations_raw`, `n_edges`, `n_core_to_core_edges`, `n_self_loops_dropped`,
plus the existing failure/warning counters unchanged.

`PipelineResult` becomes `works` / `graph` / `authors` / `author_citations`
(the `papers`/`references` attributes are removed).

## Stable ids

- Single prefix: `w-<first-author-surname>-<year>-<title-slug>`, produced by
  the existing recipe in `ids.py` (`make_paper_id` generalizes to
  `make_work_id`; `make_reference_id` is removed).
- Rationale: the prefix can no longer encode role, because a work's role
  changes over its life (cited stub → processed source) while its identity
  must not. Determinism guarantees are unchanged: same content ⇒ same id
  across runs; the first row of a cluster names the cluster.
- Author ids (`a-…`) and their generation are unchanged; existing
  `author_aliases.csv` files remain valid.

## Canonicalization (stage 4, CLI name stays `dedup`)

Input: `sources.csv` + `citations_raw.csv`. Output: `works.csv` +
`citation_graph.csv`. One clustering pass built on the existing blocked
`compare_papers` machinery in `dedup.py`, with explicit ordering:

1. **Seed source clusters first.** Every row of `sources.csv` starts (or
   joins) a cluster before any citation row is considered. Two sources that
   match each other are duplicate PDFs → merged into one work (first id wins,
   deterministically by input order — same convention as today).
   `source_duplicates.json` is still written by stage 2 as the early warning;
   stage 4 now actually resolves the duplicates instead of only warning.
2. **Citation rows match source clusters before reference clusters.** A
   citation matching a source work merges into it — this is what creates
   core→core edges. Cluster-representative metadata always prefers the
   full-text side (source metadata beats citation strings). Otherwise the
   single-pass first-match-wins clustering behaves exactly as today,
   including the year-window semantics and fail-closed year parsing.
3. **Edges.** `citing_id` = canonical id of the citing source;
   `cited_id` = canonical id of the cited cluster; de-duplicated. Self-loops
   (a work citing a variant of itself, e.g. its own preprint, which now
   clusters into it) are dropped and counted in `run_summary.json`.
4. **Rings.** BFS over the final edges: works with a `source_file` seeded in
   this corpus are ring 0; every other work gets
   `min(ring of its citers) + 1`. A discovered work that merged into a
   ring-0 cluster is simply that ring-0 work. Unreachable works cannot occur
   (every stub arrives via a citation edge). Today this yields only rings
   0 and 1; the computation is already general for future snowballing.

`compare_papers`, blocking, `DedupConfig`, and all thresholds are unchanged.

## Enrichment (stage 5)

- `Pipeline.maybe_enrich` runs the existing generic `enrich_references`
  function (renamed `enrich_works`) over **all of `works.csv`** →
  `enriched_works.csv`. No ring special-casing; one code path.
- Per-item cache compatibility: on a cache miss for `w-<slug>.json`, the
  loader falls back to the legacy `r-<slug>.json` filename (identical slug
  body, different prefix) so existing corpora do not re-crawl
  CrossRef/OpenAlex. Newly enriched core works are simply new cache entries.
- All match diagnostics, retry behavior, `EnrichConfig`, and the miss-review
  sidecars are unchanged. Do not drop the `OpenAlex_Authors` column — it is
  the author stage's best disambiguation signal, and it now covers core
  authors too.

## Authors stage (stage 4b)

- Input collapses from (references, papers, enriched_references, edges) to
  (works, enriched_works, edges). One occurrence per (author, canonical
  work); the parse/block/anchor clustering algorithm, OpenAlex/ORCID
  override rules, alias overrides, and review flags are all unchanged.
- `AuthorOccurrence.record_kind` / `citing_paper_id` are removed; consumers
  derive them from `works.ring` and the edge table.
- `authors.csv` metric columns:
  - `n_works` — canonical works this author appears on (any ring),
  - `n_core_works` — ring-0 works authored (the "prominent core author" measure),
  - `n_citations_received` — edges into any of their works,
  - `n_distinct_citing_works` — distinct citing works over all their works.
  Default sort: `n_citations_received` descending (same spirit as today).
- Because core works are now enriched, core authors gain OpenAlex/ORCID
  anchors, improving cluster precision where it matters most.

## Query API (`CitationGraph` in `graph.py`)

- Constructor: `CitationGraph(works, edges, authors=None, author_citations=None)`.
  `from_out_dir` loads `works.csv` (+ author tables when present); when only
  legacy `papers.csv`/`references.csv` exist it raises `FileNotFoundError`
  with the exact migration command (see Migration). `from_pipeline_result`
  mirrors the new `PipelineResult`.
- New idiom:
  - `g.core` (property) — DataFrame of ring-0 works,
  - `g.ring(n)` — works at ring n,
  - `g.top_authors(n=20, ring=None)` — authors ranked by distinct works
    authored, optionally restricted to a ring (`ring=0` answers "most
    prominent authors of my papers"),
  - `g.core_citations()` — edges where both ends are ring 0.
- Kept with unchanged names/semantics (now naturally spanning all works):
  `n_edges`, `top_cited`, `cited_by`, `citers_of`, `top_cited_authors`,
  `find_author`, `citations_of`, `papers_citing_author`,
  `citation_context_for_author`, `citing_papers_by_author`,
  `source_journals_citing_author`, `to_networkx`, `has_authors` and the
  actionable `RuntimeError` on missing author tables.
- Renamed/removed: `n_papers` → `n_core_works`, `n_references` → `n_works`;
  the `papers`/`references` DataFrame attributes are removed (clean break,
  no deprecated aliases). `works` is index-by-`id` (the `references.csv`
  convention wins; the old papers-side id-as-column asymmetry disappears).

## CLI, report, webui

- CLI command names are unchanged (`run`, `convert`, `metadata`,
  `references`, `dedup`, `enrich`, `authors`, `estimate`, `status`,
  `report`, `ui`) — they name actions. Help text, printed artifact paths,
  and `status` learn the new filenames; all existing knobs/flags unchanged.
- `html_report.py` / `webui.py`: `collect_report_data` reads works-based
  artifacts; report gains a per-ring count summary and a core→core citation
  table. Per-file fault tolerance is unchanged; an out_dir containing only
  legacy artifacts renders a "legacy artifacts — re-run `citegraph dedup`
  onward" finding instead of crashing.
- Cost estimation reads only `markdown/` + per-stem caches — unaffected.

## Migration of existing out_dirs

Stages 1–3 caches are stem-keyed and fully reusable — no docling and no
Gemini calls. For each existing corpus:

```bash
citegraph metadata --out ./out    # rewrites sources.csv from cached JSON
citegraph references --out ./out  # rewrites citations_raw.csv (cached)
citegraph dedup --out ./out       # works.csv + citation_graph.csv
citegraph enrich --out ./out      # enriched_works.csv (legacy cache fallback)
citegraph authors --out ./out
citegraph report --out ./out
```

Stale legacy CSVs (`papers.csv`, `references_raw.csv`, `references.csv`,
`enriched_references.csv`) are left on disk untouched; the docs note they
can be deleted. External consumers (e.g. `lfes_colombia.R`) switch from
reading `papers.csv` to `works.csv` filtered on `ring == 0`.

## Testing strategy

TDD throughout (test first, then implementation), no network in tests, the
existing `_FakeClient` prompt-sniffing pattern unchanged.

New unit coverage:

- citation row resolves into a matching source cluster (core→core edge
  produced; cluster metadata keeps the source side),
- duplicate source PDFs merge into one work; edges from both remap to it;
  warning sidecar still written,
- ring BFS (0/1 today; a synthetic 2-ring fixture for the general case),
- self-loops dropped and counted,
- `w-` id determinism and first-row-names-cluster,
- enrichment cache legacy-filename fallback,
- `top_authors(ring=0)`, `g.core`, `g.core_citations()`,
- authors metrics (`n_core_works`, `n_citations_received`) on a fixture with
  an author who is both core author and cited author,
- `from_out_dir` legacy-artifact error message.

Updated (mechanical): `test_pipeline`, `test_cli`, `test_report`,
`test_webui`, `test_packaging` docs assertions, graph/author/dedup/enrich
test fixtures to the works shape. `CLAUDE.md`, `README.md`,
`docs/USER_GUIDE.md`, `docs/UI_ARCHITECTURE.md` updated to the new model.

## Decision log

| Decision | Choice | Why |
| -------- | ------ | --- |
| Storage compat | Free restructure, no legacy views | User choice (pre-1.0; only known consumers are user-controlled R scripts). |
| Core-cites-core | Linked via unified clustering | User choice; the corpus is citation-dense. |
| Snowballing | Designed for via `ring` | User choice; costs one integer column now. |
| Id prefix | Single `w-`, role-free | Role changes over a work's life; identity must not. |
| `references_raw.csv` name | Renamed `citations_raw.csv` | Rows are citation occurrences, not canonical references. Approved. |
| `g.papers` / `g.references` | Removed, no aliases | Clean break approved; aliases would freeze the old mental model. |
| Self-loops | Dropped + counted | Almost always a preprint/published variant artifact; count keeps it auditable. |
| CLI command names | Unchanged | They describe actions; renaming breaks muscle memory for zero model benefit. |
| LLM schemas & caches | Frozen | The expensive artifacts; nothing in this redesign needs them to change. |
