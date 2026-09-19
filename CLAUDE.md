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
citegraph metadata --out ./out            # stage 2: -> sources.csv
citegraph references --out ./out          # stage 3: -> citations_raw.csv
citegraph dedup --out ./out               # stage 4: -> works.csv, citation_graph.csv
citegraph enrich --out ./out              # stage 5 (optional): CrossRef/OpenAlex -> enriched_works.csv
citegraph authors --out ./out             # stage 4b: -> authors.csv, author_citations.csv
citegraph estimate --out ./out            # pre-flight token/cost estimate (no API calls)
citegraph status --out ./out              # report which artifacts exist
citegraph report --out ./out [--open]     # write self-contained report.html QC dashboard
citegraph export-authors --out ./out      # -> core_authors.xlsx (your papers' authors, one workbook)
citegraph ui --out ./out [--port 8765]    # serve read-only live monitor on 127.0.0.1
```

`citegraph run` and `citegraph references` print a cost estimate and prompt before any LLM call. Pass `--yes`/`-y` to skip the prompt. **Always pass it when running them backgrounded, scripted, or in CI**: with no TTY on stdin the prompt resolves to "no" and the stage exits having done nothing, which in a `&&` chain looks like a silent stall rather than an error. `citegraph metadata` does not prompt.

The library mirrors the CLI: every `Pipeline` stage method accepts its input
explicitly *or* loads it from `out_dir` if called with no arguments. Missing
upstream artifacts raise `StageNotReadyError` with a hint at the prior step.

`GOOGLE_API_KEY` is required for any path that calls Gemini. `OPENALEX_API_KEY` is optional but strongly recommended for stage 5 on a real corpus — see the enrichment stage below for what it buys. Both are read from the environment or `.env` via [config.py](src/citegraph/config.py). Tests avoid the network entirely (see "Testing without network" below).

## Architecture

`citegraph` turns a folder of academic PDFs into a canonical **works** table plus a citation graph (`works.csv`, `citation_graph.csv`). Every bibliographic record is a *work* with a single role-free `w-` id and two orthogonal stored facts: **`ring`** — discovery depth, where ring 0 is the user's own PDFs (the *core*) and ring n was first discovered in a ring n−1 bibliography — and **`source_file`** — non-empty when we processed a PDF for it. Today ring 0 ⇔ has a source file; snowballing later (processing PDFs of important references) is a data change, not a schema change. The whole flow is orchestrated by `Pipeline.run()` in [pipeline.py](src/citegraph/pipeline.py); everything else is a stage it composes.

**Migrating a pre-works-model out_dir** (one that has `papers.csv`/`references.csv`): the per-stem caches are reused, so re-running `citegraph metadata && citegraph references && citegraph dedup && citegraph authors` (plus `enrich` if used — its per-item cache falls back to the legacy `r-` filenames) rebuilds everything without new docling or Gemini calls. Stale legacy CSVs can then be deleted.

**Repairing one badly-converted paper** without reprocessing the corpus: re-convert just that PDF with `convert_pdf_to_markdown(pdf, markdown_dir, overwrite=True, ocr=True, cache_stem=...)` — passing `--ocr` to the CLI would force OCR on *every* PDF and overwrite good markdown with worse. Then delete that stem's `metadata/<stem>.json` and `references/<stem>.json` (the caches are keyed by stem, and a stale cache is what pins the bad result), and re-run `metadata`, `references`, `dedup`, `enrich`, `authors`, `report`. Every other paper is served from cache, so the whole repair costs roughly one Gemini call per extraction stage. Note that the work's `id` changes once real metadata exists, so its enrichment cache entry is written fresh under the new id.

### Canonicalization reliability requirements

The reliability implementation refines the older stage descriptions below:

- Metadata retains every successful source observation and reserves IDs in
  `source_ids.json`. Duplicate PDFs retain their separate reference caches;
  their bibliographies are all processed before canonical work clustering.
- Work assessments distinguish match/review/reject. Protected negation, part
  numbers, and substantive expansions prevent unsafe title-containment merges.
  `work_id_collisions.json` retains canonical collision history, including cited
  works. Both collision registries exclude unverified legacy enrichment IDs.
- Author enrichment attaches one-to-one to extracted author positions. Global
  identity evidence is checked before unions; conflicting ORCIDs, ambiguous
  identity bridges, corporate/person conflicts, and explicit separation cannot
  be bypassed by automatic merges. Loose mode still respects hard conflicts.
  Coauthor evidence belongs to individual occurrences, not signature buckets.
- `author_overrides.csv` uses occurrence coordinates and merge/separate actions;
  `author_overrides_meta.json` binds the reviewed works snapshot. Validate all
  constraints and aliases before binding or changing generated author outputs.
- Versioned work/author audits contain actual assignments and fingerprinted
  inputs/outputs. Reports must not reconstruct historical decisions with defaults;
  missing, invalid, and stale evidence must be identified explicitly.
- Enrichment cache v2 binds original metadata; cold/cache results agree. Author
  occurrences always come from canonical extracted works, even in an enriched run.
- Empty valid stages write schema-bearing CSVs; all-failed metadata blocks
  dependent stages. The offline evaluator and copy-only migration instructions
  are in `docs/USER_GUIDE.md`; synthetic regression metrics are not corpus accuracy.

### Stage pipeline (each stage is checkpointed on disk)

1. **PDFs → markdown** — [pdf_to_markdown.py](src/citegraph/pdf_to_markdown.py) calls `docling`. Output cached as `out_dir/markdown/<stem>.md`. Idempotent: re-runs skip files unless `overwrite_markdown=True`. Docling is imported lazily inside the function so `import citegraph` stays cheap. Pass `recursive=True` (Pipeline kwarg / `--recursive` CLI flag) to walk subdirectories of `pdf_dir`; in that mode cache stems carry the relative path via `cache_stem_for` (e.g. `journal_X/paper.pdf` → `journal_X__paper`) so same-named PDFs in different folders don't clobber each other. Hidden directories (names starting with `.`) are skipped, and a stem-collision pre-check raises a clear `ValueError` before docling is ever invoked. Pass `ocr="auto"` (Pipeline kwarg / `--ocr-auto` CLI flag) to convert normally first and re-run with EasyOCR only the outputs that came out image-only *or* glyph-corrupted; pass `ocr=True` / `--ocr` to force full-page OCR for every PDF. After each conversion pass, `check_conversion_quality` in [reports.py](src/citegraph/reports.py) checks every output markdown and writes `conversion_warnings.json` when any file fails. Two independent detectors, because they fail in opposite directions: `_is_image_only` (strips `<!-- image -->` tags and `#` headers; flags files with fewer than 200 substantive chars) catches scanned PDFs, while `_has_unmappable_glyphs` catches a PDF whose embedded font carries no ToUnicode map — docling emits one `glyph<UNKNOWN>` per unmappable character, so the output is *long* and sails past the image-only check while still yielding empty metadata. The glyph rate is measured against mapped alphanumerics and deliberately kept low (≥1%, with a ≥20-occurrence floor so stray math symbols don't cry wolf): corruption is often concentrated, and a title page in a different font can be fully unmappable while the body reads fine.
2. **markdown → source-work metadata** — [extract_metadata.py](src/citegraph/extract_metadata.py) sends each markdown to Gemini with `PaperMetadata` as the structured-output schema. Cached as `out_dir/metadata/<stem>.json`; assembled into `sources.csv` (one row per processed PDF, `id` as a column, pre-canonical — duplicate PDFs are resolved at stage 4). The per-paper loop runs concurrently, capped by `Pipeline(llm_concurrency=...)` / `--llm-concurrency` / `CITEGRAPH_LLM_CONCURRENCY` (default `4`), while collecting results in input order and preserving per-paper failure isolation. After all papers are processed, `_detect_source_duplicates` in [pipeline.py](src/citegraph/pipeline.py) applies `compare_papers` (from [dedup.py](src/citegraph/dedup.py)) across the source-work records to find differently-named PDFs that contain the same paper. Results are written to `source_duplicates.json` (absent = no duplicates). The references stage reads this file to emit an informative warning instead of a cryptic "no paper id" message when it encounters a duplicate markdown.
3. **markdown → raw citations** — [extract_references.py](src/citegraph/extract_references.py) extracts `list[Reference]` per paper; the rows land in `citations_raw.csv` (one row per raw citation occurrence, with `citing_id`). Cached as `out_dir/references/<stem>.json`. Like metadata, the per-paper loop runs concurrently under `llm_concurrency` while preserving input-order CSV output and per-paper failure isolation. Three non-obvious behaviors wrap the LLM call: (a) before sending, `split_reference_sections` collects *every* bibliography section matched by a strict `##`-style header regex — multi-chapter documents (PhD theses) get one Gemini call per chapter bibliography and the results are concatenated; a non-final section ends at the next same-or-higher-level header, while the final one runs to end-of-file *unless* a same-or-higher-level header opens recognised back matter (`_BACKMATTER_HEADER_RE`: appendices, supporting information, notes, a department's list of previous doctoral theses) — a Gothenburg thesis's series back matter reads exactly like a bibliography and produced 180 phantom works before this stop list existed. The list is deliberately a *denylist* so the splitter fails closed: headers that interrupt a real reference list are common (running page heads like `## 94 ECONOMIA, Spring 2009`, repeated article titles, and in one scanned source the only regex-matching `## REFERENCES` is JSTOR front-matter boilerplate while the real bibliography sits behind an unmatched header), and terminating on those would silently drop references. When no header matches at all the full file is sent so the paper is never dropped; (b) after each response arrives, `response_was_truncated` triggers a one-shot retry at 2× output cap when the bibliography hit `max_output_tokens`, hard-capped at 80k tokens (per section, so long theses don't brush the cap); (c) `estimate_inline_citations` over the body warns at WARNING level when the LLM returned <50% of the references the body appears to cite (gated on ≥10 distinct citations and successful slicing — small bodies and unsliced papers skip the check to avoid noise). After the loop, `_write_no_references` writes `papers_no_references.json` for any paper that was successfully processed (not a failure, not a duplicate) but returned zero references — each entry has `paper_id`, `source_file`, and `title`. Common cause: image-only PDF that was not run with `--ocr-auto` / `--ocr`.
4. **canonicalization → works + edges** — `canonicalize_works` in [dedup.py](src/citegraph/dedup.py) clusters *sources and raw citations together* with a weighted rapidfuzz score over title/authors/journal plus a year window. Sources occupy the head of the combined frame, so the single-pass first-match-wins loop guarantees three things at once: duplicate source PDFs merge into one work, a citation matching a source *becomes* that work (this is what creates core→core edges), and every cluster representative — hence its metadata — is the full-text side whenever one exists. The `compare_papers` function fails *closed* on genuinely unparseable year data (e.g. a non-numeric string) — preserve that behavior. Missing-year sentinels (the schema's `Year=0`, `None`, or empty string) are treated as "year unknown" and do *not* reject the cluster; title+authors fuzzy carries the decision instead. Rings are computed by BFS from the source set over the finished edges (general for future snowballing, though today it yields only 0/1); self-loops (a work citing its own preprint variant, which clusters into it) are dropped and counted in `run_summary.json`. Output: `works.csv` (canonical, indexed by `id`, columns `ring, source_file, Title, Authors, Authors_List, Journal, Journal_Canonical, Year`) and `citation_graph.csv` (`citing_id`, `cited_id` work→work edges).

   `Journal_Canonical` comes from [journals.py](src/citegraph/journals.py) and is what you group on; `Journal` keeps what the record actually said and is never overwritten. The fold is deliberately conservative — case, diacritics, punctuation, `&`/`and`, a leading `The` — because abbreviations **cannot** be matched safely: an initials heuristic over the paper 4 corpus paired `public choice` with `protecting the commons` and `energy policy` with `economics and philosophy`. Abbreviations belong in the hand-curated `journal_aliases.csv` (`raw,canonical`), which is applied last and matched *by folded key*, so one spelling covers every variant of it; an alias matching nothing at all is reported, because a curated row that silently does not apply is worse than no row. **"At all" spans both folds, and that distinction is the whole subtlety.** The corpus spells its journals twice — as the bibliography extracted them and as the provider reports them — so a row is dead only if it matches *neither*: an alias expanding `Ecol. Econ.` matches nothing among provider titles precisely because the provider already says `Ecological Economics`, and checking one fold in isolation rejects the row for doing its job. `unused_aliases(aliases, *vocabularies)` asks the question against every vocabulary at once; `Pipeline._check_journal_aliases` assembles them, taking the other fold from the caller (stage 5 holds both frames) or from `enriched_works.csv` on disk (stage 4 does not). Dead rows always land in `journal_alias_warnings.json`; whether they *also* raise depends on whether the corpus can still grow a vocabulary — with `enrich=True` and no `enriched_works.csv` yet, the provider names are still to come and a row matching nothing yet may be perfectly good, so the sidecar is the whole report, while once every vocabulary is present (enrichment done, or never happening) a dead row is provably a typo and stops the stage. This is why the raise cannot simply be per-frame: stage 4 runs before stage 5, and enrichment is optional. Preprint repositories are folded to `Working paper`: OpenAlex reports RePEc, SSRN and publisher eBook platforms as container titles, which would otherwise rank RePEc third and SSRN ninth among this corpus's "journals" (160 works). The display name for a group is the most frequent spelling, longest breaking a tie — the same rule the author stage uses for display names. The column is computed **twice on purpose**: `works.csv` folds the extracted names so it exists without enrichment, `enriched_works.csv` folds the provider container titles, which are the better basis.
5. **optional enrichment** — `enrich_works` in [enrich.py](src/citegraph/enrich.py) hits CrossRef then OpenAlex over **all canonical works, every ring** — the user's core papers get DOIs and external author ids through the same code path as discovered references. Gated behind `enrich=True` *and* the `[crossref]` extra (httpx is imported via `_try_import_httpx` so the base install stays light). Per-work results cache under `enrichment/<work-id>.json`; a lookup miss for `w-<slug>.json` falls back to the legacy `r-<slug>.json` filename so pre-works-model corpora never re-crawl. Cached misses whose reason is `http_error` are *not* served on re-run — they record a transient outage (rate limit, 5xx, timeout), so the work is retried; only genuine no-match misses are permanent. Always pass a contact email (`--enrich-contact` / `EnrichConfig.contact_email`) for real corpora — anonymous callers get heavily rate-limited by both providers. OpenAlex additionally enforces a daily *credit* budget (free tier $0.10/day ≈ 100 searches at 10 credits each, reset midnight UTC) — a free API key from openalex.org/pricing raises it 10× and unlocks prepaid credits; pass it via `--openalex-api-key`, the `OPENALEX_API_KEY` env var / `.env`, or `EnrichConfig.openalex_api_key`. The key is sent only to OpenAlex (never CrossRef) and is redacted to `***` in `enrichment_summary.json`. Transient provider failures (429, 503, and timeouts) retry with tenacity before becoming `http_error` misses. `_best_match` scores candidates with rapidfuzz title similarity, then subtracts `EnrichConfig.year_mismatch_penalty` when both years are known and disagree; the raw and adjusted scores are both preserved so rejected same-title/different-year candidates can be audited. Author agreement is then a **veto on candidate eligibility, never a term in the score** — the title still decides acceptance, so author evidence can't push a weak title over the threshold. `author_list_agreement` (in [authors.py](src/citegraph/authors.py), so all name-folding lives in one place) compares order-free name tokens rather than `ParsedAuthor.surname_norm`: the surname path needs the corpus compound-surname lexicon, which doesn't exist yet when enrichment runs, so without it every comma-less provider name collapses to its last token and `Moros, L.` stops matching `Lina Moros Canon`. Tokens pair on equality or on `EnrichConfig.author_name_fuzz` similarity when both are ≥5 characters — that tolerance is what lets a bibliography's `Hirschmann, A.` reach `Albert O. Hirschman`, while the length floor keeps `Lee` and `Loe` apart. The check **abstains** when either side names an institution (including a bare acronym like `IFPRI`, which `_is_corporate_author` needs two tokens to catch), because a corporate-authored report's provider record legitimately lists the human chapter authors instead. Candidates are filtered before ranking, so a vetoed best candidate falls through to the next eligible one rather than sinking the lookup; when every candidate is vetoed the work misses with reason `author_mismatch`. This exists because OpenAlex holds book reviews and edited-volume chapters under the reviewed work's title verbatim — *Governing the Commons* matched a Croatian review at title score 100 — and a wrong record overwrites `Journal` and `Year`, not just the author ids. Measured on a 3,277-work corpus: 29 rejections, all OpenAlex-sourced, none in ring 0. Note the veto never re-decides a cached result — the cache stays at `schema_version` 2 and the new `enrichment_author_agreement` diagnostic is optional, so re-deciding a work means deleting its cache entry. The `_normalize_record` step unescapes HTML entities in titles, journals and author names (both providers return them raw — 132 works in the paper 4 corpus had `Journal of Economic Behavior &amp; Organization` as their canonical journal). That only cleans *freshly fetched* records, so `_apply_enrichment_result` repairs entities again on every read, cache hits included — otherwise an already-cached corpus keeps the escaped text until each work is re-crawled, which is a lookup cost for a pure string fix. `_normalize_record` also keeps OpenAlex `author.id` and ORCID alongside `display_name` in a parallel `OpenAlex_Authors` list (one dict per author, positionally aligned with `Authors_List`), plus CrossRef's structured `family`/`given` split (`None` from OpenAlex); the author-normalization stage reads this column from `enriched_works.csv`, treats the identifiers as ground truth for identity, and feeds multi-word `family` values into its compound-surname lexicon. Each enriched row also carries match diagnostics (`enrichment_status`, `enrichment_miss_reason`, `enrichment_title_score`, `enrichment_adjusted_score`, `enrichment_candidate_title`, `enrichment_year_match`, `enrichment_year_delta`), and the stage writes `enrichment_summary.json` (with a per-`ring` breakdown) plus `enrichment_misses.csv` for corpus-level review. Don't drop the column when refactoring enrichment — the author stage degrades gracefully without it but loses its single best disambiguation signal.
6. **author normalization → canonical authors + author-work edges** — [authors.py](src/citegraph/authors.py) clusters every author across the corpus (one occurrence per author per canonical work, core and cited alike) into canonical records. `parse_author` turns one raw string ("Cárdenas, J.-C.", "Adele Diamond", "García Márquez, Gabriel") into a structured `ParsedAuthor` (surname / given-names / initials / suffix) with diacritic-stripped, hyphen≡space `surname_norm` for blocking; it also detects Vancouver-style "Guerra JA" ordering (surname first, initials last), institutional authors ("The World Bank" — kept whole, `is_corporate=True`, never person-clustered), the Spanish `y` surname connector ("Ortega y Gasset"), glued "et al." tails, and OCR-spaced accent marks ("B ' enabou" → "Benabou"). Parsing is evidence-driven and two-pass: `_build_surname_lexicon` collects multi-word surnames attested by trusted forms (comma forms, hyphenated/particle compounds, enrichment `family` fields), then comma-less names re-parse against the lexicon so "Jose Alberto Guerra Forero" adopts the "Guerra Forero" boundary only when the corpus attests it — never from a Spanish-convention guess (uncorroborated cases keep the conservative last-token parse, display the full observed name, and are flagged in `author_review.json` as possible compound surnames). `normalize_authors` blocks by `surname_norm`, then within each block applies a precision-first signature algorithm: records are bucketed by their exact given-name signature (ordered full words + initials, hyphens dissolved so "Juan-Camilo" ≡ "Juan Camilo") and each bucket joins a cluster only when its *whole* signature is compatible with exactly one candidate (or co-author overlap breaks the tie) — so "Cardenas, J.C." reaches "Juan Camilo" but "Cardenas, J.P." never can. Word-vs-word comparison tolerates near-identical extraction typos ("Camillo"/"Camilo": rapidfuzz ≥90 and neither a prefix of the other, keeping Gabriel/Gabriela apart); display names use the fullest, most frequent given sequence observed so typos never name a cluster. OpenAlex / ORCID ids from the enrichment stage override string evidence — same id ⇒ same cluster (even across surname blocks), different ids ⇒ different clusters even when names look identical. Id-bearing occurrences are grouped by the *union* of every id they carry, not by one preferred key: enrichment attaches an OpenAlex id where the work matched OpenAlex but only an ORCID where it matched CrossRef, so keying on `openalex_id or orcid` would split one person in two — and their no-id name variants, being compatible with both halves, would then read as ambiguous and strand in a third cluster. External-id clusters also act as anchors for no-id variants when signatures match unambiguously. After in-block clustering, a bridging pass merges a single-surname cluster into a compound-surname cluster ("Reyes, Sandra" → "Polanía Reyes, Sandra") only on an exact shared full given name (or matching external id) — initials alone never bridge. `merge_mode="loose"` collapses by `(surname, first_initial)` regardless. Hand-curated `cluster_id,canonical_id` overrides in `out_dir/author_aliases.csv` are applied last and let the user fix anything the algorithm gets wrong without rerunning. A second curated file, `out_dir/author_external_ids.csv` (`author_id,openalex_id,orcid,note`), stamps identifiers onto canonical authors *after* identity reconciliation — it keys on the published `a-` id, and a curated value outranks the provider's, because someone who opened the provider record and checked beats a name match that failed. **The two files answer different questions and must stay separate**: aliases decide *who is the same person*, external ids decide *what that person's identifier is*; letting an identifier also force merges would put two curated files in charge of the same decision. Both fail loudly on an id that no longer exists, and one identifier claimed by two authors is an error. This exists because provider data quality can defeat name matching even when the right id is obvious to a human: OpenAlex's canonical `display_name` for Elinor Ostrom is `Элинор Остром`, so `A5003087402` never attached to the cluster built from `Ostrom, E.` despite appearing on 12 of her works. Don't reach for transliteration — across 6,785 provider author mentions in that corpus only 13 were non-Latin, covering two people. Output: `authors.csv` (canonical, indexed by `id`, `a-` prefix; metrics `n_works`, `n_core_works`, `n_citations_received`, `n_distinct_citing_works`) and `author_citations.csv` (one row per occurrence: `author_id`, `record_id` work id, `position`, `raw_author` — roles derive from joining `works.ring` and the edge table).

The cache layout is owned by `OutLayout` in [io.py](src/citegraph/io.py). If you add a new artifact, add it to `OutLayout` rather than hardcoding paths.

`citegraph report` ([html_report.py](src/citegraph/html_report.py)) reads whatever artifacts exist under `out_dir` and writes a fully self-contained `report.html` QC dashboard (no network, every file read individually fault-tolerant — corrupt artifacts become findings, not crashes); it is safe to re-run after any stage. `citegraph ui` ([webui.py](src/citegraph/webui.py)) serves the same `collect_report_data` payload as a stdlib-only live monitor on 127.0.0.1 — strictly read-only over `out_dir`, polling clients re-render as artifacts appear; the `/api/markdown` endpoint only serves bare stems that resolve inside `out_dir/markdown/`.

### One workbook for the authors of your own papers

`citegraph export-authors` ([author_export.py](src/citegraph/author_export.py))
writes `core_authors.xlsx`: every author who appears on a ring-0 work, with
everything the corpus knows about them, on sheets keyed by `author_id`
(`README`, `Authors`, `Works`, `Review_Flags`, then any curated sheet). It
exists because `authors.csv` answers a question nobody asks — it describes
thousands of authors, almost all of them encountered once in somebody else's
bibliography — while the manuscript question is *who wrote my papers, and what
do we know about them?* The answer is spread over half a dozen artifacts and,
for paper 4, over a hand-collected spreadsheet outside `out_dir` entirely.

The core set is derived by joining `author_citations.csv` to the ring-0 rows of
`works.csv`, **not** by reading `authors.csv`'s `n_core_works` column. The two
agree, but the join is what makes `--ring` mean anything, and it keeps working
on an `out_dir` whose metrics column predates a schema change. `Works` keeps a
core author's *cited* works too: an author is interesting partly because the
corpus cites their other work, and restricting the sheet to ring 0 would answer
a question nobody asked.

Curated tables from outside `out_dir` arrive through `--extra-csv NAME=PATH`
(library: `ExtraSheet`), because the package cannot know their schemas. One
filter rule covers every shape they take: **keep a row when any column named
`author_id` or ending in `_author_id` names a core author**. A per-person table
keys on the first; an edge table like `supervision.csv` has
`advisor_author_id` *and* `advisee_author_id`, and a tie is worth keeping when
either end is somebody you published — the advisor is often outside the corpus.
A CSV with no such column raises rather than being added unfiltered.
`--master-column SHEET=COLUMN` lifts a headline field onto the `Authors` sheet
and insists the source have one row per author, because lifting from a tidy
per-position table would silently multiply the master's rows.

Every curated sheet also adds an `in_<sheet>` boolean to `Authors` and a
`Core_authors_covered` cell to `README`. That column is the one to read first:
hand-collected metadata never covers everybody (paper 4: 58 of 146), and the
gap is a finding about the corpus, not a fault in the export. `a-willis-cleve`
— 3 core papers, co-citation degree 418, no institutional row — is exactly what
it is for.

Group authors on **`affiliation_field`**, never on the raw
`affiliation_department`: the department is 31 distinct strings over 58 people,
28 of them singletons, and it spells one place several ways (`Economics` 22 vs
`Department of Economics` 1, plus `Resource Economics` and `Rasmuson Chair of
Economics`). `crosswalk_fields.csv` folds those into the 8 categories
`fig1_affiliation_field` plots, and lifting both columns onto `Authors` lets a
pivot table there reproduce that figure's counts exactly.

`author_centrality.csv` is joined onto `Authors` when present, and is the one
input the pipeline never writes: [examples/paper4_figures.ipynb](examples/paper4_figures.ipynb)
does. So it is read fault-tolerantly and can silently go stale if you re-run
`citegraph authors` without re-running the notebook. Writing `.xlsx` needs
`openpyxl`, a lazily-imported `[excel]` extra, so the base install stays light.

Paper 4's workbook is rebuilt from `Dropbox/Consultoria/Maria/paper4/` with:

```bash
citegraph export-authors --out citegraph_out \
  --extra-csv Institutional=institutional/out/base_institucional_clean.csv \
  --extra-csv Career_Positions=institutional/out/career_positions.csv \
  --extra-csv Supervision=institutional/out/supervision.csv \
  --master-column Institutional=affiliation_institution \
  --master-column Institutional=affiliation_department \
  --master-column Institutional=affiliation_field \
  --master-column Institutional=affiliation_country \
  --master-column Institutional=phd_institution \
  --master-column Institutional=phd_year_end \
  --master-column Institutional=phd_field
```

### `CitationGraph` query view

[graph.py](src/citegraph/graph.py) defines `CitationGraph`, a small read-only wrapper over `works` (indexed by `id`) + `edges` plus the optional author tables. It exists to deliver on the package name — without it, the library produces edges in a CSV but offers nothing to traverse. Two constructors: `CitationGraph.from_out_dir(out_dir)` (loads CSVs; a legacy pre-works out_dir raises `FileNotFoundError` with the exact migration commands) and `CitationGraph.from_pipeline_result(result)`. Core idiom: `g.core` (ring-0 works — the user's papers), `g.ring(n)`, `g.core_citations()` (edges with both ends in the core), `n_core_works` / `n_works` / `n_edges`, `top_cited(n)` (core works cited within the corpus rank here too), `cited_by(work_id)`, `citers_of(work_id)`, `to_networkx()` (node attrs carry `ring`/`source_file`). `networkx` is a *lazy* import so the base install stays light (`pip install "citegraph[graph]"`); the import-error message tells the user how to install.

When `authors.csv` and `author_citations.csv` exist in `out_dir`, `from_out_dir` loads them too and `has_authors` becomes `True`, unlocking `top_authors(n, ring=None)` (authors ranked by distinct works authored; `ring=0` answers "who are the most prominent authors of *my* papers?"), `top_cited_authors(n)` (ranked by `n_citations_received`), `find_author(query)` (diacritic-insensitive substring match on surname or display name), `citations_of(author_id)` (every cited work the author appears on), `papers_citing_author(author_id)`, `citation_context_for_author(author_id)`, `citing_papers_by_author(author_id)`, and `source_journals_citing_author(author_id)`. The last three join through `citation_graph.csv` so citing-side journal rollups answer "which journals are doing the citing?" rather than "where were the cited works published?" These methods raise `RuntimeError` with an actionable hint when called on a graph whose author tables aren't loaded, rather than silently returning empty results.

`author_cocitation_network(min_papers=3, citing_ids=None)` is the one method that *builds* a network for analysis rather than exporting one. It returns the weighted, undirected author co-citation projection (White & Griffith 1981): two authors are joined when the same bibliography cites both, and `weight` counts how many bibliographies do. It exists because the work-level graph is one hop deep — only PDF-backed works have outgoing edges, so betweenness/closeness/PageRank over `to_networkx()` would measure the sampling design, not the literature. The projection is a real one-mode network where Freeman/Bonacich centralities are defined. A citing work that cites several works by one author contributes one mention and never a self-loop; `min_papers` prunes the once-cited tail; `citing_ids` restricts the citing side, which is how the self-citation robustness variant is built. Nodes are added in sorted order so seeded layouts are reproducible — don't "simplify" that away. **That guarantee stops at the graph this method returns**: `Graph.subgraph(nodes)` keeps its node filter in a *set*, so any subgraph or `.copy()` of it is ordered by string hashing and therefore varies with `PYTHONHASHSEED` between processes. `nx.spring_layout` seeds initial positions *by node index*, so a fixed `seed=` on a subgraph still redraws differently on every run — this is exactly how the paper 4 ego-network figure became non-reproducible while claiming a fixed seed in its own caption. Anything that draws a *subset* of this network must re-establish the order itself (`add_nodes_from(sorted(...))`, and `add_edges_from(sorted(...))` for paint order); the same trap applies to `sorted(components, key=len)`, whose stable sort leaves same-size components in hash order unless the key carries a name tiebreak. Verify by executing the notebook twice in separate processes and comparing the PNGs byte for byte. Used by section 6 of [examples/paper4_figures.ipynb](examples/paper4_figures.ipynb), which computes the centralities themselves in the notebook so the measure choices stay visible for a methods section.

The projection's atom — one row per *bibliography that cites an author* — is
`author_mentions(edges, author_citations, ...)`, a module-level function in the
same file. It exists because two callers derive different structures from the
same three operations (drop duplicate edges, join the author table, drop
duplicate mentions), and a second implementation of them would drift: the
network groups mentions by citing work to build edges, while the tie test below
needs each author's *set* of citing bibliographies.

### Hand-curated annotations join at read time

[annotations.py](src/citegraph/annotations.py) carries facts a human — or an
agent — recorded about a work rather than the pipeline: the theme a reviewer
assigned it, what kind of publication it is. They cannot live in `works.csv`,
which stage 4 regenerates, so a column typed in there survives exactly until the
next `citegraph dedup`. They live beside it in `work_annotations.csv`, keyed on
the work `id`, and `CitationGraph.from_out_dir` joins them onto `works` at load:
`g.core["Tema"]` just works, `g.has_annotations` says whether anything was
loaded, `g.annotations` is the curated frame on its own, and works nobody
annotated keep a missing value instead of dropping out of the frame.

`annotation_schema.csv` (`column,type,allowed,description`) declares what may be
recorded. Declared `enum` columns are validated on load and a bad value raises
naming the work, so an agent writing `Maybe` fails immediately; undeclared
columns are carried untouched, so a reviewer can add a question mid-review
without a code change. The `description` is the annotator's instruction, which is
why it is not decoration. Ids resolve through [identity.py](src/citegraph/identity.py)'s
redirects — an annotation written before a merge still reaches the work it became
— and an id naming no work raises rather than being silently dropped, the same
rule a dead journal alias follows. **Columns `works.csv` already has (`Title`,
`Year`, …) are context**: they are written into the file so a human can read and
edit it, and are dropped on load rather than joined back over the pipeline's own
metadata.

Paper 4's `Tema` and `Publication_Type` were imported from `core_papers.xlsx` by
`paper4/annotations/make_work_annotations.py`: a one-time fuzzy crosswalk of 93
sheet titles onto the 93 core works, refusing to write unless the assignment is
one-to-one, audited in `crosswalk_papers.csv`, with each raw spelling mapped onto
one declared value so a new category cannot arrive silently as a lone bar in a
figure.

### Does a social tie show up in the citation record?

[cocitation_stats.py](src/citegraph/cocitation_stats.py) answers one question:
given a documented tie between two authors — paper 4 supplies doctoral
supervision — are the two co-cited more than a *comparable* pair of strangers?
`tie_cocitation_test(graph, ties, ...)` scores each pair against a
degree-matched permutation null and returns per-tie results plus an
assumption-bound Fisher diagnostic. It lives in the package rather than in a notebook because it is a
manuscript claim and both paper 4 notebooks are gitignored.

Three design points are load-bearing, and two of them were got wrong first:

- **The null holds prominence fixed.** Advisors and students are prominent, and
  prominent authors are co-cited with nearly everyone, so beating an *average*
  pair proves nothing. Each end is replaced by an author cited by a similar
  number of bibliographies (`degree_tolerance`, with a one-bibliography floor
  because a 20% band around a threshold of 3 admits only exact matches), and the
  tie's own members are excluded from both strata.
- **`exclude_self_citations` defines a universe; it does not filter one side.**
  The tie's remaining bibliographies become the eligible set, and the null pairs
  are then counted *and prominence-matched inside that same set*. Filtering only
  the tie is the intuitive implementation and it is wrong: on the paper 4 corpus
  485 of 564 co-cited authors wrote none of the 93 bibliographies while one end
  of a tie wrote all six that cited them, so the tie loses most of its chances and
  its comparison pairs lose none. That version reported 2 of 16 ties surviving;
  the conditional version reports 7, against 0.8 expected by chance.
- **The ties are not independent.** Seven people appear in more than one paper
  4 tie, so neither Fisher's method nor a binomial count of significant ties is
  confirmatory evidence. Report the per-tie permutation results and the
  descriptive `n_significant` against `expected_significant`; use
  `fisher_p_value` only when the supplied ties are independent.

Each tie seeds its own RNG from its own identity, so results don't depend on the
order ties arrive in or on how many came with them, and a pair given twice, or
both ways round, is scored once. `fisher_combined_p` is implemented exactly
(even-df chi-squared has a closed form) so the package keeps no SciPy dependency.
A tie touching an author below `min_papers`, or with at least one endpoint
lacking a degree-matched substitute, is *skipped with a reason* rather than silently
scored — read `result.skipped` before reporting coverage.

### Figure styling lives in one module

[plotting.py](src/citegraph/plotting.py) is the single definition of the
economics-journal figure conventions, and **every** manuscript figure goes
through it — both paper 4 notebooks (`examples/paper4_figures.ipynb` over
`citegraph_out/`, `examples/paper4_institutional_figures.ipynb` over
`institutional/out/`) import the same helpers so the two figure sets share one
visual language by construction. Rules and rationale: the `styling-econ-figures`
skill in `.claude/skills/`.

Draw bars with `bar(ax, …)` / `barh(ax, …)`, **never `ax.bar` / `ax.barh`**.
Matplotlib takes an unspecified bar colour from the property cycle, whose first
entry under `econ_style()` is `INK` — so a bare `ax.bar` comes out *solid black*
however `patch.facecolor` is set, and no rcParam can express "black lines, gray
bars". That gap is exactly how the two notebooks drifted apart: one passed
`color=GRAYS[1]` on every call, the other trusted the documented default and
printed black bars, and a third figure invented `GRAYS[2]`. The helpers apply
`BAR_FILL` with a black edge; an explicit `color=` still wins, for stacked or
multi-series bars.

Network figures have their own shared vocabulary, so a node-link diagram and a
bar chart read as one family: `network_axes` strips the chrome, `draw_node`
carries membership in its *fill and nothing else* (`NODE_FILL` inside the
population described, `NODE_OPEN` outside it, `focus=True` for the one node a
figure is built around — never a bold label on top of that), `node_legend` names
those fills, `hairline` / `arrow_props` draw connectors, `junction_dot`
punctuates them, and `separator_rule` divides bands. `hairline` pins a solid
line because `econ_style`'s property cycle carries *line styles*, so a bare
`ax.plot` silently comes out dashed on the second call.

`junction_dot` marks a point where connectors **meet**, which is what lets bare
crossing lines mean they don't. A bracket router needs that convention the
moment the thing it draws stops being a forest: with one parent per node no two
brackets ever touch, but let a doctorate have three supervisors and one
bracket's spine has to run past rows other brackets are using, where a
T-junction is indistinguishable from an X-crossing. Unmarked, that is a figure
asserting ties nobody recorded — in the paper 4 lineages it read as Martin A.
Nowak having trained Juan Camilo Cárdenas, who share no advisor and no edge.
The caller decides which points qualify; the rule the lineage trees use is
three or more connector directions leaving one point, because a dot on every
corner is a dot that means nothing. The mark sits far below `NODE_RADIUS` on
purpose — at anything near node size it reads as an unlabelled person.

`place_node_labels` puts the names on a node-link drawing. A force-directed
layout has no idea its nodes carry text, so the naive "label just above the
marker" collides — with another name, or with somebody else's marker. Each
label is drawn, *measured against the renderer*, and moved to the next anchor
around its node until it hits neither; measuring rendered text rather than
guessing distances is what makes it hold for a different corpus. Two details are
not optional. It takes positions for **every** node but labels for only the
subset that gets text, because an unlabelled node is still an obstacle. And a
candidate is measured **without** its leader line: an annotation that owns an
arrow reports the arrow inside its extent, so a far anchor always measures as a
collision and is never chosen — the leader is attached after the anchor is
settled. Placement follows a caller-supplied `order` (prominence, normally), so
the drawing is reproducible and the caller decides who keeps the best spot;
where nothing is clean the least-overlapping anchor wins, never the preferred
one, or a whole clump's names stack in a single place.

`box_node` is `draw_node` with room for a second fact — a name over an
institution — for a lineage diagram that has to show structure and placement at
once. It fills with **`BOX_FILL`, not `NODE_FILL`**, and the difference is not
cosmetic: a box is orders of magnitude more area than a marker, and at
`NODE_FILL` the subtitle gray and the fill are *the same value*, so every second
line inside a filled box renders invisible. An area mark takes the light end of
the ramp and keeps its text black; the black edge every node carries is what
makes the tint read as a fill at all. `node_legend(..., shape="box")` follows the
same fill so a diagram of boxes is not explained by a legend of circles. The
caller measures and passes `width`/`height` — the module owns marks, the
notebook owns measurement, because a tree layout needs text widths before any
axes exist. `flow_band` draws one ribbon of a two-column flow as a straight-edged
quadrilateral with a hairline outline; straight because a ribbon has to be
countable, and outlined because that is what separates the lightest fill from
its neighbour in the grayscale proof.

A two-column flow must decide stayer-vs-mover from the **ungrouped** columns and
carry the answer in (`flow_figure(..., same_mask=...)`). Reading it off the
plotted columns looks equivalent and is not: once a tail is pooled, "Other" on
the left and "Other" on the right are different sets, and a Norwegian doctorate
now held in Switzerland gets drawn and counted as somebody who never moved.

`save_figure` writes PDF + PNG + `<stem>.caption.md` **and** a black-and-white
proof under `<figures>/gray/`, so the "legible without colour" check leaves
evidence on disk for both corpora without either notebook remembering to ask.
`show_caption` renders the same caption block in the notebook, so display and
sidecar cannot drift. Both notebooks are gitignored; regenerating them is
documented in the paper 4 figure-regeneration notes.

### Per-paper failure isolation and warning files

Both per-paper extraction loops (`Pipeline.extract_paper_metadata` and `Pipeline.extract_paper_references`) wrap the per-paper body in `try/except Exception`. A single bad PDF — or one Gemini error after retries are exhausted — is recorded as a `PaperFailure` row and the loop continues with the rest. Failures are persisted as JSONL via `_write_failures` to `out_dir/metadata_failures.jsonl` / `references_failures.jsonl`; the file is *removed* when there are no failures so `path.exists()` <=> failures occurred. The cache is never written for a failed paper, so re-runs naturally retry it.

Most warning sidecar files follow the "absent = clean" convention. Stage 5
enrichment review files are written when enrichment runs so match quality can be
inspected:

| File | Stage | Created when |
| ---- | ----- | ------------ |
| `conversion_warnings.json` | 1 (convert) | any markdown is image-only, or lost too many characters to unmappable glyphs |
| `journal_alias_warnings.json` | 4 (dedup) | a curated `journal_aliases.csv` row folds nothing *yet* — only reachable with `enrich=True` before enrichment has run; once every vocabulary exists a dead row raises instead |
| `source_duplicates.json` | 2 (metadata) | two PDFs contain the same paper |
| `papers_no_references.json` | 3 (references) | a successfully-processed paper returned 0 references |
| `metadata_failures.jsonl` | 2 (metadata) | at least one paper raised an exception |
| `references_failures.jsonl` | 3 (references) | at least one paper raised an exception |
| `enrichment_summary.json` | 5 (enrich) | enrichment ran; contains match/miss/source counts and config |
| `enrichment_misses.csv` | 5 (enrich) | enrichment ran and produced unmatched references for review |
| `author_review.json` | 6 (authors) | a cluster looks low-confidence: initial-only with several citations and no external id, a possible uncorroborated compound surname (no-comma name with multiple full given words, or a given name matching another surname block), or a detected corporate author |

Counts for the pipeline warning sidecars are surfaced in `run_summary.json` (`n_conversion_warnings`, `n_source_duplicates`, `n_papers_no_references`, `n_metadata_failures`, `n_references_failures`, `n_author_review_flags`, `n_journal_alias_warnings`), alongside the works-model counters (`n_works`, `n_core_works`, `n_citations_raw`, `n_edges`, `n_core_to_core_edges`, `n_self_loops_dropped`). The `citegraph run`, `citegraph convert`, `citegraph dedup`, and `citegraph authors` CLIs emit a yellow warning line with the file path when any are non-empty.

### Stable IDs

[ids.py](src/citegraph/ids.py) builds slug-style IDs from `(first-author-surname, year, title)` with a single role-free `w-` prefix (`make_work_id`). The prefix deliberately carries no role: a work can move from cited stub to processed source without its identity changing. IDs must be deterministic across runs (canonicalization assumes the *first* row in a cluster names the cluster, and the enrichment cache is keyed by id), so don't introduce hash-based or row-index-based IDs.

Author cluster IDs follow the same principle: `_make_author_id` in [authors.py](src/citegraph/authors.py) slugs `surname_norm + given_token` to produce `a-…` ids, where `given_token` is the fullest (then most frequent) given-name word sequence observed in the cluster (or the canonical initials if no full name is present; corporate authors slug the normalized name alone). When two genuinely distinct people slug to the same id (rare, e.g. two "John Smith" clusters split by OpenAlex), a numeric suffix `-1`, `-2`, … disambiguates. The hand-curated `author_aliases.csv` references these ids directly, so keep id generation deterministic across runs.

### Gemini wrapper

All provider-specific code lives in [llm.py](src/citegraph/llm.py). Other modules speak Pydantic models, not Gemini types. Three non-obvious behaviors:

- `tenacity` retry with exponential backoff wraps every call.
- `response_was_truncated(response)` inspects `candidates[*].finish_reason` for `MAX_TOKENS` (defensive against SDK shape variation — works whether `finish_reason` is an enum or a plain string). The references stage uses this to retry once with a 2× output cap *before* falling back to JSON repair; metadata calls don't bother (they're capped at 500 output tokens).
- `fix_incomplete_json_string` is the last-resort safety net used by `parse_structured_response` when the JSON itself fails to parse (e.g. response truncated despite the retry). If you change the JSON shape that references come back as, also update the repair function.

### Schemas double as DTOs *and* response schemas

`PaperMetadata` / `Reference` in [schemas.py](src/citegraph/schemas.py) are passed to Gemini as `response_schema` and also used as the in-memory record format and the JSON cache format. Renaming fields (`Title`, `Authors_List`, `Journal`, `Year`) breaks both the LLM contract and the on-disk cache simultaneously — bump cache compatibility deliberately. `Authors` is *not* part of the LLM contract: it's a derived `@property` (`", ".join(Authors_List)`) added back in via a `model_dump()` override so CSV/cache writers and downstream consumers (e.g. `dedup.py`) still see a single human-readable string. Old caches that contain an `Authors` key load fine because `extra="ignore"` drops it during validation; the property recomputes on access.

### Configuration

Runtime settings flow through `pydantic-settings` in [config.py](src/citegraph/config.py) (env vars + `.env`). User-facing knobs (dedup weights, threshold, year window, model, enrich, enrichment year penalty/retry settings, ocr, LLM concurrency, author merge mode) are exposed both on `Pipeline(...)` and on the relevant `citegraph` CLIs in [cli.py](src/citegraph/cli.py); keep those two surfaces in sync when adding a knob.

### Cost estimation

[cost_estimation.py](src/citegraph/cost_estimation.py) gives a pre-flight token + USD estimate for stages 2 and 3 *without making any API calls*. It walks `out_dir/markdown/`, skips files whose `metadata/<stem>.json` and `references/<stem>.json` caches already exist, and runs the same `slice_to_references_section` slicer to size the references-stage input. Surfaced as `Pipeline.estimate_extraction_cost()` and `citegraph estimate`; `citegraph run` calls it after stage 1 and prompts before any LLM call (`--yes` skips). The heuristic constants at the top of the module (`_CHARS_PER_TOKEN`, `_METADATA_OVERHEAD_CHARS`, `_TOKENS_PER_REFERENCE`, etc.) are deliberately rough — adjust them if benchmarks drift. The `_PRICING` table is dated (mid-2025 Gemini rates); update it from https://ai.google.dev/pricing rather than trusting it as ground truth. Unknown models return `cost_usd() == None` and the summary degrades gracefully rather than failing.

### `research/` is not part of the package

Scripts under [research/](research/) are kept for historical context, not installed by pip, excluded from CI, and lint-loosened in `pyproject.toml`. Don't import from `research/` in `src/`.

## Testing without network

Tests must never hit Gemini. The pattern is in [test_pipeline.py](tests/test_pipeline.py): subclass `GeminiClient` with a `_FakeClient` that overrides `generate_structured`, distinguishing metadata vs. references requests by sniffing the prompt text (`"references or bibliography section" in prompt`). Pass it via `Pipeline(client=_FakeClient())`. If you change the prompt strings in `extract_metadata.py` / `extract_references.py`, update the sniff in `_FakeClient` too.
