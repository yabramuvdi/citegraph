# Roadmap

Future improvements identified during the design review. Each item lists
the motivation, the proposed change, and any blockers that argue for
deferring it.

## Metadata extraction (`extract_metadata.py`)

### Use PDF / DOI / arXiv ground truth as a cross-check
- **Today.** Every paper is extracted from scratch by the LLM, even when
  the PDF carries usable `/Title` and `/Author` fields, or when the
  paper has a DOI we could resolve via CrossRef.
- **Proposed.** Optional fallback / validation pass: pull PDF metadata
  during the docling stage, compare with the LLM result, log
  disagreements. CrossRef-by-DOI could fill missing journals/years.
- **Note.** PDF metadata is famously unreliable, so use it as a
  *cross-check*, not a replacement.

### Concurrent / batched LLM calls
- **Today.** Both metadata and references stages iterate paper by paper
  with a single Gemini call per paper. Sequential round-trips dominate
  wall-clock time on large corpora.
- **Proposed.** Add a small thread/async pool around the per-paper
  loops. Cap concurrency via a setting (e.g. `CITEGRAPH_LLM_CONCURRENCY`).
  The cache layer already isolates each paper, so a worker failure is
  contained.

### Replace hand-wavy "no dirty characters" instruction with deterministic normalization
- **Today.** The metadata prompt asks the model to "not include any
  dirty characters that might have come from problems in the PDF."
  Unverifiable, model-dependent, silent failures.
- **Proposed.** Drop that line from the prompt. After parsing, normalize
  fields in Python: NFC unicode, strip control chars, collapse internal
  whitespace, fix common ligatures (`ﬁ`→`fi`, etc.). Apply the same
  normalization to `Reference` fields.

## Dedup (`dedup.py`)

### Smarter canonical-record construction across cluster members
- **Today.** The first row in each cluster names the cluster *and* becomes
  the canonical record. If that first row happens to have `Year=0`, an
  empty journal, or a truncated title, the canonical entry inherits those
  weaknesses even when later cluster members have better data.
- **Proposed.** Keep the first-row-wins rule for the cluster ID (preserves
  ID stability per `CLAUDE.md`), but build the canonical *record* by
  merging across all cluster members: prefer non-empty / non-sentinel
  values, optionally take the longest title, etc.
- **Why now.** Mostly relevant for references that the LLM extracted
  inconsistently across citing papers (one missing the year, another
  missing the journal). The new "missing-year is OK" tolerance in
  `compare_papers` makes these clusters form correctly — but the canonical
  row is still only as good as the first member.

### Manual review of dedup decisions (UI / terminal / auto modes)
- **Today.** Dedup is fully automatic: every cluster the algorithm forms
  is accepted. Borderline pairs (same title, slightly different authors;
  year off by 1; etc.) silently merge with no user check.
- **Proposed.** A `review_mode` parameter on `Pipeline(...)` — and a
  matching `--review` flag on `citegraph dedup` — selecting how dedup
  proposals are confirmed:
    - `auto` (default): current behaviour, no prompts.
    - `terminal`: for each proposed merge print the representative + the
      candidate it would absorb side-by-side, plus the weighted score and
      year diff, and prompt `[a]ccept / [r]eject / [s]kip / [q]uit`.
      Quitting writes pending decisions to disk so the user can resume.
    - `web`: start a tiny local web server (FastAPI or stdlib) that
      serves a single page listing pending merges, one per row, with
      accept / reject buttons; decisions stream back over a small JSON
      endpoint and the pipeline blocks until the queue drains or the
      user marks "done with the rest as auto".
- **Persistence.** Save decisions to `out_dir/dedup_decisions.jsonl`,
  keyed by the unordered pair of source-row hashes (or by the proposed
  cluster id + absorbed row id). Re-runs replay decisions instead of
  re-prompting. Without this, manual review is unworkable on a corpus of
  more than ~50 papers.
- **Scope the prompts.** Don't ask about *every* pair — only the
  borderline ones (e.g. weighted score within ±5 of the threshold, or
  year diff at the edge of the window, or pairs flagged by the new
  missing-year tolerance). Definite matches and definite non-matches
  stay automatic. Keep humans in the loop only where the algorithm is
  genuinely uncertain.
- **Visual style for the web UI.** Modeled on
  [justadesignlist.com](https://www.justadesignlist.com/):
    - **Palette.** Background `#f5f4f1` (warm off-white, not pure white).
      Primary text `#111` (near-black). Secondary text `#999` (medium
      gray — used for IDs, scores, years, secondary labels). Hairline
      dividers `#e0e0df`. No accent colour; the only "active" state is an
      inverted pill (black background, white text) for the selected filter.
    - **Typography.** A single grotesque sans-serif family doing all the
      work through weight variation — Geist covers body and labels. For
      IDs, scores, and years use JetBrains Mono. *Note:* the reference
      site is grotesque throughout; if Instrument Serif is added for page
      headings ("5 items to review.") it departs from the reference — that
      departure is intentional but worth keeping narrow (headings only,
      never row content).
    - **Row anatomy.** Four implicit columns with no explicit borders,
      separated only by whitespace rhythm:
        1. **Left gutter** — a stage/checkpoint marker (large muted letter
           or icon, small count below), analogous to the `A · 5` index
           markers in the reference.
        2. **Identifier** — filename or reference ID, left-aligned, regular
           weight.
        3. **Data columns** — extracted fields or merge candidates, spread
           across the center; secondary metadata (journal, year) in small
           uppercase `#999`.
        4. **Action** — `→` right-aligned for expand; `[a]` / `[r]` accept
           / reject affordances styled as plain text links, no button chrome.
    - **Confidence bars.** For dedup rows, render the weighted score as a
      proportional black bar (full width = 100 score), analogous to the
      category count bars in the Categories view of the reference. The bar
      sits between the data columns and the action column; the numeric score
      appears right-aligned in JetBrains Mono beside it.
    - **Filter / state pills.** Filled black pill (white text) = active
      selection; outlined pill (black text) = inactive option. Used for
      checkpoint tabs (Metadata / References / Dedup) and for
      accept/reject/skip counts in the header.
    - **Layout.** List-of-rows, never cards. One row per pending item.
      Row height ~48–56 px. Single-pixel hairline between every row.
      Generous horizontal margins; content does not touch viewport edges.
    - **No chrome.** No modals, no tooltips, no shadows, no rounded
      corners on content areas (pills are the only exception). No colour
      used for status — use weight and position instead.
    - **Vibe.** Editorial / utilitarian. "A quiet page."

## Evaluation Framework (human-in-the-loop)

### Overview: auto / terminal / web modes

- **Today.** The full pipeline runs end-to-end automatically. There is no way to spot-check LLM output mid-run or to flag suspicious extractions before they propagate downstream.
- **Proposed.** A unified `eval_mode` parameter on `Pipeline(...)` — and a matching `--eval` flag on `citegraph run` — that governs how human validation is offered at three natural checkpoints in the pipeline:
    - `auto` (default): current behaviour, no prompts.
    - `terminal`: at each checkpoint, print a random sample of extracted items alongside their source and prompt `[a]ccept / [r]eject / [e]dit / [s]kip / [q]uit`. Quitting writes pending decisions to disk so the user can resume later.
    - `web`: start a tiny local web server that presents each checkpoint as a queue of rows to review. Same visual design as the dedup review UI (see **[Dedup § Manual review](#manual-review-of-dedup-decisions-ui--terminal--auto-modes)**) — the evaluation framework is its natural extension across the whole pipeline.
- **Sample size.** The number of items drawn at each checkpoint is configurable (`--eval-sample-size`, default 5 for metadata, 3 papers × 5 refs for references). A seed is stored in `out_dir` so re-runs draw the same sample unless the seed is cleared.
- **Persistence.** Decisions are written to `out_dir/eval_decisions.jsonl`, keyed by file stem and item index. Re-runs replay accepted decisions and re-prompt only for items that were skipped or never reached.
- **Checkpoints fire after each stage's batch completes**, before output CSVs are written. If the user rejects or edits items, only the affected per-paper cache files are cleared; the rest of the batch is not re-processed.

### Web UI: design vision

**Conceptual frame: a proof-reading session.** Not a dashboard, not a wizard. The user is moving through extracted data the way a copy-editor moves through a manuscript — deliberate, linear, eyes on the source. Every decision the UI asks is: "does this match what you see on the left?" The interface should feel like that work, not like a SaaS tool.

**Entry state.**
The server starts and opens a single page. The user lands on a summary view:
- Large display heading (Instrument Serif, ~48 px): *"17 items to review."*
- Above it: a small uppercase label in `#999`, letter-spaced — `REVIEW SESSION`.
- Below the heading: three rows, one per checkpoint, using the same hairline-divided list-row anatomy the checkpoints themselves will use. Each row: checkpoint name left-aligned | item count in JetBrains Mono | a proportional black bar (same pattern as the Categories page of the visual reference) spanning the center. This gives the user an at-a-glance sense of where the work is concentrated before they begin.
- One action at the bottom, rendered as a plain text link: `Begin →`

**Navigation: sequential, chapter-by-chapter.**
The user works through checkpoints in pipeline order (Metadata → References → Dedup). No jumping, no tabs. This is a linear read.
- A fixed header bar, 40 px tall, separated from content by a single hairline:
  - Left: `citegraph / review` in small uppercase `#999`
  - Center: current checkpoint name (`METADATA`, `REFERENCES`, or `DEDUP`) in small uppercase `#111`
  - Right: remaining item count in JetBrains Mono (`7 left`), ticking down with each decision — no animation, just the number changing bare, like a typewriter counter
- Between checkpoints: a brief transitional state replaces the row list. The heading changes (*"Metadata done. 5 references to check."*) and the user advances with a single `→` keypress or link click.

**Spatial grammar.**
- Max content width: 1200 px, centered, 48 px horizontal margins.
- Header: 40 px fixed.
- First content element: 64 px below the header.
- Section label (small uppercase `#999`, 11 px, `letter-spacing: 0.12em`): sits above the display heading, like a chapter number.
- Display headings: Instrument Serif, 40–48 px, left-aligned. Used only for the entry state, chapter transitions, and the session-end screen.
- Row content: Geist at 14 px for field values; JetBrains Mono for all IDs, scores, years, counts.
- Rows span full content width. Fields are distributed across the row with the same multi-column whitespace rhythm as the visual reference's list view — identifier anchored left, data spread center, action anchored right.

**The reading line.**
A single 1 px horizontal rule (`#e0e0df`) sits *above* the active row and moves with it on a 200 ms `ease` CSS transition as the user navigates with `j` / `k`. It does not fill, highlight, or border the row — it simply marks position, like a ruler held against a page while reading. This is the only animation on the page, and it is the detail that makes the interface feel intentional.

**Keyboard grammar (shared across all checkpoints).**
- `j` / `k` — move to next / previous row
- `a` — accept current row
- `r` — reject current row
- `e` — enter inline edit mode on the focused field (underline activates)
- `u` — undo last decision; the row resets silently, no confirmation
- `→` / `Enter` — advance to next checkpoint when current queue is empty

All actions are immediately reversible with `u` within the session. No confirmation dialogs anywhere.

**Session end.**
No modal. The row list collapses and the heading transitions to:
- Display heading (Instrument Serif): *"Done."*
- Below: three summary rows — same hairline-divided anatomy — one per checkpoint, showing `accepted · edited · rejected` counts in JetBrains Mono.
- One action: `Resume pipeline →` as a plain text link, bottom of the list.

---

### Checkpoint 1 — Metadata extraction

- **When.** After `extract_paper_metadata` finishes its full batch, a random sample of N papers is drawn.
- **What the user sees.**
    - **Terminal**: a two-column table — filename on the left, extracted fields (`Title`, `Authors`, `Year`, `Journal`) on the right, one paper per screen.
    - **Web**: list-of-rows layout (same hairline-divider style as dedup). Each row shows filename left, extracted fields right. Inline edit fields allow correcting any value in place.
- **Output.** Corrections overwrite the per-paper `metadata/<stem>.json` cache. Only corrected files are re-validated; the rest proceed normally.

### Checkpoint 2 — References extraction

- **When.** After `extract_paper_references` finishes, a random sample of K papers is drawn. For each sampled paper, M references from its extracted list are shown.
- **What the user sees.**
    - **Terminal**: reference fields printed per paper, one reference per prompt, alongside the paper filename.
    - **Web**: same list-of-rows style as dedup. For each sampled paper, the **left column** renders the paper's bibliography section (loaded from `out_dir/markdown/<stem>.md`, sliced with the existing `slice_to_references_section` function) so the user can read the raw source. The **right column** lists the extracted `Reference` objects for that paper, one per row, with inline edit affordances. Showing the actual markdown source lets the user immediately spot dropped, truncated, or mis-parsed references without leaving the browser.
- **Split-column layout ("broadsheet fold").** The web view uses a two-column layout that feels like a book spread opened flat — not a dashboard panel:
    - **Proportions.** Left column ~58%, right column ~42%, divided by a single hairline (`#e0e0df`). The hairline sits in the gap between the two columns with ~24 px of breathing room on each side — a gutter, not a border.
    - **Left column ("source").** The raw bibliography markdown rendered as flowing prose — no truncation, no line clamping. The column has its own scroll context; when the user clicks a row in the right column it scrolls to bring the relevant span into view.
    - **Right column ("extracted").** One row per `Reference` object, hairline-divided. Field labels in small uppercase `#999` (TITLE · AUTHORS · YEAR · JOURNAL); field values in regular weight `#111`.
    - **Correspondence highlight.** Hovering or focusing a row in the right column applies a translucent amber wash (`rgba(180, 140, 80, 0.12)`) to the matching span of text in the left column — no border, no outline, just a highlighter-on-paper tint. This makes the source-to-extraction mapping visible at a glance and is the primary verification affordance.
    - **Edit micro-interaction.** Hovering a field value in the right column reveals a 1 px underline beneath it — no button, no border. Clicking makes the value `contenteditable`; the underline becomes solid `#111`. On blur the edit commits silently; a brief `✓` in `#999` fades out over 400 ms.
- **Output.** Corrections overwrite the per-paper `references/<stem>.json` cache. Papers flagged for full re-extraction have their cache cleared so the next `citegraph references` re-calls the LLM for those files only.

### Checkpoint 3 — Deduplication

See **[Dedup § Manual review of dedup decisions](#manual-review-of-dedup-decisions-ui--terminal--auto-modes)**. That item already specifies the terminal and web modes, scope-limiting to borderline pairs, and the `dedup_decisions.jsonl` persistence mechanism. The evaluation framework unifies it under the same `eval_mode` flag so users set one option for the entire pipeline and get consistent behaviour at all three checkpoints.

---

## Enrichment (`enrich.py`)

The five gaps (tests, per-ref caching, `EnrichConfig` exposure, concurrency, separate artifact)
have been implemented. The following are the next logical expansions once the base is solid.

### A. Semantic Scholar as a third source
- **Today.** CrossRef → OpenAlex fallback. OpenAlex covers most venues but has weaker coverage for
  preprints and CS/ML workshop papers.
- **Proposed.** Add `_semantic_scholar_lookup` as a third fallback after OpenAlex. Semantic Scholar
  returns abstracts directly (no index reconstruction), citation counts, and influence scores —
  useful for the additional-fields work below.
- **Note.** Semantic Scholar has a free API (100 req/5 min unauthenticated, 1 req/s with key). Add
  `semantic_scholar_api_key: str = ""` to `EnrichConfig` and a matching `--enrich-s2-key` CLI flag.

### B. Additional enriched fields
- **Today.** Enrichment populates `doi`, `Title`, `Authors`, `Journal`, `Year`, `enrichment_source`.
- **Proposed.** Capture the following where available:
  - `cited_by_count` — OpenAlex `cited_by_count`, CrossRef `is-referenced-by-count`. Lets users
    rank references by influence.
  - `oa_url` — OpenAlex `open_access.oa_url`. Direct link to the freely available PDF.
  - `abstract` — Semantic Scholar returns it directly; OpenAlex stores `abstract_inverted_index`
    which requires a one-pass reconstruction (`{word: positions}` → sorted string).
- **Note.** These fields are additive (new columns); the schema contract with dedup is unchanged.
  Add them to `_normalize_record` and `enriched_references.csv` only — not to `references.csv`.

### C. Source-paper enrichment
- **Today.** Only the cited references (`references.csv`) are enriched.
- **Proposed.** Run the same enrichment pass over `papers.csv` (the source papers). Output to a
  separate `enriched_papers.csv`. Source papers are typically easier to resolve (higher-quality
  LLM extraction, more likely to have DOIs in the PDF itself).
- **Note.** Add `enrich_papers: bool = False` to `Pipeline` and a `citegraph enrich --papers` flag.
  Reuse `enrich_references` — the DataFrame schema is compatible.

### D. Year-penalized scoring in `_best_match`
- **Today.** The only signal in `_best_match` is the title fuzzy-ratio. A paper with the same title
  but a different year still passes the threshold.
- **Proposed.** When the query year and candidate year are both known, subtract a penalty from the
  score if they differ by more than 1 year (e.g. `score -= 10 * max(0, abs(q_year - c_year) - 1)`).
  This keeps same-year or adjacent-year matches (reprints, preprint + journal version) while
  rejecting same-title papers from different years (e.g. annual reports, recurring workshop names).
- **Note.** Expose `year_penalty_per_year: float = 10.0` on `EnrichConfig`.

### E. Retry / rate-limit handling
- **Today.** `_crossref_lookup` / `_openalex_lookup` catch all exceptions as debug-level no-ops.
  A 429 or 503 response is silently treated as a miss.
- **Proposed.** Wrap the `client.get` calls with `tenacity` retry (already a dep from `llm.py`):
  exponential backoff on 429/503, give up after ~3 attempts. Log at WARNING when retries are
  exhausted so users know they hit a rate limit rather than a genuine no-match.
- **Note.** Use `httpx.HTTPStatusError` to inspect the status code inside the retry predicate.
