# Enrichment precision, journal normalisation, downstream regeneration, core annotations

Date: 2026-09-13
Status: proposed

Four workstreams, executed in this order because each one's output is the next
one's input. Doing them out of order means regenerating downstream artifacts
twice.

```
W1 enrichment author check ──┐
                             ├─> targeted re-crawl ─> W3 downstream regeneration
W2 journal normalisation ────┘
W4 core-paper annotations (independent; schedule anywhere)
```

---

## W1 — Enrichment author agreement

### Problem

`_best_match_report` in `enrich.py` scores candidates on title similarity alone,
then subtracts a flat `year_mismatch_penalty = 8`. With `title_match_threshold =
90`, a title score of 100 survives *any* year gap (100 − 8 = 92). Nothing ever
looks at authors.

Measured on the paper 4 corpus (2,649 matched works): **47 matches have a
completely disjoint author list. All 47 are OpenAlex-sourced; zero are CrossRef;
zero are ring 0.** The dominant failure mode is book reviews and edited-volume
chapter records, whose OpenAlex title is the reviewed work's title verbatim:

| extracted | provider |
| --- | --- |
| Ostrom, *Governing the Commons* | Grbeša; Musa (Croatian review) |
| Bowles, *Microeconomics* | david kandel (year → 2020) |
| Richerson & Boyd, *Not by Genes Alone* | Martin H. Levinson |
| Baland & Platteau, *Halting Degradation* | Neil Carter |
| Hansen & Lott, *Winner's Curse: Comment* | Cox; Dinkin; Smith (a different comment, same title) |

These overwrite `Journal` and `Year` in `enriched_works.csv`, so the damage is
not confined to author identifiers — and W2 wants to trust exactly those
columns.

### Why year is not the fix

175 matches have a year delta > 2, but **158 of them have overlapping authors**:
reprints and later editions (*The Selfish Gene* 1976/2017, *The Tragedy of the
Commons* 1968/2018). Tightening the year penalty would destroy far more than it
repairs. Author agreement has to be the discriminator.

### Design

A **veto**, not a score adjustment. Author agreement decides candidate
*eligibility*; title score continues to decide acceptance. Keeping the two
separate means the change can never let a weak title through on author evidence,
which keeps the methods description simple.

New public helper in `authors.py` (all name machinery lives there; `enrich.py`
already depends on that direction and `authors.py` does not import `enrich.py`):

```python
def author_list_agreement(extracted: list[str], provider: list[str]) -> bool | None
```

Returns `True` (at least one name pairs), `False` (usable names on both sides,
none pair), or `None` (abstain — no usable evidence).

Token-set comparison, deliberately **not** the `ParsedAuthor.surname_norm` path
that `_match_enrichment_authors` uses. That path needs the corpus surname
lexicon, which does not exist yet when enrichment runs; without it every
comma-less provider name collapses to its last token and compound surnames stop
matching. Prototyped both: the surname version vetoed 57 including many correct
matches (`Moros, L.` → `Lina Moros Canon`, `Moya, A.` → `Andrés Moya
Rodríguez`), the token-set version vetoed 35 with far fewer false vetoes.

Rules:

- Collect every alphabetic token of length > 2 from each side, dash-folded and
  diacritic-stripped (reuse `_fold_dashes` / `_strip_diacritics`), apostrophes
  removed so the OCR artifact `Ca'rdenas` reads as `cardenas`, `et al.` tails
  dropped, particles (`de`, `van`, `der`, …) dropped.
- Two tokens pair on exact equality, or `rapidfuzz.ratio >= 85` when both are
  ≥ 5 characters. That tolerance is what rescues the extraction typos —
  Hirschmann/Hirschman, Garupa/Garoupa, Sergeson/Segerson, Appelbaum/Applebaum,
  Andreoni/Andereoni — all of which are correct matches.
- **Abstain** when either side is corporate. `_is_corporate_author` catches
  named institutions; add a local acronym test (single all-caps alphabetic token
  of ≥ 3 characters) so IFPRI, IPCC, ECLAC, UNFCCC, IGAC and UNODC abstain
  rather than veto — their provider records correctly list the human chapter
  authors.
- Abstain when either side is empty.

Threading: `_best_match_report` gains an `authors: list[str]` parameter.
`_crossref_lookup_report` already receives an author string;
`_openalex_lookup_report` needs one added. `_enrich_one` has the row, so it
passes `parse_authors_list(row.get("Authors_List"))`.

Selection becomes filter-then-rank, so a vetoed winner falls through to the next
candidate instead of sinking the whole lookup:

```
score every candidate (title, year_delta, agreement)
if not scored:                       -> miss("no_<source>_candidates")   # unchanged
eligible = [c for c in scored if c.agreement is not False]
best = max(eligible or scored, key=adjusted)
if not eligible:                     -> miss("author_mismatch")
elif best.adjusted < threshold:      -> existing miss reasons, unchanged
else:                                -> match
```

`author_mismatch` ranks above `below_title_threshold` and below `http_error` in
`_select_miss_report`: it is a definite negative result, where a title-threshold
miss is merely a weak one, but it must never mask a transient outage.

Provider query strings are **not** changed. Sending the extracted authors to
OpenAlex's `raw_author_name.search` filter would bias against the very
extraction typos the fuzzy tolerance is there to absorb. Out of scope.

### Surface changes

- `enrich.py`: `_best_match_report`, `_openalex_lookup_report`, `_enrich_one`,
  `_DIAGNOSTIC_COLUMNS` (new `enrichment_author_agreement`, one of
  `match` / `mismatch` / `unknown`), `_valid_result`
  (accept the new column), `_select_miss_report` (priority for
  `author_mismatch`), `_write_enrichment_sidecars` (new column in
  `enrichment_misses.csv`).
- `authors.py`: new public `author_list_agreement`.
- `EnrichConfig`: `author_surname_fuzz: float = 85.0`. One knob, mirrored onto
  `Pipeline` and `citegraph enrich` per the CLAUDE.md sync rule.
- `CLAUDE.md`: stage 5 paragraph.

### Cache compatibility

The cache stays at `schema_version = 2`. The new diagnostic is optional and
`_with_cache_diagnostics` fills it from `_DIAGNOSTIC_COLUMNS`, so **existing
caches stay valid and no decision is re-made unless its cache entry is
deleted.** That is deliberate: it makes the re-crawl targeted instead of
corpus-wide.

The cache stores only the chosen match, never the candidate list, so re-deciding
requires re-querying. Retaining candidates (a v3 cache) would make future
rescoring free — noted, not proposed here.

### Tests (TDD, `tests/test_enrich.py`)

1. Disjoint author lists at title score 100 → miss with `author_mismatch`.
2. Fuzzy surname pairs (`Hirschmann, A.` / `Albert O. Hirschman`) → match.
3. Corporate side abstains → match preserved.
4. All-caps acronym (`IFPRI`) abstains → match preserved.
5. A vetoed best candidate falls through to an eligible lower-scoring one.
6. Empty author list on either side abstains.
7. An old cache file with no `enrichment_author_agreement` still loads and serves.
8. `enrichment_misses.csv` carries the new column and the new reason.

### Verification on the corpus

Prototyped against the real 2,649 matched rows: **29 vetoes, all OpenAlex, none
ring 0.** Roughly 26 are genuine wrong-record matches; the ~3 remaining
conservative losses are corporate-authored reports whose acronym the abstain
rule misses (PNUD spelled out, Transparency International). Rejecting is the
safe failure — they land in `enrichment_misses.csv` for review; accepting a
wrong record does not announce itself.

### Targeted re-crawl

**Corrected after implementation — only the 29 vetoed works need re-crawling,
not 419, and the 628 misses need none at all.** The veto only *removes*
candidates; it adds no bonus term. That makes the blast radius provable:

- A currently-matched work whose winner is **not** vetoed: the winner still has
  the maximum adjusted score among all candidates, so `max(eligible)` returns it
  unchanged. Re-crawling is a no-op.
- A currently-matched work whose winner **is** vetoed: the result changes. These
  are exactly the 29.
- A currently-missed work: every candidate scored below the threshold, and
  fall-through can only select a *lower*-scoring candidate. A miss cannot become
  a match. The planned second pass over the 628 misses would find nothing.

So: delete the 29 vetoed cache entries, re-run `citegraph enrich`, then re-run
`authors` because the OpenAlex/ORCID evidence changes. ~29 OpenAlex searches at
10 credits is negligible against the daily budget, where the original plan's 419
plus 628 would have consumed most of a day's allowance for no gain.

### Flagged, not in scope

Elinor Ostrom carries **no OpenAlex id and no ORCID** despite 63 works — the
conflict rules are shielding her from exactly the bad matches above. Re-check
after W1's re-crawl; it may resolve on its own.

---

## W2 — Journal name normalisation

### Problem

Smaller than `DATA_QUALITY_NOTES.md` implies. Enrichment already collapses 1,615
extracted spellings to 894 on matched rows. AER goes from 6 spellings to 3, and
those 3 differ only by a leading `The` and a `Papers & Proceedings` suffix.

### Design

Two layers, both additive — no existing column is overwritten.

1. A `Journal_Canonical` column computed by a pure function
   `canonical_journal(name) -> str` in a new `journals.py`: prefer the enriched
   container title over the extracted one when the work matched, then fold case,
   strip a leading `The`, normalise `&` ↔ `and`, collapse whitespace and
   punctuation, and strip a trailing edition/series parenthetical. Title-case
   the result for display from the most frequent observed spelling, the same
   "fullest, most frequent wins" rule the author stage uses for display names.
2. `out_dir/journal_aliases.csv` (`raw,canonical`), hand-curated, applied last —
   identical in shape and spirit to `author_aliases.csv`, for the residue that
   no rule catches.

`Journal_Canonical` is written by the dedup stage onto `works.csv` from the
extracted names, so the column exists whether or not enrichment ran, and
recomputed by the enrichment stage onto `enriched_works.csv` from the container
titles. Both call the same function; only the input differs.

W1 is a prerequisite: 29 of the container titles it currently trusts come from
the wrong record.

### Tests

Table-driven cases for the fold rules, an alias-file override case, an unknown
alias key failing loudly (mirroring `_apply_aliases`), and a works-frame
integration case.

---

## W3 — Downstream regeneration

Ordered, mechanical, one pass. Current staleness:

| Artifact | State |
| --- | --- |
| `institutional/crosswalk_people.csv` | **70/70 ids resolve** — clean re-run |
| `institutional/out/*` | built 12:38 against 14:59 `authors.csv` — stale |
| `author_centrality.csv` | 2026-09-12, **25/553 ids no longer resolve**, predates the Ekonomiska repair |
| `core_authors.xlsx` | 2026-09-09 — stale |

Steps:

1. `make_crosswalks.py` → `make_people_crosswalk.py` → `build_clean_base.py`.
   Non-destructive and append-only by design; re-check `out/review_flags.csv`
   afterwards.
2. Re-execute `examples/paper4_figures.ipynb` with nbclient (`~10 s`), which
   rewrites `author_centrality.csv`.
3. **Delete section 7's Ekonomiska exclusion filter and the cell-24 filtered
   `CitationGraph`.** The 180 records are gone from the corpus; the filter is now
   a no-op that misleads the next reader. Confirm `0 Ekonomiska works` in
   `works.csv` first.
4. Re-sync the numbers in `NETWORK_METHODOLOGY.md` to the new notebook output,
   per its standing instruction.
5. Re-run `citegraph report`.

**`core_authors.xlsx` — recommend retiring rather than regenerating.** Its four
researcher columns (`phd`, `postdoc`, `affiliation`, `notes`) are empty across
all 147 rows; the real career data lives in `base_institucional.xlsx` (58
authors) and is already cleaned by the `institutional/` pipeline. No generator
for it exists anywhere in the repo or the notebooks. Regenerating it means
writing a generator for a template nobody filled. Your call — I would delete it
and let W4's generator cover the pattern if an author-side annotation table is
ever wanted.

Notebook risk: `paper4_figures.ipynb` is gitignored and has already been lost
once. Back it up to the scratchpad before executing, and decide on force-adding
or jupytext pairing.

---

## W4 — Core-paper annotations

Design approved 2026-09-13; unbuilt. `identity.py` has landed, which unblocks
the join-key decision.

### Join key

Key on the persistent work `id`, resolved through `identity.py`'s redirect table
(`resolve_redirect`), with `source_file` kept as a human-readable column and as
the bootstrap key for the existing sheet. Retired/split ids surface as an
explicit error rather than silently transferring an annotation to an arbitrary
child — the registry already records that distinction.

### Components

- `citegraph annotate --out ./out` writes `out_dir/work_annotations.csv`: one row
  per ring-0 work, generated columns `id, source_file, Title, Authors, Year,
  Journal`, and every researcher-typed column carried over by `id`. Re-runnable
  and append-only, like the `institutional/make_*` scripts.
- `load_annotations(out_dir)` validates against `works.csv`, fails loudly on
  unknown ids, and returns a frame indexed by work id ready to `.join()` onto
  `g.core`.
- **CSV in `out_dir` is authoritative.** Diffable, no openpyxl dependency,
  consistent with `author_aliases.csv`.
- `--xlsx <path>` exports to a **new** workbook and imports from one.

### Constraint found while planning

`core_papers.xlsx` has **8 sheets**, not 1: `LFEs_COL` (93 rows, the annotation
data) plus `temas`, `experimentalistas`, `repetidos`, `MANUALS`, `LFE otros
países`, `Isaaza`, `Los que no son LFE`. The importer must read only the named
sheet, and the exporter must never write back into `core_papers.xlsx` — a naive
round-trip would destroy seven sheets of Maria's work. This is the strongest
argument for CSV-authoritative.

### One-time crosswalk

`crosswalk_papers.csv`, same shape as `crosswalk_people.csv`: fuzzy-match
`Título` → core titles with a `needs_review` flag. Measured previously: 63
exact, 22 ≥ 90, 3 ≥ 80, 4 below, before normalising case and HTML entities
(`Leaders' Distributional &amp; Efficiency Effects`). Realistically ~5 need
Maria's eye, notably "Capital social y territorio" and the Villegas thesis,
whose sheet row names a chapter rather than the dissertation.

### Tests

Template generation for a synthetic ring-0 frame, carry-over of researcher
columns across a regeneration, unknown-id failure, redirect resolution for a
merged work, xlsx round-trip preserving unrelated sheets, and a CLI smoke test.

---

## Risks

| Risk | Mitigation |
| --- | --- |
| Author veto rejects correct matches | Measured at ~3 of 2,649; all corporate reports; they land in `enrichment_misses.csv` for review |
| Re-crawl exhausts the OpenAlex daily credit budget | Targeted at 419 works first; the 628 misses are a separate, later decision |
| Re-running `authors` shifts published author ids | `author_aliases.csv` is curated and backed up; re-verify 70/70 crosswalk ids as the gate, same check as 2026-09-13 |
| Notebook is gitignored and unversioned | Back up before executing; decide on force-add or jupytext |
| W2 trusts container titles from wrong records | W1 ships first — that is why the order is fixed |

## Environment

`convert` needs `~/.pyenv/versions/3.11.10/bin/citegraph`; the repo `.venv` lacks
docling, networkx and openpyxl. No stage in this plan runs `convert`, but W3's
notebook needs networkx and openpyxl, so it runs on the pyenv interpreter.
