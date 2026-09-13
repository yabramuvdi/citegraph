# Incremental Reruns Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development for independent cache work; integrate and review in this session.

**Goal:** Adding a paper reuses expensive observations and preserves unambiguous identities and corrections.

**Architecture:** Content-aware per-item caches, row-level enrichment provenance,
and persistent slug assignment registries around existing global clustering.

**Tech Stack:** Python stdlib, existing pandas/Pydantic/pytest.

**Spec:** `docs/superpowers/specs/2026-09-13-incremental-reruns-design.md`

## Global constraints

No new dependencies or live provider calls. Preserve legacy cache reads and user
edits. All artifact paths belong to OutLayout. Never guess identity on a split.

### Task 1: Conversion and extraction cache reuse

Files: `pdf_to_markdown.py`, cache helpers in `io.py`, focused cache tests.

- [x] Write/run failing tests: renamed PDF avoids conversion; replacement causes
  conversion; repeated failed-quality OCR is reused; renamed markdown reuses
  valid extraction without accepting stale/corrupt responses.
- [x] Implement content/provenance helpers and compatibility-aware reuse without
  changing public extraction schemas. Preserve legacy stem caches.
- [x] Run cache, traversal, extraction, and cost-estimate regressions.

### Task 2: Enrichment reuse

Files: `enrich.py`, `pipeline.py`, focused enrichment tests.

- [x] Write/run failing tests: ID-only change reuses validated enrichment;
  metadata change refreshes; changed/corrupt row does not suppress valid siblings.
- [x] Add indexed content lookup of v2 enrichment caches and row provenance.
  Pipeline author reads select only independently validated rows, offline.
- [x] Run enrichment and reliability tests.

### Task 3: Persistent identities

Files: new `identity.py`, `pipeline.py`, `authors.py`, OutLayout, identity tests.

- [x] Write/run failing tests: fuller author name retains ID; source promotion
  retains work ID; collision allocation never reuses retired IDs; merges record
  redirects; splits do not silently reuse the old annotation key.
- [x] Reconcile recomputed clusters using persisted occurrence membership, with
  unchanged evidence as anchors. Keep stateless functions backward-compatible.
- [x] Integrate identity assignments into tables and audits before publication.

### Task 4: Corrections

Files: `author_overrides.py`, pipeline integration, correction tests.

- [x] Write/run failing tests: unrelated additions retain corrections; changing
  a referenced occurrence fails; aliases survive fuller names after binding;
  conflicts fail without overwriting generated author outputs.
- [x] Bind reviewed occurrences and retain validated alias memberships.
- [x] Exercise legacy migration, merges, splits, and interrupted reruns offline.

### Task 5: Integration and review

- [x] Add an end-to-end add-paper/resume check with counted expensive boundaries.
- [x] Document cache compatibility, identity redirects, and correction review.
- [x] Run `.venv/bin/python -m pytest -q` and `.venv/bin/ruff check .`.
- [x] Review diff for identity ambiguity, stale evidence, legacy migration, and
  preservation of user changes; fix actionable findings and rerun affected tests.

## Progress

Branch: `codex/incremental-reruns`. Working in place preserves existing uncommitted
reference extraction changes; no real corpus artifacts are modified.

Implementation notes: Added an active-input manifest so progressive stages do not
resurrect renamed/removed PDF caches. Identity aliases retain bound algorithm
endpoints while accepting published IDs for newly entered corrections. Legacy
author observations are validated against the prior works before adoption.

Offline real-corpus rehearsal used only a temporary copy. Its existing author
occurrence table does not match the prior extracted works population, and six
legacy alias IDs are stale under the current normalizer before adding a paper.
Those ambiguous historical corrections remain review-required; no actual corpus
files were changed. Synthetic add-paper, interrupted-resume, rename, promotion,
split/merge, and correction migration regressions pass.
