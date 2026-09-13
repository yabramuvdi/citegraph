# Pipeline Robustness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close demonstrated validation, recovery and evidence gaps with measured changes that preserve library workflows and corpus history.

**Architecture:** Extend existing helpers and the offline evaluator first. Select storage guarantees through crash tests, revalidate enrichment offline, and measure clustering before choosing its correction. Neither a generation layer nor a new membership algorithm is preapproved.

**Tech Stack:** Python >=3.11; stdlib, existing pandas/Pydantic/rapidfuzz/pytest; no new runtime dependencies.

**Spec:** [Revised robustness design](../specs/2026-09-13-pipeline-robustness-design.md).
Also read `CLAUDE.md`, `docs/RELEASE.md`, `docs/USER_GUIDE.md`, and the incremental-reruns/enrichment designs before execution.

## Global Constraints

- Python >=3.11; no new runtime dependencies for this work.
- Every new artifact path belongs to `OutLayout`.
- Preserve `PaperMetadata` and `Reference` field names and their response schemas.
- Hashes identify content, never work or author IDs.
- Canonical extracted works define the author population, including enriched runs.
- Preserve source-first metadata preference, duplicate-source bibliographies, occurrence positions, hard author conflicts, and explicit separation constraints.
- Preserve slug identity history, correction bindings, `Journal_Canonical`, journal aliases, and current enrichment checks.
- Production corpora and user edits are never modified by implementation tests.
- Tests are offline. Corpus provider refresh is separately authorized and targeted.
- No promise of zero regressions or corpus accuracy follows from synthetic tests.

## Revision and execution status

Revised against local main `8fb5e18` on 2026-09-13. Fresh review verification was
568 passed, 9 skipped (optional matplotlib/networkx unavailable), with Ruff passing.
This records a baseline, not completion of the release checklist.

The requested order remains **validation → snapshots → provenance → clustering
→ corpus migration**. Corpus comparison moves into validation. Snapshot implementation
and a broader clustering rewrite have explicit decision gates. The previous mandatory
generation directory, export/import CLI, provenance lattice, per-work acceptance CLI
and complete-component clustering rule are withdrawn.

This revision changes documents only. Start implementation in an isolated checkout
when requested. Preserve these files and unrelated user changes; commit only scoped
implementation files after each task's tests pass.

## Files and interfaces

| Existing area | Planned responsibility |
| --- | --- |
| `scripts/evaluate_canonicalization.py` | Early corpus comparison; later offline ambiguity measurement |
| `src/citegraph/io.py` | Reuse schema/fingerprint helpers; relationship checks and pure cache inspection |
| `src/citegraph/graph.py` | Enforce complete-dataset checks and selected stale-output contract |
| `src/citegraph/pipeline.py` | Existing stage composition, scoped lock/recovery integration, evidence selection |
| `src/citegraph/html_report.py`, `webui.py` | Fault-tolerant findings; incomplete/stale status consistent with analysis |
| `src/citegraph/enrich.py`, `authors.py` | Offline author-agreement revalidation and consistent evidence attachment |
| `src/citegraph/cost_estimation.py` | Pure cache inspection with unchanged reuse eligibility |
| `src/citegraph/dedup.py`, `identity.py` | Only the measurement-supported clustering/continuity changes |
| `docs/USER_GUIDE.md`, `docs/RELEASE.md` | Extend existing migration/release instructions, not duplicate them |

New production files, public configuration knobs and storage interfaces are not
committed in advance. Small shared helpers stay beside their existing counterparts.
Each task below defines its required interface; no later task depends on an unspecified
snapshot API.

## Phase 1 — Validation

### Task 1: Freeze a comparable baseline

**Files:** existing tests/evaluator; `docs/RELEASE.md` only if verification instructions need correction.
**Consumes:** current checkout and existing fixture corpus.
**Produces:** commit/environment/test record and unchanged corpus copy, when supplied.

- [ ] Record `git rev-parse HEAD`, `git status --short`, interpreter and installed optional dependencies.
- [ ] Run the existing checks, retaining actual exit codes and output:

  ```bash
  .venv/bin/python -m pytest -o addopts='' -q -ra tests
  .venv/bin/ruff check .
  ```

- [ ] Run the fixture evaluator into a fresh temporary directory:

  ```bash
  review_dir=$(mktemp -d /private/tmp/citegraph-robustness.XXXXXX)
  .venv/bin/python scripts/evaluate_canonicalization.py --fixtures tests/fixtures/canonicalization --out "$review_dir/baseline.json"
  ```

- [ ] Identify loopback sandbox failures separately; rerun with permission when needed.
  Optional dependency skips remain incomplete release coverage, not code failures.
- [ ] Locate the actual current corpus before claiming corpus counts. The reviewed
  repository's root `out/` is legacy and lacks a current works-resolution audit.
  If the current corpus is unavailable, continue synthetic/tooling work and mark
  corpus-dependent decisions unmeasured.
- [ ] Follow the existing copy-only procedure for any real corpus. Preserve original
  bytes, caches, registries and corrections. Never run an extraction/enrichment stage
  merely to obtain a baseline.
- [ ] Review the baseline record before changing semantics; no source-code commit is required.

### Task 2: Deliver corpus comparison before other changes

**Files:** `scripts/evaluate_canonicalization.py`, `tests/test_canonicalization_evaluation.py`,
`docs/USER_GUIDE.md`.
**Consumes:** existing `partition(assignments)`, audit validation and fingerprint helpers.
**Produces:** `compare_corpora(before: Path, after: Path) -> dict`;
CLI `--compare BEFORE AFTER --out REPORT`, preserving existing modes.

- [ ] Add fixture tests with known membership split, pure ID rename, changed author
  assignment, edge removal and ranking change. Keep unchanged controls.
- [ ] Separate partition changes from ID changes with the existing helper:

  ```python
  assert partition({"obs-a": "w-old", "obs-b": "w-old"}) == partition(
      {"obs-a": "w-new", "obs-b": "w-new"}
  )
  assert partition({"obs-a": "w-old", "obs-b": "w-old"}) != partition(
      {"obs-a": "w-one", "obs-b": "w-two"}
  )
  ```

- [ ] Define report keys: `schema_version`, `before_fingerprints`,
  `after_fingerprints`, `lineage_status`, `uncomparable`, `memberships`,
  `ids`, `edges`, `author_assignments`, `metrics`, `rankings`.
  Change entries include before/after values and supporting observation coordinates.
- [ ] Align unchanged source/raw observations using saved input fingerprints and
  content/occurrence coordinates. Resolve reordered unique observations explicitly;
  report duplicate/changed observations without unique alignment as uncomparable.
  Do not equate a changed canonical ID with a changed underlying person/work.
- [ ] Include redirects, retired IDs, source promotion, journal differences and
  membership-normalized edge comparisons where lineage permits. Report raw table
  deltas even when membership lineage is unavailable.
- [ ] Test identical copies produce empty changes; pure renames produce ID changes
  but no membership changes; missing/stale audits produce explicit unavailable
  comparisons rather than guessed partitions.
- [ ] Refuse before/after aliases to the same directory and report overwrite. Test
  source directories' byte inventories are unchanged and provider calls are forbidden.
- [ ] Run `python -m pytest tests/test_canonicalization_evaluation.py -q`;
  review and commit this independently useful tool before validation changes.

### Task 3: Add missing relationship checks

**Files:** `src/citegraph/io.py`, `graph.py`, `html_report.py`, `pipeline.py`;
`tests/test_graph.py`, `test_report.py`, `test_reliability_pipeline.py`.
**Consumes:** `require_columns`, `read_stage_csv`, existing column constants.
**Produces:** `validate_graph_tables(works, edges, *, authors=None, author_citations=None) -> list[dict]`.
Each finding has `code`, `artifact`, `message` and affected `coordinates`.
Reuse the report's finding convention; no new class hierarchy.

- [ ] Add this boundary regression:

  ```python
  def test_graph_rejects_dangling_endpoint():
      works = pd.DataFrame([{"id": "w-a", "Title": "A", "ring": 0}]).set_index("id")
      edges = pd.DataFrame([{"citing_id": "w-a", "cited_id": "w-missing"}])
      with pytest.raises(ValueError, match="dangling_cited_id"):
          CitationGraph(works, edges)
  ```

- [ ] Run the test and confirm it fails because current graph construction accepts
  the endpoint. Implement set-difference foreign-key checks and ID uniqueness checks.
- [ ] Cover null/blank IDs, duplicate edges, partial author-table pairs, invalid
  author/work references, noninteger/negative positions and duplicate
  `(record_id, position)`. Validate position bounds against canonical extracted
  author lists when available; retain schema-only checks when evidence is absent.
- [ ] Normalize index-versus-column ID representations on a copy. Keep existing
  optional metadata and empty legacy behavior; never silently drop malformed rows.
- [ ] Use findings in graph construction and stage output validation; reports render
  them without failing. Preserve citations-only `canonicalize_works` fixtures.
- [ ] Keep stale-audit/publication validation separate for phase 2. Do not imply
  referential checks detect stale metrics when all IDs still happen to exist.
- [ ] Run graph/report/pipeline tests and task 2 comparison on healthy fixtures;
  require unchanged semantic output. Review and commit.

## Phase 2 — Snapshots and recovery

### Task 4: Measure crash windows and select the contract

**Files:** `tests/test_reliability_pipeline.py`, `tests/test_incremental_pipeline.py`,
`tests/test_resolution_audit.py`; revise this plan/spec with the decision.
**Consumes:** existing atomic writers, registry reconciliation and saved audits.
**Produces:** failure matrix and an explicitly selected recovery contract, not a storage module.

- [ ] Inventory coupled writes in metadata, dedup and authors, including
  `source_ids.json`, collision history, work/author identities, correction bindings,
  outputs and audits. Include `_migrate_author_identity` mutations before dedup.
- [ ] Inject exceptions before and after each replacement on temporary fixtures.
  A focused injection can wrap the existing writer:

  ```python
  original = pipeline_module.write_csv
  def interrupt_after_works(path, frame, **kwargs):
      original(path, frame, **kwargs)
      if path.name == "works.csv":
          raise RuntimeError("interrupted after works")
  monkeypatch.setattr(pipeline_module, "write_csv", interrupt_after_works)
  with pytest.raises(RuntimeError, match="interrupted after works"):
      pipe.deduplicate(sources, citations)
  ```

- [ ] For each interruption record surviving artifacts, registry/output agreement,
  graph/report behavior and whether retry safely preserves identity reservations.
  Use characterization tests for current behavior; do not claim the injections
  themselves fix it or commit unexplained failing tests.
- [ ] Include first-ever runs, existing outputs, unchanged IDs with changed metrics,
  invalid overrides, failure after identity writes, and failure between author tables.
- [ ] Demonstrate that moving registries last only reverses the mismatch, and a
  writer lock does not provide rollback.
- [ ] Present the two contracts from the spec: detect/refuse/recover versus retain
  last-good readability. Ask the user to select the availability requirement.
  Do not infer selection from the word “snapshots.”
- [ ] Record the choice, supported filesystems/platforms, reader participation and
  identity recovery before task 5. If undecided, defer task 5; independent provenance
  work may continue. This is a deliberate scope gate, not an incomplete design.

### Task 5: Implement only the selected recovery design

**Files:** `io.py`, `pipeline.py`, `graph.py`, `html_report.py`, `webui.py`,
`cli.py`, relevant tests and user guide; new storage file only if task 4 justifies it.
**Consumes:** task 4's approved contract and failure matrix.
**Produces:** selected recovery implementation with coherent reader behavior.

- [ ] Before coding, replace this conditional task with the selected concrete
  artifact ownership, helper signatures, commit/recovery order and test expectations.
  Review that amendment; do not implement both alternatives.
- [ ] Reuse per-file atomic writers and existing fingerprinted audits. Add an
  out_dir lock owned by the outermost pipeline operation; nested stages and cache
  workers share ownership. Test real subprocess contention and release on exit.
  Reject unsupported locking semantics explicitly; no untested portability claims.
- [ ] For detect/refuse/recover: specify minimal completion evidence, how incomplete
  first runs differ from legacy directories, which readers refuse stale output, and
  how coupled identity state is safely restored/rebuilt. Prevent reuse of partially
  advanced identity/correction state. Do not label write reordering a transaction.
- [ ] For retained last-good availability: specify the smallest retained artifact
  set, atomic selection, full/staged publication boundaries, immutable evidence and
  reader pinning. Ship readers with writers; preserve expensive caches separately.
  Snapshot report evidence must not be reconstructed from newer mutable caches.
- [ ] In either design, distinguish missing optional authors from stale authors
  and malformed present data. Preserve report access to diagnostics. Check user
  correction changes during an operation and preserve their edits.
- [ ] Document direct-CSV consistency limits. A marker does not make multiple reads
  atomic. Use a validated quiescent copy, cooperative lock or resolved immutable
  generation where consistency is required. Do not silently turn root CSV edits
  into unsupported behavior.
- [ ] Test exception and subprocess-death boundaries from task 4, safe retry,
  staged/composed equivalence, cache reuse, reader/writer interaction and recovery
  without provider calls. State power-loss durability limits separately.
- [ ] Require zero task 2 semantic deltas on healthy inputs. Review and commit only
  the selected implementation; no unused generation/export/import infrastructure.

## Phase 3 — Provenance

### Task 6: Make extraction-cache inspection pure and unknown recipes visible

**Files:** `io.py`, `cost_estimation.py`, `pipeline.py`, `html_report.py`;
`tests/test_incremental_cache.py`, `test_cost_estimation.py`.
**Consumes:** existing `read_pydantic_cache`, `cache_is_current`, extraction recipes.
**Produces:** `inspect_pydantic_cache(path, input_path, schema, *, many=False, recipe=None) -> dict`.
Fields: `value`, `source_path`, `reusable`, `recipe_status`, `reason`.
`recipe_status` is known-compatible, unknown or incompatible; this is local cache
diagnostics, not a cross-library trust hierarchy.

- [ ] Test estimates against sidecar-less and matching sibling caches, recording
  bytes before and after. Use the existing estimate fixture and this invariant:

  ```python
  before = {str(p.relative_to(out)): p.read_bytes()
            for p in out.rglob("*") if p.is_file()}
  pipe.estimate_extraction_cost()
  after = {str(p.relative_to(out)): p.read_bytes()
           for p in out.rglob("*") if p.is_file()}
  assert after == before
  ```

- [ ] Confirm failure from sidecar writing/sibling copying. Factor selection and
  validation into the pure inspector; execution alone materializes selected reuse.
  Both paths consume the same `reusable` result. Preserve current reuse eligibility.
- [ ] Keep same-stem legacy reuse but report unknown recipes. If execution records
  current hashes, preserve that recipes/original processing are unknown. Never stamp
  the currently requested model as historical evidence. Existing unknown sidecars
  need no forced rewrite to become visibly unknown.
- [ ] Preserve input/output hash invalidation and known recipe-change rejection.
  Keep unknown-recipe cross-filename reuse disallowed. Inspect effective recipe
  fields used by callers; amend only demonstrated omissions, with call-count tests.
- [ ] Surface unknown/incompatible counts in estimates/reports. Do not add three new
  CLI policy modes merely to fix a mutating read. Any strict/refresh interface is a
  separate evidenced need with preflight cost and original-cache preservation.
- [ ] Run incremental-cache and cost tests, including estimate/execution call-count
  agreement. Run task 2 comparison; expected memberships remain unchanged. Commit.

### Task 7: Revalidate cached enrichment without a blanket recrawl

**Files:** `enrich.py`, `pipeline.py`, `authors.py`, `io.py`, `html_report.py`;
`tests/test_enrichment_author_agreement.py`, `test_incremental_enrichment.py`,
`test_reliability_pipeline.py`.
**Consumes:** `author_list_agreement(extracted, provider, *, fuzz_threshold=85.0)`,
existing per-row input/output fingerprints and cache loader.
**Produces:** offline revalidation counts/evidence and a targeted refresh inventory.

- [ ] Add cached examples for matching people, disjoint people, corporate/empty
  abstention, missing original binding, and valid cache fallback behind stale CSVs.
  Preserve the current veto semantics:

  ```python
  assert author_list_agreement(["John Smith"], ["Jane Doe"]) is False
  assert author_list_agreement(["John Smith"], ["John Smith"]) is True
  assert author_list_agreement([], ["John Smith"]) is None
  ```

- [ ] Recheck saved winners against bound canonical extracted authors. Use provider
  authors from the saved result, not fields overwritten from the original row.
  Record input/result digests, current check version/config and match/mismatch/
  abstention/unresolved outcome in existing enrichment provenance/report structures.
- [ ] Reuse still-eligible bound winners under the author-veto-only change; abstention
  is not mismatch. Label old recipe/integrity limitations without inventing history.
  This check does not certify arbitrary scoring/provider changes or recover an
  unavailable candidate list.
- [ ] Exclude confirmed mismatches from author identifiers and provider-family lexicon
  evidence; retain independently valid rows. Prefer verified per-work fallback when
  a table row is stale or incomplete. Unbound rows are unresolved, not automatically
  trusted based on slug/name similarity.
- [ ] Share eligibility across pipeline author inputs and cache/CSV paths. Audit
  direct `normalize_authors(enriched_works=...)` callers too: preserve explicit
  caller-supplied evidence semantics, but do not let internally loaded unverified
  data bypass the policy by entering through that parameter.
- [ ] Recompute corpus counts offline. The historical 29/2,649 result is context,
  not an assertion about current data. Inventory affected works whose alternate
  candidate requires a new lookup; make zero provider calls in this task.
- [ ] Quantify unresolved exceptions before specifying an acceptance interface.
  If human acceptance is needed, amend the plan for digest-bound reviewed rows,
  optionally as one batch with a reason. No mandatory per-work approval UI and no
  blanket trust for future corpus changes. Do not replace existing hard conflicts.
- [ ] Run author/enrichment/journal/override tests and task 2 comparison; explain
  intentional evidence/identity differences. Review and commit independently of
  clustering. Preserve originals before any separately authorized refresh.

## Phase 4 — Clustering

### Task 8: Measure ambiguity on original observations

**Files:** `scripts/evaluate_canonicalization.py`,
`tests/test_canonicalization_evaluation.py`, `tests/test_reliability_dedup.py`.
**Consumes:** saved assignments and raw inputs, `_candidate_index_lookup`,
`_candidate_indices`, `_assess_work_match`, existing hard-conflict predicates.
**Produces:** `measure_work_ambiguity(directory: Path) -> dict`;
CLI `--measure-ambiguity DIR --out REPORT`, without changing production clustering.

- [ ] Preserve this characterization of the current seed mechanism in the diagnostic
  fixture (the existing test module supplies `record` and `cluster`):

  ```python
  unknown, early, late = [dict(record(), Year=y) for y in (0, 2020, 2023)]
  assert len(cluster([unknown, early, late])[0]) == 1
  assert len(cluster([early, unknown, late])[0]) == 2
  ```

- [ ] Verify saved input fingerprints before using audit assignments. Missing/stale
  audits mean historical comparison unavailable; an optional fresh offline run
  must be labeled new diagnostic output, not a recovered historical decision.
- [ ] Evaluate all blocked candidate pairs for diagnostics, including rows skipped
  after assignment. Inspect every pair within existing clusters for hard conflicts,
  including year conflicts absent from the saved decisions.
- [ ] Report counts and observation coordinates for hard-conflicting members,
  non-seed positive matches, observations matching multiple groups, and review
  candidates. Distinguish a positive pair from proof that a whole group can merge.
- [ ] Add deterministic permutations and independent outside-block samples. Report
  pair counts, runtime, sampling coverage and omitted-pair uncertainty; never call
  a blocked-only scan exhaustive or use it to claim blocking recall.
- [ ] Compare resulting experimental partitions/edges/rankings using task 2. Count
  affected works and estimated review volume, not just pair totals. Keep thresholds,
  weights and production blocking fixed.
- [ ] Run evaluator/dedup tests, verify original bytes unchanged and provider calls
  forbidden, then review/commit the diagnostic tooling.
- [ ] Freeze measurement results and labeled development/held-out cases before task 9.
  If the corpus is unavailable, report scope unmeasured and do not choose a broad
  replacement from the synthetic example alone.

### Task 9: Select and implement the smallest supported clustering correction

**Files:** `dedup.py`, `pipeline.py`, `identity.py` and existing dedup/identity/audit tests;
report/evaluator only where new decision evidence requires it.
**Consumes:** task 8 measurements, task 2 comparisons and existing release positives.
**Produces:** reviewed algorithm choice and versioned correction, not an automatic rewrite.

- [ ] Record whether measured defects justify a localized cluster-wide conflict guard
  or broader deterministic membership change. Specify handling of competing matches,
  source preference, representative selection, observation coordinates and IDs before
  coding. Review precision/recall, runtime and review-volume costs.
- [ ] Do not reinstate complete-component merging, exact-group preprocessing or an
  unresolved queue without evidence. Small counts can justify a small implementation;
  they do not justify accepting a demonstrated hard conflict.
- [ ] For the chosen correction, replace the known-defect characterization with a
  regression requiring incompatible dated records to stay separate under permutations:

  ```python
  for order in itertools.permutations(range(3)):
      rows = [dict(record(), Year=y) for y in (0, 2020, 2023)]
      _, _, stats = cluster([rows[i] for i in order])
      assigned = dict(zip(order, stats["citation_cluster_ids"], strict=True))
      assert assigned[1] != assigned[2]
  ```

- [ ] Enforce hard conflicts across proposed members, not only against the seed.
  Preserve unknown-year tolerance for compatible works, protected title conflicts,
  invalid-year rejection and threshold=101 behavior. Do not require every member
  pair to pass full fuzzy similarity merely to enforce hard constraints.
- [ ] Preserve source metadata preference, all source bibliographies, audit row
  coordinates and existing identity reconciliation. Separate any representative/slug
  changes from membership deltas; version changed algorithms and retain old audits.
- [ ] Test supported abbreviations/subtitles, competing seeds, source promotion,
  CSV roundtrips, repeated/permuted inputs, ID collisions, removals/reintroduction
  and overrides. Report remaining order sensitivity if choosing a narrow guard.
- [ ] Run full offline tests/evaluator and task 2 corpus comparison. Benchmark actual
  pair counts/runtime and review load. Do not weaken must-merge positives to pass a
  conservative algorithm. Review and commit; corpus adoption remains phase 5.

## Phase 5 — Corpus migration

### Task 10: Apply existing release gates and review copied-corpus deltas

**Files:** `docs/RELEASE.md`, `docs/USER_GUIDE.md`; only tests/CI needed for the
selected storage contract and supported platforms.
**Consumes:** baseline, early comparison tool, chosen recovery tests and scientific deltas.
**Produces:** offline migration report, reviewed adoption decision and rollback instructions.

- [ ] Reuse the release checklist: full optional-dependency coverage, Ruff, fixture
  evaluation, packaging checks, supported Python versions, cached/fresh and staged/
  composed equivalence, CSV roundtrips, registry stability and labeled accuracy gates.
- [ ] Add the selected crash/recovery/locking checks, not an unconditional new
  generation test matrix or platform expansion.
- [ ] Compare separate copies for validation/storage, then provenance, then clustering.
  Require unchanged semantics for healthy storage-only runs and explain later changes
  in memberships, IDs, edges, author assignments and rankings.
- [ ] Reuse existing corpus sampling/held-out rules; report unresolved labels,
  false merges/splits, retrieval omissions, review workload and runtime. Without
  labels, report corpus accuracy unverified and leave clustering migration unadopted.
- [ ] Inventory targeted refresh needs and costs; obtain separate authorization
  before any provider call. Preserve original caches instead of deleting them.
- [ ] Keep original data and compatible code for rollback. Do not resume writes from
  old registries after newer IDs were reserved without reviewed reconciliation of
  reservations/retirements and correction history.
- [ ] Document only implemented behavior and the selected direct-CSV consistency
  contract. Handoff includes exact comparisons/tests, unresolved decisions and
  migration artifacts; no automatic production adoption or deployment.

## Self-review and scope gates

- Validation and corpus comparison are tasks 1–3; no replacement schema framework.
- Snapshot scope is decided in task 4 before task 5 interfaces or storage code.
- Pure inspection and offline enrichment revalidation are tasks 6–7; no blanket
  recrawl, automatic recipe stamping or mandatory per-work acceptance UI.
- Task 8 measures omitted comparisons; task 9 selects the correction from evidence.
- Task 10 reuses existing release/migration gates and adds selected recovery checks.
- Conditional tasks require a reviewed amendment before implementation, not invented
  APIs or an executor silently choosing the more expansive alternative.
- Parallelize independent verification where helpful, not shared io/pipeline/identity edits.
