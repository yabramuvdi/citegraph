# Canonicalization Reliability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Execute sequentially with review checkpoints; delegation is not required by this plan.

**Goal:** Fix the reproduced pipeline failures and make work/author canonicalization conservative, reproducible, auditable, and correctable.

**Architecture:** Preserve the existing stage APIs and parsing foundation. Fix record retention and serialization first, then identity attachment and merge constraints, then work matching and persisted evidence. Add correction and evaluation support after the automatic decisions are trustworthy enough to inspect.

**Tech Stack:** Python >=3.11, pandas, Pydantic, rapidfuzz, pytest, existing fake Gemini clients and HTTP transports; no new runtime dependency.

**Spec:** [Canonicalization reliability design](../specs/2026-09-11-canonicalization-reliability-design.md). Status: proposed; this document is the requested plan, not an implementation authorization or a claim of completed fixes.

## Global Constraints

- Python >=3.11; no new runtime dependency is required.
- Keep the Gemini schemas, prompts, and per-stem metadata/reference cache formats unchanged.
- Preserve existing public return shapes and CSV columns; new diagnostics use additive fields and sidecars owned by OutLayout.
- Preserve the role-free w- and a- ID prefixes; never introduce hash-based or row-index-based entity IDs.
- Unaffected IDs remain unchanged. Record necessary collision-related ID changes and never silently reuse an ambiguous enrichment cache.
- Tests make no Gemini, CrossRef, or OpenAlex calls; use fake clients and transports.
- Preserve missing-year tolerance and fail-closed behavior for malformed nonempty years.
- Keep source representatives ahead of cited-only representatives, core-to-core edges, ring computation, and self-loop removal.
- Preserve corporate-author isolation and trusted compound-surname parsing.
- Every new artifact path belongs to OutLayout. New evidence artifacts carry schema version, algorithm version, effective configuration, and input/output fingerprints.
- Hashes may fingerprint artifacts or observations; they must not name canonical works or authors.

## Execution and review gates

Use an isolated feature branch/worktree at execution time. The planning turn changes only these documents. For each task: write the stated regression, run it and observe the relevant failure, implement, run focused tests, inspect the diff, and commit only the task's files after its checks pass. Tests shown below are minimum executable examples; the accompanying case lists are required coverage. Do not weaken assertions to accommodate a regression.

Release-sized groups: A = Tasks 1–3 (data/checkpoint integrity); B = Tasks 4–5 (author identity); C = Tasks 6–7 (work matching); D = Tasks 8–9 (audit/corrections); E = Task 10 (evaluation/rollout). Each task has its own regression cycle. Task 10's representative-corpus labeling can be prepared early, but tuning must not use the held-out labels.

## File responsibilities

| File | Responsibility |
| --- | --- |
| `io.py` | OutLayout paths, table schemas, safe empty-table reading, snapshot fingerprint and atomic JSON helpers |
| `ids.py` | Existing work slug generation plus source-ID collision allocation |
| `pipeline.py` | Stage integration, source registry, consistent author inputs, evidence persistence |
| `enrich.py` | Identical fresh/cache merge semantics, cache provenance, provider array alignment |
| `authors.py` | Name parsing, list-level enrichment matching, identity constraints, author metrics |
| new `author_overrides.py` | Occurrence constraints and legacy alias validation |
| `dedup.py` | Comparison assessment, candidate indexing, cluster lineage |
| `reports.py`, `html_report.py`, `webui.py` | Summaries and saved-evidence display; no inferred historical decisions |
| new `scripts/evaluate_canonicalization.py` | Offline quality evaluator and membership comparison |
| `tests/fixtures/canonicalization/` | Fixed positive/negative/ambiguous examples and label format |

Avoid unrelated module moves. Author matching can stay in authors.py during these fixes; extract a module only if it reduces dependencies without moving the name parser.

### Task 1: Preserve schemas on empty stages

**Files:** Modify `src/citegraph/io.py`, `pipeline.py`, `authors.py`, `cli.py`; test `tests/test_io.py`, `test_pipeline.py`, `test_failure_isolation.py`, `test_cli.py`, `test_graph.py`.

**Interfaces:** Add `read_stage_csv(path: Path, *, columns: list[str], index_col: str | None = None) -> pd.DataFrame` in io.py. Share explicit column constants for sources, raw citations, works, author records, and author occurrences. Missing required files remain StageNotReadyError at the stage boundary.

- [ ] Add a disk-roundtrip regression using the existing fake metadata client and a prewritten empty reference cache:

```python
def test_zero_references_checkpoint_can_resume(tmp_path):
    p = Pipeline(pdf_dir=None, out_dir=tmp_path, client=_FakeClient(), show_progress=False)
    md = p.layout.markdown_dir / "paper.md"
    md.write_text("A paper", encoding="utf-8")
    (p.layout.references_dir / "paper.json").write_text("[]", encoding="utf-8")
    sources = p.extract_paper_metadata([md])
    p.extract_paper_references([md], sources)
    works, edges = p.deduplicate()
    assert len(works) == 1
    assert edges.empty
    assert list(edges.columns) == ["citing_id", "cited_id"]
```

- [ ] Run `.venv/bin/pytest tests/test_pipeline.py::test_zero_references_checkpoint_can_resume -v`; verify the current EmptyDataError.
- [ ] Construct empty frames with declared columns and route default and explicit CLI CSV inputs through the shared reader. Catch only pandas EmptyDataError for historical blank files; validate populated files normally. Write failures first and stop an all-failed metadata pipeline with an actionable StageNotReadyError.
- [ ] Cover blank legacy citations, populated malformed citations, zero authors roundtripping through CitationGraph, and all-failed metadata. Verify an empty upstream checkpoint cannot leave an apparent successful downstream run summary.
- [ ] Run `.venv/bin/pytest tests/test_io.py tests/test_pipeline.py tests/test_failure_isolation.py tests/test_cli.py tests/test_graph.py`; lint the changed files and commit `fix: preserve schemas for empty pipeline checkpoints`.

### Task 2: Retain source observations and allocate collision-safe IDs

**Files:** Modify `src/citegraph/ids.py`, `io.py`, `pipeline.py`, `reports.py`; test `tests/test_ids.py`, `test_pipeline.py`, `test_failure_isolation.py`.

**Interfaces:** Add `assign_source_ids(records: list[dict], previous_registry: dict | None = None) -> tuple[list[dict], dict]` in ids.py; add `OutLayout.source_ids_json`. Registry schema is defined in spec section 1. Preserve make_work_id unchanged.

- [ ] Add an end-to-end fixture with the same author/year and these distinct titles:

```python
titles = [
    "Studies in economic and social behavior: agricultural irrigation water allocation and collective farming",
    "Studies in economic and social behavior: monetary inflation central banking interest rates and currency speculation",
]
assert make_work_id(["John Smith"], 2020, titles[0]) == make_work_id(["John Smith"], 2020, titles[1])
```

Write a metadata cache and a different one-reference cache for each stem. Assert two source rows, unique provisional IDs, both references retained, and both source files represented in canonicalization lineage. Separately verify compare_papers rejects this pair.

- [ ] Run `.venv/bin/pytest tests/test_pipeline.py -k 'collision or duplicate' -v` and observe the new retention test fail before editing pipeline behavior.
- [ ] Allocate IDs before writing sources.csv; remove ID-based row dropping. Preserve all valid files in the references stage. Keep source-duplicate warnings, update their wording so they do not imply skipped processing, and let canonicalization merge true duplicates.
- [ ] Reuse saved registry allocations and reserve all historical IDs. Bootstrap existing unambiguous source allocations. Persist the registry atomically; mark collision groups for enrichment provenance checks in Task 3.
- [ ] Test reruns, input permutations with/without a registry, adding/removing a collision, identical duplicate PDFs, duplicate PDFs with complementary reference caches, and malformed same-slug records that must remain distinct. Confirm every resulting graph endpoint joins to works for valid pipeline inputs.
- [ ] Run `.venv/bin/pytest tests/test_ids.py tests/test_pipeline.py tests/test_failure_isolation.py tests/test_dedup.py`; commit `fix: retain source records across work id collisions`.

### Task 3: Make enrichment and author-stage resumption consistent

**Files:** Modify `src/citegraph/enrich.py`, `pipeline.py`, `io.py`; test `tests/test_enrich.py`, `test_pipeline.py`, `test_authors.py`.

**Interfaces:** Add `_apply_enrichment_result(row: dict, result: dict) -> dict` in enrich.py. New cache envelope: `{schema_version: 2, input_fingerprint: str, result: dict}`. Legacy cache parsing returns the same result dictionary; collision-affected legacy entries return a reported cache miss. Author occurrences always come from canonical works, with enriched_works passed separately as evidence.

- [ ] Add a fake-provider fresh/warm regression:

```python
def test_empty_provider_fields_do_not_change_warm_result(tmp_path, monkeypatch):
    from citegraph.enrich import EnrichConfig, _LookupReport, _enrich_one
    row = pd.Series(dict(Title="Study", Authors="John Smith", Authors_List=["John Smith"],
                         Journal="Known Journal", Year=2020))
    match = dict(doi="10.example/test", enrichment_source="crossref", Title="Study",
                 Authors="", Authors_List=[], Journal="", Year=None, OpenAlex_Authors=[])
    monkeypatch.setattr("citegraph.enrich._crossref_lookup_report",
                        lambda *args, **kwargs: _LookupReport(match=match))
    cold = _enrich_one("w-example", row, EnrichConfig(), None, tmp_path)
    warm = _enrich_one("w-example", row, EnrichConfig(), None, tmp_path)
    for key in ("Authors", "Authors_List", "Journal", "Year"):
        assert cold[key] == warm[key] == row[key]
```

- [ ] Run the new test and verify the warm-result mismatch.
- [ ] Apply provider results through the shared helper in every path. Build provider names and structured author entries from one filtered list. Normalize IDs consistently, preserving all observed ID evidence for later conflict reporting.
- [ ] Fingerprint normalized original inputs; read version-2 and legacy payloads without changing retry semantics. Test stale/new/collision legacy entries, r- fallback, transient http_error retry, and nonempty legitimate provider corrections.
- [ ] Make Pipeline.run use pre-enrichment canonical works for author occurrences, matching standalone authors. Persist and validate enrichment's works-input fingerprint; a stale table cannot lend external IDs to changed canonical observations. Old unverified tables may be loaded for inspection but must report uncertainty for collision-affected rows.
- [ ] Compare composed and staged author memberships using a provider result with reordered names and an extra author. Preserve extracted occurrences and flag disagreement. No live provider calls are needed.
- [ ] Run `.venv/bin/pytest tests/test_enrich.py tests/test_pipeline.py tests/test_authors.py`; commit `fix: unify cached enrichment and staged author inputs`.

### Task 4: Match enrichment authors one-to-one

**Files:** Modify `src/citegraph/authors.py`; test `tests/test_authors.py`.

**Interfaces:** Replace independent `_match_enrichment` calls with `_match_enrichment_authors(parsed: list[ParsedAuthor | None], enrichment: list[dict], known_surnames: frozenset[str], audit: list[dict]) -> list[tuple[str | None, str | None]]`. Output length and positions equal parsed; unmatched slots are `(None, None)`. Add optional `audit: list[dict] | None = None` to normalize_authors without changing its returned tuple.

- [ ] Add the reproduced missing-coauthor case:

```python
def test_partial_enrichment_does_not_merge_smith_coauthors():
    works = pd.DataFrame([dict(id="w-1", ring=0,
                              Authors_List=["John Smith", "Jane Smith"])]).set_index("id")
    enriched = pd.DataFrame([dict(id="w-1", OpenAlex_Authors=[
        dict(display_name="John Smith", openalex_id="A1", orcid=None)
    ])]).set_index("id")
    authors, occurrences, review = normalize_authors(works=works, enriched_works=enriched)
    assert occurrences.author_id.nunique() == 2
    jane_id = occurrences.loc[occurrences.raw_author == "Jane Smith", "author_id"].iloc[0]
    assert pd.isna(authors.loc[jane_id, "openalex_id"])
```

- [ ] Run the regression and observe that the current implementation collapses both names.
- [ ] Implement the admissibility predicates and mutually unique assignment rounds from spec section 3. Preserve parsed positions even when entries are unusable. Save candidate/rejection reasons; feed conflict/ambiguity findings into author_review even when another occurrence has an external ID.
- [ ] Test reversed same-surname lists, two indistinguishable names, missing middle names, particles shared by unrelated compounds, blank provider authors, unequal list lengths, corporate names, and exact/corroborated compound surname matches. Assert IDs by raw_author and position, not only cluster counts.
- [ ] Run `.venv/bin/pytest tests/test_authors.py tests/test_enrich.py tests/test_pipeline.py`; commit `fix: attach author identifiers with one-to-one matching`.

### Task 5: Prevent contradictory author-cluster merges

**Files:** Modify `src/citegraph/authors.py`; test `tests/test_authors.py`.

**Interfaces:** Add `_author_clusters_conflict(left: list[AuthorOccurrence], right: list[AuthorOccurrence]) -> bool`. It enforces full-name, external-ID, and same-work occurrence constraints under spec section 4. Record reason details through the audit collector. Apply it to every automatic merge pass rather than only the final anchor pass.

- [ ] Add a fixture with separate works carrying John Smith/A1, J. Smith/A1, and Jane Smith/no ID. Assert John and J. share an author, Jane is separate, and the prevented attachment is visible in review/audit. Repeat with input reversed.
- [ ] Add the existing supported full-name-variant positive case explicitly:

```python
def test_attested_full_name_variant_still_anchors_unidentified_author():
    works = pd.DataFrame([
        dict(id="w-full", ring=0, Authors_List=["Juan Camilo Cardenas"]),
        dict(id="w-variant", ring=0, Authors_List=["Camilo Cardenas"]),
        dict(id="w-unidentified", ring=1, Authors_List=["Camilo Cardenas"]),
    ]).set_index("id")
    enriched = pd.DataFrame([
        dict(id="w-full", OpenAlex_Authors=[
            dict(display_name="Juan Camilo Cardenas", openalex_id="A1")]),
        dict(id="w-variant", OpenAlex_Authors=[
            dict(display_name="Camilo Cardenas", openalex_id="A1")]),
    ]).set_index("id")
    _, occurrences, _ = normalize_authors(works=works, enriched_works=enriched)
    assert occurrences.author_id.nunique() == 1
    assert set(occurrences.record_id) == {"w-full", "w-variant", "w-unidentified"}
```

- [ ] Run `.venv/bin/pytest tests/test_authors.py -k 'conflict or variant or anchor' -v`; confirm the John/Jane regression fails before implementation.
- [ ] Union validated IDs first, reject conflicting ORCIDs, retain all identifier evidence, and guard weaker name-based joins. Implement full-name compatibility allowing omitted attested name parts while rejecting conflicting observed words; never use any initial-only signature to bypass that contradiction.
- [ ] Resolve coauthor-assisted ambiguous initials per occurrence rather than assigning an entire signature bucket from one overlapping coauthor. Keep same-work distinct positions separate unless direct consistent external identity supports duplication, which must be flagged.
- [ ] Test J./J.C. without full evidence, John/Jane, Gabriel/Gabriela, compound bridging, two same-name people with different IDs, union through ORCID, conflicting ORCIDs sharing an OpenAlex ID, normalized URL IDs, identical-name coauthors, and coauthor tie-breaking across two ambiguous occurrences.
- [ ] Run `.venv/bin/pytest tests/test_authors.py tests/test_graph.py tests/test_pipeline.py`; inspect metrics and review flags; commit `fix: enforce author identity conflicts across merge passes`.

### Task 6: Normalize work evidence and recover blocked candidates

**Files:** Modify `src/citegraph/dedup.py`, `io.py`; test `tests/test_dedup.py`, `test_pipeline.py`.

**Interfaces:** Add `_normalize_work_record(record: dict) -> dict` in dedup.py; parse Authors_List via existing IO utilities, derive Authors, and normalize evidence without changing stored display metadata or make_work_id. Candidate-index APIs stay internal.

- [ ] Add a pair fixture for José García/Jose Garcia, with titles Effects of monetary policy on employment / The effects of monetary policy on employment, same year/journal. Assert the matcher accepts it and canonicalization yields one cluster. Add identical Authors_List-only rows and compare in-memory versus CSV-restored inputs.
- [ ] Verify the current candidate omission and list-only mismatch with `.venv/bin/pytest tests/test_dedup.py -k 'blocking or list_only or roundtrip' -v`.
- [ ] Share normalized representations between candidate selection and scoring; add the title/author candidate routes in spec section 5. Remove asymmetric handling of unknown-author representatives. Preserve non-Latin letters with Unicode normalization; do not change existing ID slugs.
- [ ] Add a brute-force retrieval assertion to a small fixture suite:

```python
for i, left in enumerate(records):
    for j, right in enumerate(records):
        if i != j and compare_papers(left, right, DedupConfig()):
            assert j in candidates_for_row[i]
```

Build candidates_for_row through the actual index helper for the fixture DataFrame. Include spelling, accents, abbreviated first authors, authors in different orders, unknown authors, and leading title articles.

- [ ] Add a deterministic comparison-count benchmark on 1,000 diverse generated rows; compare with the n*(n-1)/2 all-pairs baseline and fail if the fixture degenerates to exhaustive comparisons. Record worst-case common-surname behavior without pretending indexing eliminates that case.
- [ ] Run `.venv/bin/pytest tests/test_dedup.py tests/test_pipeline.py`; commit `fix: normalize work matching and broaden candidate retrieval`.

### Task 7: Reject unsafe title-containment merges

**Files:** Modify `src/citegraph/dedup.py`, `reports.py`; test `tests/test_dedup.py`, `test_pipeline.py`.

**Interfaces:** Add `_assess_work_match(left: dict, right: dict, cfg: DedupConfig) -> dict` with keys decision (`match`, `review`, `reject`), reason_codes (list[str]), title_score/authors_score/journal_score/weighted_score (float), and year_ok (bool). compare_papers remains its bool wrapper. Store accepted/review assessments in the existing stats dictionary under additional keys.

- [ ] Add the direct false-merge regression:

```python
def test_negation_changes_are_not_automatic_work_matches():
    a = dict(Title="Effects of monetary policy on employment", Authors="John Smith",
             Authors_List=["John Smith"], Journal="Economics", Year=2020)
    b = dict(a, Title="No effects of monetary policy on employment")
    assert not compare_papers(a, b, DedupConfig())
```

- [ ] Verify it fails. Add part I/II, opposite substantive expansions, same-title unknown years, malformed years, explicit subtitle omission, and OCR typo positive cases.
- [ ] Implement the conservative title decision rules in spec section 5 before the weighted-score acceptance. Do not substitute a higher global threshold. Preserve existing cfg fields; this task adds no uncalibrated public tuning knobs. If later evaluation justifies a public option, expose it consistently on Pipeline and every relevant CLI in a separate reviewed change.
- [ ] Have source-duplicate detection call the same assessment policy. A review decision keeps records apart and records the pair; it must not become an asserted duplicate warning.
- [ ] Run `.venv/bin/pytest tests/test_dedup.py tests/test_pipeline.py tests/test_report.py`; enumerate positive cases intentionally moved to review in the changelog; commit `fix: require stronger evidence for title containment merges`.

### Task 8: Persist lineage and display the actual merge audit

**Files:** Modify `src/citegraph/io.py`, `pipeline.py`, `dedup.py`, `authors.py`, `reports.py`, `html_report.py`; test `tests/test_io.py`, `test_pipeline.py`, `test_report.py`, `test_webui.py`.

**Interfaces:** Add OutLayout properties `canonicalization_audit_json` and `author_resolution_audit_json`. Use `stats["citation_cluster_ids"]` plus a new `stats["source_cluster_ids"]` to serialize actual assignments. Author evidence comes from the optional audit collector added in Task 4. Shared audit header: schema_version, algorithm_version, config, input_fingerprints, output_fingerprints.

- [ ] Add a report regression with a nondefault threshold that changes membership. Persist the stage's mappings and assert the report displays precisely those mappings. Guard against recomputation:

```python
def forbidden_recanonicalization(*args, **kwargs):
    raise AssertionError("Report must not recompute historical merges")
monkeypatch.setattr("citegraph.dedup.canonicalize_works", forbidden_recanonicalization)
monkeypatch.setattr("citegraph.html_report.canonicalize_works",
                    forbidden_recanonicalization, raising=False)
```

Patch both the original function and the report's imported alias so the test also catches a future local import. raising=False keeps the test valid when the report's dependency is removed.

- [ ] Observe the current report path fail that regression. Persist source and citation lineage, match evidence, author occurrence decisions, collisions, and complete observed identifiers. Include settings/fingerprints for standalone stages as well as run. Atomic replacement prevents partially written JSON; fingerprints identify mismatched CSV snapshots.
- [ ] Render accepted merges and unresolved/conflicting evidence from saved audits. Missing legacy evidence displays unavailable; mismatched fingerprints display stale. Neither condition creates guessed historical membership. Keep existing table shape/report fault tolerance intact where possible.
- [ ] Test changed source/citation/works files, interrupted audit writes, corrupt evidence, nondefault config, standalone CLI generation, and the web UI reading the same saved evidence. Ensure a large rejected-pair space is not serialized.
- [ ] Run `.venv/bin/pytest tests/test_io.py tests/test_pipeline.py tests/test_report.py tests/test_webui.py`; commit `feat: persist canonicalization lineage and resolution evidence`.

### Task 9: Add safe separation and merge corrections

**Files:** Create `src/citegraph/author_overrides.py`; modify `authors.py`, `pipeline.py`, `io.py`, `html_report.py`; create `tests/test_author_overrides.py`; extend `tests/test_authors.py`, `test_pipeline.py`, `test_report.py`.

**Interfaces:** `load_author_constraints(path: Path, *, works: pd.DataFrame, metadata_path: Path) -> list[dict]` validates the spec's occurrence-based CSV and snapshot metadata. Add optional `constraints: list[dict] | None = None` to normalize_authors. Add OutLayout properties `author_overrides_csv` and `author_overrides_meta_json`. Retain aliases as a supported argument/file format.

- [ ] Add a merge/separate fixture using this exact CSV schema:

```csv
action,left_record_id,left_position,right_record_id,right_position,reason
separate,w-1,0,w-2,0,Two distinct people confirmed from the papers
merge,w-1,0,w-3,0,Same person confirmed from the papers
```

Assert w-1/w-3 share an author, w-2 stays separate, and all three occurrences remain present. Test the full Pipeline path, not only the parser.

- [ ] Run `.venv/bin/pytest tests/test_author_overrides.py -v` and observe missing functionality. Build forced-merge components, expand separate constraints between those components, and reject contradictions before clustering or writing output artifacts. Validate positions and work IDs against the current works snapshot.
- [ ] Apply separate constraints at every automatic union, including external-ID union and legacy aliases. Explicit merge can override name evidence but not conflicting ORCIDs or an explicit separate constraint. Reject legacy alias cycles/stale targets with an actionable diagnostic. Preserve the old file rather than rewriting it.
- [ ] Save the first validated snapshot binding atomically; require explicit rebinding after fingerprint changes. Surface the exact affected rows and current snapshot in the error so the user can review them. Record every applied correction and its reason in audit output.
- [ ] Test transitive merges, separate constraints through a merge chain, same-work positions, stale bindings, stale aliases, conflicting IDs, aliases that violate separation, repeated execution, and unchanged metrics after a no-op correction.
- [ ] Run `.venv/bin/pytest tests/test_author_overrides.py tests/test_authors.py tests/test_pipeline.py tests/test_report.py`; commit `feat: support validated author separation and merge overrides`.

### Task 10: Evaluate accuracy, document migration, and verify release

**Files:** Create `scripts/evaluate_canonicalization.py`, `tests/test_canonicalization_evaluation.py`, `tests/fixtures/canonicalization/works.json`, `authors.json`; update `docs/USER_GUIDE.md`, `docs/RELEASE.md`, and existing repository guidance describing affected behavior.

**Interfaces:** Evaluation input JSON uses records plus labeled pairs `{left: str, right: str, label: same|different|ambiguous}` and observation-to-predicted-cluster mappings. `evaluate_pairs(labels: list[dict], assignments: dict[str, str]) -> dict` returns TP/FP/FN/TN, precision, recall, and ambiguous_count; zero denominators return null rather than perfect accuracy. Ambiguous labels are excluded from scored totals. Candidate recall and review volume are reported separately by the command.

Fixture files are JSON lists of cases. Each work case has case_id, sources (normal source records), citations (normal raw-citation records plus observation_id), and labels whose operands use source observation IDs or citation observation_id. Each author case has case_id, works (canonical records with id), enriched_works (records with id and OpenAlex_Authors), and labels whose operands are `record_id:position`. Strip fixture-only observation_id before invoking production code; recover memberships from saved lineage or returned author occurrences. Do not merge unrelated cases into one corpus. CLI: `.venv/bin/python scripts/evaluate_canonicalization.py --fixtures tests/fixtures/canonicalization --out /private/tmp/citegraph-evaluation.json`. Sampling mode uses `--corpus-out <existing-output-directory> --sample-out <review.json>` and never invokes providers or assigns labels.

- [ ] Add hand-checkable metric tests before writing the evaluator:

```python
labels = [dict(left="a", right="b", label="same"),
          dict(left="a", right="c", label="different")]
metrics = evaluate_pairs(labels, {"a": "x", "b": "x", "c": "x"})
assert metrics["tp"] == 1
assert metrics["fp"] == 1
assert metrics["precision"] == 0.5
assert metrics["recall"] == 1.0
```

- [ ] Build fixed author/work fixtures from all reproduced bugs and existing supported cases. Add partition comparison independent of label spelling, fixed-input repeatability, CSV roundtrip, fresh/cache equivalence, and source-registry ID stability. For shuffled work inputs, report partition differences rather than silently claiming order invariance. Explicitly inspect any changed author partition under shuffled inputs.
- [ ] Add a deterministic stratified sample generator for the spec's representative-corpus evaluation. It outputs unlabeled review cases; it never invents ground truth. Freeze a held-out portion and require human labels for quality claims. Report false merges/false splits, retrieval omissions, review workload, and changes to citation counts/rankings. The mandatory regression gate is zero false merges on must-separate fixtures and preservation of supported positive cases.
- [ ] Document source registry semantics, provenance, new review reasons, override precedence, stale snapshot handling, exact rebuild commands, and remaining representative-order sensitivity. Rebuild only into a copy of an actual output directory. Reuse per-stem caches; report missing reference caches and invalidated enrichment before any provider calls. Scripted references use `--yes` only when that work is authorized.
- [ ] Verify in an environment containing the project's dev extras. Run `.venv/bin/pytest -o addopts=''` and `.venv/bin/ruff check .`; require optional graph/plot tests to execute for the release check. Test HTTP servers need loopback access. Distinguish environment restrictions from code failures.
- [ ] Run the offline evaluation CLI, inspect membership/ranking differences and saved audit completeness, then commit `test: add canonicalization quality evaluation and migration guidance`.

## Completion criteria

1. Every reproduced bug has a regression that failed before its fix and passes afterward.
2. No successful source or author occurrence is silently dropped; every valid pipeline graph endpoint exists in works.
3. John/Jane cannot merge through surname-only enrichment or an initials bridge; supported compound-name and external-ID positive cases still work.
4. Fresh/cached and staged/composed execution agree on canonical author membership for the same snapshot.
5. Empty valid stages resume; malformed populated checkpoints remain explicit errors.
6. Reports show saved decisions under the actual configuration, with stale/missing evidence identified.
7. Corrections can separate previously conflated occurrences without editing generated tables, and contradictory corrections fail before output mutation.
8. Lint and the full installed-extra test suite pass. Evaluation reports distinguish synthetic regression guarantees from held-out corpus accuracy.
9. Any remaining order sensitivity, unresolved ambiguity, missing human labels, ID migrations, or required API refresh is stated explicitly in the implementation handoff.
# Implementation status — 2026-09-12

Tasks 1–10 are implemented and verified. The checklists below preserve the
original execution plan; current outcomes, test evidence and remaining corpus
validation requirements are recorded in
[implementation results](2026-09-12-canonicalization-reliability-results.md).
