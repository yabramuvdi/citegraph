# Canonicalization reliability design

Status: proposed design accompanying the requested implementation plan; no implementation has started.

## Objective and approach

Prevent silent loss of source records and false author merges, make resumed stages reproduce fresh execution, and preserve enough evidence to inspect and correct canonicalization. Prefer an unresolved identity over an automatic merge that contradicts observed full names or identifiers.

Use incremental changes to the existing deterministic Python pipeline. A threshold-only patch would not fix perfect token-set scores, ID attachment errors, or checkpoint corruption. A learned entity-resolution system would require labeled training data and introduce substantial operational complexity. Neither is the recommended first step.

## Global constraints

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

## 1. Preserve sources and checkpoint schemas

Keep one source record per successfully processed file, including genuine duplicate PDFs. Give source observations unique provisional work IDs before references attach citing_id. Canonicalization subsequently decides which observations represent one work. Source-duplicate warnings are advisory and cannot remove bibliography evidence.

Keep make_work_id unchanged. Allocate its base ID or a numeric suffix using a source ID registry keyed by source_file plus the complete metadata fingerprint. Reserve previously allocated IDs, reuse an existing allocation for an unchanged observation, and allocate new collisions in a deterministic order by full metadata and source_file. Bootstrap from an existing sources.csv when its IDs can be associated unambiguously. Never resolve a collision by dropping a row. The registry contains source_file, metadata_fingerprint, and id, plus reserved historical IDs. A fresh rebuild is deterministic for an identical input set; incremental assignments are stable through the registry. Do not promise that an incremental registry and a fresh rebuild after corpus growth have identical suffixes.

All empty stage outputs retain their declared columns. Legacy completely empty CSVs can be read as schema-correct empty tables. A populated malformed CSV remains an error. An empty or all-failed metadata run raises an actionable StageNotReadyError before dependent processing, after preserving failures and a valid sources checkpoint. Zero references is a valid completed stage; zero authors is a valid result.

## 2. Apply enrichment consistently

One helper applies both cached and fresh provider results. Empty bibliographic provider fields do not erase extracted values; Authors and Authors_List remain consistent. Provider author records and display names are constructed from the same filtered entries.

Normalize external identifier representations at the boundary. New cache envelopes retain the original input fingerprint and the provider result. Unchanged legacy caches remain readable; legacy entries for IDs known to be collision-affected cannot be trusted and must be reported for refresh. A mismatched new fingerprint is a cache miss, never a silent attachment. Preserve existing transient-error retry semantics and legacy r- filename lookup.

Use the persisted canonical works table as the author-occurrence source in both Pipeline.run and standalone authors, and pass enrichment as evidence in both paths. This makes staged and composed execution agree and retains raw author spellings. Provider-only extra authors are not silently added in this change; record author-list disagreement for review. Enrichment should not create a different author population merely because the user chose run instead of individual stages.

## 3. Attach external author IDs safely

Build candidates for the whole author list. An admissible pair has compatible surname evidence and no incompatible full given-name evidence. A shared surname particle alone is insufficient. Exact normalized compound surnames or the existing corroborated single/compound relation provide surname evidence. Corporate and person records never cross-match.

First resolve mutually unique full-name matches, then mutually unique initial-compatible matches among unused entries. Position may corroborate a match but cannot override conflicting names or break an otherwise unresolved identity tie. An enrichment entry can be assigned at most once. Unresolved records keep no external ID and generate structured ambiguity/conflict evidence. A unique match is recomputed after each assignment; process a whole mutually unique round together so traversal order cannot change the result.

This deliberately abstains on two identically named coauthors when the available evidence cannot distinguish them. It does not invent an assignment from their order.

## 4. Merge author clusters with conflict checks

Separate positive compatibility from contradiction. A no-ID candidate must have supporting name evidence and must not contradict the informative full-name signatures in its proposed target. Initials cannot erase a John/Jane conflict. Apply the same conflict predicate during normal name clustering, external-anchor attachment, and compound-surname bridging.

Union directly supported external identities before attaching unidentified variants. Matching ORCIDs can reconcile multiple OpenAlex IDs, retaining and flagging the complete identifier set; conflicting known ORCIDs block automatic union even if an OpenAlex ID is shared. Distinct IDs without a direct shared identifier remain separate. Never choose one ID for display and discard contradictory evidence from the audit. Normalize IDs before comparing them.

Distinct positions in one work are evidence against a name-only merge. Preserve separate occurrences when that constraint cannot be satisfied, and flag repeated/ambiguous names. Shared external identity can explain repeated positions, but must produce a duplicate-authorship diagnostic. Coauthor tie-breaking is occurrence-specific: evidence from one J. Smith occurrence cannot assign all other J. Smith occurrences to that target.

Initial-only records may attach to a uniquely compatible full-name target. Existing attested full-name variants such as Juan Camilo/Camilo remain usable when directly supported by identity evidence; the conflict guard must not reject them simply because a name part was omitted. Add explicit positive fixtures alongside the John/Jane negative fixture.

## 5. Work matching and candidate selection

Normalize Unicode/diacritics, whitespace, author lists, and CSV representations once before scoring and blocking. Derive Authors from Authors_List when needed. Preserve non-Latin letters during comparison; normalization for matching must not mutate stored display strings or existing entity-ID generation.

Introduce an internal match assessment with decision match/review/reject, component scores, and reason codes. compare_papers remains a bool wrapper returning true only for match. Token-set containment cannot independently establish title equivalence. Use the whole title plus conservative structural checks: changes to negation, numbered parts/volumes, or a substantive expansion require review rather than automatic merging. The initial protected negation tokens are no, not, without, and non; part/volume markers include part, volume, vol, and book followed by Arabic or Roman numerals. These checks are a conservative guard, not a semantic classifier or an exhaustive multilingual guarantee.

Normalize leading articles for title comparison. Allow explicit colon/em-dash subtitle omission when the complete main titles match and author evidence supports identity, except protected-token conflicts. Preserve existing full-title fuzzy scoring for ordinary OCR/spelling variation; do not loosen thresholds to recover recall. Freeze the decision rules in fixtures and document any previously accepted positive case deliberately moved to review.

Candidate selection uses a union of normalized author keys and multiple title keys (full normalized title, article-stripped prefix, and sorted substantive title tokens). Missing-author comparisons are symmetric. Add a brute-force oracle on small fixtures to detect accepted pairs omitted by the candidate index. These routes improve tested recall; they do not guarantee exhaustive fuzzy retrieval over an arbitrary corpus. Bound performance with a deterministic comparison-count benchmark rather than a fragile CI wall-clock deadline.

Keep the current representative-based clustering model in this release, with conservative pair decisions and saved lineage. Preserve its deterministic behavior for a fixed ordered input. Test shuffled inputs to expose partition sensitivity and record differences; do not claim permutation invariance or replace the algorithm with transitive connected components. A broader clustering redesign requires corpus evaluation of those remaining differences.

## 6. Save actual decisions and support corrections

Write canonicalization_audit.json and author_resolution_audit.json at their respective stages, including standalone CLI runs. Canonicalization evidence contains every source/raw-citation observation's actual canonical ID, the representative relation, scores/reasons for accepted or review-worthy comparisons, collision assignments, and snapshot fingerprints. Do not persist every rejected comparison from a quadratic search. Author evidence contains each occurrence, normalized name, candidate attachments, assigned identifiers, conflicts, automatic decisions, and manual overrides. Keep summaries small in the HTML report; retain the complete evidence on disk.

Reports consume this evidence. Missing historical evidence is reported as unavailable; stale evidence is marked stale. Never rerun default canonicalization and label it the history of the saved graph. Existing report loading remains fault-tolerant and offline. Audit writes use temporary files and atomic replacement; fingerprints detect interrupted multi-file writes.

Preserve author_aliases.csv for merge-only compatibility. Add author_overrides.csv with columns action,left_record_id,left_position,right_record_id,right_position,reason. Each operand names an existing author occurrence in the fingerprinted works snapshot. action is merge or separate. Separate creates a cannot-link constraint; splitting a bad cluster means separating representative occurrences and reclustering the snapshot, not editing authors.csv. Expand constraints through forced-merge groups and reject inconsistent constraints before writing outputs. Invalid positions, unknown work IDs, cycles/contradictions in legacy aliases, or a stale override snapshot produce actionable errors rather than silently disappearing. Store author_overrides_meta.json with the works fingerprint when the override file is first successfully validated; later mismatches require explicit review and rebinding.

New separate constraints take precedence over heuristic and legacy-alias merges. An explicit merge may override name disagreement, with its reason recorded; it cannot override a separate constraint or conflicting ORCIDs. Full raw evidence remains available if corrected canonical IDs change.

## 7. Validation and rollout

Each fix gets a failing regression test before implementation and targeted checks afterward. Integration checks compare fresh/cached enrichment, in-memory/CSV stages, empty outputs, collision retention, manual corrections, and saved audit mappings. Tests must verify membership and identity, not just author/work counts.

Maintain labeled positive, negative, and ambiguous fixtures and an evaluator reporting pairwise precision/recall, false merges, false splits, candidate recall, and review volume separately for works/authors. Require no false merges on the must-separate regression set and preservation of existing explicitly supported positive name cases. Human-labeled corpus evaluation is a separate release gate: a deterministic stratified sample targets 150 author groups and 150 work groups (or all available if smaller), with common surnames, initials, compound surnames, identifier conflicts, similar titles, and ID collisions. Candidate pairs alone cannot measure blocking recall; include known duplicates found independently of candidate selection. Freeze a held-out subset before tuning. If labels are unavailable, report that limitation rather than inventing accuracy figures or calling quality validated.

Rebuild into a copied output directory, retaining extraction caches and original artifacts for comparison. Rebuild metadata, references, dedup, enrichment where used, authors, and report in order. References can need real calls if a formerly dropped source has no reference cache; enrichment can need real calls for invalidated or newly allocated IDs. Surface estimated calls/cost and use existing CLI confirmation behavior (scripted references require --yes). No automatic production-corpus overwrite or live paid calls is part of the implementation tests.

Compare retained sources, work/author memberships, core-to-core edges, self-loops, citation totals, and top-author rank changes. Map changed IDs using saved observation lineage. Keep old aliases intact and report entries needing migration. Source registry stability and correction validation must be checked before recommending the rebuild as a replacement for the original outputs.
