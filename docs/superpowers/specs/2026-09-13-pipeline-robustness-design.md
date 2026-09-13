# Pipeline robustness design

Status: revised after review against local main at `8fb5e18`, 2026-09-13.
Supersedes the unconditional generation layer and predetermined clustering rule
in the first draft. This is a planning revision, not authorization to change code.

## Outcome and scope

Keep the requested sequence: **validation → snapshots → provenance → clustering
→ corpus migration**. Bring corpus comparison into validation so every later
change has a baseline. Treat snapshots as a recovery-requirement decision, not
an assumed directory architecture. Measure clustering before selecting a replacement.

Verification at the reviewed commit: 568 tests passed, 9 skipped because optional
matplotlib/networkx dependencies were absent; Ruff passed. Loopback tests passed
on a permitted rerun. This is not a complete release gate; recheck HEAD and run
the existing release checklist before implementation/adoption.

## Contracts

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

## 1. Validation and early comparison

Reuse `require_columns`, table column constants, `read_stage_csv`, fingerprints,
saved audit checks, and the existing evaluator. Add relationship checks at complete
dataset boundaries: unique/nonempty IDs, valid work/author foreign keys, author
positions and occurrence uniqueness. Do not build a second schema framework.

Graph construction rejects invalid relationships; reports retain their fault-tolerant
behavior and render findings. Missing both optional author tables is valid; a partial
pair is not. Preserve schema-bearing empties, supported blank legacy files and optional
metadata omissions. Citations-only canonicalization remains a supported computational
utility; do not impose complete-source endpoint validation inside it.

Compare raw author positions only against canonical extracted evidence, never an
enriched display author's list. Do not expand public result schemas unless an actual
caller needs additional canonical evidence for a check.

Extend the evaluator now to compare two output directories: membership changes
separately from ID changes, redirects/retirements, edges, author assignments, metrics
and rankings. Align observations through saved lineage and unchanged source/raw
inputs, not fuzzy title guesses or assumed row order. Duplicate or changed observations
without unambiguous alignment are reported as uncomparable. Do not reconstruct missing
historical decisions. Report absolute changes even when lineage comparison is unavailable.

## 2. Snapshots: choose the recovery guarantee first

Current writes are individually atomic, not a multi-file transaction. Work and author
registries, correction bindings, tables and audits are coupled. Moving a registry write
last leaves the opposite mismatch possible; an output lock alone cannot repair a crash.
Audit detection is useful but is not yet enforced by every analysis reader.

Use failure injection to distinguish two contracts:

| Contract | Required behavior | Engineering consequence |
| --- | --- | --- |
| Detect and rebuild | Analysis refuses incomplete/stale output until a safe rebuild or restore; reports remain available | First assess existing audits plus minimal completion/recovery evidence, retaining root CSV paths |
| Retain last-good availability | After interruption, readers still access the previous complete dataset | Retain a coherent previous artifact set and identity state; a generation/pointer design becomes a candidate |

No contract is silently selected for the user. The decision record must show crash
results, reader behavior, recovery procedure and compatibility cost. Ask for the
required availability contract at that gate. Independent provenance diagnostics can
continue while storage implementation is deferred.

A detect-and-rebuild design must address identity recovery, not just block graph
queries: it must not reconcile from partially advanced registries or reuse reserved
IDs. If it cannot recover safely without retaining the previous coupled state,
retain that state or choose the stronger contract. An incomplete initial run must
not be mistaken for a valid legacy directory with no audit.

An out_dir writer lock is a small independent safeguard. Use stdlib OS advisory
locking for explicitly supported platforms, stable lock-file ownership, release on
process exit and actionable contention errors. The outer pipeline operation owns
the lock; nested stages and cache workers must not reacquire it. Do not claim
multi-host filesystem support or safety for independent cache writers.

Only after the decision, write the selected storage implementation details and tests.
Do not pre-create `snapshots.py`, layout overrides, generation manifests, export/import
commands or a generic transaction abstraction. If generations are selected, require
one resolution per logical read, coupled identity/output publication, staged versus
full-run semantics, stale dependency handling, correction-edit conflict checks and
separate live progress. Ship all affected readers with writers.

Direct pandas consumers of separate root CSVs have no group-atomic guarantee.
A completion marker alone does not pin multiple reads while another writer runs.
Document the need for a quiescent validated copy, a cooperative reader lock, or
one immutable resolved generation. Do not promise otherwise through compatibility
exports. Preserve manual edits and the current CSV workflow unless an explicit
compatibility migration is approved.

## 3. Provenance: local facts, offline revalidation

Estimation and execution already use the same cache reader and validity policy.
The defect is inspection-side mutation and loss of visibility into unknown recipes,
not separate policies. Legacy same-stem caches are intentionally reusable. Missing
recipe evidence must stay visibly unknown, not become evidence of the current model.
Existing input/output hashes still matter; these caches are not permanently current.

Split pure inspection from execution materialization while sharing the same selection
logic. Estimation must neither write sidecars nor copy sibling files. Continue recording
new processing recipes on actual extraction. Preserve legacy reuse without surprise
paid refresh. Surface unknown/mismatched recipes locally in estimates/reports; no
general provenance state machine threaded through every reader is required.

For enrichment, reuse `author_list_agreement` to recheck saved winners against bound
original metadata, offline. Record that this is a present compatibility check, not
proof of the original query/model/configuration. For the author-veto-only change,
a still-eligible old winner remains eligible; re-query only affected winners if an
alternative result is wanted. This does not prove equivalence after other scoring,
candidate retrieval, normalization or provider-data changes.

Distinguish mismatch from abstention: `None` is not a veto under current policy.
Preserve occurrence-level alignment and global author hard conflicts. Preserve valid
v2 evidence with its existing integrity limitations clearly labeled. Do not discard
an entire cache population solely because the new diagnostic is absent.

The committed enrichment design documents 29 vetoes among 2,649 matches; those are
historical measurements, not current counts. Recompute counts for the actual corpus.
If original binding/provider author data is unavailable, label the row unresolved;
do not infer binding from a slug or name match. Valid per-work caches can supply
evidence when a stale/partial CSV cannot. Excluded evidence must not leak through
provider family-name lexicon inputs or another author entry path.

Do not build a mandatory per-work acceptance UI. First automate checkable rows,
inventory unresolved exceptions and offer targeted refresh. If human acceptance is
actually needed, bind an explicit list of reviewed rows and their input/result
digests, with a reason. Batch review is allowed; blanket trust of future corpus
changes is not. Specify that workflow at the exception gate, before implementation.

## 4. Clustering: measurement before algorithm choice

Current work clustering is seed-based/star-shaped, not transitive closure.
A synthetic counterexample is reproduced: identical metadata dated unknown/2020/2023
forms one cluster when the unknown-year row seeds it, versus two with 2020 first.
This proves a missing cluster-wide conflict check, not its corpus prevalence.

Saved audits contain assignments and only some comparisons. Already-assigned rows,
year-pruned pairs and rejected assessments are omitted. Count hidden competitors
by evaluating original observations offline, recording those as new diagnostic
assessments rather than historical decisions.

Measure conflicting member pairs, observations compatible with multiple existing
groups, non-seed positive matches, permutation sensitivity, candidate-retrieval
omissions, affected work/edge/ranking counts, runtime and expected review volume.
Keep scores, thresholds and blocking unchanged for this measurement. Independently
sample pairs beyond blocking; a blocked scan cannot establish blocking recall.

Then choose the smallest supported correction. A cluster-wide hard-conflict guard
may suffice for a rare localized defect; a deterministic membership redesign needs
evidence of broader impact and acceptable false-split/review cost. The first draft's
complete-component rule, exact-group scheme and automatic unresolved queue are
withdrawn, not defaults awaiting implementation.

Regardless of scope, no accepted cluster may hide known-year, invalid-year or
protected-title conflicts through its representative. Preserve supported abbreviations
without requiring every member pair to pass the full fuzzy threshold. Keep representative
selection separate from membership and persistent identity reconciliation. Version any
changed algorithm; preserve caller observation coordinates and source preference.

Freeze measurement and labeled development/held-out examples before selecting a
new default. A rare failure still needs a correctness guard; frequency gates the
breadth of the rewrite, not whether a demonstrated hard conflict is acceptable.
If a narrow fix leaves order sensitivity, report it rather than claiming invariance.

## 5. Corpus adoption

Reuse `docs/RELEASE.md` and the existing copy-only migration procedure. Add only the
new comparison and chosen recovery checks. Validate storage, provenance and clustering
in separate copies/steps so scientific changes have attributable causes.

Storage changes must preserve semantic membership, IDs, edges and metrics on healthy
inputs. Explain provenance exclusions and clustering deltas separately. Require existing
must-separate/must-merge fixtures, staged/composed and cached/fresh equivalence, CSV
roundtrips, registry stability and applicable permutation tests. Human corpus accuracy
remains unverified without labels; do not adopt an unreviewed clustering migration.

Keep originals, cache observations, correction files and identity history. Do not
reset newer ID reservations to resume writes from an older dataset. Rollback uses a
compatible retained code/data pair; reconciling divergent newer history requires an
explicit reviewed recovery. No provider calls, destructive cache deletion or production
adoption are implied by executing the offline implementation plan.
