# Reliable incremental reruns

Approved scope: improve cache/enrichment reuse first, then persistent identities
and correction migration. Expensive observations survive cheap global rebuilds.

- Reuse conversion and extraction for unchanged content, including renamed PDFs.
  Detect replaced PDFs and record completed OCR attempts. Preserve legacy caches
  without forcing a corpus-wide refresh. New evidence includes input and output
  integrity and processing compatibility; hashes are cache keys, never entity IDs.
- Reuse validated enrichment by original metadata, including after ID changes.
  Author normalization retains independently verified unchanged rows even when
  other rows are new, changed, or corrupt; offline paths make no provider calls.
- Retain existing slug IDs for unambiguous work/author continuations. Clustering
  still uses current evidence. Record merges as redirects; retire split IDs and
  surface ambiguity instead of transferring annotations to an arbitrary child.
  Bootstrap from prior generated evidence when it can be validated. Existing
  standalone normalization functions retain their stateless behavior.
- Bind corrections to the referenced occurrences rather than unrelated corpus
  contents. Preserve conflict checks. Bootstrap legacy alias decisions against
  verified previous assignments; no automatic guessing from similar names.
- New artifacts belong to OutLayout. No dependencies or live provider calls.
  Preserve user edits and real corpus outputs. Legacy unverified history must be
  identified as such; do not manufacture historical assignments.

Consistent multi-file publication is a separate follow-up: this change preserves
per-file atomic writes and validates correction/identity changes before writing
author outputs. It does not introduce a snapshot directory or alter CSV paths.
