# UI and Application Boundary

Status: proposed architecture for the first researcher-facing application.
An MVP slice now exists in the package: `citegraph ui` serves a read-only
live monitor over the same data contract as `citegraph report`
(`collect_report_data`), refreshing as artifacts appear in the out
directory. The job service / control plane described below (create, cost
approval, cancellation, review decisions) remains future work and is not
implemented.

## Decision

Keep `citegraph` as the independent analysis engine and build the user
interface as a separate application that depends on it.

The first UI should be a local application: a small local service owns jobs and
calls the Python library, while a browser-based frontend presents file
selection, progress, review decisions, and exploration. This keeps PDFs and
artifacts on the researcher's computer, avoids operating a multi-user upload
service, and preserves one implementation of the scientific workflow.

A hosted multi-user service can reuse the same application contract later, but
it adds authentication, storage isolation, quotas, deletion policy, secret
management, and research-data governance. Those are deliberately outside the
first UI milestone.

Streamlit or a similar notebook-style framework is reasonable for a disposable
workflow prototype. It is not the recommended long-term boundary because job
resumption, background work, cancellation, and explicit review states are core
requirements rather than presentation details.

## Architectural boundary

```text
Researcher
    │
    ▼
Local web UI
    │ JSON commands and event stream
    ▼
citegraph application service
    ├── job state database (SQLite)
    ├── secret reference (OS keychain or process environment)
    └── citegraph Python library
            ├── filesystem artifacts and caches
            ├── Gemini
            └── optional CrossRef / OpenAlex
```

The UI must never import internal pipeline modules or infer progress by parsing
terminal output. The application service translates stable library operations
into serializable commands, states, events, review items, and result summaries.

CSV and JSON artifacts remain the portable research record. SQLite stores only
application state such as job status, user decisions, and event history; it is
not the canonical bibliographic dataset.

## Current assets and missing seams

The library already provides useful foundations:

- independently callable, checkpointed pipeline stages;
- `StageNotReadyError` with upstream-stage guidance;
- deterministic IDs and a centralized `OutLayout`;
- warning and failure sidecars;
- `run_summary.json` and `artifact_manifest.json`;
- `CitationGraph.from_out_dir()` for result exploration;
- a preflight cost estimate.

The application layer still needs:

- structured progress events instead of Rich-only progress bars;
- durable job and stage state;
- cost approval as an explicit state transition;
- cooperative cancellation between work units;
- normalized review items and recorded review decisions;
- a stable, serializable service API around `Pipeline` and `CitationGraph`.

These seams should be added without moving scientific logic into the UI.

## Domain contracts

The names below are the proposed stable application vocabulary. Exact storage
models may differ, but the public meanings should not.

### Job specification

```json
{
  "input_dir": "/data/project/pdfs",
  "out_dir": "/data/project/citegraph-output",
  "model": "gemini-3.1-flash-lite",
  "recursive": true,
  "ocr": "auto",
  "enrich": true,
  "llm_concurrency": 4,
  "author_merge_mode": "strict",
  "dedup": {
    "threshold": 85.0,
    "title_weight": 0.7,
    "authors_weight": 0.3,
    "journal_weight": 0.0,
    "year_window": 1
  }
}
```

The API key is not part of `JobSpec`. The service receives a secret reference
or reads the process environment; it never persists or returns the key.

### Job state

```text
draft
  → converting
  → estimating
  → awaiting_cost_approval
  → extracting_metadata
  → extracting_references
  → deduplicating
  → enriching            (optional)
  → normalizing_authors
  → review_required      (when actionable review items exist)
  → completed
```

Any running state may move to `cancellation_requested`, then `cancelled` at a
safe boundary. Unrecoverable service errors move the job to `failed`; isolated
paper failures remain review items and do not automatically fail the job.

Each stage also records `pending`, `running`, `completed`, `completed_with_issues`,
`failed`, `skipped`, or `cancelled`, plus start/end times and output paths.

### Progress event

```json
{
  "event_id": 184,
  "job_id": "job-01J...",
  "timestamp": "2026-09-04T10:00:00Z",
  "type": "work_item_completed",
  "stage": "references",
  "completed": 12,
  "total": 40,
  "item": "paper-12.pdf",
  "message": "References extracted"
}
```

Event types for the MVP:

- `job_state_changed`;
- `stage_started`, `stage_completed`, `stage_failed`;
- `work_item_started`, `work_item_completed`, `work_item_failed`;
- `cost_estimate_ready`, `cost_approved`;
- `review_items_changed`;
- `log_message` for diagnostics that do not drive UI state.

Events have monotonically increasing IDs per job so a reconnecting UI can ask
for everything after its last received event.

### Review item

All review surfaces share one envelope:

```json
{
  "review_id": "review-01J...",
  "job_id": "job-01J...",
  "kind": "enrichment_miss",
  "stage": "enrich",
  "severity": "warning",
  "status": "open",
  "subject_id": "r-vaswani-2017-attention-is-all-you-need",
  "summary": "No external match passed the configured threshold",
  "evidence": {},
  "allowed_decisions": ["accept_unmatched", "retry", "edit_query"]
}
```

MVP review kinds map to existing artifacts:

| Review kind | Existing source |
| --- | --- |
| `conversion_quality` | `conversion_warnings.json` |
| `source_duplicate` | `source_duplicates.json` |
| `no_references` | `papers_no_references.json` |
| `paper_failure` | metadata/references failures JSONL |
| `enrichment_miss` | `enrichment_misses.csv` |
| `author_identity` | `author_review.json` |

Every decision records timestamp, selected action, optional note, and the
resulting durable artifact. For author merges, the durable artifact is
`author_aliases.csv`; the application database is only an audit trail.

## Service interface

The first implementation can expose these operations as Python methods and map
them one-to-one to local HTTP endpoints:

```python
class CitegraphService:
    def create_job(self, spec: JobSpec) -> Job: ...
    def get_job(self, job_id: str) -> Job: ...
    def estimate(self, job_id: str) -> CostEstimate: ...
    def approve_cost(self, job_id: str, estimate_id: str) -> Job: ...
    def start(self, job_id: str) -> Job: ...
    def resume(self, job_id: str) -> Job: ...
    def request_cancellation(self, job_id: str) -> Job: ...
    def events(self, job_id: str, after: int | None = None) -> list[RunEvent]: ...
    def review_items(self, job_id: str, status: str = "open") -> list[ReviewItem]: ...
    def decide(self, job_id: str, review_id: str, decision: Decision) -> ReviewItem: ...
    def result_summary(self, job_id: str) -> ResultSummary: ...
    def graph(self, job_id: str) -> CitationGraph: ...
```

Recommended local HTTP mapping:

| Operation | Endpoint |
| --- | --- |
| Create a job | `POST /api/jobs` |
| Read job state | `GET /api/jobs/{job_id}` |
| Estimate cost | `POST /api/jobs/{job_id}/estimate` |
| Approve an estimate | `POST /api/jobs/{job_id}/cost-approval` |
| Start or resume | `POST /api/jobs/{job_id}/runs` |
| Request cancellation | `POST /api/jobs/{job_id}/cancellation` |
| Stream events | `GET /api/jobs/{job_id}/events` using server-sent events |
| List review items | `GET /api/jobs/{job_id}/reviews` |
| Record a decision | `POST /api/jobs/{job_id}/reviews/{review_id}/decision` |
| Read result summary | `GET /api/jobs/{job_id}/results` |

Server-sent events are sufficient for one-way progress updates and simpler than
WebSockets. Commands remain ordinary request/response operations.

## Execution and cancellation

The MVP runs one job at a time in a worker process. A process boundary prevents
large conversion models and a failed stage from taking down the local service.
The service writes state before dispatch and after every stage transition.

Cancellation is cooperative:

1. The service records `cancellation_requested` immediately.
2. The worker checks the flag between PDFs, Gemini calls, enrichment rows, and
   pipeline stages.
3. Completed artifacts remain valid and resumable.
4. An in-flight provider request may finish, but no new work starts afterward.
5. The job becomes `cancelled` only after the worker reaches a safe boundary.

Do not terminate a worker while a CSV or JSON file is being replaced. Artifact
writes should use a temporary sibling followed by an atomic rename before the
UI relies on cancellation.

## Researcher workflow

The MVP UI has five screens:

1. **New analysis** — select a local PDF directory, output directory, OCR mode,
   model, and optional enrichment.
2. **Conversion review** — preview converted markdown and resolve conversion or
   duplicate warnings before paid extraction.
3. **Cost approval and run** — show the estimate, provider/model, what text is
   sent externally, live stage progress, and cancellation.
4. **Quality review** — work through failures, zero-reference papers,
   enrichment misses, and author identity flags with recorded decisions.
5. **Explore and export** — corpus summary, top cited works/authors, filters,
   source-to-reference drill-down, and CSV export/open-folder actions.

The UI should not imply that a completed run is scientifically validated. It
distinguishes `completed` from `reviewed`, shows unresolved review counts, and
includes configuration/provenance in exports.

## Security and privacy requirements

- Bind the local service to loopback only by default.
- Generate a per-launch authorization token for the local frontend.
- Store API keys in the OS keychain or environment, never SQLite or job files.
- Validate that selected paths are local and resolve them before use.
- Do not serve arbitrary filesystem paths through preview endpoints.
- Escape markdown/HTML previews and use a restrictive content security policy.
- Show the external-data boundary before cost approval.
- A hosted version requires a separate threat model and data-retention design.

## Delivery sequence

### Milestone 1: application seam

- Add structured progress callbacks beside the current Rich rendering.
- Add cooperative cancellation checks at per-item and stage boundaries.
- Implement job, event, and review-domain models.
- Implement sidecar-to-review adapters.
- Prove that existing CLI behavior and artifacts remain unchanged.

### Milestone 2: local service

- Add SQLite job state and one worker process.
- Implement the service methods and local endpoints above.
- Add resume-after-restart and event-reconnect tests.
- Add keychain/environment secret resolution.

### Milestone 3: researcher UI MVP

- Implement the five-screen workflow.
- Test with selectable-text, scanned, duplicate, failed-extraction, and
  ambiguous-author corpora.
- Conduct usability sessions with researchers who do not use Python.

### Milestone 4: exploration and distribution

- Add interactive filters and citation drill-downs.
- Package the local service and frontend as a signed desktop application.
- Add update, diagnostics export, and support workflows.

Do not start hosted multi-user infrastructure until the local workflow and
review model have been validated with researchers.

## MVP acceptance criteria

The first UI milestone is successful when a researcher who does not use Python
can:

1. select a local corpus and output location;
2. understand and resolve conversion warnings;
3. see and approve estimated API cost before extraction;
4. leave and reopen the application without losing progress;
5. cancel safely and resume from completed checkpoints;
6. review every surfaced quality issue and record decisions;
7. explore source papers, cited works, and authors;
8. export the same portable artifacts produced by the Python library;
9. see which data is sent to each external provider;
10. complete the workflow without opening a notebook or terminal.
