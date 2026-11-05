#!/usr/bin/env bash
set -euo pipefail

PDF_DIR="${1:-${PDF_DIR:-}}"
OUT_DIR="${OUT_DIR:-/tmp/citegraph_smoke_out}"
PROVIDER="${PROVIDER:-${CITEGRAPH_PROVIDER:-}}"
MODEL="${MODEL:-${CITEGRAPH_MODEL:-}}"
LLM_ARGS=(--out "$OUT_DIR")
if [[ -n "$PROVIDER" ]]; then LLM_ARGS+=(--provider "$PROVIDER"); fi
if [[ -n "$MODEL" ]]; then LLM_ARGS+=(--model "$MODEL"); fi
LLM_CONCURRENCY="${LLM_CONCURRENCY:-4}"
ENRICH_CONTACT="${ENRICH_CONTACT:-}"
ENRICH_MAX_WORKERS="${ENRICH_MAX_WORKERS:-2}"
ENRICH_TIMEOUT="${ENRICH_TIMEOUT:-15}"

usage() {
  cat >&2 <<'USAGE'
Usage: scripts/smoke_test.sh /path/to/pdf-folder

Environment overrides:
  OUT_DIR=/tmp/citegraph_smoke_out
  PROVIDER=gemini  # or openai / anthropic, with an explicit MODEL
  MODEL=gemini-3.1-flash-lite
  LLM_CONCURRENCY=4
  ENRICH_CONTACT=you@example.com
  ENRICH_MAX_WORKERS=2
  ENRICH_TIMEOUT=15
USAGE
}

run_step() {
  printf '\n==> %s\n' "$*"
  "$@"
}

if [[ -z "$PDF_DIR" ]]; then
  usage
  exit 2
fi

if [[ ! -d "$PDF_DIR" ]]; then
  printf 'PDF_DIR does not exist or is not a directory: %s\n' "$PDF_DIR" >&2
  exit 2
fi

if ! command -v citegraph >/dev/null 2>&1; then
  printf 'citegraph command not found. Install the package first, for example: pip install -e ".[dev,all]"\n' >&2
  exit 127
fi

if [[ -z "$ENRICH_CONTACT" ]]; then
  printf 'Warning: ENRICH_CONTACT is empty; CrossRef polite-pool requests should include a contact email.\n' >&2
fi

printf 'Smoke test input:  %s\n' "$PDF_DIR"
printf 'Smoke test output: %s\n' "$OUT_DIR"
printf 'Provider:          %s\n' "${PROVIDER:-from citegraph settings}"
printf 'Model:             %s\n' "${MODEL:-from citegraph settings}"

run_step citegraph convert "$PDF_DIR" --out "$OUT_DIR" --recursive --ocr-auto --verbose
run_step citegraph estimate "${LLM_ARGS[@]}"
run_step citegraph metadata "${LLM_ARGS[@]}" --llm-concurrency "$LLM_CONCURRENCY"
run_step citegraph references "${LLM_ARGS[@]}" --llm-concurrency "$LLM_CONCURRENCY" --yes
run_step citegraph dedup --out "$OUT_DIR"
run_step citegraph enrich --out "$OUT_DIR" \
  --enrich-contact "$ENRICH_CONTACT" \
  --enrich-max-workers "$ENRICH_MAX_WORKERS" \
  --enrich-timeout "$ENRICH_TIMEOUT"
run_step citegraph authors --out "$OUT_DIR"
run_step citegraph status --out "$OUT_DIR"

printf '\nSmoke test complete. Load the result with:\n'
printf '  from citegraph import CitationGraph\n'
printf '  g = CitationGraph.from_out_dir(%q)\n' "$OUT_DIR"
