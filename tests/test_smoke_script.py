"""Tests for the operator smoke-test script."""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "smoke_test.sh"


def test_smoke_script_is_valid_executable_bash() -> None:
    mode = SCRIPT.stat().st_mode

    assert mode & stat.S_IXUSR
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_smoke_script_documents_and_runs_staged_pipeline() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'PDF_DIR="${1:-${PDF_DIR:-}}"' in text
    assert 'OUT_DIR="${OUT_DIR:-/tmp/citegraph_smoke_out}"' in text
    assert 'MODEL="${MODEL:-gemini-3.1-flash-lite}"' in text
    assert 'LLM_CONCURRENCY="${LLM_CONCURRENCY:-4}"' in text
    assert 'ENRICH_MAX_WORKERS="${ENRICH_MAX_WORKERS:-2}"' in text
    assert "--recursive" in text
    assert "--ocr-auto" in text
    assert "--llm-concurrency" in text
    assert "--enrich-contact" in text
    assert "--enrich-max-workers" in text

    commands = [
        "citegraph convert",
        "citegraph estimate",
        "citegraph metadata",
        "citegraph references",
        "citegraph dedup",
        "citegraph enrich",
        "citegraph authors",
        "citegraph status",
    ]
    positions = [text.index(command) for command in commands]
    assert positions == sorted(positions)


def test_smoke_script_requires_pdf_dir() -> None:
    env = os.environ.copy()
    env.pop("PDF_DIR", None)

    result = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "Usage:" in result.stderr
