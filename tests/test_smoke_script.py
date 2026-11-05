"""Tests for the operator smoke-test script."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "smoke_test.sh"


def test_smoke_script_is_valid_executable_bash() -> None:
    mode = SCRIPT.stat().st_mode

    assert mode & stat.S_IXUSR
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


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


@pytest.mark.parametrize("explicit_selection", [True, False])
def test_smoke_script_respects_provider_selection(tmp_path: Path, explicit_selection: bool) -> None:
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    binary_dir = tmp_path / "bin"
    binary_dir.mkdir()
    fake = binary_dir / "citegraph"
    fake.write_text(
        f"#!{sys.executable}\nimport json, os, sys\n"
        "with open(os.environ['CALL_LOG'], 'a') as stream:\n"
        "    stream.write(json.dumps(sys.argv[1:]) + '\\n')\n",
        encoding="utf-8",
    )
    fake.chmod(0o755)
    log = tmp_path / "calls.jsonl"
    env = {
        **os.environ, "PATH": str(binary_dir) + os.pathsep + os.defpath,
        "CALL_LOG": str(log), "OUT_DIR": str(tmp_path / "out"),
    }
    for name in ("PROVIDER", "MODEL", "CITEGRAPH_PROVIDER", "CITEGRAPH_MODEL"):
        env.pop(name, None)
    if explicit_selection:
        env.update(PROVIDER="anthropic", MODEL="selected-model")
    subprocess.run(["bash", str(SCRIPT), str(pdf_dir)], env=env, check=True, capture_output=True)
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert [args[0] for args in calls] == [
        "convert", "estimate", "metadata", "references", "dedup", "enrich", "authors", "status",
    ]
    for args in calls[1:4]:
        if explicit_selection:
            assert args[args.index("--provider") + 1] == "anthropic"
            assert args[args.index("--model") + 1] == "selected-model"
        else:
            assert "--provider" not in args  # Let the CLI honor .env and its own defaults.
            assert "--model" not in args
    assert "--yes" in calls[3]
