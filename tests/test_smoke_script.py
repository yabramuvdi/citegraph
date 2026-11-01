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
