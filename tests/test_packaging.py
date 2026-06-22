"""Packaging metadata tests."""

from __future__ import annotations

from pathlib import Path

import tomllib


def _pyproject() -> dict:
    return tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))


def test_pdf_conversion_dependency_is_optional() -> None:
    project = _pyproject()["project"]
    dependencies = project["dependencies"]
    extras = project["optional-dependencies"]

    assert not any(dep.startswith("docling") for dep in dependencies)
    assert any(dep.startswith("docling") for dep in extras["pdf"])


def test_all_extra_includes_pdf_and_enrichment_dependencies() -> None:
    extras = _pyproject()["project"]["optional-dependencies"]

    assert any(dep.startswith("docling") for dep in extras["all"])
    assert any(dep.startswith("httpx") for dep in extras["all"])
