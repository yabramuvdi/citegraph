"""Packaging metadata tests."""

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


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


def test_supported_python_versions_are_declared() -> None:
    project = _pyproject()["project"]
    classifiers = project["classifiers"]

    assert project["requires-python"] == ">=3.11"
    assert "Programming Language :: Python :: 3.10" not in classifiers
    for version in ("3.11", "3.12", "3.13"):
        assert f"Programming Language :: Python :: {version}" in classifiers


def test_sdist_excludes_repository_only_material() -> None:
    sdist = _pyproject()["tool"]["hatch"]["build"]["targets"]["sdist"]

    assert {
        "/.claude",
        "/AGENTS.md",
        "/CLAUDE.md",
        "/PROJECT_OVERVIEW.html",
        "/STATUS.html",
        "/research",
    } <= set(sdist["exclude"])


def test_package_version_is_single_sourced_with_init() -> None:
    project_version = _pyproject()["project"]["version"]
    init_text = (ROOT / "src" / "citegraph" / "__init__.py").read_text(encoding="utf-8")

    assert f'__version__ = "{project_version}"' in init_text


def test_readme_linked_workflows_exist() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    for workflow in (".github/workflows/ci.yml", ".github/workflows/publish.yml"):
        assert workflow in readme
        assert (ROOT / workflow).exists()
