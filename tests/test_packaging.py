"""Packaging metadata tests."""

from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib


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


def test_package_version_is_single_sourced_with_init() -> None:
    project_version = _pyproject()["project"]["version"]
    init_text = (ROOT / "src" / "citegraph" / "__init__.py").read_text(encoding="utf-8")

    assert f'__version__ = "{project_version}"' in init_text


def test_readme_linked_workflows_exist() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    for workflow in (".github/workflows/ci.yml", ".github/workflows/publish.yml"):
        assert workflow in readme
        assert (ROOT / workflow).exists()
