"""Content-aware reuse of expensive per-paper observations."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from citegraph.io import (
    cache_input_fingerprint,
    read_pydantic_cache,
    write_cache_fingerprint,
)
from citegraph.pdf_to_markdown import convert_directory
from citegraph.schemas import PaperMetadata


def _install_fake_docling(
    monkeypatch: pytest.MonkeyPatch,
    calls: list[tuple[str, bool]],
    *,
    poor_output: bool = False,
) -> None:
    class DocumentConverter:
        def __init__(self, **kwargs):
            self.ocr = bool(kwargs)

        def convert(self, source):
            source = Path(source)
            calls.append((source.name, self.ocr))
            if poor_output:
                markdown = "<!-- image -->"
            else:
                markdown = f"# {source.read_text(encoding='utf-8')}\n" + "x" * 300
            return SimpleNamespace(
                document=SimpleNamespace(export_to_markdown=lambda: markdown)
            )

    class PdfPipelineOptions:
        def __init__(self, **kwargs):
            pass

    docling = ModuleType("docling")
    converter_module = ModuleType("docling.document_converter")
    converter_module.DocumentConverter = DocumentConverter
    converter_module.PdfFormatOption = lambda **kwargs: kwargs
    base_models = ModuleType("docling.datamodel.base_models")
    base_models.InputFormat = SimpleNamespace(PDF="pdf")
    pipeline_options = ModuleType("docling.datamodel.pipeline_options")
    pipeline_options.EasyOcrOptions = lambda **kwargs: kwargs
    pipeline_options.PdfPipelineOptions = PdfPipelineOptions
    monkeypatch.setitem(sys.modules, "docling", docling)
    monkeypatch.setitem(sys.modules, "docling.document_converter", converter_module)
    monkeypatch.setitem(sys.modules, "docling.datamodel.base_models", base_models)
    monkeypatch.setitem(sys.modules, "docling.datamodel.pipeline_options", pipeline_options)


def test_renamed_pdf_reuses_conversion_by_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, bool]] = []
    _install_fake_docling(monkeypatch, calls)
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    pdf = pdf_dir / "old.pdf"
    pdf.write_text("same paper", encoding="utf-8")
    markdown_dir = tmp_path / "out" / "markdown"

    [old_markdown] = convert_directory(pdf_dir, markdown_dir, show_progress=False)
    pdf.rename(pdf_dir / "renamed.pdf")
    [renamed_markdown] = convert_directory(pdf_dir, markdown_dir, show_progress=False)

    assert calls == [("old.pdf", False)]
    assert renamed_markdown.name == "renamed.md"
    assert renamed_markdown.read_bytes() == old_markdown.read_bytes()


def test_replaced_pdf_refreshes_same_stem_conversion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, bool]] = []
    _install_fake_docling(monkeypatch, calls)
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    pdf = pdf_dir / "paper.pdf"
    pdf.write_text("first", encoding="utf-8")
    markdown_dir = tmp_path / "out" / "markdown"

    convert_directory(pdf_dir, markdown_dir, show_progress=False)
    pdf.write_text("replacement", encoding="utf-8")
    [markdown] = convert_directory(pdf_dir, markdown_dir, show_progress=False)

    assert calls == [("paper.pdf", False), ("paper.pdf", False)]
    assert "replacement" in markdown.read_text(encoding="utf-8")


def test_tampered_markdown_refreshes_conversion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, bool]] = []
    _install_fake_docling(monkeypatch, calls)
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    (pdf_dir / "paper.pdf").write_text("paper", encoding="utf-8")
    markdown_dir = tmp_path / "out" / "markdown"

    [markdown] = convert_directory(pdf_dir, markdown_dir, show_progress=False)
    markdown.write_text("tampered", encoding="utf-8")
    convert_directory(pdf_dir, markdown_dir, show_progress=False)

    assert calls == [("paper.pdf", False), ("paper.pdf", False)]
    assert markdown.read_text(encoding="utf-8").startswith("# paper")


def test_unknown_conversion_cache_version_refreshes_conversion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, bool]] = []
    _install_fake_docling(monkeypatch, calls)
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    (pdf_dir / "paper.pdf").write_text("paper", encoding="utf-8")
    markdown_dir = tmp_path / "out" / "markdown"

    convert_directory(pdf_dir, markdown_dir, show_progress=False)
    manifest = tmp_path / "out" / "conversion_cache.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["version"] = 999
    manifest.write_text(json.dumps(data), encoding="utf-8")
    convert_directory(pdf_dir, markdown_dir, show_progress=False)

    assert calls == [("paper.pdf", False), ("paper.pdf", False)]


def test_completed_poor_ocr_attempt_is_not_repeated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, bool]] = []
    _install_fake_docling(monkeypatch, calls, poor_output=True)
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    (pdf_dir / "scan.pdf").write_text("scan", encoding="utf-8")
    markdown_dir = tmp_path / "out" / "markdown"

    convert_directory(pdf_dir, markdown_dir, ocr="auto", show_progress=False)
    convert_directory(pdf_dir, markdown_dir, ocr="auto", show_progress=False)

    assert calls == [("scan.pdf", False), ("scan.pdf", True)]


def test_ocr_setting_refreshes_non_ocr_conversion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, bool]] = []
    _install_fake_docling(monkeypatch, calls)
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    (pdf_dir / "paper.pdf").write_text("paper", encoding="utf-8")
    markdown_dir = tmp_path / "out" / "markdown"

    convert_directory(pdf_dir, markdown_dir, show_progress=False)
    convert_directory(pdf_dir, markdown_dir, ocr=True, show_progress=False)

    assert calls == [("paper.pdf", False), ("paper.pdf", True)]


def test_forced_ocr_does_not_trust_legacy_conversion_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, bool]] = []
    _install_fake_docling(monkeypatch, calls)
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    (pdf_dir / "paper.pdf").write_text("paper", encoding="utf-8")
    markdown_dir = tmp_path / "out" / "markdown"
    markdown_dir.mkdir(parents=True)
    (markdown_dir / "paper.md").write_text("unknown legacy settings", encoding="utf-8")

    convert_directory(pdf_dir, markdown_dir, ocr=True, show_progress=False)

    assert calls == [("paper.pdf", True)]


def _metadata_cache(path: Path, title: str = "Cached") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"Title": title, "Authors_List": [], "Journal": "", "Year": 2020}),
        encoding="utf-8",
    )


def test_renamed_markdown_reuses_valid_sibling_extraction(tmp_path: Path) -> None:
    markdown_dir = tmp_path / "out" / "markdown"
    markdown_dir.mkdir(parents=True)
    old_markdown = markdown_dir / "old.md"
    old_markdown.write_text("unchanged markdown", encoding="utf-8")
    cache_dir = tmp_path / "out" / "metadata"
    old_cache = cache_dir / "old.json"
    _metadata_cache(old_cache)
    write_cache_fingerprint(old_cache, old_markdown)
    new_markdown = old_markdown.rename(markdown_dir / "renamed.md")

    cached = read_pydantic_cache(
        cache_dir / "renamed.json", new_markdown, PaperMetadata
    )

    assert cached is not None
    assert cached.Title == "Cached"
    assert (cache_dir / "renamed.json").exists()


def test_extraction_recipe_change_rejects_cache(tmp_path: Path) -> None:
    markdown = tmp_path / "out" / "markdown" / "paper.md"
    markdown.parent.mkdir(parents=True)
    markdown.write_text("markdown", encoding="utf-8")
    cache = tmp_path / "out" / "metadata" / "paper.json"
    _metadata_cache(cache)
    write_cache_fingerprint(cache, markdown, recipe={"model": "old"})

    assert (
        read_pydantic_cache(
            cache, markdown, PaperMetadata, recipe={"model": "new"}
        )
        is None
    )


def test_renamed_extraction_requires_matching_recipe(tmp_path: Path) -> None:
    markdown = tmp_path / "out" / "markdown" / "renamed.md"
    markdown.parent.mkdir(parents=True)
    markdown.write_text("markdown", encoding="utf-8")
    old_cache = tmp_path / "out" / "metadata" / "old.json"
    _metadata_cache(old_cache)
    write_cache_fingerprint(old_cache, markdown, recipe={"model": "old"})

    assert (
        read_pydantic_cache(
            old_cache.with_name("renamed.json"),
            markdown,
            PaperMetadata,
            recipe={"model": "new"},
        )
        is None
    )


def test_reused_extraction_retains_recipe(tmp_path: Path) -> None:
    markdown = tmp_path / "out" / "markdown" / "renamed.md"
    markdown.parent.mkdir(parents=True)
    markdown.write_text("markdown", encoding="utf-8")
    cache_dir = tmp_path / "out" / "metadata"
    old_cache = cache_dir / "old.json"
    new_cache = cache_dir / "renamed.json"
    _metadata_cache(old_cache)
    write_cache_fingerprint(old_cache, markdown, recipe={"model": "old"})

    assert (
        read_pydantic_cache(
            new_cache, markdown, PaperMetadata, recipe={"model": "old"}
        )
        is not None
    )
    assert (
        read_pydantic_cache(
            new_cache, markdown, PaperMetadata, recipe={"model": "new"}
        )
        is None
    )


def test_renamed_markdown_rejects_stale_sibling_extraction(tmp_path: Path) -> None:
    markdown_dir = tmp_path / "out" / "markdown"
    markdown_dir.mkdir(parents=True)
    old_markdown = markdown_dir / "old.md"
    old_markdown.write_text("old markdown", encoding="utf-8")
    old_cache = tmp_path / "out" / "metadata" / "old.json"
    _metadata_cache(old_cache)
    write_cache_fingerprint(old_cache, old_markdown)
    new_markdown = markdown_dir / "renamed.md"
    new_markdown.write_text("changed markdown", encoding="utf-8")

    assert (
        read_pydantic_cache(
            old_cache.with_name("renamed.json"), new_markdown, PaperMetadata
        )
        is None
    )


def test_renamed_markdown_rejects_tampered_sibling_extraction(tmp_path: Path) -> None:
    markdown_dir = tmp_path / "out" / "markdown"
    markdown_dir.mkdir(parents=True)
    markdown = markdown_dir / "renamed.md"
    markdown.write_text("unchanged markdown", encoding="utf-8")
    old_cache = tmp_path / "out" / "metadata" / "old.json"
    _metadata_cache(old_cache)
    write_cache_fingerprint(old_cache, markdown)
    _metadata_cache(old_cache, title="Tampered")

    assert (
        read_pydantic_cache(
            old_cache.with_name("renamed.json"), markdown, PaperMetadata
        )
        is None
    )


def test_renamed_markdown_does_not_trust_legacy_sibling_sidecar(tmp_path: Path) -> None:
    markdown = tmp_path / "out" / "markdown" / "renamed.md"
    markdown.parent.mkdir(parents=True)
    markdown.write_text("unchanged markdown", encoding="utf-8")
    old_cache = tmp_path / "out" / "metadata" / "old.json"
    _metadata_cache(old_cache)
    old_cache.with_suffix(".json.sha256").write_text(
        cache_input_fingerprint(markdown), encoding="ascii"
    )

    assert (
        read_pydantic_cache(
            old_cache.with_name("renamed.json"), markdown, PaperMetadata
        )
        is None
    )


def test_legacy_same_stem_extraction_is_upgraded_without_refresh(tmp_path: Path) -> None:
    markdown = tmp_path / "out" / "markdown" / "paper.md"
    markdown.parent.mkdir(parents=True)
    markdown.write_text("legacy markdown", encoding="utf-8")
    cache = tmp_path / "out" / "metadata" / "paper.json"
    _metadata_cache(cache)

    cached = read_pydantic_cache(cache, markdown, PaperMetadata)

    assert cached is not None
    provenance = json.loads(cache.with_suffix(".json.sha256").read_text(encoding="utf-8"))
    assert set(provenance) >= {"input_sha256", "output_sha256"}
