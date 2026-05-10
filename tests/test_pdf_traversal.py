"""Tests for the recursive PDF discovery + path-aware cache stems."""

from __future__ import annotations

from pathlib import Path

import pytest

from citegraph.pdf_to_markdown import (
    _detect_stem_collisions,
    cache_stem_for,
    convert_directory,
    convert_pdf_to_markdown,
    list_pdfs,
)


# ---------------------------------------------------------------------------
# list_pdfs
# ---------------------------------------------------------------------------
def test_list_pdfs_non_recursive_ignores_subdirs(tmp_path: Path) -> None:
    (tmp_path / "top.pdf").write_text("")
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "nested.pdf").write_text("")

    assert [p.name for p in list_pdfs(tmp_path)] == ["top.pdf"]


def test_list_pdfs_recursive_finds_nested(tmp_path: Path) -> None:
    (tmp_path / "top.pdf").write_text("")
    (tmp_path / "journal_X").mkdir()
    (tmp_path / "journal_X" / "paper1.pdf").write_text("")
    (tmp_path / "journal_X" / "deep").mkdir()
    (tmp_path / "journal_X" / "deep" / "paper2.pdf").write_text("")

    found = [str(p.relative_to(tmp_path)) for p in list_pdfs(tmp_path, recursive=True)]
    assert "top.pdf" in found
    assert "journal_X/paper1.pdf" in found
    assert "journal_X/deep/paper2.pdf" in found
    assert len(found) == 3


def test_list_pdfs_recursive_skips_hidden_directories(tmp_path: Path) -> None:
    (tmp_path / "visible.pdf").write_text("")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "hooks").mkdir()
    (tmp_path / ".git" / "hooks" / "buried.pdf").write_text("")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "site.pdf").write_text("")

    found = list_pdfs(tmp_path, recursive=True)
    assert [p.name for p in found] == ["visible.pdf"]


def test_list_pdfs_recursive_is_case_insensitive(tmp_path: Path) -> None:
    (tmp_path / "lower.pdf").write_text("")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "UPPER.PDF").write_text("")

    found = sorted(p.name for p in list_pdfs(tmp_path, recursive=True))
    assert found == ["UPPER.PDF", "lower.pdf"]


# ---------------------------------------------------------------------------
# cache_stem_for
# ---------------------------------------------------------------------------
def test_cache_stem_flat_is_unchanged(tmp_path: Path) -> None:
    """Flat layouts must produce the same stems they always did — preserves caches."""
    (tmp_path / "foo.pdf").write_text("")
    assert cache_stem_for(tmp_path / "foo.pdf", tmp_path) == "foo"


def test_cache_stem_nested_uses_relative_path(tmp_path: Path) -> None:
    p = tmp_path / "journal_X" / "paper1.pdf"
    p.parent.mkdir()
    p.write_text("")
    assert cache_stem_for(p, tmp_path) == "journal_X__paper1"


def test_cache_stem_handles_deep_nesting(tmp_path: Path) -> None:
    p = tmp_path / "a" / "b" / "c" / "doc.pdf"
    p.parent.mkdir(parents=True)
    p.write_text("")
    assert cache_stem_for(p, tmp_path) == "a__b__c__doc"


def test_cache_stem_falls_back_when_pdf_outside_dir(tmp_path: Path) -> None:
    """Defensive: if the path isn't actually under pdf_dir, just use the bare stem."""
    elsewhere = tmp_path / "outside" / "wild.pdf"
    elsewhere.parent.mkdir()
    elsewhere.write_text("")
    pdf_dir = tmp_path / "expected"
    pdf_dir.mkdir()

    assert cache_stem_for(elsewhere, pdf_dir) == "wild"


# ---------------------------------------------------------------------------
# Collision detection
# ---------------------------------------------------------------------------
def test_collision_detection_raises_with_useful_message(tmp_path: Path) -> None:
    """The contrived case where a flat name collides with a nested join key."""
    (tmp_path / "foo__bar.pdf").write_text("")
    (tmp_path / "foo").mkdir()
    (tmp_path / "foo" / "bar.pdf").write_text("")

    pdfs = list_pdfs(tmp_path, recursive=True)
    with pytest.raises(ValueError, match="collisions"):
        _detect_stem_collisions(pdfs, tmp_path)


def test_no_collision_for_distinct_nested_paths(tmp_path: Path) -> None:
    (tmp_path / "j1").mkdir()
    (tmp_path / "j2").mkdir()
    (tmp_path / "j1" / "paper.pdf").write_text("")
    (tmp_path / "j2" / "paper.pdf").write_text("")

    pdfs = list_pdfs(tmp_path, recursive=True)
    _detect_stem_collisions(pdfs, tmp_path)  # must not raise


# ---------------------------------------------------------------------------
# convert_directory + cache_stem wiring (no docling needed thanks to caching)
# ---------------------------------------------------------------------------
def test_convert_directory_recursive_writes_path_aware_cache(tmp_path: Path) -> None:
    pdf_dir = tmp_path / "pdfs"
    md_dir = tmp_path / "out" / "markdown"
    md_dir.mkdir(parents=True)

    # Two PDFs that would collide under the old flat-stem scheme.
    (pdf_dir / "j1").mkdir(parents=True)
    (pdf_dir / "j2").mkdir(parents=True)
    (pdf_dir / "j1" / "paper.pdf").write_text("")
    (pdf_dir / "j2" / "paper.pdf").write_text("")

    # Pre-write the expected markdown caches so docling never needs to run.
    (md_dir / "j1__paper.md").write_text("# from j1")
    (md_dir / "j2__paper.md").write_text("# from j2")

    out_paths = convert_directory(pdf_dir, md_dir, recursive=True, show_progress=False)

    written = sorted(p.name for p in out_paths)
    assert written == ["j1__paper.md", "j2__paper.md"]
    # Cache contents preserved (would be overwritten if collision happened).
    assert (md_dir / "j1__paper.md").read_text() == "# from j1"
    assert (md_dir / "j2__paper.md").read_text() == "# from j2"


def test_convert_directory_non_recursive_keeps_legacy_stems(tmp_path: Path) -> None:
    """Back-compat: a flat directory must still produce stem-only cache files."""
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    (pdf_dir / "foo.pdf").write_text("")

    md_dir = tmp_path / "out" / "markdown"
    md_dir.mkdir(parents=True)
    (md_dir / "foo.md").write_text("# cached")  # so docling isn't called

    out_paths = convert_directory(pdf_dir, md_dir, recursive=False, show_progress=False)
    assert [p.name for p in out_paths] == ["foo.md"]


def test_convert_directory_recursive_raises_on_collision(tmp_path: Path) -> None:
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    (pdf_dir / "foo__bar.pdf").write_text("")
    (pdf_dir / "foo").mkdir()
    (pdf_dir / "foo" / "bar.pdf").write_text("")

    md_dir = tmp_path / "out" / "markdown"
    with pytest.raises(ValueError, match="collisions"):
        convert_directory(pdf_dir, md_dir, recursive=True)


# ---------------------------------------------------------------------------
# convert_pdf_to_markdown explicit cache_stem param
# ---------------------------------------------------------------------------
def test_convert_pdf_to_markdown_respects_cache_stem(tmp_path: Path) -> None:
    pdf = tmp_path / "x.pdf"
    pdf.write_text("")
    md_dir = tmp_path / "md"
    md_dir.mkdir()
    (md_dir / "custom__name.md").write_text("# cached")  # docling won't run

    out = convert_pdf_to_markdown(pdf, md_dir, cache_stem="custom__name")
    assert out == md_dir / "custom__name.md"
