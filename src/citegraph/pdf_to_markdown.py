"""PDF -> markdown conversion using docling, with idempotent caching."""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

# ``ocr`` flag accepted by :func:`convert_directory` and the ``Pipeline``:
# ``False`` = never OCR, ``True`` = force OCR everywhere, ``"auto"`` = try
# without OCR first and only re-run the image-only outputs with OCR.
OCRMode = bool | Literal["auto"]


def list_pdfs(pdf_dir: Path | str, *, recursive: bool = False) -> list[Path]:
    """Return all ``*.pdf`` files (case-insensitive) under ``pdf_dir``.

    With ``recursive=False`` (default), looks at ``pdf_dir`` only — the
    historical behaviour. With ``recursive=True``, walks the whole tree
    but skips dotted directories (``.git``, ``.venv``, etc.) so that
    pointing at a project root doesn't drag in third-party PDFs.
    """
    pdf_dir = Path(pdf_dir)
    if not pdf_dir.is_dir():
        raise NotADirectoryError(f"PDF directory not found: {pdf_dir}")

    if not recursive:
        return sorted(p for p in pdf_dir.iterdir() if p.suffix.lower() == ".pdf")

    found: list[Path] = []
    for path in pdf_dir.rglob("*"):
        if path.suffix.lower() != ".pdf" or not path.is_file():
            continue
        rel_parts = path.relative_to(pdf_dir).parts
        if any(part.startswith(".") for part in rel_parts[:-1]):
            continue
        found.append(path)
    return sorted(found)


def cache_stem_for(pdf_path: Path | str, pdf_dir: Path | str) -> str:
    """Cache filename stem for ``pdf_path`` relative to ``pdf_dir``.

    Flat layouts produce the historical ``pdf.stem`` (so existing caches
    stay valid). Nested layouts encode the relative path — e.g.
    ``journal_X/foo.pdf`` becomes ``journal_X__foo`` — to keep two PDFs
    with the same filename in different subdirectories from clobbering
    each other's cache files.
    """
    pdf_path = Path(pdf_path)
    pdf_dir = Path(pdf_dir)
    try:
        rel = pdf_path.relative_to(pdf_dir)
    except ValueError:
        return pdf_path.stem
    parts = list(rel.with_suffix("").parts)
    # Strip filesystem separators that survived (defensive — Path.parts
    # already splits them) and join with a double underscore.
    sanitized = ["".join(c if c not in "/\\:" else "_" for c in p) for p in parts]
    return "__".join(sanitized)


def _detect_stem_collisions(pdfs: list[Path], pdf_dir: Path) -> None:
    """Raise if ``cache_stem_for`` would collide for two distinct PDFs."""
    stems = [cache_stem_for(p, pdf_dir) for p in pdfs]
    dupes = sorted(s for s, n in Counter(stems).items() if n > 1)
    if not dupes:
        return
    examples = []
    for d in dupes:
        offenders = [
            str(p.relative_to(pdf_dir)) for p, s in zip(pdfs, stems, strict=True) if s == d
        ]
        examples.append(f"  {d!r} <- {offenders}")
    raise ValueError(
        "Cache-key collisions in recursive scan:\n"
        + "\n".join(examples)
        + "\nRename one of the conflicting PDFs to disambiguate."
    )


def _is_image_only(path: Path, min_text_chars: int = 200) -> bool:
    """Return True if the markdown appears to contain only images with no real text.

    Scanned PDFs converted by docling without OCR produce files that are mostly
    ``<!-- image -->`` tags. This heuristic strips those tags and markdown
    headers to check if any substantive text survives.
    """
    text = path.read_text(encoding="utf-8")
    stripped = text.replace("<!-- image -->", "")
    lines = [ln for ln in stripped.splitlines() if not ln.lstrip().startswith("#")]
    return len("".join(lines).strip()) < min_text_chars


def convert_pdf_to_markdown(
    pdf_path: Path | str,
    markdown_dir: Path | str,
    *,
    overwrite: bool = False,
    cache_stem: str | None = None,
    ocr: bool = False,
) -> Path:
    """Convert a single PDF to markdown and write it to ``markdown_dir``.

    If the markdown file already exists and ``overwrite`` is ``False``, the
    existing file is reused (the docling conversion is the slow step we
    explicitly want to cache).

    ``cache_stem`` overrides the output filename stem; without it the PDF's
    own ``stem`` is used. Pass an explicit stem (e.g. one produced by
    :func:`cache_stem_for`) to disambiguate same-named PDFs in different
    subdirectories.

    With ``ocr=True``, configures docling to force full-page OCR via EasyOCR,
    which is needed for scanned PDFs where each page is a bitmap image.
    """
    pdf_path = Path(pdf_path)
    markdown_dir = Path(markdown_dir)
    markdown_dir.mkdir(parents=True, exist_ok=True)

    stem = cache_stem if cache_stem is not None else pdf_path.stem
    out_path = markdown_dir / f"{stem}.md"
    if out_path.exists() and not overwrite:
        logger.debug("Skipping %s (cached at %s)", pdf_path.name, out_path)
        return out_path

    logger.info("Converting %s -> markdown%s", pdf_path.name, " (OCR)" if ocr else "")
    # Imported lazily so ``import citegraph`` stays cheap and so docling is
    # only required when this function is actually called.
    from docling.document_converter import DocumentConverter

    if ocr:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import EasyOcrOptions, PdfPipelineOptions
        from docling.document_converter import PdfFormatOption

        pipeline_options = PdfPipelineOptions(
            do_ocr=True,
            ocr_options=EasyOcrOptions(force_full_page_ocr=True),
        )
        converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )
    else:
        converter = DocumentConverter()

    result = converter.convert(str(pdf_path))
    out_path.write_text(result.document.export_to_markdown(), encoding="utf-8")
    return out_path


def _convert_loop(
    pdfs: list[Path],
    pdf_dir: Path,
    markdown_dir: Path | str,
    *,
    overwrite: bool,
    recursive: bool,
    show_progress: bool,
    ocr: bool,
    description: str = "Converting PDFs",
) -> list[Path]:
    """Inner per-PDF loop shared by the single-pass and two-pass paths."""
    from citegraph._progress import iter_with_progress

    out_paths: list[Path] = []
    for pdf in iter_with_progress(
        pdfs,
        show_progress=show_progress,
        description=description,
        item_label=lambda p: p.name,
    ):
        stem = cache_stem_for(pdf, pdf_dir) if recursive else None
        out_paths.append(
            convert_pdf_to_markdown(
                pdf, markdown_dir, overwrite=overwrite, cache_stem=stem, ocr=ocr
            )
        )
    return out_paths


def convert_directory(
    pdf_dir: Path | str,
    markdown_dir: Path | str,
    *,
    overwrite: bool = False,
    recursive: bool = False,
    show_progress: bool = True,
    ocr: OCRMode = False,
) -> list[Path]:
    """Convert every PDF under ``pdf_dir`` to markdown.

    With ``recursive=True``, walks subdirectories and uses path-aware cache
    stems via :func:`cache_stem_for`. Raises ``ValueError`` upfront if two
    distinct PDFs would map to the same cache stem — better to fail loudly
    than to silently overwrite each other's caches mid-run.

    With ``show_progress=True`` (default) a rich progress bar is rendered
    while the per-PDF loop runs. Pass ``False`` for headless runs / tests
    where the extra stderr output is unwanted.

    ``ocr`` controls how OCR is applied (see :data:`OCRMode`):

    * ``False`` — never OCR.
    * ``True`` — force full-page OCR via EasyOCR for every PDF.
    * ``"auto"`` — two-pass: convert everything without OCR first, run the
      :func:`_is_image_only` heuristic on each output, then re-run only
      the flagged PDFs with OCR (their stub markdown is deleted first so
      the cache check doesn't short-circuit the retry).

    Returns the list of written markdown files in stable (sorted) order.
    """
    pdf_dir = Path(pdf_dir)
    markdown_dir = Path(markdown_dir)
    pdfs = list_pdfs(pdf_dir, recursive=recursive)
    logger.info(
        "Found %d PDFs under %s (recursive=%s, ocr=%r)", len(pdfs), pdf_dir, recursive, ocr
    )

    if recursive:
        _detect_stem_collisions(pdfs, pdf_dir)

    if ocr == "auto":
        # Pass 1: try without OCR — cheap on text PDFs, fast on cached ones.
        out_paths = _convert_loop(
            pdfs,
            pdf_dir,
            markdown_dir,
            overwrite=overwrite,
            recursive=recursive,
            show_progress=show_progress,
            ocr=False,
        )
        retry_pairs = [(pdf, md) for pdf, md in zip(pdfs, out_paths, strict=True) if _is_image_only(md)]
        if retry_pairs:
            logger.info(
                "Re-running %d image-only PDF(s) with OCR (auto fallback)", len(retry_pairs)
            )
            for _pdf, md in retry_pairs:
                md.unlink()
            _convert_loop(
                [pdf for pdf, _ in retry_pairs],
                pdf_dir,
                markdown_dir,
                overwrite=True,
                recursive=recursive,
                show_progress=show_progress,
                ocr=True,
                description=f"Re-converting {len(retry_pairs)} with OCR",
            )
        return out_paths

    return _convert_loop(
        pdfs,
        pdf_dir,
        markdown_dir,
        overwrite=overwrite,
        recursive=recursive,
        show_progress=show_progress,
        ocr=bool(ocr),
    )
