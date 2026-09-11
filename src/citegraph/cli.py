"""Command-line interface for citegraph.

Two ways to use it:

- ``citegraph run`` runs the full pipeline end-to-end.
- The per-stage commands (``convert``, ``metadata``, ``references``,
  ``dedup``, ``enrich``) run one stage at a time, reading prior outputs
  from ``--out``. ``citegraph status`` reports which artifacts exist,
  ``citegraph report`` writes a self-contained ``report.html`` QC
  dashboard summarizing whatever has been produced so far, and
  ``citegraph ui`` serves a read-only live monitor that re-renders as a
  run progresses.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import typer
from rich.logging import RichHandler

from citegraph.authors import AuthorClusterConfig
from citegraph.config import get_settings
from citegraph.dedup import DedupConfig
from citegraph.enrich import EnrichConfig
from citegraph.io import CITATION_COLUMNS, OutLayout, read_stage_csv
from citegraph.pdf_to_markdown import OCRMode
from citegraph.pipeline import Pipeline, StageNotReadyError

app = typer.Typer(
    add_completion=False,
    help="Build a deduplicated citation graph from a folder of academic PDFs.",
)


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
        force=True,
    )


# Typer option defaults come from the config dataclass so the CLI and the
# library surface can't drift apart.
_DEDUP_DEFAULTS = DedupConfig()


def _dedup_config(
    threshold: float,
    title_weight: float,
    authors_weight: float,
    journal_weight: float,
    year_window: int,
) -> DedupConfig:
    return DedupConfig(
        title_weight=title_weight,
        authors_weight=authors_weight,
        journal_weight=journal_weight,
        year_window=year_window,
        threshold=threshold,
    )


def _enrich_config(
    contact: str,
    threshold: float,
    timeout: float,
    year_penalty: float,
    retry_attempts: int,
    retry_wait: float,
    max_workers: int,
    openalex_api_key: str = "",
) -> EnrichConfig:
    if not openalex_api_key:
        openalex_api_key = get_settings().openalex_api_key or ""
    return EnrichConfig(
        contact_email=contact,
        openalex_api_key=openalex_api_key,
        title_match_threshold=threshold,
        timeout_s=timeout,
        year_mismatch_penalty=year_penalty,
        retry_attempts=retry_attempts,
        retry_wait_s=retry_wait,
        max_workers=max_workers,
    )


def _warn_conversion_quality(layout: OutLayout, *, ocr_attempted: bool) -> None:
    """Echo a yellow warning when stage 1 left image-only outputs.

    Tailors the remediation hint to whether OCR was already tried — telling
    the user to "re-run with --ocr" is unhelpful when they just did that.
    """
    if not layout.conversion_warnings_json.exists():
        return
    import json as _json

    warns = _json.loads(layout.conversion_warnings_json.read_text(encoding="utf-8"))
    if ocr_attempted:
        msg = (
            f"{len(warns)} file(s) appear image-only even after OCR; "
            "manual review needed. See:"
        )
    else:
        msg = (
            f"{len(warns)} file(s) appear image-only (scanned PDFs?). "
            "Re-run with --ocr-auto (fastest) or --ocr (force). See:"
        )
    typer.secho(msg, fg=typer.colors.YELLOW)
    typer.secho(f"  {layout.conversion_warnings_json}", fg=typer.colors.YELLOW)


def _resolve_ocr_mode(ocr: bool, ocr_auto: bool) -> OCRMode:
    """Map the two mutually-exclusive CLI flags to the Pipeline ``ocr`` value."""
    if ocr and ocr_auto:
        typer.secho(
            "--ocr and --ocr-auto are mutually exclusive; pick one.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)
    if ocr_auto:
        return "auto"
    return ocr


def _next_step_hint(msg: str) -> None:
    typer.echo(f"Next: {msg}")


def _run_stage(fn, *, next_hint: str | None) -> None:
    try:
        fn()
    except StageNotReadyError as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from e
    if next_hint:
        _next_step_hint(next_hint)


@app.command()
def run(
    pdf_dir: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True),
    out: Path = typer.Option(Path("./out"), "--out", "-o", help="Output directory."),
    enrich: bool = typer.Option(False, "--enrich", help="Resolve DOIs via CrossRef/OpenAlex."),
    enrich_contact: str = typer.Option("", "--enrich-contact", help="Contact email for CrossRef polite pool."),
    enrich_threshold: float = typer.Option(90.0, "--enrich-threshold", help="Title match threshold for enrichment."),
    enrich_timeout: float = typer.Option(15.0, "--enrich-timeout", help="HTTP timeout in seconds for enrichment."),
    enrich_year_penalty: float = typer.Option(
        8.0,
        "--enrich-year-penalty",
        help="Score penalty applied when input and candidate years differ.",
    ),
    enrich_retry_attempts: int = typer.Option(
        3,
        "--enrich-retry-attempts",
        min=1,
        help="Attempts for transient enrichment HTTP errors such as 429/503.",
    ),
    enrich_retry_wait: float = typer.Option(
        0.25,
        "--enrich-retry-wait",
        min=0.0,
        help="Base exponential-backoff wait in seconds for enrichment retries.",
    ),
    enrich_max_workers: int = typer.Option(
        8,
        "--enrich-max-workers",
        min=1,
        help="Maximum concurrent CrossRef/OpenAlex enrichment lookups.",
    ),
    openalex_api_key: str = typer.Option(
        "",
        "--openalex-api-key",
        help="OpenAlex API key (default: OPENALEX_API_KEY env var / .env).",
    ),
    model: str | None = typer.Option(None, "--model", help="Gemini model id."),
    threshold: float = typer.Option(
        _DEDUP_DEFAULTS.threshold, "--threshold", help="Dedup similarity threshold."
    ),
    title_weight: float = typer.Option(_DEDUP_DEFAULTS.title_weight, "--title-weight"),
    authors_weight: float = typer.Option(_DEDUP_DEFAULTS.authors_weight, "--authors-weight"),
    journal_weight: float = typer.Option(_DEDUP_DEFAULTS.journal_weight, "--journal-weight"),
    year_window: int = typer.Option(_DEDUP_DEFAULTS.year_window, "--year-window"),
    llm_concurrency: int | None = typer.Option(
        None,
        "--llm-concurrency",
        min=1,
        help="Maximum concurrent Gemini extraction calls (default: CITEGRAPH_LLM_CONCURRENCY or 4).",
    ),
    overwrite_markdown: bool = typer.Option(
        False, "--overwrite-markdown", help="Re-run docling even if cached."
    ),
    recursive: bool = typer.Option(
        False,
        "--recursive",
        "-r",
        help="Walk subdirectories of PDF_DIR. Cache keys are disambiguated by relative path.",
    ),
    ocr: bool = typer.Option(
        False, "--ocr", help="Force full-page OCR via EasyOCR for every PDF (for scanned PDFs)."
    ),
    ocr_auto: bool = typer.Option(
        False,
        "--ocr-auto",
        help=(
            "Two-pass OCR: convert without OCR first, then re-run image-only outputs with OCR. "
            "Cheaper than --ocr when most PDFs have selectable text."
        ),
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip cost-estimate confirmation."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run the full PDF -> citation graph pipeline."""
    _configure_logging(verbose)
    cfg = _dedup_config(threshold, title_weight, authors_weight, journal_weight, year_window)
    ecfg = _enrich_config(
        enrich_contact,
        enrich_threshold,
        enrich_timeout,
        enrich_year_penalty,
        enrich_retry_attempts,
        enrich_retry_wait,
        enrich_max_workers,
        openalex_api_key,
    )
    ocr_mode = _resolve_ocr_mode(ocr, ocr_auto)
    pipeline = Pipeline(
        pdf_dir=pdf_dir,
        out_dir=out,
        model=model,
        enrich=enrich,
        enrich_config=ecfg,
        dedup_config=cfg,
        overwrite_markdown=overwrite_markdown,
        recursive=recursive,
        ocr=ocr_mode,
        llm_concurrency=llm_concurrency,
    )

    if not yes:
        # Convert PDFs first so we can estimate token cost before any LLM calls.
        _run_stage(pipeline.convert_pdfs, next_hint=None)
        est = pipeline.estimate_extraction_cost()
        typer.echo("\nCost estimate for LLM extraction stages:\n")
        typer.echo(est.format_summary())
        typer.echo()
        if not typer.confirm("Proceed with extraction?", default=True):
            typer.echo("Aborted.")
            raise typer.Exit(code=0)
        typer.echo()

    result = pipeline.run()  # convert_pdfs() is idempotent — cached files are skipped
    n_core = (
        int((result.works["ring"] == 0).sum())
        if "ring" in result.works.columns
        else 0
    )
    typer.echo(
        f"Done. {n_core} core works, "
        f"{len(result.works)} works total, "
        f"{len(result.graph)} citation edges."
    )
    layout = pipeline.layout
    _warn_conversion_quality(layout, ocr_attempted=bool(ocr_mode))
    if layout.papers_no_references_json.exists():
        import json as _json
        entries = _json.loads(layout.papers_no_references_json.read_text(encoding="utf-8"))
        typer.secho(
            f"{len(entries)} paper(s) yielded no references (image-only or unreadable PDF?). See:",
            fg=typer.colors.YELLOW,
        )
        typer.secho(f"  {layout.papers_no_references_json}", fg=typer.colors.YELLOW)
    if layout.source_duplicates_json.exists():
        import json as _json
        groups = _json.loads(layout.source_duplicates_json.read_text(encoding="utf-8"))
        n_dup = sum(len(g.get("duplicate_source_files", [])) for g in groups)
        typer.secho(
            f"{n_dup} markdown file(s) are duplicate PDFs; remove the originals and re-run. See:",
            fg=typer.colors.YELLOW,
        )
        typer.secho(f"  {layout.source_duplicates_json}", fg=typer.colors.YELLOW)
    if layout.metadata_failures_jsonl.exists() or layout.references_failures_jsonl.exists():
        typer.secho(
            "Some papers failed; re-running will retry them. See:",
            fg=typer.colors.YELLOW,
        )
        for p in (layout.metadata_failures_jsonl, layout.references_failures_jsonl):
            if p.exists():
                typer.secho(f"  {p}", fg=typer.colors.YELLOW)
    typer.echo(f"Outputs written under: {out.resolve()}")


@app.command()
def estimate(
    out: Path = typer.Option(Path("./out"), "--out", "-o"),
    model: str | None = typer.Option(None, "--model", help="Gemini model id."),
) -> None:
    """Estimate the LLM token cost of extracting metadata and references.

    Reads existing markdown files from ``out/markdown/`` (run
    ``citegraph convert`` first) and reports how many tokens stages 2 and 3
    would consume, broken down by cached vs. uncached files.
    """
    pipeline = Pipeline(pdf_dir=None, out_dir=out, model=model)
    try:
        est = pipeline.estimate_extraction_cost()
    except StageNotReadyError as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from e
    typer.echo(est.format_summary())


@app.command()
def convert(
    pdf_dir: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True),
    out: Path = typer.Option(Path("./out"), "--out", "-o"),
    overwrite_markdown: bool = typer.Option(
        False, "--overwrite-markdown", help="Re-run docling even if cached."
    ),
    recursive: bool = typer.Option(
        False,
        "--recursive",
        "-r",
        help="Walk subdirectories of PDF_DIR. Cache keys are disambiguated by relative path.",
    ),
    ocr: bool = typer.Option(
        False, "--ocr", help="Force full-page OCR via EasyOCR for every PDF (for scanned PDFs)."
    ),
    ocr_auto: bool = typer.Option(
        False,
        "--ocr-auto",
        help=(
            "Two-pass OCR: convert without OCR first, then re-run image-only outputs with OCR. "
            "Cheaper than --ocr when most PDFs have selectable text."
        ),
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Stage 1: convert PDFs to markdown via docling."""
    _configure_logging(verbose)
    ocr_mode = _resolve_ocr_mode(ocr, ocr_auto)
    pipeline = Pipeline(
        pdf_dir=pdf_dir,
        out_dir=out,
        overwrite_markdown=overwrite_markdown,
        recursive=recursive,
        ocr=ocr_mode,
    )
    paths = pipeline.convert_pdfs()
    typer.echo(f"Converted {len(paths)} PDFs. Markdown in {pipeline.layout.markdown_dir}.")
    _warn_conversion_quality(pipeline.layout, ocr_attempted=bool(ocr_mode))
    _next_step_hint(f"`citegraph metadata --out {out}` (inspect markdown/ first if you want).")


@app.command()
def metadata(
    out: Path = typer.Option(Path("./out"), "--out", "-o"),
    model: str | None = typer.Option(None, "--model"),
    llm_concurrency: int | None = typer.Option(
        None,
        "--llm-concurrency",
        min=1,
        help="Maximum concurrent Gemini metadata extraction calls.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Stage 2: extract per-paper metadata from cached markdown."""
    _configure_logging(verbose)
    pipeline = Pipeline(
        pdf_dir=None,
        out_dir=out,
        model=model,
        llm_concurrency=llm_concurrency,
    )

    def _go() -> None:
        df = pipeline.extract_paper_metadata()
        typer.echo(f"Wrote {pipeline.layout.sources_csv} ({len(df)} sources).")
        if pipeline.layout.source_duplicates_json.exists():
            import json as _json
            groups = _json.loads(
                pipeline.layout.source_duplicates_json.read_text(encoding="utf-8")
            )
            n_dup = sum(len(g.get("duplicate_source_files", [])) for g in groups)
            typer.secho(
                f"{n_dup} markdown file(s) appear to be duplicate PDFs. See "
                f"{pipeline.layout.source_duplicates_json}",
                fg=typer.colors.YELLOW,
            )

    _run_stage(_go, next_hint=f"`citegraph references --out {out}`.")


@app.command()
def references(
    out: Path = typer.Option(Path("./out"), "--out", "-o"),
    model: str | None = typer.Option(None, "--model"),
    llm_concurrency: int | None = typer.Option(
        None,
        "--llm-concurrency",
        min=1,
        help="Maximum concurrent Gemini reference extraction calls.",
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip cost-estimate confirmation."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Stage 3: extract per-paper reference lists from cached markdown."""
    _configure_logging(verbose)
    pipeline = Pipeline(
        pdf_dir=None,
        out_dir=out,
        model=model,
        llm_concurrency=llm_concurrency,
    )

    if not yes:
        try:
            est = pipeline.estimate_extraction_cost()
        except StageNotReadyError as e:
            typer.secho(str(e), fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from e
        if est.n_references_to_process > 0:
            typer.echo("\nCost estimate for references extraction:\n")
            typer.echo(est.format_references_summary())
            typer.echo()
            if not typer.confirm("Proceed with references extraction?", default=True):
                typer.echo("Aborted.")
                raise typer.Exit(code=0)
            typer.echo()

    def _go() -> None:
        df = pipeline.extract_paper_references()
        typer.echo(f"Wrote {pipeline.layout.citations_raw_csv} ({len(df)} raw citations).")
        if pipeline.layout.papers_no_references_json.exists():
            import json as _json
            entries = _json.loads(
                pipeline.layout.papers_no_references_json.read_text(encoding="utf-8")
            )
            typer.secho(
                f"{len(entries)} paper(s) yielded no references. See "
                f"{pipeline.layout.papers_no_references_json}",
                fg=typer.colors.YELLOW,
            )

    _run_stage(_go, next_hint=f"`citegraph dedup --out {out}`.")


@app.command()
def dedup(
    citations_raw_csv: Path | None = typer.Argument(
        None,
        exists=True,
        dir_okay=False,
        help="Path to citations_raw.csv. Defaults to <out>/citations_raw.csv.",
    ),
    out: Path = typer.Option(Path("./out"), "--out", "-o"),
    threshold: float = typer.Option(_DEDUP_DEFAULTS.threshold, "--threshold"),
    title_weight: float = typer.Option(_DEDUP_DEFAULTS.title_weight, "--title-weight"),
    authors_weight: float = typer.Option(_DEDUP_DEFAULTS.authors_weight, "--authors-weight"),
    journal_weight: float = typer.Option(_DEDUP_DEFAULTS.journal_weight, "--journal-weight"),
    year_window: int = typer.Option(_DEDUP_DEFAULTS.year_window, "--year-window"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Stage 4: canonicalize sources + citations into works and the citation graph."""
    _configure_logging(verbose)
    layout = OutLayout(out)
    layout.ensure()
    cfg = _dedup_config(threshold, title_weight, authors_weight, journal_weight, year_window)

    citations_raw: pd.DataFrame | None = None
    if citations_raw_csv is not None:
        citations_raw = read_stage_csv(citations_raw_csv, columns=CITATION_COLUMNS)

    pipeline = Pipeline(pdf_dir=None, out_dir=out, dedup_config=cfg)
    try:
        works, graph = pipeline.deduplicate(citations_raw=citations_raw)
    except StageNotReadyError as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from e
    except ValueError as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from e

    n_core = int((works["ring"] == 0).sum())
    typer.echo(
        f"Canonicalized into {len(works)} works ({n_core} core, "
        f"{len(graph)} edges). Wrote {layout.works_csv} and {layout.graph_csv}."
    )
    _next_step_hint(
        f"Inspect works.csv. Optionally `citegraph enrich --out {out}` for DOI lookup."
    )


@app.command()
def enrich(
    out: Path = typer.Option(Path("./out"), "--out", "-o"),
    enrich_contact: str = typer.Option("", "--enrich-contact", help="Contact email for CrossRef polite pool."),
    enrich_threshold: float = typer.Option(90.0, "--enrich-threshold", help="Title match threshold."),
    enrich_timeout: float = typer.Option(15.0, "--enrich-timeout", help="HTTP timeout in seconds."),
    enrich_year_penalty: float = typer.Option(
        8.0,
        "--enrich-year-penalty",
        help="Score penalty applied when input and candidate years differ.",
    ),
    enrich_retry_attempts: int = typer.Option(
        3,
        "--enrich-retry-attempts",
        min=1,
        help="Attempts for transient enrichment HTTP errors such as 429/503.",
    ),
    enrich_retry_wait: float = typer.Option(
        0.25,
        "--enrich-retry-wait",
        min=0.0,
        help="Base exponential-backoff wait in seconds for enrichment retries.",
    ),
    enrich_max_workers: int = typer.Option(
        8,
        "--enrich-max-workers",
        min=1,
        help="Maximum concurrent CrossRef/OpenAlex enrichment lookups.",
    ),
    openalex_api_key: str = typer.Option(
        "",
        "--openalex-api-key",
        help="OpenAlex API key (default: OPENALEX_API_KEY env var / .env).",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Stage 5: optional CrossRef/OpenAlex enrichment of references.csv."""
    _configure_logging(verbose)
    ecfg = _enrich_config(
        enrich_contact,
        enrich_threshold,
        enrich_timeout,
        enrich_year_penalty,
        enrich_retry_attempts,
        enrich_retry_wait,
        enrich_max_workers,
        openalex_api_key,
    )
    pipeline = Pipeline(pdf_dir=None, out_dir=out, enrich=True, enrich_config=ecfg)

    def _go() -> None:
        df = pipeline.maybe_enrich()
        typer.echo(
            f"Enriched {len(df)} works → {pipeline.layout.enriched_works_csv}."
        )

    _run_stage(_go, next_hint=None)


@app.command()
def authors(
    out: Path = typer.Option(Path("./out"), "--out", "-o"),
    merge_mode: str = typer.Option(
        "strict",
        "--merge-mode",
        help="'strict' (precision-first, default) or 'loose' (collapse by surname+first-initial).",
    ),
    aliases: Path | None = typer.Option(
        None,
        "--aliases",
        help="Optional CSV of hand-curated overrides: cluster_id,canonical_id. "
        "Defaults to <out>/author_aliases.csv when it exists.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Stage 4b: cluster authors across the corpus into canonical records.

    Reads ``works.csv`` from ``--out`` and writes ``authors.csv`` +
    ``author_citations.csv``. When ``enriched_works.csv`` is present,
    OpenAlex / ORCID ids are used as ground truth for identity.
    """
    _configure_logging(verbose)
    if merge_mode not in {"strict", "loose"}:
        typer.secho(
            f"--merge-mode must be 'strict' or 'loose', got {merge_mode!r}.",
            fg=typer.colors.RED, err=True,
        )
        raise typer.Exit(code=1)

    pipeline = Pipeline(
        pdf_dir=None,
        out_dir=out,
        author_config=AuthorClusterConfig(merge_mode=merge_mode),
    )
    if aliases is not None:
        # User-specified path overrides the default. Copy the file in
        # under the canonical name so `Pipeline.normalize_authors` finds
        # it via the layout, without forcing a new code path.
        import shutil
        if aliases.exists():
            shutil.copy(aliases, pipeline.layout.author_aliases_csv)
        else:
            typer.secho(
                f"--aliases path does not exist: {aliases}",
                fg=typer.colors.RED, err=True,
            )
            raise typer.Exit(code=1)

    def _go() -> None:
        layout = pipeline.layout
        if not layout.enriched_works_csv.exists():
            n_cached = layout.enrichment_cache_count()
            if n_cached:
                typer.secho(
                    f"{n_cached} cached enrichment result(s) found in "
                    f"{layout.enrichment_dir} but enriched_works.csv is "
                    "missing. Run `citegraph enrich` first so OpenAlex/ORCID "
                    "ids anchor author clustering.",
                    fg=typer.colors.YELLOW,
                )
        authors_df, citations_df = pipeline.normalize_authors()
        typer.echo(
            f"Clustered {len(citations_df)} author occurrences into "
            f"{len(authors_df)} canonical authors. "
            f"Wrote {pipeline.layout.authors_csv}."
        )
        if pipeline.layout.author_review_json.exists():
            import json as _json
            review = _json.loads(
                pipeline.layout.author_review_json.read_text(encoding="utf-8")
            )
            typer.secho(
                f"{len(review)} cluster(s) flagged for review. See "
                f"{pipeline.layout.author_review_json}",
                fg=typer.colors.YELLOW,
            )

    _run_stage(_go, next_hint=None)


@app.command()
def report(
    out: Path = typer.Option(Path("./out"), "--out", "-o"),
    open_browser: bool = typer.Option(
        False, "--open", help="Open the written report.html in a web browser."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Write a self-contained QC dashboard (``report.html``) into ``--out``.

    Reads whatever artifacts exist — it is safe (and useful) to re-run
    after every pipeline stage. No LLM or network calls are made.
    """
    _configure_logging(verbose)
    if not out.is_dir():
        typer.secho(
            f"Output directory does not exist: {out}. "
            f"Run `citegraph convert <pdf_dir> --out {out}` first.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    from citegraph.html_report import collect_report_data, write_report

    data = collect_report_data(out)
    path = write_report(out, data=data)
    typer.echo(f"Wrote {path}.")
    n_problems = data.get("problems", {}).get("total", 0)
    if n_problems:
        typer.secho(
            f"{n_problems} problem(s) flagged — open the report for details.",
            fg=typer.colors.YELLOW,
        )
    if open_browser:
        import webbrowser

        webbrowser.open(path.resolve().as_uri())


@app.command()
def ui(
    out: Path = typer.Option(Path("./out"), "--out", "-o"),
    port: int = typer.Option(8765, "--port", help="Port to serve on (binds 127.0.0.1 only)."),
    open_browser: bool = typer.Option(
        False, "--open", help="Open the live monitor in a web browser."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Serve a read-only live monitor for ``--out`` on localhost.

    Polls the out directory as pipeline stages run and re-renders stage
    progress, problems, recent file activity, and the per-paper table.
    It never writes into ``--out`` and works before the directory exists
    (the page fills in as artifacts appear). Ctrl+C stops it.
    """
    _configure_logging(verbose)
    from citegraph.webui import create_server

    server = create_server(out, port=port, verbose=verbose)
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    typer.echo(f"citegraph live monitor: {url}  (watching {out}; Ctrl+C to stop)")
    if not out.is_dir():
        typer.secho(
            f"{out} does not exist yet — the page will fill in once a stage writes to it.",
            fg=typer.colors.YELLOW,
        )
    if open_browser:
        import webbrowser

        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        typer.echo("\nStopped.")
    finally:
        server.server_close()


@app.command()
def status(
    out: Path = typer.Option(Path("./out"), "--out", "-o"),
) -> None:
    """Report which pipeline artifacts exist under ``out``."""
    layout = OutLayout(out)

    md_count = (
        len(list(layout.markdown_dir.glob("*.md"))) if layout.markdown_dir.exists() else 0
    )
    meta_count = (
        len(list(layout.metadata_dir.glob("*.json"))) if layout.metadata_dir.exists() else 0
    )
    ref_count = (
        len(list(layout.references_dir.glob("*.json"))) if layout.references_dir.exists() else 0
    )

    def _rows(path: Path) -> str:
        if not path.exists():
            return "missing"
        try:
            return f"{len(pd.read_csv(path))} rows"
        except Exception as e:  # pragma: no cover - defensive
            return f"unreadable ({e})"

    typer.echo(f"out_dir: {out.resolve()}")
    typer.echo(f"  markdown/              {md_count} files")
    typer.echo(f"  metadata/              {meta_count} cached")
    typer.echo(f"  references/            {ref_count} cached")
    typer.echo(f"  sources.csv            {_rows(layout.sources_csv)}")
    typer.echo(f"  citations_raw.csv      {_rows(layout.citations_raw_csv)}")
    typer.echo(f"  works.csv              {_rows(layout.works_csv)}")
    typer.echo(f"  enriched_works.csv     {_rows(layout.enriched_works_csv)}")
    typer.echo(f"  citation_graph.csv     {_rows(layout.graph_csv)}")
    typer.echo(f"  authors.csv            {_rows(layout.authors_csv)}")
    typer.echo(f"  author_citations.csv   {_rows(layout.author_citations_csv)}")


if __name__ == "__main__":  # pragma: no cover
    app()
