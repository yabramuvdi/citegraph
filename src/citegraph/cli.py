"""Command-line interface for citegraph.

Two ways to use it:

- ``citegraph run`` runs the full pipeline end-to-end.
- The per-stage commands (``convert``, ``metadata``, ``references``,
  ``dedup``, ``enrich``) run one stage at a time, reading prior outputs
  from ``--out``. ``citegraph status`` reports which artifacts exist.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import typer
from rich.logging import RichHandler

from citegraph.dedup import DedupConfig, dedup_references
from citegraph.enrich import EnrichConfig
from citegraph.io import OutLayout
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
) -> EnrichConfig:
    return EnrichConfig(
        contact_email=contact,
        title_match_threshold=threshold,
        timeout_s=timeout,
    )


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
    model: str | None = typer.Option(None, "--model", help="Gemini model id."),
    threshold: float = typer.Option(85.0, "--threshold", help="Dedup similarity threshold."),
    title_weight: float = typer.Option(0.7, "--title-weight"),
    authors_weight: float = typer.Option(0.3, "--authors-weight"),
    journal_weight: float = typer.Option(0.0, "--journal-weight"),
    year_window: int = typer.Option(1, "--year-window"),
    overwrite_markdown: bool = typer.Option(
        False, "--overwrite-markdown", help="Re-run docling even if cached."
    ),
    recursive: bool = typer.Option(
        False,
        "--recursive",
        "-r",
        help="Walk subdirectories of PDF_DIR. Cache keys are disambiguated by relative path.",
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip cost-estimate confirmation."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run the full PDF -> citation graph pipeline."""
    _configure_logging(verbose)
    cfg = _dedup_config(threshold, title_weight, authors_weight, journal_weight, year_window)
    ecfg = _enrich_config(enrich_contact, enrich_threshold, enrich_timeout)
    pipeline = Pipeline(
        pdf_dir=pdf_dir,
        out_dir=out,
        model=model,
        enrich=enrich,
        enrich_config=ecfg,
        dedup_config=cfg,
        overwrite_markdown=overwrite_markdown,
        recursive=recursive,
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
    typer.echo(
        f"Done. {len(result.papers)} papers, "
        f"{len(result.references)} unique references, "
        f"{len(result.graph)} citation edges."
    )
    layout = pipeline.layout
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
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Stage 1: convert PDFs to markdown via docling."""
    _configure_logging(verbose)
    pipeline = Pipeline(
        pdf_dir=pdf_dir,
        out_dir=out,
        overwrite_markdown=overwrite_markdown,
        recursive=recursive,
    )
    paths = pipeline.convert_pdfs()
    typer.echo(f"Converted {len(paths)} PDFs. Markdown in {pipeline.layout.markdown_dir}.")
    _next_step_hint(f"`citegraph metadata --out {out}` (inspect markdown/ first if you want).")


@app.command()
def metadata(
    out: Path = typer.Option(Path("./out"), "--out", "-o"),
    model: str | None = typer.Option(None, "--model"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Stage 2: extract per-paper metadata from cached markdown."""
    _configure_logging(verbose)
    pipeline = Pipeline(pdf_dir=None, out_dir=out, model=model)

    def _go() -> None:
        df = pipeline.extract_paper_metadata()
        typer.echo(f"Wrote {pipeline.layout.papers_csv} ({len(df)} papers).")

    _run_stage(_go, next_hint=f"`citegraph references --out {out}`.")


@app.command()
def references(
    out: Path = typer.Option(Path("./out"), "--out", "-o"),
    model: str | None = typer.Option(None, "--model"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip cost-estimate confirmation."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Stage 3: extract per-paper reference lists from cached markdown."""
    _configure_logging(verbose)
    pipeline = Pipeline(pdf_dir=None, out_dir=out, model=model)

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
        typer.echo(f"Wrote {pipeline.layout.references_raw_csv} ({len(df)} raw refs).")

    _run_stage(_go, next_hint=f"`citegraph dedup --out {out}`.")


@app.command()
def dedup(
    references_raw_csv: Path | None = typer.Argument(
        None,
        exists=True,
        dir_okay=False,
        help="Path to references_raw.csv. Defaults to <out>/references_raw.csv.",
    ),
    out: Path = typer.Option(Path("./out"), "--out", "-o"),
    threshold: float = typer.Option(85.0, "--threshold"),
    title_weight: float = typer.Option(0.7, "--title-weight"),
    authors_weight: float = typer.Option(0.3, "--authors-weight"),
    journal_weight: float = typer.Option(0.0, "--journal-weight"),
    year_window: int = typer.Option(1, "--year-window"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Stage 4: deduplicate references and build the citation graph."""
    _configure_logging(verbose)
    layout = OutLayout(out)
    layout.ensure()
    cfg = _dedup_config(threshold, title_weight, authors_weight, journal_weight, year_window)

    csv_path = references_raw_csv or layout.references_raw_csv
    if not csv_path.exists():
        typer.secho(
            f"Missing {csv_path}. Run `citegraph references --out {out}` first.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    raw_refs = pd.read_csv(csv_path)
    canonical, mapping = dedup_references(raw_refs, cfg)
    graph = (
        pd.DataFrame(
            {
                "citing_id": raw_refs["citing_id"].values,
                "cited_id": mapping.values,
            }
        )
        .drop_duplicates()
        .reset_index(drop=True)
    )
    canonical.to_csv(layout.references_csv)
    graph.to_csv(layout.graph_csv, index=False)
    typer.echo(
        f"Deduplicated {len(raw_refs)} -> {len(canonical)} references "
        f"({len(graph)} edges). Wrote {layout.references_csv} and {layout.graph_csv}."
    )
    _next_step_hint(
        f"Inspect references.csv. Optionally `citegraph enrich --out {out}` for DOI lookup."
    )


@app.command()
def enrich(
    out: Path = typer.Option(Path("./out"), "--out", "-o"),
    enrich_contact: str = typer.Option("", "--enrich-contact", help="Contact email for CrossRef polite pool."),
    enrich_threshold: float = typer.Option(90.0, "--enrich-threshold", help="Title match threshold."),
    enrich_timeout: float = typer.Option(15.0, "--enrich-timeout", help="HTTP timeout in seconds."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Stage 5: optional CrossRef/OpenAlex enrichment of references.csv."""
    _configure_logging(verbose)
    ecfg = _enrich_config(enrich_contact, enrich_threshold, enrich_timeout)
    pipeline = Pipeline(pdf_dir=None, out_dir=out, enrich=True, enrich_config=ecfg)

    def _go() -> None:
        df = pipeline.maybe_enrich()
        typer.echo(
            f"Enriched {len(df)} references → {pipeline.layout.enriched_references_csv}."
        )

    _run_stage(_go, next_hint=None)


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
    typer.echo(f"  papers.csv             {_rows(layout.papers_csv)}")
    typer.echo(f"  references_raw.csv     {_rows(layout.references_raw_csv)}")
    typer.echo(f"  references.csv         {_rows(layout.references_csv)}")
    typer.echo(f"  enriched_references.csv {_rows(layout.enriched_references_csv)}")
    typer.echo(f"  citation_graph.csv     {_rows(layout.graph_csv)}")


if __name__ == "__main__":  # pragma: no cover
    app()
