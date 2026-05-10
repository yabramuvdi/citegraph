"""High-level orchestration: PDFs -> deduplicated citation graph.

The :class:`Pipeline` is the only object library users normally need to
construct. Everything is checkpointed on disk under ``out_dir`` so that
interrupted runs can resume cheaply.

Each stage method accepts its inputs explicitly (used by :meth:`run`) but
also falls back to reading the upstream artifact from ``out_dir`` if no
input is provided. That makes a progressive, stage-by-stage workflow as
easy as::

    p = Pipeline(pdf_dir, out_dir)
    p.convert_pdfs()           # inspect out_dir/markdown/
    p.extract_paper_metadata() # inspect papers.csv
    p.extract_paper_references()
    p.deduplicate()
    p.maybe_enrich()
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from citegraph.cost_estimation import ExtractionEstimate

import pandas as pd

from citegraph._progress import iter_with_progress
from citegraph.dedup import DedupConfig, dedup_references
from citegraph.enrich import EnrichConfig
from citegraph.extract_metadata import extract_metadata_from_markdown, metadata_to_record
from citegraph.extract_references import extract_references_from_markdown
from citegraph.io import OutLayout, read_json, write_json, write_pydantic, write_pydantic_list
from citegraph.llm import GeminiClient
from citegraph.pdf_to_markdown import convert_directory
from citegraph.schemas import PaperMetadata, PipelineResult, Reference

logger = logging.getLogger(__name__)


class StageNotReadyError(RuntimeError):
    """Raised when a stage is invoked but its upstream artifact is missing."""


@dataclass
class PaperFailure:
    """One per-paper failure from a stage's extraction loop."""

    source_file: str
    stage: str  # "metadata" or "references"
    error_class: str
    error_message: str


def _write_failures(path: Path, failures: list[PaperFailure]) -> None:
    """Persist ``failures`` as JSONL, or remove the file when none occurred.

    Encoding the "no failures" state as file *absence* gives a single clean
    signal: ``path.exists()`` <=> at least one paper failed in this stage.
    """
    if not failures:
        if path.exists():
            path.unlink()
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for fail in failures:
            f.write(json.dumps(asdict(fail), ensure_ascii=False) + "\n")


class Pipeline:
    """End-to-end pipeline from a folder of PDFs to a citation graph.

    Parameters
    ----------
    pdf_dir:
        Folder containing the input PDF files.
    out_dir:
        Folder where outputs and checkpoints will be written. Created if
        it doesn't exist.
    model:
        Gemini model identifier. Defaults to the value in settings.
    enrich:
        If ``True`` and the ``[crossref]`` extra is installed, run a
        CrossRef/OpenAlex enrichment pass on the deduplicated references.
    dedup_config:
        Optional :class:`DedupConfig`. If unset, sensible defaults are
        used.
    overwrite_markdown:
        If ``True``, re-run docling even when cached markdown exists.
    show_progress:
        If ``True`` (default), render a rich progress bar during the
        PDF -> markdown stage. Disable for non-interactive runs.
    client:
        Optional pre-built :class:`GeminiClient` (useful for tests).
    """

    def __init__(
        self,
        pdf_dir: str | Path | None,
        out_dir: str | Path,
        *,
        model: str | None = None,
        enrich: bool = False,
        enrich_config: EnrichConfig | None = None,
        dedup_config: DedupConfig | None = None,
        overwrite_markdown: bool = False,
        recursive: bool = False,
        show_progress: bool = True,
        client: GeminiClient | None = None,
    ) -> None:
        self.pdf_dir = Path(pdf_dir) if pdf_dir is not None else None
        self.layout = OutLayout(Path(out_dir))
        self.layout.ensure()
        self.dedup_config = dedup_config or DedupConfig()
        self.enrich = enrich
        self.enrich_config = enrich_config or EnrichConfig()
        self.overwrite_markdown = overwrite_markdown
        self.recursive = recursive
        self.show_progress = show_progress
        self._client_kwargs = {"model": model} if model else {}
        self._client = client

    @property
    def client(self) -> GeminiClient:
        if self._client is None:
            self._client = GeminiClient(**self._client_kwargs)
        return self._client

    # ------------------------------------------------------------------
    # Disk loaders (used by stages when called with no in-memory input)
    # ------------------------------------------------------------------
    def _load_markdown_paths(self) -> list[Path]:
        md_dir = self.layout.markdown_dir
        paths = sorted(md_dir.glob("*.md"))
        if not paths:
            raise StageNotReadyError(
                f"No markdown files found in {md_dir}. "
                "Run `citegraph convert <pdf_dir>` first."
            )
        return paths

    def _load_papers(self) -> pd.DataFrame:
        path = self.layout.papers_csv
        if not path.exists():
            raise StageNotReadyError(
                f"Missing {path}. Run `citegraph metadata` first."
            )
        return pd.read_csv(path)

    def _load_raw_refs(self) -> pd.DataFrame:
        path = self.layout.references_raw_csv
        if not path.exists():
            raise StageNotReadyError(
                f"Missing {path}. Run `citegraph references` first."
            )
        return pd.read_csv(path)

    def _load_references(self) -> pd.DataFrame:
        path = self.layout.references_csv
        if not path.exists():
            raise StageNotReadyError(
                f"Missing {path}. Run `citegraph dedup` first."
            )
        return pd.read_csv(path, index_col="id")

    # ------------------------------------------------------------------
    # Stage 1: PDFs -> markdown
    # ------------------------------------------------------------------
    def convert_pdfs(self) -> list[Path]:
        if self.pdf_dir is None:
            raise StageNotReadyError(
                "Pipeline was constructed without pdf_dir; cannot convert PDFs."
            )
        return convert_directory(
            self.pdf_dir,
            self.layout.markdown_dir,
            overwrite=self.overwrite_markdown,
            recursive=self.recursive,
            show_progress=self.show_progress,
        )

    # ------------------------------------------------------------------
    # Stage 2: markdown -> per-paper metadata
    # ------------------------------------------------------------------
    def extract_paper_metadata(
        self, markdown_paths: list[Path] | None = None
    ) -> pd.DataFrame:
        if markdown_paths is None:
            markdown_paths = self._load_markdown_paths()

        records: list[dict] = []
        failures: list[PaperFailure] = []
        for md in iter_with_progress(
            markdown_paths,
            show_progress=self.show_progress,
            description="Extracting metadata",
            item_label=lambda p: p.name,
        ):
            cache = self.layout.metadata_dir / f"{md.stem}.json"
            try:
                if cache.exists():
                    logger.debug("Loading metadata from cache: %s", cache.name)
                    meta = PaperMetadata.model_validate(read_json(cache))
                else:
                    logger.info("Extracting metadata from %s", md.name)
                    meta = extract_metadata_from_markdown(md, client=self.client)
                    write_pydantic(cache, meta)
                records.append(metadata_to_record(meta, source_file=md.name))
            except Exception as exc:  # noqa: BLE001 - one bad paper shouldn't kill the run
                logger.error("Metadata extraction failed for %s: %s", md.name, exc)
                failures.append(
                    PaperFailure(
                        source_file=md.name,
                        stage="metadata",
                        error_class=type(exc).__name__,
                        error_message=str(exc),
                    )
                )

        _write_failures(self.layout.metadata_failures_jsonl, failures)
        if failures:
            logger.warning(
                "Metadata stage finished with %d failure(s); see %s",
                len(failures),
                self.layout.metadata_failures_jsonl,
            )

        df = pd.DataFrame(records)
        if not df.empty:
            df = df.drop_duplicates(subset="id", keep="first").reset_index(drop=True)
        df.to_csv(self.layout.papers_csv, index=False)
        logger.info("Wrote %s (%d rows)", self.layout.papers_csv, len(df))
        return df

    # ------------------------------------------------------------------
    # Stage 3: markdown -> per-paper references
    # ------------------------------------------------------------------
    def extract_paper_references(
        self,
        markdown_paths: list[Path] | None = None,
        papers_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        if markdown_paths is None:
            markdown_paths = self._load_markdown_paths()
        if papers_df is None:
            papers_df = self._load_papers()

        source_to_id = dict(zip(papers_df["source_file"], papers_df["id"], strict=False))
        rows: list[dict] = []
        failures: list[PaperFailure] = []
        for md in iter_with_progress(
            markdown_paths,
            show_progress=self.show_progress,
            description="Extracting references",
            item_label=lambda p: p.name,
        ):
            citing_id = source_to_id.get(md.name)
            if citing_id is None:
                logger.warning("No paper id for %s; skipping references", md.name)
                continue

            cache = self.layout.references_dir / f"{md.stem}.json"
            try:
                if cache.exists():
                    refs = [Reference.model_validate(r) for r in read_json(cache)]
                    logger.debug("Loaded %d cached references for %s", len(refs), md.name)
                else:
                    logger.info("Extracting references from %s", md.name)
                    refs = extract_references_from_markdown(md, client=self.client)
                    write_pydantic_list(cache, refs)
                for ref in refs:
                    rows.append({**ref.model_dump(), "citing_id": citing_id})
            except Exception as exc:  # noqa: BLE001 - one bad paper shouldn't kill the run
                logger.error("References extraction failed for %s: %s", md.name, exc)
                failures.append(
                    PaperFailure(
                        source_file=md.name,
                        stage="references",
                        error_class=type(exc).__name__,
                        error_message=str(exc),
                    )
                )

        _write_failures(self.layout.references_failures_jsonl, failures)
        if failures:
            logger.warning(
                "References stage finished with %d failure(s); see %s",
                len(failures),
                self.layout.references_failures_jsonl,
            )

        df = pd.DataFrame(rows)
        df.to_csv(self.layout.references_raw_csv, index=False)
        logger.info("Wrote %s (%d rows)", self.layout.references_raw_csv, len(df))
        return df

    # ------------------------------------------------------------------
    # Stage 4: dedup references and build citation graph
    # ------------------------------------------------------------------
    def deduplicate(
        self,
        raw_refs: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if raw_refs is None:
            raw_refs = self._load_raw_refs()

        if raw_refs.empty:
            empty_refs = pd.DataFrame(columns=["Title", "Authors", "Year", "Journal"])
            empty_refs.index.name = "id"
            empty_graph = pd.DataFrame(columns=["citing_id", "cited_id"])
            empty_refs.to_csv(self.layout.references_csv)
            empty_graph.to_csv(self.layout.graph_csv, index=False)
            return empty_refs, empty_graph

        canonical_df, mapping = dedup_references(raw_refs, self.dedup_config)
        graph = pd.DataFrame(
            {
                "citing_id": raw_refs["citing_id"].values,
                "cited_id": mapping.values,
            }
        ).drop_duplicates().reset_index(drop=True)

        canonical_df.to_csv(self.layout.references_csv)
        graph.to_csv(self.layout.graph_csv, index=False)
        logger.info(
            "Wrote %s (%d rows) and %s (%d edges)",
            self.layout.references_csv,
            len(canonical_df),
            self.layout.graph_csv,
            len(graph),
        )
        return canonical_df, graph

    # ------------------------------------------------------------------
    # Stage 5: optional enrichment
    # ------------------------------------------------------------------
    def maybe_enrich(self, references: pd.DataFrame | None = None) -> pd.DataFrame:
        if references is None:
            references = self._load_references()
        if not self.enrich:
            return references
        from citegraph.enrich import enrich_references

        enriched = enrich_references(references, cfg=self.enrich_config, layout=self.layout)
        enriched.to_csv(self.layout.enriched_references_csv)
        logger.info("Wrote enriched %s", self.layout.enriched_references_csv)
        return enriched

    # ------------------------------------------------------------------
    # Cost estimation (no API calls)
    # ------------------------------------------------------------------
    def estimate_extraction_cost(self) -> ExtractionEstimate:
        """Estimate LLM token usage for stages 2–3 without making any API calls.

        Requires markdown files (stage 1) to exist under ``out_dir/markdown/``.
        Raises :class:`StageNotReadyError` if they are missing.
        """
        from citegraph.config import get_settings
        from citegraph.cost_estimation import estimate_extraction_cost as _estimate

        if self._client is not None:
            model = self._client.model
        else:
            model = self._client_kwargs.get("model") or get_settings().citegraph_model
        return _estimate(layout=self.layout, model=model)

    # ------------------------------------------------------------------
    # Top level
    # ------------------------------------------------------------------
    def run(self) -> PipelineResult:
        markdown_paths = self.convert_pdfs()
        papers = self.extract_paper_metadata(markdown_paths)
        raw_refs = self.extract_paper_references(markdown_paths, papers)
        references, graph = self.deduplicate(raw_refs)
        references = self.maybe_enrich(references)

        run_summary = {
            "n_papers": int(len(papers)),
            "n_references_raw": int(len(raw_refs)),
            "n_references_dedup": int(len(references)),
            "n_edges": int(len(graph)),
            "n_metadata_failures": _count_failures(self.layout.metadata_failures_jsonl),
            "n_references_failures": _count_failures(self.layout.references_failures_jsonl),
            "model": self.client.model,
            "enrich": self.enrich,
            "dedup_config": self.dedup_config.__dict__,
        }
        write_json(self.layout.out_dir / "run_summary.json", run_summary)
        return PipelineResult(papers=papers, references=references, graph=graph)


def _count_failures(path: Path) -> int:
    """Count entries in a failures.jsonl file; 0 if the file doesn't exist."""
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())
