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
from citegraph.authors import AuthorClusterConfig, load_aliases, normalize_authors
from citegraph.dedup import DedupConfig, dedup_references
from citegraph.enrich import EnrichConfig
from citegraph.extract_metadata import extract_metadata_from_markdown, metadata_to_record
from citegraph.extract_references import extract_references_from_markdown
from citegraph.io import OutLayout, read_json, write_json, write_pydantic, write_pydantic_list
from citegraph.llm import GeminiClient
from citegraph.pdf_to_markdown import OCRMode, _is_image_only, convert_directory
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
        author_config: AuthorClusterConfig | None = None,
        overwrite_markdown: bool = False,
        recursive: bool = False,
        ocr: OCRMode = False,
        show_progress: bool = True,
        client: GeminiClient | None = None,
    ) -> None:
        self.pdf_dir = Path(pdf_dir) if pdf_dir is not None else None
        self.layout = OutLayout(Path(out_dir))
        self.layout.ensure()
        self.dedup_config = dedup_config or DedupConfig()
        self.author_config = author_config or AuthorClusterConfig()
        self.enrich = enrich
        self.enrich_config = enrich_config or EnrichConfig()
        self.overwrite_markdown = overwrite_markdown
        self.recursive = recursive
        self.ocr = ocr
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
        paths = convert_directory(
            self.pdf_dir,
            self.layout.markdown_dir,
            overwrite=self.overwrite_markdown,
            recursive=self.recursive,
            show_progress=self.show_progress,
            ocr=self.ocr,
        )
        _check_conversion_quality(paths, self.layout, ocr_attempted=bool(self.ocr))
        return paths

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

        duplicates = _detect_source_duplicates(records, self.dedup_config)
        _write_source_duplicates(self.layout.source_duplicates_json, duplicates)
        if duplicates:
            n_dup_files = sum(len(g["duplicate_source_files"]) for g in duplicates)
            logger.warning(
                "%d markdown file(s) appear to be duplicate PDFs; see %s",
                n_dup_files,
                self.layout.source_duplicates_json,
            )

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

        duplicate_lookup: dict[str, dict] = {}
        if self.layout.source_duplicates_json.exists():
            for group in read_json(self.layout.source_duplicates_json):
                for dup_file in group.get("duplicate_source_files", []):
                    duplicate_lookup[dup_file] = group

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
                if md.name in duplicate_lookup:
                    group = duplicate_lookup[md.name]
                    logger.warning(
                        "Skipping %s — duplicate of '%s' (%s); remove the extra PDF to silence this",
                        md.name,
                        group["canonical_source_file"],
                        group["canonical_paper_id"],
                    )
                else:
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

        failure_source_files = {f.source_file for f in failures}
        papers_with_refs = {row["citing_id"] for row in rows}
        no_ref_entries = [
            {
                "paper_id": citing_id,
                "source_file": md_name,
                "title": papers_df.loc[papers_df["id"] == citing_id, "Title"].iloc[0]
                if not papers_df.loc[papers_df["id"] == citing_id].empty
                else "",
            }
            for md_name, citing_id in source_to_id.items()
            if md_name not in failure_source_files and citing_id not in papers_with_refs
        ]
        _write_no_references(self.layout.papers_no_references_json, no_ref_entries)
        if no_ref_entries:
            logger.warning(
                "%d paper(s) yielded no references; see %s",
                len(no_ref_entries),
                self.layout.papers_no_references_json,
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

        canonical_df, mapping = dedup_references(
            raw_refs, self.dedup_config, show_progress=self.show_progress
        )
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
    # Stage 4b: corpus-wide author normalization (after dedup)
    # ------------------------------------------------------------------
    def normalize_authors(
        self,
        references: pd.DataFrame | None = None,
        papers: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Cluster every author across the corpus into canonical records.

        Runs after :meth:`deduplicate`. Reads ``references.csv`` and
        ``papers.csv`` if no arguments are passed. When
        ``enriched_references.csv`` exists, its OpenAlex / ORCID ids are
        attached to the reference authors so that enrichment acts as
        ground truth for identity.

        Hand-curated overrides are loaded from ``author_aliases.csv`` if
        present (two-column ``cluster_id,canonical_id``) and applied
        after the algorithmic clustering.
        """
        if references is None:
            references = self._load_references()
        if papers is None:
            try:
                papers = self._load_papers()
            except StageNotReadyError:
                papers = None  # reference-only mode is fine

        enriched: pd.DataFrame | None = None
        if self.layout.enriched_references_csv.exists():
            enriched = pd.read_csv(self.layout.enriched_references_csv, index_col="id")
            # ``OpenAlex_Authors`` round-trips as a string in CSV; turn it
            # back into a list of dicts so the normalizer sees structured
            # data. Bad rows are silently dropped to keep the stage resilient.
            if "OpenAlex_Authors" in enriched.columns:
                import ast
                def _parse(val: object) -> object:
                    if isinstance(val, list):
                        return val
                    if isinstance(val, str) and val.strip().startswith("["):
                        try:
                            return ast.literal_eval(val)
                        except (ValueError, SyntaxError):
                            return None
                    return None
                enriched["OpenAlex_Authors"] = enriched["OpenAlex_Authors"].map(_parse)

        aliases = load_aliases(self.layout.author_aliases_csv)

        authors_df, citations_df, review = normalize_authors(
            references=references,
            papers=papers,
            enriched_references=enriched,
            cfg=self.author_config,
            aliases=aliases,
        )

        authors_df.to_csv(self.layout.authors_csv)
        citations_df.to_csv(self.layout.author_citations_csv, index=False)
        _write_author_review(self.layout.author_review_json, review)
        if review:
            logger.warning(
                "%d author cluster(s) flagged for review; see %s",
                len(review),
                self.layout.author_review_json,
            )
        logger.info(
            "Wrote %s (%d authors), %s (%d edges)",
            self.layout.authors_csv,
            len(authors_df),
            self.layout.author_citations_csv,
            len(citations_df),
        )
        return authors_df, citations_df

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
        authors_df, citations_df = self.normalize_authors(references=references, papers=papers)

        run_summary = {
            "n_papers": int(len(papers)),
            "n_references_raw": int(len(raw_refs)),
            "n_references_dedup": int(len(references)),
            "n_edges": int(len(graph)),
            "n_authors": int(len(authors_df)),
            "n_author_citations": int(len(citations_df)),
            "n_author_review_flags": _count_author_review(self.layout.author_review_json),
            "n_metadata_failures": _count_failures(self.layout.metadata_failures_jsonl),
            "n_references_failures": _count_failures(self.layout.references_failures_jsonl),
            "n_source_duplicates": _count_source_duplicates(self.layout.source_duplicates_json),
            "n_papers_no_references": _count_no_references(self.layout.papers_no_references_json),
            "n_conversion_warnings": _count_conversion_warnings(self.layout.conversion_warnings_json),
            "model": self.client.model,
            "enrich": self.enrich,
            "dedup_config": self.dedup_config.__dict__,
            "author_config": self.author_config.__dict__,
        }
        write_json(self.layout.out_dir / "run_summary.json", run_summary)
        return PipelineResult(papers=papers, references=references, graph=graph)


def _count_failures(path: Path) -> int:
    """Count entries in a failures.jsonl file; 0 if the file doesn't exist."""
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _detect_source_duplicates(records: list[dict], cfg: DedupConfig) -> list[dict]:
    """Find records that are duplicate conversions of the same underlying paper.

    Returns one entry per duplicate group (groups with >1 source file).  Each
    entry names the canonical file (first seen), the duplicate files, the paper
    title, and the paper id — enough for the user to identify and remove the
    extra PDFs.
    """
    from citegraph.dedup import compare_papers

    canonical_indices: list[int] = []
    duplicate_of: dict[int, int] = {}

    for i, rec in enumerate(records):
        matched = False
        for canon_idx in canonical_indices:
            if compare_papers(rec, records[canon_idx], cfg):
                duplicate_of[i] = canon_idx
                matched = True
                break
        if not matched:
            canonical_indices.append(i)

    canon_to_dupes: dict[int, list[int]] = {}
    for dup_idx, canon_idx in duplicate_of.items():
        canon_to_dupes.setdefault(canon_idx, []).append(dup_idx)

    groups = []
    for canon_idx, dup_indices in canon_to_dupes.items():
        canon = records[canon_idx]
        groups.append(
            {
                "canonical_source_file": canon["source_file"],
                "canonical_paper_id": canon["id"],
                "title": canon.get("Title", ""),
                "duplicate_source_files": [records[i]["source_file"] for i in dup_indices],
            }
        )
    return groups


def _write_no_references(path: Path, entries: list[dict]) -> None:
    """Persist papers that returned zero references; remove the file when none exist."""
    if not entries:
        if path.exists():
            path.unlink()
        return
    write_json(path, entries)


def _write_source_duplicates(path: Path, groups: list[dict]) -> None:
    """Persist duplicate groups as JSON, or remove the file when none exist.

    File absence means no duplicates were found, mirroring the failures pattern.
    """
    if not groups:
        if path.exists():
            path.unlink()
        return
    write_json(path, groups)


def _count_source_duplicates(path: Path) -> int:
    """Total number of duplicate markdown files; 0 if the file doesn't exist."""
    if not path.exists():
        return 0
    data = read_json(path)
    return sum(len(g.get("duplicate_source_files", [])) for g in data)


def _count_no_references(path: Path) -> int:
    if not path.exists():
        return 0
    return len(read_json(path))


def _check_conversion_quality(
    paths: list[Path], layout: OutLayout, *, ocr_attempted: bool = False
) -> None:
    """After conversion, flag markdown files that appear to be image-only.

    ``ocr_attempted`` tunes the remediation hint: if OCR was already tried
    (force or auto) and the output is *still* image-only, telling the user
    to "re-run with --ocr" is wrong — surface that the page is genuinely
    unreadable and needs manual review instead.
    """
    reason = (
        "image-only markdown even after OCR (manual review needed)"
        if ocr_attempted
        else "image-only markdown (possibly a scanned PDF)"
    )
    warnings = [
        {"source_file": p.name, "reason": reason}
        for p in paths
        if _is_image_only(p)
    ]
    _write_conversion_warnings(layout.conversion_warnings_json, warnings)
    if warnings:
        hint = (
            "OCR did not help; inspect the source PDFs and consider manual transcription."
            if ocr_attempted
            else "re-run with ocr=True / --ocr (or ocr='auto' / --ocr-auto) for better results."
        )
        logger.warning(
            "%d markdown file(s) appear image-only; %s See %s",
            len(warnings),
            hint,
            layout.conversion_warnings_json,
        )


def _write_conversion_warnings(path: Path, warnings: list[dict]) -> None:
    if not warnings:
        if path.exists():
            path.unlink()
        return
    write_json(path, warnings)


def _count_conversion_warnings(path: Path) -> int:
    if not path.exists():
        return 0
    return len(read_json(path))


def _write_author_review(path: Path, review: list[dict]) -> None:
    """Persist flagged author clusters, or remove the file when there are none."""
    if not review:
        if path.exists():
            path.unlink()
        return
    write_json(path, review)


def _count_author_review(path: Path) -> int:
    if not path.exists():
        return 0
    return len(read_json(path))
