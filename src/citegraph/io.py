"""Filesystem helpers for the on-disk checkpoints.

The pipeline persists every expensive intermediate step under ``out_dir``
so that re-runs can resume:

::

    out_dir/
      markdown/                 # docling output, one .md per PDF
      metadata/<paper-id>.json  # per-paper Gemini metadata response
      references/<paper-id>.json  # per-paper Gemini references response
      papers.csv                # combined paper metadata
      references_raw.csv        # all extracted references, before dedup
      references.csv            # deduplicated references
      citation_graph.csv        # (citing_id, cited_id) edges
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel


@dataclass(frozen=True)
class OutLayout:
    """Standard subdirectory / filename layout under ``out_dir``."""

    out_dir: Path

    @property
    def markdown_dir(self) -> Path:
        return self.out_dir / "markdown"

    @property
    def metadata_dir(self) -> Path:
        return self.out_dir / "metadata"

    @property
    def references_dir(self) -> Path:
        return self.out_dir / "references"

    @property
    def enrichment_dir(self) -> Path:
        return self.out_dir / "enrichment"

    @property
    def papers_csv(self) -> Path:
        return self.out_dir / "papers.csv"

    @property
    def references_raw_csv(self) -> Path:
        return self.out_dir / "references_raw.csv"

    @property
    def references_csv(self) -> Path:
        return self.out_dir / "references.csv"

    @property
    def enriched_references_csv(self) -> Path:
        return self.out_dir / "enriched_references.csv"

    @property
    def graph_csv(self) -> Path:
        return self.out_dir / "citation_graph.csv"

    @property
    def metadata_failures_jsonl(self) -> Path:
        return self.out_dir / "metadata_failures.jsonl"

    @property
    def references_failures_jsonl(self) -> Path:
        return self.out_dir / "references_failures.jsonl"

    def ensure(self) -> None:
        for d in (
            self.out_dir,
            self.markdown_dir,
            self.metadata_dir,
            self.references_dir,
            self.enrichment_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def read_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def write_pydantic(path: Path, obj: BaseModel) -> None:
    write_json(path, obj.model_dump())


def write_pydantic_list(path: Path, objs: Iterable[BaseModel]) -> None:
    write_json(path, [o.model_dump() for o in objs])
