"""Pydantic schemas used both as I/O DTOs and Gemini response schemas.

``Authors`` is a *derived* property (``", ".join(Authors_List)``), not part
of the LLM contract — only ``Authors_List`` is. The LLM used to be asked
for both, which (a) wasted output tokens and (b) created a real failure
mode where the two could disagree. Both schemas override ``model_dump`` to
include the derived ``Authors`` so CSV/cache writers see the column.

Old cache files that contain an ``Authors`` field still load cleanly: the
``extra="ignore"`` config drops the unknown key during validation, and the
property recomputes from ``Authors_List`` on access.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class PaperMetadata(BaseModel):
    """Bibliographic metadata extracted from a single source paper.

    Used both as the output of :mod:`citegraph.extract_metadata` and as the
    ``response_schema`` for Gemini's structured output.
    """

    model_config = ConfigDict(extra="ignore")

    Title: str = Field(description="The full title of the paper.")
    Authors_List: list[str] = Field(description="List of every author name.")
    Journal: str = Field(description="Journal or venue where the paper was published.")
    Year: int = Field(description="Publication year as an integer.")

    @property
    def Authors(self) -> str:
        return ", ".join(self.Authors_List)

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        data = super().model_dump(**kwargs)
        data["Authors"] = self.Authors
        return data


class Reference(BaseModel):
    """A single bibliographic reference extracted from a paper's bibliography."""

    model_config = ConfigDict(extra="ignore")

    Title: str
    Authors_List: list[str]
    Journal: str
    Year: int

    @property
    def Authors(self) -> str:
        return ", ".join(self.Authors_List)

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        data = super().model_dump(**kwargs)
        data["Authors"] = self.Authors
        return data


class CitationLink(BaseModel):
    """An edge from a citing paper to a (deduplicated) cited paper."""

    citing_id: str
    cited_id: str


@dataclass
class PipelineResult:
    """Container returned by :meth:`citegraph.Pipeline.run`.

    Three pandas DataFrames mirror the on-disk CSVs:

    - ``papers``     : one row per source paper, indexed by ``id``.
    - ``references`` : one row per *deduplicated* reference, indexed by ``id``.
    - ``graph``      : edges ``(citing_id, cited_id)``.
    """

    papers: pd.DataFrame
    references: pd.DataFrame
    graph: pd.DataFrame
    authors: pd.DataFrame | None = None
    author_citations: pd.DataFrame | None = None
