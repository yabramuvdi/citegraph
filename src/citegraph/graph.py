"""A queryable view over the pipeline's three output CSVs.

The pipeline writes everything you need to disk, but the package is called
``citegraph`` — so a small, polished class lives here to deliver on the
name. Construct from a finished pipeline run::

    from citegraph import CitationGraph
    g = CitationGraph.from_out_dir("./out")
    g.n_papers, g.n_references, g.n_edges
    g.top_cited(n=10)
    g.cited_by("p-doe-2020-some-paper")
    g.citers_of("r-smith-1968-tragedy-of-the-commons")

Or directly from a :class:`~citegraph.PipelineResult`::

    result = pipe.run()
    g = CitationGraph.from_pipeline_result(result)

The DataFrame asymmetry from the pipeline is preserved: ``papers`` keeps
``id`` as a column (matching ``papers.csv``), ``references`` is indexed by
``id`` (matching ``references.csv``). Methods do the right thing on each
side; you should rarely need to think about it.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from citegraph.io import OutLayout
from citegraph.schemas import PipelineResult

if TYPE_CHECKING:  # pragma: no cover
    import networkx as nx


class CitationGraph:
    """A read-only view over papers, references, and citation edges."""

    def __init__(
        self,
        papers: pd.DataFrame,
        references: pd.DataFrame,
        edges: pd.DataFrame,
    ) -> None:
        self.papers = papers
        self.references = references
        self.edges = edges

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_out_dir(cls, out_dir: str | Path) -> CitationGraph:
        """Load the three CSVs from a finished pipeline's ``out_dir``."""
        layout = OutLayout(Path(out_dir))
        missing = [
            p
            for p in (layout.papers_csv, layout.references_csv, layout.graph_csv)
            if not p.exists()
        ]
        if missing:
            raise FileNotFoundError(
                "Missing output(s): "
                + ", ".join(str(p) for p in missing)
                + ". Run the pipeline first (`citegraph run ...`)."
            )
        return cls(
            papers=pd.read_csv(layout.papers_csv),
            references=pd.read_csv(layout.references_csv, index_col="id"),
            edges=pd.read_csv(layout.graph_csv),
        )

    @classmethod
    def from_pipeline_result(cls, result: PipelineResult) -> CitationGraph:
        """Wrap the DataFrames returned by :meth:`Pipeline.run`."""
        return cls(papers=result.papers, references=result.references, edges=result.graph)

    # ------------------------------------------------------------------
    # Counts
    # ------------------------------------------------------------------
    @property
    def n_papers(self) -> int:
        return len(self.papers)

    @property
    def n_references(self) -> int:
        return len(self.references)

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def cited_by(self, paper_id: str) -> pd.DataFrame:
        """Return the references cited by a given source paper.

        ``paper_id`` is a ``p-…`` id from ``papers.csv``. Unknown ids return
        an empty DataFrame rather than raising — calling code can branch on
        ``len(...)``.
        """
        cited_ids = self.edges.loc[self.edges["citing_id"] == paper_id, "cited_id"]
        return self.references.loc[self.references.index.isin(cited_ids)]

    def citers_of(self, reference_id: str) -> pd.DataFrame:
        """Return the source papers that cite a given reference."""
        citing_ids = self.edges.loc[self.edges["cited_id"] == reference_id, "citing_id"]
        return self.papers[self.papers["id"].isin(citing_ids)]

    def top_cited(self, n: int = 20) -> pd.DataFrame:
        """Return the ``n`` most-cited references in this corpus.

        The returned DataFrame is the references table filtered to the top
        ``n`` rows and sorted by citation count (descending), with an extra
        ``citation_count`` column. Counts the number of *distinct papers*
        that cite each reference (the edge list is already deduplicated by
        the pipeline, so each ``(citing_id, cited_id)`` pair is one vote).
        """
        if self.edges.empty:
            empty = self.references.iloc[0:0].copy()
            empty["citation_count"] = pd.Series(dtype=int)
            return empty
        counts = self.edges.groupby("cited_id").size().sort_values(ascending=False)
        top_ids = counts.head(n).index
        out = self.references.loc[self.references.index.isin(top_ids)].copy()
        out["citation_count"] = out.index.map(counts).astype(int)
        return out.sort_values("citation_count", ascending=False)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def to_networkx(self) -> nx.DiGraph:
        """Return a ``networkx.DiGraph``: nodes are papers + references, edges go citing → cited.

        Requires ``networkx`` (not a default dependency). Each node gets a
        ``kind`` attribute (``"paper"`` or ``"reference"``) plus the row's
        metadata as additional attributes.
        """
        try:
            import networkx as nx
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "to_networkx() requires networkx. Install with: pip install networkx"
            ) from exc

        g: nx.DiGraph = nx.DiGraph()
        for _, row in self.papers.iterrows():
            attrs: dict[str, Any] = {k: v for k, v in row.items() if k != "id"}
            attrs["kind"] = "paper"
            g.add_node(row["id"], **attrs)
        for ref_id, row in self.references.iterrows():
            attrs = dict(row.items())
            attrs["kind"] = "reference"
            g.add_node(ref_id, **attrs)
        for _, edge in self.edges.iterrows():
            g.add_edge(edge["citing_id"], edge["cited_id"])
        return g

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"<CitationGraph: {self.n_papers} papers, "
            f"{self.n_references} references, {self.n_edges} edges>"
        )
