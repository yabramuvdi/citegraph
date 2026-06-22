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
        authors: pd.DataFrame | None = None,
        author_citations: pd.DataFrame | None = None,
    ) -> None:
        self.papers = papers
        self.references = references
        self.edges = edges
        # Author tables are optional — the citegraph authors stage may not
        # have been run yet. Methods that need them check `has_authors`.
        self.authors = authors if authors is not None else pd.DataFrame()
        self.author_citations = (
            author_citations if author_citations is not None else pd.DataFrame()
        )

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_out_dir(cls, out_dir: str | Path) -> CitationGraph:
        """Load the three CSVs from a finished pipeline's ``out_dir``.

        ``authors.csv`` and ``author_citations.csv`` are loaded too when
        present (i.e. after the ``citegraph authors`` stage has run).
        Their absence is silent — the rest of the API still works.
        """
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
        authors_df = (
            pd.read_csv(layout.authors_csv, index_col="id")
            if layout.authors_csv.exists() else None
        )
        author_citations_df = (
            pd.read_csv(layout.author_citations_csv)
            if layout.author_citations_csv.exists() else None
        )
        return cls(
            papers=pd.read_csv(layout.papers_csv),
            references=pd.read_csv(layout.references_csv, index_col="id"),
            edges=pd.read_csv(layout.graph_csv),
            authors=authors_df,
            author_citations=author_citations_df,
        )

    @classmethod
    def from_pipeline_result(cls, result: PipelineResult) -> CitationGraph:
        """Wrap the DataFrames returned by :meth:`Pipeline.run`."""
        return cls(
            papers=result.papers,
            references=result.references,
            edges=result.graph,
            authors=result.authors,
            author_citations=result.author_citations,
        )

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
    # Authors
    # ------------------------------------------------------------------
    @property
    def has_authors(self) -> bool:
        """True when the author-normalization stage has been run."""
        return not self.authors.empty

    def _require_authors(self) -> None:
        if not self.has_authors:
            raise RuntimeError(
                "Author tables are not loaded. Run `citegraph authors --out <dir>` "
                "to produce authors.csv / author_citations.csv first."
            )

    def top_cited_authors(self, n: int = 20) -> pd.DataFrame:
        """Return the ``n`` authors with the most reference citations.

        Counts each canonical author once per *reference appearance*
        across the corpus — i.e. the number of cited works in which the
        author's name appears. This is the metric users typically mean by
        "most cited author".
        """
        self._require_authors()
        return self.authors.sort_values("n_reference_citations", ascending=False).head(n)

    def find_author(self, query: str) -> pd.DataFrame:
        """Return canonical authors whose surname or display name matches ``query``.

        Matching is diacritic-insensitive and case-insensitive. Substring
        match, anchored to nothing — "card" matches both "Cárdenas" and
        "Cardinale". Useful for the user-facing "is this person in my
        corpus?" lookup.
        """
        self._require_authors()
        if not query:
            return self.authors.iloc[0:0]
        import unicodedata
        def _fold(s: object) -> str:
            if not isinstance(s, str):
                return ""
            return "".join(
                c for c in unicodedata.normalize("NFD", s)
                if unicodedata.category(c) != "Mn"
            ).lower()
        needle = _fold(query)
        surnames = self.authors["surname_norm"].fillna("").map(_fold)
        names = self.authors["display_name"].fillna("").map(_fold)
        mask = surnames.str.contains(needle, regex=False) | names.str.contains(needle, regex=False)
        return self.authors[mask].sort_values("n_reference_citations", ascending=False)

    def citations_of(self, author_id: str) -> pd.DataFrame:
        """Return every reference in which ``author_id`` was cited.

        Joins ``author_citations`` against ``references`` and includes
        the citing-paper id so the caller can trace back to the source.
        """
        context = self.citation_context_for_author(author_id)
        if context.empty:
            out = self.references.iloc[0:0].copy()
            out["citing_paper_ids"] = pd.Series(dtype=object)
            out["n_citing_papers"] = pd.Series(dtype=int)
            return out

        out = self.references.loc[
            self.references.index.isin(context["cited_reference_id"])
        ].copy()
        citing_ids = context.groupby("cited_reference_id")["citing_paper_id"].agg(
            lambda s: sorted(set(s))
        )
        out["citing_paper_ids"] = out.index.map(citing_ids)
        out["n_citing_papers"] = out["citing_paper_ids"].map(len)
        return out

    def papers_citing_author(self, author_id: str) -> pd.DataFrame:
        """Return source papers that cite at least one reference by ``author_id``."""
        context = self.citation_context_for_author(author_id)
        citing_ids = set(context["citing_paper_id"].dropna().unique())
        return self.papers[self.papers["id"].isin(citing_ids)]

    def citation_context_for_author(self, author_id: str) -> pd.DataFrame:
        """Return source-paper context for every citation of works by ``author_id``.

        The result is one row per ``source paper -> cited reference`` edge where
        the cited reference has ``author_id`` among its canonical authors. This
        is the audit table behind claims like "100 papers cited Juan Camilo
        Cardenas"; source-paper journal fields come from ``papers.csv``.
        """
        self._require_authors()
        columns = [
            "author_id",
            "author_display_name",
            "raw_author",
            "position",
            "citing_paper_id",
            "source_paper_title",
            "source_paper_journal",
            "source_paper_year",
            "cited_reference_id",
            "cited_reference_title",
            "cited_reference_journal",
            "cited_reference_year",
        ]
        ac = self.author_citations
        author_refs = ac[
            (ac["author_id"] == author_id)
            & (ac["record_kind"] == "reference")
        ].copy()
        if author_refs.empty or self.edges.empty:
            return pd.DataFrame(columns=columns)

        author_refs = author_refs.rename(columns={"record_id": "cited_reference_id"})
        edges = self.edges.rename(
            columns={"citing_id": "citing_paper_id", "cited_id": "cited_reference_id"}
        )
        context = edges.merge(
            author_refs[["author_id", "cited_reference_id", "position", "raw_author"]],
            on="cited_reference_id",
            how="inner",
        )
        if context.empty:
            return pd.DataFrame(columns=columns)

        papers = self.papers.rename(
            columns={
                "id": "citing_paper_id",
                "Title": "source_paper_title",
                "Journal": "source_paper_journal",
                "Year": "source_paper_year",
            }
        )
        for col in ("source_paper_title", "source_paper_journal", "source_paper_year"):
            if col not in papers.columns:
                papers[col] = pd.NA

        references = self.references.reset_index()
        references = references.rename(
            columns={
                references.columns[0]: "cited_reference_id",
                "Title": "cited_reference_title",
                "Journal": "cited_reference_journal",
                "Year": "cited_reference_year",
            }
        )
        for col in (
            "cited_reference_title",
            "cited_reference_journal",
            "cited_reference_year",
        ):
            if col not in references.columns:
                references[col] = pd.NA

        author_display = (
            self.authors.loc[author_id, "display_name"]
            if author_id in self.authors.index and "display_name" in self.authors.columns
            else ""
        )
        context["author_display_name"] = author_display
        context = context.merge(
            papers[
                [
                    "citing_paper_id",
                    "source_paper_title",
                    "source_paper_journal",
                    "source_paper_year",
                ]
            ],
            on="citing_paper_id",
            how="left",
        ).merge(
            references[
                [
                    "cited_reference_id",
                    "cited_reference_title",
                    "cited_reference_journal",
                    "cited_reference_year",
                ]
            ],
            on="cited_reference_id",
            how="left",
        )
        return context[columns].sort_values(
            ["citing_paper_id", "cited_reference_id", "position"]
        ).reset_index(drop=True)

    def citing_papers_by_author(self, author_id: str) -> pd.DataFrame:
        """Return distinct source papers that cite at least one work by ``author_id``.

        One source paper can cite several references by the same author; it
        still appears once here, with ``n_cited_references_by_author`` and the
        cited reference ids/titles preserving the evidence.
        """
        context = self.citation_context_for_author(author_id)
        columns = [
            "paper_id",
            "source_paper_title",
            "source_paper_journal",
            "source_paper_year",
            "author_id",
            "author_display_name",
            "n_cited_references_by_author",
            "cited_reference_ids",
            "cited_reference_titles",
        ]
        if context.empty:
            return pd.DataFrame(columns=columns)

        grouped = (
            context.groupby(
                [
                    "citing_paper_id",
                    "source_paper_title",
                    "source_paper_journal",
                    "source_paper_year",
                    "author_id",
                    "author_display_name",
                ],
                dropna=False,
            )
            .agg(
                n_cited_references_by_author=("cited_reference_id", "nunique"),
                cited_reference_ids=(
                    "cited_reference_id",
                    lambda s: sorted(set(s)),
                ),
                cited_reference_titles=(
                    "cited_reference_title",
                    lambda s: list(dict.fromkeys(s)),
                ),
            )
            .reset_index()
            .rename(columns={"citing_paper_id": "paper_id"})
        )
        return grouped[columns].sort_values(
            ["n_cited_references_by_author", "paper_id"],
            ascending=[False, True],
        ).reset_index(drop=True)

    def source_journals_citing_author(self, author_id: str) -> pd.DataFrame:
        """Count source-paper journals among papers that cite ``author_id``."""
        papers = self.citing_papers_by_author(author_id)
        columns = ["source_paper_journal", "n_papers", "share_of_papers"]
        if papers.empty:
            return pd.DataFrame(columns=columns)

        journal_series = (
            papers["source_paper_journal"]
            .replace("", pd.NA)
            .fillna("(unknown)")
        )
        counts = (
            journal_series.value_counts()
            .rename_axis("source_paper_journal")
            .reset_index(name="n_papers")
        )
        counts["share_of_papers"] = counts["n_papers"] / len(papers)
        return counts[columns].sort_values(
            ["n_papers", "source_paper_journal"],
            ascending=[False, True],
        ).reset_index(drop=True)

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
        author_suffix = f", {len(self.authors)} authors" if self.has_authors else ""
        return (
            f"<CitationGraph: {self.n_papers} papers, "
            f"{self.n_references} references, {self.n_edges} edges{author_suffix}>"
        )
