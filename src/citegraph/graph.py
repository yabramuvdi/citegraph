"""A queryable view over the pipeline's output CSVs.

The pipeline writes everything you need to disk, but the package is called
``citegraph`` — so a small, polished class lives here to deliver on the
name. Construct from a finished pipeline run::

    from citegraph import CitationGraph
    g = CitationGraph.from_out_dir("./out")
    g.n_core_works, g.n_works, g.n_edges
    g.core                      # your own papers (ring 0)
    g.top_cited(n=10)
    g.top_authors(n=10, ring=0) # most prominent authors of YOUR papers
    g.core_citations()          # who among your papers cites whom
    g.cited_by("w-doe-2020-some-paper")
    g.citers_of("w-smith-1968-tragedy-of-the-commons")

Or directly from a :class:`~citegraph.PipelineResult`::

    result = pipe.run()
    g = CitationGraph.from_pipeline_result(result)

Every bibliographic record is a *work* (``w-`` id), indexed by ``id`` in
``works``. Two stored facts distinguish roles: ``ring`` (0 = seeded from
your PDFs — the *core*; n = first discovered in a ring n-1 bibliography)
and ``source_file`` (non-empty when we processed a PDF for it).
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from citegraph.io import AUTHOR_CITATION_COLUMNS, AUTHOR_COLUMNS, OutLayout
from citegraph.schemas import PipelineResult

if TYPE_CHECKING:  # pragma: no cover
    import networkx as nx


class CitationGraph:
    """A read-only view over canonical works and citation edges."""

    def __init__(
        self,
        works: pd.DataFrame,
        edges: pd.DataFrame,
        authors: pd.DataFrame | None = None,
        author_citations: pd.DataFrame | None = None,
    ) -> None:
        # Accept both index-by-id (canonical) and id-as-column frames.
        if "id" in works.columns:
            works = works.set_index("id")
        self.works = works
        self.edges = edges
        # Author tables are optional — the citegraph authors stage may not
        # have been run yet. Methods that need them check `has_authors`.
        self._author_tables_loaded = authors is not None and author_citations is not None
        self.authors = authors if authors is not None else pd.DataFrame()
        self.author_citations = (
            author_citations if author_citations is not None else pd.DataFrame()
        )

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_out_dir(cls, out_dir: str | Path) -> CitationGraph:
        """Load the output CSVs from a finished pipeline's ``out_dir``.

        ``authors.csv`` and ``author_citations.csv`` are loaded too when
        present (i.e. after the ``citegraph authors`` stage has run).
        Their absence is silent — the rest of the API still works.
        """
        layout = OutLayout(Path(out_dir))
        if not layout.works_csv.exists() and (layout.out_dir / "papers.csv").exists():
            raise FileNotFoundError(
                f"{layout.works_csv} not found, but legacy papers.csv exists. "
                "This out_dir predates the works model. Re-run the cheap "
                "downstream stages (per-paper caches are reused): "
                f"citegraph metadata --out {out_dir} && "
                f"citegraph references --out {out_dir} && "
                f"citegraph dedup --out {out_dir} && "
                f"citegraph authors --out {out_dir}"
            )
        missing = [
            p for p in (layout.works_csv, layout.graph_csv) if not p.exists()
        ]
        if missing:
            raise FileNotFoundError(
                "Missing output(s): "
                + ", ".join(str(p) for p in missing)
                + ". Run the pipeline first (`citegraph run ...`)."
            )
        def read_optional(path: Path, columns: list[str], index_col: str | None = None):
            if not path.exists():
                return None
            try:
                return pd.read_csv(path, index_col=index_col)
            except pd.errors.EmptyDataError:
                empty = pd.DataFrame(columns=columns)
                return empty.set_index(index_col) if index_col else empty

        authors_df = read_optional(layout.authors_csv, AUTHOR_COLUMNS, "id")
        author_citations_df = read_optional(layout.author_citations_csv, AUTHOR_CITATION_COLUMNS)
        return cls(
            works=pd.read_csv(layout.works_csv, index_col="id"),
            edges=pd.read_csv(layout.graph_csv),
            authors=authors_df,
            author_citations=author_citations_df,
        )

    @classmethod
    def from_pipeline_result(cls, result: PipelineResult) -> CitationGraph:
        """Wrap the DataFrames returned by :meth:`Pipeline.run`."""
        return cls(
            works=result.works,
            edges=result.graph,
            authors=result.authors,
            author_citations=result.author_citations,
        )

    # ------------------------------------------------------------------
    # Counts
    # ------------------------------------------------------------------
    @property
    def n_works(self) -> int:
        return len(self.works)

    @property
    def n_core_works(self) -> int:
        return len(self.core)

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    # ------------------------------------------------------------------
    # Ring views
    # ------------------------------------------------------------------
    @property
    def core(self) -> pd.DataFrame:
        """The user's own papers: works at ring 0."""
        if "ring" not in self.works.columns:
            return self.works.iloc[0:0]
        return self.works[self.works["ring"] == 0]

    def ring(self, n: int) -> pd.DataFrame:
        """Works at discovery depth ``n`` (0 = core)."""
        if "ring" not in self.works.columns:
            return self.works.iloc[0:0]
        return self.works[self.works["ring"] == n]

    def core_citations(self) -> pd.DataFrame:
        """Edges where a core work cites another core work."""
        ids = set(self.core.index)
        return self.edges[
            self.edges["citing_id"].isin(ids) & self.edges["cited_id"].isin(ids)
        ].reset_index(drop=True)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def cited_by(self, work_id: str) -> pd.DataFrame:
        """Return the works cited by a given work.

        Unknown ids return an empty DataFrame rather than raising —
        calling code can branch on ``len(...)``.
        """
        cited_ids = self.edges.loc[self.edges["citing_id"] == work_id, "cited_id"]
        return self.works.loc[self.works.index.isin(cited_ids)]

    def citers_of(self, work_id: str) -> pd.DataFrame:
        """Return the works that cite a given work."""
        citing_ids = self.edges.loc[self.edges["cited_id"] == work_id, "citing_id"]
        return self.works.loc[self.works.index.isin(citing_ids)]

    def top_cited(self, n: int = 20) -> pd.DataFrame:
        """Return the ``n`` most-cited works in this corpus.

        The returned DataFrame is the works table filtered to the top
        ``n`` rows and sorted by citation count (descending), with an
        extra ``citation_count`` column. Counts the number of *distinct
        citing works* (the edge list is already deduplicated by the
        pipeline, so each ``(citing_id, cited_id)`` pair is one vote).
        Core works cited within the corpus rank here too.
        """
        if self.edges.empty:
            empty = self.works.iloc[0:0].copy()
            empty["citation_count"] = pd.Series(dtype=int)
            return empty
        counts = self.edges.groupby("cited_id").size().sort_values(ascending=False)
        top_ids = counts.head(n).index
        out = self.works.loc[self.works.index.isin(top_ids)].copy()
        out["citation_count"] = out.index.map(counts).astype(int)
        return out.sort_values("citation_count", ascending=False)

    # ------------------------------------------------------------------
    # Authors
    # ------------------------------------------------------------------
    @property
    def has_authors(self) -> bool:
        """True when the author-normalization stage has been run."""
        return self._author_tables_loaded

    def _require_authors(self) -> None:
        if not self.has_authors:
            raise RuntimeError(
                "Author tables are not loaded. Run `citegraph authors --out <dir>` "
                "to produce authors.csv / author_citations.csv first."
            )

    def top_authors(self, n: int = 20, ring: int | None = None) -> pd.DataFrame:
        """Authors ranked by distinct works authored, optionally one ring only.

        ``ring=0`` answers "who are the most prominent authors of *my*
        papers?". ``ring=None`` (default) spans the whole corpus. The
        result carries an ``n_works_in_selection`` column with the count
        that produced the ranking.
        """
        self._require_authors()
        ac = self.author_citations
        if ring is not None:
            if "ring" not in self.works.columns:
                ring_ids: set[str] = set()
            else:
                ring_ids = set(self.works.index[self.works["ring"] == ring])
            ac = ac[ac["record_id"].isin(ring_ids)]
        if ac.empty:
            out = self.authors.iloc[0:0].copy()
            out["n_works_in_selection"] = pd.Series(dtype=int)
            return out
        counts = ac.groupby("author_id")["record_id"].nunique().sort_values(ascending=False)
        head = counts.head(n)
        out = self.authors.loc[self.authors.index.isin(head.index)].copy()
        out["n_works_in_selection"] = out.index.map(head).astype(int)
        return out.sort_values("n_works_in_selection", ascending=False, kind="mergesort")

    def top_cited_authors(self, n: int = 20) -> pd.DataFrame:
        """Return the ``n`` authors receiving the most citations.

        Ranked by ``n_citations_received`` — the number of citation
        edges into any work the author appears on. This is the metric
        users typically mean by "most cited author", and since core
        works are citable it credits corpus-internal citations too.
        """
        self._require_authors()
        return self.authors.sort_values("n_citations_received", ascending=False).head(n)

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
        return self.authors[mask].sort_values("n_citations_received", ascending=False)

    def citations_of(self, author_id: str) -> pd.DataFrame:
        """Return every cited work in which ``author_id`` appears.

        Joins ``author_citations`` against ``works`` and includes the
        citing-work ids so the caller can trace back to the source.
        """
        context = self.citation_context_for_author(author_id)
        if context.empty:
            out = self.works.iloc[0:0].copy()
            out["citing_paper_ids"] = pd.Series(dtype=object)
            out["n_citing_papers"] = pd.Series(dtype=int)
            return out

        out = self.works.loc[
            self.works.index.isin(context["cited_reference_id"])
        ].copy()
        citing_ids = context.groupby("cited_reference_id")["citing_paper_id"].agg(
            lambda s: sorted(set(s))
        )
        out["citing_paper_ids"] = out.index.map(citing_ids)
        out["n_citing_papers"] = out["citing_paper_ids"].map(len)
        return out

    def papers_citing_author(self, author_id: str) -> pd.DataFrame:
        """Return works that cite at least one work by ``author_id``."""
        context = self.citation_context_for_author(author_id)
        citing_ids = set(context["citing_paper_id"].dropna().unique())
        return self.works.loc[self.works.index.isin(citing_ids)]

    def citation_context_for_author(self, author_id: str) -> pd.DataFrame:
        """Return citing-work context for every citation of works by ``author_id``.

        The result is one row per ``citing work -> cited work`` edge where
        the cited work has ``author_id`` among its canonical authors. This
        is the audit table behind claims like "100 papers cited Juan Camilo
        Cardenas"; citing-side journal fields come from ``works.csv``.
        Cited core works count too — corpus-internal citations of an
        author are part of their context.
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
        author_refs = ac[ac["author_id"] == author_id].copy()
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

        citing_side = self.works.reset_index()
        citing_side = citing_side.rename(
            columns={
                citing_side.columns[0]: "citing_paper_id",
                "Title": "source_paper_title",
                "Journal": "source_paper_journal",
                "Year": "source_paper_year",
            }
        )
        for col in ("source_paper_title", "source_paper_journal", "source_paper_year"):
            if col not in citing_side.columns:
                citing_side[col] = pd.NA

        cited_side = self.works.reset_index()
        cited_side = cited_side.rename(
            columns={
                cited_side.columns[0]: "cited_reference_id",
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
            if col not in cited_side.columns:
                cited_side[col] = pd.NA

        author_display = (
            self.authors.loc[author_id, "display_name"]
            if author_id in self.authors.index and "display_name" in self.authors.columns
            else ""
        )
        context["author_display_name"] = author_display
        context = context.merge(
            citing_side[
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
            cited_side[
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
        """Return distinct works that cite at least one work by ``author_id``.

        One citing work can cite several works by the same author; it
        still appears once here, with ``n_cited_references_by_author`` and the
        cited work ids/titles preserving the evidence.
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
        """Count citing-work journals among works that cite ``author_id``."""
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
    # ------------------------------------------------------------------
    # Author co-citation network
    # ------------------------------------------------------------------
    def author_cocitation_network(
        self,
        min_papers: int = 3,
        citing_ids: Iterable[str] | None = None,
    ) -> nx.Graph:
        """Return the author co-citation network as a weighted ``networkx.Graph``.

        Two authors are joined when the same bibliography cites both of
        them, and the edge ``weight`` counts how many bibliographies do
        so. This is the standard author co-citation projection of
        bibliometrics (White & Griffith 1981; White & McCain 1998).

        It exists because the raw work-level graph is one hop deep: only
        works we processed a PDF for have outgoing edges, so path-based
        centrality over :meth:`to_networkx` would measure the sampling
        design rather than the literature. The co-citation projection is
        a genuine one-mode network, so degree, eigenvector and
        betweenness centrality are all well defined on it.

        A citing work that cites several works by one author still
        contributes a single mention of that author, and never a
        self-loop.

        ``min_papers`` drops authors cited by fewer than that many
        citing works, which prunes the long tail of once-cited names.
        ``citing_ids`` restricts the citing side, which is how you build
        a variant that excludes a given author's own papers.

        Requires ``networkx`` (not a default dependency) and the author
        tables (run ``citegraph authors``). Nodes carry ``display_name``
        and ``n_citing_papers``; nodes are inserted in sorted order so
        that seeded layouts are reproducible across runs.
        """
        self._require_authors()
        try:
            import networkx as nx
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "author_cocitation_network() requires networkx. "
                "Install with: pip install networkx"
            ) from exc

        edges = self.edges[["citing_id", "cited_id"]].drop_duplicates()
        if citing_ids is not None:
            edges = edges[edges["citing_id"].isin(set(citing_ids))]

        work_authors = self.author_citations[["author_id", "record_id"]].drop_duplicates()
        # One mention per (citing work, cited author): citing three works
        # by the same author is one appearance of that author, not three.
        mentions = (
            edges.merge(work_authors, left_on="cited_id", right_on="record_id")[
                ["citing_id", "author_id"]
            ]
            .drop_duplicates()
        )

        n_citing_papers = mentions.groupby("author_id")["citing_id"].nunique()
        keep = n_citing_papers[n_citing_papers >= min_papers]
        mentions = mentions[mentions["author_id"].isin(set(keep.index))]

        names = (
            self.authors["display_name"]
            if "display_name" in self.authors.columns
            else pd.Series(dtype=object)
        )

        g: nx.Graph = nx.Graph()
        for author_id in sorted(keep.index):
            g.add_node(
                author_id,
                display_name=names.get(author_id, author_id),
                n_citing_papers=int(keep[author_id]),
            )
        for _, group in mentions.groupby("citing_id"):
            for a, b in itertools.combinations(sorted(group["author_id"]), 2):
                if g.has_edge(a, b):
                    g[a][b]["weight"] += 1
                else:
                    g.add_edge(a, b, weight=1)
        return g

    def to_networkx(self) -> nx.DiGraph:
        """Return a ``networkx.DiGraph``: nodes are works, edges go citing → cited.

        Requires ``networkx`` (not a default dependency). Each node
        carries the work's row metadata as attributes — including
        ``ring`` and ``source_file`` where present.
        """
        try:
            import networkx as nx
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "to_networkx() requires networkx. Install with: pip install networkx"
            ) from exc

        g: nx.DiGraph = nx.DiGraph()
        for work_id, row in self.works.iterrows():
            attrs: dict[str, Any] = dict(row.items())
            g.add_node(work_id, **attrs)
        for _, edge in self.edges.iterrows():
            g.add_edge(edge["citing_id"], edge["cited_id"])
        return g

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        author_suffix = f", {len(self.authors)} authors" if self.has_authors else ""
        return (
            f"<CitationGraph: {self.n_core_works} core works, "
            f"{self.n_works} works, {self.n_edges} edges{author_suffix}>"
        )
