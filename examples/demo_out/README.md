# Demo output directory

A finished `citegraph` run you can explore without an API key, docling, or
any network access:

```python
from citegraph import CitationGraph

g = CitationGraph.from_out_dir("examples/demo_out")
```

or `citegraph report --out examples/demo_out --open`.

The corpus is built around one experimental economist's 93 papers, chapters
and working papers on collective action and common-pool resources, and
everything those papers cite: 3,277 canonical works, 4,532 citation edges,
4,738 canonical authors.

Only the generated tables are here: `sources.csv`, `works.csv`,
`citation_graph.csv`, `authors.csv` and `author_citations.csv`. The PDFs, the
converted markdown and the per-paper extraction caches are the papers' own
text and are not distributed. Enrichment output, curated correction files and
reviewer annotations are also left out, so `enriched_works.csv` is reported
missing and `g.has_annotations` is `False`. Bibliographic facts about
published works are not copyrightable; the tables are shared under the
repository's MIT license.

The tables were produced by the pipeline at the version recorded in the git
history of this directory, with hand-curated `author_aliases.csv`,
`author_external_ids.csv` and `journal_aliases.csv` applied. Rerunning the
pipeline on the same PDFs would reproduce them only with those files present.
