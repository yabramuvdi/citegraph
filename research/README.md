# Research scripts (not part of the published package)

This folder contains the original ad-hoc analysis scripts that motivated
`citegraph` but are not part of its public API:

- `find_authors.py` &mdash; search the citations CSV for specific authors
  (e.g. Ostrom).
- `word_counts.py` &mdash; count occurrences of dictionary terms across the
  paper corpus, in English or Spanish.
- `preprocess.py` &mdash; older `PyMuPDF`-based two-column PDF text
  extraction (now superseded by `docling` in the main pipeline).
- `dicts.yaml` &mdash; the keyword dictionaries used by `word_counts.py`.

These scripts are kept in the repository for the original project but are
**not installed** by `pip install citegraph` and are excluded from CI.
They expect to be run with the working directory set to `research/`, e.g.

```bash
cd research && python word_counts.py
```

They also depend on `../../data/...` paths from the original project tree.
