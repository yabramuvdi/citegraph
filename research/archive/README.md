# Archive

These are the original, pre-package scripts. Their functionality has been
migrated into the `citegraph` package; they live here only as historical
reference and to make diffs easy.

| File | Replaced by |
| ---- | ----------- |
| `pdf_extraction.py`     | [`citegraph/pdf_to_markdown.py`](../../src/citegraph/pdf_to_markdown.py) |
| `label_papers.py`       | [`citegraph/extract_metadata.py`](../../src/citegraph/extract_metadata.py) and [`citegraph/extract_references.py`](../../src/citegraph/extract_references.py) |
| `label_papers_old.py`   | The dedup logic moved to [`citegraph/dedup.py`](../../src/citegraph/dedup.py); the legacy plotting was dropped. |
| `helper_functions.py`   | `fix_incomplete_json_string` is now in [`citegraph/llm.py`](../../src/citegraph/llm.py); `transform_date` was unrelated and has been removed. |

These scripts are **not maintained** &mdash; if you change something in
the package, do not also update them.
