# Evaluation harness

A small offline evaluation that lets you verify how well the pipeline
performs on your corpus before publishing it to other researchers.

## Layout

```
evaluation/
  README.md
  evaluate.py
  RESULTS.md          # written here after a run
  gold/
    paper_id.json     # one file per gold-labelled paper:
                      #   {
                      #     "source_file": "filename.md",
                      #     "metadata": { ...PaperMetadata... },
                      #     "references": [ ...list[Reference]... ]
                      #   }
```

## Building gold labels

Run the pipeline once on your full corpus, then for ~5 papers you trust,
copy the auto-extracted JSON into `evaluation/gold/<id>.json` and edit it
by hand to fix any mistakes. That's your gold standard.

A starter file is included at [`gold/sample_paper.json`](gold/sample_paper.json)
matching the test fixture.

## Running

```bash
python evaluation/evaluate.py \
    --gold evaluation/gold \
    --out ./out
```

Reports go to stdout and to [`RESULTS.md`](RESULTS.md):

- **Metadata accuracy** &mdash; per-field exact / fuzzy match against gold.
- **Reference recall / precision** &mdash; matched against gold by fuzzy
  title (≥ 90).
- **Dedup F1** &mdash; only computed if `gold/dedup_pairs.json` exists with
  pairs of references that should cluster.

## Caveats

- The gold set should ideally cover at least 5 papers and ~150 references
  for stable numbers.
- Reference matching is fuzzy (rapidfuzz, threshold 90 by default), which
  is forgiving of typos but can miss heavy formatting differences.
