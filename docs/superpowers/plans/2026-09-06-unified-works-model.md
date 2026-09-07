# Unified Works Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the papers/references split with a single canonical **work** entity carrying `ring` (0 = the user's core PDFs) and `source_file`, so core papers get the same dedup/enrichment/author machinery as everything else and core-cites-core edges become first-class.

**Architecture:** One clustering pass canonicalizes source papers *and* extracted citations together (sources seeded first, so citations of core papers resolve into them). Downstream stages (enrich, authors, graph API, CLI, report, webui) consume a single `works.csv` + `citation_graph.csv`. The LLM layer and per-stem caches are frozen; ids move to a single role-free `w-` prefix.

**Tech Stack:** Python 3.11+, pandas, rapidfuzz, pydantic, typer, pytest. No network in tests (`_FakeClient` pattern in `tests/test_pipeline.py`).

**Spec:** `docs/superpowers/specs/2026-09-06-unified-works-model-design.md` — read it first; every task below implements a section of it.

## Global Constraints

- **LLM layer frozen:** `PaperMetadata`/`Reference` in `schemas.py`, prompts, `llm.py`, and the per-stem `metadata/<stem>.json` / `references/<stem>.json` caches must not change shape.
- **Deterministic ids:** same content ⇒ same id across runs; first row of a cluster names the cluster. Never hash- or index-based.
- **No legacy aliases in the final state:** old artifact properties, `make_paper_id`/`make_reference_id`, `dedup_references`, `g.papers`/`g.references` are all *removed* in Task 13 — but each intermediate task must leave the full test suite green, so old code paths survive until their consumers have migrated.
- **Tests never hit the network.** Enrichment tests use the existing fake-transport pattern in `tests/test_enrich.py`.
- **Every task:** run `pytest` (full suite) and `ruff check .` before its commit.
- **Work on a feature branch** (`git checkout -b works-model` from main, or a worktree via superpowers:using-git-worktrees). First commit on the branch should add the spec + this plan (`git add docs/superpowers && git commit -m "docs: works-model spec and plan"`).
- New CSV columns orders matter where specified (`works.csv`: `ring, source_file, Title, Authors, Authors_List, Journal, Year` after the `id` index).
- `sources.csv` keeps `id` as a *column* (stage-2 checkpoint convention); `works.csv` is *indexed* by `id` (canonical convention).

---

### Task 1: `make_work_id` in `ids.py`

**Files:**
- Modify: `src/citegraph/ids.py`
- Test: `tests/test_ids.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `make_work_id(authors: str | list[str], year: int | str | None, title: str) -> str` returning `w-<surname>-<year>-<slug>`. `make_paper_id`/`make_reference_id` stay (deleted in Task 13).

- [ ] **Step 1: Write the failing tests** (append to `tests/test_ids.py`)

```python
from citegraph.ids import make_paper_id, make_work_id


def test_make_work_id_prefix_and_determinism():
    a = make_work_id("Ostrom, E.", 1990, "Governing the Commons")
    b = make_work_id("Ostrom, E.", 1990, "Governing the Commons")
    assert a == "w-ostrom-1990-governing-the-commons"
    assert a == b


def test_make_work_id_same_slug_body_as_legacy_ids():
    # The enrichment-cache fallback (Task 7) renames r-<slug> -> w-<slug>,
    # which only works if the slug body is identical across prefixes.
    legacy = make_paper_id("Cárdenas, J.C.", 2000, "Real wealth and experimental cooperation")
    work = make_work_id("Cárdenas, J.C.", 2000, "Real wealth and experimental cooperation")
    assert work == "w-" + legacy.removeprefix("p-")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_ids.py -v`
Expected: FAIL with `ImportError: cannot import name 'make_work_id'`

- [ ] **Step 3: Implement** (append to `src/citegraph/ids.py`)

```python
def make_work_id(
    authors: str | list[str],
    year: int | str | None,
    title: str,
) -> str:
    """Role-free work id (``w-`` prefix).

    A work's role (core source vs cited stub) changes over its life;
    its identity must not, so the prefix carries no role information.
    """
    return make_paper_id(authors, year, title, prefix="w")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_ids.py -v` — Expected: PASS. Then `pytest && ruff check .`

- [ ] **Step 5: Commit**

```bash
git add src/citegraph/ids.py tests/test_ids.py
git commit -m "feat: add role-free make_work_id (w- prefix)"
```

---

### Task 2: New artifact paths on `OutLayout`

**Files:**
- Modify: `src/citegraph/io.py` (add properties after the existing CSV properties, around `io.py:98`; also extend the module docstring layout diagram)
- Test: `tests/test_io.py`

**Interfaces:**
- Produces: `OutLayout.sources_csv` → `out_dir/sources.csv`; `OutLayout.citations_raw_csv` → `out_dir/citations_raw.csv`; `OutLayout.works_csv` → `out_dir/works.csv`; `OutLayout.enriched_works_csv` → `out_dir/enriched_works.csv`. Old properties (`papers_csv`, `references_raw_csv`, `references_csv`, `enriched_references_csv`) remain until Task 13.

- [ ] **Step 1: Write the failing test** (append to `tests/test_io.py`)

```python
def test_outlayout_works_model_paths(tmp_path):
    from citegraph.io import OutLayout

    layout = OutLayout(tmp_path)
    assert layout.sources_csv == tmp_path / "sources.csv"
    assert layout.citations_raw_csv == tmp_path / "citations_raw.csv"
    assert layout.works_csv == tmp_path / "works.csv"
    assert layout.enriched_works_csv == tmp_path / "enriched_works.csv"
```

- [ ] **Step 2: Run** `pytest tests/test_io.py -v` — Expected: FAIL (`AttributeError: sources_csv`)

- [ ] **Step 3: Implement** — add to `OutLayout` (same `@property` style as `papers_csv` at `io.py:64-66`):

```python
    @property
    def sources_csv(self) -> Path:
        return self.out_dir / "sources.csv"

    @property
    def citations_raw_csv(self) -> Path:
        return self.out_dir / "citations_raw.csv"

    @property
    def works_csv(self) -> Path:
        return self.out_dir / "works.csv"

    @property
    def enriched_works_csv(self) -> Path:
        return self.out_dir / "enriched_works.csv"
```

Update the module docstring diagram (`io.py:6-16`) to list `sources.csv`, `citations_raw.csv`, `works.csv` instead of `papers.csv` / `references_raw.csv` / `references.csv`.

- [ ] **Step 4: Run** `pytest tests/test_io.py -v && pytest && ruff check .` — Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/citegraph/io.py tests/test_io.py
git commit -m "feat: OutLayout paths for works-model artifacts"
```

---

### Task 3: `canonicalize_works` in `dedup.py`

The heart of the redesign. Refactor the single-pass clustering loop out of `dedup_references` into a shared helper, then build `canonicalize_works` on it.

**Files:**
- Modify: `src/citegraph/dedup.py`
- Test: `tests/test_dedup.py`

**Interfaces:**
- Consumes: `make_work_id` (Task 1); existing `compare_papers`, `_candidate_index_lookup`, `_candidate_indices`, `_years_can_match`, `_row_to_dict`, `DedupConfig`.
- Produces:
  - `canonicalize_works(sources: pd.DataFrame, citations_raw: pd.DataFrame, cfg: DedupConfig | None = None, *, show_progress: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, dict]`
    - `sources` columns: `id` (`w-…`), `source_file`, `Title`, `Authors`, `Authors_List`, `Journal`, `Year`.
    - `citations_raw` columns: `Title`, `Authors`, `Authors_List`, `Journal`, `Year`, `citing_id`.
    - Returns `(works_df, edges_df, stats)`: `works_df` indexed by `id` with columns `ring, source_file, Title, Authors, Authors_List, Journal, Year`; `edges_df` columns `citing_id, cited_id` (deduplicated, no self-loops); `stats = {"n_self_loops_dropped": int, "n_source_duplicates_merged": int}`.
  - `_compute_rings(ring0_ids: set[str], edges: pd.DataFrame) -> dict[str, int]` (module-private, unit-tested).
  - `dedup_references` keeps working unchanged (delegates to the shared loop) until Task 13.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_dedup.py`)

```python
import pandas as pd

from citegraph.dedup import DedupConfig, _compute_rings, canonicalize_works


def _sources_df():
    return pd.DataFrame(
        [
            {
                "id": "w-cardenas-2000-real-wealth",
                "source_file": "real wealth.pdf",
                "Title": "Real wealth and experimental cooperation",
                "Authors": "Cardenas, Juan Camilo",
                "Authors_List": ["Cardenas, Juan Camilo"],
                "Journal": "Journal of Development Economics",
                "Year": 2000,
            },
            {
                "id": "w-ostrom-1990-governing-the-commons",
                "source_file": "governing.pdf",
                "Title": "Governing the Commons",
                "Authors": "Ostrom, Elinor",
                "Authors_List": ["Ostrom, Elinor"],
                "Journal": "CUP",
                "Year": 1990,
            },
        ]
    )


def test_citation_of_a_source_resolves_to_the_source_work():
    citations = pd.DataFrame(
        [
            {
                "Title": "Real wealth and experimental cooperation",
                "Authors": "Cardenas, J.C.",
                "Authors_List": ["Cardenas, J.C."],
                "Journal": "J Dev Econ",
                "Year": 2000,
                "citing_id": "w-ostrom-1990-governing-the-commons",
            }
        ]
    )
    works, edges, stats = canonicalize_works(
        _sources_df(), citations, DedupConfig(), show_progress=False
    )
    # No new work was minted for the citation — it merged into the source.
    assert len(works) == 2
    assert list(edges.itertuples(index=False)) == [
        ("w-ostrom-1990-governing-the-commons", "w-cardenas-2000-real-wealth")
    ]
    # Cluster metadata kept the full-text side, not the citation string.
    assert works.loc["w-cardenas-2000-real-wealth", "Title"] == (
        "Real wealth and experimental cooperation"
    )
    assert works.loc["w-cardenas-2000-real-wealth", "Authors"] == "Cardenas, Juan Camilo"
    assert works.loc["w-cardenas-2000-real-wealth", "ring"] == 0
    assert stats["n_self_loops_dropped"] == 0


def test_unmatched_citation_becomes_ring1_stub():
    citations = pd.DataFrame(
        [
            {
                "Title": "A completely different treatise on fisheries",
                "Authors": "Schlager, E.",
                "Authors_List": ["Schlager, E."],
                "Journal": "Land Economics",
                "Year": 1994,
                "citing_id": "w-cardenas-2000-real-wealth",
            }
        ]
    )
    works, edges, _stats = canonicalize_works(
        _sources_df(), citations, DedupConfig(), show_progress=False
    )
    stub = works[works["ring"] == 1]
    assert len(stub) == 1
    assert stub.index[0].startswith("w-schlager-1994-")
    assert stub.iloc[0]["source_file"] == ""
    assert len(edges) == 1


def test_duplicate_source_pdfs_merge_and_edges_remap():
    sources = _sources_df()
    dup = sources.iloc[[0]].copy()
    dup["id"] = "w-cardenas-2000-real-wealth-and-exp"  # different slug, same paper
    dup["source_file"] = "real wealth (copy).pdf"
    dup["Title"] = "Real wealth and experimental cooperation "
    sources = pd.concat([sources, dup], ignore_index=True)
    citations = pd.DataFrame(
        [
            {
                "Title": "Governing the Commons",
                "Authors": "Ostrom, E.",
                "Authors_List": ["Ostrom, E."],
                "Journal": "CUP",
                "Year": 1990,
                "citing_id": "w-cardenas-2000-real-wealth-and-exp",  # cites via the dup id
            }
        ]
    )
    works, edges, stats = canonicalize_works(
        sources, citations, DedupConfig(), show_progress=False
    )
    assert stats["n_source_duplicates_merged"] == 1
    assert "w-cardenas-2000-real-wealth-and-exp" not in works.index
    # The edge's citing side remapped onto the canonical source id.
    assert list(edges.itertuples(index=False)) == [
        ("w-cardenas-2000-real-wealth", "w-ostrom-1990-governing-the-commons")
    ]
    # First member keeps the cluster's source_file.
    assert works.loc["w-cardenas-2000-real-wealth", "source_file"] == "real wealth.pdf"


def test_self_citation_is_dropped_and_counted():
    citations = pd.DataFrame(
        [
            {
                "Title": "Real wealth and experimental cooperation",
                "Authors": "Cardenas, Juan Camilo",
                "Authors_List": ["Cardenas, Juan Camilo"],
                "Journal": "working paper",  # preprint variant of itself
                "Year": 2000,
                "citing_id": "w-cardenas-2000-real-wealth",
            }
        ]
    )
    _works, edges, stats = canonicalize_works(
        _sources_df(), citations, DedupConfig(), show_progress=False
    )
    assert edges.empty
    assert stats["n_self_loops_dropped"] == 1


def test_compute_rings_general_bfs():
    edges = pd.DataFrame(
        [
            {"citing_id": "w-a", "cited_id": "w-b"},
            {"citing_id": "w-b", "cited_id": "w-c"},   # ring-1 work citing (snowball case)
            {"citing_id": "w-a", "cited_id": "w-d"},
        ]
    )
    rings = _compute_rings({"w-a"}, edges)
    assert rings == {"w-a": 0, "w-b": 1, "w-d": 1, "w-c": 2}
```

- [ ] **Step 2: Run** `pytest tests/test_dedup.py -v` — Expected: FAIL (`ImportError: canonicalize_works`)

- [ ] **Step 3: Refactor the clustering loop into a shared helper**

In `src/citegraph/dedup.py`, extract the body of `dedup_references`'s loop (`dedup.py:236-272`) into:

```python
from collections.abc import Callable


def _cluster_rows(
    df: pd.DataFrame,
    cfg: DedupConfig,
    make_cluster_id: Callable[[int, pd.Series], str],
    *,
    show_progress: bool,
    description: str,
) -> tuple[list[str], list[tuple[str, dict]]]:
    """Single-pass blocked clustering shared by dedup and canonicalization.

    Returns the per-row cluster id list and ``(cluster_id, representative
    dict)`` pairs in first-seen order. The representative is always the
    *first* member of its cluster.
    """
    cluster_ids: list[str | None] = [None] * len(df)
    representatives: list[tuple[str, dict]] = []
    author_blocks, title_blocks, unknown_author = _candidate_index_lookup(df)

    for i in iter_with_progress(
        list(range(len(df))),
        show_progress=show_progress,
        description=description,
        item_label=lambda idx: f"row {idx + 1}/{len(df)}",
    ):
        if cluster_ids[i] is not None:
            continue
        paper_i = _row_to_dict(df.iloc[i])
        cluster_id = make_cluster_id(i, df.iloc[i])
        cluster_ids[i] = cluster_id
        representatives.append((cluster_id, paper_i))

        candidate_js = _candidate_indices(
            df.iloc[i],
            author_blocks=author_blocks,
            title_blocks=title_blocks,
            unknown_author=unknown_author,
        )
        for j in sorted(idx for idx in candidate_js if idx > i):
            if cluster_ids[j] is not None:
                continue
            if not _years_can_match(df.iloc[i].get("Year"), df.iloc[j].get("Year"), cfg):
                continue
            if compare_papers(paper_i, _row_to_dict(df.iloc[j]), cfg):
                cluster_ids[j] = cluster_id
    return cluster_ids, representatives  # all entries are filled by construction
```

Rewrite `dedup_references` to delegate (behavior identical — its existing tests must stay green):

```python
    # inside dedup_references, replacing dedup.py:236-272
    df = df.reset_index(drop=True)

    def _ref_cluster_id(_i: int, row: pd.Series) -> str:
        return make_reference_id(
            row.get("Authors_List") or row.get("Authors", ""),
            row.get("Year"),
            row.get("Title", ""),
        )

    cluster_id_list, representatives = _cluster_rows(
        df, cfg, _ref_cluster_id,
        show_progress=show_progress,
        description="Deduplicating references",
    )
    mapping = pd.Series(cluster_id_list, index=df.index, name="cited_id", dtype=object)
```

- [ ] **Step 4: Implement `canonicalize_works` and `_compute_rings`** (append to `dedup.py`; import `make_work_id` from `citegraph.ids`)

```python
def _compute_rings(ring0_ids: set[str], edges: pd.DataFrame) -> dict[str, int]:
    """BFS discovery depth from the ring-0 seed set over citation edges."""
    rings = {wid: 0 for wid in ring0_ids}
    out_edges: dict[str, set[str]] = defaultdict(set)
    for citing, cited in zip(edges["citing_id"], edges["cited_id"], strict=False):
        out_edges[str(citing)].add(str(cited))
    frontier = set(ring0_ids)
    depth = 0
    while frontier:
        depth += 1
        nxt = {c for w in frontier for c in out_edges.get(w, ()) if c not in rings}
        for c in nxt:
            rings[c] = depth
        frontier = nxt
    return rings


_WORK_COLUMNS = ["ring", "source_file", "Title", "Authors", "Authors_List", "Journal", "Year"]


def canonicalize_works(
    sources: pd.DataFrame,
    citations_raw: pd.DataFrame,
    cfg: DedupConfig | None = None,
    *,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Cluster sources + raw citations into canonical works.

    Sources occupy the head of the combined frame, so the single-pass
    first-match-wins loop guarantees: (a) two matching sources merge
    (duplicate PDFs), (b) a citation matching a source becomes that
    work, and (c) every cluster representative — hence its metadata —
    is the full-text side whenever one exists.
    """
    cfg = cfg or DedupConfig()
    require_columns(sources, ["id", "source_file", "Title", "Year"], artifact="sources")
    require_columns(citations_raw, ["Title", "Year", "citing_id"], artifact="citations_raw")
    for frame, name in ((sources, "sources"), (citations_raw, "citations_raw")):
        if "Authors" not in frame.columns and "Authors_List" not in frame.columns:
            raise ValueError(f"{name} is missing required column: Authors or Authors_List")

    src = sources.reset_index(drop=True)
    cit = citations_raw.reset_index(drop=True)
    n_src = len(src)
    combined = pd.concat([src, cit], ignore_index=True, sort=False)

    def _cluster_id(i: int, row: pd.Series) -> str:
        if i < n_src:
            return str(row["id"])
        return make_work_id(
            row.get("Authors_List") or row.get("Authors", ""),
            row.get("Year"),
            row.get("Title", ""),
        )

    cluster_ids, representatives = _cluster_rows(
        combined, cfg, _cluster_id,
        show_progress=show_progress,
        description="Canonicalizing works",
    )

    source_canonical = {str(src.iloc[i]["id"]): cluster_ids[i] for i in range(n_src)}
    ring0_ids = set(cluster_ids[:n_src])

    edges = pd.DataFrame(
        {
            "citing_id": [
                source_canonical.get(str(c), str(c)) for c in cit["citing_id"]
            ],
            "cited_id": cluster_ids[n_src:],
        }
    )
    n_before = len(edges)
    edges = edges[edges["citing_id"] != edges["cited_id"]]
    n_self_loops = n_before - len(edges)
    edges = edges.drop_duplicates().reset_index(drop=True)

    source_file_by_cluster: dict[str, str] = {}
    for i in range(n_src):
        source_file_by_cluster.setdefault(cluster_ids[i], str(src.iloc[i]["source_file"]))

    rings = _compute_rings(ring0_ids, edges)
    records = []
    for cluster_id, rep in representatives:
        record = dict(rep)
        record["id"] = cluster_id
        record["ring"] = rings.get(cluster_id, 1)
        record["source_file"] = source_file_by_cluster.get(cluster_id, "")
        records.append(record)
    works = pd.DataFrame(records).set_index("id")[_WORK_COLUMNS]

    stats = {
        "n_self_loops_dropped": int(n_self_loops),
        "n_source_duplicates_merged": int(n_src - len(ring0_ids)),
    }
    logger.info(
        "Canonicalized %d sources + %d citations into %d works (%d edges)",
        n_src, len(cit), len(works), len(edges),
    )
    return works, edges, stats
```

Note the `rings.get(cluster_id, 1)` default: a work whose only inbound edge was a dropped self-loop *is* its ring-0 cluster (already in `rings`); the default only covers hypothetical orphans defensively.

- [ ] **Step 5: Run** `pytest tests/test_dedup.py -v && pytest && ruff check .` — Expected: PASS (both new tests and the untouched `dedup_references` tests)

- [ ] **Step 6: Commit**

```bash
git add src/citegraph/dedup.py tests/test_dedup.py
git commit -m "feat: canonicalize_works — unified source+citation clustering with rings"
```

---

### Task 4: Stage 2 writes `sources.csv` with `w-` ids

**Files:**
- Modify: `src/citegraph/extract_metadata.py:76-81` (`metadata_to_record`), `src/citegraph/pipeline.py:172-178` (`_load_papers` → `_load_sources`), `pipeline.py:275-279` (CSV write)
- Test: `tests/test_pipeline.py` (and any `test_failure_isolation.py` assertions on `papers.csv`)

**Interfaces:**
- Consumes: `make_work_id` (Task 1), `OutLayout.sources_csv` (Task 2).
- Produces: `metadata_to_record` returns dict with `id = make_work_id(...)`; `Pipeline._load_sources() -> pd.DataFrame` (raises `StageNotReadyError` mentioning `citegraph metadata`); `extract_paper_metadata` writes `sources.csv`. Downstream stage 3 (Task 5) consumes both.

- [ ] **Step 1: Write the failing test** (add to `tests/test_pipeline.py`, using its existing `_FakeClient` fixture pattern)

```python
def test_metadata_stage_writes_sources_csv_with_work_ids(tmp_path, fake_pipeline):
    # `fake_pipeline` = however the file's existing tests build
    # Pipeline(client=_FakeClient()) with markdown fixtures in tmp_path.
    pipeline = fake_pipeline
    df = pipeline.extract_paper_metadata()
    assert (pipeline.layout.out_dir / "sources.csv").exists()
    assert not (pipeline.layout.out_dir / "papers.csv").exists()
    assert df["id"].str.startswith("w-").all()
    assert {"id", "source_file", "Title", "Authors", "Authors_List", "Journal", "Year"} <= set(df.columns)
```

(Adapt the fixture name to what `test_pipeline.py` actually uses — it builds pipelines inline; follow the file's existing setup helper.)

- [ ] **Step 2: Run** `pytest tests/test_pipeline.py -v -k sources_csv` — Expected: FAIL

- [ ] **Step 3: Implement**

1. `extract_metadata.py`: import `make_work_id` instead of `make_paper_id`; in `metadata_to_record` set `record["id"] = make_work_id(meta.Authors_List or meta.Authors, meta.Year, meta.Title)`.
2. `pipeline.py`: rename `_load_papers` → `_load_sources`, reading `self.layout.sources_csv` with message `f"Missing {path}. Run `citegraph metadata` first."`.
3. `pipeline.py:278`: `df.to_csv(self.layout.sources_csv, index=False)` and update the log line.
4. Fix every in-repo caller of `_load_papers` (`extract_paper_references` at `pipeline.py:303-304`) and any test that asserted on `papers.csv` after stage 2 — update them to `sources.csv` / `w-` prefixes. `detect_source_duplicates` (in `reports.py`) consumes the records list unchanged — no edit there; its JSON keys (`canonical_paper_id`) stay as-is.

- [ ] **Step 4: Run** `pytest && ruff check .` — Expected: PASS (fix any stage-2 assertions still expecting `p-` ids)

- [ ] **Step 5: Commit**

```bash
git add -A src tests
git commit -m "feat: stage 2 writes sources.csv with w- work ids"
```

---

### Task 5: Stage 3 writes `citations_raw.csv`

**Files:**
- Modify: `src/citegraph/pipeline.py:180-186` (`_load_raw_refs` → `_load_citations_raw`), `pipeline.py:296-412` (`extract_paper_references`: param rename `papers_df` → `sources_df`, CSV write at `pipeline.py:410`)
- Test: `tests/test_pipeline.py`, `tests/test_failure_isolation.py`

**Interfaces:**
- Consumes: `_load_sources` (Task 4), `OutLayout.citations_raw_csv` (Task 2).
- Produces: `Pipeline.extract_paper_references(markdown_paths=None, sources_df=None) -> pd.DataFrame` writing `citations_raw.csv` (columns: `Title, Authors_List, Journal, Year, Authors, citing_id` — same as today's `references_raw.csv`); `Pipeline._load_citations_raw()` with hint `citegraph references`.

- [ ] **Step 1: Write the failing test** (add to `tests/test_pipeline.py`)

```python
def test_references_stage_writes_citations_raw_csv(tmp_path, fake_pipeline):
    pipeline = fake_pipeline
    sources = pipeline.extract_paper_metadata()
    raw = pipeline.extract_paper_references(None, sources)
    assert (pipeline.layout.out_dir / "citations_raw.csv").exists()
    assert not (pipeline.layout.out_dir / "references_raw.csv").exists()
    assert raw["citing_id"].str.startswith("w-").all()
```

- [ ] **Step 2: Run** `pytest tests/test_pipeline.py -v -k citations_raw` — Expected: FAIL

- [ ] **Step 3: Implement** — mechanical renames listed in Files above; keep the per-paper failure isolation, duplicate-skip warning, and `papers_no_references.json` logic byte-identical (its `paper_id` key now carries a `w-` id; that's fine, the key name stays).

- [ ] **Step 4: Run** `pytest && ruff check .` — Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add -A src tests
git commit -m "feat: stage 3 writes citations_raw.csv keyed by work ids"
```

---

### Task 6: Stage 4 canonicalizes into `works.csv`

**Files:**
- Modify: `src/citegraph/pipeline.py:417-460` (`deduplicate`), `pipeline.py:188-194` (`_load_references` → `_load_works`, reading `works_csv` with hint `citegraph dedup`), `pipeline.py:36` (import `canonicalize_works`)
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `canonicalize_works` (Task 3), `_load_sources` (Task 4), `_load_citations_raw` (Task 5), `OutLayout.works_csv`.
- Produces: `Pipeline.deduplicate(sources=None, citations_raw=None) -> tuple[pd.DataFrame, pd.DataFrame]` (works, edges) writing `works.csv` (index `id`) + `citation_graph.csv`; stashes `self._canonicalize_stats: dict` (default `{}` set in `__init__`) for `run()`; `Pipeline._load_works()`.

- [ ] **Step 1: Write the failing test**

```python
def test_deduplicate_produces_works_with_rings(tmp_path, fake_pipeline):
    pipeline = fake_pipeline
    sources = pipeline.extract_paper_metadata()
    raw = pipeline.extract_paper_references(None, sources)
    works, graph = pipeline.deduplicate(sources, raw)
    assert (pipeline.layout.out_dir / "works.csv").exists()
    assert works.index.str.startswith("w-").all()
    assert set(works.columns) >= {"ring", "source_file", "Title", "Authors", "Journal", "Year"}
    assert (works["ring"] == 0).sum() == len(sources)
    assert graph["citing_id"].isin(works.index).all()
    assert graph["cited_id"].isin(works.index).all()
```

- [ ] **Step 2: Run** `pytest tests/test_pipeline.py -v -k works_with_rings` — Expected: FAIL

- [ ] **Step 3: Implement** — replace the body of `deduplicate`:

```python
    def deduplicate(
        self,
        sources: pd.DataFrame | None = None,
        citations_raw: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if sources is None:
            sources = self._load_sources()
        if citations_raw is None:
            citations_raw = self._load_citations_raw()

        if citations_raw.empty:
            citations_raw = pd.DataFrame(
                columns=["Title", "Authors", "Authors_List", "Journal", "Year", "citing_id"]
            )

        works, graph, stats = canonicalize_works(
            sources, citations_raw, self.dedup_config, show_progress=self.show_progress
        )
        self._canonicalize_stats = stats
        works.to_csv(self.layout.works_csv)
        graph.to_csv(self.layout.graph_csv, index=False)
        logger.info(
            "Wrote %s (%d works) and %s (%d edges); %d self-loop(s) dropped, "
            "%d duplicate source(s) merged",
            self.layout.works_csv, len(works),
            self.layout.graph_csv, len(graph),
            stats["n_self_loops_dropped"], stats["n_source_duplicates_merged"],
        )
        return works, graph
```

The old `require_columns`/empty-frame special case moves inside `canonicalize_works` (already done in Task 3); an empty *sources* frame is a real error (`StageNotReadyError` came from `_load_sources`). Delete the now-unused `dedup_references` import once nothing in `pipeline.py` uses it.

- [ ] **Step 4: Run** `pytest && ruff check .` — fix `test_pipeline.py` end-to-end assertions (they will still call `maybe_enrich`/`normalize_authors`, untouched until Tasks 7-9; if any break on column expectations, adjust only what this task changed: works.csv shape and method signature).

- [ ] **Step 5: Commit**

```bash
git add -A src tests
git commit -m "feat: stage 4 canonicalizes sources + citations into works.csv"
```

---

### Task 7: Enrichment over works + legacy cache fallback

**Files:**
- Modify: `src/citegraph/enrich.py` (`enrich_references` → `enrich_works` at `enrich.py:566`, `_enrich_one` cache lookup at `enrich.py:446-451`, `_write_enrichment_sidecars` for the per-ring breakdown), `src/citegraph/pipeline.py:548-558` (`maybe_enrich`)
- Test: `tests/test_enrich.py`, `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `works.csv` shape (Task 6), `OutLayout.enriched_works_csv` (Task 2).
- Produces: `enrich_works(df, cfg=None, layout=None) -> pd.DataFrame` (identical behavior to today's `enrich_references`, new name; a `ring` column, when present, is passed through untouched and summarized); `Pipeline.maybe_enrich(works=None)` writing `enriched_works.csv`; cache fallback: missing `w-<slug>.json` reads legacy `r-<slug>.json`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_enrich.py`, reusing its existing offline fake-client/transport helpers)

```python
def test_enrich_one_falls_back_to_legacy_cache_filename(tmp_path):
    import json
    import pandas as pd
    from citegraph.enrich import EnrichConfig, _enrich_one

    legacy = {"doi": "10.1/x", "enrichment_source": "crossref", "Title": "Cached"}
    (tmp_path / "r-ostrom-1990-governing.json").write_text(json.dumps(legacy))
    row = pd.Series({"Title": "Governing", "Authors": "Ostrom", "Year": 1990})
    out = _enrich_one("w-ostrom-1990-governing", row, EnrichConfig(), client=None,
                      enrichment_dir=tmp_path)
    assert out["doi"] == "10.1/x"  # served from the legacy r- cache, no network


def test_enrichment_summary_breaks_down_by_ring(tmp_path):
    # Build a 2-row works frame (ring 0 and ring 1) and run enrich_works with
    # the file's existing fake transport that returns one match and one miss;
    # then assert enrichment_summary.json contains {"by_ring": {"0": ..., "1": ...}}.
    ...  # follow the existing summary-assertion test in this file as the template
```

(The second test's body follows the existing `enrichment_summary.json` test in `test_enrich.py` — copy its fake-transport setup, add a `ring` column, assert the new `by_ring` key.)

- [ ] **Step 2: Run** `pytest tests/test_enrich.py -v -k "legacy or by_ring"` — Expected: FAIL

- [ ] **Step 3: Implement**

1. Rename `enrich_references` → `enrich_works` (update the docstring: "references" → "works"). Keep the log line but say "works".
2. In `_enrich_one` (`enrich.py:446-451`) add the fallback before the network path:

```python
    if enrichment_dir is not None:
        cache_path = enrichment_dir / f"{ref_id}.json"
        if not cache_path.exists() and ref_id.startswith("w-"):
            legacy = enrichment_dir / f"r-{ref_id[2:]}.json"
            if legacy.exists():
                cache_path = legacy
        if cache_path.exists():
            ...  # unchanged
```

New results still write to the `w-` filename (`_write_cache(enrichment_dir / f"{ref_id}.json", ...)` — unchanged, `ref_id` is already `w-…`).
3. In `_write_enrichment_sidecars`, when the frame has a `ring` column add to the summary dict: `summary["by_ring"] = {str(ring): {"n": int(len(g)), "n_matched": int((g["enrichment_status"] == "matched").sum())} for ring, g in enriched.groupby("ring")}`.
4. `pipeline.maybe_enrich`: parameter `works`, loads via `self._load_works()`, calls `enrich_works`, writes `self.layout.enriched_works_csv`.

- [ ] **Step 4: Run** `pytest && ruff check .` — update remaining `test_enrich.py` imports of `enrich_references` to `enrich_works`.

- [ ] **Step 5: Commit**

```bash
git add -A src tests
git commit -m "feat: enrichment runs over all works with legacy r- cache fallback"
```

---

### Task 8: Author stage over works

**Files:**
- Modify: `src/citegraph/authors.py` — `AuthorOccurrence` (`authors.py:295-311`), `normalize_authors` signature (`authors.py:973`), `_collect_occurrences` (`authors.py:1145-1232`), `_clusters_to_authors_df` (`authors.py:946-978`), `_clusters_to_citations_df` (`authors.py:981-996`), `_reference_citers` (`authors.py:~930-943`), `_review_reason` (adapt if it reads `record_kind`)
- Test: `tests/test_authors.py`

**Interfaces:**
- Consumes: `works.csv` shape (index `id`, `ring` column), `enriched_works.csv`, edges.
- Produces:
  - `normalize_authors(*, works, enriched_works=None, citation_edges=None, cfg=None, aliases=None) -> tuple[authors_df, citations_df, review]`
  - `AuthorOccurrence` fields: `parsed, record_id, position, co_author_keys, openalex_id, orcid` (`record_kind` and `citing_paper_id` removed).
  - `authors_df` columns: `display_name, surname, surname_norm, canonical_given, initials, openalex_id, orcid, n_works, n_core_works, n_citations_received, n_distinct_citing_works`; index `id`; sorted by `n_citations_received` desc.
  - `citations_df` columns: `author_id, record_id, position, raw_author`.

- [ ] **Step 1: Write the failing test** (add to `tests/test_authors.py`; a minimal works-shaped fixture)

```python
import pandas as pd

from citegraph.authors import normalize_authors


def _works_fixture():
    works = pd.DataFrame(
        {
            "ring": [0, 0, 1],
            "source_file": ["a.pdf", "b.pdf", ""],
            "Title": ["Core paper A", "Core paper B", "External classic"],
            "Authors": ["Cardenas, Juan Camilo", "Cardenas, Juan Camilo, Ostrom, Elinor", "Ostrom, Elinor"],
            "Authors_List": [
                ["Cardenas, Juan Camilo"],
                ["Cardenas, Juan Camilo", "Ostrom, Elinor"],
                ["Ostrom, Elinor"],
            ],
            "Journal": ["JDE", "WD", "CUP"],
            "Year": [2000, 2004, 1990],
        },
        index=pd.Index(["w-a", "w-b", "w-c"], name="id"),
    )
    edges = pd.DataFrame(
        [
            {"citing_id": "w-a", "cited_id": "w-c"},
            {"citing_id": "w-b", "cited_id": "w-c"},
            {"citing_id": "w-a", "cited_id": "w-b"},  # core cites core
        ]
    )
    return works, edges


def test_author_metrics_span_core_and_cited_roles():
    works, edges = _works_fixture()
    authors, citations, _review = normalize_authors(works=works, citation_edges=edges)

    cardenas = authors[authors["surname_norm"] == "cardenas"].iloc[0]
    assert cardenas["n_works"] == 2           # authored w-a and w-b
    assert cardenas["n_core_works"] == 2
    assert cardenas["n_citations_received"] == 1   # w-b is cited once (by w-a)
    assert cardenas["n_distinct_citing_works"] == 1

    ostrom = authors[authors["surname_norm"] == "ostrom"].iloc[0]
    assert ostrom["n_works"] == 2             # w-b (co-author) and w-c
    assert ostrom["n_core_works"] == 1        # only w-b is ring 0
    assert ostrom["n_citations_received"] == 3  # 2 into w-c + 1 into w-b

    assert set(citations.columns) == {"author_id", "record_id", "position", "raw_author"}
    assert "record_kind" not in citations.columns
```

- [ ] **Step 2: Run** `pytest tests/test_authors.py -v -k span_core` — Expected: FAIL (signature mismatch)

- [ ] **Step 3: Implement**

1. `AuthorOccurrence`: delete `record_kind` and `citing_paper_id` fields; update the docstring and the two construction sites in `_collect_occurrences`.
2. `_collect_occurrences(*, works, enriched_works)`: replace the two loops (`ref_rows`/`paper_rows`) with **one** loop over `works.iterrows()` (index is the id). The `enrich_map` builds from `enriched_works` exactly as it did from `enriched_references` (`authors.py:1163-1169`), and `_match_enrichment` now applies to *every* work — that is the whole point (core authors gain external ids). The surname lexicon builds from the single row list.
3. `_clusters_to_authors_df(clusters, *, works, citation_edges)`:

```python
def _clusters_to_authors_df(
    clusters: list[AuthorCluster],
    *,
    works: pd.DataFrame,
    citation_edges: pd.DataFrame | None = None,
) -> pd.DataFrame:
    ring_by_work = works["ring"].to_dict() if "ring" in works.columns else {}
    in_edges: dict[str, set[str]] = defaultdict(set)
    n_in_edges: dict[str, int] = defaultdict(int)
    if citation_edges is not None and not citation_edges.empty:
        for _, row in citation_edges.iterrows():
            cited, citing = str(row.get("cited_id")), str(row.get("citing_id"))
            if cited and citing:
                in_edges[cited].add(citing)
                n_in_edges[cited] += 1

    rows = []
    for c in clusters:
        work_ids = {o.record_id for o in c.occurrences}
        citing: set[str] = set()
        for w in work_ids:
            citing |= in_edges.get(w, set())
        rows.append(
            {
                "id": c.id,
                "display_name": c.display_name,
                "surname": c.surname,
                "surname_norm": c.surname_norm,
                "canonical_given": c.canonical_given,
                "initials": c.initials,
                "openalex_id": c.openalex_id,
                "orcid": c.orcid,
                "n_works": len(work_ids),
                "n_core_works": sum(1 for w in work_ids if ring_by_work.get(w) == 0),
                "n_citations_received": sum(n_in_edges.get(w, 0) for w in work_ids),
                "n_distinct_citing_works": len(citing),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df.set_index(pd.Index([], name="id"))
    return df.set_index("id").sort_values("n_citations_received", ascending=False)
```

Delete `_reference_citers` (superseded). `_clusters_to_citations_df` drops the `record_kind` and `citing_paper_id` keys.
4. `normalize_authors`: new keyword-only signature (see Interfaces); pass `works=works` through to `_clusters_to_authors_df`. Update its docstring.
5. `_review_reason`: if it counts via `record_kind == "reference"` occurrences, count *all* occurrences instead (same intent: "initial-only cluster with several appearances and no external id"). Verify by reading it before editing.
6. Sweep `tests/test_authors.py`: replace `references=`/`papers=`/`enriched_references=` kwargs with `works=`/`enriched_works=`. Reference-style fixtures become works frames (add `ring` = 1, `source_file` = "" defaults; paper-style rows get `ring` = 0). Metric assertions rename `n_reference_citations` → `n_citations_received`, `n_distinct_papers_citing` → `n_distinct_citing_works`, `n_occurrences` → `n_works`. This is the largest mechanical sweep in the plan (~38k-line test file) — do it with careful search-and-replace plus a full test run, not by hand-rewriting tests.

- [ ] **Step 4: Run** `pytest tests/test_authors.py -v && pytest && ruff check .` — Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add -A src tests
git commit -m "feat: author normalization runs over unified works"
```

---

### Task 9: Pipeline wiring — `normalize_authors`, `run()`, `PipelineResult`

**Files:**
- Modify: `src/citegraph/pipeline.py:465-543` (`normalize_authors`), `pipeline.py:581-636` (`run()`), `src/citegraph/schemas.py:73-89` (`PipelineResult`), `graph.py:100-108` (`from_pipeline_result` — minimal touch here, full rework in Task 10)
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: Tasks 4-8 signatures.
- Produces:
  - `Pipeline.normalize_authors(works=None, graph=None) -> tuple[pd.DataFrame, pd.DataFrame]` reading `works.csv`, `enriched_works.csv` (warning text now names `enriched_works.csv`), `citation_graph.csv`.
  - `PipelineResult(works, graph, authors=None, author_citations=None)` — `papers`/`references` fields removed.
  - `run()` summary keys: `n_works`, `n_core_works`, `n_citations_raw`, `n_edges`, `n_core_to_core_edges`, `n_self_loops_dropped`, plus all existing failure/warning counters, `model`, `enrich`, `dedup_config`, `author_config`.

- [ ] **Step 1: Write the failing test**

```python
def test_run_summary_reports_works_counts(tmp_path, fake_pipeline):
    import json
    pipeline = fake_pipeline
    result = pipeline.run()
    summary = json.loads((pipeline.layout.out_dir / "run_summary.json").read_text())
    assert summary["n_works"] == len(result.works)
    assert summary["n_core_works"] == int((result.works["ring"] == 0).sum())
    assert "n_core_to_core_edges" in summary and "n_self_loops_dropped" in summary
    assert not hasattr(result, "papers")
```

- [ ] **Step 2: Run** — Expected: FAIL

- [ ] **Step 3: Implement**

1. `normalize_authors`: load `works` via `_load_works()`; drop the `papers` parameter/loader entirely; read `enriched_works_csv` (keep the `OpenAlex_Authors` re-parse and the interrupted-enrichment warning, with updated filename in the message); call the new `authors.normalize_authors(works=..., enriched_works=..., citation_edges=..., cfg=..., aliases=...)`.
2. `run()`:

```python
        markdown_paths = self.convert_pdfs()
        sources = self.extract_paper_metadata(markdown_paths)
        citations_raw = self.extract_paper_references(markdown_paths, sources)
        works, graph = self.deduplicate(sources, citations_raw)
        works = self.maybe_enrich(works)
        authors_df, citations_df = self.normalize_authors(works=works, graph=graph)

        ring0 = set(works.index[works["ring"] == 0]) if "ring" in works.columns else set()
        n_core_to_core = int(
            (graph["citing_id"].isin(ring0) & graph["cited_id"].isin(ring0)).sum()
        )
        run_summary = {
            "n_works": int(len(works)),
            "n_core_works": int(len(ring0)),
            "n_citations_raw": int(len(citations_raw)),
            "n_edges": int(len(graph)),
            "n_core_to_core_edges": n_core_to_core,
            "n_self_loops_dropped": int(
                self._canonicalize_stats.get("n_self_loops_dropped", 0)
            ),
            # ... existing failure/warning counters, model, enrich, configs unchanged
        }
```

Manifest artifact keys: `sources`, `citations_raw`, `works`, `citation_graph`, `authors`, `author_citations`, `run_summary`.
3. `schemas.py` `PipelineResult`: fields `works`, `graph`, `authors`, `author_citations`; update docstring. `graph.py:from_pipeline_result` → `cls(works=result.works, edges=result.graph, ...)` (full class rework next task; here only keep imports/tests green — if Task 10 hasn't run yet, adjust the constructor call to whatever keeps the current class importable, or do Tasks 9 and 10 as one commit if the seam is too tangled; prefer two commits, constructor-kwarg shim in this one).
4. Update `pipeline.py` module docstring (`pipeline.py:1-18`) to the new stage narrative.

- [ ] **Step 4: Run** `pytest && ruff check .`

- [ ] **Step 5: Commit**

```bash
git add -A src tests
git commit -m "feat: pipeline runs end-to-end on the works model"
```

---

### Task 10: `CitationGraph` works-based API

**Files:**
- Modify: `src/citegraph/graph.py` (whole class)
- Test: `tests/test_graph.py`

**Interfaces:**
- Consumes: `works.csv` (indexed by `id`), `citation_graph.csv`, author tables (Task 8 shapes).
- Produces:
  - `CitationGraph(works, edges, authors=None, author_citations=None)`; attributes `works`, `edges`, `authors`, `author_citations`.
  - `n_works: int`, `n_core_works: int`, `n_edges: int` (properties). `n_papers`/`n_references` removed.
  - `core: pd.DataFrame` (property, `ring == 0`), `ring(n: int) -> pd.DataFrame`, `core_citations() -> pd.DataFrame`, `top_authors(n: int = 20, ring: int | None = None) -> pd.DataFrame` (adds `n_works_in_selection` column).
  - Unchanged names, works-backed: `cited_by`, `citers_of`, `top_cited`, `has_authors`, `top_cited_authors` (sorts by `n_citations_received`), `find_author` (sorts likewise), `citations_of`, `papers_citing_author`, `citation_context_for_author`, `citing_papers_by_author`, `source_journals_citing_author`, `to_networkx`.
  - `from_out_dir` loads `works.csv`; if absent but `papers.csv` exists, raises `FileNotFoundError` whose message contains the word `legacy` and the command sequence `citegraph metadata`, `citegraph references`, `citegraph dedup`.

- [ ] **Step 1: Write the failing tests** (rework `tests/test_graph.py`; keep its author-method tests, port fixtures to works shape). New coverage:

```python
def test_core_ring_and_top_authors(works_graph):
    # works_graph: CitationGraph over the Task 8 _works_fixture shapes + author tables
    g = works_graph
    assert set(g.core.index) == {"w-a", "w-b"}
    assert set(g.ring(1).index) == {"w-c"}
    cc = g.core_citations()
    assert list(cc.itertuples(index=False)) == [("w-a", "w-b")]

    top = g.top_authors(5, ring=0)
    assert top.iloc[0]["n_works_in_selection"] == 2  # Cárdenas authored both core works

    top_all = g.top_authors(5)
    assert "n_works_in_selection" in top_all.columns


def test_from_out_dir_legacy_layout_raises_with_migration_hint(tmp_path):
    import pandas as pd
    import pytest
    from citegraph.graph import CitationGraph

    pd.DataFrame({"id": ["p-x"]}).to_csv(tmp_path / "papers.csv", index=False)
    with pytest.raises(FileNotFoundError, match="legacy"):
        CitationGraph.from_out_dir(tmp_path)


def test_top_cited_includes_core_works(works_graph):
    top = works_graph.top_cited(5)
    assert "w-b" in top.index          # a core paper cited within the corpus
    assert top.loc["w-c", "citation_count"] == 2
```

- [ ] **Step 2: Run** `pytest tests/test_graph.py -v` — Expected: FAIL

- [ ] **Step 3: Implement** — rework `graph.py`:

1. Constructor stores `self.works` (must be indexed by `id`; `set_index("id")` defensively when an `id` column is present), `self.edges`, author tables as today.
2. `from_out_dir`: require `works_csv` + `graph_csv`; legacy branch:

```python
        if not layout.works_csv.exists() and layout.papers_csv.exists():
            raise FileNotFoundError(
                f"{layout.works_csv} not found, but legacy papers.csv exists. "
                "This out_dir predates the works model. Re-run the cheap "
                "downstream stages (caches are reused): citegraph metadata "
                f"--out {out_dir} && citegraph references --out {out_dir} && "
                f"citegraph dedup --out {out_dir} && citegraph authors --out {out_dir}"
            )
```

(Task 13 removes `layout.papers_csv`; this branch then tests `(layout.out_dir / "papers.csv").exists()` directly — write it that way now so Task 13 doesn't touch it.)
3. New members:

```python
    @property
    def core(self) -> pd.DataFrame:
        return self.works[self.works["ring"] == 0]

    def ring(self, n: int) -> pd.DataFrame:
        return self.works[self.works["ring"] == n]

    def core_citations(self) -> pd.DataFrame:
        ids = set(self.core.index)
        return self.edges[
            self.edges["citing_id"].isin(ids) & self.edges["cited_id"].isin(ids)
        ].reset_index(drop=True)

    def top_authors(self, n: int = 20, ring: int | None = None) -> pd.DataFrame:
        """Authors ranked by distinct works authored (optionally one ring only)."""
        self._require_authors()
        ac = self.author_citations
        if ring is not None:
            ring_ids = set(self.works.index[self.works["ring"] == ring])
            ac = ac[ac["record_id"].isin(ring_ids)]
        counts = ac.groupby("author_id")["record_id"].nunique().sort_values(ascending=False)
        head = counts.head(n)
        out = self.authors.loc[self.authors.index.isin(head.index)].copy()
        out["n_works_in_selection"] = out.index.map(head).astype(int)
        return out.sort_values("n_works_in_selection", ascending=False)
```

4. Port existing methods: every `self.references` → `self.works`; every `self.papers` (id-as-column) → `self.works` (id-as-index) — e.g. `citers_of` becomes `self.works.loc[self.works.index.isin(citing_ids)]`. In `citation_context_for_author` drop the `record_kind == "reference"` filter (`graph.py:266-269`) — the author's records join against `edges.cited_id` directly, which naturally covers cited core works; the citing-side metadata merge reads from `self.works.reset_index()` renamed to the same `citing_paper_id`/`source_paper_*` output columns (public column names unchanged). `top_cited_authors`/`find_author` sort by `n_citations_received`.
5. `to_networkx`: single loop over `self.works.iterrows()`; node attrs = row dict (includes `ring`, `source_file`); drop the `kind` attribute.
6. `__repr__`: `<CitationGraph: {n_core_works} core works, {n_works} works, {n_edges} edges{author_suffix}>`.

- [ ] **Step 4: Run** `pytest tests/test_graph.py -v && pytest && ruff check .`

- [ ] **Step 5: Commit**

```bash
git add -A src tests
git commit -m "feat: works-based CitationGraph with core/ring query idiom"
```

---

### Task 11: CLI surface

**Files:**
- Modify: `src/citegraph/cli.py` — echo strings and artifact args at `cli.py:362, 417, 434-484, 538, 564-601, 734-738`; docstrings mentioning old filenames
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: all pipeline signatures from Tasks 4-9.
- Produces: same command names and flags; `dedup` command's optional path argument becomes `citations_raw_csv` (default `<out>/citations_raw.csv`); `status` lists `sources.csv`, `citations_raw.csv`, `works.csv`, `enriched_works.csv`, `citation_graph.csv`, `authors.csv`, `author_citations.csv`.

- [ ] **Step 1: Write/adjust failing tests** — update `tests/test_cli.py` expectations: `metadata` echoes `sources.csv`; `references` echoes `citations_raw.csv`; `dedup` echoes `works.csv` (and `works`/`edges` counts); `enrich` echoes `enriched_works.csv`; `status` output lists the new filenames. Add one new assertion:

```python
def test_status_lists_works_artifacts(tmp_path, runner):
    result = runner.invoke(app, ["status", "--out", str(tmp_path)])
    assert "works.csv" in result.output
    assert "sources.csv" in result.output
    assert "papers.csv" not in result.output
```

- [ ] **Step 2: Run** `pytest tests/test_cli.py -v` — Expected: FAIL

- [ ] **Step 3: Implement** — mechanical: swap layout properties and strings; `dedup` command calls `pipeline.deduplicate(...)` via the two-frame path (read `citations_raw_csv` argument if given, else pipeline loaders); keep the Rich-stable assertion style the file already uses.

- [ ] **Step 4: Run** `pytest && ruff check .`

- [ ] **Step 5: Commit**

```bash
git add -A src tests
git commit -m "feat: CLI speaks works-model artifacts"
```

---

### Task 12: Report + web UI

**Files:**
- Modify: `src/citegraph/html_report.py` (artifact reads at `html_report.py:275-292`, checklist rows at `377-386`, stage-progress hints at `597-620`, plus section renderers naming papers/references), `src/citegraph/webui.py` (labels only — it serves `collect_report_data`'s payload)
- Test: `tests/test_report.py`, `tests/test_webui.py`

**Interfaces:**
- Consumes: works-model artifacts (Tasks 2-9).
- Produces: `collect_report_data` reads `sources.csv`/`citations_raw.csv`/`works.csv`/`enriched_works.csv`; payload gains `n_core_works`, `n_works`, `ring_counts` (dict ring→count) and `core_to_core_edges` (list of `{citing_id, cited_id}`); report renders a "Core corpus" summary block and a core→core citation table; an out_dir containing `papers.csv` but no `works.csv` yields a finding titled `legacy artifacts` (not a crash).

- [ ] **Step 1: Write failing tests**

```python
def test_report_data_includes_ring_summary(populated_out_dir):
    # populated_out_dir: the file's existing fixture, ported to works.csv shape
    data = collect_report_data(OutLayout(populated_out_dir))
    assert data["ring_counts"].get(0, 0) > 0
    assert "core_to_core_edges" in data


def test_report_flags_legacy_out_dir(tmp_path):
    import pandas as pd
    pd.DataFrame({"id": ["p-x"]}).to_csv(tmp_path / "papers.csv", index=False)
    data = collect_report_data(OutLayout(tmp_path))
    assert any("legacy" in str(f).lower() for f in data["errors"])
```

(Adapt `data["errors"]` to the container `collect_report_data` actually uses for findings — read the function first; findings-not-crashes is its existing contract.)

- [ ] **Step 2: Run** `pytest tests/test_report.py -v -k "ring or legacy"` — Expected: FAIL

- [ ] **Step 3: Implement** — swap layout reads; compute `ring_counts` / `core_to_core_edges` from the works+edges frames when present; add the legacy finding when `papers.csv` exists without `works.csv`; rename user-visible section labels ("Papers" → "Core works", "References" → "Works"); update `webui.py` static labels to match. Update `tests/test_report.py` / `tests/test_webui.py` fixtures to write works-model CSVs.

- [ ] **Step 4: Run** `pytest && ruff check .`

- [ ] **Step 5: Commit**

```bash
git add -A src tests
git commit -m "feat: report and web UI render the works model"
```

---

### Task 13: Remove legacy code paths + documentation sweep

**Files:**
- Modify: `src/citegraph/ids.py` (delete `make_paper_id`/`make_reference_id`; keep `_first_author_token`; fold the slug builder into `make_work_id`), `src/citegraph/dedup.py` (delete `dedup_references` and its docstring example), `src/citegraph/io.py` (delete `papers_csv`, `references_raw_csv`, `references_csv`, `enriched_references_csv` properties), `src/citegraph/enrich.py` (delete any leftover `enrich_references` name), plus whatever the sweep finds
- Docs: `CLAUDE.md`, `README.md`, `docs/USER_GUIDE.md`, `docs/UI_ARCHITECTURE.md`
- Test: whole suite

**Interfaces:** none new — this task deletes.

- [ ] **Step 1: Sweep for stragglers**

Run: `grep -rn "make_paper_id\|make_reference_id\|dedup_references\|papers_csv\|references_raw_csv\|references_csv\|enriched_references\|record_kind\|n_reference_citations" src/ tests/ docs/ README.md CLAUDE.md`

Every hit must be deleted, renamed, or (in docs) rewritten. Two intentional survivors: the legacy-detection literals `"papers.csv"` in `graph.from_out_dir` and `html_report.py` (they *test for* the legacy layout by filename on purpose), and the `r-` prefix literal in `_enrich_one`'s cache fallback.

- [ ] **Step 2: Delete + fix** — `ids.py` ends up with `_first_author_token` and a `make_work_id` that owns the slug logic directly (move the body of `make_paper_id` into it, drop the `prefix` parameter). Update `test_ids.py` accordingly (the Task 1 cross-prefix test now hardcodes the expected legacy string instead of calling `make_paper_id`).

- [ ] **Step 3: Documentation sweep** — rewrite the affected sections of `CLAUDE.md` (stage pipeline, artifacts table, IDs, CitationGraph, testing), `README.md` usage/outputs, `docs/USER_GUIDE.md` (outputs, migration note with the exact re-run command sequence and the note that stale legacy CSVs can be deleted), `docs/UI_ARCHITECTURE.md` (payload fields). State the three-line R migration (`works.csv`, `filter(ring == 0)`).

- [ ] **Step 4: Run** `pytest && ruff check .` and `python -c "import citegraph"` — Expected: all green; `pytest tests/test_packaging.py -v` confirms docs assertions still hold.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor: remove legacy papers/references code paths; docs for works model"
```

---

### Task 14: End-to-end verification on a real corpus (manual gate)

**Files:** none (verification only)

- [ ] **Step 1:** Run the full suite one more time from a clean state: `pytest -q && ruff check .`
- [ ] **Step 2:** Dry-run the migration path on one of the user's real out_dirs (ask the user which; do NOT run enrichment against the network without their go-ahead): `citegraph metadata --out <dir> && citegraph references --out <dir> && citegraph dedup --out <dir> && citegraph status --out <dir>`. Confirm: `works.csv` exists, `n_core_works` equals the old paper count, `citation_graph.csv` contains at least one edge whose `cited_id` is a ring-0 work (core-cites-core actually fires on real data).
- [ ] **Step 3:** `citegraph report --out <dir> --open` — eyeball the ring summary and core→core table.
- [ ] **Step 4:** Report results to the user; hand off to superpowers:finishing-a-development-branch.
