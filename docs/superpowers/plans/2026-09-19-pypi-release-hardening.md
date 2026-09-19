# PyPI Release Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve the four verified PyPI release blockers while preserving the base install and existing per-tie analysis.

**Architecture:** Patch each behavior at its existing shared boundary: package metadata owns OCR dependencies, `TieCocitationTest` owns statistical summaries, `tie_cocitation_test` owns null eligibility, and `build_core_author_workbook` owns sheet-name safety. Each fix begins with one focused regression and adds no new subsystem.

**Tech Stack:** Python 3.11+, pytest, pandas/openpyxl, Hatchling, Ruff, Twine

**Spec:** `docs/superpowers/specs/2026-09-19-pypi-release-hardening-design.md`

## Global Constraints

- Keep the base installation free of Docling and openpyxl.
- Do not add Docling to the `dev` extra; conversion tests continue to use fakes.
- Do not replace the removed binomial aggregate with another inferential method.
- Keep per-tie permutation p-values, `n_significant`, `expected_significant`, and the assumption-bound Fisher diagnostic.
- Validate workbook names before reading/filtering extra-sheet data or opening the output file.
- Preserve every unrelated working-tree change; the affected files already contain uncommitted user work.
- Do not create implementation commits from the dirty tree unless the user separately asks to commit the combined changes.

---

### Task 1: Make the PDF Extras Install EasyOCR

**Files:**
- Modify: `tests/test_packaging.py:15-28`
- Modify: `pyproject.toml:42-64`

**Interfaces:**
- Consumes: PEP 621 optional-dependency lists under `project.optional-dependencies`.
- Produces: `citegraph[pdf]` and `citegraph[all]` requirements containing `docling[easyocr]>=2.0`.

- [ ] **Step 1: Tighten the packaging test to require Docling's EasyOCR extra**

Replace the loose Docling assertions with exact requirements:

```python
def test_pdf_conversion_dependency_is_optional() -> None:
    project = _pyproject()["project"]
    dependencies = project["dependencies"]
    extras = project["optional-dependencies"]

    assert not any(dep.startswith("docling") for dep in dependencies)
    assert "docling[easyocr]>=2.0" in extras["pdf"]


def test_all_extra_includes_pdf_and_enrichment_dependencies() -> None:
    extras = _pyproject()["project"]["optional-dependencies"]

    assert "docling[easyocr]>=2.0" in extras["all"]
    assert any(dep.startswith("httpx") for dep in extras["all"])
```

- [ ] **Step 2: Run the packaging regressions and confirm the missing extra**

Run: `.venv/bin/python -m pytest tests/test_packaging.py -v`

Expected: the two Docling assertions fail because the current value is `docling>=2.0`.

- [ ] **Step 3: Change only the two public PDF-capable extras**

In `pyproject.toml`, use the same requirement in both lists:

```toml
pdf = [
  "docling[easyocr]>=2.0",
]

all = [
  "docling[easyocr]>=2.0",
  "httpx>=0.27",
  "matplotlib>=3.8",
  "networkx>=3.0",
  "openpyxl>=3.1",
]
```

- [ ] **Step 4: Re-run the packaging tests**

Run: `.venv/bin/python -m pytest tests/test_packaging.py -v`

Expected: all packaging tests pass.

---

### Task 2: Remove the Invalid Binomial Aggregate

**Files:**
- Modify: `tests/test_cocitation_stats.py:12-20, 400-428`
- Modify: `src/citegraph/cocitation_stats.py:29-34, 50-57, 141-160, 337-end`

**Interfaces:**
- Consumes: existing `TieCocitationTest` per-tie results.
- Produces: no `binomial_p_value` property and no public `binomial_tail_p` helper; descriptive counts and `fisher_p_value` remain.

- [ ] **Step 1: Replace binomial-value tests with a public-API absence regression**

Remove `binomial_tail_p` from the test import and delete its numeric unit tests. Add:

```python
import citegraph.cocitation_stats as cocitation_stats


def test_dependent_binomial_aggregate_is_not_exposed(paired_corpus: CitationGraph) -> None:
    result = tie_cocitation_test(
        paired_corpus,
        [("a-adv", "a-stu")],
        min_papers=1,
        n_draws=50,
        seed=0,
    )

    assert not hasattr(result, "binomial_p_value")
    assert not hasattr(cocitation_stats, "binomial_tail_p")
    assert "binomial_tail_p" not in cocitation_stats.__all__
```

Update the empty-result test to assert only `fisher_p_value is None`, while retaining the descriptive-count assertions.

- [ ] **Step 2: Run the new API regression and confirm it fails**

Run: `.venv/bin/python -m pytest tests/test_cocitation_stats.py::test_dependent_binomial_aggregate_is_not_exposed -v`

Expected: FAIL because both the property and helper still exist.

- [ ] **Step 3: Delete the invalid API and narrow Fisher's documentation**

In `src/citegraph/cocitation_stats.py`:

- remove `"binomial_tail_p"` from `__all__`;
- delete `TieCocitationTest.binomial_p_value`;
- delete `binomial_tail_p`;
- replace the module-level combined-statistics discussion with a warning that
  Fisher assumes independent p-values and is not confirmatory for overlapping
  ties;
- change `fisher_p_value`'s docstring to:

```python
"""Fisher's combination of the per-tie p-values.

This assumes independent p-values. Supervision ties that share people do
not satisfy that assumption, so this is an assumption-bound diagnostic,
not confirmatory evidence for overlapping ties.
"""
```

- [ ] **Step 4: Run all co-citation tests**

Run: `.venv/bin/python -m pytest tests/test_cocitation_stats.py -v`

Expected: all remaining tests pass and no test imports or accesses the removed API.

---

### Task 3: Skip Nulls Missing Either Endpoint Stratum

**Files:**
- Modify: `tests/test_cocitation_stats.py` near the existing skipped-tie tests
- Modify: `src/citegraph/cocitation_stats.py:275-299`

**Interfaces:**
- Consumes: `source_pool` and `target_pool`, each created by `stratum`.
- Produces: one `SkippedTie` and no `TieCocitation` when either endpoint lacks a degree-matched substitute.

- [ ] **Step 1: Add the reproduced empty-endpoint-stratum regression**

```python
def test_tie_is_skipped_when_either_endpoint_has_no_degree_matched_substitute() -> None:
    mentions = {f"c{i}": ["a-u"] for i in range(20)}
    for author, cited_by in {
        "a-v": range(3),
        "a-x": range(3, 6),
        "a-y": range(6, 9),
    }.items():
        for i in cited_by:
            mentions[f"c{i}"].append(author)
    graph = _graph_from_mentions(mentions)

    result = tie_cocitation_test(graph, [("a-u", "a-v")])

    assert result.ties == ()
    assert len(result.skipped) == 1
    assert "no degree-matched substitute" in result.skipped[0].reason
    assert "a-u" in result.skipped[0].reason
```

- [ ] **Step 2: Run the regression and confirm the false significance**

Run: `.venv/bin/python -m pytest tests/test_cocitation_stats.py::test_tie_is_skipped_when_either_endpoint_has_no_degree_matched_substitute -v`

Expected: FAIL because the current implementation scores the tie with `pool_size == 0`.

- [ ] **Step 3: Require both pools and remove cross-pool fallback sampling**

Immediately after building the pools:

```python
empty_ends = [
    author
    for author, pool in ((source, source_pool), (target, target_pool))
    if not pool
]
if empty_ends:
    skipped.append(
        SkippedTie(
            source,
            target,
            "no degree-matched substitute for: " + ", ".join(empty_ends),
        )
    )
    continue
```

Keep the existing union-size distinct-pair guard after this check. In the draw loop, replace the fallbacks with:

```python
first = rng.choice(source_pool)
second = rng.choice(target_pool)
```

- [ ] **Step 4: Run all co-citation tests again**

Run: `.venv/bin/python -m pytest tests/test_cocitation_stats.py -v`

Expected: all co-citation tests pass, including existing self-citation, determinism, and skip-reason coverage.

---

### Task 4: Reject Workbook Sheet Collisions Before Output Mutation

**Files:**
- Modify: `tests/test_author_export.py` before the optional-artifact tests
- Modify: `src/citegraph/author_export.py:21-80`

**Interfaces:**
- Consumes: `tuple[ExtraSheet, ...]` passed to `build_core_author_workbook`.
- Produces: `ValueError` before any data filtering or output-file opening when names collide case-insensitively with generated or extra sheets.

- [ ] **Step 1: Add regressions for reserved and duplicate names**

```python
@pytest.mark.parametrize("name", ["README", "Authors", "Works", "Review_Flags", "authors"])
def test_reserved_extra_sheet_name_is_rejected_before_output_changes(
    tmp_path: Path, name: str
) -> None:
    out = _make_out_dir(tmp_path)
    dest = tmp_path / "existing.xlsx"
    dest.write_bytes(b"sentinel")
    extra = ExtraSheet(name=name, frame=pd.DataFrame([{"author_id": "a-one"}]))

    with pytest.raises(ValueError, match="conflicts with workbook sheet"):
        build_core_author_workbook(out, dest, extra_sheets=(extra,))

    assert dest.read_bytes() == b"sentinel"


def test_extra_sheet_names_must_be_unique_case_insensitively(tmp_path: Path) -> None:
    first = ExtraSheet(name="Careers", frame=pd.DataFrame([{"author_id": "a-one"}]))
    second = ExtraSheet(name="careers", frame=pd.DataFrame([{"author_id": "a-two"}]))

    with pytest.raises(ValueError, match="conflicts with workbook sheet"):
        build_core_author_workbook(_make_out_dir(tmp_path), extra_sheets=(first, second))
```

- [ ] **Step 2: Run the two regressions and confirm current overwrite behavior**

Run: `.venv/bin/python -m pytest tests/test_author_export.py -k 'reserved_extra or unique_case' -v`

Expected: failures because no shared library validation exists.

- [ ] **Step 3: Add one small shared validator and call it first**

```python
_GENERATED_SHEETS = ("README", "Authors", "Works", "Review_Flags")


def _validate_extra_sheet_names(extra_sheets: tuple[ExtraSheet, ...]) -> None:
    used = {name.casefold(): name for name in _GENERATED_SHEETS}
    for sheet in extra_sheets:
        folded = sheet.name.casefold()
        if folded in used:
            raise ValueError(
                f"Extra sheet {sheet.name!r} conflicts with workbook sheet {used[folded]!r}"
            )
        used[folded] = sheet.name
```

Make it the first statement in `build_core_author_workbook`, before `_require_openpyxl()`:

```python
_validate_extra_sheet_names(extra_sheets)
_require_openpyxl()
```

- [ ] **Step 4: Run the author-export tests**

Run: `.venv/bin/python -m pytest tests/test_author_export.py -v`

Expected: all author-export library and CLI tests pass.

---

### Task 5: Align Research Documentation and Run the Release Gate

**Files:**
- Modify: `CLAUDE.md:212-249`
- Modify: `docs/superpowers/specs/2026-09-15-supervision-cocitation-figures-design.md:37-78`
- Verify: `README.md:23-35` (no text change required because the public extra names stay the same)

**Interfaces:**
- Consumes: the final public behavior from Tasks 1-4.
- Produces: documentation with no invalid binomial claim and explicit unmatched-stratum skipping.

- [ ] **Step 1: Rewrite the co-citation guidance**

In both documents:

- say the API returns per-tie results plus an assumption-bound Fisher diagnostic;
- state that overlapping ties have no confirmatory combined p-value;
- describe `n_significant` and `expected_significant` as descriptive only;
- state that either endpoint lacking a matched stratum causes a skipped tie;
- remove the reported binomial p-values and every `binomial_tail_p` reference.

Preserve the observed corpus counts (`9 of 16`, `7 of 16`, and `0.8 expected`) but explicitly label them descriptive.

- [ ] **Step 2: Prove the removed claim is absent**

Run:

```bash
rg -n "binomial_p_value|binomial_tail_p|binomial p" src tests CLAUDE.md docs README.md
```

Expected: no matches.

- [ ] **Step 3: Run focused verification**

Run:

```bash
.venv/bin/python -m pytest \
  tests/test_packaging.py \
  tests/test_cocitation_stats.py \
  tests/test_author_export.py -v
```

Expected: all focused tests pass with no skips.

- [ ] **Step 4: Run the complete test and lint gates**

Run:

```bash
.venv/bin/python -m pytest -o addopts=''
.venv/bin/ruff check .
git diff --check
```

Expected: 708 or more tests pass, no skips, Ruff reports `All checks passed!`, and `git diff --check` emits nothing.

- [ ] **Step 5: Build and validate clean distributions outside the repository `dist/`**

Run:

```bash
release_dir=$(mktemp -d /private/tmp/citegraph-release-hardening.XXXXXX)
UV_CACHE_DIR=/private/tmp/citegraph-release-hardening-uv-cache \
  uv build --out-dir "$release_dir/dist"
UV_CACHE_DIR=/private/tmp/citegraph-release-hardening-uv-cache \
  uv tool run --from twine twine check --strict "$release_dir"/dist/*
```

Expected: one wheel and one sdist build successfully and both pass strict Twine checks.

- [ ] **Step 6: Review the final diff without committing unrelated work**

Run:

```bash
git diff -- \
  pyproject.toml \
  src/citegraph/cocitation_stats.py \
  src/citegraph/author_export.py \
  tests/test_packaging.py \
  tests/test_cocitation_stats.py \
  tests/test_author_export.py \
  CLAUDE.md \
  docs/superpowers/specs/2026-09-15-supervision-cocitation-figures-design.md
```

Expected: only the four approved behavior changes and their tests/documentation appear in the relevant hunks; pre-existing changes elsewhere remain untouched.
