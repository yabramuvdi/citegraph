"""Regression tests for canonicalization pipeline integrity (offline)."""
import json

import pandas as pd
import pytest

from citegraph.pipeline import Pipeline, StageNotReadyError
from tests.test_pipeline import _FakeClient


def setup_cached(tmp_path, title="Study", names=None):
    p = Pipeline(pdf_dir=None, out_dir=tmp_path, client=_FakeClient(), show_progress=False)
    md = p.layout.markdown_dir / "paper.md"
    md.write_text("A paper", encoding="utf-8")
    (p.layout.metadata_dir / "paper.json").write_text(json.dumps(dict(
        Title=title, Authors_List=names if names is not None else ["John Smith"],
        Journal="Journal", Year=2020)))
    (p.layout.references_dir / "paper.json").write_text("[]")
    return p, md


def test_zero_references_checkpoint_can_resume(tmp_path):
    p, md = setup_cached(tmp_path)
    sources = p.extract_paper_metadata([md])
    p.extract_paper_references([md], sources)
    works, edges = p.deduplicate()
    assert len(works) == 1
    assert edges.empty
    assert list(edges.columns) == ["citing_id", "cited_id"]


def test_legacy_blank_reference_checkpoint_can_resume(tmp_path):
    p, md = setup_cached(tmp_path)
    p.extract_paper_metadata([md])
    p.layout.citations_raw_csv.write_text("\n")
    works, edges = p.deduplicate()
    assert len(works) == 1
    assert edges.empty


def test_populated_malformed_checkpoint_is_not_empty(tmp_path):
    p, md = setup_cached(tmp_path)
    p.extract_paper_metadata([md])
    p.layout.citations_raw_csv.write_text("wrong\nvalue\n")
    with pytest.raises(ValueError, match="missing required column"):
        p.deduplicate()


def test_all_failed_metadata_preserves_failure_then_blocks_references(tmp_path):
    p, md = setup_cached(tmp_path)
    (p.layout.metadata_dir / "paper.json").write_text("broken")
    sources = p.extract_paper_metadata([md])
    assert sources.empty
    assert p.layout.metadata_failures_jsonl.exists()
    assert list(pd.read_csv(p.layout.sources_csv).columns) == [
        "id", "source_file", "Title", "Authors", "Authors_List", "Journal", "Year"]
    with pytest.raises(StageNotReadyError, match="metadata"):
        p.extract_paper_references([md], sources)


def test_source_slug_collision_retains_both_bibliographies(tmp_path):
    p, md = setup_cached(tmp_path)
    titles = [
        "Studies in economic and social behavior: agricultural irrigation water allocation and collective farming",
        "Studies in economic and social behavior: monetary inflation central banking interest rates and currency speculation",
    ]
    paths = []
    for i, title in enumerate(titles):
        path = p.layout.markdown_dir / f"collision{i}.md"
        path.write_text("Paper")
        paths.append(path)
        (p.layout.metadata_dir / f"collision{i}.json").write_text(json.dumps(dict(
            Title=title, Authors_List=["John Smith"], Journal="Journal", Year=2020)))
        (p.layout.references_dir / f"collision{i}.json").write_text(json.dumps([dict(
            Title=f"Unique reference {i}", Authors_List=[f"Author{i}, A."], Journal="J", Year=1990)]))
    sources = p.extract_paper_metadata(paths)
    assert len(sources) == 2
    assert sources.id.nunique() == 2
    refs = p.extract_paper_references(paths, sources)
    assert set(refs.Title) == {"Unique reference 0", "Unique reference 1"}
    original = sources.set_index("source_file").id.to_dict()
    reversed_sources = p.extract_paper_metadata(paths[::-1])
    assert reversed_sources.set_index("source_file").id.to_dict() == original
    p.extract_paper_metadata(paths[:1])
    restored = p.extract_paper_metadata(paths)
    assert restored.set_index("source_file").id.to_dict() == original
    works, edges = p.deduplicate(restored, refs)
    assert set(edges.citing_id) <= set(works.index)
    assert set(edges.cited_id) <= set(works.index)


def test_identical_source_files_retain_complementary_references(tmp_path):
    p, md = setup_cached(tmp_path)
    copy = p.layout.markdown_dir / "copy.md"
    copy.write_text("same paper")
    (p.layout.metadata_dir / "copy.json").write_text((p.layout.metadata_dir / "paper.json").read_text())
    (p.layout.references_dir / "copy.json").write_text(json.dumps([dict(
        Title="Only in better conversion", Authors_List=["Mary Jones"], Journal="Other", Year=2005)]))
    sources = p.extract_paper_metadata([md, copy])
    assert len(sources) == 2
    refs = p.extract_paper_references([md, copy], sources)
    assert len(refs) == 1
    works, edges = p.deduplicate(sources, refs)
    assert len(works[works.ring == 0]) == 1
    assert len(edges) == 1


def test_empty_provider_fields_do_not_change_warm_result(tmp_path, monkeypatch):
    from citegraph.enrich import EnrichConfig, _enrich_one, _LookupReport
    row = pd.Series(dict(Title="Study", Authors="John Smith", Authors_List=["John Smith"],
                         Journal="Known Journal", Year=2020))
    match = dict(doi="10.example/test", enrichment_source="crossref", Title="Study",
                 Authors="", Authors_List=[], Journal="", Year=None, OpenAlex_Authors=[])
    monkeypatch.setattr("citegraph.enrich._crossref_lookup_report",
                        lambda *args, **kwargs: _LookupReport(match=match))
    cold = _enrich_one("w-example", row, EnrichConfig(), None, tmp_path)
    warm = _enrich_one("w-example", row, EnrichConfig(), None, tmp_path)
    for key in ("Authors", "Authors_List", "Journal", "Year"):
        assert cold[key] == warm[key] == row[key]


def test_changed_input_does_not_reuse_enrichment_identity(tmp_path, monkeypatch):
    from citegraph.enrich import EnrichConfig, _enrich_one, _LookupReport
    row = pd.Series(dict(Title="Study", Authors="John Smith", Authors_List=["John Smith"], Year=2020))
    def lookup(title, *args):
        return _LookupReport(match=dict(doi=f"10.example/{title}", enrichment_source="crossref", Title=title))
    monkeypatch.setattr("citegraph.enrich._crossref_lookup_report", lookup)
    _enrich_one("w-example", row, EnrichConfig(), None, tmp_path)
    row["Title"] = "Changed"
    result = _enrich_one("w-example", row, EnrichConfig(), None, tmp_path)
    assert result["doi"] == "10.example/Changed"


def test_composed_and_resumed_authors_use_extracted_population(tmp_path, monkeypatch):
    p, md = setup_cached(tmp_path)
    p.enrich = True
    monkeypatch.setattr(p, "convert_pdfs", lambda: [md])
    def fake_enrich(works, **kwargs):
        out = works.copy()
        out["Authors_List"] = [["John Smith", "Extra Author"]]
        out["Authors"] = "John Smith, Extra Author"
        out["OpenAlex_Authors"] = [[dict(display_name="John Smith", openalex_id="A1"),
                                     dict(display_name="Extra Author", openalex_id="A2")]]
        return out
    monkeypatch.setattr("citegraph.enrich.enrich_works", fake_enrich)
    result = p.run()
    resumed, occurrences = p.normalize_authors()
    assert set(result.author_citations.raw_author) == set(occurrences.raw_author) == {"John Smith"}
    assert set(result.authors.index) == set(resumed.index)
    assert resumed.iloc[0].openalex_id == "A1"


def test_tampered_enriched_table_cannot_reassign_identity(tmp_path, monkeypatch):
    p, md = setup_cached(tmp_path)
    sources = p.extract_paper_metadata([md])
    refs = p.extract_paper_references([md], sources)
    works, edges = p.deduplicate(sources, refs)
    p.enrich = True
    def fake_enrich(works, **kwargs):
        out = works.copy()
        out["OpenAlex_Authors"] = [[dict(display_name="John Smith", openalex_id="A1")]]
        return out
    monkeypatch.setattr("citegraph.enrich.enrich_works", fake_enrich)
    p.maybe_enrich(works)
    text = p.layout.enriched_works_csv.read_text().replace("A1", "A_WRONG")
    p.layout.enriched_works_csv.write_text(text)
    authors, _ = p.normalize_authors()
    assert pd.isna(authors.iloc[0].openalex_id)


def test_author_positions_survive_csv_roundtrip():
    from citegraph.authors import _row_authors
    row = pd.Series({"Authors_List": ["", "John Smith", "Jane Smith"]})
    csv_row = pd.read_csv(__import__('io').StringIO(pd.DataFrame([row]).to_csv(index=False))).iloc[0]
    assert _row_authors(row) == _row_authors(csv_row) == ["", "John Smith", "Jane Smith"]


def test_failed_run_removes_previous_success_summary(tmp_path, monkeypatch):
    p, md = setup_cached(tmp_path)
    p.layout.run_summary_json.write_text('{"n_works": 99}')
    (p.layout.metadata_dir / "paper.json").write_text("broken")
    monkeypatch.setattr(p, "convert_pdfs", lambda: [md])
    with pytest.raises(StageNotReadyError):
        p.run()
    assert not p.layout.run_summary_json.exists()


def test_pipeline_corrections_and_failure_leave_saved_outputs_unchanged(tmp_path):
    from tests.test_author_overrides import correction, works
    p, _ = setup_cached(tmp_path)
    frame = works(["Smith, John", "Smith, John", "Jones, Jane"]).rename_axis("id")
    frame.to_csv(p.layout.works_csv)
    rows = [correction("separate", "w-1", "w-2"), correction("merge", "w-1", "w-3")]
    pd.DataFrame(rows).to_csv(p.layout.author_overrides_csv, index=False)
    _, occurrences = p.normalize_authors(works=frame)
    ids = occurrences.set_index("record_id").author_id
    assert ids["w-1"] == ids["w-3"] != ids["w-2"]
    assert p.layout.author_overrides_meta_json.exists()
    artifacts = [p.layout.authors_csv, p.layout.author_citations_csv,
                 p.layout.author_resolution_audit_json, p.layout.author_overrides_meta_json]
    saved = {path: path.read_bytes() for path in artifacts}
    rows.append(correction("merge", "w-1", "w-2"))
    pd.DataFrame(rows).to_csv(p.layout.author_overrides_csv, index=False)
    with pytest.raises(ValueError, match="contradiction"):
        p.normalize_authors(works=frame)
    assert {path: path.read_bytes() for path in artifacts} == saved


def test_first_invalid_correction_does_not_bind_snapshot(tmp_path):
    from tests.test_author_overrides import correction, works
    p, _ = setup_cached(tmp_path)
    frame = works(["Smith, John", "Smith, John"])
    pd.DataFrame([correction("merge", "w-1", "w-2"),
                  correction("separate", "w-1", "w-2")]).to_csv(p.layout.author_overrides_csv, index=False)
    with pytest.raises(ValueError, match="contradiction"):
        p.normalize_authors(works=frame)
    assert not p.layout.author_overrides_meta_json.exists()
    assert not p.layout.authors_csv.exists()


def test_graph_accepts_legacy_blank_author_checkpoints(tmp_path):
    from citegraph.graph import CitationGraph
    p, md = setup_cached(tmp_path, names=[])
    sources = p.extract_paper_metadata([md])
    refs = p.extract_paper_references([md], sources)
    p.deduplicate(sources, refs)
    p.layout.authors_csv.write_text("\n")
    p.layout.author_citations_csv.write_text("\n")
    graph = CitationGraph.from_out_dir(tmp_path)
    assert graph.authors.empty and graph.author_citations.empty
    assert graph.n_works == 1


def test_cited_collision_history_blocks_legacy_cache_and_identifiers(tmp_path, monkeypatch):
    from citegraph.enrich import EnrichConfig, _enrich_one, _LookupReport
    p, md = setup_cached(tmp_path)
    sources = p.extract_paper_metadata([md])
    prefix = "Studies in economic and social behavior: "
    titles = [prefix + "agricultural irrigation water allocation and collective farming",
              prefix + "monetary inflation central banking interest rates and currency speculation"]
    refs = pd.DataFrame([dict(Title=title, Authors="John Smith", Authors_List=["John Smith"],
                              Journal="J", Year=2020, citing_id=sources.iloc[0].id) for title in titles])
    refs.to_csv(p.layout.citations_raw_csv, index=False)
    works, _ = p.deduplicate(sources, refs)
    base = works.index[works.Title == titles[0]][0]
    legacy = dict(Title=titles[1], doi="10.old/wrong", Authors_List=["John Smith"],
                  OpenAlex_Authors=[dict(display_name="John Smith", openalex_id="WRONG")])
    cache_path = p.layout.enrichment_dir / f"{base}.json"
    cache_path.write_text(json.dumps(legacy))
    calls = []
    def lookup(*args, **kwargs):
        calls.append(args[0])
        return _LookupReport(match=dict(Title=titles[0], doi="10.new/correct"))
    monkeypatch.setattr("citegraph.enrich._crossref_lookup_report", lookup)
    result = _enrich_one(base, works.loc[base], EnrichConfig(), None, p.layout.enrichment_dir)
    assert result["doi"] == "10.new/correct"
    assert calls == [titles[0]]
    # Removing the other collider must not rehabilitate the old cache/CSV.
    works, _ = p.deduplicate(sources, refs.iloc[:1])
    pd.DataFrame([dict(id=base, **legacy)]).set_index("id").to_csv(p.layout.enriched_works_csv)
    authors, _ = p.normalize_authors(works=works)
    assert "WRONG" not in set(authors.openalex_id.dropna())
    cache_path.write_text(json.dumps(legacy))
    _enrich_one(base, works.loc[base], EnrichConfig(), None, p.layout.enrichment_dir)
    assert len(calls) == 2
