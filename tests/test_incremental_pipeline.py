"""Incremental runs preserve verified evidence and reviewed identity offline."""
import json

import pandas as pd
import pytest

from citegraph.pipeline import Pipeline
from tests.test_author_overrides import correction
from tests.test_pipeline import _FakeClient


def pipeline(tmp_path):
    return Pipeline(pdf_dir=None, out_dir=tmp_path, client=_FakeClient(), show_progress=False)


def works(names):
    return pd.DataFrame([
        dict(id=f'w-{i}', Title=f'Distinct study {i}', Authors_List=[name],
             Authors=name, Journal='J', Year=2000 + i, ring=0, source_file=f'{i}.md')
        for i, name in enumerate(names)
    ]).set_index('id')


def test_added_work_keeps_verified_enrichment_for_existing_author(tmp_path, monkeypatch):
    p = pipeline(tmp_path)
    original = works(['John Smith'])
    p.enrich = True

    def enrich(frame, **kwargs):
        return frame.assign(OpenAlex_Authors=[[dict(display_name='John Smith', openalex_id='A1')]])

    monkeypatch.setattr('citegraph.enrich.enrich_works', enrich)
    p.maybe_enrich(original)
    authors, _ = p.normalize_authors(works=works(['John Smith', 'Jane Jones']))
    assert authors.loc['a-smith-john', 'openalex_id'] == 'A1'
    assert pd.isna(authors.loc['a-jones-jane', 'openalex_id'])


def test_tampered_enrichment_row_does_not_discard_valid_sibling(tmp_path, monkeypatch):
    p = pipeline(tmp_path)
    frame = works(['John Smith', 'Jane Jones'])
    p.enrich = True

    def enrich(frame, **kwargs):
        result = frame.copy()
        result['OpenAlex_Authors'] = [[dict(display_name=n, openalex_id=a)]
                                     for n, a in [('John Smith', 'A1'), ('Jane Jones', 'A2')]]
        return result

    monkeypatch.setattr('citegraph.enrich.enrich_works', enrich)
    p.maybe_enrich(frame)
    path = p.layout.enriched_works_csv
    path.write_text(path.read_text().replace('A1', 'WRONG'))
    authors, _ = p.normalize_authors(works=frame)
    assert pd.isna(authors.loc['a-smith-john', 'openalex_id'])
    assert authors.loc['a-jones-jane', 'openalex_id'] == 'A2'


def test_fuller_name_updates_display_without_renaming_author(tmp_path):
    p = pipeline(tmp_path)
    first, _ = p.normalize_authors(works=works(['Smith, J.']))
    second, edges = p.normalize_authors(works=works(['Smith, J.', 'John Smith']))
    assert list(second.index) == list(first.index)
    assert second.iloc[0].display_name == 'John Smith'
    assert edges.author_id.nunique() == 1


def test_alias_survives_fuller_name_after_binding(tmp_path):
    p = pipeline(tmp_path)
    frame = works(['Smith, J.', 'Jane Jones'])
    p.layout.author_aliases_csv.write_text('cluster_id,canonical_id\na-smith-j,a-jones-jane\n')
    first, _ = p.normalize_authors(works=frame)
    second, edges = p.normalize_authors(works=works(['Smith, J.', 'Jane Jones', 'John Smith']))
    assert list(first.index) == list(second.index) == ['a-jones-jane']
    assert edges.author_id.nunique() == 1


def test_unrelated_addition_keeps_occurrence_corrections(tmp_path):
    p = pipeline(tmp_path)
    frame = works(['John Smith', 'John Smith'])
    pd.DataFrame([correction('separate', 'w-0', 'w-1')]).to_csv(p.layout.author_overrides_csv, index=False)
    first, _ = p.normalize_authors(works=frame)
    second, _ = p.normalize_authors(works=works(['John Smith', 'John Smith', 'Jane Jones']))
    assert set(first.index) <= set(second.index)
    before = p.layout.authors_csv.read_bytes()
    changed = works(['Jane Smith', 'John Smith', 'Jane Jones'])
    with pytest.raises(ValueError, match='snapshot|occurrence'):
        p.normalize_authors(works=changed)
    assert p.layout.authors_csv.read_bytes() == before


def test_source_promotion_keeps_previous_work_identity(tmp_path):
    p = pipeline(tmp_path)
    source = works(['Mary Jones']).reset_index().drop(columns='ring')
    cited = dict(Title='Collective irrigation behavior', Authors='Smith, J.',
                 Authors_List=['Smith, J.'], Journal='J', Year=1995, citing_id='w-0')
    refs = pd.DataFrame([cited])
    first, _ = p.deduplicate(source, refs)
    old_id = first.index[first.ring == 1][0]
    new_source = dict(id='w-smith-source', source_file='new.md', **{
        key: value for key, value in cited.items() if key != 'citing_id'})
    second, graph = p.deduplicate(pd.concat([source, pd.DataFrame([new_source])]), refs)
    assert old_id in second.index
    assert second.loc[old_id, 'ring'] == 0
    assert second.loc[old_id, 'source_file'] == 'new.md'
    assert graph.iloc[0].cited_id == old_id
    audit = json.loads(p.layout.canonicalization_audit_json.read_text())
    assert audit['citation_cluster_ids'] == [old_id]


def test_legacy_aliases_are_bound_before_new_works_replace_snapshot(tmp_path):
    p = pipeline(tmp_path)
    frame = works(['Smith, J.', 'Jane Jones'])
    frame.to_csv(p.layout.works_csv)
    p.layout.author_aliases_csv.write_text('cluster_id,canonical_id\na-smith-j,a-jones-jane\n')
    p.normalize_authors(works=frame)
    p.layout.author_identity_json.unlink()  # prior library wrote only CSVs/audit
    added = works(['Smith, J.', 'Jane Jones', 'John Smith'])
    refs = pd.DataFrame(columns=['Title', 'Authors_List', 'Year', 'citing_id'])
    p.deduplicate(added.reset_index().drop(columns='ring'), refs)
    authors, citations = p.normalize_authors()
    assert list(authors.index) == ['a-jones-jane']
    assert citations.author_id.nunique() == 1


def test_extraction_recipe_change_refreshes_and_estimate_agrees(tmp_path):
    from tests.test_pipeline import _FakeClient

    class CountingClient(_FakeClient):
        calls = 0

        def generate_structured(self, **kwargs):
            self.calls += 1
            return super().generate_structured(**kwargs)

    client = CountingClient()
    p = Pipeline(pdf_dir=None, out_dir=tmp_path, client=client, show_progress=False)
    md = p.layout.markdown_dir / 'paper.md'
    md.write_text('A scientific paper')
    p.extract_paper_metadata()
    assert client.calls == 1
    client.model = 'different-model'
    assert p.estimate_extraction_cost().n_metadata_to_process == 1
    p.extract_paper_metadata()
    assert client.calls == 2


def test_add_paper_and_resume_reuses_all_completed_expensive_work(tmp_path, monkeypatch):
    from citegraph.enrich import _LookupReport
    from tests.test_incremental_cache import _install_fake_docling
    from tests.test_pipeline import _FakeResponse

    conversions = []
    _install_fake_docling(monkeypatch, conversions)

    class CountingClient(_FakeClient):
        calls = 0

        def generate_structured(self, **kwargs):
            self.calls += 1
            response = super().generate_structured(**kwargs)
            if not isinstance(response.parsed, list) and 'ADDED_PAPER' in kwargs['prompt']:
                response = _FakeResponse(response.parsed.model_copy(update={
                    'Title': 'A different experiment on marine ecosystems'}))
            return response

    lookups = []
    interrupted = False

    def lookup(title, *args):
        lookups.append(title)
        if interrupted and title == 'A different experiment on marine ecosystems':
            raise RuntimeError('interrupted provider operation')
        return _LookupReport(miss_reason='no_crossref_candidates')

    monkeypatch.setattr('citegraph.enrich._crossref_lookup_report', lookup)
    monkeypatch.setattr('citegraph.enrich._openalex_lookup_report',
                        lambda *args: _LookupReport(miss_reason='no_openalex_candidates'))
    pdfs = tmp_path / 'pdfs'
    pdfs.mkdir()
    (pdfs / 'first.pdf').write_text('FIRST_PAPER')
    client = CountingClient()
    p = Pipeline(pdf_dir=pdfs, out_dir=tmp_path / 'out', client=client,
                 enrich=True, show_progress=False)
    first = p.run()
    assert client.calls == 2
    assert len(lookups) == 3  # one source and two cited works
    (pdfs / 'added.pdf').write_text('ADDED_PAPER')
    interrupted = True
    with pytest.raises(RuntimeError, match='interrupted'):
        p.run()
    assert client.calls == 4
    assert len(conversions) == 2
    interrupted = False
    result = p.run()
    assert client.calls == 4
    assert len(conversions) == 2
    assert len(lookups) == 5  # only the failed new work is retried
    assert (result.works.ring == 0).sum() == 2
    assert set(first.authors.index) <= set(result.authors.index)
    # Renaming a PDF also reuses extraction; source identity stays the same.
    (pdfs / 'first.pdf').rename(pdfs / 'renamed.pdf')
    renamed = p.run()
    assert client.calls == 4
    assert len(conversions) == 2
    assert len(lookups) == 5
    assert set(renamed.works.index) == set(result.works.index)


def test_split_retires_annotation_key_and_new_ids_accept_aliases(tmp_path):
    p = pipeline(tmp_path)
    frame = works(['John Smith', 'John Smith'])
    original, _ = p.normalize_authors(works=frame)
    pd.DataFrame([correction('separate', 'w-0', 'w-1')]).to_csv(p.layout.author_overrides_csv, index=False)
    split, edges = p.normalize_authors(works=frame)
    assert not set(original.index) & set(split.index)
    assert len(split) == 2
    repeated, _ = p.normalize_authors(works=frame)
    assert set(repeated.index) == set(split.index)
    p.layout.author_overrides_csv.unlink()
    source, target = edges.author_id.tolist()
    p.layout.author_aliases_csv.write_text(f'cluster_id,canonical_id\n{source},{target}\n')
    merged, _ = p.normalize_authors(works=frame)
    assert len(merged) == 1
    repeated, _ = p.normalize_authors(works=frame)
    assert list(repeated.index) == list(merged.index)


def test_self_alias_is_rejected_even_with_persistent_identities(tmp_path):
    p = pipeline(tmp_path)
    p.layout.author_aliases_csv.write_text('cluster_id,canonical_id\na-smith-john,a-smith-john\n')
    with pytest.raises(ValueError, match='cycle'):
        p.normalize_authors(works=works(['John Smith']))


def test_new_alias_can_use_published_id_after_undoing_an_old_alias(tmp_path):
    p = pipeline(tmp_path)
    frame = works(['John Smith', 'Jane Jones'])
    path = p.layout.author_aliases_csv
    path.write_text('cluster_id,canonical_id\na-smith-john,a-jones-jane\n')
    p.normalize_authors(works=frame)
    path.unlink()
    authors, _ = p.normalize_authors(works=frame)
    by_name = authors.reset_index().set_index('display_name')['id']
    path.write_text('cluster_id,canonical_id\n'
                    f'{by_name["John Smith"]},{by_name["Jane Jones"]}\n')
    merged, _ = p.normalize_authors(works=frame)
    assert len(merged) == 1


def test_legacy_assignment_tables_can_seed_ids_without_historical_audit(tmp_path):
    p = pipeline(tmp_path)
    frame = works(['Smith, J.'])
    frame.to_csv(p.layout.works_csv)
    original, _ = p.normalize_authors(works=frame)
    p.layout.author_identity_json.unlink()
    p.layout.author_resolution_audit_json.unlink()
    added = works(['Smith, J.', 'John Smith'])
    added.loc['w-1', 'Title'] = 'A different experiment on marine ecosystems'
    p.deduplicate(added.reset_index().drop(columns='ring'),
                  pd.DataFrame(columns=['Title', 'Authors_List', 'Year', 'citing_id']))
    authors, _ = p.normalize_authors()
    assert list(authors.index) == list(original.index)


def test_progressive_run_uses_current_pdf_selection_after_rename(tmp_path, monkeypatch):
    from tests.test_incremental_cache import _install_fake_docling

    calls = []
    _install_fake_docling(monkeypatch, calls)
    pdfs = tmp_path / 'pdfs'
    pdfs.mkdir()
    (pdfs / 'old.pdf').write_text('Paper')
    p = Pipeline(pdf_dir=pdfs, out_dir=tmp_path / 'out', client=_FakeClient(), show_progress=False)
    p.convert_pdfs()
    p.extract_paper_metadata()
    (pdfs / 'old.pdf').rename(pdfs / 'new.pdf')
    p.convert_pdfs()
    sources = p.extract_paper_metadata()
    assert sources.source_file.tolist() == ['new.md']
    assert p.estimate_extraction_cost().n_files == 1
    assert len(calls) == 1


def test_alias_preserves_explicit_target_even_when_its_public_id_was_remapped(tmp_path):
    p = pipeline(tmp_path)
    frame = works(['Jane Adams', 'John Zebra'])
    path = p.layout.author_aliases_csv
    path.write_text('cluster_id,canonical_id\na-adams-jane,a-zebra-john\n')
    p.normalize_authors(works=frame)
    path.unlink()
    split, _ = p.normalize_authors(works=frame)
    ids = split.reset_index().set_index('display_name')['id']
    target = ids['John Zebra']
    path.write_text(f'cluster_id,canonical_id\n{ids["Jane Adams"]},{target}\n')
    merged, _ = p.normalize_authors(works=frame)
    assert list(merged.index) == [target]


def test_fresh_enrichment_survives_csv_float_roundtrip(tmp_path, monkeypatch):
    p = pipeline(tmp_path)
    frame = works(['John Smith'])
    p.enrich = True

    def enrich(frame, **kwargs):
        return frame.assign(
            OpenAlex_Authors=[[dict(display_name='John Smith', openalex_id='A1')]],
            enrichment_title_score=90.21739130434783)

    monkeypatch.setattr('citegraph.enrich.enrich_works', enrich)
    p.maybe_enrich(frame)
    authors, _ = p.normalize_authors(works=frame)
    assert authors.loc['a-smith-john', 'openalex_id'] == 'A1'
