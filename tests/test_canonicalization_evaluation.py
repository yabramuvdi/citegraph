"""Offline evaluation must distinguish regression labels from human evidence."""
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / 'scripts/evaluate_canonicalization.py'


def evaluator():
    assert SCRIPT.exists(), 'offline evaluator has not been implemented'
    spec = importlib.util.spec_from_file_location('evaluation', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_hand_checked_metrics_and_ambiguity():
    m = evaluator().evaluate_pairs([
        dict(left='a', right='b', label='same'),
        dict(left='a', right='c', label='different'),
        dict(left='b', right='d', label='same'),
        dict(left='c', right='d', label='different'),
        dict(left='a', right='d', label='ambiguous'),
    ], dict(a='x', b='x', c='x', d='y'))
    assert [m[k] for k in ('tp', 'fp', 'fn', 'tn')] == [1, 1, 1, 1]
    assert m['precision'] == m['recall'] == 0.5
    assert m['ambiguous_count'] == 1


def test_empty_metrics_are_unknown_and_missing_observations_fail():
    e = evaluator()
    assert e.evaluate_pairs([], {})['precision'] is None
    assert e.evaluate_pairs([], {})['recall'] is None
    with pytest.raises(ValueError, match='missing'):
        e.evaluate_pairs([dict(left='a', right='b', label='same')], {'a': 'x'})


def test_partition_ignores_cluster_names():
    assert evaluator().partition({'a': 'x', 'b': 'x'}) == evaluator().partition({'b': 'z', 'a': 'z'})


def test_regression_fixtures():
    report = evaluator().evaluate_fixtures(Path(__file__).parent / 'fixtures/canonicalization')
    for kind in ('works', 'authors'):
        assert report[kind]['metrics']['fp'] == 0
        assert report[kind]['metrics']['fn'] == 0
        assert all(c['repeat_equal'] and c['csv_roundtrip_equal'] for c in report[kind]['cases'])
    assert all(c['reversed_partition_equal'] for c in report['authors']['cases'])
    sensitive = next(c for c in report['works']['cases'] if c['case_id'] == 'representative_order_sensitivity')
    assert not sensitive['reversed_partition_equal']  # Known limitation is measured, not hidden.


def test_sampling_is_unlabeled_deterministic_and_frozen(tmp_path):
    import pandas as pd
    e = evaluator()
    pd.DataFrame([dict(id=f'w-{i}', Title=f'Policy study {i}', Year=2020,
                       Authors_List='["J. Smith"]', ring=0, source_file=f'{i}.pdf')
                  for i in range(170)]).to_csv(tmp_path / 'works.csv', index=False)
    first = e.sample_corpus(tmp_path)
    assert first == e.sample_corpus(tmp_path)
    for kind in ('works', 'authors'):
        assert len(first[kind]) == 150
        assert all(g['label'] is None for g in first[kind])
        assert {g['split'] for g in first[kind]} == {'development', 'held_out'}
    output = tmp_path / 'review.json'
    e.write_sample(tmp_path, output)
    with pytest.raises(FileExistsError):
        e.write_sample(tmp_path, output)


def test_sample_contains_similar_title_and_identifier_conflict_context(tmp_path):
    import json

    import pandas as pd
    pd.DataFrame([
        dict(id='w-a', Title='Policy and employment', Authors_List='["John Smith"]'),
        dict(id='w-b', Title='Policy and unemployment', Authors_List='["Jane Smith"]'),
    ]).to_csv(tmp_path / 'works.csv', index=False)
    (tmp_path / 'canonicalization_audit.json').write_text(json.dumps({
        'id_collisions': [{'base_id': 'w-a', 'assigned_id': 'w-b'}]}))
    pd.DataFrame([
        dict(id='w-a', OpenAlex_Authors=json.dumps([
            dict(display_name='John Smith', openalex_id='A1', orcid='0000-0001')])),
        dict(id='w-b', OpenAlex_Authors=json.dumps([
            dict(display_name='Jane Smith', openalex_id='A1', orcid='0000-0002')])),
    ]).to_csv(tmp_path / 'enriched_works.csv', index=False)
    result = evaluator().sample_corpus(tmp_path)
    assert any('id_collision' in g['strata'] for g in result['works'])
    assert any('similar_titles' in g['strata'] for g in result['works'])
    assert any('identifier_conflict' in g['strata'] for g in result['authors'])
    assert result['authors'][0]['enrichment_context']


def test_sampling_preserves_empty_author_coordinates(tmp_path):
    import pandas as pd
    pd.DataFrame([dict(id='w-1', Title='Study', Authors_List=['', 'Jane Smith'])]).to_csv(
        tmp_path / 'works.csv', index=False)
    sample = evaluator().sample_corpus(tmp_path)
    assert [r['observation_id'] for r in sample['authors']] == ['w-1:1']


def test_sampling_includes_merged_away_work_observations(tmp_path):
    from citegraph.dedup import DedupConfig
    from tests.test_resolution_audit import corpus
    p = corpus(tmp_path)
    p.dedup_config = DedupConfig()
    p.deduplicate()
    sample = evaluator().sample_corpus(tmp_path)
    assert sample['work_audit_status'] == 'current'
    merged = next(group for group in sample['works'] if len(group['observed_members']) == 2)
    assert {member['observation_id'] for member in merged['observed_members']} == {'citation:0', 'citation:1'}


def test_historical_collisions_are_sampling_strata(tmp_path):
    import json

    import pandas as pd
    pd.DataFrame([dict(id='w-1', Title='Study', Authors_List=['Jane Smith'])]).to_csv(
        tmp_path / 'works.csv', index=False)
    (tmp_path / 'work_id_collisions.json').write_text(json.dumps({'collision_ids': ['w-1']}))
    assert 'id_collision' in evaluator().sample_corpus(tmp_path)['works'][0]['strata']
