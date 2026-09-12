import pandas as pd
import pytest

from citegraph.authors import normalize_authors


def works(names):
    return pd.DataFrame({'Authors_List': [[n] for n in names], 'ring': 0},
                        index=[f'w-{i+1}' for i in range(len(names))])


def correction(action, left, right, lp=0, rp=0):
    return dict(action=action, left_record_id=left, left_position=lp,
                right_record_id=right, right_position=rp, reason='Verified in paper')


def test_merge_and_separate_govern_automatic_joins():
    frame = works(['Smith, John', 'Smith, John', 'Jones, Jane'])
    constraints = [correction('separate', 'w-1', 'w-2'), correction('merge', 'w-1', 'w-3')]
    for _ in range(2):
        audit = []
        _, edges, _ = normalize_authors(works=frame, constraints=constraints, audit=audit)
        ids = edges.set_index('record_id').author_id
        assert ids['w-1'] == ids['w-3'] != ids['w-2']
        assert len(edges) == 3
        assert len([a for a in audit if a['decision'] == 'manual_override']) == 2


def test_transitive_contradiction_rejected():
    with pytest.raises(ValueError, match='contradict'):
        normalize_authors(works=works(['Smith, John'] * 3), constraints=[
            correction('merge', 'w-1', 'w-2'), correction('merge', 'w-2', 'w-3'),
            correction('separate', 'w-1', 'w-3')])


def test_same_work_positions_can_be_separated():
    frame = works(['Smith, John'])
    frame.at['w-1', 'Authors_List'] = ['Smith, John', 'Smith, John']
    _, edges, _ = normalize_authors(works=frame, constraints=[
        correction('separate', 'w-1', 'w-1', 0, 1)])
    assert edges.author_id.nunique() == 2


def test_snapshot_binding_is_deferred_and_changes_rejected(tmp_path):
    from citegraph.author_overrides import bind_author_constraints, load_author_constraints
    path, meta = tmp_path / 'overrides.csv', tmp_path / 'meta.json'
    pd.DataFrame([correction('separate', 'w-1', 'w-2')]).to_csv(path, index=False)
    frame = works(['Smith, John'] * 2)
    load_author_constraints(path, works=frame, metadata_path=meta)
    assert not meta.exists()
    bind_author_constraints(path, works=frame, metadata_path=meta)
    load_author_constraints(path, works=frame, metadata_path=meta)
    frame.at['w-1', 'Authors_List'] = ['Smith, Jane']
    with pytest.raises(ValueError, match='snapshot'):
        load_author_constraints(path, works=frame, metadata_path=meta)


@pytest.mark.parametrize('aliases', [{'stale': 'a-smith-john'},
                                   {'a-smith-john': 'a-jones-jane', 'a-jones-jane': 'a-smith-john'}])
def test_invalid_legacy_aliases_rejected(aliases):
    with pytest.raises(ValueError):
        normalize_authors(works=works(['Smith, John', 'Jones, Jane']), aliases=aliases)


def test_override_snapshot_includes_unnamed_work_ids(tmp_path):
    from citegraph.author_overrides import bind_author_constraints, load_author_constraints
    path, meta = tmp_path / 'overrides.csv', tmp_path / 'meta.json'
    pd.DataFrame([correction('merge', 'w-1', 'w-2')]).to_csv(path, index=False)
    frame = works(['Smith, John'] * 2)
    bind_author_constraints(path, works=frame, metadata_path=meta)
    frame.index = ['w-2', 'w-1']
    with pytest.raises(ValueError, match='snapshot'):
        load_author_constraints(path, works=frame, metadata_path=meta)


@pytest.mark.parametrize('action', ['merge', 'separate'])
def test_constraints_precede_shared_external_id(action):
    frame = works(['Smith, John'] * 2)
    enriched = pd.DataFrame({'OpenAlex_Authors': [[{
        'display_name': 'John Smith', 'openalex_id': 'A1'}]] * 2}, index=frame.index)
    _, edges, _ = normalize_authors(works=frame, enriched_works=enriched,
        constraints=[correction(action, 'w-1', 'w-2')])
    assert edges.author_id.nunique() == (2 if action == 'separate' else 1)


def test_explicit_merge_cannot_override_orcid_conflict():
    frame = works(['Smith, John'] * 2)
    enriched = pd.DataFrame({'OpenAlex_Authors': [[{
        'display_name': 'John Smith', 'orcid': orcid}]
        for orcid in ['0000-1', '0000-2']]}, index=frame.index)
    with pytest.raises(ValueError, match='ORCID'):
        normalize_authors(works=frame, enriched_works=enriched,
            constraints=[correction('merge', 'w-1', 'w-2')])


def test_alias_cannot_override_separation():
    frame = works(['Smith, John'] * 2)
    with pytest.raises(ValueError, match='separate'):
        normalize_authors(works=frame, constraints=[correction('separate', 'w-1', 'w-2')],
                          aliases={'a-smith-john-1': 'a-smith-john'})


def test_noop_merge_preserves_metrics():
    frame = works(['Smith, John'] * 2)
    expected = normalize_authors(works=frame)[0]
    actual = normalize_authors(works=frame, constraints=[correction('merge', 'w-1', 'w-2')])[0]
    pd.testing.assert_frame_equal(expected, actual)


@pytest.mark.parametrize('change', [{'left_position': -1}, {'right_record_id': 'w-missing'},
                                  {'action': 'split'}, {'reason': ''}])
def test_invalid_coordinates_and_actions_rejected(change):
    row = correction('merge', 'w-1', 'w-2') | change
    with pytest.raises(ValueError, match='row 2'):
        normalize_authors(works=works(['Smith, John'] * 2), constraints=[row])


def test_invalid_header_rejected(tmp_path):
    from citegraph.author_overrides import load_author_constraints
    path = tmp_path / 'overrides.csv'
    path.write_text('action,reason\n')
    with pytest.raises(ValueError, match='columns'):
        load_author_constraints(path, works=works([]), metadata_path=tmp_path / 'meta.json')


def test_conflicting_alias_rows_rejected(tmp_path):
    from citegraph.authors import load_aliases
    path = tmp_path / 'aliases.csv'
    path.write_text('cluster_id,canonical_id\na-one,a-two\na-one,a-three\n')
    with pytest.raises(ValueError, match='conflict'):
        load_aliases(path)
