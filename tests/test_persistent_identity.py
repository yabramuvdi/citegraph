"""Identity history must never move annotations onto an arbitrary split child."""
import pytest


def test_merge_redirect_split_and_id_reservation():
    from citegraph.identity import reconcile

    ids, first, _ = reconcile({'a-smith-j': {'one'}, 'a-smith-john': {'two'}})
    assert ids == {'a-smith-j': 'a-smith-j', 'a-smith-john': 'a-smith-john'}
    ids, merged, changes = reconcile({'a-smith-john': {'one', 'two', 'three'}}, first)
    assert set(ids.values()) == {'a-smith-john'}
    assert merged['redirects'] == {'a-smith-j': 'a-smith-john'}
    ids, split, changes = reconcile({'a-smith-john': {'one'}, 'a-smith-john-1': {'two', 'three'}}, merged)
    assert 'a-smith-john' not in ids.values()
    assert 'a-smith-j' not in ids.values()
    assert any(c['decision'] == 'identity_split' for c in changes)
    repeated, _, _ = reconcile({'a-smith-john': {'one'}, 'a-smith-john-1': {'two', 'three'}}, split)
    assert repeated == ids


def test_removed_entity_can_return_but_unrelated_entity_cannot_steal_id():
    from citegraph.identity import reconcile

    _, old, _ = reconcile({'w-original': {'old observation'}})
    ids, state, _ = reconcile({'w-original': {'different observation'}}, old)
    assert ids['w-original'] != 'w-original'
    ids, _, _ = reconcile({'w-new-spelling': {'old observation'}}, state)
    assert ids['w-new-spelling'] == 'w-original'


def test_invalid_registry_fails_closed():
    from citegraph.identity import reconcile

    with pytest.raises(ValueError, match='identity'):
        reconcile({'a-one': {'one'}}, {'schema_version': 987})
