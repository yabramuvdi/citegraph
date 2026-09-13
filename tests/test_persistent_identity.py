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


def _prior_frames():
    import pandas as pd

    works = pd.DataFrame(
        [{'id': 'w-1', 'ring': 0, 'source_file': 'a.pdf', 'Title': 'T1',
          'Authors_List': "['Ibáñez Díaz, Marcela']", 'Authors': 'Ibáñez Díaz, Marcela',
          'Journal': 'J', 'Year': 2007},
         {'id': 'w-2', 'ring': 1, 'source_file': '', 'Title': 'T2',
          'Authors_List': "['Smith, John']", 'Authors': 'Smith, John',
          'Journal': 'J', 'Year': 2008}]).set_index('id')
    authors = pd.DataFrame(
        [{'id': 'a-ibanez-diaz-marcela'}, {'id': 'a-smith-john'}]).set_index('id')
    citations = pd.DataFrame(
        # The first row is legacy drift: older code split the comma form, so the
        # stored raw_author is a prefix of what the works table yields today.
        [{'author_id': 'a-ibanez-diaz-marcela', 'record_id': 'w-1', 'position': 0,
          'raw_author': 'Ibáñez Díaz'},
         {'author_id': 'a-smith-john', 'record_id': 'w-2', 'position': 0,
          'raw_author': 'Smith, John'}])
    return works, authors, citations


def test_seeding_skips_unverifiable_occurrences_instead_of_discarding_registry():
    """0.5% legacy drift must not throw away every prior assignment."""
    from citegraph.identity import seed_author_registry

    works, authors, citations = _prior_frames()
    skipped: list = []
    registry = seed_author_registry(authors, citations, works=works, skipped=skipped)

    assert [s[0] for s in skipped] == ['a-ibanez-diaz-marcela']
    assert 'a-smith-john' in registry['entries']
    assert registry['entries']['a-smith-john']


def test_seeding_still_rejects_a_genuinely_corrupt_registry():
    from citegraph.identity import seed_author_registry

    works, authors, citations = _prior_frames()
    citations.loc[0, 'author_id'] = 'a-nobody'
    with pytest.raises(ValueError, match='unknown author'):
        seed_author_registry(authors, citations, works=works, skipped=[])


def test_user_merge_claims_its_target_id_even_when_the_prior_group_split():
    """A curated alias is the strongest identity evidence we have.

    Without a pin, a prior group that fans out is retired so no arbitrary child
    inherits its annotations. An alias names the child explicitly, so it is not
    arbitrary — the published ID must continue on it, or every crosswalk keyed
    to that ID breaks on the next run.
    """
    from citegraph.identity import reconcile

    _, prior, _ = reconcile({'a-lopez-maria-claudia': {'one', 'two', 'three'}})
    groups = {'a-lopez-maria-claudia': {'one', 'two'}, 'a-lopez-m': {'three'}}

    unpinned, _, _ = reconcile(groups, prior)
    assert unpinned['a-lopez-maria-claudia'] != 'a-lopez-maria-claudia'

    ids, state, changes = reconcile(
        groups, prior, pinned={'a-lopez-maria-claudia': 'a-lopez-maria-claudia'})
    assert ids['a-lopez-maria-claudia'] == 'a-lopez-maria-claudia'
    assert ids['a-lopez-m'] != 'a-lopez-maria-claudia'
    assert any(c['decision'] == 'identity_split_resolved' for c in changes)
    assert state['entries']['a-lopez-maria-claudia']


def test_two_clusters_cannot_claim_the_same_pinned_id():
    from citegraph.identity import reconcile

    _, prior, _ = reconcile({'a-shared': {'one', 'two'}})
    ids, _, _ = reconcile({'a-x': {'one'}, 'a-y': {'two'}}, prior,
                          pinned={'a-x': 'a-shared', 'a-y': 'a-shared'})
    assert ids['a-x'] != ids['a-y']
    assert sorted(ids.values()).count('a-shared') == 1
