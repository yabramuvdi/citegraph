"""Persist slug identities using observed membership, without freezing clusters."""
from __future__ import annotations

import json
from collections import defaultdict

from citegraph.io import fingerprint, metadata_fingerprint


def reconcile(groups: dict[str, set[str]], previous: dict | None = None, *,
              pinned: dict[str, str] | None = None) -> tuple[dict, dict, list]:
    """Keep unambiguous continuations; retire splits and redirect merged IDs.

    Missing entities remain reserved so removing/reintroducing input cannot give
    an old annotation key to somebody else. Membership hashes are evidence only;
    new entity IDs are always the supplied deterministic slugs (with suffixes).

    ``pinned`` maps a group key to an ID a *user* asserted it continues — an
    alias merge naming both spellings of one person. That is the strongest
    identity evidence available, so a pin claims its ID even when the prior
    group otherwise reads as a split: the child is not arbitrary, it is the
    one the user pointed at. The split is still recorded, as
    ``identity_split_resolved``, and the remaining children get fresh IDs.
    Two groups can never claim the same pin.
    """
    previous = previous if previous is not None else {
        'schema_version': 1, 'entries': {}, 'reserved_ids': [], 'redirects': {}, 'history': []}
    if (not isinstance(previous, dict) or previous.get('schema_version') != 1
            or not isinstance(previous.get('entries'), dict)
            or not isinstance(previous.get('redirects'), dict)
            or not isinstance(previous.get('reserved_ids'), list)
            or not isinstance(previous.get('history'), list)):
        raise ValueError('Invalid identity registry; restore its last valid copy')
    entries = previous['entries']
    if any(not isinstance(key, str) or not isinstance(members, list)
           or any(not isinstance(member, str) for member in members)
           for key, members in entries.items()):
        raise ValueError('Invalid identity membership; restore its last valid copy')
    redirects = dict(previous['redirects'])
    if any(not isinstance(k, str) or not isinstance(v, str) for k, v in redirects.items()):
        raise ValueError('Invalid identity redirects')
    for key in redirects:
        resolve_redirect(key, redirects)
    reserved = set(previous['reserved_ids']) | set(entries) | set(redirects) | set(redirects.values())
    old_by_member: dict[str, set[str]] = defaultdict(set)
    for old, members in entries.items():
        for member in members:
            old_by_member[member].add(old)
    ancestors = {new: set().union(*(old_by_member[m] for m in members))
                 for new, members in groups.items()}
    descendants: dict[str, set[str]] = defaultdict(set)
    for new, old_ids in ancestors.items():
        for old in old_ids:
            descendants[old].add(new)
    splits = {old for old, children in descendants.items() if len(children) > 1}
    assigned: dict[str, str] = {}
    # A user-curated merge outranks the split rule; honour it before anything
    # else can reserve the ID. Pinned IDs stay in ``splits`` so no *other*
    # child can pick them up as an ordinary continuation.
    claimed: dict[str, str] = {}
    for new in sorted(groups):
        target = pinned.get(new) if pinned else None
        if target is None or target in claimed:
            continue
        claimed[target] = new
        assigned[new] = target
    # Reserve all continuations before allocating any new collision suffix.
    for new in sorted(groups):
        if new in assigned:
            continue
        candidates = ancestors[new] - splits - set(claimed)
        if candidates:
            assigned[new] = new if new in candidates else min(candidates)
    used = set(assigned.values())
    for new in sorted(groups):
        if new in assigned:
            continue
        candidate, suffix = new, 1
        while candidate in reserved or candidate in used:
            candidate = f'{new}-{suffix}'
            suffix += 1
        assigned[new] = candidate
        used.add(candidate)
    changes = []
    retained = {old: list(members) for old, members in entries.items()
                if old not in descendants}
    for new, cid in assigned.items():
        # Preserve disappeared members of a continuation for future restoration.
        members = set(groups[new])
        inherited = ancestors[new] - splits
        if claimed.get(cid) == new and cid in entries:
            inherited = inherited | {cid}
        for old in inherited:
            members.update(entries[old])
            if old != cid:
                redirects[old] = cid
                changes.append({'decision': 'identity_merge', 'old_id': old, 'new_id': cid})
        retained[cid] = sorted(members)
    for old in sorted(splits):
        entry = {'decision': 'identity_split', 'old_id': old,
                 'new_ids': sorted(assigned[new] for new in descendants[old])}
        if old in claimed:
            # A user merge decided which child continues; say so in the audit
            # rather than reporting the ID as retired.
            entry['decision'] = 'identity_split_resolved'
            entry['assigned_id'] = old
        changes.append(entry)
    # Redirects ending at a split remain retired, never choose a split child.
    state = {'schema_version': 1, 'entries': retained,
             'reserved_ids': sorted(reserved | used), 'redirects': redirects,
             'history': previous['history'] + changes}
    return assigned, state, changes


def resolve_redirect(identifier: str, redirects: dict[str, str]) -> str:
    seen = set()
    while identifier in redirects:
        if identifier in seen:
            raise ValueError('Invalid identity redirect cycle')
        seen.add(identifier)
        identifier = redirects[identifier]
    return identifier


def occurrence_key(record_id: str, position: int, raw_author: str) -> str:
    return json.dumps([str(record_id), int(position), raw_author], ensure_ascii=False)


def remap_author_registry(state: dict, work_redirects: dict) -> dict:
    """Apply confirmed work redirects to author evidence, never split mappings."""
    result = dict(state)
    for section in ('algorithm', 'published'):
        if section not in state:
            continue
        registry = dict(state[section])
        entries = {}
        for aid, members in registry['entries'].items():
            entries[aid] = []
            for member in members:
                wid, position, raw = json.loads(member)
                entries[aid].append(occurrence_key(resolve_redirect(wid, work_redirects), position, raw))
        registry['entries'] = entries
        result[section] = registry
    return result


def work_groups(works, sources, citations, stats) -> dict[str, set[str]]:
    groups = {str(wid): set() for wid in works.index}
    metadata_groups: dict[str, set[str]] = defaultdict(set)
    for row, cid in zip(sources.to_dict('records'), stats['source_cluster_ids'], strict=True):
        signature = metadata_fingerprint(row)
        groups[cid].add(fingerprint(['source', row['id'], signature]))
        metadata_groups[signature].add(cid)
    positions: dict[str, int] = defaultdict(int)
    for row, cid in zip(citations.to_dict('records'), stats['citation_cluster_ids'], strict=True):
        signature = metadata_fingerprint(row)
        source = str(row['citing_id'])
        groups[cid].add(fingerprint(['citation', source, positions[source], signature]))
        positions[source] += 1
        metadata_groups[signature].add(cid)
    # An exact metadata observation is an anchor only if it names one cluster.
    # This also bootstraps old works tables without inventing past decisions.
    for signature, ids in metadata_groups.items():
        if len(ids) == 1:
            groups[next(iter(ids))].add('metadata:' + signature)
    return groups


def seed_work_registry(works) -> dict:
    groups = {str(wid): {'metadata:' + metadata_fingerprint(row.to_dict())}
              for wid, row in works.iterrows()}
    return reconcile(groups)[1]


def seed_author_registry(authors, citations, *, works=None, skipped=None) -> dict:
    from citegraph.authors import _row_authors

    if not authors.index.is_unique or citations.duplicated(['record_id', 'position']).any():
        raise ValueError('Invalid prior author identity: duplicate IDs or occurrences')
    groups = {str(aid): set() for aid in authors.index}
    for row in citations.itertuples(index=False):
        if row.author_id not in groups:
            raise ValueError('Invalid prior author identity: occurrence references an unknown author')
        if works is not None:
            names = _row_authors(works.loc[row.record_id]) if row.record_id in works.index else []
            position = int(row.position)
            if position != row.position or not 0 <= position < len(names) or names[position] != row.raw_author:
                # Older code split comma-bearing author strings differently, so
                # a handful of legacy occurrences no longer describe the works
                # table. Callers that pass `skipped` keep the rest of the
                # registry: dropping an occurrence only removes an anchor, and
                # a cluster with no surviving ancestor is allocated a fresh ID
                # rather than silently continuing the wrong one.
                if skipped is None:
                    raise ValueError('Invalid prior author identity: occurrence differs from previous works')
                skipped.append((row.author_id, row.record_id, position, row.raw_author))
                continue
        groups[row.author_id].add(occurrence_key(row.record_id, row.position, row.raw_author))
    return reconcile(groups)[1]
