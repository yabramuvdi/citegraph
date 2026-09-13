"""Occurrence-based corrections bound to the works snapshot they describe."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd

from citegraph.io import OutLayout, fingerprint, frame_fingerprint, read_json, write_json


def _redirects(path: Path) -> dict:
    registry = OutLayout(Path(path).parent).work_identity_json
    return read_json(registry).get('redirects', {}) if registry.exists() else {}


def _occurrence_evidence(rows: list[dict], works: pd.DataFrame) -> dict[str, str]:
    from citegraph.authors import _row_authors

    evidence = {}
    for row in rows:
        for side in ('left', 'right'):
            wid, position = str(row[f'{side}_record_id']), int(row[f'{side}_position'])
            work = works.loc[wid]
            evidence[json.dumps([wid, position])] = fingerprint({
                'raw_author': _row_authors(work)[position],
                'Title': work.get('Title'), 'Year': work.get('Year'),
            })
    return evidence


def validate_constraints(rows: list[dict], coordinates: set[tuple[str, int]]) -> list[dict]:
    result = []
    for number, row in enumerate(rows, 2):
        row = dict(row)
        try:
            if row['action'] not in {'merge', 'separate'} or not str(row['reason']).strip():
                raise ValueError('action must be merge/separate and reason must be nonempty')
            for side in ('left', 'right'):
                value = str(row[f'{side}_position'])
                if not value.isdigit():
                    raise ValueError('position must be a nonnegative integer')
                row[f'{side}_position'] = int(value)
                coordinate = (str(row[f'{side}_record_id']), int(value))
                if coordinate not in coordinates:
                    raise ValueError(f'unknown author occurrence {coordinate}')
        except (KeyError, ValueError) as exc:
            raise ValueError(f'Author override row {number}: {row}: {exc}') from exc
        result.append(row)
    return result


def load_author_constraints(path: Path, *, works: pd.DataFrame, metadata_path: Path) -> list[dict]:
    from citegraph.authors import _collect_occurrences

    path, metadata_path = Path(path), Path(metadata_path)
    if not path.exists():
        return []
    with path.open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        required = ['action', 'left_record_id', 'left_position', 'right_record_id',
                    'right_position', 'reason']
        if reader.fieldnames != required:
            raise ValueError(f'Author override columns must be {required}; got {reader.fieldnames}')
        rows = list(reader)
    from citegraph.identity import resolve_redirect

    redirects = _redirects(path)
    for row in rows:
        for side in ('left', 'right'):
            row[f'{side}_record_id'] = resolve_redirect(row[f'{side}_record_id'], redirects)
    current = frame_fingerprint(works.rename_axis('id'))
    coordinates = {(o.record_id, o.position) for o in _collect_occurrences(
        works=works, enriched_works=None)}
    rows = validate_constraints(rows, coordinates)
    if metadata_path.exists():
        binding = json.loads(metadata_path.read_text())
        if binding.get('schema_version') == 2:
            stored = binding.get('occurrences')
            if not isinstance(stored, dict):
                raise ValueError('Invalid author override occurrence binding; restore metadata')
            previous = {}
            for key, value in stored.items():
                wid, position = json.loads(key)
                key = json.dumps([resolve_redirect(wid, redirects), position])
                if key in previous and previous[key] != value:
                    raise ValueError(f'Author override occurrence became ambiguous: {key}')
                previous[key] = value
            changed = [key for key, value in _occurrence_evidence(rows, works).items()
                       if key in previous and previous[key] != value]
            if changed:
                raise ValueError(f'Author override snapshot changed at occurrences: {changed}. '
                                 f'Review and explicitly rebind by removing {metadata_path}.')
        elif binding.get('schema_version') != 1 or binding.get('works_fingerprint') != current:
            raise ValueError(f'Author override snapshot changed to {current}; affected rows: {rows}. '
                             f'Review these occurrences and explicitly rebind by removing {metadata_path}.')
    return rows


def bind_author_constraints(path: Path, *, works: pd.DataFrame, metadata_path: Path) -> None:
    """Bind only after the caller successfully validates normalization and aliases."""
    if Path(path).exists():
        rows = load_author_constraints(path, works=works, metadata_path=metadata_path)
        write_json(metadata_path, {'schema_version': 2, 'occurrences': _occurrence_evidence(rows, works),
                                   'works_fingerprint': frame_fingerprint(works.rename_axis('id'))})


def prepare_constraints(occurrences: list, rows: list[dict]) -> list[list]:
    """Seed forced groups and attach expanded cannot-links before any automatic union."""
    by_key = {(o.record_id, o.position): o for o in occurrences}
    rows = validate_constraints(rows, set(by_key))
    parent = {key: key for key in by_key}

    def root(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def pair(row):
        return [(row[f'{side}_record_id'], row[f'{side}_position']) for side in ('left', 'right')]

    for row in rows:
        if row['action'] == 'merge':
            left, right = pair(row)
            parent[root(right)] = root(left)
    groups = {}
    for key, occurrence in by_key.items():
        groups.setdefault(root(key), []).append(occurrence)
    for group in groups.values():
        if len({o.orcid for o in group if o.orcid}) > 1:
            raise ValueError(f'Author merge has conflicting ORCIDs: {[(o.record_id, o.position) for o in group]}')
    for row in rows:
        if row['action'] == 'separate':
            left, right = (root(key) for key in pair(row))
            if left == right:
                raise ValueError(f'Author override contradiction through merge chain: {row}')
            for own, other in ((left, right), (right, left)):
                forbidden = {(o.record_id, o.position) for o in groups[other]}
                for occurrence in groups[own]:
                    occurrence.cannot_link.update(forbidden)
    return list(groups.values())
