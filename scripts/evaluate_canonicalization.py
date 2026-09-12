"""Offline synthetic regression evaluation and frozen, unlabeled corpus sampling."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from citegraph.authors import _row_authors, normalize_authors, parse_author
from citegraph.dedup import _candidate_index_lookup, _candidate_indices, canonicalize_works
from citegraph.html_report import _collect_resolution_audit
from citegraph.io import (
    WORK_AUDIT_INPUTS,
    WORK_AUDIT_OUTPUTS,
    OutLayout,
    parse_openalex_authors,
    unverified_collision_ids,
)


def evaluate_pairs(labels: list[dict], assignments: dict[str, str]) -> dict:
    counts = dict(tp=0, fp=0, fn=0, tn=0, ambiguous_count=0)
    for pair in labels:
        if pair['label'] not in {'same', 'different', 'ambiguous'}:
            raise ValueError(f"Unknown label: {pair['label']}")
        if any(pair[side] not in assignments for side in ('left', 'right')):
            raise ValueError(f'missing observation in assignments: {pair}')
        if pair['label'] == 'ambiguous':
            counts['ambiguous_count'] += 1
            continue
        predicted = assignments[pair['left']] == assignments[pair['right']]
        actual = pair['label'] == 'same'
        counts[('tp' if actual else 'fp') if predicted else ('fn' if actual else 'tn')] += 1
    return _metrics(counts)


def _metrics(counts):
    counts = dict(counts)
    counts['precision'] = counts['tp'] / (counts['tp'] + counts['fp']) if counts['tp'] + counts['fp'] else None
    counts['recall'] = counts['tp'] / (counts['tp'] + counts['fn']) if counts['tp'] + counts['fn'] else None
    return counts


def partition(assignments):
    groups = defaultdict(set)
    for observation, cluster in assignments.items():
        groups[cluster].add(observation)
    return frozenset(frozenset(group) for group in groups.values())


def _frame(records, columns=(), *, csv=False):
    frame = pd.DataFrame(records) if records else pd.DataFrame(columns=columns)
    if csv:
        frame = pd.read_csv(io.StringIO(frame.to_csv(index=False)), keep_default_na=False)
    for col, parser in [('OpenAlex_Authors', parse_openalex_authors)]:
        if col in frame:
            frame[col] = frame[col].map(parser)
    return frame


def _run(case, kind, *, reverse=False, csv=False):
    if kind == 'works':
        source = list(case['sources'])
        citation = list(case['citations'])
        if reverse:
            source.reverse()
            citation.reverse()
        ids = [r.get('observation_id', r.get('id')) for r in source + citation]
        if len(ids) != len(set(ids)) or None in ids:
            raise ValueError('Fixture observation IDs must be unique and nonempty')
        def clean(rows):
            return [{k: v for k, v in r.items() if k != 'observation_id'} for r in rows]
        sources = _frame(clean(source), ['id', 'source_file', 'Title', 'Authors_List', 'Year'], csv=csv)
        citations = _frame(clean(citation), ['citing_id', 'Title', 'Authors_List', 'Year'], csv=csv)
        _, edges, stats = canonicalize_works(sources, citations, show_progress=False)
        assignments = dict(zip(ids, stats['source_cluster_ids'] + stats['citation_cluster_ids'], strict=True))
        combined = pd.concat([sources, citations], ignore_index=True)
        authors, titles, unknown = _candidate_index_lookup(combined)
        index = {value: i for i, value in enumerate(ids)}
        positives = [p for p in case['labels'] if p['label'] == 'same']
        found = sum(index[p['right']] in _candidate_indices(combined.iloc[index[p['left']]],
                    author_blocks=authors, title_blocks=titles, unknown_author=unknown) for p in positives)
        return assignments, dict(review_count=sum(d['decision'] == 'review' for d in stats['decisions']),
                                candidate_found=found, candidate_total=len(positives),
                                edge_count=len(edges), citation_counts=edges.cited_id.value_counts().to_dict())
    records = list(case['works'])
    if reverse:
        records.reverse()
    works = _frame(records, csv=csv).set_index('id')
    enriched = _frame(case.get('enriched_works', []), ['id', 'OpenAlex_Authors'], csv=csv).set_index('id')
    audit = []
    _, occurrences, review = normalize_authors(works=works, enriched_works=enriched, audit=audit)
    assignments = {f'{r.record_id}:{r.position}': r.author_id for r in occurrences.itertuples()}
    return assignments, dict(review_count=len(review), candidate_found=None, candidate_total=None)


def evaluate_fixtures(directory: Path) -> dict:
    result = {'scope': 'synthetic regression fixtures; no held-out corpus accuracy claim'}
    for kind in ('works', 'authors'):
        reports, totals = [], Counter(dict(tp=0, fp=0, fn=0, tn=0, ambiguous_count=0))
        for case in json.loads((directory / f'{kind}.json').read_text()):
            assignments, extra = _run(case, kind)
            metrics = evaluate_pairs(case['labels'], assignments)
            totals.update({k: metrics[k] for k in totals})
            reversed_assignments, reversed_extra = _run(case, kind, reverse=True)
            reports.append(dict(case_id=case['case_id'], metrics=metrics, assignments=assignments,
                                repeat_equal=assignments == _run(case, kind)[0],
                                csv_roundtrip_equal=partition(assignments) == partition(_run(case, kind, csv=True)[0]),
                                reversed_partition_equal=partition(assignments) == partition(reversed_assignments),
                                reversed_assignments=reversed_assignments,
                                reversed_edge_count=reversed_extra.get('edge_count'), **extra))
        candidate_total = sum(c['candidate_total'] or 0 for c in reports)
        result[kind] = dict(metrics=_metrics(totals), cases=reports,
                            review_count=sum(c['review_count'] for c in reports),
                            candidate_recall=(sum(c['candidate_found'] or 0 for c in reports) / candidate_total
                                              if candidate_total else None))
    return result


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _sample(groups, limit):
    buckets = defaultdict(list)
    for group in groups:
        group['sample_id'] = _digest(group)
        buckets[group['stratum']].append(group)
    for bucket in buckets.values():
        bucket.sort(key=lambda g: g['sample_id'])
    selected = []
    while len(selected) < limit and any(buckets.values()):
        for key in sorted(buckets):
            if buckets[key] and len(selected) < limit:
                selected.append(buckets[key].pop(0))
    # Freeze precisely one fifth before labels exist; no labels or model decisions drive selection.
    for i, group in enumerate(sorted(selected, key=lambda g: g['sample_id'])):
        group.update(split='held_out' if i % 5 == 0 else 'development', label=None)
    return selected


def sample_corpus(directory: Path, limit=150) -> dict:
    """Sample observations with context, independent of the production candidate index.

    These are review groups, not automatically labeled pairs. Humans must seek
    duplicates outside each group to assess blocking omissions.
    """
    works = pd.read_csv(directory / 'works.csv', keep_default_na=False)
    authors, work_groups = [], []
    enriched = {}
    enrichment_path = directory / 'enriched_works.csv'
    if enrichment_path.exists():
        enriched = {r['id']: parse_openalex_authors(r.get('OpenAlex_Authors')) or []
                    for r in pd.read_csv(enrichment_path, keep_default_na=False).to_dict('records')}
    external = defaultdict(set)
    for entries in enriched.values():
        for entry in entries:
            if entry.get('openalex_id') and entry.get('orcid'):
                external[entry['openalex_id']].add(entry['orcid'])
    conflicts = {key for key, values in external.items() if len(values) > 1}
    collision_ids = unverified_collision_ids(OutLayout(directory))
    audit_path = directory / 'canonicalization_audit.json'
    if audit_path.exists():
        for collision in json.loads(audit_path.read_text()).get('id_collisions', []):
            collision_ids.update([collision['base_id'], collision['assigned_id']])
    errors = []
    audit = _collect_resolution_audit(audit_path, errors, inputs=WORK_AUDIT_INPUTS,
                                    outputs=WORK_AUDIT_OUTPUTS)
    observed = defaultdict(list)
    if audit['audit_status'] == 'current':
        for kind, filename in [('source', 'sources.csv'), ('citation', 'citations_raw.csv')]:
            records = pd.read_csv(directory / filename, keep_default_na=False).to_dict('records')
            mapping = audit['saved_audit'].get(f'{kind}_cluster_ids')
            if not isinstance(mapping, list) or len(mapping) != len(records):
                raise ValueError('Saved observation mapping does not match the corpus')
            for i, (record, cid) in enumerate(zip(records, mapping, strict=True)):
                if cid not in set(works.id):
                    raise ValueError(f'Saved observation refers to unknown work: {cid}')
                observed[cid].append(dict(observation_id=f'{kind}:{i}', record=record))
    # Independent coarse title grouping, deliberately not the production candidate index.
    title_groups = defaultdict(list)
    for row in works.to_dict('records'):
        title_groups[' '.join(str(row.get('Title', '')).lower().split()[:2])].append(row['id'])
    all_names = [name for _, row in works.iterrows() for name in _row_authors(row)]
    surnames = Counter(parse_author(name).surname_norm for name in all_names if parse_author(name))
    for row in works.to_dict('records'):
        title_tokens = str(row.get('Title', '')).lower().split()
        related = title_groups[' '.join(title_tokens[:2])]
        strata = []
        if row['id'] in collision_ids:
            strata.append('id_collision')
        if len(related) > 1:
            strata.append('similar_titles')
        strata.append('short_title' if len(title_tokens) < 6 else 'long_title')
        work_groups.append(dict(observation_id=row['id'], record=row, related_work_ids=related,
                                observed_members=observed[row['id']], stratum=strata[0], strata=strata))
        for pos, name in enumerate(_row_authors(pd.Series(row))):
            if not name.strip():
                continue
            parsed = parse_author(name)
            if not parsed:
                stratum = 'unparsed'
            elif ' ' in parsed.surname_norm:
                stratum = 'compound_surname'
            elif surnames[parsed.surname_norm] > 3:
                stratum = 'common_surname'
            elif '.' in name:
                stratum = 'initials'
            else:
                stratum = 'other'
            strata = [stratum]
            if '.' in name and 'initials' not in strata:
                strata.append('initials')
            context = enriched.get(row['id'], [])
            if any(entry.get('openalex_id') in conflicts for entry in context):
                strata.insert(0, 'identifier_conflict')
            authors.append(dict(observation_id=f"{row['id']}:{pos}", raw_author=name,
                                work=row, stratum=strata[0], strata=strata,
                                enrichment_context=context))
    return dict(schema_version=1, corpus_fingerprint=_digest(works.to_dict('records')),
                work_audit_status=audit['audit_status'], audit_errors=errors,
                limitation='Unlabeled observations with context. Human review and independently found '
                           'duplicates are required; this is not an accuracy estimate.',
                works=_sample(work_groups, limit), authors=_sample(authors, limit))


def write_sample(directory: Path, output: Path):
    sample = sample_corpus(directory)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(sample, stream, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--corpus-out', type=Path)
    parser.add_argument('--sample-out', type=Path)
    args = parser.parse_args()
    if args.corpus_out and args.sample_out:
        write_sample(args.corpus_out, args.sample_out)
    elif args.fixtures and args.out:
        args.out.write_text(json.dumps(evaluate_fixtures(args.fixtures), indent=2, ensure_ascii=False))
    else:
        parser.error('Supply --fixtures/--out or --corpus-out/--sample-out')


if __name__ == '__main__':
    main()
