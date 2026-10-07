#!/usr/bin/env python3
"""Verify an authored audit package without fetching or requiring article PDFs.

Optional source verification is an explicit, separate operation. No model scores
are read, and successful register verification is not anatomical validation.
"""
import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-dir', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--verify-source-files', action='store_true')
    parser.add_argument('--source-root', type=Path, default=Path('.'))
    parser.add_argument('--source-map', type=Path,
                        help='Optional JSON object mapping source IDs or original file paths to local file paths.')
    args = parser.parse_args()
    base = args.audit_dir.resolve()
    read = lambda name: json.loads((base / name).read_text())
    manifest = read('assembly_manifest.json')
    assert manifest['status'] == 'PASS'
    checked = 0
    for filename, expected in manifest['outputs'].items():
        assert digest(base / filename) == expected, ('output hash mismatch', filename)
        checked += 1
    for filename, expected in manifest['inputs'].items():
        assert digest(base / filename) == expected, ('review input hash mismatch', filename)
        checked += 1
    assert digest(base / 'REVIEW_PLAN.json') == manifest['frozen_plan_sha256']
    assert digest(base / 'selected_endpoints.json') == manifest['frozen_sample_sha256']
    sample = read('completed_documentary_sample.json')
    population = read('endpoint_register_completed.json')
    selected = {r['endpoint_id']: r for r in read('selected_endpoints.json')}
    assert len(sample) == len(selected) == 100
    assert len(population) == 2254
    assert len({r['endpoint_id'] for r in population}) == 2254
    assert Counter(r['documentary_review_status'] for r in population) == {'REVIEWED': 100, 'NOT_REVIEWED': 2154}
    assert Counter(r['history'] for r in sample) == {k: 20 for k in ['tetrapods', 'dinosaurs', 'ichthyosaurs', 'gnathostomes', 'squamates']}
    for row in sample:
        original = selected[row['endpoint_id']]
        for field in ['history', 'original_taxon_label', 'old_character_index_zero_based',
                      'new_character_index_zero_based', 'later_state', 'selection_digest']:
            assert row[field] == original[field], (row['endpoint_id'], field)
        assert row['later_primary_token'] == row['later_state']
        assert row['expert_review_status'] == 'NOT_PERFORMED'
        assert not row['exact_anatomical_state_reobserved_by_this_audit']
    experts = read('expert_blank_register.json')
    assert {r['endpoint_id'] for r in experts} == set(selected)
    for row in experts:
        assert row['review_status'] == 'NOT_PERFORMED'
        assert all(value == '' for key, value in row.items() if key not in ['endpoint_id', 'review_status'])
    assert {r['endpoint_id'] for r in read('expert_source_locator_packet.json')} == set(selected)
    assert {r['endpoint_id'] for r in read('expert_later_state_reveal.json')} == set(selected)
    sources = read('source_index.json')['sources']
    assert len({s['source_id'] for s in sources}) == len(sources)
    assert all(re.fullmatch('[a-f0-9]{64}', s['sha256']) for s in sources)
    source_result = {'status': 'NOT_CHECKED', 'reason': 'Article PDFs and other external source files are not required for register integrity checks; no offline source-verification claim is made.'}
    if args.verify_source_files:
        mapping = json.loads(args.source_map.read_text()) if args.source_map else {}
        passed, unavailable, mismatched = [], [], []
        for source in sources:
            original = source['optional_original_local_file']
            location = Path(mapping.get(source['source_id'], mapping.get(original, original)))
            path = location if location.is_absolute() else args.source_root / location
            if not path.is_file():
                unavailable.append(source['source_id'])
            elif digest(path) != source['sha256']:
                mismatched.append(source['source_id'])
            else:
                passed.append(source['source_id'])
        source_result = {'status': 'PASS' if not unavailable and not mismatched else 'INCOMPLETE_OR_FAILED',
                         'verified': passed, 'unavailable': unavailable, 'hash_mismatch': mismatched}
    result = {'register_integrity': 'PASS', 'sample_n': 100, 'registered_n': 2254,
              'authored_file_hashes_checked': checked + 2,
              'source_files': source_result, 'independent_expert_validation': 'NOT_PERFORMED'}
    print(json.dumps(result, indent=2))
    if source_result['status'] == 'INCOMPLETE_OR_FAILED':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
