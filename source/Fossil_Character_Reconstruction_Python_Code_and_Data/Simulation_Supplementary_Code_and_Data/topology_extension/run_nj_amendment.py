#!/usr/bin/env python3
"""Add a data-derived NJ start to the saved three-random-start NNI analysis.

This is an explicitly post hoc amendment prompted by failure of the original
search to recover short trees even for truth-complete matrices. The original
results are retained. The generating topology is never supplied to NJ/search.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
from nni_search import hillclimb, score_search
from nj_start import matrix_nj_start, combine_endpoints
from run_topology_extension import (sha, serialized_search, summarize_matrix,
                                    scenario_summaries, paired_scheme_summaries, MAX_STEPS)


def amend_one(arguments):
    root_s, original_s, out_s, identifier = arguments
    root = Path(root_s); original = Path(original_s); out = Path(out_s)
    receipt_path = original / 'search_records' / f'{identifier}.json.gz'
    with gzip.open(receipt_path, 'rt', encoding='utf8') as handle:
        previous = json.load(handle)
    base = root / 'simulation/results'
    paths = {'masked': base/'inputs'/f'{identifier}_masked.npz',
             'truth': base/'truth'/f'{identifier}_truth.npz',
             'predictions': base/'predictions'/f'{identifier}.npz',
             'evaluation': base/'evaluation'/f'{identifier}.npz'}
    assert {key: sha(path) for key, path in paths.items()} == previous['source_hashes']
    with np.load(paths['masked'], allow_pickle=False) as z: masked = z['X']
    with np.load(paths['truth'], allow_pickle=False) as z: truth = z['X_true']; truth_edges = z['edges']
    with np.load(paths['predictions'], allow_pickle=False) as z:
        probabilities = z['probabilities']; targets = z['target_indices']
        draws = np.unpackbits(z['missing_states_packed'], axis=2, count=len(targets))
    assert np.array_equal(targets, np.argwhere(masked == -1))
    assert np.array_equal(masked[masked >= 0], truth[masked >= 0])
    schemes = ('mode', 'knn5', 'logistic', 'extra_trees', 'nj_mk', 'nj_mk_joint')
    cache = {}; searches = {}; rows = []; started = time.monotonic()
    for old in previous['treatment_results']:
        key = old['searched_matrix_sha256']
        if key not in cache:
            if old['kind'] == 'baseline':
                matrix = masked if old['scheme'] == 'source_incomplete' else truth
            else:
                index = schemes.index(old['scheme'])
                matrix = masked.copy()
                if old['kind'] == 'map':
                    states = np.argmax(probabilities[min(index, 4)], axis=-1)
                    matrix[targets[:, 0], targets[:, 1]] = states[targets[:, 0], targets[:, 1]]
                elif old['kind'] == 'draw':
                    matrix[targets[:, 0], targets[:, 1]] = draws[index, old['draw_index']]
                else:
                    raise ValueError(old['kind'])
            matrix = np.asarray(matrix, dtype=np.int8)
            import hashlib
            assert hashlib.sha256(matrix.tobytes()).hexdigest() == key
            nj_edges, nj_diagnostic = matrix_nj_start(matrix)
            added = hillclimb(matrix, nj_edges, max_steps=MAX_STEPS)
            found = [{**item, 'edges': np.asarray(item['edges'], dtype=np.int16)}
                     for item in previous['searches'][key]['starts']]
            assert len(found) == 3
            result = combine_endpoints(found + [added], len(matrix))
            assert result['minimum_found_length'] <= previous['searches'][key]['minimum_found_length']
            # Generating topology enters only this separate post-search scorer.
            cache[key] = score_search(result, truth_edges, len(matrix))
            searches[key] = {**serialized_search(result),
                             'nj_start_edges': nj_edges.tolist(), 'nj_diagnostics': nj_diagnostic}
        rows.append({**old, **cache[key]})
    baseline = {item['scheme']: item['mean_normalized_RF'] for item in rows if item['kind'] == 'baseline'}
    for row in rows:
        row['RF_difference_from_source_incomplete'] = row['mean_normalized_RF'] - baseline['source_incomplete']
        row['RF_difference_from_truth_complete'] = row['mean_normalized_RF'] - baseline['truth_complete']
        row['worse_RF_than_source_incomplete'] = int(row['RF_difference_from_source_incomplete'] > 1e-12)
        row['worse_RF_than_truth_complete'] = int(row['RF_difference_from_truth_complete'] > 1e-12)
        row['D_exact_and_RF_worse_than_truth_complete'] = row['fixed_D_exact_match'] * row['worse_RF_than_truth_complete']
    receipt = {'matrix_id': identifier, 'source_hashes': previous['source_hashes'],
               'original_search_receipt_sha256': sha(receipt_path),
               'original_start_seed_sequences': previous['start_seed_sequences'],
               'original_start_edges': previous['start_edges'],
               'draw_indices': previous['draw_indices'], 'n_starts_per_input': 4,
               'n_recorded_treatments': len(rows),
               'n_distinct_input_matrices_searched': len(searches),
               'n_new_NJ_starts_searched': len(searches),
               'seconds': time.monotonic() - started, 'searches': searches, 'treatment_results': rows}
    with gzip.open(out/'search_records'/f'{identifier}.json.gz', 'wt', encoding='utf8') as handle:
        json.dump(receipt, handle, separators=(',', ':'), allow_nan=True)
    return rows, {key: value for key, value in receipt.items()
                  if key not in ('searches', 'treatment_results', 'original_start_edges')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--original-results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--limit', type=int, help='Engineering check only; omit for all600 matrices.')
    args = parser.parse_args()
    root = args.root.resolve(); original = args.original_results.resolve(); out = args.output.resolve()
    if args.workers < 1: parser.error('--workers must be positive')
    if out == root or root in out.parents or out == original or original in out.parents:
        raise ValueError('Output must be outside the source archive and original results')
    if out.exists() and any(out.iterdir()): raise FileExistsError('Use a new empty output directory')
    out.mkdir(parents=True, exist_ok=True); (out/'search_records').mkdir()
    old_manifest = json.loads((original/'RUN_MANIFEST.json').read_text())
    assert old_manifest['complete_600_matrix_analysis'] and old_manifest['status'] == 'PASS'
    design = json.loads((Path(__file__).parent/'DESIGN_AMENDMENT.json').read_text())
    assert design['n_starts_total'] == 4 and design['max_strict_NNI_steps_per_start'] == MAX_STEPS
    registry = pd.read_csv(root/'simulation/results/matrix_registry.csv').sort_values(['scenario_id', 'replicate'])
    if args.limit: registry = registry.head(args.limit)
    payload = [(str(root), str(original), str(out), identifier) for identifier in registry.matrix_id]
    started = time.monotonic(); rows = []; receipts = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, (part, receipt) in enumerate(pool.map(amend_one, payload), 1):
            rows.extend(part); receipts.append(receipt)
            if i % 25 == 0 or i == len(payload):
                print(f'NJ amendment complete: {i}/{len(payload)} matrices; {time.monotonic()-started:.1f}s', flush=True)
    raw = pd.DataFrame(rows); matrix, metrics = summarize_matrix(rows)
    raw.to_csv(out/'selected_completion_topology.csv', index=False)
    matrix.to_csv(out/'matrix_topology_summary.csv', index=False)
    scenario_summaries(matrix, metrics).to_csv(out/'scenario_topology_summary.csv', index=False)
    paired_scheme_summaries(matrix).to_csv(out/'paired_scheme_contrasts.csv', index=False)
    marginal = matrix[(matrix.kind == 'map') & (matrix.scheme == 'nj_mk')].set_index('matrix_id')
    joint = matrix[(matrix.kind == 'map') & (matrix.scheme == 'nj_mk_joint')].set_index('matrix_id').loc[marginal.index]
    np.testing.assert_allclose(marginal[metrics], joint[metrics], equal_nan=True)
    manifest = {'status': 'PASS', 'analysis': 'Explicitly post hoc four-start amendment',
                'n_source_matrices': len(registry), 'complete_600_matrix_analysis': len(registry) == 600 and args.limit is None,
                'n_treatment_records': len(raw), 'n_matrix_summary_rows': len(matrix),
                'n_new_NJ_starts_searched': sum(item['n_new_NJ_starts_searched'] for item in receipts),
                'n_treatment_records_with_step_limit': int(raw.any_start_step_limit.sum()),
                'seconds': time.monotonic()-started, 'design': design,
                'design_sha256': sha(Path(__file__).parent/'DESIGN_AMENDMENT.json'),
                'search_code_sha256': sha(Path(__file__).parent/'nni_search.py'),
                'NJ_code_sha256': sha(Path(__file__).parent/'nj_start.py'), 'driver_sha256': sha(__file__),
                'original_run_manifest_sha256': sha(original/'RUN_MANIFEST.json'),
                'python': sys.version, 'numpy': np.__version__, 'pandas': pd.__version__,
                'search_is_heuristic': True, 'generating_tree_used_only_for_post_search_metrics': True,
                'independent_inference_unit': 'matrix; eight selected draws averaged before MCSE',
                'matrix_receipts': receipts}
    (out/'RUN_MANIFEST.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')
    print(json.dumps({key: manifest[key] for key in ('status', 'n_source_matrices', 'n_treatment_records',
                                                   'n_new_NJ_starts_searched', 'n_treatment_records_with_step_limit', 'seconds')}, indent=2))


if __name__ == '__main__':
    main()
