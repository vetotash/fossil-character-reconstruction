#!/usr/bin/env python3
"""Audit tip/row binding, independent scores/splits, and baseline search gaps."""
from __future__ import annotations
import argparse
import gzip
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd
from nni_search import full_character_scores, split_set


def cut_edge_splits(edges, n_tips):
    """Independent split implementation: remove each edge and traverse a side."""
    adj = [[] for _ in range(2*n_tips-2)]
    for a, b in edges:
        adj[int(a)].append(int(b)); adj[int(b)].append(int(a))
    result = set(); full = (1 << n_tips) - 1
    for a, b in edges:
        a, b = int(a), int(b); seen = {a}; queue = [a]
        for node in queue:
            for neighbour in adj[node]:
                if (node == a and neighbour == b) or (node == b and neighbour == a): continue
                if neighbour not in seen: seen.add(neighbour); queue.append(neighbour)
        tips = [node for node in seen if node < n_tips]
        if 1 < len(tips) < n_tips-1:
            mask = sum(1 << node for node in tips)
            result.add(min(mask, full ^ mask))
    assert len(result) == n_tips-3
    return frozenset(result)


def audit(root, original, amended, output):
    spec = importlib.util.spec_from_file_location('independent_original_scoring', root/'simulation/experiment.py')
    source = importlib.util.module_from_spec(spec); spec.loader.exec_module(source)
    base = root/'simulation/results'; rows = []; n_endpoints = 0
    for registry in pd.read_csv(base/'matrix_registry.csv').itertuples():
        identifier = registry.matrix_id
        with np.load(base/'truth'/f'{identifier}_truth.npz', allow_pickle=False) as archive:
            truth = archive['X_true']; truth_edges = archive['edges']
            assert np.array_equal(truth, archive['node_states'][:len(truth)])
        with np.load(base/'inputs'/f'{identifier}_masked.npz', allow_pickle=False) as archive:
            masked = archive['X']
        assert np.array_equal(masked[masked >= 0], truth[masked >= 0])
        true_splits = cut_edge_splits(truth_edges, len(truth))
        assert true_splits == split_set(truth_edges, len(truth))
        for label, directory in [('three_random_starts', original), ('four_starts_with_NJ', amended)]:
            with gzip.open(directory/'search_records'/f'{identifier}.json.gz', 'rt', encoding='utf8') as handle:
                receipt = json.load(handle)
            for scheme, matrix in [('source_incomplete', masked), ('truth_complete', truth)]:
                row = next(item for item in receipt['treatment_results'] if item['scheme'] == scheme)
                record = receipt['searches'][row['searched_matrix_sha256']]
                generating_scores = full_character_scores(matrix, truth_edges)
                np.testing.assert_array_equal(generating_scores, source.parsimony_costs(matrix.T, truth_edges))
                errors = []
                for edges in record['retained_edges']:
                    scores = source.parsimony_costs(matrix.T, edges)
                    np.testing.assert_array_equal(scores, full_character_scores(matrix, edges))
                    assert int(scores.sum()) == record['minimum_found_length']
                    estimated = cut_edge_splits(edges, len(matrix))
                    assert estimated == split_set(edges, len(matrix))
                    errors.append(len(estimated ^ true_splits)/(2*(len(matrix)-3)))
                    n_endpoints += 1
                assert abs(np.mean(errors)-row['mean_normalized_RF']) < 1e-12
                rows.append(dict(matrix_id=identifier, scenario_id=registry.scenario_id,
                                 search_design=label, baseline=scheme,
                                 generating_tree_length=int(generating_scores.sum()),
                                 minimum_found_length=record['minimum_found_length'],
                                 length_minus_generating=int(record['minimum_found_length']-generating_scores.sum()),
                                 mean_normalized_RF=float(np.mean(errors))))
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows); frame.to_csv(output/'baseline_recovery_audit.csv', index=False)
    summary = frame.groupby(['search_design', 'scenario_id', 'baseline']).agg(
        n_matrices=('matrix_id', 'size'), mean_length_gap=('length_minus_generating', 'mean'),
        n_length_above_generating=('length_minus_generating', lambda x: int((x>0).sum())),
        mean_normalized_RF=('mean_normalized_RF', 'mean')).reset_index()
    summary.to_csv(output/'baseline_recovery_by_scenario.csv', index=False)
    receipt = dict(status='PASS', source_matrices=600, baseline_design_records=len(frame),
                   independently_rescored_retained_endpoints=n_endpoints,
                   X_true_equals_first_48_node_states=True, observed_cells_match_truth=True,
                   score_check='New Fitch score equals original binary Sankoff per character',
                   split_check='Independent edge-cut traversal matches recursive bit-mask splits',
                   RF_check='Independent edge-cut split RF matches all baseline reported values',
                   interpretation='A score above the generating-tree score certifies failure to reach a global minimum; a score at or below it does not prove global optimality.')
    (output/'BASELINE_AUDIT.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--original-results', type=Path, required=True)
    parser.add_argument('--amended-results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit(args.root.resolve(), args.original_results.resolve(), args.amended_results.resolve(), args.output.resolve())
