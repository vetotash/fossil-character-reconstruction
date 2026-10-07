#!/usr/bin/env python3
"""R17 paired empirical-layout simulations, generation and evaluation only."""
from __future__ import annotations
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import time
import numpy as np

HERE = Path(__file__).resolve().parent
PANELS = ('tetrapods', 'dinosaurs', 'ichthyosaurs')
METHODS = ('mode', 'knn5', 'logistic', 'extra_trees', 'nj_mk')
SCHEMES = (*METHODS, 'nj_mk_joint')
ARMS = ('empirical', 'permuted')
GENERATION_SEED = 2026100417
MASK_SEED = 2026100418
N_DRAWS = 512
LOG_FLOOR = 1e-12

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2) + '\n')

def write_csv(path, rows):
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def orient(edges, n_tips, root=None):
    edges = np.asarray(edges, dtype=np.int16)
    n_nodes = int(edges.max()) + 1
    root = n_tips if root is None else int(root)
    adjacency = [[] for _ in range(n_nodes)]
    for a, b in edges:
        adjacency[int(a)].append(int(b)); adjacency[int(b)].append(int(a))
    assert len(edges) == n_nodes - 1
    assert all(len(adjacency[i]) == 1 for i in range(n_tips))
    parent = np.full(n_nodes, -1, dtype=np.int16)
    order, children = [root], [[] for _ in range(n_nodes)]
    visited = {root}
    for a in order:
        for b in sorted(adjacency[a]):
            if b == parent[a]:
                continue
            assert b not in visited
            visited.add(b); parent[b] = a; children[a].append(b); order.append(b)
    assert len(order) == n_nodes
    return root, order, children

def generate_states(edges, n_tips, n_characters, q, rng, root=None):
    root, order, children = orient(edges, n_tips, root)
    states = np.zeros((len(order), n_characters), dtype=np.uint8)
    states[root] = rng.integers(0, 2, n_characters, dtype=np.uint8)
    flip = -np.expm1(-2*q)/2
    for node in order:
        for child in children[node]:
            states[child] = states[node] ^ (rng.random(n_characters) < flip)
    return states

def score_patterns(patterns, edges, root=None):
    patterns = np.atleast_2d(np.asarray(patterns, dtype=np.int8))
    n_patterns, n_tips = patterns.shape
    root, order, children = orient(edges, n_tips, root)
    costs = [None] * len(order)
    for node in reversed(order):
        if node < n_tips:
            costs[node] = np.column_stack((patterns[:, node] == 1, patterns[:, node] == 0)).astype(np.int16) * (n_tips+1)
        else:
            value = np.zeros((n_patterns, 2), dtype=np.int16)
            for child in children[node]:
                cc = costs[child]
                value[:, 0] += np.minimum(cc[:, 0], cc[:, 1]+1)
                value[:, 1] += np.minimum(cc[:, 1], cc[:, 0]+1)
                costs[child] = None
            costs[node] = value
    return costs[root].min(1)

def minima(patterns):
    patterns = np.atleast_2d(patterns)
    return (np.any(patterns == 0, axis=1) & np.any(patterns == 1, axis=1)).astype(np.int16)

def total_costs(matrix, edges, root=None):
    return int(score_patterns(matrix.T, edges, root).sum()), int(minima(matrix.T).sum())

def completion_costs(masked, target_mask, states, edges, root=None):
    """Only target columns require rescoring; non-target missing remain -1."""
    targets = np.argwhere(target_mask)
    columns = np.unique(targets[:, 1])
    states = np.atleast_2d(states)
    assert states.shape[1] == len(targets)
    col_lookup = np.full(masked.shape[1], -1, dtype=int)
    col_lookup[columns] = np.arange(len(columns))
    sub = masked[:, columns]
    old_l, old_m = total_costs(masked, edges, root)
    sub_l, sub_m = total_costs(sub, edges, root)
    completed = np.broadcast_to(sub, (len(states), *sub.shape)).copy()
    completed[:, targets[:, 0], col_lookup[targets[:, 1]]] = states
    assert np.all(completed[:, sub == -1][:, ~target_mask[:, columns][sub == -1]] == -1)
    patterns = completed.transpose(0, 2, 1).reshape(-1, masked.shape[0])
    L = score_patterns(patterns, edges, root).reshape(len(states), len(columns)).sum(1) + old_l-sub_l
    M = minima(patterns).reshape(len(states), len(columns)).sum(1) + old_m-sub_m
    return L, M

def generation(base, config_file, frozen_design):
    base, config_file, frozen_design = Path(base), Path(config_file), Path(frozen_design)
    config = json.loads(config_file.read_text())
    if not frozen_design.is_file():
        raise FileNotFoundError('A root-approved frozen design is required')
    if base.exists():
        raise FileExistsError(base)
    (base/'inputs').mkdir(parents=True); (base/'truth').mkdir()
    entries, registry, scenarios, scaffolds = [], [], [], {}
    for panel_index, panel in enumerate(PANELS):
        specification = config['panels'][panel]
        template_path = (config_file.parent / specification['template']).resolve()
        trees_path = (config_file.parent / specification['trees']).resolve()
        assert sha256(template_path) == specification['template_sha256']
        assert sha256(trees_path) == specification['trees_sha256']
        with np.load(template_path) as data:
            labels = data['labels'].astype(np.uint8)
            names = data['names']
            original_columns = data['original_column_indices']
        trees = json.loads(trees_path.read_text())
        tree = next(t for t in trees if t['name'] == specification['tree_name'])
        edges, root = np.asarray(tree['edges'], dtype=np.int16), int(tree['root'])
        n, p = labels.shape
        actual_root, traversal, children = orient(edges, n, root)
        scaffolds[panel] = {'tree_name': tree['name'], 'root': actual_root, 'traversal': traversal, 'children': children, 'tip_order': names.tolist(), 'edges': edges.tolist(), 'edge_lengths': [1.0]*len(edges)}
        assert n == tree['n_tips'] and np.isin(labels, [0,1,2]).all()
        assert np.all((labels == 0).sum(0) > 0)
        for q_index, q in enumerate(config['rates']):
            ordinal = 2*panel_index + q_index
            scenario_id = f's{ordinal:02d}'
            scenario = {'scenario_id': scenario_id, 'scenario_ordinal': ordinal, 'panel': panel, 'q': q,
                        'n_tips': n, 'n_characters': p, 'n_targets': int((labels == 2).sum()),
                        'target_columns': int((labels == 2).any(0).sum()), 'replicates': config['replicates'],
                        'tree_name': tree['name'], 'template_sha256': specification['template_sha256']}
            scenarios.append(scenario)
            for rep in range(config['replicates']):
                pair_id = f'{scenario_id}_r{rep:03d}'
                rng = np.random.default_rng(np.random.SeedSequence([GENERATION_SEED, panel_index, q_index, rep]))
                node_states = generate_states(edges, n, p, q, rng, root)
                truth = node_states[:n]
                truth_path = base/'truth'/f'{pair_id}.npz'
                np.savez_compressed(truth_path, X_true=truth, node_states=node_states, edges=edges,
                                    root=np.array(root), q=np.array(q), names=names,
                                    original_column_indices=original_columns)
                shuffle_rng = np.random.default_rng(np.random.SeedSequence([MASK_SEED, panel_index, q_index, rep]))
                permuted = np.column_stack([shuffle_rng.permutation(labels[:, j]) for j in range(p)])
                assert np.array_equal(np.stack([(permuted == k).sum(0) for k in range(3)]),
                                      np.stack([(labels == k).sum(0) for k in range(3)]))
                for arm_index, arm_labels in enumerate((labels, permuted)):
                    matrix_id = f'{pair_id}_a{arm_index}'
                    masked = truth.astype(np.int8).copy(); masked[arm_labels != 0] = -1
                    path = base/'inputs'/f'{matrix_id}.npz'
                    np.savez_compressed(path, X=masked, target_mask=arm_labels == 2, labels=arm_labels)
                    entry = {'matrix_id': matrix_id, 'pair_id': pair_id, 'scenario_ordinal': ordinal,
                             'replicate': rep, 'panel_ordinal': panel_index, 'q_ordinal': q_index, 'arm_ordinal': arm_index, 'arm': ARMS[arm_index],
                             'masked_file': str(path.relative_to(base)), 'masked_sha256': sha256(path)}
                    entries.append(entry)
                    registry.append({**scenario, **entry, 'truth_sha256': sha256(truth_path)})
    write_json(base/'input_manifest.json', entries)
    write_csv(base/'matrix_registry.csv', registry); write_csv(base/'scenario_registry.csv', scenarios)
    write_json(base/'GENERATION_MANIFEST.json', {'paired_matrices': len(entries)//2, 'arm_matrices': len(entries),
               'frozen_design_sha256': sha256(frozen_design), 'config_sha256': sha256(config_file),
               'script_sha256': sha256(__file__), 'generation_seed': GENERATION_SEED, 'mask_seed': MASK_SEED,
               'config': config, 'scaffolds': scaffolds})
    print(json.dumps({'generated_pairs': len(entries)//2, 'generated_arms': len(entries)}), flush=True)

def metrics(probability, y):
    ptrue = probability[np.arange(len(y)), y]
    return {'accuracy': float(np.mean(probability.argmax(1) == y)),
            'brier': float(np.mean(np.sum((probability-np.eye(2)[y])**2, axis=1))),
            'log_loss': float(np.mean(-np.log(np.maximum(ptrue, LOG_FLOOR)))),
            'zero_true_probability_count': int(np.count_nonzero(ptrue == 0)),
            'log_floor_count': int(np.count_nonzero(ptrue < LOG_FLOOR)),
            'mean_max_probability': float(np.mean(probability.max(1)))}

def evaluate_one(base, entry):
    base = Path(base); matrix_id = entry['matrix_id']
    meta_path = base/'predictions'/f'{matrix_id}.json'
    meta = json.loads(meta_path.read_text())
    pred_path = base/'predictions'/f'{matrix_id}.npz'
    assert sha256(pred_path) == meta['prediction_sha256']
    assert sha256(base/entry['masked_file']) == entry['masked_sha256']
    inputs = np.load(base/entry['masked_file'])
    masked, target_mask = inputs['X'], inputs['target_mask'].astype(bool)
    targets = np.argwhere(target_mask); prediction = np.load(pred_path)
    assert np.array_equal(targets, prediction['target_indices'])
    # Truth is opened only after all arms have fitted and hashes are verified.
    truth_data = np.load(base/'truth'/f"{entry['pair_id']}.npz")
    truth, edges, root = truth_data['X_true'], truth_data['edges'], int(truth_data['root'])
    y = truth[target_mask]
    old_l, old_m = total_costs(masked, edges, root)
    true_l, true_m = completion_costs(masked, target_mask, y[None, :], edges, root)
    true_l, true_m = int(true_l[0]), int(true_m[0]); true_d = true_l-old_l-(true_m-old_m)
    rows, ds, ls, ms, hards = [], [], [], [], []
    for k, scheme in enumerate(SCHEMES):
        probability = prediction['probabilities'][min(k, 4)]
        hard = probability.argmax(1).astype(np.uint8)
        hard_l, hard_m = completion_costs(masked, target_mask, hard[None, :], edges, root)
        hard_d = int(hard_l[0]-old_l-(hard_m[0]-old_m))
        draws = np.unpackbits(prediction['missing_states_packed'][k], axis=1, count=len(targets))
        assert hashlib.sha256(draws.tobytes()).hexdigest() == meta['completion_sha256'][k]
        l, m = completion_costs(masked, target_mask, draws, edges, root)
        d = l-old_l-(m-old_m)
        assert np.all(d >= 0) and true_d >= 0 and hard_d >= 0
        low, high = np.quantile(d, [.025, .975], method='linear')
        error = float(d.mean()-true_d)
        rows.append({'matrix_id': matrix_id, 'pair_id': entry['pair_id'], 'scenario_id': f"s{entry['scenario_ordinal']:02d}",
                     'replicate': entry['replicate'], 'arm': entry['arm'], 'scheme': scheme,
                     'n_targets': len(targets), **metrics(probability, y), 'old_L': old_l, 'old_M': old_m,
                     'true_L': true_l, 'true_M': true_m, 'true_D': true_d, 'hard_D': hard_d,
                     'hard_D_error': hard_d-true_d, 'hard_D_absolute_error': abs(hard_d-true_d),
                     'mean_D': float(d.mean()), 'mean_D_error': error, 'mean_D_absolute_error': abs(error),
                     'mean_D_error_per_100_targets': 100*error/len(targets),
                     'mean_D_absolute_error_per_100_targets': 100*abs(error)/len(targets),
                     'sd_D': float(d.std(ddof=1)), 'mean_D_mcse': float(d.std(ddof=1)/math.sqrt(N_DRAWS)),
                     'lower95': float(low), 'upper95': float(high), 'interval_width': float(high-low),
                     'truth_in_interval': int(low <= true_d <= high), 'truth_below_interval': int(true_d < low),
                     'truth_above_interval': int(true_d > high),
                     'constant_response_count': meta['constant_response_count'],
                     'convergence_warning_count': meta['convergence_warning_count'], 'n_draws': N_DRAWS})
        ds.append(d); ls.append(l); ms.append(m); hards.append(hard_d)
    out = base/'evaluation'/f'{matrix_id}.npz'
    np.savez_compressed(out, D_samples=np.asarray(ds, dtype=np.int32), L_samples=np.asarray(ls, dtype=np.int32),
                        M_samples=np.asarray(ms, dtype=np.int32), hard_D=np.asarray(hards, dtype=np.int32),
                        old_L=np.array(old_l), old_M=np.array(old_m), true_L=np.array(true_l), true_M=np.array(true_m),
                        true_D=np.array(true_d))
    return rows

def wilson(k, n):
    z = 1.959963984540054; p = k/n
    center = (p+z*z/(2*n))/(1+z*z/n)
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return center-half, center+half

METRICS = ('accuracy', 'brier', 'log_loss', 'mean_max_probability', 'true_D', 'hard_D', 'hard_D_error',
           'hard_D_absolute_error', 'mean_D', 'mean_D_error', 'mean_D_absolute_error',
           'mean_D_error_per_100_targets', 'mean_D_absolute_error_per_100_targets', 'sd_D',
           'mean_D_mcse', 'interval_width', 'constant_response_count', 'convergence_warning_count')
PAIRED_METRICS = ('accuracy', 'brier', 'log_loss', 'mean_D_error', 'mean_D_absolute_error',
                  'mean_D_error_per_100_targets', 'mean_D_absolute_error_per_100_targets',
                  'interval_width', 'truth_in_interval')

def summarize(base, rows):
    scenarios = list(csv.DictReader((base/'scenario_registry.csv').open()))
    summaries, paired = [], []
    for scenario, arm, scheme in itertools.product(scenarios, ARMS, SCHEMES):
        subset = [r for r in rows if r['scenario_id'] == scenario['scenario_id'] and r['arm'] == arm and r['scheme'] == scheme]
        n = len(subset); hit = sum(r['truth_in_interval'] for r in subset)
        low, high = wilson(hit, n)
        row = {**scenario, 'arm': arm, 'scheme': scheme, 'n_matrices': n, 'interval_coverage': hit/n,
               'coverage_mcse': math.sqrt((hit/n)*(1-hit/n)/n), 'coverage_wilson_lower95': low,
               'coverage_wilson_upper95': high, 'count_truth_below_interval': sum(r['truth_below_interval'] for r in subset),
               'count_truth_above_interval': sum(r['truth_above_interval'] for r in subset)}
        for key in METRICS:
            values = np.asarray([r[key] for r in subset])
            row[key+'_mean'] = float(values.mean()); row[key+'_mcse'] = float(values.std(ddof=1)/math.sqrt(n))
        row['zero_true_probability_total'] = sum(r['zero_true_probability_count'] for r in subset)
        row['log_floor_total'] = sum(r['log_floor_count'] for r in subset)
        summaries.append(row)
    lookup = {(r['pair_id'], r['arm'], r['scheme']): r for r in rows}
    for scenario, scheme in itertools.product(scenarios, SCHEMES):
        ids = [r['pair_id'] for r in rows if r['scenario_id'] == scenario['scenario_id'] and r['arm'] == 'empirical' and r['scheme'] == scheme]
        record = {**scenario, 'scheme': scheme, 'n_pairs': len(ids), 'difference_direction': 'empirical_minus_permuted'}
        for metric in PAIRED_METRICS:
            differences = np.asarray([lookup[(i,'empirical',scheme)][metric]-lookup[(i,'permuted',scheme)][metric] for i in ids])
            record[metric+'_difference_mean'] = float(differences.mean())
            record[metric+'_difference_mcse'] = float(differences.std(ddof=1)/math.sqrt(len(ids)))
        paired.append(record)
    write_csv(base/'matrix_results.csv', rows); write_csv(base/'scenario_summary.csv', summaries)
    write_csv(base/'paired_layout_contrasts.csv', paired)

def evaluation(base):
    base = Path(base)
    fit = json.loads((base/'FIT_MANIFEST.json').read_text())
    entries = json.loads((base/'input_manifest.json').read_text())
    assert not fit['failed_arms'] and fit['completed_arms'] == len(entries)
    for record in fit['records']:
        assert sha256(base/'predictions'/f"{record['matrix_id']}.npz") == record['prediction_sha256']
        assert sha256(base/'predictions'/f"{record['matrix_id']}.json") == record['metadata_sha256']
    (base/'evaluation').mkdir(exist_ok=False)
    started = time.monotonic(); rows = []
    for i, entry in enumerate(entries):
        rows.extend(evaluate_one(base, entry))
        if (i+1) % 10 == 0:
            print(json.dumps({'evaluated_arms': i+1, 'total_arms': len(entries), 'seconds': round(time.monotonic()-started,2)}), flush=True)
    summarize(base, rows)
    write_json(base/'EVALUATION_MANIFEST.json', {'arm_matrices': len(entries), 'paired_matrices': len(entries)//2,
               'rows': len(rows), 'seconds': time.monotonic()-started, 'script_sha256': sha256(__file__),
               'fit_manifest_sha256': sha256(base/'FIT_MANIFEST.json')})

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['generate','evaluate'])
    p.add_argument('--output', type=Path, required=True); p.add_argument('--config', type=Path)
    p.add_argument('--frozen-design', type=Path); a = p.parse_args()
    generation(a.output, a.config, a.frozen_design) if a.action == 'generate' else evaluation(a.output)
