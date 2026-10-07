"""Numerical validation of saved simulation arrays, with optional model refits.

Portable: python validation/validate_results.py --root PATH --no-refit
No scientific production functions are imported. Optional predetermined refits
call the adapter only; all scoring and summary arithmetic are independent.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
from independent_engine import matrix_lengths, self_test, wilson

METHODS = ('mode', 'knn5', 'logistic', 'extra_trees', 'nj_mk')
SCHEMES = METHODS + ('nj_mk_joint',)
GEN_SEED, MASK_SEED, PROP_SEED = 2026100403, 2026100404, 2026100405
ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_close(actual, expected, label, atol=2e-10):
    a, b = np.asarray(actual), np.asarray(expected)
    assert a.shape == b.shape, (label, a.shape, b.shape)
    if not np.allclose(a, b, rtol=1e-11, atol=atol):
        raise AssertionError((label, float(np.max(np.abs(a-b))), a, b))


def replay_inputs(s, rep, q, k, mode):
    tree_rng = np.random.default_rng(np.random.SeedSequence([GEN_SEED, s, rep, 0]))
    edges = [[0, 48], [1, 48], [2, 48]]
    for tip in range(3, 48):
        chosen = int(tree_rng.integers(0, len(edges)))
        a, b = edges[chosen]
        del edges[chosen]
        node = 48 + tip - 2
        edges += [[a, node], [b, node], [tip, node]]
    edges = np.asarray(edges, dtype=int)
    adjacency = {n: [] for n in range(94)}
    for a, b in edges:
        adjacency[a].append(b); adjacency[b].append(a)
    parents = {48: None}; queue = [48]
    children = {}
    for parent in queue:
        children[parent] = [x for x in sorted(adjacency[parent]) if x != parents[parent]]
        for child in children[parent]:
            assert child not in parents
            parents[child] = parent
            queue.append(child)
    rng = np.random.default_rng(np.random.SeedSequence([GEN_SEED, s, rep, 1]))
    states = np.empty((94, 40), dtype=np.uint8)
    states[48] = rng.integers(0, 2, size=40, dtype=np.uint8)
    flip = -np.expm1(-2*q)/2
    for parent in queue:
        for child in children[parent]:
            states[child] = np.where(rng.random(40) < flip, 1-states[parent], states[parent])
    rng = np.random.default_rng(np.random.SeedSequence([MASK_SEED, s, rep]))
    mask = np.zeros((48, 40), dtype=bool)
    anchor = -1
    if mode == 'structured':
        anchor = int(rng.integers(48))
        distance = {anchor: 0}; nodes = [anchor]
        for node in nodes:
            for child in adjacency[node]:
                if child not in distance:
                    distance[child] = distance[node] + 1
                    nodes.append(child)
        tips = sorted(range(48), key=lambda x: (distance[x], x))[:k]
        mask[tips, 8:] = True
    else:
        for char in range(8, 40):
            probability = None
            if mode == 'statebiased':
                probability = np.where(states[:48, char] == 1, 4., 1.)
                probability /= sum(probability)
            tips = rng.choice(48, size=k, replace=False, p=probability)
            mask[tips, char] = True
    x = states[:48].astype(np.int8).copy()
    x[mask] = -1
    return edges, states, x, anchor


def independent_joint_replay(x, models, seed):
    """Replay binary conditional sampling without importing the Mk module."""
    locations = np.argwhere(x == -1)
    output = np.empty((512, len(locations)), dtype=np.uint8)
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    for char in range(8, 40):
        model = models[char]
        columns = np.flatnonzero(locations[:, 1] == char)
        tips = locations[columns, 0]
        if model['constant_fallback']:
            output[:, columns] = int(model['classes'][0])
            continue
        assert model['classes'] == ['0', '1']
        edges = model['edges']; count = 1 + max(max(a, b) for a, b, t in edges)
        adjacency = {i: [] for i in range(count)}
        for a, b, length in edges:
            adjacency[a].append((b, length)); adjacency[b].append((a, length))
        root = count-1
        queue = [root]; parent = {root: None}; children = {}; branch = {}
        for node in queue:
            children[node] = []
            for child, length in adjacency[node]:
                if child != parent[node]:
                    assert child not in parent
                    parent[child] = node; children[node].append(child)
                    branch[child] = length; queue.append(child)
        emissions = np.ones((count, 2))
        train = np.flatnonzero(x[:, char] != -1)
        emissions[train] = np.eye(2)[x[train, char]]
        upward = np.ones((count, 2)); message = np.ones((count, 2)); transition = {}
        for node in reversed(queue):
            likelihood = emissions[node].copy()
            for child in children[node]:
                likelihood *= message[child]
            upward[node] = likelihood / likelihood.max()
            if node != root:
                e = np.exp(-2*model['rate']*branch[node])
                change = (1-e)/2
                transition[node] = np.array([[change+e, change], [change, change+e]])
                message[node] = transition[node] @ upward[node]
        draw_states = np.empty((512, count), dtype=np.uint8)
        root_p = upward[root]/upward[root].sum()
        draw_states[:, root] = rng.choice(2, size=512, p=root_p)
        for node in queue:
            for child in children[node]:
                weights = transition[child] * upward[child]
                prob_zero = weights[:, 0] / weights.sum(axis=1)
                draw_states[:, child] = rng.random(512) > prob_zero[draw_states[:, node]]
        assert np.array_equal(draw_states[:, train], np.broadcast_to(x[train, char], (512, len(train))))
        output[:, columns] = draw_states[:, tips]
    return output


def audit_matrix(arguments):
    root_s, entry, scenario, refit = arguments
    root = Path(root_s); base = root / 'simulation/results'
    identifier = entry['matrix_id']
    s, rep = entry['scenario_ordinal'], entry['replicate']
    x_file = base / entry['masked_file']
    p_file = base / 'predictions' / (identifier + '.npz')
    p_meta = json.loads((p_file.with_suffix('.json')).read_text())
    t_file = base / 'truth' / (identifier + '_truth.npz')
    e_file = base / 'evaluation' / (identifier + '.npz')
    e_meta = json.loads(e_file.with_suffix('.json').read_text())
    assert digest(x_file) == entry['masked_sha256'] == p_meta['masked_sha256']
    assert digest(p_file) == p_meta['prediction_sha256'] == e_meta['prediction_sha256_verified_before_truth_read']
    assert digest(e_file) == e_meta['result_sha256']
    assert p_meta['adapter_sha256'] == digest(root / 'methods/fit_adapter.py')
    assert p_meta['mk_module_sha256'] == digest(root / 'methods/mk_baseline.py')
    x = np.load(x_file)['X']
    with np.load(t_file) as t:
        truth, edges = t['X_true'], t['edges']
        re_edges, re_states, re_x, re_anchor = replay_inputs(s, rep, scenario['q'], scenario['missing_per_response'], scenario['mask_mode'])
        assert np.array_equal(edges, re_edges)
        assert np.array_equal(t['node_states'], re_states)
        assert np.array_equal(truth, re_states[:48])
        assert np.array_equal(x, re_x)
        assert float(t['q']) == scenario['q'] and int(t['root']) == 48
        assert int(t['mask_anchor']) == re_anchor
        assert np.all(t['edge_lengths'] == 1)
    mask = x == -1
    assert not mask[:, :8].any()
    assert np.all(mask[:, 8:].sum(0) == scenario['missing_per_response'])
    assert np.array_equal(x[~mask], truth[~mask])
    assert x.shape == truth.shape == (48, 40)
    locations = np.argwhere(mask); n_targets = len(locations)
    with np.load(p_file) as p:
        probabilities = p['probabilities']
        assert np.array_equal(p['target_indices'], locations)
        packed = p['missing_states_packed']
    assert probabilities.shape == (5, 48, 40, 2)
    assert packed.shape[:2] == (6, 512)
    assert np.isfinite(probabilities).all() and (probabilities >= 0).all()
    check_close(probabilities.sum(-1), np.ones((5, 48, 40)), identifier + ' probability normalization')
    expected_observed = np.eye(2)[truth[~mask]]
    for i in range(5):
        assert np.array_equal(probabilities[i][~mask], expected_observed)
    expected_seeds = [PROP_SEED, s, rep, 0]
    assert p_meta['marginal_uniform_seed_sequence'] == e_meta['marginal_uniform_seed_sequence'] == expected_seeds
    assert p_meta['joint_seed_sequence'] == e_meta['joint_seed_sequence'] == [PROP_SEED, s, rep, 1]
    uniform = np.random.default_rng(np.random.SeedSequence(expected_seeds)).random((512, n_targets))
    saved_draws = np.unpackbits(packed, axis=-1, count=n_targets).astype(np.uint8)
    for i in range(5):
        expected = uniform < probabilities[i, locations[:, 0], locations[:, 1], 1]
        assert np.array_equal(saved_draws[i], expected)
    replayed_joint = independent_joint_replay(x, p_meta['nj_models'], [PROP_SEED, s, rep, 1])
    assert np.array_equal(saved_draws[5], replayed_joint), (identifier, 'joint draw replay')
    for i, scheme in enumerate(SCHEMES):
        d_sha = hashlib.sha256(saved_draws[i].tobytes()).hexdigest()
        assert d_sha == p_meta['completion_sha256'][i] == e_meta['completion_sha256_by_scheme'][scheme]
    # Verify visible-response and class contracts for every fitted character.
    assert p_meta['status'] == 'completed' and not p_meta['failures']
    assert [d['character'] for d in p_meta['diagnostics']] == list(range(8, 40))
    counters = {'constant_response_count': 0, 'convergence_warning_count': 0,
                'logistic_limit_count': 0, 'mk_failed_optimizer_count': 0,
                'mk_boundary_count': 0, 'nj_zero_overlap_pairs': 0,
                'nj_negative_branches': 0, 'nj_truncated_branches': 0}
    for diagnostic in p_meta['diagnostics']:
        char = diagnostic['character']; model = p_meta['nj_models'][char]
        observed = np.flatnonzero(~mask[:, char])
        labels = x[observed, char].astype(str)
        classes = np.unique(labels).tolist()
        assert diagnostic['status'] == 'completed'
        assert diagnostic['n_train'] == len(observed) and diagnostic['n_test'] == scenario['missing_per_response']
        assert diagnostic['observed_classes'] == classes == model['classes']
        assert model['train_indices'] == observed.tolist() and model['train_states'] == labels.tolist()
        is_constant = len(classes) == 1
        assert diagnostic['constant_fallback'] == model['constant_fallback'] == is_constant
        counters['constant_response_count'] += int(is_constant)
        if is_constant:
            for method in range(5):
                assert np.all(probabilities[method, mask[:, char], char, int(classes[0])] == 1)
                assert np.all(probabilities[method, mask[:, char], char, 1-int(classes[0])] == 0)
        else:
            values = np.where(x == -1, '?', x.astype(str))
            categories = [np.unique(col).tolist() for col in np.delete(values, char, axis=1)[observed].T]
            assert diagnostic['onehot_categories'] == categories
        logistic = diagnostic['classifiers']['logistic']
        counters['convergence_warning_count'] += int(logistic.get('convergence_warning', False))
        counters['logistic_limit_count'] += int(logistic.get('reached_max_iter', False))
        mk = diagnostic['nj_mk']; optimizer = mk['optimizer']
        assert optimizer['ran'] == (not is_constant)
        counters['mk_failed_optimizer_count'] += int(optimizer['ran'] and not optimizer['success'])
        counters['mk_boundary_count'] += int(mk['rate_boundary'])
        counters['nj_zero_overlap_pairs'] += int(mk['zero_overlap_pairs'])
        counters['nj_negative_branches'] += int(mk['negative_branches'])
        counters['nj_truncated_branches'] += int(mk['truncated_branches'])
    assert counters['constant_response_count'] == p_meta['constant_response_count']
    assert counters['convergence_warning_count'] == p_meta['convergence_warning_count']
    refit_count = 0
    if refit and rep == 0:
        sys.path.insert(0, str(root / 'methods'))
        import fit_adapter
        values = np.where(x == -1, '?', x.astype(str))
        for char in (8, 39):
            fitted = fit_adapter.fit_character(values, char)
            targets = np.flatnonzero(mask[:, char])
            for method, name in enumerate(METHODS):
                cls, p = fitted['methods'][name]
                expected = np.zeros((len(targets), 2))
                for col, state in enumerate(cls):
                    expected[:, int(state)] = p[:, col]
                assert np.array_equal(probabilities[method, targets, char], expected)
            refit_count += 1
    old_l, old_m = matrix_lengths(x, edges)
    true_l, true_m = matrix_lengths(truth, edges)
    true_d = int(true_l-old_l-true_m+old_m)
    eval_saved = np.load(e_file)
    for name, expected in [('old_L', old_l), ('old_M', old_m), ('true_L', true_l), ('true_M', true_m), ('true_D', true_d)]:
        assert int(eval_saved[name]) == int(expected), (identifier, name)
    target_truth = truth[mask]
    rows = []
    for i, scheme in enumerate(SCHEMES):
        method = min(i, 4)
        p = probabilities[method][mask]
        hard = np.argmax(p, axis=1).astype(np.uint8)
        completion = x.copy(); completion[mask] = hard
        hard_l, hard_m = matrix_lengths(completion, edges)
        hard_d = int(hard_l-old_l-hard_m+old_m)
        assert hard_d == int(eval_saved['hard_D'][i]), (identifier, scheme, 'hard_D')
        completions = np.broadcast_to(x, (512, 48, 40)).copy()
        completions[:, locations[:, 0], locations[:, 1]] = saved_draws[i]
        lengths, minimum = matrix_lengths(completions, edges)
        sampled_d = lengths-old_l-minimum+old_m
        assert np.array_equal(lengths, eval_saved['L_samples'][i]), (identifier, scheme, 'L_samples')
        assert np.array_equal(minimum, eval_saved['M_samples'][i]), (identifier, scheme, 'M_samples')
        assert np.array_equal(sampled_d, eval_saved['D_samples'][i]), (identifier, scheme, 'D_samples')
        assert np.all(sampled_d >= 0)
        ptrue = p[np.arange(n_targets), target_truth]
        y = np.eye(2)[target_truth]
        lower, upper = np.quantile(sampled_d, [.025, .975], method='linear')
        sd = float(np.std(sampled_d, ddof=1)); mean = float(np.mean(sampled_d))
        rows.append({'matrix_id': identifier, 'scenario_id': scenario['scenario_id'], 'replicate': rep, 'scheme': scheme,
                     'n_targets': n_targets, 'accuracy': float(np.mean(hard == target_truth)),
                     'brier': float(np.mean(np.sum((p-y)**2, axis=1))),
                     'log_loss': float(-np.log(np.maximum(ptrue, 1e-12)).mean()),
                     'zero_true_probability_count': int(np.sum(ptrue == 0)), 'log_floor_count': int(np.sum(ptrue < 1e-12)),
                     'mean_max_probability': float(np.max(p, axis=1).mean()), 'old_L': int(old_l), 'old_M': int(old_m),
                     'true_L': int(true_l), 'true_M': int(true_m), 'true_D': true_d, 'hard_D': hard_d,
                     'hard_D_error': hard_d-true_d, 'hard_D_absolute_error': abs(hard_d-true_d),
                     'mean_D': mean, 'mean_D_error': mean-true_d, 'mean_D_absolute_error': abs(mean-true_d),
                     'sd_D': sd, 'mean_D_mcse': sd/math.sqrt(512), 'lower95': lower, 'upper95': upper,
                     'interval_width': upper-lower, 'truth_in_interval': int(lower <= true_d <= upper),
                     'truth_below_interval': int(true_d < lower), 'truth_above_interval': int(true_d > upper),
                     'constant_response_count': counters['constant_response_count'],
                     'nonconstant_response_count': 32-counters['constant_response_count'],
                     'convergence_warning_count': counters['convergence_warning_count'], 'n_draws': 512})
    return rows, counters, refit_count


def verify_summaries(base, rows):
    independent = pd.DataFrame(rows)
    stored = pd.read_csv(base / 'matrix_results.csv')
    keys = ['matrix_id', 'scheme']
    independent = independent.sort_values(keys).reset_index(drop=True)
    stored = stored.sort_values(keys).reset_index(drop=True)
    assert len(stored) == 3600
    for field in independent.columns:
        if independent[field].dtype == object:
            assert independent[field].tolist() == stored[field].tolist(), field
        else:
            check_close(independent[field], stored[field], 'matrix '+field)
    summary = pd.read_csv(base / 'scenario_summary.csv')
    assert len(summary) == 72 and (summary.n_matrices == 50).all()
    for _, row in summary.iterrows():
        selected = independent[(independent.scenario_id == row.scenario_id) & (independent.scheme == row.scheme)]
        assert len(selected) == 50
        for field in summary.columns:
            if field.endswith('_mean') and field[:-5] in selected:
                check_close(row[field], selected[field[:-5]].mean(), field)
            elif field.endswith('_mcse') and field[:-5] in selected:
                check_close(row[field], selected[field[:-5]].std(ddof=1)/math.sqrt(50), field)
        hits = int(selected.truth_in_interval.sum())
        lo, hi = wilson(hits, 50)
        check_close(row.interval_coverage, hits/50, 'coverage')
        check_close(row.coverage_mcse, math.sqrt((hits/50)*(1-hits/50)/50), 'coverage mcse')
        check_close(row.coverage_wilson_lower95, lo, 'wilson lower')
        check_close(row.coverage_wilson_upper95, hi, 'wilson upper')
        assert row.count_truth_below_interval == selected.truth_below_interval.sum()
        assert row.count_truth_above_interval == selected.truth_above_interval.sum()
        assert row.zero_true_probability_total == selected.zero_true_probability_count.sum()
        assert row.log_floor_total == selected.log_floor_count.sum()
    paired = pd.read_csv(base / 'paired_method_contrasts.csv')
    assert len(paired) == 12*15 and (paired.n_matrices == 50).all()
    for _, row in paired.iterrows():
        ref = independent[(independent.scenario_id == row.scenario_id) & (independent.scheme == row.reference)].set_index('matrix_id')
        cand = independent[(independent.scenario_id == row.scenario_id) & (independent.scheme == row.candidate)].set_index('matrix_id')
        assert set(ref.index) == set(cand.index)
        for field in paired.columns:
            suffix = '_difference_mean'
            if field.endswith(suffix):
                metric = field[:-len(suffix)]
                difference = cand[metric] - ref[metric]
                check_close(row[field], difference.mean(), field)
                check_close(row[metric+'_difference_mcse'], difference.std(ddof=1)/math.sqrt(50), metric+' contrast mcse')
    return {'matrix_records': len(independent), 'scenario_records': len(summary), 'paired_records': len(paired)}


def verify_diagnostic_exports(base, rows, counters):
    summary = json.loads((base / 'FIT_DIAGNOSTICS_SUMMARY.json').read_text())
    assert summary['completed_matrices'] == 600 and summary['failed_matrix_count'] == 0
    assert summary['response_fits'] == 19200
    mapping = {'constant_response_fits': 'constant_response_count',
               'logistic_convergence_warnings': 'convergence_warning_count',
               'logistic_reached_max_iter': 'logistic_limit_count',
               'mk_optimizer_failures': 'mk_failed_optimizer_count',
               'mk_rate_boundaries': 'mk_boundary_count'}
    for external, independent in mapping.items():
        assert summary[external] == counters[independent], external
    assert summary['mk_optimizer_calls'] == 19200-counters['constant_response_count']
    char_table = pd.read_csv(base / 'fit_character_diagnostics.csv')
    assert len(char_table) == 19200
    assert len(char_table[['matrix_id', 'character']].drop_duplicates()) == 19200
    records = char_table.set_index(['matrix_id', 'character']).to_dict('index')
    warning_total = 0
    for identifier in sorted(char_table.matrix_id.unique()):
        meta = json.loads((base / 'predictions' / (identifier+'.json')).read_text())
        for d in meta['diagnostics']:
            saved = records[(identifier, d['character'])]
            mk = d['nj_mk']; log = d['classifiers']['logistic']; opt = mk['optimizer']
            expected = {'n_train': d['n_train'], 'n_test': d['n_test'],
                        'constant_fallback': int(d['constant_fallback']), 'logistic_fit_ran': int(log['fit_ran']),
                        'logistic_convergence_warning': int(log.get('convergence_warning', False)),
                        'logistic_reached_max_iter': int(log.get('reached_max_iter', False)),
                        'nj_zero_overlap_pairs': mk['zero_overlap_pairs'], 'nj_negative_branches': mk['negative_branches'],
                        'nj_truncated_branches': mk['truncated_branches'], 'mk_rate_boundary': int(mk['rate_boundary']),
                        'mk_optimizer_ran': int(opt['ran']), 'mk_warning_count': len(mk['warnings'])}
            for key, value in expected.items():
                assert saved[key] == value, (identifier, d['character'], key)
            if mk['rate'] is None:
                assert pd.isna(saved['mk_rate'])
            else:
                check_close(saved['mk_rate'], mk['rate'], 'mk_rate')
            assert str(saved['mk_optimizer_success']) == (str(int(opt['success'])) if opt['ran'] else 'not_run')
            warning_total += len(mk['warnings'])
    assert summary['mk_warning_count'] == warning_total
    exported = pd.read_csv(base / 'fit_diagnostics_by_scenario.csv')
    assert len(exported) == 12
    for _, record in exported.iterrows():
        subset = char_table[char_table.scenario_id == record.scenario_id]
        assert len(subset) == record.response_fits == 1600
        for field in exported.columns:
            if field in ('scenario_id', 'response_fits'):
                continue
            if field == 'mk_optimizer_failed':
                expected = int(((subset.mk_optimizer_ran == 1) & (subset.mk_optimizer_success.astype(str) != '1')).sum())
            else:
                expected = int(subset[field].sum())
            assert record[field] == expected, (record.scenario_id, field)
    zero_table = pd.read_csv(base / 'zero_probability_by_scenario.csv')
    assert len(zero_table) == 72
    metrics = pd.DataFrame(rows)
    for _, record in zero_table.iterrows():
        subset = metrics[(metrics.scenario_id == record.scenario_id) & (metrics.scheme == record.scheme)]
        assert len(subset) == record.n_matrices == 50
        n = int(subset.n_targets.sum()); zero = int(subset.zero_true_probability_count.sum()); floor = int(subset.log_floor_count.sum())
        assert record.n_target_cells == n and record.zero_true_probability_count == zero and record.log_floor_count == floor
        check_close(record.zero_true_probability_proportion, zero/n, 'zero proportion')
        check_close(record.log_floor_proportion, floor/n, 'floor proportion')
    return {'character_rows': len(char_table), 'scenario_rows': len(exported), 'zero_probability_rows': len(zero_table)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--no-refit', action='store_true')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = args.root.resolve(); base = root / 'simulation/results'
    started = time.monotonic()
    frozen = json.loads((root / 'simulation/design.json').read_text())
    assert frozen['matrices'] == 600 and frozen['propagation_draws'] == 512
    assert frozen['generating_seed'] == GEN_SEED and frozen['mask_seed'] == MASK_SEED and frozen['propagation_seed'] == PROP_SEED
    manifests = {key: json.loads((base / (key+'_MANIFEST.json')).read_text()) for key in ['GENERATION', 'FIT', 'EVALUATION']}
    assert manifests['GENERATION']['frozen_design_sha256'] in {frozen['source_protocol_sha256'], digest(root / 'simulation/design.json')}
    assert manifests['GENERATION']['generation_seed'] == GEN_SEED
    assert manifests['GENERATION']['mask_seed'] == MASK_SEED
    assert manifests['GENERATION']['propagation_seed'] == PROP_SEED
    assert manifests['GENERATION']['script_sha256'] == digest(root / 'simulation/experiment.py')
    assert manifests['GENERATION']['matrix_count'] == manifests['FIT']['completed_matrices'] == manifests['EVALUATION']['completed_matrices'] == 600
    assert not manifests['FIT']['failed_matrices']
    assert len(manifests['FIT']['records']) == 600
    for record in manifests['FIT']['records']:
        assert record['status'] == 'completed'
        stem = base / 'predictions' / record['matrix_id']
        assert digest(stem.with_suffix('.npz')) == record['prediction_sha256']
        assert digest(stem.with_suffix('.json')) == record['metadata_sha256']
    assert manifests['FIT']['adapter_sha256'] == digest(root / 'methods/fit_adapter.py')
    assert manifests['FIT']['mk_module_sha256'] == digest(root / 'methods/mk_baseline.py')
    assert manifests['FIT']['fit_worker_sha256'] == digest(root / 'simulation/fit_worker.py')
    assert manifests['FIT']['input_manifest_sha256'] == digest(base / 'input_manifest.json')
    assert manifests['EVALUATION']['fit_manifest_sha256'] == digest(base / 'FIT_MANIFEST.json')
    assert manifests['EVALUATION']['script_sha256'] == digest(root / 'simulation/experiment.py')
    entries = json.loads((base / 'input_manifest.json').read_text())
    assert len(entries) == len({r['matrix_id'] for r in entries}) == 600
    scenarios = pd.read_csv(base / 'scenario_registry.csv').set_index('scenario_id').to_dict('index')
    assert len(scenarios) == 12
    for ordinal, (q, missing, mode) in enumerate(itertools.product((.03, .3), (14, 29), ('random', 'structured', 'statebiased'))):
        record = scenarios[f's{ordinal:02d}']
        assert record['ordinal'] == ordinal and record['q'] == q
        assert record['missing_per_response'] == missing and record['mask_mode'] == mode
        assert record['n_tips'] == 48 and record['n_characters'] == 40 and record['n_complete_anchors'] == 8
        assert record['replicates'] == 50 and record['draws'] == 512
    registry = pd.read_csv(base / 'matrix_registry.csv').set_index('matrix_id')
    assert len(registry) == 600
    for entry in entries:
        identifier = entry['matrix_id']
        assert digest(base / 'truth' / (identifier+'_truth.npz')) == registry.loc[identifier, 'truth_sha256']
    arguments = []
    for entry in entries:
        scenario_id = f"s{entry['scenario_ordinal']:02d}"
        scenario = {'scenario_id': scenario_id, **scenarios[scenario_id]}
        arguments.append((str(root), entry, scenario, not args.no_refit))
    rows, all_counters, n_refits = [], {}, 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for idx, (matrix_rows, counters, refits) in enumerate(pool.map(audit_matrix, arguments)):
            rows.extend(matrix_rows); n_refits += refits
            for name, value in counters.items():
                all_counters[name] = all_counters.get(name, 0) + value
            if (idx+1) % 50 == 0:
                print(json.dumps({'matrices_independently_checked': idx+1, 'seconds': round(time.monotonic()-started, 1)}), flush=True)
    summaries = verify_summaries(base, rows)
    diagnostic_exports = verify_diagnostic_exports(base, rows, all_counters)
    report = {'status': 'PASS', 'design_sha256': digest(root / 'simulation/design.json'),
              'matrices': 600, 'response_fits_accounted': 600*32, 'schemes': list(SCHEMES),
              'draws_per_scheme_per_matrix': 512, 'sampled_completions_independently_scored': 600*6*512,
              'hard_completions_independently_scored': 600*6,
              'all_seed_generation_and_masks_replayed': True, 'all_marginal_draws_replayed': True,
              'joint_draws_decoded_hashed_and_independently_replayed': True,
              'truth_and_observed_contract_checked': True, 'all_native_binary_minima_checked': True,
              'independent_score_algorithm': 'bitset Fitch, compared with production Sankoff',
              'toy_exhaustive_check': self_test(), 'selected_refits': n_refits,
              'selected_refit_rule': 'scenario each replicate 0, responses 8 and 39',
              'refit_skipped_for_portable_replay': args.no_refit,
              'diagnostic_totals': all_counters, 'summary_checks': summaries,
              'diagnostic_export_checks': diagnostic_exports,
              'seconds': time.monotonic()-started}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
