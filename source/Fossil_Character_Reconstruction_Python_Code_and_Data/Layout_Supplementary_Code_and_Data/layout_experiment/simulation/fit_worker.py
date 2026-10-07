#!/usr/bin/env python3
"""R17 masked-only fits. Only historical target cells are propagated.

The five estimators and joint Mk sampler are byte-identical R16 imports.
This wrapper fits target columns, retains probabilities only at target cells,
and leaves all other missing cells unknown in downstream completion matrices.
"""
from __future__ import annotations
import os
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'frozen_methods'))
from fit_adapter import METHODS, fit_character, sample_mk_joint

N_DRAWS = 512
PROPAGATION_SEED = 2026100419

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2) + '\n')

def fit_targets(masked, target_mask):
    masked = np.asarray(masked, dtype=np.int8)
    target_mask = np.asarray(target_mask, dtype=bool)
    assert masked.shape == target_mask.shape
    assert np.isin(masked, [-1, 0, 1]).all()
    assert np.all(masked[target_mask] == -1)
    targets = np.argwhere(target_mask)
    columns = np.unique(targets[:, 1])
    matrix = np.where(masked == -1, '?', masked.astype(str))
    probabilities = np.zeros((5, len(targets), 2), dtype=float)
    models = [None] * matrix.shape[1]
    diagnostics, failures = [], []
    for j in columns:
        j = int(j)
        try:
            fitted = fit_character(matrix, j)
        except Exception as exception:
            failures.append({'character': j, 'type': type(exception).__name__, 'message': str(exception)})
            continue
        target_positions = np.flatnonzero(targets[:, 1] == j)
        rows = targets[target_positions, 0]
        fitted_row_index = {int(row): i for i, row in enumerate(fitted['target_indices'])}
        selected = [fitted_row_index[int(row)] for row in rows]
        for method_index, method in enumerate(METHODS):
            classes, values = fitted['methods'][method]
            for k, state in enumerate(classes):
                probabilities[method_index, target_positions, int(state)] = values[selected, k]
        models[j] = fitted['nj_model']
        diagnostics.append({'character': j, 'n_historical_targets': len(rows), **fitted['diagnostics']})
    return {'probabilities': probabilities, 'models': models, 'diagnostics': diagnostics,
            'failures': failures, 'target_indices': targets, 'target_columns': columns}

def joint_targets(masked, target_mask, models, n_draws, seed):
    """Run unchanged sampler on response columns and discard background draws.

    Unknown non-target tips remain latent in the fitted Mk tree. Sampling them
    is mathematically required for joint target draws, but those states are
    never inserted into a scored matrix.
    """
    targets = np.argwhere(target_mask)
    columns = np.unique(targets[:, 1])
    matrix = np.where(masked == -1, '?', masked.astype(str))[:, columns]
    models_subset = [models[int(j)] for j in columns]
    all_unknowns = np.argwhere(matrix == '?')
    original_columns = columns[all_unknowns[:, 1]]
    keep = target_mask[all_unknowns[:, 0], original_columns]
    selected = np.column_stack((all_unknowns[keep, 0], original_columns[keep]))
    assert np.array_equal(selected, targets)
    return sample_mk_joint(matrix, models_subset, n_draws, seed)[:, keep]

def fit_file(masked_path, entry, output):
    start = time.monotonic()
    masked_path, output = Path(masked_path), Path(output)
    assert sha256(masked_path) == entry['masked_sha256']
    with np.load(masked_path) as data:
        masked, target_mask = data['X'], data['target_mask'].astype(bool)
    fitted = fit_targets(masked, target_mask)
    metadata = {'matrix_id': entry['matrix_id'], 'masked_sha256': entry['masked_sha256'],
                'diagnostics': fitted['diagnostics'], 'nj_models': fitted['models'],
                'failures': fitted['failures'], 'target_columns': fitted['target_columns'].tolist(),
                'adapter_sha256': sha256(HERE / 'frozen_methods/fit_adapter.py'),
                'mk_module_sha256': sha256(HERE / 'frozen_methods/mk_baseline.py')}
    if fitted['failures']:
        metadata.update(status='failed', seconds=time.monotonic()-start)
        write_json(output / f"{entry['matrix_id']}.json", metadata)
        return {'matrix_id': entry['matrix_id'], 'status': 'failed', 'failures': fitted['failures']}
    probability = fitted['probabilities']
    assert np.isfinite(probability).all() and np.allclose(probability.sum(2), 1)
    # Separate independent propagation streams by arm; data matrices are paired.
    marginal_seed = [PROPAGATION_SEED, entry['panel_ordinal'], entry['q_ordinal'], entry['replicate'], entry['arm_ordinal'], 0]
    joint_seed = [PROPAGATION_SEED, entry['panel_ordinal'], entry['q_ordinal'], entry['replicate'], entry['arm_ordinal'], 1]
    rng = np.random.default_rng(np.random.SeedSequence(marginal_seed))
    uniforms = rng.random((N_DRAWS, len(fitted['target_indices'])))
    draws = [(uniforms < probability[i, :, 1][None, :]).astype(np.uint8) for i in range(5)]
    draws.append(joint_targets(masked, target_mask, fitted['models'], N_DRAWS, np.random.SeedSequence(joint_seed)))
    packed = np.asarray([np.packbits(x, axis=1) for x in draws])
    path = output / f"{entry['matrix_id']}.npz"
    np.savez_compressed(path, probabilities=probability, missing_states_packed=packed,
                        target_indices=fitted['target_indices'].astype(np.int16))
    metadata.update(status='completed', seconds=time.monotonic()-start,
                    prediction_sha256=sha256(path),
                    constant_response_count=sum(d['constant_fallback'] for d in fitted['diagnostics']),
                    convergence_warning_count=sum(d['classifiers']['logistic'].get('convergence_warning', False) for d in fitted['diagnostics']),
                    marginal_uniform_seed_sequence=marginal_seed, joint_seed_sequence=joint_seed,
                    n_draws=N_DRAWS, n_targets=len(fitted['target_indices']),
                    schemes=[*METHODS, 'nj_mk_joint'],
                    completion_sha256=[hashlib.sha256(x.tobytes()).hexdigest() for x in draws],
                    boundary='Only masked X and historical target indicator read; no truth, generating tree, or rate')
    write_json(output / f"{entry['matrix_id']}.json", metadata)
    return {'matrix_id': entry['matrix_id'], 'status': 'completed', 'seconds': metadata['seconds'],
            'prediction_sha256': metadata['prediction_sha256'],
            'metadata_sha256': sha256(output / f"{entry['matrix_id']}.json")}

def run(base, workers):
    base = Path(base)
    output = base / 'predictions'
    output.mkdir(exist_ok=False)
    entries = json.loads((base / 'input_manifest.json').read_text())
    start = time.monotonic()
    records = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fit_file, base / e['masked_file'], e, output): e for e in entries}
        for future in as_completed(futures):
            entry = futures[future]
            try:
                records.append(future.result())
            except Exception as exception:
                failure = {'matrix_id': entry['matrix_id'], 'status': 'worker_exception',
                           'type': type(exception).__name__, 'message': str(exception), 'traceback': traceback.format_exc()}
                records.append(failure)
                write_json(output / f"{entry['matrix_id']}_failure.json", failure)
            print(json.dumps({'finished_arms': len(records), 'total_arms': len(entries),
                              'failed_arms': sum(r['status'] != 'completed' for r in records),
                              'seconds': round(time.monotonic()-start, 2)}), flush=True)
    records.sort(key=lambda r: r['matrix_id'])
    manifest = {'completed_arms': sum(r['status'] == 'completed' for r in records),
                'failed_arms': [r for r in records if r['status'] != 'completed'],
                'workers': workers, 'seconds': time.monotonic()-start,
                'fit_worker_sha256': sha256(__file__), 'input_manifest_sha256': sha256(base / 'input_manifest.json'),
                'records': records}
    write_json(base / 'FIT_MANIFEST.json', manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k != 'records'}), flush=True)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=6)
    a = p.parse_args()
    run(a.output, a.workers)
