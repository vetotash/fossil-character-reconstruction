#!/usr/bin/env python3
"""Score frozen Round 17 D draws; no generation, fitting, or truth imputation.

Requires an explicitly prepared registry referencing saved arrays, with per-file
hashes and source-declared method order. It never constructs draws from summary
statistics. Each arm is scored against its OWN target-only saved true_D.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from validated_crps import aggregate, distribution_metrics

SCHEMES = ('mode', 'knn5', 'logistic', 'extra_trees', 'nj_mk', 'nj_mk_joint')
METRICS = ('crps_empirical', 'crps_fair_ensemble', 'signed_bias',
           'absolute_mean_error', 'width95', 'truth_in_95',
           'truth_below_95', 'truth_above_95')
ARMS = ('empirical', 'permuted')
REQUIRED = ('condition_id', 'latent_id', 'panel', 'q', 'replicate', 'arm',
            'npz_path', 'npz_sha256', 'draw_key', 'truth_key', 'schemes_json')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_registry(registry):
    missing = set(REQUIRED) - set(registry.columns)
    if missing:
        raise ValueError(f'Missing explicit registry fields: {sorted(missing)}')
    if registry[list(REQUIRED)].isna().any().any():
        raise ValueError('Registry contains missing required values')
    if set(registry.arm) != set(ARMS):
        raise ValueError('Exactly empirical and permuted arms are required')
    if registry[['latent_id', 'arm']].duplicated().any():
        raise ValueError('Duplicate latent-matrix/arm records')
    if len(registry) != 360 or registry.latent_id.nunique() != 180:
        raise ValueError('Scientific run requires all 180 paired latent matrices (360 arms)')
    if registry.condition_id.nunique() != 6 or registry.panel.nunique() != 3:
        raise ValueError('Expected six conditions from three panels')
    if set(registry.q.astype(float)) != {.03, .3}:
        raise ValueError('Expected q=0.03 and q=0.3')
    if len(registry[['panel', 'q', 'condition_id']].drop_duplicates()) != 6:
        raise ValueError('Condition IDs do not uniquely identify panel/rate combinations')
    for condition, group in registry.groupby('condition_id'):
        if len(group) != 60 or group.latent_id.nunique() != 30:
            raise ValueError(f'{condition}: expected 30 paired replicates')
        if group.panel.nunique() != 1 or group.q.nunique() != 1:
            raise ValueError(f'{condition}: multiple panel/rate settings')
        for arm in ARMS:
            subset = group[group.arm == arm]
            if len(subset) != 30 or subset.replicate.nunique() != 30:
                raise ValueError(f'{condition}/{arm}: incomplete replicate set')
    for latent, group in registry.groupby('latent_id'):
        if len(group) != 2 or set(group.arm) != set(ARMS):
            raise ValueError(f'{latent}: paired arm absent')
        for col in ('condition_id', 'panel', 'q', 'replicate'):
            if group[col].nunique() != 1:
                raise ValueError(f'{latent}: inconsistent pair metadata {col}')


def score_registry(registry, source_root):
    """Return one record per arm/scheme and a record of verified source hashes."""
    validate_registry(registry)
    records, checks = [], []
    source_root = Path(source_root).resolve()
    for row in registry.sort_values(['condition_id', 'replicate', 'arm']).to_dict('records'):
        path = (source_root / str(row['npz_path'])).resolve()
        if source_root not in path.parents:
            raise ValueError('Array source must be within the explicit source root')
        digest = sha256(path)
        if digest != row['npz_sha256']:
            raise ValueError(f'Array hash mismatch: {path}')
        declared_schemes = tuple(json.loads(row['schemes_json']))
        if len(declared_schemes) != 6 or set(declared_schemes) != set(SCHEMES):
            raise ValueError('Registry must declare all six unique schemes in saved array order')
        with np.load(path, allow_pickle=False) as arrays:
            draws = np.array(arrays[row['draw_key']], dtype=float, copy=True)
            saved_truth = np.asarray(arrays[row['truth_key']])
        if draws.shape != (6, 512):
            raise ValueError(f'{path}: need saved (6,512) D draws, got {draws.shape}')
        if not np.isfinite(draws).all() or saved_truth.size != 1:
            raise ValueError(f'{path}: finite draws and scalar target-only true_D required')
        truth = float(saved_truth.reshape(-1)[0])
        if not np.isfinite(truth):
            raise ValueError(f'{path}: nonfinite true_D')
        common = {key: row[key] for key in ('condition_id', 'latent_id', 'panel', 'q', 'replicate', 'arm')}
        common.update(matrix_id=f"{row['latent_id']}:{row['arm']}", n_draws=512,
                      true_D=truth, source_npz=row['npz_path'], source_npz_sha256=digest)
        if 'original_matrix_id' in row:
            common['original_matrix_id'] = row['original_matrix_id']
        if 'n_targets' in row:
            common['n_targets'] = int(row['n_targets'])
        for i, scheme in enumerate(declared_schemes):
            records.append({**common, 'scheme': scheme, **distribution_metrics(draws[i], truth)})
        checks.append({'latent_id': row['latent_id'], 'arm': row['arm'], 'file': row['npz_path'],
                       'sha256': digest, 'draw_key': row['draw_key'], 'truth_key': row['truth_key'],
                       'shape': list(draws.shape), 'scheme_order': list(declared_schemes)})
    return pd.DataFrame(records), checks


def paired_row(values, **metadata):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or len(values) < 2:
        raise ValueError('Paired contrast requires finite independent-matrix differences')
    return {**metadata, 'n_paired_matrices': len(values),
            'mean_difference': float(values.mean()),
            'paired_MCSE': float(values.std(ddof=1) / math.sqrt(len(values)))}


def paired_layout_contrasts(frame):
    rows, differences = [], []
    for (condition, panel, q, scheme), group in frame.groupby(['condition_id', 'panel', 'q', 'scheme'], sort=True):
        a = group[group.arm == 'empirical'].set_index('latent_id')
        b = group[group.arm == 'permuted'].set_index('latent_id')
        if set(a.index) != set(b.index) or a.index.has_duplicates or b.index.has_duplicates:
            raise ValueError('Layout pairs are not one-to-one')
        b = b.loc[a.index]
        for metric in METRICS:
            values = a[metric].to_numpy(float) - b[metric].to_numpy(float)
            metadata = {'condition_id': condition, 'panel': panel, 'q': q,
                        'scheme': scheme, 'metric': metric,
                        'difference_direction': 'empirical_minus_permuted'}
            rows.append(paired_row(values, **metadata))
            differences.extend({**metadata, 'latent_id': latent, 'difference': float(value)}
                               for latent, value in zip(a.index, values))
    return pd.DataFrame(rows), pd.DataFrame(differences)


def paired_method_contrasts(frame):
    rows = []
    for (condition, panel, q, arm), group in frame.groupby(['condition_id', 'panel', 'q', 'arm'], sort=True):
        for reference, candidate in itertools.combinations(SCHEMES, 2):
            a = group[group.scheme == reference].set_index('latent_id')
            b = group[group.scheme == candidate].set_index('latent_id')
            if set(a.index) != set(b.index) or a.index.has_duplicates or b.index.has_duplicates:
                raise ValueError('Method pairs are not one-to-one')
            b = b.loc[a.index]
            for metric in METRICS:
                rows.append(paired_row(b[metric].to_numpy(float) - a[metric].to_numpy(float),
                            condition_id=condition, panel=panel, q=q, arm=arm,
                            reference=reference, candidate=candidate, metric=metric,
                            difference_direction='candidate_minus_reference'))
    return pd.DataFrame(rows)


def run(registry_path, source_root, output, source_audit=None):
    source_root, output = Path(source_root).resolve(), Path(output).resolve()
    if output == source_root or source_root in output.parents:
        raise ValueError('Output must be outside the read-only source archive')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a new/empty output directory; existing results are never overwritten')
    registry = pd.read_csv(registry_path, dtype={'latent_id': str, 'condition_id': str})
    audit = None
    if source_audit:
        audit = json.loads(Path(source_audit).read_text())
        if audit.get('status') != 'PASS' or audit.get('normalized_registry_sha256') != sha256(registry_path):
            raise ValueError('Source audit does not verify the supplied normalized registry')
    frame, checks = score_registry(registry, source_root)
    summary = aggregate(frame, ['condition_id', 'panel', 'q', 'arm', 'scheme'], list(METRICS))
    contrasts, matrix_differences = paired_layout_contrasts(frame)
    methods = paired_method_contrasts(frame)
    if len(frame) != 2160 or len(summary) != 72 or len(contrasts) != 36 * len(METRICS):
        raise AssertionError('Incomplete scientific scope')
    output.mkdir(parents=True, exist_ok=True)
    files = {'matrix_distribution_scores.csv': frame, 'arm_distribution_summary.csv': summary,
             'paired_layout_contrasts.csv': contrasts, 'matrix_paired_layout_differences.csv': matrix_differences,
             'paired_method_contrasts.csv': methods}
    for filename, data in files.items():
        data.to_csv(output / filename, index=False)
    manifest = {
        'status': 'PASS', 'analysis': 'post_hoc_scoring_of_saved_distributions_no_refitting',
        'independent_latent_matrices': 180, 'fitted_arms': 360, 'schemes': list(SCHEMES),
        'forecast_rows': 2160, 'draws_per_distribution': 512,
        'independent_matrices_per_panel_rate': 30, 'arm_summaries': 72,
        'layout_contrast_groups': 36, 'within_arm_method_pair_groups': 180,
        'registry_sha256': sha256(registry_path),
        'source_audit_sha256': sha256(source_audit) if source_audit else None,
        'source_recovery_scope': audit.get('source_recovery_scope') if audit else None,
        'script_sha256': sha256(__file__), 'validated_crps_sha256': sha256(Path(__file__).with_name('validated_crps.py')),
        'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
        'primary_score': 'CRPS of each issued 512-draw empirical distribution; lower is better',
        'secondary_score': 'Fair finite-ensemble CRPS sensitivity; different estimand from empirical-distribution CRPS',
        'target_truth': 'Each arm uses its own saved target-only true_D; unavailable U cells stay missing',
        'layout_difference': 'empirical minus permuted within the same latent replicate after arm-specific scoring',
        'method_difference': 'candidate minus reference within the same latent replicate and arm',
        'mcse': 'sample standard deviation across 30 matrix values or paired differences divided by sqrt(30)',
        'inclusion': 'Saved-draw linear 2.5%/97.5% quantiles, inclusive endpoints; Wilson limits for finite-matrix inclusion frequency',
        'limitations': ['Post hoc additional scores, not a prespecified Round17 primary endpoint.',
                       'No overall cross-panel ranking or pooled average; D and CRPS scales depend on panel and target geometry.',
                       'Layout permutation changes target identities as well as observed/unavailable arrangements.',
                       'Saved D-array scoring does not by itself re-audit biological source mappings or anatomical validity.'],
        'source_arrays_checked': checks,
        'output_hashes': {name: sha256(output / name) for name in files},
    }
    (output / 'analysis_manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registry', required=True, type=Path)
    p.add_argument('--source-root', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--source-audit', type=Path)
    a = p.parse_args()
    result = run(a.registry, a.source_root, a.output, a.source_audit)
    print(json.dumps({k: result[k] for k in ('status', 'independent_latent_matrices', 'forecast_rows', 'arm_summaries')}, indent=2))
