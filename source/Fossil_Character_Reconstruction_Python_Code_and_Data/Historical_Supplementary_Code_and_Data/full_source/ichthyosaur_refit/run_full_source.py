"""Full-source sensitivity for the frozen 2017 to 2019 ichthyosaur history.

Only the old singleton response training rows change relative to the shared-fit
analysis: no old control is concealed. Original estimators and parameters are
called directly. Later labels are joined only after all prediction fits finish.
"""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import shutil
import sys
import time

import numpy as np
import pandas as pd
import scipy
import sklearn

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
ARCHIVE = ROOT / 'new_project_search/round12/restored/analysis'
OLD = ARCHIVE / 'new_project_search/round10/ichthyosaur_shared_fit'
CURATION = ARCHIVE / 'new_project_search/round10/ichthyosaur_reaudit'
MODULE = ARCHIVE / 'new_project_search/round10/method_pipeline/fit_frozen_history.py'
spec = importlib.util.spec_from_file_location('r10_fitter', MODULE)
fitter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fitter)
MODELS = fitter.MODELS
VERSIONS = {'numpy': np.__version__, 'pandas': pd.__version__, 'scipy': scipy.__version__, 'sklearn': sklearn.__version__}

def now(): return datetime.now(timezone.utc).isoformat()
def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def record(p):
    p = Path(p).resolve()
    return {'path': str(p), 'bytes': p.stat().st_size, 'sha256': digest(p)}
def dump(name, value): (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False))


def prepare():
    assert not (OUT / 'FIT_FREEZE.json').exists()
    assert not (OUT / 'FIT_STARTED.json').exists()
    old_freeze = json.loads((OLD / 'FIT_FREEZE.json').read_text())
    assert VERSIONS == old_freeze['versions']
    source = CURATION / 'source_model_inputs.npz'
    assert digest(source) == digest(OLD / 'source_training.npz')
    assert digest(source) == 'a389564fb359cca5342e730247e9b99b332f80fc7759dfdac81bd6c0fafb5e0e'
    z = np.load(source, allow_pickle=False)
    a, names, chars = z['A'], z['old_names'].tolist(), z['old_char_numbers'].tolist()
    assert a.shape == (115, 242)
    assert len(set(names)) == 115 and len(set(chars)) == 242
    targets = pd.read_csv(OLD / 'frozen_targets.csv', keep_default_na=False, dtype={'true': str})
    targets = targets.loc[targets.scenario.eq('later_publication')].copy().reset_index(drop=True)
    expected = pd.read_csv(CURATION / 'frozen_future_targets.csv', keep_default_na=False, dtype={'true': str})
    keys = ['character', 'taxon', 'true']
    assert targets[keys].sort_values(keys).reset_index(drop=True).equals(expected[keys].sort_values(keys).reset_index(drop=True))
    assert len(targets) == 196 and targets.character.nunique() == 130
    assert not targets.duplicated(['character', 'taxon']).any()
    lookup, cmap = {x: i for i, x in enumerate(names)}, {int(c): j for j, c in enumerate(chars)}
    counts = []
    previous = pd.read_csv(OLD / 'training_counts.csv').set_index('character')
    for char, g in targets.groupby('character', sort=True):
        j = cmap[int(char)]
        inds = np.array([lookup[x] for x in g.taxon])
        assert np.all(a[inds, j] == '?')
        n = int(fitter.mk.known(a[:, j]).sum())
        assert n >= 15 and n == int(previous.loc[char, 'n_original_train'])
        assert n - int(previous.loc[char, 'n_shared_train']) == int(previous.loc[char, 'n_controls'])
        counts.append({'character': int(char), 'n_original_train': n, 'n_full_source_train': n,
                       'n_old_shared_train': int(previous.loc[char, 'n_shared_train']),
                       'n_restored_old_controls': int(previous.loc[char, 'n_controls']), 'n_future': len(g)})
    shutil.copyfile(source, OUT / 'source_training.npz')
    targets.to_csv(OUT / 'frozen_targets.csv', index=False)
    # Deliberately separate a key-only prediction input from later labels.
    targets[['lineage', 'character', 'taxon']].to_csv(OUT / 'prediction_keys.csv', index=False)
    targets[keys].to_csv(OUT / 'evaluation_labels.csv', index=False)
    pd.DataFrame(counts).to_csv(OUT / 'training_counts.csv', index=False)
    dependencies = fitter.source_dependencies()
    files = [Path(__file__), OUT / 'PROTOCOL.txt', *dependencies,
             OUT / 'source_training.npz', OUT / 'frozen_targets.csv', OUT / 'prediction_keys.csv',
             OUT / 'evaluation_labels.csv', OUT / 'training_counts.csv']
    inputs = [source, CURATION / 'frozen_future_targets.csv', CURATION / 'CURATION_FREEZE.json',
              OLD / 'FIT_FREEZE.json', OLD / 'frozen_targets.csv', OLD / 'training_counts.csv',
              OLD / 'predictions.csv', OLD / 'estimated_trees.json']
    dump('FIT_FREEZE.json', {'created_utc': now(), 'analysis': 'full_source_sensitivity',
         'lineage': old_freeze['lineage'], 'n_source_taxa': 115, 'n_predictor_characters': 242,
         'n_future': 196, 'n_target_characters': 130, 'n_controls_concealed': 0,
         'models': MODELS, 'seed': fitter.SEED, 'versions': VERSIONS,
         'new_targets': False, 'new_hyperparameters': False,
         'fit_function': 'r10 fit_frozen_history.instrumented_predict and instrumented_mk',
         'files': [record(p) for p in files], 'original_inputs': [record(p) for p in inputs]})
    print('PREPARED: 196 fixed targets; 130 characters; no old controls concealed.', flush=True)


def run():
    assert not (OUT / 'FIT_STARTED.json').exists(), 'Refuse to overwrite a fit attempt'
    freeze = json.loads((OUT / 'FIT_FREEZE.json').read_text())
    for item in freeze['files'] + freeze['original_inputs']:
        assert digest(item['path']) == item['sha256'], item['path']
    assert VERSIONS == freeze['versions']
    dump('FIT_STARTED.json', {'started_utc': now(), 'freeze_sha256': digest(OUT / 'FIT_FREEZE.json')})
    started = time.monotonic()
    z = np.load(OUT / 'source_training.npz', allow_pickle=False)
    a, names = z['A'], z['old_names'].tolist()
    chars = z['old_char_numbers'].tolist()
    cmap, lookup = {int(c): j for j, c in enumerate(chars)}, {x: i for i, x in enumerate(names)}
    # This is the sole target table loaded while fitting: it contains no labels.
    keys = pd.read_csv(OUT / 'prediction_keys.csv', keep_default_na=False)
    assert list(keys.columns) == ['lineage', 'character', 'taxon']
    totals = fitter.mk.distance_totals(a)
    previous_trees = {int(x['character']): x for x in json.loads((OLD / 'estimated_trees.json').read_text())}
    predictions, fits, diagnostics, trees, mk_models = [], [], [], [], []
    for number, (char, g) in enumerate(keys.groupby('character', sort=True)):
        char, j = int(char), cmap[int(char)]
        train = np.flatnonzero(fitter.mk.known(a[:, j]))
        test = np.array([lookup[x] for x in g.taxon])
        assert np.all(a[test, j] == '?') and not set(train) & set(test)
        matrix = a.copy()
        assert np.array_equal(matrix, a)
        distance, zero = fitter.mk.excluding_distance(a, j, totals)
        edges, njdiag = fitter.mk.neighbor_join(distance)
        # Target exclusion means restoring that response cannot alter its NJ tree.
        assert np.array_equal(np.asarray(edges), np.asarray(previous_trees[char]['edges']))
        trees.append({'character': char, 'taxa': names, 'edges': edges})
        methods, clfdiag = fitter.instrumented_predict(matrix, j, train, test)
        labels, probability, mkfit, mkdiag = fitter.instrumented_mk(edges, len(names), matrix[:, j], train, test)
        methods['nj_mk'] = labels, probability
        diagnostics.append({'character': char, 'classifiers': clfdiag, 'mk': mkdiag})
        fits.append({'lineage': freeze['lineage'], 'character': char, 'n_train': len(train),
                     'n_original_train': len(train), 'n_controls': 0, 'n_future': len(test),
                     'zero_overlap_pairs': zero, **njdiag, **mkfit})
        mk_models.append({'character': char, 'taxa': names, 'edges': edges,
                          'character_matrix_index': j, 'observed': a[:, j].tolist(),
                          'train': train.tolist(), 'test': test.tolist(), 'classes': labels.tolist(),
                          **mkfit})
        for model in MODELS:
            classes, p = methods[model]
            assert np.isfinite(p).all() and np.all(p >= 0) and np.allclose(p.sum(axis=1), 1)
            hard = classes[p.argmax(axis=1)]
            for k, tip in enumerate(test):
                predictions.append({'lineage': freeze['lineage'], 'scenario': 'later_target',
                     'character': char, 'taxon': names[tip], 'model': model, 'pred': str(hard[k]),
                     'n_train': len(train), 'confidence': float(p[k].max()),
                     'state_probabilities': json.dumps(dict(zip(classes.tolist(), p[k].tolist())))})
        dump('fit_progress.json', {'completed_characters': number + 1, 'last_character': char,
                                   'seconds': time.monotonic() - started})
        dump('fit_execution_diagnostics.json', diagnostics)
        if number % 40 == 0:
            print(json.dumps({'completed_characters': number + 1, 'seconds': round(time.monotonic()-started, 2)}), flush=True)
    # Prediction artifact is written before loading later labels for evaluation.
    result = pd.DataFrame(predictions)
    result.to_csv(OUT / 'predictions_before_evaluation.csv', index=False)
    rawhash = digest(OUT / 'predictions_before_evaluation.csv')
    truth = pd.read_csv(OUT / 'evaluation_labels.csv', keep_default_na=False, dtype={'true': str})
    result = result.merge(truth, on=['character', 'taxon'], how='left', validate='many_to_one', sort=False)
    assert len(result) == 980 and not result['true'].isna().any()
    probs = result.state_probabilities.map(json.loads)
    result['p_true'] = [float(p.get(t, 0.0)) for p, t in zip(probs, result['true'])]
    result['brier'] = [float(sum(x*x for x in p.values()) - 2*pt + 1) for p, pt in zip(probs, result.p_true)]
    result['log_loss'] = -np.log(np.maximum(result.p_true.to_numpy(), 1e-12))
    columns = pd.read_csv(OLD / 'predictions.csv', nrows=0).columns.tolist()
    assert set(columns) == set(result.columns)
    result = result[columns]
    result.to_csv(OUT / 'predictions.csv', index=False)
    pd.DataFrame(fits).to_csv(OUT / 'fit_diagnostics.csv', index=False)
    dump('estimated_trees.json', trees)
    dump('mk_models.json', mk_models)
    assert digest(OUT / 'predictions_before_evaluation.csv') == rawhash
    assert np.array_equal(a, np.load(OUT / 'source_training.npz')['A'])
    summary = {'completed_utc': now(), 'seconds': time.monotonic()-started,
        'characters': len(fits), 'future_targets': 196, 'prediction_rows': len(result),
        'same_original_model_functions': True, 'new_hyperparameter_tuning': False,
        'future_labels_loaded_during_fitting': False,
        'prediction_before_evaluation_sha256': rawhash,
        'old_singletons_restored': 196, 'old_control_targets_predicted': False,
        'n_source_taxa': len(names), 'n_predictor_characters': len(chars),
        'all_estimated_nj_trees_identical_to_shared_fit': True,
        'all_training_counts_equal_original_singleton_counts': all(x['n_train'] == x['n_original_train'] for x in fits),
        'logistic_convergence_warning_characters': [x['character'] for x in diagnostics if any(y.get('convergence_warning') for y in x['classifiers'] if y['model']=='logistic')],
        'logistic_max_iter_characters': [x['character'] for x in diagnostics if any(y.get('reached_max_iter') for y in x['classifiers'] if y['model']=='logistic')],
        'mk_rate_boundary_characters': [x['character'] for x in fits if x['rate_boundary']],
        'mk_optimizer_unsuccessful_characters': [x['character'] for x in diagnostics if x['mk']['optimizer'] and not x['mk']['optimizer']['success']]}
    dump('FIT_COMPLETED.json', summary)
    dump('OUTPUT_MANIFEST.json', [record(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name != 'OUTPUT_MANIFEST.json'])
    print(json.dumps(summary, indent=2), flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=['prepare', 'run'])
    args = parser.parse_args()
    (prepare if args.operation == 'prepare' else run)()
