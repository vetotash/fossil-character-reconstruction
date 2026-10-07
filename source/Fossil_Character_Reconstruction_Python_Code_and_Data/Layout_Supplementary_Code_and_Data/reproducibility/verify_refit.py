#!/usr/bin/env python3
"""Read-only integrity and exact archived-result comparison for a complete refit."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from refit_original import PACKAGE,SOURCE,ARCHIVE,validate_binding,check_prediction,sha

def verify(base):
    base=Path(base).resolve()
    saved_binding=json.loads((base/'REFIT_SOURCE_BINDING.json').read_text())
    validate_binding(saved_binding,base)
    entries=json.loads((base/'input_manifest.json').read_text())
    assert entries==json.loads((ARCHIVE/'input_manifest.json').read_text()) and len(entries)==360
    fit=json.loads((base/'FIT_MANIFEST.json').read_text())
    assert fit['completed_arms']==360 and not fit['failed_arms']
    assert sha(base/'input_manifest.json')==fit['input_manifest_sha256']
    records={r['matrix_id']:r for r in fit['records']}
    originals={r['matrix_id']:r for r in json.loads((ARCHIVE/'FIT_MANIFEST.json').read_text())['records']}
    registry=pd.read_csv(ARCHIVE/'matrix_registry.csv').set_index('matrix_id')
    for e in entries:
        mid=e['matrix_id'];r=check_prediction(base/'predictions',e)
        assert r is not None and r==records[mid],mid
        assert r['prediction_sha256']==originals[mid]['prediction_sha256'],('prediction differs from archive',mid)
        assert sha(base/e['masked_file'])==e['masked_sha256'],mid
        assert sha(base/'truth'/f"{e['pair_id']}.npz")==registry.loc[mid,'truth_sha256'],mid
        with np.load(base/e['masked_file'],allow_pickle=False) as masked,np.load(base/'predictions'/f'{mid}.npz',allow_pickle=False) as pred:
            assert np.array_equal(np.argwhere(masked['target_mask']),pred['target_indices'])
        with np.load(base/'evaluation'/f'{mid}.npz',allow_pickle=False) as new,np.load(ARCHIVE/'evaluation'/f'{mid}.npz',allow_pickle=False) as old:
            assert set(new.files)==set(old.files)
            for key in old.files:assert np.array_equal(new[key],old[key]),(mid,key)
            assert new['D_samples'].shape==(6,512)
            assert np.array_equal(new['D_samples'],new['L_samples']-new['old_L']-(new['M_samples']-new['old_M']))
            assert new['true_D']==new['true_L']-new['old_L']-(new['true_M']-new['old_M'])
        checkpoint=json.loads((base/'evaluation_rows'/f'{mid}.json').read_text())
        assert checkpoint['prediction_sha256']==r['prediction_sha256']
        assert checkpoint['evaluation_sha256']==sha(base/'evaluation'/f'{mid}.npz')
        assert checkpoint['scoring_source_sha256']==sha(SOURCE/'experiment.py')
    evaluation=json.loads((base/'EVALUATION_MANIFEST.json').read_text())
    assert evaluation['arm_matrices']==360 and evaluation['rows']==2160
    assert evaluation['fit_manifest_sha256']==sha(base/'FIT_MANIFEST.json')
    tables={}
    for name in ('matrix_results.csv','paired_layout_contrasts.csv'):
        new=pd.read_csv(base/name);old=pd.read_csv(ARCHIVE/name)
        pd.testing.assert_frame_equal(new,old,check_exact=False,atol=1e-12,rtol=1e-12)
        tables[name]=len(new)
    from compare_condition_summary import compare_condition_summary
    summary_check=compare_condition_summary(base/'scenario_summary.csv',ARCHIVE/'matrix_results.csv',ARCHIVE/'scenario_registry.csv')
    assert summary_check['equal_at_1e-12']
    comparison=json.loads((base/'REFIT_COMPARISON.json').read_text())
    assert comparison['status']=='pass' and comparison['n_arms']==360
    assert comparison['all_prediction_bytes_match_archived_hashes'] and comparison['all_evaluation_arrays_exactly_equal']
    return {'status':'pass','new_refit_arms':360,'prediction_arrays_and_packed_states_match_original_SHA256':360,
            'new_model_metadata_files_verified':360,'truth_files':180,'masked_inputs':360,
            'new_evaluation_files_exactly_equal_to_original':360,'two_original_result_tables_equal_at_1e-12':tables,
            'new_condition_summary_against_original_matrix_rows':summary_check,
            'scope':'Read-only verification of new complete refit and bindings; fitting/scoring are not repeated by this command'}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--input',type=Path,default=PACKAGE/'refit_20261006')
    args=parser.parse_args();print(json.dumps(verify(args.input),indent=2))
