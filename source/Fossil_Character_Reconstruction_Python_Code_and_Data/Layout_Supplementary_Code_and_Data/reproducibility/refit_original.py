#!/usr/bin/env python3
"""Refit all original layout arms with the frozen estimator code, with checkpoints.

The archived experiment is never overwritten. A new destination is generated
using the frozen original generator, and every input is checked against the
archived SHA256 registry before fitting. Completed outputs can be resumed.
The optional --adopt directory accepts only bound outputs of this same frozen
fit_file routine, allowing a timed pilot to become part of the complete run.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import time
import traceback
import numpy as np
import pandas as pd

PACKAGE=Path(__file__).resolve().parents[1]
SOURCE=PACKAGE/'layout_experiment/simulation'
ARCHIVE=SOURCE/'results'
sys.path.insert(0,str(SOURCE))
import fit_worker
import experiment

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
def binding():
    files=['experiment.py','fit_worker.py','frozen_methods/fit_adapter.py','frozen_methods/mk_baseline.py','config.json']
    return {'source_files':{p:sha(SOURCE/p) for p in files},'wrapper_sha256':sha(__file__),
            'input_manifest_sha256':sha(ARCHIVE/'input_manifest.json'),
            'python':sys.version.split()[0],
            'dependencies':{p:importlib.metadata.version(p) for p in ('numpy','scipy','scikit-learn','pandas')},
            'operation':'Fresh refitting from hash-verified masked inputs; not recovered original metadata'}


def validate_binding(saved,base):
    current=binding()
    if saved==current:return
    # The initial execution finished all fits and scores, then failed because its
    # comparator assumed an original condition-summary file had been archived.
    # Permit only this documented wrapper correction, with all scientific source
    # and dependency bindings unchanged and the executed wrapper retained verbatim.
    base=Path(base);fix=json.loads((base/'POSTPROCESSING_FIX.json').read_text())
    assert fix['executed_wrapper_sha256']==saved['wrapper_sha256']
    assert fix['corrected_wrapper_sha256']==current['wrapper_sha256']
    assert sha(base/fix['executed_wrapper_source'])==saved['wrapper_sha256']
    assert {k:v for k,v in saved.items() if k!='wrapper_sha256'}=={k:v for k,v in current.items() if k!='wrapper_sha256'}

def check_prediction(directory,entry):
    """Validate a fit without opening generating truth or evaluation records."""
    directory=Path(directory);mid=entry['matrix_id'];jp=directory/f'{mid}.json';npz=directory/f'{mid}.npz'
    if not jp.exists() or not npz.exists():return None
    meta=json.loads(jp.read_text())
    if meta.get('status')!='completed':return None
    assert meta['matrix_id']==mid and meta['masked_sha256']==entry['masked_sha256']
    assert meta['adapter_sha256']==sha(SOURCE/'frozen_methods/fit_adapter.py')
    assert meta['mk_module_sha256']==sha(SOURCE/'frozen_methods/mk_baseline.py')
    assert sha(npz)==meta['prediction_sha256']
    with np.load(npz,allow_pickle=False) as z:
        p=z['probabilities'];targets=z['target_indices'];packed=z['missing_states_packed']
        assert p.shape==(5,len(targets),2) and np.isfinite(p).all() and (p>=0).all() and np.allclose(p.sum(2),1)
        assert packed.shape==(6,512,(len(targets)+7)//8)
        for i in range(6):
            assert hashlib.sha256(np.unpackbits(packed[i],axis=1,count=len(targets)).tobytes()).hexdigest()==meta['completion_sha256'][i]
    return {'matrix_id':mid,'status':'completed','seconds':meta['seconds'],
            'prediction_sha256':sha(npz),'metadata_sha256':sha(jp)}

def evaluate_checkpoint(base,entry):
    base=Path(base);mid=entry['matrix_id'];checkpoint=base/'evaluation_rows'/f'{mid}.json';evaluation=base/'evaluation'/f'{mid}.npz'
    psha=sha(base/'predictions'/f'{mid}.npz')
    if checkpoint.exists() and evaluation.exists():
        old=json.loads(checkpoint.read_text())
        if old['prediction_sha256']==psha and old['evaluation_sha256']==sha(evaluation) and old['scoring_source_sha256']==sha(SOURCE/'experiment.py'):
            return old['rows']
    rows=experiment.evaluate_one(base,entry)
    write(checkpoint,{'matrix_id':mid,'prediction_sha256':psha,'evaluation_sha256':sha(evaluation),
                      'scoring_source_sha256':sha(SOURCE/'experiment.py'),'rows':rows})
    return rows

def compare(base,entries):
    base=Path(base);oldfit={r['matrix_id']:r for r in json.loads((ARCHIVE/'FIT_MANIFEST.json').read_text())['records']}
    records=[]
    for e in entries:
        mid=e['matrix_id'];now=base/'evaluation'/f'{mid}.npz';old=ARCHIVE/'evaluation'/f'{mid}.npz'
        with np.load(now,allow_pickle=False) as a,np.load(old,allow_pickle=False) as b:
            same={key:bool(np.array_equal(a[key],b[key])) for key in b.files}
        records.append({'matrix_id':mid,'prediction_sha256_matches_original':sha(base/'predictions'/f'{mid}.npz')==oldfit[mid]['prediction_sha256'],
                        'evaluation_sha256_matches_original':sha(now)==sha(old),'evaluation_arrays_equal':same})
    tables={}
    for name in ('matrix_results.csv','paired_layout_contrasts.csv'):
        current=pd.read_csv(base/name);old=pd.read_csv(ARCHIVE/name)
        try:
            pd.testing.assert_frame_equal(current,old,check_exact=False,atol=1e-12,rtol=1e-12)
            tables[name]={'equal_at_1e-12':True,'rows':len(current)}
        except AssertionError as error:
            tables[name]={'equal_at_1e-12':False,'rows':len(current),'difference':str(error)[:12000]}
    from compare_condition_summary import compare_condition_summary
    tables['scenario_summary.csv']=compare_condition_summary(base/'scenario_summary.csv',ARCHIVE/'matrix_results.csv',ARCHIVE/'scenario_registry.csv')
    allarrays=all(all(r['evaluation_arrays_equal'].values()) for r in records)
    allpred=all(r['prediction_sha256_matches_original'] for r in records)
    result={'status':'pass' if allarrays and allpred and all(v['equal_at_1e-12'] for v in tables.values()) else 'differences_found',
            'scope':'New 360-arm masked-only fits, newly sampled completions and newly computed parsimony compared with frozen original results',
            'n_arms':len(entries),'all_prediction_bytes_match_archived_hashes':allpred,
            'all_evaluation_arrays_exactly_equal':allarrays,'tables':tables,'records':records,
            'metadata_note':'JSON includes new elapsed times and is newly generated; it is not recovered original metadata.',
            'original_results_overwritten':False}
    write(base/'REFIT_COMPARISON.json',result)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=6)
    p.add_argument('--resume',action='store_true');p.add_argument('--adopt',type=Path);a=p.parse_args()
    if not 1<=a.workers<=64:p.error('--workers must be between 1 and 64')
    base=a.output.resolve()
    if base==ARCHIVE.resolve() or ARCHIVE.resolve() in base.parents:raise ValueError('Cannot write into original archived results')
    started=time.monotonic();identity=binding()
    if base.exists():
        if not a.resume:raise FileExistsError('Existing destination requires --resume')
        validate_binding(json.loads((base/'REFIT_SOURCE_BINDING.json').read_text()),base)
    else:
        experiment.generation(base,SOURCE/'config.json',PACKAGE/'layout_experiment/ANALYSIS_PLAN.txt')
        write(base/'REFIT_SOURCE_BINDING.json',identity)
    entries=json.loads((base/'input_manifest.json').read_text());archived=json.loads((ARCHIVE/'input_manifest.json').read_text())
    assert entries==archived and len(entries)==360
    registry=pd.read_csv(ARCHIVE/'matrix_registry.csv')
    for r in registry.itertuples(index=False):
        assert sha(base/r.masked_file)==r.masked_sha256
        assert sha(base/'truth'/f'{r.pair_id}.npz')==r.truth_sha256
    pred=base/'predictions';pred.mkdir(exist_ok=True)
    adopted=[]
    if a.adopt:
        for e in entries:
            if check_prediction(pred,e):continue
            r=check_prediction(a.adopt,e)
            if r:
                for suffix in ('.json','.npz'):shutil.copy2(a.adopt/f"{e['matrix_id']}{suffix}",pred/f"{e['matrix_id']}{suffix}")
                adopted.append(e['matrix_id'])
        write(base/'ADOPTED_PILOT.json',{'operation':'Reuse newly executed pilot fits from the same frozen code; not original recovered files','matrix_ids':adopted})
    records={};todo=[]
    for e in entries:
        r=check_prediction(pred,e)
        if r:records[e['matrix_id']]=r
        else:todo.append(e)
    print(json.dumps({'stage':'fitting','already_complete':len(records),'remaining':len(todo),'workers':a.workers}),flush=True)
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures={pool.submit(fit_worker.fit_file,base/e['masked_file'],e,pred):e for e in todo}
        for future in as_completed(futures):
            e=futures[future]
            try:r=future.result()
            except Exception as error:r={'matrix_id':e['matrix_id'],'status':'worker_exception','error':str(error),'traceback':traceback.format_exc()}
            records[e['matrix_id']]=r
            checkpoint={'completed_arms':sum(r['status']=='completed' for r in records.values()),'failed_arms':[r for r in records.values() if r['status']!='completed'],
                        'workers':a.workers,'seconds':time.monotonic()-started,'fit_worker_sha256':sha(SOURCE/'fit_worker.py'),
                        'input_manifest_sha256':sha(base/'input_manifest.json'),'records':sorted(records.values(),key=lambda r:r['matrix_id'])}
            write(base/'FIT_MANIFEST.json',checkpoint)
            print(json.dumps({'stage':'fitting','completed_arms':checkpoint['completed_arms'],'failed_arms':len(checkpoint['failed_arms']),'total_arms':360,'seconds':round(time.monotonic()-started,1)}),flush=True)
    # Refresh bound records even on a no-work resume.
    finalrecords=[check_prediction(pred,e) for e in entries]
    assert all(finalrecords),'Incomplete fitting; resume after resolving recorded failures'
    fit={'completed_arms':360,'failed_arms':[],'workers':a.workers,'seconds':time.monotonic()-started,'fit_worker_sha256':sha(SOURCE/'fit_worker.py'),
         'input_manifest_sha256':sha(base/'input_manifest.json'),'records':finalrecords,'operation':'New complete refit with per-arm checkpoint reuse'}
    write(base/'FIT_MANIFEST.json',fit)
    (base/'evaluation').mkdir(exist_ok=True);(base/'evaluation_rows').mkdir(exist_ok=True)
    evaluated={};evaluation_start=time.monotonic()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures={pool.submit(evaluate_checkpoint,base,e):e for e in entries}
        for future in as_completed(futures):
            e=futures[future];evaluated[e['matrix_id']]=future.result()
            if len(evaluated)%10==0:print(json.dumps({'stage':'evaluation','completed_arms':len(evaluated),'total_arms':360,'seconds':round(time.monotonic()-evaluation_start,1)}),flush=True)
    rows=[row for e in entries for row in evaluated[e['matrix_id']]]
    experiment.summarize(base,rows)
    write(base/'EVALUATION_MANIFEST.json',{'arm_matrices':360,'paired_matrices':180,'rows':len(rows),'seconds':time.monotonic()-evaluation_start,
          'script_sha256':sha(SOURCE/'experiment.py'),'fit_manifest_sha256':sha(base/'FIT_MANIFEST.json'),
          'scheduling_wrapper_sha256':sha(__file__),'workers':a.workers,
          'arithmetic':'Unchanged experiment.evaluate_one and summarize; no evaluation RNG; per-arm checkpoints'})
    comparison=compare(base,entries)
    print(json.dumps({k:v for k,v in comparison.items() if k not in ('records','tables')}),flush=True)
    if comparison['status']!='pass':raise SystemExit(2)
if __name__=='__main__':main()
