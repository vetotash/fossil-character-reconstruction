#!/usr/bin/env python3
"""Frozen-target O/U control: prepare -> masked-only fit -> evaluate -> audit.

Source estimators, joint conditioning and Sankoff scoring are imported unchanged
from the required recovered Round17 sources. This extension changes the observation mask.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv, hashlib, importlib.util, json, math, shutil, time, traceback
from pathlib import Path
import numpy as np
from fixed_target_design import randomize_observed_fixed_target, masked_input, PERMUTATION_SEED, PROPAGATION_SEED, N_DRAWS

HERE=Path(__file__).resolve().parent
METHODS=('mode','knn5','logistic','extra_trees','nj_mk')
SCHEMES=(*METHODS,'nj_mk_joint')
ARMS=('empirical','fixed_target_permuted')
CORE=None; WORKER=None

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,obj): Path(path).write_text(json.dumps(obj,indent=2)+'\n')
def read(path): return json.loads(Path(path).read_text())
def rows(path):
    with Path(path).open() as f:return list(csv.DictReader(f))
def csvwrite(path,data):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path); result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result

def initialize(source):
    global CORE,WORKER
    source=Path(source)
    CORE=module('fixed_control_original_scoring',source/'simulation/experiment.py')
    WORKER=module('fixed_control_original_fitting',source/'simulation/fit_worker.py')
    assert tuple(WORKER.METHODS)==METHODS and WORKER.N_DRAWS==512

def bind_source(source):
    files=['ANALYSIS_PLAN.txt','DESIGN_FREEZE.json','simulation/config.json','simulation/experiment.py','simulation/fit_worker.py',
           'simulation/frozen_methods/fit_adapter.py','simulation/frozen_methods/mk_baseline.py',
           'simulation/results/GENERATION_MANIFEST.json','simulation/results/input_manifest.json',
           'simulation/results/FIT_MANIFEST.json','simulation/results/EVALUATION_MANIFEST.json',
           'simulation/results/matrix_registry.csv','simulation/results/matrix_results.csv']
    return {f:sha(source/f) for f in files}

def bound_source(base,override=None):
    """Resolve relocated source without changing recorded manifest provenance."""
    binding=read(Path(base)/'SOURCE_BINDING.json')
    candidate=Path(override) if override is not None else Path(binding['source'])
    candidate=candidate.resolve()
    for name,expected in binding['source_files'].items():
        actual=sha(candidate/name)
        if actual!=expected:raise ValueError(f'Bound source hash mismatch: {name}')
    return candidate

def prepare(source,base,replayed):
    source,base=Path(source).resolve(),Path(base).resolve()
    replayed=Path(replayed).resolve()
    replay_check=read(replayed/'REPLAY_VERIFICATION.json')
    assert replay_check['all_truth_and_mask_sha256_match_original_registry']
    freeze=read(HERE/'DESIGN_FREEZE.json')
    assert sha(HERE/'ANALYSIS_PLAN.txt')==freeze['analysis_plan_sha256']
    assert sha(HERE/'fixed_target_design.py')==freeze['randomization_code_sha256']
    source_hashes=bind_source(source)
    if base.exists():raise FileExistsError(base)
    original=source/'simulation/results'
    entries=read(original/'input_manifest.json'); fit=read(original/'FIT_MANIFEST.json')
    assert len(entries)==360 and fit['completed_arms']==360 and not fit['failed_arms']
    selected=[e for e in entries if e['arm']=='empirical']
    assert len(selected)==180
    by_fit={r['matrix_id']:r for r in fit['records']}
    registry={r['matrix_id']:r for r in rows(original/'matrix_registry.csv')}
    for folder in ('inputs','truth','baseline_predictions','baseline_evaluation'):(base/folder).mkdir(parents=True,exist_ok=True)
    new_entries=[]; baseline_entries=[]; bindings=[]
    for e in selected:
        source_mask=original/e['masked_file']; record=registry[e['matrix_id']]
        assert sha(source_mask)==e['masked_sha256']
        source_truth=replayed/'truth'/f"{e['pair_id']}.npz"
        assert sha(source_truth)==record['truth_sha256']
        with np.load(source_mask) as d:
            old_masked=d['X']; target=d['target_mask'].astype(bool); labels=d['labels']
        with np.load(source_truth) as d: latent=d['X_true']
        observed=labels==0
        assert np.array_equal(target,labels==2)
        assert np.array_equal(old_masked, np.where(observed,latent.astype(np.int8),-1))
        seed=[PERMUTATION_SEED,e['panel_ordinal'],e['q_ordinal'],e['replicate']]
        new_observed=randomize_observed_fixed_target(observed,target,seed)
        new_masked,keys=masked_input(latent,new_observed,target)
        new_labels=np.where(target,2,np.where(new_observed,0,1)).astype(np.uint8)
        assert np.array_equal(np.stack([(labels==k).sum(0) for k in range(3)]),np.stack([(new_labels==k).sum(0) for k in range(3)]))
        assert np.array_equal(np.argwhere(target),keys)
        name=e['pair_id']+'_a2'; out=base/'inputs'/f'{name}.npz'
        np.savez_compressed(out,X=new_masked,target_mask=target,labels=new_labels)
        new={**e,'matrix_id':name,'arm':'fixed_target_permuted','arm_ordinal':2,
             'masked_file':str(out.relative_to(base)),'masked_sha256':sha(out)}
        new_entries.append(new)
        # Preserve source empirical arrays without changing their model or draw.
        baseline_mask=base/e['masked_file'];shutil.copyfile(source_mask,baseline_mask)
        shutil.copyfile(source_truth,base/'truth'/source_truth.name)
        f=by_fit[e['matrix_id']]
        assert f['status']=='completed'
        ev=original/'evaluation'/(e['matrix_id']+'.npz')
        shutil.copyfile(ev,base/'baseline_evaluation'/ev.name)
        baseline_entries.append(e)
        bindings.append({'pair_id':e['pair_id'],'empirical_id':e['matrix_id'],'new_id':name,
                         'source_truth_sha256':sha(source_truth),'source_input_sha256':sha(source_mask),
                         'source_prediction_sha256':f['prediction_sha256'],'source_prediction_metadata_sha256':f['metadata_sha256'],
                         'source_evaluation_sha256':sha(ev),'new_input_sha256':sha(out),'permutation_seed':seed})
    assert len({e['pair_id'] for e in new_entries})==180
    shutil.copyfile(replayed/'scenario_registry.csv',base/'scenario_registry.csv')
    shutil.copyfile(original/'matrix_results.csv',base/'baseline_matrix_results.csv')
    shutil.copyfile(replayed/'REPLAY_VERIFICATION.json',base/'REPLAY_VERIFICATION.json')
    dump(base/'input_manifest.json',new_entries);dump(base/'baseline_manifest.json',baseline_entries)
    dump(base/'SOURCE_BINDING.json',{'source':str(source),'source_files':source_hashes,'pairs':bindings,
         'original_truth_recovered_by_exact_replay':True,'replay_verification_sha256':sha(base/'REPLAY_VERIFICATION.json'),
         'baseline_state_metrics_source':'Original matrix_results.csv; original probability arrays were not in recovered ZIP prefix',
         'baseline_matrix_results_sha256':sha(base/'baseline_matrix_results.csv')})
    dump(base/'GENERATION_MANIFEST.json',{'new_arms':180,'baseline_arms':180,'new_independent_latent_matrices_generated':0,'original_truth_files_replayed_byte_identically':180,
         'source':str(source),'plan_sha256':sha(HERE/'ANALYSIS_PLAN.txt'),'design_freeze_sha256':sha(HERE/'DESIGN_FREEZE.json'),
         'preparation_code_sha256':sha(__file__),'mask_code_sha256':sha(HERE/'fixed_target_design.py'),
         'input_manifest_sha256':sha(base/'input_manifest.json'),'source_binding_sha256':sha(base/'SOURCE_BINDING.json'),
         'permutation_seed':PERMUTATION_SEED,'propagation_seed':PROPAGATION_SEED})
    print(json.dumps({'prepared_new_arms':180,'retained_original_empirical_arms':180,'path':str(base)}),flush=True)

def fit_one(base,entry):
    # No truth, scaffold, q, or source-registry file is opened by this worker.
    start=time.monotonic();base=Path(base);masked_path=base/entry['masked_file'];out=base/'predictions'
    assert sha(masked_path)==entry['masked_sha256']
    with np.load(masked_path) as d:masked=d['X'];target=d['target_mask'].astype(bool)
    result=WORKER.fit_targets(masked,target)
    meta={'matrix_id':entry['matrix_id'],'masked_sha256':entry['masked_sha256'],'diagnostics':result['diagnostics'],
          'nj_models':result['models'],'failures':result['failures'],'target_columns':result['target_columns'].tolist(),
          'adapter_sha256':sha(WORKER.HERE/'frozen_methods/fit_adapter.py'),'mk_module_sha256':sha(WORKER.HERE/'frozen_methods/mk_baseline.py')}
    if result['failures']:
        meta.update(status='failed',seconds=time.monotonic()-start);dump(out/f"{entry['matrix_id']}.json",meta)
        return {'matrix_id':entry['matrix_id'],'status':'failed','failures':result['failures']}
    p=result['probabilities'];assert np.isfinite(p).all() and np.allclose(p.sum(2),1)
    seed=[PROPAGATION_SEED,entry['panel_ordinal'],entry['q_ordinal'],entry['replicate']]
    marginal=seed+[0];joint=seed+[1]
    uniforms=np.random.default_rng(np.random.SeedSequence(marginal)).random((N_DRAWS,len(result['target_indices'])))
    draws=[(uniforms<p[i,:,1][None,:]).astype(np.uint8) for i in range(5)]
    draws.append(WORKER.joint_targets(masked,target,result['models'],N_DRAWS,np.random.SeedSequence(joint)))
    packed=np.asarray([np.packbits(d,axis=1) for d in draws]);path=out/f"{entry['matrix_id']}.npz"
    np.savez_compressed(path,probabilities=p,missing_states_packed=packed,target_indices=result['target_indices'].astype(np.int16))
    meta.update(status='completed',seconds=time.monotonic()-start,prediction_sha256=sha(path),
       constant_response_count=sum(d['constant_fallback'] for d in result['diagnostics']),
       convergence_warning_count=sum(d['classifiers']['logistic'].get('convergence_warning',False) for d in result['diagnostics']),
       marginal_uniform_seed_sequence=marginal,joint_seed_sequence=joint,n_draws=N_DRAWS,n_targets=len(result['target_indices']),
       schemes=list(SCHEMES),completion_sha256=[hashlib.sha256(d.tobytes()).hexdigest() for d in draws],
       boundary='Worker opens masked X and target mask only; original fitting functions unchanged')
    meta_path=out/f"{entry['matrix_id']}.json";dump(meta_path,meta)
    return {'matrix_id':entry['matrix_id'],'status':'completed','seconds':meta['seconds'],
            'prediction_sha256':sha(path),'metadata_sha256':sha(meta_path)}

def fit(base,workers,source_override=None):
    base=Path(base);source=bound_source(base,source_override);entries=read(base/'input_manifest.json')
    assert len(entries)==180 and sha(base/'input_manifest.json')==read(base/'GENERATION_MANIFEST.json')['input_manifest_sha256']
    (base/'predictions').mkdir(exist_ok=False);start=time.monotonic();records=[]
    with ProcessPoolExecutor(max_workers=workers,initializer=initialize,initargs=(source,)) as pool:
        jobs={pool.submit(fit_one,base,e):e for e in entries}
        for future in as_completed(jobs):
            e=jobs[future]
            try:result=future.result()
            except Exception as error:
                result={'matrix_id':e['matrix_id'],'status':'worker_exception','type':type(error).__name__,
                        'message':str(error),'traceback':traceback.format_exc()};dump(base/'predictions'/f"{e['matrix_id']}_failure.json",result)
            records.append(result)
            print(json.dumps({'finished_new_arms':len(records),'total_new_arms':180,'failed':sum(r['status']!='completed' for r in records),'seconds':time.monotonic()-start}),flush=True)
    records.sort(key=lambda r:r['matrix_id'])
    dump(base/'FIT_MANIFEST.json',{'completed_arms':sum(r['status']=='completed' for r in records),
      'failed_arms':[r for r in records if r['status']!='completed'],'workers':workers,'seconds':time.monotonic()-start,
      'script_sha256':sha(__file__),'source_binding_sha256':sha(base/'SOURCE_BINDING.json'),
      'input_manifest_sha256':sha(base/'input_manifest.json'),'records':records})

def crps(draws,truth,fair=False):
    x=np.sort(np.asarray(draws,dtype=float));n=len(x)
    pair=float(np.dot(2*np.arange(1,n+1)-n-1,x))
    return float(np.abs(x-truth).mean()-pair/(n*(n-1) if fair else n*n))

def evaluate_one(base,entry):
    result=CORE.evaluate_one(base,entry)
    with np.load(Path(base)/'evaluation'/f"{entry['matrix_id']}.npz") as saved:
        draws=saved['D_samples'];truth=int(saved['true_D'])
    for k,row in enumerate(result):
        row['crps_empirical']=crps(draws[k],truth)
        row['crps_fair_ensemble']=crps(draws[k],truth,True)
    if entry['arm']=='empirical':
        with np.load(Path(base)/'baseline_evaluation'/f"{entry['matrix_id']}.npz") as original, np.load(Path(base)/'evaluation'/f"{entry['matrix_id']}.npz") as current:
            assert set(original.files)==set(current.files)
            for key in current.files:assert np.array_equal(original[key],current[key]),(entry['matrix_id'],key)
    return result

def baseline_rows(base):
    binding=read(base/'SOURCE_BINDING.json')
    assert sha(base/'baseline_matrix_results.csv')==binding['baseline_matrix_results_sha256']
    records={}
    for raw in rows(base/'baseline_matrix_results.csv'):
        if raw['arm']!='empirical':continue
        converted={}
        for key,value in raw.items():
            try: converted[key]=float(value) if any(c in value for c in '.eE') else int(value)
            except ValueError: converted[key]=value
        records[(raw['matrix_id'],raw['scheme'])]=converted
    bindings={b['empirical_id']:b for b in binding['pairs']};result=[]
    for entry in read(base/'baseline_manifest.json'):
        mid=entry['matrix_id'];b=bindings[mid]
        assert sha(base/entry['masked_file'])==b['source_input_sha256']
        with np.load(base/entry['masked_file']) as d:masked=d['X'];target=d['target_mask'].astype(bool)
        truth_path=base/'truth'/f"{entry['pair_id']}.npz"
        assert sha(truth_path)==b['source_truth_sha256']
        with np.load(truth_path) as d:truth=d['X_true'];edges=d['edges'];root=int(d['root'])
        oldl,oldm=CORE.total_costs(masked,edges,root)
        tl,tm=CORE.completion_costs(masked,target,truth[target][None,:],edges,root)
        true_d=int(tl[0]-oldl-(tm[0]-oldm))
        path=base/'baseline_evaluation'/f'{mid}.npz';assert sha(path)==b['source_evaluation_sha256']
        with np.load(path) as d:
            for key,value in [('old_L',oldl),('old_M',oldm),('true_L',int(tl[0])),('true_M',int(tm[0])),('true_D',true_d)]:
                assert int(d[key])==value,(mid,key)
            ds=d['D_samples'];assert ds.shape==(6,512)
            assert np.array_equal(ds,d['L_samples']-oldl-(d['M_samples']-oldm))
            hard=d['hard_D']
        for k,scheme in enumerate(SCHEMES):
            row=records[(mid,scheme)];draws=ds[k];low,high=np.quantile(draws,[.025,.975],method='linear')
            checks={'true_D':true_d,'hard_D':hard[k],'mean_D':draws.mean(),'mean_D_error':draws.mean()-true_d,
                'mean_D_absolute_error':abs(draws.mean()-true_d),'sd_D':draws.std(ddof=1),
                'lower95':low,'upper95':high,'interval_width':high-low,
                'truth_in_interval':int(low<=true_d<=high),'truth_below_interval':int(true_d<low),'truth_above_interval':int(true_d>high)}
            for key,value in checks.items():assert np.isclose(row[key],value,atol=1e-12,rtol=0),(mid,scheme,key)
            row['crps_empirical']=crps(draws,true_d);row['crps_fair_ensemble']=crps(draws,true_d,True);result.append(row)
        shutil.copyfile(path,base/'evaluation'/path.name)
    assert len(result)==1080
    return result

def summarize(base,data):
    scenarios=rows(base/'scenario_registry.csv');summaries=[];contrasts=[]
    metrics=(*CORE.METRICS,'crps_empirical','crps_fair_ensemble')
    paired_metrics=(*CORE.PAIRED_METRICS,'crps_empirical','crps_fair_ensemble')
    lookup={(r['pair_id'],r['arm'],r['scheme']):r for r in data}
    for scenario in scenarios:
      for scheme in SCHEMES:
        for arm in ARMS:
          subset=[r for r in data if r['scenario_id']==scenario['scenario_id'] and r['arm']==arm and r['scheme']==scheme]
          assert len(subset)==30
          hits=sum(r['truth_in_interval'] for r in subset);lower,upper=CORE.wilson(hits,30)
          out={**scenario,'arm':arm,'scheme':scheme,'n_matrices':30,'inclusion_count':hits,'inclusion_fraction':hits/30,
               'inclusion_wilson_lower95':lower,'inclusion_wilson_upper95':upper,
               'truth_below_count':sum(r['truth_below_interval'] for r in subset),'truth_above_count':sum(r['truth_above_interval'] for r in subset)}
          for key in metrics:
            v=np.array([r[key] for r in subset]);out[key+'_mean']=float(v.mean());out[key+'_mcse']=float(v.std(ddof=1)/np.sqrt(30))
          out['zero_true_probability_total']=sum(r['zero_true_probability_count'] for r in subset)
          out['log_floor_total']=sum(r['log_floor_count'] for r in subset);summaries.append(out)
        ids=[r['pair_id'] for r in data if r['scenario_id']==scenario['scenario_id'] and r['arm']=='empirical' and r['scheme']==scheme]
        assert len(ids)==len(set(ids))==30
        out={**scenario,'scheme':scheme,'n_pairs':30,'direction':'empirical_minus_fixed_target_permuted'}
        for key in paired_metrics:
          v=np.array([lookup[(i,ARMS[0],scheme)][key]-lookup[(i,ARMS[1],scheme)][key] for i in ids])
          out[key+'_difference_mean']=float(v.mean());out[key+'_difference_mcse']=float(v.std(ddof=1)/np.sqrt(30))
        contrasts.append(out)
    csvwrite(base/'matrix_results.csv',data);csvwrite(base/'scenario_summary.csv',summaries);csvwrite(base/'paired_layout_contrasts.csv',contrasts)

def evaluate(base,workers,source_override=None):
    base=Path(base);source=bound_source(base,source_override);initialize(source)
    fit=read(base/'FIT_MANIFEST.json');assert fit['completed_arms']==180 and not fit['failed_arms']
    assert sha(base/'input_manifest.json')==fit['input_manifest_sha256']
    for r in fit['records']:
        assert sha(base/'predictions'/f"{r['matrix_id']}.npz")==r['prediction_sha256']
        assert sha(base/'predictions'/f"{r['matrix_id']}.json")==r['metadata_sha256']
    entries=read(base/'input_manifest.json');(base/'evaluation').mkdir(exist_ok=False)
    started=time.monotonic();by_id={}
    with ProcessPoolExecutor(max_workers=workers,initializer=initialize,initargs=(source,)) as pool:
        jobs={pool.submit(evaluate_one,base,e):e for e in entries}
        for f in as_completed(jobs):
            e=jobs[f];by_id[e['matrix_id']]=f.result()
            if len(by_id)%10==0:print(json.dumps({'evaluated_new_arms':len(by_id),'total_new_arms':180,'seconds':time.monotonic()-started}),flush=True)
    data=baseline_rows(base)+[r for e in entries for r in by_id[e['matrix_id']]];summarize(base,data)
    dump(base/'EVALUATION_MANIFEST.json',{'new_arms':180,'original_empirical_D_truth_and_summaries_rechecked':180,'rows':len(data),
      'original_empirical_D_truth_recomputed':True,'original_empirical_D_summaries_match':True,
      'original_empirical_probabilities_and_completions_available':False,'workers':workers,'seconds':time.monotonic()-started,
      'script_sha256':sha(__file__),'fit_manifest_sha256':sha(base/'FIT_MANIFEST.json'),
      'source_binding_sha256':sha(base/'SOURCE_BINDING.json'),
      'evaluation_files':{e['matrix_id']:sha(base/'evaluation'/f"{e['matrix_id']}.npz") for e in entries+read(base/'baseline_manifest.json')}})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','fit','evaluate'])
    p.add_argument('--source',type=Path);p.add_argument('--replayed-inputs',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=6)
    a=p.parse_args()
    if a.action=='prepare':prepare(a.source,a.output,a.replayed_inputs)
    elif a.action=='fit':fit(a.output,a.workers,a.source)
    else:evaluate(a.output,a.workers,a.source)
