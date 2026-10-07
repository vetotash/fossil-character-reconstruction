#!/usr/bin/env python3
"""Frozen old-label clade-block diagnostic using authenticated archived fitters.

Later labels are never read. ``plan`` uses target-ledger column identities only;
``fit`` masks old response labels and saves forecasts; ``evaluate`` joins the
held-out old labels only after complete forecast artifacts exist.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
import time
import traceback

import numpy as np
import pandas as pd
import scipy
import sklearn

PACKAGE_ROOT=Path(__file__).resolve().parents[1]
SEED=2026100407
PANELS=('tetrapods','dinosaurs','ichthyosaurs')
MODELS=('mode','knn5','logistic','extra_trees','nj_mk')
TREE_NAMES={'tetrapods':'O1','dinosaurs':'baron','ichthyosaurs':'moon_source_0001'}
SINGLETONS=set('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat()
def stable_key(*parts):
    return hashlib.sha256(json.dumps([SEED,*parts],separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def write_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def known(matrix): return np.isin(matrix,list(SINGLETONS))


def newick_descendants(text):
    """Parse labels/topology only; retain serialized source rooting and clades."""
    text=re.sub(r'\[[^\]]*\]','',text)
    tokens=re.findall(r"'(?:[^']|'')*'|[(),:;]|[^\s(),:;]+",text)
    position=0;internal=[];leaves=[]
    def subtree():
        nonlocal position
        if tokens[position]=='(':
            position+=1;desc=list(subtree())
            while tokens[position]==',':
                position+=1;desc.extend(subtree())
            assert tokens[position]==')';position+=1
            if position<len(tokens) and tokens[position] not in ',():;': position+=1
            internal.append(tuple(sorted(desc)))
        else:
            token=tokens[position];position+=1
            if token.startswith("'"): token=token[1:-1].replace("''", "'")
            desc=[token];leaves.append(token)
        if position<len(tokens) and tokens[position]==':': position+=2
        return desc
    all_tips=subtree()
    if position<len(tokens) and tokens[position]==';':position+=1
    assert position==len(tokens) and len(all_tips)==len(set(all_tips))
    return leaves,sorted(set(internal))


def source_paths(source_root,panel):
    directory=source_root/('ichthyosaur_refit' if panel=='ichthyosaurs' else f'old_refits/inputs/{panel}')
    return {'matrix':directory/'source_training.npz','target_ledger':directory/'frozen_targets.csv',
            'grouping_trees':source_root/f'downstream/{panel}/reference_trees.nex'}


def load_panel(source_root,panel):
    paths=source_paths(source_root,panel)
    with np.load(paths['matrix'],allow_pickle=False) as source:
        matrix=source['A'];taxa=source['old_names'].tolist();characters=source['old_char_numbers'].astype(int).tolist()
    assert matrix.shape==(len(taxa),len(characters)) and len(set(taxa))==len(taxa)
    assert len(set(characters))==len(characters)
    # Deliberately read identities only, never target 'true' or other outcomes.
    ledger=pd.read_csv(paths['target_ledger'],usecols=['scenario','character'])
    responses=sorted(set(ledger.loc[ledger.scenario.eq('later_publication'),'character'].astype(int)))
    assert responses and set(responses)<=set(characters)
    tree_records=re.findall(r'(?im)^TREE\s+(\S+)\s*=\s*(.*?);',paths['grouping_trees'].read_text())
    source_tree=dict(tree_records)[TREE_NAMES[panel]]+';'
    tips,clades=newick_descendants(source_tree)
    assert set(tips)<=set(taxa), 'Tree labels must be exact source-label matches'
    return matrix,taxa,characters,responses,tips,clades,paths


def select_clades(matrix,taxa,characters,responses,clades,panel,
                  max_clades=4,max_columns=12,min_inside=4,min_outside=15):
    row_lookup={t:i for i,t in enumerate(taxa)}
    col_lookup={c:j for j,c in enumerate(characters)}
    available=known(matrix);total=available.sum(axis=0)
    max_tips=min(12,len(taxa)//4)
    candidates=[];exclusions={'below_minimum_tips':0,'above_maximum_tips':0,'no_eligible_response':0}
    for clade in sorted(set(tuple(sorted(c)) for c in clades)):
        if len(clade)<4:exclusions['below_minimum_tips']+=1;continue
        if len(clade)>max_tips:exclusions['above_maximum_tips']+=1;continue
        rows=np.array([row_lookup[t] for t in clade],dtype=int)
        inside=available[rows].sum(axis=0);eligible=[];column_exclusions={'too_few_inside':0,'too_few_outside':0}
        for character in responses:
            j=col_lookup[character]
            if inside[j]<min_inside:column_exclusions['too_few_inside']+=1;continue
            if total[j]-inside[j]<min_outside:column_exclusions['too_few_outside']+=1;continue
            eligible.append({'character':character,'column_index':j,
                             'inside_labels':int(inside[j]),'outside_labels':int(total[j]-inside[j])})
        if not eligible:exclusions['no_eligible_response']+=1;continue
        key=stable_key('clade',panel,list(clade))
        eligible.sort(key=lambda c:stable_key('response',panel,key,c['character']))
        candidates.append({'clade_key':key,'taxa':list(clade),'source_row_indices':rows.tolist(),
                           'eligible_response_count':len(eligible),'eligible_responses':eligible,
                           'response_exclusion_counts':column_exclusions})
    candidates.sort(key=lambda c:c['clade_key'])
    selected=[];used=set()
    for candidate in candidates:
        candidate['selected']=False
        if len(selected)>=max_clades:candidate['selection_exclusion']='four_clade_limit';continue
        if used.intersection(candidate['taxa']):candidate['selection_exclusion']='overlaps_selected_clade';continue
        candidate['selected']=True;used.update(candidate['taxa'])
        selected.append({**candidate,'clade_id':f'{panel}_clade{len(selected)+1:02d}',
                         'selected_responses':candidate['eligible_responses'][:max_columns]})
    return selected,{'candidate_clades':candidates,'topology_exclusions':exclusions,
                    'internal_descendant_sets':len(set(tuple(sorted(c)) for c in clades)),
                    'eligible_clade_count':len(candidates),'selected_clade_count':len(selected),
                    'source_response_pool_count':len(responses),'max_clade_tips':max_tips}


def matched_tasks(matrix,taxa,selected,panel,random_controls=3):
    available=known(matrix);tasks=[]
    for clade in selected:
        for response in clade['selected_responses']:
            j=response['column_index'];character=response['character']
            inside=np.array(clade['source_row_indices'],int)
            blocked=sorted(inside[available[inside,j]].tolist())
            pool=np.flatnonzero(available[:,j])
            case=f"{clade['clade_id']}_char{character:04d}"
            base={'panel':panel,'clade_id':clade['clade_id'],'clade_key':clade['clade_key'],
                  'case_id':case,'character':character,'column_index':j,
                  'heldout_count':len(blocked),'old_singleton_count':len(pool),
                  'clade_taxa':clade['taxa'],'clade_row_indices':clade['source_row_indices']}
            tasks.append({**base,'task_id':case+'_blocked','treatment':'blocked','control_index':None,
                          'target_row_indices':blocked,'target_taxa':[taxa[i] for i in blocked]})
            for control in range(random_controls):
                seed=int(stable_key('random_control',panel,clade['clade_key'],character,control)[:16],16)
                rng=np.random.default_rng(seed)
                rows=sorted(rng.choice(pool,size=len(blocked),replace=False).tolist())
                tasks.append({**base,'task_id':case+f'_random{control+1:02d}',
                              'treatment':'random','control_index':control+1,'random_seed':seed,
                              'target_row_indices':rows,'target_taxa':[taxa[i] for i in rows]})
    assert len({t['task_id'] for t in tasks})==len(tasks)
    return tasks


def prepare(source_root):
    freeze=json.loads((PACKAGE_ROOT/'protocol_freeze.json').read_text())
    assert digest(PACKAGE_ROOT/'DESIGN.md')==freeze['protocol_sha256']
    destination=PACKAGE_ROOT/'split_plan.json'
    if destination.exists():raise FileExistsError('Split plan already frozen')
    manifest=[];panels={};tasks=[]
    for panel in PANELS:
        matrix,taxa,characters,responses,tips,clades,paths=load_panel(source_root,panel)
        selected,audit=select_clades(matrix,taxa,characters,responses,clades,panel)
        panel_tasks=matched_tasks(matrix,taxa,selected,panel)
        panels[panel]={'shape':list(matrix.shape),'taxa':taxa,'characters':characters,
                       'source_tree':TREE_NAMES[panel],'tree_tip_count':len(tips),
                       'source_taxa_absent_from_grouping_tree':sorted(set(taxa)-set(tips)),
                       'selected_clades':selected,'selection_audit':audit,'task_count':len(panel_tasks)}
        tasks.extend(panel_tasks)
        for kind,path in paths.items():manifest.append({'panel':panel,'kind':kind,
                 'source_relative_path':str(path.relative_to(source_root)),
                 'sha256':digest(path),'bytes':path.stat().st_size})
    vendor_records=[]
    source_by_name={'instrumentation.py':source_root/'old_refits/run_refits.py',
                    **{name:source_root/'old_refits/vendor'/name for name in ['pilot.py','dinosaur_matched_pilot.py','mk_baseline.py']}}
    for name,source in source_by_name.items():
        local=PACKAGE_ROOT/'vendor'/name
        assert digest(source)==digest(local)
        vendor_records.append({'local':'vendor/'+name,'source_relative_path':str(source.relative_to(source_root)),
                               'sha256':digest(local)})
    output={'created_utc':now(),'protocol_sha256':freeze['protocol_sha256'],
            'source_root':str(source_root.resolve()),'source_manifest':manifest,
            'fitters':vendor_records,'panels':panels,'tasks':tasks,
            'versions':{x.__name__:x.__version__ for x in [np,pd,scipy,sklearn]},
            'later_state_values_read':False,'prediction_outcomes_inspected':False,
            'task_count':len(tasks),'maximum_method_fits':len(tasks)*len(MODELS)}
    write_json(destination,output)
    write_json(PACKAGE_ROOT/'split_plan_freeze.json',{'created_utc':now(),'split_plan_sha256':digest(destination),
                                            'task_count':len(tasks)})
    print(json.dumps({p:{'eligible_clades':panels[p]['selection_audit']['eligible_clade_count'],
                          'selected_clades':len(panels[p]['selected_clades']),
                          'selected_response_splits':panels[p]['task_count']//4,
                          'tasks':panels[p]['task_count']} for p in PANELS},indent=2))


def load_fitters():
    sys.dont_write_bytecode=True
    sys.path.insert(0,str(PACKAGE_ROOT/'vendor'))
    import dinosaur_matched_pilot as dp
    import mk_baseline as mk
    spec=importlib.util.spec_from_file_location('blocked_archived_instrumentation',PACKAGE_ROOT/'vendor/instrumentation.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return dp,mk,module


def masked_input(matrix,column,targets):
    targets=np.asarray(targets,dtype=int)
    assert len(targets)==len(set(targets.tolist())) and known(matrix[targets,column]).all()
    result=matrix.copy();result[targets,column]='?'
    train=np.flatnonzero(known(result[:,column]))
    assert not set(train)&set(targets)
    assert np.all(result[targets,column]=='?')
    assert np.array_equal(np.delete(matrix,column,axis=1),np.delete(result,column,axis=1))
    return result,train


def fit_task(payload):
    source_root_s,task=payload;source_root=Path(source_root_s)
    path=source_paths(source_root,task['panel'])['matrix']
    with np.load(path,allow_pickle=False) as archive:
        source=archive['A'];taxa=archive['old_names'].tolist()
    j=task['column_index'];test=np.array(task['target_row_indices'],int)
    matrix,train=masked_input(source,j,test)
    assert len(train)>=15
    if task['treatment']=='blocked':
        clade=np.array(task['clade_row_indices'],int)
        assert not known(matrix[clade,j]).any()
        assert sorted(clade[known(source[clade,j])].tolist())==test.tolist()
    dp,mk,instrument=load_fitters()
    started=time.monotonic();predictions=[];failures=[];diagnostics={}
    try:
        methods,diagnostics['classifiers']=instrument.instrumented_predict(dp,matrix,j,train,test)
    except Exception as exc:
        methods={};failures.append({'models':list(MODELS[:4]),'error':repr(exc),'traceback':traceback.format_exc()})
    try:
        # Rebuild from this masked matrix and remove the entire response column.
        totals=mk.distance_totals(matrix)
        distances,zero=mk.excluding_distance(matrix,j,totals)
        edges,njdiag=mk.neighbor_join(distances)
        labels,prob,fit,diag=instrument.instrumented_mk(mk,edges,len(taxa),matrix[:,j],train,test)
        methods['nj_mk']=(labels,prob)
        diagnostics['nj_mk']={'zero_overlap_pairs':zero,**njdiag,**fit,**diag,
                               'edges_sha256':hashlib.sha256(json.dumps(edges).encode()).hexdigest(),
                               'response_character_excluded':True,'tree_rebuilt_for_this_treatment':True}
    except Exception as exc:
        failures.append({'models':['nj_mk'],'error':repr(exc),'traceback':traceback.format_exc()})
    expected_classes=sorted(set(matrix[train,j].tolist()))
    for method,(labels,prob) in methods.items():
        assert labels.tolist()==expected_classes
        assert prob.shape==(len(test),len(labels)) and np.isfinite(prob).all()
        assert np.all(prob>=0) and np.allclose(prob.sum(1),1,rtol=0,atol=1e-12)
        for k,row in enumerate(test):
            predictions.append({'task_id':task['task_id'],'panel':task['panel'],'case_id':task['case_id'],
                'clade_id':task['clade_id'],'treatment':task['treatment'],'control_index':task['control_index'],
                'character':task['character'],'column_index':j,'taxon':taxa[row],'row_index':int(row),
                'method':method,'training_count':len(train),'classes':labels.tolist(),
                'probabilities':prob[k].tolist(),'predicted_state':str(labels[np.argmax(prob[k])])})
    result={'task':task,'source_sha256':digest(path),'training_indices':train.tolist(),
            'training_classes':expected_classes,'target_labels_masked':True,
            'later_labels_used':False,'predictions':predictions,'diagnostics':diagnostics,
            'failures':failures,'seconds':time.monotonic()-started,
            'fitters_sha256':{str(p.name):digest(p) for p in (PACKAGE_ROOT/'vendor').glob('*.py')}}
    out=PACKAGE_ROOT/'results/tasks'/f"{task['task_id']}.json"
    if out.exists():raise FileExistsError(out)
    write_json(out,result)
    return {'task_id':task['task_id'],'seconds':result['seconds'],'failures':len(failures)}


def frozen_plan(source_root_override=None):
    plan=PACKAGE_ROOT/'split_plan.json';freeze=json.loads((PACKAGE_ROOT/'split_plan_freeze.json').read_text())
    assert digest(plan)==freeze['split_plan_sha256']
    data=json.loads(plan.read_text())
    source=Path(data['source_root']) if source_root_override is None else Path(source_root_override).resolve()
    for record in data['source_manifest']:
        assert digest(source/record['source_relative_path'])==record['sha256']
    for record in data['fitters']:assert digest(PACKAGE_ROOT/record['local'])==record['sha256']
    # A path override changes storage location only; every source hash is still
    # verified against the unchanged frozen plan and all split keys are retained.
    data['source_root']=str(source)
    return data


def fit_all(workers=2,limit=None,source_root_override=None):
    plan=frozen_plan(source_root_override);pending=[t for t in plan['tasks'] if not (PACKAGE_ROOT/'results/tasks'/f"{t['task_id']}.json").exists()]
    if limit is not None:pending=pending[:limit]
    print(f'Pending character treatments: {len(pending)}',flush=True)
    start=time.monotonic();out=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for k,result in enumerate(pool.map(fit_task,[(plan['source_root'],t) for t in pending]),1):
            out.append(result)
            if k%12==0 or k==len(pending):
                print(json.dumps({'completed_now':k,'remaining_now':len(pending)-k,
                                  'elapsed_seconds':round(time.monotonic()-start,2),
                                  'failed_tasks':sum(r['failures']>0 for r in out)}),flush=True)
    write_json(PACKAGE_ROOT/'results/last_fit_run.json',{'completed_utc':now(),'tasks':out,'seconds':time.monotonic()-start})


def evaluate(source_root_override=None):
    plan=frozen_plan(source_root_override);source=Path(plan['source_root']);all_predictions=[];task_results=[]
    for task in plan['tasks']:
        path=PACKAGE_ROOT/'results/tasks'/f"{task['task_id']}.json"
        if not path.exists():raise RuntimeError(f'Missing prediction artifact: {path}')
        result=json.loads(path.read_text());assert result['task']==task
        all_predictions.extend(result['predictions']);task_results.append(result)
    # Freeze the forecast-only artifact before opening old label arrays for scoring.
    pred_path=PACKAGE_ROOT/'results/predictions_before_evaluation.json'
    write_json(pred_path,all_predictions);forecast_hash=digest(pred_path)
    matrices={p:np.load(source_paths(source,p)['matrix'],allow_pickle=False)['A'] for p in PANELS}
    cell_rows=[]
    for p in all_predictions:
        truth=str(matrices[p['panel']][p['row_index'],p['column_index']])
        assert truth in SINGLETONS
        probabilities=dict(zip(p['classes'],p['probabilities']));pt=float(probabilities.get(truth,0.0))
        cell_rows.append({**{k:v for k,v in p.items() if k not in ('classes','probabilities')},
                         'true_old_label':truth,'accuracy':int(p['predicted_state']==truth),
                         'brier':sum(v*v for v in probabilities.values())-2*pt+1,
                         'log_loss':-math.log(max(pt,1e-12)),
                         'zero_true_probability':int(pt==0),'training_unseen_state':int(truth not in probabilities),
                         'constant_training':int(len(probabilities)==1),'p_true':pt})
    assert digest(pred_path)==forecast_hash
    cells=pd.DataFrame(cell_rows);cells.to_csv(PACKAGE_ROOT/'results/cell_scores.csv',index=False)
    metrics=['accuracy','brier','log_loss','zero_true_probability','training_unseen_state','constant_training']
    keys=['panel','clade_id','case_id','task_id','treatment','method']
    summaries=cells.groupby(keys,sort=True).agg(n_targets=('accuracy','size'),
                    **{metric:(metric,'mean') for metric in metrics}).reset_index()
    summaries.to_csv(PACKAGE_ROOT/'results/treatment_scores.csv',index=False)
    differences=[];missing=[]
    for task in [t for t in plan['tasks'] if t['treatment']=='blocked']:
        case=task['case_id']
        for method in MODELS:
            group=summaries[(summaries.case_id==case)&(summaries.method==method)]
            blocked=group[group.treatment.eq('blocked')];random=group[group.treatment.eq('random')]
            if len(blocked)!=1 or len(random)!=3:
                missing.append({'case_id':case,'method':method,'blocked_records':len(blocked),'random_records':len(random)});continue
            assert (group.n_targets==task['heldout_count']).all()
            record={'panel':task['panel'],'clade_id':task['clade_id'],'case_id':case,
                    'character':task['character'],'method':method,'n_targets_per_treatment':task['heldout_count'],
                    'random_controls':3,'difference_direction':'blocked_minus_mean_random'}
            for metric in metrics:
                value=float(blocked.iloc[0][metric]);control=float(random[metric].mean())
                record[metric+'_blocked']=value;record[metric+'_random_mean']=control
                record[metric+'_difference']=value-control
            differences.append(record)
    paired=pd.DataFrame(differences);paired.to_csv(PACKAGE_ROOT/'results/paired_split_scores.csv',index=False)
    value_columns=[c for c in paired if any(c.endswith(suffix) for suffix in ('_blocked','_random_mean','_difference'))]
    clades=paired.groupby(['panel','clade_id','method'],sort=True).agg(
        n_matched_characters=('case_id','size'),**{c:(c,'mean') for c in value_columns}).reset_index()
    clades.to_csv(PACKAGE_ROOT/'results/clade_scores.csv',index=False)
    panels=clades.groupby(['panel','method'],sort=True).agg(
        n_clades=('clade_id','size'),**{c:(c,'mean') for c in value_columns}).reset_index()
    panels.to_csv(PACKAGE_ROOT/'results/panel_scores.csv',index=False)
    manifest={'completed_utc':now(),'status':'PASS' if not missing and all(not r['failures'] for r in task_results) else 'COMPLETED_WITH_RECORDED_FAILURES',
              'protocol_sha256':plan['protocol_sha256'],'split_plan_sha256':digest(PACKAGE_ROOT/'split_plan.json'),
              'prediction_artifact_sha256':forecast_hash,'tasks':len(task_results),'prediction_rows':len(cells),
              'method_task_failures':[{'task_id':r['task']['task_id'],'failures':r['failures']} for r in task_results if r['failures']],
              'incomplete_matched_comparisons':missing,'later_labels_used':False,
              'summary_weighting':'Equal character weights within each clade, then equal clade weights within panel; no pooled-cell SE or population-level CI.',
              'inference_scope':'Old full-source label extrapolation diagnostic; distinct from historical-target accuracy and original matched-control masking.',
              'versions':{x.__name__:x.__version__ for x in [np,pd,scipy,sklearn]},
              'script_sha256':digest(__file__)}
    manifest['source_root_used']=str(source)
    write_json(PACKAGE_ROOT/'results/EVALUATION_MANIFEST.json',manifest)
    print(json.dumps(manifest,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('operation',choices=['plan','fit','evaluate'])
    p.add_argument('--source-root',type=Path);p.add_argument('--workers',type=int,default=2);p.add_argument('--limit',type=int)
    args=p.parse_args()
    if args.operation=='plan':
        if args.source_root is None:p.error('--source-root is required for plan')
        prepare(args.source_root.resolve())
    elif args.operation=='fit':fit_all(args.workers,args.limit,args.source_root)
    else:evaluate(args.source_root)


if __name__=='__main__':main()
