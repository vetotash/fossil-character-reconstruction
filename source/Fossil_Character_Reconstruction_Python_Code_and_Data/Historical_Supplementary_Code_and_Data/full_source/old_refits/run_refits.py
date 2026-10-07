"""Round 14 diagnostic replay and full-source sensitivity with frozen predictors.

No parameter tuning or new targets. Models call byte-identical original modules.
Two conditions per history: shared hides frozen old controls, full_source restores
all observed old responses. Later labels join prediction scoring only.
"""
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor
import argparse, hashlib, json, os, shutil, sys, time, warnings
import numpy as np
import pandas as pd
import scipy, sklearn
from sklearn.exceptions import ConvergenceWarning

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]/'round12/restored/analysis/morphology_workspace/morphology_project'
MODELS=['mode','knn5','logistic','extra_trees','nj_mk']
SHA=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
NOW=lambda:datetime.now(timezone.utc).isoformat()

def prepare():
    assert not (HERE/'FIT_FREEZE.json').exists(), 'Preparation exists; do not overwrite'
    assert PROJECT.is_dir(),PROJECT
    (HERE/'vendor').mkdir(exist_ok=True)
    originals=[]
    for rel in ['dinosaur_matched_pilot.py','pilot.py','round2/mk_baseline.py','round2/run_comparison.py','round3/shared_training/run_shared.py']:
        src=PROJECT/rel;dst=HERE/'vendor'/Path(rel).name
        shutil.copyfile(src,dst);originals.append({'source':str(src),'local':str(dst.relative_to(HERE)),'sha256':SHA(src)})
    for lineage in ['tetrapods','dinosaurs']:
        dest=HERE/'inputs'/lineage;dest.mkdir(parents=True,exist_ok=True)
        files={'source_training.npz':PROJECT/'round2/inputs'/f'{lineage}.npz',
               'frozen_targets.csv':PROJECT/'round2/inputs'/f'{lineage}_frozen_targets.csv',
               'estimated_trees.json':PROJECT/'round2'/lineage/'estimated_trees.json',
               'archived_shared_predictions.csv':PROJECT/'round3/shared_training'/lineage/'predictions.csv',
               'archived_full_predictions.csv':PROJECT/'round2'/lineage/'predictions.csv'}
        for name,src in files.items():
            dst=dest/name;shutil.copyfile(src,dst)
            originals.append({'source':str(src),'local':str(dst.relative_to(HERE)),'sha256':SHA(src)})
    versions={'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__}
    assert versions=={'numpy':'2.3.5','pandas':'2.2.3','scipy':'1.17.0','sklearn':'1.8.0'},versions
    freeze={'created_utc':NOW(),'models':MODELS,'conditions':['shared','full_source'],
      'parameters':'Original functions, unchanged parameters and seeds',
      'future_truth_use':'Prediction scoring only; no fitting or parameter selection',
      'source_tree_use':'Archived old-source NJ trees, response character excluded',
      'scope':'All original future targets 350 tetrapods and 1333 dinosaurs. Downstream endpoints independently frozen at 349 and 1333.',
      'historical_full_source_predictions':'Already present in round2 later_publication; replay with diagnostics. New downstream sensitivity is performed separately.',
      'versions':versions,'files':originals,'runner_sha256':SHA(__file__)}
    (HERE/'FIT_FREEZE.json').write_text(json.dumps(freeze,indent=2))
    print(json.dumps(freeze,indent=2))


def load_modules():
    sys.path.insert(0,str(HERE/'vendor'))
    import dinosaur_matched_pilot as dp
    import mk_baseline as mk
    return dp,mk


def instrumented_predict(dp,matrix,j,train,test):
    records=[];originals={'extra_trees':dp.ExtraTreesClassifier,'logistic':dp.LogisticRegression}
    def factory(name):
        def construct(*args,**kwargs):
            model=originals[name](*args,**kwargs);bound_fit=model.fit
            def captured_fit(*args,**kwargs):
                start=time.monotonic()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always');result=bound_fit(*args,**kwargs)
                ni=getattr(model,'n_iter_',None)
                records.append({'model':name,'seconds':time.monotonic()-start,
                    'n_iter':None if ni is None else np.asarray(ni).tolist(),
                    'reached_max_iter':None if ni is None else bool(np.any(np.asarray(ni)>=model.max_iter)),
                    'convergence_warning':any(issubclass(w.category,ConvergenceWarning) for w in caught),
                    'warnings':[{'category':w.category.__name__,'message':str(w.message)} for w in caught],
                    'parameters':model.get_params(),'classes':model.classes_.tolist(),
                    'n_features_in':int(model.n_features_in_),'fit_ran':True})
                return result
            model.fit=captured_fit;return model
        return construct
    dp.ExtraTreesClassifier,dp.LogisticRegression=factory('extra_trees'),factory('logistic')
    try: methods=dp.predict(matrix,j,train,test)
    finally:dp.ExtraTreesClassifier=originals['extra_trees'];dp.LogisticRegression=originals['logistic']
    if len(np.unique(matrix[train,j]))==1:
        records=[{'model':n,'fit_ran':False,'constant_class_fallback':True,'n_iter':None,'convergence_warning':False,'reached_max_iter':False,'warnings':[]} for n in originals]
    return methods,records


def instrumented_mk(mk,edges,ntaxa,observed,train,test):
    diag={};original=mk.minimize_scalar
    def capture(*args,**kwargs):
        result=original(*args,**kwargs)
        diag.update({'success':bool(result.success),'message':str(result.message),'nit':int(result.nit),
                     'nfev':int(result.nfev),'refined_logq':float(result.x),'refined_negative_loglik':float(result.fun),
                     'bounds':list(kwargs['bounds']),'method':kwargs['method'],'options':kwargs['options']})
        return result
    mk.minimize_scalar=capture;start=time.monotonic()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:labels,prob,fit=mk.predict_mk(edges,ntaxa,observed,train,test)
        finally:mk.minimize_scalar=original
    return labels,prob,fit,{'seconds':time.monotonic()-start,'optimizer':diag,
        'warnings':[{'category':w.category.__name__,'message':str(w.message)} for w in caught]}


def compare_archive(result,path,condition):
    old=pd.read_csv(path,keep_default_na=False,dtype={'true':str,'pred':str})
    if condition=='full_source':
        old=old[old.scenario.eq('later_publication')].copy();old['scenario']='later_target'
    keys=['lineage','scenario','character','taxon','model']
    paired=result.merge(old,on=keys,suffixes=('_refit','_archive'),validate='one_to_one')
    assert len(paired)==len(result)==len(old),(len(paired),len(result),len(old))
    assert paired.true_refit.eq(paired.true_archive).all()
    assert paired.pred_refit.eq(paired.pred_archive).all()
    assert paired.n_train_refit.eq(paired.n_train_archive).all()
    rows=[]
    for row in paired.itertuples():
        p,q=json.loads(row.state_probabilities_refit),json.loads(row.state_probabilities_archive)
        assert set(p)==set(q)
        delta=max(abs(p[s]-q[s]) for s in p)
        rows.append({'lineage':row.lineage,'scenario':row.scenario,'character':row.character,'taxon':row.taxon,'model':row.model,'max_probability_abs_difference':delta})
    errors=pd.DataFrame(rows)
    assert errors.max_probability_abs_difference.max()<1e-10,errors.max_probability_abs_difference.max()
    return errors,{'prediction_rows':len(result),'hard_predictions_exact':True,'state_sets_exact':True,
                  'target_truth_exact':True,'training_counts_exact':True,
                  'max_probability_abs_difference':float(errors.max_probability_abs_difference.max())}


def fit(lineage):
    dp,mk=load_modules();freeze=json.loads((HERE/'FIT_FREEZE.json').read_text())
    assert SHA(__file__)==freeze['runner_sha256']
    for record in freeze['files']:assert SHA(HERE/record['local'])==record['sha256'],record['local']
    inputs=HERE/'inputs'/lineage
    z=np.load(inputs/'source_training.npz',allow_pickle=False);a=z['A'];names=z['old_names'].tolist()
    charmap={int(c):j for j,c in enumerate(z['old_char_numbers'])};lookup={n:i for i,n in enumerate(names)}
    frozen=pd.read_csv(inputs/'frozen_targets.csv',keep_default_na=False,dtype={'true':str,'pred':str})
    future=frozen[frozen.model.eq('mode')&frozen.scenario.eq('later_publication')].copy()
    control=frozen[frozen.model.eq('mode')&frozen.scenario.eq('matched_concealment')].copy()
    tree_data=json.loads((inputs/'estimated_trees.json').read_text());trees={int(t['character']):t for t in tree_data}
    for condition in ['shared','full_source']:
        output=HERE/lineage/condition;output.mkdir(parents=True,exist_ok=True)
        assert not (output/'FIT_STARTED.json').exists(),'Existing attempt must be preserved'
        (output/'FIT_STARTED.json').write_text(json.dumps({'started_utc':NOW(),'freeze_sha256':SHA(HERE/'FIT_FREEZE.json')},indent=2))
        rows=[];fits=[];diagnostics=[];mk_models=[];started=time.monotonic()
        for number,(character,fg) in enumerate(future.groupby('character',sort=True)):
            character=int(character);j=charmap[character];cg=control[control.character.eq(character)]
            ci=np.array([lookup[n] for n in cg.taxon]);fi=np.array([lookup[n] for n in fg.taxon])
            assert len(ci)>0 and len(fi)>0 and not set(ci)&set(fi)
            assert np.array_equal(a[ci,j],cg.true.to_numpy()) and np.all(a[fi,j]=='?')
            matrix=a.copy()
            if condition=='shared':matrix[ci,j]='?';test=np.r_[ci,fi];truth=np.r_[cg.true.to_numpy(),fg.true.to_numpy()]
            else:test=fi;truth=fg.true.to_numpy()
            train=np.flatnonzero(mk.known(matrix[:,j]));assert not set(train)&set(test)
            edges=trees[character]['edges'];assert trees[character]['taxa']==names
            methods,classifier_diag=instrumented_predict(dp,matrix,j,train,test)
            labels,prob,mkfit,optimizer_diag=instrumented_mk(mk,edges,len(names),matrix[:,j],train,test)
            methods['nj_mk']=(labels,prob)
            diagnostics.append({'character':character,'classifiers':classifier_diag,'mk':optimizer_diag})
            fits.append({'lineage':lineage,'condition':condition,'character':character,'n_train':len(train),
                         'n_original_train':int(mk.known(a[:,j]).sum()),'n_controls':len(ci),'n_future':len(fi),**mkfit})
            mk_models.append({'character':character,'taxa':names,'edges':edges,'classes':labels.tolist(),
                              'train_indices':train.tolist(),'train_states':matrix[train,j].tolist(),
                              'target_indices':fi.tolist(),'control_indices':ci.tolist(),'rate':mkfit['rate'],
                              'constant_fallback':mkfit['constant_fallback']})
            for model in MODELS:
                labels,prob=methods[model]
                assert np.isfinite(prob).all() and np.all(prob>=0) and np.allclose(prob.sum(1),1)
                pred=labels[prob.argmax(1)]
                for k,tip in enumerate(test):
                    yy=truth[k];p=prob[k];ptrue=float(p[np.flatnonzero(labels==yy)[0]]) if yy in labels else 0.
                    rows.append({'lineage':lineage,'scenario':'control_old_score' if condition=='shared' and k<len(ci) else 'later_target',
                                 'character':character,'taxon':names[tip],'model':model,'true':yy,'pred':pred[k],
                                 'n_train':len(train),'confidence':float(p.max()),'p_true':ptrue,
                                 'brier':float(np.sum(p*p)-2*ptrue+1),'log_loss':-float(np.log(max(ptrue,1e-12))),
                                 'state_probabilities':json.dumps(dict(zip(labels.tolist(),p.tolist())))})
            (output/'fit_progress.json').write_text(json.dumps({'completed_characters':number+1,'last_character':character,'seconds':time.monotonic()-started},indent=2))
            (output/'fit_execution_diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
            if number%40==0:print(json.dumps({'lineage':lineage,'condition':condition,'completed_characters':number+1,'seconds':round(time.monotonic()-started,2)}),flush=True)
        result=pd.DataFrame(rows);assert not result.duplicated(['scenario','character','taxon','model']).any()
        result.to_csv(output/'predictions.csv',index=False)
        pd.DataFrame(fits).to_csv(output/'fit_diagnostics.csv',index=False)
        (output/'estimated_trees.json').write_text(json.dumps([trees[int(c)] for c in sorted(future.character.unique())]))
        (output/'mk_models.json').write_text(json.dumps(mk_models))
        archive=inputs/('archived_shared_predictions.csv' if condition=='shared' else 'archived_full_predictions.csv')
        errors,comparison=compare_archive(result,archive,condition);errors.to_csv(output/'archive_probability_comparison.csv',index=False)
        summary={'completed_utc':NOW(),'lineage':lineage,'condition':condition,'characters':len(fits),
                 'prediction_rows':len(result),'future_cells':len(future),'seconds':time.monotonic()-started,
                 'future_labels_used_for_fitting':False,'new_hyperparameter_tuning':False,'archive_comparison':comparison,
                 'logistic_convergence_warning_characters':[r['character'] for r in diagnostics if any(x.get('convergence_warning') for x in r['classifiers'] if x['model']=='logistic')],
                 'logistic_max_iter_characters':[r['character'] for r in diagnostics if any(x.get('reached_max_iter') for x in r['classifiers'] if x['model']=='logistic')],
                 'mk_rate_boundary_characters':[r['character'] for r in fits if r['rate_boundary']],
                 'mk_optimizer_unsuccessful_characters':[r['character'] for r in diagnostics if r['mk']['optimizer'] and not r['mk']['optimizer']['success']]}
        (output/'FIT_COMPLETED.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)
    return lineage

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('operation',choices=['prepare','fit']);parser.add_argument('lineages',nargs='*');args=parser.parse_args()
    if args.operation=='prepare':prepare()
    else:
        with ProcessPoolExecutor(max_workers=2) as pool:list(pool.map(fit,args.lineages or ['tetrapods','dinosaurs']))
