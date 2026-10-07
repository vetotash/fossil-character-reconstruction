"""Recompute three-history summaries from frozen scores; no fits, draws or native scoring."""
from pathlib import Path
import argparse,hashlib,json,math,re,shutil,sys
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
INPUT=HERE/'inputs'
SCHEMES=['mode','knn5','logistic','extra_trees','nj_mk','nj_mk_joint']
MODELS=SCHEMES[:-1]
LABELS={'mode':'Frequency','knn5':'kNN','logistic':'Logistic','extra_trees':'Extra Trees','nj_mk':'Mk marginal','nj_mk_joint':'Mk joint'}
HISTORIES={
 'tetrapods':dict(label='Early tetrapods',folder='new_project_search/round8/probability_propagation/future95',summary='new_project_search/round8/probability_propagation/summary.csv',
   prediction='morphology_workspace/morphology_project/round3/shared_training/tetrapods/predictions.csv',matrix='morphology_workspace/morphology_project/round5/tetrapod_consequence/exploratory_later_reference_trees/inputs/source.nex',ntax=95,ncells=349,chars=87,nrefs=5),
 'dinosaurs':dict(label='Dinosaurs',folder='new_project_search/round8/replication/probability',summary='new_project_search/round8/replication/probability/summary.csv',
   prediction='morphology_workspace/morphology_project/round3/shared_training/dinosaurs/predictions.csv',matrix='new_project_search/round8/replication/inputs/source.nex',ntax=74,ncells=1333,chars=367,nrefs=2),
 'ichthyosaurs':dict(label='Ichthyosaurs',folder='new_project_search/round10/ichthyosaur_propagation',summary='new_project_search/round10/ichthyosaur_propagation/summary.csv',
   prediction='new_project_search/round10/ichthyosaur_shared_fit/predictions.csv',matrix='new_project_search/round10/ichthyosaur_reaudit/score_inputs/panel.npz',ntax=101,ncells=192,chars=128,nrefs=5)}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def paths():
    out=set()
    for h,s in HISTORIES.items():
        out.update([s['summary'],s['prediction'],s['matrix'],s['folder']+'/cells.csv',s['folder']+'/hard_scores.npz'])
        for scheme in SCHEMES:
            out.add('new_project_search/round8/replication/joint_probability/scores.npz' if h=='dinosaurs' and scheme=='nj_mk_joint' else s['folder']+'/'+scheme+'_scores.npz')
    out.update(['new_project_search/round8/replication/joint_probability/summary.csv','new_project_search/round8/replication/joint_probability/cells.csv',
       'new_project_search/round9/taxon_ablation/source_symbol_keys.csv',
       'new_project_search/round8/run_probability_propagation.py','new_project_search/round8/replication/run_probability.py',
       'new_project_search/round10/ichthyosaur_propagation/run_probability.py'])
    return sorted(out)

def acquire(source):
    assert not INPUT.exists(),'Preserve inputs; verify or rebuild using existing copied inputs instead.'
    records=[]
    for rel in paths():
        p=source/rel;assert p.is_file(),str(p)
        q=INPUT/rel;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
        records.append(dict(source_relative_path=rel,package_relative_path=str(q.relative_to(HERE)),bytes=q.stat().st_size,sha256=sha(q)))
    (HERE/'INPUT_MANIFEST.json').write_text(json.dumps(dict(source_archive='Round10 immutable archive restored for round12',
        source_archive_sha256='3056263e2ab77e471229000e05fba0fd591b5206d2e5906beb1e365236520a5f',
        scope='Frozen score-array arithmetic and endpoint-matched prediction summaries only; no refits, draws or native scoring',files=records),indent=2)+'\n')

def verify_inputs():
    j=json.loads((HERE/'INPUT_MANIFEST.json').read_text())
    for r in j['files']:
        p=HERE/r['package_relative_path'];assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],str(p)
    return len(j['files'])

def read_matrix(p):
    if p.suffix=='.npz':
        z=np.load(p,allow_pickle=False);return dict(zip(z['names'].tolist(),z['A'].tolist()))
    text=p.read_text();ntax=int(re.search(r'NTAX\s*=\s*(\d+)',text,re.I)[1]);nc=int(re.search(r'NCHAR\s*=\s*(\d+)',text,re.I)[1])
    block=re.search(r'^\s*MATRIX\s*\n(.*?);',text,re.M|re.S|re.I)[1];raw={}
    for line in block.strip().splitlines():
        m=re.match(r"\s*('(?:[^']|'')*'|\S+)\s+(.+)",line);name=m[1].strip("'")
        row=re.findall(r'\([^)]*\)|\{[^}]*\}|.',re.sub(r'\s+','',m[2]));assert len(row)==nc and name not in raw
        raw[name]=row
    assert len(raw)==ntax;return raw

def old_summary(h,s):
    d=pd.read_csv(INPUT/s['summary'])
    if h=='tetrapods':
        assert len(d)==54;d=d[d.panel=='future95'].copy();assert len(d)==30
        d=d.rename(columns={'tree':'reference','reference_added_excess_steps':'reference_excess_steps','point_added_excess_steps':'point_excess_steps',
          'sim_mean_added_excess_steps':'sim_mean_excess_steps','sim_sd_added_excess_steps':'sim_sd_excess_steps','sim_mean_added_steps':'mean_added_steps'})
    elif h=='dinosaurs':
        assert len(d)==10;extra=pd.read_csv(INPUT/'new_project_search/round8/replication/joint_probability/summary.csv');assert len(extra)==2
        d=pd.concat([d,extra],ignore_index=True)
    assert len(d)==s['nrefs']*6 and not d.duplicated(['model','reference']).any()
    return d

def score_synthesis():
    records=[];reps=[];checks=[]
    for h,s in HISTORIES.items():
        folder=INPUT/s['folder'];z=np.load(folder/'hard_scores.npz',allow_pickle=False)
        hs,hm=z['steps'],z['minima'];trees=z['trees'].tolist();chars=z['characters'].tolist();hard=z['treatments'].tolist()
        assert len(trees)==s['nrefs'] and len(chars)==s['chars'] and hs.shape==(len(trees),7,len(chars))
        assert hard[0]=='source' and hard[1] in ['reference','later_codes'] and hard[2:]==MODELS
        obs=(hs[:,1]-hs[:,0]).sum(1)-(hm[1]-hm[0]).sum();archived=old_summary(h,s)
        for scheme in SCHEMES:
            q=INPUT/'new_project_search/round8/replication/joint_probability/scores.npz' if h=='dinosaurs' and scheme=='nj_mk_joint' else folder/(scheme+'_scores.npz')
            a=np.load(q,allow_pickle=False);assert a['characters'].tolist()==chars and a['trees'].tolist()==trees
            steps,mins=a['steps'],a['minima'];assert steps.shape==(len(trees),1000,len(chars)) and mins.shape==(1000,len(chars))
            add=(steps-hs[:,0,None,:]).sum(2);low=(mins-hm[0]).sum(1);v=add-low[None,:]
            i=hard.index('nj_mk' if scheme=='nj_mk_joint' else scheme)
            point=(hs[:,i]-hs[:,0]).sum(1)-(hm[i]-hm[0]).sum()
            for ti,ref in enumerate(trees):
                x=v[ti];quant=np.quantile(x,[.025,.975]);baseline=archived[(archived.model==scheme)&(archived.reference==ref)];assert len(baseline)==1;b=baseline.iloc[0]
                result=dict(history=h,history_label=s['label'],model=scheme,model_label=LABELS[scheme],reference=ref,n_target_cells=s['ncells'],n_target_characters=s['chars'],n_scoring_taxa=s['ntax'],
                  replicates=1000,reference_excess_steps=int(obs[ti]),point_excess_steps=int(point[ti]),sim_mean_excess_steps=float(x.mean()),sim_sd_excess_steps=float(x.std(ddof=1)),sim_q025=float(quant[0]),sim_q975=float(quant[1]),
                  mean_added_steps=float(add[ti].mean()),mean_added_minimum_steps=float(low.mean()),reference_in_central95=bool(quant[0]<=obs[ti]<=quant[1]),
                  fraction_at_least_reference=float((x>=obs[ti]).mean()),fraction_at_most_reference=float((x<=obs[ti]).mean()))
                for k in set(result)&set(archived.columns):
                    if k in ['reference','model']:assert result[k]==b[k]
                    elif isinstance(result[k],bool):assert result[k]==bool(b[k])
                    elif isinstance(result[k],(float,int)):assert abs(result[k]-float(b[k]))<1e-10,(h,ref,scheme,k)
                result.update(mean_minus_later=result['sim_mean_excess_steps']-result['reference_excess_steps'],
                  point_minus_later=result['point_excess_steps']-result['reference_excess_steps'],q025_minus_later=result['sim_q025']-result['reference_excess_steps'],
                  q975_minus_later=result['sim_q975']-result['reference_excess_steps'],mean_minus_point=result['sim_mean_excess_steps']-result['point_excess_steps'],
                  reference_position='below' if obs[ti]<quant[0] else 'above' if obs[ti]>quant[1] else 'inside')
                records.append(result)
                reps.extend(dict(history=h,model=scheme,reference=ref,replicate=j,added_steps=int(add[ti,j]),added_minimum_steps=int(low[j]),D=int(x[j])) for j in range(1000))
        checks.append(dict(history=h,complete_rows=len(archived),references=trees,all_archived_summary_fields_reproduced=True))
    assert len(records)==72 and len(reps)==72000
    pd.DataFrame(records).to_csv(HERE/'main72_summary.csv',index=False)
    pd.DataFrame(reps).to_csv(HERE/'main72000_replicates.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    return pd.DataFrame(records),checks

def prediction_synthesis():
    rows=[];registry=[];scores=[];checks=[]
    sourceflags=pd.read_csv(INPUT/'new_project_search/round9/taxon_ablation/source_symbol_keys.csv',keep_default_na=False,dtype={'later_published_state':str})
    flagkeys=set(zip(sourceflags.taxon,sourceflags.character));assert len(flagkeys)==25
    for h,s in HISTORIES.items():
        raw=read_matrix(INPUT/s['matrix']);assert len(raw)==s['ntax']
        cells=pd.read_csv(INPUT/s['folder']/'cells.csv',keep_default_na=False,dtype={'true':str,'pred':str})
        excluded=cells[~cells.taxon.isin(raw)].copy();admitted=cells[cells.taxon.isin(raw)].copy()
        assert len(admitted)==s['ncells'] and not admitted.duplicated(['character','taxon']).any()
        if h=='tetrapods':assert excluded[['taxon','character']].to_dict('records')==[{'taxon':'Paleothyris','character':265}]
        else:assert len(excluded)==0
        if h=='dinosaurs':
            jc=pd.read_csv(INPUT/'new_project_search/round8/replication/joint_probability/cells.csv');assert jc[['character','taxon']].equals(admitted[['character','taxon']].reset_index(drop=True))
            keys=set(zip(admitted.taxon,admitted.character));assert flagkeys.issubset(keys)
            cross=sourceflags.merge(admitted[['taxon','character','true']],on=['taxon','character'],validate='one_to_one')
            assert len(cross)==25 and cross.later_published_state.eq(cross.true).all()
        prediction=pd.read_csv(INPUT/s['prediction'],keep_default_na=False,dtype={'true':str,'pred':str})
        prediction=prediction[prediction.scenario=='later_target'].copy()
        for r in admitted.itertuples():
            assert raw[r.taxon][int(r.character)-1]=='?',(h,r.taxon,r.character)
            source='original_dash_normalized_to_question' if h=='dinosaurs' and (r.taxon,r.character) in flagkeys else 'literal_question_target'
            registry.append(dict(history=h,taxon=r.taxon,character=int(r.character),later_published_state=r.true,source_scope=source))
        for model in MODELS:
            p=prediction[prediction.model==model];assert not p.duplicated(['taxon','character']).any()
            joined=admitted[['taxon','character','true']].merge(p,on=['taxon','character'],how='left',validate='one_to_one',indicator=True,suffixes=('_target',''))
            assert len(joined)==s['ncells'] and joined['_merge'].eq('both').all() and joined.true.eq(joined.true_target).all()
            bs=[];ls=[];ok=[];zerop=0
            for r in joined.itertuples():
                probabilities=json.loads(r.state_probabilities);vals=np.array(list(probabilities.values()));assert (vals>=0).all() and np.isfinite(vals).all() and abs(vals.sum()-1)<1e-12
                truth=str(r.true);pred=str(r.pred);pt=float(probabilities.get(truth,0));b=sum((float(probabilities.get(k,0))-(1 if k==truth else 0))**2 for k in set(probabilities)|{truth})
                loss=-math.log(max(pt,1e-12));correct=int(pred==truth)
                assert pred in probabilities and abs(float(probabilities[pred])-float(vals.max()))<1e-10
                assert abs(b-float(r.brier))<1e-10 and abs(loss-float(r.log_loss))<1e-10 and abs(pt-float(r.p_true))<1e-10
                bs.append(b);ls.append(loss);ok.append(correct);zerop+=int(pt==0)
                scores.append(dict(history=h,model=model,taxon=r.taxon,character=int(r.character),true=truth,pred=pred,agreement=correct,brier=b,log_loss=loss,p_true=pt))
            rows.append(dict(history=h,history_label=s['label'],model=model,model_label=LABELS[model],endpoints=len(joined),target_taxa=joined.taxon.nunique(),target_characters=joined.character.nunique(),
                agreement_count=sum(ok),agreement=float(np.mean(ok)),agreement_percent=100*float(np.mean(ok)),mean_brier=float(np.mean(bs)),mean_log_loss=float(np.mean(ls)),zero_true_probability_cells=zerop,
                brier_definition='sum over all states of squared probability error',logloss_definition='natural log, true-state probability clipped below at1e-12',scope='exact downstream endpoints'))
        checks.append(dict(history=h,archived_cells_rows=len(cells),admitted_downstream_cells=len(admitted),excluded_cells=excluded[['taxon','character']].to_dict('records'),
            matched_prediction_models=5,one_to_one_keys=True,all_later_states_equal=True,all_source_targets_question_mark=True,
            all_stored_probability_metrics_reproduced=True,dinosaur_original_dash_targets=25 if h=='dinosaurs' else 0))
    assert len(registry)==1874 and len(scores)==9370 and len(rows)==15
    pd.DataFrame(registry).to_csv(HERE/'endpoint_registry1874.csv',index=False)
    pd.DataFrame(scores).to_csv(HERE/'endpoint_prediction_scores9370.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    pd.DataFrame(rows).to_csv(HERE/'matched_prediction_metrics15.csv',index=False)
    return pd.DataFrame(rows),checks

def table2(d):
    rows=[]
    for h,s in HISTORIES.items():
        for model in SCHEMES:
            g=d[(d.history==h)&(d.model==model)];assert len(g)==s['nrefs']
            rows.append(dict(history=h,history_label=s['label'],model=model,model_label=LABELS[model],n_reference_trees=len(g),
                propagated_mean_minus_later_min=float(g.mean_minus_later.min()),propagated_mean_minus_later_max=float(g.mean_minus_later.max()),
                cellwise_MAP_minus_later_min=float(g.point_minus_later.min()),cellwise_MAP_minus_later_max=float(g.point_minus_later.max()),
                later_below_simulated_central95=int(g.reference_position.eq('below').sum()),later_inside_simulated_central95=int(g.reference_position.eq('inside').sum()),later_above_simulated_central95=int(g.reference_position.eq('above').sum())))
    t=pd.DataFrame(rows);assert len(t)==18;t.to_csv(HERE/'table2_main.csv',index=False)
    lines=['Table 2. Descriptive behavior across the fixed reference trees','',
      'History | Scheme | References | Mean D minus later D range | Cellwise MAP D minus later D range | Later D below / inside / above central95',
      '']
    for r in t.itertuples():lines.append(f'{r.history_label} | {r.model_label} | {r.n_reference_trees} | {r.propagated_mean_minus_later_min:.3f} to {r.propagated_mean_minus_later_max:.3f} | {r.cellwise_MAP_minus_later_min:.0f} to {r.cellwise_MAP_minus_later_max:.0f} | {r.later_below_simulated_central95} / {r.later_inside_simulated_central95} / {r.later_above_simulated_central95}')
    lines+=['','Ranges are the minimum and maximum of reference-specific means, not uncertainty intervals or pooled effects. Below/inside/above counts describe the later-code D position relative to conditional central 95% simulation ranges. They are not coverage estimates or independent replicated tests. The reference trees are correlated sensitivity conditions. D is excess fixed-reference length change after subtracting completion-specific native minimum change. Target counts and matrix structures differ across histories; no cross-history mean is estimated. Joint Mk uses the same cellwise MAP marker as marginal Mk; it is not a jointly optimized MAP completion.']
    (HERE/'table2_main.txt').write_text('\n'.join(lines)+'\n')
    return t

def run():
    n=verify_inputs();d,scorechecks=score_synthesis();metrics,metricchecks=prediction_synthesis();t=table2(d)
    report=dict(input_files_verified=n,main_rows=72,main_draw_tree_values=72000,table2_rows=18,matched_prediction_rows=15,unique_matched_endpoints=1874,endpoint_prediction_rows=9370,
       model_fits=0,new_simulations=0,native_score_calls=0,score_checks=scorechecks,prediction_checks=metricchecks,
       caveats=['Same 349/1333/192 downstream endpoints matched within each history, not homologous endpoints across histories',
                'Dinosaurs retain25 original dashes normalized in the historical1333-target cohort',
                'Natural-logloss clips zero true-state probabilities at1e-12',
                'No bootstrap, pooled meta-effect or algorithm winner inferred'])
    (HERE/'SYNTHESIS_CHECKS.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['score_checks','prediction_checks']},indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--copy-inputs-from',type=Path);a=ap.parse_args()
    if a.copy_inputs_from:acquire(a.copy_inputs_from.resolve())
    run()
