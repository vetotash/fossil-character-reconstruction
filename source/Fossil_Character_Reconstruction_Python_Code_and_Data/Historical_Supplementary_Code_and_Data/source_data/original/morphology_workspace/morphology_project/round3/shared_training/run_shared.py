from pathlib import Path
import os,sys,json,time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

P=Path(__file__).resolve().parent;R2=P.parents[1]/'round2'
sys.path.insert(0,str(R2));os.environ['FOSSIL_USE_SNAPSHOT']='1'
import run_comparison as rc
from mk_baseline import known,predict_mk

LINEAGES=['dinosaurs','tetrapods','crocodilians','gnathostomes','whales']

def quality_excluded(lineage,frame):
    if lineage=='whales':return frame.taxon.eq('Eschrichtius_robustus')
    if lineage=='crocodilians':return frame.taxon.eq('Portugalosuchus_azenhae')&frame.character.isin([129,134])
    return pd.Series(False,index=frame.index)

def metrics_for(frame):
    out=[]
    for (lineage,scenario,model),g in frame.groupby(['lineage','scenario','model']):
        ok=g.true.eq(g.pred)
        out.append(dict(lineage=lineage,scenario=scenario,model=model,n=len(g),taxa=g.taxon.nunique(),characters=g.character.nunique(),accuracy=ok.mean(),brier=g.brier.mean(),macro_character_accuracy=g.assign(correct=ok).groupby('character').correct.mean().mean(),macro_character_brier=g.groupby('character').brier.mean().mean(),confidence=g.confidence.mean()))
    return pd.DataFrame(out)

def run(lineage):
    started=time.time();dest=P/lineage;dest.mkdir(exist_ok=True,parents=True)
    names,a,frozen,charmap=rc.dataset(lineage);lookup={n:i for i,n in enumerate(names)}
    prior=pd.read_csv(R2/lineage/'predictions.csv',dtype={'true':str,'pred':str})
    tree_data=json.loads((R2/lineage/'estimated_trees.json').read_text());trees={int(r['character']):r for r in tree_data}
    target=frozen[frozen.model.eq('mode')&frozen.scenario.eq('later_publication')].copy()
    target=target.loc[~quality_excluded(lineage,target)]
    controls=frozen[frozen.model.eq('mode')&frozen.scenario.eq('matched_concealment')].copy()
    output=[];fits=[];excluded=[];reproduced=0;maxerr=0.
    for num,(character,future) in enumerate(target.groupby('character')):
        control=controls[controls.character.eq(character)]
        if len(control)==0:
            excluded.extend([dict(lineage=lineage,character=int(character),taxon=n,reason='no_frozen_control_for_character') for n in future.taxon]);continue
        j=charmap[int(character)];ci=np.array([lookup[n] for n in control.taxon]);fi=np.array([lookup[n] for n in future.taxon])
        assert np.array_equal(a[ci,j],control.true.to_numpy());assert np.all(a[fi,j]=='?')
        assert len(set(ci))==len(ci) and not set(ci)&set(fi)
        mat=a.copy();mat[ci,j]='?';train=np.flatnonzero(known(mat[:,j]))
        assert not set(train)&set(np.r_[ci,fi])
        test=np.r_[ci,fi];truth=np.r_[control.true.to_numpy(),future.true.to_numpy()]
        edges=trees[int(character)]['edges'];assert trees[int(character)]['taxa']==names
        methods=rc.dp.predict(mat,j,train,test)
        labels,probs,fit=predict_mk(edges,len(names),mat[:,j],train,test);methods['nj_mk']=(labels,probs)
        fits.append(dict(lineage=lineage,character=int(character),n_train=len(train),n_original_train=int(known(a[:,j]).sum()),n_controls=len(ci),n_future=len(fi),**fit))
        for model,(labels,prob) in methods.items():
            pred=labels[prob.argmax(1)]
            oldcontrol=prior[prior.scenario.eq('matched_concealment')&prior.character.eq(character)&prior.model.eq(model)].set_index('taxon').loc[control.taxon]
            assert np.array_equal(pred[:len(ci)],oldcontrol.pred.to_numpy()),(lineage,character,model,'control labels changed')
            for k,raw in enumerate(oldcontrol.state_probabilities):
                oldp=json.loads(raw);assert set(oldp)==set(labels)
                err=max(abs(oldp[s]-prob[k,l]) for l,s in enumerate(labels));maxerr=max(maxerr,err);assert err<1e-10
            reproduced+=len(ci)
            for k,row in enumerate(test):
                yy=truth[k];p=prob[k];ptrue=float(p[np.flatnonzero(labels==yy)[0]]) if yy in labels else 0.
                output.append(dict(lineage=lineage,scenario='control_old_score' if k<len(ci) else 'later_target',character=int(character),taxon=names[row],model=model,true=yy,pred=pred[k],n_train=len(train),confidence=float(p.max()),p_true=ptrue,brier=float(np.sum(p*p)-2*ptrue+1),log_loss=-float(np.log(max(ptrue,1e-12))),state_probabilities=json.dumps(dict(zip(labels.tolist(),p.tolist())))))
        if num%75==0:print(lineage,num+1,round(time.time()-started,1),flush=True)
    output=pd.DataFrame(output);output.to_csv(dest/'predictions.csv',index=False)
    pd.DataFrame(fits).to_csv(dest/'fit_diagnostics.csv',index=False)
    pd.DataFrame(excluded,columns=['lineage','character','taxon','reason']).to_csv(dest/'excluded_targets.csv',index=False)
    metrics_for(output).to_csv(dest/'metrics.csv',index=False)
    # Same natural target, two old training sets. New labels are reference only.
    new=output[output.scenario.eq('later_target')].copy()
    old=prior[prior.scenario.eq('later_publication')].copy()
    paired=new.merge(old,on=['lineage','character','taxon','model'],suffixes=('_shared','_full'),validate='one_to_one')
    assert len(paired)==len(new) and paired.true_shared.eq(paired.true_full).all()
    paired['brier_change']=paired.brier_shared-paired.brier_full
    paired['accuracy_change']=paired.true_shared.eq(paired.pred_shared).astype(int)-paired.true_full.eq(paired.pred_full).astype(int)
    paired[['lineage','character','taxon','model','n_train_shared','n_train_full','accuracy_change','brier_change']].to_csv(dest/'training_reduction_effect.csv',index=False)
    result=dict(lineage=lineage,controls_exactly_reproduced=reproduced,max_control_probability_error=maxerr,eligible_future_cells=int(new.model.eq('mode').sum()),excluded_future_cells=len(excluded),all_same_training=True,seconds=time.time()-started)
    (dest/'checks.json').write_text(json.dumps(result,indent=2));print(result,flush=True)

if __name__=='__main__':
    with ProcessPoolExecutor(max_workers=3) as pool:list(pool.map(run,sys.argv[1:] or LINEAGES))
    m=pd.concat([pd.read_csv(P/n/'metrics.csv') for n in LINEAGES if (P/n/'metrics.csv').exists()]);m.to_csv(P/'metrics.csv',index=False)
    print(m.to_string(index=False))
