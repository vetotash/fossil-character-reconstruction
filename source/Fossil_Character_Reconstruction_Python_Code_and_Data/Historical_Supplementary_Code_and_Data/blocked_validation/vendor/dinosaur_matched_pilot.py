"""Matched-missingness exploratory reanalysis of the documented dinosaur pair.
Inherited character numbers are externally documented; complete semantic audit pending.
"""
from pathlib import Path
import ast,re,json
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
ROOT=Path(__file__).resolve().parent
source=ast.parse((ROOT/'pilot.py').read_text())
exec(compile(ast.Module(body=[n for n in source.body if isinstance(n,ast.FunctionDef) and n.name in ['parse','known','canonical']],type_ignores=[]),'helpers','exec'))

def predict(matrix,j,tr,te):
    y=matrix[tr,j];labels,counts=np.unique(y,return_counts=True)
    p=np.tile(counts/counts.sum(),(len(te),1));out={'mode':(labels,p)}
    if len(labels)==1:return {k:(labels,p) for k in ['mode','knn5','logistic','extra_trees']}
    xx=np.delete(matrix,j,axis=1);xx=np.where(known(xx),xx,'?')
    enc=OneHotEncoder(handle_unknown='ignore',sparse_output=False)
    xtr=enc.fit_transform(xx[tr]);xte=enc.transform(xx[te])
    for name,model in [('extra_trees',ExtraTreesClassifier(n_estimators=100,min_samples_leaf=2,max_features='sqrt',random_state=20261002,n_jobs=1)),('logistic',LogisticRegression(C=1,max_iter=300,random_state=20261002))]:
        model.fit(xtr,y);out[name]=(model.classes_,model.predict_proba(xte))
    probs=[]
    for row in xx[te]:
        mask=(xx[tr]!='?')&(row[None,:]!='?')
        d=np.divide(((xx[tr]!=row)&mask).sum(1),mask.sum(1),out=np.full(len(tr),np.inf),where=mask.sum(1)>0)
        inds=np.argsort(d,kind='stable')[:5]
        probs.append([(y[inds]==label).mean() for label in labels])
    out['knn5']=(labels,np.array(probs));return out

def append_predictions(out,mat,j,tr,te,truth,names,scenario,missing):
    for model,(labels,prob) in predict(mat,j,tr,te).items():
        pred=labels[np.argmax(prob,axis=1)]
        for k,i in enumerate(te):
            ptrue=prob[k,np.where(labels==truth[k])[0][0]] if truth[k] in labels else 0.
            brier=float(np.sum(prob[k]**2)-2*ptrue+1)
            out.append(dict(scenario=scenario,character=j+1,taxon=names[i],true=truth[k],pred=pred[k],model=model,confidence=float(prob[k].max()),p_true=float(ptrue),brier=brier,log_loss=-float(np.log(max(ptrue,1e-12))),source_missing=float(missing[i])))

if __name__=='__main__':
    names,old=parse(ROOT/'matrices/Baron_etal_2017a.nex');names=canonical(names)
    newnames,new=parse(ROOT/'matrices/Langer_etal_2017a.nex');newnames=canonical(newnames)
    common=sorted(set(names)&set(newnames));oi=np.array([names.index(n) for n in common]);ni=np.array([newnames.index(n) for n in common])
    A=old[oi];B=new[ni];filled=(A=='?')&known(B);missing=(~known(old)).mean(1)
    out=[];matches=[];rng=np.random.default_rng(20261002)
    for j in np.where(filled.any(0))[0]:
        tr=np.where(known(old[:,j]))[0];tc=np.where(filled[:,j])[0];te=oi[tc]
        if len(tr)<15:continue
        append_predictions(out,old,j,tr,te,B[tc,j],names,'later_publication',missing)
        # One-to-one row-completeness matching within the same character.
        # Cap at a quarter of source observations, randomly subsample natural
        # targets before matching if necessary. Uses source missingness only.
        n=min(len(te),max(1,len(tr)//4));chosen=rng.choice(te,n,replace=False)
        cost=np.abs(missing[chosen,None]-missing[tr][None,:])
        r,c=linear_sum_assignment(cost);rt=tr[c]
        for ti,ri in zip(chosen[r],rt):matches.append(dict(character=j+1,later_taxon=names[ti],mask_taxon=names[ri],later_missing=float(missing[ti]),mask_missing=float(missing[ri]),abs_difference=float(abs(missing[ti]-missing[ri]))))
        train=np.setdiff1d(tr,rt);masked=old.copy();masked[rt,j]='?'
        append_predictions(out,masked,j,train,rt,old[rt,j],names,'matched_mask',missing)
        if j%75==0:print('completed',j,flush=True)
    d=pd.DataFrame(out);d.to_csv(ROOT/'dinosaur_matched_predictions.csv',index=False)
    m=pd.DataFrame(matches);m.to_csv(ROOT/'dinosaur_mask_matching.csv',index=False)
    metrics=[]
    for (scenario,model),g in d.groupby(['scenario','model']):
        ok=(g.true==g.pred);bins=pd.cut(g.confidence,bins=np.linspace(0,1,6),include_lowest=True)
        tab=g.assign(correct=ok).groupby(bins,observed=True).agg(n=('correct','size'),accuracy=('correct','mean'),confidence=('confidence','mean'))
        metrics.append(dict(scenario=scenario,model=model,n=len(g),characters=g.character.nunique(),taxa=g.taxon.nunique(),accuracy=float(ok.mean()),macro_character_accuracy=float(g.assign(correct=ok).groupby('character').correct.mean().mean()),brier=float(g.brier.mean()),log_loss=float(g.log_loss.mean()),ece_5bins=float((tab.n/len(g)*abs(tab.accuracy-tab.confidence)).sum()),mean_confidence=float(g.confidence.mean())))
    metrics=pd.DataFrame(metrics);metrics.to_csv(ROOT/'dinosaur_matched_metrics.csv',index=False)
    print(metrics.to_string(index=False));print('Matching abs row missingness median,mean,max:',m.abs_difference.median(),m.abs_difference.mean(),m.abs_difference.max())
