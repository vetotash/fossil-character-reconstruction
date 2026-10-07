"""Exploratory longitudinal imputation feasibility pilot. No tuning on later matrix."""
from pathlib import Path
import re,json
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score,balanced_accuracy_score

ROOT=Path(__file__).resolve().parent
def parse(path):
    text=path.read_text()
    if '\\n' in text and '\n' not in text:
        text=text.replace('\\n','\n').replace('\\t','\t').replace('\\"','"')
    ntax=int(re.search(r'NTAX\s*=\s*(\d+)',text,re.I)[1])
    nchar=int(re.search(r'NCHAR\s*=\s*(\d+)',text,re.I)[1])
    text=re.sub(r'\[.*?\]','',text,flags=re.S)
    block=re.search(r'\bMATRIX\b(.*?);',text,re.S|re.I)[1]
    taxa=[]; rows=[]
    for line in block.strip().splitlines():
        line=line.strip()
        if not line: continue
        m=re.match(r"('(?:[^']|'')*'|\S+)\s+(.+)",line)
        if not m: raise ValueError(line[:100])
        tax=m[1].strip("'"); raw=re.sub(r'\s+','',m[2])
        cells=re.findall(r'\([^)]*\)|\{[^}]*\}|.',raw)
        assert len(cells)==nchar,(path.name,tax,len(cells),nchar)
        taxa.append(tax);rows.append(cells)
    assert len(taxa)==ntax,(path.name,len(taxa),ntax)
    assert len(taxa)==len(set(taxa))
    return taxa,np.array(rows)

def known(x):
    return np.vectorize(lambda s:bool(re.fullmatch('[0-9A-Z]',s)))(x)

audit=[]
for p in (ROOT/'matrices').glob('*.nex'):
    try:
        names,a=parse(p)
        audit.append(dict(matrix=p.name,taxa=a.shape[0],characters=a.shape[1],single_known=int(known(a).sum()),missing=int((a=='?').sum()),inapplicable=int((a=='-').sum()),other=int((~known(a)&(a!='?')&(a!='-')).sum()),character_labels=False))
    except Exception as e: audit.append(dict(matrix=p.name,error=str(e)))
pd.DataFrame(audit).to_csv(ROOT/'matrix_audit.csv',index=False)

old_names,old=parse(ROOT/'matrices'/'Baron_etal_2017a.nex')
new_names,new=parse(ROOT/'matrices'/'Langer_etal_2017a.nex')
# The later matrix abbreviates most single-species genera. Preserve species for
# multispesific genera; repair the documented orthographic difference explicitly.
def canonical(names):
    genus=[n.split('_')[0] for n in names]
    out=[]
    for n,g in zip(names,genus):
        if g=='Lewisuchus': out.append('Lewisuchus_Pseudolagosuchus')
        elif g in ['Dromomeron','Massospondylus']: out.append(n.replace('kaalea','kaalae'))
        else: out.append(g)
    assert len(out)==len(set(out))
    return out
old_raw_names=old_names.copy();new_raw_names=new_names.copy()
old_names=canonical(old_names);new_names=canonical(new_names)
pd.DataFrame([dict(matrix='Baron_2017',raw=r,canonical=c) for r,c in zip(old_raw_names,old_names)]+[dict(matrix='Langer_2017',raw=r,canonical=c) for r,c in zip(new_raw_names,new_names)]).to_csv(ROOT/'taxon_mapping.csv',index=False)
common=sorted(set(old_names)&set(new_names))
A=old[[old_names.index(n) for n in common]]
B=new[[new_names.index(n) for n in common]]
both=known(A)&known(B)
filled=(A=='?')&known(B)
summary=dict(old_ntaxa=len(old_names),new_ntaxa=len(new_names),shared_taxa=len(common),nchar=A.shape[1],both_known=int(both.sum()),rescored=int((both&(A!=B)).sum()),old_unknown_later_known=int(filled.sum()),new_unknown_old_known=int((known(A)&(B=='?')).sum()))
summary['all_token_differences']=int((A!=B).sum())
assert len(common)==74
assert int((A!=B).sum())==3366, 'Must match Cerny and Simonoff 2023 source-data audit'
old_eval_ix=np.array([old_names.index(n) for n in common])
rng=np.random.default_rng(20261002)
out=[]

def models_predict(matrix,j,tr,te):
    y=matrix[tr,j]
    labels,counts=np.unique(y,return_counts=True)
    mode=labels[np.argmax(counts)]
    result={'mode':np.repeat(mode,len(te))}
    if len(labels)==1:
        return dict(mode=result['mode'],knn5=result['mode'],extra_trees=result['mode'],logistic=result['mode'])
    xx=np.delete(matrix,j,axis=1)
    # Treat all unresolved states as an explicit missing category for one hot models.
    xx=np.where(known(xx),xx,'?')
    enc=OneHotEncoder(handle_unknown='ignore',sparse_output=False)
    xtr=enc.fit_transform(xx[tr]);xte=enc.transform(xx[te])
    model=ExtraTreesClassifier(n_estimators=100,min_samples_leaf=2,max_features='sqrt',random_state=20261002,n_jobs=1)
    model.fit(xtr,y);result['extra_trees']=model.predict(xte)
    model=LogisticRegression(C=1,max_iter=300,random_state=20261002)
    model.fit(xtr,y);result['logistic']=model.predict(xte)
    preds=[]
    for row in xx[te]:
        mask=(xx[tr]!='?')&(row[None,:]!='?')
        d=np.divide(((xx[tr]!=row)&mask).sum(1),mask.sum(1),out=np.full(len(tr),np.inf),where=mask.sum(1)>0)
        inds=np.argsort(d,kind='stable')[:5]
        vals,cnts=np.unique(y[inds],return_counts=True);preds.append(vals[np.argmax(cnts)])
    result['knn5']=np.array(preds)
    return result

for j in np.where(filled.any(0))[0]:
    train=np.where(known(old[:,j]))[0]
    target_common=np.where(filled[:,j])[0]
    test=old_eval_ix[target_common]
    if len(train)<15:continue
    for model,pred in models_predict(old,j,train,test).items():
        for ix,pi in zip(target_common,pred):
            out.append(dict(scenario='later_publication',character=j+1,taxon=common[ix],true=B[ix,j],pred=pi,model=model))
    # Matched per-character random concealment, target old observations.
    n=min(len(test),max(1,len(train)//4))
    test_random=rng.choice(train,size=n,replace=False)
    train_random=np.setdiff1d(train,test_random)
    masked=old.copy();masked[test_random,j]='?'
    for model,pred in models_predict(masked,j,train_random,test_random).items():
        for ix,pi in zip(test_random,pred):
            out.append(dict(scenario='random_concealment',character=j+1,taxon=old_names[ix],true=old[ix,j],pred=pi,model=model))
    if j%50==0: print('completed char',j,flush=True)
df=pd.DataFrame(out);df.to_csv(ROOT/'pilot_predictions.csv',index=False)
metrics=[]
for (scenario,model),g in df.groupby(['scenario','model']):
    perchar=g.groupby('character').apply(lambda v:(v.true==v.pred).mean(),include_groups=False)
    metrics.append(dict(scenario=scenario,model=model,n=len(g),characters=g.character.nunique(),taxa=g.taxon.nunique(),accuracy=accuracy_score(g.true,g.pred),mean_character_accuracy=float(perchar.mean())))
pd.DataFrame(metrics).to_csv(ROOT/'pilot_metrics.csv',index=False)
(ROOT/'pilot_summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2));print(pd.DataFrame(metrics).to_string(index=False))
