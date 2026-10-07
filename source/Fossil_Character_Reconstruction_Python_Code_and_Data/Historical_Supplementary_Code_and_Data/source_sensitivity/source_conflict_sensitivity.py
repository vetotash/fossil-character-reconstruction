#!/usr/bin/env python3
"""Post hoc exclusion sensitivity for the documented Laquintasaura366 conflict.

The archived matrix coding is preserved. The single conflicted cell remains
unknown in both the reference and every completion. No predictor is refitted.
Only the binary unordered affected character is rescored in Python; every
pre-exclusion score is first checked against the archived native score arrays.
"""
from pathlib import Path
import argparse,hashlib,json,re
import numpy as np
import pandas as pd

SCHEMES=['mode','knn5','logistic','extra_trees','nj_mk','nj_mk_joint']
TAXON='Laquintasaura';CHARACTER=366

def matrix(path):
    text=path.read_text();nchar=int(re.search(r'NCHAR\s*=\s*(\d+)',text,re.I)[1])
    block=re.search(r'^\s*MATRIX\s*\n(.*?);',text,re.I|re.M|re.S)[1]
    rows={}
    for line in block.strip().splitlines():
        name,seq=line.strip().split(None,1)
        tokens=re.findall(r'\([^)]*\)|\{[^}]*\}|.',re.sub(r'\s+','',seq))
        assert len(tokens)==nchar;rows[name]=tokens
    return rows

def parse_tree(newick):
    tokens=re.findall(r'[(),;]|[^(),;\s]+',newick);i=0
    def node():
        nonlocal i
        if tokens[i]=='(':
            i+=1;children=[node()]
            while tokens[i]==',':i+=1;children.append(node())
            assert tokens[i]==')';i+=1
            return children
        label=tokens[i];i+=1
        assert ':' not in label and '[' not in label
        return label
    tree=node();assert tokens[i:]==[';'];return tree

def score(patterns,names,tree):
    index={n:i for i,n in enumerate(names)}
    def rec(node):
        if isinstance(node,str):
            x=patterns[:,index[node]]
            return np.stack((x==1,x==0),axis=1).astype(np.int32)*(len(names)+1)
        total=np.zeros((len(patterns),2),dtype=np.int32)
        for child in node:
            c=rec(child)
            total[:,0]+=np.minimum(c[:,0],c[:,1]+1)
            total[:,1]+=np.minimum(c[:,1],c[:,0]+1)
        return total
    return rec(tree).min(1)

def minimum(patterns):return (np.any(patterns==0,axis=1)&np.any(patterns==1,axis=1)).astype(np.int32)
def column(raw,names):
    tokens=[raw[n][CHARACTER-1] for n in names];assert set(tokens)<={'0','1','?'}
    return np.array([-1 if t=='?' else int(t) for t in tokens],dtype=np.int8)

def metrics(d,y):
    d=np.asarray(d,dtype=float);lo,hi=np.quantile(d,[.025,.975]);ordered=np.sort(d);n=len(d)
    crps=np.mean(abs(d-y))-np.dot(2*np.arange(1,n+1)-n-1,ordered)/(n*n)
    return {'reference_D':int(y),'mean_D':float(d.mean()),'mean_minus_reference':float(d.mean()-y),'q025':float(lo),'q975':float(hi),'width':float(hi-lo),'reference_in_central95':bool(lo<=y<=hi),'CRPS_against_later_codes':float(crps)}

def run(source,out):
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    base=source/'new_project_search/round8/replication';prob=base/'probability'
    raw=matrix(base/'inputs/source.nex');names=list(raw);taxon_index=names.index(TAXON)
    assert raw[TAXON][CHARACTER-1]=='?'
    source_column=column(raw,names);old_min=int(minimum(source_column[None,:])[0])
    trees={name:parse_tree(newick) for name,newick in re.findall(r'TREE\s+(\w+)\s*=\s*\[&U\]\s*(.*?;)',(base/'reference_trees.nex').read_text(),re.I)}
    with np.load(prob/'hard_scores.npz',allow_pickle=False) as a:
        hs=a['steps'];hm=a['minima'];tree_names=a['trees'].tolist();chars=a['characters'].tolist();treatments=a['treatments'].tolist()
    ci=chars.index(CHARACTER);assert list(trees)==tree_names
    reference=column(matrix(base/'inputs/later_codes.nex'),names)
    assert reference[taxon_index]==0
    reference_excluded=reference.copy();reference_excluded[taxon_index]=-1
    old_native=hs[:,0,:].sum(1)-hm[0].sum()
    original_y=hs[:,1,:].sum(1)-hm[1].sum()-old_native
    reference_scores=np.array([score(reference[None,:],names,trees[t])[0] for t in tree_names])
    assert np.array_equal(reference_scores,hs[:,1,ci])
    excluded_scores=np.array([score(reference_excluded[None,:],names,trees[t])[0] for t in tree_names])
    new_y=original_y+(excluded_scores-reference_scores)-(minimum(reference_excluded[None,:])[0]-hm[1,ci])
    hard_patterns=np.stack([column(matrix(base/'inputs'/f'{t}.nex'),names) for t in ['source','later_codes','mode','knn5','logistic','extra_trees','nj_mk']])
    for i,t in enumerate(tree_names):assert np.array_equal(score(hard_patterns,names,trees[t]),hs[i,:,ci])
    assert np.array_equal(minimum(hard_patterns),hm[:,ci])
    rows=[];draw_rows=[];checked=14
    for scheme in SCHEMES:
        folder=base/'joint_probability' if scheme=='nj_mk_joint' else prob
        cells=pd.read_csv(folder/'cells.csv',keep_default_na=False)
        selected=np.flatnonzero(cells.character.to_numpy()==CHARACTER)
        conflict=np.flatnonzero((cells.character==CHARACTER)&(cells.taxon==TAXON));assert len(conflict)==1
        draws=np.load(folder/('aligned_draws.npy' if scheme=='nj_mk_joint' else f'{scheme}_draws.npy'),allow_pickle=False)
        patterns=np.tile(source_column,(len(draws),1))
        for j in selected:patterns[:,names.index(cells.iloc[j].taxon)]=draws[:,j].astype(np.int8)
        excluded=patterns.copy();excluded[:,taxon_index]=-1
        with np.load(folder/('scores.npz' if scheme=='nj_mk_joint' else f'{scheme}_scores.npz'),allow_pickle=False) as a:
            assert a['characters'].tolist()==chars and a['trees'].tolist()==tree_names
            allsteps=a['steps'];mins=a['minima'];d=allsteps.sum(2)-mins.sum(1)[None,:]-old_native[:,None]
        assert np.array_equal(minimum(patterns),mins[:,ci])
        delta_min=minimum(excluded)-mins[:,ci]
        for i,t in enumerate(tree_names):
            original=score(patterns,names,trees[t]);assert np.array_equal(original,allsteps[i,:,ci]);checked+=len(draws)
            new_d=d[i]+score(excluded,names,trees[t])-original-delta_min
            for label,values,y in [('original1333',d[i],original_y[i]),('exclude_conflict1332',new_d,new_y[i])]:
                rows.append({'tree':t,'scheme':scheme,'cohort':label,**metrics(values,y)})
                draw_rows.extend({'tree':t,'scheme':scheme,'cohort':label,'draw':k,'D':int(v),'reference_D':int(y)} for k,v in enumerate(values))
    pd.DataFrame(rows).to_csv(out/'D_exclusion_summary.csv',index=False)
    pd.DataFrame(draw_rows).to_csv(out/'D_exclusion_draws.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    predictions=pd.read_csv(source/'morphology_workspace/morphology_project/round3/shared_training/dinosaurs/predictions.csv',keep_default_na=False)
    predictions=predictions[predictions.scenario=='later_target'];state=[]
    for model,g in predictions.groupby('model'):
        assert len(g)==1333
        for label,h in [('original1333',g),('exclude_conflict1332',g[~((g.taxon==TAXON)&(g.character==CHARACTER))])]:
            state.append(dict(model=model,cohort=label,n=len(h),agreement=float((h.pred==h.true).mean()),brier=float(h.brier.mean()),log_loss=float(h.log_loss.mean())))
    pd.DataFrame(state).to_csv(out/'state_exclusion_summary.csv',index=False)
    facts={'status':'PASS','rule':'Exclude Laquintasaura character366 because the published modification list conflicts with the matrix coding and original binary definition','selection':'post hoc source-triggered; not chosen from model errors','frozen_label_preserved':0,'unsupported_alternative_not_imputed':2,'source_targets':1333,'sensitivity_targets':1332,'native_character_tree_scores_reproduced':checked,'native_minima_reproduced':True,'predictor_refits':0,'scoring_scope':'Only the binary unordered affected column is rescored; all other archived native character contributions remain unchanged','code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/'analysis_manifest.json').write_text(json.dumps(facts,indent=2)+'\n');print(json.dumps(facts,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.source.resolve(),a.output.resolve())
