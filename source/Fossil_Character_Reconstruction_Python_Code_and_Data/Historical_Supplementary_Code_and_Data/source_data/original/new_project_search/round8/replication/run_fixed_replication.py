"""Frozen two-reference native PAUP scores with independent Sankoff verification."""
from pathlib import Path
from collections import Counter
from decimal import Decimal
import csv,json,os,re,subprocess,datetime,argparse
import numpy as np
import pandas as pd
from inventory import HERE,PROJECT,parse,canonical,parse_tree,leaves,sha

TREATMENTS=['source','later_codes','mode','knn5','logistic','extra_trees','nj_mk']
TREES=['baron','langer']
TOOLS=PROJECT/'round5/tetrapod_consequence/tools'
def prune(t,keep):
    if isinstance(t,str):return t if t in keep else None
    out=[c for x in t if (c:=prune(x,keep)) is not None]
    if not out:return None
    return out[0] if len(out)==1 else out
def serialize(t):return t if isinstance(t,str) else '('+','.join(map(serialize,t))+')'
def splits(t,keep):
    out=set()
    def rec(node):
        if isinstance(node,str):return {node}
        d=set()
        for child in node:
            z=rec(child);d|=z;a=frozenset(z&keep);b=frozenset(keep-a)
            if min(len(a),len(b))>=2:out.add(frozenset([a,b]))
        return d
    rec(t);return out
def source_ordered(text):
    raw=re.search(r'TYPESET.*?=\s*ord:\s*([^,]+),',text,re.S|re.I)[1]
    indices=set()
    for a in raw.split():
        if '-' in a:
            l,r=map(int,a.split('-'));indices.update(range(l,r+1))
        else:indices.add(int(a))
    assert len(indices)==39
    return indices
def prepare():
    assert (HERE/'PROTOCOL.txt').exists()
    cfg=json.loads((HERE/'scope_decision.json').read_text())
    report=HERE/cfg['review_file'];assert report.exists()
    old,A,text=parse(HERE/'raw/baron_original.nex');on=list(map(canonical,old))
    later,B,_=parse(HERE/'raw/langer_original.nex');nn=list(map(canonical,later))
    trees={stem:parse_tree((HERE/f'raw/ML_unconstrained/ML_unconstrained/{stem}_unconstrained.treefile').read_text()) for stem in TREES}
    keep=set(on)&set(nn)&set(leaves(trees['baron']))&set(leaves(trees['langer']))
    keep-=set(cfg['exclude_canonical_taxa'])
    names=[x for x in on if x in keep]
    assert len(names)==cfg['expected_taxa']
    predictions=pd.read_csv(PROJECT/'round3/shared_training/dinosaurs/predictions.csv')
    pred=predictions[(predictions.scenario=='later_target')&predictions.taxon.isin(keep)].copy()
    mode=pred[pred.model=='mode'];targets={(r.taxon,r.character) for r in mode.itertuples()}
    assert len(targets)==len(mode)==cfg['expected_targets']
    assert len(pred)==len(targets)*5 and set(pred.model)==set(TREATMENTS[2:])
    ordered=source_ordered(text)
    body=re.search(r'\bMATRIX\b(.*?);',text,re.S|re.I)
    source=A[[on.index(x) for x in names]]
    (HERE/'inputs').mkdir(exist_ok=True)
    inputs={}
    for treatment in TREATMENTS:
        m=source.copy()
        if treatment!='source':
            use=mode if treatment=='later_codes' else pred[pred.model==treatment]
            for r in use.itertuples():
                i=names.index(r.taxon);j=r.character-1
                assert m[i,j]=='?'
                assert B[nn.index(r.taxon),j]==str(r.true)
                m[i,j]=str(r.true if treatment=='later_codes' else r.pred)
        diffs={(names[i],j+1) for i,j in zip(*np.where(m!=source))}
        assert diffs==(set() if treatment=='source' else targets)
        derivative=text[:body.start(1)]+'\n'+''.join(n+' '+''.join(row)+'\n' for n,row in zip(names,m))+text[body.end(1):]
        derivative=re.sub(r'(NTAX\s*=\s*)75\b',lambda v:v[1]+str(len(names)),derivative,flags=re.I)
        p=HERE/'inputs'/f'{treatment}.nex';p.write_text(derivative);pn,pa,_=parse(p);assert pn==names and np.array_equal(pa,m)
        inputs[treatment]=sha(p)
    newtrees={}
    for name,t in trees.items():
        n=prune(t,keep);assert set(leaves(n))==keep and len(leaves(n))==len(keep)
        assert splits(t,keep)==splits(n,keep)
        newtrees[name]=n
    (HERE/'reference_trees.nex').write_text('#NEXUS\nBEGIN TREES;\n'+''.join(f'TREE {name} = [&U] {serialize(t)};\n' for name,t in newtrees.items())+'END;\n')
    targettable=mode[['taxon','character','true']].copy();targettable.to_csv(HERE/'retained_targets.csv',index=False)
    gate=dict(time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scores_existed_before_gate=(HERE/'native').exists(),n_taxa=len(names),n_targets=len(targets),n_target_taxa=mode.taxon.nunique(),n_target_characters=mode.character.nunique(),
       tree_names=TREES,all_treatment_matrices_differ_only_at_targets=True,retained_taxa=names,
       ordered_characters=sorted(ordered),inputs=inputs,predictions_sha256=sha(PROJECT/'round3/shared_training/dinosaurs/predictions.csv'),
       protocol_sha256=sha(HERE/'PROTOCOL.txt'),scope_review_sha256=sha(report),scope_decision_sha256=sha(HERE/'scope_decision.json'),
       treefile_sha256=sha(HERE/'reference_trees.nex'),script_sha256=sha(Path(__file__)),all_pruned_edge_bipartitions_verified=True)
    assert not gate['scores_existed_before_gate']
    (HERE/'PRESCORE_GATE.json').write_text(json.dumps(gate,indent=2))
    return gate,newtrees

def sankoff(matrix,tree,names,ordered):
    """Independent all-states dynamic program; no inference or tree searching."""
    out=[]
    for j in range(matrix.shape[1]):
        known={int(x) for x in matrix[:,j] if x.isdigit()}
        # All0..4 original allowed states; these do not make unobserved states cheaper.
        states=np.arange(5);cost=np.abs(states[:,None]-states[None,:]) if j+1 in ordered else (states[:,None]!=states[None,:]).astype(int)
        def rec(t):
            if isinstance(t,str):
                value=matrix[names.index(t),j]
                if value in '?-':return np.zeros(5)
                assert value.isdigit(),value
                return np.where(states==int(value),0,1e9)
            return sum(np.min(rec(ch)[None,:]+cost,axis=1) for ch in t)
        out.append(int(np.min(rec(tree))))
    return out
def read_scores(p):
    rows=list(csv.reader(p.open(),delimiter='\t'));assert rows[0]==['Tree','Length','Character','Length']
    totals={};characters={};i=None
    for r in rows[1:]:
        if not any(r):continue
        if r[0]:i=int(r[0]);totals[i]=Decimal(r[1]);characters[i]={}
        else:characters[i][int(r[2])]=Decimal(r[3])
    assert set(totals)=={1,2}
    for i in totals:assert set(characters[i])==set(range(1,458)) and sum(characters[i].values())==totals[i]
    return totals,characters
def run():
    gate=json.loads((HERE/'PRESCORE_GATE.json').read_text());assert gate['script_sha256']==sha(Path(__file__))
    assert gate['treefile_sha256']==sha(HERE/'reference_trees.nex')
    treetext=(HERE/'reference_trees.nex').read_text()
    trees={m[1]:parse_tree(m[2]) for m in re.finditer(r'TREE\s+(\w+)\s*=\s*\[&U\]\s*(.*?);',treetext)}
    assert list(trees)==TREES
    totalsrows=[];characterrows=[];data={}
    for treatment in TREATMENTS:
        matrix=HERE/'inputs'/f'{treatment}.nex';assert sha(matrix)==gate['inputs'][treatment]
        names,A,text=parse(matrix);assert names==gate['retained_taxa']
        out=HERE/'native'/treatment;out.mkdir(parents=True,exist_ok=True)
        driver=out/'score.nex'
        driver.write_text(text+f'''\nBEGIN PAUP;
set autoclose=yes increase=no maxtrees=100;
outgroup Euparkeria / only;
pset mstaxa=variable opt=deltran;
gettrees file='{(HERE/'reference_trees.nex').resolve()}';
pscores all / single=all tl=yes scorefile=native_scores.tsv replace=yes;
quit;
END;\n''')
        env=os.environ.copy();env['LD_LIBRARY_PATH']=str((TOOLS/'runtime/usr/lib/x86_64-linux-gnu').resolve());env['OMP_NUM_THREADS']='1'
        with (out/'stdout.txt').open('w') as f:
            p=subprocess.run([str((TOOLS/'paup4a169_ubuntu64').resolve()),'-n','-u','-L','native.log','-r','score.nex'],cwd=out,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=300)
        assert p.returncode==0
        log=(out/'native.log').read_text();assert not re.search(r'\berrors?\b|execution terminated|processing.*terminated',log,re.I)
        totals,scores=read_scores(out/'native_scores.tsv')
        data[treatment]=scores
        for i,name in enumerate(TREES,1):
            independent=sankoff(A,trees[name],names,set(gate['ordered_characters']))
            assert all(scores[i][j]==independent[j-1] for j in range(1,458)),(treatment,name)
            totalsrows.append(dict(treatment=treatment,reference=name,total_steps=int(totals[i])))
            for ch,v in scores[i].items():characterrows.append(dict(treatment=treatment,reference=name,character=ch,steps=int(v)))
        (out/'validation.json').write_text(json.dumps(dict(native_returncode=0,all457_character_sums_verified=True,all914_character_scores_independently_reproduced=True,matrix_sha256=sha(matrix),native_tsv_sha256=sha(out/'native_scores.tsv'),driver_sha256=sha(driver)),indent=2))
        print(treatment,{TREES[i-1]:str(x) for i,x in totals.items()},flush=True)
    targetchars=set(pd.read_csv(HERE/'retained_targets.csv').character)
    for r in totalsrows:
        i=TREES.index(r['reference'])+1;t=r['treatment']
        delta={j:data[t][i][j]-data['source'][i][j] for j in range(1,458)}
        assert all(v>=0 for v in delta.values())
        assert all(delta[j]==0 for j in range(1,458) if j not in targetchars)
        r['source_steps']=int(sum(data['source'][i].values()));r['added_steps']=int(sum(delta.values()));r['later_minus_treatment']=int(sum(data['later_codes'][i].values())-r['total_steps'])
    pd.DataFrame(totalsrows).to_csv(HERE/'fixed_tree_totals.csv',index=False)
    pd.DataFrame(characterrows).to_csv(HERE/'character_scores.csv',index=False)
    (HERE/'checks.json').write_text(json.dumps(dict(scores=14,n_taxa=gate['n_taxa'],n_targets=gate['n_targets'],n_characters=457,
       all6398_native_character_scores_independently_reproduced=True,all_nontarget_character_changes_zero=True,
       all_source_completion_increments_nonnegative=True,no_tree_search=True,no_model_refitting=True),indent=2))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('operation',choices=['prepare','score']);args=ap.parse_args()
    if args.operation=='prepare':prepare()
    else:run()
