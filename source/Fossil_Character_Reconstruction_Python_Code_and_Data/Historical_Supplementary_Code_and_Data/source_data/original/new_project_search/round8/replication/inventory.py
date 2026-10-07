"""Read-only input and topology inventory. Does not calculate parsimony scores."""
from pathlib import Path
import re, json, csv, hashlib
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[2]/'morphology_workspace/morphology_project'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def parse(p):
    text=p.read_text(); ntax=int(re.search(r'NTAX\s*=\s*(\d+)',text,re.I)[1]); nchar=int(re.search(r'NCHAR\s*=\s*(\d+)',text,re.I)[1])
    clean=re.sub(r'\[.*?\]','',text,flags=re.S)
    block=re.search(r'\bMATRIX\b(.*?);',clean,re.S|re.I)[1]
    names=[]; rows=[]
    for line in block.strip().splitlines():
        if not line.strip():continue
        m=re.match(r"\s*('(?:[^']|'')*'|\S+)\s+(.+)",line)
        names.append(m[1].strip("'")); cells=re.findall(r'\([^)]*\)|\{[^}]*\}|.',re.sub(r'\s+','',m[2]))
        assert len(cells)==nchar,(names[-1],len(cells)); rows.append(cells)
    assert len(names)==len(set(names))==ntax
    return names,np.array(rows),text
def canonical(n):
    g=n.split('_')[0]
    if g=='Lewisuchus':return 'Lewisuchus_Pseudolagosuchus'
    if g in ['Dromomeron','Massospondylus']:return n.replace('kaalea','kaalae')
    return g
def parse_tree(text):
    tok=re.findall(r"'(?:[^']|'')*'|[(),:;]|[^\s(),:;]+",text); ix=0
    def parse():
        nonlocal ix
        if tok[ix]=='(':
            ix+=1; node=[parse()]
            while tok[ix]==',':ix+=1;node.append(parse())
            assert tok[ix]==')';ix+=1
            if ix<len(tok) and tok[ix] not in ',):;':ix+=1
        else:node=canonical(tok[ix].strip("'"));ix+=1
        if ix<len(tok) and tok[ix]==':':ix+=2
        return node
    result=parse()
    if ix<len(tok) and tok[ix]==';':ix+=1
    assert ix==len(tok)
    return result
def leaves(t): return [t] if isinstance(t,str) else [x for ch in t for x in leaves(ch)]
def nodes(t): return [] if isinstance(t,str) else [t]+[x for ch in t for x in nodes(ch)]
def run():
    old,A,txt=parse(HERE/'raw/baron_original.nex'); new,B,_=parse(HERE/'raw/langer_original.nex')
    on=list(map(canonical,old)); nn=list(map(canonical,new)); assert len(set(on))==75 and len(set(nn))==83
    old_npz=np.load(PROJECT/'round2/inputs/dinosaurs.npz',allow_pickle=True)
    assert on==old_npz['old_names'].tolist()
    # Existing arrays simplify non-singleton tokens to '?'; original NEXUS remains authoritative.
    simple=np.where(np.vectorize(lambda x:bool(re.fullmatch('[0-9A-Z]',x)))(A),A,'?')
    assert np.array_equal(simple,old_npz['A'])
    predictions=pd.read_csv(PROJECT/'round3/shared_training/dinosaurs/predictions.csv')
    pred=predictions[predictions.scenario=='later_target']; targets=pred[pred.model=='mode']
    assert len(targets)==1333
    for r in targets.itertuples():
        assert A[on.index(r.taxon),r.character-1]=='?'
        assert B[nn.index(r.taxon),r.character-1]==str(r.true)
    trees={}; trrows=[]
    for stem in ['baron','langer']:
        p=HERE/f'raw/ML_unconstrained/ML_unconstrained/{stem}_unconstrained.treefile'
        tree=parse_tree(p.read_text());tips=leaves(tree);ct=tips;assert len(ct)==len(set(ct))
        trees[stem]=set(ct)
        trrows.append(dict(name=stem,file=str(p.relative_to(HERE)),sha256=sha(p),tips=len(tips),internal_nodes=len(nodes(tree)),internal_degrees=sorted(set(map(len,nodes(tree))))))
    common=set(on)&set(nn)&trees['baron']&trees['langer']
    m=[]
    for a,c in zip(old,on):
        m.append(dict(canonical=c,old_name=a,later_name=new[nn.index(c)] if c in nn else '',in_both_reference_trees=c in common,targets=int((targets.taxon==c).sum())))
    pd.DataFrame(m).to_csv(HERE/'taxon_inventory.csv',index=False)
    ordered=re.search(r'TYPESET.*?ORD:\s*([^;]+);',txt,re.I|re.S)
    output=dict(old_shape=list(A.shape),later_shape=list(B.shape),frozen_array_matches_simplified_original=True,
        frozen_targets_all1333_exact_old_missing_later_singleton=True,old_non_singleton_tokens=int(np.sum(~np.isin(A,list('01234?-')))),
        common_taxa=len(common),common_targets=int(targets.taxon.isin(common).sum()),common_target_characters=int(targets.loc[targets.taxon.isin(common),'character'].nunique()),
        unmatched_source_names=sorted(set(on)-common),trees=trrows,
        source_sha256=sha(HERE/'raw/baron_original.nex'),later_sha256=sha(HERE/'raw/langer_original.nex'),predictions_sha256=sha(PROJECT/'round3/shared_training/dinosaurs/predictions.csv'))
    (HERE/'input_inventory.json').write_text(json.dumps(output,indent=2));print(json.dumps(output,indent=2))
if __name__=='__main__':run()
