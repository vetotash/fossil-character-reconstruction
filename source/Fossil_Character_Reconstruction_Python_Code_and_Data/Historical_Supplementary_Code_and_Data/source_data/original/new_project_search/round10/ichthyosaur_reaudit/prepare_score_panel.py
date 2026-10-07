"""Freeze exact common scoring panel and prune prespecified source trees; no scores."""
from pathlib import Path
import importlib.util,re,json,hashlib,datetime
import numpy as np
import pandas as pd

P=Path(__file__).resolve().parent
ROOT=P.parents[2]
TP=P.parent/'ichthyosaur_trees'
OUT=P/'score_inputs'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def prune(t,keep):
    if isinstance(t,str):return t if t in keep else None
    children=[z for c in t if (z:=prune(c,keep)) is not None]
    return None if not children else children[0] if len(children)==1 else children
def serialize(t):return t if isinstance(t,str) else '('+','.join(map(serialize,t))+')'
def splits(t,keep):
    out=set()
    def rec(n):
        if isinstance(n,str):return {n}
        tips=set()
        for c in n:
            desc=rec(c);tips|=desc
            left=frozenset(desc&keep);right=frozenset(keep-left)
            if min(len(left),len(right))>=2:out.add(frozenset([left,right]))
        return tips
    rec(t);return out

if __name__=='__main__':
    assert not (OUT/'PRESCORE_GATE.json').exists(),'Do not overwrite score freeze'
    OUT.mkdir(exist_ok=True)
    spec=importlib.util.spec_from_file_location('original_inventory',ROOT/'new_project_search/round8/replication/inventory.py')
    inv=importlib.util.module_from_spec(spec);spec.loader.exec_module(inv);inv.canonical=lambda s:s
    cf=json.loads((P/'CURATION_FREEZE.json').read_text())
    for fn,h in cf['hashes'].items():assert sha(P/fn)==h,(fn,'curation hash mismatch')
    z=np.load(P/'raw_parsed.npz');A=z['old_tokens'];B=z['later_tokens'];old=z['old_names'].tolist();later=z['later_names'].tolist()
    archive_names,archive_A,_=inv.parse(TP/'archive_extract/matrix.nex')
    assert archive_names==old and np.array_equal(archive_A,A),'Immutable archive mismatch'
    tf=json.loads((TP/'TREE_FREEZE.json').read_text());trees={}
    for item in tf['selected']:
        f=ROOT/item['path'];assert sha(f)==item['sha256']
        name=f'moon_source_{item["index_one_based"]:04d}'
        trees[name]=inv.parse_tree(f.read_text())
    keep=set(old)&set(later)
    for tree in trees.values():keep &= set(inv.leaves(tree))
    keep-={'Suevoleviathan_integer'}
    names=[n for n in old if n in keep]
    assert 'Hupehsuchus_nanchangensis' in names
    source=A[[old.index(n) for n in names],:]
    targets=pd.read_csv(P/'frozen_future_targets.csv',dtype={'true':str,'later_published_state':str,'source_symbol':str})
    excluded=targets[~targets.taxon.isin(keep)].copy()
    targets=targets[targets.taxon.isin(keep)].copy().reset_index(drop=True)
    targets.insert(0,'target_index',np.arange(len(targets)))
    targets['row_index']=[names.index(n) for n in targets.taxon]
    targets['column_index']=targets.character-1
    for r in targets.itertuples():
        assert source[r.row_index,r.column_index]=='?'
        assert len(r.true)==1 and r.true.isdigit()
        assert B[later.index(r.taxon),r.column_index]==r.true
    known_model_chars=np.load(P/'source_model_inputs.npz')['old_char_numbers']
    assert set(targets.character)<=set(known_model_chars)
    ordered=[99,203,204,218]
    original=(P/'raw/tjsp_a_1394922_sm0622.nex').read_text(encoding='cp1252')
    typecmd=re.search(r'TYPESET.*?;',original,re.I|re.S).group(0)
    assert re.search(r'\bord:\s*99\s+203\s*-\s*204\s+218\s*;',typecmd,re.I)
    np.savez_compressed(OUT/'panel.npz',A=source,names=np.array(names),characters=np.arange(1,288),ordered_characters=np.array(ordered),target_characters=np.sort(targets.character.unique()),target_rows=targets.row_index.to_numpy(),target_columns=targets.column_index.to_numpy(),later_states=targets.true.to_numpy(dtype='<U1'))
    targets.to_csv(OUT/'target_cells.csv',index=False)
    excluded.to_csv(OUT/'targets_outside_panel.csv',index=False)
    (OUT/'names.txt').write_text('\n'.join(names)+'\n')
    text='#NEXUS\nBEGIN DATA;\nDIMENSIONS NTAX='+str(len(names))+' NCHAR=287;\nFORMAT DATATYPE=STANDARD MISSING=? GAP=- SYMBOLS="0123";\nMATRIX\n'+''.join(n+' '+''.join(row)+'\n' for n,row in zip(names,source))+';\nEND;\nBEGIN ASSUMPTIONS;\n'+typecmd+'\nEND;\n'
    (OUT/'source_full287.nex').write_text(text)
    pn,pa,_=inv.parse(OUT/'source_full287.nex');assert pn==names and np.array_equal(pa,source)
    lines=['#NEXUS','BEGIN TREES;'];treemeta=[]
    for name,tree in trees.items():
        pruned=prune(tree,keep)
        assert set(inv.leaves(pruned))==keep and len(inv.leaves(pruned))==len(keep)
        assert splits(tree,keep)==splits(pruned,keep)
        tree_string=serialize(pruned)+';'
        reparsed=inv.parse_tree(tree_string)
        assert splits(reparsed,keep)==splits(pruned,keep)
        (OUT/(name+'.nwk')).write_text(tree_string+'\n')
        lines.append(f'TREE {name} = [&U] {tree_string}')
        treemeta.append(dict(name=name,tips=len(keep),root_degree=len(pruned),internal_degrees=sorted(set(map(len,inv.nodes(pruned)))),bipartitions_preserved=True))
    (OUT/'reference_trees.nex').write_text('\n'.join(lines+['END;'])+'\n')
    rows=[]
    for i,n in enumerate(old):
        reasons=[]
        if n not in later:reasons.append('absent_in2019_matrix')
        if any(n not in inv.leaves(t) for t in trees.values()):reasons.append('absent_in_frozen_source_trees')
        if n=='Suevoleviathan_integer':reasons.append('2019_OTU_merger')
        rows.append(dict(taxon=n,retained=n in keep,reason=';'.join(reasons),model_targets=int((pd.read_csv(P/'frozen_future_targets.csv').taxon==n).sum())))
    pd.DataFrame(rows).to_csv(OUT/'taxon_scope.csv',index=False)
    polys=[dict(taxon=old[i],character=j+1,token=A[i,j],retained_in_panel=old[i] in keep) for i,j in zip(*np.where(np.vectorize(lambda x:len(x)>1)(A)))]
    pd.DataFrame(polys).to_csv(OUT/'original_polymorphic_tokens.csv',index=False)
    metadata=dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),native_score_run=False,panel_dimensions=list(source.shape),n_targets=len(targets),n_target_taxa=targets.taxon.nunique(),n_target_characters=targets.character.nunique(),n_model_targets=196,n_excluded_model_targets=len(excluded),source_ordered_characters=ordered,target_ordered_characters=sorted(set(targets.character)&set(ordered)),polymorphic_tokens=polys,source_tokens=dict(zip(*np.unique(source,return_counts=True))),all_target_original_question_mark=True,all_targets2019singleton_verified=True,full287_characters_preserved=True,all_source_tokens_preserved=True,original_publisher_equals_immutable2019_archive=True,tree_origin='Five preselected source-study posterior topologies, NOT Maxwell2019 result trees',tree_scope='Exact common old/later matrix names and all5 source-tree tip sets, excluding2019 merged Suevoleviathan_integer; no tips added',outgroup='Hupehsuchus_nanchangensis',phalarodon_scope_note='Phalarodon_major remains in both archived matrices and source tree sample;2019 text reports its exclusion from that later analysis. Retained here only as old-background tip; not a future target and not a claim of2019 analysis-tree coverage.',tree_metadata=treemeta,hashes={str(f.relative_to(P)):sha(f) for f in sorted(OUT.iterdir()) if f.is_file()},curation_freeze_sha256=sha(P/'CURATION_FREEZE.json'),tree_selection_freeze_sha256=sha(TP/'TREE_FREEZE.json'),script_sha256=sha(Path(__file__)))
    (OUT/'PRESCORE_GATE.json').write_text(json.dumps(metadata,indent=2,default=lambda x:int(x)))
    print(json.dumps(metadata,indent=2,default=lambda x:int(x)))
