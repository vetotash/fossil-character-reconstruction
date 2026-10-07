"""Frozen marginal and joint probability propagation using native PAUP scores.

Does not fit predictors, search trees, or implement parsimony in Python.
Run after the round8 native concatenation and joint-Mk checks have passed.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, gzip, hashlib, importlib.util, json, os, time
import numpy as np
import pandas as pd

P=Path(__file__).resolve().parent
ROOT=P.parents[1]
WORK=ROOT/'morphology_workspace/morphology_project/round5/tetrapod_consequence'
PRED=WORK.parents[1]/'round3/shared_training/tetrapods/predictions.csv'
OUT=P/'probability_propagation'
spec=importlib.util.spec_from_file_location('lower',P/'method_review/audit_native_minima.py')
lower=importlib.util.module_from_spec(spec);spec.loader.exec_module(lower)
MODELS=['mode','knn5','logistic','extra_trees','nj_mk']
B=1000

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def initialize():
    freeze=json.loads((P/'PROBABILITY_FREEZE.json').read_text())
    for name,h in freeze['files'].items():assert sha(Path(name))==h,name
    check=json.loads((P/'method_review/native_lower_bounds/audit_manifest.json').read_text())
    for n in ['102','95']:
        assert check[n]['concatenation']['all_concatenated_scores_match']
        assert check[n]['concatenation']['all_concatenated_minima_match']
    d=pd.read_csv(PRED,keep_default_na=False,dtype={'true':str,'pred':str})
    rng=np.random.default_rng(20261003)
    sets={}
    for kind,scenario in [('future','later_target'),('control','control_old_score')]:
        base=d[(d.scenario==scenario)&(d.model=='mode')].sort_values(['character','taxon']).reset_index(drop=True)
        assert len(base)==350 and not base.duplicated(['character','taxon']).any()
        u=rng.random((B,len(base)))
        draws={};points={};probs={}
        for model in MODELS:
            sub=d[(d.scenario==scenario)&(d.model==model)].sort_values(['character','taxon']).reset_index(drop=True)
            assert sub[['character','taxon']].equals(base[['character','taxon']])
            assert sub.true.tolist()==base.true.tolist()
            pp=[json.loads(s) for s in sub.state_probabilities]
            assert all(set(x)=={'0','1'} and all(0<=v<=1 for v in x.values())
                       and abs(sum(x.values())-1)<1e-12 for x in pp)
            p0=np.array([x['0'] for x in pp])
            draws[model]=(u>=p0[None,:]).astype(np.uint8).astype('U1')
            points[model]=sub.pred.to_numpy(dtype='U1')
            probs[model]=p0
        jp=P/'joint_mk'/f'{kind}_joint_draws.npy'
        ji=pd.read_csv(P/'joint_mk'/f'{kind}_cells.csv',keep_default_na=False)
        lookup={(int(r.character),r.taxon):int(r.draw_column) for r in ji.itertuples()}
        order=[lookup[(int(r.character),r.taxon)] for r in base.itertuples()]
        joint=np.load(jp,allow_pickle=False)[:,order]
        assert joint.shape==(B,350) and np.isin(joint,['0','1']).all()
        draws['nj_mk_joint']=joint
        sets[kind]={'cells':base,'draws':draws,'points':points,'p0':probs}
    chars=sorted(sets['future']['cells'].character.unique().tolist())
    assert len(chars)==87
    assert sets['future']['cells'].groupby('character').size().to_dict()==sets['control']['cells'].groupby('character').size().to_dict()
    assert set(zip(sets['future']['cells'].taxon,sets['future']['cells'].character)).isdisjoint(
           zip(sets['control']['cells'].taxon,sets['control']['cells'].character))
    return sets,chars

def make_matrix(raw,chars,cells,draws):
    """Repeat only independent target columns; leave all other tokens intact."""
    names=list(raw);ntax=len(names)
    nrep=1 if draws is None else len(draws)
    lookup={j:k for k,j in enumerate(chars)}
    positions={n:[] for n in names}
    if draws is not None:
        for i,r in enumerate(cells.itertuples()):
            if r.taxon in positions:
                assert raw[r.taxon][r.character-1]=='?',(r.taxon,r.character)
                positions[r.taxon].append((lookup[r.character],i))
    strings={}
    for n in names:
        arr=np.tile(np.array([raw[n][j-1] for j in chars],dtype='U12'),(nrep,1))
        for column,i in positions[n]:arr[:,column]=draws[:,i]
        strings[n]=''.join(arr.ravel())
    nc=nrep*len(chars)
    text='#NEXUS\nBEGIN DATA;\n'+f'DIMENSIONS NTAX={ntax} NCHAR={nc};\n'+\
        'FORMAT DATATYPE=STANDARD GAP=- MISSING=? SYMBOLS="01234";\nMATRIX\n'+\
        ''.join(n+' '+strings[n]+'\n' for n in names)+';\nEND;\n'+\
        f'BEGIN ASSUMPTIONS; OPTIONS DEFTYPE=UNORD POLYTCOUNT=MINSTEPS; TYPESET * joined=unord: 1-{nc}; END;\n'
    return text

def score(raw,chars,cells,draws,trees,dest):
    nrep=1 if draws is None else len(draws)
    matrix=make_matrix(raw,chars,cells,draws)
    diag,scores,metadata=lower.run(matrix,trees,dest,nrep*len(chars))
    values=np.array([[int(v[j]) for j in range(1,nrep*len(chars)+1)]
                     for _,v in sorted(scores.items())],dtype=np.int16).reshape(len(scores),nrep,len(chars))
    minima=np.array([int(diag[j]['native_minimum_steps']) for j in range(1,nrep*len(chars)+1)],dtype=np.int16).reshape(nrep,len(chars))
    assert np.all(values>=minima[None,:,:])
    # Keep exact native evidence compressed; inputs can also be reconstructed from draws.
    compressed={}
    for name in ['diagnose.nex','native.log','native_scores.tsv','stdout.txt','lower_bounds.csv']:
        path=dest/name;rawbytes=path.read_bytes();data=gzip.compress(rawbytes,compresslevel=6,mtime=0)
        path.with_suffix(path.suffix+'.gz').write_bytes(data)
        compressed[name]={'uncompressed_bytes':len(rawbytes),'uncompressed_sha256':hashlib.sha256(rawbytes).hexdigest(),
                          'gzip_sha256':hashlib.sha256(data).hexdigest()}
        path.unlink()
    (dest/'native_evidence.json').write_text(json.dumps({'run':metadata,'compressed':compressed},indent=2)+'\n')
    return values,minima

def summarize(panel,tree_names,hard_names,hs,hm,sim,mins,rows,rep_rows):
    base=hs[:,0,:].sum(axis=-1);base_m=hm[0].sum()
    observed=hs[:,1,:].sum(axis=-1)-base
    observed_extra=observed-(hm[1].sum()-base_m)
    for model,x in sim.items():
        ds=x.sum(axis=-1)-base[:,None]
        dm=mins[model].sum(axis=-1)-base_m
        extra=ds-dm[None,:]
        point_model='nj_mk' if model=='nj_mk_joint' else model
        pi=hard_names.index(point_model)
        point=hs[:,pi,:].sum(axis=-1)-base
        point_extra=point-(hm[pi].sum()-base_m)
        for i,tree in enumerate(tree_names):
            e=extra[i];q=np.quantile(e,[.025,.975])
            rows.append({'panel':panel,'tree':tree,'model':model,'replicates':len(e),
                'reference_added_steps':int(observed[i]),'reference_added_excess_steps':int(observed_extra[i]),
                'point_added_steps':int(point[i]),'point_added_excess_steps':int(point_extra[i]),
                'sim_mean_added_steps':float(ds[i].mean()),'sim_mean_added_excess_steps':float(e.mean()),
                'sim_sd_added_excess_steps':float(e.std(ddof=1)),'sim_q025':float(q[0]),'sim_q975':float(q[1]),
                'fraction_at_least_reference':float((e>=observed_extra[i]).mean()),
                'fraction_at_most_reference':float((e<=observed_extra[i]).mean()),
                'reference_in_central95':bool(q[0]<=observed_extra[i]<=q[1]),
                'replicates_with_changed_minimum':int((dm!=0).sum())})
            rep_rows.extend({'panel':panel,'tree':tree,'model':model,'replicate':k,
                             'added_steps':int(ds[i,k]),'added_minimum_steps':int(dm[k]),
                             'added_excess_steps':int(e[k])} for k in range(len(e)))

def main(batch_size):
    OUT.mkdir(exist_ok=True);sets,chars=initialize()
    original=lower.cells((WORK/'inputs/source.nex').read_text())
    scope95=lower.cells((WORK/'exploratory_later_reference_trees/inputs/source.nex').read_text())
    common={n:row.copy() for n,row in original.items()}
    for r in sets['control']['cells'].itertuples():
        assert common[r.taxon][r.character-1]==r.true
        common[r.taxon][r.character-1]='?'
    panels=[('future95','future',scope95,WORK/'exploratory_later_reference_trees/all_five_reference_trees.tre',['O1','O2','R1','R2','R3']),
            ('commonbase_future102','future',common,WORK/'character_diagnostics/published_O1_O2.tre',['O1','O2']),
            ('commonbase_control102','control',common,WORK/'character_diagnostics/published_O1_O2.tre',['O1','O2'])]
    rows=[];rep_rows=[];inventory={};t0=time.monotonic()
    for panel,kind,raw,trees,tree_names in panels:
        folder=OUT/panel;folder.mkdir(exist_ok=True)
        bundle=sets[kind];cells=bundle['cells'];hard_names=['source','reference']+MODELS
        cells.to_csv(folder/'cells.csv',index=False)
        hard_draws=np.stack([cells.true.to_numpy(dtype='U1')]+[bundle['points'][m] for m in MODELS])
        source_steps,source_min=score(raw,chars,cells,None,trees,folder/'native_source')
        hard_steps,hard_min=score(raw,chars,cells,hard_draws,trees,folder/'native_hard')
        hs=np.concatenate([source_steps,hard_steps],axis=1);hm=np.concatenate([source_min,hard_min])
        np.savez_compressed(folder/'hard_scores.npz',steps=hs,minima=hm,characters=chars,trees=tree_names,treatments=hard_names)
        sims={};minima={}
        for model,draws in bundle['draws'].items():
            drawpath=folder/f'{model}_draws.npy';np.save(drawpath,draws,allow_pickle=False)
            pieces=[];mpieces=[]
            for start in range(0,B,batch_size):
                end=min(start+batch_size,B)
                v,m=score(raw,chars,cells,draws[start:end],trees,folder/'native'/model/f'batch_{start:04d}_{end:04d}')
                pieces.append(v);mpieces.append(m)
            sims[model]=np.concatenate(pieces,axis=1);minima[model]=np.concatenate(mpieces)
            sp=folder/f'{model}_scores.npz'
            np.savez_compressed(sp,steps=sims[model],minima=minima[model],characters=chars,trees=tree_names)
            inventory[f'{panel}/{model}']={'draw_sha256':sha(drawpath),'scores_sha256':sha(sp),'n_native_trees':len(tree_names),'n_replicates':B}
            print(json.dumps({'panel':panel,'model':model,'complete_replicates':B,'elapsed_seconds':round(time.monotonic()-t0,2)}),flush=True)
        summarize(panel,tree_names,hard_names,hs,hm,sims,minima,rows,rep_rows)
        pd.DataFrame(rows).to_csv(OUT/'summary.csv',index=False)
        pd.DataFrame(rep_rows).to_csv(OUT/'replicate_totals.csv',index=False)
    (OUT/'run_manifest.json').write_text(json.dumps({'completed_utc':datetime.now(timezone.utc).isoformat(),
        'seconds':time.monotonic()-t0,'replicates':B,'seed':20261003,'batch_size':batch_size,
        'model_refits':0,'tree_searches':0,'protocol_sha256':sha(P/'PROBABILITY_PROTOCOL.txt'),
        'script_sha256':sha(Path(__file__)),'cells_by_scenario':{k:len(v['cells']) for k,v in sets.items()},
        'target_characters':chars,'inventory':inventory},indent=2)+'\n')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--batch-size',type=int,default=200)
    main(parser.parse_args().batch_size)
