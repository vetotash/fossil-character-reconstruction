#!/usr/bin/env python3
"""Recompute historical summaries from archived probabilities and score arrays."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,math
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
SCHEMES=['mode','knn5','logistic','extra_trees','nj_mk','nj_mk_joint']
EXPECTED={'tetrapods':350,'dinosaurs':1333,'ichthyosaurs':196,'gnathostomes':212,'squamates':163}

def verify():
    manifest=json.loads((ROOT/'MANIFEST.json').read_text())
    for row in manifest['files']:
        path=ROOT/row['path']
        if path.stat().st_size!=row['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:
            raise ValueError(f"Integrity mismatch: {row['path']}")
    print(f"Verified {len(manifest['files'])} files")

def compare_csv(actual,expected,keys):
    a=pd.read_csv(actual).sort_values(keys).reset_index(drop=True)
    b=pd.read_csv(expected).sort_values(keys).reset_index(drop=True)
    assert list(a.columns)==list(b.columns) and a.shape==b.shape
    for col in a:
        if pd.api.types.is_numeric_dtype(a[col]) and not pd.api.types.is_bool_dtype(a[col]):
            np.testing.assert_allclose(a[col],b[col],rtol=0,atol=1e-10,equal_nan=True)
        else:assert a[col].equals(b[col]),col

def primary(out):
    path=ROOT/'primary/build_synthesis.py'
    spec=importlib.util.spec_from_file_location('archived_primary_synthesis',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.verify_inputs()
    module.HERE=out;out.mkdir()
    scores,_=module.score_synthesis();module.prediction_synthesis();module.table2(scores)
    compare_csv(out/'main72_summary.csv',ROOT/'primary/main72_summary.csv',['history','model','reference'])
    compare_csv(out/'matched_prediction_metrics15.csv',ROOT/'primary/matched_prediction_metrics15.csv',['history','model'])
    return {'reference_summaries':72,'draw_tree_values':72000,'matched_D_targets':1874}

def states(out):
    records=[]
    for history,count in EXPECTED.items():
        data=pd.read_csv(ROOT/'predictions'/f'{history}.csv',keep_default_na=False,dtype={'true':str,'pred':str})
        for (scenario,model),group in data.groupby(['scenario','model'],sort=True):
            assert not group.duplicated(['taxon','character']).any()
            if scenario=='later_target':assert len(group)==count
            bs=[];losses=[];correct=[]
            for row in group.itertuples():
                prob=json.loads(row.state_probabilities)
                assert all(math.isfinite(v) and v>=0 for v in prob.values()) and abs(sum(prob.values())-1)<1e-10
                pt=prob.get(row.true,0.)
                b=sum((prob.get(s,0.)-int(s==row.true))**2 for s in set(prob)|{row.true})
                loss=-math.log(max(pt,1e-12))
                assert abs(b-row.brier)<1e-10 and abs(loss-row.log_loss)<1e-10
                assert row.pred in prob and abs(prob[row.pred]-max(prob.values()))<1e-10
                bs.append(b);losses.append(loss);correct.append(int(row.pred==row.true))
            records.append(dict(history=history,scenario=scenario,model=model,n_targets=len(group),agreement_count=sum(correct),agreement=np.mean(correct),brier=np.mean(bs),log_loss=np.mean(losses)))
    pd.DataFrame(records).to_csv(out/'state_prediction_summary.csv',index=False)
    return {'historical_targets':sum(EXPECTED.values()),'prediction_rows_scored':sum(r['n_targets'] for r in records),'summary_rows':len(records)}

def full_source(out):
    base=ROOT/'full_source/downstream';reference=pd.read_csv(base/'full_source72_summary.csv')
    rows=[];replicates=[]
    for history in ['tetrapods','dinosaurs','ichthyosaurs']:
        folder=base/history
        with np.load(folder/'hard_scores.npz',allow_pickle=False) as hard:
            steps=hard['steps'].astype(np.int64);minimum=hard['minima'].astype(np.int64)
            trees=hard['trees'].tolist();characters=hard['characters'].tolist();treatments=hard['treatments'].tolist()
        old_excess=(steps[:,0,:]-minimum[0]).sum(1)
        later_D=(steps[:,1,:]-minimum[1]).sum(1)-old_excess
        for scheme in SCHEMES:
            with np.load(folder/f'{scheme}_scores.npz',allow_pickle=False) as saved:
                assert saved['trees'].tolist()==trees and saved['characters'].tolist()==characters
                values=(saved['steps'].astype(np.int64)-saved['minima'][None,:,:]).sum(2)-old_excess[:,None]
            index=treatments.index('nj_mk' if scheme=='nj_mk_joint' else scheme)
            hard_D=(steps[:,index,:]-minimum[index]).sum(1)-old_excess
            for i,tree in enumerate(trees):
                d=values[i];lower,upper=np.quantile(d,[.025,.975])
                row=dict(history=history,model=scheme,reference=tree,reference_excess_steps=int(later_D[i]),point_excess_steps=int(hard_D[i]),sim_mean_excess_steps=float(d.mean()),sim_sd_excess_steps=float(d.std(ddof=1)),sim_q025=float(lower),sim_q975=float(upper),reference_in_central95=bool(lower<=later_D[i]<=upper))
                orig=reference[(reference.history==history)&(reference.model==scheme)&(reference.reference==tree)]
                assert len(orig)==1
                for key,value in row.items():
                    if isinstance(value,(float,int)):assert abs(value-float(orig.iloc[0][key]))<1e-10,key
                    else:assert value==orig.iloc[0][key],key
                rows.append(row)
                replicates.extend(dict(history=history,model=scheme,reference=tree,replicate=j,D=int(v)) for j,v in enumerate(d))
    assert len(rows)==72 and len(replicates)==72000
    pd.DataFrame(rows).to_csv(out/'full_source_summary.csv',index=False)
    pd.DataFrame(replicates).to_csv(out/'full_source_draws.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    return {'reference_summaries':len(rows),'draw_tree_values':len(replicates)}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['verify','replay'])
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    verify()
    if args.command=='verify':return
    if args.output is None:parser.error('replay requires --output NEW_DIRECTORY')
    out=args.output.resolve()
    if out==ROOT or ROOT in out.parents:raise ValueError('Output must be outside the supplied archive')
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    results={'primary':primary(out/'primary'),'all_five_histories':states(out),'full_source_sensitivity':full_source(out),'scope':'Python replay of frozen probabilities and native character-score arrays; no refitting or new native parsimony run'}
    (out/'replay_manifest.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))

if __name__=='__main__':main()
