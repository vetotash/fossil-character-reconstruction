"""Score independently generated exact conditional NJ Mk draws, without refitting."""
from pathlib import Path
import datetime,json,time
import numpy as np
import pandas as pd
from inventory import HERE,parse,sha
from run_fixed_replication import source_ordered,TREES,TREATMENTS
from run_probability import score

INP=HERE.parent/'joint_mk_dinosaurs';OUT=HERE/'joint_probability';B=1000
def main():
    assert (HERE/'probability/run_manifest.json').exists(),'All5 prespecified marginal schemes must complete first'
    recon=json.loads((INP/'RECONSTRUCTION_CHECK.json').read_text());assert recon['all_marginals_reproduced'] and recon['maximum_probability_error']<1e-10
    assert recon['refitted_rates']==recon['reestimated_trees']==0 and recon['seed']==2026100318 and recon['n_draws']==B
    assert not OUT.exists();OUT.mkdir()
    cells=pd.read_csv(HERE/'probability/cells.csv',keep_default_na=False,dtype={'true':str,'pred':str})
    ji=pd.read_csv(INP/'future_cells.csv',keep_default_na=False)
    lookup={(int(r.character),r.taxon):int(r.draw_column) for r in ji.itertuples()};assert len(lookup)==1333
    order=[lookup[(int(r.character),r.taxon)] for r in cells.itertuples()]
    original=np.load(INP/'future_joint_draws.npy',allow_pickle=False);draws=original[:,order]
    assert draws.shape==(B,1333) and np.isin(draws,list('01234')).all()
    np.save(OUT/'aligned_draws.npy',draws,allow_pickle=False)
    cells[['character','taxon']].to_csv(OUT/'cells.csv',index=False)
    oldfreeze=json.loads((HERE/'probability/FREEZE.json').read_text());assert sha(HERE/'run_probability.py')==oldfreeze['script_sha256']
    gate=dict(time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),native_scores_present=False,seed=2026100318,n_draws=B,n_cells=1333,
      files={str(x.relative_to(HERE.parent)):sha(x) for x in [INP/'RECONSTRUCTION_CHECK.json',INP/'PROTOCOL.txt',INP/'FREEZE.json',INP/'future_joint_draws.npy',INP/'future_cells.csv',INP/'character_state_map.csv',HERE/'JOINT_PROTOCOL.txt',Path(__file__),HERE/'run_probability.py',HERE/'reference_trees.nex',HERE/'inputs/source.nex']},aligned_draws_sha256=sha(OUT/'aligned_draws.npy'))
    (OUT/'FREEZE.json').write_text(json.dumps(gate,indent=2))
    names,raw,text=parse(HERE/'inputs/source.nex');ordered=source_ordered(text);chars=sorted(cells.character.unique().tolist())
    hard=np.load(HERE/'probability/hard_scores.npz',allow_pickle=False);hs=hard['steps'];hm=hard['minima'];assert list(hard['characters'])==chars
    pieces=[];mpieces=[];t0=time.monotonic()
    for start in range(0,B,50):
        end=min(start+50,B);s,m=score(raw,names,chars,cells,draws[start:end],ordered,OUT/'native'/f'batch_{start:04d}_{end:04d}');pieces.append(s);mpieces.append(m)
    s=np.concatenate(pieces,axis=1);m=np.concatenate(mpieces)
    np.savez_compressed(OUT/'scores.npz',steps=s,minima=m,characters=chars,trees=TREES)
    base=hs[:,0,:].sum(axis=1);base_m=hm[0].sum();ds=s.sum(axis=2)-base[:,None];dm=m.sum(axis=1)-base_m;extra=ds-dm[None,:]
    obs=hs[:,1,:].sum(axis=1)-base-(hm[1].sum()-base_m)
    k=TREATMENTS.index('nj_mk');point=hs[:,k,:].sum(axis=1)-base-(hm[k].sum()-base_m)
    rows=[];reps=[]
    for ti,tr in enumerate(TREES):
        e=extra[ti];q=np.quantile(e,[.025,.975]);rows.append(dict(model='nj_mk_joint',reference=tr,replicates=B,reference_excess_steps=int(obs[ti]),point_excess_steps=int(point[ti]),sim_mean_excess_steps=float(e.mean()),sim_sd_excess_steps=float(e.std(ddof=1)),sim_q025=float(q[0]),sim_q975=float(q[1]),fraction_at_least_reference=float((e>=obs[ti]).mean()),fraction_at_most_reference=float((e<=obs[ti]).mean()),reference_in_central95=bool(q[0]<=obs[ti]<=q[1]),mean_added_steps=float(ds[ti].mean()),mean_added_minimum_steps=float(dm.mean())))
        reps.extend(dict(model='nj_mk_joint',reference=tr,replicate=i,added_steps=int(ds[ti,i]),added_minimum_steps=int(dm[i]),added_excess_steps=int(e[i])) for i in range(B))
    pd.DataFrame(rows).to_csv(OUT/'summary.csv',index=False);pd.DataFrame(reps).to_csv(OUT/'replicate_totals.csv',index=False)
    combined=pd.concat([pd.read_csv(HERE/'probability/summary.csv'),pd.DataFrame(rows)],ignore_index=True);combined.to_csv(HERE/'probability_summary_all_schemes.csv',index=False)
    (OUT/'run_manifest.json').write_text(json.dumps(dict(completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),seconds=time.monotonic()-t0,n_replicates=B,n_schemes=1,seed=2026100318,native_calls=20,refits=0,tree_searches=0,freeze_sha256=sha(OUT/'FREEZE.json'),score_sha256=sha(OUT/'scores.npz')),indent=2))
    print(pd.DataFrame(rows).to_string(index=False))
if __name__=='__main__':main()
