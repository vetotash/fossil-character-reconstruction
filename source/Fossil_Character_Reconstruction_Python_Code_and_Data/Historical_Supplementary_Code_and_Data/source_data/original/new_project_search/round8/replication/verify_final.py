"""Independent output-level checks, without rerunning predictors or native scores."""
import json
import numpy as np
import pandas as pd
from inventory import HERE,sha
from run_fixed_replication import TREES,TREATMENTS

def main():
    summary=pd.read_csv(HERE/'probability_summary_all_schemes.csv');assert len(summary)==12
    hard=np.load(HERE/'probability/hard_scores.npz',allow_pickle=False);hs=hard['steps'];hm=hard['minima'];chars=hard['characters']
    assert hs.shape==(2,7,367) and hm.shape==(7,367)
    checks=[]
    for model in ['mode','knn5','logistic','extra_trees','nj_mk','nj_mk_joint']:
        folder=HERE/('joint_probability' if model=='nj_mk_joint' else 'probability')
        file=folder/('scores.npz' if model=='nj_mk_joint' else f'{model}_scores.npz')
        data=np.load(file,allow_pickle=False);s=data['steps'];m=data['minima']
        assert s.shape==(2,1000,367) and m.shape==(1000,367) and np.array_equal(data['characters'],chars)
        assert (s>=m[None]).all()
        ds=s.sum(2)-hs[:,0,:].sum(1)[:,None];dm=m.sum(1)-hm[0].sum();e=ds-dm[None]
        assert (ds>=0).all()
        for ti,tr in enumerate(TREES):
            row=summary[(summary.model==model)&(summary.reference==tr)].iloc[0];q=np.quantile(e[ti],[.025,.975])
            assert abs(row.sim_mean_excess_steps-e[ti].mean())<1e-9
            assert np.allclose([row.sim_q025,row.sim_q975],q,atol=1e-10,rtol=0)
            obs=hs[ti,1].sum()-hs[ti,0].sum()-(hm[1].sum()-hm[0].sum())
            assert row.reference_excess_steps==obs
            assert row.reference_in_central95==bool(q[0]<=obs<=q[1])
        checks.append(dict(model=model,replicates=1000,native_character_values=int(s.size),minimum_values=int(m.size),score_sha256=sha(file)))
    proofs=list((HERE/'probability').glob('native*/**/verification.json'))+list((HERE/'joint_probability/native').glob('*/verification.json'))
    assert len(proofs)==122,len(proofs)
    for path in proofs:
        d=json.loads(path.read_text());assert d['all_character_sums_verified'] and d['all_constants_verified']
        assert d['ordered_columns']==35*d['nrep']
        for filename,evidence in d['evidence'].items():assert sha(path.parent/(filename+'.gz'))==evidence['gzip_sha256']
    report=dict(all12_summary_rows_recomputed_from_arrays=True,all12000_replicate_tree_scores_verified=True,
      simulation_character_tree_values=4404000,simulation_minimum_values=2202000,
      all122_native_evidence_hash_sets_verified=True,all_repeated_ordered_columns_correct=True,
      no_new_predictor_fit_or_tree_search=True,checks=checks)
    (HERE/'FINAL_VERIFICATION.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
