"""Lightweight relocation check of saved exact PMFs and all saved MPT bitsets.

This does not regenerate simulations, probability fits, conditioning, topology
scores or missingness. Full independent scientific audits are archived apart.
"""
from pathlib import Path
import argparse
import json
import time
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
METHODS=['oracle_joint','product_marginals','sharpened_product']


def same(a,b,tol=1e-10):
    a,b=np.asarray(a),np.asarray(b)
    assert a.shape==b.shape
    assert np.allclose(a,b,atol=tol,rtol=0,equal_nan=True)


def main(output):
    started=time.time()
    primary=ROOT/'simulation/results'
    topology=ROOT/'topology/results'
    rows=pd.read_csv(primary/'matrix_results.csv').set_index(['scenario_id','replicate','method'])
    topology_rows=pd.read_csv(topology/'topology_matrix_summary.csv').set_index(['scenario_id','matrix_index','method'])
    assert len(rows)==14400 and len(topology_rows)==7200
    design=json.loads((primary/'design.json').read_text())
    pmfs_checked=topology_sets_checked=0
    for scenario in design['scenarios']:
        sid=scenario['scenario_id']
        with np.load(primary/f'{sid}_matrices.npz') as z:
            pmf=z['matrix_D_pmfs']; true=z['true_D']; hard=z['marginal_map_D']
        same(pmf.sum(axis=-1),np.ones((100,3)))
        assert pmf.min()>=0 and np.isfinite(pmf).all()
        axis=np.arange(pmf.shape[-1]); mean=pmf@axis
        variance=(pmf*(axis-mean[...,None])**2).sum(axis=-1)
        cdf=pmf.cumsum(axis=-1)
        lower=(cdf<.025).sum(axis=-1); upper=(cdf<.975).sum(axis=-1)
        oracle_cdf=np.pad(cdf[:,0,:],((0,0),(1,0)))
        rindex=np.arange(100)[:,None]
        coverage=oracle_cdf[rindex,upper+1]-oracle_cdf[rindex,lower]
        own_cdf=np.pad(cdf,((0,0),(0,0),(1,0)))
        mindex=np.arange(3)[None,:]
        own_mass=own_cdf[rindex,mindex,upper+1]-own_cdf[rindex,mindex,lower]
        expected={'predicted_D_mean':mean,'predicted_D_variance':variance,
                  'D_q025':lower,'D_q975':upper,'interval_width':upper-lower,
                  'true_D':np.repeat(true[:,None],3,axis=1),
                  'marginal_map_D':np.repeat(hard[:,None],3,axis=1),
                  'mean_minus_truth':mean-true[:,None],
                  'mean_minus_oracle':mean-mean[:,0,None],
                  'oracle_conditional_interval_mass':coverage,'own_interval_mass':own_mass,
                  'realized_covered':(lower<=true[:,None])&(true[:,None]<=upper),
                  'D_total_variation_from_oracle':abs(pmf-pmf[:,0,None,:]).sum(axis=-1)/2}
        for field,values in expected.items():
            for m,method in enumerate(METHODS):
                actual=rows.loc[(sid,slice(None),method),field].to_numpy()
                same(values[:,m],actual)
        pmfs_checked+=300
        if scenario['n_tips']!=6:continue
        with np.load(topology/'arrays'/f'{sid}_topology.npz') as z: a={k:z[k] for k in z.files}
        for prefix in ['full','source','map','draw']:
            bits=np.unpackbits(a[prefix+'_mpt_bits'],axis=-1,bitorder='little')
            assert not bits[...,105:].any()
            bits=bits[...,:105]
            count=bits.sum(axis=-1)
            assert count.min()>0
            rf=(bits*a['rf_numerators']).sum(axis=-1)/(6*count)
            same(count,a[prefix+'_mpt_count'],0)
            same(rf,a[prefix+'_mean_normalized_rf'])
            same(bits[...,int(a['truth_tree_index'])],a[prefix+'_truth_in_mpt'],0)
            topology_sets_checked+=int(count.size)
        same(abs(a['draw_D']-a['truth_D'][:,None,None]),a['draw_absolute_D_error'],0)
        for m,method in enumerate(METHODS):
            values={
                'draw_mean_normalized_rf':a['draw_mean_normalized_rf'][:,m].mean(axis=-1),
                'draw_truth_mpt_probability':a['draw_truth_in_mpt'][:,m].mean(axis=-1),
                'draw_mean_mpt_count':a['draw_mpt_count'][:,m].mean(axis=-1),
                'draw_mean_absolute_D_error':a['draw_absolute_D_error'][:,m].mean(axis=-1),
                'draw_mean_D':a['draw_D'][:,m].mean(axis=-1)}
            for field,value in values.items():same(value,topology_rows.loc[(sid,slice(None),method),field].to_numpy())
    result={'status':'PASS','scope':'saved PMF arithmetic and saved optimum-set bitset arithmetic only',
            'exact_matrix_method_pmfs_checked':pmfs_checked,'optimum_set_bitsets_checked':topology_sets_checked,
            'new_simulations':0,'new_native_scoring':0,'new_tree_search':0,
            'runtime_seconds':time.time()-started}
    assert pmfs_checked==14400 and topology_sets_checked==472800
    output=Path(output)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args().output)
