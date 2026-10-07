"""Reconstruct all retained MPT sets using the independently verified table."""
from pathlib import Path
import hashlib
import json
import math
import time
import numpy as np
import pandas as pd
from audit_topology_registry import edge_splits

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
RESULTS=ROOT/'topology/results'
METHODS=['oracle_joint','product_marginals','sharpened_product']
MAX_ERROR={}


def equal(name,a,b,tolerance=1e-11):
    a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    assert a.shape==b.shape,(name,a.shape,b.shape)
    assert np.array_equal(np.isnan(a),np.isnan(b)),(name,'NaN mismatch')
    finite=np.isfinite(a)
    delta=float(np.max(abs(a[finite]-b[finite]))) if finite.any() else 0.
    MAX_ERROR[name]=max(MAX_ERROR.get(name,0.),delta)
    assert delta<=tolerance,(name,delta)


def all_mpts(scores, rf, true_index):
    minimum=scores.min(axis=-1)
    members=scores==minimum[...,None]
    count=members.sum(axis=-1)
    return {'minimum_tree_length':minimum,'mpt_count':count,
            'mean_normalized_rf':(members*rf).sum(axis=-1)/count,
            'truth_in_mpt':members[...,true_index],
            'mpt_bits':np.packbits(members,axis=-1,bitorder='little')}


def check_stored_summary(saved, prefix, reconstructed):
    for name,value in reconstructed.items():
        equal(prefix+'_'+name,value,saved[prefix+'_'+name],tolerance=0 if name!='mean_normalized_rf' else 1e-12)


def main():
    started=time.time()
    registry_check=json.loads((HERE/'TOPOLOGY_REGISTRY_INDEPENDENT_AUDIT.json').read_text())
    assert registry_check['status']=='PASS'
    registry_path=ROOT/'topology/registry/six_tip_tree_registry.json'
    table_path=ROOT/'topology/registry/six_tip_cost_registry.npz'
    for path in (registry_path,table_path):
        assert hashlib.sha256(path.read_bytes()).hexdigest()==registry_check['input_sha256'][str(path.relative_to(ROOT))]
    registry=json.loads(registry_path.read_text())
    with np.load(table_path) as z: table={k:z[k] for k in z.files}
    fullcost=table['costs'][:,table['full_to_ternary']]
    manifest=json.loads((ROOT/'topology/topology_input_manifest.json').read_text())
    matrix=pd.read_csv(RESULTS/'topology_matrix_summary.csv').set_index(['scenario_id','matrix_index','method'])
    summary=pd.read_csv(RESULTS/'topology_scenario_summary.csv')
    paired=pd.read_csv(RESULTS/'topology_paired_contrasts.csv')
    assert len(matrix)==7200 and len(summary)==72 and len(paired)==72
    input_files=[]
    baseline_matrices_checked=draws_checked=0
    for scenario in manifest['scenarios']:
        sid=scenario['scenario_id']
        input_path=ROOT/'simulation/results'/f'{sid}_matrices.npz'
        result_path=RESULTS/'arrays'/f'{sid}_topology.npz'
        input_files += [input_path,result_path]
        with np.load(input_path) as z: a={k:z[k] for k in z.files}
        with np.load(result_path) as z: saved={k:z[k] for k in z.files}
        assert hashlib.sha256(input_path.read_bytes()).hexdigest()==scenario['input_sha256']
        true_splits=edge_splits(scenario['truth_edges'],list(range(6)))
        splits=[tuple(sorted(tuple(x) for x in t['splits'])) for t in registry['trees']]
        assert splits.count(true_splits)==1
        true_index=splits.index(true_splits)
        equal('truth_tree_index',true_index,saved['truth_tree_index'],0)
        rf=np.array([len(set(s).symmetric_difference(true_splits))/6 for s in splits])
        equal('RF_numerators',rf*6,saved['rf_numerators'],1e-12)
        full_scores=fullcost[:,a['full_pattern_ids']].sum(axis=-1).T
        known_bits=(((a['observed_state_bits'][...,None] >> np.arange(6))&1)).astype(int)
        source=known_bits.copy()
        source[((a['missing_mask_bits'][...,None] >> np.arange(6))&1).astype(bool)]=2
        ternary_ids=source@(3**np.arange(6))
        source_scores=table['costs'][:,ternary_ids].sum(axis=-1).T
        map_scores=np.moveaxis(fullcost[:,a['map_pattern_ids']].sum(axis=-1),0,-1)
        expected_full=all_mpts(full_scores,rf,true_index)
        expected_source=all_mpts(source_scores,rf,true_index)
        expected_map=all_mpts(map_scores,rf,true_index)
        for prefix,scores,reconstructed in [('full',full_scores,expected_full),('source',source_scores,expected_source),('map',map_scores,expected_map)]:
            equal(prefix+'_all_tree_scores',scores,saved[prefix+'_tree_scores'],0)
            check_stored_summary(saved,prefix,reconstructed)
        equal('truth_D',a['true_D'],saved['truth_D'],0)
        equal('map_D',np.repeat(a['marginal_map_D'][:,None],3,axis=1),saved['map_D'],0)
        equal('map_absolute_D_error',abs(saved['map_D']-saved['truth_D'][:,None]),saved['map_absolute_D_error'],0)
        draw_reconstructed={name:[] for name in ('minimum_tree_length','mpt_count','mean_normalized_rf','truth_in_mpt','mpt_bits')}
        for r in range(100):
            scores=np.moveaxis(fullcost[:,a['draw_pattern_ids'][r]].sum(axis=-1),0,-1)
            result=all_mpts(scores,rf,true_index)
            for name,value in result.items():draw_reconstructed[name].append(value)
        draw_reconstructed={k:np.asarray(v) for k,v in draw_reconstructed.items()}
        check_stored_summary(saved,'draw',draw_reconstructed)
        equal('draw_D',a['draw_D'],saved['draw_D'],0)
        equal('draw_absolute_D_error',abs(a['draw_D']-a['true_D'][:,None,None]),saved['draw_absolute_D_error'],0)
        for r in range(100):
            for m,method in enumerate(METHODS):
                row=matrix.loc[(sid,r,method)]
                score_rf=draw_reconstructed['mean_normalized_rf'][r,m]
                absent=~draw_reconstructed['truth_in_mpt'][r,m]
                exact=a['draw_D'][r,m]==a['true_D'][r]
                changed=(draw_reconstructed['mpt_bits'][r,m]!=expected_full['mpt_bits'][r]).any(axis=-1)
                worse=score_rf>expected_full['mean_normalized_rf'][r]+1e-12
                count=int(exact.sum())
                expected={'n_draws':64,'truth_D':a['true_D'][r],
                          'map_absolute_D_error':abs(a['marginal_map_D'][r]-a['true_D'][r]),
                          'draw_mean_normalized_rf':score_rf.mean(),
                          'draw_truth_mpt_probability':(~absent).mean(),
                          'draw_mean_mpt_count':draw_reconstructed['mpt_count'][r,m].mean(),
                          'draw_mean_absolute_D_error':abs(a['draw_D'][r,m]-a['true_D'][r]).mean(),
                          'draw_mean_D':a['draw_D'][r,m].mean(),
                          'D_exact_match_count':count,'D_exact_match_probability':count/64,
                          'D_exact_match_truth_absent_joint_probability':(exact&absent).mean(),
                          'D_exact_match_nonzero_rf_joint_probability':(exact&(score_rf>0)).mean(),
                          'D_exact_match_mpt_differs_full_joint_probability':(exact&changed).mean(),
                          'D_exact_match_rf_worse_than_full_count':(exact&worse).sum(),
                          'D_exact_match_rf_worse_than_full_joint_probability':(exact&worse).mean(),
                          'baseline_truth_absent_probability':absent.mean(),'baseline_nonzero_rf_probability':(score_rf>0).mean(),
                          'baseline_mpt_differs_full_probability':changed.mean(),'baseline_rf_worse_than_full_probability':worse.mean(),
                          'truth_absent_given_D_exact_match':absent[exact].mean() if count else np.nan,
                          'mean_rf_given_D_exact_match':score_rf[exact].mean() if count else np.nan,
                          'mpt_differs_full_given_D_exact_match':changed[exact].mean() if count else np.nan,
                          'rf_worse_than_full_given_D_exact_match':worse[exact].mean() if count else np.nan}
                for prefix,b in [('full',expected_full),('source',expected_source),('map',expected_map)]:
                    for metric in ['mean_normalized_rf','truth_in_mpt','mpt_count']:
                        expected[prefix+'_'+metric]=b[metric][r,m] if prefix=='map' else b[metric][r]
                for name,value in expected.items():equal('matrix_csv_'+name,value,row[name])
        baseline_matrices_checked+=500 # 100 full,100 source,300 common MAP labels.
        draws_checked+=100*3*64
        print(sid,'all MPT sets and per-matrix summaries PASS',flush=True)
    flat=matrix.reset_index()
    numeric=[c for c in matrix.columns if c!='n_draws']
    for _,r in summary.iterrows():
        part=flat[(flat.scenario_id==r.scenario_id)&(flat.method==r.method)]
        assert len(part)==100
        for field in numeric:
            values=part[field].dropna()
            mean=values.mean() if len(values) else np.nan
            se=values.std(ddof=1)/math.sqrt(len(values)) if len(values)>1 else np.nan
            equal('scenario_mean_'+field,mean,r[field+'_mean'])
            equal('scenario_mcse_'+field,se,r[field+'_mcse'])
            if field+'_n_valid' in summary.columns:
                equal('scenario_n_valid_'+field,len(values),r[field+'_n_valid'])
        exact=part.D_exact_match_count.sum()
        equal('pooled_exact_match_count',exact,r.pooled_exact_match_count,0)
        for field,numerator in [('pooled_truth_absent_given_D_exact_match',part.D_exact_match_truth_absent_joint_probability.sum()*64),
                                ('pooled_mpt_differs_full_given_D_exact_match',part.D_exact_match_mpt_differs_full_joint_probability.sum()*64),
                                ('pooled_rf_worse_than_full_given_D_exact_match',part.D_exact_match_rf_worse_than_full_count.sum())]:
            equal(field,numerator/exact if exact else np.nan,r[field])
        equal('matrices_with_exact_D_and_rf_worse_than_full',(part.D_exact_match_rf_worse_than_full_count>0).sum(),r.matrices_with_exact_D_and_rf_worse_than_full,0)
    for _,r in paired.iterrows():
        left=flat[(flat.scenario_id==r.scenario_id)&(flat.method==r.left_method)].sort_values('matrix_index')
        right=flat[(flat.scenario_id==r.scenario_id)&(flat.method==r.right_method)].sort_values('matrix_index')
        assert left.matrix_index.tolist()==right.matrix_index.tolist()==list(range(100))
        for field in [x[:-16] for x in paired.columns if x.endswith('_difference_mean')]:
            d=left[field].to_numpy()-right[field].to_numpy()
            equal('paired_mean_'+field,d.mean(),r[field+'_difference_mean'])
            equal('paired_mcse_'+field,d.std(ddof=1)/10,r[field+'_difference_mcse'])
    result={'status':'PASS','scenarios':24,'independent_matrices':2400,'saved_completion_draws_rescored':draws_checked,
            'baseline_and_common_MAP_matrices_rescored':baseline_matrices_checked,
            'tree_count':105,'every_tied_optimum_retained':True,
            'cost_table_previously_exhaustively_verified':True,'matrix_summary_rows_checked':7200,
            'scenario_summary_rows_checked':72,'paired_summary_rows_checked':72,
            'maximum_absolute_errors':MAX_ERROR,'runtime_seconds':time.time()-started,
            'input_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in input_files+[RESULTS/'topology_matrix_summary.csv',RESULTS/'topology_scenario_summary.csv',RESULTS/'topology_paired_contrasts.csv']}}
    (HERE/'TOPOLOGY_RESULTS_INDEPENDENT_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('input_sha256','maximum_absolute_errors')},indent=2))


if __name__=='__main__':main()
