#!/usr/bin/env python3
"""Pair archived cell scores with newly calculated D-distribution scores."""
import argparse,itertools,json,math
from pathlib import Path
import numpy as np
import pandas as pd
from score_draws import SCHEMES,sha256


def stats(d):
    return float(d.mean()),float(d.std(ddof=1)/math.sqrt(len(d)))


def run(root,results,output):
    if output.exists() and any(output.iterdir()):raise FileExistsError(output)
    original=root/'simulation/results/matrix_results.csv'
    frame=pd.read_csv(original)
    source_contrasts=pd.read_csv(root/'simulation/results/paired_layout_contrasts.csv')
    distribution=pd.read_csv(results/'matrix_distribution_scores.csv')
    scores=distribution.merge(frame[['matrix_id','scheme','brier','log_loss','accuracy']],left_on=['original_matrix_id','scheme'],right_on=['matrix_id','scheme'],validate='one_to_one',suffixes=('','_archived'))
    metrics=['brier','log_loss','accuracy','crps_empirical','crps_fair_ensemble','signed_bias','width95','truth_in_95']
    layout=[];method=[]
    for (condition,panel,q,scheme),g in scores.groupby(['condition_id','panel','q','scheme']):
        row=dict(condition_id=condition,panel=panel,q=q,scheme=scheme,n_pairs=30,difference_direction='empirical_minus_permuted')
        for metric in metrics:
            p=g.pivot(index='latent_id',columns='arm',values=metric);mean,se=stats(p.empirical-p.permuted)
            row[metric+'_difference_mean']=mean;row[metric+'_difference_MCSE']=se
        original_row=source_contrasts[(source_contrasts.scenario_id==condition)&(source_contrasts.scheme==scheme)].iloc[0]
        for metric in ('brier','log_loss','accuracy'):
            np.testing.assert_allclose([row[metric+'_difference_mean'],row[metric+'_difference_MCSE']],[original_row[metric+'_difference_mean'],original_row[metric+'_difference_mcse']],atol=1e-12)
        row['brier_CRPS_point_estimate_direction_agrees']=bool(np.sign(row['brier_difference_mean'])==np.sign(row['crps_empirical_difference_mean']))
        row['empirical_fair_CRPS_point_estimate_direction_agrees']=bool(np.sign(row['crps_empirical_difference_mean'])==np.sign(row['crps_fair_ensemble_difference_mean']))
        layout.append(row)
    for (condition,panel,q,arm),g in scores.groupby(['condition_id','panel','q','arm']):
        for reference,candidate in itertools.combinations(SCHEMES,2):
            row=dict(condition_id=condition,panel=panel,q=q,arm=arm,reference=reference,candidate=candidate,n_pairs=30,difference_direction='candidate_minus_reference')
            for metric in metrics:
                p=g.pivot(index='latent_id',columns='scheme',values=metric);mean,se=stats(p[candidate]-p[reference])
                row[metric+'_difference_mean']=mean;row[metric+'_difference_MCSE']=se
            method.append(row)
    layout=pd.DataFrame(layout);method=pd.DataFrame(method)
    output.mkdir(parents=True,exist_ok=True)
    layout.to_csv(output/'paired_layout_cell_and_distribution_scores.csv',index=False)
    method.to_csv(output/'paired_method_cell_and_distribution_scores.csv',index=False)
    manifest={'status':'PASS','layout_groups':36,'method_pair_groups':180,'metrics':metrics,
              'cell_scores_source':'Archived per-arm per-method matrix_results.csv; no probability refitting/recalculation',
              'distribution_scores_source':'New CRPS scoring of all original saved 512-draw distributions',
              'brier_CRPS_direction_disagreements':int((~layout.brier_CRPS_point_estimate_direction_agrees).sum()),
              'empirical_fair_CRPS_layout_direction_disagreements':int((~layout.empirical_fair_CRPS_point_estimate_direction_agrees).sum()),
              'script_sha256':sha256(__file__),'original_metrics_sha256':sha256(original),
              'new_distribution_scores_sha256':sha256(results/'matrix_distribution_scores.csv'),
              'output_hashes':{p.name:sha256(p) for p in output.glob('*.csv')},
              'interpretation':'Signs describe point estimates, not multiplicity-adjusted significance tests. All paired MCSE use 30 independent latent matrices.'}
    (output/'endpoint_comparison_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',type=Path,required=True);p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.source_root,a.results,a.output)
