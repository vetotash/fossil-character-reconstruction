#!/usr/bin/env python3
"""Independent checks of saved-array metrics, CRPS and paired Monte Carlo errors."""
import argparse,json,math
from pathlib import Path
import numpy as np
import pandas as pd
from score_draws import sha256


def run(source_root,results,output):
    output=Path(output)
    if output.exists(): raise FileExistsError(output)
    frame=pd.read_csv(results/'matrix_distribution_scores.csv')
    old=pd.read_csv(source_root/'simulation/results/matrix_results.csv')
    joined=frame.merge(old,left_on=['original_matrix_id','scheme'],right_on=['matrix_id','scheme'],suffixes=('_new','_old'),validate='one_to_one')
    if len(joined)!=2160: raise ValueError('Incomplete metric match')
    mapping={'true_D':'true_D','mean_D':'mean_D','signed_bias':'mean_D_error','absolute_mean_error':'mean_D_absolute_error','sd_D':'sd_D','lower95':'lower95','upper95':'upper95','width95':'interval_width','truth_in_95':'truth_in_interval','truth_below_95':'truth_below_interval','truth_above_95':'truth_above_interval'}
    differences={}
    for new,previous in mapping.items():
        a=joined[new+'_new' if new in old.columns else new].to_numpy(float)
        b=joined[previous+'_old' if previous in frame.columns else previous].to_numpy(float)
        differences[new]=float(np.max(np.abs(a-b)))
        np.testing.assert_allclose(a,b,atol=1e-11,rtol=1e-12)
    # Direct O(M^2) pairwise-distance formula on all six methods in all six
    # conditions and both arms of the first replicate: 72 whole distributions.
    checked=0;crps_error=0.;fair_error=0.
    for row in frame[frame.replicate==0].to_dict('records'):
        with np.load(source_root/row['source_npz'],allow_pickle=False) as x:
            schemes=['mode','knn5','logistic','extra_trees','nj_mk','nj_mk_joint']
            d=x['D_samples'][schemes.index(row['scheme'])].astype(float)
            truth=float(x['true_D'])
        half_sum=.5*np.abs(d[:,None]-d[None,:]).sum()
        empirical=np.abs(d-truth).mean()-half_sum/(len(d)**2)
        fair=np.abs(d-truth).mean()-half_sum/(len(d)*(len(d)-1))
        crps_error=max(crps_error,abs(empirical-row['crps_empirical']))
        fair_error=max(fair_error,abs(fair-row['crps_fair_ensemble']))
        np.testing.assert_allclose([empirical,fair],[row['crps_empirical'],row['crps_fair_ensemble']],atol=1e-11,rtol=1e-12)
        checked+=1
    layout=pd.read_csv(results/'paired_layout_contrasts.csv')
    methods=pd.read_csv(results/'paired_method_contrasts.csv')
    contrast_checks=0
    for table,kind in ((layout,'layout'),(methods,'method')):
        for row in table.to_dict('records'):
            subset=frame[frame.condition_id==row['condition_id']]
            if kind=='layout':
                subset=subset[subset.scheme==row['scheme']]
                p=subset.pivot(index='latent_id',columns='arm',values=row['metric'])
                d=p.empirical-p.permuted
            else:
                subset=subset[subset.arm==row['arm']]
                p=subset.pivot(index='latent_id',columns='scheme',values=row['metric'])
                d=p[row['candidate']]-p[row['reference']]
            if len(d)!=30: raise ValueError('Contrast does not have 30 paired matrices')
            np.testing.assert_allclose([d.mean(),d.std(ddof=1)/math.sqrt(30)],[row['mean_difference'],row['paired_MCSE']],atol=1e-11,rtol=1e-12)
            contrast_checks+=1
    manifest={'status':'PASS','forecast_rows_matched_to_frozen_matrix_results':2160,
              'largest_metric_absolute_difference':differences,'direct_pairwise_crps_checks':checked,
              'max_empirical_crps_difference':crps_error,'max_fair_crps_difference':fair_error,
              'paired_contrast_mean_and_mcse_checks':contrast_checks,
              'source_metric_table_sha256':sha256(source_root/'simulation/results/matrix_results.csv'),
              'matrix_scores_sha256':sha256(results/'matrix_distribution_scores.csv'),'script_sha256':sha256(__file__),
              'scope':'Post hoc distribution scoring and arithmetic verification only; no new fit or generation.'}
    output.write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',required=True,type=Path);p.add_argument('--results',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();run(a.source_root,a.results,a.output)
