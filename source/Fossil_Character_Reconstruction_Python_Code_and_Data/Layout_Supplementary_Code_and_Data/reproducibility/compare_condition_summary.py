"""Check a newly generated condition summary against archived matrix-level rows.

The original archive did not retain scenario_summary.csv. This comparison uses
independent pandas reductions and explicit interval arithmetic, not the new
matrix-result table or the production experiment.summarize function.
"""
from pathlib import Path
import math
import numpy as np
import pandas as pd
import experiment

def compare_condition_summary(current_path, archived_matrix_path, archived_registry_path):
    current=pd.read_csv(current_path)
    archived=pd.read_csv(archived_matrix_path)
    scenarios=pd.read_csv(archived_registry_path).set_index('scenario_id')
    expected_keys={(s,a,m) for s in scenarios.index for a in experiment.ARMS for m in experiment.SCHEMES}
    actual_keys=set(zip(current.scenario_id,current.arm,current.scheme))
    assert len(current)==len(expected_keys)==len(actual_keys) and actual_keys==expected_keys
    largest=0.0;checked=0;z=1.959963984540054
    for row in current.to_dict('records'):
        group=archived[(archived.scenario_id==row['scenario_id'])&(archived.arm==row['arm'])&(archived.scheme==row['scheme'])]
        n=len(group);hits=int(group.truth_in_interval.sum());p=hits/n
        den=1+z*z/n; center=(p+z*z/(2*n))/den
        half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
        expected={'n_matrices':n,'interval_coverage':p,'coverage_mcse':math.sqrt(p*(1-p)/n),
                  'coverage_wilson_lower95':center-half,'coverage_wilson_upper95':center+half,
                  'count_truth_below_interval':int(group.truth_below_interval.sum()),
                  'count_truth_above_interval':int(group.truth_above_interval.sum()),
                  'zero_true_probability_total':int(group.zero_true_probability_count.sum()),
                  'log_floor_total':int(group.log_floor_count.sum())}
        for metric in experiment.METRICS:
            expected[metric+'_mean']=float(group[metric].mean())
            expected[metric+'_mcse']=float(group[metric].std(ddof=1)/math.sqrt(n))
        for key,value in expected.items():
            diff=abs(float(row[key])-float(value));largest=max(largest,diff);checked+=1
            assert np.isclose(row[key],value,rtol=1e-12,atol=1e-12),(row['scenario_id'],row['arm'],row['scheme'],key,row[key],value)
        for key,value in scenarios.loc[row['scenario_id']].to_dict().items():
            if isinstance(value,(int,float,np.number)):assert np.isclose(row[key],value,rtol=1e-12,atol=1e-12),(key,row[key],value)
            else:assert row[key]==value,(key,row[key],value)
    return {'equal_at_1e-12':True,'rows':len(current),'summary_values_checked':checked,
            'largest_absolute_difference':largest,
            'reference':'Independent aggregation of original archived matrix_results.csv and scenario_registry.csv; an original scenario_summary.csv was not present',
            'original_condition_summary_file_available':False}
