#!/usr/bin/env python3
"""Build Table S28 and matrix-level descriptive checks from completed searches."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def summarize(results, output):
    scenario = pd.read_csv(results/'scenario_topology_summary.csv')
    raw = pd.read_csv(results/'selected_completion_topology.csv')
    matrix = pd.read_csv(results/'matrix_topology_summary.csv')
    assert len(raw) == 33600 and len(matrix) == 8400 and len(scenario) == 168
    assert raw.matrix_id.nunique() == 600 and (raw.groupby('matrix_id').size() == 56).all()
    assert (scenario.n_independent_matrices == 50).all()
    np.testing.assert_allclose(raw.mean_split_recall, 1-raw.mean_normalized_RF, atol=1e-12)
    schemes = ('mode', 'knn5', 'logistic', 'extra_trees', 'nj_mk', 'nj_mk_joint')
    baseline = scenario[scenario.kind == 'baseline'].set_index(['scenario_id', 'scheme'])
    rows = []; compact = []
    for identifier, group in scenario[scenario.kind == 'draw'].groupby('scenario_id', sort=True):
        source = baseline.loc[(identifier, 'source_incomplete')]
        truth = baseline.loc[(identifier, 'truth_complete')]
        first = group.iloc[0]
        short = dict(scenario_id=identifier, q=first.q,
                     missing_per_response=int(first.missing_per_response), mask_mode=first.mask_mode,
                     source_incomplete_RF=f'{source.mean_normalized_RF_mean:.4f} ({source.mean_normalized_RF_MCSE:.4f})',
                     truth_complete_RF=f'{truth.mean_normalized_RF_mean:.4f} ({truth.mean_normalized_RF_MCSE:.4f})')
        for scheme in schemes:
            r = group[group.scheme == scheme].iloc[0]
            row = dict(scenario_id=identifier, q=r.q, missing_per_response=int(r.missing_per_response),
                       mask_mode=r.mask_mode, scheme=scheme, n_independent_matrices=50,
                       sampled_RF_mean=r.mean_normalized_RF_mean, sampled_RF_MCSE=r.mean_normalized_RF_MCSE,
                       source_incomplete_RF_mean=source.mean_normalized_RF_mean,
                       source_incomplete_RF_MCSE=source.mean_normalized_RF_MCSE,
                       truth_complete_RF_mean=truth.mean_normalized_RF_mean,
                       truth_complete_RF_MCSE=truth.mean_normalized_RF_MCSE,
                       RF_difference_from_source_incomplete_mean=r.RF_difference_from_source_incomplete_mean,
                       RF_difference_from_source_incomplete_paired_MCSE=r.RF_difference_from_source_incomplete_MCSE,
                       RF_difference_from_truth_complete_mean=r.RF_difference_from_truth_complete_mean,
                       RF_difference_from_truth_complete_paired_MCSE=r.RF_difference_from_truth_complete_MCSE)
            rows.append(row)
            short[scheme+'_RF_change_from_incomplete'] = (
                f'{r.RF_difference_from_source_incomplete_mean:+.4f} ({r.RF_difference_from_source_incomplete_MCSE:.4f})')
        compact.append(short)
    output.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(output/'Table_S28.csv', index=False)
    pd.DataFrame(compact).to_csv(output/'Table_S28_compact.tsv', sep='\t', index=False)
    draws = raw[raw.kind == 'draw']
    hits = draws[draws.D_exact_and_RF_worse_than_truth_complete == 1]
    exact = draws[draws.fixed_D_exact_match == 1]
    descriptor = dict(status='PASS', n_matrices=600, n_scenarios=12,
                      n_draw_treatment_records=len(draws),
                      n_random_or_statebiased_conditions_worse_than_incomplete=int((
                          (table.mask_mode != 'structured') & (table.RF_difference_from_source_incomplete_mean > 0)).sum()),
                      n_structured_conditions_lower_than_incomplete=int((
                          (table.mask_mode == 'structured') & (table.RF_difference_from_source_incomplete_mean < 0)).sum()),
                      n_sampled_condition_means_worse_than_truth_complete=int((table.RF_difference_from_truth_complete_mean > 0).sum()),
                      n_selected_draws_exact_D=len(exact), n_selected_draws_exact_D_and_worse_RF=len(hits),
                      n_distinct_source_matrices_with_exact_D_and_worse_RF=hits.matrix_id.nunique(),
                      n_undefined_consensus_precision_records=int(raw.consensus_split_precision.isna().sum()),
                      caveat='Draw counts are descriptive; independent sampling units are600source matrices,50percondition. This script does not perform hypothesis tests.')
    (output/'TOPOLOGY_SUMMARY.json').write_text(json.dumps(descriptor, indent=2)+'\n')
    print(json.dumps(descriptor, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summarize(args.results.resolve(), args.output.resolve())
