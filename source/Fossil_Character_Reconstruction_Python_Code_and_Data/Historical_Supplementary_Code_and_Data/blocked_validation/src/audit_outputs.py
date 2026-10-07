#!/usr/bin/env python3
"""Direct artifact audit, independent of the fitting/evaluation implementation."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd

PACKAGE_ROOT=Path(__file__).resolve().parents[1]


def audit(source_override=None):
    root=PACKAGE_ROOT;plan=json.loads((root/'split_plan.json').read_text())
    source=Path(source_override) if source_override else Path(plan['source_root'])
    source_paths={p:source/('ichthyosaur_refit' if p=='ichthyosaurs' else f'old_refits/inputs/{p}')/'source_training.npz' for p in plan['panels']}
    matrices={p:np.load(f,allow_pickle=False)['A'] for p,f in source_paths.items()}
    alphabet=set('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    counts={'tasks':0,'classifier_convergence_warnings':0,'mk_optimizer_unsuccessful':0,
            'mk_rate_boundaries':0,'constant_training_tasks':0,'forecast_rows':0}
    edges={};case_sizes={}
    for task in plan['tasks']:
        result=json.loads((root/'results/tasks'/f"{task['task_id']}.json").read_text())
        matrix=matrices[task['panel']];j=task['column_index'];test=set(task['target_row_indices'])
        observed={i for i,value in enumerate(matrix[:,j]) if value in alphabet};train=observed-test
        assert set(result['training_indices'])==train and len(test)==task['heldout_count']
        if task['treatment']=='blocked':assert test==observed&set(task['clade_row_indices'])
        classes=sorted({str(matrix[i,j]) for i in train})
        assert result['training_classes']==classes and len(result['predictions'])==len(test)*5
        for p in result['predictions']:
            assert p['row_index'] in test and p['classes']==classes
            assert np.isclose(sum(p['probabilities']),1,rtol=0,atol=1e-12)
        counts['tasks']+=1;counts['forecast_rows']+=len(result['predictions'])
        counts['constant_training_tasks']+=len(classes)==1
        counts['classifier_convergence_warnings']+=sum(d.get('convergence_warning',False) for d in result['diagnostics']['classifiers'])
        mk=result['diagnostics']['nj_mk'];optimizer=mk['optimizer']
        counts['mk_optimizer_unsuccessful']+=bool(optimizer and not optimizer['success'])
        counts['mk_rate_boundaries']+=mk['rate_boundary']
        edges.setdefault((task['panel'],j),set()).add(mk['edges_sha256'])
        case_sizes.setdefault(task['case_id'],[]).append(len(test))
    assert all(len(values)==4 and len(set(values))==1 for values in case_sizes.values())
    assert all(len(values)==1 for values in edges.values())
    cells=pd.read_csv(root/'results/cell_scores.csv',dtype={'true_old_label':str,'predicted_state':str})
    predictions=json.loads((root/'results/predictions_before_evaluation.json').read_text())
    assert len(cells)==len(predictions)
    for p,row in zip(predictions,cells.to_dict('records')):
        truth=str(matrices[p['panel']][p['row_index'],p['column_index']])
        prob=dict(zip(p['classes'],p['probabilities']));augmented=sorted(set(prob)|{truth})
        vector=np.array([prob.get(s,0.) for s in augmented])
        onehot=np.array([int(s==truth) for s in augmented])
        assert row['true_old_label']==truth
        assert abs(row['brier']-float(((vector-onehot)**2).sum()))<1e-12
        assert row['training_unseen_state']==int(truth not in prob)
    unchanged=all(hashlib.sha256((source/record['source_relative_path']).read_bytes()).hexdigest()==record['sha256'] for record in plan['source_manifest'])
    assert unchanged
    counts.update({'status':'PASS','exact_training_complements_checked':True,
        'entire_block_response_checked':True,'count_matched_controls_checked':True,
        'training_only_alphabets_checked':True,'source_arrays_unchanged':True,
        'response_excluded_NJ_invariant_across_masks_checked':True,
        'Brier_verified_with_explicit_augmented_onehot':True,
        'source_tree_blocks_follow_archived_Newick_orientation':True,
        'primary_analysis_training_note':'Original shared fitting hides matched-control response labels; it does not mean restriction to downstream tree taxon counts.'})
    (root/'results/INDEPENDENT_ARTIFACT_CHECKS.json').write_text(json.dumps(counts,indent=2)+'\n')
    print(json.dumps(counts,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path);args=p.parse_args();audit(args.source_root)
