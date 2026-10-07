"""Generate a compact numerical interpretation from completed diagnostics."""
from pathlib import Path
import json
import pandas as pd

HERE=Path(__file__).resolve().parent

def main():
    check=json.loads((HERE/'DIAGNOSTIC_CHECKS.json').read_text())
    assert check['status']=='PASS' and check['score_conditions']==144
    metric=pd.read_csv(HERE/'prediction_diagnostics.csv')
    pair=pd.read_csv(HERE/'paired_prediction_diagnostics15.csv')
    scores=pd.read_csv(HERE/'probability_and_D_diagnostics.csv')
    old='archived_shared_training';full='full_source_training'
    lines=['ROUND 14 PROBABILITY DIAGNOSTIC FINDINGS','',
        'Scope: 1,874 unique matched endpoints, 18,740 old/full prediction rows,',
        '30 probability summaries and 144 old/full reference-score conditions.',
        'These counts do not denote independent biological replicates.','',
        '1. TETRAPOD ILLUSTRATION AT REFERENCE O1','',
        'Training | Model | Agreement | Entropy nats | Mean max p | Expected non-MAP cells | Hard D | Mean D | Mean D minus hard D | Later D']
    for regime in [old,full]:
        for model in ['logistic','extra_trees']:
            r=scores[(scores.training_regime==regime)&(scores.history=='tetrapods')&(scores.reference=='O1')&(scores.model==model)].iloc[0]
            lines.append(f'{regime} | {model} | {int(r.agreement_count)}/{int(r.n_endpoints)} | {r.mean_entropy_nats:.6f} | {r.mean_max_probability:.6f} | {r.expected_non_map_cells:.6f} | {int(r.D_MAP)} | {r.mean_D:.3f} | {r.mean_D_minus_D_MAP:.3f} | {int(r.later_D)}')
    lines+=['',
        'The identical 301/349 agreement and identical O1 hard D of 15 occur in',
        'the archived shared-training condition only. Restoring source scores',
        'changes individual predictions and their agreement; it must not be',
        'described as preserving that exact equality. The paired tables quantify',
        'the full-source sensitivity without selecting new endpoints.','',
        'Probability concentration is compatible with part of the propagated',
        'behavior, but these comparisons do not identify a causal entropy effect.',
        'Changing training also changes hard assignments, state alphabets in some',
        'characters and possibly the placement of probability mass across cells.','',
        '2. ALPHABET AND HARD-ASSIGNMENT CHANGES','',
        'History | Model | Alphabet changed cells | Hard prediction changed cells']
    for r in pair.itertuples():
        lines.append(f'{r.history} | {r.model} | {r.changed_alphabet_cells} | {r.changed_hard_prediction_cells}')
    lines+=['',
        'Dinosaur alphabets expand at 17 targets across four characters:',
        '107 (4 targets): {0,1} to {0,1,2};',
        '269 (3 targets): {0,2,3} to {0,1,2,3};',
        '296 (8 targets): {0} to {0,1};',
        '336 (2 targets): {1,2} to {0,1,2}.',
        'Consequently, old/full normalized entropy has a changing denominator at',
        'these targets. The raw entropy, alphabet and maximum probability are',
        'retained so the normalized metric is not interpreted in isolation.','',
        '3. REFERENCE-CONDITIONAL POSITIONS OF LATER D','',
        'Training | History | Scheme | Later D below / inside / above central 95%']
    for keys,g in scores.groupby(['training_regime','history','model'],sort=False):
        counts=g.reference_position.value_counts()
        lines.append(' | '.join(keys)+f' | {counts.get("below",0)} / {counts.get("inside",0)} / {counts.get("above",0)}')
    lines+=['',
        'Position counts describe fixed-reference conditional simulation ranges.',
        'They are not confidence interval coverage, independent replication or',
        'a test of anatomical truth.','',
        '4. SAME MARGINALS, DIFFERENT JOINT REALIZATIONS','',
        'Training | History | Joint minus marginal mean D across references']
    for regime in [old,full]:
        for history in ['tetrapods','dinosaurs','ichthyosaurs']:
            g=scores[(scores.training_regime==regime)&(scores.history==history)]
            mar=g[g.model=='nj_mk'].set_index('reference')
            joint=g[g.model=='nj_mk_joint'].set_index('reference').reindex(mar.index)
            delta=joint.mean_D-mar.mean_D
            lines.append(f'{regime} | {history} | {delta.min():.3f} to {delta.max():.3f}')
    lines+=['',
        'Mk joint and marginal schemes have identical per-cell probability',
        'summaries. Their scored means can differ, which prevents treating',
        'marginal entropy as a complete description of downstream behavior.',
        'These differences include the variation of the recorded simulations;',
        'no significance test or causal estimate is made.']
    pairs=pd.read_csv(HERE/'paired_score_diagnostics72.csv')
    fish=pairs[(pairs.history=='ichthyosaurs')&(pairs.model=='extra_trees')]
    assert len(fish)==5
    lines+=['',
        '5. AN INCREMENT IS NOT A DIRECT ENTROPY MEASURE','',
        f'For ichthyosaur Extra Trees, mean entropy changes only from {fish.mean_entropy_nats_old.iloc[0]:.6f}',
        f'to {fish.mean_entropy_nats_full.iloc[0]:.6f} nats. Yet mean D minus hard D increases',
        f'by {fish.change_mean_D_minus_D_MAP.min():.3f} to {fish.change_mean_D_minus_D_MAP.max():.3f} steps across the five references.',
        f'Hard D changes by {fish.change_D_MAP.min():.0f} to {fish.change_D_MAP.max():.0f} steps while mean D changes by',
        f'{fish.change_mean_D.min():.3f} to {fish.change_mean_D.max():.3f} steps. This is an arithmetic consequence',
        'of subtracting a different hard completion, not evidence that the',
        'predictive probabilities became substantially more diffuse.','',
        '6. REFERENCE-POSITION CHANGES','',
        'History | Scheme | Reference | Old position | Full-source position']
    changes=pairs[pairs.reference_position_old.ne(pairs.reference_position_full)]
    for r in changes.itertuples():
        lines.append(f'{r.history} | {r.model} | {r.reference} | {r.reference_position_old} | {r.reference_position_full}')
    lines+=['',
        'Validation: all probability/key/label/archive and independent training',
        'alphabet checks passed. The exact old/full score partition passed:',
        'change(mean D) = change(hard D) + change(mean D minus hard D).',
        'This equality is algebraic, not a causal decomposition.']
    (HERE/'FINDINGS.txt').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
