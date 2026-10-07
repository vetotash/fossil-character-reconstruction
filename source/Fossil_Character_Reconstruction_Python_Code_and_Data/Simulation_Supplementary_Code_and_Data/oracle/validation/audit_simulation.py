"""Independent reconstruction of every Round 15 primary PMF and saved draw.

No production scientific functions are imported. Generating probabilities and
costs come from enumeration of internal states; conditioning and convolution
use independent_engine. Saved random draws are replayed with recorded seeds.
"""
from pathlib import Path
import csv
import hashlib
import json
import math
import time
import numpy as np
import pandas as pd
from independent_engine import (enumerate_internal_states, conditions,
                                character_pmf, convolve_pmfs,
                                summarize_distribution, oracle_interval_mass)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / 'simulation/results'
METHODS = ['oracle_joint', 'product_marginals', 'sharpened_product']
MAX_ERROR = {}


def equal(name, expected, actual, atol=2e-10):
    expected, actual = np.asarray(expected), np.asarray(actual)
    assert expected.shape == actual.shape, (name, expected.shape, actual.shape)
    delta = float(np.max(np.abs(expected.astype(float) - actual.astype(float)))) if expected.size else 0.
    MAX_ERROR[name] = max(MAX_ERROR.get(name, 0.), delta)
    assert delta <= atol, (name, delta, atol)


def entropy(p):
    p = np.asarray(p, float)
    return -(p * np.log2(np.clip(p, 1e-300, 1)) + (1-p) * np.log2(np.clip(1-p, 1e-300, 1)))


def main():
    started = time.time()
    assert json.loads((ROOT / 'DESIGN_FREEZE.json').read_text())['status'] == 'DESIGN_FROZEN'
    production_check = json.loads((RESULTS / 'RUN_CHECKS.json').read_text())
    assert production_check['status'] == 'PASS'
    design = json.loads((RESULTS / 'design.json').read_text())
    trees = json.loads((RESULTS / 'trees.json').read_text())
    registry = np.load(RESULTS / 'pattern_registry.npz')
    rows = pd.read_csv(RESULTS / 'matrix_results.csv')
    summary = pd.read_csv(RESULTS / 'scenario_summary.csv')
    assert len(rows) == 14400 and len(summary) == 144
    assert list(rows.groupby('scenario_id').size()) == [300] * 48
    assert len(rows[['scenario_id', 'replicate', 'method']].drop_duplicates()) == 14400
    probability_lookup, cost_lookup = {}, {}
    for key, tree in trees.items():
        n = tree['n_tips']
        tips = list(range(n))
        assert len(set(tuple(sorted(e)) for e in tree['edges'])) == 2*n-3
        assert tree['edge_lengths'] == [1.] * (2*n-3)
        patterns = registry[f'n{n}_patterns']
        for qi, q in enumerate((.02, .1, .5)):
            prob, cost = enumerate_internal_states(patterns, tree['edges'], tips, q)
            equal('unconditional_probability_normalization', prob.sum(), 1.)
            prob /= prob.sum()
            equal('unconditional_pattern_probabilities', prob, registry[f'{key}_q{qi}_probabilities'])
            equal('complete_pattern_steps', cost, registry[f'{key}_full_steps'], atol=0)
            minimum = np.any(patterns != patterns[:, :1], axis=1).astype(int)
            equal('complete_pattern_minima', minimum, registry[f'{key}_full_minima'], atol=0)
            probability_lookup[(key, q)] = prob
            cost_lookup[key] = cost
    draw_count = 0
    conditional_count = 0
    input_files = []
    for scenario in design['scenarios']:
        sid = scenario['scenario_id']
        ordinal, n, nchar, k = [scenario[x] for x in ('ordinal', 'n_tips', 'n_characters', 'missing_per_character')]
        inputfile = RESULTS / f'{sid}_matrices.npz'
        input_files.append(inputfile)
        with np.load(inputfile) as loaded:
            a = {key: loaded[key] for key in loaded.files}
        patterns = registry[f'n{n}_patterns']
        tree = trees[scenario['tree_key']]
        prior = probability_lookup[(scenario['tree_key'], scenario['q'])]
        costs = cost_lookup[scenario['tree_key']]
        full_minima = np.any(patterns != patterns[:, :1], axis=1).astype(int)
        rng_truth = np.random.default_rng(np.random.SeedSequence([2026100401, ordinal, 0]))
        # This prior has independently passed the complete enumeration comparison.
        archived_prior = registry[f"{scenario['tree_key']}_q{(.02,.1,.5).index(scenario['q'])}_probabilities"]
        truth = rng_truth.choice(len(patterns), size=(100, nchar), p=archived_prior)
        equal('truth_seed_replay', truth, a['full_pattern_ids'], atol=0)
        rng_mask = np.random.default_rng(np.random.SeedSequence([2026100401, ordinal, 1]))
        masks = np.zeros((100, nchar), int)
        anchors = np.full(100, -1)
        for r in range(100):
            if scenario['mask_mode'] == 'clustered':
                anchors[r] = anchor = int(rng_mask.integers(n))
                selected = sorted(range(n), key=lambda tip: (tree['tip_edge_distances'][anchor][tip], tip))[:k]
                masks[r] = sum(2**tip for tip in selected)
            else:
                for c in range(nchar):
                    masks[r, c] = sum(2**int(tip) for tip in rng_mask.choice(n, k, replace=False))
        equal('mask_seed_replay', masks, a['missing_mask_bits'], atol=0)
        equal('cluster_anchor_replay', anchors, a['cluster_anchors'], atol=0)
        observed_ids = truth & (((1 << n) - 1) ^ masks)
        equal('observed_state_bits', observed_ids, a['observed_state_bits'], atol=0)
        local = rows[rows.scenario_id == sid].set_index(['replicate', 'method'])
        rng_draw = np.random.default_rng(np.random.SeedSequence([2026100402, ordinal, 2]))
        cache = {}
        for r in range(100):
            uniforms = rng_draw.random((64, nchar)) if n == 6 else None
            pmfs = [[], [], []]
            source_L = source_M = true_D = hard_D = 0
            realized_brier, expected_brier, ent, maxp = [np.zeros(3) for _ in range(4)]
            correct = expected_correct = total_correlation = 0.
            replay_draw_D = np.zeros((3,64), int)
            for c in range(nchar):
                cachekey = (int(masks[r,c]), int(observed_ids[r,c]))
                if cachekey not in cache:
                    observed = patterns[observed_ids[r,c]].astype(np.int16)
                    observed[((masks[r,c] >> np.arange(n)) & 1).astype(bool)] = -1
                    item = conditions(patterns, prior, costs, observed)
                    item['pattern_ids'] = item['support'] @ (2 ** np.arange(n))
                    item['missing_tips'] = np.flatnonzero(observed < 0)
                    p1 = item['marginal_one']
                    sharp = p1 ** 2 / (p1 ** 2 + (1-p1) ** 2)
                    item['forecasts'] = np.stack([p1, p1, sharp])
                    item['pmfs'] = [character_pmf(item['D'], item[law]) for law in ('oracle','product','power2')]
                    item['hard_id'] = int(item['hard'] @ (2 ** np.arange(n)))
                    item['tc'] = float(entropy(p1).sum() + (item['oracle'] * np.log2(item['oracle'])).sum())
                    cache[cachekey] = item
                item = cache[cachekey]
                source_L += item['partial_cost']; source_M += item['partial_minimum']
                actual_id = truth[r,c]
                true_D += int(costs[actual_id] - full_minima[actual_id] - item['partial_cost'] + item['partial_minimum'])
                hard_D += item['hard_D']
                equal('map_pattern_ids', np.array([item['hard_id']]*3), a['map_pattern_ids'][r,:,c], atol=0)
                equal('missing_tip_indices', item['missing_tips'], a['missing_tip_indices'][r,c], atol=0)
                equal('oracle_marginals', item['marginal_one'], a['oracle_missing_p1'][r,c])
                equal('sharpened_marginals', item['forecasts'][2], a['sharpened_missing_p1'][r,c])
                actual = patterns[actual_id, item['missing_tips']]
                f = item['forecasts']; p1 = item['marginal_one']
                realized_brier += 2 * ((f-actual)**2).sum(axis=1)
                expected_brier += 2 * (p1*(1-f)**2+(1-p1)*f**2).sum(axis=1)
                ent += entropy(f).sum(axis=1)
                maxp += np.maximum(f,1-f).sum(axis=1)
                chosen = p1 > .5 + 1e-12
                correct += (chosen == actual).sum()
                expected_correct += np.where(chosen,p1,1-p1).sum()
                total_correlation += item['tc']
                for m, law in enumerate(('oracle','product','power2')):
                    pmfs[m].append(item['pmfs'][m])
                    if n == 6:
                        cdf = np.cumsum(item[law]); cdf[-1] = 1.
                        selected = np.searchsorted(cdf,uniforms[:,c],side='right')
                        equal('propagation_seed_replay', item['pattern_ids'][selected], a['draw_pattern_ids'][r,m,:,c], atol=0)
                        replay_draw_D[m] += item['D'][selected]
            equal('source_L', source_L, a['source_L'][r], atol=0)
            equal('source_M', source_M, a['source_M'][r], atol=0)
            equal('true_D', true_D, a['true_D'][r], atol=0)
            equal('hard_D', hard_D, a['marginal_map_D'][r], atol=0)
            reconstructed = [convolve_pmfs(p) for p in pmfs]
            oracle = reconstructed[0]
            for m, method in enumerate(METHODS):
                p = reconstructed[m]
                padded = np.pad(p,(0,a['matrix_D_pmfs'].shape[-1]-len(p)))
                equal('matrix_D_pmfs', padded, a['matrix_D_pmfs'][r,m])
                s = summarize_distribution(p)
                equal('matrix_probability_normalization', s['mass'], 1.)
                axis = np.arange(len(p))
                opad = np.pad(oracle,(0,len(p)-len(oracle))) if len(p)>=len(oracle) else oracle
                ppad = np.pad(p,(0,len(oracle)-len(p))) if len(p)<len(oracle) else p
                expected = {
                    'target_cells': k*nchar, 'true_D': true_D, 'marginal_map_D': hard_D,
                    'predicted_D_mean': s['mean'], 'oracle_D_mean': summarize_distribution(oracle)['mean'],
                    'predicted_D_variance': float(p @ (axis-s['mean'])**2),
                    'mean_minus_oracle': s['mean']-summarize_distribution(oracle)['mean'],
                    'mean_minus_truth': s['mean']-true_D,
                    'D_q025': s['lower'], 'D_q975': s['upper'], 'interval_width':s['upper']-s['lower'],
                    'realized_covered': int(s['lower']<=true_D<=s['upper']),
                    'oracle_conditional_interval_mass': oracle_interval_mass(oracle,s['lower'],s['upper']),
                    'own_interval_mass': float(p[s['lower']:s['upper']+1].sum()),
                    'D_total_variation_from_oracle':float(abs(ppad-opad).sum()/2),
                    'realized_brier':realized_brier[m]/(k*nchar),
                    'oracle_expected_brier':expected_brier[m]/(k*nchar),
                    'realized_accuracy':correct/(k*nchar),'oracle_expected_accuracy':expected_correct/(k*nchar),
                    'forecast_mean_max_probability':maxp[m]/(k*nchar),'mean_marginal_entropy_bits':ent[m]/(k*nchar),
                    'oracle_total_correlation_bits_per_character':total_correlation/nchar}
                for field,value in expected.items():
                    equal('matrix_csv_'+field,value,float(local.loc[(r,method),field]))
            if n == 6:
                equal('draw_D',replay_draw_D,a['draw_D'][r],atol=0)
                draw_count += 3*64
        conditional_count += len(cache)
        print(sid,'independent reconstruction PASS',flush=True)
    # Recompute all means and standard errors from per-matrix rows, with no
    # pooling of completion draws or tree ties as independent matrices.
    scenario_columns = set(design['scenarios'][0])
    numeric = [key for key in rows if key not in scenario_columns and key not in ('replicate','matrix_id','method')]
    for _,record in summary.iterrows():
        part = rows[(rows.scenario_id==record.scenario_id)&(rows.method==record.method)]
        assert len(part)==100
        for field in numeric:
            equal('summary_mean_'+field,part[field].mean(),record[field+'_mean'])
            equal('summary_se_'+field,part[field].std(ddof=1)/10,record[field+'_se'])
        successes = int(part.realized_covered.sum())
        equal('coverage_count',successes,record.coverage_count,atol=0)
        z=1.959963984540054; rate=successes/100; den=1+z*z/100
        center=(rate+z*z/200)/den; half=z*math.sqrt(rate*(1-rate)/100+z*z/40000)/den
        equal('coverage_wilson_low',center-half,record.coverage_wilson_low)
        equal('coverage_wilson_high',center+half,record.coverage_wilson_high)
    freeze=json.loads((ROOT/'DESIGN_FREEZE.json').read_text())
    assert hashlib.sha256((ROOT/'ANALYSIS_PLAN.txt').read_bytes()).hexdigest()==freeze['plan_sha256']
    assert production_check['frozen_design_sha256']==hashlib.sha256((ROOT/'DESIGN_FREEZE.json').read_bytes()).hexdigest()
    result={'status':'PASS','independent_algorithm':'complete internal-state enumeration; compatible-tip-pattern minimization; explicit shift-and-add convolution',
            'matrix_count':4800,'matrix_method_pmfs_reconstructed':14400,'scenario_method_summaries_checked':144,
            'generating_models_checked':12,'conditional_states_reconstructed':conditional_count,
            'saved_completion_draws_replayed':draw_count,'truth_and_mask_rng_replay':'exact',
            'freeze_metadata':{'frozen_design_sha256_refers_to':'DESIGN_FREEZE.json','plan_sha256_refers_to':'ANALYSIS_PLAN.txt','both_verified':True},
            'all_sampled_observed_states_preserved':True,'maximum_absolute_errors':MAX_ERROR,
            'runtime_seconds':time.time()-started,
            'input_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in input_files+[RESULTS/'matrix_results.csv',RESULTS/'scenario_summary.csv',RESULTS/'pattern_registry.npz',RESULTS/'trees.json']}}
    (HERE/'SIMULATION_INDEPENDENT_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('input_sha256','maximum_absolute_errors')},indent=2))


if __name__=='__main__':
    main()
