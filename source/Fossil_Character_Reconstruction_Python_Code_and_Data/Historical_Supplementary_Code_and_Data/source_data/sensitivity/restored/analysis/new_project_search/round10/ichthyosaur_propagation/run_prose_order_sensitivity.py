"""Prespecified addition of ordered characters 100 and 219, without new draws."""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import native_all_ordered as native

HERE = Path(__file__).resolve().parent
OUT = HERE / 'prose_order'


def run():
    native.check_freeze()
    prespec = json.loads((HERE / 'PROSE_ORDER_FREEZE.json').read_text())
    for item in prespec['inputs']:
        assert native.sha(item['path']) == item['sha256']
    assert not OUT.exists(), 'Preserve existing results'
    OUT.mkdir()
    dependencies = [Path(__file__), HERE/'run_probability.py', HERE/'native_all_ordered.py', HERE/'PROSE_ORDER_FREEZE.json',
                    HERE/'RUN_MANIFEST.json', HERE/'hard_scores.npz', HERE/'hard_full_scores.npz']
    dependencies += [HERE/f'{s}_scores.npz' for s in native.SCHEMES]
    freeze = {'created_utc': native.now(), 'prespecified_before_main_results': True, 'main_qualitative_summary_received_from_parent_before_parser_fix': True,
              'design_prespecified_in': str(HERE/'PROSE_ORDER_FREEZE.json'),
              'input_files': [{'path': str(p), 'sha256': native.sha(p)} for p in dependencies]}
    (OUT/'EXECUTION_FREEZE.json').write_text(json.dumps(freeze, indent=2))
    names, raw, ordered, cells, trees = native.load_panel()
    alternative = ordered | {100, 219}
    chars = sorted(cells.character.unique().tolist()); ix = chars.index(100)
    assert 219 not in chars
    target = cells[cells.character.eq(100)]
    assert len(target) == 1 and target.iloc[0].taxon == 'Platypterygius_americanus'
    assert str(target.iloc[0].true) == '2'
    hard = np.load(HERE/'hard_assignments.npy', allow_pickle=False)
    original_hard = np.load(HERE/'hard_scores.npz', allow_pickle=False)
    hs, hm = original_hard['steps'].copy(), original_hard['minima'].copy()
    # Verify character-slice native calls against already frozen original arrays.
    old_s, old_m = native.score(raw,names,[100],cells,hard,ordered,OUT/'validation'/'old_order_hard100')
    assert np.array_equal(old_s[:,:,0], hs[:,:,ix]) and np.array_equal(old_m[:,0],hm[:,ix])
    new_s, new_m = native.score(raw,names,[100],cells,hard,alternative,OUT/'native'/'new_order_hard100')
    hs[:,:,ix] = new_s[:,:,0]; hm[:,ix] = new_m[:,0]
    np.savez_compressed(OUT/'hard_scores.npz',steps=hs,minima=hm,characters=chars,trees=trees,treatments=native.TREATMENTS)
    # Untargeted 219 affects only the fixed full-matrix background.
    s219, m219 = native.score(raw,names,[219],cells,None,ordered,OUT/'validation'/'old_order_source219')
    n219, nm219 = native.score(raw,names,[219],cells,None,alternative,OUT/'native'/'new_order_source219')
    full = np.load(HERE/'hard_full_scores.npz',allow_pickle=False)
    assert np.array_equal(s219[:,0,0],full['steps'][:,0,218])
    assert int(m219[0,0]) == int(full['minima'][0,218])
    fulls, fullm = full['steps'].copy(), full['minima'].copy()
    fulls[:,:,99] = new_s[:,:,0]; fullm[:,99] = new_m[:,0]
    fulls[:,:,218] = n219[:,0,0][:,None]; fullm[:,218] = nm219[0,0]
    np.savez_compressed(OUT/'hard_full_scores.npz',steps=fulls,minima=fullm,characters=np.arange(1,288),trees=trees,treatments=native.TREATMENTS)
    rows_hard=[]
    for ti,tree in enumerate(trees):
        for k,treatment in enumerate(native.TREATMENTS):
            ds=int(fulls[ti,k].sum()-fulls[ti,0].sum()); dm=int(fullm[k].sum()-fullm[0].sum())
            assert ds == int(hs[ti,k].sum()-hs[ti,0].sum())
            assert dm == int(hm[k].sum()-hm[0].sum())
            rows_hard.append({'reference':tree,'treatment':treatment,'full_tree_steps':int(fulls[ti,k].sum()),
              'full_minimum_steps':int(fullm[k].sum()),'added_steps':ds,'added_minimum_steps':dm,'added_excess_steps':ds-dm})
    pd.DataFrame(rows_hard).to_csv(OUT/'fixed_tree_totals.csv',index=False)
    baseline_s,baseline_m=hs[:,0,:],hm[0]
    observed=hs[:,1,:].sum(axis=1)-baseline_s.sum(axis=1)-(hm[1].sum()-baseline_m.sum())
    rows=[];replicates=[];checks={};started=time.monotonic()
    for scheme in native.SCHEMES:
        draws=np.load(HERE/f'{scheme}_draws.npy',allow_pickle=False)
        src=np.load(HERE/f'{scheme}_scores.npz',allow_pickle=False)
        scores,mins=src['steps'].copy(),src['minima'].copy()
        vs,vm=native.score(raw,names,[100],cells,draws[:1],ordered,OUT/'validation'/f'{scheme}_first_old_order100')
        assert np.array_equal(vs[:,0,0],scores[:,0,ix]) and int(vm[0,0])==int(mins[0,ix])
        for start in range(0,native.B,native.BATCH):
            stop=min(native.B,start+native.BATCH)
            ss,mm=native.score(raw,names,[100],cells,draws[start:stop],alternative,
                              OUT/'native'/scheme/f'batch_{start:04d}_{stop:04d}')
            scores[:,start:stop,ix]=ss[:,:,0];mins[start:stop,ix]=mm[:,0]
        unchanged=np.arange(len(chars))!=ix
        assert np.array_equal(scores[:,:,unchanged],src['steps'][:,:,unchanged])
        assert np.array_equal(mins[:,unchanged],src['minima'][:,unchanged])
        np.savez_compressed(OUT/f'{scheme}_scores.npz',steps=scores,minima=mins,characters=chars,trees=trees)
        ds=scores.sum(axis=2)-baseline_s.sum(axis=1)[:,None];dm=mins.sum(axis=1)-baseline_m.sum()
        extra=ds-dm[None,:]
        model='nj_mk' if scheme=='nj_mk_joint' else scheme;k=native.TREATMENTS.index(model)
        point=hs[:,k,:].sum(axis=1)-baseline_s.sum(axis=1)-(hm[k].sum()-baseline_m.sum())
        for ti,tree in enumerate(trees):
            v=extra[ti];q=np.quantile(v,[.025,.975])
            rows.append({'model':scheme,'reference':tree,'replicates':native.B,'reference_excess_steps':int(observed[ti]),
                'point_excess_steps':int(point[ti]),'sim_mean_excess_steps':float(v.mean()),'sim_sd_excess_steps':float(v.std(ddof=1)),
                'sim_q025':float(q[0]),'sim_q975':float(q[1]),'fraction_at_least_reference':float((v>=observed[ti]).mean()),
                'fraction_at_most_reference':float((v<=observed[ti]).mean()),'reference_in_central95':bool(q[0]<=observed[ti]<=q[1]),
                'mean_added_steps':float(ds[ti].mean()),'mean_added_minimum_steps':float(dm.mean())})
            replicates.extend({'model':scheme,'reference':tree,'replicate':b,'added_steps':int(ds[ti,b]),
                'added_minimum_steps':int(dm[b]),'added_excess_steps':int(v[b])} for b in range(native.B))
        checks[scheme]={'all_other_character_scores_unchanged':True,'all_other_character_minima_unchanged':True,
                       'original_first_draw_char100_equal':True,'scores_sha256':native.sha(OUT/f'{scheme}_scores.npz')}
        print(json.dumps({'scheme':scheme,'completed':native.B}),flush=True)
    pd.DataFrame(rows).to_csv(OUT/'summary.csv',index=False)
    pd.DataFrame(replicates).to_csv(OUT/'replicate_totals.csv',index=False)
    primary=pd.read_csv(HERE/'summary.csv')
    comparison=primary.merge(pd.DataFrame(rows),on=['model','reference'],suffixes=('_main','_prose'))
    comparison['central95_status_changed']=comparison.reference_in_central95_main.ne(comparison.reference_in_central95_prose)
    comparison.to_csv(OUT/'comparison_to_primary.csv',index=False)
    report={'completed_utc':native.now(),'seconds':time.monotonic()-started,'no_new_models_or_draws':True,
            'ordered_characters':sorted(alternative),'only_changed_target_character':100,'summary_rows':len(rows),
            'changed_central95_classifications':int(comparison.central95_status_changed.sum()),
            'char219_old_steps':s219[:,0,0].tolist(),'char219_new_steps':n219[:,0,0].tolist(),
            'char219_old_minimum':int(m219[0,0]),'char219_new_minimum':int(nm219[0,0]),
            'char219_fixed_background_verified_to_cancel':True,'checks':checks}
    for item in freeze['input_files']:
        assert native.sha(item['path'])==item['sha256']
    native.check_freeze()
    (OUT/'RUN_MANIFEST.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    run()
