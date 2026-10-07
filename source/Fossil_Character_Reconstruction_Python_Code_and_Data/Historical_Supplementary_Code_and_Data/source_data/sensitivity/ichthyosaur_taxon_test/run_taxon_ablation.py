"""Frozen paired target-block ablation, with unchanged round 10 native scorer."""
from pathlib import Path
import argparse, datetime, hashlib, importlib.util, json, sys, time
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
R12 = HERE.parent
ROOT = R12/'restored/analysis'
OLD = ROOT/'new_project_search/round10/ichthyosaur_propagation'
PANEL = OLD.parent/'ichthyosaur_reaudit/score_inputs'
TOOLS = R12/'software/tools'
NATIVE = OLD/'native_all_ordered.py'
spec = importlib.util.spec_from_file_location('round10_native_unchanged', NATIVE)
native = importlib.util.module_from_spec(spec); spec.loader.exec_module(native)
native.TOOLS = TOOLS; native.PANEL = PANEL
SCHEMES = ['mode','knn5','logistic','extra_trees','nj_mk','nj_mk_joint']
HARD = ['source','later_codes','mode','knn5','logistic','extra_trees','nj_mk']
ORDERS = {'source_order':{99,203,204,218}, 'prose_order':{99,100,203,204,218,219}}
B, BATCH = 1000, 50
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
now = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()


def panel():
    names,raw,ordered,cells,trees=native.load_panel()
    saved=pd.read_csv(OLD/'cells.csv',keep_default_na=False,dtype={'true':str})
    assert saved[['character','taxon','true']].equals(cells[['character','taxon','true']])
    chars=sorted(cells.character.unique().tolist())
    assert len(chars)==128
    assignments=np.load(OLD/'hard_assignments.npy',allow_pickle=False)
    assert assignments.shape==(7,192) and np.all(assignments[0]=='?')
    assert np.array_equal(assignments[1],cells.true.to_numpy())
    draws={s:np.load(OLD/f'{s}_draws.npy',allow_pickle=False) for s in SCHEMES}
    assert all(a.shape==(1000,192) and np.isin(a,list('0123')).all() for a in draws.values())
    counts=cells.groupby('taxon',as_index=False).size().rename(columns={'size':'target_count'})
    counts=counts.sort_values(['target_count','taxon'],ascending=[False,True]).reset_index(drop=True)
    counts['rank']=np.arange(1,len(counts)+1); selected=counts.head(3).taxon.tolist()
    assert counts.head(3).target_count.tolist()==[56,33,22]
    assert not cells.loc[cells.taxon.isin(selected),'character'].isin([100,219]).any()
    base={}
    for ordering in ORDERS:
        folder=OLD if ordering=='source_order' else OLD/'prose_order'
        hz=np.load(folder/'hard_scores.npz',allow_pickle=False)
        full=np.load(folder/'hard_full_scores.npz',allow_pickle=False)
        assert hz['characters'].tolist()==chars and hz['treatments'].tolist()==HARD
        assert full['characters'].tolist()==list(range(1,288))
        bs={s:{k:np.load(folder/f'{s}_scores.npz',allow_pickle=False)[k] for k in ['steps','minima']} for s in SCHEMES}
        assert all(a['steps'].shape==(5,1000,128) and a['minima'].shape==(1000,128) for a in bs.values())
        base[ordering]=dict(hs=hz['steps'],hm=hz['minima'],fulls=full['steps'],fullm=full['minima'],models=bs,summary=pd.read_csv(folder/'summary.csv'))
    return dict(names=names,raw=raw,cells=cells,chars=chars,trees=trees,assignments=assignments,draws=draws,counts=counts,selected=selected,base=base)


def freeze():
    assert not (HERE/'FREEZE.json').exists() and not (HERE/'cases').exists()
    p=panel();p['counts'].to_csv(HERE/'all_taxon_target_counts.csv',index=False)
    p['cells'][p['cells'].taxon.isin(p['selected'])].to_csv(HERE/'selected_cells.csv',index=False)
    paths=[Path(__file__),HERE/'PROTOCOL.txt',HERE/'all_taxon_target_counts.csv',HERE/'selected_cells.csv',
           NATIVE,OLD/'run_probability.py',OLD/'run_prose_order_sensitivity.py',OLD/'FREEZE.json',OLD/'PROSE_ORDER_FREEZE.json',
           PANEL/'panel.npz',PANEL/'target_cells.csv',PANEL/'reference_trees.nex',PANEL/'PRESCORE_GATE.json',
           OLD/'hard_assignments.npy',OLD/'cells.csv',TOOLS/'paup4a169_ubuntu64',
           TOOLS/'runtime/usr/lib/x86_64-linux-gnu/libgfortran.so.4',TOOLS/'runtime/usr/lib/x86_64-linux-gnu/libpython2.7.so.1.0']
    paths += [OLD/f'{s}_draws.npy' for s in SCHEMES]
    for folder in [OLD,OLD/'prose_order']:
        paths += [folder/f for f in ['hard_scores.npz','hard_full_scores.npz','summary.csv','RUN_MANIFEST.json']]
        paths += [folder/f'{s}_scores.npz' for s in SCHEMES]
    gate=dict(created_utc=now(),analysis_type='historical sensitivity analysis, not prospective preregistration',prior_round10_results_known=True,
        round12_ablation_results_present=False,selection_rule='top three counts descending, literal taxon label ascending for ties',
        selected=p['counts'].head(3).to_dict('records'),all_taxa_retained=101,full_character_background=287,targets=192,
        target_characters=128,replicates=1000,models_refitted=0,new_draws=0,trees_searched=0,schemes=SCHEMES,orders={k:sorted(v) for k,v in ORDERS.items()},
        changed_order_characters_disjoint_from_selected_targets=True,
        files=[dict(relative_to_round12=str(x.resolve().relative_to(R12.resolve())),sha256=sha(x),bytes=x.stat().st_size) for x in paths])
    (HERE/'FREEZE.json').write_text(json.dumps(gate,indent=2)); print(json.dumps({k:v for k,v in gate.items() if k!='files'},indent=2))


def gatecheck():
    gate=json.loads((HERE/'FREEZE.json').read_text())
    for r in gate['files']:
        path=R12/r['relative_to_round12']
        assert sha(path)==r['sha256'],str(path)
    return gate


def score(p,chars,draws,ordering,dest):
    return native.score(p['raw'],p['names'],chars,p['cells'],draws,ORDERS[ordering],dest,
                        tree_file=PANEL/'reference_trees.nex',tree_names=p['trees'])


def summary(p,hs,hm,steps,mins,scheme):
    ds=steps.sum(2)-hs[:,0].sum(1)[:,None];dm=mins.sum(1)-hm[0].sum();ex=ds-dm[None,:]
    obs=hs[:,1].sum(1)-hs[:,0].sum(1)-(hm[1].sum()-hm[0].sum())
    k=HARD.index('nj_mk' if scheme=='nj_mk_joint' else scheme)
    point=hs[:,k].sum(1)-hs[:,0].sum(1)-(hm[k].sum()-hm[0].sum())
    rows=[]
    for i,tr in enumerate(p['trees']):
        v=ex[i];q=np.quantile(v,[.025,.975]);res=float(v.mean()-obs[i])
        rows.append(dict(model=scheme,reference=tr,replicates=B,reference_excess_steps=int(obs[i]),point_excess_steps=int(point[i]),
            sim_mean_excess_steps=float(v.mean()),sim_sd_excess_steps=float(v.std(ddof=1)),sim_q025=float(q[0]),sim_q975=float(q[1]),
            fraction_at_least_reference=float((v>=obs[i]).mean()),fraction_at_most_reference=float((v<=obs[i]).mean()),
            reference_in_central95=bool(q[0]<=obs[i]<=q[1]),mean_added_steps=float(ds[i].mean()),mean_added_minimum_steps=float(dm.mean()),
            mean_minus_reference=res,discrepancy_sign=int(np.sign(res)),reference_position='below' if obs[i]<q[0] else 'above' if obs[i]>q[1] else 'inside'))
    return rows,ds,dm,ex


def check_summary(rows,old):
    new=pd.DataFrame(rows).set_index(['model','reference']).sort_index()
    old=old.set_index(['model','reference']).sort_index()
    assert new.index.equals(old.index)
    for c in old.columns:
        if old[c].dtype==bool:assert new[c].equals(old[c]),c
        elif np.issubdtype(old[c].dtype,np.number):assert np.allclose(new[c],old[c],rtol=0,atol=1e-12),c
        else:assert new[c].equals(old[c]),c


def validate():
    gatecheck();p=panel();assert not (HERE/'VALIDATION.json').exists() and not (HERE/'cases').exists()
    started=time.monotonic();checks=[];ix=np.array(p['chars'])-1
    for ordering in ORDERS:
        b=p['base'][ordering]
        s,m=score(p,list(range(1,288)),p['assignments'],ordering,HERE/'validation'/ordering/'hard_full287')
        assert np.array_equal(s,b['fulls']) and np.array_equal(m,b['fullm'])
        assert np.array_equal(s[:,:,ix],b['hs']) and np.array_equal(m[:,ix],b['hm'])
        nontarget=np.array([x for x in range(287) if x not in ix])
        assert np.all(s[:,:,nontarget]==s[:,0:1,nontarget]) and np.all(m[:,nontarget]==m[0:1,nontarget])
        checks.append(dict(ordering=ordering,full287_hard_treatments=7,all_scores_and_minima_exact=True))
    base=p['base']['source_order'];prose=p['base']['prose_order'];i100=p['chars'].index(100)
    unchanged=np.array([i for i,c in enumerate(p['chars']) if c!=100])
    allsum={'source_order':[],'prose_order':[]}
    for scheme in SCHEMES:
        for start in range(0,B,BATCH):
            stop=start+BATCH
            s,m=score(p,p['chars'],p['draws'][scheme][start:stop],'source_order',HERE/'validation/source_order'/scheme/f'batch_{start:04d}_{stop:04d}')
            assert np.array_equal(s,base['models'][scheme]['steps'][:,start:stop])
            assert np.array_equal(m,base['models'][scheme]['minima'][start:stop])
            ps,pm=score(p,[100],p['draws'][scheme][start:stop],'prose_order',HERE/'validation/prose_order'/scheme/f'character100_batch_{start:04d}_{stop:04d}')
            assert np.array_equal(ps[:,:,0],prose['models'][scheme]['steps'][:,start:stop,i100])
            assert np.array_equal(pm[:,0],prose['models'][scheme]['minima'][start:stop,i100])
        assert np.array_equal(base['models'][scheme]['steps'][:,:,unchanged],prose['models'][scheme]['steps'][:,:,unchanged])
        assert np.array_equal(base['models'][scheme]['minima'][:,unchanged],prose['models'][scheme]['minima'][:,unchanged])
        for order in ORDERS:
            b=p['base'][order];z=b['models'][scheme];allsum[order]+=summary(p,b['hs'],b['hm'],z['steps'],z['minima'],scheme)[0]
        checks.append(dict(scheme=scheme,replicates=1000,source_all128_target_scores_and_minima_exact=True,
            prose_character100_scores_and_minima_exact=True,prose_other127_characters_bitwise_identical=True))
        print(json.dumps(dict(validation_scheme=scheme,passed=True,elapsed_seconds=round(time.monotonic()-started,1))),flush=True)
    for ordering in ORDERS:check_summary(allsum[ordering],p['base'][ordering]['summary'])
    gatecheck()
    report=dict(completed_utc=now(),seconds=time.monotonic()-started,all_baselines_passed=True,
        baseline_summary_rows_exact=60,baseline_draw_tree_totals_exact=60000,
        source_native_per_character_steps_checked=5*1000*128*6,source_native_minima_checked=1000*128*6,
        prose_changed_target_per_character_steps_checked=5*1000*6,prose_changed_target_minima_checked=1000*6,
        full_hard_steps_checked=5*7*287*2,full_hard_minima_checked=7*287*2,checks=checks,freeze_sha256=sha(HERE/'FREEZE.json'))
    (HERE/'VALIDATION.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


def replace(a,new,ci):
    out=a.copy();out[...,ci]=new;other=[i for i in range(a.shape[-1]) if i not in ci]
    assert np.array_equal(out[...,other],a[...,other]);return out


def run():
    gate=gatecheck();assert json.loads((HERE/'VALIDATION.json').read_text())['all_baselines_passed']
    assert not (HERE/'cases').exists();p=panel();allrows=[];allreps=[];inventory=[];started=time.monotonic()
    for taxon in p['selected']:
        focal=np.where(p['cells'].taxon.eq(taxon))[0];chars=sorted(p['cells'].iloc[focal].character.unique().tolist())
        ci=[p['chars'].index(c) for c in chars];assert not set(chars)&{100,219}
        hard=p['assignments'].copy();hard[:,focal]='?'
        other=[i for i in range(192) if i not in focal]
        assert np.array_equal(hard[:,other],p['assignments'][:,other])
        case=HERE/'cases'/taxon;case.mkdir(parents=True)
        p['cells'].iloc[focal].to_csv(case/'restored_target_cells.csv',index=False)
        hs,hm=score(p,chars,hard,'source_order',case/'native_hard')
        savedhard={}
        for ordering in ORDERS:
            out=case/ordering;out.mkdir();base=p['base'][ordering]
            fullhs=replace(base['hs'],hs,ci);fullhm=replace(base['hm'],hm,ci)
            assert np.array_equal(fullhs[:,0],base['hs'][:,0]) and np.array_equal(fullhm[0],base['hm'][0])
            assert np.all(fullhs<=base['hs']) and np.all(fullhm<=base['hm'])
            np.savez_compressed(out/'hard_scores.npz',steps=fullhs,minima=fullhm,characters=p['chars'],trees=p['trees'],treatments=HARD)
            fulls=replace(base['fulls'],hs,np.array(chars)-1);fullm=replace(base['fullm'],hm,np.array(chars)-1)
            np.savez_compressed(out/'hard_full_scores.npz',steps=fulls,minima=fullm,characters=np.arange(1,288),trees=p['trees'],treatments=HARD)
            assert np.array_equal(fulls[:,:,np.array(p['chars'])-1],fullhs)
            savedhard[ordering]=(fullhs,fullhm,fulls,fullm)
        for scheme in SCHEMES:
            original=p['draws'][scheme];draw=original.copy();draw[:,focal]='?'
            assert np.array_equal(draw[:,other],original[:,other]) and np.all(draw[:,focal]=='?')
            ss=[];mm=[]
            for start in range(0,B,BATCH):
                stop=start+BATCH;s,m=score(p,chars,draw[start:stop],'source_order',case/'native'/scheme/f'batch_{start:04d}_{stop:04d}')
                ss.append(s);mm.append(m)
            affected=np.concatenate(ss,axis=1);minimum=np.concatenate(mm,axis=0)
            for ordering in ORDERS:
                base=p['base'][ordering];old=base['models'][scheme]
                steps=replace(old['steps'],affected,ci);mins=replace(old['minima'],minimum,ci)
                assert np.all(steps<=old['steps']) and np.all(mins<=old['minima'])
                out=case/ordering/f'{scheme}_scores.npz'
                np.savez_compressed(out,steps=steps,minima=mins,characters=p['chars'],trees=p['trees'],affected_character_indices=ci)
                hss,hmm,fulls,fullm=savedhard[ordering]
                rows,ds,dm,ex=summary(p,hss,hmm,steps,mins,scheme)
                orig,ods,odm,oex=summary(p,base['hs'],base['hm'],old['steps'],old['minima'],scheme)
                # Full 287-background difference equals target-character calculation exactly.
                non=np.array([c-1 for c in range(1,288) if c not in p['chars']])
                background_s=base['fulls'][:,0,non].sum(1);background_m=base['fullm'][0,non].sum()
                assert np.array_equal(steps.sum(2)+background_s[:,None]-fulls[:,0].sum(1)[:,None],ds)
                assert np.array_equal(mins.sum(1)+background_m-fullm[0].sum(),dm)
                for ti,(r,o) in enumerate(zip(rows,orig)):
                    r.update(ordering=ordering,focal_taxon=taxon,restored_target_cells=len(focal),remaining_target_cells=192-len(focal),affected_characters=len(chars))
                    for key in ['reference_excess_steps','point_excess_steps','sim_mean_excess_steps','sim_q025','sim_q975','mean_minus_reference','reference_in_central95','reference_position','discrepancy_sign']:
                        r['original_'+key]=o[key]
                    for key in ['reference_excess_steps','point_excess_steps','sim_mean_excess_steps','mean_minus_reference']:
                        r['delta_'+key]=r[key]-o[key]
                    r['interval_status_changed']=r['reference_in_central95']!=o['reference_in_central95']
                    r['mean_discrepancy_sign_changed']=r['discrepancy_sign']!=o['discrepancy_sign'];allrows.append(r)
                    allreps.extend(dict(ordering=ordering,focal_taxon=taxon,model=scheme,reference=p['trees'][ti],replicate=i,
                        added_steps=int(ds[ti,i]),added_minimum_steps=int(dm[i]),added_excess_steps=int(ex[ti,i]),
                        original_added_excess_steps=int(oex[ti,i]),paired_delta_excess_steps=int(ex[ti,i]-oex[ti,i])) for i in range(B))
                inventory.append(dict(ordering=ordering,focal_taxon=taxon,scheme=scheme,score_sha256=sha(out),replicates=B,
                    affected_chars=len(chars),untouched_slices_bitwise_equal=True,all_taxa_retained=True,full_background_identity_verified=True))
            pd.DataFrame(allrows).to_csv(HERE/'summary.csv',index=False)
            pd.DataFrame(allreps).to_csv(HERE/'replicate_totals.csv',index=False)
            (HERE/'progress_manifest.json').write_text(json.dumps(inventory,indent=2))
            print(json.dumps(dict(taxon=taxon,scheme=scheme,completed_replicates=B,both_orderings=True,elapsed_seconds=round(time.monotonic()-started,1))),flush=True)
    assert len(allrows)==180 and len(allreps)==180000 and len(inventory)==36
    gatecheck();report=dict(completed_utc=now(),seconds=time.monotonic()-started,summary_rows=180,replicate_rows=180000,
        native_ablation_draw_blocks=3*6*20,hard_native_blocks=3,schemes=6,reference_trees=5,cases=3,orderings=2,
        models_refitted=0,new_draws=0,tree_searches=0,native_affected_scores_reused_between_orders_because_extra_ordered_characters_not_ablated=True,
        freeze_sha256=sha(HERE/'FREEZE.json'),validation_sha256=sha(HERE/'VALIDATION.json'),inventory=inventory)
    (HERE/'RUN_MANIFEST.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='inventory'},indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('operation',choices=['freeze','validate','score']);args=ap.parse_args()
    {'freeze':freeze,'validate':validate,'score':run}[args.operation]()
