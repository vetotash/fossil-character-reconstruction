"""Extend frozen targets with a source-only tree model and comparable probabilities."""
from pathlib import Path
import sys,ast,re,json,time,os,hashlib
import numpy as np,pandas as pd
from concurrent.futures import ProcessPoolExecutor
from mk_baseline import known,distance_totals,excluding_distance,neighbor_join,predict_mk
P=Path(__file__).resolve().parent;ROOT=P.parent
sys.path.insert(0,str(ROOT))
import dinosaur_matched_pilot as dp

def dataset(name):
    # Portable, checked snapshots are opt-in. Original source loaders remain
    # available for re-curation; the snapshots contain only the old predictors.
    if os.environ.get('FOSSIL_USE_SNAPSHOT')=='1':
        snap=P/'inputs'/f'{name}.npz';target=P/'inputs'/f'{name}_frozen_targets.csv'
        manifests=json.loads((P/'inputs/snapshot_manifest.json').read_text())
        record=next(r for r in manifests if r['lineage']==name)
        assert hashlib.sha256(snap.read_bytes()).hexdigest()==record['matrix_sha256']
        assert hashlib.sha256(target.read_bytes()).hexdigest()==record['frozen_targets_sha256']
        z=np.load(snap)
        return z['old_names'].tolist(),z['A'],pd.read_csv(target,dtype={'true':str,'pred':str}),{int(c):j for j,c in enumerate(z['old_char_numbers'])}
    if name=='dinosaurs':
        names,a=dp.parse(ROOT/'matrices/Baron_etal_2017a.nex');names=dp.canonical(names)
        frozen=pd.read_csv(ROOT/'dinosaur_matched_predictions.csv',dtype={'true':str,'pred':str})
        charmap={j+1:j for j in range(a.shape[1])}
    elif name=='tetrapods':
        z=np.load(ROOT/'tetrapod_pairs/parsed_matrices.npz');a=z['A'];names=z['old_names'].tolist()
        frozen=pd.read_csv(ROOT/'tetrapod_pairs/pilot_predictions.csv',dtype={'true':str,'pred':str}).rename(columns={'old_char':'character'})
        charmap={j+1:j for j in range(a.shape[1])}
    elif name=='crocodilians':
        ns={'re':re,'np':np}
        tree=ast.parse((ROOT/'mammal_pairs/audit_croc.py').read_text())
        exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='parse_tnt'],type_ignores=[]),'parse_tnt','exec'),ns)
        names,a,chars=ns['parse_tnt'](ROOT/'mammal_pairs/peerj-09-12094-s023.tnt')
        fix={'Tomistoma_dowsoni':'Sutekhsuchus_dowsoni','Moroccosuchus_zennaroi':'Maroccosuchus_zennaroi'}
        names=[fix.get(n,n) for n in names]
        keep=[j for j,c in enumerate(chars) if c not in {80,86,116,216}]
        a=a[:,keep];chars=np.array(chars)[keep];charmap={int(c):j for j,c in enumerate(chars)}
        frozen=pd.read_csv(ROOT/'mammal_pairs/croc_predictions.csv',dtype={'true':str,'pred':str})
    elif name=='gnathostomes':
        z=np.load(P/'independent_expansion/gnathostome_mk_input.npz')
        a=z['A'];names=z['old_names'].tolist();charmap={int(c):j for j,c in enumerate(z['old_char_numbers'])}
        frozen=pd.read_csv(P/'independent_expansion/gnathostome_predictions.csv',dtype={'true':str,'pred':str})
    elif name=='whales':
        z=np.load(P/'cenozoic_expansion/whale_parsed.npz')
        a=z['A'];names=z['old_names'].tolist();charmap={int(c):j for j,c in enumerate(z['old_char_numbers'])}
        frozen=pd.read_csv(P/'cenozoic_expansion/predictions.csv',dtype={'true':str,'pred':str})
        frozen['scenario']=frozen['scenario'].replace({'later':'later_publication'})
    else:raise ValueError(name)
    frozen['scenario']=np.where(frozen.scenario=='later_publication','later_publication','matched_concealment')
    return names,a,frozen,charmap

def run_dataset(name):
    start=time.time();names,a,frozen,charmap=dataset(name)
    outdir=P/name;outdir.mkdir(exist_ok=True)
    lookup={n:i for i,n in enumerate(names)};totals=distance_totals(a)
    targets=frozen[frozen.model=='mode'].copy();out=[];diagnostics=[]
    tree_data=[];reproduced=0
    for number,(character,groups) in enumerate(targets.groupby('character')):
        j=charmap[int(character)];d,zero=excluding_distance(a,j,totals);edges,diag=neighbor_join(d)
        tree_data.append({'character':int(character),'taxa':names,'edges':edges})
        for scenario,g in groups.groupby('scenario'):
            te=np.array([lookup[n] for n in g.taxon]);truth=g.true.to_numpy();mat=a.copy()
            if scenario=='later_publication':
                assert np.all(a[te,j]=='?')
            else:
                assert np.array_equal(a[te,j],truth)
                mat[te,j]='?'
            tr=np.flatnonzero(known(mat[:,j]));assert not set(tr)&set(te)
            methods=dp.predict(mat,j,tr,te)
            classes,prob,fit=predict_mk(edges,len(names),mat[:,j],tr,te)
            methods['nj_mk']=(classes,prob)
            diagnostics.append({'lineage':name,'character':int(character),'scenario':scenario,
                                'n_train':len(tr),'n_test':len(te),'zero_overlap_pairs':zero,**diag,**fit})
            for model,(classes,prob) in methods.items():
                predicted=classes[prob.argmax(1)]
                if model!='nj_mk':
                    previous=frozen[(frozen.character==character)&(frozen.scenario==scenario)&(frozen.model==model)].set_index('taxon').loc[g.taxon]
                    assert np.array_equal(predicted,previous.pred.to_numpy()),(name,character,scenario,model,'Frozen label mismatch')
                    reproduced+=len(te)
                for i,(row,yy,yp,pp) in enumerate(zip(te,truth,predicted,prob)):
                    unseen=yy not in classes
                    ptrue=float(pp[np.flatnonzero(classes==yy)[0]]) if not unseen else 0.
                    out.append({'lineage':name,'scenario':scenario,'character':int(character),'taxon':names[row],
                                'true':yy,'pred':yp,'model':model,'n_train':len(tr),
                                'confidence':float(pp.max()),'p_true':ptrue,'unseen_state':unseen,
                                'brier':float(np.sum(pp**2)-2*ptrue+1),'log_loss':-float(np.log(max(ptrue,1e-12))),
                                'state_probabilities':json.dumps(dict(zip(classes.tolist(),pp.tolist())))})
        if number%50==0:print(name,'characters',number+1,'elapsed',round(time.time()-start,1),flush=True)
    out=pd.DataFrame(out);out.to_csv(outdir/'predictions.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(outdir/'model_diagnostics.csv',index=False)
    (outdir/'estimated_trees.json').write_text(json.dumps(tree_data))
    assert not out.duplicated(['lineage','scenario','character','taxon','model']).any()
    metrics=[]
    for (scenario,model),g in out.groupby(['scenario','model']):
        correct=g.true==g.pred
        metrics.append({'lineage':name,'scenario':scenario,'model':model,'n':len(g),'taxa':g.taxon.nunique(),
                        'characters':g.character.nunique(),'accuracy':float(correct.mean()),
                        'macro_character_accuracy':float(g.assign(correct=correct).groupby('character').correct.mean().mean()),
                        'brier':float(g.brier.mean()),'log_loss':float(g.log_loss.mean()),
                        'mean_confidence':float(g.confidence.mean()),'unseen_states':int(g.unseen_state.sum())})
    pd.DataFrame(metrics).to_csv(outdir/'metrics.csv',index=False)
    checks={'lineage':name,'all_old_predictions_reproduced':reproduced,'new_model_predictions':int((out.model=='nj_mk').sum()),
            'no_training_target_overlap':True,'target_excluded_from_tree_distances':True,
            'seconds':time.time()-start}
    (outdir/'checks.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks),flush=True)
    return metrics

if __name__=='__main__':
    lineages=sys.argv[1:] or ['dinosaurs','tetrapods','crocodilians']
    with ProcessPoolExecutor(max_workers=3) as ex:
        list(ex.map(run_dataset,lineages))
    metric_paths=[P/name/'metrics.csv' for name in ['dinosaurs','tetrapods','crocodilians','gnathostomes','whales']]
    m=pd.concat([pd.read_csv(p) for p in metric_paths if p.exists()],ignore_index=True)
    m.to_csv(P/'comparison_metrics.csv',index=False)
    print(m.to_string(index=False))
