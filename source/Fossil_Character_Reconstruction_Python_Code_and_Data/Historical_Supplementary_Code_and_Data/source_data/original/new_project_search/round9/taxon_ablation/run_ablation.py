"""Exact native re-scoring after restoring one focal taxon's targets to '?'."""
from pathlib import Path
import argparse,datetime,gzip,hashlib,importlib.util,json,re,sys,time,os,subprocess
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];R8=ROOT/'new_project_search/round8'
REP=R8/'replication';sys.path.insert(0,str(REP))
dspec=importlib.util.spec_from_file_location('dino_local_adapter',HERE/'dino_native_adapter.py')
dino=importlib.util.module_from_spec(dspec);dspec.loader.exec_module(dino)
from run_fixed_replication import source_ordered
spec=importlib.util.spec_from_file_location('tetprop',R8/'run_probability_propagation.py')
tet=importlib.util.module_from_spec(spec);spec.loader.exec_module(tet)
SCHEMES=['mode','knn5','logistic','extra_trees','nj_mk','nj_mk_joint']
HARD=['source','reference','mode','knn5','logistic','extra_trees','nj_mk']
B=1000
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read_matrix(path):
    text=path.read_text();ntax=int(re.search(r'NTAX\s*=\s*(\d+)',text,re.I)[1]);nc=int(re.search(r'NCHAR\s*=\s*(\d+)',text,re.I)[1])
    block=re.search(r'^\s*MATRIX\s*\n(.*?);',text,re.M|re.S|re.I)[1]
    raw={}
    for line in block.strip().splitlines():
        m=re.match(r"\s*('(?:[^']|'')*'|\S+)\s+(.+)",line);name=m[1].strip("'")
        row=re.findall(r'\([^)]*\)|\{[^}]*\}|.',re.sub(r'\s+','',m[2]));assert len(row)==nc
        assert name not in raw;raw[name]=row
    assert len(raw)==ntax
    return raw,text
def load(history):
    work=ROOT/'morphology_workspace/morphology_project'
    if history=='tetrapods':
        folder=R8/'probability_propagation/future95';matrix=work/'round5/tetrapod_consequence/exploratory_later_reference_trees/inputs/source.nex'
        tree=matrix.parent.parent/'all_five_reference_trees.tre';trees=['O1','O2','R1','R2','R3'];predpath=work/'round3/shared_training/tetrapods/predictions.csv'
    else:
        folder=REP/'probability';matrix=REP/'inputs/source.nex';tree=REP/'reference_trees.nex';trees=['baron','langer'];predpath=work/'round3/shared_training/dinosaurs/predictions.csv'
    raw,text=read_matrix(matrix);names=list(raw)
    allcells=pd.read_csv(folder/'cells.csv',keep_default_na=False,dtype={'true':str,'pred':str})
    index=np.where(allcells.taxon.isin(names))[0];cells=allcells.iloc[index].reset_index(drop=True)
    assert len(cells)==(349 if history=='tetrapods' else 1333)
    oldhard=np.load(folder/'hard_scores.npz',allow_pickle=False);hs=oldhard['steps'];hm=oldhard['minima'];chars=oldhard['characters'].tolist()
    assert hs.shape==(len(trees),7,len(chars)) and hm.shape==(7,len(chars))
    assert set(cells.character)==set(chars)
    pred=pd.read_csv(predpath,keep_default_na=False,dtype={'true':str,'pred':str});pred=pred[pred.scenario=='later_target']
    targetkeys=[(int(r.character),r.taxon) for r in cells.itertuples()]
    hard_values=[cells.true.to_numpy(dtype='U1')]
    for model in HARD[2:]:
        pdict={(int(r.character),r.taxon):r.pred for r in pred[pred.model==model].itertuples()}
        hard_values.append(np.array([pdict[k] for k in targetkeys],dtype='U1'))
    hard_values=np.stack(hard_values)
    models={};inputs=[matrix,tree,folder/'cells.csv',folder/'hard_scores.npz',predpath]
    for model in SCHEMES:
        if history=='dinosaurs' and model=='nj_mk_joint':dp=REP/'joint_probability/aligned_draws.npy';sp=REP/'joint_probability/scores.npz'
        else:dp=folder/f'{model}_draws.npy';sp=folder/f'{model}_scores.npz'
        draw=np.load(dp,allow_pickle=False);assert draw.shape==(B,len(allcells))
        old=np.load(sp,allow_pickle=False);assert old['steps'].shape==(len(trees),B,len(chars)) and old['minima'].shape==(B,len(chars))
        assert np.array_equal(old['characters'],chars)
        models[model]=dict(draws=draw[:,index],steps=old['steps'],minima=old['minima'])
        inputs.extend([dp,sp])
    return dict(history=history,raw=raw,text=text,names=names,cells=cells,chars=chars,tree=tree,trees=trees,hard_steps=hs,hard_minima=hm,hard_values=hard_values,models=models,
        inputs=inputs,ordered=source_ordered(text) if history=='dinosaurs' else set())
def source_summary(h,hs,hm,steps,minimum,model):
    base=hs[:,0].sum(1);bmin=hm[0].sum();ds=steps.sum(2)-base[:,None];dm=minimum.sum(1)-bmin;ex=ds-dm[None]
    oi=1;pi=HARD.index('nj_mk' if model=='nj_mk_joint' else model)
    observed=hs[:,oi].sum(1)-base-(hm[oi].sum()-bmin);point=hs[:,pi].sum(1)-base-(hm[pi].sum()-bmin)
    rows=[]
    for i,tr in enumerate(h['trees']):
        e=ex[i];q=np.quantile(e,[.025,.975]);rows.append(dict(history=h['history'],reference=tr,scheme=model,reference_excess_steps=int(observed[i]),point_excess_steps=int(point[i]),sim_mean_excess_steps=float(e.mean()),sim_sd_excess_steps=float(e.std(ddof=1)),sim_q025=float(q[0]),sim_q975=float(q[1]),fraction_at_least_reference=float((e>=observed[i]).mean()),fraction_at_most_reference=float((e<=observed[i]).mean()),reference_in_central95=bool(q[0]<=observed[i]<=q[1]),mean_added_steps=float(ds[i].mean()),mean_added_minimum_steps=float(dm.mean())))
    return rows,ds,dm,ex
def native(h,chars,cells,draws,dest):
    try:
        if h['history']=='tetrapods':return tet_native(h,chars,cells,draws,dest)
        return dino.score(np.array(list(h['raw'].values())),h['names'],chars,cells,draws,h['ordered'],dest)
    finally:
        # Native adapters compress on success; also preserve/compress failed evidence.
        if dest.exists():
            leftovers=[]
            for p in dest.iterdir():
                if p.suffix not in ['.nex','.log','.tsv','.txt','.csv']:continue
                data=p.read_bytes();q=p.with_suffix(p.suffix+'.gz')
                assert not q.exists()
                q.write_bytes(gzip.compress(data,compresslevel=6,mtime=0));leftovers.append(dict(name=p.name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),gzip_sha256=sha(q)));p.unlink()
            if leftovers:(dest/'cleanup_evidence.json').write_text(json.dumps(leftovers,indent=2))
def tet_native(h,chars,cells,draws,dest):
    """Same round8 PAUP commands; scope-aware check when no multi-state tokens exist."""
    assert not dest.exists();dest.mkdir(parents=True)
    nr=1 if draws is None else len(draws);nc=nr*len(chars)
    matrix=tet.make_matrix(h['raw'],chars,cells,draws)
    multi=any(row[j-1].startswith(('(', '{')) for row in h['raw'].values() for j in chars)
    driver=dest/'diagnose.nex';driver.write_text(matrix+f'''\nBEGIN PAUP;
set autoclose=yes increase=no maxtrees=100;
outgroup Eusthenopteron / only;
pset mstaxa=variable minforfit=minlength opt=deltran;
gettrees file='{h['tree'].resolve()}';
pscores all / single=all tl=yes scorefile=native_scores.tsv replace=yes;
describe 1 / plot=none diag=yes;
quit;
END;\n''')
    env=os.environ.copy();env['LD_LIBRARY_PATH']=str(tet.WORK/'tools/runtime/usr/lib/x86_64-linux-gnu');env['OMP_NUM_THREADS']='1'
    t=time.monotonic()
    with (dest/'stdout.txt').open('w') as f:r=subprocess.run([str(tet.WORK/'tools/paup4a169_ubuntu64'),'-n','-u','-L','native.log','-r',driver.name],cwd=dest,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=60)
    assert r.returncode==0
    log=(dest/'native.log').read_text();assert not re.search(r'\berrors?\b|execution terminated|processing.*terminated',log,re.I)
    if multi:assert 'Multistate taxa interpretation depends on "{}" versus "()" designation' in log
    diag=tet.lower.parse_diag(log,nc);totals,scores=tet.lower.native.read_scorefile(dest/'native_scores.tsv',nc)
    assert len(scores)==5
    s=np.array([[int(v[j]) for j in range(1,nc+1)] for _,v in sorted(scores.items())],dtype=np.int16)
    minimum=np.array([int(diag[j]['native_minimum_steps']) for j in range(1,nc+1)],dtype=np.int16)
    assert all(int(diag[j]['native_assigned_steps'])==s[0,j-1] for j in range(1,nc+1)) and (s>=minimum[None]).all()
    pd.DataFrame(diag.values()).to_csv(dest/'lower_bounds.csv',index=False)
    compressed={}
    for p in [driver,dest/'native.log',dest/'native_scores.tsv',dest/'stdout.txt',dest/'lower_bounds.csv']:
        data=p.read_bytes();q=p.with_suffix(p.suffix+'.gz');q.write_bytes(gzip.compress(data,compresslevel=6,mtime=0));compressed[p.name]=dict(uncompressed_bytes=len(data),uncompressed_sha256=sha(p),gzip_sha256=sha(q));p.unlink()
    (dest/'native_evidence.json').write_text(json.dumps(dict(seconds=time.monotonic()-t,nchar=nc,nrep=nr,original_multistate_tokens_present=multi,all_character_sums_verified=True,all_minima_parsed=True,commands_same_as_round8=True,compressed=compressed),indent=2))
    return s.reshape(5,nr,len(chars)),minimum.reshape(nr,len(chars))
def case_indices(h,taxon):
    if taxon=='original_question_mark_only':
        keys=pd.read_csv(HERE/'source_symbol_keys.csv',keep_default_na=False)
        keyset=set(zip(keys.taxon,keys.character));assert len(keyset)==25
        original_focal=np.array([(r.taxon,int(r.character)) in keyset for r in h['cells'].itertuples()]);assert original_focal.sum()==25
    else:original_focal=(h['cells'].taxon==taxon).to_numpy()
    chars=sorted(h['cells'].loc[original_focal,'character'].unique().tolist())
    ci=[h['chars'].index(x) for x in chars];cols=np.where(h['cells'].character.isin(chars))[0]
    cells=h['cells'].iloc[cols].reset_index(drop=True);focal=np.where(original_focal[cols])[0]
    return chars,ci,cols,cells,focal
def replace(old,new,ci):
    out=old.copy();out[...,ci]=new
    unaffected=[j for j in range(old.shape[-1]) if j not in ci]
    assert np.array_equal(out[...,unaffected],old[...,unaffected])
    return out
def freeze(histories):
    selected=json.loads((HERE/'SELECTION_FREEZE.json').read_text())
    files={str(p.relative_to(ROOT)):sha(p) for h in histories for p in h['inputs']}
    for p in [HERE/'PROTOCOL.txt',HERE/'SOURCE_SYMBOL_ADDENDUM.txt',HERE/'ADAPTER_NOTE.txt',HERE/'SOURCE_SYMBOL_FREEZE.json',HERE/'source_symbol_keys.csv',HERE/'SELECTION_FREEZE.json',HERE/'selected_cells.csv',HERE/'all_taxon_target_counts.csv',Path(__file__),HERE/'dino_native_adapter.py',REP/'run_probability.py',R8/'run_probability_propagation.py',R8/'method_review/audit_native_minima.py']:
        files[str(p.relative_to(ROOT))]=sha(p)
    assert not (HERE/'cases').exists()
    selected['selected']['dinosaurs'].append('original_question_mark_only')
    gate=dict(time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=files,native_ablation_scores_present=False,prior_no_ablation_validation_attempt_documented=True,selected=selected['selected'],existing_draws=1000,new_draws=0,new_model_fits=0,new_trees=0)
    (HERE/'FREEZE.json').write_text(json.dumps(gate,indent=2));return gate
def validate(histories,gate):
    rows=[];summary=[]
    for h in histories:
        for taxon in gate['selected'][h['history']]:
            chars,ci,cols,cells,focal=case_indices(h,taxon);folder=HERE/'validation'/h['history']/taxon
            bs,bm=native(h,chars,cells,None,folder/'source')
            hs,hm=native(h,chars,cells,h['hard_values'][:,cols],folder/'hard_no_ablation')
            assert np.array_equal(bs[:,0],h['hard_steps'][:,0,ci])
            assert np.array_equal(bm[0],h['hard_minima'][0,ci])
            assert np.array_equal(hs,h['hard_steps'][:,1:][:,:,ci]) and np.array_equal(hm,h['hard_minima'][1:][:,ci])
            first=np.stack([h['models'][m]['draws'][0,cols] for m in SCHEMES]);fs,fm=native(h,chars,cells,first,folder/'first_draw_no_ablation')
            for k,m in enumerate(SCHEMES):
                old=h['models'][m]
                assert np.array_equal(fs[:,k],old['steps'][:,0,ci]) and np.array_equal(fm[k],old['minima'][0,ci])
                restored=replace(old['steps'],old['steps'][...,ci],ci);restoredmin=replace(old['minima'],old['minima'][...,ci],ci)
                assert np.array_equal(restored,old['steps']) and np.array_equal(restoredmin,old['minima'])
                a=source_summary(h,h['hard_steps'],h['hard_minima'],restored,restoredmin,m)[0]
                b=source_summary(h,h['hard_steps'],h['hard_minima'],old['steps'],old['minima'],m)[0];assert a==b
            rows.append(dict(history=h['history'],taxon=taxon,focal_cells=len(focal),affected_characters=chars,native_source_and_all_hard_reproduced=True,native_first_draw_all6_schemes_reproduced=True,unmodified_slice_replacement_reproduces_full_arrays_and_summaries=True))
            print(json.dumps(dict(validation=h['history'],taxon=taxon,characters=len(chars),passed=True)),flush=True)
    (HERE/'VALIDATION.json').write_text(json.dumps(dict(time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),cases=rows,all7_cases_passed=True),indent=2))
    return rows
def execute(histories,gate):
    assert json.loads((HERE/'VALIDATION.json').read_text())['all7_cases_passed']
    allsummary=[];allreps=[];manifest=[];t0=time.monotonic()
    for h in histories:
        for taxon in gate['selected'][h['history']]:
            chars,ci,cols,cells,focal=case_indices(h,taxon);out=HERE/'cases'/h['history']/taxon;out.mkdir(parents=True)
            hdraw=h['hard_values'][:,cols].copy();hdraw[:,focal]='?'
            hs,hm=native(h,chars,cells,hdraw,out/'native_hard')
            fullhs=h['hard_steps'].copy();fullhm=h['hard_minima'].copy()
            fullhs[:,1:]=replace(fullhs[:,1:],hs,ci);fullhm[1:]=replace(fullhm[1:],hm,ci)
            np.savez_compressed(out/'hard_scores.npz',steps=fullhs,minima=fullhm,characters=h['chars'],trees=h['trees'],treatments=HARD)
            pd.DataFrame(dict(character=chars,original_array_index=ci)).to_csv(out/'affected_characters.csv',index=False)
            for model in SCHEMES:
                old=h['models'][model];draws=old['draws'][:,cols].copy();draws[:,focal]='?'
                assert (draws[:,focal]=='?').all()
                other=[j for j in range(len(cols)) if j not in focal];assert np.array_equal(draws[:,other],old['draws'][:,cols][:,other])
                batch=200 if h['history']=='tetrapods' else 50;parts=[];mparts=[]
                for start in range(0,B,batch):
                    end=min(start+batch,B);s,m=native(h,chars,cells,draws[start:end],out/'native'/model/f'batch_{start:04d}_{end:04d}');parts.append(s);mparts.append(m)
                affected=np.concatenate(parts,axis=1);minimum=np.concatenate(mparts)
                fulls=replace(old['steps'],affected,ci);fullm=replace(old['minima'],minimum,ci)
                # Relaxing one cell may not increase a fixed-tree or attainable minimum cost.
                assert (fulls<=old['steps']).all() and (fullm<=old['minima']).all()
                scorefile=out/f'{model}_scores.npz';np.savez_compressed(scorefile,steps=fulls,minima=fullm,characters=h['chars'],trees=h['trees'],affected_character_indices=ci)
                rows,ds,dm,ex=source_summary(h,fullhs,fullhm,fulls,fullm,model)
                original=source_summary(h,h['hard_steps'],h['hard_minima'],old['steps'],old['minima'],model)[0]
                for ti,(r,orig) in enumerate(zip(rows,original)):
                    r.update(focal_taxon=taxon,restored_target_cells=len(focal),remaining_target_cells=len(h['cells'])-len(focal),affected_characters=len(chars),replicates=B,
                        case_type='source_symbol_filter' if taxon=='original_question_mark_only' else 'single_target_taxon_unfilled',
                        original_reference_excess_steps=orig['reference_excess_steps'],original_point_excess_steps=orig['point_excess_steps'],original_sim_mean_excess_steps=orig['sim_mean_excess_steps'],original_reference_in_central95=orig['reference_in_central95'],
                        delta_reference_excess_steps=r['reference_excess_steps']-orig['reference_excess_steps'],delta_point_excess_steps=r['point_excess_steps']-orig['point_excess_steps'],delta_sim_mean_excess_steps=r['sim_mean_excess_steps']-orig['sim_mean_excess_steps'],
                        interval_status_changed=r['reference_in_central95']!=orig['reference_in_central95'])
                    allsummary.append(r)
                    allreps.extend(dict(history=h['history'],focal_taxon=taxon,scheme=model,reference=h['trees'][ti],replicate=i,added_steps=int(ds[ti,i]),added_minimum_steps=int(dm[i]),added_excess_steps=int(ex[ti,i])) for i in range(B))
                manifest.append(dict(history=h['history'],focal_taxon=taxon,scheme=model,native_characters_recomputed=len(chars),replicates=B,unaffected_slices_bitwise_equal=True,all_restored_cells_missing=True,all_other_frozen_draw_values_unchanged=True,score_sha256=sha(scorefile)))
                pd.DataFrame(allsummary).to_csv(HERE/'summary.csv',index=False);pd.DataFrame(allreps).to_csv(HERE/'replicate_totals.csv',index=False)
                (HERE/'progress_manifest.json').write_text(json.dumps(manifest,indent=2))
                print(json.dumps(dict(history=h['history'],taxon=taxon,scheme=model,complete=1000,elapsed_seconds=round(time.monotonic()-t0,1))),flush=True)
    assert len(allsummary)==138 and len(allreps)==138000 and len(manifest)==42
    (HERE/'RUN_MANIFEST.json').write_text(json.dumps(dict(completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),seconds=time.monotonic()-t0,summary_rows=len(allsummary),replicate_rows=len(allreps),schemes=SCHEMES,cases=7,replicates=1000,new_draws=0,refits=0,tree_searches=0,freeze_sha256=sha(HERE/'FREEZE.json'),inventory=manifest),indent=2))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('operation',choices=['validate','score']);args=ap.parse_args()
    histories=[load('tetrapods'),load('dinosaurs')]
    if args.operation=='validate':gate=freeze(histories);validate(histories,gate)
    else:
        gate=json.loads((HERE/'FREEZE.json').read_text())
        for p,s in gate['files'].items():assert sha(ROOT/p)==s,p
        execute(histories,gate)
