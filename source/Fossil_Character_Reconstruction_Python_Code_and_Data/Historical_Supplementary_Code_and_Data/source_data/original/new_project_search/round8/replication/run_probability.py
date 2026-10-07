"""Native fixed-tree propagation of frozen dinosaur categorical probabilities."""
from pathlib import Path
from decimal import Decimal
import datetime,gzip,json,os,re,subprocess,time,argparse
import numpy as np
import pandas as pd
from inventory import HERE,PROJECT,parse,sha
from run_fixed_replication import TOOLS,TREATMENTS,TREES,source_ordered

OUT=HERE/'probability';MODELS=TREATMENTS[2:];B=1000;SEED=2026100317
PRED=PROJECT/'round3/shared_training/dinosaurs/predictions.csv'
def generate():
    assert not OUT.exists();OUT.mkdir()
    allp=pd.read_csv(PRED,keep_default_na=False,dtype={'true':str,'pred':str})
    p=allp[allp.scenario=='later_target'];base=p[p.model=='mode'].sort_values(['character','taxon']).reset_index(drop=True)
    assert len(base)==1333 and not base.duplicated(['character','taxon']).any()
    uniforms=np.random.default_rng(SEED).random((B,len(base)));np.save(OUT/'common_uniforms.npy',uniforms)
    base.to_csv(OUT/'cells.csv',index=False)
    draws={};hashes={}
    for model in MODELS:
        sub=p[p.model==model].sort_values(['character','taxon']).reset_index(drop=True)
        assert sub[['taxon','character','true']].equals(base[['taxon','character','true']])
        sample=np.empty((B,len(base)),dtype='U1')
        for i,r in enumerate(sub.itertuples()):
            probs=json.loads(r.state_probabilities);states=sorted(probs,key=int)
            values=np.array([probs[s] for s in states]);assert np.isfinite(values).all() and np.all(values>=0) and np.all(values<=1) and abs(values.sum()-1)<1e-12
            assert all(s in '01234' and len(s)==1 for s in states)
            choices=np.searchsorted(np.cumsum(values),uniforms[:,i],side='right');assert choices.max()<len(states)
            sample[:,i]=np.array(states)[choices]
        path=OUT/f'{model}_draws.npy';np.save(path,sample,allow_pickle=False);hashes[model]=sha(path)
        draws[model]=sample
    gate=dict(time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),seed=SEED,replicates=B,n_targets=1333,n_characters=367,n_taxa=74,
      models=MODELS,trees=TREES,probability_source_sha256=sha(PRED),protocol_sha256=sha(HERE/'PROBABILITY_PROTOCOL.txt'),
      script_sha256=sha(Path(__file__)),common_uniforms_sha256=sha(OUT/'common_uniforms.npy'),draw_sha256=hashes,
      reference_tree_sha256=sha(HERE/'reference_trees.nex'),source_matrix_sha256=sha(HERE/'inputs/source.nex'),
      previous_hard_scores_sha256=sha(HERE/'fixed_tree_totals.csv'),previous_minima_sha256=sha(HERE/'character_lower_bounds.csv'),simulated_native_scores_present=False)
    (OUT/'FREEZE.json').write_text(json.dumps(gate,indent=2));print(json.dumps(gate,indent=2))

def build_matrix(raw,names,chars,cells,draws,ordered):
    nr=1 if draws is None else len(draws);m=np.tile(raw[:,np.array(chars)-1],(1,nr))
    lookup={c:j for j,c in enumerate(chars)}
    if draws is not None:
        for i,r in enumerate(cells.itertuples()):
            row=names.index(r.taxon);ch=lookup[r.character];assert raw[row,r.character-1]=='?'
            m[row,ch+np.arange(nr)*len(chars)]=draws[:,i]
    nc=m.shape[1]
    oc=[rep*len(chars)+j+1 for rep in range(nr) for j,ch in enumerate(chars) if ch in ordered]
    uc=[rep*len(chars)+j+1 for rep in range(nr) for j,ch in enumerate(chars) if ch not in ordered]
    assert len(oc)+len(uc)==nc and len(oc)==nr*sum(c in ordered for c in chars)
    text='#NEXUS\nBEGIN DATA;\n'+f'DIMENSIONS NTAX={len(names)} NCHAR={nc};\nFORMAT DATATYPE=STANDARD SYMBOLS="01234" GAP=- MISSING=?;\nMATRIX\n'
    text+=''.join(n+' '+''.join(row)+'\n' for n,row in zip(names,m))+';\nEND;\n'
    text+='BEGIN ASSUMPTIONS; OPTIONS DEFTYPE=UNORD POLYTCOUNT=MINSTEPS;\nTYPESET * repeated=ord: '+' '.join(map(str,oc))+', unord: '+' '.join(map(str,uc))+';\nEND;\n'
    return m,text
def score(raw,names,chars,cells,draws,ordered,dest):
    assert not dest.exists();dest.mkdir(parents=True)
    matrix,text=build_matrix(raw,names,chars,cells,draws,ordered);nc=matrix.shape[1];nr=nc//len(chars)
    driver=dest/'diagnose.nex';driver.write_text(text+f'''\nBEGIN PAUP;
set autoclose=yes increase=no maxtrees=100;
outgroup Euparkeria / only;
pset mstaxa=variable minforfit=minlength opt=deltran;
gettrees file='{(HERE/'reference_trees.nex').resolve()}';
pscores all / single=all tl=yes scorefile=native_scores.tsv replace=yes;
describe 1 / plot=none diag=yes;
quit;
END;\n''')
    env=os.environ.copy();env['LD_LIBRARY_PATH']=str((TOOLS/'runtime/usr/lib/x86_64-linux-gnu').resolve());env['OMP_NUM_THREADS']='1'
    start=time.monotonic()
    with (dest/'stdout.txt').open('w') as f:r=subprocess.run([str((TOOLS/'paup4a169_ubuntu64').resolve()),'-n','-u','-L','native.log','-r',driver.name],cwd=dest,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=60)
    assert r.returncode==0
    log=(dest/'native.log').read_text();assert not re.search(r'\berrors?\b|execution terminated|processing.*terminated',log,re.I)
    # Verify native character-type counts as well as the text mapping.
    expected_ordered=nr*sum(c in ordered for c in chars)
    assert re.search(r'\b'+str(expected_ordered)+r" characters are of type 'ord'",log)
    scoretable=pd.read_csv(dest/'native_scores.tsv',sep='\t',header=0,dtype=str,keep_default_na=False)
    s=np.zeros((2,nc),dtype=np.int16);totals={};ti=None;counts=[0,0]
    for row in scoretable.itertuples(index=False,name=None):
        if row[0]:ti=int(row[0])-1;totals[ti]=int(Decimal(row[1]))
        elif row[2]:j=int(row[2])-1;s[ti,j]=int(Decimal(row[3]));counts[ti]+=1
    assert counts==[nc,nc] and all(int(s[i].sum())==totals[i] for i in range(2))
    block=log.split('Character diagnostics:\n',1)[1];mins={};printed=0
    for line in block.splitlines():
        m=re.match(r'^\s*(\d+)\s+(.*)$',line)
        if not m:continue
        vals=m[2].split()[-9:]
        if len(vals)<9:continue
        j=int(m[1])-1;ra,mi,tr,ma=map(Decimal,vals[:4]);assert j not in mins and 0<=j<nc
        assert tr==int(s[0,j]);mins[j]=int(mi);printed+=1
    missing=set(range(nc))-set(mins);constant=re.search(r'\((\d+) constant characters not shown\)',block)
    assert len(missing)==(int(constant[1]) if constant else 0)
    for j in missing:
        observed=set(matrix[:,j])-{'?','-'};assert len(observed)<=1 and (s[:,j]==0).all();mins[j]=0
    minimum=np.array([mins[j] for j in range(nc)],dtype=np.int16)
    assert (s>=minimum[None,:]).all()
    evidence={}
    for file in [driver,dest/'native.log',dest/'native_scores.tsv',dest/'stdout.txt']:
        data=file.read_bytes();cp=file.with_suffix(file.suffix+'.gz');cp.write_bytes(gzip.compress(data,compresslevel=6,mtime=0));evidence[file.name]=dict(bytes=len(data),uncompressed_sha256=sha(file),gzip_sha256=sha(cp));file.unlink()
    (dest/'verification.json').write_text(json.dumps(dict(seconds=time.monotonic()-start,nchar=nc,nrep=nr,ordered_columns=expected_ordered,native_constant_omissions=len(missing),all_constants_verified=True,native_totals=totals,all_character_sums_verified=True,evidence=evidence),indent=2))
    return s.reshape(2,nr,len(chars)),minimum.reshape(nr,len(chars))

def execute(batch):
    gate=json.loads((OUT/'FREEZE.json').read_text());assert sha(Path(__file__))==gate['script_sha256']
    assert sha(PRED)==gate['probability_source_sha256'] and sha(HERE/'reference_trees.nex')==gate['reference_tree_sha256']
    names,raw,text=parse(HERE/'inputs/source.nex');ordered=source_ordered(text)
    cells=pd.read_csv(OUT/'cells.csv',keep_default_na=False,dtype={'true':str,'pred':str});chars=sorted(cells.character.unique().tolist());assert len(chars)==367
    p=pd.read_csv(PRED,keep_default_na=False,dtype={'true':str,'pred':str});p=p[p.scenario=='later_target']
    hard=np.stack([cells.true.to_numpy(dtype='U1')]+[p[p.model==m].sort_values(['character','taxon']).pred.to_numpy(dtype='U1') for m in MODELS])
    bs,bm=score(raw,names,chars,cells,None,ordered,OUT/'native_source')
    hs,hm=score(raw,names,chars,cells,hard,ordered,OUT/'native_hard_concat')
    allhs=np.concatenate([bs,hs],axis=1);allhm=np.concatenate([bm,hm])
    old=pd.read_csv(HERE/'character_scores.csv');oldmin=pd.read_csv(HERE/'character_lower_bounds.csv')
    for k,t in enumerate(TREATMENTS):
        for ti,tr in enumerate(TREES):
            ref=old[(old.treatment==t)&(old.reference==tr)].set_index('character').loc[chars,'steps'].to_numpy()
            assert np.array_equal(allhs[ti,k],ref)
        refmin=oldmin[oldmin.treatment==t].set_index('character').loc[chars,'native_minimum_steps'].to_numpy()
        assert np.array_equal(allhm[k],refmin)
    (OUT/'CONCATENATION_CHECK.json').write_text(json.dumps(dict(time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),all5138_target_character_tree_scores_equal_separate_runs=True,all2569_minima_equal_separate_runs=True,n_target_characters=367,n_ordered_target_characters=sum(c in ordered for c in chars),n_treatments=7),indent=2))
    np.savez_compressed(OUT/'hard_scores.npz',steps=allhs,minima=allhm,characters=chars,trees=TREES,treatments=TREATMENTS)
    rows=[];replicate_rows=[];inventory={};t0=time.monotonic()
    for model in MODELS:
        path=OUT/f'{model}_draws.npy';assert sha(path)==gate['draw_sha256'][model];draws=np.load(path,allow_pickle=False)
        pieces=[];mpieces=[]
        for start in range(0,B,batch):
            end=min(start+batch,B);s,m=score(raw,names,chars,cells,draws[start:end],ordered,OUT/'native'/model/f'batch_{start:04d}_{end:04d}')
            pieces.append(s);mpieces.append(m)
        s=np.concatenate(pieces,axis=1);m=np.concatenate(mpieces)
        np.savez_compressed(OUT/f'{model}_scores.npz',steps=s,minima=m,characters=chars,trees=TREES)
        ds=s.sum(axis=2)-bs[:,0,:].sum(axis=1)[:,None];dm=m.sum(axis=1)-bm[0].sum();extra=ds-dm[None,:]
        observed=allhs[:,1,:].sum(axis=1)-bs[:,0,:].sum(axis=1)-(allhm[1].sum()-bm[0].sum())
        k=TREATMENTS.index(model);point=allhs[:,k,:].sum(axis=1)-bs[:,0,:].sum(axis=1)-(allhm[k].sum()-bm[0].sum())
        for ti,tr in enumerate(TREES):
            e=extra[ti];q=np.quantile(e,[.025,.975]);rows.append(dict(model=model,reference=tr,replicates=B,reference_excess_steps=int(observed[ti]),point_excess_steps=int(point[ti]),sim_mean_excess_steps=float(e.mean()),sim_sd_excess_steps=float(e.std(ddof=1)),sim_q025=float(q[0]),sim_q975=float(q[1]),fraction_at_least_reference=float((e>=observed[ti]).mean()),fraction_at_most_reference=float((e<=observed[ti]).mean()),reference_in_central95=bool(q[0]<=observed[ti]<=q[1]),mean_added_steps=float(ds[ti].mean()),mean_added_minimum_steps=float(dm.mean())))
            replicate_rows.extend(dict(model=model,reference=tr,replicate=i,added_steps=int(ds[ti,i]),added_minimum_steps=int(dm[i]),added_excess_steps=int(e[i])) for i in range(B))
        pd.DataFrame(rows).to_csv(OUT/'summary.csv',index=False);pd.DataFrame(replicate_rows).to_csv(OUT/'replicate_totals.csv',index=False)
        inventory[model]=dict(draw_sha256=sha(path),scores_sha256=sha(OUT/f'{model}_scores.npz'),completed_replicates=B)
        print(json.dumps(dict(model=model,completed=B,elapsed_seconds=round(time.monotonic()-t0,2))),flush=True)
    (OUT/'run_manifest.json').write_text(json.dumps(dict(completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),seconds=time.monotonic()-t0,batch_size=batch,seed=SEED,replicates=B,model_refits=0,tree_searches=0,independent_marginal_sampling=True,minimum_steps_verified_natively=True,freeze_sha256=sha(OUT/'FREEZE.json'),inventory=inventory),indent=2))
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('operation',choices=['generate','score']);a.add_argument('--batch-size',type=int,default=50);args=a.parse_args()
    if args.operation=='generate':generate()
    else:execute(args.batch_size)
