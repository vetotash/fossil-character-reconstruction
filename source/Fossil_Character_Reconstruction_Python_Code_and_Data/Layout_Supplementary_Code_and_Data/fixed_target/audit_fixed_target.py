#!/usr/bin/env python3
"""Independent bitset scoring and input/seed audit of the fixed-target control.

No estimator, production scoring routine or production mask randomizer is used
in the numerical checks below. Run only after all fits and evaluation finish.
"""
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import hashlib,json,math,time
import numpy as np

HERE=Path(__file__).resolve().parent
SCHEMES=('mode','knn5','logistic','extra_trees','nj_mk','nj_mk_joint')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def dump(path,x):Path(path).write_text(json.dumps(x,indent=2)+'\n')

def independent_fitch(patterns,edges,root):
    """Binary unordered patterns, -1 unknown. Root has at most three children."""
    p=np.atleast_2d(np.asarray(patterns,dtype=np.int8));ntips=p.shape[1]
    edges=np.asarray(edges,dtype=int);nodes=int(edges.max())+1
    adj=[[] for _ in range(nodes)]
    for a,b in edges:adj[a].append(b);adj[b].append(a)
    assert len(edges)==nodes-1 and all(len(adj[t])==1 for t in range(ntips))
    parents={int(root):-1};order=[int(root)]
    for n in order:
        for child in adj[n]:
            if child!=parents[n]:
                assert child not in parents;parents[child]=n;order.append(child)
    assert len(order)==nodes
    states={};cost=np.zeros(len(p),dtype=np.int16)
    for node in reversed(order):
        if node<ntips:
            states[node]=np.where(p[:,node]<0,3,np.where(p[:,node]==0,1,2)).astype(np.uint8)
        else:
            children=[c for c in adj[node] if parents.get(c)==node]
            assert len(children)==(3 if node==root else 2) or (node==root and len(children)==2)
            value=states.pop(children[0])
            for child in children[1:]:
                other=states.pop(child);intersection=value&other;disjoint=intersection==0
                cost+=disjoint;value=np.where(disjoint,value|other,intersection).astype(np.uint8)
            states[node]=value
    minimum=((p==0).any(1)&(p==1).any(1)).astype(np.int16)
    return cost,minimum

def completed_totals(masked,target,states,edges,root):
    states=np.atleast_2d(states);positions=np.argwhere(target)
    complete=np.broadcast_to(masked,(len(states),*masked.shape)).copy()
    complete[:,positions[:,0],positions[:,1]]=states
    assert np.all(complete[:,(~target)&(masked==-1)]==-1)
    patterns=complete.transpose(0,2,1).reshape(-1,len(masked))
    length,minimum=independent_fitch(patterns,edges,root)
    return length.reshape(len(states),-1).sum(1),minimum.reshape(len(states),-1).sum(1)

def audit_one(base,entry,binding):
    base=Path(base);mid=entry['matrix_id'];oldid=binding['empirical_id']
    newpath=base/entry['masked_file'];assert sha(newpath)==entry['masked_sha256']==binding['new_input_sha256']
    with np.load(newpath) as d:new=d['X'];target=d['target_mask'].astype(bool);labels=d['labels']
    oldpath=base/'inputs'/f'{oldid}.npz';assert sha(oldpath)==binding['source_input_sha256']
    with np.load(oldpath) as d:old=d['X'];oldtarget=d['target_mask'].astype(bool);oldlabels=d['labels']
    truthpath=base/'truth'/f"{entry['pair_id']}.npz";assert sha(truthpath)==binding['source_truth_sha256']
    with np.load(truthpath) as d:truth=d['X_true'];edges=d['edges'];root=int(d['root'])
    assert np.array_equal(target,oldtarget) and np.array_equal(target,labels==2)
    assert np.array_equal(np.stack([(labels==k).sum(0) for k in range(3)]),np.stack([(oldlabels==k).sum(0) for k in range(3)]))
    assert np.array_equal(new,np.where(labels==0,truth.astype(np.int8),-1))
    rng=np.random.default_rng(np.random.SeedSequence(binding['permutation_seed']))
    replay=oldlabels.copy()
    for j in range(labels.shape[1]):
        available=np.flatnonzero(oldlabels[:,j]!=2)
        # Independent label-vector implementation (production permutes O booleans).
        replay[available,j]=oldlabels[rng.permutation(available),j]
    assert np.array_equal(replay,labels)
    oldtruths=[]
    for inputmatrix,name in [(old,oldid),(new,mid)]:
        l,m=independent_fitch(inputmatrix.T,edges,root);l0,m0=int(l.sum()),int(m.sum())
        lt,mt=completed_totals(inputmatrix,target,truth[target][None,:],edges,root)
        td=int(lt[0]-l0-(mt[0]-m0));oldtruths.append(td)
        with np.load(base/'evaluation'/f'{name}.npz') as ev:
            assert int(ev['old_L'])==l0 and int(ev['old_M'])==m0
            assert int(ev['true_L'])==lt[0] and int(ev['true_M'])==mt[0] and int(ev['true_D'])==td
    meta=read(base/'predictions'/f'{mid}.json');predpath=base/'predictions'/f'{mid}.npz'
    assert sha(predpath)==meta['prediction_sha256']
    assert meta['status']=='completed' and not meta['failures']
    assert len(meta['nj_models'])==new.shape[1]
    for j,model in enumerate(meta['nj_models']):
        if not target[:,j].any():
            assert model is None
            continue
        visible=np.flatnonzero(new[:,j]>=0)
        assert model['train_indices']==visible.tolist()
        assert model['train_states']==new[visible,j].astype(str).tolist()
        assert model['classes']==np.unique(new[visible,j].astype(str)).tolist()
        assert model['constant_fallback']==(len(model['classes'])==1)
    with np.load(predpath) as p:
        probs=p['probabilities'];packed=p['missing_states_packed'];positions=p['target_indices']
    assert np.array_equal(positions,np.argwhere(target))
    assert probs.shape==(5,len(positions),2) and np.isfinite(probs).all() and (probs>=0).all()
    assert np.allclose(probs.sum(2),1,atol=1e-12,rtol=0)
    marginal=np.random.default_rng(np.random.SeedSequence(meta['marginal_uniform_seed_sequence'])).random((512,len(positions)))
    l,m=independent_fitch(new.T,edges,root);l0,m0=int(l.sum()),int(m.sum())
    with np.load(base/'evaluation'/f'{mid}.npz') as ev:
        for k,scheme in enumerate(SCHEMES):
            draws=np.unpackbits(packed[k],axis=1,count=len(positions))
            assert hashlib.sha256(draws.tobytes()).hexdigest()==meta['completion_sha256'][k]
            if k<5:assert np.array_equal(draws,(marginal<probs[k,:,1]).astype(np.uint8))
            length,minimum=completed_totals(new,target,draws,edges,root)
            assert np.array_equal(length,ev['L_samples'][k])
            assert np.array_equal(minimum,ev['M_samples'][k])
            assert np.array_equal(length-l0-(minimum-m0),ev['D_samples'][k])
            hard=probs[min(k,4)].argmax(1)
            hl,hm=completed_totals(new,target,hard[None,:],edges,root)
            assert int(ev['hard_D'][k])==int(hl[0]-l0-(hm[0]-m0))
    return {'pair_id':entry['pair_id'],'new_matrix_id':mid,'target_count':len(positions),
      'new_completions_checked':3072,'hard_completions_checked':6,'truth_completions_checked':2,
      'empirical_true_D':oldtruths[0],'fixed_target_true_D':oldtruths[1],
      'source_empirical_prediction_arrays_available':False,'status':'pass'}

def run(base,workers,report_path=None):
    base=Path(base)
    report_path=Path(report_path) if report_path is not None else base/'INDEPENDENT_SCORING_AUDIT.json'
    if report_path.exists():raise FileExistsError(report_path)
    report_path.parent.mkdir(parents=True,exist_ok=True)
    fit=read(base/'FIT_MANIFEST.json');evaluation=read(base/'EVALUATION_MANIFEST.json')
    assert fit['completed_arms']==180 and not fit['failed_arms']
    assert evaluation['rows']==2160
    bindings={b['new_id']:b for b in read(base/'SOURCE_BINDING.json')['pairs']}
    entries=read(base/'input_manifest.json');start=time.monotonic();records=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        fs=[pool.submit(audit_one,base,e,bindings[e['matrix_id']]) for e in entries]
        for f in as_completed(fs):
            records.append(f.result())
            if len(records)%10==0:print(json.dumps({'audited_pairs':len(records),'total_pairs':180,'seconds':time.monotonic()-start}),flush=True)
    records.sort(key=lambda x:x['pair_id'])
    report={'status':'pass','pairs':180,'new_raw_completions_independently_scored':sum(r['new_completions_checked'] for r in records),
       'hard_completions_independently_scored':1080,'target_truth_completions_independently_scored':360,
       'all_targets_fixed':True,'all_column_category_counts_preserved':True,'all_new_mask_seeds_replayed':True,
       'all_marginal_draws_replayed':True,'all_saved_Mk_training_labels_match_visible_input_only':True,'joint_draws_independently_scored_not_replayed_here':True,
       'empirical_D_truth_recomputed':True,'empirical_prediction_arrays_not_available':True,
       'pairs_with_changed_D_truth':sum(r['empirical_true_D']!=r['fixed_target_true_D'] for r in records),
       'script_sha256':sha(__file__),'fit_manifest_sha256':sha(base/'FIT_MANIFEST.json'),
       'evaluation_manifest_sha256':sha(base/'EVALUATION_MANIFEST.json'),'seconds':time.monotonic()-start,'records':records}
    dump(report_path,report)
    print(json.dumps({k:v for k,v in report.items() if k!='records'}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=6);p.add_argument('--report',type=Path)
    a=p.parse_args();run(a.output,a.workers,a.report)
