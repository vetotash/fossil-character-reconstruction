#!/usr/bin/env python3
"""Read-only numerical/input-boundary review of the post hoc NJ amendment."""
from pathlib import Path
import argparse, gzip, hashlib, json, math, sys
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def scalar_distance(x):
 n=len(x);d=np.full((n,n),np.nan);finite=[];zeros=0
 for a in range(n):
  for b in range(a+1,n):
   pairs=[(int(u),int(v)) for u,v in zip(x[a],x[b]) if u>=0 and v>=0]
   if pairs:
    val=sum(u!=v for u,v in pairs)/len(pairs);finite.append(val);d[a,b]=d[b,a]=val
   else:zeros+=1
 fallback=sum(finite)/len(finite) if finite else .5
 d[~np.isfinite(d)]=fallback;np.fill_diagonal(d,0)
 return d,zeros,fallback

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--root',type=Path,required=True,help='Package root containing simulation/results.')
 parser.add_argument('--topology-dir',type=Path,required=True,help='Directory containing NJ/search Python modules and DESIGN_AMENDMENT.json.')
 parser.add_argument('--original-results',type=Path,required=True,help='Original three-start result directory containing search_records/.')
 parser.add_argument('--amended-results',type=Path,required=True,help='Four-start result directory containing search_records/s00_r000.json.gz.')
 parser.add_argument('--output',type=Path,required=True,help='New review receipt JSON; existing files are not overwritten.')
 args=parser.parse_args()
 if args.output.exists():parser.error('--output must name a new file')
 code=args.topology_dir.resolve()
 sys.path.insert(0,str(code))
 global matrix_distance,neighbor_join,matrix_nj_start,combine_endpoints
 global enumerate_trees,split_set,score_search,full_score,validate_tree,random_start_tree
 from nj_start import matrix_distance,neighbor_join,matrix_nj_start,combine_endpoints
 from nni_search import enumerate_trees,split_set,score_search,full_score,validate_tree,random_start_tree
 rng=np.random.default_rng(39731);distance_checks=0
 cases=[np.full((4,3),-1,dtype=np.int8),np.zeros((4,0),dtype=np.int8),np.zeros((4,5),dtype=np.int8),
  np.array([[0,-1,1],[1,0,-1],[-1,-1,-1],[0,1,1]],dtype=np.int8)]
 for n in [4,7,48]:
  for width in [1,3,40]:
   for _ in range(6):cases.append(rng.integers(-1,2,(n,width),dtype=np.int8))
 for x in cases:
  before=x.copy();expected,zero,fallback=scalar_distance(x);actual,info=matrix_distance(x)
  np.testing.assert_allclose(actual,expected,atol=1e-15)
  assert info['zero_overlap_pairs']==zero and math.isclose(info['zero_overlap_fallback'],fallback)
  np.testing.assert_array_equal(x,before);validate_tree(neighbor_join(actual),len(x));distance_checks+=1
 # Independent additive path-distance example checks NJ leaf IDs and row reordering.
 n=7;edges=random_start_tree(n,73691);adj=validate_tree(edges,n)
 weights={tuple(sorted(e)):float(i+1) for i,e in enumerate(edges)}
 dist=np.zeros((n,n))
 for a in range(n):
  queue=[(a,-1,0.)]
  for node,parent,length in queue:
   if node<n:dist[a,node]=length
   for nxt in adj[node]:
    if nxt!=parent:queue.append((nxt,node,length+weights[tuple(sorted((node,nxt)))]))
 assert split_set(neighbor_join(dist),n)==split_set(edges,n)
 perm=rng.permutation(n);permuted=neighbor_join(dist[np.ix_(perm,perm)])
 mapped=permuted.copy();mapped[mapped<n]=perm[mapped[mapped<n]]
 assert split_set(mapped,n)==split_set(edges,n)
 # Duplicate starts must not increase endpoint weights; truth is evaluated only later.
 trees=enumerate_trees(4)
 def found(e,score=0):return {'edges':e,'score':score,'steps':0,'reached_step_limit':False}
 union=combine_endpoints([found(trees[0]),found(trees[0]),found(trees[1]),found(trees[2],1)],4)
 assert union['n_starts']==4 and union['n_distinct_best_endpoints']==2
 retained_before=[split_set(x['edges'],4) for x in union['endpoints']]
 score=score_search(union,trees[0],4)
 assert score['mean_normalized_RF']==.5 and score['min_normalized_RF']==0 and score['max_normalized_RF']==1
 score_search(union,trees[2],4)
 assert retained_before==[split_set(x['edges'],4) for x in union['endpoints']]
 improved=combine_endpoints([found(trees[0],2),found(trees[1],1),found(trees[2],1),found(trees[0],0)],4)
 assert improved['minimum_found_length']==0 and len(improved['endpoints'])==1
 # Reconstruct every input for the already-saved one-matrix driver check.
 source=args.root.resolve()/'simulation/results'
 identifier='s00_r000'
 def receipt(folder):
  with gzip.open(folder/'search_records'/f'{identifier}.json.gz','rt') as h:return json.load(h)
 old=receipt(args.original_results.resolve());new=receipt(args.amended_results.resolve())
 paths={'masked':source/'inputs'/f'{identifier}_masked.npz','truth':source/'truth'/f'{identifier}_truth.npz',
        'predictions':source/'predictions'/f'{identifier}.npz','evaluation':source/'evaluation'/f'{identifier}.npz'}
 assert {k:sha(v) for k,v in paths.items()}==old['source_hashes']==new['source_hashes']
 with np.load(paths['masked'],allow_pickle=False) as z:x=z['X']
 with np.load(paths['truth'],allow_pickle=False) as z:truth=z['X_true']
 with np.load(paths['predictions'],allow_pickle=False) as z:
  probs=z['probabilities'];targets=z['target_indices'];packed=z['missing_states_packed']
 np.testing.assert_array_equal(targets,np.argwhere(x==-1));assert x.shape==truth.shape==(48,40)
 assert np.isin(x,[-1,0,1]).all() and np.isin(truth,[0,1]).all()
 # Independently unpack big-endian bits, instead of using np.unpackbits.
 shifts=7-(np.arange(len(targets))%8)
 decoded=(packed[:,:,np.arange(len(targets))//8]>>shifts)&1
 schemes=['mode','knn5','logistic','extra_trees','nj_mk','nj_mk_joint']
 assert new['draw_indices']==old['draw_indices']==[0,73,146,219,292,365,438,511]
 matrices={}
 for row in old['treatment_results']:
  if row['kind']=='baseline':m=x.copy() if row['scheme']=='source_incomplete' else truth.copy()
  else:
   m=x.copy();i=schemes.index(row['scheme'])
   if row['kind']=='map':states=np.argmax(probs[min(i,4)],axis=-1)[targets[:,0],targets[:,1]]
   else:states=decoded[i,row['draw_index']]
   m[targets[:,0],targets[:,1]]=states
  m=m.astype(np.int8);key=hashlib.sha256(m.tobytes()).hexdigest();assert key==row['searched_matrix_sha256']
  matrices[key]=m
 assert len(old['treatment_results'])==len(new['treatment_results'])==56
 assert len(matrices)==len(new['searches'])==55
 endpoint_scores=0
 for key,m in matrices.items():
  r=new['searches'][key];prior=old['searches'][key]
  assert len(r['starts'])==4 and r['starts'][:3]==prior['starts']
  nj,diagnostic=matrix_nj_start(m);np.testing.assert_array_equal(nj,r['nj_start_edges']);assert diagnostic==r['nj_diagnostics']
  expected,zero,_=scalar_distance(m);np.testing.assert_allclose(matrix_distance(m)[0],expected)
  assert zero==0
  actualscores=[];splits=[]
  for item in r['starts']:
   value=full_score(m,np.array(item['edges']));assert value==item['score'];actualscores.append(value)
   splits.append(split_set(np.array(item['edges']),len(m)));endpoint_scores+=1
  minimum=min(actualscores)
  best={s for s,v in zip(splits,actualscores) if v==minimum}
  assert r['minimum_found_length']==minimum<=prior['minimum_found_length']
  assert best=={split_set(np.array(e),len(m)) for e in r['retained_edges']}
 # Inspect binary/anchor preconditions for all production masked inputs, no searches rerun.
 n_source=0;min_complete=40
 for p in sorted((source/'inputs').glob('*_masked.npz')):
  with np.load(p,allow_pickle=False) as z:m=z['X']
  assert m.shape==(48,40) and np.isin(m,[-1,0,1]).all()
  complete=int(np.all(m>=0,axis=0).sum());min_complete=min(min_complete,complete);n_source+=1
 assert n_source==600 and min_complete>=8
 result={'status':'PASS','scope':'Independent read-only review of NJ initialization and four-start merge; no production files altered.',
  'substantive_defects_found':[],
  'independent_distance_and_fallback_cases':distance_checks,
  'all_unknown_fallback':.5,'fallback_excludes_diagonal':True,
  'additive_tree_leaf_ID_and_row_permutation_checks':True,
  'distinct_equal_weight_ties_and_lower_score_replacement_checks':True,
  'post_search_truth_change_cannot_modify_retained_endpoints':True,
  'source_matrix':'s00_r000','input_treatment_hashes_reconstructed':56,'unique_inputs_checked':55,
  'stored_endpoint_scores_rescored':endpoint_scores,'original_three_endpoints_exactly_retained':True,
  'matrix_derived_NJ_edges_reproduced':55,'big_endian_completion_bits_independently_decoded':True,
  'production_masked_inputs_checked':600,'minimum_complete_anchor_columns':min_complete,
  'read_only_findings':['NJ receives only the matrix being searched, including both old unknown entries and predicted completion states as appropriate.',
   'Leaf IDs are matrix row indices; target_indices binds to argwhere(masked == -1), and reconstructed byte hashes match original receipts.',
   'Four endpoints compete by parsimony length only; duplicate minimum topologies are counted once and tied distinct topologies receive equal weight.',
   'Generating topology reaches only post-search metrics; it does not select an NJ join, NNI move or retained endpoint.',
   'The amendment was chosen after the three-random-start failure was inspected and is explicitly labelled post hoc; this does not make the search global.'],
  'limitations':['No repeat of the full 600-matrix amended search; all-input preconditions plus small/toy cases and one complete saved driver case were reviewed.',
   'As already documented, consensus precision is undefined for an empty consensus and summary means omit those undefined values.'],
  'hashes':{f:sha(code/f) for f in ['nj_start.py','run_nj_amendment.py','nni_search.py','run_topology_extension.py','DESIGN_AMENDMENT.json']}}
 out=args.output.resolve();out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k not in ['hashes','read_only_findings','limitations']},indent=2))
if __name__=='__main__':main()
