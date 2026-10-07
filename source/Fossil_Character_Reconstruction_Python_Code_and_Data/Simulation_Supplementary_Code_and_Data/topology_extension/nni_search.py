"""Deterministic-budget binary unordered parsimony NNI heuristic.

Search accepts only a character matrix and supplied independent starting trees.
No generating topology is used by search. Fitch directed-edge messages give
exact scores for every NNI neighbour. Only strictly improving moves are taken;
ties between improving moves use internal-node/alternative order. Distinct
best-scoring endpoints across starts are retained, not claimed to be all MPTs.
"""
from __future__ import annotations
import numpy as np


def validate_tree(edges, n_tips):
    edges = np.asarray(edges, dtype=int)
    n_nodes = 2*n_tips-2
    if edges.shape != (2*n_tips-3, 2) or np.any(edges < 0) or np.any(edges >= n_nodes):
        raise ValueError("Invalid binary unrooted tree edge dimensions/IDs")
    adj = [[] for _ in range(n_nodes)]
    for a, b in edges:
        if a == b or b in adj[a]: raise ValueError("Repeated/self edge")
        adj[a].append(int(b)); adj[b].append(int(a))
    if any(len(adj[i]) != (1 if i < n_tips else 3) for i in range(n_nodes)):
        raise ValueError("Tips must have degree one and internal nodes degree three")
    seen, queue = {0}, [0]
    for a in queue:
        for b in adj[a]:
            if b not in seen: seen.add(b); queue.append(b)
    if len(seen) != n_nodes: raise ValueError("Disconnected tree")
    return [sorted(a) for a in adj]


def edges_from_adj(adj):
    return np.array([(a,b) for a, nbr in enumerate(adj) for b in sorted(nbr) if a < b], dtype=np.int16)


def random_start_tree(n_tips, seed):
    """Uniform labelled binary tree via edge insertion; independent search RNG."""
    rng = np.random.default_rng(seed)
    edges = [(0,n_tips),(1,n_tips),(2,n_tips)]
    for tip in range(3,n_tips):
        a,b = edges.pop(int(rng.integers(len(edges))))
        node = n_tips+tip-2
        edges.extend([(a,node),(b,node),(tip,node)])
    return np.asarray(edges,dtype=np.int16)


def _orient(adj, root):
    parent = [-1]*len(adj)
    order = [root]
    for a in order:
        for b in adj[a]:
            if b != parent[a]: parent[b]=a; order.append(b)
    return parent,order


def _combine(a,b):
    intersection = a & b
    return np.where(intersection != 0,intersection,a | b)


def full_character_scores(matrix, edges):
    x = np.asarray(matrix,dtype=np.int8)
    if x.ndim != 2 or not np.isin(x,[-1,0,1]).all(): raise ValueError("Binary matrix with -1 unknown required")
    n,p = x.shape
    adj=validate_tree(edges,n);parent,order=_orient(adj,n)
    support=[None]*len(adj)
    score=np.zeros(p,dtype=np.int64)
    for a in reversed(order):
        if a < n:
            support[a]=np.where(x[a]<0,3,1 << x[a]).astype(np.uint8)
        else:
            children=[b for b in adj[a] if b != parent[a]]
            value=support[children[0]].copy()
            for b in children[1:]:
                inter=value & support[b]
                score+=(inter==0)
                value=np.where(inter!=0,inter,value | support[b])
            support[a]=value
    return score


def full_score(matrix,edges):
    return int(full_character_scores(matrix,edges).sum())


def directed_support(matrix,adj):
    n,p=matrix.shape;parent,order=_orient(adj,n);message={}
    for a in reversed(order[1:]):
        if a < n: value=np.where(matrix[a]<0,3,1 << matrix[a]).astype(np.uint8)
        else:
            other=[b for b in adj[a] if b!=parent[a]]
            value=_combine(message[other[0],a],message[other[1],a])
        message[a,parent[a]]=value
    for a in order:
        for b in adj[a]:
            if b==parent[a]: continue
            if a < n:
                value=np.where(matrix[a]<0,3,1 << matrix[a]).astype(np.uint8)
            else:
                other=[c for c in adj[a] if c!=b]
                value=_combine(message[other[0],a],message[other[1],a])
            message[a,b]=value
    return message


def quartet_cost(a,b,c,d):
    """Fitch cost connecting four fixed subtree support arrays as ab|cd."""
    i1=a & b;i2=c & d
    s1=np.where(i1!=0,i1,a | b);s2=np.where(i2!=0,i2,c | d)
    return (i1==0).astype(np.int16)+(i2==0)+(s1 & s2==0)


def nni_neighbors(matrix,adj,current_score):
    """Return all 2(n-3) alternatives and their exact scores, vectorized."""
    n=matrix.shape[0];messages=directed_support(matrix,adj)
    move=[];a=[];b=[];c=[];d=[]
    for u in range(n,len(adj)):
        for v in sorted(adj[u]):
            if v <= u or v < n: continue
            left=sorted(w for w in adj[u] if w!=v)
            right=sorted(w for w in adj[v] if w!=u)
            move.append((u,v,left[1],right[0],right[1]))
            a.append(messages[left[0],u]);b.append(messages[left[1],u])
            c.append(messages[right[0],v]);d.append(messages[right[1],v])
    a,b,c,d=map(np.asarray,(a,b,c,d))
    old=quartet_cost(a,b,c,d).sum(axis=1)
    first=quartet_cost(a,c,b,d).sum(axis=1)-old+current_score
    second=quartet_cost(a,d,b,c).sum(axis=1)-old+current_score
    return move,np.column_stack((first,second)).reshape(-1)


def apply_move(adj,move,variant):
    u,v,b,c,d=move
    swap=c if variant==0 else d
    adj[u].remove(b);adj[u].append(swap);adj[u].sort()
    adj[v].remove(swap);adj[v].append(b);adj[v].sort()
    adj[b].remove(u);adj[b].append(v);adj[b].sort()
    adj[swap].remove(v);adj[swap].append(u);adj[swap].sort()


def split_set(edges,n_tips):
    """Canonical nontrivial unrooted splits as Python integer bit masks."""
    adj=validate_tree(edges,n_tips);parent,order=_orient(adj,n_tips)
    masks=[0]*len(adj);full=(1<<n_tips)-1;result=set()
    for a in reversed(order):
        mask=(1<<a) if a<n_tips else 0
        for b in adj[a]:
            if b != parent[a]: mask |= masks[b]
        masks[a]=mask
        if parent[a]>=0 and 1<mask.bit_count()<n_tips-1:
            result.add(min(mask,full^mask))
    if len(result)!=n_tips-3: raise AssertionError("Binary tree should have n-3 splits")
    return frozenset(result)


def hillclimb(matrix,start_edges,max_steps=200):
    x=np.asarray(matrix,dtype=np.int8)
    if x.ndim!=2 or x.shape[0]<4 or not np.isin(x,[-1,0,1]).all(): raise ValueError("Invalid masked binary matrix")
    adj=validate_tree(start_edges,len(x));score=full_score(x,start_edges)
    trajectory=[score];candidate_evaluations=0;reached_limit=False
    for step in range(max_steps):
        moves,scores=nni_neighbors(x,adj,score)
        candidate_evaluations+=len(scores)
        winner=int(np.argmin(scores))
        if scores[winner]>=score: break
        apply_move(adj,moves[winner//2],winner%2)
        score=int(scores[winner]);trajectory.append(score)
    else: reached_limit=True
    edges=edges_from_adj(adj)
    assert full_score(x,edges)==score
    _,scores=nni_neighbors(x,adj,score)
    local_optimum=bool(np.all(scores>=score))
    return {"edges":edges,"score":score,"steps":len(trajectory)-1,
            "score_trajectory":trajectory,"candidate_evaluations":candidate_evaluations,
            "local_optimum":local_optimum,"reached_step_limit":reached_limit and not local_optimum}


def search_matrix(matrix,starts,max_steps=200):
    """Return all DISTINCT lowest-score endpoints found across fixed starts."""
    found=[hillclimb(matrix,start,max_steps) for start in starts]
    minimum=min(r["score"] for r in found)
    retained={}
    for result in found:
        if result["score"]==minimum:
            splits=split_set(result["edges"],len(matrix))
            retained.setdefault(splits,result)
    endpoints=list(retained.values())
    return {"minimum_found_length":minimum,"endpoints":endpoints,"starts":found,
            "n_distinct_best_endpoints":len(endpoints),
            "n_starts":len(starts),
            "any_start_step_limit":any(r["reached_step_limit"] for r in found)}


def score_search(result,truth_edges,n_tips):
    """Post-search evaluation only; no truth-based choice among tied endpoints."""
    true=split_set(truth_edges,n_tips)
    estimated=[split_set(r["edges"],n_tips) for r in result["endpoints"]]
    recalls=np.array([len(s & true)/len(true) for s in estimated])
    rf=np.array([len(s ^ true)/(2*(n_tips-3)) for s in estimated])
    consensus=frozenset.intersection(*estimated)
    correct=len(consensus & true)
    return {"minimum_found_length":result["minimum_found_length"],
            "n_distinct_best_endpoints":len(estimated),
            "mean_normalized_RF":float(rf.mean()),"min_normalized_RF":float(rf.min()),"max_normalized_RF":float(rf.max()),
            "mean_split_recall":float(recalls.mean()),
            "truth_among_retained_endpoints":int(true in estimated),
            "consensus_n_splits":len(consensus),"consensus_resolution":len(consensus)/(n_tips-3),
            "consensus_split_recall":correct/len(true),
            "consensus_split_precision":correct/len(consensus) if consensus else np.nan,
            "any_start_step_limit":int(result["any_start_step_limit"]),
            "mean_accepted_moves":float(np.mean([r["steps"] for r in result["starts"]]))}


def enumerate_trees(n_tips):
    """All labelled unrooted binary trees, intended only for small-tree tests."""
    trees=[[(0,n_tips),(1,n_tips),(2,n_tips)]]
    for tip in range(3,n_tips):
        node=n_tips+tip-2;new=[]
        for edges in trees:
            for index,(a,b) in enumerate(edges):
                new.append(edges[:index]+edges[index+1:]+[(a,node),(b,node),(tip,node)])
        trees=new
    return [np.asarray(edges,dtype=np.int16) for edges in trees]
