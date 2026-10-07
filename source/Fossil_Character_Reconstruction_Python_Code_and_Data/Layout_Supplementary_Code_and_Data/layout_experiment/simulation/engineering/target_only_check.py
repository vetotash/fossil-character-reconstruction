"""Small non-study fixture checks before scientific generation."""
import itertools
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from experiment import score_patterns, completion_costs, total_costs
from fit_worker import fit_targets, joint_targets, sample_mk_joint

edges = np.array([[0,4],[1,4],[4,5],[2,5],[3,5]])
patterns = np.asarray(list(itertools.product((-1,0,1), repeat=4)))
brute=[]
for tips in patterns:
    scores=[]
    for internal in itertools.product((0,1), repeat=2):
        unknown=np.flatnonzero(tips==-1)
        for assign in itertools.product((0,1),repeat=len(unknown)):
            leaves=tips.copy(); leaves[unknown]=assign
            states=np.r_[leaves, internal]
            scores.append(sum(states[a]!=states[b] for a,b in edges))
    brute.append(min(scores))
assert np.array_equal(score_patterns(patterns,edges),brute)
masked=np.array([[0,-1,-1],[1,0,-1],[-1,1,0],[-1,-1,1]],dtype=np.int8)
target=np.zeros_like(masked,dtype=bool); target[0,1]=True; target[3,0]=True
states=np.asarray(list(itertools.product((0,1),repeat=2)),dtype=np.uint8)
l,m=completion_costs(masked,target,states,edges)
for k,row in enumerate(states):
    completed=masked.copy(); completed[target]=row
    assert np.array_equal(completed[(masked==-1)&~target],masked[(masked==-1)&~target])
    assert tuple(total_costs(completed,edges))==(l[k],m[k])
fitted=fit_targets(masked,target); assert not fitted['failures']
draws=joint_targets(masked,target,fitted['models'],64,170001)
cols=np.unique(np.argwhere(target)[:,1]); strings=np.where(masked==-1,'?',masked.astype(str))[:,cols]
all_unknown=np.argwhere(strings=='?'); keep=target[all_unknown[:,0],cols[all_unknown[:,1]]]
all_draws=sample_mk_joint(strings,[fitted['models'][j] for j in cols],64,170001)
assert np.array_equal(draws,all_draws[:,keep])
result={'status':'PASS','ternary_patterns_vs_brute_force':len(patterns),'target_only_completions':len(states),
        'joint_subset_byte_identical':True,'production_generated':False}
(Path(__file__).parent/'TARGET_ONLY_CHECK.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
