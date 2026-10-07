"""An exact explanatory counterexample, not a new fitted prediction method.

For ((A,B),(C,D)), observed B=0 and D=1, unknown A and C each have
P(0)=P(1)=1/2. Compare three joint distributions with identical marginals.
"""
from pathlib import Path
import itertools, json, csv
from fractions import Fraction

HERE=Path(__file__).resolve().parent/'dependence_example'

def score(a,c):
    # All states at the three internal nodes of a rooted display of the quartet.
    # Rooting introduces a degree-two node but does not change parsimony length.
    vals=[]
    for left,right,root in itertools.product([0,1],repeat=3):
        vals.append((left!=a)+(left!=0)+(right!=c)+(right!=1)
                    +(root!=left)+(root!=right))
    length=min(vals)
    minimum=1  # Observed B=0,D=1 guarantee both states in every completion.
    return length,minimum,length-minimum

def main():
    HERE.mkdir(exist_ok=True)
    states=list(itertools.product([0,1],repeat=2))
    distributions={
      'same_states':{(0,0):Fraction(1,2),(1,1):Fraction(1,2)},
      'independent':{s:Fraction(1,4) for s in states},
      'opposite_states':{(0,1):Fraction(1,2),(1,0):Fraction(1,2)}}
    rows=[];summaries={}
    for name,probs in distributions.items():
        marginals={tip:sum(p for s,p in probs.items() if s[i]==1)
                   for i,tip in enumerate(['A','C'])}
        assert all(p==Fraction(1,2) for p in marginals.values())
        expected=sum(p*score(*s)[2] for s,p in probs.items())
        summaries[name]={'P_A1':str(marginals['A']),'P_C1':str(marginals['C']),
                         'expected_added_excess_steps':str(expected)}
        for a,c in states:
            length,minimum,excess=score(a,c)
            rows.append({'distribution':name,'A':a,'B':0,'C':c,'D':1,
                         'probability':str(probs.get((a,c),Fraction(0))),
                         'tree_steps':length,'minimum_steps':minimum,
                         'added_excess_steps':excess})
    with (HERE/'exact_assignments.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    result={'tree':'((A,B),(C,D))','source_states':{'A':'?','B':'0','C':'?','D':'1'},
            'source_steps':1,'source_minimum_steps':1,'source_excess_steps':0,
            'summaries':summaries,'enumeration':'all 2^3 internal-node assignments for each completion',
            'status':'Explanatory construction, not a fossil observation or novelty claim'}
    (HERE/'exact_result.json').write_text(json.dumps(result,indent=2)+'\n')
    (HERE/'EXPLANATION.txt').write_text('''Why cell probabilities do not determine a downstream compatibility distribution

Take the four-tip tree ((A,B),(C,D)). B is observed in state0 and D in
state1. A and C are missing. Both predicted marginal probabilities are
one half for each state. The source needs one step and has minimum1.

If A and C always receive the same state, complete patterns0001 and1011
have equal probability. Both need one step, so added excess steps are0.
If they are independent, all four completions have probability1/4.
Only1001 needs two steps, yielding mean added excess steps1/4.
If A and C always receive opposite states,0011 and1001 each occur with
probability1/2, yielding mean added excess steps1/2.

Every scheme has exactly the same per-cell state probabilities. Both
states already occur among observed taxa, so the minimum is1 throughout.
This is a direct finite counterexample: marginal probabilities alone do
not identify the expected added excess steps on even a fixed tree.
The script enumerates every internal-node assignment with exact rational
probabilities. These synthetic values are not included in the empirical
fossil results, and the example is not presented as a novel theorem.

Across independent character scoring blocks, expectation of total cost is
the sum of expected character costs by linearity of expectation. Correlation
between characters can alter uncertainty in the total but not its mean
when the per-character joint distributions are held fixed. Dependence
among taxa within a character can alter both the mean and distribution.
This distinction motivates the conditional joint Mk sensitivity without
claiming that it solves every error in historical fossil prediction.
''')
    print(json.dumps(summaries))

if __name__=='__main__':main()
