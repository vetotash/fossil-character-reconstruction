import itertools, unittest
import numpy as np
from audit_fixed_target import independent_fitch,completed_totals
from run_fixed_target import crps

class IndependentScoringTests(unittest.TestCase):
    def test_all_ternary_patterns_against_internal_state_enumeration(self):
        edges=np.array([(0,4),(1,4),(4,5),(2,5),(3,5)])
        patterns=np.array(list(itertools.product((-1,0,1),repeat=4)))
        scored,mins=independent_fitch(patterns,edges,4)
        expected=[]
        for pattern in patterns:
            values=[]
            for a,b in itertools.product((0,1),repeat=2):
                cost=int(a!=b)
                for tip,parentstate in enumerate((a,a,b,b)):
                    if pattern[tip]>=0:cost+=int(pattern[tip]!=parentstate)
                values.append(cost)
            expected.append(min(values))
        np.testing.assert_array_equal(scored,expected)
        np.testing.assert_array_equal(mins,[(0 in p and 1 in p) for p in patterns])

    def test_target_completion_keeps_other_unknown_cells(self):
        edges=np.array([(0,4),(1,4),(4,5),(2,5),(3,5)])
        masked=np.array([[-1],[0],[-1],[1]],dtype=np.int8)
        target=np.array([[1],[0],[0],[0]],dtype=bool)
        l,m=completed_totals(masked,target,[[0],[1]],edges,4)
        np.testing.assert_array_equal(l,[1,1]);np.testing.assert_array_equal(m,[1,1])

    def test_crps_sorted_pair_formula_against_direct_pairwise(self):
        for x,y in [([0,2,2,7],3),([1,1,1],1),([1,4,8,9],-2)]:
            a=np.array(x);n=len(a);pair=np.abs(a[:,None]-a[None,:]).sum()
            self.assertAlmostEqual(crps(a,y),np.abs(a-y).mean()-pair/(2*n*n))
            self.assertAlmostEqual(crps(a,y,True),np.abs(a-y).mean()-pair/(2*n*(n-1)))

if __name__=='__main__':unittest.main()
