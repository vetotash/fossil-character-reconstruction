import unittest
import itertools
import numpy as np
from fixed_target_design import randomize_observed_fixed_target, masked_input


class FixedTargetTests(unittest.TestCase):
    def setUp(self):
        self.observed = np.array([[1,0,1],[0,1,0],[1,0,0],[0,1,1],[0,0,0]], dtype=bool)
        self.target = np.array([[0,1,0],[1,0,0],[0,0,1],[0,0,0],[1,0,0]], dtype=bool)

    def test_preserves_target_and_column_counts_without_mutation(self):
        old_o, old_t = self.observed.copy(), self.target.copy()
        new = randomize_observed_fixed_target(self.observed, self.target, [7,2,4])
        np.testing.assert_array_equal(new.sum(0), old_o.sum(0))
        self.assertFalse((new & self.target).any())
        np.testing.assert_array_equal(self.observed, old_o)
        np.testing.assert_array_equal(self.target, old_t)
        np.testing.assert_array_equal(new, randomize_observed_fixed_target(old_o, old_t,[7,2,4]))

    def test_all_admissible_layouts_reachable_small_example(self):
        # One fixed target, one O and two U: exactly three conditional layouts.
        obs=np.array([[1],[0],[0],[0]],bool)
        target=np.array([[0],[0],[0],[1]],bool)
        patterns={tuple(randomize_observed_fixed_target(obs,target,[i]).ravel()) for i in range(100)}
        self.assertEqual(patterns,{(True,False,False,False),(False,True,False,False),(False,False,True,False)})

    def test_hidden_truth_never_changes_fitting_input(self):
        x=np.zeros(self.observed.shape,dtype=np.int8)
        changed=x.copy(); changed[~self.observed]=1
        a,keys=masked_input(x,self.observed,self.target)
        b,newkeys=masked_input(changed,self.observed,self.target)
        np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(keys,newkeys)
        self.assertTrue(np.all(a[~self.observed]==-1))
        np.testing.assert_array_equal(keys,np.argwhere(self.target))

    def test_fixed_target_does_not_imply_fixed_D_truth(self):
        # Independent internal-state enumeration on a five-tip binary tree.
        # Tip 0 remains the sole target; three O tips and one U tip per arm.
        edges=[(0,5),(1,5),(5,6),(2,6),(6,7),(3,7),(4,7)]
        truth=np.array([0,0,1,0,1])
        def length_and_minimum(pattern):
            length=99
            for internal in itertools.product((0,1),repeat=3):
                labels={5+i:v for i,v in enumerate(internal)}
                labels.update({i:int(v) for i,v in enumerate(pattern) if v>=0})
                score=sum(labels[a]!=labels[b] for a,b in edges if a in labels and b in labels)
                length=min(length,score)
            minimum=int(0 in pattern and 1 in pattern)
            return length,minimum
        results=[]
        for observed_rows in [(1,2,3),(2,3,4)]:
            baseline=np.full(5,-1); baseline[list(observed_rows)]=truth[list(observed_rows)]
            completed=baseline.copy(); completed[0]=truth[0]
            l0,m0=length_and_minimum(baseline)
            l1,m1=length_and_minimum(completed)
            results.append((l1-l0)-(m1-m0))
        self.assertEqual(results,[0,1])

    def test_invalid_overlap_rejected(self):
        with self.assertRaises(ValueError):
            randomize_observed_fixed_target(self.observed,self.observed,[0])

    def test_observed_values_can_change_with_fixed_target(self):
        x=np.arange(15).reshape(5,3)%2
        found=False
        original,_=masked_input(x,self.observed,self.target)
        for seed in range(20):
            new=randomize_observed_fixed_target(self.observed,self.target,[seed])
            changed,keys=masked_input(x,new,self.target)
            if not np.array_equal(changed,original):
                found=True; break
        self.assertTrue(found)
        np.testing.assert_array_equal(keys,np.argwhere(self.target))


if __name__=='__main__':
    unittest.main()
