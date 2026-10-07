from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from blocked_validation import (newick_descendants,select_clades,matched_tasks,
                                masked_input,load_fitters,known)


class BlockedValidationTests(unittest.TestCase):
    def test_newick_clades_preserve_source_labels(self):
        tips,clades=newick_descendants("[&U] (('a one':1,b:2)90:3,(c:1,d:1));")
        self.assertEqual(tips,['a one','b','c','d'])
        self.assertIn(('a one','b'),clades)
        self.assertIn(('c','d'),clades)

    def test_disjoint_selection_and_matched_random_masks(self):
        taxa=[f't{i}' for i in range(24)]
        matrix=np.tile(np.array(list('010101')), (24,1))
        clades=[tuple(taxa[a:a+4]) for a in range(0,16,4)]
        clades.append(tuple(taxa[:5]))  # overlaps the first clade.
        selected,audit=select_clades(matrix,taxa,list(range(1,7)),list(range(1,7)),clades,'toy')
        flattened=[t for c in selected for t in c['taxa']]
        self.assertEqual(len(flattened),len(set(flattened)))
        tasks=matched_tasks(matrix,taxa,selected,'toy')
        self.assertEqual(tasks,matched_tasks(matrix,taxa,selected,'toy'))
        for case in {t['case_id'] for t in tasks}:
            group=[t for t in tasks if t['case_id']==case]
            self.assertEqual(len(group),4)
            self.assertEqual(len({len(t['target_row_indices']) for t in group}),1)
            self.assertTrue(all(len(set(t['target_row_indices']))==t['heldout_count'] for t in group))

    def test_mask_removes_entire_clade_response_without_changing_other_characters(self):
        source=np.array([list('010'),list('110'),list('?00'),list('100'),list('010')])
        before=source.copy();masked,train=masked_input(source,0,[0,1,3])
        self.assertFalse(known(masked[[0,1,2,3],0]).any())
        np.testing.assert_array_equal(source,before)
        np.testing.assert_array_equal(masked[:,1:],source[:,1:])
        np.testing.assert_array_equal(train,[4])

    def test_archived_fitters_keep_unseen_heldout_state_out_of_training_alphabet(self):
        dp,mk,instrument=load_fitters()
        rng=np.random.default_rng(17)
        source=rng.integers(0,2,(24,6)).astype(str)
        source[:4,0]='2';matrix,train=masked_input(source,0,[0,1,2,3])
        methods,diag=instrument.instrumented_predict(dp,matrix,0,train,np.arange(4))
        d,zero=mk.excluding_distance(matrix,0,mk.distance_totals(matrix))
        edges,_=mk.neighbor_join(d)
        labels,prob,fit,diag=instrument.instrumented_mk(mk,edges,24,matrix[:,0],train,np.arange(4))
        methods['nj_mk']=(labels,prob)
        for labels,prob in methods.values():
            self.assertNotIn('2',labels)
            self.assertTrue(np.isfinite(prob).all())
            np.testing.assert_allclose(prob.sum(1),1)
        # Excluding the response makes NJ distances invariant to its labels.
        altered=matrix.copy();altered[:,0]='9'
        d2,_=mk.excluding_distance(altered,0,mk.distance_totals(altered))
        np.testing.assert_allclose(d,d2,rtol=0,atol=0)

    def test_archived_constant_fallback_remains_visible(self):
        dp,mk,instrument=load_fitters()
        source=np.tile(np.array(list('010101')),(24,1));source[:4,0]='1'
        matrix,train=masked_input(source,0,[0,1,2,3])
        methods,diag=instrument.instrumented_predict(dp,matrix,0,train,np.arange(4))
        for labels,prob in methods.values():
            self.assertEqual(labels.tolist(),['0'])
            np.testing.assert_array_equal(prob,np.ones((4,1)))
        self.assertTrue(all(d['constant_class_fallback'] for d in diag))


if __name__=='__main__':unittest.main(verbosity=2)
