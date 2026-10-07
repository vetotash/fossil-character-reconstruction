import copy
import itertools
import unittest
import numpy as np
from nni_search import (validate_tree,random_start_tree,full_score,full_character_scores,
                       nni_neighbors,apply_move,edges_from_adj,search_matrix,split_set,
                       score_search,enumerate_trees)


def brute_score(matrix,edges):
    n,p=matrix.shape;nn=int(edges.max())+1;total=0
    for j in range(p):
        unknown=np.flatnonzero(matrix[:,j]<0)
        best=np.inf
        for bits in itertools.product((0,1),repeat=nn-n+len(unknown)):
            states=np.r_[matrix[:,j],bits[:nn-n]].copy()
            states[unknown]=bits[nn-n:]
            best=min(best,int(np.sum(states[edges[:,0]]!=states[edges[:,1]])))
        total+=best
    return int(total)


class NNITests(unittest.TestCase):
    def test_all_four_tip_partial_patterns_match_bruteforce(self):
        matrix=np.array(list(itertools.product((-1,0,1),repeat=4)),dtype=np.int8).T
        for edges in enumerate_trees(4):
            expected=[brute_score(matrix[:,j:j+1],edges) for j in range(matrix.shape[1])]
            np.testing.assert_array_equal(full_character_scores(matrix,edges),expected)

    def test_every_NNI_score_matches_full_recalculation(self):
        rng=np.random.default_rng(62)
        for n in (5,6,9,48):
            matrix=rng.integers(-1,2,(n,14),dtype=np.int8)
            tree=random_start_tree(n,88)
            adj=validate_tree(tree,n)
            moves,scores=nni_neighbors(matrix,adj,full_score(matrix,tree))
            self.assertEqual(len(scores),2*(n-3))
            for index,expected in enumerate(scores):
                candidate=copy.deepcopy(adj)
                apply_move(candidate,moves[index//2],index%2)
                self.assertEqual(full_score(matrix,edges_from_adj(candidate)),expected)

    def test_tree_enumeration_unique_split_sets(self):
        for n,expected in ((4,3),(5,15),(6,105)):
            trees=enumerate_trees(n)
            self.assertEqual(len(trees),expected)
            self.assertEqual(len({split_set(t,n) for t in trees}),expected)

    def test_all_ternary_single_character_five_tip_searches_are_exact(self):
        trees=enumerate_trees(5)
        patterns=np.array(list(itertools.product((-1,0,1),repeat=5)),np.int8).T
        scores=np.array([full_character_scores(patterns,t) for t in trees])
        for j in range(patterns.shape[1]):
            result=search_matrix(patterns[:,j:j+1],trees[:3])
            self.assertEqual(result['minimum_found_length'],scores[:,j].min())
            self.assertTrue(all(r['local_optimum'] for r in result['starts']))

    def test_tied_endpoint_average_does_not_select_closest_to_truth(self):
        matrix=np.zeros((5,3),dtype=np.int8)
        trees=enumerate_trees(5)
        result=search_matrix(matrix,[trees[0],trees[1],trees[2],trees[0]])
        self.assertEqual(result['n_distinct_best_endpoints'],3)
        assessed=score_search(result,trees[0],5)
        expected=np.mean([len(split_set(t,5)^split_set(trees[0],5))/4 for t in trees[:3]])
        self.assertEqual(assessed['mean_normalized_RF'],expected)
        self.assertGreater(assessed['mean_normalized_RF'],assessed['min_normalized_RF'])

    def test_source_immutability_determinism_and_strict_improvement(self):
        rng=np.random.default_rng(66)
        matrix=rng.integers(-1,2,(9,10),dtype=np.int8)
        starts=[random_start_tree(9,s) for s in (11,12,13)]
        saved=matrix.copy();saved_starts=[t.copy() for t in starts]
        first=search_matrix(matrix,starts);second=search_matrix(matrix,starts)
        self.assertEqual(first['minimum_found_length'],second['minimum_found_length'])
        self.assertEqual([split_set(r['edges'],9) for r in first['endpoints']],
                         [split_set(r['edges'],9) for r in second['endpoints']])
        for result in first['starts']:
            self.assertTrue(np.all(np.diff(result['score_trajectory'])<0))
        np.testing.assert_array_equal(matrix,saved)
        for t,s in zip(starts,saved_starts): np.testing.assert_array_equal(t,s)


if __name__=='__main__':unittest.main(verbosity=2)
