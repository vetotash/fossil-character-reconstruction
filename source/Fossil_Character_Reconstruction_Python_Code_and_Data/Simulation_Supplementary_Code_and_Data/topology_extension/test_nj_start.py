import unittest
import numpy as np
from nj_start import matrix_distance, neighbor_join, matrix_nj_start, combine_endpoints
from nni_search import split_set, hillclimb


class NJStartTests(unittest.TestCase):
    def test_additive_quartets_and_tie_determinism(self):
        for paired in ((0, 1), (0, 2), (0, 3)):
            other = [i for i in range(4) if i not in paired]
            d = np.full((4, 4), 4.0); np.fill_diagonal(d, 0)
            d[paired[0], paired[1]] = d[paired[1], paired[0]] = 2
            d[other[0], other[1]] = d[other[1], other[0]] = 2
            split = min(sum(1 << i for i in paired), sum(1 << i for i in other))
            self.assertEqual(split_set(neighbor_join(d), 4), frozenset([split]))
        np.testing.assert_array_equal(neighbor_join(np.zeros((4, 4))), neighbor_join(np.zeros((4, 4))))

    def test_masked_distance_and_no_overlap(self):
        x = np.array([[0, -1, 1], [1, 0, -1], [-1, -1, -1], [0, 1, 1]])
        d, diagnostic = matrix_distance(x)
        self.assertEqual(d[0, 1], 1); self.assertEqual(d[0, 3], 0)
        self.assertEqual(d[1, 3], 1); self.assertEqual(diagnostic['zero_overlap_pairs'], 3)
        self.assertAlmostEqual(d[0, 2], 2 / 3)
        self.assertTrue(np.all(np.diag(d) == 0))

    def test_original_endpoints_retained_if_better_and_inputs_unchanged(self):
        x = np.array([[0, 0], [0, 0], [1, 1], [1, 1]], dtype=np.int8)
        original = x.copy(); start, _ = matrix_nj_start(x)
        one = hillclimb(x, start)
        combined = combine_endpoints([one, one], 4)
        self.assertEqual(combined['n_distinct_best_endpoints'], 1)
        self.assertEqual(combined['minimum_found_length'], 2)
        np.testing.assert_array_equal(x, original)


if __name__ == '__main__':
    unittest.main()
