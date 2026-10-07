import unittest
import numpy as np
import pandas as pd

from analyze_distributions import crps_empirical, distribution_metrics, paired_contrasts, state_group_masks


def crps_by_cdf_integration(values, truth):
    # Independent piecewise-constant CDF integral, not the pairwise identity.
    breaks = np.unique(np.r_[values, truth])
    total = 0.
    for low, high in zip(breaks[:-1], breaks[1:]):
        midpoint = (low+high)/2
        difference = np.mean(values <= midpoint)-int(truth <= midpoint)
        total += (high-low)*difference**2
    return total


class CRPSTests(unittest.TestCase):
    def test_hand_calculated_cases(self):
        self.assertEqual(crps_empirical([0, 2], 1), .5)
        self.assertEqual(crps_empirical([0, 2], 1, fair=True), 0.)
        self.assertEqual(crps_empirical([3], 7), 4.)
        self.assertEqual(crps_empirical([3, 3, 3], 7), 4.)
        self.assertEqual(crps_empirical([1, 1], 1), 0.)

    def test_matches_independent_CDF_integral_and_direct_pairwise_formula(self):
        rng = np.random.default_rng(20261004)
        for size in (2, 3, 9, 31):
            for _ in range(8):
                values = rng.integers(-5, 12, size).astype(float)
                truth = float(rng.uniform(-8, 15))
                got = crps_empirical(values, truth)
                direct = np.abs(values-truth).mean()-np.abs(values[:, None]-values[None, :]).mean()/2
                self.assertAlmostEqual(got, direct, places=12)
                self.assertAlmostEqual(got, crps_by_cdf_integration(values, truth), places=12)
                self.assertGreaterEqual(got, -1e-12)

    def test_translation_scaling_and_oracle_centering(self):
        values = np.array([-2, 0, 1, 4, 10.])
        self.assertAlmostEqual(crps_empirical(values, 7), crps_empirical(values+40, 47))
        self.assertAlmostEqual(crps_empirical(values*3, 21), 3*crps_empirical(values, 7))
        original = distribution_metrics(values, 7)
        oracle = distribution_metrics(values-original['signed_bias'], 7)
        self.assertAlmostEqual(oracle['signed_bias'], 0.)
        self.assertAlmostEqual(oracle['width95'], original['width95'])

    def test_invalid_arguments(self):
        for bad in ([], [np.nan], [[0, 1]]):
            with self.assertRaises(ValueError): crps_empirical(bad, 0)
        with self.assertRaises(ValueError): crps_empirical([0], 0, fair=True)

    def test_unseen_constant_zero_groups_are_disjoint(self):
        masked = np.array([[0, 0], [0, 1], [-1, -1], [-1, -1]])
        truth = np.array([[0, 0], [0, 1], [1, 1], [0, 0]])
        target = np.argwhere(masked == -1)
        groups, constant, unseen = state_group_masks(masked, truth, target, np.array([0., 0., 1., .5]))
        self.assertEqual({k: int(v.sum()) for k, v in groups.items()}, {
            'training_unseen_state': 1, 'constant_training_seen_state': 1,
            'nonconstant_training_zero_true_probability': 1,
            'nonconstant_training_positive_true_probability': 1})
        np.testing.assert_array_equal(constant, [True, False])
        np.testing.assert_array_equal(unseen, [True, False, False, False])

    def test_paired_MCSE_uses_within_matrix_differences(self):
        from analyze_distributions import SCHEMES
        rows = []
        for replicate in range(3):
            for index, scheme in enumerate(SCHEMES):
                rows.append({'matrix_id':str(replicate), 'scenario_id':'s00', 'scheme':scheme,
                             'crps':100*replicate+index})
        contrast = paired_contrasts(pd.DataFrame(rows), ['crps'])
        self.assertTrue(np.all(contrast.paired_MCSE == 0))
        self.assertTrue(np.all(contrast.n_paired_matrices == 3))


if __name__ == '__main__':
    unittest.main(verbosity=2)
