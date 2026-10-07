"""Engineering checks only; these synthetic values are not scientific replicates."""
import unittest
import numpy as np
import pandas as pd
from score_draws import SCHEMES, METRICS, paired_layout_contrasts, paired_method_contrasts, validate_registry
from validated_crps import crps_empirical, distribution_metrics


class ScoreTests(unittest.TestCase):
    def test_crps_against_independent_cdf_integral(self):
        rng = np.random.default_rng(21803)
        for m in [2, 7, 31, 512]:
            x = rng.integers(-20, 50, m).astype(float)
            y = 10.4
            breaks = np.unique(np.r_[x, y])
            mids = (breaks[:-1] + breaks[1:]) / 2
            integral = sum((hi-lo) * (np.mean(x <= mid)-int(y <= mid))**2
                           for lo, hi, mid in zip(breaks[:-1], breaks[1:], mids))
            self.assertAlmostEqual(crps_empirical(x, y), integral, places=11)

    def test_own_arm_truth_is_required(self):
        empirical = distribution_metrics(np.array([9., 10., 11.]), 10.)
        permuted = distribution_metrics(np.array([99., 100., 101.]), 100.)
        self.assertAlmostEqual(empirical['crps_empirical'], permuted['crps_empirical'])
        self.assertEqual(empirical['signed_bias'], permuted['signed_bias'])
        wrong = distribution_metrics(np.array([99., 100., 101.]), 10.)
        self.assertGreater(wrong['crps_empirical'], permuted['crps_empirical'])

    def test_pairing_uses_latent_ids_and_paired_variance(self):
        rows = []
        for i in range(30):
            for arm, shift in [('empirical', 3.), ('permuted', 0.)]:
                for method, scheme in enumerate(SCHEMES):
                    rows.append(dict(condition_id='c', panel='panel', q=.03, latent_id=f'id{i}',
                                     arm=arm, scheme=scheme,
                                     **{metric:1000.*i+shift+method for metric in METRICS}))
        frame = pd.DataFrame(rows).sample(frac=1, random_state=1)
        contrasts, _ = paired_layout_contrasts(frame)
        self.assertTrue(np.all(contrasts.mean_difference == 3))
        self.assertTrue(np.all(contrasts.paired_MCSE == 0))
        self.assertTrue(np.all(contrasts.n_paired_matrices == 30))
        method_contrasts = paired_method_contrasts(frame)
        self.assertTrue(np.all(method_contrasts.mean_difference > 0))
        self.assertTrue(np.all(method_contrasts.paired_MCSE == 0))

    def test_missing_or_duplicate_pair_fails(self):
        with self.assertRaises(ValueError):
            validate_registry(pd.DataFrame())
        base = dict(condition_id='c', panel='panel', q=.03, latent_id='one', arm='empirical', scheme='mode')
        rows = [{**base, **{metric:1 for metric in METRICS}},
                {**base, **{metric:2 for metric in METRICS}}]
        with self.assertRaises(ValueError):
            paired_layout_contrasts(pd.DataFrame(rows))


if __name__ == '__main__':
    unittest.main(verbosity=2)
