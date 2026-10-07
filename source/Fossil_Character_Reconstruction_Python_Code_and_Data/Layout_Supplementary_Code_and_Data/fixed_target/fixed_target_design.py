"""Truth-blind fixed-target O/U randomization and fitting-input projection.

Masks use boolean arrays. Observed and target are disjoint; U is their complement.
No latent state is an input to the randomization itself.
"""
from __future__ import annotations
import numpy as np

PERMUTATION_SEED = 2026100420
PROPAGATION_SEED = 2026100421
N_DRAWS = 512


def validate_masks(observed, target):
    observed = np.asarray(observed)
    target = np.asarray(target)
    if observed.dtype != np.bool_ or target.dtype != np.bool_:
        raise TypeError('Observed and target masks must have boolean dtype')
    if observed.ndim != 2 or observed.shape != target.shape:
        raise ValueError('Observed and target masks must have matching 2D shape')
    if np.any(observed & target):
        raise ValueError('Observed and target masks must be disjoint')
    return observed, target


def randomize_observed_fixed_target(observed, target, seed):
    """Uniformly permute O/U labels independently per column with T fixed."""
    observed, target = validate_masks(observed, target)
    randomized = observed.copy()
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    for j in range(observed.shape[1]):
        rows = np.flatnonzero(~target[:, j])
        randomized[rows, j] = observed[rng.permutation(rows), j]
    assert not np.any(randomized & target)
    assert np.array_equal(randomized.sum(axis=0), observed.sum(axis=0))
    return randomized


def masked_input(latent, observed, target):
    """Project only O values; return row-major target keys, never target labels."""
    observed, target = validate_masks(observed, target)
    latent = np.asarray(latent)
    if latent.shape != observed.shape or not np.isin(latent, [0, 1]).all():
        raise ValueError('Latent input must be a matching binary array')
    masked = np.full(latent.shape, -1, dtype=np.int8)
    masked[observed] = latent[observed]
    return masked, np.argwhere(target).astype(np.int16)
