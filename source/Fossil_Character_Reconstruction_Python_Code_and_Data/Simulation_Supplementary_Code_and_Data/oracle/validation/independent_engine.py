"""Independent finite-state checks; no production Fitch or pruning imports.

This file only defines functions. Importing it does not generate scientific
samples. Binary tip patterns use the explicitly supplied array ordering.
"""

import itertools
import numpy as np


def bits(count):
    """Lexicographic binary patterns, first column most significant."""
    return ((np.arange(2 ** count, dtype=np.uint64)[:, None]
             >> np.arange(count - 1, -1, -1, dtype=np.uint64)) & 1).astype(np.int8)


def enumerate_internal_states(patterns, edges, tips, rate):
    """Return P(tip pattern) and its minimum edge changes by direct enumeration.

    Uniform stationary root gives probability 1/2 times product of all
    unoriented edge transition factors for a complete node assignment.
    For a binary symmetric stationary process this does not depend on root.
    """
    patterns = np.asarray(patterns, dtype=np.int8)
    nodes = list(dict.fromkeys(x for edge in edges for x in edge))
    internals = [x for x in nodes if x not in tips]
    assert len(internals) == len(tips) - 2
    assert len(edges) == 2 * len(tips) - 3
    assert patterns.shape == (2 ** len(tips), len(tips))
    internal = bits(len(internals))
    lookup = {node: (True, i) for i, node in enumerate(tips)}
    lookup.update({node: (False, i) for i, node in enumerate(internals)})
    changes = np.zeros((len(patterns), len(internal)), dtype=np.int16)
    for a, b in edges:
        ta, ia = lookup[a]
        tb, ib = lookup[b]
        va = patterns[:, ia, None] if ta else internal[None, :, ia]
        vb = patterns[:, ib, None] if tb else internal[None, :, ib]
        changes += va != vb
    flip = -np.expm1(-2 * rate) / 2
    probabilities = (0.5 * flip ** changes
                     * (1 - flip) ** (len(edges) - changes)).sum(axis=1)
    return probabilities, changes.min(axis=1)


def compatible(patterns, observed):
    observed = np.asarray(observed, dtype=np.int8)
    known = observed >= 0
    return np.all(patterns[:, known] == observed[None, known], axis=1)


def conditions(patterns, probabilities, costs, observed):
    """Independent posterior, product distributions and excess costs."""
    keep = compatible(patterns, observed)
    states = patterns[keep]
    posterior = probabilities[keep].copy()
    posterior /= posterior.sum()
    observed = np.asarray(observed, dtype=np.int8)
    missing = np.flatnonzero(observed < 0)
    marginal_one = posterior @ states[:, missing]
    product = np.prod(np.where(states[:, missing], marginal_one,
                               1 - marginal_one), axis=1)
    squared = marginal_one ** 2 / (marginal_one ** 2 + (1 - marginal_one) ** 2)
    power2 = np.prod(np.where(states[:, missing], squared, 1 - squared), axis=1)
    partial_cost = int(costs[keep].min())
    observed_states = np.unique(observed[observed >= 0])
    partial_minimum = int(len(observed_states) == 2)
    complete_minima = np.any(states != states[:, :1], axis=1).astype(np.int16)
    increments = costs[keep] - complete_minima - partial_cost + partial_minimum
    assert increments.min() >= 0
    hard = observed.copy()
    hard[missing] = (marginal_one > .5 + 1e-12).astype(np.int8)
    hard_index = np.flatnonzero(np.all(states == hard, axis=1))
    assert len(hard_index) == 1
    return {
        'support': states, 'oracle': posterior, 'product': product, 'power2': power2,
        'marginal_one': marginal_one, 'D': increments,
        'partial_cost': partial_cost, 'partial_minimum': partial_minimum,
        'hard': hard, 'hard_D': int(increments[hard_index[0]])}


def character_pmf(increments, probabilities):
    return np.bincount(np.asarray(increments, dtype=np.int64),
                       weights=probabilities, minlength=int(max(increments)) + 1)


def convolve_pmfs(character_distributions):
    total = np.array([1.0])
    for mass in character_distributions:
        # Straight convolution deliberately avoids a production FFT or scipy.
        next_total = np.zeros(len(total) + len(mass) - 1)
        for shift, value in enumerate(mass):
            next_total[shift:shift + len(total)] += value * total
        total = next_total
    return total


def summarize_distribution(pmf):
    pmf = np.asarray(pmf, dtype=float)
    cdf = np.cumsum(pmf)
    lo, hi = (int(np.searchsorted(cdf, p, side='left')) for p in (.025, .975))
    return {'mass': float(pmf.sum()),
            'mean': float(np.arange(len(pmf)) @ pmf),
            'lower': lo, 'upper': hi}


def oracle_interval_mass(oracle_pmf, lower, upper):
    return float(np.asarray(oracle_pmf)[lower:upper + 1].sum())


def normalized_rf(splits_a, splits_b, n_tips):
    return len(set(splits_a).symmetric_difference(set(splits_b))) / (2 * (n_tips - 3))
