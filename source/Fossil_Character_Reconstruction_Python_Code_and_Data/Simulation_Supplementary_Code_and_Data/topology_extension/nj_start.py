"""Data-derived NJ initialization for the explicitly post hoc search amendment."""
from __future__ import annotations
import numpy as np
from nni_search import validate_tree, split_set


def matrix_distance(matrix):
    """Mismatch proportion across jointly observed characters; -1 is unknown.

    This is a starting-tree distance, not an evolutionary-distance estimate.
    The production matrices have eight complete anchors, so every pair has
    overlap. The general zero-overlap fallback is the mean defined off-diagonal
    distance, or 0.5 if no pair is defined; such pairs are explicitly counted.
    """
    x = np.asarray(matrix, dtype=np.int8)
    if x.ndim != 2 or len(x) < 4 or not np.isin(x, [-1, 0, 1]).all():
        raise ValueError("Expected a binary matrix with -1 unknown")
    valid = (x[:, None, :] >= 0) & (x[None, :, :] >= 0)
    denominator = valid.sum(axis=2)
    numerator = ((x[:, None, :] != x[None, :, :]) & valid).sum(axis=2)
    d = np.divide(numerator, denominator, out=np.full(denominator.shape, np.nan), where=denominator > 0)
    upper = np.triu(np.ones(d.shape, dtype=bool), 1)
    defined = upper & np.isfinite(d)
    fallback = float(d[defined].mean()) if defined.any() else 0.5
    zero = int((upper & ~np.isfinite(d)).sum())
    d[~np.isfinite(d)] = fallback
    np.fill_diagonal(d, 0.0)
    return d, {"zero_overlap_pairs": zero, "zero_overlap_fallback": fallback}


def neighbor_join(distance):
    """NJ topology, with the historical implementation's deterministic Q ties.

    Branch lengths are not used by unordered parsimony. The topology arithmetic
    and row-major upper-triangle tie order match methods/mk_baseline.py.
    """
    d = np.array(distance, dtype=float, copy=True)
    n = len(d)
    if d.shape != (n, n) or n < 4 or not np.isfinite(d).all() or not np.allclose(d, d.T):
        raise ValueError("Finite symmetric distance matrix required")
    active = list(range(n)); edges = []; next_node = n
    while len(active) > 2:
        m = len(active); sums = d.sum(axis=1)
        q = (m - 2) * d - sums[:, None] - sums[None, :]
        q[np.tril_indices(m)] = np.inf
        i, j = np.unravel_index(np.argmin(q), q.shape)
        edges.extend([(next_node, active[i]), (next_node, active[j])])
        keep = [v for v in range(m) if v not in (i, j)]
        new_distance = (d[i, keep] + d[j, keep] - d[i, j]) / 2
        sub = d[np.ix_(keep, keep)]
        d = np.zeros((m - 1, m - 1))
        d[:-1, :-1] = sub; d[-1, :-1] = new_distance; d[:-1, -1] = new_distance
        active = [active[v] for v in keep] + [next_node]; next_node += 1
    edges.append((active[0], active[1]))
    edges = np.asarray(edges, dtype=np.int16)
    validate_tree(edges, n)
    return edges


def matrix_nj_start(matrix):
    distance, diagnostic = matrix_distance(matrix)
    return neighbor_join(distance), diagnostic


def combine_endpoints(found, n_tips):
    """Retain distinct minimum-score endpoints; never takes a truth argument."""
    minimum = min(item["score"] for item in found)
    retained = {}
    for item in found:
        if item["score"] == minimum:
            retained.setdefault(split_set(item["edges"], n_tips), item)
    endpoints = list(retained.values())
    return {"minimum_found_length": minimum, "endpoints": endpoints, "starts": found,
            "n_distinct_best_endpoints": len(endpoints), "n_starts": len(found),
            "any_start_step_limit": any(item["reached_step_limit"] for item in found)}
