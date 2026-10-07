"""Independent Round 16 scoring and summary checks.

No production generator, fitter, likelihood, score, or summarizer is imported.
Binary Fitch bit sets provide an algorithm independent of the production
Sankoff dynamic program; brute-force internal-state enumeration checks toys.
"""
from __future__ import annotations

import itertools
import math
import numpy as np


def orient_edges(edges, n_tips):
    edges = np.asarray(edges, dtype=int)
    nodes = sorted(set(edges.ravel().tolist()))
    if nodes != list(range(len(nodes))):
        raise AssertionError("Node IDs must be contiguous")
    adj = [[] for _ in nodes]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    assert len(edges) == len(nodes) - 1
    assert all(len(adj[t]) == 1 for t in range(n_tips))
    assert all(len(adj[t]) == 3 for t in range(n_tips, len(nodes)))
    root = n_tips
    parent = {root: None}
    order = [root]
    for a in order:
        for b in adj[a]:
            if b == parent[a]:
                continue
            assert b not in parent
            parent[b] = a
            order.append(b)
    assert len(order) == len(nodes)
    children = {a: [b for b in adj[a] if parent.get(b) == a] for a in nodes}
    return root, order, children


def fitch_character_lengths(patterns, edges):
    """Lengths for N patterns × n labelled tips; -1 is unknown."""
    patterns = np.asarray(patterns, dtype=np.int8)
    assert patterns.ndim == 2 and np.isin(patterns, [-1, 0, 1]).all()
    n_patterns, n_tips = patterns.shape
    root, order, children = orient_edges(edges, n_tips)
    states = np.empty((len(order), n_patterns), dtype=np.uint8)
    states[:n_tips] = np.where(patterns.T == -1, 3, 1 << patterns.T)
    scores = np.zeros(n_patterns, dtype=np.int64)
    for a in reversed(order):
        if a < n_tips:
            continue
        child = children[a]
        merged = states[child[0]].copy()
        for b in child[1:]:
            inter = merged & states[b]
            disjoint = inter == 0
            scores += disjoint
            merged = np.where(disjoint, merged | states[b], inter)
        states[a] = merged
    return scores


def matrix_lengths(matrices, edges):
    """Return L and completion-specific M for (..., tips, chars)."""
    matrices = np.asarray(matrices, dtype=np.int8)
    assert matrices.ndim >= 2
    n_tips, n_chars = matrices.shape[-2:]
    leading = matrices.shape[:-2]
    patterns = np.swapaxes(matrices, -1, -2).reshape(-1, n_tips)
    per = fitch_character_lengths(patterns, edges).reshape(leading + (n_chars,))
    minima = (np.any(matrices == 0, axis=-2) & np.any(matrices == 1, axis=-2)).sum(-1)
    return per.sum(-1), minima


def brute_force_lengths(patterns, edges):
    patterns = np.asarray(patterns, dtype=np.int8)
    n_tips = patterns.shape[1]
    n_nodes = 1 + int(np.max(edges))
    assert n_nodes - n_tips <= 12
    outcomes = []
    for pattern in patterns:
        missing = np.flatnonzero(pattern == -1)
        best = math.inf
        for internal in itertools.product((0, 1), repeat=n_nodes - n_tips):
            for hidden in itertools.product((0, 1), repeat=len(missing)):
                states = np.r_[pattern, internal].astype(int)
                states[missing] = hidden
                value = sum(states[a] != states[b] for a, b in edges)
                best = min(best, value)
        outcomes.append(best)
    return np.asarray(outcomes, dtype=int)


def wilson(successes, total, z=1.959963984540054):
    p = successes / total
    denominator = 1 + z*z/total
    mid = (p + z*z/(2*total)) / denominator
    half = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denominator
    return mid-half, mid+half


def self_test():
    edges = np.array([[4, 0], [4, 1], [4, 5], [5, 2], [5, 3]])
    patterns = np.array(list(itertools.product((-1, 0, 1), repeat=4)), dtype=np.int8)
    assert np.array_equal(fitch_character_lengths(patterns, edges), brute_force_lengths(patterns, edges))
    complete = patterns[np.all(patterns >= 0, axis=1)]
    assert np.array_equal(fitch_character_lengths(complete, edges), brute_force_lengths(complete, edges))
    return {"toy_tips": 4, "all_partial_patterns": len(patterns), "passed": True}


if __name__ == "__main__":
    print(self_test())
