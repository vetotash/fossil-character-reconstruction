#!/usr/bin/env python3
"""Exact six-tip unordered binary parsimony, with every optimal tree retained.

This module neither generates character matrices nor samples completions. The
scientific runner consumes the main simulation's saved draw identities.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

NTIPS = 6
METHODS = ("oracle_joint", "product_marginals", "sharpened_product")


def normalize_edges(edges):
    return tuple(sorted(tuple(sorted(map(int, edge))) for edge in edges))


def enumerate_trees(n=NTIPS):
    """Grow labelled unrooted binary trees by inserting the next leaf on an edge."""
    if n < 3:
        raise ValueError("At least three tips required")
    trees = [normalize_edges(((0, n), (1, n), (2, n)))]
    for tip in range(3, n):
        node = n + tip - 2
        new_trees = []
        for edges in trees:
            for a, b in edges:
                rest = [e for e in edges if e != (a, b)]
                new_trees.append(normalize_edges(rest + [(a, node), (b, node), (tip, node)]))
        trees = new_trees
    # Sort by topology, not arbitrary internal-node labels.
    trees.sort(key=lambda edges: split_key(edges, n))
    keys = [split_key(e, n) for e in trees]
    assert len(keys) == len(set(keys))
    return trees


def adjacency(edges):
    out = {}
    for a, b in edges:
        out.setdefault(int(a), []).append(int(b))
        out.setdefault(int(b), []).append(int(a))
    return out


def split_key(edges, n=NTIPS):
    graph = adjacency(edges)
    tips = set(range(n))
    splits = []
    for a, b in edges:
        seen, todo = {b}, [a]
        while todo:
            node = todo.pop()
            if node in seen:
                continue
            seen.add(node)
            todo.extend(graph[node])
        one = tips.intersection(seen - {b})
        other = tips - one
        if min(len(one), len(other)) >= 2:
            # Canonical orientation chooses the smaller side, then lexicographic.
            sides = [tuple(sorted(one)), tuple(sorted(other))]
            splits.append(min(sides, key=lambda s: (len(s), s)))
    return tuple(sorted(splits))


def tree_to_edges(tree, n=NTIPS):
    """A degree-three root with nested tuples/lists describes an unrooted tree."""
    counter = itertools.count(n)
    edges = []

    def descend(node):
        if isinstance(node, (int, np.integer)):
            return int(node)
        parent = next(counter)
        for child in node:
            edges.append((parent, descend(child)))
        return parent

    descend(tree)
    return normalize_edges(edges)


def ternary_patterns(n=NTIPS):
    ids = np.arange(3**n, dtype=np.int64)
    return np.asarray([(ids // 3**j) % 3 for j in range(n)], dtype=np.uint8).T


def binary_patterns(n=NTIPS):
    ids = np.arange(2**n, dtype=np.int64)
    return np.asarray([(ids >> j) & 1 for j in range(n)], dtype=np.uint8).T


def ternary_ids(patterns):
    patterns = np.asarray(patterns)
    if patterns.shape[-1] != NTIPS or np.any((patterns < 0) | (patterns > 2)):
        raise ValueError("Expected six tips taking states 0, 1, or 2 (unknown)")
    return np.sum(patterns.astype(np.int64) * 3**np.arange(NTIPS), axis=-1)


def fitch_pattern_scores(edges, patterns, n=NTIPS):
    """Return Fitch score for each pattern; 2 encodes missing, admitting {0,1}."""
    graph = adjacency(edges)
    patterns = np.asarray(patterns)
    root = min(v for v in graph if v >= n)

    def visit(node, parent):
        if node < n:
            value = patterns[:, node]
            sets = np.where(value == 2, 3, value + 1).astype(np.uint8)
            return sets, np.zeros(len(patterns), dtype=np.int16)
        child_values = [visit(c, node) for c in sorted(graph[node]) if c != parent]
        state, cost = child_values[0]
        for other, other_cost in child_values[1:]:
            intersection = state & other
            empty = intersection == 0
            state = np.where(empty, state | other, intersection).astype(np.uint8)
            cost = cost + other_cost + empty
        return state, cost

    return visit(root, -1)[1]


def minimum_steps(patterns):
    patterns = np.asarray(patterns)
    return (np.any(patterns == 0, axis=-1) & np.any(patterns == 1, axis=-1)).astype(np.uint8)


def build_registry():
    trees = enumerate_trees()
    patterns = ternary_patterns()
    costs = np.stack([fitch_pattern_scores(edges, patterns) for edges in trees])
    assert costs.shape == (105, 729)
    assert all(len(split_key(tree)) == 3 for tree in trees)
    full_ids = ternary_ids(binary_patterns())
    # Constants and complete empty-source patterns provide simple exact gates.
    assert np.all(costs[:, [0, (3**NTIPS - 1) // 2, 3**NTIPS - 1]] == 0)
    return {
        "edges": trees,
        "split_keys": [split_key(tree) for tree in trees],
        "ternary_patterns": patterns,
        "costs": costs.astype(np.uint8),
        "full_to_ternary": full_ids,
        "minimum_steps": minimum_steps(patterns),
    }


def save_registry(registry, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination / "six_tip_cost_registry.npz",
                        ternary_patterns=registry["ternary_patterns"],
                        costs=registry["costs"],
                        full_to_ternary=registry["full_to_ternary"],
                        minimum_steps=registry["minimum_steps"])
    rows = [{"tree_index": i, "edges": edges, "splits": registry["split_keys"][i]}
            for i, edges in enumerate(registry["edges"])]
    (destination / "six_tip_tree_registry.json").write_text(
        json.dumps({"ntips": NTIPS, "ntrees": 105, "rf_denominator": 6,
                    "pattern_tip_order": list(range(NTIPS)), "trees": rows}, indent=2) + "\n")


def find_truth_index(registry, truth_edges):
    key = split_key(truth_edges)
    hits = [i for i, item in enumerate(registry["split_keys"]) if item == key]
    if len(hits) != 1:
        raise ValueError(f"Generating topology not found exactly once: {hits}")
    return hits[0]


def rf_numerators(registry, truth_index):
    truth = set(registry["split_keys"][truth_index])
    return np.asarray([len(truth.symmetric_difference(set(key)))
                       for key in registry["split_keys"]], dtype=np.uint8)


def matrix_tree_scores(registry, pattern_ids):
    """Input (..., ncharacters) ternary ids; output (..., 105) tree scores."""
    ids = np.asarray(pattern_ids, dtype=np.int64)
    costs = registry["costs"][:, ids].sum(axis=-1, dtype=np.int32)
    return np.moveaxis(costs, 0, -1)


def summarize_mpts(tree_scores, rf_numerator, truth_index):
    scores = np.asarray(tree_scores)
    best = np.min(scores, axis=-1)
    selected = scores == best[..., None]
    count = np.sum(selected, axis=-1)
    assert np.all(count > 0)
    return {
        "minimum_tree_length": best,
        "mpt_count": count,
        "mean_normalized_rf": np.sum(selected * rf_numerator, axis=-1) / (6.0 * count),
        "truth_in_mpt": selected[..., truth_index],
        "mpt_bits": np.packbits(selected, axis=-1, bitorder="little"),
    }


def source_pattern_ids(full_pattern_ids, missing_mask_bits):
    full_pattern_ids = np.asarray(full_pattern_ids, dtype=np.int64)
    missing_mask_bits = np.asarray(missing_mask_bits, dtype=np.int64)
    if full_pattern_ids.shape != missing_mask_bits.shape:
        raise ValueError("Full pattern and missing mask arrays differ in shape")
    states = (full_pattern_ids[..., None] >> np.arange(NTIPS)) & 1
    missing = ((missing_mask_bits[..., None] >> np.arange(NTIPS)) & 1).astype(bool)
    return ternary_ids(np.where(missing, 2, states))


def check_observed_cells(full_ids, completion_ids, missing_mask_bits):
    observed = ((1 << NTIPS) - 1) ^ np.asarray(missing_mask_bits, dtype=np.int64)
    changed = (np.asarray(full_ids, dtype=np.int64) ^ np.asarray(completion_ids, dtype=np.int64)) & observed
    if np.any(changed):
        raise AssertionError("Completion changes an observed cell")


def evaluate_scenario(registry, truth_edges, full_ids, missing_masks, map_ids, draw_ids,
                      expected_draw_D=None):
    """Evaluate one simulation scenario, preserving arrays' matrix/method/draw axes."""
    full_ids = np.asarray(full_ids)
    missing_masks = np.asarray(missing_masks)
    map_ids = np.asarray(map_ids)
    draw_ids = np.asarray(draw_ids)
    nm, nc = full_ids.shape
    if missing_masks.shape != (nm, nc) or map_ids.shape != (nm, 3, nc):
        raise ValueError("Expected full/missing matrices [matrix,char], MAP [matrix,3,char]")
    if draw_ids.shape != (nm, 3, 64, nc):
        raise ValueError("Expected saved draws [matrix,3,64,char]")
    check_observed_cells(full_ids[:, None, :], map_ids, missing_masks[:, None, :])
    check_observed_cells(full_ids[:, None, None, :], draw_ids, missing_masks[:, None, None, :])
    if not (np.array_equal(map_ids[:, 0], map_ids[:, 1]) and np.array_equal(map_ids[:, 0], map_ids[:, 2])):
        raise AssertionError("The three marginal cellwise MAP completions should coincide")
    truth_index = find_truth_index(registry, truth_edges)
    rf = rf_numerators(registry, truth_index)
    full_ternary = registry["full_to_ternary"][full_ids]
    source_ternary = source_pattern_ids(full_ids, missing_masks)
    map_ternary = registry["full_to_ternary"][map_ids]
    full_scores = matrix_tree_scores(registry, full_ternary)
    source_scores = matrix_tree_scores(registry, source_ternary)
    map_scores = matrix_tree_scores(registry, map_ternary)
    output = {"truth_tree_index": np.asarray(truth_index), "rf_numerators": rf}
    for prefix, scores in (("full", full_scores), ("source", source_scores), ("map", map_scores)):
        for key, value in summarize_mpts(scores, rf, truth_index).items():
            output[prefix + "_" + key] = value
    full_m = registry["minimum_steps"][full_ternary].sum(axis=-1)
    source_m = registry["minimum_steps"][source_ternary].sum(axis=-1)
    source_length = source_scores[:, truth_index]
    truth_D = (full_scores[:, truth_index] - source_length) - (full_m.astype(int) - source_m.astype(int))
    output["truth_D"] = truth_D
    map_m = registry["minimum_steps"][map_ternary].sum(axis=-1)
    output["map_D"] = (map_scores[..., truth_index] - source_length[:, None]) - (map_m.astype(int) - source_m[:, None].astype(int))
    output["map_absolute_D_error"] = np.abs(output["map_D"] - truth_D[:, None])
    # Chunk by matrix; temporary score arrays remain small even for large exports.
    rows = []
    for matrix in range(nm):
        ternary = registry["full_to_ternary"][draw_ids[matrix]]
        scores = matrix_tree_scores(registry, ternary)
        result = summarize_mpts(scores, rf, truth_index)
        minima = registry["minimum_steps"][ternary].sum(axis=-1).astype(int)
        result["D"] = (scores[..., truth_index] - source_length[matrix]) - (minima - int(source_m[matrix]))
        result["absolute_D_error"] = np.abs(result["D"] - truth_D[matrix])
        rows.append(result)
    for key in rows[0]:
        output["draw_" + key] = np.stack([row[key] for row in rows])
    if expected_draw_D is not None and not np.array_equal(output["draw_D"], expected_draw_D):
        delta = np.max(np.abs(output["draw_D"] - expected_draw_D))
        raise AssertionError(f"Generating-tree D differs from simulation saved draws; max={delta}")
    output["full_tree_scores"] = full_scores
    output["source_tree_scores"] = source_scores
    output["map_tree_scores"] = map_scores
    return output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry-only", type=Path, required=True)
    args = parser.parse_args()
    registry = build_registry()
    save_registry(registry, args.registry_only)
    print(json.dumps({"status": "PASS", "trees": 105, "ternary_patterns": 729,
                      "scientific_simulations_generated": 0}))
