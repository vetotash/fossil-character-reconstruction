#!/usr/bin/env python3
"""Known-model binary-character experiment with exact conditional propagation.

No empirical models are fitted. The only completion laws are the generating
conditional joint distribution, its product of exact marginals, and the product
of power-two sharpened marginals. The common hard prediction is marginal MAP.
Run --self-test before execution. Production requires an explicit frozen design.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import time

import numpy as np

METHODS = ("oracle_joint", "product_marginals", "sharpened_product")
BASE_SEED = 2026100401
PROPAGATION_SEED = 2026100402
MAP_TIE_TOLERANCE = 1e-12
REPLICATES = 100
TOPOLOGY_DRAWS = 64
Q_VALUES = (0.02, 0.1, 0.5)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def to_nested_tuple(value):
    return tuple(map(to_nested_tuple, value)) if isinstance(value, (tuple, list)) else int(value)


def tree_for(n, shape):
    if shape == "balanced":
        return ((0, 1), (2, 3), (4, 5)) if n == 6 else ((0, (1, 2)), (3, (4, 5)), ((6, 7), (8, 9)))
    tail = (n - 2, n - 1)
    for tip in range(n - 3, 1, -1):
        tail = (tip, tail)
    return (0, 1, tail)


def tree_record(n, shape):
    nested = tree_for(n, shape)
    edges = []
    next_node = n
    def add(node):
        nonlocal next_node
        if isinstance(node, int):
            return node
        parent = next_node
        next_node += 1
        for child in node:
            edges.append([parent, add(child)])
        return parent
    root = add(nested)
    adjacency = {i: [] for i in range(next_node)}
    for a, b in edges:
        adjacency[a].append(b)
        adjacency[b].append(a)
    assert next_node == 2 * n - 2 and len(edges) == 2 * n - 3
    assert all(len(adjacency[i]) == (1 if i < n else 3) for i in adjacency)
    distances = np.empty((n, n), dtype=np.int16)
    for tip in range(n):
        discovered = {tip: 0}
        queue = [tip]
        for vertex in queue:
            for neighbor in adjacency[vertex]:
                if neighbor not in discovered:
                    discovered[neighbor] = discovered[vertex] + 1
                    queue.append(neighbor)
        distances[tip] = [discovered[j] for j in range(n)]
    def newick(node):
        if isinstance(node, int):
            return f"t{node}"
        return "(" + ",".join(newick(child) + ":1" for child in node) + ")"
    return {"key": f"n{n}_{shape}", "n_tips": n, "shape": shape,
            "tip_labels": [f"t{i}" for i in range(n)], "nested": nested,
            "root": root, "edges": edges, "edge_lengths": [1.0] * len(edges),
            "newick": newick(nested) + ";", "tip_edge_distances": distances.tolist()}


def patterns_for(n):
    ids = np.arange(2 ** n, dtype=np.uint16)
    return ((ids[:, None] >> np.arange(n)) & 1).astype(np.uint8)


def pattern_probabilities(patterns, nested, q):
    """Pruning likelihood with uniform root and Q=[[-q,q],[q,-q]]."""
    flip = -np.expm1(-2 * q) / 2
    transition = np.array([[1 - flip, flip], [flip, 1 - flip]])
    def recurse(node):
        if isinstance(node, int):
            return np.column_stack((patterns[:, node] == 0, patterns[:, node] == 1)).astype(float)
        result = np.ones((len(patterns), 2))
        for child in node:
            result *= recurse(child) @ transition.T
        return result
    p = recurse(to_nested_tuple(nested)).mean(axis=1)
    assert abs(p.sum() - 1) < 1e-12 and np.all(p > 0)
    return p / p.sum()


def parsimony_costs(states, nested):
    """Sankoff costs on a fixed tree; -1 denotes unknown, cost zero at both states."""
    states = np.atleast_2d(states)
    def recurse(node):
        if isinstance(node, int):
            return np.column_stack((states[:, node] == 1, states[:, node] == 0)).astype(np.int16) * 1000
        result = np.zeros((len(states), 2), dtype=np.int16)
        for child in node:
            costs = recurse(child)
            result[:, 0] += np.minimum(costs[:, 0], costs[:, 1] + 1)
            result[:, 1] += np.minimum(costs[:, 1], costs[:, 0] + 1)
        return result
    return recurse(to_nested_tuple(nested)).min(axis=1)


def minima(states):
    states = np.atleast_2d(states)
    return (np.any(states == 0, axis=1) & np.any(states == 1, axis=1)).astype(np.int16)


def conditional_distribution(patterns, prior, full_steps, full_minima, nested, missing_bits, observed_bits):
    """Return exact laws on complete-pattern IDs compatible with observations.

    Pattern ID bit j is tip j. observed_bits has zeroes at missing tips. This
    function has no random state and supports independent downstream replay.
    """
    n = patterns.shape[1]
    all_bits = (1 << n) - 1
    known_bits = all_bits ^ int(missing_bits)
    if int(observed_bits) & int(missing_bits):
        raise ValueError("observed_bits must be zero at missing tips")
    ids = np.arange(len(patterns), dtype=np.uint16)
    support = ids[(ids & known_bits) == int(observed_bits)]
    conditional_joint = prior[support].copy()
    evidence_probability = conditional_joint.sum()
    conditional_joint /= evidence_probability
    missing_tips = np.flatnonzero((int(missing_bits) >> np.arange(n)) & 1)
    support_states = patterns[support][:, missing_tips]
    p1 = conditional_joint @ support_states
    p0 = 1 - p1
    marginal_product = np.prod(np.where(support_states, p1, p0), axis=1)
    p1_sharp = p1 * p1 / (p1 * p1 + p0 * p0)
    sharpened_product = np.prod(np.where(support_states, p1_sharp, 1 - p1_sharp), axis=1)
    probabilities = np.stack((conditional_joint, marginal_product, sharpened_product))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    assert np.max(np.abs(probabilities.sum(axis=1) - 1)) < 1e-12
    old_states = patterns[int(observed_bits)].astype(np.int16)
    old_states[missing_tips] = -1
    old_steps = int(parsimony_costs(old_states, nested)[0])
    old_minimum = int(minima(old_states)[0])
    D_support = (full_steps[support] - old_steps) - (full_minima[support] - old_minimum)
    assert np.all(D_support >= 0)
    D_pmfs = np.stack([np.bincount(D_support, weights=p, minlength=int(D_support.max()) + 1)
                      for p in probabilities])
    marginal_map_bits = int(observed_bits)
    # Ties are deliberately mapped to zero in all methods. No joint MAP is used.
    for tip, prob in zip(missing_tips, p1):
        marginal_map_bits |= int(prob > 0.5 + MAP_TIE_TOLERANCE) << int(tip)
    marginal_map_D = int((full_steps[marginal_map_bits] - old_steps) - (full_minima[marginal_map_bits] - old_minimum))
    forecast_p1 = np.stack((p1, p1, p1_sharp))
    forecast_entropy = -(np.where(forecast_p1 > 0, forecast_p1 * np.log2(np.maximum(forecast_p1, 1e-300)), 0)
                         + np.where(forecast_p1 < 1, (1 - forecast_p1) * np.log2(np.maximum(1 - forecast_p1, 1e-300)), 0))
    joint_entropy = -float(np.dot(conditional_joint, np.log2(conditional_joint)))
    total_correlation = float(forecast_entropy[0].sum() - joint_entropy)
    assert total_correlation > -1e-10
    return {"support": support, "probabilities": probabilities, "p1": p1,
            "forecast_p1": forecast_p1, "forecast_entropy": forecast_entropy,
            "missing_tips": missing_tips, "old_steps": old_steps, "old_minimum": old_minimum,
            "D_support": D_support, "D_pmfs": D_pmfs, "marginal_map_bits": marginal_map_bits,
            "marginal_map_D": marginal_map_D, "evidence_probability": float(evidence_probability),
            "total_correlation_bits": max(0.0, total_correlation)}


def convolve_pmfs(pmfs):
    result = np.ones(1)
    for pmf in pmfs:
        result = np.convolve(result, pmf)
    if not np.isfinite(result).all() or np.min(result) < -1e-14:
        raise ArithmeticError("Invalid convolved distribution")
    result = np.maximum(result, 0)
    result /= result.sum()
    return result


def pmf_interval(pmf):
    cdf = np.cumsum(pmf)
    return int(np.searchsorted(cdf, 0.025)), int(np.searchsorted(cdf, 0.975))


def scenario_records():
    records = []
    for n, nchar, missing_counts in ((6, 32, (2, 4)), (10, 64, (3, 7))):
        for shape, q, k, mask_mode in itertools.product(("balanced", "caterpillar"), Q_VALUES, missing_counts, ("random", "clustered")):
            ordinal = len(records)
            records.append({"scenario_id": f"s{ordinal:02d}", "ordinal": ordinal, "n_tips": n,
                            "n_characters": nchar, "tree_key": f"n{n}_{shape}", "shape": shape,
                            "q": q, "missing_per_character": k, "mask_mode": mask_mode,
                            "replicates": REPLICATES, "topology_draws": TOPOLOGY_DRAWS if n == 6 else 0})
    return records


def write_csv(path, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def wilson(successes, n):
    z = 1.959963984540054
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return center - half, center + half


def design_description():
    return {"title": "Known model tests of conditional character propagation",
            "version": 1, "base_seed": BASE_SEED, "propagation_seed": PROPAGATION_SEED,
            "map_tie_tolerance": MAP_TIE_TOLERANCE, "methods": list(METHODS),
            "replicates_per_scenario": REPLICATES, "scenario_count": 48,
            "trees": [tree_record(n, shape) for n, shape in itertools.product((6, 10), ("balanced", "caterpillar"))],
            "scenarios": scenario_records(),
            "generator": "Independent binary symmetric Mk characters; uniform root; all unrooted edges length 1; transition flip=(1-exp(-2q))/2; constants retained; no ascertainment conditioning.",
            "mask": "Independent of character states. Random: k uniformly sampled tips independently for each character. Clustered: one uniform anchor per matrix, k nearest tips including anchor, shortest graph distances then ascending tip index break ties; same subset for every character.",
            "propagation": "Exact conditional per-character D distributions convolved over independent characters. Intervals are discrete equal-tail central 95 percent (inverse CDF at .025 and .975), not confidence intervals.",
            "D": "(L_complete-L_observed)-(M_complete-M_observed); L fixed generating-tree unordered binary parsimony; M=1 iff both 0 and 1 observed, else 0.",
            "hard_prediction": "All methods use the same exact conditional marginal MAP; ties within 1e-12 of p1=0.5 select state 0. It is not a joint MAP.",
            "brier": "Mean over missing cells of the two-class sum of squared errors; exact expected Brier also evaluated under oracle conditional marginals.",
            "accuracy": "Realized marginal-MAP agreement and its oracle conditional expectation; forecast maximum probability recorded separately.",
            "scope": "Controlled known tree, rates, model and mask; no parameter fitting, anatomical validation, historical-score truth claim, or universal method ranking.",
            "topology_draws": "For 6 tips, 64 completions per method and matrix; common uniforms across three methods; separate RNG stream from truth and mask; source/all-truth/marginal MAP also saved. Fixed-tree D main analysis remains exact.",
            "rng": "NumPy default_rng SeedSequence([base_seed, scenario_ordinal, stream]); base_seed2026100401 with stream 0 truth, 1 masks; independent base2026100402 with stream 2 topology completions."}


def run(output, frozen_design):
    output = Path(output)
    frozen_design = Path(frozen_design)
    if not frozen_design.is_file():
        raise FileNotFoundError("Production requires the root-approved frozen design artifact")
    if output.exists():
        raise FileExistsError("Use a new output directory; existing evidence is never overwritten")
    output.mkdir(parents=True)
    started = time.time()
    design = design_description()
    trees = {record["key"]: record for record in design["trees"]}
    json_write(output / "design.json", design)
    json_write(output / "trees.json", trees)
    registry = {}
    models = {}
    for n in (6, 10):
        registry[f"n{n}_patterns"] = patterns_for(n)
    for tree_key, tree in trees.items():
        n = tree["n_tips"]
        patterns = registry[f"n{n}_patterns"]
        steps = parsimony_costs(patterns, tree["nested"])
        minimum = minima(patterns)
        registry[f"{tree_key}_full_steps"] = steps
        registry[f"{tree_key}_full_minima"] = minimum
        for qi, q in enumerate(Q_VALUES):
            prior = pattern_probabilities(patterns, tree["nested"], q)
            registry[f"{tree_key}_q{qi}_probabilities"] = prior
            models[(tree_key, q)] = (patterns, prior, steps, minimum)
    np.savez_compressed(output / "pattern_registry.npz", **registry)
    write_csv(output / "scenario_registry.csv", scenario_records())
    all_rows = []
    summary_rows = []
    cache_max = 0
    for scenario in scenario_records():
        scenario_started = time.time()
        sid, ordinal = scenario["scenario_id"], scenario["ordinal"]
        n, nchar, k = (scenario[key] for key in ("n_tips", "n_characters", "missing_per_character"))
        tree = trees[scenario["tree_key"]]
        patterns, prior, full_steps, full_minima = models[(scenario["tree_key"], scenario["q"])]
        truth_rng, mask_rng = [np.random.default_rng(np.random.SeedSequence([BASE_SEED, ordinal, stream])) for stream in range(2)]
        completion_rng = np.random.default_rng(np.random.SeedSequence([PROPAGATION_SEED, ordinal, 2]))
        full_ids = truth_rng.choice(len(patterns), size=(REPLICATES, nchar), p=prior).astype(np.uint16)
        missing_masks = np.zeros((REPLICATES, nchar), dtype=np.uint16)
        cluster_anchors = np.full(REPLICATES, -1, dtype=np.int16)
        for r in range(REPLICATES):
            if scenario["mask_mode"] == "clustered":
                anchor = int(mask_rng.integers(n))
                cluster_anchors[r] = anchor
                tips = sorted(range(n), key=lambda tip: (tree["tip_edge_distances"][anchor][tip], tip))[:k]
                missing_masks[r, :] = sum(1 << int(tip) for tip in tips)
            else:
                for c in range(nchar):
                    missing_masks[r, c] = sum(1 << int(tip) for tip in mask_rng.choice(n, k, replace=False))
        observed_ids = full_ids & (((1 << n) - 1) ^ missing_masks)
        map_ids = np.zeros((REPLICATES, 3, nchar), dtype=np.uint16)
        # D has support no greater than nchar * (floor(n/2)-1).
        pmf_width = nchar * (n // 2 - 1) + 1
        matrix_D_pmfs = np.zeros((REPLICATES, 3, pmf_width))
        true_D = np.zeros(REPLICATES, dtype=np.int16)
        source_L = np.zeros(REPLICATES, dtype=np.int16)
        source_M = np.zeros(REPLICATES, dtype=np.int16)
        map_D = np.zeros(REPLICATES, dtype=np.int16)
        missing_tip_indices = np.zeros((REPLICATES, nchar, k), dtype=np.uint8)
        oracle_missing_p1 = np.zeros((REPLICATES, nchar, k))
        sharpened_missing_p1 = np.zeros((REPLICATES, nchar, k))
        if n == 6:
            draws = np.zeros((REPLICATES, 3, TOPOLOGY_DRAWS, nchar), dtype=np.uint8)
            draw_D = np.zeros((REPLICATES, 3, TOPOLOGY_DRAWS), dtype=np.int16)
        cache = {}
        scenario_rows = []
        for r in range(REPLICATES):
            per_character = []
            realized_brier = np.zeros(3)
            expected_brier = np.zeros(3)
            entropy = np.zeros(3)
            forecast_max = np.zeros(3)
            actual_correct = expected_correct = 0.0
            total_correlation = 0.0
            uniforms = completion_rng.random((TOPOLOGY_DRAWS, nchar)) if n == 6 else None
            for c in range(nchar):
                key = (int(missing_masks[r, c]), int(observed_ids[r, c]))
                if key not in cache:
                    cache[key] = conditional_distribution(patterns, prior, full_steps, full_minima, tree["nested"], *key)
                item = cache[key]
                per_character.append(item)
                actual = patterns[full_ids[r, c], item["missing_tips"]]
                p1 = item["p1"]
                forecast = item["forecast_p1"]
                missing_tip_indices[r, c] = item["missing_tips"]
                oracle_missing_p1[r, c] = p1
                sharpened_missing_p1[r, c] = forecast[2]
                realized_brier += 2 * np.sum((forecast - actual) ** 2, axis=1)
                expected_brier += 2 * np.sum(p1 * (1 - forecast) ** 2 + (1 - p1) * forecast ** 2, axis=1)
                entropy += item["forecast_entropy"].sum(axis=1)
                forecast_max += np.maximum(forecast, 1 - forecast).sum(axis=1)
                chosen_one = p1 > 0.5 + MAP_TIE_TOLERANCE
                expected_correct += np.where(chosen_one, p1, 1 - p1).sum()
                actual_correct += np.sum(chosen_one == actual)
                total_correlation += item["total_correlation_bits"]
                map_ids[r, :, c] = item["marginal_map_bits"]
                map_D[r] += item["marginal_map_D"]
                source_L[r] += item["old_steps"]
                source_M[r] += item["old_minimum"]
                true_D[r] += int((full_steps[full_ids[r, c]] - item["old_steps"]) - (full_minima[full_ids[r, c]] - item["old_minimum"]))
                if n == 6:
                    for m in range(3):
                        cdf = np.cumsum(item["probabilities"][m])
                        cdf[-1] = 1.0
                        indices = np.searchsorted(cdf, uniforms[:, c], side="right")
                        draws[r, m, :, c] = item["support"][indices]
                        draw_D[r, m] += item["D_support"][indices]
            for m in range(3):
                pmf = convolve_pmfs([item["D_pmfs"][m] for item in per_character])
                if len(pmf) > pmf_width:
                    raise ArithmeticError("Unexpected fixed-tree D upper bound")
                matrix_D_pmfs[r, m, :len(pmf)] = pmf
            oracle_pmf = matrix_D_pmfs[r, 0]
            D_axis = np.arange(pmf_width)
            oracle_mean = float(oracle_pmf @ D_axis)
            for m, method in enumerate(METHODS):
                pmf = matrix_D_pmfs[r, m]
                lo, hi = pmf_interval(pmf)
                D_mean = float(pmf @ D_axis)
                row = dict(scenario)
                row.update({"replicate": r, "matrix_id": f"{sid}_r{r:03d}", "method": method,
                            "target_cells": k * nchar, "true_D": int(true_D[r]), "marginal_map_D": int(map_D[r]),
                            "predicted_D_mean": D_mean, "oracle_D_mean": oracle_mean,
                            "predicted_D_variance": float(pmf @ ((D_axis - D_mean) ** 2)),
                            "mean_minus_oracle": D_mean - oracle_mean,
                            "mean_minus_truth": D_mean - int(true_D[r]),
                            "D_q025": lo, "D_q975": hi, "interval_width": hi - lo,
                            "realized_covered": int(lo <= true_D[r] <= hi),
                            "oracle_conditional_interval_mass": float(oracle_pmf[lo:hi+1].sum()),
                            "own_interval_mass": float(pmf[lo:hi+1].sum()),
                            "D_total_variation_from_oracle": float(np.abs(pmf - oracle_pmf).sum() / 2),
                            "realized_brier": float(realized_brier[m] / (k * nchar)),
                            "oracle_expected_brier": float(expected_brier[m] / (k * nchar)),
                            "realized_accuracy": actual_correct / (k * nchar),
                            "oracle_expected_accuracy": expected_correct / (k * nchar),
                            "forecast_mean_max_probability": float(forecast_max[m] / (k * nchar)),
                            "mean_marginal_entropy_bits": float(entropy[m] / (k * nchar)),
                            "oracle_total_correlation_bits_per_character": total_correlation / nchar})
                scenario_rows.append(row)
            assert np.array_equal(map_ids[r, 0], map_ids[r, 1]) and np.array_equal(map_ids[r, 0], map_ids[r, 2])
        stored = {"full_pattern_ids": full_ids, "missing_mask_bits": missing_masks,
                  "observed_state_bits": observed_ids, "cluster_anchors": cluster_anchors,
                  "map_pattern_ids": map_ids, "matrix_D_pmfs": matrix_D_pmfs,
                  "missing_tip_indices": missing_tip_indices, "oracle_missing_p1": oracle_missing_p1,
                  "sharpened_missing_p1": sharpened_missing_p1,
                  "true_D": true_D, "source_L": source_L, "source_M": source_M, "marginal_map_D": map_D}
        if n == 6:
            stored.update(draw_pattern_ids=draws, draw_D=draw_D)
        np.savez_compressed(output / f"{sid}_matrices.npz", **stored)
        write_csv(output / f"{sid}_results.csv", scenario_rows)
        all_rows.extend(scenario_rows)
        for method in METHODS:
            rows = [row for row in scenario_rows if row["method"] == method]
            record = dict(scenario, method=method)
            numeric_keys = [key for key in rows[0] if key not in scenario and key not in ("replicate", "matrix_id", "method")]
            for key in numeric_keys:
                values = np.array([row[key] for row in rows], dtype=float)
                record[f"{key}_mean"] = float(values.mean())
                record[f"{key}_se"] = float(values.std(ddof=1) / math.sqrt(REPLICATES))
            successes = sum(row["realized_covered"] for row in rows)
            lo, hi = wilson(successes, REPLICATES)
            record.update(coverage_count=successes, coverage_wilson_low=lo, coverage_wilson_high=hi)
            summary_rows.append(record)
        cache_max = max(cache_max, len(cache))
        print(f"{sid} complete {len(cache)} conditional states {time.time()-scenario_started:.2f}s", flush=True)
    write_csv(output / "matrix_results.csv", all_rows)
    write_csv(output / "scenario_summary.csv", summary_rows)
    checks = {"status": "PASS", "matrix_count": 48 * REPLICATES, "method_matrix_rows": len(all_rows),
              "scenario_method_rows": len(summary_rows), "methods": list(METHODS),
              "all_scenarios_retained": len(summary_rows) == 144,
              "known_model_oracle_not_anatomical_truth": True,
              "oracle_interval_mass_min": min(row["oracle_conditional_interval_mass"] for row in all_rows if row["method"] == METHODS[0]),
              "oracle_minimum_expected_brier_verified": True,
              "maximum_conditional_cache_entries": cache_max,
              "runtime_seconds": time.time() - started, "python": platform.python_version(),
              "numpy": np.__version__, "script_sha256": sha256(__file__),
              "frozen_design_sha256": sha256(frozen_design)}
    assert len(all_rows) == 14400 and len(summary_rows) == 144
    assert checks["oracle_interval_mass_min"] >= 0.95 - 1e-12
    for i in range(0, len(all_rows), 3):
        block = all_rows[i:i+3]
        assert abs(block[0]["realized_brier"] - block[1]["realized_brier"]) < 1e-14
        assert block[2]["oracle_expected_brier"] >= block[0]["oracle_expected_brier"] - 1e-14
        assert len({row["realized_accuracy"] for row in block}) == 1
    json_write(output / "RUN_CHECKS.json", checks)
    json_write(output / "MANIFEST.json", {path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
                                          for path in sorted(output.iterdir()) if path.is_file()})
    print(json.dumps(checks), flush=True)


def self_test():
    # Pre-design correctness checks only: one tiny topology, arbitrary q=0.3.
    tree = (0, 1, 2)
    patterns = patterns_for(3)
    p = pattern_probabilities(patterns, tree, 0.3)
    flip = (1 - math.exp(-0.6)) / 2
    direct = np.zeros(8)
    for i, states in enumerate(patterns):
        for root in (0, 1):
            direct[i] += 0.5 * np.prod([flip if state != root else 1 - flip for state in states])
    np.testing.assert_allclose(p, direct, rtol=1e-13, atol=1e-15)
    cost = parsimony_costs(patterns, tree)
    np.testing.assert_array_equal(cost, [min(sum(row), 3 - sum(row)) for row in patterns])
    item = conditional_distribution(patterns, p, cost, minima(patterns), tree, 3, 4)
    assert item["support"].tolist() == [4, 5, 6, 7]
    assert np.all(item["D_support"] == 0)
    np.testing.assert_allclose(item["probabilities"].sum(axis=1), 1)
    np.testing.assert_allclose(item["probabilities"][1] @ patterns[item["support"]][:, :2], item["p1"])
    np.testing.assert_allclose(convolve_pmfs([[.25, .75], [.4, .6]]), [.1, .45, .45])
    for n in (6, 10):
        for shape in ("balanced", "caterpillar"):
            tree_record(n, shape)
    print(json.dumps({"status": "PASS", "scope": "tiny algebra/interface tests only, no scenario outcomes generated"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--describe", type=Path)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--frozen-design", type=Path)
    args = parser.parse_args()
    if args.describe:
        json_write(args.describe, design_description())
    if args.self_test:
        self_test()
    if args.output:
        if not args.frozen_design:
            parser.error("--output requires --frozen-design")
        run(args.output, args.frozen_design)
