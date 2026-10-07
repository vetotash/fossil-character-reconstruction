#!/usr/bin/env python3
"""Random matrix generation and evaluation, separate from masked-data fitting.

Run generation, fitting and evaluation as separate commands. The fit worker is
in fit_worker.py and receives masked-file paths only. Scientific production
requires a frozen design file; self-test uses a small engineering fixture.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import time

import numpy as np

METHODS = ("mode", "knn5", "logistic", "extra_trees", "nj_mk")
SCHEMES = (*METHODS, "nj_mk_joint")
N_TIPS = 48
N_CHARS = 40
N_ANCHORS = 8
N_REPLICATES = 50
N_DRAWS = 512
GENERATION_SEED = 2026100403
MASK_SEED = 2026100404
PROPAGATION_SEED = 2026100405
FIXED_MODEL_SEED = 20261002
MAP_TIE_TOLERANCE = 0.0
LOG_FLOOR = 1e-12


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def write_csv(path, rows):
    with Path(path).open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def scenarios():
    rows = []
    for q, missing, mask in itertools.product((0.03, 0.3), (14, 29), ("random", "structured", "statebiased")):
        ordinal = len(rows)
        rows.append(dict(scenario_id=f"s{ordinal:02d}", ordinal=ordinal, q=q,
                         missing_per_response=missing, mask_mode=mask,
                         n_tips=N_TIPS, n_characters=N_CHARS, n_complete_anchors=N_ANCHORS,
                         replicates=N_REPLICATES, draws=N_DRAWS))
    return rows


def orient_tree(edges, n_tips, root=None):
    edges = np.asarray(edges, dtype=np.int16)
    n_nodes = int(edges.max()) + 1
    root = n_tips if root is None else int(root)
    adjacency = [[] for _ in range(n_nodes)]
    for a, b in edges:
        adjacency[int(a)].append(int(b))
        adjacency[int(b)].append(int(a))
    assert n_nodes == 2 * n_tips - 2
    assert len(edges) == 2 * n_tips - 3
    assert all(len(adjacency[i]) == (1 if i < n_tips else 3) for i in range(n_nodes))
    parents = np.full(n_nodes, -1, dtype=np.int16)
    order = [root]
    children = [[] for _ in range(n_nodes)]
    for a in order:
        for b in sorted(adjacency[a]):
            if b != parents[a]:
                parents[b] = a
                children[a].append(b)
                order.append(b)
    assert len(order) == n_nodes and len(set(order)) == n_nodes
    return root, order, children, adjacency


def random_tree(n_tips, rng):
    """Uniform edge insertion from a labelled 3-tip star, length 1 per edge."""
    edges = [(0, n_tips), (1, n_tips), (2, n_tips)]
    for tip in range(3, n_tips):
        edge_index = int(rng.integers(len(edges)))
        a, b = edges.pop(edge_index)
        internal = n_tips + tip - 2
        edges.extend([(a, internal), (b, internal), (tip, internal)])
    edges = np.asarray(edges, dtype=np.int16)
    orient_tree(edges, n_tips)
    return edges


def generate_states(edges, n_tips, n_characters, q, rng):
    root, order, children, _ = orient_tree(edges, n_tips)
    states = np.zeros((2 * n_tips - 2, n_characters), dtype=np.uint8)
    states[root] = rng.integers(0, 2, n_characters, dtype=np.uint8)
    flip_probability = -np.expm1(-2 * q) / 2
    for parent in order:
        for child in children[parent]:
            states[child] = states[parent] ^ (rng.random(n_characters) < flip_probability)
    return states, root


def make_mask(truth, edges, k, mode, rng, n_anchors=N_ANCHORS):
    n_tips, n_characters = truth.shape
    mask = np.zeros_like(truth, dtype=bool)
    anchor = -1
    if mode == "structured":
        anchor = int(rng.integers(n_tips))
        _, _, _, adjacency = orient_tree(edges, n_tips)
        distances = {anchor: 0}
        queue = [anchor]
        for node in queue:
            for neighbor in adjacency[node]:
                if neighbor not in distances:
                    distances[neighbor] = distances[node] + 1
                    queue.append(neighbor)
        selected = sorted(range(n_tips), key=lambda i: (distances[i], i))[:k]
        mask[np.asarray(selected), n_anchors:] = True
    else:
        for j in range(n_anchors, n_characters):
            weights = None
            if mode == "statebiased":
                weights = np.where(truth[:, j] == 1, 4.0, 1.0)
                weights /= weights.sum()
            elif mode != "random":
                raise ValueError(mode)
            selected = rng.choice(n_tips, k, replace=False, p=weights)
            mask[selected, j] = True
    assert not mask[:, :n_anchors].any()
    assert np.all(mask[:, n_anchors:].sum(axis=0) == k)
    return mask, anchor


def parsimony_costs(states, edges, root=None):
    """Exact binary unordered Sankoff score, vectorized over state patterns."""
    states = np.atleast_2d(np.asarray(states, dtype=np.int8))
    n_patterns, n_tips = states.shape
    root, order, children, _ = orient_tree(edges, n_tips, root)
    costs = [None] * (2 * n_tips - 2)
    inf = n_tips + 1
    for node in reversed(order):
        if node < n_tips:
            costs[node] = np.column_stack((states[:, node] == 1, states[:, node] == 0)).astype(np.int16) * inf
        else:
            value = np.zeros((n_patterns, 2), dtype=np.int16)
            for child in children[node]:
                child_cost = costs[child]
                value[:, 0] += np.minimum(child_cost[:, 0], child_cost[:, 1] + 1)
                value[:, 1] += np.minimum(child_cost[:, 1], child_cost[:, 0] + 1)
                costs[child] = None
            costs[node] = value
    return costs[root].min(axis=1)


def minima(states):
    states = np.atleast_2d(states)
    return (np.any(states == 0, axis=1) & np.any(states == 1, axis=1)).astype(np.int16)


def total_costs(matrix, edges, root=None):
    return int(parsimony_costs(matrix.T, edges, root).sum()), int(minima(matrix.T).sum())


def sample_costs(masked, missing_states, edges, root=None):
    """Complete only the target cohort and return per-completion L and M."""
    missing_states = np.asarray(missing_states, dtype=np.uint8)
    n_draws = len(missing_states)
    targets = np.argwhere(masked == -1)
    assert missing_states.shape == (n_draws, len(targets))
    complete = np.broadcast_to(masked, (n_draws, *masked.shape)).copy()
    complete[:, targets[:, 0], targets[:, 1]] = missing_states
    patterns = complete.transpose(0, 2, 1).reshape(-1, masked.shape[0])
    l = parsimony_costs(patterns, edges, root).reshape(n_draws, masked.shape[1]).sum(axis=1)
    m = minima(patterns).reshape(n_draws, masked.shape[1]).sum(axis=1)
    return l, m


def generation(base, frozen_design):
    base, frozen_design = Path(base), Path(frozen_design)
    if not frozen_design.is_file():
        raise FileNotFoundError("Production requires an existing root-approved frozen design")
    if base.exists():
        raise FileExistsError("Use a new output directory; previous data are never overwritten")
    (base / "inputs").mkdir(parents=True)
    (base / "truth").mkdir()
    registry, input_manifest = [], []
    for scenario in scenarios():
        s = scenario["ordinal"]
        for rep in range(N_REPLICATES):
            matrix_id = f"{scenario['scenario_id']}_r{rep:03d}"
            tree_rng = np.random.default_rng(np.random.SeedSequence([GENERATION_SEED, s, rep, 0]))
            states_rng = np.random.default_rng(np.random.SeedSequence([GENERATION_SEED, s, rep, 1]))
            mask_rng = np.random.default_rng(np.random.SeedSequence([MASK_SEED, s, rep]))
            edges = random_tree(N_TIPS, tree_rng)
            states, root = generate_states(edges, N_TIPS, N_CHARS, scenario["q"], states_rng)
            truth = states[:N_TIPS]
            mask, anchor = make_mask(truth, edges, scenario["missing_per_response"], scenario["mask_mode"], mask_rng)
            masked = truth.astype(np.int8)
            masked[mask] = -1
            masked_path = base / "inputs" / f"{matrix_id}_masked.npz"
            truth_path = base / "truth" / f"{matrix_id}_truth.npz"
            np.savez_compressed(masked_path, X=masked)
            np.savez_compressed(truth_path, X_true=truth, node_states=states, edges=edges,
                                edge_lengths=np.ones(len(edges)), root=np.array(root),
                                q=np.array(scenario["q"]), mask_anchor=np.array(anchor))
            input_manifest.append({"matrix_id": matrix_id, "scenario_ordinal": s, "replicate": rep,
                                   "masked_file": str(masked_path.relative_to(base)),
                                   "masked_sha256": sha256(masked_path)})
            registry.append({**scenario, "matrix_id": matrix_id, "replicate": rep,
                             "masked_sha256": sha256(masked_path), "truth_sha256": sha256(truth_path)})
    write_csv(base / "scenario_registry.csv", scenarios())
    write_csv(base / "matrix_registry.csv", registry)
    write_json(base / "input_manifest.json", input_manifest)
    write_json(base / "GENERATION_MANIFEST.json", {"frozen_design_sha256": sha256(frozen_design),
               "matrix_count": len(registry), "generation_seed": GENERATION_SEED,
               "mask_seed": MASK_SEED, "propagation_seed": PROPAGATION_SEED,
               "fixed_model_seed": FIXED_MODEL_SEED, "script_sha256": sha256(__file__)})
    print(json.dumps({"generated": len(registry), "output": str(base)}), flush=True)


def metric_record(probabilities, y):
    ptrue = probabilities[np.arange(len(y)), y]
    hard = np.argmax(probabilities, axis=1)
    onehot = np.eye(2)[y]
    return {"accuracy": float(np.mean(hard == y)),
            "brier": float(np.mean(np.sum((probabilities-onehot)**2, axis=1))),
            "log_loss": float(np.mean(-np.log(np.maximum(ptrue, LOG_FLOOR)))),
            "zero_true_probability_count": int(np.count_nonzero(ptrue == 0)),
            "log_floor_count": int(np.count_nonzero(ptrue < LOG_FLOOR)),
            "mean_max_probability": float(np.mean(np.max(probabilities, axis=1)))}


def evaluate_matrix(base, entry):
    base = Path(base)
    matrix_id = entry["matrix_id"]
    pred_path = base / "predictions" / f"{matrix_id}.npz"
    pred_meta_path = base / "predictions" / f"{matrix_id}.json"
    pred_meta = json.loads(pred_meta_path.read_text())
    if sha256(pred_path) != pred_meta["prediction_sha256"]:
        raise ValueError(f"Prediction hash mismatch before truth read: {matrix_id}")
    masked_path = base / entry["masked_file"]
    if sha256(masked_path) != entry["masked_sha256"]:
        raise ValueError(f"Masked input changed: {matrix_id}")
    masked = np.load(masked_path)["X"]
    predictions = np.load(pred_path)
    probabilities = predictions["probabilities"]
    targets = np.argwhere(masked == -1)
    n_targets = len(targets)
    # Truth is first read after the saved prediction artifact and hash checks.
    with np.load(base / "truth" / f"{matrix_id}_truth.npz") as archive:
        truth, edges, root = archive["X_true"], archive["edges"], int(archive["root"])
    y = truth[targets[:, 0], targets[:, 1]]
    old_l, old_m = total_costs(masked, edges, root)
    true_l, true_m = total_costs(truth, edges, root)
    true_d = true_l - old_l - (true_m - old_m)
    all_d, all_l, all_m, all_hard_d, completion_hashes = [], [], [], [], []
    rows = []
    for scheme_index, scheme in enumerate(SCHEMES):
        method_index = min(scheme_index, len(METHODS)-1)
        target_probabilities = probabilities[method_index, targets[:, 0], targets[:, 1], :]
        hard_states = np.argmax(target_probabilities, axis=1).astype(np.uint8)
        hard_l, hard_m = sample_costs(masked, hard_states[None, :], edges, root)
        hard_d = int(hard_l[0] - old_l - (hard_m[0] - old_m))
        sampled = np.unpackbits(predictions["missing_states_packed"][scheme_index], axis=1, count=n_targets).astype(np.uint8)
        sampled_l, sampled_m = sample_costs(masked, sampled, edges, root)
        d = sampled_l - old_l - (sampled_m - old_m)
        assert np.all(d >= 0) and true_d >= 0 and hard_d >= 0
        lower, upper = np.quantile(d, [0.025, 0.975], method="linear")
        rows.append({"matrix_id": matrix_id, "scenario_id": f"s{entry['scenario_ordinal']:02d}",
                     "replicate": entry["replicate"], "scheme": scheme, "n_targets": n_targets,
                     **metric_record(target_probabilities, y), "old_L": old_l, "old_M": old_m,
                     "true_L": true_l, "true_M": true_m, "true_D": true_d,
                     "hard_D": hard_d, "hard_D_error": hard_d-true_d,
                     "hard_D_absolute_error": abs(hard_d-true_d),
                     "mean_D": float(np.mean(d)), "mean_D_error": float(np.mean(d)-true_d),
                     "mean_D_absolute_error": abs(float(np.mean(d)-true_d)),
                     "sd_D": float(np.std(d, ddof=1)), "mean_D_mcse": float(np.std(d, ddof=1)/math.sqrt(N_DRAWS)),
                     "lower95": float(lower), "upper95": float(upper),
                     "interval_width": float(upper-lower),
                     "truth_in_interval": int(lower <= true_d <= upper),
                     "truth_below_interval": int(true_d < lower),
                     "truth_above_interval": int(true_d > upper),
                     "constant_response_count": pred_meta["constant_response_count"],
                     "nonconstant_response_count": 32-pred_meta["constant_response_count"],
                     "convergence_warning_count": pred_meta["convergence_warning_count"],
                     "n_draws": N_DRAWS})
        all_d.append(d)
        all_l.append(sampled_l)
        all_m.append(sampled_m)
        all_hard_d.append(hard_d)
        completion_hashes.append(hashlib.sha256(sampled.tobytes()).hexdigest())
    result_path = base / "evaluation" / f"{matrix_id}.npz"
    np.savez_compressed(result_path, D_samples=np.asarray(all_d, dtype=np.int16),
                        L_samples=np.asarray(all_l, dtype=np.int16), M_samples=np.asarray(all_m, dtype=np.int16),
                        hard_D=np.asarray(all_hard_d, dtype=np.int16),
                        old_L=np.array(old_l), old_M=np.array(old_m),
                        true_L=np.array(true_l), true_M=np.array(true_m), true_D=np.array(true_d))
    write_json(base / "evaluation" / f"{matrix_id}.json", {
        "prediction_sha256_verified_before_truth_read": pred_meta["prediction_sha256"],
        "result_sha256": sha256(result_path), "completion_sha256_by_scheme": dict(zip(SCHEMES, completion_hashes)),
        "marginal_uniform_seed_sequence": [PROPAGATION_SEED, entry["scenario_ordinal"], entry["replicate"], 0],
        "joint_seed_sequence": [PROPAGATION_SEED, entry["scenario_ordinal"], entry["replicate"], 1]})
    return rows


def wilson(count, n):
    z = 1.959963984540054
    p = count/n
    center = (p+z*z/(2*n))/(1+z*z/n)
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return center-half, center+half


def summarize(base, rows):
    by_scenario = {s["scenario_id"]: s for s in scenarios()}
    summary, paired = [], []
    for scenario_id, scheme in itertools.product(by_scenario, SCHEMES):
        subset = [r for r in rows if r["scenario_id"] == scenario_id and r["scheme"] == scheme]
        n = len(subset)
        if not n:
            continue
        hit = sum(r["truth_in_interval"] for r in subset)
        low, high = wilson(hit, n)
        record = {**by_scenario[scenario_id], "scheme": scheme, "n_matrices": n,
                  "interval_coverage": hit/n, "coverage_mcse": math.sqrt((hit/n)*(1-hit/n)/n),
                  "coverage_wilson_lower95": low, "coverage_wilson_upper95": high,
                  "count_truth_below_interval": sum(r["truth_below_interval"] for r in subset),
                  "count_truth_above_interval": sum(r["truth_above_interval"] for r in subset)}
        for name in ("accuracy", "brier", "log_loss", "mean_max_probability", "true_D", "hard_D", "hard_D_error",
                     "hard_D_absolute_error", "mean_D", "mean_D_error", "mean_D_absolute_error", "sd_D", "mean_D_mcse",
                     "interval_width", "constant_response_count", "convergence_warning_count"):
            values = np.asarray([r[name] for r in subset])
            record[name+"_mean"] = float(values.mean())
            record[name+"_mcse"] = float(values.std(ddof=1)/math.sqrt(n)) if n > 1 else None
        record["zero_true_probability_total"] = sum(r["zero_true_probability_count"] for r in subset)
        record["log_floor_total"] = sum(r["log_floor_count"] for r in subset)
        summary.append(record)
    index = {(r["matrix_id"], r["scheme"]): r for r in rows}
    for scenario_id, reference, candidate in itertools.product(by_scenario, SCHEMES, SCHEMES):
        if SCHEMES.index(reference) >= SCHEMES.index(candidate):
            continue
        ids = [r["matrix_id"] for r in rows if r["scenario_id"] == scenario_id and r["scheme"] == reference]
        if not ids:
            continue
        record = {"scenario_id": scenario_id, "reference": reference, "candidate": candidate, "n_matrices": len(ids)}
        for name in ("accuracy", "brier", "log_loss", "mean_D_error", "mean_D_absolute_error", "interval_width", "truth_in_interval"):
            differences = np.asarray([index[(i, candidate)][name]-index[(i, reference)][name] for i in ids])
            record[name+"_difference_mean"] = float(differences.mean())
            record[name+"_difference_mcse"] = float(differences.std(ddof=1)/math.sqrt(len(ids))) if len(ids)>1 else None
        paired.append(record)
    write_csv(base / "matrix_results.csv", rows)
    write_csv(base / "scenario_summary.csv", summary)
    write_csv(base / "paired_method_contrasts.csv", paired)


def evaluation(base):
    base = Path(base)
    (base / "evaluation").mkdir(exist_ok=False)
    inputs = json.loads((base / "input_manifest.json").read_text())
    fit_manifest = json.loads((base / "FIT_MANIFEST.json").read_text())
    if fit_manifest["failed_matrices"]:
        raise RuntimeError("Fit failures retained in FIT_MANIFEST; resolve or explicitly account before evaluation")
    if fit_manifest["completed_matrices"] != len(inputs):
        raise RuntimeError("Incomplete fit predictions")
    started = time.time()
    rows = []
    for ordinal, entry in enumerate(inputs):
        rows.extend(evaluate_matrix(base, entry))
        if (ordinal+1) % 50 == 0:
            print(json.dumps({"evaluated_matrices": ordinal+1, "seconds": round(time.time()-started, 2)}), flush=True)
    summarize(base, rows)
    write_json(base / "EVALUATION_MANIFEST.json", {"completed_matrices": len(inputs), "schemes": SCHEMES,
               "rows": len(rows), "draws_per_scheme_matrix": N_DRAWS,
               "seconds": time.time()-started, "script_sha256": sha256(__file__),
               "fit_manifest_sha256": sha256(base / "FIT_MANIFEST.json")})


def self_test(output):
    """Engineering check only: two toy trees, no study-scenario outcome pilot."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    import itertools as it
    edges = random_tree(6, np.random.default_rng(7201))
    patterns = np.asarray(list(it.product((0, 1), repeat=6)), dtype=np.int8)
    scores = parsimony_costs(patterns, edges)
    # Independent internal assignments in this small unit check.
    brute = []
    for pattern in patterns:
        candidate_scores = []
        for internals in it.product((0, 1), repeat=4):
            all_states = np.concatenate((pattern, internals))
            candidate_scores.append(sum(all_states[a] != all_states[b] for a,b in edges))
        brute.append(min(candidate_scores))
    assert np.array_equal(scores, brute)
    assert scores[0] == scores[-1] == 0
    assert np.array_equal(scores, scores[::-1])
    truth, root = generate_states(edges, 6, 12, 0.1, np.random.default_rng(7202))
    masked = truth[:6].astype(np.int8)
    mask, anchor = make_mask(masked, edges, 2, "structured", np.random.default_rng(7203), n_anchors=2)
    masked[mask] = -1
    assert total_costs(masked, edges)[0] <= total_costs(truth[:6], edges)[0]
    targets = np.argwhere(mask)
    completed_l, completed_m = sample_costs(masked, truth[:6][mask][None,:], edges)
    assert (completed_l[0], completed_m[0]) == total_costs(truth[:6], edges)
    np.savez_compressed(output / "engineering_fixture.npz", edges=edges, truth=truth[:6], masked=masked,
                        root=np.array(root), patterns=patterns, expected_costs=scores)
    write_json(output / "ENGINEERING_CHECKS.json", {"status": "PASS", "n_complete_patterns": len(patterns),
               "fixed_tree_score_against_internal_enumeration": True,
               "missing_state_monotonicity": True, "completion_preserves_observed": True,
               "production_results_generated": False})
    print("Engineering checks PASS", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["generate", "evaluate", "self-test"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frozen-design", type=Path)
    args = parser.parse_args()
    if args.action == "generate":
        generation(args.output, args.frozen_design)
    elif args.action == "evaluate":
        evaluation(args.output)
    else:
        self_test(args.output)
