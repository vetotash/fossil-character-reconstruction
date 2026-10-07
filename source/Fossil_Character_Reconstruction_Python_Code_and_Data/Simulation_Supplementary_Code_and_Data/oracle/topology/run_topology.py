#!/usr/bin/env python3
"""Run the frozen exact six-tip topology extension on saved simulation arrays.

No new random draws, optimization heuristics, or fitted models are used.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from exact_topology import METHODS, build_registry, evaluate_scenario, save_registry


HERE = Path(__file__).resolve().parent


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError(f"No rows for {path}")
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def matrix_rows(scenario_id, result):
    nm = len(result["truth_D"])
    out = []
    for matrix in range(nm):
        for m, method in enumerate(METHODS):
            exact = result["draw_D"][matrix, m] == result["truth_D"][matrix]
            absent = ~result["draw_truth_in_mpt"][matrix, m]
            rf = result["draw_mean_normalized_rf"][matrix, m]
            different = np.any(result["draw_mpt_bits"][matrix, m] != result["full_mpt_bits"][matrix], axis=-1)
            worse_full = rf > result["full_mean_normalized_rf"][matrix] + 1e-12
            n_exact = int(np.sum(exact))
            row = {
                "scenario_id": scenario_id,
                "matrix_index": matrix,
                "method": method,
                "n_draws": 64,
                "truth_D": int(result["truth_D"][matrix]),
                "full_mean_normalized_rf": float(result["full_mean_normalized_rf"][matrix]),
                "source_mean_normalized_rf": float(result["source_mean_normalized_rf"][matrix]),
                "map_mean_normalized_rf": float(result["map_mean_normalized_rf"][matrix, m]),
                "full_truth_in_mpt": int(result["full_truth_in_mpt"][matrix]),
                "source_truth_in_mpt": int(result["source_truth_in_mpt"][matrix]),
                "map_truth_in_mpt": int(result["map_truth_in_mpt"][matrix, m]),
                "full_mpt_count": int(result["full_mpt_count"][matrix]),
                "source_mpt_count": int(result["source_mpt_count"][matrix]),
                "map_mpt_count": int(result["map_mpt_count"][matrix, m]),
                "map_absolute_D_error": int(result["map_absolute_D_error"][matrix, m]),
                "draw_mean_normalized_rf": float(np.mean(rf)),
                "draw_truth_mpt_probability": float(np.mean(~absent)),
                "draw_mean_mpt_count": float(np.mean(result["draw_mpt_count"][matrix, m])),
                "draw_mean_absolute_D_error": float(np.mean(result["draw_absolute_D_error"][matrix, m])),
                "draw_mean_D": float(np.mean(result["draw_D"][matrix, m])),
                "D_exact_match_count": n_exact,
                "D_exact_match_probability": n_exact / 64,
                "D_exact_match_truth_absent_joint_probability": float(np.mean(exact & absent)),
                "D_exact_match_nonzero_rf_joint_probability": float(np.mean(exact & (rf > 0))),
                "D_exact_match_mpt_differs_full_joint_probability": float(np.mean(exact & different)),
                "D_exact_match_rf_worse_than_full_count": int(np.sum(exact & worse_full)),
                "D_exact_match_rf_worse_than_full_joint_probability": float(np.mean(exact & worse_full)),
                "baseline_truth_absent_probability": float(np.mean(absent)),
                "baseline_nonzero_rf_probability": float(np.mean(rf > 0)),
                "baseline_mpt_differs_full_probability": float(np.mean(different)),
                "baseline_rf_worse_than_full_probability": float(np.mean(worse_full)),
                "truth_absent_given_D_exact_match": float(np.mean(absent[exact])) if n_exact else float("nan"),
                "mean_rf_given_D_exact_match": float(np.mean(rf[exact])) if n_exact else float("nan"),
                "mpt_differs_full_given_D_exact_match": float(np.mean(different[exact])) if n_exact else float("nan"),
                "rf_worse_than_full_given_D_exact_match": float(np.mean(worse_full[exact])) if n_exact else float("nan"),
            }
            out.append(row)
    return out


def scenario_summary(rows, metadata):
    outputs = []
    numeric_keys = [k for k in rows[0] if k not in {"scenario_id", "matrix_index", "method", "n_draws"}]
    for method in METHODS:
        selected = [r for r in rows if r["method"] == method]
        output = dict(metadata)
        output.update({"method": method, "n_matrices": len(selected), "draws_per_matrix": 64})
        for key in numeric_keys:
            values = np.asarray([r[key] for r in selected], dtype=float)
            valid = values[np.isfinite(values)]
            output[key + "_mean"] = float(np.mean(valid)) if len(valid) else float("nan")
            output[key + "_mcse"] = float(np.std(valid, ddof=1) / np.sqrt(len(valid))) if len(valid) > 1 else float("nan")
            if len(valid) != len(selected):
                output[key + "_n_valid"] = int(len(valid))
        # Pooled rates are a descriptive ratio of within-matrix counts; they
        # carry no confidence interval and are not treated as independent draws.
        n_exact = sum(r["D_exact_match_count"] for r in selected)
        n_bad = sum(r["D_exact_match_truth_absent_joint_probability"] * 64 for r in selected)
        n_diff = sum(r["D_exact_match_mpt_differs_full_joint_probability"] * 64 for r in selected)
        n_worse = sum(r["D_exact_match_rf_worse_than_full_count"] for r in selected)
        output["pooled_exact_match_count"] = int(n_exact)
        output["pooled_truth_absent_given_D_exact_match"] = float(n_bad / n_exact) if n_exact else float("nan")
        output["pooled_mpt_differs_full_given_D_exact_match"] = float(n_diff / n_exact) if n_exact else float("nan")
        output["pooled_rf_worse_than_full_given_D_exact_match"] = float(n_worse / n_exact) if n_exact else float("nan")
        output["matrices_with_exact_D_and_rf_worse_than_full"] = int(sum(r["D_exact_match_rf_worse_than_full_count"] > 0 for r in selected))
        outputs.append(output)
    return outputs


def paired_summary(rows, metadata):
    outputs = []
    metrics = ("draw_mean_normalized_rf", "draw_truth_mpt_probability", "draw_mean_mpt_count",
               "draw_mean_absolute_D_error", "D_exact_match_truth_absent_joint_probability")
    for i, left in enumerate(METHODS):
        for right in METHODS[i + 1:]:
            a = {r["matrix_index"]: r for r in rows if r["method"] == left}
            b = {r["matrix_index"]: r for r in rows if r["method"] == right}
            if a.keys() != b.keys():
                raise AssertionError("Paired matrices do not match")
            out = dict(metadata)
            out.update({"left_method": left, "right_method": right, "difference": "left_minus_right", "n_matrices": len(a)})
            for metric in metrics:
                difference = np.asarray([a[j][metric] - b[j][metric] for j in sorted(a)])
                out[metric + "_difference_mean"] = float(np.mean(difference))
                out[metric + "_difference_mcse"] = float(np.std(difference, ddof=1) / np.sqrt(len(difference)))
            outputs.append(out)
    return outputs


def run(manifest, output):
    manifest = Path(manifest).resolve()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    spec = json.loads(manifest.read_text())
    if spec.get("methods") != list(METHODS):
        raise ValueError("Manifest method order does not match frozen scheme order")
    if spec.get("design_status") != "DESIGN_FROZEN":
        raise ValueError("Scientific run requires DESIGN_FROZEN manifest")
    scenarios = spec["scenarios"]
    if len(scenarios) != 24:
        raise ValueError("Frozen topology design requires 24 six-tip scenarios")
    registry = build_registry()
    save_registry(registry, output / "registry")
    all_matrix, all_scenarios, all_pairs, checks = [], [], [], []
    for index, entry in enumerate(scenarios):
        scenario_id = entry["scenario_id"]
        source = (manifest.parent / entry["input_npz"]).resolve()
        if sha256(source) != entry["input_sha256"]:
            raise AssertionError(f"Input hash changed after manifest: {source}")
        metadata = {k: entry[k] for k in ("scenario_id", "ntips", "nchar", "shape", "q", "k_missing", "mask_regime")}
        with np.load(source, allow_pickle=False) as data:
            if data["full_pattern_ids"].shape != (100, 32):
                raise ValueError("Frozen six-tip scenario must contain 100 matrices x 32 characters")
            result = evaluate_scenario(registry, entry["truth_edges"],
                                       data["full_pattern_ids"], data["missing_mask_bits"],
                                       data["map_pattern_ids"], data["draw_pattern_ids"], data["draw_D"])
            if not np.array_equal(result["truth_D"], data["true_D"]):
                raise AssertionError("Truth D disagrees with main simulation")
            if not np.array_equal(result["map_D"][:, 0], data["marginal_map_D"]):
                raise AssertionError("Cellwise MAP D disagrees with main simulation")
        target = output / "arrays" / f"{scenario_id}_topology.npz"
        target.parent.mkdir(exist_ok=True)
        np.savez_compressed(target, **result)
        rows = matrix_rows(scenario_id, result)
        all_matrix.extend(rows)
        all_scenarios.extend(scenario_summary(rows, metadata))
        all_pairs.extend(paired_summary(rows, metadata))
        checks.append({"scenario_id": scenario_id, "input_sha256": sha256(source), "output_sha256": sha256(target),
                       "input_npz": str(source), "output_npz": str(target), "matrices": 100,
                       "draws_scored": 100 * 3 * 64, "truth_tree_index": int(result["truth_tree_index"]),
                       "saved_draw_D_exact_match": True, "common_MAP_identical": True,
                       "observed_cells_preserved": True})
        print(f"{index + 1}/24 {scenario_id}: 19,200 draws scored, all tied MPTs retained", flush=True)
    write_csv(output / "topology_matrix_summary.csv", all_matrix)
    # Summary rows can have varying n_valid fields; make all fields explicit.
    all_keys = list(dict.fromkeys(k for r in all_scenarios for k in r))
    for row in all_scenarios:
        for key in all_keys:
            if key not in row:
                row[key] = row["n_matrices"] if key.endswith("_n_valid") else float("nan")
    all_scenarios = [{k: r[k] for k in all_keys} for r in all_scenarios]
    write_csv(output / "topology_scenario_summary.csv", all_scenarios)
    write_csv(output / "topology_paired_contrasts.csv", all_pairs)
    final = {"status": "PASS", "independent_audit": "PENDING", "scenarios": 24, "independent_matrices": 2400,
             "draws_scored": 460800, "tree_count": 105, "methods": list(METHODS),
             "topology_averages": "All tied MPTs retained; normalized RF denominator 6",
             "uncertainty": "Monte Carlo standard errors across independent matrices; no draw-level inference",
             "manifest_sha256": sha256(manifest), "checks": checks}
    (output / "TOPOLOGY_CHECKS.json").write_text(json.dumps(final, indent=2) + "\n")
    print(json.dumps({k: final[k] for k in ("status", "scenarios", "independent_matrices", "draws_scored")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=HERE / "results")
    args = parser.parse_args()
    run(args.manifest, args.output)
