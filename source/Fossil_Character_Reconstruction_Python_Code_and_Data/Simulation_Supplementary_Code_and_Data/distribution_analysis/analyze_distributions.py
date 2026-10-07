#!/usr/bin/env python3
"""Post hoc evaluation of frozen Round 16 simulation forecasts; no fitting.

CRPS of the ISSUED EMPIRICAL distribution of M draws x_i, lower is better:
    mean_i |x_i-y| - (1/(2 M^2)) sum_i sum_j |x_i-x_j|.
The optional fair-ensemble estimate uses M(M-1) in the pairwise denominator.
The two are different estimands; the empirical score is the primary result.

The oracle-centred diagnostic subtracts each forecast's realised mean error,
using that SAME matrix's truth. It is not a valid forecast or a fitted
calibration method. Its results are saved separately from original forecasts.

State losses partition into disjoint cell groups. Signed D bias partitions
additively across disjoint CHARACTER groups, not individual cells. No additive
decomposition of total CRPS into character contributions is claimed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import time
import types

import numpy as np
import pandas as pd

METHODS = ("mode", "knn5", "logistic", "extra_trees", "nj_mk")
SCHEMES = METHODS + ("nj_mk_joint",)
LOG_FLOOR = 1e-12
ENGINE = None
ENGINE_PATH = None


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def crps_empirical(draws, truth, *, fair=False):
    """O(M log M) CRPS; supports ties, negative draws, and point forecasts."""
    values = np.asarray(draws, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or not np.isfinite(truth):
        raise ValueError("CRPS requires a finite nonempty vector and finite truth")
    m = len(values)
    if fair and m < 2:
        raise ValueError("Fair-ensemble estimate requires at least two draws")
    ordered = np.sort(values)
    # Subtract the minimum to avoid cancellation under a common large offset.
    pair_sum_half = np.dot(2 * np.arange(1, m + 1) - m - 1, ordered - ordered[0])
    denominator = m * (m - 1) if fair else m * m
    return float(np.mean(np.abs(values - truth)) - pair_sum_half / denominator)


def distribution_metrics(draws, truth):
    draws = np.asarray(draws, dtype=np.float64)
    low, high = np.quantile(draws, [0.025, 0.975], method="linear")
    bias = float(draws.mean() - truth)
    return {
        "crps_empirical": crps_empirical(draws, truth),
        "crps_fair_ensemble": crps_empirical(draws, truth, fair=True),
        "mean_D": float(draws.mean()), "signed_bias": bias,
        "absolute_mean_error": abs(bias), "sd_D": float(draws.std(ddof=1)),
        "lower95": float(low), "upper95": float(high), "width95": float(high-low),
        "truth_in_95": int(low <= truth <= high),
        "truth_below_95": int(truth < low), "truth_above_95": int(truth > high),
    }


def state_group_masks(masked, truth, targets, p_true):
    """Disjoint cell groups, defined from old training labels and saved p_true."""
    targets = np.asarray(targets, int)
    train_states = [set(masked[masked[:, j] >= 0, j].tolist()) for j in range(masked.shape[1])]
    constants = np.array([len(states) == 1 for states in train_states])
    y = truth[targets[:, 0], targets[:, 1]]
    unseen = np.array([int(v) not in train_states[j] for v, j in zip(y, targets[:, 1])])
    constant_target = constants[targets[:, 1]]
    if not np.all(~unseen | constant_target):
        raise AssertionError("Binary unseen states should occur only in constant-training columns")
    if not np.all(p_true[unseen] == 0):
        raise AssertionError("Training-unseen state received nonzero stored probability")
    groups = {
        "training_unseen_state": unseen,
        "constant_training_seen_state": constant_target & ~unseen,
        "nonconstant_training_zero_true_probability": ~constant_target & (p_true == 0),
        "nonconstant_training_positive_true_probability": ~constant_target & (p_true > 0),
    }
    if not np.all(np.sum(list(groups.values()), axis=0) == 1):
        raise AssertionError("Cell contribution groups are not a partition")
    return groups, constants, unseen


def load_engine(root):
    """Load original independent scorer without writing source-directory caches."""
    global ENGINE, ENGINE_PATH
    path = Path(root) / "validation" / "independent_engine.py"
    if ENGINE is None or ENGINE_PATH != str(path):
        ENGINE = types.ModuleType("frozen_independent_scorer")
        exec(compile(path.read_text(), str(path), "exec"), ENGINE.__dict__)
        ENGINE_PATH = str(path)
    return ENGINE


def character_excess(matrices, edges, engine):
    matrices = np.asarray(matrices, dtype=np.int8)
    n_tips, n_chars = matrices.shape[-2:]
    leading = matrices.shape[:-2]
    patterns = np.swapaxes(matrices, -1, -2).reshape(-1, n_tips)
    lengths = engine.fitch_character_lengths(patterns, edges).reshape(leading + (n_chars,))
    minima = np.any(matrices == 0, axis=-2) & np.any(matrices == 1, axis=-2)
    return lengths - minima.astype(int)


def audit_one(argument):
    root_s, registry_row = argument
    root = Path(root_s)
    base = root / "simulation" / "results"
    matrix_id = registry_row["matrix_id"]
    paths = {
        "masked": base / "inputs" / f"{matrix_id}_masked.npz",
        "truth": base / "truth" / f"{matrix_id}_truth.npz",
        "prediction": base / "predictions" / f"{matrix_id}.npz",
        "prediction_metadata": base / "predictions" / f"{matrix_id}.json",
        "evaluation": base / "evaluation" / f"{matrix_id}.npz",
        "evaluation_metadata": base / "evaluation" / f"{matrix_id}.json",
    }
    metadata = json.loads(paths["prediction_metadata"].read_text())
    evalmeta = json.loads(paths["evaluation_metadata"].read_text())
    hashes = {key: sha256(path) for key, path in paths.items()}
    assert hashes["masked"] == registry_row["masked_sha256"] == metadata["masked_sha256"]
    assert hashes["truth"] == registry_row["truth_sha256"]
    assert hashes["prediction"] == metadata["prediction_sha256"] == evalmeta["prediction_sha256_verified_before_truth_read"]
    assert hashes["evaluation"] == evalmeta["result_sha256"]
    assert metadata["status"] == "completed" and not metadata["failures"]
    assert tuple(metadata["methods"]) == METHODS and tuple(metadata["schemes"]) == SCHEMES
    with np.load(paths["masked"], allow_pickle=False) as z:
        masked = z["X"]
    with np.load(paths["truth"], allow_pickle=False) as z:
        truth, edges = z["X_true"], z["edges"]
    with np.load(paths["prediction"], allow_pickle=False) as z:
        probabilities, targets, packed = z["probabilities"], z["target_indices"], z["missing_states_packed"]
    with np.load(paths["evaluation"], allow_pickle=False) as z:
        D, true_D, old_L, old_M = z["D_samples"], float(z["true_D"]), float(z["old_L"]), float(z["old_M"])
    assert np.array_equal(targets, np.argwhere(masked == -1))
    assert probabilities.shape == (5, *masked.shape, 2)
    assert np.isfinite(probabilities).all() and (probabilities >= 0).all()
    assert np.allclose(probabilities.sum(axis=-1), 1, atol=1e-12)
    assert np.array_equal(masked[masked >= 0], truth[masked >= 0])
    for method_probs in probabilities:
        for state in (0, 1):
            assert np.array_equal(method_probs[:, :, state][masked >= 0], (masked[masked >= 0] == state).astype(float))
    y = truth[targets[:, 0], targets[:, 1]]
    n_targets, n_draws = len(targets), D.shape[1]
    assert D.shape == (6, 512) and n_draws == metadata["n_draws"]
    draws = np.unpackbits(packed, axis=2, count=n_targets)
    engine = load_engine(root)
    old_by_character = character_excess(masked, edges, engine)
    truth_by_character = character_excess(truth, edges, engine) - old_by_character
    assert old_by_character.sum() == old_L-old_M
    assert truth_by_character.sum() == true_D
    response_columns = np.any(masked == -1, axis=0)
    common = {key: registry_row[key] for key in ("matrix_id", "scenario_id", "replicate", "q", "missing_per_response", "mask_mode")}
    common.update(n_targets=n_targets, n_draws=n_draws, true_D=true_D)
    distributions, oracle, state_rows, component_rows = [], [], [], []
    training_classes = [set(masked[masked[:, j] >= 0, j].tolist()) for j in range(masked.shape[1])]
    diagnostic_by_col = {int(d["character"]): d for d in metadata["diagnostics"]}
    for j in np.flatnonzero(response_columns):
        assert bool(diagnostic_by_col[int(j)]["constant_fallback"]) == (len(training_classes[j]) == 1)
    for si, scheme in enumerate(SCHEMES):
        method_index = si if si < 5 else 4
        p = probabilities[method_index, targets[:, 0], targets[:, 1], :]
        p_true = p[np.arange(n_targets), y]
        brier = np.sum((p - np.eye(2)[y]) ** 2, axis=1)
        logloss = -np.log(np.maximum(p_true, LOG_FLOOR))
        errors = np.argmax(p, axis=1) != y
        groups, constants, unseen = state_group_masks(masked, truth, targets, p_true)
        metrics = distribution_metrics(D[si], true_D)
        state = {"brier": float(brier.mean()), "log_loss": float(logloss.mean()),
                 "error_rate": float(errors.mean()), "training_unseen_fraction": float(unseen.mean()),
                 "constant_target_fraction": float(constants[targets[:, 1]].mean()),
                 "zero_true_probability_fraction": float((p_true == 0).mean()),
                 "constant_response_count": int(np.sum(constants & response_columns))}
        distributions.append({**common, "scheme": scheme, **metrics, **state})
        centred = D[si].astype(float)-metrics["signed_bias"]
        centred_metrics = distribution_metrics(centred, true_D)
        assert abs(centred_metrics["signed_bias"]) < 1e-9
        assert np.isclose(centred_metrics["width95"], metrics["width95"], atol=1e-9)
        oracle.append({**common, "scheme": scheme,
                       "diagnostic": "same_matrix_truth_oracle_mean_shift_not_a_valid_forecast",
                       "applied_shift": -metrics["signed_bias"], **centred_metrics,
                       "crps_change_from_original": centred_metrics["crps_empirical"]-metrics["crps_empirical"],
                       "inclusion_change_from_original": centred_metrics["truth_in_95"]-metrics["truth_in_95"]})
        for group, select in groups.items():
            count = int(select.sum())
            state_rows.append({**common, "scheme": scheme, "group": group, "group_n_targets": count,
                "target_fraction": count/n_targets, "error_count": int(errors[select].sum()),
                "brier_sum": float(brier[select].sum()), "log_loss_sum": float(logloss[select].sum()),
                "error_fraction_all_targets": float(errors[select].sum()/n_targets),
                "brier_contribution_all_targets": float(brier[select].sum()/n_targets),
                "log_loss_contribution_all_targets": float(logloss[select].sum()/n_targets),
                "brier_within_group": float(brier[select].mean()) if count else np.nan,
                "log_loss_within_group": float(logloss[select].mean()) if count else np.nan,
                "zero_true_probability_count": int(np.sum(p_true[select] == 0))})

        # Independent Fitch reconstruction for all six saved completion arrays.
        completed = np.broadcast_to(masked, (n_draws, *masked.shape)).copy()
        completed[:, targets[:, 0], targets[:, 1]] = draws[si]
        per_character = character_excess(completed, edges, engine)-old_by_character[None, :]
        assert np.array_equal(per_character.sum(axis=1), D[si])
        assert np.all(per_character[:, constants] == 0), "A constant-class completion must add zero excess cost"
        columns_unseen = np.zeros(masked.shape[1], bool)
        columns_zero = np.zeros(masked.shape[1], bool)
        columns_unseen[targets[unseen, 1]] = True
        columns_zero[targets[p_true == 0, 1]] = True
        character_groups = {
            "constant_training_with_unseen_targets": response_columns & constants & columns_unseen,
            "constant_training_without_unseen_targets": response_columns & constants & ~columns_unseen,
            "nonconstant_training_with_zero_probability_targets": response_columns & ~constants & columns_zero,
            "nonconstant_training_without_zero_probability_targets": response_columns & ~constants & ~columns_zero,
        }
        assert np.array_equal(np.sum(list(character_groups.values()), axis=0), response_columns.astype(int))
        component_bias_sum = 0.0
        for group, select in character_groups.items():
            component_truth = float(truth_by_character[select].sum())
            component_mean = float(per_character[:, select].sum(axis=1).mean())
            component_bias_sum += component_mean-component_truth
            component_rows.append({**common, "scheme": scheme, "character_group": group,
                "n_response_characters": int(select.sum()),
                "true_D_component": component_truth, "mean_D_component": component_mean,
                "signed_bias_contribution": component_mean-component_truth})
        assert np.isclose(component_bias_sum, metrics["signed_bias"], atol=1e-10)
    checks = {"matrix_id": matrix_id, "hashes": hashes, "target_count": n_targets,
              "source_probability_checks": True, "source_constant_fallback_checks": True,
              "all_six_D_distributions_independently_reconstructed": True,
              "D_character_bias_partition_checked": True,
              "oracle_translation_preserves_width": True}
    return distributions, oracle, state_rows, component_rows, checks


def aggregate(frame, groups, metrics):
    """One independent matrix per group row; MCSE is sample SD / sqrt(n)."""
    rows = []
    for keys, subset in frame.groupby(groups, sort=True, dropna=False):
        if not isinstance(keys, tuple): keys = (keys,)
        result = dict(zip(groups, keys))
        assert subset.matrix_id.nunique() == len(subset)
        result["n_matrices"] = len(subset)
        for metric in metrics:
            values = subset[metric].to_numpy(float)
            finite = values[np.isfinite(values)]
            result[metric+"_n"] = len(finite)
            result[metric+"_mean"] = float(finite.mean()) if len(finite) else np.nan
            result[metric+"_MCSE"] = float(finite.std(ddof=1)/math.sqrt(len(finite))) if len(finite) > 1 else np.nan
        if "truth_in_95" in metrics:
            successes = int(subset.truth_in_95.sum())
            result["inclusion_count"] = successes
            result["inclusion_Wilson_low"], result["inclusion_Wilson_high"] = wilson(successes, len(subset))
        rows.append(result)
    return pd.DataFrame(rows)


def wilson(successes, total):
    z = 1.959963984540054
    p = successes/total
    d = 1+z*z/total
    midpoint = (p+z*z/(2*total))/d
    half = z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/d
    return midpoint-half, midpoint+half


def paired_contrasts(frame, metrics):
    rows = []
    for scenario, subset in frame.groupby("scenario_id", sort=True):
        for reference, candidate in itertools.combinations(SCHEMES, 2):
            left = subset[subset.scheme == reference].set_index("matrix_id")
            right = subset[subset.scheme == candidate].set_index("matrix_id")
            assert set(left.index) == set(right.index)
            right = right.loc[left.index]
            for metric in metrics:
                difference = right[metric].to_numpy(float)-left[metric].to_numpy(float)
                rows.append({"scenario_id": scenario, "reference": reference, "candidate": candidate,
                    "difference_direction": "candidate_minus_reference", "metric": metric,
                    "n_paired_matrices": len(difference), "mean_difference": float(difference.mean()),
                    "paired_MCSE": float(difference.std(ddof=1)/math.sqrt(len(difference)))})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int, help="Engineering subset only; omit for scientific results")
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    # The source archive is always read-only, including rejection of nested output.
    if output == root or root in output.parents:
        raise ValueError("Output must not be inside the source archive")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use an empty/new output directory to preserve prior results")
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    registry_path = root / "simulation/results/matrix_registry.csv"
    registry = pd.read_csv(registry_path).sort_values(["scenario_id", "replicate"])
    if args.limit: registry = registry.head(args.limit)
    engine = load_engine(root)
    toy_check = engine.self_test()
    payload = [(str(root), row) for row in registry.to_dict("records")]
    records, oracle_records, state_records, component_records, checks = [], [], [], [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, result in enumerate(pool.map(audit_one, payload), 1):
            distributions, oracle, state, components, checked = result
            records.extend(distributions); oracle_records.extend(oracle)
            state_records.extend(state); component_records.extend(components); checks.append(checked)
            if i % 50 == 0 or i == len(payload): print(f"Verified and analysed {i}/{len(payload)} matrices", flush=True)
    original = pd.DataFrame(records)
    oracle = pd.DataFrame(oracle_records)
    states = pd.DataFrame(state_records)
    components = pd.DataFrame(component_records)
    metric_names = ["crps_empirical", "crps_fair_ensemble", "signed_bias", "absolute_mean_error", "width95", "truth_in_95", "truth_below_95", "truth_above_95", "brier", "log_loss", "error_rate", "training_unseen_fraction", "constant_target_fraction", "zero_true_probability_fraction", "constant_response_count"]
    original.to_csv(output / "matrix_distribution_scores.csv", index=False)
    aggregate(original, ["scenario_id", "q", "missing_per_response", "mask_mode", "scheme"], metric_names).to_csv(output / "scenario_distribution_scores.csv", index=False)
    paired_contrasts(original, metric_names[:11]).to_csv(output / "paired_method_differences.csv", index=False)
    oracle.to_csv(output / "matrix_oracle_centered_diagnostic.csv", index=False)
    aggregate(oracle, ["scenario_id", "q", "missing_per_response", "mask_mode", "scheme"], metric_names[:8]+["crps_change_from_original", "inclusion_change_from_original"]).to_csv(output / "scenario_oracle_centered_diagnostic.csv", index=False)
    states.to_csv(output / "matrix_state_loss_contributions.csv", index=False)
    state_metrics = ["group_n_targets", "target_fraction", "error_fraction_all_targets", "brier_contribution_all_targets", "log_loss_contribution_all_targets", "zero_true_probability_count"]
    aggregate(states, ["scenario_id", "q", "missing_per_response", "mask_mode", "scheme", "group"], state_metrics).to_csv(output / "scenario_state_loss_contributions.csv", index=False)
    components.to_csv(output / "matrix_D_bias_components.csv", index=False)
    aggregate(components, ["scenario_id", "q", "missing_per_response", "mask_mode", "scheme", "character_group"], ["n_response_characters", "true_D_component", "mean_D_component", "signed_bias_contribution"]).to_csv(output / "scenario_D_bias_components.csv", index=False)
    # Cross-check every existing metric re-evaluated here against the frozen table.
    old = pd.read_csv(root / "simulation/results/matrix_results.csv")
    merged = original.merge(old, on=["matrix_id", "scheme"], suffixes=("_new", "_source"), validate="one_to_one")
    maximum_errors = {}
    for new, old_name in [("signed_bias", "mean_D_error"), ("width95", "interval_width"), ("truth_in_95", "truth_in_interval"), ("brier_new", "brier_source"), ("log_loss_new", "log_loss_source")]:
        error = float(np.max(np.abs(merged[new].to_numpy(float)-merged[old_name].to_numpy(float))))
        assert error < 1e-10, (new, error)
        maximum_errors[new] = error
    manifest = {
        "status": "PASS", "matrices": len(registry), "complete_600_matrix_analysis": len(registry) == 600 and args.limit is None,
        "forecast_rows": len(original), "state_partition_rows": len(states), "D_partition_rows": len(components),
        "draws_per_distribution": 512, "source_registry_sha256": sha256(registry_path),
        "source_independent_engine_sha256": sha256(root / "validation/independent_engine.py"),
        "analysis_script_sha256": sha256(__file__), "python": sys.version,
        "numpy": np.__version__, "pandas": pd.__version__, "seconds": time.monotonic()-started,
        "source_scorer_toy_check": toy_check, "existing_metric_maximum_errors": maximum_errors,
        "crps_empirical_formula": "M^-1 sum_i |D_i-y| - (2M^2)^-1 sum_i sum_j |D_i-D_j|",
        "crps_fair_formula": "M^-1 sum_i |D_i-y| - [2M(M-1)]^-1 sum_i sum_j |D_i-D_j|",
        "primary_score": "CRPS of frozen empirical distribution; lower is better",
        "fair_estimator_note": "Secondary finite-ensemble sensitivity for independent Monte Carlo draws, not the score of the empirical forecast distribution",
        "inference_unit": "Independent matrix; 50 matrices per scenario. Paired scheme differences use the same matrices",
        "post_hoc_status": "All added analyses specified after original results were available",
        "oracle_diagnostic_warning": "Uses same-matrix true D to remove mean bias. Not a valid forecast, estimated calibration, or out-of-sample correction",
        "D_partition_warning": "Signed bias decomposes over characters. CRPS and inclusion do not have an additive character decomposition",
        "references": ["Gneiting & Raftery 2007, doi:10.1198/016214506000001437", "Ferro 2014, doi:10.1002/qj.2270"],
        "matrices_checked": checks,
    }
    (output / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k: manifest[k] for k in ("status", "matrices", "forecast_rows", "seconds", "existing_metric_maximum_errors")}, indent=2))


if __name__ == "__main__":
    main()
