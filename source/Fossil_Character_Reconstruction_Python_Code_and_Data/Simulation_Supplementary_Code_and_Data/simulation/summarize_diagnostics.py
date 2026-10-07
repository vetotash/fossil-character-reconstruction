#!/usr/bin/env python3
"""Collect fit diagnostics and exact-zero forecast frequencies without refitting."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path


def run(base):
    base = Path(base)
    manifest = json.loads((base / "FIT_MANIFEST.json").read_text())
    records = []
    for item in manifest["records"]:
        if item["status"] != "completed":
            continue
        meta = json.loads((base / "predictions" / f"{item['matrix_id']}.json").read_text())
        for d in meta["diagnostics"]:
            nj = d["nj_mk"]
            optimizer = nj.get("optimizer", {})
            logistic = d["classifiers"]["logistic"]
            records.append({
                "matrix_id": item["matrix_id"], "scenario_id": item["matrix_id"].split("_")[0],
                "character": d["character"], "n_train": d["n_train"], "n_test": d["n_test"],
                "constant_fallback": int(d["constant_fallback"]),
                "logistic_fit_ran": int(logistic["fit_ran"]),
                "logistic_convergence_warning": int(logistic.get("convergence_warning", False)),
                "logistic_reached_max_iter": int(logistic.get("reached_max_iter", False)),
                "nj_zero_overlap_pairs": nj["zero_overlap_pairs"],
                "nj_negative_branches": nj["negative_branches"],
                "nj_truncated_branches": nj["truncated_branches"],
                "mk_rate": nj["rate"], "mk_rate_boundary": int(nj["rate_boundary"]),
                "mk_optimizer_ran": int(optimizer.get("ran", False)),
                "mk_optimizer_success": int(optimizer.get("success", False)) if optimizer.get("ran", False) else "not_run",
                "mk_warning_count": len(nj.get("warnings", [])),
            })
    with (base / "fit_character_diagnostics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    scenario_summary = []
    for scenario in sorted(set(r["scenario_id"] for r in records)):
        subset = [r for r in records if r["scenario_id"] == scenario]
        counts = {key: sum(r[key] for r in subset) for key in (
            "constant_fallback", "logistic_fit_ran", "logistic_convergence_warning", "logistic_reached_max_iter",
            "nj_zero_overlap_pairs", "nj_negative_branches", "nj_truncated_branches", "mk_rate_boundary",
            "mk_optimizer_ran", "mk_warning_count")}
        counts["mk_optimizer_failed"] = sum(r["mk_optimizer_ran"] and not r["mk_optimizer_success"] for r in subset)
        scenario_summary.append({"scenario_id": scenario, "response_fits": len(subset), **counts})
    with (base / "fit_diagnostics_by_scenario.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(scenario_summary[0]))
        writer.writeheader()
        writer.writerows(scenario_summary)
    with (base / "matrix_results.csv").open(newline="") as file:
        metrics = list(csv.DictReader(file))
    zeros = []
    for scenario, scheme in sorted(set((r["scenario_id"], r["scheme"]) for r in metrics)):
        subset = [r for r in metrics if r["scenario_id"] == scenario and r["scheme"] == scheme]
        n = sum(int(r["n_targets"]) for r in subset)
        zero = sum(int(r["zero_true_probability_count"]) for r in subset)
        floored = sum(int(r["log_floor_count"]) for r in subset)
        zeros.append({"scenario_id": scenario, "scheme": scheme, "n_matrices": len(subset),
                      "n_target_cells": n, "zero_true_probability_count": zero,
                      "zero_true_probability_proportion": zero/n,
                      "log_floor_count": floored, "log_floor_proportion": floored/n})
    with (base / "zero_probability_by_scenario.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(zeros[0]))
        writer.writeheader()
        writer.writerows(zeros)
    summary = {"completed_matrices": manifest["completed_matrices"],
               "failed_matrix_count": len(manifest["failed_matrices"]),
               "response_fits": len(records),
               "constant_response_fits": sum(r["constant_fallback"] for r in records),
               "logistic_convergence_warnings": sum(r["logistic_convergence_warning"] for r in records),
               "logistic_reached_max_iter": sum(r["logistic_reached_max_iter"] for r in records),
               "mk_optimizer_calls": sum(r["mk_optimizer_ran"] for r in records),
               "mk_optimizer_failures": sum(r["mk_optimizer_ran"] and not r["mk_optimizer_success"] for r in records),
               "mk_rate_boundaries": sum(r["mk_rate_boundary"] for r in records),
               "mk_warning_count": sum(r["mk_warning_count"] for r in records),
               "interpretation": "Rates and branch-boundary diagnostics are retained; none are filtered out. Exact-zero frequencies are cell counts; inferential replication remains the matrix."}
    (base / "FIT_DIAGNOSTICS_SUMMARY.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.output)
