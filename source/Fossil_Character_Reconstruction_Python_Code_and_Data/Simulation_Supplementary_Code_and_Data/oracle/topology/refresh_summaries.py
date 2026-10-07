#!/usr/bin/env python3
"""Rebuild descriptive tables from fixed verified score arrays without rescoring."""
import json
from pathlib import Path

import numpy as np

from run_topology import matrix_rows, scenario_summary, paired_summary, write_csv, sha256


def main():
    here = Path(__file__).resolve().parent
    output = here / "results"
    manifest = json.loads((here / "topology_input_manifest.json").read_text())
    checks = json.loads((output / "TOPOLOGY_CHECKS.json").read_text())
    hashes = {row["scenario_id"]: row["output_sha256"] for row in checks["checks"]}
    matrix, scenario, paired = [], [], []
    for entry in manifest["scenarios"]:
        scenario_id = entry["scenario_id"]
        source = output / "arrays" / f"{scenario_id}_topology.npz"
        if sha256(source) != hashes[scenario_id]:
            raise AssertionError(f"Saved score arrays changed: {source}")
        metadata = {k: entry[k] for k in ("scenario_id", "ntips", "nchar", "shape", "q", "k_missing", "mask_regime")}
        with np.load(source, allow_pickle=False) as data:
            fixed_arrays = {key: data[key] for key in data.files}
        rows = matrix_rows(scenario_id, fixed_arrays)
        matrix.extend(rows)
        scenario.extend(scenario_summary(rows, metadata))
        paired.extend(paired_summary(rows, metadata))
    write_csv(output / "topology_matrix_summary.csv", matrix)
    keys = list(dict.fromkeys(k for row in scenario for k in row))
    scenario = [{k: row.get(k, row["n_matrices"] if k.endswith("_n_valid") else float("nan")) for k in keys} for row in scenario]
    write_csv(output / "topology_scenario_summary.csv", scenario)
    write_csv(output / "topology_paired_contrasts.csv", paired)
    log = {"status": "PASS", "saved_arrays_unchanged": True, "new_simulations_or_draws": 0,
           "matrix_rows": len(matrix), "scenario_rows": len(scenario),
           "additional_description": "D equals truth D and mean tied-MPT RF exceeds truth-complete RF by >1e-12",
           "reason": "Separate endpoint non-identifiability from the complete matrix's own finite-character recovery error",
           "timing": "Requested descriptive summary after inspecting initial topology results; all scenarios retained"}
    (output / "SUMMARY_REFRESH.json").write_text(json.dumps(log, indent=2) + "\n")
    print(json.dumps(log))


if __name__ == "__main__":
    main()
