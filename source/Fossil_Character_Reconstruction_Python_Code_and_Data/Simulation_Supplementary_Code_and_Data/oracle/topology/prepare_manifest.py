#!/usr/bin/env python3
"""Translate the frozen simulation registry into explicit topology inputs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from exact_topology import METHODS


def main(simulation, output):
    simulation, output = Path(simulation).resolve(), Path(output).resolve()
    rows = list(csv.DictReader((simulation / "scenario_registry.csv").open()))
    trees = json.loads((simulation / "trees.json").read_text())
    scenarios = []
    for row in rows:
        if int(row["n_tips"]) != 6:
            continue
        input_file = simulation / f"{row['scenario_id']}_matrices.npz"
        if not input_file.is_file():
            raise FileNotFoundError(input_file)
        import os
        scenarios.append({
            "scenario_id": row["scenario_id"],
            "input_npz": os.path.relpath(input_file, output.parent),
            "input_sha256": hashlib.sha256(input_file.read_bytes()).hexdigest(),
            "truth_edges": trees[row["tree_key"]]["edges"],
            "ntips": int(row["n_tips"]),
            "nchar": int(row["n_characters"]),
            "shape": row["shape"],
            "q": float(row["q"]),
            "k_missing": int(row["missing_per_character"]),
            "mask_regime": row["mask_mode"],
        })
    if len(scenarios) != 24:
        raise ValueError(f"Expected 24 six-tip scenarios; found {len(scenarios)}")
    plan = simulation.parent.parent / "DESIGN_FREEZE.json"
    frozen = json.loads(plan.read_text())
    if frozen["status"] != "DESIGN_FROZEN":
        raise ValueError("Design not frozen")
    output.write_text(json.dumps({"design_status": "DESIGN_FROZEN", "methods": list(METHODS),
                                  "design_freeze_sha256": hashlib.sha256(plan.read_bytes()).hexdigest(),
                                  "scenarios": scenarios}, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulation-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.simulation_results, args.output)
