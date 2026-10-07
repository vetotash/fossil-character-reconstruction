#!/usr/bin/env python3
"""Export Figure S1 from checked saved arrays into a new external directory.

This additive entry point does not reconstruct missing historical audit receipts,
refit models, generate simulations or search trees. The original plotting and
analysis sources remain unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
METHODS = ("oracle_joint", "product_marginals", "sharpened_product")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_figure_values() -> dict:
    """Check every plotted summary against its saved matrix-level values."""
    simulation = ROOT / "oracle/simulation/results"
    topology = ROOT / "oracle/topology/results"
    matrix = pd.read_csv(simulation / "matrix_results.csv")
    summary = pd.read_csv(simulation / "scenario_summary.csv")
    assert len(matrix) == 14400 and len(summary) == 144
    assert not summary.duplicated(["scenario_id", "method"]).any()
    checked = 0
    for row in summary.itertuples(index=False):
        group = matrix[(matrix.scenario_id == row.scenario_id) & (matrix.method == row.method)]
        assert len(group) == group.replicate.nunique() == 100
        for field in ("n_tips", "shape", "q", "missing_per_character", "mask_mode"):
            assert (group[field] == getattr(row, field)).all(), (row.scenario_id, field)
        np.testing.assert_allclose(
            row.oracle_conditional_interval_mass_mean,
            group.oracle_conditional_interval_mass.mean(), atol=1e-10, rtol=0,
        )
        checked += 1

    matrix = pd.read_csv(topology / "topology_matrix_summary.csv")
    summary = pd.read_csv(topology / "topology_scenario_summary.csv")
    design = json.loads((simulation / "design.json").read_text())
    scenarios = {r["scenario_id"]: r for r in design["scenarios"]}
    assert len(matrix) == 7200 and len(summary) == 72
    assert not summary.duplicated(["scenario_id", "method"]).any()
    for row in summary.itertuples(index=False):
        group = matrix[(matrix.scenario_id == row.scenario_id) & (matrix.method == row.method)]
        group = group.sort_values("matrix_index")
        assert group.matrix_index.tolist() == list(range(100))
        scenario = scenarios[row.scenario_id]
        for target, source in (("ntips", "n_tips"), ("nchar", "n_characters"),
                               ("shape", "shape"), ("q", "q"),
                               ("k_missing", "missing_per_character"), ("mask_regime", "mask_mode")):
            assert getattr(row, target) == scenario[source], (row.scenario_id, target)
        method_index = METHODS.index(row.method)
        with np.load(topology / "arrays" / f"{row.scenario_id}_topology.npz", allow_pickle=False) as arrays:
            for prefix in ("full", "source", "map", "draw"):
                field = prefix + "_mean_normalized_rf"
                expected = arrays[field]
                if prefix == "draw":
                    expected = expected[:, method_index].mean(axis=-1)
                elif expected.ndim == 2:
                    expected = expected[:, method_index]
                np.testing.assert_allclose(group[field].to_numpy(), expected, atol=1e-10, rtol=0)
                np.testing.assert_allclose(getattr(row, field + "_mean"), expected.mean(), atol=1e-10, rtol=0)
                checked += 1
    return {"plotted_summary_values_checked": checked,
            "coverage_scenario_method_rows": 144, "topology_scenario_method_rows": 72,
            "independent_matrices_per_scenario": 100}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="New directory outside this delivered package.")
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output == ROOT or ROOT in output.parents:
        parser.error("--output must be outside the delivered package")
    if output.exists():
        parser.error("--output must be a new directory")
    output.mkdir(parents=True)

    replay = output / "saved_array_replay.json"
    subprocess.run([sys.executable, "-B", str(ROOT / "oracle/validation/replay_saved_arrays.py"),
                    "--output", str(replay)], cwd=ROOT, check=True)
    checked = json.loads(replay.read_text())
    assert checked["status"] == "PASS"
    summary_check = check_figure_values()

    source = ROOT / "oracle/figures/plot_supplementary_s1.py"
    spec = importlib.util.spec_from_file_location("figure_s1_original_plot", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                               "pdf.fonttype": 42, "ps.fonttype": 42})
    simulation_summary = ROOT / "oracle/simulation/results/scenario_summary.csv"
    topology_summary = ROOT / "oracle/topology/results/topology_scenario_summary.csv"
    pdf = output / "supplementary_figure_s1.pdf"
    outputs = [pdf]
    with module.PdfPages(pdf, metadata={"Title": "Supplementary Figure S1", "Author": "",
                                      "Subject": "Saved-array replay of the supplied figure"}) as handle:
        pages = ((module.coverage_page, simulation_summary), (module.topology_page, topology_summary))
        for number, (make, data) in enumerate(pages, 1):
            figure = make(module.rows(data))
            handle.savefig(figure)
            png = output / f"supplementary_figure_s1_page{number}.png"
            figure.savefig(png, dpi=200, facecolor="white")
            module.plt.close(figure)
            outputs.append(png)
    inputs = [simulation_summary, topology_summary,
              ROOT / "oracle/simulation/results/matrix_results.csv",
              ROOT / "oracle/topology/results/topology_matrix_summary.csv"]
    receipt = {
        "status": "PASS",
        "scope": "Fresh saved-array and plotted-summary arithmetic checks followed by figure export; not a recovered historical audit or a new scientific simulation.",
        "new_simulations": 0, "new_model_refits": 0, "new_tree_searches": 0,
        "historical_receipts_created_or_modified": False,
        "original_plotting_source_modified": False,
        "saved_array_check": checked, "figure_input_check": summary_check,
        "original_plotting_source_sha256": digest(source), "wrapper_sha256": digest(Path(__file__)),
        "input_summary_hashes": {str(p.relative_to(ROOT)): digest(p) for p in inputs},
        "output_hashes": {p.name: digest(p) for p in outputs},
        "rendering_note": "Numerical inputs are checked; font/backend/PDF metadata differences can prevent binary identity with archived PDFs.",
    }
    (output / "FIGURE_S1_REPLAY.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "output": str(output), "figure_input_check": summary_check}, indent=2))


if __name__ == "__main__":
    main()
