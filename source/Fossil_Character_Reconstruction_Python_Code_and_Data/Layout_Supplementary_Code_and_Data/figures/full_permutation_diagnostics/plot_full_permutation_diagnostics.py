#!/usr/bin/env python3
"""Reproduce the retained full O/U/T permutation diagnostics.

The original A/B numbers are copied without rounding from the archived plot
table. Panel C is joined by scenario and scheme, then independently checked
against the 30 saved within-matrix CRPS differences for every plotted point.
Run: python plot_full_permutation_diagnostics.py --output-dir PATH
Requires Python 3.12 and matplotlib; all numerical work uses the standard library.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

HERE = Path(__file__).resolve().parent
SCHEMES = ("mode", "knn5", "logistic", "extra_trees", "nj_mk", "nj_mk_joint")
LABELS = ("Frequency", "kNN5", "Logistic", "Extra Trees", "Mk marginal", "Mk joint")
COLORS = ("#555555", "#333333", "#E69F00", "#009E73", "#CC79A7", "#882255")
MARKERS = ("o", "s", "^", "D", "v", "P")
METRICS = ("brier", "mean_D_absolute_error", "crps_empirical")
SOURCE_AB = HERE / "inputs/original_figure2_input_table.csv"
SOURCE_C = HERE / "inputs/paired_layout_contrasts.csv"
SOURCE_C_PAIRS = HERE / "inputs/matrix_paired_layout_differences.csv"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def close(actual, expected, key):
    if not math.isclose(float(actual), float(expected), rel_tol=2e-12, abs_tol=2e-12):
        raise AssertionError((key, actual, expected))


def prepare():
    original = read_csv(SOURCE_AB)
    scores = [r for r in read_csv(SOURCE_C) if r["metric"] == "crps_empirical"]
    pairs = [r for r in read_csv(SOURCE_C_PAIRS) if r["metric"] == "crps_empirical"]
    assert len(original) == len(scores) == 36
    assert len(pairs) == 1080
    lookup = {(r["condition_id"], r["scheme"]): r for r in scores}
    assert len(lookup) == 36
    output = []
    checks = []
    for source in original:
        row = dict(source)
        key = (row["scenario_id"], row["scheme"])
        score = lookup[key]
        assert row["panel"] == score["panel"] and row["q"] == score["q"]
        assert row["difference_direction"] == score["difference_direction"] == "empirical_minus_permuted"
        assert int(row["n_pairs"]) == int(score["n_paired_matrices"]) == 30
        observations = [p for p in pairs if (p["condition_id"], p["scheme"]) == key]
        assert len(observations) == len({p["latent_id"] for p in observations}) == 30
        assert all(p["difference_direction"] == "empirical_minus_permuted" for p in observations)
        differences = [float(p["difference"]) for p in observations]
        close(score["mean_difference"], statistics.mean(differences), (key, "CRPS mean"))
        close(score["paired_MCSE"], statistics.stdev(differences) / math.sqrt(30), (key, "CRPS paired MCSE"))
        row["crps_empirical_difference_mean"] = score["mean_difference"]
        row["crps_empirical_difference_mcse"] = score["paired_MCSE"]
        assert all(row[k] == v for k, v in source.items()), "Original fields changed"
        output.append(row)
        checks.append({"scenario_id": key[0], "scheme": key[1], "n_pairs": 30})
    by_key = {(r["scenario_id"], r["scheme"]): r for r in output}
    scenarios = []
    for sid in sorted({r["scenario_id"] for r in output}, key=lambda s: int(s[1:])):
        one = by_key[(sid, "mode")]
        assert {r["scheme"] for r in output if r["scenario_id"] == sid} == set(SCHEMES)
        for suffix in ("mean", "mcse"):
            close(by_key[(sid, "nj_mk")]["brier_difference_" + suffix],
                  by_key[(sid, "nj_mk_joint")]["brier_difference_" + suffix], (sid, "shared Mk Brier", suffix))
        scenarios.append({"scenario_id": sid, "panel": one["panel"], "q": one["q"]})
    return scenarios, output, checks


def make_plot(scenarios, rows, output):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.4,
                         "axes.labelsize": 9.4, "axes.titlesize": 10,
                         "xtick.labelsize": 9.4, "ytick.labelsize": 9.4,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 7.3), sharey=True)
    fig.subplots_adjust(left=0.175, right=0.985, top=0.825, bottom=0.245, wspace=0.25)
    titles = ("Target-state\nBrier score", "Absolute error\nof mean D", "CRPS of the\nD distribution")
    by_key = {(r["scenario_id"], r["scheme"]): r for r in rows}
    ylabels = [s["panel"].capitalize() + "\nq = " + s["q"] for s in scenarios]
    for panel, (ax, metric, title) in enumerate(zip(axes, METRICS, titles)):
        extrema = [0.0]
        ax.axvline(0, color="#444444", linewidth=0.8, linestyle="--", zorder=1)
        for i, scenario in enumerate(scenarios):
            if i % 2 == 0:
                ax.axhspan(i - 0.49, i + 0.49, facecolor="#F1F1F1", zorder=0)
            for j, scheme in enumerate(SCHEMES):
                if panel == 0 and scheme == "nj_mk_joint":
                    continue
                row = by_key[(scenario["scenario_id"], scheme)]
                mean = float(row[metric + "_difference_mean"])
                mcse = float(row[metric + "_difference_mcse"])
                assert math.isfinite(mean) and math.isfinite(mcse) and mcse >= 0
                y = i + (j - 2.5) * 0.115
                ax.errorbar(mean, y, xerr=mcse, fmt=MARKERS[j], color=COLORS[j],
                            markersize=4.0, capsize=1.8, elinewidth=0.9,
                            linewidth=0, markeredgewidth=0.5, zorder=3)
                extrema.extend((mean - mcse, mean + mcse))
        low, high = min(extrema), max(extrema)
        span = max(high - low, 0.01)
        ax.set_xlim(low - 0.12 * span, high + 0.12 * span)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.set_title(chr(65 + panel) + "  " + title, loc="left", pad=11, fontweight="bold")
        ax.set_xlabel("Paired difference" + ("" if panel == 0 else " (steps)"), labelpad=7)
        ax.set_yticks(range(6), ylabels)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", alpha=0.15)
        ax.set_ylim(5.5, -0.5)
        if panel:
            ax.tick_params(axis="y", labelleft=False)
    handles = [Line2D([0], [0], color=COLORS[j], marker=MARKERS[j], markersize=4.5,
                      linestyle="none", label=LABELS[j]) for j in range(6)]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.58, 0.080),
               ncol=3, frameon=False, fontsize=9.4, columnspacing=1.5, handletextpad=0.5)
    fig.text(0.175, 0.966, "Historical observation patterns in paired simulations",
             fontsize=12, weight="bold", va="top")
    fig.text(0.175, 0.929, "30 latent matrix pairs per condition; mean difference and one paired MCSE",
             fontsize=9.4)
    fig.text(0.58, 0.155, "Empirical minus randomized pattern", ha="center", fontsize=9.4)
    fig.text(0.175, 0.018,
             "Negative values indicate lower error or CRPS under the empirical pattern.\n"
             "Each point compares patterns within a scheme, using each arm’s target-only truth.",
             fontsize=9.4, linespacing=1.55)
    stem = output / "full_permutation_diagnostics"
    fig.savefig(stem.with_suffix(".pdf"), metadata={
        "Title": "Full O/U/T permutation diagnostics",
        "Subject": "Original Brier and mean-D error contrasts with paired empirical CRPS"})
    fig.savefig(stem.with_suffix(".png"), dpi=400)
    plt.close(fig)
    return [stem.with_suffix(".pdf"), stem.with_suffix(".png")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    scenarios, rows, checks = prepare()
    table = output / "full_permutation_input_table.csv"
    with table.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    files = make_plot(scenarios, rows, output)
    receipt = {
        "status": "NUMERICAL_CHECKS_PASS_VISUAL_REVIEW_SEPARATE",
        "conditions": 6, "latent_pairs": 180,
        "plotted_points": {"A_brier": 30, "B_mean_D_absolute_error": 36, "C_empirical_CRPS": 36},
        "original_A_B_fields_retained_exactly": True,
        "CRPS_paired_contrasts_recalculated": len(checks),
        "pairing": "Empirical minus original column-randomised arm, within each latent matrix and scheme",
        "error_bars": "One paired MCSE across 30 independent matrices, not 512 independent draws",
        "input_sha256": {str(p.relative_to(HERE)): sha(p) for p in (SOURCE_AB, SOURCE_C, SOURCE_C_PAIRS)},
        "output_sha256": {p.name: sha(p) for p in files + [table]},
        "matplotlib_version": matplotlib.__version__,
    }
    (output / "DIAGNOSTIC_CHECKS.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
