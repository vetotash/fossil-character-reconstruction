#!/usr/bin/env python3
"""Plot both historical observation-pattern contrasts from saved results.

Run from the Layout code/data archive with:
  python figures/figure2/plot_figure2.py --source-root .
No models, completions, or parsimony scores are recomputed.  Means and paired
Monte Carlo standard errors are independently reconstructed from saved
matrix-level results and checked against the published summary CSVs.
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
from matplotlib.ticker import FixedLocator, FormatStrFormatter

HERE = Path(__file__).resolve().parent
SCHEMES = ("mode", "knn5", "logistic", "extra_trees", "nj_mk", "nj_mk_joint")
LABELS = ("Frequency", "kNN5", "Logistic", "Extra Trees", "Mk marginal", "Mk joint")
COLORS = ("#555555", "#0072B2", "#E69F00", "#009E73", "#CC79A7", "#882255")
METRICS = ("brier", "crps_empirical")
LAYOUTS = ("full_permutation", "fixed_target_permutation")
SOURCES = {
    "full_state_summary": "layout_experiment/simulation/results/paired_layout_contrasts.csv",
    "full_state_matrices": "layout_experiment/simulation/results/matrix_results.csv",
    "full_distribution_summary": "distribution_analysis/results/paired_layout_contrasts.csv",
    "full_distribution_matrices": "distribution_analysis/results/matrix_distribution_scores.csv",
    "full_distribution_pairs": "distribution_analysis/results/matrix_paired_layout_differences.csv",
    "fixed_summary": "fixed_target/results/paired_layout_contrasts.csv",
    "fixed_matrices": "fixed_target/results/matrix_results.csv",
}

CAPTION = """Figure 2. Historical observation patterns under two randomisation controls.
Points compare the empirical pattern with full within-column O/U/T randomisation
(filled circles) or O/U randomisation with target positions T held fixed (open
squares). Rows show all six dataset–rate conditions; colours identify the six
completion schemes. Columns report differences in target-state Brier score (A)
and CRPS of the D distribution, in parsimony steps (B). Every difference is
empirical minus the specified randomised treatment, calculated within each of
30 latent matrices; horizontal bars are one paired Monte Carlo standard error.
Negative values indicate a lower score under the empirical pattern. Both
randomisation controls reuse the same empirical results. Each arm's D score
uses its own target-only truth completion and observed background, including
when targets are fixed. Thus the difference between the two plotted contrasts
does not isolate the contribution of target relocation. Marginal and joint Mk
share identical cell probabilities, so their Brier contrasts are identical and
are displayed at both scheme positions. All 36 scheme–condition comparisons
are shown for each control. Absolute mean-D errors are reported in Supplementary Information S10 and S13.
"""


def read_csv(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(actual, expected, key):
    assert math.isclose(float(actual), float(expected), rel_tol=2e-12, abs_tol=2e-12), (key, actual, expected)


def indexed(rows, fields):
    result = {tuple(row[field] for field in fields): row for row in rows}
    assert len(result) == len(rows), ("duplicate keys", fields)
    return result


def prepare(root):
    data = {name: read_csv(root / relative) for name, relative in SOURCES.items()}
    full_summary = indexed(data["full_state_summary"], ("scenario_id", "scheme"))
    fixed_summary = indexed(data["fixed_summary"], ("scenario_id", "scheme"))
    distribution_summary = indexed(
        [r for r in data["full_distribution_summary"] if r["metric"] == "crps_empirical"],
        ("condition_id", "scheme"))
    expected = {(f"s{i:02d}", scheme) for i in range(6) for scheme in SCHEMES}
    assert set(full_summary) == set(fixed_summary) == set(distribution_summary) == expected
    full_state = indexed(data["full_state_matrices"], ("pair_id", "scheme", "arm"))
    full_distribution = indexed(data["full_distribution_matrices"], ("latent_id", "scheme", "arm"))
    fixed = indexed(data["fixed_matrices"], ("pair_id", "scheme", "arm"))
    distribution_pairs = indexed(
        [r for r in data["full_distribution_pairs"] if r["metric"] == "crps_empirical"],
        ("latent_id", "scheme"))
    assert len(full_state) == len(full_distribution) == len(fixed) == 2160
    assert len(distribution_pairs) == 1080
    assert set(full_state) == set(full_distribution)
    assert {(p, s, "permuted" if a == "fixed_target_permuted" else a)
            for p, s, a in fixed} == set(full_state)

    rows, scenarios, checks = [], [], []
    for sid in sorted({sid for sid, scheme in expected}):
        template = full_summary[(sid, "mode")]
        scenarios.append({"scenario_id": sid, "panel": template["panel"], "q": template["q"]})
        for scheme in SCHEMES:
            key = (sid, scheme)
            a, b, c = full_summary[key], fixed_summary[key], distribution_summary[key]
            for field in ("panel", "q"):
                assert a[field] == b[field] == c[field], (key, field)
            assert a["difference_direction"] == c["difference_direction"] == "empirical_minus_permuted"
            assert b["direction"] == "empirical_minus_fixed_target_permuted"
            assert int(a["n_pairs"]) == int(b["n_pairs"]) == int(c["n_paired_matrices"]) == 30
            pair_ids = sorted(p for p, s, arm in full_state
                              if s == scheme and arm == "empirical" and full_state[(p, s, arm)]["scenario_id"] == sid)
            assert pair_ids == [f"{sid}_r{i:03d}" for i in range(30)]
            for metric in METRICS:
                source = full_state if metric == "brier" else full_distribution
                differences = {layout: [] for layout in LAYOUTS}
                for pid in pair_ids:
                    empirical = float(source[(pid, scheme, "empirical")][metric])
                    close(empirical, fixed[(pid, scheme, "empirical")][metric], (key, pid, metric, "shared empirical arm"))
                    for layout, arm, values in (
                            (LAYOUTS[0], "permuted", source),
                            (LAYOUTS[1], "fixed_target_permuted", fixed)):
                        diff = empirical - float(values[(pid, scheme, arm)][metric])
                        differences[layout].append(diff)
                        if metric == "crps_empirical" and layout == LAYOUTS[0]:
                            pair = distribution_pairs[(pid, scheme)]
                            assert pair["difference_direction"] == "empirical_minus_permuted"
                            assert pair["condition_id"] == sid and pair["panel"] == a["panel"] and pair["q"] == a["q"]
                            close(diff, pair["difference"], (key, pid, "saved paired CRPS"))
                for layout in LAYOUTS:
                    ds = differences[layout]
                    if layout == LAYOUTS[1]:
                        mean, mcse = b[metric + "_difference_mean"], b[metric + "_difference_mcse"]
                        summary_path = SOURCES["fixed_summary"]
                    elif metric == "brier":
                        mean, mcse = a["brier_difference_mean"], a["brier_difference_mcse"]
                        summary_path = SOURCES["full_state_summary"]
                    else:
                        mean, mcse = c["mean_difference"], c["paired_MCSE"]
                        summary_path = SOURCES["full_distribution_summary"]
                    close(mean, statistics.mean(ds), (key, layout, metric, "mean"))
                    close(mcse, statistics.stdev(ds) / math.sqrt(30), (key, layout, metric, "MCSE"))
                    assert math.isfinite(float(mean)) and math.isfinite(float(mcse)) and float(mcse) >= 0
                    rows.append({"scenario_id": sid, "panel": a["panel"], "q": a["q"],
                                 "scheme": scheme, "randomisation": layout, "metric": metric,
                                 "difference_direction": "empirical_minus_randomised",
                                 "n_pairs": 30, "mean_difference": mean, "paired_MCSE": mcse,
                                 "source_summary": summary_path})
                    checks.append({"scenario_id": sid, "scheme": scheme, "metric": metric,
                                   "randomisation": layout, "n_pairs": len(ds)})
    lookup = indexed(rows, ("scenario_id", "scheme", "randomisation", "metric"))
    assert len(lookup) == 144
    for sid in (s["scenario_id"] for s in scenarios):
        for layout in LAYOUTS:
            for field in ("mean_difference", "paired_MCSE"):
                close(lookup[(sid, "nj_mk", layout, "brier")][field],
                      lookup[(sid, "nj_mk_joint", layout, "brier")][field],
                      (sid, layout, "shared Mk Brier", field))
    return scenarios, rows, checks


def make_plot(scenarios, rows, output):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.4,
                         "axes.labelsize": 9.4, "axes.titlesize": 10.2,
                         "xtick.labelsize": 9.4, "ytick.labelsize": 9.4,
                         "legend.fontsize": 9.4, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 8.15), sharey=True)
    fig.subplots_adjust(left=0.175, right=0.985, top=0.84, bottom=0.204, wspace=0.30)
    lookup = indexed(rows, ("scenario_id", "scheme", "randomisation", "metric"))
    titles = ("A  Target-state Brier score", "B  CRPS of the D distribution")
    ylabels = [s["panel"].capitalize() + "\nq = " + s["q"] for s in scenarios]
    plotted = []
    for col, (ax, metric) in enumerate(zip(axes, METRICS)):
        ax.axvline(0, color="#444444", linewidth=0.8, linestyle="--", zorder=1)
        for i, scenario in enumerate(scenarios):
            if i % 2 == 0:
                ax.axhspan(i - 0.49, i + 0.49, facecolor="#f0f1f2", zorder=0)
            for j, scheme in enumerate(SCHEMES):
                center = i + (j - 2.5) * 0.142
                for layout, shift, marker, fill in (
                        (LAYOUTS[0], -0.029, "o", COLORS[j]),
                        (LAYOUTS[1], 0.029, "s", "white")):
                    key = (scenario["scenario_id"], scheme, layout, metric)
                    row = lookup[key]
                    mean, mcse = float(row["mean_difference"]), float(row["paired_MCSE"])
                    ax.errorbar(mean, center + shift, xerr=mcse, fmt=marker,
                                color=COLORS[j], markerfacecolor=fill, markeredgecolor=COLORS[j],
                                markersize=3.15, markeredgewidth=0.85, capsize=0,
                                elinewidth=0.9, linewidth=0, zorder=3)
                    plotted.append(key)
        ax.set_title(titles[col], loc="left", pad=9, fontweight="bold")
        ax.set_xlabel("Paired difference" + ("" if col == 0 else " (steps)"), labelpad=7)
        ax.set_yticks(range(6), ylabels)
        ax.tick_params(axis="y", length=0, pad=7)
        ax.grid(axis="x", alpha=0.15)
        ax.set_ylim(5.5, -0.5)
        if col == 0:
            ax.set_xlim(-0.037, 0.085)
            ax.xaxis.set_major_locator(FixedLocator([-0.03, 0, 0.03, 0.06]))
            ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        else:
            ax.set_xlim(-33, 30)
            ax.xaxis.set_major_locator(FixedLocator([-30, -15, 0, 15, 30]))
            ax.tick_params(axis="y", labelleft=False)
        # Assert that all means and their MCSE bars lie inside the chosen scale.
        lo, hi = ax.get_xlim()
        for row in rows:
            if row["metric"] == metric:
                mean, se = float(row["mean_difference"]), float(row["paired_MCSE"])
                assert lo < mean - se <= mean + se < hi, (metric, row, (lo, hi))
    assert len(plotted) == len(set(plotted)) == len(rows) == 144
    fig.text(0.175, 0.972, "Historical observation patterns in paired simulations",
             fontsize=11.5, weight="bold", va="top")
    fig.text(0.175, 0.941, "30 latent matrix pairs per condition; means ± one paired MCSE", fontsize=9.4)
    control_handles = [
        Line2D([0], [0], color="#444444", marker="o", markersize=4,
               linestyle="none", label="Full O/U/T permutation"),
        Line2D([0], [0], color="#444444", marker="s", markerfacecolor="white", markersize=4,
               linestyle="none", label="O/U permutation; T fixed")]
    fig.legend(handles=control_handles, loc="upper center", bbox_to_anchor=(0.575, 0.924),
               ncol=2, frameon=False, columnspacing=1.7, handletextpad=0.45)
    method_handles = [Line2D([0], [0], color=COLORS[j], linewidth=2.2, label=LABELS[j]) for j in range(6)]
    fig.legend(handles=method_handles, loc="lower center", bbox_to_anchor=(0.57, 0.063),
               ncol=3, frameon=False, columnspacing=1.6, handlelength=1.15, handletextpad=0.5)
    fig.text(0.575, 0.126, "Empirical minus the specified randomised pattern", ha="center", fontsize=9.4)
    fig.text(0.175, 0.025, "Negative values favour the empirical pattern. Mk Brier contrasts are identical.", fontsize=9.4)
    fig.canvas.draw()
    # Detect text extending beyond the page; the PDF remains exactly 7.6 × 8.15 in.
    renderer = fig.canvas.get_renderer()
    outside = []
    for artist in fig.findobj(matplotlib.text.Text):
        if artist.get_visible() and artist.get_text():
            box = artist.get_window_extent(renderer)
            if box.width and box.height and (box.x0 < -0.5 or box.y0 < -0.5 or
                    box.x1 > fig.bbox.width + 0.5 or box.y1 > fig.bbox.height + 0.5):
                outside.append(artist.get_text())
    assert not outside, ("text outside page", outside)
    stem = output / "figure2_paired_layout_contrasts"
    fig.savefig(stem.with_suffix(".pdf"), metadata={
        "Title": "Figure 2. Paired historical observation-pattern contrasts",
        "Subject": "Full O/U/T and fixed-target O/U randomisation; Brier and CRPS"})
    fig.savefig(stem.with_suffix(".png"), dpi=400)
    plt.close(fig)
    return [stem.with_suffix(".pdf"), stem.with_suffix(".png")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=HERE.parent.parent)
    parser.add_argument("--output-dir", type=Path, default=HERE)
    args = parser.parse_args()
    root, output = args.source_root.resolve(), args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    scenarios, rows, checks = prepare(root)
    table = output / "figure2_input_table.csv"
    with table.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    files = make_plot(scenarios, rows, output)
    caption = output / "caption.txt"
    caption.write_text(CAPTION)
    receipt = {
        "status": "NUMERICAL_AND_PAGE_BOUNDS_CHECKS_PASS_VISUAL_REVIEW_PENDING",
        "conditions": 6, "latent_matrices": 180, "pairs_per_condition": 30,
        "schemes": list(SCHEMES), "randomisation_controls": list(LAYOUTS),
        "scheme_condition_comparisons_per_control": 36,
        "plotted_points": {"A_brier": 72, "B_crps_empirical": 72},
        "all_paired_means_and_MCSEs_reconstructed": len(checks),
        "shared_empirical_arm_verified": True,
        "shared_Mk_Brier_means_and_MCSEs_verified": True,
        "matrix_results_only_no_model_or_parsimony_rerun": True,
        "error_bars": "One paired MCSE across 30 independent latent matrices",
        "figure_inches": [7.6, 8.15], "minimum_font_points": 9.4,
        "minimum_font_at_6_5_inches": 9.4 * 6.5 / 7.6,
        "text_within_page_bounds": True,
        "source_sha256": {relative: sha(root / relative) for relative in SOURCES.values()},
        "output_sha256": {p.name: sha(p) for p in files + [table, caption]},
        "matplotlib_version": matplotlib.__version__,
    }
    (output / "validation.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: receipt[k] for k in ("status", "plotted_points", "all_paired_means_and_MCSEs_reconstructed", "minimum_font_at_6_5_inches")}, indent=2))


if __name__ == "__main__":
    main()
