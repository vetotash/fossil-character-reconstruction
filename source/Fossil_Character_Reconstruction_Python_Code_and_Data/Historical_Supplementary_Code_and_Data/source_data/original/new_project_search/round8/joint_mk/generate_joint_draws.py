"""Conditional state simulation from already frozen NJ-Mk trees and rates.

No model is fitted here. Later truth columns are deliberately not loaded.
"""
from pathlib import Path
import hashlib
import json
import platform
import sys

import numpy as np
import pandas as pd
import scipy

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
PROJECT = ROOT / "morphology_workspace/morphology_project"
sys.path.insert(0, str(PROJECT / "round2"))
from mk_baseline import known, orient, pruning, transition


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def upward(tree, emission, rate):
    root, order, children, lengths = tree
    up = np.ones_like(emission)
    msg = np.ones_like(emission)
    k = emission.shape[1]
    for node in reversed(order):
        value = emission[node].copy()
        for child in children[node]:
            value *= msg[child]
        up[node] = value / value.max()
        if node != root:
            msg[node] = transition(rate, lengths[node], k) @ up[node]
    return up


def joint_sample(tree, emission, rate, n, rng):
    root, order, children, lengths = tree
    k = emission.shape[1]
    up = upward(tree, emission, rate)
    states = np.empty((n, len(order)), dtype=np.int16)
    probability = up[root] / up[root].sum()  # uniform stationary root prior
    states[:, root] = rng.choice(k, size=n, p=probability)
    for parent in order:
        for child in children[parent]:
            conditional = transition(rate, lengths[child], k) * up[child][None, :]
            conditional /= conditional.sum(axis=1, keepdims=True)
            cdf = np.cumsum(conditional[states[:, parent]], axis=1)
            cdf[:, -1] = 1.0
            states[:, child] = (rng.random(n)[:, None] > cdf).sum(axis=1)
    return states


def main():
    freeze = json.loads((OUT / "FREEZE.json").read_text())
    for item in freeze["inputs"]:
        assert sha(ROOT / item["path"]) == item["sha256"], item["path"]
    n_draws, seed = freeze["n_draws"], freeze["seed"]
    snap = PROJECT / "round2/inputs/tetrapods.npz"
    snapshot_manifest = json.loads((snap.parent / "snapshot_manifest.json").read_text())
    record = next(row for row in snapshot_manifest if row["lineage"] == "tetrapods")
    assert sha(snap) == record["matrix_sha256"]
    z = np.load(snap)
    names, old = z["old_names"].tolist(), z["A"]
    name_index = {name: i for i, name in enumerate(names)}
    char_index = {int(c): j for j, c in enumerate(z["old_char_numbers"])}
    archive = pd.read_csv(
        PROJECT / "round3/shared_training/tetrapods/predictions.csv",
        usecols=["scenario", "character", "taxon", "model", "pred", "state_probabilities"],
        dtype={"pred": str},
    )
    archive = archive.loc[archive.model.eq("nj_mk")].copy()
    archive = archive.sort_values(["character", "taxon", "scenario"]).reset_index(drop=True)
    assert len(archive) == 700
    assert not archive.duplicated(["character", "taxon"]).any()
    fits = pd.read_csv(PROJECT / "round3/shared_training/tetrapods/fit_diagnostics.csv").set_index("character")
    trees = {
        int(row["character"]): row
        for row in json.loads((PROJECT / "round2/tetrapods/estimated_trees.json").read_text())
    }
    configurations, diagnostics, marginal_rows = [], [], []
    max_error = 0.0
    # All 700 stored probabilities must reproduce before any draw is generated.
    for character, targets in archive.groupby("character", sort=True):
        j = char_index[int(character)]
        control = targets.loc[targets.scenario.eq("control_old_score")]
        future = targets.loc[targets.scenario.eq("later_target")]
        ci = np.array([name_index[x] for x in control.taxon])
        fi = np.array([name_index[x] for x in future.taxon])
        assert len(ci) == len(fi) and np.all(old[fi, j] == "?")
        assert known(old[ci, j]).all()
        observed = old[:, j].copy()
        observed[ci] = "?"
        train = np.flatnonzero(known(observed))
        assert len(train) == fits.loc[character, "n_train"]
        assert len(ci) == fits.loc[character, "n_controls"]
        assert len(fi) == fits.loc[character, "n_future"]
        row = trees[int(character)]
        assert row["taxa"] == names
        tree = orient(row["edges"])
        classes = np.unique(observed[train])
        k = len(classes)
        emission = np.ones((len(tree[1]), k))
        for i in train:
            emission[i] = (classes == observed[i]).astype(float)
        rate = float(fits.loc[character, "rate"])
        if k == 1:
            assert bool(fits.loc[character, "constant_fallback"])
            posterior = np.ones_like(emission)
            likelihood = 0.0
        else:
            assert np.isfinite(rate)
            likelihood, posterior = pruning(tree, emission, rate, True)
        assert abs(likelihood - fits.loc[character, "training_loglik"]) < 1e-9
        indices = np.array([name_index[x] for x in targets.taxon])
        character_max_error = 0.0
        for position, target in targets.iterrows():
            tip = name_index[target.taxon]
            probabilities = posterior[tip]
            stored = json.loads(target.state_probabilities)
            assert set(stored) == set(classes)
            error = max(abs(stored[s] - p) for s, p in zip(classes, probabilities))
            character_max_error = max(error, character_max_error)
            assert error < 1e-10, (character, target.taxon, error)
            assert classes[probabilities.argmax()] == target.pred
            for state, probability in zip(classes, probabilities):
                marginal_rows.append({"global_cell_index": position, "scenario": target.scenario,
                                      "character": int(character), "taxon": target.taxon,
                                      "state": state, "probability": float(probability)})
        max_error = max(max_error, character_max_error)
        diagnostics.append({"character": int(character), "n_train": len(train),
                            "n_controls": len(ci), "n_future": len(fi), "n_states": k,
                            "rate": rate if np.isfinite(rate) else None,
                            "training_loglik": likelihood,
                            "maximum_marginal_discrepancy": character_max_error})
        configurations.append((character, targets.index.to_numpy(), indices, tree, emission, rate, classes, train, observed))
    pd.DataFrame(diagnostics).to_csv(OUT / "model_reconstruction.csv", index=False)
    marginals = pd.DataFrame(marginal_rows)
    marginals.to_csv(OUT / "reproduced_marginals.csv", index=False)
    verification = {"all_marginals_reproduced": True, "cells": len(archive),
                    "characters": len(configurations), "maximum_probability_error": max_error,
                    "reestimated_trees": 0, "refitted_rates": 0, "later_truth_columns_loaded": False,
                    "n_draws": n_draws, "seed": seed}
    (OUT / "RECONSTRUCTION_CHECK.json").write_text(json.dumps(verification, indent=2))
    rng = np.random.Generator(np.random.PCG64(seed))
    draws = np.empty((n_draws, len(archive)), dtype="<U1")
    for character, positions, tips, tree, emission, rate, classes, train, observed in configurations:
        if len(classes) == 1:
            states = np.zeros((n_draws, len(tree[1])), dtype=np.int16)
        else:
            states = joint_sample(tree, emission, rate, n_draws, rng)
        # Conditioning is enforced exactly in every realization.
        assert np.all(classes[states[:, train]] == observed[train][None, :])
        draws[:, positions] = classes[states[:, tips]]
    arrays = {}
    for scenario, label in [("control_old_score", "control"), ("later_target", "future")]:
        panel = archive.loc[archive.scenario.eq(scenario), ["scenario", "character", "taxon"]].copy()
        panel["global_cell_index"] = panel.index
        panel = panel.reset_index(drop=True)
        panel.insert(0, "draw_column", np.arange(len(panel)))
        assert len(panel) == 350
        panel.to_csv(OUT / f"{label}_cells.csv", index=False)
        arrays[label] = draws[:, panel.global_cell_index.to_numpy()]
        np.save(OUT / f"{label}_joint_draws.npy", arrays[label], allow_pickle=False)
    # NPZ is convenient; .npy files have deterministic bytes/hashes for each panel.
    np.savez_compressed(OUT / "joint_draws.npz", **arrays, draw_number=np.arange(n_draws))
    frequencies = []
    for row in marginals.itertuples():
        p = row.probability
        empirical = float((draws[:, row.global_cell_index] == row.state).mean())
        se = np.sqrt(p * (1 - p) / n_draws)
        frequencies.append({**row._asdict(), "empirical_probability": empirical,
                            "empirical_minus_analytic": empirical - p,
                            "binomial_monte_carlo_se": se,
                            "standardized_deviation": (empirical - p) / se if se > 0 else 0.0})
    freq = pd.DataFrame(frequencies).drop(columns="Index")
    freq.to_csv(OUT / "monte_carlo_marginal_check.csv", index=False)
    verification.update({"python": platform.python_version(), "numpy": np.__version__,
                         "pandas": pd.__version__, "scipy": scipy.__version__,
                         "rng": "numpy.random.Generator(PCG64)",
                         "all_training_tips_match_in_every_draw": True,
                         "maximum_monte_carlo_abs_marginal_discrepancy": float(freq.empirical_minus_analytic.abs().max()),
                         "maximum_monte_carlo_abs_standardized_deviation": float(freq.standardized_deviation.abs().max()),
                         "joint_control_future_by_character": True,
                         "model_uncertainty_integrated": False})
    (OUT / "DRAW_CHECK.json").write_text(json.dumps(verification, indent=2))
    manifest = [{"file": p.name, "bytes": p.stat().st_size, "sha256": sha(p)}
                for p in sorted(OUT.iterdir()) if p.is_file() and p.name != "MANIFEST.json"]
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
