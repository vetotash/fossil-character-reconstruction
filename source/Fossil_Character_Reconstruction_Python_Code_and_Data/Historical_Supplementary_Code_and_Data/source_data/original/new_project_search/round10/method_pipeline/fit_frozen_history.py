"""Freeze then fit the unchanged five-method model family to a new history.

Preparation and fitting are separate operations. No work runs on import.
The original predictor and Mk functions are called without changing parameters;
temporary factories capture estimator diagnostics without changing estimators.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import warnings

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.optimize import linear_sum_assignment
from sklearn.exceptions import ConvergenceWarning

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PROJECT = ROOT / "morphology_workspace/morphology_project"
sys.path.insert(0, str(PROJECT / "round2"))
import run_comparison as rc
import mk_baseline as mk

dp = rc.dp
MODELS = ["mode", "knn5", "logistic", "extra_trees", "nj_mk"]
SEED = 20261002
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def source_dependencies():
    return [Path(__file__), PROJECT / "round2/run_comparison.py",
            PROJECT / "round2/mk_baseline.py", PROJECT / "dinosaur_matched_pilot.py",
            PROJECT / "pilot.py"]


def record_file(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def prepare(source_npz, future_csv, candidate_freeze, output, lineage):
    """Freeze controls with the original fish rule; this does not fit a model."""
    output = Path(output).resolve()
    assert not output.exists(), "Refuse to replace an existing prepared analysis"
    source_npz, future_csv, candidate_freeze = map(Path, (source_npz, future_csv, candidate_freeze))
    assert candidate_freeze.is_file(), "A candidate scope freeze is required first"
    z = np.load(source_npz, allow_pickle=False)
    a, names, chars = z["A"], z["old_names"].tolist(), z["old_char_numbers"].tolist()
    assert a.shape == (len(names), len(chars))
    assert len(names) == len(set(names)) and len(chars) == len(set(chars))
    assert list(map(int, chars)) == sorted(map(int, chars)), "Preserve original ascending character order"
    lookup = {name: i for i, name in enumerate(names)}
    cmap = {int(c): j for j, c in enumerate(chars)}
    future = pd.read_csv(future_csv, keep_default_na=False, dtype={"true": str})
    assert {"taxon", "character", "true"}.issubset(future.columns)
    assert not future.duplicated(["taxon", "character"]).any()
    future = future[["taxon", "character", "true"]].sort_values(["character", "taxon"]).reset_index(drop=True)
    missing = (~mk.known(a)).mean(axis=1)
    rng = np.random.default_rng(SEED)
    targets, controls, matches, counts = [], [], [], []
    for character, group in future.groupby("character", sort=True):
        character = int(character)
        j = cmap[character]
        tr = np.flatnonzero(mk.known(a[:, j]))
        assert len(tr) >= 15, (character, len(tr), "Candidate not source-admissible")
        te = np.array([lookup[n] for n in group.taxon])
        assert np.all(a[te, j] == "?"), (character, "Future targets must be original literal ?")
        assert mk.known(group.true.to_numpy()).all()
        for r in group.itertuples():
            targets.append({"lineage": lineage, "scenario": "later_publication",
                            "character": character, "taxon": r.taxon, "true": r.true})
        ncontrol = min(len(te), max(1, len(tr) // 4))
        chosen = rng.choice(te, ncontrol, replace=False)
        rows, columns = linear_sum_assignment(abs(missing[chosen, None] - missing[tr][None, :]))
        for ti, ci in zip(chosen[rows], tr[columns]):
            controls.append({"lineage": lineage, "scenario": "matched_concealment",
                             "character": character, "taxon": names[ci], "true": a[ci, j]})
            matches.append({"character": character, "later_taxon": names[ti], "mask_taxon": names[ci],
                            "later_missing": float(missing[ti]), "mask_missing": float(missing[ci]),
                            "abs_difference": float(abs(missing[ti] - missing[ci]))})
        counts.append({"character": character, "n_original_train": len(tr), "n_future": len(te),
                       "n_controls": ncontrol, "n_shared_train": len(tr) - ncontrol})
    output.mkdir(parents=True)
    shutil.copyfile(source_npz, output / "source_training.npz")
    pd.DataFrame(targets + controls).to_csv(output / "frozen_targets.csv", index=False)
    pd.DataFrame(matches).to_csv(output / "control_matching.csv", index=False)
    pd.DataFrame(counts).to_csv(output / "training_counts.csv", index=False)
    protocol = HERE / "NEW_FIT_PROTOCOL.txt"
    assert protocol.is_file()
    freeze = {"created_utc": timestamp(), "lineage": lineage, "seed": SEED,
              "models": MODELS, "control_rule": "original_ichthyosaur_quarter_cap_Hungarian",
              "source_missingness_scope": "all admitted predictor characters in source_training.npz",
              "n_source_taxa": len(names), "n_predictor_characters": len(chars),
              "n_future": len(targets), "n_controls": len(controls), "n_target_characters": len(counts),
              "fitting_started": False, "candidate_freeze": record_file(candidate_freeze),
              "candidate_inputs": [record_file(source_npz), record_file(future_csv)],
              "versions": {"numpy": np.__version__, "pandas": pd.__version__,
                           "scipy": scipy.__version__, "sklearn": sklearn.__version__},
              "files": [record_file(p) for p in source_dependencies() + [protocol,
                        output / "source_training.npz", output / "frozen_targets.csv",
                        output / "control_matching.csv", output / "training_counts.csv"]]}
    (output / "FIT_FREEZE.json").write_text(json.dumps(freeze, indent=2))
    print(json.dumps({k: v for k, v in freeze.items() if k not in {"files", "candidate_inputs"}}, indent=2))


def instrumented_predict(matrix, j, train, test):
    """Call the original classifier function; record, do not alter, fitting."""
    records = []
    originals = {"extra_trees": dp.ExtraTreesClassifier, "logistic": dp.LogisticRegression}

    def factory(name):
        def construct(*args, **kwargs):
            model = originals[name](*args, **kwargs)
            bound_fit = model.fit

            def captured_fit(*fit_args, **fit_kwargs):
                start = time.monotonic()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    result = bound_fit(*fit_args, **fit_kwargs)
                n_iter = getattr(model, "n_iter_", None)
                convergence = any(issubclass(w.category, ConvergenceWarning) for w in caught)
                records.append({"model": name, "seconds": time.monotonic() - start,
                    "n_iter": None if n_iter is None else np.asarray(n_iter).tolist(),
                    "reached_max_iter": None if n_iter is None else bool(np.any(np.asarray(n_iter) >= model.max_iter)),
                    "convergence_warning": convergence,
                    "warnings": [{"category": w.category.__name__, "message": str(w.message)} for w in caught],
                    "parameters": model.get_params(), "classes": model.classes_.tolist(),
                    "n_features_in": int(model.n_features_in_), "fit_ran": True})
                return result

            model.fit = captured_fit
            return model
        return construct

    dp.ExtraTreesClassifier, dp.LogisticRegression = factory("extra_trees"), factory("logistic")
    try:
        methods = dp.predict(matrix, j, train, test)
    finally:
        dp.ExtraTreesClassifier = originals["extra_trees"]
        dp.LogisticRegression = originals["logistic"]
    if len(np.unique(matrix[train, j])) == 1:
        for name in originals:
            records.append({"model": name, "fit_ran": False, "constant_class_fallback": True,
                            "n_iter": None, "convergence_warning": None, "warnings": []})
    return methods, records


def instrumented_mk(edges, ntaxa, observed, train, test):
    """Record the original optimizer result without changing its call or return."""
    optimizer_record = {}
    original = mk.minimize_scalar

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        optimizer_record.update({"success": bool(result.success), "message": str(result.message),
                                 "nit": int(result.nit), "nfev": int(result.nfev),
                                 "refined_logq": float(result.x), "refined_negative_loglik": float(result.fun),
                                 "bounds": list(kwargs["bounds"]), "method": kwargs["method"],
                                 "options": kwargs["options"]})
        return result

    mk.minimize_scalar = capture
    start = time.monotonic()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            labels, probability, fit = mk.predict_mk(edges, ntaxa, observed, train, test)
        finally:
            mk.minimize_scalar = original
    diagnostic = {"seconds": time.monotonic() - start, "optimizer": optimizer_record,
                  "warnings": [{"category": w.category.__name__, "message": str(w.message)} for w in caught]}
    return labels, probability, fit, diagnostic


def fit(output):
    output = Path(output).resolve()
    assert not (output / "FIT_STARTED.json").exists(), "Do not overwrite an existing fit attempt"
    freeze = json.loads((output / "FIT_FREEZE.json").read_text())
    for record in freeze["files"] + freeze["candidate_inputs"] + [freeze["candidate_freeze"]]:
        assert sha(record["path"]) == record["sha256"], record["path"]
    expected = {"numpy": np.__version__, "pandas": pd.__version__,
                "scipy": scipy.__version__, "sklearn": sklearn.__version__}
    assert expected == freeze["versions"]
    assert expected == {"numpy": "2.3.5", "pandas": "2.2.3", "scipy": "1.17.0", "sklearn": "1.8.0"}
    (output / "FIT_STARTED.json").write_text(json.dumps({"started_utc": timestamp(),
        "freeze_sha256": sha(output / "FIT_FREEZE.json"), "models": MODELS}, indent=2))
    z = np.load(output / "source_training.npz", allow_pickle=False)
    a, names = z["A"], z["old_names"].tolist()
    cmap = {int(c): j for j, c in enumerate(z["old_char_numbers"])}
    lookup = {name: i for i, name in enumerate(names)}
    frozen = pd.read_csv(output / "frozen_targets.csv", keep_default_na=False, dtype={"true": str})
    totals = mk.distance_totals(a)
    predictions, trees, fits, diagnostics = [], [], [], []
    started = time.monotonic()
    for number, (character, group) in enumerate(frozen.groupby("character", sort=True)):
        character, j = int(character), cmap[int(character)]
        controls = group[group.scenario.eq("matched_concealment")]
        future = group[group.scenario.eq("later_publication")]
        ci = np.array([lookup[n] for n in controls.taxon])
        fi = np.array([lookup[n] for n in future.taxon])
        assert len(ci) and len(fi) and not set(ci) & set(fi)
        assert np.array_equal(a[ci, j], controls.true.to_numpy()) and np.all(a[fi, j] == "?")
        matrix = a.copy(); matrix[ci, j] = "?"
        train = np.flatnonzero(mk.known(matrix[:, j])); test = np.r_[ci, fi]
        assert not set(train) & set(test)
        # Truth is joined only to output scores; no future values enter predictors.
        truth = np.r_[controls.true.to_numpy(), future.true.to_numpy()]
        distance, zero = mk.excluding_distance(a, j, totals)
        edges, nj_diag = mk.neighbor_join(distance)
        trees.append({"character": character, "taxa": names, "edges": edges})
        methods, classifier_diag = instrumented_predict(matrix, j, train, test)
        labels, probs, mk_fit, optimizer_diag = instrumented_mk(edges, len(names), matrix[:, j], train, test)
        methods["nj_mk"] = (labels, probs)
        diagnostics.append({"character": character, "classifiers": classifier_diag, "mk": optimizer_diag})
        fits.append({"lineage": freeze["lineage"], "character": character,
                     "n_train": len(train), "n_original_train": int(mk.known(a[:, j]).sum()),
                     "n_controls": len(ci), "n_future": len(fi), "zero_overlap_pairs": zero,
                     **nj_diag, **mk_fit})
        for model in MODELS:
            classes, probability = methods[model]
            assert np.isfinite(probability).all() and np.all(probability >= 0)
            assert np.allclose(probability.sum(axis=1), 1)
            hard = classes[probability.argmax(axis=1)]
            for k, tip in enumerate(test):
                p = probability[k]; yy = truth[k]
                ptrue = float(p[np.flatnonzero(classes == yy)[0]]) if yy in classes else 0.0
                predictions.append({"lineage": freeze["lineage"],
                    "scenario": "control_old_score" if k < len(ci) else "later_target",
                    "character": character, "taxon": names[tip], "model": model, "true": yy,
                    "pred": hard[k], "n_train": len(train), "confidence": float(p.max()),
                    "p_true": ptrue, "brier": float(np.sum(p*p)-2*ptrue+1),
                    "log_loss": -float(np.log(max(ptrue, 1e-12))),
                    "state_probabilities": json.dumps(dict(zip(classes.tolist(), p.tolist())))})
        # Durable partial evidence is retained if a later character fails.
        (output / "fit_progress.json").write_text(json.dumps({"completed_characters": number + 1,
            "last_character": character, "seconds": time.monotonic() - started}, indent=2))
        (output / "fit_execution_diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
        if number % 40 == 0:
            print(json.dumps({"completed_characters": number + 1, "seconds": round(time.monotonic()-started, 2)}), flush=True)
    result = pd.DataFrame(predictions)
    assert len(result) == 5 * (freeze["n_future"] + freeze["n_controls"])
    assert not result.duplicated(["scenario", "taxon", "character", "model"]).any()
    result.to_csv(output / "predictions.csv", index=False)
    pd.DataFrame(fits).to_csv(output / "fit_diagnostics.csv", index=False)
    (output / "estimated_trees.json").write_text(json.dumps(trees))
    summary = {"completed_utc": timestamp(), "seconds": time.monotonic()-started,
        "characters": len(fits), "prediction_rows": len(result),
        "same_original_model_functions": True, "new_hyperparameter_tuning": False,
        "future_labels_used_for_fitting": False,
        "logistic_convergence_warning_characters": [r["character"] for r in diagnostics
             if any(d.get("convergence_warning") for d in r["classifiers"] if d["model"] == "logistic")],
        "logistic_max_iter_characters": [r["character"] for r in diagnostics
             if any(d.get("reached_max_iter") for d in r["classifiers"] if d["model"] == "logistic")],
        "mk_rate_boundary_characters": [r["character"] for r in fits if r["rate_boundary"]],
        "mk_optimizer_unsuccessful_characters": [r["character"] for r in diagnostics
             if r["mk"]["optimizer"] and not r["mk"]["optimizer"]["success"]]}
    (output / "FIT_COMPLETED.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="operation", required=True)
    prep = sub.add_parser("prepare")
    for flag in ["source-npz", "future-csv", "candidate-freeze", "output", "lineage"]:
        prep.add_argument("--" + flag, required=True)
    run = sub.add_parser("fit"); run.add_argument("--output", required=True)
    args = vars(parser.parse_args()); operation = args.pop("operation")
    if operation == "prepare":
        prepare(**args)
    else:
        fit(**args)
