#!/usr/bin/env python3
"""Refit frozen historical targets with authenticated archived estimators.

This new orchestrator relocates input paths; it never edits frozen sources or
claims to recover original logs. Outputs must be new and outside the archive.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
import platform
import sys
import time
import traceback

sys.dont_write_bytecode = True
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np
import pandas as pd
import scipy
import sklearn

ROOT = Path(__file__).resolve().parent
OLD = ROOT / "full_source/old_refits"
FISH = ROOT / "full_source/ichthyosaur_refit"
SOURCE = ROOT / "source_data/original"
MODELS = ["mode", "knn5", "logistic", "extra_trees", "nj_mk"]
VERSIONS = {"numpy": np.__version__, "pandas": pd.__version__,
            "scipy": scipy.__version__, "sklearn": sklearn.__version__}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    def native(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        raise TypeError(type(x).__name__)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False,
                                   default=native) + "\n")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def locate_fish_record(path):
    """Map only the two documented archival prefixes, never guess by basename."""
    if "/round14/ichthyosaur_refit/" in path:
        return FISH / path.split("/round14/ichthyosaur_refit/", 1)[1]
    if "/round12/restored/analysis/" in path:
        return SOURCE / path.split("/round12/restored/analysis/", 1)[1]
    raise ValueError("Unrecognised frozen path: " + path)


def authenticate(jobs):
    manifest = json.loads((ROOT / "MANIFEST.json").read_text())
    catalog = {r["path"]: r for r in manifest["files"]}
    verified, missing_documents = {}, []

    def check(path, expected, origin):
        path = Path(path).resolve()
        require(path.is_relative_to(ROOT), "Input escaped archive: " + str(path))
        require(path.is_file(), "Required fitting input absent: " + str(path))
        actual = sha(path)
        require(actual == expected, "Frozen SHA-256 mismatch: " + str(path))
        verified[str(path.relative_to(ROOT))] = {
            "sha256": actual, "bytes": path.stat().st_size,
            "expected_hash_source": origin}

    def package_check(path):
        rel = str(path.relative_to(ROOT))
        require(rel in catalog, "Input absent from package manifest: " + rel)
        check(path, catalog[rel]["sha256"], "MANIFEST.json")

    old_freeze, fish_freeze = None, None
    if any(h != "ichthyosaurs" for h, _ in jobs):
        package_check(OLD / "FIT_FREEZE.json")
        old_freeze = json.loads((OLD / "FIT_FREEZE.json").read_text())
        require(VERSIONS == old_freeze["versions"], "Library versions differ from old-refit freeze")
        check(OLD / "run_refits.py", old_freeze["runner_sha256"],
              "full_source/old_refits/FIT_FREEZE.json:runner_sha256")
        for r in old_freeze["files"]:
            check(OLD / r["local"], r["sha256"], "full_source/old_refits/FIT_FREEZE.json")

    if any(h == "ichthyosaurs" for h, _ in jobs):
        package_check(FISH / "FIT_FREEZE.json")
        fish_freeze = json.loads((FISH / "FIT_FREEZE.json").read_text())
        require(VERSIONS == fish_freeze["versions"], "Library versions differ from ichthyosaur freeze")
        for r in fish_freeze["files"] + fish_freeze["original_inputs"]:
            path = locate_fish_record(r["path"])
            if path == FISH / "PROTOCOL.txt" and not path.exists():
                missing_documents.append({"path": str(path.relative_to(ROOT)),
                    "expected_sha256": r["sha256"], "status": "NOT_CHECKED_NOT_SUPPLIED",
                    "scope": "Documentary protocol, not executable or matrix input; original whole-run gate is not claimed to pass."})
                continue
            check(path, r["sha256"], "full_source/ichthyosaur_refit/FIT_FREEZE.json")
        package_check(FISH / "predictions.csv")
    return {"verified_files": verified, "unavailable_documentary_inputs": missing_documents,
            "input_hash_status": "PASS", "original_whole_run_gate_replayed": False,
            "scope": "New orchestrator; all required fitting input/code hashes checked, no original check disabled.",
            "library_versions": VERSIONS, "python": platform.python_version(),
            "platform": platform.platform(), "wrapper_sha256": sha(__file__)}, old_freeze, fish_freeze


def read_csv(path):
    return pd.read_csv(path, keep_default_na=False, dtype={"true": str, "pred": str})


def compare(result, archived):
    keys = ["lineage", "scenario", "character", "taxon", "model"]
    require(not result.duplicated(keys).any(), "Duplicate forecast key")
    paired = result.merge(archived, on=keys, suffixes=("_new", "_archived"),
                          how="left", validate="one_to_one", indicator=True)
    require(paired["_merge"].eq("both").all(), "Forecast key missing from archived predictions")
    for field in ["true", "pred", "n_train"]:
        require(paired[field + "_new"].eq(paired[field + "_archived"]).all(),
                "Archived " + field + " mismatch")
    differences = []
    for row in paired.itertuples():
        p = json.loads(row.state_probabilities_new)
        q = json.loads(row.state_probabilities_archived)
        require(set(p) == set(q), "Archived state alphabet differs")
        delta = max(abs(p[s] - q[s]) for s in p)
        require(delta <= 1e-10, "Archived probability mismatch exceeds 1e-10")
        differences.append({k: getattr(row, k) for k in keys} |
                           {"max_probability_abs_difference": delta})
    return pd.DataFrame(differences)


def fit_job(history, condition, mode, character, output, old_freeze, fish_freeze):
    started = time.monotonic()
    output.mkdir()
    fish = history == "ichthyosaurs"
    # Remove cached import names so each family resolves its own authenticated files.
    for name in ["run_comparison", "mk_baseline", "dinosaur_matched_pilot"]:
        sys.modules.pop(name, None)
    if fish:
        module = load_module("portable_fish_estimators",
                             SOURCE / "new_project_search/round10/method_pipeline/fit_frozen_history.py")
        mk = module.mk
        folder, lineage = FISH, fish_freeze["lineage"]
        future = read_csv(FISH / "prediction_keys.csv")
        require(list(future.columns) == ["lineage", "character", "taxon"], "Unexpected fish key columns")
        controls = None
        treefile = SOURCE / "new_project_search/round10/ichthyosaur_shared_fit/estimated_trees.json"
    else:
        module = load_module("portable_old_estimators", OLD / "run_refits.py")
        dp, mk = module.load_modules()
        folder, lineage = OLD / "inputs" / history, history
        frozen = read_csv(folder / "frozen_targets.csv")
        selected = frozen[frozen.model.eq("mode")]
        future = selected[selected.scenario.eq("later_publication")][["character", "taxon"]].copy()
        controls = selected[selected.scenario.eq("matched_concealment")].copy()
        treefile = folder / "estimated_trees.json"
    with np.load(folder / "source_training.npz", allow_pickle=False) as z:
        matrix = z["A"].copy()
        names, chars = z["old_names"].tolist(), z["old_char_numbers"].tolist()
    lookup, cmap = {n: i for i, n in enumerate(names)}, {int(c): j for j, c in enumerate(chars)}
    trees = {int(t["character"]): t for t in json.loads(treefile.read_text())}
    eligible = sorted(map(int, future.character.unique()))
    if mode == "smoke":
        chosen = eligible[0] if character is None else character
        require(chosen in eligible, "Requested smoke character is not an archived future target")
        eligible = [chosen]
    future = future[future.character.isin(eligible)]
    totals = mk.distance_totals(matrix) if fish else None
    rows, diagnostics = [], []
    dump(output / "FIT_STARTED.json", {"started_utc": now(), "history": history,
         "condition": condition, "mode": mode, "selected_characters": eligible,
         "new_execution_log": True, "original_historical_log": False})
    for char, group in future.groupby("character", sort=True):
        char, j = int(char), cmap[int(char)]
        fi = np.array([lookup[t] for t in group.taxon], dtype=int)
        current = matrix.copy()
        require(np.all(current[fi, j] == "?"), "Future cell is not an original literal question mark")
        if not fish:
            cg = controls[controls.character.eq(char)]
            ci = np.array([lookup[t] for t in cg.taxon], dtype=int)
            require(len(ci) > 0 and not set(ci) & set(fi), "Invalid frozen control selection")
            require(np.array_equal(matrix[ci, j], cg.true.to_numpy()), "Control labels differ from old matrix")
            if condition == "shared":
                current[ci, j] = "?"
                test = np.r_[ci, fi]
                scenarios = ["control_old_score"] * len(ci) + ["later_target"] * len(fi)
            else:
                test, scenarios = fi, ["later_target"] * len(fi)
            edges = trees[char]["edges"]
            require(trees[char]["taxa"] == names, "Tree taxon order differs")
        else:
            test, scenarios = fi, ["later_target"] * len(fi)
            distance, zero = mk.excluding_distance(matrix, j, totals)
            edges, njdiag = mk.neighbor_join(distance)
            require(np.array_equal(np.asarray(edges), np.asarray(trees[char]["edges"])),
                    "Rebuilt fish NJ tree differs from archived response-excluded tree")
        train = np.flatnonzero(mk.known(current[:, j]))
        require(not set(train) & set(test), "Training/target overlap")
        if fish:
            methods, clfdiag = module.instrumented_predict(current, j, train, test)
            labels, probability, mkfit, mkdiag = module.instrumented_mk(edges, len(names), current[:, j], train, test)
        else:
            methods, clfdiag = module.instrumented_predict(dp, current, j, train, test)
            labels, probability, mkfit, mkdiag = module.instrumented_mk(mk, edges, len(names), current[:, j], train, test)
        methods["nj_mk"] = (labels, probability)
        diagnostics.append({"character": char, "n_train": len(train), "n_test": len(test),
                            "classifiers": clfdiag, "mk_fit": mkfit, "mk_execution": mkdiag})
        for model in MODELS:
            labels, probability = methods[model]
            require(np.isfinite(probability).all() and (probability >= 0).all()
                    and np.allclose(probability.sum(axis=1), 1), "Invalid forecast probabilities")
            hard = labels[probability.argmax(axis=1)]
            for k, tip in enumerate(test):
                rows.append({"lineage": lineage, "scenario": scenarios[k], "character": char,
                    "taxon": names[tip], "model": model, "pred": str(hard[k]), "n_train": len(train),
                    "confidence": float(probability[k].max()),
                    "state_probabilities": json.dumps(dict(zip(labels.tolist(), probability[k].tolist())))})
        dump(output / "fit_execution_diagnostics.json", diagnostics)
        dump(output / "fit_progress.json", {"completed_characters": len(diagnostics), "last_character": char})
        print(json.dumps({"history": history, "condition": condition, "character": char,
                          "completed_characters": len(diagnostics)}), flush=True)
    result = pd.DataFrame(rows)
    forecast_path = output / "predictions_before_evaluation.csv"
    result.to_csv(forecast_path, index=False)
    forecast_hash = sha(forecast_path)
    if fish:
        truth = read_csv(FISH / "evaluation_labels.csv")
        result = result.merge(truth, on=["character", "taxon"], validate="many_to_one", how="left")
        archived = read_csv(FISH / "predictions.csv")
    else:
        key = ["character", "taxon", "scenario"]
        truth = selected[key + ["true"]].copy()
        truth["scenario"] = truth.scenario.replace({"later_publication": "later_target", "matched_concealment": "control_old_score"})
        result = result.merge(truth, on=key, validate="many_to_one", how="left")
        archived = read_csv(folder / ("archived_shared_predictions.csv" if condition == "shared" else "archived_full_predictions.csv"))
        if condition == "full_source":
            archived = archived[archived.scenario.eq("later_publication")].copy()
            archived["scenario"] = "later_target"
    require(not result["true"].isna().any(), "Missing evaluation label")
    probs = result.state_probabilities.map(json.loads)
    result["p_true"] = [p.get(t, 0.) for p, t in zip(probs, result["true"])]
    result["brier"] = [sum(v*v for v in p.values()) - 2*pt + 1 for p, pt in zip(probs, result.p_true)]
    result["log_loss"] = -np.log(np.maximum(result.p_true.to_numpy(), 1e-12))
    comparison = compare(result, archived)
    result.to_csv(output / "predictions.csv", index=False)
    comparison.to_csv(output / "archive_probability_comparison.csv", index=False)
    require(sha(forecast_path) == forecast_hash, "Forecast-only artifact changed during evaluation")
    summary = {"status": "PASS", "completed_utc": now(), "history": history, "condition": condition,
        "mode": mode, "response_characters": len(diagnostics), "method_character_forecasts": len(diagnostics)*5,
        "prediction_rows": len(result), "max_probability_abs_difference": float(comparison.max_probability_abs_difference.max()),
        "probability_tolerance": 1e-10, "forecast_sha256_before_evaluation": forecast_hash,
        "seconds": time.monotonic()-started, "native_calls": 0,
        "new_execution_log": True, "original_historical_log": False}
    dump(output / "FIT_COMPLETED.json", summary)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("operation", choices=["verify", "smoke", "fit"])
    p.add_argument("--history", choices=["tetrapods", "dinosaurs", "ichthyosaurs", "all"], default="tetrapods")
    p.add_argument("--condition", choices=["shared", "full_source", "all"], default=None)
    p.add_argument("--character", type=int, help="One archived response character, smoke only; default first")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    require(args.character is None or args.operation == "smoke", "--character is smoke-only")
    require(args.operation != "smoke" or args.history != "all", "Smoke must select one history")
    histories = ["tetrapods", "dinosaurs", "ichthyosaurs"] if args.history == "all" else [args.history]
    jobs = []
    for h in histories:
        condition = args.condition or ("full_source" if h == "ichthyosaurs" else "shared")
        conditions = (["full_source"] if h == "ichthyosaurs" else ["shared", "full_source"]) if condition == "all" else [condition]
        require(h != "ichthyosaurs" or conditions == ["full_source"], "Ichthyosaur refit supports full_source only")
        jobs.extend((h, c) for c in conditions)
    require(args.operation != "smoke" or len(jobs) == 1, "Smoke selects one history-condition case")
    output = args.output.resolve()
    require(output != ROOT and not output.is_relative_to(ROOT), "Output must be outside the delivered archive")
    require(not output.exists(), "Output directory already exists; preserve previous attempts")
    output.mkdir(parents=True)
    with (output / "execution.log").open("w") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        dump(output / "RUN_STARTED.json", {"started_utc": now(), "command": sys.argv,
             "operation": args.operation, "jobs": jobs, "wrapper_sha256": sha(__file__),
             "scope": "New execution; not a recovered historical log"})
        try:
            receipt, old_freeze, fish_freeze = authenticate(jobs)
            dump(output / "INPUT_VERIFICATION.json", receipt)
            results = []
            if args.operation != "verify":
                for history, condition in jobs:
                    results.append(fit_job(history, condition, args.operation, args.character,
                                   output / (history + "_" + condition), old_freeze, fish_freeze))
            # Recheck every authenticated file after execution to detect input mutation.
            for rel, entry in receipt["verified_files"].items():
                require(sha(ROOT / rel) == entry["sha256"], "Input changed during execution: " + rel)
            dump(output / "RUN_COMPLETED.json", {"status": "PASS", "completed_utc": now(),
                 "operation": args.operation, "new_execution_log": True, "native_calls": 0,
                 "original_inputs_unchanged": True, "results": results,
                 "unavailable_documentary_inputs": receipt["unavailable_documentary_inputs"]})
        except Exception as exc:
            traceback.print_exc()
            dump(output / "RUN_FAILED.json", {"status": "FAIL", "failed_utc": now(),
                 "error": str(exc), "new_execution_log": True})
            raise
    print("PASS: " + str(output / "RUN_COMPLETED.json"))


if __name__ == "__main__":
    main()
