#!/usr/bin/env python3
"""Masked-only fitting and propagation phase. This module never reads truth.

Only input_manifest.json and each entry's masked matrix are opened. Generating
rates, generating trees and hidden labels are not arguments of fit_matrix.
"""
from __future__ import annotations

import os
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_variable] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np

METHODS_PATH = Path(__file__).resolve().parents[1] / "methods"
sys.path.insert(0, str(METHODS_PATH))
from fit_adapter import METHODS, fit_matrix, sample_mk_joint

N_DRAWS = 512
PROPAGATION_SEED = 2026100405


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def fit_file(masked_path, entry, output_directory):
    """No truth directory/path/data is received or opened by this worker."""
    masked_path, output_directory = Path(masked_path), Path(output_directory)
    start = time.monotonic()
    matrix_id = entry["matrix_id"]
    if sha256(masked_path) != entry["masked_sha256"]:
        raise RuntimeError("Masked input hash mismatch")
    with np.load(masked_path) as saved:
        masked = saved["X"]
    string_matrix = np.where(masked == -1, "?", masked.astype(str))
    fitted = fit_matrix(string_matrix)
    metadata = {"matrix_id": matrix_id, "masked_sha256": entry["masked_sha256"],
                "methods": list(METHODS), "diagnostics": fitted["diagnostics"],
                "nj_models": fitted["nj_models"], "failures": fitted["failures"],
                "adapter_sha256": sha256(METHODS_PATH / "fit_adapter.py"),
                "mk_module_sha256": sha256(METHODS_PATH / "mk_baseline.py")}
    if fitted["failures"]:
        metadata.update(status="failed", seconds=time.monotonic()-start)
        json_write(output_directory / f"{matrix_id}.json", metadata)
        return {"matrix_id": matrix_id, "status": "failed", "failures": fitted["failures"]}
    probabilities = np.stack([fitted["probabilities"][m] for m in METHODS])
    if not np.all(np.isfinite(probabilities)) or not np.allclose(probabilities.sum(axis=-1), 1):
        raise ArithmeticError("Invalid saved probabilities")
    targets = np.argwhere(masked == -1)
    n_targets = len(targets)
    marginal_seed = [PROPAGATION_SEED, entry["scenario_ordinal"], entry["replicate"], 0]
    joint_seed = [PROPAGATION_SEED, entry["scenario_ordinal"], entry["replicate"], 1]
    rng = np.random.default_rng(np.random.SeedSequence(marginal_seed))
    uniforms = rng.random((N_DRAWS, n_targets))
    packed, completion_hashes = [], []
    for i, method in enumerate(METHODS):
        p1 = probabilities[i, targets[:, 0], targets[:, 1], 1]
        missing_states = (uniforms < p1[None, :]).astype(np.uint8)
        completion_hashes.append(hashlib.sha256(missing_states.tobytes()).hexdigest())
        packed.append(np.packbits(missing_states, axis=1))
    joint = sample_mk_joint(string_matrix, fitted["nj_models"], N_DRAWS, np.random.SeedSequence(joint_seed))
    assert joint.shape == (N_DRAWS, n_targets)
    assert np.isin(joint, [0, 1]).all()
    completion_hashes.append(hashlib.sha256(joint.tobytes()).hexdigest())
    packed.append(np.packbits(joint, axis=1))
    prediction_path = output_directory / f"{matrix_id}.npz"
    np.savez_compressed(prediction_path, probabilities=probabilities,
                        missing_states_packed=np.asarray(packed, dtype=np.uint8),
                        target_indices=targets.astype(np.int16))
    metadata.update(status="completed", seconds=time.monotonic()-start,
                    prediction_sha256=sha256(prediction_path),
                    constant_response_count=sum(d["constant_fallback"] for d in fitted["diagnostics"]),
                    convergence_warning_count=sum(d["classifiers"]["logistic"].get("convergence_warning", False) for d in fitted["diagnostics"]),
                    marginal_uniform_seed_sequence=marginal_seed, joint_seed_sequence=joint_seed,
                    schemes=[*METHODS, "nj_mk_joint"], n_draws=N_DRAWS, n_targets=n_targets,
                    completion_sha256=completion_hashes,
                    prediction_information_boundary="Only masked array supplied; all prediction/completion outputs saved before evaluation phase")
    json_write(output_directory / f"{matrix_id}.json", metadata)
    return {"matrix_id": matrix_id, "status": "completed", "seconds": metadata["seconds"],
            "prediction_sha256": metadata["prediction_sha256"],
            "metadata_sha256": sha256(output_directory / f"{matrix_id}.json")}


def run(base, workers):
    base = Path(base)
    output = base / "predictions"
    output.mkdir(exist_ok=False)
    entries = json.loads((base / "input_manifest.json").read_text())
    started = time.monotonic()
    records = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fit_file, base / entry["masked_file"], entry, output): entry for entry in entries}
        for future in as_completed(futures):
            entry = futures[future]
            try:
                records.append(future.result())
            except Exception as exception:
                failure = {"matrix_id": entry["matrix_id"], "status": "worker_exception",
                           "type": type(exception).__name__, "message": str(exception),
                           "traceback": traceback.format_exc()}
                records.append(failure)
                json_write(output / f"{entry['matrix_id']}_failure.json", failure)
            if len(records) % 25 == 0:
                print(json.dumps({"finished_matrices": len(records), "total_matrices": len(entries),
                                  "failed_matrices": sum(r["status"] != "completed" for r in records),
                                  "seconds": round(time.monotonic()-started, 2)}), flush=True)
    records.sort(key=lambda record: record["matrix_id"])
    manifest = {"completed_matrices": sum(r["status"] == "completed" for r in records),
                "failed_matrices": [r for r in records if r["status"] != "completed"],
                "workers": workers, "seconds": time.monotonic()-started,
                "fit_worker_sha256": sha256(__file__),
                "input_manifest_sha256": sha256(base / "input_manifest.json"),
                "adapter_sha256": sha256(METHODS_PATH / "fit_adapter.py"),
                "mk_module_sha256": sha256(METHODS_PATH / "mk_baseline.py"),
                "records": records}
    json_write(base / "FIT_MANIFEST.json", manifest)
    print(json.dumps({k:v for k,v in manifest.items() if k != "records"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    run(args.output, args.workers)
