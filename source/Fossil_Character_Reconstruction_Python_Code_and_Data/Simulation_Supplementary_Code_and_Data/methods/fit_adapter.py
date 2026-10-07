"""Frozen empirical fitting methods for masked binary simulation matrices.

The public fitting interface receives only the masked matrix, a response column,
and optionally its observed labels. It has no true matrix, generating tree,
generating rate, or hidden response input. Evaluation belongs in the caller.

The copied mk_baseline.py is byte-identical to the Round 14 historical module.
The four tabular methods below reproduce dinosaur_matched_pilot.predict while
adding input validation and execution diagnostics, not model changes.
"""
from __future__ import annotations

import time
import warnings
from typing import Mapping

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

import mk_baseline as mk

METHODS = ("mode", "knn5", "logistic", "extra_trees", "nj_mk")
RANDOM_STATE = 20261002


def _instrumented_mk(edges, n_tips, observed, train, test):
    """Capture optimizer diagnostics without modifying historical arithmetic.

    Use process-level (not same-module threaded) parallelism for matrix fits.
    """
    optimizer = {"ran": False, "success": None, "message": "single-class fallback",
                 "nit": 0, "nfev": 0}
    original = mk.minimize_scalar

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        optimizer.update({"ran": True, "success": bool(result.success),
                          "message": str(result.message), "nit": int(result.nit),
                          "nfev": int(result.nfev), "refined_logq": float(result.x),
                          "refined_negative_loglik": float(result.fun),
                          "bounds": list(kwargs["bounds"]), "method": kwargs["method"],
                          "options": kwargs["options"]})
        return result

    mk.minimize_scalar = capture
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            labels, probabilities, diagnostic = mk.predict_mk(edges, n_tips, observed, train, test)
    finally:
        mk.minimize_scalar = original
    diagnostic["optimizer"] = optimizer
    diagnostic["warnings"] = [{"category": w.category.__name__, "message": str(w.message)} for w in caught]
    return labels, probabilities, diagnostic


def _validate(masked_matrix, target_character, observed_labels):
    matrix = np.asarray(masked_matrix)
    if matrix.ndim != 2 or matrix.shape[0] < 3 or matrix.shape[1] < 2:
        raise ValueError("Expected >=3 taxa and >=2 characters")
    if not np.isin(matrix, ["0", "1", "?"]).all():
        raise ValueError("Adapter supports 0, 1, ? only; encode missing as ?")
    matrix = matrix.astype("U1", copy=True)
    if not isinstance(target_character, (int, np.integer)):
        raise TypeError("target_character must be a zero-based integer")
    j = int(target_character)
    if not 0 <= j < matrix.shape[1]:
        raise IndexError(j)
    train = np.flatnonzero(matrix[:, j] != "?")
    test = np.flatnonzero(matrix[:, j] == "?")
    if len(train) == 0 or len(test) == 0:
        raise ValueError("Need at least one observed and one hidden response")
    if observed_labels is not None:
        if not isinstance(observed_labels, Mapping):
            raise TypeError("observed_labels must map observed row indices to states")
        supplied = {int(i): str(state) for i, state in observed_labels.items()}
        expected = {int(i): str(matrix[i, j]) for i in train}
        if supplied != expected:
            raise ValueError("observed_labels must match exactly the visible response cells")
    return matrix, j, train, test


def fit_character(masked_matrix, target_character, observed_labels=None):
    """Fit all five methods, returning predictions for each missing response.

    Parameters
    ----------
    masked_matrix : array of '0', '1', '?', shape (taxa, characters)
    target_character : zero-based column index
    observed_labels : optional mapping of every observed response row to state
        Its only purpose is to verify the fitting boundary. Hidden rows cannot
        be supplied. The response is always read from masked_matrix.

    Returns
    -------
    dict with train_indices, target_indices, methods, diagnostics, nj_model.
    methods[method] is (classes, probability_matrix). Classes are exclusively
    observed training states; callers map absent binary states to probability
    zero for evaluation. The one-class fallback therefore has one column.
    nj_model retains inferred edges/rate for optional downstream propagation.
    No method receives held-out labels or generating parameters.
    """
    start = time.monotonic()
    matrix, j, train, test = _validate(masked_matrix, target_character, observed_labels)
    y = matrix[train, j]
    classes, counts = np.unique(y, return_counts=True)
    base = np.tile(counts / counts.sum(), (len(test), 1))
    methods = {"mode": (classes.copy(), base.copy())}
    diagnostics = {
        "n_train": len(train), "n_test": len(test),
        "observed_classes": classes.tolist(), "constant_fallback": len(classes) == 1,
        "timings": {}, "classifiers": {},
    }
    if len(classes) == 1:
        for name in ("knn5", "logistic", "extra_trees"):
            methods[name] = (classes.copy(), base.copy())
        diagnostics["classifiers"] = {
            name: {"fit_ran": False, "constant_fallback": True, "warnings": []}
            for name in ("logistic", "extra_trees")
        }
    else:
        xx = np.delete(matrix, j, axis=1)
        # Keeping ? as a nominal feature and ignoring unseen categories matches
        # the empirical code. No truth-based or complete-matrix preprocessing.
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        xtr = encoder.fit_transform(xx[train])
        xte = encoder.transform(xx[test])
        diagnostics["onehot_categories"] = [c.tolist() for c in encoder.categories_]
        for name, model in (
            ("extra_trees", ExtraTreesClassifier(
                n_estimators=100, min_samples_leaf=2, max_features="sqrt",
                bootstrap=False, random_state=RANDOM_STATE, n_jobs=1)),
            ("logistic", LogisticRegression(C=1, max_iter=300, random_state=RANDOM_STATE)),
        ):
            fitted_start = time.monotonic()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model.fit(xtr, y)
            methods[name] = (model.classes_.copy(), model.predict_proba(xte))
            n_iter = getattr(model, "n_iter_", None)
            diagnostics["timings"][name] = time.monotonic() - fitted_start
            diagnostics["classifiers"][name] = {
                "fit_ran": True, "constant_fallback": False,
                "n_iter": None if n_iter is None else np.asarray(n_iter).tolist(),
                "reached_max_iter": False if n_iter is None else bool(np.any(n_iter >= model.max_iter)),
                "convergence_warning": any(issubclass(w.category, ConvergenceWarning) for w in caught),
                "warnings": [{"category": w.category.__name__, "message": str(w.message)} for w in caught],
            }
        fitted_start = time.monotonic()
        probs = []
        no_overlap = []
        for row in xx[test]:
            jointly_observed = (xx[train] != "?") & (row[None, :] != "?")
            denominator = jointly_observed.sum(1)
            d = np.divide(((xx[train] != row) & jointly_observed).sum(1), denominator,
                          out=np.full(len(train), np.inf), where=denominator > 0)
            # Preserve the original stable tie rule, including inf distances.
            indices = np.argsort(d, kind="stable")[:5]
            probs.append([(y[indices] == label).mean() for label in classes])
            no_overlap.append(int((denominator == 0).sum()))
        methods["knn5"] = (classes.copy(), np.asarray(probs))
        diagnostics["timings"]["knn5"] = time.monotonic() - fitted_start
        diagnostics["knn_no_overlap_training_rows_per_target"] = no_overlap

    fitted_start = time.monotonic()
    distances, zero_pairs = mk.excluding_distance(matrix, j, mk.distance_totals(matrix))
    edges, nj_diag = mk.neighbor_join(distances)
    mk_classes, probabilities, mk_diag = _instrumented_mk(edges, len(matrix), matrix[:, j], train, test)
    methods["nj_mk"] = (mk_classes, probabilities)
    diagnostics["timings"]["nj_mk"] = time.monotonic() - fitted_start
    diagnostics["nj_mk"] = {"zero_overlap_pairs": zero_pairs, **nj_diag, **mk_diag}
    diagnostics["seconds"] = time.monotonic() - start
    for name in METHODS:
        labels, p = methods[name]
        assert p.shape == (len(test), len(labels)), (name, p.shape)
        assert np.isfinite(p).all() and (p >= 0).all() and np.allclose(p.sum(1), 1)
    return {
        "train_indices": train, "target_indices": test,
        "methods": {name: methods[name] for name in METHODS},
        "diagnostics": diagnostics,
        "nj_model": {"edges": edges, "classes": mk_classes.tolist(), "rate": mk_diag["rate"],
                     "constant_fallback": mk_diag["constant_fallback"],
                     "train_indices": train.tolist(), "train_states": y.tolist()},
    }


def fit_matrix(masked_matrix):
    """Fit each response column; return method arrays with shape n x p x 2.

    Observed entries have their fixed one-hot probability. All hidden entries
    are predictions. Diagnostic and fitted Mk records follow column order.
    Only columns containing ? are fitted. Completely observed anchor columns
    are carried as one-hot probabilities and are never response fits.
    This convenience interface is likewise blind to the generating truth.
    """
    matrix = np.asarray(masked_matrix)
    if matrix.ndim != 2 or not np.isin(matrix, ["0", "1", "?"]).all():
        raise ValueError("Expected 2D array with 0, 1, ? only")
    n, p = matrix.shape
    probability = {name: np.zeros((n, p, 2), dtype=float) for name in METHODS}
    observed = matrix != "?"
    for name in METHODS:
        for state in range(2):
            probability[name][:, :, state][observed] = (matrix[observed] == str(state))
    diagnostics = []
    nj_models = [None] * p
    failures = []
    for j in np.flatnonzero((matrix == "?").any(0)):
        j = int(j)
        try:
            fitted = fit_character(matrix, j)
        except Exception as exception:
            failure = {"character": j, "status": "fit_failure", "type": type(exception).__name__,
                       "message": str(exception)}
            failures.append(failure)
            diagnostics.append(failure)
            for name in METHODS:
                probability[name][matrix[:, j] == "?", j] = np.nan
            continue
        for name, (classes, probs) in fitted["methods"].items():
            for k, state in enumerate(classes):
                probability[name][fitted["target_indices"], j, int(state)] = probs[:, k]
        diagnostics.append({"character": j, "status": "completed", **fitted["diagnostics"]})
        nj_models[j] = fitted["nj_model"]
    return {"probabilities": probability, "diagnostics": diagnostics,
            "nj_models": nj_models, "failures": failures}


def sample_mk_joint(masked_matrix, nj_models, n_draws, seed):
    """Conditional Mk draws at missing cells in np.argwhere row-major order.

    This follows the archived Round 8 joint sampler's upward messages then
    conditional node sampling. Characters remain independent. The tree and
    rate supplied here must be fitted nj_models, never generating parameters.
    """
    matrix = np.asarray(masked_matrix)
    locations = np.argwhere(matrix == "?")
    output = np.zeros((int(n_draws), len(locations)), dtype=np.uint8)
    rng = np.random.default_rng(seed)
    for j in np.unique(locations[:, 1]):
        model = nj_models[int(j)]
        if model is None:
            raise ValueError(f"No successfully fitted Mk model for column {j}")
        classes = np.asarray(model["classes"])
        columns = np.flatnonzero(locations[:, 1] == j)
        tips = locations[columns, 0]
        if model["constant_fallback"]:
            output[:, columns] = int(classes[0])
            continue
        tree = mk.orient(model["edges"])
        root, order, children, lengths = tree
        k = len(classes)
        emission = np.ones((len(order), k))
        observed_rows = np.flatnonzero(matrix[:, j] != "?")
        assert observed_rows.tolist() == model["train_indices"]
        assert matrix[observed_rows, j].tolist() == model["train_states"]
        for i in observed_rows:
            emission[i] = (classes == matrix[i, j]).astype(float)
        up = np.ones_like(emission)
        messages = np.ones_like(emission)
        for node in reversed(order):
            value = emission[node].copy()
            for child in children[node]:
                value *= messages[child]
            up[node] = value / value.max()
            if node != root:
                messages[node] = mk.transition(model["rate"], lengths[node], k) @ up[node]
        states = np.empty((int(n_draws), len(order)), dtype=np.int16)
        root_probability = up[root] / up[root].sum()
        states[:, root] = rng.choice(k, size=int(n_draws), p=root_probability)
        for parent in order:
            for child in children[parent]:
                conditional = mk.transition(model["rate"], lengths[child], k) * up[child][None, :]
                conditional /= conditional.sum(axis=1, keepdims=True)
                cdf = np.cumsum(conditional[states[:, parent]], axis=1)
                cdf[:, -1] = 1.0
                states[:, child] = (rng.random(int(n_draws))[:, None] > cdf).sum(axis=1)
        assert np.array_equal(classes[states[:, observed_rows]],
                              np.broadcast_to(matrix[observed_rows, j], (int(n_draws), len(observed_rows))))
        output[:, columns] = classes[states[:, tips]].astype(np.uint8)
    return output
