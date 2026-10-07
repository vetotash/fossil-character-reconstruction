"""Unmodified scoring/aggregation functions extracted from the validated 600-matrix analysis.
Source SHA256: ea0b4d4a1bef18698a96936cbbdc89480440a87f13ee66db826f5bb8d0e76d2e
The extraction copies full AST source spans verbatim; no fitting functions are included.
"""

import math
import numpy as np
import pandas as pd

def crps_empirical(draws, truth, *, fair=False):
    """O(M log M) CRPS; supports ties, negative draws, and point forecasts."""
    values = np.asarray(draws, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or not np.isfinite(truth):
        raise ValueError("CRPS requires a finite nonempty vector and finite truth")
    m = len(values)
    if fair and m < 2:
        raise ValueError("Fair-ensemble estimate requires at least two draws")
    ordered = np.sort(values)
    # Subtract the minimum to avoid cancellation under a common large offset.
    pair_sum_half = np.dot(2 * np.arange(1, m + 1) - m - 1, ordered - ordered[0])
    denominator = m * (m - 1) if fair else m * m
    return float(np.mean(np.abs(values - truth)) - pair_sum_half / denominator)

def distribution_metrics(draws, truth):
    draws = np.asarray(draws, dtype=np.float64)
    low, high = np.quantile(draws, [0.025, 0.975], method="linear")
    bias = float(draws.mean() - truth)
    return {
        "crps_empirical": crps_empirical(draws, truth),
        "crps_fair_ensemble": crps_empirical(draws, truth, fair=True),
        "mean_D": float(draws.mean()), "signed_bias": bias,
        "absolute_mean_error": abs(bias), "sd_D": float(draws.std(ddof=1)),
        "lower95": float(low), "upper95": float(high), "width95": float(high-low),
        "truth_in_95": int(low <= truth <= high),
        "truth_below_95": int(truth < low), "truth_above_95": int(truth > high),
    }

def aggregate(frame, groups, metrics):
    """One independent matrix per group row; MCSE is sample SD / sqrt(n)."""
    rows = []
    for keys, subset in frame.groupby(groups, sort=True, dropna=False):
        if not isinstance(keys, tuple): keys = (keys,)
        result = dict(zip(groups, keys))
        assert subset.matrix_id.nunique() == len(subset)
        result["n_matrices"] = len(subset)
        for metric in metrics:
            values = subset[metric].to_numpy(float)
            finite = values[np.isfinite(values)]
            result[metric+"_n"] = len(finite)
            result[metric+"_mean"] = float(finite.mean()) if len(finite) else np.nan
            result[metric+"_MCSE"] = float(finite.std(ddof=1)/math.sqrt(len(finite))) if len(finite) > 1 else np.nan
        if "truth_in_95" in metrics:
            successes = int(subset.truth_in_95.sum())
            result["inclusion_count"] = successes
            result["inclusion_Wilson_low"], result["inclusion_Wilson_high"] = wilson(successes, len(subset))
        rows.append(result)
    return pd.DataFrame(rows)

def wilson(successes, total):
    z = 1.959963984540054
    p = successes/total
    d = 1+z*z/total
    midpoint = (p+z*z/(2*total))/d
    half = z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/d
    return midpoint-half, midpoint+half
