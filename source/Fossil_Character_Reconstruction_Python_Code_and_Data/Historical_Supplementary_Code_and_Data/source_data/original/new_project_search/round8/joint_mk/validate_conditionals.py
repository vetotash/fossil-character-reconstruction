"""Exact small-tree enumeration check for the forward conditional formula."""
from pathlib import Path
from itertools import product
import json
import numpy as np
from generate_joint_draws import orient, transition, pruning, upward

out = Path(__file__).resolve().parent
edges = [(4, 0, .1), (4, 1, .2), (4, 5, .15), (5, 2, .3), (5, 3, .07)]
tree = orient(edges)
root, order, children, lengths = tree
results = []
for k in [2, 3]:
    rate = .73
    emission = np.ones((6, k))
    emission[0] = np.arange(k) == 0
    emission[1] = np.arange(k) == k - 1
    up = upward(tree, emission, rate)
    _, marginal = pruning(tree, emission, rate, True)
    assignments = np.array(list(product(range(k), repeat=6)))
    direct = np.ones(len(assignments)) / k
    forward = up[root, assignments[:, root]] / up[root].sum()
    for node in order:
        direct *= emission[node, assignments[:, node]]
        for child in children[node]:
            p = transition(rate, lengths[child], k)
            direct *= p[assignments[:, node], assignments[:, child]]
            conditional = p * up[child][None, :]
            conditional /= conditional.sum(axis=1, keepdims=True)
            forward *= conditional[assignments[:, node], assignments[:, child]]
    direct /= direct.sum()
    assert np.isclose(forward.sum(), 1)
    joint_error = float(np.max(np.abs(direct - forward)))
    enumerated_marginal = np.array([[direct[assignments[:, node] == s].sum()
                                    for s in range(k)] for node in range(6)])
    marginal_error = float(np.max(np.abs(enumerated_marginal - marginal)))
    assert joint_error < 1e-12 and marginal_error < 1e-12
    results.append({"states": k, "all_node_assignments": len(assignments),
                    "maximum_exact_joint_probability_error": joint_error,
                    "maximum_exact_marginal_probability_error": marginal_error})
(out / "EXACT_ENUMERATION_CHECK.json").write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
