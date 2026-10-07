"""Exhaustive independent audit of the 105-tree finite lookup table."""
from pathlib import Path
import hashlib
import json
import numpy as np
from independent_engine import bits, compatible, enumerate_internal_states

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def edge_splits(edges, tips):
    vertices = set(x for e in edges for x in e)
    adjacency = {x: set() for x in vertices}
    for a, b in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)
    assert all(len(adjacency[t]) == 1 for t in tips)
    assert all(len(adjacency[t]) == 3 for t in vertices - set(tips))
    splits = []
    for a, b in edges:
        visited, frontier = {a}, [a]
        while frontier:
            current = frontier.pop()
            for nxt in adjacency[current]:
                if {current, nxt} == {a, b} or nxt in visited:
                    continue
                visited.add(nxt)
                frontier.append(nxt)
        part = tuple(sorted(set(tips) & visited))
        complement = tuple(sorted(set(tips) - set(part)))
        if min(len(part), len(complement)) < 2:
            continue
        splits.append(min((part, complement), key=lambda x: (len(x), x)))
    return tuple(sorted(splits))


def main():
    manifest = ROOT / 'topology/registry/six_tip_tree_registry.json'
    arrayfile = ROOT / 'topology/registry/six_tip_cost_registry.npz'
    registry = json.loads(manifest.read_text())
    data = np.load(arrayfile)
    tips = registry['pattern_tip_order']
    ternary = data['ternary_patterns']
    full = bits(len(tips))
    assert len(registry['trees']) == 105
    assert ternary.shape == (729, 6)
    assert len({tuple(p) for p in ternary}) == 729
    minima = ((ternary == 0).any(axis=1) & (ternary == 1).any(axis=1)).astype(int)
    np.testing.assert_array_equal(minima, data['minimum_steps'])
    compatible_indexes = [np.flatnonzero(compatible(full, np.where(p == 2, -1, p))) for p in ternary]
    all_splits = []
    max_cost_difference = 0
    for record in registry['trees']:
        graph_splits = edge_splits(record['edges'], tips)
        assert graph_splits == tuple(sorted(tuple(s) for s in record['splits']))
        assert len(graph_splits) == 3
        all_splits.append(graph_splits)
        _, full_cost = enumerate_internal_states(full, record['edges'], tips, .1)
        partial_cost = np.array([full_cost[idx].min() for idx in compatible_indexes])
        delta = abs(partial_cost - data['costs'][record['tree_index']]).max()
        max_cost_difference = max(max_cost_difference, int(delta))
        assert delta == 0
    assert len(set(all_splits)) == 105
    expected_full = ((np.arange(64)[:, None] >> np.arange(6)) & 1)
    np.testing.assert_array_equal(ternary[data['full_to_ternary']], expected_full)
    result = {
        'status': 'PASS', 'algorithm': 'enumerate internal assignments, then minimize compatible complete tip patterns',
        'tree_count': 105, 'partial_pattern_count_per_tree': 729,
        'cost_values_checked': 105 * 729,
        'unique_split_sets': 105, 'nontrivial_splits_per_tree': 3,
        'max_absolute_cost_difference': max_cost_difference,
        'minimum_and_pattern_maps_checked': True,
        'input_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (manifest, arrayfile)},
    }
    (HERE / 'TOPOLOGY_REGISTRY_INDEPENDENT_AUDIT.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
