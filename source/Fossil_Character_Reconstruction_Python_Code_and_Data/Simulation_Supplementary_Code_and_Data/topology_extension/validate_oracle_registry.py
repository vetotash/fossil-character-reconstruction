#!/usr/bin/env python3
"""Check new NNI scorer against the supplied exhaustive six-tip registry.

This validates all 105 topologies x729 partial binary-state patterns and every
registered split set. It does not assert global optimality of 48-tip searches.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from nni_search import full_character_scores,split_set


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    root=args.oracle_root.resolve();output=args.output.resolve()
    if root==output or root in output.parents:raise ValueError("Do not write inside the source oracle archive")
    base=root/"topology/registry"
    registry_path=base/"six_tip_tree_registry.json"
    costs_path=base/"six_tip_cost_registry.npz"
    registry=json.loads(registry_path.read_text())
    with np.load(costs_path,allow_pickle=False) as z:
        patterns=np.where(z["ternary_patterns"]==2,-1,z["ternary_patterns"]).astype(np.int8)
        costs=z["costs"]
    assert patterns.shape==(729,6) and costs.shape==(105,729)
    assert registry["ntips"]==6 and registry["ntrees"]==105
    for index,record in enumerate(registry["trees"]):
        edges=np.asarray(record["edges"],dtype=int)
        observed=full_character_scores(patterns.T,edges)
        np.testing.assert_array_equal(observed,costs[index])
        full=(1<<6)-1
        expected=set()
        for split in record["splits"]:
            bitmask=sum(1<<int(tip) for tip in split)
            expected.add(min(bitmask,full^bitmask))
        assert split_set(edges,6)==expected
    result={"status":"PASS","trees":105,"partial_patterns":729,
            "exact_character_scores_compared":105*729,"split_sets_compared":105,
            "maximum_score_difference":0,
            "input_sha256":{registry_path.name:sha(registry_path),costs_path.name:sha(costs_path)},
            "search_code_sha256":sha(Path(__file__).parent/"nni_search.py"),
            "validation_code_sha256":sha(__file__),
            "scope":"Exact small-tree scoring and split validation; not proof of global optimality of the large-tree heuristic"}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
