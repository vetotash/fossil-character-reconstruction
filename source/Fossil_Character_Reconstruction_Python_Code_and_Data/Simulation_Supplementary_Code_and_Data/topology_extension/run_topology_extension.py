#!/usr/bin/env python3
"""Fixed-design NNI topology assessment of frozen fitted-method completions."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from nni_search import random_start_tree,search_matrix,score_search,split_set

METHODS=("mode","knn5","logistic","extra_trees","nj_mk")
SCHEMES=METHODS+("nj_mk_joint",)
SEED=2026100407
DRAW_INDICES=(0,73,146,219,292,365,438,511)
N_STARTS=3
MAX_STEPS=200


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def serialized_search(result):
    return {"minimum_found_length":result["minimum_found_length"],
            "n_distinct_best_endpoints":result["n_distinct_best_endpoints"],
            "any_start_step_limit":result["any_start_step_limit"],
            "starts":[{**r,"edges":r["edges"].tolist()} for r in result["starts"]],
            "retained_edges":[r["edges"].tolist() for r in result["endpoints"]]}


def run_one(arguments):
    source_s,out_s,registry_row=arguments
    source=Path(source_s);out=Path(out_s);base=source/"simulation/results"
    identifier=registry_row["matrix_id"]
    paths={"masked":base/"inputs"/f"{identifier}_masked.npz",
           "truth":base/"truth"/f"{identifier}_truth.npz",
           "predictions":base/"predictions"/f"{identifier}.npz",
           "evaluation":base/"evaluation"/f"{identifier}.npz"}
    hashes={key:sha(path) for key,path in paths.items()}
    meta=json.loads((base/"predictions"/f"{identifier}.json").read_text())
    emeta=json.loads((base/"evaluation"/f"{identifier}.json").read_text())
    assert hashes["masked"]==registry_row["masked_sha256"]==meta["masked_sha256"]
    assert hashes["truth"]==registry_row["truth_sha256"]
    assert hashes["predictions"]==meta["prediction_sha256"]
    assert hashes["evaluation"]==emeta["result_sha256"]
    with np.load(paths["masked"],allow_pickle=False) as z:masked=z["X"]
    with np.load(paths["truth"],allow_pickle=False) as z:truth=z["X_true"];truth_edges=z["edges"]
    with np.load(paths["predictions"],allow_pickle=False) as z:
        probabilities=z["probabilities"];targets=z["target_indices"]
        draws=np.unpackbits(z["missing_states_packed"],axis=2,count=len(targets))
    with np.load(paths["evaluation"],allow_pickle=False) as z:
        saved_D=z["D_samples"];true_D=float(z["true_D"]);hard_D=z["hard_D"]
    assert np.array_equal(targets,np.argwhere(masked==-1))
    assert np.array_equal(masked[masked>=0],truth[masked>=0])
    ordinal=int(registry_row["ordinal"]);replicate=int(registry_row["replicate"])
    starts=[random_start_tree(len(masked),np.random.SeedSequence([SEED,ordinal,replicate,i])) for i in range(N_STARTS)]
    # Only incomplete/completed character matrices and independent starts enter
    # search. Generating edges are supplied to the POST-SEARCH scoring function.
    cache={};fitted_records={};rows=[];start_time=time.monotonic()
    common={key:registry_row[key] for key in ("matrix_id","scenario_id","replicate","q","missing_per_response","mask_mode")}
    common.update(n_tips=len(masked),n_characters=masked.shape[1],n_targets=len(targets),true_D=true_D)
    def fit_and_score(matrix,kind,scheme,draw_id,fixed_D):
        matrix=np.asarray(matrix,dtype=np.int8)
        key=hashlib.sha256(matrix.tobytes()).hexdigest()
        if key not in cache:
            result=search_matrix(matrix,starts,max_steps=MAX_STEPS)
            # This is the only point at which the generating topology is used.
            cache[key]=score_search(result,truth_edges,len(matrix))
            fitted_records[key]=serialized_search(result)
        rows.append({**common,"kind":kind,"scheme":scheme,"draw_index":draw_id,
                     "searched_matrix_sha256":key,"fixed_scaffold_D":float(fixed_D),
                     "fixed_scaffold_absolute_D_error":float(abs(fixed_D-true_D)),
                     **cache[key]})
    fit_and_score(masked,"baseline","source_incomplete",-1,0.)
    fit_and_score(truth,"baseline","truth_complete",-1,true_D)
    for si,scheme in enumerate(SCHEMES):
        mi=si if si<5 else 4
        completed=np.where(masked>=0,masked,np.argmax(probabilities[mi],axis=-1)).astype(np.int8)
        fit_and_score(completed,"map",scheme,-1,hard_D[si])
    for si,scheme in enumerate(SCHEMES):
        for draw_id in DRAW_INDICES:
            completed=masked.copy();completed[targets[:,0],targets[:,1]]=draws[si,draw_id]
            fit_and_score(completed,"draw",scheme,draw_id,saved_D[si,draw_id])
    source_RF=rows[0]["mean_normalized_RF"];truth_RF=rows[1]["mean_normalized_RF"]
    for row in rows:
        row["RF_difference_from_source_incomplete"]=row["mean_normalized_RF"]-source_RF
        row["RF_difference_from_truth_complete"]=row["mean_normalized_RF"]-truth_RF
        row["worse_RF_than_source_incomplete"]=int(row["mean_normalized_RF"]>source_RF+1e-12)
        row["worse_RF_than_truth_complete"]=int(row["mean_normalized_RF"]>truth_RF+1e-12)
        row["fixed_D_exact_match"]=int(row["fixed_scaffold_D"]==true_D)
        row["D_exact_and_RF_worse_than_truth_complete"]=row["fixed_D_exact_match"]*row["worse_RF_than_truth_complete"]
    receipt={"matrix_id":identifier,"source_hashes":hashes,
             "start_seed_sequences":[[SEED,ordinal,replicate,i] for i in range(N_STARTS)],
             "start_edges":[e.tolist() for e in starts],
             "draw_indices":list(DRAW_INDICES),"n_recorded_treatments":len(rows),
             "n_distinct_input_matrices_searched":len(fitted_records),
             "seconds":time.monotonic()-start_time,
             "searches":fitted_records,"treatment_results":rows}
    target=out/"search_records"/f"{identifier}.json.gz"
    with gzip.open(target,"wt",encoding="utf8") as handle:json.dump(receipt,handle,separators=(",",":"),allow_nan=True)
    return rows,{k:v for k,v in receipt.items() if k not in ("searches","treatment_results","start_edges")}


def summarize_matrix(rows):
    frame=pd.DataFrame(rows)
    metrics=["minimum_found_length","n_distinct_best_endpoints","mean_normalized_RF","min_normalized_RF","max_normalized_RF",
             "mean_split_recall","truth_among_retained_endpoints","consensus_n_splits","consensus_resolution",
             "consensus_split_recall","consensus_split_precision","any_start_step_limit","mean_accepted_moves",
             "fixed_scaffold_absolute_D_error","RF_difference_from_source_incomplete","RF_difference_from_truth_complete",
             "worse_RF_than_source_incomplete","worse_RF_than_truth_complete","fixed_D_exact_match","D_exact_and_RF_worse_than_truth_complete"]
    keys=["matrix_id","scenario_id","replicate","q","missing_per_response","mask_mode","kind","scheme"]
    result=[]
    for key,part in frame.groupby(keys,sort=True):
        row=dict(zip(keys,key));row["n_selected_completions"]=len(part)
        for metric in metrics:row[metric]=float(part[metric].mean())
        row["source_incomplete_RF"]=float(part.mean_normalized_RF.iloc[0]-part.RF_difference_from_source_incomplete.iloc[0])
        row["truth_complete_RF"]=float(part.mean_normalized_RF.iloc[0]-part.RF_difference_from_truth_complete.iloc[0])
        result.append(row)
    return pd.DataFrame(result),metrics


def scenario_summaries(matrix,metrics):
    keys=["scenario_id","q","missing_per_response","mask_mode","kind","scheme"]
    rows=[]
    for key,part in matrix.groupby(keys,sort=True):
        row=dict(zip(keys,key));row["n_independent_matrices"]=len(part)
        assert len(part)==part.matrix_id.nunique()
        for metric in metrics:
            vals=part[metric].dropna().to_numpy(float)
            row[metric+"_mean"]=float(vals.mean()) if len(vals) else np.nan
            row[metric+"_MCSE"]=float(vals.std(ddof=1)/np.sqrt(len(vals))) if len(vals)>1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def paired_scheme_summaries(matrix):
    rows=[]
    for (scenario,kind),part in matrix[matrix.kind.isin(["map","draw"])].groupby(["scenario_id","kind"],sort=True):
        for reference,candidate in itertools.combinations(SCHEMES,2):
            a=part[part.scheme==reference].set_index("matrix_id")
            b=part[part.scheme==candidate].set_index("matrix_id").loc[a.index]
            for metric in ["mean_normalized_RF","consensus_split_recall","consensus_resolution","fixed_scaffold_absolute_D_error"]:
                diff=b[metric].to_numpy()-a[metric].to_numpy()
                rows.append({"scenario_id":scenario,"kind":kind,"reference":reference,"candidate":candidate,
                             "metric":metric,"direction":"candidate_minus_reference","n_paired_matrices":len(diff),
                             "mean_difference":float(diff.mean()),"paired_MCSE":float(diff.std(ddof=1)/np.sqrt(len(diff)))})
    return pd.DataFrame(rows)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root",type=Path,required=True);ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--workers",type=int,default=4);ap.add_argument("--limit",type=int)
    args=ap.parse_args();root=args.root.resolve();out=args.output.resolve()
    if out==root or root in out.parents:raise ValueError("Output must be separate from original archive")
    if out.exists() and any(out.iterdir()):raise FileExistsError("Use a new empty output directory")
    out.mkdir(parents=True,exist_ok=True);(out/"search_records").mkdir()
    design=json.loads((Path(__file__).parent/"DESIGN.json").read_text())
    assert design["draw_indices"]==list(DRAW_INDICES) and design["n_starts"]==N_STARTS
    assert design["max_strict_NNI_steps_per_start"]==MAX_STEPS and design["search_seed"]==SEED
    registry=pd.read_csv(root/"simulation/results/matrix_registry.csv").sort_values(["scenario_id","replicate"])
    if args.limit:registry=registry.head(args.limit)
    started=time.monotonic();all_rows=[];receipts=[]
    payload=[(str(root),str(out),r) for r in registry.to_dict("records")]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i,(rows,receipt) in enumerate(pool.map(run_one,payload),1):
            all_rows.extend(rows);receipts.append(receipt)
            if i%25==0 or i==len(payload):print(f"Topology searches complete: {i}/{len(payload)} matrices; {time.monotonic()-started:.1f}s",flush=True)
    raw=pd.DataFrame(all_rows);matrix,metrics=summarize_matrix(all_rows)
    raw.to_csv(out/"selected_completion_topology.csv",index=False)
    matrix.to_csv(out/"matrix_topology_summary.csv",index=False)
    scenario_summaries(matrix,metrics).to_csv(out/"scenario_topology_summary.csv",index=False)
    paired_scheme_summaries(matrix).to_csv(out/"paired_scheme_contrasts.csv",index=False)
    # The MAP annotation for joint Mk must be an exact alias of marginal Mk.
    map_m=matrix[(matrix.kind=="map")&(matrix.scheme=="nj_mk")].set_index("matrix_id")
    map_j=matrix[(matrix.kind=="map")&(matrix.scheme=="nj_mk_joint")].set_index("matrix_id").loc[map_m.index]
    np.testing.assert_allclose(map_m[metrics],map_j[metrics],equal_nan=True)
    manifest={"status":"PASS","n_source_matrices":len(registry),"complete_600_matrix_analysis":len(registry)==600 and args.limit is None,
              "n_treatment_records":len(raw),"n_matrix_summary_rows":len(matrix),
              "n_distinct_input_matrices_searched":sum(r["n_distinct_input_matrices_searched"] for r in receipts),
              "retained_endpoint_count_total":int(raw.n_distinct_best_endpoints.sum()),
              "n_treatment_records_with_step_limit":int(raw.any_start_step_limit.sum()),
              "seconds":time.monotonic()-started,"design":design,
              "design_sha256":sha(Path(__file__).parent/"DESIGN.json"),
              "search_code_sha256":sha(Path(__file__).parent/"nni_search.py"),"driver_sha256":sha(__file__),
              "python":sys.version,"numpy":np.__version__,"pandas":pd.__version__,
              "search_is_heuristic":True,"minimum_found_is_not_claimed_global":True,
              "generating_tree_used_only_for_post_search_metrics":True,
              "joint_MAP_exactly_shared_with_marginal_Mk":True,
              "independent_inference_unit":"matrix; selected completion draws are averaged within matrix before MCSE",
              "matrix_receipts":receipts}
    (out/"RUN_MANIFEST.json").write_text(json.dumps(manifest,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:manifest[k] for k in ("status","n_source_matrices","n_treatment_records","n_distinct_input_matrices_searched","n_treatment_records_with_step_limit","seconds")},indent=2))


if __name__=="__main__":main()
