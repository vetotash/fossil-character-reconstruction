#!/usr/bin/env python3
"""Scheduling-only wrapper for frozen R17 evaluation arithmetic.

No RNG is used in evaluation. This wrapper changes process scheduling only,
retains input-manifest result order, and binds its own and the core hashes.
"""
import os
for variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[variable]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time
import experiment

def run(base, workers):
    base=Path(base)
    fit=json.loads((base/'FIT_MANIFEST.json').read_text())
    entries=json.loads((base/'input_manifest.json').read_text())
    assert not fit['failed_arms'] and fit['completed_arms']==len(entries)
    for record in fit['records']:
        assert experiment.sha256(base/'predictions'/f"{record['matrix_id']}.npz")==record['prediction_sha256']
        assert experiment.sha256(base/'predictions'/f"{record['matrix_id']}.json")==record['metadata_sha256']
    (base/'evaluation').mkdir(exist_ok=False)
    started=time.monotonic(); by_id={}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures={executor.submit(experiment.evaluate_one,base,e):e['matrix_id'] for e in entries}
        for future in as_completed(futures):
            matrix_id=futures[future]
            by_id[matrix_id]=future.result()
            if len(by_id)%10==0:
                print(json.dumps({'evaluated_arms':len(by_id),'total_arms':len(entries),
                                  'seconds':round(time.monotonic()-started,2)}),flush=True)
    rows=[row for entry in entries for row in by_id[entry['matrix_id']]]
    experiment.summarize(base,rows)
    experiment.write_json(base/'EVALUATION_MANIFEST.json',{
        'arm_matrices':len(entries),'paired_matrices':len(entries)//2,'rows':len(rows),
        'seconds':time.monotonic()-started,'script_sha256':experiment.sha256(experiment.__file__),
        'fit_manifest_sha256':experiment.sha256(base/'FIT_MANIFEST.json'),
        'scheduling_wrapper_sha256':experiment.sha256(__file__),'workers':workers,
        'arithmetic':'Frozen experiment.evaluate_one and summarize; no RNG; input-manifest result ordering'})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path)
    p.add_argument('--workers',type=int,default=8);a=p.parse_args();run(a.output,a.workers)
