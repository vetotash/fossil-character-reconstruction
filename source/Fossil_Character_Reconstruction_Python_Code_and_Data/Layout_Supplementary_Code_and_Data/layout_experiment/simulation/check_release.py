"""Production bookkeeping checks; independent scientific audit is separate."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import numpy as np

def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main(base):
    base=Path(base); here=Path(__file__).resolve().parent
    generation=json.loads((base/'GENERATION_MANIFEST.json').read_text())
    fits=json.loads((base/'FIT_MANIFEST.json').read_text())
    evaluation=json.loads((base/'EVALUATION_MANIFEST.json').read_text())
    assert generation['paired_matrices']==180 and generation['arm_matrices']==360
    assert fits['completed_arms']==360 and not fits['failed_arms']
    assert fits['fit_worker_sha256']==digest(here/'fit_worker.py')
    assert generation['script_sha256']==evaluation['script_sha256']==digest(here/'experiment.py')
    assert evaluation['fit_manifest_sha256']==digest(base/'FIT_MANIFEST.json')
    tables={name:list(csv.DictReader((base/name).open())) for name in ['matrix_results.csv','scenario_summary.csv','paired_layout_contrasts.csv']}
    assert [len(tables[name]) for name in tables]==[2160,72,36]
    assert all(int(x['n_matrices'])==30 for x in tables['scenario_summary.csv'])
    assert all(int(x['n_pairs'])==30 for x in tables['paired_layout_contrasts.csv'])
    for rec in fits['records']:
        mid=rec['matrix_id']; pred=base/'predictions'/f'{mid}.npz'; meta=base/'predictions'/f'{mid}.json'
        assert digest(pred)==rec['prediction_sha256'] and digest(meta)==rec['metadata_sha256']
        d=np.load(pred); assert d['probabilities'].shape[0]==5 and d['missing_states_packed'].shape[:2]==(6,512)
        assert np.isfinite(d['probabilities']).all() and np.allclose(d['probabilities'].sum(-1),1)
    checks={'status':'PASS','latent_matrices':180,'arm_matrices':360,'completion_draws':1105920,
            'failed_fits':0,'script_hashes_stable':True,'raw_prediction_and_metadata_hashes_stable':True,
            'table_row_counts':{name:len(rows) for name,rows in tables.items()},
            'note':'Bookkeeping checks only. Independent numerical audit receipt is separate.'}
    (base/'PRODUCTION_CHECKS.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(checks))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);main(p.parse_args().output)
