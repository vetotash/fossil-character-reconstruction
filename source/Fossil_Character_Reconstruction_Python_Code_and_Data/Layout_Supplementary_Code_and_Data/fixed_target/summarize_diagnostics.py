#!/usr/bin/env python3
"""Summarize every new fixed-target fit; do not filter boundary fits or warnings."""
import argparse,csv,json,hashlib
from pathlib import Path

KEYS=('constant_fallback','logistic_convergence_warning','logistic_iteration_limit',
      'mk_optimizer_ran','mk_optimizer_failed','mk_rate_boundary','mk_warning_count',
      'nj_zero_overlap_pair_events','nj_negative_branch_events','nj_truncated_branch_events')
def read(p):return json.loads(p.read_text())
def write_csv(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def run(base):
 fit=read(base/'FIT_MANIFEST.json');assert fit['completed_arms']==180 and not fit['failed_arms']
 rows=[]
 for item in fit['records']:
  path=base/'predictions'/f"{item['matrix_id']}.json"
  assert hashlib.sha256(path.read_bytes()).hexdigest()==item['metadata_sha256']
  meta=read(path)
  for d in meta['diagnostics']:
   m=d['nj_mk'];o=m.get('optimizer',{});l=d['classifiers']['logistic']
   rows.append({'matrix_id':item['matrix_id'],'scenario_id':item['matrix_id'][:3],
    'arm':'fixed_target_permuted','character':d['character'],'observed_training_rows':d['n_train'],
    'n_targets':d['n_historical_targets'],'constant_fallback':int(d['constant_fallback']),
    'logistic_convergence_warning':int(l.get('convergence_warning',False)),
    'logistic_iteration_limit':int(l.get('reached_max_iter',False)),
    'mk_optimizer_ran':int(o.get('ran',False)),
    'mk_optimizer_failed':int(o.get('ran',False) and not o.get('success',False)),
    'mk_rate':m['rate'],'mk_rate_boundary':int(m['rate_boundary']),'mk_warning_count':len(m.get('warnings',[])),
    'nj_zero_overlap_pair_events':m['zero_overlap_pairs'],'nj_negative_branch_events':m['negative_branches'],
    'nj_truncated_branch_events':m['truncated_branches']})
 assert len(rows)==34920
 write_csv(base/'fit_character_diagnostics.csv',rows)
 summaries=[]
 for scenario in sorted({r['scenario_id'] for r in rows}):
  rs=[r for r in rows if r['scenario_id']==scenario]
  summaries.append({'scenario_id':scenario,'arm':'fixed_target_permuted','response_fit_events':len(rs),
   **{key:sum(r[key] for r in rs) for key in KEYS},
   'minimum_observed_training_rows':min(r['observed_training_rows'] for r in rs),
   'maximum_observed_training_rows':max(r['observed_training_rows'] for r in rs)})
 write_csv(base/'fit_diagnostics_by_condition.csv',summaries)
 summary={'completed_arms':180,'failed_arms':0,'response_fit_events':len(rows),
  **{key:sum(r[key] for r in rows) for key in KEYS},
  'scope':'New fixed-target arms only. Original empirical diagnostic summaries remain in the source archive. Pair/branch totals are repeated fit events, not distinct branches or independent replicates.',
  'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 (base/'FIT_DIAGNOSTICS_SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.output)
