from pathlib import Path
import json,hashlib
import pandas as pd
HERE=Path(__file__).resolve().parent
rows=[]
for lineage in ['tetrapods','dinosaurs']:
 for condition in ['shared','full_source']:
  p=HERE/lineage/condition
  summary=json.loads((p/'FIT_COMPLETED.json').read_text())
  diag=json.loads((p/'fit_execution_diagnostics.json').read_text())
  logs=[d for r in diag for d in r['classifiers'] if d['model']=='logistic' and d['fit_ran']]
  nit=[n for d in logs for n in d['n_iter']]
  rows.append(dict(lineage=lineage,condition=condition,characters=summary['characters'],
    logistic_actual_fits=len(logs),logistic_constant_fallbacks=len(diag)-len(logs),
    logistic_max_n_iter=max(nit),logistic_warnings=len(summary['logistic_convergence_warning_characters']),
    logistic_maxiter_reached=len(summary['logistic_max_iter_characters']),
    mk_failed=len(summary['mk_optimizer_unsuccessful_characters']),mk_boundary=len(summary['mk_rate_boundary_characters']),
    prediction_rows=summary['prediction_rows'],future_cells=summary['future_cells'],
    max_archived_probability_difference=summary['archive_comparison']['max_probability_abs_difference'],
    seconds=summary['seconds']))
pd.DataFrame(rows).to_csv(HERE/'diagnostic_summary.csv',index=False)
assert sum(r['logistic_warnings']+r['logistic_maxiter_reached']+r['mk_failed'] for r in rows)==0
out={'status':'PASS','all_four_fits_completed':True,'all_archive_probabilities_exact':all(r['max_archived_probability_difference']==0 for r in rows),
     'total_response_condition_fits':sum(r['characters'] for r in rows),'total_prediction_rows':sum(r['prediction_rows'] for r in rows),
     'further_logistic_iteration_sensitivity_triggered':False,
     'reason':'No logistic convergence warning or max_iter reach in any actual fit',
     'historical_scope':'Both shared and full-source historical predictions existed previously; this rerun recovers exact probabilities and diagnostics. Full-source downstream propagation is conducted separately.',
     'rows':rows}
(HERE/'REFIT_CHECKS.json').write_text(json.dumps(out,indent=2))
manifest=[]
for p in sorted(HERE.rglob('*')):
 if p.is_file() and '__pycache__' not in p.parts and p.name!='OUTPUT_MANIFEST.json':
  manifest.append({'path':str(p.relative_to(HERE)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
(HERE/'OUTPUT_MANIFEST.json').write_text(json.dumps(manifest,indent=2))
print(pd.DataFrame(rows).to_string(index=False));print(json.dumps({k:v for k,v in out.items() if k!='rows'},indent=2))
