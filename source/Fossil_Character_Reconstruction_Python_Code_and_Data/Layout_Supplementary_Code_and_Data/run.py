#!/usr/bin/env python3
"""Portable entry points for frozen historical-layout simulations and diagnostics."""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,subprocess,sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
EXPERIMENT=HERE/'layout_experiment'
RESULTS=EXPERIMENT/'simulation/results'
DISTRIBUTION=HERE/'distribution_analysis'
FIXED=HERE/'fixed_target'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj

def verify_inputs(base=RESULTS):
    base=Path(base)
    original=pd.read_csv(RESULTS/'matrix_registry.csv')
    assert len(original)==360 and original.pair_id.nunique()==180
    for r in original.itertuples(index=False):
        assert sha(base/r.masked_file)==r.masked_sha256,('mask',r.matrix_id)
        assert sha(base/'truth'/f'{r.pair_id}.npz')==r.truth_sha256,('truth',r.pair_id)
    assert len(list((base/'inputs').glob('*.npz')))==360
    assert len(list((base/'truth').glob('*.npz')))==180
    return {'truth_files':180,'mask_files':360,'all_sha256_match_original_registry':True}

def verify_fixed():
    base=FIXED/'results'
    binding=json.loads((base/'SOURCE_BINDING.json').read_text())
    for relative,expected in binding['source_files'].items():
        assert sha(EXPERIMENT/relative)==expected,relative
    fit=json.loads((base/'FIT_MANIFEST.json').read_text())
    evaluation=json.loads((base/'EVALUATION_MANIFEST.json').read_text())
    assert fit['completed_arms']==180 and not fit['failed_arms']
    assert evaluation['rows']==2160 and len(evaluation['evaluation_files'])==360
    assert sha(base/'input_manifest.json')==fit['input_manifest_sha256']
    assert sha(base/'SOURCE_BINDING.json')==fit['source_binding_sha256']
    assert sha(base/'FIT_MANIFEST.json')==evaluation['fit_manifest_sha256']
    for record in fit['records']:
        mid=record['matrix_id']
        assert sha(base/'predictions'/f'{mid}.npz')==record['prediction_sha256'],mid
        assert sha(base/'predictions'/f'{mid}.json')==record['metadata_sha256'],mid
    for mid,expected in evaluation['evaluation_files'].items():
        assert sha(base/'evaluation'/f'{mid}.npz')==expected,mid
    for name in ('input_manifest.json','baseline_manifest.json'):
        for record in json.loads((base/name).read_text()):
            assert sha(base/record['masked_file'])==record['masked_sha256'],record['matrix_id']
    for record in binding['pairs']:
        assert sha(base/'truth'/f"{record['pair_id']}.npz")==record['source_truth_sha256']
        assert sha(base/'baseline_evaluation'/f"{record['empirical_id']}.npz")==record['source_evaluation_sha256']
    audit=json.loads((base/'INDEPENDENT_SCORING_AUDIT.json').read_text())
    assert audit['status']=='pass' and audit['pairs']==180
    assert audit['fit_manifest_sha256']==sha(base/'FIT_MANIFEST.json')
    assert audit['evaluation_manifest_sha256']==sha(base/'EVALUATION_MANIFEST.json')
    return {'new_fitted_arms':180,'new_prediction_files':180,'new_model_metadata_files':180,
            'input_files':360,'truth_files':180,'evaluation_files':360,
            'all_bound_hashes_match':True,'empirical_baseline':'Saved original arrays, not refitted'}

def verify():
    pinned=json.loads((HERE/'provenance/PINNED_ORIGINAL_CODE.json').read_text())
    for record in pinned['files']:
        assert sha(EXPERIMENT/record['path'])==record['sha256'],record['path']
    config=json.loads((EXPERIMENT/'simulation/config.json').read_text())
    for panel,settings in config['panels'].items():
        for field in ('template','trees'):
            path=(EXPERIMENT/'simulation'/settings[field]).resolve()
            assert EXPERIMENT in path.parents
            assert sha(path)==settings[field+'_sha256'],(panel,field)
    checked=verify_inputs()
    recovery=json.loads((HERE/'provenance/ORIGINAL_MEMBER_RECOVERY.json').read_text())
    members={r['path']:r for r in recovery['records']}
    registry=pd.read_csv(RESULTS/'matrix_registry.csv')
    for row in registry.itertuples(index=False):
        path=RESULTS/'evaluation'/f'{row.matrix_id}.npz'
        assert sha(path)==members[f'round17/simulation/results/evaluation/{row.matrix_id}.npz']['sha256']
        with np.load(path,allow_pickle=False) as x:
            assert x['D_samples'].shape==(6,512)
            assert np.array_equal(x['D_samples'],x['L_samples']-x['old_L']-(x['M_samples']-x['old_M']))
            assert x['true_D']==x['true_L']-x['old_L']-(x['true_M']-x['old_M'])
    assert len(list((RESULTS/'evaluation').glob('*.npz')))==360
    result={'status':'PASS','scope':'input/template/source and saved-distribution integrity; not a refit',**checked,'frozen_template_and_tree_files':6,'saved_evaluation_files':360,'forecast_distributions':2160,'unchanged_original_source_files':len(pinned['files']),
            'archive_scope':'Required 360 evaluation members recovered completely; complete original ZIP not verified; original model/probability/completion files are not included.'}
    if (FIXED/'results').exists():result['fixed_target']=verify_fixed()
    if (HERE/'refit_20261006/REFIT_COMPARISON.json').exists():
        checked=subprocess.run([sys.executable,str(HERE/'reproducibility/verify_refit.py'),
                                '--input',str(HERE/'refit_20261006')],
                               check=True,capture_output=True,text=True)
        result['new_original_layout_refit']=json.loads(checked.stdout)
    if (HERE/'MANIFEST.json').exists():
        manifest=json.loads((HERE/'MANIFEST.json').read_text())
        for record in manifest['files']:
            assert sha(HERE/record['path'])==record['sha256'],record['path']
        result['package_manifest_files_checked']=len(manifest['files'])
    return result

def rescore(output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Use a fresh output directory')
    source_check=verify();output.mkdir(parents=True)
    sys.path.insert(0,str(DISTRIBUTION))
    import score_draws
    core=module('original_layout_experiment',EXPERIMENT/'simulation/experiment.py')
    assert tuple(core.SCHEMES)==score_draws.SCHEMES
    original=pd.read_csv(RESULTS/'matrix_registry.csv')
    rows=[]
    for r in original.sort_values('matrix_id').to_dict('records'):
        relative=f"simulation/results/evaluation/{r['matrix_id']}.npz"
        rows.append(dict(condition_id=r['scenario_id'],latent_id=r['pair_id'],original_matrix_id=r['matrix_id'],panel=r['panel'],q=r['q'],replicate=r['replicate'],arm=r['arm'],n_targets=r['n_targets'],npz_path=relative,npz_sha256=sha(EXPERIMENT/relative),draw_key='D_samples',truth_key='true_D',schemes_json=json.dumps(core.SCHEMES)))
    registry=output/'normalized_registry.csv';pd.DataFrame(rows).to_csv(registry,index=False)
    audit=output/'source_audit.json';audit.write_text(json.dumps({'status':'PASS','normalized_registry_sha256':sha(registry),'source_recovery_scope':source_check['archive_scope'],'source_integrity':source_check,'package_entrypoint_sha256':sha(__file__)},indent=2)+'\n')
    manifest=score_draws.run(registry,EXPERIMENT,output/'results',audit)
    subprocess.run([sys.executable,str(DISTRIBUTION/'verify_results.py'),'--source-root',str(EXPERIMENT),'--results',str(output/'results'),'--output',str(output/'results/verification.json')],check=True)
    subprocess.run([sys.executable,str(DISTRIBUTION/'endpoint_concordance.py'),'--source-root',str(EXPERIMENT),'--results',str(output/'results'),'--output',str(output/'endpoint_comparisons')],check=True)
    comparisons={}
    for name in ['matrix_distribution_scores.csv','arm_distribution_summary.csv','paired_layout_contrasts.csv','matrix_paired_layout_differences.csv','paired_method_contrasts.csv']:
        saved=pd.read_csv(DISTRIBUTION/'results'/name);current=pd.read_csv(output/'results'/name)
        # The source-array path is intentionally made package-relative.
        cols=[c for c in saved.columns if c!='source_npz']
        pd.testing.assert_frame_equal(saved[cols],current[cols],check_exact=False,atol=1e-12,rtol=1e-12)
        comparisons[name]={'rows':len(current),'agrees':True}
    result={'status':'PASS','scope':'replay of saved original D distributions; no fitting','all_saved_numeric_results_agree':True,'tables':comparisons,'entrypoint_sha256':sha(__file__)}
    (output/'REPLAY_COMPARISON.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

def generate(output):
    output=Path(output).resolve()
    subprocess.run([sys.executable,str(EXPERIMENT/'simulation/experiment.py'),'generate','--output',str(output),'--config',str(EXPERIMENT/'simulation/config.json'),'--frozen-design',str(EXPERIMENT/'ANALYSIS_PLAN.txt')],check=True)
    result={'status':'PASS','operation':'new execution of original generator; no fitting',**verify_inputs(output)}
    (output/'REPLAY_VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

def original(stage,output,workers):
    output=Path(output).resolve()
    script={'fit-original':'fit_worker.py','evaluate-original':'evaluation_parallel.py'}[stage]
    subprocess.run([sys.executable,str(EXPERIMENT/'simulation'/script),'--output',str(output),'--workers',str(workers)],check=True)

def figures(output):
    subprocess.run([sys.executable,str(DISTRIBUTION/'plot_diagnostics.py'),'--scores600',str(DISTRIBUTION/'reference600/matrix_distribution_scores.csv'),'--scores180',str(DISTRIBUTION/'results/matrix_distribution_scores.csv'),'--output',str(Path(output).resolve())],check=True)

def figure2(output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Use a fresh output directory')
    subprocess.run([sys.executable,str(HERE/'figures/figure2/plot_figure2.py'),
                    '--source-root',str(HERE),'--output-dir',str(output)],check=True)

def fixed(action,output,workers):
    command=[sys.executable,str(FIXED/'run_fixed_target.py'),action,
             '--source',str(EXPERIMENT),'--output',str(Path(output).resolve()),
             '--workers',str(workers)]
    if action=='prepare':command.extend(['--replayed-inputs',str(RESULTS)])
    subprocess.run(command,check=True)

def audit_fixed(output,report,workers):
    subprocess.run([sys.executable,str(FIXED/'audit_fixed_target.py'),
                    '--output',str(Path(output).resolve()),'--report',str(Path(report).resolve()),
                    '--workers',str(workers)],check=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('verify',help='Check all input/truth/draw hashes without writing to this package')
    for cmd in ('rescore','generate','figures','figure2'):
        p=sub.add_parser(cmd);p.add_argument('--output',type=Path,required=True)
    for cmd in ('fit-original','evaluate-original'):
        p=sub.add_parser(cmd);p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=4)
    for cmd in ('prepare-fixed','fit-fixed','evaluate-fixed'):
        p=sub.add_parser(cmd);p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=4)
    p=sub.add_parser('audit-fixed');p.add_argument('--output',type=Path,required=True);p.add_argument('--report',type=Path,required=True);p.add_argument('--workers',type=int,default=4)
    a=parser.parse_args()
    if a.command=='verify':print(json.dumps(verify(),indent=2))
    elif a.command=='rescore':rescore(a.output)
    elif a.command=='generate':generate(a.output)
    elif a.command=='figures':figures(a.output)
    elif a.command=='figure2':figure2(a.output)
    elif a.command in ('prepare-fixed','fit-fixed','evaluate-fixed'):fixed(a.command.split('-')[0],a.output,a.workers)
    elif a.command=='audit-fixed':audit_fixed(a.output,a.report,a.workers)
    else:original(a.command,a.output,a.workers)
