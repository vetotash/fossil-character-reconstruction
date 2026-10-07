"""Replay character summaries after extraction without changing frozen files.

The original script's absolute freeze keys are resolved to this extracted root.
A mapped copy of the freeze is written only inside a NEW replay output folder.
No original source, manifest, probability, or native-score file is modified.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
ORIGINAL=HERE/'character_robustness'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main(output):
    freeze=json.loads((ORIGINAL/'FREEZE.json').read_text());mapped={};mapping=[]
    for old,meta in freeze['inputs'].items():
        parts=Path(old).parts
        anchors=[i for i,p in enumerate(parts) if p in {'new_project_search','morphology_workspace'}]
        if len(anchors)!=1:raise ValueError(f'Unsupported archived path: {old}')
        relative=Path(*parts[anchors[0]:]);current=ROOT/relative
        assert current.is_file() and sha(current)==meta['sha256'],str(current)
        mapped[str(current)]=meta
        mapping.append({'archived_absolute_path':old,'current_root_relative_path':relative.as_posix(),'sha256':meta['sha256']})
    output=output.resolve()
    if output.exists():raise FileExistsError('Use a new replay directory: '+str(output))
    output.mkdir(parents=True)
    (output/'PROTOCOL.txt').write_bytes((ORIGINAL/'PROTOCOL.txt').read_bytes())
    replay_freeze={**freeze,'inputs':mapped,'runtime_path_mapping_only':True,
        'original_freeze_sha256':sha(ORIGINAL/'FREEZE.json')}
    (output/'FREEZE.json').write_text(json.dumps(replay_freeze,indent=2)+'\n')
    spec=importlib.util.spec_from_file_location('frozen_character_robustness',ORIGINAL/'analyze_character_robustness.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.HERE=output
    # Its already resolved R8 variable continues to point into the extracted archive.
    assert module.R8.resolve()==(ROOT/'new_project_search/round8').resolve()
    module.main()
    outputs=['full_summary.csv','per_character.csv','leave_one_character_out.csv','adversarial_deletions.csv',
      'mean_direction_fragility.csv','full_residual_draws.csv','biological_character_index.csv',
      'biological_across_reference_index.csv','full_total_validation.csv','robustness_summary.csv']
    checks={name:sha(output/name)==sha(ORIGINAL/name) for name in outputs}
    assert all(checks.values()),checks
    result={'frozen_source_unchanged':True,'original_analysis_script_sha256':sha(ORIGINAL/'analyze_character_robustness.py'),
      'wrapper_sha256':sha(Path(__file__)),'all_ten_csv_outputs_byte_identical':True,'checks':checks,'path_mapping':mapping,
      'no_new_fits_draws_native_runs':True,'output':str(output)}
    (output/'REPLAY_CHECK.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'replay_verified':True,'output':str(output)}))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args().output)
