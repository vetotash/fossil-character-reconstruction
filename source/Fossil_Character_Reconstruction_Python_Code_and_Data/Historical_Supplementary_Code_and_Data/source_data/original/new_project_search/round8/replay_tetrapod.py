"""Portable entry point for the byte-identical frozen tetrapod script.

Only exact absolute keys in PROBABILITY_FREEZE.json are remapped for the sha()
lookup. The original source file, inputs, seeds and executed functions are kept.
Default checks inputs and recreates draws in memory without scoring or writing.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def load():
    script = HERE/'run_probability_propagation.py'
    gate = json.loads((HERE/'IMPLEMENTATION_FREEZE.json').read_text())
    expected = gate['files']['new_project_search/round8/run_probability_propagation.py']
    assert hashlib.sha256(script.read_bytes()).hexdigest()==expected, 'Frozen code hash changed'
    spec = importlib.util.spec_from_file_location('frozen_tetrapod_propagation', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    frozen = json.loads((HERE/'PROBABILITY_FREEZE.json').read_text())['files']
    aliases = {}
    for name in frozen:
        old = Path(name)
        assert old.is_absolute(), name
        parts = old.parts
        candidates = [i for i,x in enumerate(parts) if x in ('new_project_search','morphology_workspace')]
        assert len(candidates)==1, name
        aliases[str(old)] = ROOT/Path(*parts[candidates[0]:])
    original_sha = module.sha
    def portable_hash(path):
        return original_sha(aliases.get(str(path), path))
    module.sha = portable_hash
    return module


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--rerun', action='store_true', help='Run PAUP after manually backing up the existing result directory')
    a.add_argument('--batch-size', type=int, default=200)
    args = a.parse_args()
    module = load()
    if args.rerun:
        if module.OUT.exists():
            raise RuntimeError('Rename probability_propagation to a backup first; existing evidence will not be overwritten')
        module.main(args.batch_size)
    else:
        sets, chars = module.initialize()
        print(json.dumps({'frozen_code_hash_valid': True, 'absolute_hash_paths_remapped': True,
                          'characters': len(chars), 'cells': {k: len(v['cells']) for k,v in sets.items()},
                          'native_calls': 0, 'files_written': 0}))


if __name__=='__main__':
    main()
