#!/usr/bin/env python3
"""Unified Python entry point. All new outputs must be outside this archive."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PACKAGES = {
    'historical': ROOT / 'Historical_Supplementary_Code_and_Data',
    'simulation': ROOT / 'Simulation_Supplementary_Code_and_Data',
    'layout': ROOT / 'Layout_Supplementary_Code_and_Data',
}

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def integrity():
    manifest = json.loads((ROOT / 'SHA256_MANIFEST.json').read_text())
    failures = []
    for item in manifest['files']:
        path = ROOT / item['path']
        if not path.is_file() or digest(path) != item['sha256']:
            failures.append(item['path'])
    if failures:
        raise RuntimeError('Missing or changed release files: ' + ', '.join(failures))
    return {'status': 'PASS', 'files_checked': len(manifest['files']),
            'scope': 'SHA-256 file integrity; not a model refit'}

def external_output(raw):
    path = Path(raw).expanduser().resolve()
    if path == ROOT or ROOT in path.parents:
        raise ValueError('Choose an output directory outside the delivered package.')
    if path.exists():
        raise FileExistsError('Choose a new output directory: ' + str(path))
    path.mkdir(parents=True)
    return path

def invoke(script, arguments, log=None):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    command = [sys.executable, '-B', str(script), *map(str, arguments)]
    print('Running:', script.relative_to(ROOT), flush=True)
    if log is None:
        subprocess.run(command, cwd=script.parent, env=env, check=True)
    else:
        with log.open('w', encoding='utf-8') as stream:
            stream.write('New execution of supplied Python code. This is not a recovered historical log.\n')
            stream.write(json.dumps(command) + '\n')
            stream.flush()
            subprocess.run(command, cwd=script.parent, env=env, stdout=stream,
                           stderr=subprocess.STDOUT, check=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('integrity', help='Check every distributed file; standard library only.')
    verify = sub.add_parser('verify', help='Run all three existing saved-result checks; no refits.')
    verify.add_argument('--output', required=True)
    verify.add_argument('--workers', type=int, default=2)
    replay = sub.add_parser('replay', help='Recalculate historical, benchmark and layout summaries from saved arrays.')
    replay.add_argument('--output', required=True)
    replay.add_argument('--workers', type=int, default=2)
    for name in PACKAGES:
        child = sub.add_parser(name, help='Pass arguments to the original module entry point.')
        child.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command in PACKAGES:
        name = args.command
        forwarded = args.arguments
        if forwarded[:1] == ['--']:
            forwarded = forwarded[1:]
        if not forwarded:
            forwarded = ['--help']
        # Resolve user paths before switching into the original module directory.
        # Otherwise ../output would silently become an output inside this archive.
        for index, value in enumerate(forwarded[:-1]):
            if value in ('--output', '--report', '--output-dir'):
                destination = Path(forwarded[index + 1]).expanduser().resolve()
                if destination == ROOT or ROOT in destination.parents:
                    raise ValueError('Choose outputs outside the delivered package.')
                forwarded[index + 1] = str(destination)
        # Original figure and specialised analysis commands may write in place;
        # see module README before running them in a separate working copy.
        script = PACKAGES[name] / ('run.py' if name == 'layout' else 'run_analysis.py')
        invoke(script, forwarded)
        return
    checked = integrity()
    if args.command == 'integrity':
        print(json.dumps(checked, indent=2))
        return
    output = external_output(args.output)
    stages = []
    if args.command == 'verify':
        jobs = [
            ('historical', PACKAGES['historical'] / 'run_analysis.py', ['verify']),
            ('simulation', PACKAGES['simulation'] / 'run_analysis.py', ['verify', '--workers', args.workers]),
            ('layout', PACKAGES['layout'] / 'run.py', ['verify']),
        ]
    else:
        jobs = [
            ('historical', PACKAGES['historical'] / 'run_analysis.py', ['replay', '--output', output / 'historical']),
            ('benchmark', PACKAGES['simulation'] / 'distribution_analysis/analyze_distributions.py',
             ['--root', PACKAGES['simulation'], '--output', output / 'benchmark', '--workers', args.workers]),
            ('oracle', PACKAGES['simulation'] / 'oracle/validation/replay_saved_arrays.py',
             ['--output', output / 'oracle_replay.json']),
            ('layout', PACKAGES['layout'] / 'run.py', ['rescore', '--output', output / 'layout']),
        ]
    for name, script, arguments in jobs:
        invoke(script, arguments, output / (name + '.log'))
        stages.append({'stage': name, 'status': 'PASS', 'log': name + '.log'})
    report = {'status': 'PASS', 'operation': args.command, 'integrity': checked,
              'stages': stages, 'new_model_fitting': False, 'new_native_PAUP_scoring': False}
    (output / 'RUN_REPORT.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
