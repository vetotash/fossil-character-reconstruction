#!/usr/bin/env python3
"""Run the simulation code and regenerate supplementary outputs."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parent

def run(script,*arguments):
    env=dict(os.environ)
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
        env[name]='1'
    subprocess.run([sys.executable,str(ROOT/script),*map(str,arguments)],cwd=ROOT,env=env,check=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    subs=parser.add_subparsers(dest='action',required=True)
    verify=subs.add_parser('verify',help='Validate all 600 saved matrices, predictions, completions and summaries.')
    verify.add_argument('--workers',type=int,default=2)
    verify.add_argument('--refit',action='store_true',help='Additionally refit the predetermined response columns.')
    subs.add_parser('figures',help='Regenerate Supplementary Figure S2 from saved results.')
    subs.add_parser('self-test',help='Check binary parsimony against exhaustive enumeration.')
    reproduce=subs.add_parser('reproduce',help='Generate, fit and evaluate all 600 matrices in a new directory.')
    reproduce.add_argument('--output',type=Path,required=True)
    reproduce.add_argument('--workers',type=int,default=2)
    args=parser.parse_args()
    if hasattr(args,'workers') and args.workers<1:parser.error('--workers must be positive')
    if args.action=='verify':
        flags=[] if args.refit else ['--no-refit']
        run('validation/validate_results.py','--root',ROOT,'--workers',args.workers,*flags)
    elif args.action=='figures':
        run('figures/plot_supplementary_s2.py')
    elif args.action=='self-test':
        with tempfile.TemporaryDirectory(prefix='character_simulation_test_') as tmp:
            run('simulation/experiment.py','self-test','--output',Path(tmp)/'checks')
    else:
        output=args.output.expanduser().resolve()
        if output.exists():parser.error('--output must be a new directory')
        output.parent.mkdir(parents=True,exist_ok=True)
        run('simulation/experiment.py','generate','--output',output,'--frozen-design',ROOT/'simulation/design.json')
        run('simulation/fit_worker.py','--output',output,'--workers',args.workers)
        run('simulation/experiment.py','evaluate','--output',output)
        run('simulation/summarize_diagnostics.py','--output',output)

if __name__=='__main__':main()
