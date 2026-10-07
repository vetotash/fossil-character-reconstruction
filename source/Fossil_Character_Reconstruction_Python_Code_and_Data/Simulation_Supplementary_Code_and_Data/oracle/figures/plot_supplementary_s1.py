#!/usr/bin/env python3
"""Plot every frozen R15 scenario from audited summaries, without model selection."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
METHODS = ('oracle_joint', 'product_marginals', 'sharpened_product')
COLORS = ('#333333', '#009E73', '#D55E00')
LABELS = ('Oracle joint', 'Product marginals', 'Sharpened product')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def rows(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))

def passed(path):
    obj = json.loads(Path(path).read_text())
    if obj.get('status') != 'PASS':
        raise RuntimeError(f'Final PASS required: {path}')
    return obj

def style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#999999')
    ax.tick_params(length=2.5, color='#777777')
    ax.grid(axis='x', color='#e8e8e8', linewidth=.6)
    ax.set_axisbelow(True)

def key(row, topology=False):
    return (float(row['q']), int(row['k_missing' if topology else 'missing_per_character']),
            row['mask_regime' if topology else 'mask_mode'])

def label(row, topology=False):
    q, k, mode = key(row, topology)
    return f'{q:g} | {k} | {mode}'

def coverage_page(data):
    fig, axes = plt.subplots(2, 2, figsize=(11.7, 8.3), sharex=True)
    for r, shape in enumerate(('balanced', 'caterpillar')):
        for c, n in enumerate((6, 10)):
            ax = axes[r, c]
            panel = [x for x in data if int(x['n_tips']) == n and x['shape'] == shape]
            base = sorted([x for x in panel if x['method'] == METHODS[0]], key=key)
            if len(base) != 12:
                raise AssertionError('Expected 12 scenarios per coverage panel')
            for j, method in enumerate(METHODS):
                lookup = {x['scenario_id']: x for x in panel if x['method'] == method}
                vals = [float(lookup[x['scenario_id']]['oracle_conditional_interval_mass_mean']) for x in base]
                ax.scatter(vals, np.arange(12) + (j - 1) * .19, color=COLORS[j], s=20,
                           edgecolor='white', linewidth=.3, zorder=3)
            for boundary in (3.5, 7.5):
                ax.axhline(boundary, color='#dddddd', linewidth=.65)
            ax.axvline(.95, color='#555555', linestyle=':', linewidth=1)
            ax.set_yticks(range(12), [label(x) for x in base], fontsize=8)
            ax.invert_yaxis()
            ax.set_xlim(-.025, 1.025)
            ax.set_xticks(np.linspace(0, 1, 5))
            ax.xaxis.set_major_formatter(PercentFormatter(1))
            ax.set_title(f'{chr(65 + 2*r + c)}  {n} tips, {shape}', loc='left', fontsize=11, fontweight='bold')
            ax.set_ylabel('q | hidden tips per character | masking', fontsize=8.5)
            if r == 1:
                ax.set_xlabel('Mean true conditional mass of each predicted interval', fontsize=9)
            style(ax)
    fig.suptitle('Supplementary Figure S1. Controlled conditional prediction', x=.04, y=.985,
                 ha='left', fontsize=14, fontweight='bold')
    fig.text(.04, .94, 'All 48 scenarios. Each point averages exact conditional interval mass over 100 independently generated matrices.', fontsize=9)
    handles = [Line2D([], [], marker='o', linestyle='', color=c, label=l) for c, l in zip(COLORS, LABELS)]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.53, .042), ncol=3, frameon=False, fontsize=9)
    fig.text(.04, .019, 'Dotted line: 95%. The oracle interval has at least 95% conditional mass by construction; discreteness can increase it.', fontsize=8.5)
    fig.subplots_adjust(left=.17, right=.975, top=.88, bottom=.15, wspace=.60, hspace=.32)
    return fig

def topology_page(data):
    fig, axes = plt.subplots(1, 2, figsize=(11.7, 8.3), sharex=True)
    all_delta = []
    for shape in ('balanced', 'caterpillar'):
        for row in [x for x in data if x['shape'] == shape]:
            full = float(row['full_mean_normalized_rf_mean'])
            all_delta.extend([float(row[col]) - full for col in
                              ('source_mean_normalized_rf_mean', 'map_mean_normalized_rf_mean', 'draw_mean_normalized_rf_mean')])
    bound_low, bound_high = min(all_delta + [0]), max(all_delta + [0])
    pad = max(.025, .08 * (bound_high - bound_low))
    for c, shape in enumerate(('balanced', 'caterpillar')):
        ax = axes[c]
        panel = [x for x in data if x['shape'] == shape]
        base = sorted([x for x in panel if x['method'] == METHODS[0]], key=lambda x:key(x, True))
        if len(base) != 12:
            raise AssertionError('Expected 12 scenarios per topology panel')
        y = np.arange(12)
        for col, offset, marker, color in [('source_mean_normalized_rf_mean', -.28, 's', '#777777'),
                                           ('map_mean_normalized_rf_mean', -.14, '^', '#111111')]:
            vals = [float(x[col]) - float(x['full_mean_normalized_rf_mean']) for x in base]
            ax.scatter(vals, y + offset, marker=marker, color=color, s=22, zorder=3)
        for j, method in enumerate(METHODS):
            lookup = {x['scenario_id']: x for x in panel if x['method'] == method}
            vals = [float(lookup[x['scenario_id']]['draw_mean_normalized_rf_mean']) - float(x['full_mean_normalized_rf_mean']) for x in base]
            ax.scatter(vals, y + j*.14, color=COLORS[j], s=22, edgecolor='white', linewidth=.3, zorder=3)
        for boundary in (3.5, 7.5):
            ax.axhline(boundary, color='#dddddd', linewidth=.65)
        ax.axvline(0, color='#555555', linestyle=':', linewidth=1)
        ax.set_yticks(range(12), [label(x, True) for x in base], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlim(bound_low - pad, bound_high + pad)
        ax.set_title(f'{chr(69+c)}  6 tips, {shape}', loc='left', fontsize=12, fontweight='bold')
        ax.set_xlabel('Mean normalized RF minus full truth baseline', fontsize=10)
        ax.set_ylabel('q | hidden tips per character | masking', fontsize=9)
        style(ax)
    fig.suptitle('Supplementary Figure S1. Topology relative to complete data', x=.04, y=.985,
                 ha='left', fontsize=14, fontweight='bold')
    fig.text(.04, .935, 'All 24 six-tip scenarios. Every tied most-parsimonious tree is retained among all 105 labelled binary topologies.', fontsize=9)
    fig.text(.04, .905, 'Positive differences indicate greater mean distance from the generating topology than the full truth matrix yields.', fontsize=9)
    handles = [Line2D([], [], marker='s', linestyle='', color='#777777', label='Partial background'),
               Line2D([], [], marker='^', linestyle='', color='#111111', label='Common marginal MAP')]
    handles += [Line2D([], [], marker='o', linestyle='', color=c, label=l) for c,l in zip(COLORS,LABELS)]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.51, .078), ncol=3, frameon=False, fontsize=9)
    fig.text(.04, .052, 'RF is averaged over all tied optimal trees and divided by 6. For each scheme, 64 draws are first averaged within each matrix.', fontsize=8.5)
    fig.text(.04, .032, 'Points then average paired differences over 100 independent matrices. Draws are not independent datasets.', fontsize=8.5)
    fig.text(.04, .013, 'All predictions condition on the known generating tree. Negative differences can reflect this extra information; full truth need not recover that tree.', fontsize=8.5)
    fig.subplots_adjust(left=.165, right=.975, top=.85, bottom=.19, wspace=.62)
    return fig

def main(a):
    sim = ROOT / 'simulation/results/scenario_summary.csv'
    topo = ROOT / 'topology/results/topology_scenario_summary.csv'
    for gate in (ROOT/'simulation/results/RUN_CHECKS.json', ROOT/'topology/results/TOPOLOGY_CHECKS.json',
                 a.simulation_audit, a.topology_audit):
        passed(gate)
    sd, td = rows(sim), rows(topo)
    assert len(sd) == 144 and len(td) == 72
    assert len({x['scenario_id'] for x in sd}) == 48 and len({x['scenario_id'] for x in td}) == 24
    assert all(set(x['method'] for x in sd if x['scenario_id'] == sid) == set(METHODS) for sid in {x['scenario_id'] for x in sd})
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'ps.fonttype':42})
    outputs=[]
    pdf = HERE/'supplementary_figure_s1.pdf'
    with PdfPages(pdf, metadata={'Title':'Supplementary Figure S1','Author':'','Subject':'Controlled conditional prediction and exhaustive six-tip topology'}) as handle:
        for page, make, data in ((1, coverage_page, sd), (2, topology_page, td)):
            fig=make(data)
            handle.savefig(fig)
            png=HERE/f'supplementary_figure_s1_page{page}.png'
            fig.savefig(png, dpi=200, facecolor='white')
            outputs.append(png)
            plt.close(fig)
    outputs.append(pdf)
    record={'status':'NUMERICAL_INPUT_CHECK_PASS_VISUAL_REVIEW_PENDING','all_simulation_scenarios':48,
            'all_topology_scenarios':24,'replication_unit':'independently generated matrix','matrices_per_scenario':100,
            'error_bars':'none','input_hashes':{str(p.relative_to(ROOT)):sha(p) for p in (sim,topo)},
            'audit_hashes':{str(p):sha(p) for p in (a.simulation_audit,a.topology_audit)},
            'outputs':{p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in outputs},
            'script_sha256':sha(__file__)}
    (HERE/'FIGURE_CHECKS.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--simulation-audit',type=Path,required=True)
    p.add_argument('--topology-audit',type=Path,required=True)
    main(p.parse_args())
