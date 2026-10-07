#!/usr/bin/env python3
"""Publication diagnostics from scored distributions; separate experiments and rates.

Makes a multipage vector PDF plus one SVG and 300-dpi PNG per page. Every page
retains all six schemes and all conditions at one rate. No pooled ranking.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from validated_crps import aggregate

SCHEMES = ['mode', 'knn5', 'logistic', 'extra_trees', 'nj_mk', 'nj_mk_joint']
LABELS = {'mode':'Frequency', 'knn5':'kNN', 'logistic':'Logistic', 'extra_trees':'Extra Trees',
          'nj_mk':'Mk marginal', 'nj_mk_joint':'Mk joint'}
COLORS = ['#555555', '#333333', '#E69F00', '#009E73', '#CC79A7', '#882255']
MARKERS = ['o', 's', '^', 'D', 'v', 'P']
METRICS = ['crps_empirical', 'signed_bias', 'width95', 'truth_in_95']
TITLES = ['A  Empirical CRPS', 'B  Signed mean D error', 'C  95% range width', 'D  Truth inclusion']
AXLABELS = ['Steps; lower is better', 'Steps; zero is unbiased', 'Steps; descriptive', 'Fraction of matrices']


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare600(path):
    frame = pd.read_csv(path)
    if frame.matrix_id.nunique() != 600 or len(frame) != 3600:
        raise ValueError('Expected full 600-matrix score table')
    summary = aggregate(frame, ['scenario_id', 'q', 'missing_per_response', 'mask_mode', 'scheme'], METRICS)
    summary['experiment'] = '600-matrix benchmark'
    summary['condition_order'] = summary.scenario_id.str.replace('s', '').astype(int)
    summary['condition_label'] = summary.apply(lambda r:f"k = {r.missing_per_response}\n{r.mask_mode.replace('_', '-').replace('statebiased', 'state-biased')}", axis=1)
    return summary


def prepare180(path):
    frame = pd.read_csv(path)
    if frame.latent_id.nunique() != 180 or len(frame) != 2160:
        raise ValueError('Expected full 180-latent-matrix, two-arm score table')
    summary = aggregate(frame, ['condition_id', 'panel', 'q', 'arm', 'scheme'], METRICS)
    summary['experiment'] = '180-matrix paired layout experiment'
    # Stable source-derived panel order, not sorted by outcomes.
    names = list(dict.fromkeys(frame.sort_values('condition_id').panel))
    ranks = {name:i for i,name in enumerate(names)}
    summary['condition_order'] = [ranks[p]*2 + int(a == 'permuted') for p,a in zip(summary.panel,summary.arm)]
    def panel_name(name):
        lower = name.lower()
        if 'tetra' in lower: return 'Early tetrapods'
        if 'dino' in lower: return 'Dinosaurs'
        if 'ichth' in lower: return 'Ichthyosaurs'
        return name
    summary['condition_label'] = [f"{panel_name(p)}\n{'Empirical' if a == 'empirical' else 'Randomized'}" for p,a in zip(summary.panel,summary.arm)]
    return summary


def draw_page(summary, experiment, q):
    data = summary[summary.q == q].copy()
    condition_keys = data[['condition_order', 'condition_label']].drop_duplicates().sort_values('condition_order')
    if len(condition_keys) != 6:
        raise ValueError('Expected six displayed condition/arm rows per rate')
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':9, 'axes.titlesize':10,
                         'axes.labelsize':9, 'pdf.fonttype':42, 'ps.fonttype':42,
                         'svg.fonttype':'none', 'axes.spines.top':False, 'axes.spines.right':False})
    fig, axes = plt.subplots(1,4,figsize=(13.0,9.2),sharey=True)
    fig.subplots_adjust(left=.172,right=.982,bottom=.14,top=.79,wspace=.24)
    offsets = np.linspace(-.32,.32,6)
    for j,metric in enumerate(METRICS):
        ax = axes[j]
        for y,(order, label) in enumerate(condition_keys.itertuples(index=False,name=None)):
            if y%2 == 0: ax.axhspan(y-.48,y+.48,color='#F3F3F3',zorder=0)
            rows = data[data.condition_order == order].set_index('scheme')
            if len(rows) != 6 or set(rows.index) != set(SCHEMES):
                raise ValueError('Incomplete scheme set for plotted condition')
            for k,scheme in enumerate(SCHEMES):
                row = rows.loc[scheme]
                mean = float(row[metric+'_mean'])
                if metric == 'truth_in_95':
                    lo,hi = row.inclusion_Wilson_low,row.inclusion_Wilson_high
                    err = np.maximum(0,[[mean-lo],[hi-mean]])
                else:
                    err = float(row[metric+'_MCSE'])
                ax.errorbar(mean,y+offsets[k],xerr=err,fmt=MARKERS[k],ms=4.2,
                            color=COLORS[k],mec=COLORS[k],mew=.65,elinewidth=1.,capsize=2,zorder=3)
        ax.set_title(TITLES[j],loc='left',fontweight='bold',pad=13)
        ax.set_xlabel(AXLABELS[j],labelpad=9)
        ax.grid(axis='x',color='#DBDBDB',linewidth=.6,zorder=1)
        ax.tick_params(axis='y',length=0)
        ax.spines['left'].set_visible(False)
        ax.set_ylim(5.65,-.65)
        if metric == 'signed_bias': ax.axvline(0,color='#777777',linestyle='--',linewidth=1)
        elif metric == 'truth_in_95':
            ax.axvline(.95,color='#777777',linestyle=':',linewidth=1)
            ax.set_xlim(-.03,1.03)
            ax.set_xticks([0,.25,.5,.75,1])
        else: ax.set_xlim(left=0)
    axes[0].set_yticks(range(6),condition_keys.condition_label)
    axes[0].tick_params(axis='y',pad=11,labelsize=10)
    for ax in axes[1:]: ax.tick_params(labelleft=False)
    n = int(data.n_matrices.iloc[0])
    fig.suptitle(f'{experiment}: q = {q:g}',x=.172,y=.956,ha='left',fontsize=17,fontweight='bold')
    subtitle = ('Eight fully observed anchors; 50 independent matrices per condition' if experiment.startswith('600')
                else '30 independent latent matrices per panel and rate; each arm uses its own target-only truth')
    fig.text(.172,.913,subtitle,ha='left',fontsize=10.2)
    handles=[Line2D([],[],color=c,marker=m,linestyle='none',markersize=5,label=LABELS[s])
             for s,c,m in zip(SCHEMES,COLORS,MARKERS)]
    fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.162,.885),ncol=6,
               frameon=False,columnspacing=1.35,handletextpad=.45,fontsize=9.8)
    fig.text(.172,.068,'A-C: means with one Monte Carlo standard error across independent matrices.  D: inclusion fractions with Wilson 95% intervals.',fontsize=9)
    fig.text(.172,.044,'All forecasts use 512 saved draws. The 95% inclusion reference is descriptive; no pooled ranking or cross-experiment average is shown.',fontsize=9)
    return fig


def run(scores600, scores180, output):
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a new/empty figure output directory')
    output.mkdir(parents=True,exist_ok=True)
    inputs = {'scores600':{'path':str(Path(scores600).resolve()),'sha256':sha256(scores600)}}
    frames = [prepare600(scores600)]
    if scores180:
        inputs['scores180']={'path':str(Path(scores180).resolve()),'sha256':sha256(scores180)}
        frames.append(prepare180(scores180))
    tables = pd.concat(frames,ignore_index=True)
    tables.to_csv(output/'figure_input_table.csv',index=False)
    pages=[]
    pdf_path=output/'distribution_diagnostics.pdf'
    with PdfPages(pdf_path,metadata={'Title':'Distributional prediction diagnostics','Author':'Analysis workflow','Subject':'Frozen simulation forecasts: CRPS, bias, range width and truth inclusion'}) as pdf:
        for frame in frames:
            experiment=frame.experiment.iloc[0]
            for q in [.03,.3]:
                fig=draw_page(frame,experiment,q)
                prefix=('benchmark600' if experiment.startswith('600') else 'layout180')+f'_q{str(q).replace(".","p")}'
                pdf.savefig(fig)
                fig.savefig(output/f'{prefix}.svg',format='svg')
                fig.savefig(output/f'{prefix}.png',dpi=300)
                plt.close(fig)
                pages.append({'name':prefix,'experiment':experiment,'q':q})
    manifest={'status':'CREATED_PENDING_VISUAL_REVIEW','inputs':inputs,'pages':pages,
              'source_180_available':bool(scores180),'plot_script_sha256':sha256(__file__),
              'output_hashes':{f.name:sha256(f) for f in sorted(output.iterdir()) if f.is_file()},
              'scope_note':'Distinct experiments retained on separate pages, with no fabricated or summary-derived 180-matrix CRPS.'}
    (output/'figure_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'pdf':str(pdf_path),'pages':len(pages),'contains180':bool(scores180)},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scores600',required=True,type=Path)
    p.add_argument('--scores180',type=Path)
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();run(a.scores600,a.scores180,a.output)
