#!/usr/bin/env python3
"""Plot all fitted-method scenarios after checking the saved numerical inputs."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
METHODS = ('mode', 'knn5', 'logistic', 'extra_trees', 'nj_mk')
SCHEMES = METHODS + ('nj_mk_joint',)
LABELS = ('Frequency', 'kNN', 'Logistic', 'Extra Trees', 'Mk marginal', 'Mk joint')
COLORS = ('#777777', '#333333', '#D55E00', '#009E73', '#CC79A7', '#6E42A6')
MASK_LABELS = {'random':'random', 'structured':'structured', 'statebiased':'state biased'}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def rows(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))

def wilson(k, n):
    z = 1.959963984540054
    p = k/n
    den = 1+z*z/n
    center = (p+z*z/(2*n))/den
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return center-half, center+half

def check_manifests(base):
    fit_path=base/'FIT_MANIFEST.json'
    evaluation=json.loads((base/'EVALUATION_MANIFEST.json').read_text())
    fit=json.loads(fit_path.read_text())
    inputs=json.loads((base/'input_manifest.json').read_text())
    assert fit['completed_matrices']==evaluation['completed_matrices']==600
    assert not fit['failed_matrices']
    assert evaluation['rows']==3600 and evaluation['draws_per_scheme_matrix']==512
    assert tuple(evaluation['schemes'])==SCHEMES
    assert evaluation['fit_manifest_sha256']==sha(fit_path)
    assert fit['input_manifest_sha256']==sha(base/'input_manifest.json')
    assert len(inputs)==len(fit['records'])==600
    assert all(record['status']=='completed' for record in fit['records'])
    ids={entry['matrix_id'] for entry in inputs}
    assert len(ids)==600 and ids=={record['matrix_id'] for record in fit['records']}
    return {entry['matrix_id']:f"s{entry['scenario_ordinal']:02d}" for entry in inputs}

def check_and_table(base):
    input_scenarios=check_manifests(base)
    data = rows(base/'matrix_results.csv')
    official = rows(base/'scenario_summary.csv')
    assert len(data) == 3600 and len(official) == 72
    assert {r['matrix_id'] for r in data}==set(input_scenarios)
    assert all(r['scenario_id']==input_scenarios[r['matrix_id']] for r in data)
    assert len({(r['matrix_id'],r['scheme']) for r in data})==3600
    index = {(r['scenario_id'],r['scheme']):r for r in official}
    assert set(index)=={(f's{i:02d}',scheme) for i in range(12) for scheme in SCHEMES}
    table=[]
    for sid in [f's{i:02d}' for i in range(12)]:
        for scheme in SCHEMES:
            r = index[sid,scheme]
            group=[a for a in data if a['scenario_id']==sid and a['scheme']==scheme]
            assert len(group)==50 and len({a['matrix_id'] for a in group})==50
            assert int(r['n_matrices'])==50
            record={k:r[k] for k in ('scenario_id','q','missing_per_response','mask_mode','scheme')}
            record['n_matrices']=50
            for metric in ('accuracy','brier','mean_D_absolute_error'):
                values=np.array([float(a[metric]) for a in group])
                assert np.isfinite(values).all() and np.all(values>=0)
                if metric=='accuracy':assert np.all(values<=1)
                if metric=='brier':assert np.all(values<=2)
                mean=float(np.mean(values))
                se=float(np.std(values,ddof=1)/np.sqrt(50))
                assert np.isfinite([mean,se,float(r[metric+'_mean']),float(r[metric+'_mcse'])]).all()
                assert float(r[metric+'_mcse'])>=0
                assert np.isclose(mean,float(r[metric+'_mean']),rtol=1e-12,atol=1e-12)
                assert np.isclose(se,float(r[metric+'_mcse']),rtol=1e-12,atol=1e-12)
                record[metric+'_mean']=mean
                record[metric+'_mcse']=se
            for a in group:
                assert np.isclose(float(a['mean_D_absolute_error']),abs(float(a['mean_D'])-float(a['true_D'])),rtol=1e-12,atol=1e-12)
                assert int(a['n_draws'])==512
                lower,upper,truth,mean=map(float,(a['lower95'],a['upper95'],a['true_D'],a['mean_D']))
                assert np.isfinite([lower,upper,truth,mean]).all() and lower<=upper
                assert int(a['truth_in_interval'])==int(lower<=truth<=upper)
                assert int(a['truth_below_interval'])==int(truth<lower)
                assert int(a['truth_above_interval'])==int(truth>upper)
            hits=sum(int(a['truth_in_interval']) for a in group)
            p=hits/50
            lo,hi=wilson(hits,50)
            assert np.isclose(p,float(r['interval_coverage']),rtol=1e-12,atol=1e-12)
            assert np.isclose(lo,float(r['coverage_wilson_lower95']),rtol=1e-12,atol=1e-12)
            assert np.isclose(hi,float(r['coverage_wilson_upper95']),rtol=1e-12,atol=1e-12)
            record.update(coverage_count=hits,coverage=p,wilson_lower95=lo,wilson_upper95=hi)
            table.append(record)
    # The two Mk schemes have exactly the same single-cell predictions.
    for sid in [f's{i:02d}' for i in range(12)]:
        a=index[sid,'nj_mk']; b=index[sid,'nj_mk_joint']
        for key in ('brier_mean','brier_mcse','accuracy_mean','accuracy_mcse'):
            assert float(a[key])==float(b[key])
    path=HERE/'figure_s2_input_table.csv'
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]))
        writer.writeheader(); writer.writerows(table)
    return table,path

def style(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.spines[['left','bottom']].set_color('#999999')
    ax.tick_params(length=3,color='#888888')
    ax.grid(axis='x',color='#e8e8e8',linewidth=.6)
    ax.set_axisbelow(True)

def label(r):
    return f"{float(r['q']):g} | {int(r['missing_per_response'])} | {MASK_LABELS[r['mask_mode']]}"

def base_page(page,schemes,title,subtitle):
    fig,axes=plt.subplots(1,2,figsize=(13.6,12.0))
    fig.suptitle(title,x=.045,y=.978,ha='left',fontsize=16,fontweight='bold')
    fig.text(.045,.942,subtitle,fontsize=10.5)
    handles=[Line2D([],[],marker='o',linestyle='',color=COLORS[i],label=LABELS[i] if schemes==SCHEMES else ('Fitted Mk' if i==4 else LABELS[i]),markersize=5)
             for i in range(len(schemes))]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.52,.929),ncol=len(schemes),frameon=False,fontsize=10)
    fig.text(.045,.04,'Scenario labels: q | hidden tips per response character | masking. All 12 scenarios are retained.',fontsize=9)
    fig.text(.045,.022,'Replication unit: independently generated matrix (n = 50 per scenario). Completion draws are nested, not independent replicates.',fontsize=9)
    fig.text(.97,.022,f'{page} / 2',ha='right',fontsize=9)
    fig.subplots_adjust(left=.19,right=.975,top=.864,bottom=.145,wspace=.70)
    return fig,axes

def decorate(ax,base,letter,title,xlabel):
    positions=np.arange(12,dtype=float)*1.22
    ax.set_yticks(positions,[label(r) for r in base],fontsize=9.1)
    ax.set_ylim(positions[-1]+.62,-.62)
    for boundary in (2.5,5.5,8.5):
        ax.axhline(boundary*1.22,color='#d1d1d1',linewidth=.8)
    ax.set_title(letter+'  '+title,loc='left',fontsize=11.2,fontweight='bold',pad=11)
    ax.set_xlabel(xlabel,fontsize=10,labelpad=8)
    style(ax)
    return positions

def prediction_page(table):
    base=[r for r in table if r['scheme']==METHODS[0]]
    index={(r['scenario_id'],r['scheme']):r for r in table}
    fig,axes=base_page(1,METHODS,'Supplementary Figure S2. Fitted prediction under random simulation',
                      'Five fitted predictors. Each point summarizes hidden-state performance across 50 independent matrices.')
    for ax,metric,letter,title,xlabel in ((axes[0],'brier','A','Brier score','Mean Brier score (lower is better)'),
                                         (axes[1],'accuracy','B','State prediction','Mean accuracy')):
        positions=decorate(ax,base,letter,title,xlabel)
        largest=0
        for j,method in enumerate(METHODS):
            values=np.array([index[r['scenario_id'],method][metric+'_mean'] for r in base])
            errors=np.array([index[r['scenario_id'],method][metric+'_mcse'] for r in base])
            y=positions+(j-2)*.155
            if metric=='brier':
                ax.errorbar(values,y,xerr=errors,fmt='o',color=COLORS[j],markersize=4.1,elinewidth=.95,capsize=1.5,zorder=3)
            else:
                ax.scatter(values,y,color=COLORS[j],s=19,zorder=3)
            largest=max(largest,float(np.max(values+errors)))
        if metric=='accuracy':
            ax.set_xlim(-.025,1.025); ax.set_xticks(np.linspace(0,1,5)); ax.xaxis.set_major_formatter(PercentFormatter(1))
        else:
            ax.set_xlim(-.025,max(.25,largest*1.08))
    fig.text(.045,.063,'Brier error bars: one Monte Carlo standard error on either side. Accuracy is shown as a mean; both metrics first average within matrices.',fontsize=9)
    return fig

def downstream_page(table):
    base=[r for r in table if r['scheme']==SCHEMES[0]]
    index={(r['scenario_id'],r['scheme']):r for r in table}
    fig,axes=base_page(2,SCHEMES,'Supplementary Figure S2. Fixed-tree compatibility and interval inclusion',
                      'Six propagation schemes. Every matrix uses 512 completed draws per scheme and its own generating reference tree.')
    positions=decorate(axes[0],base,'C','Absolute discrepancy','Mean |propagated mean D - truth D| (steps)')
    decorate(axes[1],base,'D','Truth inside the central range','Inclusion proportion with Wilson 95% interval')
    largest=0
    for j,scheme in enumerate(SCHEMES):
        y=positions+(j-2.5)*.143
        selected=[index[r['scenario_id'],scheme] for r in base]
        errors=np.array([r['mean_D_absolute_error_mcse'] for r in selected])
        vals=np.array([r['mean_D_absolute_error_mean'] for r in selected])
        largest=max(largest,float(np.max(vals+errors)))
        axes[0].errorbar(vals,y,xerr=errors,fmt='o',color=COLORS[j],markersize=4.1,elinewidth=.95,capsize=1.5,zorder=3)
        vals=np.array([r['coverage'] for r in selected])
        lo=np.array([r['wilson_lower95'] for r in selected])
        hi=np.array([r['wilson_upper95'] for r in selected])
        # Protect zero-width Wilson endpoints against roundoff at p = 0 or 1.
        axes[1].errorbar(vals,y,xerr=np.maximum(0,np.stack([vals-lo,hi-vals])),fmt='o',color=COLORS[j],markersize=3.8,elinewidth=.8,capsize=1.1,zorder=3)
        for row,yy in zip(selected,y):
            axes[1].text(1.055,yy,f"{row['coverage_count']}/50",ha='left',va='center',fontsize=6.9,color=COLORS[j])
    axes[0].set_xlim(-max(.5,largest*.025),largest*1.10)
    axes[1].set_xlim(-.04,1.22)
    axes[1].set_xticks(np.linspace(0,1,5)); axes[1].xaxis.set_major_formatter(PercentFormatter(1))
    axes[1].axvline(.95,color='#555555',linestyle=':',linewidth=1)
    fig.text(.045,.063,'Panel C error bars: one Monte Carlo standard error. Panel D: inclusive empirical 2.5% and 97.5% quantiles; dotted line = 95%.',fontsize=9)
    return fig

def main(args):
    table,input_table=check_and_table(args.results)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'ps.fonttype':42})
    output=[]
    pdf=HERE/'supplementary_figure_s2.pdf'
    with PdfPages(pdf,metadata={'Title':'Supplementary Figure S2','Subject':'Fitted-method simulation in fossil character prediction','Author':''}) as handle:
        for number,make in ((1,prediction_page),(2,downstream_page)):
            fig=make(table)
            handle.savefig(fig)
            if args.png:
                png=HERE/f'supplementary_figure_s2_page{number}.png'
                fig.savefig(png,dpi=200,facecolor='white')
                output.append(png)
            plt.close(fig)
    output.append(pdf)
    for path in [input_table,*output]:print(path)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=ROOT/'simulation/results')
    parser.add_argument('--png',action='store_true',help='Also write 200 dpi PNGs for both PDF pages')
    main(parser.parse_args())
