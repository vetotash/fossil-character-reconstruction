"""Static, descriptive paired probability and compatibility diagnostic figure."""
from pathlib import Path
import os
HERE=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(HERE/'plot_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

MODELS=['mode','knn5','logistic','extra_trees','nj_mk','nj_mk_joint']
LABELS=['Frequency','kNN','Logistic','Extra Trees','Mk marginal','Mk joint']
HISTORIES=[('tetrapods','Early tetrapods',349),('dinosaurs','Dinosaurs',1333),('ichthyosaurs','Ichthyosaurs',192)]
REGIMES=['archived_shared_training','full_source_training']
COLORS={'archived_shared_training':'#606060','full_source_training':'#BB5B2B'}
MARKERS={'archived_shared_training':'o','full_source_training':'s'}

def main():
    data=pd.read_csv(HERE/'probability_and_D_diagnostics.csv')
    assert len(data)==144 and set(data.training_regime)==set(REGIMES)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.titlesize':9,
        'axes.labelsize':8.5,'xtick.labelsize':8,'ytick.labelsize':8,'pdf.fonttype':42,
        'ps.fonttype':42,'axes.linewidth':.6,'savefig.dpi':350})
    fig,axs=plt.subplots(3,2,figsize=(7.15,8.6),gridspec_kw={'width_ratios':[1,1.1]})
    for ri,(history,title,n) in enumerate(HISTORIES):
        h=data[data.history.eq(history)]
        assert len(h)==(60 if history!='dinosaurs' else 24)
        for ci,ax in enumerate(axs[ri]):
            ax.set_ylim(5.6,-.6);ax.set_yticks(range(6),LABELS)
            ax.grid(axis='x',color='#DCDCDC',lw=.5,zorder=0)
            ax.spines[['top','right']].set_visible(False)
            ax.tick_params(length=3,width=.6)
            ax.set_title(f'{chr(65+ri*2+ci)}  {title} ({n} targets)',loc='left',fontweight='bold',pad=10)
        left,right=axs[ri]
        left.set_xlim(0,1);left.set_xticks([0,.25,.5,.75,1])
        left.set_xlabel('Mean normalized entropy')
        right.set_xlabel('Mean D minus cellwise MAP D')
        # Per-history scales avoid treating a target-normalized value as a
        # common effect size across nonhomologous matrices.
        xmax=float(h.mean_D_minus_D_MAP.max())
        xmin=min(0.,float(h.mean_D_minus_D_MAP.min()))
        right.set_xlim(xmin*1.1,xmax*1.1)
        for mi,model in enumerate(MODELS):
            g=h[h.model.eq(model)]
            references=sorted(g.reference.unique())
            offsets=np.linspace(-.17,.17,len(references)) if len(references)>1 else np.array([0.])
            o=g[g.training_regime.eq(REGIMES[0])].set_index('reference')
            f=g[g.training_regime.eq(REGIMES[1])].set_index('reference')
            assert o.mean_normalized_entropy.nunique()==1 and f.mean_normalized_entropy.nunique()==1
            xo=float(o.mean_normalized_entropy.iloc[0]);xf=float(f.mean_normalized_entropy.iloc[0])
            left.plot([xo,xf],[mi,mi],color='#8A8A8A',lw=.8,zorder=2)
            for regime,x in zip(REGIMES,[xo,xf]):
                left.scatter([x],[mi],s=27,marker=MARKERS[regime],facecolors='white' if regime==REGIMES[0] else COLORS[regime],edgecolors=COLORS[regime],linewidths=.9,zorder=3)
            for reference,off in zip(references,offsets):
                xo=float(o.loc[reference,'mean_D_minus_D_MAP']);xf=float(f.loc[reference,'mean_D_minus_D_MAP'])
                right.plot([xo,xf],[mi+off,mi+off],color='#A0A0A0',lw=.65,zorder=2)
                for regime,x in zip(REGIMES,[xo,xf]):
                    right.scatter([x],[mi+off],s=16,marker=MARKERS[regime],facecolors='white' if regime==REGIMES[0] else COLORS[regime],edgecolors=COLORS[regime],linewidths=.7,zorder=3)
    handles=[Line2D([0],[0],marker='o',lw=0,markerfacecolor='white',markeredgecolor=COLORS[REGIMES[0]],label='Archived shared training'),
             Line2D([0],[0],marker='s',lw=0,color=COLORS[REGIMES[1]],label='All eligible source scores')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.51,.991),ncol=2,frameon=False,fontsize=8.5)
    fig.subplots_adjust(left=.14,right=.985,top=.93,bottom=.13,hspace=.57,wspace=.64)
    fig.text(.14,.02,'Within each row, models share the same targets. Right panels show every reference tree.\nConnecting lines mark paired training conditions, not uncertainty intervals or causal effects.',fontsize=8,color='#4A4A4A',ha='left',va='bottom')
    fig.savefig(HERE/'figure_probability_concentration.png',dpi=350)
    fig.savefig(HERE/'figure_probability_concentration.pdf')
    plt.close(fig)

if __name__=='__main__':main()
