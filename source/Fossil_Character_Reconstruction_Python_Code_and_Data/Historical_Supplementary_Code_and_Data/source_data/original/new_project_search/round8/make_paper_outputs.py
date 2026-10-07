"""Present completed native results without fitting or selecting a method."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE=Path(__file__).resolve().parent
OUT=HERE/'deliverables'
MODELS=['mode','knn5','logistic','extra_trees','nj_mk','nj_mk_joint']
LABELS=['Frequency','kNN','Logistic','Extra Trees','Mk marginal','Mk joint']

def main():
    tet=pd.read_csv(HERE/'probability_propagation/summary.csv')
    dino=pd.read_csv(HERE/'replication/probability_summary_all_schemes.csv')
    assert len(tet)==54 and len(dino)==12
    rows=[]
    for r in tet[tet.panel=='future95'].itertuples():
        rows.append(dict(history='tetrapods',reference=r.tree,model=r.model,
            targets=349,reference_excess=r.reference_added_excess_steps,
            point_excess=r.point_added_excess_steps,mean=r.sim_mean_added_excess_steps,
            q025=r.sim_q025,q975=r.sim_q975))
    for r in dino.itertuples():
        rows.append(dict(history='dinosaurs',reference=r.reference,model=r.model,
            targets=1333,reference_excess=r.reference_excess_steps,
            point_excess=r.point_excess_steps,mean=r.sim_mean_excess_steps,
            q025=r.sim_q025,q975=r.sim_q975))
    d=pd.DataFrame(rows)
    d.to_csv(OUT/'all_reference_results.csv',index=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                         'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,4,figsize=(17,8.5),layout='constrained')
    pairs=[('tetrapods',t) for t in ['O1','O2','R1','R2','R3']]+[('dinosaurs',t) for t in ['baron','langer']]
    for idx,((hist,tree),ax) in enumerate(zip(pairs,axes.flat)):
        s=d[(d.history==hist)&(d.reference==tree)].set_index('model').loc[MODELS]
        y=np.arange(6)
        ax.errorbar(s['mean'],y,xerr=np.vstack([s['mean']-s.q025,s.q975-s['mean']]),
                    fmt='o',color='#166b93',capsize=3,markersize=5,lw=1.4)
        ax.scatter(s.point_excess,y+0.16,marker='D',s=29,color='#cf6c31',zorder=4)
        ref=int(s.reference_excess.iloc[0]);ax.axvline(ref,color='#292d31',lw=1.5)
        ax.set_yticks(y,LABELS);ax.invert_yaxis();ax.set_ylim(5.65,-.7)
        ax.grid(axis='x',alpha=.18)
        ax.set_xlabel('Added excess steps')
        name='Tetrapods' if hist=='tetrapods' else 'Dinosaurs'
        ax.set_title(f'{chr(65+idx)}  {name} {tree}\nLater reference = {ref}',loc='left',fontsize=11)
        ax.set_xlim(left=0,right=80 if hist=='tetrapods' else 410)
    ax=axes.flat[-1];ax.axis('off')
    handles=[Line2D([0],[0],color='#166b93',marker='o',label='Simulation mean and central 95% range'),
             Line2D([0],[0],color='#cf6c31',marker='D',linestyle='',label='Single predicted completion'),
             Line2D([0],[0],color='#292d31',label='Later published coding reference')]
    ax.legend(handles=handles,loc='upper left',bbox_to_anchor=(-.15,.94),frameon=False,fontsize=9)
    ax.text(-.1,.48,'1,000 draws per scheme\n\n349 tetrapod targets\n1,333 dinosaur targets\n\nAll seven reference trees shown\nTwo matrix histories\n\nRanges condition on fixed models.\nThey are not biological confidence\nintervals or estimates of coverage.',
            transform=ax.transAxes,va='top',fontsize=10,linespacing=1.45)
    fig.suptitle('Historical character predictions and their phylogenetic consequences',fontsize=16)
    fig.savefig(OUT/'historical_prediction_results.png',dpi=180)
    fig.savefig(OUT/'historical_prediction_results.pdf')
    plt.close(fig)

    def table(hist):
        lines=['Reference | Method | Single completion | Mean | Central 95% range | Later reference']
        for r in d[d.history==hist].itertuples():
            lines.append(f'{r.reference} | {LABELS[MODELS.index(r.model)]} | {r.point_excess} | {r.mean:.3f} | {r.q025:.3f} to {r.q975:.3f} | {r.reference_excess}')
        return '\n'.join(lines)
    text='''RESULTS

Historical targets and scoring verification

The two downstream histories contain 1,682 eligible target cells: 349 in the 95 unit tetrapod panel and 1,333 in the 74 unit dinosaur panel. Native minimum character lengths were unchanged by all seven tetrapod hard treatments, so their added tree lengths were also added excess steps. Later dinosaur assignments increased the minimum by four steps, one each in characters 245, 313, 336 and 370. Consequently their ordinary additions of 225 and 195 steps became excess additions of 221 and 191 on the two reference trees. The minima of the five deterministic dinosaur predictions were unchanged. Minimum corrections were also applied separately to every simulation.

The earlier tetrapod constrained searches returned original scores of 1,621 and 1,622 steps. All five predicted completions and the later coding reference retained the direction of the backbone preference, with differences of one or two steps. We therefore found no reversal of that constrained preference. These restricted backbones are not complete competing hypotheses of lissamphibian origin.

Deterministic and probabilistic completions

On tetrapod reference O1, both logistic regression and extremely randomized trees produced deterministic excess additions of 15, compared with 37 for the later coding reference. Their propagated mean additions differed: 25.284 for logistic regression, with a central simulation range of 19 to 32, and 52.101 for extremely randomized trees, with a range of 41 to 63.025. Thus identical deterministic aggregate costs concealed distributions on opposite sides of the later reference. Frequency sampling also exceeded the later reference on all five tetrapod trees. The two Mk propagation schemes included the later reference within their central ranges on all five trees; this is a conditional observation for one history, not an estimate of repeated coverage.

The dinosaur replication showed method dependent behavior. Frequency and extremely randomized tree sampling exceeded the later reference on both trees. The central ranges for kNN and logistic regression included it on both trees. The complete Mk results and all deterministic and propagated scores are retained below. No scheme, reference topology or negative result is excluded.

Tetrapod results

'''+table('tetrapods')+'\n\nDinosaur results\n\n'+table('dinosaurs')+'''

Common background control diagnostic

In the 102 unit diagnostic, later target assignments added 36 excess steps on each source reference, whereas restoration of the old control assignments added 13 and 12. All simulated central ranges for the controls lay above these old reference additions. This contrast survives identical scoring backgrounds and character counts, but the reference topologies themselves were inferred using old control states. It therefore cannot establish a causal effect of the missing data mechanism.

Interpretation of replication

The result that deterministic and propagated completions can lead to different downstream assessments is reproducible in the second matrix history. The relative positions of particular methods are not universal across histories or topologies. A thousand simulations quantify the behavior of a fixed predictive scheme; they do not provide a thousand independent fossil datasets.
'''
    (OUT/'results_section.txt').write_text(text)
    core=(OUT/'manuscript_core.txt').read_text()
    abstract='''ABSTRACT

Predictions of missing fossil characters are usually evaluated by their agreement with withheld states, yet paleontological use also depends on how reconstructed states affect phylogenetic evidence. We evaluate frozen predictions against later published assignments in audited tetrapod and dinosaur matrix histories. The downstream panels contain 1,682 originally unknown entries. We compare five prediction methods and six uncertainty propagation schemes on seven fixed reference topologies, with 1,000 completions per scheme and panel. Native minimum step corrections distinguish added character diversity from added excess transformations. Deterministic completions and their propagated distributions yield different evaluations: in tetrapods, two methods adding the same 15 excess steps produce propagated means of 25.3 and 52.1, around a later reference of 37. Both higher and lower compatibility costs occur, while relative performance depends on the matrix history and topology. Conditional joint Mk draws retain within-character tip dependence but do not integrate parameter or tree uncertainty. These historical tests show how later fossil scoring can evaluate both state prediction and conditional downstream behavior, without equating published coding with anatomical truth or declaring a universally superior method.

'''
    core=core.replace('INTRODUCTION\n',abstract+'INTRODUCTION\n',1)
    core=core.replace('RESULTS\n\nThe separate automatically generated results section and tables accompany this core draft. They report all methods and all seven reference topologies rather than selecting a favourable tree.\n',text)
    (OUT/'fossil_character_manuscript_draft.txt').write_text(core)
    print(json.dumps({'rows':len(d),'references':len(pairs),'histories':2,'files_created':5}))

if __name__=='__main__':main()
