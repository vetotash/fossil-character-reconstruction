"""Descriptive evaluation of frozen fits; no fitting or target selection.

Intervals resample taxa and characters independently, with product weights.
They describe sensitivity to the represented rows and characters, conditional
on fitted predictions, not repeated-study uncertainty or a test of novelty.
All ten pairs are reported; no winner is selected for subsequent processing.
"""
from pathlib import Path
import argparse, itertools, json, hashlib
import numpy as np
import pandas as pd

MODELS = ['mode', 'knn5', 'logistic', 'extra_trees', 'nj_mk']

def evaluate(src, out, reps=10000, seed=2026100312):
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(src)
    assert set(df.model) == set(MODELS)
    assert not df.duplicated(['scenario','taxon','character','model']).any()
    df['correct'] = (df.true.astype(str) == df.pred.astype(str)).astype(float)
    summary, confidence, pairs, skipped = [], [], [], []
    rng = np.random.default_rng(seed)
    for scenario, group in df.groupby('scenario'):
        for model, sub in group.groupby('model'):
            summary.append(dict(scenario=scenario,model=model,n=len(sub),
                taxa=sub.taxon.nunique(),characters=sub.character.nunique(),
                agreement=sub.correct.mean(),brier=sub.brier.mean(),
                log_loss=sub.log_loss.mean(),mean_confidence=sub.confidence.mean()))
            for low,high in [(0,.5),(.5,.7),(.7,.9),(.9,1.00000001)]:
                s=sub[(sub.confidence>=low)&(sub.confidence<high)]
                confidence.append(dict(scenario=scenario,model=model,
                    confidence_lower=low,confidence_upper=min(high,1),n=len(s),
                    mean_confidence=s.confidence.mean(),agreement=s.correct.mean()))
        ordered=group.set_index(['taxon','character','model']).sort_index()
        key=group[['taxon','character']].drop_duplicates().sort_values(['taxon','character'])
        ti=pd.Categorical(key.taxon).codes; ci=pd.Categorical(key.character).codes
        nt,nc=ti.max()+1,ci.max()+1
        assert len(group)==len(key)*len(MODELS)
        if nt < 5 or nc < 5:
            skipped.append({'scenario':scenario,'taxa':int(nt),'characters':int(nc),
                'reason':'Fewer than five represented taxa or characters; omit crossed intervals rather than present sparse cluster resampling as general evidence.'})
            continue
        arrays={}
        for metric in ['correct','brier','log_loss']:
            arrays[metric]=np.stack([ordered.xs(m,level='model').reindex(
                pd.MultiIndex.from_frame(key)).loc[:,metric].to_numpy() for m in MODELS],axis=1)
            assert np.isfinite(arrays[metric]).all()
        boot={metric:[] for metric in arrays}
        valid=0
        for start in range(0,reps,500):
            n=min(500,reps-start)
            tw=rng.multinomial(nt,np.ones(nt)/nt,size=n)
            cw=rng.multinomial(nc,np.ones(nc)/nc,size=n)
            w=tw[:,ti]*cw[:,ci]; denom=w.sum(axis=1)
            good=denom>0; w=w[good]; denom=denom[good]; valid+=len(denom)
            for metric, a in arrays.items():
                boot[metric].append((w@a)/denom[:,None])
        for metric,a in arrays.items():
            b=np.concatenate(boot[metric])
            for i,j in itertools.combinations(range(len(MODELS)),2):
                delta=b[:,i]-b[:,j]
                lo,hi=np.quantile(delta,[.025,.975])
                pairs.append(dict(scenario=scenario,metric=metric,model_a=MODELS[i],
                    model_b=MODELS[j],difference_a_minus_b=float((a[:,i]-a[:,j]).mean()),
                    lower=float(lo),upper=float(hi),replicates=valid,
                    excludes_zero=bool(lo>0 or hi<0)))
    pd.DataFrame(summary).to_csv(out/'predictive_metrics.csv',index=False)
    pd.DataFrame(confidence).to_csv(out/'confidence_bins.csv',index=False)
    pd.DataFrame(pairs).to_csv(out/'crossed_bootstrap_pairs.csv',index=False)
    future=df[df.scenario=='later_target'].drop_duplicates(['taxon','character'])
    future.groupby('taxon').size().sort_values(ascending=False).rename('n_targets').to_csv(out/'target_taxon_counts.csv')
    future.groupby('character').size().sort_values(ascending=False).rename('n_targets').to_csv(out/'target_character_counts.csv')
    manifest={'input':str(src),'input_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),
        'seed':seed,'bootstrap_replicates':reps,'methods':MODELS,
        'scope':'Descriptive metrics and exploratory crossed-resampling sensitivity. Later scores are references, not anatomical truth. Intervals are conditional on frozen fitted predictions and do not account for phylogenetic dependence or between-study generalization.',
        'intervals':'Pointwise percentile intervals for all ten method pairs; not multiplicity-adjusted and not a basis for selecting a universal winner.',
        'confidence_bins':'lower inclusive, upper exclusive; last includes one; no recalibration',
        'crossed_intervals_omitted':skipped,
        'analysis_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/'METRICS_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('predictions',type=Path);p.add_argument('output',type=Path)
    args=p.parse_args();evaluate(args.predictions,args.output)
