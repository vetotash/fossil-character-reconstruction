"""Additive, outcome-informed character deletion diagnostics; no new fits or draws."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, math
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
OUT=HERE/'character_robustness'
SCHEMES=['mode','knn5','logistic','extra_trees','nj_mk','nj_mk_joint']
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
direction=lambda v:'positive' if v>0 else ('negative' if v<0 else 'zero')


def stats(observed,point,draws):
    q=np.quantile(draws,[.025,.975],method='linear');residual=draws-observed
    return dict(observed_excess=int(observed),point_excess=int(point),sim_mean=float(draws.mean()),
        sim_sd=float(draws.std(ddof=1)),sim_q025=float(q[0]),sim_q975=float(q[1]),
        mean_residual=float(residual.mean()),point_residual=int(point-observed),
        mean_residual_direction=direction(int(residual.sum())),
        reference_in_central95=bool(q[0]<=observed<=q[1]),
        reference_interval_position='below' if observed<q[0] else ('above' if observed>q[1] else 'inside'),
        fraction_at_least_reference=float((draws>=observed).mean()),
        fraction_at_most_reference=float((draws<=observed).mean()))


def main():
    assert not OUT.exists(),'Do not overwrite a completed diagnostic'
    OUT.mkdir()
    protocol='''Character contribution robustness for ichthyosaur 2017 to 2019 panel.
This extension is requested after primary results were available. Apply exactly
the round9 deletion diagnostic to the frozen primary NEXUS-order arrays.
For every one of 30 scheme/reference combinations delete each of 128 characters
once, updating both simulated and later-reference endpoints. Also delete the
top 1, 3, 5 and ceiling(10%*128)=13 characters ranked by absolute mean simulation
minus later-reference contribution, with smaller original character ID breaking
ties. The ranking is outcome-informed, a concentration stress check, not an
independent test or a method-selection rule. Retain all original fits, draws,
scoring minima, trees, taxa and characters. No new native scoring is performed.
'''
    (OUT/'PROTOCOL.txt').write_text(protocol)
    paths=[Path(__file__),OUT/'PROTOCOL.txt',HERE/'hard_scores.npz',HERE/'summary.csv']
    paths += [HERE/f'{s}_scores.npz' for s in SCHEMES]
    freeze={'created_utc':datetime.now(timezone.utc).isoformat(),'primary_results_already_seen':True,
            'selection_outcome_informed':True,'inputs':[{'path':str(p),'sha256':sha(p)} for p in paths]}
    (OUT/'FREEZE.json').write_text(json.dumps(freeze,indent=2))
    hard=np.load(HERE/'hard_scores.npz',allow_pickle=False)
    hs,hm=hard['steps'].astype(np.int64),hard['minima'].astype(np.int64)
    chars=hard['characters'];trees=hard['trees'];treat=list(hard['treatments']);n=len(chars)
    assert n==128 and len(trees)==5
    observed=hs[:,1]-hm[1][None,:]-(hs[:,0]-hm[0][None,:])
    saved=pd.read_csv(HERE/'summary.csv')
    full=[];perchar=[];loo=[];stress=[];fragility=[];all_draw=[];all_point=[]
    for scheme in SCHEMES:
        z=np.load(HERE/f'{scheme}_scores.npz',allow_pickle=False)
        assert np.array_equal(z['characters'],chars) and np.array_equal(z['trees'],trees)
        s=z['steps'].astype(np.int64);m=z['minima'].astype(np.int64)
        assert s.shape==(5,1000,128) and m.shape==(1000,128)
        draw=s-m[None,:,:]-(hs[:,0]-hm[0][None,:])[:,None,:]
        k=treat.index('nj_mk' if scheme=='nj_mk_joint' else scheme)
        point=hs[:,k]-hm[k][None,:]-(hs[:,0]-hm[0][None,:])
        all_draw.append(draw);all_point.append(point)
        for ti,tree in enumerate(trees):
            o,p,x=observed[ti],point[ti],draw[ti]
            ot,pt,xt=int(o.sum()),int(p.sum()),x.sum(axis=1);fs=stats(ot,pt,xt)
            tags=dict(lineage='ichthyosaurs_2017_2019',reference=str(tree),scheme=scheme,
                      ntax=101,active_cells=192,nchar=128,replicates=1000)
            full.append({**tags,**fs})
            sr=saved[(saved.model==scheme)&(saved.reference==tree)].iloc[0]
            for a,b in {'observed_excess':'reference_excess_steps','point_excess':'point_excess_steps',
              'sim_mean':'sim_mean_excess_steps','sim_sd':'sim_sd_excess_steps','sim_q025':'sim_q025',
              'sim_q975':'sim_q975','fraction_at_least_reference':'fraction_at_least_reference',
              'fraction_at_most_reference':'fraction_at_most_reference'}.items():
                assert abs(float(fs[a])-float(sr[b]))<1e-9,(scheme,tree,a)
            assert fs['reference_in_central95']==bool(sr.reference_in_central95)
            numerator=x.sum(axis=0)-1000*o;total=int(numerator.sum())
            order=sorted(range(n),key=lambda i:(-abs(int(numerator[i])),int(chars[i])))
            rank={i:r for r,i in enumerate(order,1)}
            for ci,ch in enumerate(chars):
                perchar.append({**tags,'character':int(ch),'observed_excess':int(o[ci]),
                  'point_excess':int(p[ci]),'sim_mean_excess':float(x[:,ci].mean()),
                  'sim_sd_excess':float(x[:,ci].std(ddof=1)),'mean_residual':int(numerator[ci])/1000,
                  'mean_residual_numerator':int(numerator[ci]),'absolute_residual_rank':rank[ci]})
                ss=stats(ot-int(o[ci]),pt-int(p[ci]),xt-x[:,ci])
                loo.append({**tags,'deleted_character':int(ch),'remaining_characters':127,**ss,
                  'direction_same_as_full':ss['mean_residual_direction']==fs['mean_residual_direction'],
                  'inclusion_same_as_full':ss['reference_in_central95']==fs['reference_in_central95']})
            for count in [1,3,5,math.ceil(.1*n)]:
                ix=order[:count];ss=stats(ot-int(o[ix].sum()),pt-int(p[ix].sum()),xt-x[:,ix].sum(axis=1))
                stress.append({**tags,'deleted_k':count,'deleted_characters':json.dumps([int(chars[i]) for i in ix]),
                  'selection':'largest_absolute_mean_residual_outcome_informed','remaining_characters':n-count,**ss,
                  'direction_same_as_full':ss['mean_residual_direction']==fs['mean_residual_direction'],
                  'inclusion_same_as_full':ss['reference_in_central95']==fs['reference_in_central95']})
            same=[i for i in order if int(numerator[i])*total>0];rem=total;zero=None;reverse=None;removed=[]
            for count,ci in enumerate(same,1):
                rem-=int(numerator[ci])
                if rem==0 and zero is None:zero=count
                if rem*total<0 and n-count>0:
                    reverse=count;removed=[int(chars[i]) for i in same[:count]];break
            fragility.append({**tags,'full_mean_residual':total/1000,'full_mean_residual_direction':direction(total),
              'minimum_k_strict_reversal':reverse,'fraction_characters_to_reverse':reverse/n if reverse is not None else None,
              'first_k_exact_zero_before_reversal':zero,'same_signed_character_count':len(same),
              'characters_for_minimum_reversal':json.dumps(removed),'no_strict_reversal_possible':reverse is None,
              'selection':'descending_same_signed_mean_residual_outcome_informed'})
    for name,rows in [('full_summary',full),('per_character',perchar),('leave_one_character_out',loo),
                       ('adversarial_deletions',stress),('mean_direction_fragility',fragility)]:
        pd.DataFrame(rows).to_csv(OUT/f'{name}.csv',index=False)
    np.savez_compressed(OUT/'ichthyosaurs_excess_contributions.npz',observed=observed,
      points=np.stack(all_point),draws=np.stack(all_draw),characters=chars,references=trees,schemes=SCHEMES)
    f,l,a=pd.DataFrame(full),pd.DataFrame(loo),pd.DataFrame(stress);summaries=[]
    for (reference,scheme),g in l.groupby(['reference','scheme'],sort=False):
        ag=a[(a.reference==reference)&(a.scheme==scheme)];fg=f[(f.reference==reference)&(f.scheme==scheme)].iloc[0]
        summaries.append(dict(reference=reference,scheme=scheme,full_direction=fg.mean_residual_direction,
          full_reference_in_interval=bool(fg.reference_in_central95),
          single_deletions_changing_direction=int((~g.direction_same_as_full).sum()),
          single_deletions_changing_inclusion=int((~g.inclusion_same_as_full).sum()),
          adversarial_sizes_changing_direction=int((~ag.direction_same_as_full).sum()),
          adversarial_sizes_changing_inclusion=int((~ag.inclusion_same_as_full).sum())))
    pd.DataFrame(summaries).to_csv(OUT/'robustness_summary.csv',index=False)
    for r in freeze['inputs']:assert sha(r['path'])==r['sha256']
    check=dict(completed_utc=datetime.now(timezone.utc).isoformat(),new_fits=0,new_native_scores=0,new_draws=0,
      full_comparisons=len(f),per_character_rows=len(perchar),leave_one_out_rows=len(l),adversarial_rows=len(a),
      all_full_summaries_match=True,single_deletions_changing_direction=int((~l.direction_same_as_full).sum()),
      single_deletions_changing_inclusion=int((~l.inclusion_same_as_full).sum()),
      adversarial_sizes_changing_direction=int((~a.direction_same_as_full).sum()),
      adversarial_sizes_changing_inclusion=int((~a.inclusion_same_as_full).sum()),
      n_independent_histories=1,n_references=5,selection_outcome_informed=True)
    (OUT/'CHECKS.json').write_text(json.dumps(check,indent=2));print(json.dumps(check,indent=2))


if __name__=='__main__':main()
