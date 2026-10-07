"""Additive character deletion diagnostics using frozen native-score arrays only."""
from pathlib import Path
import hashlib,json,math
from datetime import datetime,timezone
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent;R8=HERE.parents[1]/'round8'
MODELS=['mode','knn5','logistic','extra_trees','nj_mk','nj_mk_joint']
CONFIGS={'tetrapods':dict(folder=R8/'probability_propagation/future95',nchar=87,ntax=95,ncells=349),
 'dinosaurs':dict(folder=R8/'replication/probability',nchar=367,ntax=74,ncells=1333)}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def direction(n):return 'positive' if n>0 else ('negative' if n<0 else 'zero')
def stats(observed,point,draws):
    q=np.quantile(draws,[.025,.975],method='linear')
    resid=draws-observed
    return dict(observed_excess=int(observed),point_excess=int(point),sim_mean=float(draws.mean()),
      sim_sd=float(draws.std(ddof=1)),sim_q025=float(q[0]),sim_q975=float(q[1]),
      mean_residual=float(resid.mean()),point_residual=int(point-observed),
      mean_residual_direction=direction(int(resid.sum())),
      reference_in_central95=bool(q[0]<=observed<=q[1]),
      reference_interval_position='below' if observed<q[0] else ('above' if observed>q[1] else 'inside'),
      fraction_at_least_reference=float((draws>=observed).mean()),
      fraction_at_most_reference=float((draws<=observed).mean()))
def ranks(values,chars,absolute=False):
    order=sorted(range(len(chars)),key=lambda i:(-abs(int(values[i])) if absolute else -int(values[i]),int(chars[i])))
    out=np.empty(len(chars),dtype=int)
    for rank,i in enumerate(order,1):out[i]=rank
    return order,out
def write(name,rows):pd.DataFrame(rows).to_csv(HERE/name,index=False)

def main():
    freeze=json.loads((HERE/'FREEZE.json').read_text())
    for p,meta in freeze['inputs'].items():assert sha(Path(p))==meta['sha256']
    full=[];perchar=[];loo=[];stress=[];fragile=[];replicates=[];biology=[];bio_avg=[];validations=[];panel_arrays={}
    for lineage,cfg in CONFIGS.items():
        folder=cfg['folder'];hard=np.load(folder/'hard_scores.npz',allow_pickle=False)
        hs=hard['steps'].astype(np.int64);hm=hard['minima'].astype(np.int64);chars=hard['characters'];trees=hard['trees'];treat=list(hard['treatments']);n=len(chars)
        assert n==cfg['nchar'] and len(set(chars))==n
        obs=hs[:,1]-hm[1][None,:]-(hs[:,0]-hm[0][None,:])
        arrays={};points={};raw_arrays={};minimum_arrays={}
        saved=pd.read_csv(R8/'probability_propagation/summary.csv') if lineage=='tetrapods' else pd.concat([pd.read_csv(folder/'summary.csv'),pd.read_csv(R8/'replication/joint_probability/summary.csv')])
        for model in MODELS:
            path=R8/'replication/joint_probability/scores.npz' if lineage=='dinosaurs' and model=='nj_mk_joint' else folder/f'{model}_scores.npz'
            z=np.load(path,allow_pickle=False);s=z['steps'].astype(np.int64);m=z['minima'].astype(np.int64)
            assert np.array_equal(z['characters'],chars) and np.array_equal(z['trees'],trees)
            assert s.shape==(len(trees),1000,n) and m.shape==(1000,n)
            draw=s-m[None,:,:]-(hs[:,0]-hm[0][None,:])[:,None,:]
            point_index=treat.index('nj_mk' if model=='nj_mk_joint' else model)
            point=hs[:,point_index]-hm[point_index][None,:]-(hs[:,0]-hm[0][None,:])
            arrays[model]=draw;points[model]=point;raw_arrays[model]=s-hs[:,0,None,:];minimum_arrays[model]=m-hm[0,None,:]
            for ti,tree in enumerate(trees):
                o=obs[ti];p=point[ti];x=draw[ti];ot=int(o.sum());pt=int(p.sum());xt=x.sum(axis=1);fs=stats(ot,pt,xt)
                tags=dict(lineage=lineage,reference=str(tree),scheme=model,ntax=cfg['ntax'],active_cells=cfg['ncells'],nchar=n,replicates=len(x))
                full.append({**tags,**fs})
                sr=saved[(saved.model==model)&((saved.tree==tree)&(saved.panel=='future95') if lineage=='tetrapods' else (saved.reference==tree))].iloc[0]
                mapping={'observed_excess':'reference_added_excess_steps' if lineage=='tetrapods' else 'reference_excess_steps',
                    'point_excess':'point_added_excess_steps' if lineage=='tetrapods' else 'point_excess_steps',
                    'sim_mean':'sim_mean_added_excess_steps' if lineage=='tetrapods' else 'sim_mean_excess_steps',
                    'sim_sd':'sim_sd_added_excess_steps' if lineage=='tetrapods' else 'sim_sd_excess_steps',
                    'sim_q025':'sim_q025','sim_q975':'sim_q975','fraction_at_least_reference':'fraction_at_least_reference','fraction_at_most_reference':'fraction_at_most_reference'}
                for k,v in mapping.items():assert abs(float(fs[k])-float(sr[v]))<1e-9,(tags,k,fs[k],sr[v])
                assert fs['reference_in_central95']==bool(sr.reference_in_central95)
                assert abs(float(raw_arrays[model][ti].sum(axis=1).mean())-float(sr['sim_mean_added_steps' if lineage=='tetrapods' else 'mean_added_steps']))<1e-9
                if lineage=='dinosaurs':assert abs(float(minimum_arrays[model].sum(axis=1).mean())-float(sr.mean_added_minimum_steps))<1e-9
                validations.append({**tags,'all_frozen_summary_fields_match':True,'sum_per_character_equals_full_native_endpoint':True})
                residual_num=x.sum(axis=0)-1000*o
                order,absrank=ranks(residual_num,chars,True);_,signedrank=ranks(residual_num,chars);_,orank=ranks(o,chars)
                total_num=int(residual_num.sum());assert total_num==int((xt-ot).sum())
                for ci,ch in enumerate(chars):
                    perchar.append({**tags,'character':int(ch),'observed_excess':int(o[ci]),'point_excess':int(p[ci]),
                      'sim_mean_excess':float(x[:,ci].mean()),'sim_sd_excess':float(x[:,ci].std(ddof=1)),
                      'mean_residual':int(residual_num[ci])/1000,'mean_residual_numerator':int(residual_num[ci]),
                      'mean_minus_point':float(x[:,ci].mean()-p[ci]),'absolute_residual_rank':int(absrank[ci]),
                      'signed_residual_rank':int(signedrank[ci]),'later_contribution_rank':int(orank[ci])})
                    ss=stats(ot-int(o[ci]),pt-int(p[ci]),xt-x[:,ci])
                    loo.append({**tags,'deleted_character':int(ch),'remaining_characters':n-1,**ss,
                      'direction_same_as_full':ss['mean_residual_direction']==fs['mean_residual_direction'],
                      'inclusion_same_as_full':ss['reference_in_central95']==fs['reference_in_central95']})
                for k in sorted(set([1,3,5,math.ceil(.1*n)])):
                    ix=order[:k];ss=stats(ot-int(o[ix].sum()),pt-int(p[ix].sum()),xt-x[:,ix].sum(axis=1))
                    stress.append({**tags,'deleted_k':k,'deleted_characters':json.dumps([int(chars[i]) for i in ix]),
                      'selection':'largest_absolute_mean_residual_outcome_informed','remaining_characters':n-k,**ss,
                      'direction_same_as_full':ss['mean_residual_direction']==fs['mean_residual_direction'],
                      'inclusion_same_as_full':ss['reference_in_central95']==fs['reference_in_central95']})
                ss_order=[i for i in order if int(residual_num[i])*total_num>0]
                rem=total_num;first_zero=None;first_reverse=None;rev_removed=[]
                for k,ci in enumerate(ss_order,1):
                    rem-=int(residual_num[ci])
                    if rem==0 and first_zero is None:first_zero=k
                    if rem*total_num<0 and n-k>0:
                        first_reverse=k;rev_removed=[int(chars[i]) for i in ss_order[:k]];break
                fragile.append({**tags,'full_mean_residual':total_num/1000,'full_mean_residual_direction':direction(total_num),
                  'minimum_k_strict_reversal':first_reverse,'fraction_characters_to_reverse':first_reverse/n if first_reverse is not None else None,
                  'first_k_exact_zero_before_reversal':first_zero,'same_signed_character_count':len(ss_order),
                  'characters_for_minimum_reversal':json.dumps(rev_removed),'no_strict_reversal_possible':first_reverse is None,
                  'selection':'descending_same_signed_mean_residual_outcome_informed'})
                replicates.extend({**{k:tags[k] for k in ['lineage','reference','scheme']},'replicate':b,
                  'observed_excess':ot,'point_excess':pt,'simulation_excess':int(xt[b]),'simulation_residual':int(xt[b]-ot)} for b in range(1000))
        # Biological index without model-dependent repetition of observed increments.
        b_rows=[]
        for ti,tree in enumerate(trees):
            delta=(arrays['extra_trees'][ti]-arrays['logistic'][ti]).sum(axis=0)
            _,rank=ranks(delta,chars);_,arank=ranks(delta,chars,True);_,orank=ranks(obs[ti],chars)
            for ci,ch in enumerate(chars):
                r=dict(lineage=lineage,reference=str(tree),character=int(ch),observed_excess=int(obs[ti,ci]),
                  observed_contribution_rank=int(orank[ci]),extra_trees_mean=float(arrays['extra_trees'][ti,:,ci].mean()),
                  logistic_mean=float(arrays['logistic'][ti,:,ci].mean()),et_minus_logistic_mean=int(delta[ci])/1000,
                  et_minus_logistic_signed_rank=int(rank[ci]),et_minus_logistic_absolute_rank=int(arank[ci]))
                b_rows.append(r);biology.append(r)
        bframe=pd.DataFrame(b_rows)
        g=bframe.groupby('character',sort=True).agg(reference_count=('reference','size'),observed_mean=('observed_excess','mean'),
          observed_min=('observed_excess','min'),observed_max=('observed_excess','max'),et_minus_logistic_mean=('et_minus_logistic_mean','mean'),
          et_minus_logistic_min=('et_minus_logistic_mean','min'),et_minus_logistic_max=('et_minus_logistic_mean','max')).reset_index()
        g['lineage']=lineage
        for col,dest,absolute in [('observed_mean','observed_mean_rank',False),('et_minus_logistic_mean','et_minus_logistic_signed_rank',False),('et_minus_logistic_mean','et_minus_logistic_absolute_rank',True)]:
            # All references have 1000 draws; integer sums preserve exact ties.
            numerator=np.rint(g[col].to_numpy()*len(trees)*1000).astype(np.int64)
            order=sorted(g.index,key=lambda i:(-abs(int(numerator[i])) if absolute else -int(numerator[i]),g.at[i,'character']))
            for rank,i in enumerate(order,1):g.at[i,dest]=rank
        bio_avg.extend(g.to_dict('records'))
        panel_arrays[lineage]={'observed':obs,'points':np.stack([points[m] for m in MODELS]),
          'draws':np.stack([arrays[m] for m in MODELS]),'characters':chars,'references':trees,'schemes':np.array(MODELS)}
    for name,rows in [('full_summary.csv',full),('per_character.csv',perchar),('leave_one_character_out.csv',loo),('adversarial_deletions.csv',stress),
      ('mean_direction_fragility.csv',fragile),('full_residual_draws.csv',replicates),('biological_character_index.csv',biology),('biological_across_reference_index.csv',bio_avg),('full_total_validation.csv',validations)]:write(name,rows)
    for lineage,data in panel_arrays.items():np.savez_compressed(HERE/f'{lineage}_excess_contributions.npz',**data)
    f=pd.DataFrame(full);l=pd.DataFrame(loo);a=pd.DataFrame(stress)
    robustness=[]
    for keys,g in l.groupby(['lineage','reference','scheme'],sort=False):
        tag=dict(zip(['lineage','reference','scheme'],keys));sg=a[(a.lineage==keys[0])&(a.reference==keys[1])&(a.scheme==keys[2])]
        fg=f[(f.lineage==keys[0])&(f.reference==keys[1])&(f.scheme==keys[2])].iloc[0]
        robustness.append({**tag,'full_direction':fg.mean_residual_direction,'full_reference_in_interval':bool(fg.reference_in_central95),
          'all_single_deletions_preserve_direction':bool(g.direction_same_as_full.all()),'single_deletions_changing_direction':int((~g.direction_same_as_full).sum()),
          'all_single_deletions_preserve_inclusion':bool(g.inclusion_same_as_full.all()),'single_deletions_changing_inclusion':int((~g.inclusion_same_as_full).sum()),
          'all_adversarial_sizes_preserve_direction':bool(sg.direction_same_as_full.all()),'all_adversarial_sizes_preserve_inclusion':bool(sg.inclusion_same_as_full.all())})
    write('robustness_summary.csv',robustness)
    check=dict(completed_utc=datetime.now(timezone.utc).isoformat(),all_frozen_input_hashes_unchanged=True,
      new_native_scores=0,new_model_fits=0,new_random_draws=0,all_full_totals_match_frozen_summaries=True,
      full_comparisons=len(full),per_character_rows=len(perchar),leave_one_out_rows=len(loo),adversarial_rows=len(stress),
      full_residual_draw_rows=len(replicates),biology_rows=len(biology),n_independent_histories=2,
      references=7,schemes=6,protocol_sha256=sha(HERE/'PROTOCOL.txt'),script_sha256=sha(Path(__file__)),
      character_additivity_only=True,adversarial_selection_outcome_informed=True,
      files={p.name:sha(p) for p in sorted(HERE.glob('*.csv'))})
    (HERE/'CHECKS.json').write_text(json.dumps(check,indent=2)+'\n');print(json.dumps(check,indent=2))

if __name__=='__main__':main()
