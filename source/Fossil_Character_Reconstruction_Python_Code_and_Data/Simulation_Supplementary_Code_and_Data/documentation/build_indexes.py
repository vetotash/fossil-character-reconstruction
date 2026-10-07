#!/usr/bin/env python3
"""Regenerate FILE_CATALOG.tsv and DATA_DICTIONARY.tsv from this package.
Reads archived data for schema/shape/missing-value metadata; never fits models,
changes scientific outputs, or updates SHA256SUMS.txt. The semantic mappings are
transcribed from package READMEs and inspected producing code. Numpy is required.
"""
from pathlib import Path
from collections import defaultdict,Counter
import csv,json,gzip,hashlib,re,math,zipfile
import numpy as np
import argparse
parser=argparse.ArgumentParser(description='Rebuild documentation indexes from supplied files without rerunning scientific analyses.')
parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
ROOT=parser.parse_args().root.resolve()
GEN={'FILE_CATALOG.tsv','DATA_DICTIONARY.tsv'}

def family(n):
 n=re.sub(r's\d{2}_r\d{3}', '{matrix_id}',n)
 n=re.sub(r's\d{2}(?=_)','{scenario_id}',n)
 return n

def typ(v):
 if v is None:return 'null'
 if isinstance(v,bool):return 'boolean'
 if isinstance(v,int):return 'integer'
 if isinstance(v,float):return 'number'
 if isinstance(v,str):return 'string'
 return type(v).__name__

def shape(v):
 if not isinstance(v,list):return []
 if not v:return [0]
 sh=[shape(e) for e in v]
 return [len(v)]+(sh[0] if all(x==sh[0] for x in sh) else ['ragged'])

def keyname(k):
 if re.fullmatch(r'[0-9a-f]{64}',str(k)):return '{sha256}'
 if re.fullmatch(r'\d+',str(k)):return '{index}'
 if '/' in k or re.search(r'\.(?:py|csv|json|npz|txt|tsv|gz|pdf|md)$',k):return '{file_path}'
 if re.fullmatch(r's\d{2}_r\d{3}',k):return '{matrix_id}'
 return k

def rec_field(fields,k,v,depth=0):
 if isinstance(v,dict):
  if not v: record(fields,k,'object',(),None,'empty_object')
  for kk,vv in v.items():rec_field(fields,k+'.'+keyname(kk) if k else keyname(kk),vv,depth+1)
 elif isinstance(v,list):
  if any(isinstance(x,dict) for x in v):
   for x in v:rec_field(fields,k+'[*]',x,depth+1)
   if not v:record(fields,k,'array',tuple([0]),None,'empty_array')
  else:
   flat=[]
   def flatlist(a):
    for x in a:
     if isinstance(x,list):flatlist(x)
     else:flat.append(x)
   flatlist(v)
   types=sorted(set(typ(x) for x in flat))
   record(fields,k,'array['+'|'.join(types)+']',tuple(shape(v)),None,'empty_array' if not flat else '')
   fields[k]['null']+=sum(x is None for x in flat)
   fields[k]['nan']+=sum(isinstance(x,float) and not math.isfinite(x) for x in flat)
   for x in flat[:6]:addsample(fields[k],x)
 else: record(fields,k,typ(v),(),v,'')

def addsample(s,v):
 if isinstance(v,(str,int,float,bool)) and len(s['examples'])<5:
  st=str(v)
  if len(st)>140:st=st[:137]+'...'
  if st not in s['examples']:s['examples'].append(st)

def record(fields,k,t,sh,v,note):
 s=fields.setdefault(k,{'types':set(),'shapes':set(),'n':0,'null':0,'nan':0,'empty':0,'examples':[]})
 s['types'].add(t);s['shapes'].add(str(sh));s['n']+=1
 if v is None and t=='null':s['null']+=1
 if isinstance(v,float) and not math.isfinite(v):s['nan']+=1
 if note:s['empty']+=1
 addsample(s,v)

schemas={};files=[]
paths=sorted(p for p in ROOT.rglob('*') if p.is_file() and p.name not in GEN)
for idx,p in enumerate(paths):
 n=p.relative_to(ROOT).as_posix();fam=family(n)
 fmt='json.gz' if n.endswith('.json.gz') else p.suffix[1:]
 if fmt in ['csv','tsv','npz','json','json.gz']:
  sid=fmt.upper().replace('.','_')+'_'+hashlib.sha256(fam.encode()).hexdigest()[:12]
  s=schemas.setdefault(sid,{'family':fam,'format':fmt,'files':[],'fields':{}});s['files'].append(n)
  fields=s['fields'];rows=''
  if fmt in ['csv','tsv']:
   with p.open(newline='') as f:
    reader=csv.DictReader(f,delimiter='\t' if fmt=='tsv' else ',');rows=0
    for row in reader:
     rows+=1
     for k,v in row.items():
      if v is None:record(fields,k,'null',(),None,'');continue
      if not v: record(fields,k,'blank',(),'', 'empty_string');continue
      try:
       vv=float(v);tt='integer' if re.fullmatch(r'[-+]?\d+',v) else 'number';record(fields,k,tt,(),vv,'')
      except ValueError:record(fields,k,'string',(),v,'')
  elif fmt=='npz':
   with np.load(p,allow_pickle=False) as data:
    rows=len(data.files)
    for k in data.files:
     a=data[k];record(fields,k,str(a.dtype),tuple(a.shape),None,'empty_array' if a.size==0 else '')
     if np.issubdtype(a.dtype,np.floating):fields[k]['nan']+=int(np.sum(~np.isfinite(a)))
  else:
   with (gzip.open(p,'rt') if fmt=='json.gz' else p.open()) as f:o=json.load(f)
   rec_field(fields,'',o);rows=len(o) if isinstance(o,(dict,list)) else 1
  files.append({'path':n,'format':fmt,'schema_id':sid,'rows_or_members':str(rows),'bytes':p.stat().st_size})
 else:files.append({'path':n,'format':fmt or 'text','schema_id':'NOT_TABULAR','rows_or_members':'','bytes':p.stat().st_size})
 if idx%1000==0:print('Inspected',idx,flush=True)
for s in schemas.values():
 for f in s['fields'].values(): f['types']=sorted(f['types']);f['shapes']=sorted(f['shapes'])
I={'files':files,'schemas':schemas}

D={}
def add(keys,desc,unit='dimensionless'):
 for k in keys.split('|'):D[k]=(desc,unit)
add('scenario_id','Factorial scenario identifier.','identifier')
add('matrix_id','Independent simulated matrix identifier: sSS_rRRR.','identifier')
add('ordinal|replicate|matrix_index|character|draw_index','Zero-based index of the named scenario, replicate, character, or completion draw; draw_index=-1 denotes baseline/MAP.','index')
add('q','Instantaneous off-diagonal binary symmetric Mk generating rate; not per geological year.','per unit generating branch length')
add('mk_rate|rate','Fitted Mk off-diagonal transition rate.','per unit inferred branch length')
add('method|scheme|reference|candidate|left_method|right_method','Probability method/completion scheme or named member of a paired comparison; ordering is stated in REPRODUCIBILITY.txt.','category')
add('mask|mask_mode|mask_regime','Masking rule: benchmark random/structured/statebiased, or oracle random/clustered.','category')
add('kind','Treatment category distinguishing source incomplete, truth complete, MAP, and sampled completions.','category')
add('baseline','Incomplete-source or truth-complete matrix comparator.','category')
add('search_design','Original three-random-start or amended three-random-plus-NJ search design.','category')
add('shape','Generating tree shape (balanced/caterpillar in oracle design).','category')
add('tree_key','Key identifying a generating tree in the tree registry.','identifier')
add('group','Disjoint target-cell loss group; see distribution_analysis/README.txt.','category')
add('character_group','Disjoint response-character D-contribution group; see distribution_analysis/README.txt.','category')
add('metric','Name of the metric being contrasted; associated numeric differences retain that metric\'s units.','identifier')
add('difference|direction|difference_direction','Recorded subtraction order for a paired comparison.','text')
add('diagnostic','Truth-dependent oracle-centering diagnostic label; not a forecast.','category')
add('accuracy|realized_accuracy','Fraction of hidden target cells with correct hard prediction.','proportion')
add('brier|realized_brier','Mean sum of two squared class-probability errors over target cells (0 to 2).')
add('log_loss','Mean negative log probability of the true target state, with probability floored at 1e-12.','nats')
add('error_rate','Fraction of target-cell hard predictions that are incorrect.','proportion')
add('mean_max_probability|forecast_mean_max_probability','Mean largest predicted class probability across hidden target cells.','probability')
add('old_L|source_L','Incomplete-source matrix parsimony length on the fixed generating tree.','parsimony steps')
add('old_M|source_M','Sum of incomplete-source per-character minimum parsimony lengths.','parsimony steps')
add('true_L','Truth-complete matrix parsimony length on the fixed generating tree.','parsimony steps')
add('true_M','Sum of truth-complete per-character minimum parsimony lengths.','parsimony steps')
add('true_D|truth_D','True target completion excess cost: (L_complete-old_L)-(M_complete-old_M).','parsimony steps')
add('hard_D|marginal_map_D','Fixed-tree D of the common cellwise maximum-probability completion; joint Mk MAP is not joint MAP.','parsimony steps')
add('hard_D_error','hard_D minus true_D.','parsimony steps')
add('hard_D_absolute_error|map_absolute_D_error','Absolute difference between marginal-MAP D and true_D.','parsimony steps')
add('mean_D|predicted_D_mean','Predictive mean D: mean over saved draws in benchmark, exact probability-weighted mean in oracle.','parsimony steps')
add('mean_D_error|signed_bias|mean_minus_truth','Predictive mean D minus realized true_D.','parsimony steps')
add('mean_D_absolute_error|absolute_mean_error','Absolute difference between predictive mean D and realized true_D.','parsimony steps')
add('sd_D','Sample standard deviation of saved D draws.','parsimony steps')
add('mean_D_mcse','Within-matrix SD of D draws divided by sqrt(n_draws); scenario-summary naming differs as documented.','parsimony steps')
add('lower95|upper95','Lower/upper empirical 0.025/0.975 quantile of saved D draws, using linear interpolation.','parsimony steps')
add('D_q025|D_q975','Discrete exact-PMF 0.025/0.975 quantile of D.','parsimony steps')
add('interval_width|width95','Upper95 minus lower95; oracle uses its discrete D quantiles.','parsimony steps')
add('truth_in_interval|truth_in_95|realized_covered','Indicator that realized true_D lies inside the central interval, including endpoints.','0/1 indicator')
add('truth_below_interval|truth_below_95','Indicator that realized true_D is strictly below the central interval.','0/1 indicator')
add('truth_above_interval|truth_above_95','Indicator that realized true_D is strictly above the central interval.','0/1 indicator')
add('coverage|interval_coverage','Fraction of independent matrices with true_D in the central interval.','proportion')
add('coverage_mcse','Binomial Monte Carlo standard error sqrt(p*(1-p)/n_matrices).','proportion')
add('coverage_count|inclusion_count|count_truth_below_interval|count_truth_above_interval','Number of independent matrices satisfying the coverage/inclusion/outside-range condition named in the field.','count of matrices')
add('coverage_wilson_low|coverage_wilson_high|coverage_wilson_lower95|coverage_wilson_upper95|wilson_lower95|wilson_upper95|inclusion_Wilson_low|inclusion_Wilson_high','Lower/upper Wilson 95% confidence limit for the matrix-level inclusion proportion.','proportion')
add('constant_fallback','Indicator that a response has only one observed training state and uses the constant-state fallback.','0/1 indicator')
add('constant_response_count|nonconstant_response_count','Number of response characters with one/multiple observed training states, respectively.','count of characters')
add('convergence_warning_count|logistic_convergence_warning','Number of logistic response fits with ConvergenceWarning; character-level values are 0/1.','count/indicator')
add('logistic_fit_ran|mk_optimizer_ran','Whether the indicated fit/optimizer was attempted; scenario diagnostics sum response-level flags.','count/indicator')
add('logistic_reached_max_iter','Iteration-limit flag for logistic regression; summed in scenario diagnostics.','count/indicator')
add('mk_optimizer_success','Optimizer success flag; not_run if optimization was not attempted.','0/1 or not_run')
add('mk_rate_boundary','Flag abs(log(mk_rate)) >= 7.999; summed in scenario diagnostics.','count/indicator')
add('mk_optimizer_failed','Count of attempted Mk optimizations without success.','count')
add('mk_warning_count','Number of recorded Mk warning messages.','count')
add('nj_zero_overlap_pairs','Number of taxon pairs having no jointly observed predictors.','count of taxon pairs')
add('nj_negative_branches','Number of negative neighbour-joining branches before truncation.','count of branches')
add('nj_truncated_branches','Number of NJ branches floored to 1e-8 for likelihood calculations.','count of branches')
add('zero_true_probability_count|zero_true_probability_total','Number of hidden target truths assigned exactly zero probability before flooring.','count of target cells')
add('zero_true_probability_proportion|zero_true_probability_fraction','Fraction of hidden target truths assigned exactly zero probability before flooring.','proportion')
add('log_floor_count|log_floor_total','Number of target truths whose assigned probability is below 1e-12.','count of target cells')
add('log_floor_proportion','Fraction of target truths whose assigned probability is below 1e-12.','proportion')
add('oracle_D_mean','Exact mean D under the true joint conditional completion distribution.','parsimony steps')
add('predicted_D_variance','Exact variance of the condition\'s D probability mass function.','squared parsimony steps')
add('mean_minus_oracle','Predicted exact D mean minus oracle-joint exact D mean.','parsimony steps')
add('oracle_conditional_interval_mass','True oracle-joint probability mass inside the condition\'s reported D interval.','probability')
add('own_interval_mass','The condition\'s own D-PMF mass inside its reported central interval.','probability')
add('D_total_variation_from_oracle','One-half sum of absolute differences between condition and oracle D PMFs.','probability')
add('oracle_expected_brier','Expected two-class-sum Brier loss integrating the forecast against true oracle conditional marginals.')
add('oracle_expected_accuracy','Expected accuracy of the common hard decision under true oracle marginals.','probability')
add('mean_marginal_entropy_bits','Mean binary entropy of missing-cell marginal probabilities.','bits per target cell')
add('oracle_total_correlation_bits_per_character','Sum of oracle missing-tip marginal entropies minus joint entropy conditional on observations, averaged over characters.','bits per character')
add('crps_empirical','CRPS of saved empirical D distribution: mean|draw-truth|-sum pairwise distances/(2*M^2).','parsimony steps')
add('crps_fair_ensemble','Finite-ensemble sensitivity CRPS using M*(M-1) in the pairwise denominator; different estimand from empirical CRPS.','parsimony steps')
add('training_unseen_fraction','Fraction of target true states absent from their response\'s observed training labels.','proportion')
add('constant_target_fraction','Fraction of targets belonging to responses with one observed training state.','proportion')
add('target_fraction','Number of targets in this disjoint cell group divided by all matrix targets.','proportion')
add('error_count','Number of incorrect hard target predictions within the specified cell group.','count of target cells')
add('brier_sum','Sum of two-class-sum Brier losses over targets in this cell group.')
add('log_loss_sum','Sum of negative natural-log true-state probabilities within the group; floor 1e-12.','nats')
add('error_fraction_all_targets','Group error count divided by all matrix target cells.','proportion')
add('brier_contribution_all_targets','Group Brier sum divided by all matrix target cells.')
add('log_loss_contribution_all_targets','Group log-loss sum divided by all matrix target cells.','nats')
add('brier_within_group','Group Brier sum divided by group target count; undefined for empty groups.')
add('log_loss_within_group','Group log-loss sum divided by group target count; undefined for empty groups.','nats')
add('true_D_component|mean_D_component','Sum of characterwise true/mean D over the named disjoint character group.','parsimony steps')
add('signed_bias_contribution','mean_D_component minus true_D_component; group contributions sum to full signed bias.','parsimony steps')
add('applied_shift','Negative original signed bias added to each draw using its own evaluation truth; diagnostic only.','parsimony steps')
add('crps_change_from_original','Oracle-centered diagnostic empirical CRPS minus original empirical CRPS.','parsimony steps')
add('inclusion_change_from_original','Oracle-centered diagnostic inclusion indicator minus original inclusion indicator.','difference of 0/1 indicators')
add('marginal_Mk_CRPS|joint_Mk_CRPS','Empirical CRPS of marginal/joint Mk scheme.','parsimony steps')
add('joint_minus_marginal_CRPS','Within-matrix joint Mk minus marginal Mk empirical CRPS, then mean over 50 matrices.','parsimony steps')
add('marginal_Mk_inclusion_out_of_50|joint_Mk_inclusion_out_of_50','Number of the 50 independent matrices whose true_D is inside the corresponding Mk central interval.','count of matrices')
add('all_scheme_signed_bias_min|all_scheme_signed_bias_max','Minimum/maximum across scheme-level mean signed biases in the scenario.','parsimony steps')
add('minimum_found_length','Lowest ordinary unweighted parsimony length encountered by finite NNI searches; not proof of a global optimum.','parsimony steps')
add('generating_tree_length','Ordinary parsimony length of the searched matrix on its generating tree.','parsimony steps')
add('length_minus_generating|mean_length_gap','Encountered minimum length minus the generating-tree length (averaged in scenario audit).','parsimony steps')
add('n_length_above_generating','Count of matrices whose found baseline length exceeds the generating-tree length.','count of matrices')
add('n_distinct_best_endpoints','Number of distinct tied endpoint topologies at the best length encountered.','count of topologies')
add('mean_normalized_RF|min_normalized_RF|max_normalized_RF','Mean/minimum/maximum normalized split-set symmetric-difference distance to the generating tree across distinct retained endpoint topologies.','normalized RF, 0 to 1')
add('mean_split_recall','Fraction of generating splits recovered, averaged across retained endpoints; for binary trees equals 1-normalized RF.','proportion')
add('truth_among_retained_endpoints','Whether the generating topology is among distinct best encountered endpoint topologies.','0/1 indicator')
add('consensus_n_splits','Number of nontrivial splits shared by all retained best endpoints.','count of splits')
add('consensus_resolution','Strict-consensus split count divided by n_tips-3.','proportion')
add('consensus_split_recall','Number of generating splits in the strict consensus divided by n_tips-3.','proportion')
add('consensus_split_precision','Number of correct strict-consensus splits divided by consensus split count; undefined when the count is zero.','proportion')
add('any_start_step_limit','Whether any starting-tree search reached the 200-accepted-move cap.','0/1 indicator')
add('mean_accepted_moves','Mean accepted strictly improving NNI moves across starting trees.','count of moves')
add('fixed_scaffold_D','D of this searched completion on the fixed generating scaffold; search minimizes ordinary length, not D.','parsimony steps')
add('fixed_scaffold_absolute_D_error','Absolute fixed-scaffold D minus true_D; sampled summaries average draw-level absolute errors.','parsimony steps')
add('RF_difference_from_source_incomplete|RF_difference_from_truth_complete','Paired normalized RF minus that of the named baseline for the same matrix.','normalized RF difference')
add('worse_RF_than_source_incomplete|worse_RF_than_truth_complete','Whether mean normalized RF exceeds the paired named baseline.','0/1 indicator')
add('fixed_D_exact_match','Whether fixed-scaffold D equals realized true_D.','0/1 indicator')
add('D_exact_and_RF_worse_than_truth_complete','Conjunction of exact fixed-D match and RF worse than paired truth-complete baseline.','0/1 indicator')
add('source_incomplete_RF|truth_complete_RF|sampled_RF','Mean normalized RF for the named baseline or sampled completion treatment.','normalized RF, 0 to 1')
add('full_mean_normalized_rf|source_mean_normalized_rf|map_mean_normalized_rf','Mean normalized RF over all tied most-parsimonious trees for the named complete/source/MAP matrix.','normalized RF, 0 to 1')
add('full_truth_in_mpt|source_truth_in_mpt|map_truth_in_mpt','Whether generating tree belongs to the exhaustive most-parsimonious tree set for the named matrix.','0/1 indicator')
add('full_mpt_count|source_mpt_count|map_mpt_count','Number of exhaustive tied most-parsimonious trees for the named matrix.','count of trees')
add('draw_mean_normalized_rf','Mean across saved completion draws of their all-ties normalized RF.','normalized RF, 0 to 1')
add('draw_truth_mpt_probability','Fraction of completion draws whose exhaustive optimum set contains the generating tree.','proportion')
add('draw_mean_mpt_count','Mean number of exhaustive most-parsimonious trees across completion draws.','count of trees')
add('draw_mean_absolute_D_error','Mean absolute fixed-generating-tree D error across completion draws.','parsimony steps')
add('draw_mean_D','Mean fixed-generating-tree D across saved completion draws.','parsimony steps')
add('D_exact_match_count|pooled_exact_match_count','Number of saved draws matching realized truth D; pooled version sums across matrices.','count of draws')
add('D_exact_match_probability','Fraction of saved completion draws matching realized truth D.','proportion')
add('D_exact_match_rf_worse_than_full_count','Number of draws with exact realized-D match and mean all-ties RF greater than paired complete-matrix RF by >1e-12.','count of draws')
add('matrices_with_exact_D_and_rf_worse_than_full','Number of matrices having at least one exact-D-matching draw with RF worse than paired complete matrix.','count of matrices')
add('truth_absent_given_D_exact_match|pooled_truth_absent_given_D_exact_match','Among exact-D-matching draws, proportion whose exhaustive optimum set excludes the generating topology; pooled version divides summed counts.','conditional proportion')
add('mean_rf_given_D_exact_match','Mean all-ties normalized RF among draws matching realized truth D.','normalized RF, 0 to 1')
add('mpt_differs_full_given_D_exact_match|pooled_mpt_differs_full_given_D_exact_match','Among exact-D-matching draws, proportion whose exhaustive optimum set differs from that of the complete matrix; pooled version uses summed counts.','conditional proportion')
add('rf_worse_than_full_given_D_exact_match|pooled_rf_worse_than_full_given_D_exact_match','Among exact-D-matching draws, proportion with all-ties RF > paired complete-data RF + 1e-12; pooled version uses summed counts.','conditional proportion')
for event,desc in [('truth_absent','generating tree absent from the exhaustive optimum set'),('nonzero_rf','nonzero mean normalized RF'),('mpt_differs_full','optimum set differs from the complete matrix optimum set'),('rf_worse_than_full','RF exceeds paired complete-matrix RF by >1e-12')]:
 add('baseline_'+event+'_probability','Unconditional fraction of all saved draws with '+desc+'.','proportion')
 add('D_exact_match_'+event+'_joint_probability','Fraction of all saved draws having both exact realized-D match and '+desc+'.','joint proportion')

COUNT={'n_tips':'tips','ntips':'tips','n_characters':'characters','nchar':'characters','n_complete_anchors':'complete anchor characters','n_targets':'hidden target cells','n_target_cells':'target cells','target_cells':'hidden target cells','group_n_targets':'target cells in the named group','n_train':'observed training cells','n_test':'hidden test cells','n_matrices':'independent matrices','n_independent_matrices':'independent matrices','n_paired_matrices':'paired independent matrices','n_response_characters':'response characters in the named group','n_selected_completions':'selected completions in this matrix/treatment','replicates':'independent matrices per scenario','n_draws':'saved completion draws','draws':'saved completion draws per scheme','draws_per_matrix':'saved completion draws per matrix/method','topology_draws':'oracle six-tip completion draws','missing_per_response':'hidden tips per response character','hidden_tips_per_response':'hidden tips per response character','missing_per_character':'hidden tips per character','k_missing':'hidden tips per character','response_fits':'response-character fit records'}
for k,v in COUNT.items():add(k,'Number of '+v+'.','count')

# Execution/model metadata retained in JSON and compressed search records.
add('maxiter|max_accepted_moves|max_strict_NNI_steps_per_start','Configured optimizer/search iteration or accepted-move ceiling.','count of iterations/moves')
add('xatol','Absolute optimizer tolerance for the bounded log-rate search.','log-rate units')
add('bounds','Optimization bounds on natural-log Mk rate.','log-rate units')
add('root','Computational tree-root vertex ID.','node identifier')
add('tree_index','Zero-based index in the exhaustive tree registry.','tree index')
add('train_indices','Zero-based tip indices with observed response labels used for fitting.','tip index')
add('train_states','Observed binary response states aligned with train_indices.','binary states')
add('tip_edge_distances','Pairwise tip-to-tip graph distances measured in unit generating edges.','edge count')
add('map_tie_tolerance','Absolute probability tolerance for treating marginal predictions as tied; ties choose 0.','probability')
add('refined_negative_loglik','Negative training log likelihood at the refined Mk rate.','natural-log likelihood')
add('negative_branches','Number of negative NJ branch estimates before truncation.','count of branches')
add('truncated_branches','Number of NJ branch estimates floored for likelihood calculations.','count of branches')
add('zero_overlap_pairs','Number of taxon pairs lacking jointly observed predictor characters.','count of taxon pairs')
add('knn_no_overlap_training_rows_per_target','For each target, number of candidate training rows sharing no observed predictor.','count of training rows')
add('rf_denominator','RF normalization denominator, 2*(n_tips-3).','count of split differences')
add('generating_length|nj_endpoint_length|start_length','Ordinary parsimony length of the named generating/start/NJ-endpoint tree.','parsimony steps')
add('estimated_CPU_seconds_600','Engineering projection of CPU time for 600 matrices, not a completed execution time.','CPU seconds')
add('brier_new','Brier loss recomputed by the distribution-analysis verification.','dimensionless')
add('log_loss_new','Natural-log loss recomputed by the distribution-analysis verification.','nats')
add('start_seed_sequence','NumPy SeedSequence components for a recorded random starting tree.','integer seed components')
add('maxiter','Maximum optimizer iterations.','count of iterations')

add('zero_overlap_fallback|all_unknown_fallback','Pairwise NJ-distance fallback: mean defined off-diagonal mismatch proportion, or 0.5 when no pair is defined.','mismatch proportion')
add('RF','Normalized split symmetric-difference distance to the generating topology.','normalized RF, 0 to 1')
add('draw_indices|pattern_tip_order','Ordered zero-based saved-draw or tip indices used by the named module.','index')
add('scenario_ordinal','Zero-based position of the scenario in the frozen registry.','index')
add('hidden_per_response','Design values for hidden tips per response character.','count of tips')
add('warnings|failures|failed_matrices|substantive_defects_found','List of recorded warning/failure/defect messages or matrix identifiers; an empty list records none.','message/identifier list')
for k in '''D_partition_rows baseline_design_records step_limits complete_anchors completed_matrices constant_response_fits characters independent_matrices_per_scenario matrices scenarios draws_per_distribution draws_per_scheme_matrix exact_character_scores_compared forecast_rows independent_distance_and_fallback_cases independently_rescored_retained_endpoints input_treatment_hashes_reconstructed logistic_convergence_warnings matrices_per_scenario matrix_derived_NJ_edges_reproduced matrix_rows minimum_complete_anchor_columns mk_optimizer_calls mk_optimizer_failures mk_rate_boundaries new_simulations_or_draws ntrees partial_patterns production_masked_inputs_checked propagation_draws propagation_schemes registry_parsimony_scores registry_split_sets replicates_per_scenario responses retained_endpoint_count_total rows scenario_rows source_matrices all_partial_patterns toy_tips split_sets_compared state_partition_rows stored_endpoint_scores_rescored topology_scenarios trees unique_inputs_checked workers'''.split():
 add(k,'Recorded count of '+k.replace('_',' ')+'.','count')

add('source_id','Identifier for the third-party dependency/source notice.','identifier')
add('source_url','Recorded public source or upstream licence URL.','URL')
add('evidence_file','Package-relative source/documentation file supporting this provenance statement.','relative path')
add('licence_as_recorded','Original licence information recorded for the external source; no additional rights are inferred.','licence text')
add('package_grant_applies','Whether this release\'s own licence grant applies to the named item.','permission scope text')
add('note','Explanation of source provenance or licence scope.','text')
add('scope','Description of the scope of the recorded design, provenance or permission statement.','text')

# Known NPZ array encodings; array shape is also enumerated from every archived member.
NPZ={
'X':('Masked binary character matrix; -1=unknown, 0/1=observed state.','categorical states','tip x character'),
'X_true':('Complete generated true tip-state matrix.','binary states','tip x character'),
'node_states':('Generated states at every tree node.','binary states','node x character'),
'edges':('Pairs of zero-based graph vertex IDs; first n_tips vertices correspond to ordered tips.','node identifiers','edge x endpoint(2)'),
'edge_lengths':('Generating tree edge lengths (unit branches in the benchmark).','generating branch-length units','edge'),
'root':('Computational root vertex ID; benchmark root=48.','node identifier','scalar'),
'mask_anchor':('Anchor tip for structured masking; -1 for other masking rules.','tip index','scalar'),
'probabilities':('Predictive categorical probabilities; method order mode,knn5,logistic,extra_trees,nj_mk; state order 0,1.','probability','method x tip x character x state'),
'target_indices':('Row-major argwhere(X == -1); positions of hidden cells.','zero-based indices','target x (tip,character)'),
'missing_states_packed':('Packed binary completion states; NumPy big bit order; unpack count=len(target_indices); scheme order in REPRODUCIBILITY.','packed bits','scheme x draw x packed target-byte'),
'D_samples':('Saved completion D=(L-old_L)-(M-old_M) on generating scaffold.','parsimony steps','scheme x draw'),
'L_samples':('Saved completion ordinary parsimony length on generating scaffold.','parsimony steps','scheme x draw'),
'M_samples':('Saved completion sum of minimum character lengths.','parsimony steps','scheme x draw'),
'hard_D':('Cellwise-MAP fixed-scaffold D; joint Mk shares marginal Mk.','parsimony steps','scheme'),
'full_pattern_ids':('Complete binary character patterns; bit j is tip j.','bit-encoded state pattern ID','independent matrix x character'),
'missing_mask_bits':('Bit j=1 marks a missing tip in this character.','bit-encoded missingness mask','independent matrix x character'),
'observed_state_bits':('Complete pattern ID with missing bits cleared; use with missing_mask_bits; zero is not missing.','bit-encoded observed-state pattern','independent matrix x character'),
'cluster_anchors':('Anchor tip for clustered oracle masking; -1 for random masking.','tip index','independent matrix'),
'map_pattern_ids':('Cellwise marginal MAP binary patterns; identical over three methods; ties within 1e-12 choose 0.','bit-encoded state pattern ID','independent matrix x method x character'),
'matrix_D_pmfs':('Normalized exact PMF of D; final-axis index is D value; zero padding up to nchar*(floor(n/2)-1).','probability','independent matrix x method x D-support index'),
'missing_tip_indices':('Ascending missing-tip IDs for each matrix and character.','zero-based tip indices','independent matrix x character x missing tip'),
'oracle_missing_p1':('Exact true joint-model marginal probability that each missing tip is state 1.','probability','independent matrix x character x missing tip'),
'sharpened_missing_p1':('Power-two sharpened state-1 marginal probabilities.','probability','independent matrix x character x missing tip'),
'draw_pattern_ids':('Saved six-tip topology-completion patterns; method order oracle_joint,product_marginals,sharpened_product.','bit-encoded state pattern ID','independent matrix x method x draw x character'),
'draw_D':('Fixed-generating-tree D for the identical saved topology completion draws.','parsimony steps','independent matrix x method x draw'),
'costs':('Exact unordered binary parsimony scores for every six-tip tree and ternary partial pattern; ternary digit 2=unknown.','parsimony steps','tree(105) x ternary pattern(729)'),
'full_to_ternary':('Map complete binary pattern ID to corresponding ternary-pattern registry ID.','pattern index','binary pattern(64)'),
}

def semantic(field,fam,fmt,stats):
 leaf=field.split('.')[-1].replace('[*]',''); key=leaf
 if field.startswith('existing_metric_maximum_errors.'):
  units={'brier_new':'dimensionless Brier-score discrepancy','log_loss_new':'nats','signed_bias':'parsimony steps','truth_in_95':'absolute difference of 0/1 inclusion flags','width95':'parsimony steps'}
  return 'Maximum absolute discrepancy between the recomputed '+key+' and its corresponding archived original metric, across all checked matrix/scheme rows; verification residual, not the raw metric.',units.get(key,'units of the compared metric'),'scalar maximum over checked rows'
 if key=='maximum_score_difference':return 'Maximum absolute parsimony-score discrepancy in the exhaustive six-tip registry comparison; recorded zero after exact equality checks.','parsimony steps','scalar maximum over tree/pattern comparisons'

 if fam.endswith('Table_S28_compact.tsv') and ('_RF' in key):
  if key.endswith('_RF_change_from_incomplete'):return 'Display cell: selected-draw mean normalized-RF difference for '+key.split('_RF')[0]+' minus source-incomplete baseline, followed by paired MCSE in parentheses.','normalized RF difference','formatted mean (MCSE), scalar per scenario'
  return 'Display cell: mean normalized RF of the named baseline followed by its between-matrix MCSE in parentheses.','normalized RF, 0 to 1','formatted mean (MCSE), scalar per scenario'
 if key=='edges' and field.startswith('nj_models[*].'):return 'Fitted NJ tree edges: ordered triples (vertex_a,vertex_b,branch_length); lengths floored to 1e-8 where necessary.','node IDs, node IDs, inferred branch-length units','edge x (vertex_a,vertex_b,length)'
 if key=='edges' and all(t in ('integer','number') for t in stats['types']):return 'Number of fitted tree edges.','count of edges','scalar'
 if key=='nj_models':return 'Per-character fitted Mk model placeholder; null for complete anchor columns that require no response fit.','model object or null','character-indexed list element'
 if fmt=='npz':
  if 'oracle/topology/' in fam:
   if key=='truth_tree_index':return 'Index of the generating topology in the 105-tree registry.','zero-based tree index','scalar'
   if key=='rf_numerators':return 'Split-set symmetric-difference count of each enumerated topology versus generating tree; divide by 6 for normalized RF.','count of nontrivial split differences','tree ID(105)'
   if key=='ternary_patterns':return 'Partial six-tip patterns: 0/1 observed states, 2=unknown.','categorical states','ternary pattern ID(729) x tip(6)'
   if key=='minimum_steps':return 'Minimum possible parsimony length for each partial pattern across all 105 trees.','parsimony steps','ternary pattern ID(729)'
   prefix=key.split('_',1)[0];tail=key.split('_',1)[-1]
   ax={'full':'matrix','source':'matrix','map':'matrix x method','draw':'matrix x method x draw'}.get(prefix,'matrix')
   if tail=='minimum_tree_length':return 'Minimum ordinary parsimony length over the exhaustive 105-tree set for this treatment.','parsimony steps',ax
   if tail=='mpt_count':return 'Number of exhaustive tied most-parsimonious trees for this treatment.','count of trees',ax
   if tail=='mean_normalized_rf':return 'Normalized RF to the generating topology averaged over every tied most-parsimonious tree for this treatment.','normalized RF, 0 to 1',ax
   if tail=='truth_in_mpt':return 'Generating topology belongs to the exhaustive optimum set for this treatment.','boolean',ax
   if tail=='mpt_bits':return 'Membership of every tied most-parsimonious tree; little-endian packed bits, 14 bytes for 105 trees; unused final bits are padding.','packed membership bits',ax+' x packed tree-byte(14)'
   if tail=='tree_scores':return 'Ordinary parsimony length on each enumerated six-tip tree.','parsimony steps',ax+' x tree ID(105)'
   if key in ('truth_D','map_D','draw_D'):return 'Fixed-generating-tree D of the named truth/MAP/draw treatment.','parsimony steps',ax
   if tail=='absolute_D_error':return 'Absolute treatment D minus realized true_D.','parsimony steps',ax
  if key in NPZ:return NPZ[key]
  if key in ('old_L','old_M','true_L','true_M','true_D','source_L','source_M','marginal_map_D','q'):
   return (*D[key], 'independent matrix' if fam.startswith('oracle/') else 'scalar')
 if key=='mean_D_mcse_mean' and fam=='simulation/results/scenario_summary.csv':return 'Mean across 50 independent matrices of each matrix\'s finite-draw MCSE: SD(512 D draws)/sqrt(512).','parsimony steps','scenario x scheme'
 if key=='mean_D_mcse_mcse' and fam=='simulation/results/scenario_summary.csv':return 'Between-matrix MCSE of the 50 within-matrix finite-draw MCSEs: sample SD of [SD(512 D draws)/sqrt(512)] divided by sqrt(50).','parsimony steps','scenario x scheme'
 if key=='mean_D_mcse' and fam=='simulation/results/scenario_summary.csv':return 'Between-matrix MCSE of mean_D, computed as sample SD of the 50 matrix means divided by sqrt(50).','parsimony steps','scenario x scheme'
 if key in ('mean_difference','paired_MCSE'):
  if 'S11_summary' in fam:return ('Paired within-matrix MCSE of joint-minus-marginal Mk CRPS.','parsimony steps','scalar per scenario')
  return ('Mean paired difference, using the recorded direction.' if key=='mean_difference' else 'Sample SD of paired within-matrix differences divided by sqrt(n_paired_matrices).','units of metric column','scalar per comparison')
 if key in D:return (*D[key],'scalar per row' if fmt in ('csv','tsv') else 'scalar or nested field; observed shape supplied')
 # Summary suffixes: precedence preserves within-draw mean_D_mcse versus scenario summary.
 for suf,label in [('_n_valid','Number of independent matrices with a defined '),('_paired_MCSE','MCSE of paired within-matrix differences in '),('_difference_mcse','MCSE of paired candidate-minus-reference differences in '),('_difference_mean','Mean paired candidate-minus-reference difference in '),('_MCSE','Between-independent-matrix MCSE (sample SD/sqrt(n)) of '),('_mcse','Between-independent-matrix MCSE (sample SD/sqrt(n)) of '),('_se','Standard error across 100 independent oracle matrices of '),('_mean','Mean across independent matrices of '),('_n','Count of finite independent-matrix values for ')]:
  if key.endswith(suf):
   base=key[:-len(suf)];d,u,_=semantic(base,fam,fmt,stats)
   if suf in ('_n','_n_valid'):u='count of matrices'
   elif u in ('0/1 indicator','boolean'):u='proportion'
   if 'difference' in suf or suf.lower().endswith(('mcse','se')):u=u.replace(', 0 to 1','')
   if 'difference' in suf and 'proportion' in u:u='proportion difference' if suf.endswith('_mean') else 'proportion units (MCSE of paired differences)'
   if key=='mean_D_mcse' and fam.endswith('scenario_summary.csv'):d='mean_D (the within-matrix average D)';u='parsimony steps'
   if fam.startswith('oracle/topology/') and 'candidate-minus-reference' in label:label=label.replace('candidate-minus-reference','left-minus-right')
   return label+d,u,'scalar per summary row'
 if key in ('mean_difference','paired_MCSE'):
  if 'S11_summary' in fam:return ('Paired within-matrix MCSE of joint-minus-marginal Mk CRPS.','parsimony steps','scalar per scenario')
  return ('Mean paired difference.' if key=='mean_difference' else 'Sample SD of paired within-matrix differences divided by sqrt(n_paired_matrices).','units of metric column','scalar per comparison')
 if 'sha256' in key.lower() or key=='{sha256}':return 'SHA256 digest binding the named source/output bytes.','hexadecimal checksum','scalar string'
 if fmt=='npz':
  if re.fullmatch(r'n\d+_patterns',key):return 'Binary patterns; pattern ID bit j equals tip j state.','binary states','pattern ID x tip'
  if key.endswith('_full_steps'):return 'Ordinary parsimony length of each complete binary pattern on the tree named by the prefix.','parsimony steps','binary pattern ID'
  if key.endswith('_full_minima'):return 'Minimum possible unordered parsimony length for each complete binary pattern.','parsimony steps','binary pattern ID'
  if key.endswith('_probabilities'):return 'Exact unconditional probability of each complete binary pattern; q0=.02,q1=.1,q2=.5.','probability','binary pattern ID'
  if key.endswith('_mpt_bits'):return 'All most-parsimonious tree membership; little-endian packed bits, 14 bytes covering 105 trees.','packed membership bits',key.split('_')[0]+' treatment axes x packed tree-byte'
  if key.endswith('_scores'):return 'Parsimony length on each of the 105 enumerated labelled six-tip trees.','parsimony steps',key.split('_')[0]+' treatment axes x tree ID'
  if key.startswith(('full_','source_','map_','draw_')):
   base=key.split('_',1)[1]
   if base in ('mean_normalized_rf','min_normalized_rf','max_normalized_rf'):return 'Normalized RF to generating topology over all tied optimum trees; '+base.split('_')[0]+' over ties.','normalized RF, 0 to 1','matrix; MAP adds method; draw adds method x draw'
   if base=='mpt_count':return 'Number of tied global most-parsimonious trees among exhaustive 105-tree set.','count of trees','matrix; MAP adds method; draw adds method x draw'
   if base=='truth_in_mpt':return 'Whether the generating tree belongs to the exhaustive optimum set.','boolean','matrix; MAP adds method; draw adds method x draw'
 if key.endswith('_n_valid'):return 'Number of matrices where the conditional quantity is defined.','count','scalar'
 # JSON provenance and execution metadata.
 low=key.lower(); phrase=key.replace('_',' ').replace('{','').replace('}','')
 if key in ('seconds','elapsed_seconds') or '.timings.' in field:return 'Recorded elapsed runtime for the named operation.','seconds','scalar'
 if 'seed' in low:return 'Recorded deterministic random generator seed or NumPy SeedSequence components.','integer seed','ordered scalar/vector'
 if key in ('edges','start_edges','original_start_edges','nj_start_edges','generating_edges','tree_edges','truth_edges','retained_edges'):return 'Tree edges as pairs of node IDs; nested outer axes distinguish starting trees or retained endpoints.','node identifiers','edge x endpoint(2), with optional leading start/endpoint axis'
 if key=='nested':return 'Nested representation of the tree topology; terminal integers are zero-based tip IDs and nested lists group descendant subtrees.','tip identifiers and tree structure','ragged recursive tree'
 if key in ('splits','consensus_splits','best_splits','canonical_splits'):return 'Canonical nontrivial tree bipartitions, encoded by tip bit masks.','split bit masks','set/list of splits'
 if 'path' in low or key in ('source','output','file','{file_path}'):return 'Recorded path or filename for the named source/output; old absolute paths are provenance, not portable entry points.','path/text','scalar string'
 if 'version' in low or key in ('python','numpy','pandas','scipy','sklearn','scikit_learn','matplotlib','platform'):return 'Recorded software/environment version or platform identifier.','version/text','scalar string'
 if key in ('bytes','size_bytes'):return 'Size of the referenced file at recording time.','bytes','scalar integer'
 if key in ('score','score_trajectory','minimum_found_length','generating_tree_score'):return 'Ordinary unweighted unordered parsimony length; trajectory retains each accepted improving score.','parsimony steps','scalar or ordered search trajectory'
 if key in ('steps','candidate_evaluations','n_iter','nit','nfev','accepted_moves'):return 'Count of '+phrase+' for the named search/optimizer.','count','scalar or vector'
 if key in ('training_loglik','log_likelihood','loglik'):return 'Training-label log likelihood under the fitted Mk model.','natural-log likelihood','scalar'
 if key in ('refined_logq','logq'):return 'Natural logarithm of fitted Mk transition rate.','log rate','scalar'
 if key in ('observed_classes','classes','observed_labels','onehot_categories'):return 'Observed categorical training states or one-hot category labels; ? is missing-observation category.','categorical state labels','response/predictor/category lists'
 if key in ('observed_tips','target_tips','tip_order','tip_labels','tips','character_indices'):return 'Ordered tip/character labels or zero-based indices for the named data structure.','identifier/index','vector'
 if key in ('edge_lengths','branch_lengths'):return 'Branch lengths of the named generating/fitted tree.','branch-length units','edge'
 if key in ('n_tips','nchar','ntips') or low.startswith(('n_','count_')) or low.endswith(('_count','_counts')):return 'Recorded count of '+phrase+'.','count','scalar or labelled collection'
 if all(t in ('string','array[string]','blank') for t in stats['types']):return 'Recorded '+phrase+'; descriptive metadata preserved from the source document.','text/category','scalar or list'
 if ('boolean' in stats['types'] and all(t in ('boolean','null') for t in stats['types'])) or low.startswith(('is_','has_','any_')):return 'Truth value of the recorded '+phrase+' condition.','boolean','scalar'
 if 'probabilit' in low or low.endswith(('_p1','_mass')):return 'Recorded '+phrase+' in the named model/check context.','probability','scalar or ordered array'
 if 'error' in low or 'discrepanc' in low or 'difference' in low or low.startswith('max_'):return 'Recorded '+phrase+' for the named verification/check; interpretation follows the associated producer.','units of checked quantity; see producer','scalar or labelled collection'
 return 'Recorded '+phrase+' in '+field.rsplit('.',1)[0]+'; exact nesting and values are retained in the referenced file.','not a physical measurement; metadata/structural value','see observed shape and source'

def module(n):
 if n.startswith('oracle/simulation/'):return 'exact oracle simulation'
 if n.startswith('oracle/topology/'):return 'exhaustive six-tip topology'
 if n.startswith('oracle/'):return 'oracle design/validation'
 if n.startswith('topology_extension/'):return 'heuristic 48-tip topology'
 if n.startswith('distribution_analysis/'):return 'post hoc distribution diagnostics'
 if n.startswith('simulation/') or n.startswith('methods/'):return '600-matrix fitted benchmark'
 if n.startswith('validation/'):return 'benchmark numerical validation'
 if n.startswith('figures/'):return 'supplementary figures'
 return 'package documentation/environment'

def source(n):
 if n.startswith('documentation/') or n in ('DATA_DICTIONARY.tsv','FILE_CATALOG.tsv','REPRODUCIBILITY.txt'):return 'REPRODUCIBILITY.txt; documentation/build_indexes.py'
 if n=='THIRD_PARTY_NOTICES.tsv' or n.startswith('LICENS'):return 'LICENSING.txt; THIRD_PARTY_NOTICES.tsv'
 if n.startswith('oracle/simulation/'):return 'oracle/simulation/README.txt; oracle/simulation/exact_simulation.py'
 if n.startswith('oracle/topology/'):return 'oracle/topology/README.txt; oracle/topology/run_topology.py; oracle/topology/refresh_summaries.py'
 if n.startswith('oracle/'):return 'oracle/simulation/README.txt; oracle/topology/README.txt'
 if n.startswith('topology_extension/'):return 'topology_extension/README.txt; topology_extension/run_topology_extension.py; topology_extension/run_nj_amendment.py'
 if n.startswith('distribution_analysis/'):return 'distribution_analysis/README.txt; distribution_analysis/analyze_distributions.py'
 return 'README.txt; simulation/experiment.py; methods/fit_adapter.py; methods/mk_baseline.py'

def role(n,fmt):
 if n.startswith('documentation/'):return 'Documentation index generator; reads existing artifacts without fitting models or changing scientific outputs.'
 if n=='THIRD_PARTY_NOTICES.tsv' or n.startswith('LICENS'):return 'Release licence and third-party scope documentation.'
 if fmt=='py':return 'Executable analysis/validation/plotting source; see module README for intended entry point.'
 if fmt=='pdf':return 'Rendered supplementary figure; numeric inputs are catalogued separately.'
 if fmt in ('txt','md'):
  if 'SHA256' in n:return 'Integrity digest list; compare file bytes, not scientific findings.'
  if 'requirements' in n:return 'Pinned external package dependency versions.'
  return 'Human-readable documentation, caption, or frozen analysis plan.'
 if 'search_records' in n:return 'Per-matrix compressed search provenance, every start trajectory, retained endpoints, treatment metrics.'
 if '/truth/' in n:return 'Simulated truth reserved for evaluation.'
 if '/inputs/' in n:return 'Masked observed inputs to fitting.'
 if '/predictions/' in n:return 'Fitted probabilities/completion arrays or recorded model metadata.'
 if '/evaluation/' in n:return 'Saved evaluation D/L/M arrays or hash/seed receipt.'
 if 'matrix' in n or 'scenario' in n:return 'Scientific inputs/results or per-matrix/scenario summaries; use schema dictionary.'
 if 'registry' in n:return 'Index of design conditions, patterns, trees or exact scoring tables.'
 if fmt in ('json','json.gz'):return 'Design/provenance/check metadata; recorded checks describe original execution, not a new rerun.'
 return 'Numeric scientific table; row meaning and fields are documented by schema.'

rows=[];unknown=[]
for sid,s in sorted(I['schemas'].items(),key=lambda kv:kv[1]['family']):
 for field,st in sorted(s['fields'].items()):
  desc,unit,axes=semantic(field,s['family'],s['format'],st)
  if s['family'].endswith('matrix_topology_summary.csv') and s['family'].startswith('topology_extension/') and field not in ('matrix_id','scenario_id','replicate','q','missing_per_response','mask_mode','kind','scheme','n_selected_completions'):
   desc+=' In this table, average selected draw-level values within the same matrix/treatment; baseline/MAP has one completion.'
   if unit in ('0/1 indicator','boolean'):unit='proportion of selected completions'
  if s['format'] in ('csv','tsv') and s['family'].endswith('paired_method_contrasts.csv') and '_difference_' in field:unit=unit.replace(', 0 to 1','')
  if s['format'] in ('csv','tsv') and 'Recorded '==desc[:9]:unknown.append((s['family'],field))
  missing=[]
  if st['null']:missing.append('JSON null / missing parser value observed: '+str(st['null']))
  if st['nan']:missing.append('Non-finite numeric values observed: '+str(st['nan']))
  if st['empty']:missing.append('Blank fields or empty arrays/objects observed: '+str(st['empty']))
  if not missing:missing.append('No null/blank/non-finite value observed in this field; optional JSON keys need not occur in every object')
  if field in ('X','mask_anchor','cluster_anchors','draw_index'):missing.append('-1 sentinel: unknown character, non-applicable mask anchor, or baseline/MAP draw index as specified')
  if 'consensus_split_precision' in field:missing.append('Original search precision undefined when consensus has zero splits: JSON NaN / CSV blank')
  if 'given_D_exact_match' in field and not field.endswith('_n_valid'):missing.append('Conditional quantity undefined when there are no exact-D-matching draws; summary reports valid-matrix counts')
  if field in ('brier_within_group','log_loss_within_group'):missing.append('Blank when the cell group is empty')
  if field=='mk_optimizer_success':missing.append('not_run denotes optimization not attempted')
  rows.append({'schema_id':sid,'path_pattern':s['family'],'format':s['format'],'field':field,'definition':desc,'units':unit,'dtype_observed':'|'.join(st['types']),'shape_observed':' | '.join(st['shapes']),'axes_or_row_unit':axes,'missingness':'; '.join(missing),'scope':'All catalogue paths sharing schema_id; JSON [*]=list element, {sha256}/{index}/{file_path}=dynamic key; zero is a value unless explicitly a sentinel','source':source(s['family'])})
with (ROOT/'DATA_DICTIONARY.tsv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t');w.writeheader();w.writerows(rows)
cat=[]
for r in I['files']:
 n=r['path'];r=dict(r);r['bytes']='' if n.startswith('SHA256') else (ROOT/n).stat().st_size;r.update(module=module(n),role=role(n,r['format']),dictionary='DATA_DICTIONARY.tsv:'+r['schema_id'] if r['schema_id']!='NOT_TABULAR' else 'not applicable: prose/source/figure',source=source(n));cat.append(r)
# Final root additions are indexed even though their text does not have scientific fields.
known={r['path'] for r in cat}
for p in sorted(ROOT.iterdir()):
 if p.is_file() and p.name not in known:
  cat.append(dict(path=p.name,format=p.suffix[1:],schema_id='DOCUMENTATION',rows_or_members='',bytes='' if p.name=='FILE_CATALOG.tsv' else p.stat().st_size,module='package documentation/environment',role='Submission documentation/index; not a new scientific result.',dictionary='Column definitions in REPRODUCIBILITY.txt for catalogue/dictionary; otherwise prose.',source='REPRODUCIBILITY.txt'))
if 'FILE_CATALOG.tsv' not in {r['path'] for r in cat}:cat.append(dict(path='FILE_CATALOG.tsv',format='tsv',schema_id='DOCUMENTATION',rows_or_members='',bytes='',module='package documentation/environment',role='Exact relative-path index; self byte count intentionally blank.',dictionary='Column definitions in REPRODUCIBILITY.txt',source='REPRODUCIBILITY.txt'))
with (ROOT/'FILE_CATALOG.tsv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=['path','format','module','role','schema_id','dictionary','rows_or_members','bytes','source'],delimiter='\t');w.writeheader();w.writerows(sorted(cat,key=lambda r:r['path']))
print('Documentation generated:',len(cat),'catalogued paths;',len(I['schemas']),'schemas;',len(rows),'dictionary fields.')
