DISTRIBUTION SCORES AND ERROR COMPONENTS FOR THE 600-MATRIX BENCHMARK

Purpose
Reanalyse the frozen predictions, sampled completions and excess-parsimony
costs in the supplied simulation archive. The script does not generate data,
fit predictors, alter probabilities, or change sampled states. The additions
are post hoc. The 180 historical-layout simulations are outside this analysis.

Files
analyze_distributions.py: analysis and source-array checks.
test_distribution_analysis.py: six independent numerical/behavioural tests.
Methods and results are reported in Supplementary Information S11.
S11_summary_table.csv: compact 12-condition summary for Supplementary
Information; the complete 72 scheme-by-condition summary is in results/.

Requirements
Python 3.10 or later, NumPy and pandas. The completed run used Python 3.12,
NumPy 2.3.5 and pandas 2.2.3. The source archive must contain simulation/results
and validation/independent_engine.py. No fitted-model dependencies are needed
because saved predictions are read directly. The scorer is loaded from source
without creating cache files inside the source archive.

Run from this directory, substituting the actual source archive path:
  python -m unittest discover -s . -p test_distribution_analysis.py -v
  python analyze_distributions.py --root /path/to/Simulation_Supplementary_Code_and_Data --output /path/to/new_results --workers 3

The output directory must be empty or new and must lie outside the source
archive. A numerical engineering subset can be run with --limit 4, but the
scientific outputs supplied here used all 600 matrices. Source files are read
only. The completed analysis checked hashes, reconstructed every saved D draw
with independent Fitch scoring, and matched the original summary metrics.
No new fitted predictions were calculated.

Conventions
Scheme order: mode, knn5, logistic, extra_trees, nj_mk, nj_mk_joint. mode denotes
observed-state frequencies. nj_mk_joint shares cell probabilities with nj_mk.
Each condition has 50 independent matrices. Every predictive distribution has
512 draws. q is the generating rate; missing_per_response is 14 or 29; masks
are random, structured, or statebiased. Statebiased favours hiding state 1.
All costs and CRPS values are in parsimony steps. Smaller CRPS is better.

Primary CRPS of the saved empirical distribution:
  (1/M) sum_i |d_i-y| - (1/(2*M^2)) sum_i sum_j |d_i-d_j|.
crps_fair_ensemble uses M*(M-1) instead of M^2 as a secondary finite-M estimate
for independent Monte Carlo draws. It is a different estimand from the score
of the saved empirical forecast. All principal statements use crps_empirical.
The primary and secondary scores gave the same scenario-level mean rankings.

Output files and columns

1. matrix_distribution_scores.csv
One row per matrix and original scheme (3,600 rows).
Identifiers: matrix_id, scenario_id, replicate, q, missing_per_response,
mask_mode, scheme. n_targets is the number of hidden cells and n_draws is 512.
true_D is the target-truth completion's change in excess parsimony cost.
mean_D is the saved-draw mean. signed_bias = mean_D-true_D.
absolute_mean_error = abs(signed_bias). sd_D is the draw SD.
lower95 and upper95 are linear-interpolated 0.025 and 0.975 quantiles;
width95 is their difference. truth_in_95, truth_below_95 and truth_above_95
are mutually exclusive 0/1 indicators with inclusive interval endpoints.
crps_empirical and crps_fair_ensemble use the formulas above.
brier is the average TWO-CLASS-SUM Brier loss. log_loss uses natural logarithms
and a true-state probability floor of 1e-12. error_rate is the target error
fraction. training_unseen_fraction counts target truths absent from old
training labels; constant_target_fraction counts targets in characters with
one observed training state; zero_true_probability_fraction counts exactly
zero saved probabilities before flooring. constant_response_count counts
single-state training response characters.

2. scenario_distribution_scores.csv
One row per scenario and scheme (72 rows). For each metric, *_mean is the
mean across independent matrices and *_MCSE is sample SD/sqrt(n). *_n reports
the number of finite matrix-level values. n_matrices is 50. inclusion_count
and inclusion_Wilson_low/high describe counts out of 50 and their Wilson
proportion intervals. Draws are never used as independent replicates.

3. paired_method_differences.csv
All 15 scheme pairs in every scenario for the reported distribution and state
metrics. mean_difference = candidate-reference within the same matrix, then
averaged. paired_MCSE is SD of those differences/sqrt(50).
n_paired_matrices is 50. Negative CRPS differences favour the candidate.

4. matrix_state_loss_contributions.csv
Four disjoint target-cell groups per matrix and scheme (14,400 rows):
  training_unseen_state;
  constant_training_seen_state;
  nonconstant_training_zero_true_probability;
  nonconstant_training_positive_true_probability.
group_n_targets and target_fraction describe group size. error_count,
brier_sum and log_loss_sum are additive totals. *_all_targets columns divide
the group totals by ALL targets in the matrix, not only the group. They sum
across groups to the full-matrix metric. brier_within_group and
log_loss_within_group divide by group size and are blank for empty groups.
zero_true_probability_count is counted before the log floor.
scenario_state_loss_contributions.csv gives means and MCSEs across matrices;
it retains zero-contribution groups rather than discarding them.

5. matrix_D_bias_components.csv
Four disjoint CHARACTER groups per matrix and scheme (14,400 rows):
  constant_training_with_unseen_targets;
  constant_training_without_unseen_targets;
  nonconstant_training_with_zero_probability_targets;
  nonconstant_training_without_zero_probability_targets.
n_response_characters gives group size. true_D_component and mean_D_component
sum characterwise truth and mean costs. signed_bias_contribution is their
difference and sums across groups to the full matrix's signed_bias.
scenario_D_bias_components.csv gives group means and MCSEs across matrices.
This accounting does not attribute a character's entire error causally to
one cell. Total CRPS and inclusion do not decompose additively in this way.

6. matrix_oracle_centered_diagnostic.csv and
   scenario_oracle_centered_diagnostic.csv
These are SEPARATE, truth-dependent diagnostics. For each matrix,
applied_shift = -(original mean_D-true_D) is added to every saved D draw.
The same matrix's evaluation truth therefore fixes the new mean. Range width
is unchanged; the translated values need not be attainable original costs.
crps_change_from_original and inclusion_change_from_original compare this
diagnostic with its original distribution. These are not valid forecasts,
calibrated uncertainty, or demonstrated out-of-sample improvements.

7. analysis_manifest.json
Records software versions, source and script hashes, per-matrix verification,
and maximum discrepancies from the original metric table. It distinguishes
the complete 600-matrix analysis from an engineering subset.

Completed checks
Six new unit checks passed, including hand calculations, independent CDF
integration, direct pairwise CRPS, group partitioning and paired MCSEs. The
source independent scorer's exhaustive 81-pattern toy check passed. All 600
matrices and all 3,600 saved distributions were read and reconstructed. Every
saved D draw was matched exactly. Differences from published CSV metrics were
zero for signed bias and inclusion, at most 7.2e-15 for width and at most
9e-16 for state losses. The analysis did not refit any of the 600 matrices.
