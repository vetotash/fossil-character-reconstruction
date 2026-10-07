ROUND 15 EXACT CONDITIONAL SIMULATION

Purpose
This controlled experiment asks what a fixed-tree D distribution can diagnose
when a generating tree and a binary symmetric Mk model are completely known.
It tests loss of conditional dependence and a deliberate marginal sharpening
perturbation. It is not a refit or ranking of the empirical classifiers, an
anatomical validation, or an estimate of complete phylogenetic uncertainty.

Reproduction
Python 3.12.14 and NumPy 2.3.5 were used. The scientific script needs only the
Python standard library and NumPy. Run from any directory using actual paths:

  python exact_simulation.py --self-test
  python exact_simulation.py --output NEW_EMPTY_OUTPUT_DIRECTORY \
      --frozen-design ../DESIGN_FREEZE.json

Existing output directories are refused. The design is hard-coded and also
written to design.json. No original workspace or previous project round is
needed. The root-level ANALYSIS_PLAN.txt and DESIGN_FREEZE.json preserve the
prospective within-session design decision, made after the earlier empirical
results were already known. It is not an external preregistration.

Execution
All 48 factorial scenarios and all 100 independent matrices per scenario were
retained. Each scenario has its own truth and masking RNG. Simulation base seed
is 2026100401; truth uses SeedSequence([base, scenario ordinal, 0]) and masking
uses SeedSequence([base, scenario ordinal, 1]). The 6-tip topology completion
draws use the separate base seed 2026100402 and stream 2. A matrix's three
probability conditions use common inverse-CDF uniforms, but these are not its
generating truth. Nothing is sampled to estimate the primary D distributions:
per-character conditional PMFs and matrix convolutions are exact to floating
point arithmetic. The 64 sampled completions only support the topology module.

The script was not changed after the production run started. Tiny pre-run
correctness checks used a three-tip example and did not inspect any factorial
scenario outcome. No technical correction or performance-based parameter
change was needed during the production run.

Data contracts
The three method axes always have this order:
  0 oracle_joint
  1 product_marginals
  2 sharpened_product

All pattern IDs encode tip j as bit j, with state 1 represented by a set bit.
Tip order is t0,t1,... . All stored matrices use characters as their last axis
unless specifically described below. Missing-tip vectors use ascending tip ID.

results/trees.json
  Dictionary tree_key -> complete labelled graph edges, root, nested tree,
  unit edge lengths, Newick, tip labels and graph distances. Every internal
  graph vertex has degree 3; the computational root has no special edge length.

results/pattern_registry.npz
  n6_patterns [64,6] and n10_patterns [1024,10].
  <tree_key>_full_steps and <tree_key>_full_minima [2**n].
  <tree_key>_q<i>_probabilities [2**n], q0=.02, q1=.1, q2=.5.

results/scenario_registry.csv
  All 48 scenarios. missing_per_character is k; mask_mode is random/clustered.
  n_tips=6 has 32 independent characters; n_tips=10 has 64.

results/sXX_matrices.npz
  full_pattern_ids [100,nchar]: independently generated complete character IDs.
  missing_mask_bits [100,nchar]: a set bit marks a missing tip.
  observed_state_bits [100,nchar]: full_pattern_ids with missing bits cleared;
    this integer alone is NOT a complete character and needs its missing mask.
  cluster_anchors [100]: anchor tip for clustered masking, -1 for random masks.
  map_pattern_ids [100,3,nchar]: identical across three methods, the common
    exact marginal MAP with numerical ties within 1e-12 assigned to state 0.
    This is never the joint MAP.
  matrix_D_pmfs [100,3,width]: complete normalized matrix D probabilities,
    padded with zeroes through nchar*(floor(n/2)-1).
  true_D [100], source_L [100], source_M [100], marginal_map_D [100].
  missing_tip_indices [100,nchar,k]: ascending indices for each target vector.
  oracle_missing_p1 [100,nchar,k]: exact single-tip probabilities of state 1.
  sharpened_missing_p1 [100,nchar,k]: power-two sharpened probabilities.
  draw_pattern_ids [100,3,64,nchar]: six-tip only, downstream completions.
  draw_D [100,3,64]: exact fixed-generating-tree D for those same completions.

results/matrix_results.csv
  14,400 rows, one matrix x probability condition. Contains D mean/variance,
  discrete central .025/.975 quantiles, realized truth D, coverage indicator,
  TRUE conditional probability mass in each condition's interval, and the
  condition's own interval mass. The mean-minus-oracle endpoint isolates a
  conditional propagation discrepancy without sampling the generating truth.
  D_total_variation_from_oracle compares the two D distributions, not their
  underlying full-character joint distributions. D is a many-to-one summary.

  Brier is the sum of two squared class-probability errors, averaged over
  missing cells. oracle_expected_brier integrates that score under the TRUE
  conditional marginal. realized_accuracy is common across three conditions;
  oracle_expected_accuracy evaluates the common decision under the oracle.
  forecast_mean_max_probability is a forecast's own confidence, not necessarily
  its true expected accuracy. mean_marginal_entropy_bits is binary entropy.
  oracle_total_correlation_bits_per_character is the sum of missing-tip
  marginal entropies minus their joint entropy, conditional on observed tips.

results/scenario_summary.csv
  144 rows. Every numeric matrix endpoint has a mean and standard error over
  100 INDEPENDENT MATRICES; completion draws are not independent replicates.
  Realized coverage additionally has a Wilson 95% interval over those 100
  generating realizations. The exact conditional probability masses provide
  a separate, less noisy assessment of the interval's conditional meaning.

results/topology_input_manifest.json
  Paths and generating edges for all 24 six-tip scenario files. This is the
  input contract for the separately implemented exhaustive 105-tree module.

RUN_CHECKS.json is the producer's internal check, not the independent audit.
MANIFEST.json records result hashes and byte counts at production completion.
The manuscript release additionally requires the independent audit artifacts.

Interpretation boundaries
The oracle's central interval contains at least .95 true conditional mass by
construction, with discrete conservatism. This is a numerical control, not a
scientific discovery. An observed coverage below .95 across 100 matrices is
possible even for this exact oracle, and is not alone evidence of miscalibration.
Correct marginal probabilities do not determine a joint completion law. The
product and oracle have identical Brier scores and hard decisions, but can
induce different D distributions. A D match still need not imply joint or
topological correctness. The perturbation does not identify the mechanism of
any fitted historical classifier. Root parameters, tree, character independence
and state-independent masks are imposed, not estimated from fossil evidence.
