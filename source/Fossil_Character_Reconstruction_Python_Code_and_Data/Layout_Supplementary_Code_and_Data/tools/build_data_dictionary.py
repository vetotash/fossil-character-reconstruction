#!/usr/bin/env python3
"""Rebuild the complete file catalogue and field dictionary from packaged files.

Run after adding scientific outputs, before rebuilding MANIFEST.json. No scientific
files are changed. CSV types describe the actual export; NPZ types/shapes come
from NPY headers without unpickling. JSON paths use [*] for list elements. Empty
lists retain their container entry; no element schema is invented. Units and
scientific definitions below were checked against the shipped analysis source.
Opaque provenance fields retain their source key without invented semantics.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import re
import zipfile
import numpy as np

DHEADER = ['file_path', 'field_or_json_path', 'format', 'stored_type', 'shape_or_extent',
           'unit', 'missing_values', 'definition', 'source_or_scope']
CHEADER = ['file_path', 'format', 'role', 'dictionary_entries', 'scope']
OUTPUTS = {'DATA_DICTIONARY.tsv', 'FILE_CATALOG.tsv'}

# Shared field definitions. Dimensionless quantities use proportions, not percent.
DEFS = {
 'scenario_id': ('identifier', 'Simulation condition identifier s00 through s05 in the layout experiment; use the accompanying registry.'),
 'condition_id': ('identifier', 'Simulation condition identifier; interpreted within its experiment and registry.'),
 'matrix_id': ('identifier', 'Matrix/arm identifier; scenario, replicate and arm are encoded in the original IDs.'),
 'original_matrix_id': ('identifier', 'Original simulation matrix/arm identifier retained across downstream analyses.'),
 'pair_id': ('identifier', 'Latent matrix identifier shared by the paired observation-layout arms.'),
 'latent_id': ('identifier', 'Independent latent matrix identifier shared by paired arms.'),
 'panel': ('category', 'Historical observation-layout template: tetrapods, dinosaurs or ichthyosaurs.'),
 'arm': ('category', 'empirical = original observation labels; permuted = full O/U/T column permutation; fixed_target_permuted = O/U permutation outside fixed T.'),
 'scheme': ('category', 'Completion scheme; order in arrays: mode, knn5, logistic, extra_trees, nj_mk, nj_mk_joint.'),
 'schemes': ('ordered categories', 'Order of six completion schemes: mode, knn5, logistic, extra_trees, nj_mk, nj_mk_joint.'),
 'schemes_json': ('JSON-encoded categories', 'JSON string giving the order of schemes in the associated D_samples array.'),
 'method': ('category', 'Predictor, estimator or optimiser name in the enclosing record.'),
 'candidate': ('category', 'Candidate scheme in a candidate-minus-reference paired contrast.'),
 'reference': ('category', 'Reference scheme in a candidate-minus-reference paired contrast.'),
 'q': ('rate per unit branch length', 'Generating binary Mk rate; 0.03 or 0.3 for the layout experiments; not calendar time.'),
 'rates': ('rate per unit branch length', 'Generating Mk rates in the experiment configuration.'),
 'mk_rate': ('rate per fitted branch-length unit', 'Estimated Mk rate on the fitted neighbour-joining scaffold.'),
 'rate': ('rate per fitted branch-length unit', 'Estimated Mk rate in fitted model metadata; null for a constant-response fallback.'),
 'replicate': ('zero-based index', 'Independent latent replicate index within a simulation condition.'),
 'replicates': ('matrices per condition', 'Number of independent latent replicates specified for a condition.'),
 'scenario_ordinal': ('zero-based index', 'Scenario position in the generation configuration.'),
 'panel_ordinal': ('zero-based index', 'Historical panel position used in deterministic seed sequences.'),
 'q_ordinal': ('zero-based index', 'Generating-rate position used in deterministic seed sequences.'),
 'arm_ordinal': ('zero-based index', 'Arm seed index: 0 empirical, 1 full permutation, 2 fixed-target permutation.'),
 'condition_order': ('zero-based index', 'Plot ordering of conditions.'),
 'condition_label': ('text', 'Display label for a plotting condition.'),
 'randomisation': ('category', 'Control shown in Figure 2: full label permutation or fixed-target O/U permutation.'),
 'experiment': ('category', 'Source experiment; distinguishes the 600-matrix reference benchmark from the 180-matrix layout study.'),
 'n_tips': ('taxa', 'Number of terminal taxa in the matrix/scaffold.'),
 'n_taxa': ('taxa', 'Number of taxa.'),
 'n_characters': ('characters', 'Number of character columns in the relevant matrix.'),
 'n_cells': ('cells', 'Total number of matrix cells.'),
 'n_targets': ('cells', 'Number of target cells to be predicted in the relevant matrix or character record.'),
 'n_historical_targets': ('cells', 'Historical target cells for the response character.'),
 'target_columns': ('characters or zero-based column indices', 'CSV scalar: count of target-containing columns; array/list: zero-based indices of target-containing columns.'),
 'target_characters': ('characters', 'Number of characters containing historical targets.'),
 'target_taxa': ('taxa', 'Number of taxa containing historical targets.'),
 'n_draws': ('draws per scheme', 'Number of completion draws; 512 in the saved experiments.'),
 'n_matrices': ('independent matrices', 'Independent matrices contributing to the within-condition summary.'),
 'n_pairs': ('independent paired matrices', 'Independent latent matrix pairs contributing to a paired contrast.'),
 'n_paired_matrices': ('independent paired matrices', 'Independent latent matrix pairs contributing to a paired contrast.'),
 'n_train': ('taxa', 'Rows with an observed response state used for fitting.'),
 'n_test': ('taxa', 'All rows with a hidden response state; may include unavailable cells that are not historical targets.'),
 'observed_training_rows': ('taxa', 'Observed response rows used for the character fit.'),
 'tree_name': ('identifier', 'Name of the fixed historical topology/scaffold in tree_topologies.json.'),
 'difference_direction': ('text', 'Explicit subtraction order for contrasts; retained verbatim in the table.'),
 'direction': ('text', 'Explicit subtraction order for contrasts; retained verbatim in the table.'),
 'metric': ('category', 'Name of the response metric; its definition and unit determine the associated contrast units.'),
 'mean_difference': ('same as metric column', 'Mean within-latent-matrix difference, with subtraction direction given in the same record.'),
 'difference': ('same as metric column', 'Within-latent-matrix difference in the named metric.'),
 'paired_MCSE': ('same as metric column', 'Sample SD of within-matrix differences divided by sqrt(number of independent paired matrices).'),
 'accuracy': ('proportion', 'Fraction of target cells whose argmax state equals the latent true state; argmax breaks ties by first state.'),
 'error_rate': ('proportion', 'Fraction of target-cell argmax predictions different from the true state; 1 minus accuracy.'),
 'brier': ('dimensionless, range 0 to 2', 'Mean across target cells of sum over binary states of (predicted probability minus one-hot truth)^2; both state terms are included.'),
 'log_loss': ('natural-log units', 'Mean negative log predicted probability of the true target state, with the scoring code probability floor applied.'),
 'zero_true_probability_count': ('cells', 'Target cells assigned exactly zero probability to the true state before log-loss flooring.'),
 'log_floor_count': ('cells', 'Target probabilities requiring the log-loss floor.'),
 'mean_max_probability': ('probability', 'Mean over target cells of the largest predicted state probability.'),
 'old_L': ('parsimony steps', 'Total parsimony length of the masked matrix on its fixed scaffold before target completion.'),
 'old_M': ('steps', 'Sum of minimum required changes from observed state counts before completion.'),
 'true_L': ('parsimony steps', 'Length after revealing only target cells with their latent true states; other unavailable cells remain unknown.'),
 'true_M': ('steps', 'Minimum required changes after revealing only the true target states.'),
 'true_D': ('excess parsimony steps', 'Target-truth change in excess cost: (true_L-old_L) - (true_M-old_M); arm-specific.'),
 'hard_D': ('excess parsimony steps', 'D after filling target cells with argmax states; array follows the six-scheme order.'),
 'hard_D_error': ('excess parsimony steps', 'hard_D minus true_D.'),
 'hard_D_absolute_error': ('excess parsimony steps', 'Absolute value of hard_D minus true_D.'),
 'mean_D': ('excess parsimony steps', 'Arithmetic mean of the saved 512 D completion draws.'),
 'mean_D_error': ('excess parsimony steps', 'Mean of D draws minus true_D.'),
 'mean_D_absolute_error': ('excess parsimony steps', 'Absolute value of mean(D draws) minus true_D.'),
 'mean_D_error_per_100_targets': ('excess steps per 100 target cells', '100 times (mean D minus true_D) divided by number of target cells.'),
 'mean_D_absolute_error_per_100_targets': ('excess steps per 100 target cells', '100 times absolute(mean D minus true_D) divided by number of target cells.'),
 'sd_D': ('excess parsimony steps', 'Sample standard deviation of D draws, using ddof=1.'),
 'mean_D_mcse': ('excess parsimony steps', 'In a matrix-results row, sample SD of 512 D draws divided by sqrt(512); in a scenario-summary row, MCSE across independent matrix mean_D values.'),
 'lower95': ('excess parsimony steps', '2.5% quantile of D draws, using NumPy linear interpolation.'),
 'upper95': ('excess parsimony steps', '97.5% quantile of D draws, using NumPy linear interpolation.'),
 'interval_width': ('excess parsimony steps', 'upper95 minus lower95.'),
 'width95': ('excess parsimony steps', 'upper95 minus lower95.'),
 'truth_in_interval': ('indicator 0/1', '1 when lower95 <= true_D <= upper95, otherwise 0.'),
 'truth_in_95': ('indicator 0/1', '1 when lower95 <= true_D <= upper95, otherwise 0.'),
 'truth_below_interval': ('indicator 0/1', '1 when true_D < lower95.'),
 'truth_below_95': ('indicator 0/1', '1 when true_D < lower95.'),
 'truth_above_interval': ('indicator 0/1', '1 when true_D > upper95.'),
 'truth_above_95': ('indicator 0/1', '1 when true_D > upper95.'),
 'crps_empirical': ('excess parsimony steps', 'CRPS of the actual finite saved distribution: mean|D-y| - (1/(2 m^2)) sum_ij |D_i-D_j|.'),
 'crps_fair_ensemble': ('excess parsimony steps', 'Fair ensemble CRPS sensitivity: mean|D-y| - (1/(2 m(m-1))) sum_ij |D_i-D_j|; targets a different ensemble estimand.'),
 'signed_bias': ('excess parsimony steps', 'mean(D draws) minus true_D for one matrix and scheme.'),
 'absolute_mean_error': ('excess parsimony steps', 'Absolute value of mean(D draws) minus true_D.'),
 'inclusion_count': ('matrices', 'Number of matrices for which the central 95% D interval contains true_D inclusively.'),
 'inclusion_fraction': ('proportion', 'Fraction of independent matrices whose central 95% D interval contains true_D.'),
 'interval_coverage': ('proportion', 'Fraction of independent matrices whose central 95% D interval contains true_D.'),
 'coverage_mcse': ('proportion', 'Binomial Monte Carlo standard error sqrt(p*(1-p)/n) of the inclusion fraction.'),
 'constant_response_count': ('characters', 'Response characters using a constant-state fallback because one response class was observed.'),
 'constant_fallback': ('indicator or count', 'Whether a response fit used the single-observed-class fallback; aggregate tables sum events.'),
 'convergence_warning_count': ('warnings', 'Number of response fits with logistic-regression convergence warnings.'),
 'training_unseen_fraction': ('proportion of targets', 'Target cells whose true state did not occur in the observed response training labels.'),
 'constant_target_fraction': ('proportion of targets', 'Target cells belonging to a response character with a constant observed training state.'),
 'zero_true_probability_fraction': ('proportion of targets', 'Target cells assigned zero true-state probability before log-loss flooring.'),
 'taxon': ('label', 'Taxon label as retained in the source matrix.'),
 'row_zero_based': ('zero-based row index', 'Row index into the associated matrix/taxon names.'),
 'column_zero_based': ('zero-based column index', 'Column index into the associated character matrix.'),
 'character_one_based': ('one-based character number', 'Character number in the associated original source panel.'),
 'character': ('zero-based column index unless source record states otherwise', 'In fitting diagnostics: response column index into the analysed input matrix; source-evidence records retain their own character numbering.'),
 'taxa': ('taxa', 'Number of matrix rows contributing to the character summary.'),
 'characters': ('characters or one-based identifiers', 'CSV scalar: number of columns for a taxon summary; NPZ array: original one-based character identifiers.'),
 'simple_observed': ('cells', 'Source cells holding one unambiguous single state token.'),
 'unavailable': ('cells', 'Source cells not treated as simple observed states.'),
 'question': ('cells', 'Source cells represented by the literal unknown token ?.'),
 'nonsingleton_or_gap': ('cells', 'Source cells with a non-singleton token or gap, treated as unavailable in the binary layout experiment.'),
 'historical_targets': ('cells', 'Historical documentary target cells in the source/template scope.'),
 'is_target_character': ('boolean', 'Whether the character contains any historical target cell.'),
 'missing_fraction': ('proportion', 'Unavailable cells divided by total cells in the taxon or character summary.'),
 'historical_target_fraction': ('proportion', 'Historical target cells divided by all character cells in the taxon row.'),
 'original_ordered': ('boolean', 'Whether the character was ordered in the source panel; simulated characters are binary.'),
 'X': ('binary state code', 'Masked input matrix, axes taxa by characters: 0/1 observed state, -1 hidden/unavailable; all target cells are -1.'),
 'X_true': ('binary state code', 'Complete simulated latent terminal-state matrix, axes taxa by characters; values 0/1.'),
 'labels': ('observation-layout code', 'Matrix labels: 0 observed O, 1 unavailable U, 2 historical target T; not biological state codes.'),
 'target_mask': ('boolean', 'True at target cells T, false elsewhere; same row/column order as X.'),
 'observed_mask': ('boolean', 'True at simple observed O cells in the frozen source layout.'),
 'source_question_mask': ('boolean', 'True where the original source token is the literal ?.'),
 'nonsingleton_or_gap_mask': ('boolean', 'True where the original source cell is a non-singleton token or gap.'),
 'raw': ('source token', 'Raw source matrix tokens, taxa by characters; ? unknown, - gap, and bracketed sets retain source semantics.'),
 'names': ('taxon labels', 'Taxon labels in matrix row order.'),
 'original_column_indices': ('zero-based indices', 'Map of retained matrix columns to the original full source-panel columns.'),
 'original_ordered_characters': ('one-based character identifiers', 'Ordered character numbers retained from the original source panel.'),
 'ordered': ('one-based character identifiers', 'Ordered character numbers in the source panel; empty means no listed ordered characters.'),
 'admitted_predictor_columns': ('zero-based column indices', 'Source columns admitted to the layout predictor matrix.'),
 'admitted_predictor_characters': ('one-based character identifiers', 'Original character identifiers for the admitted predictor columns.'),
 'node_states': ('binary state code', 'Latent simulated states for all tree nodes by character, including internal nodes.'),
 'root': ('zero-based node index', 'Root node index used to orient a topology.'),
 'edges': ('node indices; fitted metadata also branch length', 'Latent topology arrays contain undirected endpoint pairs; nj_models metadata contains endpoint1, endpoint2, fitted branch length triples.'),
 'probabilities': ('probability', 'Array axes: five methods (mode, knn5, logistic, extra_trees, nj_mk), target cells, binary states [0,1]; nj_mk_joint shares nj_mk marginal probabilities.'),
 'target_indices': ('zero-based row,column indices', 'Array rows give target (taxon row, character column), in np.argwhere(target_mask) order.'),
 'missing_states_packed': ('packed binary bits', 'Axes: six schemes, 512 draws, ceil(n_targets/8) bytes. Decode np.unpackbits(array, axis=-1, count=n_targets, bitorder="big"); padding bits are not target data.'),
 'D_samples': ('excess parsimony steps', 'Axes six schemes by 512 draws; D=(L_samples-old_L)-(M_samples-old_M).'),
 'L_samples': ('parsimony steps', 'Length after each target-only completion; axes six schemes by 512 draws.'),
 'M_samples': ('steps', 'Minimum required state changes after each target-only completion; axes six schemes by 512 draws.'),
 'nj_models': ('list indexed by zero-based character', 'Fitted NJ/Mk model records used for joint completion; null at columns not fitted as responses.'),
 'train_indices': ('zero-based taxon indices', 'Observed response row indices used in the fitted model.'),
 'train_states': ('state labels', 'Observed response state tokens in train_indices order.'),
 'classes': ('state labels', 'Observed response classes in the stored probability/model order.'),
 'observed_classes': ('state labels', 'Distinct observed response states used for fitting.'),
 'onehot_categories': ('state tokens by predictor', 'Categories fitted by the one-hot encoder for predictor columns, including ? as a predictor category.'),
 'knn_no_overlap_training_rows_per_target': ('training rows', 'For each hidden response row, count of training rows with no jointly observed predictor.'),
 'n_iter': ('iterations', 'Recorded classifier iteration counts; null when estimator does not expose an iteration count.'),
 'nit': ('iterations', 'Optimiser iteration count reported by SciPy.'),
 'nfev': ('function evaluations', 'Optimiser objective evaluation count reported by SciPy.'),
 'training_loglik': ('natural-log units', 'Mk log likelihood on observed response states at the fitted rate.'),
 'refined_negative_loglik': ('natural-log units', 'Negative fitted Mk training log likelihood returned by the optimiser.'),
 'refined_logq': ('natural-log rate', 'Optimised natural logarithm of Mk rate.'),
 'draw_key': ('array key', 'Name of the draw array in the source NPZ.'),
 'truth_key': ('array key', 'Name of the true-value array in the source NPZ.'),
 'missing_per_response': ('cells per response character', 'Hidden cells per response in the separate 600-matrix reference benchmark.'),
 'mask_mode': ('category', 'Mask-generation pattern in the separate 600-matrix reference benchmark.'),
}

DEFS.update({
 'equal_at_1e-12': ('boolean', 'True when the stated numerical comparison passes with absolute and relative tolerances both 1e-12.'),
 'summary_values_checked': ('numeric summary entries', 'Number of condition-summary values independently recalculated from the original archived matrix-level rows and compared with the new summary.'),
 'largest_absolute_difference': ('native units of the checked numeric fields', 'Largest absolute numerical difference across the checked summary entries; when fields have different units this is a numerical consistency bound, not a pooled scientific effect.'),
 'original_condition_summary_file_available': ('boolean', 'Whether an original scenario_summary.csv was present in the archived package; False here, so its new counterpart is checked by independent aggregation of original matrix-level results.'),
 'executed_wrapper_source': ('package-relative path', 'Verbatim source snapshot of the wrapper that actually performed all fitting and scoring before the comparison-only correction.'),
 'scientific_fitting_or_scoring_code_changed': ('boolean', 'Whether the frozen scientific fitting or scoring functions changed in the documented postprocessing correction; False.'),
 'fits_recomputed_due_to_this_correction': ('fitted arms', 'Number of arm fits repeated solely to fix the comparison-stage missing-table assumption; zero.'),
 'scores_recomputed_due_to_this_correction': ('scored arms', 'Number of arm evaluations repeated solely to fix the comparison-stage missing-table assumption; zero.'),
 'original_archived_outputs_modified': ('boolean', 'Whether the correction changed original archived scientific outputs; False.'),
 'correction': ('text', 'Description of the comparator correction: two archived tables are directly compared and the new condition summary is independently checked from archived matrix rows.'),
 'reason': ('text', 'Explicit reason for the documented correction, exclusion or recorded event; it is explanatory metadata, not a scientific score.'),
 'stage': ('pipeline stage', 'Named execution stage at which the documented event occurred.'),
 'documented_excluded_original_members': ('file records', 'Original archive members explicitly excluded from the final package, with identifying hashes and reasons; scientific source/output files remain subject to unchanged-byte checks.'),
 'unexpected_missing_original_members': ('file paths', 'Original archive members absent without an explicit, individually validated exclusion; this list must be empty for the audit to pass.'),
 'missing_original_members': ('file paths', 'All original members absent from the final package, including any explicitly documented non-scientific temporary residue.'),
 'canonical_model_path': ('package-relative path', 'Path to the intact canonical fitted-model JSON retained in the package in place of an incomplete temporary duplicate.'),
})

DEFS.update({
 'status': ('status label', 'Outcome or recorded state of the enclosing operation/check; interpret the literal value together with its scope and explanatory fields.'),
 'scope': ('text', 'Explicit boundary of what the enclosing operation or verification checks; it is not a claim of unlisted computations.'),
 'metadata_note': ('text', 'Explanation of provenance or interpretation of the stored metadata, including distinctions between new execution records and lost originals.'),
 'records': ('record collection', 'Individual records whose fields are defined by the child JSON paths; their order and entity type follow the enclosing manifest or report.'),
 'tables': ('table-check records', 'Mapping from result-table filename to its row count, comparison reference and numerical-agreement result.'),
 'evaluation_arrays_equal': ('array-agreement flags', 'Map from each saved evaluation array name to an exact elementwise equality flag comparing newly computed and original archived arrays.'),
 'rows': ('rows', 'Number of rows in the named result table or checked output.'),
 'masked_inputs': ('input files', 'Number of masked matrix files whose identities and target-position bindings were verified.'),
 'truth_files': ('truth files', 'Number of complete latent-truth files with verified identities.'),
 'new_evaluation_files_exactly_equal_to_original': ('evaluation NPZ files', 'Number of newly computed evaluation files for which all arrays equal the original archived arrays exactly.'),
 'new_model_metadata_files_verified': ('model-diagnostic JSON files', 'Number of newly generated fitted-model/diagnostic metadata files whose bound hashes and source identifiers were verified.'),
 'new_refit_arms': ('fitted matrix arms', 'Number of original-design masked matrix arms newly fitted in the reconstruction.'),
 'prediction_arrays_and_packed_states_match_original_SHA256': ('prediction NPZ files', 'Number of new probability/packed-state NPZ files whose full-file SHA256 digest matches the digest in the original FIT_MANIFEST; this integer is a count, not a digest.'),
 'two_original_result_tables_equal_at_1e-12': ('table verification record', 'Direct comparison of the two available original result tables; child filename keys report matched row counts at absolute/relative tolerance 1e-12.'),
 'new_condition_summary_against_original_matrix_rows': ('aggregation verification record', 'Independent numerical verification of the new condition summary from original archived matrix-level rows and the condition registry.'),
 'bytes': ('bytes', 'Uncompressed file size in bytes.'),
 'path': ('file path', 'Path of the named file in the enclosing record; package-relative unless that record explicitly records a historical absolute path.'),
 'original_members_checked': ('files', 'Number of file members in the original ZIP examined by the update audit.'),
 'byte_identical_original_members': ('files', 'Original ZIP members present in the updated directory with identical size and SHA256 bytes.'),
 'changed_original_members': ('file paths', 'Original members whose bytes changed; the audit allows only specified documentation, entry-point or integrity-file updates.'),
 'unexpected_changed_original_members': ('file paths', 'Changed original members outside the explicitly allowed set; this list must be empty for the audit to pass.'),
 'added_paths_at_check': ('file paths', 'New file paths present at the time of the original-archive comparison.'),
 'original_archive': ('archive filename', 'Filename of the unchanged original ZIP used as the audit reference.'),
 'checked_on': ('calendar date', 'Date recorded for the audit, in YYYY-MM-DD format.'),
 'root_integrity_note': ('text', 'Explanation of checksum/manifest generation order and avoidance of recursive self-checksums.'),
 'license_scope': ('text', 'Summary of the separate licensing scopes for original code, derived data and upstream material; governing notices are in the named license files.'),
 'new_refit_evidence': ('package-relative evidence paths', 'Paths to the complete-refit comparison and verification records supporting the new reconstruction.'),
})

FLAGS = {'logistic_convergence_warning','logistic_iteration_limit','mk_optimizer_ran',
         'mk_optimizer_failed','mk_rate_boundary','mk_warning_count','nj_zero_overlap_pair_events',
         'nj_negative_branch_events','nj_truncated_branch_events','response_fit_events',
         'response_fit_count','arm_count','logistic_reached_max_iter_count',
         'logistic_convergence_warning_count','mk_nonconstant_fit_count',
         'mk_optimizer_unsuccessful_count','mk_rate_boundary_count',
         'nj_zero_overlap_pairs_sum','nj_negative_branches_sum','nj_truncated_branches_sum',
         'zero_overlap_pairs','negative_branches','truncated_branches',
         'fit_ran','reached_max_iter','convergence_warning','rate_boundary'}

def describe(key, path='', dtype=''):
    key = re.sub(r'\[\*\]', '', key).rsplit('.', 1)[-1]
    # A JSON leaf may reuse a scientific field name inside a diagnostic record.
    # Resolve the parent context before consulting the shared metric glossary.
    is_json = path.startswith('$')
    if is_json and '.evaluation_arrays_equal.' in path:
        return ('boolean', 'True when every element of the named newly computed evaluation array exactly equals the corresponding original archived array; this is an agreement flag, not the array value or its scientific unit.')
    if is_json and path.startswith('$.tables.') and path.endswith('.csv'):
        return ('table verification record', 'Comparison record for the result-table filename given in this JSON key, with numerical agreement and row-count fields below.')
    if is_json and '.two_original_result_tables_equal_at_1e-12.' in path:
        return ('rows', 'Number of rows in this archived result table that match the newly computed table within the stated 1e-12 tolerance.')
    if is_json and key == 'reference' and ('scenario_summary.csv' in path or 'new_condition_summary_against_original_matrix_rows' in path):
        return ('comparison provenance', 'Names the original matrix-level rows and registry used for independent verification of the newly generated condition summary; this is not a reference completion scheme.')
    if is_json and '.largest_metric_absolute_difference.' in path:
        unit = DEFS.get(key, ('same units as the named metric', ''))[0]
        if unit.startswith('indicator'):
            unit = 'absolute difference between 0/1 indicators'
        return (unit, 'Largest absolute difference in ' + key + ' between the recomputed and frozen recorded metric values across matched rows; this is a numerical agreement check, not the metric value itself.')
    if is_json and key == 'largest_metric_absolute_difference':
        return ('per-metric agreement diagnostics', 'Map from metric name to the largest absolute recomputed-versus-recorded difference across matched rows.')
    if is_json and key == 'edges':
        if '.nj_models[*].edges' in path:
            return ('node indices and fitted branch-length units', 'Fitted NJ model topology. Each edge triple is [endpoint node index, endpoint node index, fitted branch length]; list-element paths describe the same triples or their scalar entries.')
        if '.diagnostics[*].nj_mk.edges' in path and dtype == 'integer':
            return ('edges (count)', 'Number of edges in the inferred neighbour-joining tree, recorded as a scalar fitting diagnostic; this field does not contain edge endpoint pairs or branch lengths.')
        if dtype == 'integer' and not path.endswith('[*]'):
            return ('edges (count)', 'Scalar count of tree edges in the enclosing JSON record; topology endpoint arrays are stored separately.')
        return ('zero-based node indices', 'Fixed/latent topology edge endpoint pairs [node_a, node_b]; these pairs do not include fitted branch lengths. List-element paths describe pairs or their node indices.')
    if key == 'edges' and path.endswith('.npz'):
        return ('zero-based node indices', 'Latent topology edge endpoint pairs [node_a, node_b]; array axes are edges by two node indices. These pairs contain no fitted branch lengths.')
    if is_json and key == 'character':
        if '.frozen_targets_excluded_by_downstream_scope[*].character' in path:
            return ('one-based source character number', 'Original source character number of a frozen historical target excluded from the downstream subset.')
        if '.diagnostics[*].character' in path:
            return ('zero-based matrix column index', 'Response character column in the analysed masked input; diagnostics are indexed by the zero-based fitting column.')
    if is_json and key == 'target_columns':
        if dtype in ('integer', 'number') and not path.endswith('[*]'):
            return ('characters (count)', 'Scalar count of matrix columns containing target cells in this JSON record; an array/list with this name instead stores zero-based column indices.')
        return ('zero-based column indices', 'Ordered list of matrix columns containing target cells, or an element of that list; this JSON field is not the CSV count of target-containing columns.')
    if is_json and '.optimizer.method' in path:
        return ('optimiser method name', 'Name of the scalar optimisation algorithm recorded by the Mk rate optimiser; this is not a prediction/completion scheme.')
    if is_json and key in ('max_empirical_crps_difference', 'max_fair_crps_difference'):
        return ('excess parsimony steps', 'Largest absolute discrepancy between the independently evaluated direct pairwise-distance CRPS formula and the saved CRPS score for the checked distributions.')
    # A verification flag can contain words such as "seed" or "sha256" in its
    # name without storing seed material or a digest. Stored type takes priority.
    if is_json and dtype == 'boolean' and key not in DEFS and key not in FLAGS:
        return ('boolean', 'True/false metadata or verification flag named "' + key + '"; it records the enclosing check/status, not a seed value, checksum, count or scientific score.')
    if key in DEFS:
        return DEFS[key]
    kl = key.lower()
    if key in FLAGS:
        meaning = key.replace('_', ' ')
        unit = 'boolean' if dtype == 'boolean' else 'indicator in a fit; event count in aggregate tables'
        return (unit, meaning + '; count/flag recorded by the shipped fitting code, without treating response fits as independent latent matrices.')
    if kl.startswith(('minimum_observed_training','maximum_observed_training','min_observed_training','max_observed_training')):
        return ('taxa', 'Minimum or maximum observed response training rows across the recorded fits.')
    if ('wilson' in kl) or kl.startswith(('inclusion_wilson','coverage_wilson')):
        return ('proportion', 'Specified lower/upper endpoint of the two-sided 95% Wilson interval for the independent-matrix truth-inclusion fraction.')
    if kl in ('truth_below_count','truth_above_count','count_truth_below_interval','count_truth_above_interval'):
        return ('matrices', 'Count of independent matrices whose true_D lies on the specified side of the central 95% draw interval.')
    if kl == 'zero_true_probability_total':
        return ('cells', 'Sum of zero_true_probability_count across the condition, arm and scheme.')
    if kl == 'log_floor_total':
        return ('cells', 'Sum of log_floor_count across the condition, arm and scheme.')
    if kl.endswith('_point_estimate_direction_agrees'):
        return ('boolean', 'Whether the signs of the two named paired mean contrasts agree; not an inferential test.')
    if kl.endswith('_sha256') or key == 'sha256':
        return ('SHA-256 hex digest', 'SHA-256 digest identifying the named input, code, output or record; not a DOI or accession.')
    if 'seed' in kl:
        return ('integer seed or ordered seed sequence', 'Deterministic pseudorandom seed material, interpreted by the named generator in the associated source code.')
    if kl in ('seconds','elapsed_seconds') or 'seconds' in kl or '.timings.' in path:
        return ('seconds', 'Recorded elapsed execution time; platform-dependent and not a scientific result.')
    if kl.endswith(('_path','_file')) or kl in ('source_npz','source_summary','template','trees','npz_path','masked_file'):
        return ('path', 'File path retained by the source record; historical absolute paths are provenance, not guaranteed executable locations.')
    if kl.endswith('_difference_mean') or kl.endswith('_difference_mcse'):
        stem = key[:key.lower().rfind('_difference_')]
        unit, definition = describe(stem, path, dtype)
        suffix = 'Mean of paired within-matrix differences; direction is in this record.' if kl.endswith('_mean') else 'Sample SD of paired within-matrix differences divided by sqrt(n_pairs).'
        return (unit, definition + ' ' + suffix)
    if kl.endswith('_mcse') and key not in DEFS:
        unit, definition = describe(key[:-5], path, dtype)
        return (unit, definition + ' Monte Carlo standard error across independent matrices in the condition: sample SD/sqrt(n).')
    if kl.endswith('_mean') and key not in DEFS:
        unit, definition = describe(key[:-5], path, dtype)
        return (unit, definition + ' Arithmetic mean across independent matrices within the condition/arm/scheme.')
    if kl.endswith('_n'):
        return ('matrices', 'Number of nonmissing independent-matrix values contributing to the named metric summary.')
    if kl.endswith(('_count','_total')) or kl.startswith('n_'):
        return ('count (entity indicated by field name)', 'Recorded count for ' + key.replace('_', ' ') + '; inspect the enclosing record and named source for its scope.')
    if dtype == 'boolean':
        return ('boolean', 'True/false metadata flag with its source field name retained; no additional biological interpretation is assigned.')
    return ('not specified in source metadata', 'Source field "' + key + '" retained as recorded. Its structure/type is documented here; semantics beyond the original key/context are not independently specified.')


def role(rel):
    s = str(rel)
    if s in OUTPUTS or s.startswith('tools/build_data_dictionary'):
        return 'data documentation'
    if s.endswith(('.py','.sh')): return 'analysis or verification source code'
    if 'reference600' in s: return 'frozen score table from the separate 600-matrix benchmark'
    if 'refit' in s.lower(): return 'new refit/reconstruction output; consult its provenance before comparing to archived results'
    if 'baseline_evaluation' in s or 'baseline_matrix' in s: return 'original empirical-arm baseline reused in the fixed-target control'
    if '/truth/' in s: return 'complete simulated latent truth; evaluation only'
    if '/inputs/' in s and s.endswith('.npz'): return 'masked fitting input and observation labels'
    if '/predictions/' in s: return 'fitted predictions, target completions or fitted-model metadata'
    if '/evaluation/' in s: return 'per-completion parsimony evaluation and target-truth score'
    if 'empirical_bridge' in s: return 'frozen historical source/layout template or provenance'
    if s.endswith(('.png','.pdf','.svg')): return 'figure asset'
    if 'figure' in s: return 'figure data, caption or validation'
    if s.endswith('.csv'): return 'analysis registry, result or summary table'
    if 'provenance' in s or s.endswith('.json'): return 'configuration, provenance or verification metadata'
    if s.endswith(('.txt','.md','.cff','.yml','.yaml')): return 'documentation or dependency/metadata declaration'
    if s.endswith('.nex'): return 'original source tree in NEXUS format'
    return 'supplementary package file'


def walk_json(value, location='$', found=None):
    if found is None: found={}
    typ = 'null' if value is None else ('boolean' if isinstance(value,bool) else 'object' if isinstance(value,dict) else 'array' if isinstance(value,list) else 'integer' if isinstance(value,int) else 'number' if isinstance(value,float) else 'string')
    stat=found.setdefault(location, {'types':set(),'lengths':set(),'null':False})
    stat['types'].add(typ)
    if value is None: stat['null']=True
    if isinstance(value,(list,dict)): stat['lengths'].add(len(value))
    if isinstance(value,dict):
        for k,v in value.items():
            # Files/hashes are data keys, not scientific fields. Preserve a wildcard.
            k = '{file_key}' if '/' in k or re.fullmatch(r'[0-9a-f]{64}',k) else k
            walk_json(v,location+'.'+k,found)
    elif isinstance(value,list) and value:
        if all(not isinstance(x,(dict,list)) for x in value):
            # Aggregate primitive elements in linear time without repeated recursion.
            child=found.setdefault(location+'[*]',{'types':set(),'lengths':set(),'null':False})
            for x in value:
                t='null' if x is None else 'boolean' if isinstance(x,bool) else 'integer' if isinstance(x,int) else 'number' if isinstance(x,float) else 'string'
                child['types'].add(t); child['null'] |= x is None
        else:
            for v in value: walk_json(v,location+'[*]',found)
    return found


def csv_type(values):
    vals=[x for x in values if x.strip() not in ('','NA','NaN','nan','null','None')]
    if not vals: return 'unspecified (all blank/missing)'
    if all(x in ('True','False','true','false') for x in vals): return 'boolean'
    if all(re.fullmatch(r'[+-]?\d+',x) for x in vals): return 'integer'
    try:
        [float(x) for x in vals]; return 'number'
    except (ValueError,TypeError): return 'string'


def npz_headers(file):
    with zipfile.ZipFile(file) as z:
        for name in z.namelist():
            if not name.endswith('.npy'): continue
            with z.open(name) as f:
                version=np.lib.format.read_magic(f)
                if version == (1,0): shape,fortran,dtype=np.lib.format.read_array_header_1_0(f)
                elif version == (2,0): shape,fortran,dtype=np.lib.format.read_array_header_2_0(f)
                else: raise ValueError(f'Unsupported NPY version {version}: {file}:{name}')
            yield name[:-4],str(dtype),shape,fortran


def fields_for(file,rel):
    suffix=file.suffix.lower(); fmt=suffix.lstrip('.') or 'text'
    if suffix in ('.csv','.tsv'):
        with file.open(encoding='utf-8-sig',newline='') as f:
            reader=csv.DictReader(f,delimiter='\t' if suffix=='.tsv' else ',')
            names=reader.fieldnames or []; rows=list(reader)
        for name in names:
            vals=[r.get(name,'') or '' for r in rows]; typ=csv_type(vals)
            missing=sorted({x for x in vals if x.strip() in ('','NA','NaN','nan','null','None')})
            missing_text=('Observed tokens: '+json.dumps(missing)+'. Blank is unavailable/not applicable as defined by the source table; numeric zero is valid.') if missing else 'No blank/NA token observed; numeric zero is valid.'
            unit,definition=describe(name,str(rel),typ)
            if str(rel) in OUTPUTS: unit,definition='metadata', 'Documentation column: '+name.replace('_',' ')+'.'
            yield [str(rel),name,fmt,typ,f'{len(rows)} rows',unit,missing_text,definition,'CSV/TSV header and exported values; analysis definitions in shipped source']
    elif suffix=='.npz':
        for name,dtype,shape,fortran in npz_headers(file):
            unit,definition=describe(name,str(rel),dtype)
            missing='No missing-value sentinel defined; see masks and field definition. Zero is valid.'
            if name=='X': missing='-1 denotes unavailable/hidden states; 0 and 1 are biological state codes.'
            elif name=='raw': missing='? unknown; - gap; bracketed sets retain their original source coding.'
            elif dtype=='bool': missing='No missing values; False is a valid mask value.'
            elif name=='missing_states_packed': missing='Trailing padded bits must be discarded with count=n_targets; zero bits are valid states.'
            yield [str(rel),name,fmt,dtype,json.dumps(shape),unit,missing,definition,'NPY header; C order' if not fortran else 'NPY header; Fortran order']
    elif suffix=='.json':
        with file.open(encoding='utf-8') as f: value=json.load(f)
        for key,stat in sorted(walk_json(value).items()):
            typ='|'.join(sorted(stat['types'])); lens=stat['lengths']
            extent=('length '+str(min(lens)) if len(lens)==1 else f'length range {min(lens)}..{max(lens)}') if lens else 'scalar'
            unit,definition=describe(key,key,typ)
            if key=='$': unit,definition='JSON container','Root JSON record. [*] denotes each list element; {file_key} denotes a path/hash-keyed metadata entry.'
            elif typ in ('object','array'): unit='structured record'; definition='Container for the fields listed below this JSON path. '+definition
            missing='JSON null observed; means absent/not fitted/not applicable according to its parent record.' if stat['null'] else 'No JSON null observed at this path; an empty list/object is a valid empty collection.'
            yield [str(rel),key,fmt,typ,extent,unit,missing,definition,'Exact JSON structure; list element types aggregated across all records in this file']
    else:
        definition={'nex':'NEXUS tree file retaining source topology labels and taxon translation; parse as NEXUS rather than as a CSV.',
                    'py':'Executable source text. This dictionary does not assign biological units to code variables.',
                    'png':'Raster figure image; numeric source tables are catalogued separately.',
                    'pdf':'Publication figure or document; numeric source tables are catalogued separately.',
                    'svg':'Vector figure; numeric source tables are catalogued separately.'}.get(fmt,'Plain-text documentation, package metadata, or other file; interpret according to its filename and README.')
        yield [str(rel),'(whole file)',fmt,'file','not tabular','not applicable','not applicable',definition,role(rel)]


def include(rel):
    return not any(p.startswith('.') or p=='__pycache__' for p in rel.parts) and rel.suffix not in {'.pyc','.pyo','.tmp'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);args=parser.parse_args()
    root=args.root.resolve(); files=sorted(p for p in root.rglob('*') if p.is_file() and include(p.relative_to(root)) and p.name not in OUTPUTS)
    rows=[]; catalog=[]
    for i,file in enumerate(files):
        rel=file.relative_to(root); items=list(fields_for(file,rel));rows.extend(items)
        catalog.append([str(rel),file.suffix.lower().lstrip('.') or 'text',role(rel),len(items),'File present in package; historical provenance paths embedded in records may refer to earlier locations.'])
    for name,header in [('DATA_DICTIONARY.tsv',DHEADER),('FILE_CATALOG.tsv',CHEADER)]:
        for field in header:
            rows.append([name,field,'tsv','string','documentation rows','not applicable','None defined','Documentation column: '+field.replace('_',' ')+'.','Generated by tools/build_data_dictionary.py'])
        catalog.append([name,'tsv','data documentation',len(header),'Generated catalogue/dictionary; integrity digests are in MANIFEST.json.'])
    for name,header,records in [('DATA_DICTIONARY.tsv',DHEADER,rows),('FILE_CATALOG.tsv',CHEADER,sorted(catalog))]:
        temp=root/(name+'.tmp')
        with temp.open('w',encoding='utf-8',newline='') as f:
            writer=csv.writer(f,delimiter='\t',lineterminator='\n');writer.writerow(header);writer.writerows(records)
        temp.replace(root/name)
    covered={r[0] for r in rows}; actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and include(p.relative_to(root))}
    assert actual==covered, {'missing':sorted(actual-covered),'extraneous':sorted(covered-actual)}
    print(json.dumps({'catalogued_files':len(catalog),'dictionary_entries':len(rows),'coverage':'all nonhidden noncache package files','path_coverage_complete':True},indent=2))

if __name__=='__main__': main()
