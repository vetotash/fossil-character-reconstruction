"""Documentation-only inventory. Does not execute scientific analyses or alter inputs.
Run from any directory with Python 3.12 and the pinned NumPy dependency.
All headers, JSON structures and arrays are read from this package on each run.
No external cache is required. New documentation is included in the catalog;
root integrity manifests are deliberately left to the packaging owner.
"""
from pathlib import Path
from collections import defaultdict,Counter
import ast,csv,json,re,hashlib

import sys,tempfile
sys.dont_write_bytecode=True
from inspect_inputs import inspect_inputs,json_fields
ROOT=Path(__file__).resolve().parents[1]
_TEMP=tempfile.TemporaryDirectory(prefix='historical_documentation_')
WORK=Path(_TEMP.name)
TABLES,JSONS,ARRAYS=inspect_inputs(ROOT)

# These definitions use the delivered README, native-scoring scripts, prediction
# fitters, distribution diagnostics and documentary registers. Machine expressions
# below are retained alongside definitions for traceability, not executed.
BASE_TEXT=r'''
history|Historical matrix pair: tetrapods, dinosaurs, ichthyosaurs, gnathostomes or squamates; these are distinct documentary cohorts.|categorical label
lineage|Historical matrix pair identifying the fitted/evaluated lineage; same grouping role as history.|categorical label
panel|Analysis panel identifying the retained matrix/taxon cohort; may differ from the full prediction cohort.|categorical label
taxon|Taxon/OTU label for a matrix row; matching names do not establish unchanged specimen scope.|taxon label
character|Original source character number, normally one-based; use explicitly zero-based column_index for array offsets.|character identifier
row_index|Zero-based row position in the associated input matrix/taxon list.|zero-based index
column_index|Zero-based column position in the associated input matrix; not necessarily source character number minus one after filtering.|zero-based index
target_index|Position of an endpoint in the frozen target/cell registry; align with target_rows/target_columns or cells.csv.|zero-based index
draw_column|Zero-based column of a saved completion-draw matrix, mapped to a target cell by this row.|zero-based index
global_cell_index|Zero-based index into the full control/future prediction-cell registry.|zero-based index
original_array_index|Zero-based position of a retained source character in the full character-score array.|zero-based index
model|Predictor/completion identifier: mode=observed frequency, knn5=5-neighbour classifier, logistic, extra_trees, nj_mk=marginal Mk, nj_mk_joint=joint conditional Mk.|categorical label
method|Prediction method identifier; blocked validation compares five fitted predictors on the same held-out old labels.|categorical label
scheme|Completion sampling scheme; joint Mk changes within-character dependence, not marginal probabilities.|categorical label
prediction_model|Cell-probability model underlying a completion scheme; joint and marginal Mk use the same cell model.|categorical label
reference|Fixed reference-tree identifier; correlated trees within a history are sensitivity references, not independent biological replicates.|tree identifier
tree|Fixed reference-tree identifier used for native parsimony evaluation.|tree identifier
tree_index|PAUP/native reference-tree ordinal; native output uses one-based tree numbers.|one-based index
replicate|Saved conditional completion draw identifier; draws are computational replicates, not independent historical datasets.|draw identifier
draw|Saved conditional completion draw identifier within a scheme.|draw identifier
replicates|Number of saved completion draws or bootstrap replicates; interpretation follows the enclosing result table.|count
scenario|Evaluation cohort, usually withheld control versus later_publication endpoint; does not identify independent experiments by itself.|categorical label
treatment|Completion/validation treatment: source, later_codes, deterministic method, blocked withholding or matched random withholding as indicated by the record.|categorical label
condition|Training condition used for refitting, notably frozen shared-training versus all source observations.|categorical label
training_regime|Which earlier observations were allowed during fitting: shared/control-withheld or full_source/control-restored.|categorical label
cohort|Evaluation endpoint subset: full original cohort or source-triggered exclusion subset; fits are unchanged for exclusion sensitivities.|categorical label
true|Recorded evaluation state: published later code, or held-out earlier label in controls; not independently established anatomical truth.|categorical state symbol
true_old_label|Earlier singleton state hidden for blocked/random validation and revealed only for evaluation.|categorical state symbol
pred|Cellwise maximum-probability predicted state from the recorded categorical distribution.|categorical state symbol
predicted_state|Cellwise maximum-probability predicted state; tie handling follows the archived fitter.|categorical state symbol
state|Original categorical character state symbol; preserve labels rather than treating them as numerical magnitudes.|categorical state symbol
state_symbol|Original categorical state represented by the associated state_index.|categorical state symbol
state_index|Zero-based position in the character-specific state vocabulary; not the source state label itself.|zero-based index
state_probabilities|JSON object encoded in a CSV cell, mapping original categorical state strings to fitted probabilities; observed training states define the alphabet.|probability, 0–1
probabilities|Categorical probability vector aligned with the record's classes/state labels; values sum to one within a prediction.|probability, 0–1
probability|Probability of the named state/assignment under the specified fitted or exact distribution.|probability, 0–1
p_true|Probability assigned to the recorded evaluation state; zero when that state is outside the fitted alphabet.|probability, 0–1
confidence|Largest fitted state probability for the cell; this is model confidence, not validated anatomical certainty.|probability, 0–1
agreement|Indicator or mean fraction of cellwise predictions equal to the recorded evaluation labels; grouping determines cell versus summary level.|proportion, 0–1
accuracy|Indicator or mean fraction of predictions equal to the withheld/evaluation state; agreement is with documentary labels in historical data.|proportion, 0–1
agreement_count|Number of predictions whose maximum-probability state equals the recorded reference label.|count of cells
agreement_percent|Agreement fraction multiplied by 100.|percent
brier|Class-sum Brier loss: sum over admissible states of (p_s - 1{s=y})^2; reported means average this cell loss.|squared probability, 0–2
log_loss|Negative natural logarithm of the true/reference-state probability, floored at 1e-12 for scoring.|nats
entropy_nats|Categorical predictive entropy -sum_s p_s log(p_s), with zero-probability terms contributing zero.|nats
normalized_entropy|Predictive entropy divided by log(number of represented states); constant one-state predictions require the code's zero convention.|dimensionless
max_probability|Maximum fitted probability over the cell's represented categorical states.|probability, 0–1
expected_non_map_cells|Sum over target cells of (1 - maximum state probability), the expected count differing from cellwise MAP under marginal sampling.|expected count of cells
expected_non_map_indicator|One minus the maximum fitted state probability for an individual endpoint.|probability, 0–1
k_states|Number of categorical states represented in an endpoint's fitted probability distribution.|count of states
one_state_cells|Number of endpoints whose fitted alphabet contains a single training-observed state.|count of cells
unseen_state|Whether the evaluation label is absent from the training-observed state alphabet.|boolean
training_unseen_state|Indicator/fraction of validation targets whose old reference state is absent from the remaining training alphabet.|indicator or proportion
zero_true_probability|Indicator/fraction of evaluation cells assigned exactly zero probability to the reference state.|indicator or proportion
zero_true_probability_cells|Number of evaluation endpoints assigned exactly zero probability to the reference state.|count of cells
true_state_outside_alphabet_cells|Number of endpoints whose reference state is not present in the fitted state's alphabet.|count of cells
constant_training|Indicator/fraction of predictions using a training response with one observed state, hence a constant forecast.|indicator or proportion
constant_fallback|Whether the response had one training-observed state and all models returned a constant distribution rather than fitting a classifier/rate.|boolean
D|Change in excess parsimony cost: (L_completed-L_source) - (M_completed-M_source), with completion-specific minimum M on the fixed tree.|parsimony steps
reference_D|D calculated for the later-code reference completion on the same fixed tree and endpoint cohort.|parsimony steps
later_D|D of the completion using recorded later codes; a documentary comparison rather than anatomical truth.|parsimony steps
D_MAP|D of the deterministic cellwise maximum-probability completion, not a jointly optimised state assignment.|parsimony steps
reference_excess_steps|D of the recorded later-score completion: reference added length minus reference added minimum.|parsimony steps
point_excess_steps|D of the cellwise maximum-probability completion, with its own minimum-length correction.|parsimony steps
observed_excess|D contribution or total for the later published-state reference, as defined by the table's character/reference grouping.|parsimony steps
point_excess|D contribution or total for the deterministic cellwise MAP completion.|parsimony steps
simulation_excess|D for one saved sampled completion, possibly one character or its sum according to the row's grouping.|parsimony steps
added_excess_steps|Added parsimony length minus added completion-specific minimum; same D endpoint as excess-step summaries.|parsimony steps
added_extra_steps|Added parsimony length minus added minimum length; an excess-step decomposition.|parsimony steps
added_steps|L_completed minus L_source on the specified fixed reference tree.|parsimony steps
added_tree_steps|L_completed minus L_source on the specified fixed reference tree.|parsimony steps
added_minimum_steps|M_completed minus M_source, recalculated for each completion under native character rules.|parsimony steps
steps|Native parsimony length for the specified character, treatment and reference tree; preserves native ordering/multistate conventions.|parsimony steps
native_steps|Native PAUP character length on the named tree/treatment.|parsimony steps
native_minimum_steps|Native lower bound for a character under that treatment, used to correct added tree length into excess cost.|parsimony steps
native_assigned_steps|Native character length assigned in the lower-bound/diagnosis report.|parsimony steps
native_maximum_steps|Native upper/reference bound recorded by PAUP's character-diagnosis output, not a predictive interval.|parsimony steps
minimum_steps|Completion-specific character minimum length, summed when the row describes a whole matrix.|parsimony steps
total_steps|Sum of native character lengths on the specified reference tree and treatment.|parsimony steps
tree_steps|Parsimony length on the specified fixed tree.|parsimony steps
source_steps|Parsimony length of the unchanged incomplete source matrix.|parsimony steps
source_minimum|Minimum-length contribution of the unchanged incomplete source matrix.|parsimony steps
treatment_minimum|Minimum-length contribution after applying the named completion/treatment.|parsimony steps
minimum_increment|Treatment minimum minus source minimum.|parsimony steps
full_tree_steps|Parsimony length of the complete treatment used in the local decomposition/check.|parsimony steps
full_minimum_steps|Completion-specific minimum corresponding to full_tree_steps.|parsimony steps
sim_mean_excess_steps|Mean D across saved draws for the row's scheme/reference/cohort.|parsimony steps
sim_sd_excess_steps|Sample standard deviation of D across saved completion draws.|parsimony steps
sim_mean_excess|Mean simulated character-level excess-cost contribution.|parsimony steps
sim_sd_excess|Standard deviation of simulated character-level excess-cost contribution.|parsimony steps
sim_mean|Mean D across retained saved draws in the sensitivity result.|parsimony steps
sim_sd|Standard deviation of D across retained saved draws in the sensitivity result.|parsimony steps
mean_D|Arithmetic mean of the saved D distribution.|parsimony steps
q025|Linear-interpolated 2.5th percentile of the saved D distribution.|parsimony steps
q975|Linear-interpolated 97.5th percentile of the saved D distribution.|parsimony steps
sim_q025|Linear-interpolated 2.5th percentile of the saved D draws; lower central 95% endpoint.|parsimony steps
sim_q975|Linear-interpolated 97.5th percentile of the saved D draws; upper central 95% endpoint.|parsimony steps
width|Upper minus lower endpoint of the central 95% D range.|parsimony steps
reference_in_central95|Whether the later-reference D lies inclusively between the saved 2.5th and 97.5th percentiles.|boolean
fraction_at_least_reference|Fraction of completion draws with D greater than or equal to later-reference D.|proportion, 0–1
fraction_at_most_reference|Fraction of completion draws with D less than or equal to later-reference D.|proportion, 0–1
reference_position|Later-reference D classified below, inside or above the saved central 95% D range.|categorical label
reference_interval_position|Later-reference D classified relative to the saved central 95% range.|categorical label
mean_minus_later|Mean sampled D minus later-reference D; positive values indicate more excess steps than the later completion.|parsimony steps
mean_minus_reference|Mean sampled D minus later-reference D.|parsimony steps
mean_residual|Mean sampled D minus the later-reference D, at whole-panel or per-character level as grouped.|parsimony steps
point_minus_later|Cellwise MAP D minus later-reference D.|parsimony steps
point_residual|Cellwise MAP D minus later-reference D.|parsimony steps
mean_minus_point|Mean sampled D minus cellwise MAP D.|parsimony steps
q025_minus_later|Lower central 95% D endpoint minus later-reference D.|parsimony steps
q975_minus_later|Upper central 95% D endpoint minus later-reference D.|parsimony steps
simulation_residual|One saved draw's D minus the later-reference D.|parsimony steps
CRPS_against_later_codes|Empirical CRPS of D draws against the later-code D: mean|D-y| minus half the mean pairwise absolute distance.|parsimony steps
rate|Fitted symmetric Mk off-diagonal transition rate per inferred branch-length unit; not a geological rate.|inverse inferred branch-length unit
training_loglik|Log likelihood of the observed training response under the fitted Mk tree and rate.|natural-log likelihood
rate_boundary|Whether the fitted log-rate lies at/near the optimisation boundary used by the archived fitter.|boolean
zero_overlap_pairs|Number of taxon pairs with no jointly resolved predictor states for distance construction.|count of taxon pairs
negative_branches|Number of neighbour-joining edges inferred with negative branch lengths before truncation.|count of edges
truncated_branches|Number of inferred NJ edges replaced by the implementation's positive floor.|count of edges
max_probability_abs_difference|Largest absolute difference between reproduced and archived categorical probabilities for this prediction record.|probability difference
max_archived_probability_difference|Maximum absolute difference from archived prediction probabilities in the diagnostic refit.|probability difference
max_marginal_error|Maximum absolute discrepancy between reconstructed joint-model marginals and archived fitted marginal probabilities.|probability difference
maximum_marginal_discrepancy|Largest absolute error between reconstructed conditional marginals and the supplied cell probabilities.|probability difference
empirical|Frequency of a named state among saved Monte Carlo draws, compared with its model probability.|proportion, 0–1
n_train|Number of singleton response labels available to this response-specific fit after relevant control masking.|count of taxa
n_original_train|Number of earlier singleton response labels before withholding the historical matched controls.|count of taxa
n_controls|Number of selected earlier-label control cells for that response character.|count of cells
n_future|Number of future/later-publication evaluation endpoints for the response character.|count of cells
n_full_source_train|Training singleton labels when previously withheld controls are restored.|count of taxa
n_old_shared_train|Training singleton labels under the frozen shared/control-withheld fit.|count of taxa
n_restored_old_controls|Earlier control labels restored in the full-source training sensitivity.|count of cells
training_count|Number of remaining training response labels used for this prediction.|count of taxa
n_states|Number of training-observed states in the response alphabet.|count of states
n_character_states|Size of the character-specific categorical vocabulary.|count of states
n_target_cells|Number of retained taxon-by-character endpoints in the downstream panel/treatment.|count of cells
n_target_characters|Number of source characters containing retained target cells.|count of characters
n_scoring_taxa|Number of retained taxon rows entering fixed-tree parsimony scoring.|count of taxa
n_endpoints|Number of retained evaluation cells contributing to the metric summary.|count of cells
ntax|Number of taxon/OTU rows in the associated matrix/panel.|count of taxa
nchar|Number of characters/columns in the associated matrix or native scoring block.|count of characters
active_cells|Retained target cells after the named deletion/sensitivity treatment.|count of cells
source_missing_fraction|Fraction of unavailable/unknown source matrix entries under the relevant source coding convention.|proportion, 0–1
source_missing|Source missingness value recorded for the fit; inspect sample/type and generating expression for fraction versus boolean use.|as encoded; code determines
old_singletons|Number of earlier observed singleton states for the character before target prediction.|count of taxa
source_singleton_training_taxa|Number of earlier singleton observations in the response character used for the source eligibility threshold.|count of taxa
source_training_meets15|Whether the response contains at least 15 admissible earlier singleton observations before masking.|boolean
singleton_and_source_training_meets15|Whether a later singleton endpoint also satisfies the earlier-source threshold of at least 15 singleton labels.|boolean
state_range|Native character-state range reported during minimum/maximum diagnosis; categorical coding range is preserved verbatim.|state-symbol range
endpoint_id|Stable documentary endpoint identifier linking a taxon-character cell across audit records; not an anatomical truth label.|identifier
selection_digest|Hash used for deterministic fixed-rule endpoint sampling independently of model predictions/errors.|hexadecimal digest
original_primary_token|Verbatim original published/source matrix token before modelling recoding.|categorical token
later_primary_token|Verbatim later published matrix token used in documentary comparison.|categorical token
token_history|Documentary assessment of original/later token identity and missingness interpretation.|categorical assessment/text
character_correspondence|Documentary evidence status supporting the old-to-later character-definition correspondence.|categorical assessment/text
taxon_scope|Documented continuity or change in the meaning/composition of the OTU label.|categorical assessment/text
specimen_continuity|Evidence status for unchanged specimen material between the historical codings; not implied by identical taxon names.|categorical assessment/text
later_state_support|Documentary support category for the later recorded state; does not constitute independent expert anatomical validation.|categorical assessment/text
support_mode|Type of documentary evidence supporting the later score, e.g. scoring instruction versus direct specimen description.|categorical assessment/text
explicit_published_specimen_description_support|Whether published specimen descriptions directly support the later state in this documentary review.|boolean
complete_documentary_chain_supported|Whether all required documentary links were supported; this is not an independent anatomical correctness verdict.|boolean
expert_review_status|Completion status of the separate independent-expert review; blank/uncompleted records must not be called completed validation.|categorical status
review_status|Recorded review completion/status, separate from model success or anatomical correctness.|categorical status
provisional_anatomical_state|State to be entered by the independent reviewer before later-label reveal; blank means review not completed.|categorical state; may be blank
provisional_confidence|Independent reviewer's confidence in the provisional anatomical assessment; no completed expert scale is inferred from blank forms.|reviewer supplied
old_specimens|Specimen identifiers/material supporting the earlier matrix assignment, when recoverable.|free text/list of specimen identifiers
later_specimens|Specimen identifiers/material supporting the later matrix assignment, when recoverable.|free text/list of specimen identifiers
source_reason|Explanation of documentary evidence and its limits for this endpoint.|free text
exclusion_reason|Recorded reason for failing a source eligibility, correspondence or taxon-scope requirement; blank means no reason recorded, not proof of anatomy.|free text
evidence|Supporting documentary statement, source locator or machine-check result; interpretation follows the enclosing audit record.|structured/text evidence
reason|Authored explanation for a decision, assessment or check outcome.|free text
scope|Explicit scope of the stored analysis/receipt, including what was and was not checked.|free text
rule|Recorded decision/selection/exclusion rule used to define this record.|free text
doi|Digital Object Identifier of the source publication/data record.|DOI string
source_url|Public URL of a source publication, matrix or repository record.|URL
pdf_page|Page ordinal within the cited PDF, usually one-based; printed-page locators are recorded separately where needed.|PDF page number
short_quote|Short source quotation retained to document the stated discrepancy or assessment.|quoted text
source_locator|Page/figure/character/section location of the supporting published statement.|document locator
archive_locator|Relative archived member or code location supporting the implementation decision.|file/record locator
frozen_later_state|Later published state held fixed as the original evaluation label, including unresolved source conflicts.|categorical state symbol
modification_list_state|State specified in the later publication's prose score-modification list, which may conflict with the matrix.|categorical state symbol
additional_inheritance_support|Whether follow-up tracing supplied support via the published definition-inheritance chain, separate from original Def grading.|boolean
original_record_changed|Whether the original audit record was altered by follow-up; the additive follow-up preserves the original records.|boolean
later_state_conflict_retained|Whether an unresolved conflict between published later-state sources remains in the record.|boolean
duplicate_is_singleton|Whether the source-declared duplicate character already has one earlier singleton state in this taxon.|boolean
duplicate_equals_later_state|Whether that duplicate's earlier singleton state matches the later response label.|boolean
excluded_as_source_declared_duplicate|Whether the endpoint is removed in the source-triggered duplicate-character sensitivity, without refitting predictors.|boolean
seconds|Elapsed wall-clock runtime recorded by the named instrumented operation. Records under reproducibility/diagnostic_smoke_20261006 are the explicitly dated new smoke; other historical receipts retain their original scope.|seconds
sha256|SHA-256 digest of the referenced file's byte content; a historical digest is not evidence the file is currently included.|64-character hexadecimal digest
bytes|Recorded file length in bytes; may describe an omitted original artifact identified by a historical manifest.|bytes
path|Recorded relative or original absolute file path; historical absolute paths are provenance, not current execution locations.|path string
created_utc|Recorded UTC creation timestamp of the source operation/manifest.|ISO timestamp
freeze_time_utc|UTC timestamp at which the stated plan/input selection was frozen; not retrospective proof of preregistration.|ISO timestamp
version|Recorded source/software/schema version identifier.|version string
status|Stored completion/validation/availability status of the named check or artifact; not a new verification result.|categorical status
passed|Boolean result recorded for the named check; consult the receipt's scope and time.|boolean
random_seed|Integer seed supplied to the pseudo-random generator for the named task; reproducibility remains version/platform dependent.|integer seed
random_state|Estimator pseudo-random seed/state parameter recorded by the fitted classifier.|integer seed or null
classes|Ordered original categorical labels defining the probability-vector/state-index alphabet.|ordered state labels
training_classes|Sorted states observed in the permitted training responses.|ordered state labels
train_indices|Zero-based taxon-row indices conditioned on during fitting, aligned with train_states.|zero-based indices
training_indices|Zero-based matrix-row indices retained for fitting after target masking.|zero-based indices
train_states|Original singleton response labels corresponding to train_indices.|categorical state symbols
target_indices|Zero-based taxon rows predicted for this response-specific model; use the record's character for the response column.|zero-based indices
control_indices|Zero-based taxon-row positions chosen as old-label controls for this response character.|zero-based indices
edges|NJ/Mk tree-edge records (parent-node, child-node, branch length) when an array; edge count when scalar. Tips index taxa, internal node IDs follow tips.|edge triples or count
optimizer_success|Whether the recorded Mk optimiser reported successful termination.|boolean
logistic_actual_fits|Number of response characters for which logistic regression actually fitted, excluding constant fallbacks.|count of fits
logistic_constant_fallbacks|Number of response characters handled by a one-state constant predictor rather than logistic fitting.|count of characters
logistic_max_n_iter|Largest optimiser iteration count among instrumented logistic fits.|iterations
logistic_warnings|Number of recorded logistic convergence warnings.|count of warnings
logistic_maxiter_reached|Number of instrumented logistic fits reaching the iteration cap.|count of fits
mk_failed|Number of attempted Mk optimisations reporting failure in the diagnostic refit.|count of fits
mk_boundary|Number of fitted Mk rates flagged at the optimisation boundary.|count of fits
logistic_convergence_warning|Whether the instrumented logistic fit emitted a convergence warning.|boolean
convergence_warning|Whether the instrumented classifier emitted a convergence warning.|boolean
n_iter|Optimiser iteration count(s) recorded by the fitted classifier; null for estimators that do not expose this attribute.|iterations
reached_max_iter|Whether the recorded optimiser iteration count reached its configured cap; null if not applicable.|boolean or null
fit_ran|Whether the nonconstant classifier/Mk optimisation was actually executed.|boolean
warnings|Captured warning messages from the instrumented operation; an empty list means none recorded.|list of text
all_train_tips_conditioned|Whether reconstructed joint Mk sampling conditions on every permitted training tip label.|boolean
'''
# Split only first two separators; CRPS prose contains absolute-value bars.
BASE={}
for line in BASE_TEXT.strip().splitlines():
 key,desc,unit=line.split('|',2)
 if key=='CRPS_against_later_codes':desc='Empirical CRPS of D draws against later-code D: mean absolute(D-y) minus half the mean pairwise absolute draw distance.';unit='parsimony steps'
 BASE[key]=(desc,unit)

EXTRA=r'''
reviewer_qualification|Independent reviewer's relevant anatomical/taxonomic expertise; deliberately blank until that expert completes the form.|free text
full_prediction_column|Zero-based column in the full saved prediction/completion assignment order corresponding to this retained downstream cell.|zero-based index
difference|Empirical sampled-state frequency minus the fitted conditional marginal probability in the Monte Carlo marginal check.|probability difference
excess_steps|D: increase in fixed-tree parsimony length minus increase in the completion-specific, tree-independent minimum.|parsimony steps
hard|Deterministic cellwise maximum-probability completion's D, evaluated separately on each reference tree.|parsimony steps
mean|Mean D across the saved completion draws in the indicated scheme/reference sensitivity comparison.|parsimony steps
low|Lower central 95% D endpoint (2.5th percentile) in the named original/ablated comparison.|parsimony steps
high|Upper central 95% D endpoint (97.5th percentile) in the named original/ablated comparison.|parsimony steps
remaining_targets|Number of retained target cells after restoring the selected target block to unknown.|count of cells
interval_changes|Number of scheme-reference comparisons whose later-reference inclusion status changes after the stated sensitivity.|count of comparisons
later_inside|Number of reference-tree comparisons placing later-code D inside the conditional central 95% D range.|count of trees/comparisons
later_below|Number of reference-tree comparisons placing later-code D below the conditional central 95% D range.|count of trees/comparisons
later_above|Number of reference-tree comparisons placing later-code D above the conditional central 95% D range.|count of trees/comparisons
later_below_simulated_central95|Number of correlated reference-tree comparisons in which later-reference D is below the lower draw quantile.|count of tree comparisons
later_inside_simulated_central95|Number of correlated reference-tree comparisons in which later-reference D is inclusively between the draw quantiles.|count of tree comparisons
later_above_simulated_central95|Number of correlated reference-tree comparisons in which later-reference D is above the upper draw quantile.|count of tree comparisons
reference_in_main95|Number of fixed-reference cases whose later-code D lies inside the main/source-order central 95% range.|count of reference cases
reference_in_prose95|Number of fixed-reference cases whose later-code D lies inside the alternative prose-order central 95% range.|count of reference cases
frozen_350_target_n|Number of originally frozen tetrapod future endpoints associated with the named taxon, before the 349-cell downstream restriction.|count of cells
observed_mean|Mean later-code per-character D contribution across the correlated reference trees used for the anatomical-priority index.|parsimony steps
extra_trees_mean|Mean sampled Extra Trees D contribution for the character/reference row.|parsimony steps
logistic_mean|Mean sampled logistic-regression D contribution for the character/reference row.|parsimony steps
mean_residual_numerator|Sum across saved draws of the per-character residual (sampled D minus later-code D); dividing by draws gives mean_residual.|sum of parsimony-step residuals
selection_value|Numerical contribution/rank criterion used by the stated selection_reason to identify a character for documentary examination.|units of named selection criterion
langer_section|Section number in Langer et al.'s supplementary explanation of scoring changes.|document section identifier
pdf_symbols_before_alignment|Number of parsed printed character symbols before the source-to-archive alignment adjustment.|count of tokens
alignment_prefix_removed|Number of leading parsed tokens removed by the recorded source-to-archive alignment.|count of tokens
aligned_characters|Number of character positions remaining after source-to-archive alignment.|count of characters
pdf_dash_to_archive_question|Number of original printed dash tokens represented as question marks in the numerical archive.|count of cells
other_symbol_differences|Number of original/archived token differences apart from the explicitly counted dash-to-question conversion.|count of cells
frozen_target_taxa|Taxon labels contributing the originally frozen target cells for the character.|taxon-label list
scheme_reference_rows|Number of completion-scheme by fixed-reference rows in the focal-taxon sensitivity comparison.|count of comparisons
minimum_reference_change|Minimum change in later-reference D among the reference trees after the stated target restoration.|parsimony steps
maximum_reference_change|Maximum change in later-reference D among the reference trees after the stated target restoration.|parsimony steps
candidate_cells|Number of taxon-character cells meeting the preliminary correspondence/missingness definition before final eligibility filtering.|count of cells
shared_old_unknown|Number of earlier unknown cells within the admitted historical character/taxon correspondence.|count of cells
later_singletons|Number of matched earlier-unknown cells receiving a singleton token in the later matrix.|count of cells
later_unknown|Number of matched earlier-unknown cells still unknown in the later matrix.|count of cells
later_polymorphic|Number of matched cells carrying later polymorphic token notation under the source convention.|count of cells
later_uncertain_set|Number of matched cells carrying a later uncertainty-set token under the source convention.|count of cells
later_gap|Number of matched cells carrying a later gap/dash token.|count of cells
downstream_cells|Number of candidate/frozen cells retained in the common-tree downstream scoring panel.|count of cells
singleton|Number of matrix entries parsed as a single observed state.|count of cells
unknown|Number of matrix entries parsed as unknown, preserving the source token convention.|count of cells
gap|Number of matrix entries carrying the source gap/dash token; biological meaning is not inferred from the symbol alone.|count of cells
polymorphic|Number of matrix entries parsed as polymorphic under the source notation.|count of cells
uncertain_set|Number of matrix entries parsed as uncertainty sets under the source notation.|count of cells
denominator|Number of eligible/source cells forming the denominator for the indicated eligibility-flow stage.|count of cells
fraction|Stage count divided by the explicitly supplied denominator; blank when a proportion is not applicable.|proportion, 0–1
confidence_lower|Lower edge of the maximum-predicted-probability bin used for the calibration/score summary.|probability, 0–1
confidence_upper|Upper edge of the maximum-predicted-probability bin used for the calibration/score summary.|probability, 0–1
lower|Lower endpoint of the crossed taxon-by-character bootstrap interval for the indicated model_a-minus-model_b metric contrast.|same units as metric column
upper|Upper endpoint of the crossed taxon-by-character bootstrap interval for the indicated model_a-minus-model_b metric contrast.|same units as metric column
abs_difference|Absolute difference in source missingness between the matched control and target character used in control matching.|proportion difference
source_singleton_taxa|Number of source taxon rows with an admissible singleton response state.|count of taxa
number_targets|Number of admitted later-publication target cells mapped to the indicated source character.|count of cells
nj_mk_pred|Cellwise maximum-probability state from the NJ+Mk model in the selected disagreement case.|categorical state symbol
nj_mk_confidence|Largest NJ+Mk state probability in the selected disagreement case.|probability, 0–1
nj_mk_p_true|NJ+Mk probability assigned to the recorded later state in the selected disagreement case.|probability, 0–1
knn_pred|Cellwise maximum-probability state from the five-neighbour model in the selected disagreement case.|categorical state symbol
knn_confidence|Largest five-neighbour state probability in the selected disagreement case.|probability, 0–1
accuracy_confidence_at_least_090|Agreement among predictions whose largest state probability is at least 0.90.|proportion, 0–1
endpoint2019_truth|Recorded 2019 published reference code in the earlier 195-cell ichthyosaur cohort; the name does not establish anatomical truth.|categorical state symbol
adversarial_sizes_changing_inclusion|Number of prespecified selected-character deletion sizes changing later-reference central-range inclusion relative to the full result.|count of deletion sizes
panel_ntax|Number of taxon rows in the common scoring panel for the minimum-length check.|count of taxa
target_characters_with_minimum_change|Number of characters containing target cells whose minimum parsimony length changes on completion.|count of characters
non_target_characters_with_minimum_change|Number of characters without target cells whose minimum changes; a diagnostic that should detect unintended background changes.|count of characters
selection_value|Value of the named character-priority criterion used to choose the row for follow-up; interpret with selection_reason.|criterion-dependent numeric value
probability_tolerance|Allowed absolute discrepancy when comparing two categorical probabilities in the recorded verification.|absolute probability tolerance
tolerance|Absolute numerical tolerance used by the named historical validation check.|units of compared quantity
line|One-based line locator in the cited text/source extract.|line number
dimensions_pixels|Raster figure width and height reported during historical figure QA.|pixels, [width,height]
print_size_inches|Figure width and height used for the historical publication raster rendering.|inches, [width,height]
dpi|Raster rendering resolution recorded in historical figure QA.|pixels per inch
root_degree|Number of graph edges incident on the stored/root display node of the topology.|edge count
internal_degrees|Degrees of internal topology nodes under the stored graph representation.|edge counts
internal_nodes|Number of internal vertices in the stored phylogenetic topology.|node count
tips|Number of terminal taxon tips in the stored topology.|taxon count
article|Provider-specific article/deposit identifier used in the source acquisition manifest.|repository record identifier
native_returncode|Exit status returned by the archived native PAUP process; zero normally indicates successful process termination, subject to content checks.|process exit integer
nrep|Number of matrix/treatment replicates concatenated for this native scoring batch.|count of replicates
storageSize|Total byte size reported for the archived Dryad dataset/version.|bytes
versionNumber|Provider's dataset/version number, not a model parameter.|version integer
defined_type|Figshare provider resource-type enumeration retained verbatim; this package does not reinterpret the external enum.|provider enumeration
is_user_defined|Provider flag recording whether the corresponding category/resource metadata was user defined.|provider boolean/integer flag
handle|Provider handle identifier, if assigned; blank means no handle recorded.|persistent-identifier string
grant_code|Funding/grant identifier returned in provider metadata; blank means none recorded.|identifier string
value|Literal value of the enclosing provider/custom-metadata field; its sibling name identifies the metadata item.|provider field value
order|Provider-record ordering index within the archived metadata list.|ordering integer
index|Provider's record ordering/index value in its metadata; not a matrix array offset.|provider index
revision|Provider's record revision number at acquisition.|revision integer
prior_version|Earlier archive/revision number referenced by the continuation record.|version integer
current_version_number|Provider's current version number recorded during source retrieval.|version integer
future|Number of later-publication target cells in the enclosing historical fit/sampling manifest.|count of cells
control|Number of earlier-label control cells in the enclosing historical fit/sampling manifest.|count of cells
'''
for line in EXTRA.strip().splitlines():
 key,desc,unit=line.split('|',2);BASE[key]=(desc,unit)
BASE['D']=('Change in excess parsimony cost: (L_completed-L_source) - (M_completed-M_source); L is scored on the fixed reference tree and M is the completion-specific, tree-independent minimum.','parsimony steps')
BASE.update({
 'random_controls':('Number of count-matched dispersed withholding controls paired with this topology-defined split.','count of controls'),
 'column':('One-based character-column ordinal in the concatenated native PAUP scoring input; map through the batch layout before interpreting a source character number.','one-based native column'),
 'single_deletions_changing_inclusion':('Number of one-character deletions changing the full-data later-reference central-range inclusion status.','count of deletions'),
 'frozen_label_preserved':('Original later-matrix state retained as the evaluation label despite the published prose conflict; this integer is a state, not a boolean.','categorical state symbol'),
 'unsupported_alternative_not_imputed':('Conflicting alternative state mentioned in the publication but not substituted into the matrix or predictions.','categorical state symbol'),
 'ordered_characters':('One-based original source character numbers declared ordered for native scoring.','one-based character numbers'),
 'additional_ordered_characters':('One-based original source characters additionally treated as ordered in the alternative prose-order sensitivity.','one-based character numbers'),
 'source_ordered_characters':('One-based source character IDs declared ordered in the full source-file matrix.','one-based character numbers'),
 'target_ordered_characters':('One-based ordered source character IDs present among the scoring target characters.','one-based character numbers'),
 'target_character_to_rescore':('One-based source character whose native contribution is recomputed under the alternative ordering convention.','one-based character number'),
 'unmapped_old_characters':('One-based earlier source character IDs for which the candidate screen did not admit a later correspondence.','one-based character numbers'),
 'seed':('Integer pseudo-random seed frozen for the named selection or computational operation.','integer seed'),
 'maxiter':('Maximum iterations permitted in the recorded scalar Mk-rate optimisation options.','iterations'),
 'max_metric_error':('Maximum absolute difference between recomputed and frozen evaluation metric values in the archived clean replay.','units of the compared score'),
 'max_probability_error':('Maximum absolute discrepancy between recomputed and archived categorical probabilities.','absolute probability difference'),
 'source_minimum_steps':('Minimum parsimony length of the incomplete source background in the exact illustrative example.','parsimony steps'),
 'source_excess_steps':('Source-tree length minus source minimum in the exact illustrative example; this is baseline excess, not completion-added D.','parsimony steps'),
 'control_index':('Withholding-treatment identifier: 0 for the blocked treatment and 1, 2 or 3 for its matched random controls. It is not a taxon-row offset.','control identifier, 0/1/2/3'),
 'character_matrix_index':('Zero-based position of the response character in the stored training matrix; character supplies its original source number.','zero-based column index'),
 'mean_D_minus_D_MAP_per_target':('Difference between mean sampled D and cellwise MAP D divided by n_target_cells: (mean_D - D_MAP)/n_target_cells.','parsimony steps per target cell'),
 'mean_D_minus_later_D_per_target':('Difference between mean sampled D and later-code D divided by n_target_cells: (mean_D - later_D)/n_target_cells.','parsimony steps per target cell'),
})
COUNT_KEYS='reference_summaries draw_tree_values historical_targets prediction_rows_scored summary_rows tasks forecast_rows reviewers_appointed packets_sent raw_old_nchar raw_new_nchar below_minimum_tips above_maximum_tips original_unresolved dinosaur_endpoints_checked additional_inheritance_routes ichthyosaur_candidate_rows Suevole­viathan_candidates_excluded Suevoleviathan_candidates_excluded panels original_batch_size batch_size sensitivity_targets source_targets predictor_refits mk_optimizer_unsuccessful mk_rate_boundaries constant_training_tasks native_ablation_draw_blocks schemes reference_trees cases orderings tree_searches affected_chars between_order_interval_changes uncompressed_duplicates_removed expected_ntax frozen_prediction_rows target_rows character_map rows keys focal_cells refits native_calls native_constant_omissions total source_taxa source_characters future_targets expected_taxa expected_targets simulation_character_tree_values common_target_characters completed_replicates'.split()
for k in COUNT_KEYS:
 BASE.setdefault(k,('Recorded number of '+k.replace('_',' ')+'. The enclosing manifest/check defines the cohort or batch; this is a count of stored computational items, not independent anatomical replications.','count'))
BASE.update({
 'licence_as_recorded':('Upstream licence statement retained from the cited source record; this is not a new permission grant and later addenda may clarify its status.','licence identifier/text'),
 'licence':('Explicit licence associated with the identified source item; it is separate from the package grant for original research materials.','licence identifier/text'),
 'package_grant_applies':('Whether this package can grant MIT/CC BY rights for the named item; upstream content remains subject to its own rights.','scope statement'),
 'repository_version':('Repository version identifier used to retrieve and verify the named upstream item.','repository version identifier'),
 'repository_file_id':('Provider file-record identifier for the exact upstream file checked.','repository identifier'),
 'repository_sha256_matches_local':('Whether the recorded provider content digest equals the packaged file SHA-256 in this targeted licensing review.','boolean'),
 'conclusion':('Explicit interpretation of the targeted source/release check, limited to the evidence and items named in the record.','free text'),
 'matrix_taxa':('Number of taxon rows read from the upstream matrix in the targeted source/licence verification.','count of taxa'),
 'matrix_characters':('Number of character columns read from the upstream matrix in the targeted source/licence verification.','count of characters'),
 'original_members_checked':('Number of file members from the original supplied ZIP checked against this revised package by the packaging validation.','count of files'),
 'byte_identical_original_members':('Number of checked original ZIP file members whose SHA-256 digest and byte size are both unchanged in this revised package; the explicitly documented README revision is excluded from this unchanged count.','count of files'),
 'response_characters':('Number of response characters actually fitted in this explicitly dated portable run; the packaged smoke contains one.','count of characters'),
 'method_character_forecasts':('Number of method-by-response forecast tasks completed in this portable run; one response with five predictors gives five.','count of method-character tasks'),
 'completed_characters':('Response characters completed so far in this new portable run, as recorded by the progress log.','count of characters'),
 'last_character':('One-based original source number of the most recently completed response character in the new run.','one-based character number'),
 'nit':('Number of iterations used by the recorded SciPy scalar Mk-rate optimiser.','iterations'),
 'nfev':('Number of objective-function evaluations used by the recorded scalar optimiser.','function evaluations'),
 'refined_logq':('Refined optimum in natural-log Mk rate coordinates after bounded local optimisation within the selected grid bracket.','natural log of rate'),
 'refined_negative_loglik':('Negative training log likelihood at the refined Mk-rate optimum; minimised by the scalar optimiser.','negative natural-log likelihood'),
 'bounds':('Lower and upper log-rate endpoints of the local bounded Mk optimisation bracket recorded here; not the full coarse grid range.','natural-log rate endpoints [lower,upper]'),
 'xatol':('Absolute tolerance in the scalar log-rate parameter used to stop the bounded Mk optimisation.','natural-log rate tolerance'),
})

PARAMS={
 'C':('Inverse regularisation strength of logistic regression; larger values weaken regularisation.','dimensionless'),
 'max_iter':('Maximum optimiser iterations allowed for logistic regression.','iterations'),
 'tol':('Numerical convergence tolerance used by the configured optimiser.','solver tolerance'),
 'solver':('Logistic-regression optimisation algorithm, recorded from estimator parameters.','algorithm name'),
 'penalty':('Recorded logistic regularisation parameter; in scikit-learn 1.8 this may be the deprecated marker, so use l1_ratio/C for the effective setting.','parameter value'),
 'l1_ratio':('Elastic-net mixing parameter; zero corresponds to an L2-only penalty in the recorded logistic setting.','fraction, 0–1'),
 'dual':('Whether the dual logistic-regression formulation is enabled.','boolean'),
 'fit_intercept':('Whether an intercept term is fitted in logistic regression.','boolean'),
 'intercept_scaling':('Scale of the synthetic intercept feature for solvers using that feature.','dimensionless'),
 'class_weight':('Weights assigned to response classes during fitting; null means no additional class weights.','mapping or null'),
 'n_estimators':('Number of extremely randomised trees fitted in the ensemble.','count of trees'),
 'criterion':('Tree-split impurity criterion used by the classifier.','criterion name'),
 'min_samples_leaf':('Minimum number/fraction of training samples required in a terminal tree leaf.','sample count or fraction'),
 'min_samples_split':('Minimum samples/fraction required to split an internal decision-tree node.','sample count or fraction'),
 'max_features':('Predictor-subset size/rule considered at each randomised tree split.','count, fraction or rule'),
 'max_depth':('Maximum permitted decision-tree depth; null means no configured depth bound.','levels or null'),
 'max_leaf_nodes':('Maximum permitted terminal leaf count; null means unbounded.','count or null'),
 'min_weight_fraction_leaf':('Minimum fraction of total sample weight required in a terminal leaf.','fraction'),
 'min_impurity_decrease':('Minimum impurity reduction required to accept a tree split.','impurity units'),
 'ccp_alpha':('Cost-complexity pruning parameter for decision trees; zero disables this pruning penalty.','impurity penalty'),
 'bootstrap':('Whether individual ensemble trees use bootstrap-resampled observations.','boolean'),
 'oob_score':('Whether out-of-bag predictions are used to calculate an auxiliary score.','boolean'),
 'max_samples':('Bootstrap sample-size rule, applicable when bootstrap is enabled.','count/fraction or null'),
 'monotonic_cst':('Monotonic constraints on predictor effects; null means none supplied.','constraint vector or null'),
 'n_jobs':('Number of parallel estimator workers; this is a computational setting, not replication.','workers'),
 'verbose':('Estimator diagnostic-output verbosity level.','integer level'),
 'warm_start':('Whether a subsequent fit reuses an already fitted estimator state.','boolean'),
 'n_features_in':('Number of preprocessed predictor features presented to the fitted estimator.','feature count'),
}

CODE=defaultdict(list)
for f in ROOT.rglob('*.py'):
 if 'documentation' in f.relative_to(ROOT).parts:continue
 text=f.read_text(errors='replace');source_lines=text.splitlines();rel=str(f.relative_to(ROOT))
 try:tree=ast.parse(text)
 except SyntaxError:continue
 for node in ast.walk(tree):
  if isinstance(node,ast.Dict):
   for key,value in zip(node.keys,node.values):
    if isinstance(key,ast.Constant) and isinstance(key.value,str):
     if value.lineno==value.end_lineno:expr=source_lines[value.lineno-1][value.col_offset:value.end_col_offset]
     else:expr='\n'.join([source_lines[value.lineno-1][value.col_offset:]]+source_lines[value.lineno:value.end_lineno-1]+[source_lines[value.end_lineno-1][:value.end_col_offset]])
     expr=re.sub(r'\s+',' ',expr)
     if len(expr)<=650:CODE[key.value].append((rel,value.lineno,expr))

def evidence(field,rel):
 key=field.replace('[]','').split('.')[-1].strip('[]"')
 found=[v for v in CODE.get(key,[]) if not re.fullmatch(r'(?:[-+]?\d+(?:\.\d+)?|True|False|None|[\'\"].*?[\'\"])',v[2])]
 if not found:return '', ''
 parts=Path(rel).parts
 def score(v):
  path,line,expr=v;pp=Path(path).parts;common=0
  for a,b in zip(parts,pp):
   if a!=b:break
   common+=1
  return common*20+(10 if Path(rel).stem in Path(path).stem else 0)-len(expr)/1000
 best=max(found,key=score)
 common=0
 for a,b in zip(parts,Path(best[0]).parts):
  if a!=b:break
  common+=1
 if common<1 or (rel.startswith('source_data/') and common<4):return '', ''
 return f'{best[0]}:{best[1]}',best[2]

def words(s):
 return re.sub(r'\s+',' ',re.sub(r'[_-]+',' ',s)).strip()

def describe(field,rel,examples=(),types=(),position=None,depth=0):
 """Concrete metric definitions compose with named aggregation/cohort operators."""
 key=field.replace('[]','').split('.')[-1].strip('[]"')
 low=key.lower();sample='; '.join(str(x) for x in examples if str(x))[:400]
 ev,expr=evidence(key,rel)
 status='documented metric or identifier'
 numeric=set(types).intersection({'int','float'})
 if field.endswith('edges[][]'):
  d='One component of a three-entry edge record [node_u, node_v, branch_length]. Components 0 and 1 are node IDs; component 2 is the inferred edge length. Tips use the zero-based taxa order; internal IDs follow the tips.';u='mixed: node index, node index, inferred branch-length unit';status='model edge tuple convention'
 elif field.endswith('edges[]'):
  d='One three-component phylogenetic edge [node_u, node_v, branch_length]; order within the triple is significant.';u='two zero-based node IDs plus branch length';status='model edge tuple convention'
 elif field.startswith('$') and 'native_totals' in field and key=='*':d='Whole-tree native parsimony length for the tree ID used as this mapping key; individual-character lengths sum to this total.';u='parsimony steps'
 elif field.startswith('$') and numeric and any(s in field.lower() for s in ('symbol_counts','source_tokens','old_native_missing_tokens','token_counts')):
  d='Number of matrix/source entries having the literal token identified by the final JSON key. The parent path specifies old/later matrix or selected cohort; unknown/gap/set tokens retain source meaning.';u='count of cells';status='parent-keyed token count'
 elif field.startswith('$') and numeric and any(s in field.lower() for s in ('target_taxon','targets_per_taxon','by_taxon','taxon_counts','eligible_taxa')):
  d='Number of retained target/end-point cells for the taxon named by the final JSON key; the parent record identifies the admitted cohort.';u='count of cells';status='parent-keyed taxon count'
 elif field.startswith('$') and numeric and ('counts' in field.lower() or key in ('UNRESOLVED','MATRIX_ENTRY_ONLY','TENTATIVE_OR_CONTRADICTORY','SOURCE_SCORING_RATIONALE','INHERITED_MATRIX_WITHOUT_CELL_RATIONALE')):
  d='Number of documentary/audit records in the category identified by this JSON path. Categories are assessment dimensions and may overlap; these counts are not independent anatomical validations.';u='count of records';status='parent-keyed category count'
 elif field.endswith(('old195_removed_keys[][]','old195_added_keys[][]')):
  d='Component of a two-item endpoint key [taxon_label, source_character_number] identifying an addition/removal relative to the earlier 195-cell ichthyosaur cohort. Character numbers are one-based.';u='mixed taxon label and one-based character number';status='endpoint-key tuple convention'
 elif field=='$[][]' and 'placoderm_pdf_definition_differences' in rel:
  d='Component of a four-item definition-comparison record [old_character, later_character, old_definition_text, later_definition_text]. The first two components are published one-based character numbers; the latter two preserve source prose.';u='mixed character numbers and text';status='definition-comparison tuple convention'
 elif field.startswith('$') and 'interval_changes_by_order' in field and numeric:
  d='Number of focal-taxon scheme/reference comparisons whose interval inclusion changes under the character ordering identified by this key.';u='count of comparisons';status='parent-keyed interval count'
 elif field.startswith('$') and 'mean_sign_changes_by_order' in field and numeric:
  d='Number of focal-taxon scheme/reference comparisons whose mean-D-minus-later-D discrepancy changes sign after ablation, under the character ordering identified by this key.';u='count of comparisons';status='parent-keyed sign-change count'
 elif field.startswith('$.orders.') and field.endswith('[]') and 'ichthyosaur_taxon_test' in rel:
  d='One-based source character number treated as ordered under the source_order or prose_order convention named by the parent key. The sets are separate documented ordering assumptions.';u='one-based character number';status='explicit ordering-set member'
 elif field.startswith('$') and 'date-parts' in field:
  d='Repository/Crossref publication-date components in order year, month and day where supplied; partial dates may omit later components.';u='calendar component';status='provider date format'
 elif key in ('source_index','later_index','old_index') and ('taxon_crosswalk_independent' in rel or 'panel_taxon_mapping' in rel):
  d='One-based source/later taxon ordinal as recorded in this NEXUS TAXLABELS/MATRIX/TRANSLATE crosswalk; Eusthenopteron is ordinal 1. This is not the zero-based NumPy row convention.';u='one-based taxon ordinal';status='recorded NEXUS crosswalk convention'
 elif key in ('old_index','later_index','admitted_character_index') and 'fish_frozen' in rel:
  d='Zero-based '+('column position in the admitted-character model matrix; old_character/later_character preserve the one-based published numbers.' if key=='admitted_character_index' else 'taxon row position in the earlier or later parsed matrix, as specified by the field name; OTU labels are in the corresponding raw-name list.');u='zero-based array index';status='matrix-alignment convention'
 elif key in BASE:d,u=BASE[key]
 elif key in PARAMS:d,u=PARAMS[key]
 elif key.isdigit():d=f'Original categorical token for source character {key} in the taxon row; numeric column labels are one-based character numbers, not numeric measurements.';u='categorical state/token'
 elif key in ('A','B','C') and 'dependence_example' in rel:d=f'Binary state assigned to toy-example tip {key} in this enumerated joint assignment.';u='state 0 or 1'
 elif low in ('tree','character','length') and 'native_scores' in rel:
  if key=='Tree':d='One-based native PAUP tree number on a tree-total record; blank on following per-character records.';u='one-based tree index'
  elif key=='Character':d='One-based character number on a per-character record; blank on the preceding tree-total record.';u='one-based character index'
  elif position==2:d='Total native PAUP tree length, present beside Tree; distinct from the second Length column.';u='parsimony steps'
  else:d='Per-character native PAUP length, present beside Character; distinct from total tree length in column 2.';u='parsimony steps'
 elif re.search(r'(?:sha256|digest|checksum|_hash)$',low):d='Recorded content digest binding the named input/output/code artifact or deterministic selection to its bytes; use the enclosing key for the artifact identity. Historical entries may refer to omitted files.';u='hexadecimal digest'
 elif low.endswith(('_one_based','_1based')) or low=='character_one_based':d='One-based published/source '+words(re.sub(r'_(one_based|1based)$','',key))+' identifier; subtract one only when indexing the corresponding unfiltered array.';u='one-based index'
 elif low.endswith('_zero_based'):d='Zero-based '+words(key[:-11])+' position in the identified earlier/later matrix; a source label may differ after filtering.';u='zero-based index'
 elif re.search(r'(row|column|array|training|train|target|control)_indices$',low):d='Zero-based positions of '+words(key[:-8])+' in the enclosing matrix/taxon/character order; the supplied labels define that order.';u='zero-based indices'
 elif low.endswith('_index') or low in ('old_index','later_index','source_index'):
  d='Stored '+words(key)+' identifier/position. Explicit *_zero_based and *_one_based fields take precedence; this legacy field must be interpreted with its generating expression rather than converted automatically.';u='legacy index; see expression';status='legacy index, convention from code'
 elif low in ('old_character','later_character','revised_character','new_character','new_char','target_character'):
  d=words(key).capitalize()+' number in the corresponding published matrix correspondence; historical tabular character numbers are one-based unless explicitly suffixed otherwise.';u='one-based character number'
 elif low.endswith(('_token','_symbol','_state','_states','_score')) and not low.endswith(('review_state','random_state')):
  d='Recorded '+words(key)+' for the identified historical cell; retain categorical symbols verbatim, including unknown/gap/set notation. Later states are documentary assignments.';u='categorical state/token';status='explicit source field semantics'
 elif any(x in low for x in ('taxon','otu','specimen')) and any(x in low for x in ('label','name','otu_labels','specimen')):
  d='Recorded '+words(key)+' identifying taxonomic or specimen material; labels alone do not establish material continuity.';u='name/identifier text'
 elif low.endswith(('_label','_name','_title','_heading')) or low in ('name','title','label','heading'):
  d='Human-readable '+words(key)+' attached to the enclosing record, retained to identify its taxon, character, method or source.';u='text label'
 elif 'page' in low:
  d='Source locator recording '+words(key)+'. PDF-page ordinals and printed-page labels are distinct; explicitly 1based fields use one-based counting.';u='page number/range or printed label'
 elif low.endswith(('_doi','_url','_uri')) or low in ('url','urls','dois','links','link','download_url'):
  d='Public source/repository locator for '+words(key)+'. Availability or reuse permission is not inferred from the link alone.';u='DOI/URL string'
 elif low.endswith(('_path','_file','_directory','_root','_locator')) or low in ('file','filename','files','source_relative_path','package_relative_path'):
  d='File or record location for '+words(key)+'. Original absolute paths document the source workspace; use package-relative paths for portable replay.';u='path/locator'
 elif low.endswith(('_utc','_date','_time','timestamp')) or low in ('date','created','modified','published_date'):
  d='Recorded date/time of '+words(key)+' for the historical operation or source record; not evidence of an operation performed in this documentation revision.';u='timestamp/date as encoded'
 elif depth<4 and any(low.endswith(s) for s in ('_shared','_old','_full','_main','_prose','_source')):
  suffix=next(s for s in ('_shared','_old','_full','_main','_prose','_source') if low.endswith(s))
  base=key[:-len(suffix)];bd,bu,bs,_,_=describe(base,rel,examples,types,depth=depth+1)
  meanings={'_shared':'shared/control-withheld training result','_old':'earlier shared-training/baseline result','_full':'full-source/control-restored training result','_main':'primary/source-file character-order result','_prose':'alternative published-prose character ordering','_source':'source-file character ordering'}
  d=bd+' This column is the '+meanings[suffix]+'.';u=bu;status=bs
 elif depth<4 and low.startswith(('original_','ablated_')):
  prefix='original_' if low.startswith('original_') else 'ablated_';bd,bu,bs,_,_=describe(key[len(prefix):],rel,examples,types,depth=depth+1)
  d=('Original unmodified-cohort value: ' if prefix=='original_' else 'Value after the specified target/character ablation: ')+bd;u=bu;status=bs
 elif depth<4 and low.startswith(('delta_','change_','paired_delta_')):
  prefix=next(s for s in ('paired_delta_','delta_','change_') if low.startswith(s));bd,bu,bs,_,_=describe(key[len(prefix):],rel,examples,types,depth=depth+1)
  direction='full-source minus shared-training' if 'probability_diagnostics' in rel else 'modified/ablated minus original' if ('ablation' in rel or 'taxon_test' in rel) else 'candidate minus stated baseline'
  d=f'Within-matched-record difference ({direction}) in: '+bd;u=bu;status=bs
 elif depth<4 and any(low.endswith(s) and key[:-len(s)] in BASE for s in ('_change','_min','_max')):
  suffix=next(s for s in ('_change','_min','_max') if low.endswith(s) and key[:-len(s)] in BASE);bd,bu=BASE[key[:-len(suffix)]]
  d={'_change':'Full-source minus shared-training change in: ','_min':'Minimum across the indicated reference/group values of: ','_max':'Maximum across the indicated reference/group values of: '}[suffix]+bd;u=bu
 elif depth<4 and low.startswith(('mean_','sim_mean_','sim_sd_')):
  prefix=next(s for s in ('sim_mean_','sim_sd_','mean_') if low.startswith(s));bd,bu,bs,_,_=describe(key[len(prefix):],rel,examples,types,depth=depth+1)
  d=('Standard deviation across completion draws of: ' if prefix=='sim_sd_' else 'Arithmetic mean over the group identified by this row of: ')+bd;u=bu;status=bs
 elif depth<4 and low.endswith(('_blocked','_random_mean','_difference','_min','_max','_change')):
  suffix=next(s for s in ('_random_mean','_difference','_blocked','_change','_min','_max') if low.endswith(s));bd,bu,bs,_,_=describe(key[:-len(suffix)],rel,examples,types,depth=depth+1)
  operation={'_blocked':'Tree-block withholding value of: ','_random_mean':'Mean over matched random-withholding controls of: ','_difference':'Blocked-minus-matched-random mean difference in: ' if 'blocked_validation' in rel else 'Recorded paired difference in: ','_min':'Minimum across the group/reference trees of: ','_max':'Maximum across the group/reference trees of: ','_change':'Full-source minus shared-training change in: '}
  d=operation[suffix]+bd;u=bu;status=bs
 elif '_minus_' in low:
  left,right=key.split('_minus_',1);d='Signed difference: '+words(left)+' minus '+words(right)+'. For D/cost quantities this retains fixed-tree parsimony-step units; use the code expression for the stated grouping.';u='parsimony steps' if any(x in low for x in ('_D','d_','excess','point','later','reference','residual')) else 'same as compared quantity';status='explicit difference convention'
 elif low.startswith('n_') or low.endswith(('_count','_counts','_n')) or low in ('n','count','cells','taxa','characters','columns','endpoints','targets','edges','prediction_rows','future_cells','target_taxa','target_characters'):
  thing=re.sub(r'^n_','',key);thing=re.sub(r'_counts?$','',thing)
  d='Number of '+words(thing)+' retained in the enclosing row/group/record. This is a computational/sample count, not a count of independent evolutionary replications.';u='count'
 elif any(x in low for x in ('target_cells','training_taxa','control_cells','future_cells','affected_characters','remaining_characters','deleted_characters','restored_cells','ordered_columns','omitted_constants','failed_methods','tests_passed','predictions_compared','rows_exact','totals_exact','scores_checked','steps_checked','minima_checked','hard_treatments','replicate_rows','native_blocks','native_batches','character_values','minimum_values','draw_tree_totals','uncompressed_files','compressed_files','files_processed','files_checked','leftovers','new_draws','new_trees','existing_draws','new_topology_searches','new_parameter_fits','trees_searched','paup_calls','new_paup_scores','new_native_scores','native_characters_recomputed','character_background','target_otus','future_taxa','future_characters','common_taxa','common_targets','shared_taxa','shared_otus','source_feature_columns','source_response_characters','unreviewed_of','additional_targets','additional_documented','remaining_unresolved','original_unresolved','new_stochastic_completions','old_non_singleton','question_to_singleton','excluded_original','old195_excluded','normalized_character_candidates')):
  d='Recorded '+words(key)+' in the specified endpoint/character selection. Numeric scalar values are counts; lists contain the original character/taxon identifiers or offsets shown by the source record.';u='count or identifier list, by type'
 elif low.endswith(('_bytes','sizebytes')) or low=='totalsize':d='Recorded storage size for '+words(key)+', referring to the named compressed/uncompressed representation.';u='bytes'
 elif low.endswith('_shape') or low in ('panel_dimensions','dimensions','shape'):
  d='Stored matrix dimensions in array order [taxon rows, character columns]; these are dimensions, not individual character IDs.';u='axis lengths';status='matrix dimension convention'
 elif low.endswith('_seed'):
  d='Integer pseudo-random seed for '+words(key)+' in the stated original run; identical seeds alone do not guarantee cross-version floating-point identity.';u='integer seed'
 elif low.startswith(('min_','max_')) and any(x in low for x in ('clade','response_columns','heldout','remaining_singleton','controls','batch_size')):
  d='Protocol bound on '+words(key)+': a limit/minimum used to select withheld blocks or retained training labels, not an observed model performance metric.';u='count threshold'
 elif low in ('random_controls_per_split','batch_size_remaining','score_conditions'):
  d='Configured/recorded count of '+words(key)+' in the enclosing computation; not independent biological replication.';u='count'
 elif low=='size' and ('stash:files' in field or 'figshare' in rel or 'cerny2023_metadata' in rel):d='Recorded byte length of the repository file, or total deposited-file size at record level, as supplied by the archived repository API.';u='bytes'
 elif low.endswith(('views','downloads','citations')):
  d='Repository usage counter for '+words(key)+' captured at metadata retrieval; this is a historical provider statistic.';u='count of events'
 elif 'rank' in low:
  d='Within-group ordinal ranking by '+words(key.replace('_rank',''))+'. Absolute ranks use magnitude, signed ranks retain direction; the generating expression defines ties/order.';u='rank'
 elif low.endswith(('_fraction','_proportion','_probability')) or low.startswith('fraction_'):
  d='Fraction of the explicitly named eligible observations/draws satisfying '+words(key)+'. The enclosing result table defines numerator and denominator; this is not a posterior probability of anatomical truth.';u='proportion, 0–1'
 elif 'bool' in types or (examples and all(str(x) in ('True','False','true','false') for x in examples)):
  d='Boolean check of whether '+words(key)+'. True records that this named condition held in the archived operation; it does not broaden the scope of that operation.';u='boolean';status='explicit recorded condition'
 elif any(x in low for x in ('status','classification','category','class','relation','position','direction','sign','mode','stage','decision','selected','eligible','admitted','retained','match','agree','verified','support','continuity','conflict','changed','omitted','exposure')):
  d='Recorded assessment/check of '+words(key)+' within '+Path(rel).stem+'. Use the literal categories/values rather than treating this as a new anatomical validation.';u='categorical assessment/flag';status='explicit recorded condition'
 elif any(x in low for x in ('reason','note','scope','caution','rule','basis','evidence','description','definition','limitation','limit','protocol','instruction','interpretation')):
  d='Authored explanation of '+words(key)+' for this scientific/provenance record; the text states the applicable rule, evidence or limitation.';u='free text/structured explanation'
 elif low.endswith(('_id','_key')) or low in ('id','key','canonical','clade_key'):
  d='Stable '+words(key)+' identifying the associated task, cohort, endpoint, clade or source record; preserve exactly for joins.';u='identifier'
 elif low in ('python','numpy','scipy','pandas','sklearn','scikit_learn','matplotlib','paup','platform','system','machine','executable'):
  d='Recorded runtime/software identity for '+words(key)+' in the original environment; dependencies are not bundled merely because a version is recorded.';u='version/platform string'
 elif expr:
  d='Recorded/computed '+words(key)+' in '+Path(rel).stem+'. The exact generating assignment is supplied in definition_expression; interpret it together with the enclosing record and identifiers.';u='as implemented; see expression';status='code-derived; expression authoritative'
 elif sample and not set(types).intersection({'int','float','bool'}):
  d='Source-authored '+words(key)+' entry in '+Path(rel).stem+'. Its literal value provides the record-specific definition/context: '+sample;u='text/category as recorded';status='explicit record wording'
 else:
  d='UNRESOLVED legacy '+words(key)+' in '+Path(rel).stem+': the delivered value and key are preserved, but no sufficiently explicit generating definition was located. Do not assume a denominator, unit or indexing convention beyond the recorded type/example.';u='not established from delivered evidence';status='UNRESOLVED'
 if key=='edges' and not field.endswith('edges[][]') and not any(t in types for t in ('array','object')) and examples and all(str(x).isdigit() for x in examples):d='Number of edges in the inferred NJ tree used for the response-specific Mk fit.';u='count of edges'
 if key=='D' and 'dependence_example' in rel:d='Binary state of toy-example tip D in this enumerated assignment; this column is not the excess-parsimony metric.';u='state 0 or 1'
 if any(t in low for t in ('difference','delta_','change_','_change','_minus_','residual')) and not any(t in low for t in ('absolute','abs_difference','max_probability_abs')):
  if '0–1' in u or u=='indicator or proportion':u='difference in proportion/probability, -1 to 1'
  elif '0–2' in u:u='difference in class-sum Brier loss, -2 to 2'
  elif u=='percent':u='percentage-point difference, -100 to 100'
 return d,u,status,ev,expr

def missingness(field,rel,typ):
 low=field.lower()
 if 'expert_blank' in rel:return 'Blank fields are deliberately uncompleted independent-expert review entries; do not fill or interpret them as completed/negative reviews.'
 if 'native_scores' in rel:return 'Structural blanks separate tree-total records (columns 1–2) from character records (columns 3–4); do not coerce blanks to zero.'
 if any(x in low for x in ('token','symbol','state')) and not any(x in low for x in ('probab','random','unseen')):return "Preserve '?' unknown and '-' source-specific gap/inapplicability tokens; parentheses/braces retain source polymorphism/uncertainty conventions. Blank/null means unrecorded/not applicable, not state 0."
 if 'rate' in low or 'optimizer' in low:return 'Null/blank can mean optimisation not run for a constant training response; do not substitute zero rate.'
 if any(x in low for x in ('path','file','sha256','bytes')):return 'A stored path/hash/size may describe an omitted historical artifact; absence is not replaced by fabricated content.'
 return 'Blank CSV cells and JSON null denote unrecorded/not applicable values; empty arrays/objects are explicit empty containers. No blanket zero or mean imputation is authorised.'

def stage(rel):
 if rel.startswith('reproducibility/diagnostic_smoke_20261006/'):
  return 'New 2026-10-06 representative input verification/single-response smoke; not original logs or a full historical/PAUP rerun'
 if rel.startswith('anatomical_sources/definition_followup/'):return 'Additive documentary definition/OTU follow-up; preserves original audit and frozen labels'
 if rel.startswith('anatomical_sources/'):return 'Historical source documentation and fixed-rule endpoint audit; independent expert forms remain uncompleted'
 if rel.startswith('blocked_validation/'):return 'Frozen topology-block withholding and matched random controls using earlier observed labels'
 if rel.startswith('full_source/downstream/'):return 'Fixed-tree propagation under all-earlier-observation/full-source refits'
 if rel.startswith('full_source/old_refits/'):return 'Instrumented diagnostic refits of frozen tetrapod/dinosaur panels'
 if rel.startswith('full_source/ichthyosaur_refit/'):return 'All-source ichthyosaur response fitting and leakage-separated evaluation'
 if rel.startswith('full_source/probability_diagnostics/'):return 'Post hoc comparison of shared/full-source probability and propagated-cost summaries'
 if rel.startswith('primary/'):return 'Endpoint-matched primary historical comparison and frozen native-score replay'
 if rel.startswith('predictions/'):return 'Primary categorical prediction table for one historical cohort'
 if rel.startswith('source_sensitivity/'):return 'Source-triggered exclusion sensitivity using existing frozen predictions; no refitting'
 if 'character_robustness' in rel:return 'Archived single-character and selected-character deletion sensitivity'
 if 'taxon_ablation' in rel or 'taxon_test' in rel:return 'Archived target-block ablation/native-score evidence; taxa retained, selected targets restored to unknown'
 if 'denominator_audit' in rel:return 'Archived reconstruction of conditional eligible unknown-cell denominators and source symbols'
 if 'anatomy' in rel or 'definition' in rel:return 'Archived documentary anatomical-source and character-correspondence evidence'
 if 'method_specification' in rel:return 'Archived fitter/selection implementation documentation and preserved source-availability limits'
 if 'reaudit' in rel or 'candidates' in rel:return 'Archived raw-source parsing, eligibility/admission and historical correspondence curation'
 if 'joint' in rel:return 'Archived within-character joint Mk reconstruction/sampling and conditional-marginal checks'
 if 'probability' in rel or 'propagation' in rel:return 'Archived sampled-completion propagation and native fixed-tree scoring'
 if 'prediction_evaluation' in rel or 'named_species' in rel:return 'Archived probability scoring/bootstrap or taxonomic-scope sensitivity'
 if 'fit' in rel or 'shared_training' in rel:return 'Archived fitted predictions, source training snapshots and fitted-tree/model records'
 if 'raw' in rel:return 'Preserved downloaded/published source data and acquisition metadata'
 return 'Package provenance, portable entry points or preserved historical analysis evidence'

def purpose(rel):
 path=Path(rel);name=path.name;stem=re.sub(r'\.(csv|tsv|npy|npz|json|nex|nxs|gz|py|txt|md)$','',name)
 if rel in ('DATA_DICTIONARY.tsv','FILE_CATALOG.tsv','ARRAY_LAYOUTS.tsv','DOCUMENTATION_GUIDE.txt','DOCUMENTATION_COVERAGE.json','REPRODUCIBILITY.txt','RELEASE_NOTES.txt','PACKAGE_UPDATE_VALIDATION.json','LICENSING.txt','LICENSE_CODE_MIT.txt','LICENSE_DATA_CC_BY_4_0.txt','THIRD_PARTY_NOTICES.tsv','UPSTREAM_LICENCE_REVIEW.json'):return 'Submission documentation added in the current packaging revision; does not change scientific results. '+words(stem)
 if rel.startswith('documentation/'):
  return 'Documentation generator/helper added in this packaging revision; reads actual package structures and rewrites only the five top-level documentation indexes/guide/report, without scientific fitting or external caches.'
 if rel=='portable_diagnostic_refit.py':return 'New portable orchestrator for frozen diagnostic fitters: verifies recorded input/code hashes, relocates paths explicitly and writes new fitting logs/results to a fresh external directory; does not reconstruct historical logs.'
 if name in ('MANIFEST.json','SHA256SUMS'):return 'Delivered/root integrity listing; regenerated by packaging owner after documentation additions. Nested historical manifests remain historical receipts.'
 if name=='native_scores.tsv.gz':return 'Gzip-compressed native PAUP output with interleaved tree-total and per-character records; two distinct Length columns.'
 if 'lower_bounds' in name:return 'Native per-character lower/assigned/upper lengths used to derive completion-specific minimum corrections.'
 if path.suffix=='.py':return 'Python implementation for '+words(path.stem)+'. '+stage(rel)
 if 'README' in name:return 'Human-readable scope, file interpretation and execution commands for this analysis directory.'
 if path.suffix=='.npz':return 'Named NumPy arrays; see ARRAY_LAYOUTS.tsv for every member dtype/shape. '+stage(rel)
 if path.suffix=='.npy':return ('Shared uniform random variates aligned to target cells and draws; used to couple marginal sampling.' if 'uniform' in name else 'Saved deterministic cell assignments.' if 'hard' in name else 'Saved sampled categorical completion assignments.')+' '+stage(rel)
 if path.suffix.lower() in ('.nex','.nxs') or (path.suffix=='.txt' and 'MOESM' in name):return 'NEXUS source/analysis content: matrix tokens, character declarations, taxon translation and/or tree/PAUP commands as present. '+stage(rel)
 if path.suffix.lower() in ('.nwk','.tre'):return 'Newick/NEXUS topology representation; preserve labels and branch lengths/rooting annotations as supplied. '+stage(rel)
 if path.suffix=='.json':return 'Structured '+words(path.stem)+' record. '+stage(rel)
 if '.csv' in name or '.tsv' in name:return 'Tabular '+words(stem.replace('.csv','').replace('.tsv',''))+' records; field order and definitions are in the assigned schema. '+stage(rel)
 return words(stem)+'. '+stage(rel)

ROWS=[];MAP={};SCHEMA_INFO=[]
FIELDS=['schema_id','format','field','column_position_1based','description','units','datatype','shape_axes_or_indexing','missingness','definition_basis','evidence_file','definition_expression','representative_file','observed_example']
def add(sid,fmt,field,rel,examples=(),types=(),position='',axis='',override=None):
 d,u,s,ev,expr=describe(field,rel,examples,types,position)
 if override:d,u,axis=override;s='array format and source implementation';ev=rel;expr=''
 ROWS.append(dict(zip(FIELDS,[sid,fmt,field,position,d,u,'/'.join(types),axis or ('one scalar per row; column position is one-based' if fmt in ('CSV','TSV') else 'JSON path; [] denotes array elements and .* a dictionary keyed by recorded identifiers'),missingness(field,rel,types),s,ev or rel,expr,rel,'; '.join(map(str,examples))[:600]])))

for sid,g in TABLES.items():
 rel=g['files'][0];fmt='TSV' if '.tsv' in rel else 'CSV'
 for i,k in enumerate(g['fields']):
  ex=[r[i] for r in g['samples'] if i<len(r)]
  types=[]
  for v in ex:
   if v=='':t='blank'
   elif v in ('True','False'):t='bool'
   else:
    try:int(v);t='int'
    except ValueError:
     try:float(v);t='float'
     except ValueError:t='str'
   if t not in types:types.append(t)
  add(sid,fmt,k,rel,ex,types,i+1)
 for f in g['files']:MAP[f]=sid
 SCHEMA_INFO.append({'schema_id':sid,'format':fmt,'files':len(g['files']),'fields':len(g['fields'])})

for sid,g in JSONS.items():
 rel=g['files'][0]
 for k,spec in g['fields'].items():
  if spec['types']==['object'] or spec['types']==['array']:
   noun='keyed object' if spec['types']==['object'] else 'ordered list'
   label=k[2:] if k.startswith('$.') else k
   desc=f'{noun.capitalize()} containing '+(words(label)+' records' if label not in ('$', '$[]') else 'the complete records in this file')+'. Nested dictionary entries document each stored component; order is significant for taxa, states, draws and edge triples.'
   add(sid,'JSON',k,rel,spec['examples'],spec['types'],axis='Observed container sizes in representative: '+str(spec['container_sizes']),override=(desc,'container; child fields carry units','JSON path container; [] array elements; .* variable dictionary key'))
  else:add(sid,'JSON',k,rel,spec['examples'],spec['types'])
 for f in g['files']:MAP[f]=sid
 SCHEMA_INFO.append({'schema_id':sid,'format':'JSON','files':len(g['files']),'fields':len(g['fields'])})

ARRAY_DEF={
 'A':('Historical training/scoring character tokens, preserving singleton, unknown, gap and multistate symbols.','categorical token','taxon row x character column; align rows with old_names/names and columns with old_char_numbers/characters'),
 'raw':('Original incomplete tokens for the fixed-tree common scoring panel.','categorical token','taxon row x source-character column; names labels rows'),
 'old_tokens':('Parsed earlier published/source matrix tokens before completion.','categorical token','earlier taxon row x source-character column; old_names labels rows'),
 'later_tokens':('Parsed later published matrix tokens used for documentary correspondence/evaluation.','categorical token','later taxon row x later-character column; later_names labels rows'),
 'old_A':('Full parsed earlier gnathostome matrix, before the admitted predictor-column restriction.','categorical token','earlier taxon row x earlier source-character column'),
 'later_B':('Full parsed later gnathostome matrix before correspondence filtering.','categorical token','later taxon row x later source-character column'),
 'names':('Taxon labels in the exact row order of the associated matrix.','taxon label','one label per matrix row; zero-based array positions'),
 'old_names':('Earlier-source taxon labels in matrix row order.','taxon label','one label per earlier matrix row'),
 'later_names':('Later-source taxon labels in matrix row order.','taxon label','one label per later matrix row'),
 'old_char_numbers':('One-based original source character numbers corresponding to retained training columns.','one-based character number','one value per training-matrix column'),
 'old_characters':('One-based original source character numbers, retained as compatibility/alias metadata for model inputs.','one-based character number','one value per admitted training-matrix column'),
 'characters':('One-based original character labels defining the character axis of the accompanying matrix/score array.','one-based character number','one value per character-axis position; positions themselves are zero-based'),
 'ordered':('Source characters treated as ordered by the native scorer; empty means no ordered characters declared.','one-based character number','one-dimensional list of source character IDs; check generator for conversion to offsets'),
 'ordered_characters':('One-based source character numbers treated as ordered in this scoring input.','one-based character number','list of ordered source characters'),
 'target_characters':('One-based source character numbers containing retained prediction targets.','one-based character number','character subset, not an array-position list'),
 'target_rows':('Matrix row indices of retained prediction endpoints.','zero-based index','endpoint order; paired elementwise with target_columns and later_states'),
 'target_columns':('Matrix column indices of retained prediction endpoints.','zero-based index','endpoint order; paired elementwise with target_rows and later_states'),
 'later_states':('Recorded later singleton labels aligned with retained target cells.','categorical state symbol','one per target_rows/target_columns pair'),
 'steps':('Native PAUP parsimony lengths for each reference tree, treatment/draw and character.','parsimony steps','reference tree x treatment/draw x character; trees, treatments and characters label axes'),
 'minima':('Native minimum length for each treatment/draw and character; independent of reference-tree choice.','parsimony steps','treatment/draw x character, aligned to steps second/third axes'),
 'trees':('Names of reference trees in the first axis of steps.','tree identifier','one-dimensional reference-tree order'),
 'treatments':('Ordered deterministic treatments, including source, later_codes and five cellwise MAP completions.','treatment identifier','labels the treatment axis of steps/minima; joint Mk uses marginal-Mk MAP'),
 'affected_character_indices':('Positions of characters affected by restoring a focal taxon target block to unknown.','zero-based index','indices into the accompanying characters/score-array character axis'),
 'observed':('Per-character D contributions of later-code reference completions.','parsimony steps','reference tree x character'),
 'points':('Per-character D contributions of deterministic cellwise MAP completions.','parsimony steps','scheme x reference tree x character'),
 'draws':('Per-character D contributions of saved sampled completions.','parsimony steps','scheme x reference tree x draw x character'),
 'references':('Ordered fixed-reference names labelling the reference-tree axis.','tree identifier','one name per reference-tree position'),
 'schemes':('Ordered completion-scheme identifiers labelling the scheme axis.','scheme identifier','one name per scheme position'),
 'control':('Joint Mk categorical assignments for matched held-out earlier-label control cells.','categorical state symbol','draw x control cell; match cells/target registry and draw_number'),
 'future':('Joint Mk categorical assignments for later-publication target cells.','categorical state symbol','draw x future target cell; map cells via future_cells.csv'),
 'draw_number':('Stored zero-based draw identifiers for joint-sampling arrays.','zero-based index','one identifier per first-axis draw'),
}
array_layout=[]
for sid,g in ARRAYS['schemas'].items():
 # NPY assignments are split by meaning even when their physical schema matches.
 subgroups=defaultdict(list)
 for f in g['files']:
  role='uniforms' if 'common_uniforms' in f else 'hard'+str(ARRAYS['files'][f]['fields']['array']['shape'][0]) if f.endswith('hard_assignments.npy') else 'draws' if f.endswith('.npy') else ''
  subgroups[role].append(f)
 for role,files in subgroups.items():
  newsid=sid+('_'+role if role else '');rel=files[0]
  for key,spec in ARRAYS['files'][rel]['fields'].items():
   override=ARRAY_DEF.get(key)
   if key.endswith('_mask'):
    conditions={'old_question_mask':"token is '?'",'old_gap_mask':"token is '-'",'old_singleton_mask':'token is one admissible singleton state','old_brace_set_mask':'token uses source brace-set notation','old_parenthesis_set_mask':'token uses source parenthesis-set notation'}
    override=('Boolean token-class mask: '+conditions.get(key,words(key))+'. Source set conventions are preserved without conflating uncertainty and polymorphism.','boolean','earlier taxon row x character column, same shape as old_tokens')
   if key=='array':
    if role=='uniforms':override=('Saved common uniform random variates coupling independent-marginal completion schemes for the same target cells.','uniform variate in [0,1)','draw x target cell, aligned to cells.csv/frozen target order')
    elif role.startswith('hard'):
     order='later_codes, mode, knn5, logistic, extra_trees, nj_mk' if role=='hard6' else 'source, later_codes, mode, knn5, logistic, extra_trees, nj_mk' if role=='hard7' else 'treatment order requires adjacent generating source'
     override=('Recorded later-state reference and deterministic maximum-probability assignments (plus the source unknown row where present). Joint and marginal Mk share the same cellwise MAP row.','categorical state symbol','treatment x target cell; row order: '+order)
    else:override=('Saved sampled state assignments for the completion scheme named in the file. Joint Mk retains within-character tip dependence; other schemes use independent marginals.','categorical state symbol','draw x target cell; align targets to adjacent cells.csv/future_cells.csv registry')
   add(newsid,'NumPy',key,rel,spec['sample'],[spec['dtype']],axis=str(spec['shape']),override=override)
  for f in files:
   MAP[f]=newsid
   for k,spec in ARRAYS['files'][f]['fields'].items():array_layout.append({'path':f,'schema_id':newsid,'array_key':k,'dtype':spec['dtype'],'shape':json.dumps(spec['shape']),'axis_definition_field':k,'example':'; '.join(spec['sample'])})
  SCHEMA_INFO.append({'schema_id':newsid,'format':'NumPy','files':len(files),'fields':len(g['fields'])})

# Non-tabular scientific and documentation formats also receive explicit schemas.
FORMAT_DEFS={
 'NEXUS':[('DIMENSIONS','Declared NTAX/NCHAR counts for the associated taxon-character data block.','counts'),('TAXLABELS/TRANSLATE','Taxon labels or numeric-tree-label translation map; preserve order and exact labels.','taxon identifiers'),('FORMAT','Source DATATYPE, SYMBOLS, MISSING, GAP and optional EQUATE/RESPECTCASE declarations; these govern token interpretation.','coding conventions'),('MATRIX','Taxon-by-character categorical observations/completions; preserve unknown/gap/multistate notation.','categorical tokens'),('CHARSTATELABELS/CHARLABELS','Character and state definitions/labels as supplied; source numbering is one-based.','character/state identifiers'),('ASSUMPTIONS/CTYPE/USERTYPE','Native ordered/unordered/state-cost assumptions, if present.','character costs/ordering'),('TREES','Newick topology statements and branch lengths; TRANSLATE maps labels when present.','topology and branch-length units as supplied'),('PAUP','Native PAUP command block when this member is an analysis script; executable dependency is external.','commands')],
 'TREE':[('topology','Parenthesised Newick topology, terminal labels and optional branch lengths/comments; source rooting conventions remain unchanged.','topology; branch units as supplied')],
 'PYTHON':[('source_text','Archived Python implementation; use the portable top-level entry points and recorded source-root overrides. Existing absolute source-workspace paths can be provenance-only.','source code')],
 'TEXT':[('text','Human-authored narrative, protocol, dependency list, licence or bibliography. Interpret the document in its recorded scope; a historical receipt is not a new run.','text')],
 'INTEGRITY':[('digest_and_path','Integrity records associating content digests with relative file paths; consult their scope for files intentionally excluded from self-hashing.','SHA-256 digest and path')],
}
for sid,defs in FORMAT_DEFS.items():
 for field,d,u in defs:add(sid,'text format',field,'README.txt',override=(d,u,'Format-dependent text blocks, not a flat numeric array'))

DOC_SCHEMAS={
 'DOC_DICTIONARY':('DATA_DICTIONARY.tsv',{
 'schema_id':'Join key identifying the shared field structure used by files in FILE_CATALOG.tsv.',
 'format':'Stored format to which the field definition applies, e.g. CSV, TSV, JSON, NumPy or text blocks.',
 'field':'Exact source column/array key or structural JSON path, preserving duplicate-column names with a separate position.',
 'column_position_1based':'One-based ordinal of a CSV/TSV column; blank for named array/JSON/text fields.',
 'description':'Concrete interpretation of the field, including score formulas, cohort context or explicit unresolved status.',
 'units':'Measurement units or categorical representation; signed differences are distinguished from bounded original scores.',
 'datatype':'Observed scalar type(s) or NumPy dtype in the representative member; examples are not a formal exhaustive type restriction.',
 'shape_axes_or_indexing':'Array-axis order, position convention or JSON container/row interpretation needed for alignment.',
 'missingness':'Meaning of blank/null/unknown/gap or empty-container values for this field; no values were imputed.',
 'definition_basis':'Whether the meaning comes from source definitions, exact code, explicit record wording, format conventions, or remains unresolved.',
 'evidence_file':'Relative source member, optionally with a source-code line locator, supporting the definition.',
 'definition_expression':'Exact archived Python generating expression when a sufficiently relevant assignment was located; blank otherwise.',
 'representative_file':'Existing member whose inspected structure supplies this schema and illustrative types/values.',
 'observed_example':'Short literal values from the representative member, not a complete allowed-value list.'}),
 'DOC_CATALOG':('FILE_CATALOG.tsv',{
 'path':'Exact path relative to the package root; one row for every present file, including documentation.',
 'format':'Filename storage format, retaining gzip compression suffix for compressed tables.',
 'purpose':'Role of this member in source curation, fitting, native-score evidence, summaries or documentation.',
 'source_or_provenance':'Authenticated source-copy pointer or upstream source DOI/URL where available; documentation additions are identified separately.',
 'schema_id':'Join key into DATA_DICTIONARY.tsv for this member.',
 'dictionary':'Relative dictionary file and schema identifier used to interpret the member.',
 'array_layout':'Per-file ARRAY_LAYOUTS.tsv lookup for NPZ/NPY data; otherwise explicitly not an array file.',
 'nearby_documentation':'Nearest existing README/manifest/freeze files supplying local scope and scientific provenance.'}),
 'DOC_ARRAY_LAYOUTS':('ARRAY_LAYOUTS.tsv',{
 'path':'Exact NPZ/NPY package-relative member path.',
 'schema_id':'Associated array-meaning schema in DATA_DICTIONARY.tsv.',
 'array_key':'Named NPZ array key, or array for a standalone NPY member.',
 'dtype':'Exact NumPy dtype read with allow_pickle=False.',
 'shape':'Actual array dimensions as a JSON list in storage-axis order.',
 'axis_definition_field':'Field key in the referenced dictionary schema defining axis meanings and indexing.',
 'example':'First up to three flattened array values for identification; not a statistical sample.'}),
}
for sid,(rel,defs) in DOC_SCHEMAS.items():
 MAP[rel]=sid
 for n,(k,d) in enumerate(defs.items(),1):add(sid,'TSV',k,rel,position=n,override=(d,'documentation field; see description','one value per documentation row'))

MAP['DOCUMENTATION_COVERAGE.json']='DOC_COVERAGE'
for key,definition in {
 'scope':'Precisely delimits documentation inspection from scientific reruns or recovery of absent artifacts.',
 'file_count':'Number of actual files mapped by FILE_CATALOG.',
 'catalog_paths_unique':'Whether catalog relative paths are unique.',
 'catalog_covers_all_present_paths':'Whether catalog paths exactly equal the directory file set at generation time.',
 'table_schema_count':'Number of distinct observed table-header schemas including new notices.',
 'json_schema_count':'Number of distinct observed source/new-root JSON key structures, excluding this self-report.',
 'array_schema_count':'Number of array semantic schemas, distinguishing six/seven hard-treatment rows and sampled draws.',
 'dictionary_rows':'Number of field-definition rows written to DATA_DICTIONARY.tsv.',
 'array_layout_rows':'Number of named arrays inventoried across NPZ/NPY files.',
 'array_files':'Number of NPZ/NPY files inspected.',
 'schema_summary[]':'List of inspected scientific/new-root structures with schema_id, format, files count and fields count.',
 'schema_summary[].schema_id':'Structure identifier joining catalog and dictionary.',
 'schema_summary[].format':'CSV/TSV/JSON/NumPy source representation.',
 'schema_summary[].files':'Count of package members assigned to this structure.',
 'schema_summary[].fields':'Number of distinct field keys in this structure.',
 'definition_basis_counts.*':'Count of dictionary rows assigned to the evidence-basis category used as key.',
 'unresolved_fields[]':'Explicit list of semantic definitions still not established; each entry identifies schema_id, field, representative_file and observed_example.',
 'unresolved_fields[].schema_id':'Schema containing the unresolved field.',
 'unresolved_fields[].field':'Unresolved source field/path.',
 'unresolved_fields[].representative_file':'Existing member to inspect for that unresolved field.',
 'unresolved_fields[].observed_example':'Retained example value; does not establish unknown units or semantics.',
 'indexing_caution':'Documents distinctions between legacy indices, zero-based array offsets and one-based native/source labels.',
 'self_inventory_note':'Explains why the catalog includes itself but leaves bytes/hashes to the outer delivery integrity manifest.'
}.items():add('DOC_COVERAGE','JSON','$.'+key,'DOCUMENTATION_COVERAGE.json',override=(definition,'documentation count, boolean, identifier or explanation as described','JSON report hierarchy'))

NEWFILES=['DATA_DICTIONARY.tsv','FILE_CATALOG.tsv','ARRAY_LAYOUTS.tsv','DOCUMENTATION_GUIDE.txt','DOCUMENTATION_COVERAGE.json']
for n in NEWFILES:
 if not (ROOT/n).exists():(ROOT/n).write_text('')

def write_tsv(path,fields,rows):
 with path.open('w',newline='',encoding='utf-8') as h:
  w=csv.DictWriter(h,fieldnames=fields,delimiter='\t',lineterminator='\n',extrasaction='ignore');w.writeheader();w.writerows(rows)

write_tsv(ROOT/'DATA_DICTIONARY.tsv',FIELDS,ROWS)
write_tsv(ROOT/'ARRAY_LAYOUTS.tsv',['path','schema_id','array_key','dtype','shape','axis_definition_field','example'],array_layout)

provenance=json.loads((ROOT/'SOURCE_PROVENANCE.json').read_text())
origin={r['path']:r for r in provenance.get('files',[])}
source_map=json.loads((ROOT/'anatomical_sources/packaged_matrix_source_map.json').read_text())
source_index={r['source_id']:r for r in json.loads((ROOT/'anatomical_sources/source_index.json').read_text())['sources']}
primary_origins={path:(sid,source_index.get(sid,{})) for sid,path in source_map.items()}

CAT=[]
for f in sorted(ROOT.rglob('*')):
 if not f.is_file():continue
 rel=str(f.relative_to(ROOT));sid=MAP.get(rel)
 if sid is None:
  if f.suffix.lower() in ('.nex','.nxs') or (f.suffix=='.txt' and 'MOESM' in f.name):sid='NEXUS'
  elif f.suffix.lower() in ('.tre','.nwk'):sid='TREE'
  elif f.suffix=='.py':sid='PYTHON'
  elif f.name.startswith('SHA256'):sid='INTEGRITY'
  else:sid='TEXT'
 src=origin.get(rel)
 if rel in primary_origins:
  source_id,record=primary_origins[rel];source='Published primary matrix '+source_id+'; '+', '.join(record.get('dois',[])+record.get('urls',[]))
 elif src:source='Authenticated copied member '+src.get('source','')+'; SOURCE_PROVENANCE.json sha256='+src.get('sha256','')
 elif rel.startswith(('documentation/','reproducibility/diagnostic_smoke_20261006/')):source='New packaging documentation or explicitly dated representative smoke/input-check evidence; not an original historical run record'
 elif rel in NEWFILES or rel in ('REPRODUCIBILITY.txt','RELEASE_NOTES.txt','PACKAGE_UPDATE_VALIDATION.json','LICENSING.txt','LICENSE_CODE_MIT.txt','LICENSE_DATA_CC_BY_4_0.txt','THIRD_PARTY_NOTICES.tsv','UPSTREAM_LICENCE_REVIEW.json','portable_diagnostic_refit.py'):source='Documentation/portable orchestration added in this packaging revision; inherited scientific result bytes are unchanged'
 else:source='Delivered package-authored or inherited member; use local enclosing manifest and SOURCE_PROVENANCE.json where present'
 parents=list(f.parents)
 nearby=[]
 for parent in parents:
  if parent==ROOT.parent:break
  for name in ('README.txt','README.md','MANIFEST.json','OUTPUT_MANIFEST.json','INPUT_MANIFEST.json','RUN_MANIFEST.json','FREEZE.json'):
   q=parent/name
   if q.exists() and q!=f:nearby.append(str(q.relative_to(ROOT)))
  if len(nearby)>=3:break
 CAT.append({'path':rel,'format':''.join(f.suffixes[-2:]) if f.suffix=='.gz' else f.suffix or 'text','purpose':purpose(rel),'source_or_provenance':source,'schema_id':sid,'dictionary':'DATA_DICTIONARY.tsv#schema_id='+sid,'array_layout':'ARRAY_LAYOUTS.tsv#path='+rel if f.suffix in ('.npz','.npy') else 'not an array file','nearby_documentation':'; '.join(nearby[:4])})
write_tsv(ROOT/'FILE_CATALOG.tsv',['path','format','purpose','source_or_provenance','schema_id','dictionary','array_layout','nearby_documentation'],CAT)

unresolved=[{k:r[k] for k in ('schema_id','field','representative_file','observed_example')} for r in ROWS if r['definition_basis']=='UNRESOLVED']
coverage={'scope':'Documentation-only inspection. This generator runs no models or native PAUP. The separately labelled reproducibility/diagnostic_smoke_20261006 files contain a new single-response verification, not recovered historical artifacts or a full refit.','file_count':len(CAT),'catalog_paths_unique':len({r['path'] for r in CAT})==len(CAT),'catalog_covers_all_present_paths':set(r['path'] for r in CAT)==set(str(f.relative_to(ROOT)) for f in ROOT.rglob('*') if f.is_file()),'table_schema_count':len(TABLES),'json_schema_count':len(JSONS),'array_schema_count':len([s for s in SCHEMA_INFO if s['format']=='NumPy']),'dictionary_rows':len(ROWS),'array_layout_rows':len(array_layout),'array_files':len(ARRAYS['files']),'schema_summary':SCHEMA_INFO,'definition_basis_counts':dict(Counter(r['definition_basis'] for r in ROWS)),'unresolved_fields':unresolved,'indexing_caution':'Legacy *_index fields without explicit base are not silently converted; array rows/columns and native PAUP tree/character ordinals are distinguished.','self_inventory_note':'FILE_CATALOG includes itself and added documentation; file sizes/hashes are left to the final package manifest to avoid circular self-hashing.'}
(ROOT/'DOCUMENTATION_COVERAGE.json').write_text(json.dumps(coverage,indent=2,ensure_ascii=False)+'\n')
(ROOT/'DOCUMENTATION_GUIDE.txt').write_text('''HOW TO USE THE PACKAGE-WIDE DATA DICTIONARY

This documentation was assembled from actual delivered table headers, JSON keys,
NumPy arrays, source code and existing READMEs. It does not modify scientific
data or claim a new model fit, PAUP run or anatomical review.

REBUILDING THESE INDEXES
From the package root run: python -B documentation/build_indexes.py
Or invoke that script by its absolute path from any directory. It locates the
package relative to __file__, scans actual members with allow_pickle=False,
and rewrites DATA_DICTIONARY.tsv, FILE_CATALOG.tsv, ARRAY_LAYOUTS.tsv,
DOCUMENTATION_GUIDE.txt and DOCUMENTATION_COVERAGE.json. No outside cache or
original workspace path is required. It does not run analysis scripts, alter
source results, regenerate fitting logs, or update the root MANIFEST/SHA files.
Install the package's pinned NumPy dependency before running the index builder.

New evidence under reproducibility/diagnostic_smoke_20261006 is separately
labelled: one actual response-character smoke with five methods, plus input hash
verification. Those records are new 2026-10-06 verification, not recovery of
original fitting logs or a full historical refit/native PAUP rerun.

Start with FILE_CATALOG.tsv. Every present path has a purpose, provenance pointer,
schema_id, DATA_DICTIONARY.tsv link and, for NumPy files, an ARRAY_LAYOUTS.tsv link.
Files sharing field structure use a common schema instead of repeating identical
definitions. DATA_DICTIONARY rows identify their representative member. Examples
are illustrations from that representative, not exhaustive allowed-value sets.
Column positions are one-based. Duplicate names are distinguished by position.

DATA_DICTIONARY.tsv supplies meaning, units, type, axis/index convention,
missing-value treatment, definition basis and an exact generating expression plus
source-code location when found. Definitions of differences/aggregations identify
the quantity and comparison; the source expression remains authoritative for
grouping. 'UNRESOLVED' identifies a legacy field whose semantics could not be
established from delivered evidence. These entries preserve its observed value
and locator and do not invent scientific units or missing historical records.
DOCUMENTATION_COVERAGE.json lists every unresolved entry explicitly.

JSON paths use dot notation for normal identifiers, quoted brackets for literal
keys containing punctuation, [] for all elements of a list, and .* for a mapping
keyed by variable recorded identifiers (for example character IDs or filenames).
Container rows describe hierarchy; child rows describe the values. A schema is
based on observed key paths, not a restriction against future optional keys.
JSON type/shape examples are from the representative file; ARRAY_LAYOUTS records
the actual dtype and shape of every member in every supplied NPZ/NPY file.

NUMPY ALIGNMENT
Load NPZ/NPY with allow_pickle=False. Taxon row order comes from names/old_names;
character positions come from characters/old_char_numbers. Published character
numbers are one-based; target_rows/target_columns are zero-based offsets.
Native score arrays use tree x treatment-or-draw x character for steps, and
treatment-or-draw x character for minima. Hard treatments are explicitly named.
Draw arrays align to the local cell registry, not an alphabetically sorted list.
Do not infer state symbols from numeric array positions.

SPECIAL NATIVE PAUP TSV FORMAT
native_scores.tsv(.gz) has header Tree, Length, Character, Length. Tree-total
records populate columns 1-2. Subsequent character records populate columns 3-4.
The two Length columns are different quantities. Structural blanks are expected;
do not read them as zero or discard the tree block association. gzip changes only
the storage encoding. PAUP minimum-length tables retain the source conventions
for ordered characters and multistate tokens; Python replay only recombines them.

STATE TOKENS, EMPTY VALUES AND INTERPRETATION
Keep original state strings. '?' is unknown; '-' must follow the source's declared
gap/inapplicability convention. Parentheses and braces are not interchangeable.
Do not automatically turn blank CSV fields or JSON null into zero. Null model
rates/diagnostics can mean no optimisation ran for a constant training response.
Blank expert-review fields are intentionally uncompleted reviews.

Brier is the class-sum squared probability loss (0-2); log loss uses natural logs
and a 1e-12 scoring floor. D is added tree length minus added completion-specific
minimum, in parsimony steps. Central 95% draw ranges are conditional simulation
ranges; draws and correlated trees are not independent biological replications.
Later scores remain documentary reference assignments, not established anatomy.

HISTORICAL PATHS AND RECEIPTS
An old manifest may name a file not included here. A path/hash/version receipt is
not recovery of that file or a claim of a fresh check. The final root manifest
describes this delivery. REPRODUCIBILITY.txt distinguishes numerical replay,
new fitting and external native PAUP regeneration, and records unrecovered items.
Original numeric data and source code have not been changed for this dictionary.
''',encoding='utf-8')
print(json.dumps({k:v for k,v in coverage.items() if k not in ('schema_summary','unresolved_fields')},indent=2))
print('UNRESOLVED',len(unresolved))
(WORK/'historical_unresolved.json').write_text(json.dumps(unresolved,indent=2,ensure_ascii=False))
