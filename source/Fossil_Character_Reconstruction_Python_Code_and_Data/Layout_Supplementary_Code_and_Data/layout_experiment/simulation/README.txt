ROUND 17 EMPIRICAL OBSERVATION LAYOUT SIMULATION

Scientific design is frozen in ../ANALYSIS_PLAN.txt and ../DESIGN_FREEZE.json.
This experiment is distinct from the oracle R15 and generic random-matrix R16
experiments. It does not simulate fossil preservation or publication processes.

RUN FROM THE ROUND 17 DIRECTORY
  python simulation/experiment.py generate --output simulation/results \
    --config simulation/config.json --frozen-design ANALYSIS_PLAN.txt
  python simulation/fit_worker.py --output simulation/results --workers 8
  python simulation/evaluation_parallel.py --output simulation/results --workers 8
  python simulation/check_release.py --output simulation/results
  python simulation/summarize_diagnostics.py --output simulation/results

Generation, prediction and evaluation are separate phases. Generation refuses
to overwrite its destination. Fitting refuses an existing predictions directory.
Evaluation requires all 360 arm fits, verifies saved prediction/metadata hashes,
and refuses an existing evaluation directory. Use a fresh destination for a full
rerun. The frozen config resolves template paths relative to its own location.
Requirements are listed in requirements.txt. Each numerical library and model
is restricted to one thread inside the fitting worker. The evaluation wrapper
uses the unchanged core arithmetic and no RNG, preserves manifest result order,
and records core and wrapper hashes. Process scheduling does
not change any random stream.

DESIGN AND REPLICATION
Three panel templates, two rates, 30 paired replicates per panel-rate condition
give 180 independently generated latent matrices and 360 arm inputs. There are
512 completions under six propagation schemes per arm, totaling 1,105,920 draws.
Matrices, not target cells or completion draws, are the replication units.

Arm empirical retains the old observed/background-unknown/historical-target
layout. Arm permuted independently permutes each column's entire label vector.
Both arms use identical generated states and the same fixed generating scaffold.
Each arm has its own set of target positions and therefore its own truth D.

Known-state generators use independent binary characters, symmetric Mk rates
0.03 or 0.3, stationary root probabilities, and unit-length edges on one archived
reference topology per panel. No complete predictor anchor columns are added.
All 87, 367 and 128 target response columns are fitted. The fish predictor panel
contains 242 admitted definitions; its 287-character source background remains
archived separately for provenance. Prediction receives no generated tree or q.

SEEDS
State seed: SeedSequence([2026100417, panel_index, q_index, replicate_index]).
Permutation seed: SeedSequence([2026100418, panel_index, q_index, replicate_index]).
Propagation: SeedSequence([2026100419, panel_index, q_index, replicate_index,
                          arm_index, stream_index]).
Stream 0 gives uniforms shared by the five marginal schemes within one arm;
stream 1 drives the joint Mk sampler. Arms have separate propagation streams.
Estimator seed remains 20261002. Panel order: tetrapods, dinosaurs, ichthyosaurs.
Rate order: 0.03, 0.3. Generation traverses the stored root and sorted-child BFS
order, recorded explicitly in GENERATION_MANIFEST.json.

METHODS
frozen_methods/fit_adapter.py and mk_baseline.py are byte-identical R16 files.
Methods: mode (empirical frequency distribution), knn5, logistic, extra_trees,
nj_mk. Sixth propagation scheme nj_mk_joint uses the same fitted NJ+Mk models.
fit_worker.fit_targets calls fit_character for each target response column,
then retains probabilities only at historical targets. Other missing tips may
be integrated/sampled internally by Mk, but their states are discarded. Only
historical target states are written into completed matrices for scoring.
Single observed-state fallbacks, zero probabilities and convergence warnings
are retained. Missing binary states are not supplied to the fitted alphabet.

FILES AND ARRAY SCHEMAS
config.json: fixed 30 replicates, rates and source template/tree identities.
results/input_manifest.json: masked input identity/hash and deterministic stream
indices only; no generating tree, rate values or hidden states.
results/scenario_registry.csv: six panel-rate conditions and template metadata.
results/matrix_registry.csv: all 360 arm identities, their pair IDs and hashes.
results/GENERATION_MANIFEST.json: plan/config/script hashes, seeds, scaffold
tip orders, edges, root, traversal and unit lengths.

inputs/{matrix_id}.npz:
  X int8[n_taxa,n_characters]: 0/1 at visible cells; -1 elsewhere.
  target_mask bool[same shape]: historical targets only.
  labels uint8[same shape]: 0 visible; 1 background unknown; 2 target.
matrix_id is s00_r000_a0, where scenario_id is s00 and pair_id is s00_r000.
Only X and target_mask are read by fitting. Labels are included for audit.

truth/{pair_id}.npz:
  X_true uint8[n_taxa,n_characters], complete latent tip states.
  node_states uint8[n_nodes,n_characters], including internal states.
  edges int16[n_edges,2], root scalar, q scalar, names, original_column_indices.
Truth is shared by both arms, stored separately and not read during fitting.

predictions/{matrix_id}.npz:
  probabilities float64[5,n_targets,2], raw probabilities in method order.
  target_indices int16[n_targets,2], np.argwhere(target_mask) row-major order.
  missing_states_packed uint8[6,512,ceil(n_targets/8)], np.packbits big-bit order.
  np.unpackbits(...,axis=1,count=n_targets) restores each completion cohort.
predictions/{matrix_id}.json:
  Per-character diagnostics, full fitted Mk models in admitted-column order,
  source adapter hashes, stream indices, raw-completion hashes and status.
  No fitted model exists for a column without historical targets.
FIT_MANIFEST.json binds all raw probabilities, model metadata and worker code.

evaluation/{matrix_id}.npz:
  D_samples, L_samples, M_samples int32[6,512]; hard_D int32[6].
  old_L, old_M, true_L, true_M, true_D scalar values.
  True values reveal targets only and leave every background unknown as -1.
  The full latent matrix is never used as the target-completion benchmark.

matrix_results.csv: 2160 rows, one per arm and propagation scheme.
  Brier is the target-average sum over both binary classes.
  mean_D_absolute_error is abs(mean(D_samples)-own_target_true_D), not the mean
  absolute error of individual draws. Per-100-target errors are scale aids.
  Central 95% endpoints use numpy.quantile(method='linear'); truth inclusion
  is inclusive. These conditional prediction ranges are not assumed calibrated.

scenario_summary.csv: 72 rows (six conditions x two arms x six schemes).
  Each metric mean has MCSE from the 30 independent latent matrices in that arm.
  Interval inclusion includes Wilson limits, binomial MCSE and failure counts.
paired_layout_contrasts.csv: 36 rows (six conditions x six schemes).
  Every contrast is EMPIRICAL MINUS PERMUTED within the same latent matrix,
  followed by mean paired difference and SD(differences)/sqrt(30).
  Raw D values are not the primary arm comparison because each arm changes
  target identity. Compare errors relative to each arm's own target truth.

The joint scheme shares the NJ Mk single-cell probabilities, so Brier and
accuracy appear twice in six-scheme tables. This is one fitted predictor with
two propagation methods, not six separately fitted predictors.

VERIFICATION
engineering/ contains IID execution timings and toy correctness checks only;
it contains no scientific study performance pilot. The 81 ternary-state leaf
patterns on a four-tip tree were checked against explicit enumeration of every
missing and internal state. Target-only costs were compared against direct
completion, and the unchanged Mk sampler's selected draws agree byte-for-byte.
Independent release checks are in ../independent_review, including generation,
mask counts, all raw completions/costs, summary arithmetic and frozen refits.
Do not infer successful verification merely from this README; read receipts.

LIMITS
These three templates are existing histories, not new biological replication.
The layout contrast combines several forms of structure and changes target
identities; it cannot identify a single preservation or publication mechanism.
Binary independent evolution and unit edges do not reproduce native multistate,
ordered, polymorphic or inapplicable-character semantics. Historical documentary
endpoints remain distinct from anatomical truth. No topology search is included.
