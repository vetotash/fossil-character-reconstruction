LAYOUT SIMULATIONS: SUPPLEMENTARY CODE AND DATA

Verification
See AUTHOR_VERIFICATION_STATEMENT.txt for the declared AI assistance and the
author's reported verification of this release. Archived receipts retain the
dates and scope of the work they record.

This package contains the frozen 180-matrix paired historical-layout experiment,
its post hoc distributional diagnostics, and the fixed-target observation-layout
control described separately in fixed_target/.
Use Python 3.12 and the exact versions in requirements.txt. Numerical dependencies
are not vendored. The code and included data need no network access at runtime.

The archived experiment has three historical templates, two binary Mk rates
(q=0.03 and 0.3), and 30 independent latent replicates per panel/rate. Its two arms
are empirical and whole-column-label permutation, giving 360 inputs. Each arm
has six saved 512-draw distributions. The true D is arm-specific: only that arm's
target cells are revealed, and other unavailable cells remain unknown. Layout
permutation changes target identities as well as observed/unavailable positions.

RECOVERY AND SCOPE

All 360 original evaluation NPZ members were recovered intact from an interrupted
transfer and checked against ZIP-member length, CRC32 and SHA256. The complete
original ZIP was not verified. The 180 original latent truth files and 360 mask
files were regenerated using unchanged source, frozen templates, topologies,
settings and seeds. Every regenerated file is byte-identical by SHA256 to the
original matrix registry. Required simulation inputs and the full saved D-draw
cohort are therefore included.

The original 360 fitted model JSON files and probability/packed-completion arrays
were not recovered as a complete cohort. A complete new refit is now included in
refit_20261006/: 360 model-diagnostic JSON files, 360 probability/packed-completion
NPZ files, 360 newly scored evaluation NPZ files and the three regenerated result
tables. All 360 new prediction NPZ files match the original FIT_MANIFEST SHA256
hashes exactly. All original and new evaluation arrays are exactly equal, and
the two available original result tables agree to 1e-12. The new 72-row condition
summary was independently checked against reaggregation of the original matrix
results (maximum numerical difference 2.85e-14). The new model diagnostics contain new
execution times and are not recovered original logs. Original archived outputs
remain unchanged. REFIT_COMPARISON.json and provenance/COMPLETE_REFIT_VERIFICATION.json
record the comparisons and separate read-only integrity check.

Rescoring the original saved D arrays alone remains a replay, not a model fit.
Do not run the original full-prediction release check against the reduced original
results directory: it deliberately requires the omitted original model files.
Use run.py verify to check both the original archive and the complete new refit.

QUICK VERIFICATION AND REPLAY

Run these commands from the extracted package directory, or invoke run.py by its
absolute path from another working directory. Output destinations must be fresh;
relative output paths are resolved against the caller's working directory.

  python -m pip install -r requirements.txt
  python run.py verify
  python run.py rescore --output ../layout_crps_replay
  python run.py figures --output ../layout_diagnostic_figures
  python run.py figure2 --output ../figure2_replay
  python -m unittest discover -s distribution_analysis -p 'test_*.py' -v

The integrity check verifies all 180 truth hashes, 360 mask hashes, 360 evaluation
hashes, frozen template/tree identities and unchanged original code. It also
checks the new fixed-target probability/model pairs, both-arm inputs and evaluation
arrays, their audit bindings, and every file listed in MANIFEST.json. Rescore
recomputes all 2160 issued-distribution scores, empirical-minus-permuted layout
contrasts and all within-arm method pairs. It independently verifies 72 complete
CRPS distributions by the direct pairwise-distance formula and 1728 contrast
means/MCSEs, and checks every saved numeric result. The figures command uses the
included 180 score table and a frozen 600-benchmark score table; it does not refit
models or reconstruct draws from summaries. Full 600 benchmark raw data belong to
the separate Simulation supplementary package. The figure2 command redraws the
two-column main figure using the full O/U/T and fixed-target O/U controls. It
independently checks all 144 plotted paired means and MCSEs against saved
matrix-level values, including the shared empirical baseline and shared Mk Brier
probabilities. It fits no models and recomputes no completions or parsimony scores.

ORIGINAL GENERATION AND FULL REFITTING

  python run.py generate --output ../layout_original_rerun
  python run.py fit-original --output ../layout_original_rerun --workers 4
  python run.py evaluate-original --output ../layout_original_rerun --workers 4

Generation creates all 180 truth files and 360 masked inputs and checks all 540
SHA256 identities against the original registry. Fitting reads masked X and
target_mask only, using a fitting manifest without true states, tree or rate. Evaluation opens truth after fitting.
A full refit is computationally expensive and may depend on platform/library
floating-point behaviour. Source seeds and implementation versions are pinned;
matching scores are checked, not assumed merely because code is present.

A resumable wrapper executes the same frozen generation, fitting and scoring
functions, preserves every fitted probability and completion array, and compares
its outputs directly with the original archived hashes and result tables:

  python reproducibility/refit_original.py --output ../layout_complete_refit --workers 8
  python reproducibility/refit_original.py --output ../layout_complete_refit --workers 8 --resume
  python reproducibility/verify_refit.py --input ../layout_complete_refit

The --resume command is needed only after interruption. Source or dependency
changes prevent resume; completed files are validated before reuse. See
REPRODUCIBILITY.txt and refit_20261006/RECONSTRUCTION_NOTE.txt for exact scope.

FIXED-TARGET CONTROL

This additional experiment holds each empirical target cell T fixed and shuffles
observed O versus unavailable U labels independently within each column outside T.
It preserves each column's O/U/T counts and uses the same 180 latent matrices and
scaffolds. It does not preserve which nontarget cells are visible. Each arm has
its own true D because changing O/U can change the observed parsimony baseline.

The additional 180 fixed-target arms were fitted from their masked inputs. Five
state predictors and six completion schemes were evaluated. Their probabilities,
model metadata, packed 512-draw completions, evaluation arrays and all condition
summaries are included. The paired empirical baseline reuses original saved D
arrays and archived cell-score rows; it is not a new empirical-arm refit. The new
fits and the replayed baseline are identified separately in the manifests.

  python run.py audit-fixed --output fixed_target/results --report ../fixed_target_audit.json --workers 4
  python run.py prepare-fixed --output ../fixed_target_rerun
  python run.py fit-fixed --output ../fixed_target_rerun --workers 4
  python run.py evaluate-fixed --output ../fixed_target_rerun --workers 4
  python run.py audit-fixed --output ../fixed_target_rerun --report ../fixed_target_rerun_audit.json --workers 4
  python -m unittest discover -s fixed_target -p 'test_*.py' -v

The first command independently checks the shipped results without changing them.
The remaining commands create and fit a fresh control. The fitting/evaluation
wrapper supplies the relocated original source directory and verifies all 13
source hashes before running. Original absolute paths in SOURCE_BINDING.json are
retained as provenance and are not needed by these wrapper commands. Archived
preparation/fitting code is preserved in execution_source/; CODE_PROVENANCE.json
records the small portability changes made after fitting. Existing outputs are
not overwritten. Do not evaluate into the shipped results directory.

ORGANISATION

DATA_DICTIONARY.tsv and FILE_CATALOG.tsv: complete per-file field/array definitions
  and package catalogue; regenerate with tools/build_data_dictionary.py after
  adding files, then update the package manifest.
refit_20261006/: complete new reconstruction of all 360 original fitted arms,
  with masked inputs, latent truth, probabilities, packed completion draws,
  model diagnostics, fresh parsimony evaluations and exact comparisons.
reproducibility/: resumable reconstruction wrapper and read-only verifier.
layout_experiment/ANALYSIS_PLAN.txt and DESIGN_FREEZE.json: original design.
layout_experiment/empirical_bridge/: frozen anatomical-layout templates, source
  matrices, topology files and their provenance. The original template-building
  script is preserved for provenance; rebuilding templates from the earlier
  historical audit requires that separate historical source archive. The included
  frozen templates are sufficient to regenerate all simulations in this package.
layout_experiment/simulation/: unchanged generation, fitting and evaluation code.
layout_experiment/simulation/results/: original archived registries/manifests,
  all saved evaluation arrays, hash-matched regenerated masks/truth, cell scores.
distribution_analysis/: CRPS scoring, independent arithmetic checks, tests, full
  condition/method tables and included reference 600-matrix score table for plotting.
figures/: four-page distribution diagnostics (Figure S3) with SVG and 300 dpi PNG
  counterparts. figure_manifest.json describes these Figure S3 assets only.
figures/figure2/: current main-text Figure 2, comparing both full O/U/T and fixed-T
  O/U randomisation in Brier and CRPS columns. All 36 scheme/condition contrasts
  are shown for each control, with one paired MCSE. Includes the plotting script,
  a 144-row numerical input table, publication PDF/PNG, caption and validation.
  The script reads saved sources elsewhere in this archive; no source table is
  inferred from figure graphics. Run it via run.py figure2 --output NEW_DIRECTORY.
figures/full_permutation_diagnostics/: retained three-metric diagnostics for the
  full O/U/T control only (Brier, absolute mean-D error and CRPS), with original
  numerical inputs, renamed PDF/PNG and a separate plotting script. These are
  not the current main-text Figure 2. Absolute mean-D errors are also reported
  in Supplementary Information S10 and S13.
provenance/: recovered-member manifest, original source bindings and input-replay
  receipt. Recorded old absolute paths are provenance only; run.py resolves all
  execution paths relative to this extracted package.
fixed_target/: separate additional experiment, its frozen design, complete final
  outputs and tests. Use the fixed-target commands above.

SCORING AND REPLICATION

Empirical CRPS scores the actual 512-draw distribution. Fair finite-ensemble CRPS
is a sensitivity score addressing a different estimand. Central 95% intervals use
linear 2.5/97.5% quantiles and inclusive truth checks. Width alone is not a measure
of calibration. All summaries retain conditions; no cross-panel score pooling is
performed. Each layout-experiment mean/MCSE uses 30 independent latent matrices;
paired comparisons use the SD of within-matrix differences divided by sqrt(30).
The 600-matrix reference figure uses 50 matrices per condition. Completion draws,
cells, methods and fixed topologies are not additional independent replicates.

SCIENTIFIC LIMITS

Binary independent-character evolution, unit branches and fixed scaffolds do not
recreate fossil-preservation, publication or native multistate/inapplicability
processes. Documentary historical endpoints are not specimen-level anatomical
truth. A lower score or higher truth inclusion does not establish universal
calibration or a universally preferable completion scheme. The added CRPS and
fixed-target analyses are retrospective extensions, not original preregistration.

Consolidated-release note (7 October 2026)
This module is distributed inside one combined code/data archive. See ../README.txt
and ../CLEANUP_MANIFEST.tsv. Redundant rendered figures and superseded packaging
receipts were omitted; scientific source files, arrays, original execution logs
and source-binding manifests retain their original bytes. Figure scripts and
figure data tables remain. Review limits recorded above still apply.
