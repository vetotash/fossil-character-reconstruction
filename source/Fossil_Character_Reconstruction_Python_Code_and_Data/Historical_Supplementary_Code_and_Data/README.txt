Historical character prediction code and data

Current author verification
The author has rerun the analysis code and independently verified the
source-based judgements and scientific interpretations. The author confirmed numerical
agreement with all retained manuscript, figure, table and archive data. Previously
prepared outputs were retained because the rerun outputs could not be integrated
into the assembled submission. See AUTHOR_VERIFICATION_STATEMENT.txt for the
AI-use declaration and the scope of this completed independent human review.
Earlier archived receipts retain the scope and date of the checks they record.

This archive accompanies Historical information patterns and probabilistic
character reconstruction in fossil vertebrates. All new analysis scripts are Python.
The original published matrix states are documentary references rather than
independently established anatomical truths.

Start with [REPRODUCIBILITY.txt](REPRODUCIBILITY.txt) for the recorded Python and
PAUP versions, commands, evidence locations and limits of numerical replay,
diagnostic refitting, all 532 blocked treatments and native rescoring.
[DATA_DICTIONARY.tsv](DATA_DICTIONARY.tsv) describes data fields and array axes;
[FILE_CATALOG.tsv](FILE_CATALOG.tsv) locates the supplied files.
Read [DOCUMENTATION_GUIDE.txt](DOCUMENTATION_GUIDE.txt) for schema conventions,
the special native-PAUP TSV layout and missing-value rules.
[ARRAY_LAYOUTS.tsv](ARRAY_LAYOUTS.tsv) lists the actual shape and dtype of every
supplied NumPy array; [DOCUMENTATION_COVERAGE.json](DOCUMENTATION_COVERAGE.json)
records coverage and any field semantics still unresolved from the delivered evidence.
[RELEASE_NOTES.txt](RELEASE_NOTES.txt) describes this documentation/portability
revision; [consolidated RELEASE_VALIDATION.json](../RELEASE_VALIDATION.json)
records the packaging checks and their stated scope.
The self-contained index builder is
[documentation/build_indexes.py](documentation/build_indexes.py); run
`python -B documentation/build_indexes.py` from this root to rescan actual files
and regenerate the dictionary/catalog/array layouts/guide/coverage report.
It requires no external inventory cache and leaves scientific results and the
root integrity manifests unchanged.

Licensing: [LICENSING.txt](LICENSING.txt),
[LICENSE_CODE_MIT.txt](LICENSE_CODE_MIT.txt),
[LICENSE_DATA_CC_BY_4_0.txt](LICENSE_DATA_CC_BY_4_0.txt),
[THIRD_PARTY_NOTICES.tsv](THIRD_PARTY_NOTICES.tsv) and
[UPSTREAM_LICENCE_REVIEW.json](UPSTREAM_LICENCE_REVIEW.json).
Original code is MIT and original derived materials are CC BY 4.0, subject to
source-specific upstream exceptions: S020 has no established open redistribution
licence and is excluded from blanket grants; S021 retains the article-level
CC BY-NC 4.0 restriction, with no separate matrix-specific licence established.
See the new upstream review for source-specific evidence and unresolved scope.

The archived Python environment is 3.12.14; requirements.txt pins its recorded
scientific-library versions. Numerical replay does not need PAUP. Native
rescoring requires the separately installed PAUP 4.0a169 Ubuntu64 executable and
its documented runtime libraries. Original tetrapod control selections survive
as frozen records, but the original selector and complete original fitting logs
have not been recovered in this package; see REPRODUCIBILITY.txt.

Quick numerical replay

  python -m pip install -r requirements.txt
  python -B run_analysis.py verify
  python -B run_analysis.py replay --output ../historical_replay

The output directory must not already exist. Replay verifies the supplied file
hashes, recomputes state scores for all five histories from archived categorical
probabilities, and reconstructs the 72 primary and 72 full-source fixed-tree
summaries from character-level parsimony scores. It checks every summary against
the archived values. Each fixed-tree regime has 72,000 draw-tree observations;
these are conditional computational draws, not independent biological samples.

Portable diagnostic refitting

[portable_diagnostic_refit.py](portable_diagnostic_refit.py) calls the archived
estimators using automatically located, hash-checked frozen inputs and writes
new predictions and execution logs outside this archive. From this root:

  python -B portable_diagnostic_refit.py verify --history all --condition all --output ../historical_refit_input_check
  python -B portable_diagnostic_refit.py smoke --history tetrapods --condition shared --output ../historical_refit_smoke
  python -B portable_diagnostic_refit.py fit --history all --condition all --output ../historical_diagnostic_refit

Each destination must be new. Smoke fits one response character with five
methods; fit covers both training conditions for tetrapods/dinosaurs and the
full-source ichthyosaur condition. The new wrapper preserves code/input hash
checks and records newly generated logs, not recovered original logs. The
unavailable ichthyosaur PROTOCOL.txt is explicitly NOT_CHECKED_NOT_SUPPLIED;
all executable and fitting-input hashes must pass. This does not claim that
the original fish runner's entire historical gate has been replayed. See
REPRODUCIBILITY.txt for details and the limited smoke-validation scope.
The actual new records are included under
[reproducibility/diagnostic_smoke_20261006/](reproducibility/diagnostic_smoke_20261006/):
smoke/ contains the one-character, five-method run, and input_check/ contains
the separate 38-file input verification. These are new records from this
packaging revision, not a complete historical refit or recovered original logs.

Data organization

predictions/ contains five primary prediction tables. Each row identifies a
history, control or later-target scenario, original taxon, one-based character
number, method, recorded reference state, predicted state, training sample size,
categorical probabilities and scores. The later-target cohorts contain 350
tetrapod, 1333 dinosaur, 196 ichthyosaur, 212 gnathostome and 163 squamate entries.
An entry is a taxon-by-character cell. State labels are categorical strings.

primary/ contains the exact 1874 endpoints used for fixed-tree comparisons,
their frozen inputs, the original numerical synthesis code and published
summaries. There are 349 tetrapod, 1333 dinosaur and 192 ichthyosaur endpoints.
State-prediction and fixed-tree cohorts are deliberately distinguished.

full_source/ contains the full-source training sensitivity, old training
matrices, target registries, fitted probabilities, frozen completion draws,
reference trees and character score arrays. source_data/original/ and
source_data/sensitivity/ retain the original relative structure of scientific
inputs, Python sources and numerical evidence for historical source curation,
ordering conventions, target-block analyses and state-only comparisons.
SOURCE_PROVENANCE.json maps copied members to their authenticated source
archives. The present MANIFEST.json lists the final delivered files and hashes.

Binary NPZ files are NumPy ZIP containers and must be read with
numpy.load(path, allow_pickle=False). Historical character matrices store
original single-state, unknown, uncertainty or polymorphic tokens as strings;
do not convert polymorphism to uncertainty or treat '-' as an observed state
without checking the associated source convention. In character-score NPZ
files, steps has dimensions tree x treatment/draw x character; minima has
dimensions treatment/draw x character. characters and trees define the axes.
The hard-score treatment axis contains the source, later codes and five MAP
completions. The joint Mk scheme shares the marginal Mk cellwise MAP marker.

Brier scores sum squared errors over all categorical states. Log loss uses the
natural logarithm with the recorded floor of 1e-12. D is the change in fixed-tree
parsimony length minus the change in completion-specific minimum length, in
steps. Original arrays use native PAUP character lengths and minima, preserving
ordered and polymorphic character conventions. The supplied Python replay
recomputes their arithmetic; it does not reimplement the native scorer or claim
a new PAUP run. Historical native regeneration requires the documented external
PAUP installation. No third-party executables are redistributed. Archived
source scripts may retain historical paths; use the portable entry points and
the explicitly supplied source-root arguments for the new analyses.

Additional analyses

blocked_validation/ contains the frozen old-label tree-block diagnostic, all
532 treatment forecasts, the paired random controls, scores, original fitters
and semantic tests. Its README.md gives refit and evaluation commands. Supply
the relocated full_source directory with --source-root. The blocks use the
display orientation of the published unrooted trees. Other character
observations remain available, and no later label is used in these fits.

source_sensitivity/ contains the two source-triggered exclusion analyses.
The first leaves the documented Laquintasaura character366 conflict unknown in
all completions and the later reference. Its Python scorer is checked against
all 12,014 archived native lengths for the binary unordered affected character
before replacing that character's contribution. The second excludes all13
gnathostome character448 targets whose recorded states were already observed
in source-declared duplicate450. Neither analysis changes published labels or
refits a predictor; both retain the full original cohort alongside the subset.

From this archive's root, run either analysis into a new sibling directory:

  python -B source_sensitivity/source_conflict_sensitivity.py --source source_data/original --output ../historical_conflict_replay
  python -B source_sensitivity/duplicate_character_sensitivity.py --predictions predictions/gnathostomes.csv --keys source_sensitivity/gnathostome_duplicate448_verified_keys.csv --output ../historical_duplicate_replay

Both output directories must not already exist. The first command verifies the
affected character against the archived native scores before calculating the
exclusion; it is not a replacement implementation of the full native scorer.

anatomical_sources/ contains the complete endpoint register, fixed100-entry
documentary sample, source locators, evidence summaries and machine-readable
review fields. Its initial documentary assessment was AI-assisted; the author
has subsequently independently verified the source-based judgements and
scientific interpretations. This completed human review is separate from the
archived, uncompleted anatomical-expert review forms. An unresolved entry is not
classified as anatomically incorrect. Blank expert fields remain historical
records and must not be interpreted as completed anatomical-expert reviews.

Run the portable audit integrity and primary-matrix checks from this root:

  python -B anatomical_sources/verify_documentary_register.py
  python -B anatomical_sources/verify_packaged_matrices.py

The first command checks authored registers, frozen selections and labels, and
blank expert records; it explicitly leaves external article files NOT_CHECKED.
The second verifies the ten included old/later primary matrices. source_index.json
records DOI/URL, page/figure/character locators and hashes for all 24 inspected
sources. Article PDFs and full-text extracts are not redistributed, so neither
command claims offline verification of omitted full texts or anatomical truth.
local_source_verification_report.json is a receipt from the original audit
workspace, where those source files were available, not a claim about this ZIP.
The original per-history audit scripts preserve their acquisition-workspace
paths; use verify_documentary_register.py for portable register verification.

Additive documentary follow-up

anatomical_sources/definition_followup/ records the subsequent AI-assisted
documentary review of the twenty sampled dinosaur character correspondences and
the history-specific OTU admission and exclusion decisions. The author has
independently verified these source-based judgements and their scientific
interpretation; see AUTHOR_VERIFICATION_STATEMENT.txt. The revised SI
presents the definition follow-up in S16.2 and the Inherited column of Table S30,
and the OTU decisions in S16.5. The nineteen additional inheritance routes are
distinct from the original Def evidence classification and from independent
anatomical validation. Original audit records, frozen labels, model inputs,
predictions and numerical results are preserved. Source URLs, hashes and precise
locators are supplied; publisher PDFs and derived full texts are not included.

  python -B anatomical_sources/definition_followup/verify_followup.py

This command checks the original audit hashes, endpoint identity, archived matrix
tokens, evidence-route counts and recovered merger-exclusion accounting. It does
not repeat documentary reading, establish anatomical truth or validate model
performance. The directory's README.txt explains the additional evidence and
remaining limits. PACKAGE_VALIDATION.json and the older audit validation records
remain historical receipts of their original checks; they have not been replaced
or relabelled as new validation results for this revision.

Fresh blocked refitting

The supplied blocked fit command resumes missing tasks; with all supplied task
artifacts present it does not fit again. To recompute all 532 treatments without
altering this archive, run these commands from the archive's parent directory:

  cp -a Historical_Supplementary_Code_and_Data Historical_Supplementary_Code_and_Data_refit
  cd Historical_Supplementary_Code_and_Data_refit/blocked_validation
  mv results results_supplied
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -B src/blocked_validation.py fit --source-root ../full_source --workers 2
  python -B src/blocked_validation.py evaluate --source-root ../full_source
  python -B -m unittest discover -s tests -v

Keep the frozen split plan and vendor files; do not invoke plan again. The first
fit should report 532 pending treatments and creates a new results directory.
The source-root override changes only storage location: all source and fitter
hashes are checked against the frozen plan. A full refit takes substantially
longer than the numerical replays. The copied archive's original outer manifest
will no longer describe regenerated result files; preserve the delivered copy
and compare new summaries with results_supplied before drawing conclusions.
The archived EVALUATION_MANIFEST.json records all 532 completed treatments.
PACKAGE_VALIDATION.json records evaluation of those saved outputs and a fresh
relocation probe of one treatment only, not a second complete 532-treatment fit.

The associated Simulation_Supplementary_Code_and_Data.zip contains the600-matrix
benchmark, oracle simulations and fitted-completion topology searches.
Layout_Supplementary_Code_and_Data.zip contains the180-matrix historical-layout
experiment, fixed-target observation control, CRPS analyses and figure code.
Each archive records its own computational and data-access limits.

Consolidated-release note (7 October 2026)
This module is distributed inside one combined code/data archive. See ../README.txt
and ../CLEANUP_MANIFEST.tsv. Redundant rendered figures and superseded packaging
receipts were omitted; scientific source files, arrays, original execution logs
and source-binding manifests retain their original bytes. Figure scripts and
figure data tables remain. Review limits recorded above still apply.
