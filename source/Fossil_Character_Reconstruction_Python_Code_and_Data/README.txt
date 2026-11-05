REPRODUCIBILITY DATA AND PYTHON CODE
Historical information patterns and probabilistic character reconstruction in fossil vertebrates

1. PURPOSE AND CONTENTS
This consolidated archive contains the original published matrix inputs used by
the study, processed analysis inputs, derived numerical results, Python source
code, reproducibility evidence and data documentation. It is a cleaned data/code
release, not the manuscript or Supplementary Information submission file.
No scientific numeric values, matrix states, analysis algorithms or original
execution logs were changed during this consolidation.

The three module directory names and their internal scientific paths are retained
because frozen plans and source bindings depend on them:
  Historical_Supplementary_Code_and_Data/
    Five historical state-prediction datasets and three fixed-tree comparisons;
    original and revised matrices; documentary source checks; original and
    full-source fits; blocked validation and reported sensitivity analyses.
  Simulation_Supplementary_Code_and_Data/
    The 600-matrix benchmark, exact oracle studies, distribution diagnostics,
    original and amended fitted-completion topology searches.
  Layout_Supplementary_Code_and_Data/
    The 180-matrix historical-pattern experiment, 360 original-design arms,
    fixed-target control, saved distributions and separate 2026-10-06 refit.

Repeated rendered figures and superseded package-edit receipts were omitted.
Figure source code and numerical figure tables are retained. Final figures are
supplied with the manuscript/Supplementary Information. CLEANUP_MANIFEST.tsv lists
every omission and its reason. Original source ZIPs are not nested in this ZIP.
Large fitted-model JSON/NPZ files and records of original versus amended analyses
are necessary evidence, not unused drafts. SOURCE_DATASETS.tsv maps the empirical
datasets to upstream sources and distinguishes article DOI from data DOI.

2. QUICK START (PYTHON)
Extract the complete ZIP before running; keep its directory structure unchanged.
Recorded and checked environment: Python 3.12.14, NumPy 2.3.5, pandas 2.2.3,
SciPy 1.17.0, scikit-learn 1.8.0 and matplotlib 3.10.8.
Create an environment outside the extracted archive, then from this root run:

  python -m pip install -r requirements.txt
  python -B run.py integrity
  python -B run.py verify --output ../verification_new --workers 2
  python -B run.py replay --output ../replay_new --workers 2

Each output directory must be new and outside this archive. Paths containing
spaces should be quoted. integrity uses only the Python standard library.
verify checks the supplied inputs, bindings and saved predictions/results using
all three original module verifiers. replay recalculates historical summaries,
benchmark distribution diagnostics, exact-oracle checks and layout summaries.
These commands do not refit every model or regenerate the native historical
parsimony scores. Their logs are explicitly new execution records.

The unified entry point also forwards module-specific commands, for example:
  python -B run.py historical -- replay --output ../historical_new
  python -B run.py simulation -- reproduce --output ../benchmark_new --workers 2
  python -B run.py layout -- figure2 --output ../figure2_new

Read each module's README.txt and REPRODUCIBILITY.txt for complete fitting,
simulation, fixed-target, topology and plotting commands. Recomputing all fits
and topology searches takes substantially longer than saved-output verification.
Commands documented as writing in place must be run in a separate working copy.
For Supplementary Figure S1, use the additional Python replay_figure_s1.py in
the Simulation module; PORTABLE_FIGURE_S1.txt explains its saved-array checks.

3. HOW TO INTERPRET EVERY FILE AND FIELD
FILE_INDEX.tsv lists every distributed file with its module, purpose, format,
and location of the relevant data dictionary. For individual data files consult:
  Historical module: FILE_CATALOG.tsv + DATA_DICTIONARY.tsv + ARRAY_LAYOUTS.tsv
    Join catalogue schema_id to dictionary schema_id. For native PAUP TSV files,
    column_position_1based distinguishes repeated column names. ARRAY_LAYOUTS.tsv
    gives the actual dtype, shape and axis names of every NPY/NPZ array.
  Simulation module: FILE_CATALOG.tsv + DATA_DICTIONARY.tsv
    Join schema_id; JSON [*] denotes list members, and dynamic keys are described
    in the dictionary. Observed dtypes, shapes and missingness are retained.
  Layout module: FILE_CATALOG.tsv + DATA_DICTIONARY.tsv
    The dictionary maps file_path and field/key to definition, units, dtype,
    shape and notes. Per-replicate shape/null distinctions are retained.

The original matrices are morphological character data, not new measurements
or raw specimen imagery. Source-provided state symbols and polymorphic or
uncertain entries remain unchanged. Definitions follow the original character
descriptions identified in the source records. Published later states are
reference assignments, not independently established anatomical truth.

Common conventions (file-specific dictionaries take precedence):
  Morphological states: categorical and dimensionless; ? means unknown;
    '-' follows the source's convention and is not automatically an observed 0.
    Braces and parentheses preserve original polymorphism/uncertainty syntax.
  Simulated matrix states: 0/1, with -1 unavailable where documented.
  Historical-layout labels: O observed, U unavailable, T target; stored labels
    0/1/2 respectively in the Layout template arrays.
  Probability: dimensionless, 0 to 1. Brier score is a sum across states;
    its binary range is 0 to 2. Log loss uses natural logarithms (nats).
  L and M: parsimony steps. D=(L-old_L)-(M-old_M), in steps.
    D errors, interval widths and CRPS therefore also have units of steps.
  q: transition rate per unit branch length, not per geological year.
  Normalised RF: dimensionless split distance. Runtime fields: seconds.
  Native source characters commonly use one-based numbering; array positions
    are generally zero-based. The per-file dictionary specifies the convention.
  Empty/NaN: unavailable or mathematically undefined as documented, not zero.
  NPZ/NPY: read using numpy.load(..., allow_pickle=False). Packed-state bit order
    differs between modules; use the documented bit order rather than guessing.

4. PROCESSING AND REUSE
Both archived inputs and processed/derived data are supplied, with Python code
for the processing and analyses. Documentary curation that was not recoverable
is explicitly identified in the Historical module. Licences are described in
LICENSING.txt and each module's upstream notices; third-party data are not
relicensed by this combined archive. Publisher full-text articles are not bundled.

5. INDEPENDENT AUTHOR VERIFICATION AND REPRODUCTION LIMITS
The sole author reports that the analysis code was rerun and source-based
judgements and scientific interpretations independently checked. Numerical
agreement was confirmed with the retained manuscript, tables, figures and data.
AUTHOR_VERIFICATION_STATEMENT.txt records this verification report and its scope.
The accompanying manuscript contains consolidated tool-use and contribution
disclosures. Existing historical logs and blank expert-review forms are retained
with their original scope; they are not rewritten to claim later work occurred.

The current packaging checks are separately recorded in RELEASE_VALIDATION.json.
They check data preservation, structure and executable saved-result replays;
they are not a new full scientific refit or independent anatomical examination.

Historical native rescoring needs a separately installed PAUP 4.0a169 and its
documented runtime libraries. Python saved-output replay needs no PAUP. The
Simulation and Layout binary simulations/scorers are Python. No unvalidated
Python substitute has replaced historical ordered/polymorphic native scoring.
Original tetrapod selection code and complete original execution logs remain
unavailable. Frozen selections and the available genuine records are preserved.

6. SUBMISSION AND DATA AVAILABILITY
Deposit this ZIP with an appropriate research-data repository, obtain an actual
reviewer-access URL, and enter that URL in the submission form. Source dataset
DOIs identify upstream datasets, not this new code/data release. Do not represent
an unassigned accession as already deposited. State in the data availability
declaration that original source matrices and processed/derived analysis data
are supplied. Include the actual repository DOI when it has been assigned.

Root-level TSV column definitions are in documentation/ROOT_FIELD_DICTIONARY.tsv.

Optional requirements-source-review.txt lists extra document-reading packages
for retained legacy source-acquisition scripts. These are not needed for the
standard numerical entry points. Some archived acquisition scripts retain their
original workspace paths; source-document reconstruction is not advertised as
a complete portable pipeline. Use the supplied source matrices and documented
portable checks for the preserved analysis scope. SOURCE_DATASETS.tsv URLs are
taken from the actual acquisition records; publisher availability may change.
