ACTUAL TREE-INFERENCE EXTENSION FOR THE 600 FITTED-METHOD SIMULATIONS

Scope and analysis history
This directory contains the topology assessment reported in Supplementary
Information S14 and Table S28. It reuses the same 600 source matrices and saved
completion arrays; it is not a new set of 600 independent simulations.

The main reported results are in results_amended/. The original three-random-
start results are retained in results/ as a search-sensitivity record. The
original strict-NNI search recovered poor trees even from truth-complete data:
575/600 such endpoints had length above the generating-tree length. After this
was observed, the design was explicitly amended to add a data-derived NJ start.
DESIGN.json records the original choices. DESIGN_AMENDMENT.json records the
post hoc amendment; neither analysis is claimed to be externally preregistered.
These frozen JSON records and production-code bytes retain their run hashes.

Inputs and treatments
Each source matrix has 48 tips and 40 unordered binary characters, including
eight complete anchor columns. The treatments are the source incomplete matrix,
its truth-complete matrix, five distinct cellwise maximum-probability completions,
and eight fixed saved draws from each of six original completion schemes.
Joint Mk shares marginal Mk's deterministic completion; its MAP label is an
alias, not a sixth fitted predictor. Saved draw indices are 0, 73, 146, 219,
292, 365, 438 and 511 (zero-based), fixed before topology evaluation. No new
prediction fits, completion draws, masks or generating trees are produced.

Search
The score is ordinary total unweighted unordered parsimony length; -1 denotes
an unknown state. Exact Fitch messages score all 2(n-3) NNI neighbours at each
step. The lowest-score strictly improving move is accepted, with fixed edge
and alternative order breaking move ties. Searches stop when no lower-scoring
NNI neighbour remains or after 200 accepted moves. Neutral moves are not taken.

Three independent random labelled starting trees are shared by all treatments
within a source matrix. Their NumPy SeedSequence is [2026100407,
scenario_ordinal, replicate, start_index]. The amended search additionally
uses one NJ tree inferred from each incomplete or completed matrix itself.
NJ uses the mismatch proportion over jointly observed characters, all 40
columns, and the historical implementation's upper-triangle row-major Q ties.
Every production pair has at least eight complete anchor observations. The
implementation also explicitly records a generic zero-overlap fallback.
Branch lengths do not enter the parsimony search. The added NJ endpoint is
combined with the original three endpoints without rerunning those searches.

All distinct endpoint topologies at the minimum length encountered across
starts receive equal weights after topology deduplication. Truth never breaks
a tie. The generating tree is supplied only to a separate post-search scorer;
it never initializes NJ or NNI. Retained endpoints are not all globally
most-parsimonious trees or posterior samples. Four starts and strict NNI remain
a heuristic; no global optimality is claimed. The amended truth-complete
endpoint still exceeded generating-tree length in 30/600 matrices. A length
at or below the generating-tree score does not itself prove global optimality.

Metrics
Normalized unrooted RF is the size of the symmetric difference between the
estimated and generating nontrivial split sets divided by 2(n-3). RF is averaged
over distinct tied endpoints. All endpoints are binary, so split recall is
1-RF rather than independent evidence. Strict-consensus resolution, split
recall and precision describe only the encountered endpoint ties. Precision
is undefined for a consensus without any split. For sampled treatments its
average first excludes undefined draw values, then excludes matrices without
a defined value. All amended records have defined precision. Primary RF always
includes every draw and matrix.

The eight selected draws are averaged within each matrix and scheme. Scenario
means and MCSEs use 50 independent matrices; MCSE is sample SD/sqrt(50).
Differences from both baselines and between schemes are paired within matrix.
Draws, retained endpoints and starting trees are not independent replicates.
The truth-complete matrix is a finite-data comparator, not an error-free bound.
The fixed-scaffold D values are retained for descriptive comparison only;
search never minimizes D. Exact-D-match/worse-RF counts are not a D-RF
correlation analysis. For sampled summaries, fixed_scaffold_absolute_D_error
is the mean of eight draw-level absolute errors, not the absolute error of
the original 512-draw mean D.

Files
nni_search.py: Fitch scores, all-NNI scoring, random starts and heuristic search.
nj_start.py: matrix-derived NJ start and truth-independent endpoint union.
run_topology_extension.py: original three-random-start analysis.
run_nj_amendment.py: add NJ starts while reusing original saved endpoints.
test_nni_search.py and test_nj_start.py: nine computational tests.
validate_oracle_registry.py: exhaustive six-tip score/split verification.
audit_baseline_recovery.py: node/row binding, independent scores and split/RF audit.
summarize_topology_extension.py: recreate Table S28 and descriptive counts.
DESIGN.json and DESIGN_AMENDMENT.json: frozen choices and analysis history.
ENGINEERING_FEASIBILITY.json and NJ_AMENDMENT_ENGINEERING.json: fixed-case checks.
ORACLE_REGISTRY_VALIDATION.json: completed exhaustive score/split check.
baseline_audit/: baseline audit results and computational receipt.
validation/: independent NJ distance, label, completion and endpoint checks.
PACKAGE_RELOCATION_VERIFICATION.json: portable entry-point checks.
summary/: Table_S28.csv, Table_S28_compact.tsv and TOPOLOGY_SUMMARY.json.
results/: original search; results_amended/: amended main search.

Requirements and portable commands
Python 3.10 or later, NumPy and pandas. The baseline audit also imports the
original supplied Sankoff scorer. No phylogenetics executable or R is needed.
Run the following commands from the enclosing archive root, which contains
simulation/, methods/, oracle/ and topology_extension/:

  python -m unittest discover -s topology_extension -p 'test_*.py' -v
  python topology_extension/validate_oracle_registry.py --oracle-root oracle --output ../topology_registry_check.json
  python topology_extension/summarize_topology_extension.py --results topology_extension/results_amended --output ../table_s28_replay
  python topology_extension/audit_baseline_recovery.py --root . --original-results topology_extension/results --amended-results topology_extension/results_amended --output ../topology_baseline_replay

Independent NJ/input-boundary replay (write a new receipt file):

  python topology_extension/validation/review_nj_amendment.py --root . --topology-dir topology_extension --original-results topology_extension/results --amended-results topology_extension/results_amended --output ../nj_amendment_review.json

To repeat only the added NJ searches using the supplied original endpoints:

  python topology_extension/run_nj_amendment.py --root . --original-results topology_extension/results --output ../topology_amended_replay --workers 6

To reproduce both search stages in new directories:

  python topology_extension/run_topology_extension.py --root . --output ../topology_three_start_replay --workers 2
  python topology_extension/run_nj_amendment.py --root . --original-results ../topology_three_start_replay --output ../topology_four_start_replay --workers 6

Search output directories must be new/empty and outside the source archive.
Worker count affects runtime, not seed or tie decisions. --limit is an
engineering option only; omit it for the full 600-matrix analysis. Scientific
results are not regenerated by the top-level 600-matrix verify command alone.

Result formats (the same four CSVs in each results directory)
selected_completion_topology.csv: 33,600 baseline/MAP/draw treatment rows.
Identifying fields are matrix_id, scenario_id, q, missing_per_response,
mask_mode, kind, scheme and draw_index (-1 for baseline/MAP). A SHA256 binds each
searched int8 matrix. minimum_found_length and endpoint/consensus metrics
refer to the finite search. RF_difference_from_source_incomplete and
RF_difference_from_truth_complete are paired differences; positive means
worse than that baseline. Binary worse_RF_than_* and fixed_D_exact_match flags
are 0/1. D_exact_and_RF_worse_than_truth_complete is their conjunction.

matrix_topology_summary.csv: 8,400 rows, averaging selected draws within matrix.
n_selected_completions is eight for sampled treatments and one otherwise.
Most numeric fields therefore become means; draws do not increase the number
of independent matrices. source_incomplete_RF and truth_complete_RF retain
paired comparators. The joint-Mk MAP row duplicates marginal-Mk MAP numerically.

scenario_topology_summary.csv: 168 rows, with 50 matrices per row. For each
metric, *_mean and *_MCSE are the matrix-level mean and standard error.
paired_scheme_contrasts.csv: 1,440 rows. mean_difference is candidate minus
reference and paired_MCSE uses the within-matrix differences. Negative RF
differences favour the candidate. kind distinguishes MAP from draw means.

search_records/*.json.gz retain source hashes, original seed/start edges,
every start's endpoint/score/decreasing trajectory and search diagnostics,
all retained best endpoints, and per-treatment metrics. Amended records also
include NJ start edges, NJ diagnostics and the original receipt hash. Exact
identical input matrices reuse their deterministic search result but retain
all treatment labels. NaN encodes undefined precision in the original JSON;
CSV numeric blanks carry the same meaning. RUN_MANIFEST.json records actual
counts, code/design hashes, environment versions and step caps.

Completed verification
Nine tests passed. These cover exhaustive partial four-tip scoring, all NNI
scores versus full recalculation on 5/6/9/48 tips, small-tree enumeration,
five-tip single-character exhaustive search, immutable inputs, deterministic
moves, distinct-endpoint ties, additive NJ quartets and masked distances.
The supplied six-tip registry matched exactly: 105 trees x 729 partial
patterns = 76,545 character scores, plus all 105 split sets. This checks the
score/representation, not global 48-tip optimisation. The baseline audit
checked 600 row bindings and independently rescored 2,458 retained endpoints
across both designs using original Sankoff and edge-cut split calculations.
All checks passed. The amended production run added 32,998 NJ searches and
contained no step-cap records.

An additional independent replay checked 58 scalar-distance/fallback cases,
additive-tree labels and row permutations, distinct endpoint ties, all 56
treatment inputs and 220 endpoint scores for s00_r000, and the complete-anchor
precondition of all 600 source matrices. Its script and receipt are supplied.
