Round 17 observed matrix geometry and historical endpoint templates
=================================================================

Purpose and boundaries
----------------------
This extraction uses the three already audited Round 14 downstream panels.
It reads no method performance and performs no simulation or fitted comparison.
It provides actual observed availability patterns, audited historical target
locations, and fixed reference topologies for a simulation design to freeze.
Later published states are deliberately absent from the portable templates.
Generated binary labels at these positions will be simulation truth, not the
actual later anatomical observations. The target mask inherits the historical
endpoint scope and its documented exclusions; it is not an unbiased sample of
all cells that could have been researched in future.

The recommended files are <panel>/admitted_template.npz. Their dimensions are:
                     taxa  predictor columns  target cells  target columns
tetrapods              95                339           349              87
dinosaurs              74                457          1333             367
ichthyosaurs          101                242           192             128

There are no completely observed singleton predictor columns in any template.
This is a direct difference from the previous 48 x 40 simulation's eight fully
observed anchor columns. In the archived downstream matrices there also happen
to be no columns without a literal question mark, but the precise scientific
description remains singleton availability, not lack of anatomical observation.

Labels in all template files
----------------------------
0 = a simple singleton source state matching [0-9A-Z]. This location becomes a
    generated binary 0/1 visible to the fitting code. Actual fossil state values
    are not recoded or interpreted as binary by this extraction.
1 = a non-target unavailable model entry. This includes source '?' and, where
    present, '-' and compound state tokens. It remains '?' in the binary fit.
2 = an already audited historical endpoint. All such positions are literal '?'
    in the archived downstream source panel. They remain '?' during fitting;
    simulated hidden states are used only at evaluation/partial completion.

The term unavailable concerns this specific singleton-state fitting interface.
It does not mean that a polymorphic or uncertain anatomical observation did not
exist. Parentheses are NEXUS polymorphism notation, braces are state uncertainty
notation, '?' is missing, and '-' is the declared NEXUS gap symbol. We do not
infer a biological preservation mechanism from any one of these symbols. The
binary simulation treats all non-singleton tokens as unavailable, a stated
simplification. The native empirical scorer retains their original distinctions.
These masks come from archived model/downstream preparation, not an untouched
original publisher file; in particular normalized dinosaur '?' tokens must not
be used to infer which original entries were dashes or biological missingness.

Exact source-token inventories for the full downstream panel are in each
description.json. Tetrapods include 84 compound entries (35 parentheses and 49
braces); dinosaurs contain only simple states and '?' in this prepared panel;
ichthyosaurs include 708 '-' and two parenthesized entries in the 287-column
panel. The admitted 242-column projection is separately preserved.

NPZ schema
----------
admitted_template.npz (preferred for the simulation):
  names: 1D Unicode array in exact old downstream taxon order.
  characters: original one-based source character IDs, length p.
  labels: int8 n x p array of the three classes above.
  observed_mask: bool n x p, exactly labels == 0.
  target_mask: bool n x p, exactly labels == 2.
  source_question_mask: bool n x p retaining literal archived '?' locations.
  nonsingleton_or_gap_mask: bool n x p retaining other unavailable locations.
  target_columns: zero-based columns in THIS projected template with >=1 target.
  original_column_indices: zero-based columns in the complete downstream panel.
  original_ordered_characters: original one-based ordered IDs for provenance;
      generated binary states use the frozen simulation cost model, not a
      claim that original multistate ordering has been reproduced.

template.npz: full downstream column set (339/457/287 respectively); adds
  admitted_predictor_columns (zero-based full-panel positions) and
  admitted_predictor_characters (one-based original IDs).
  Its target_columns refers to its OWN full column space.

source_panel.npz is a byte-exact copy of Round 14 panel.npz for provenance.
Do not supply its fossil state values to a synthetic-data estimator.
target_positions.csv gives full-panel row/column positions and original IDs,
with no later observed labels. Use characters mapping when projecting columns.
by_taxon.csv and by_character.csv describe the FULL archived downstream panel;
description.json additionally reports admitted predictor counts and distributions.

Fish character-space distinction
-------------------------------
The original fitting matrix has 115 taxa x 242 admitted source character IDs.
The fixed-tree native scoring background has 101 taxa x 287 source characters.
All 101 common taxa and 242 common character columns are exactly equal between
these objects. The extra 45 downstream columns are preserved for provenance,
but excluded from the recommended synthetic predictor template to match the
admitted historical fitting character set. Their old source definitions could
in principle support a different source-only simulation; that would expand the
historical predictor design and is not done here. In the historical downstream
score, only the 128 response characters contribute, so preserving 287 columns
did not make the extra 45 into fitted response endpoints.

Likewise historical full-source fitting used 102 tetra/75 dino taxa, versus
95/74 in the downstream trees. The templates therefore represent fixed-tree
panel geometry, NOT a re-run of full-source historical fitting. All shared cells
are checked identical and all targets are subsets of frozen historical keys.
Only this row/predictor-scope distinction is introduced; no scored future label
is imported as a predictor. Target source counts are deduplicated by taxon and
character because old prediction files repeat keys across methods.

Trees and explicit branch-length convention
------------------------------------------
tree_topologies.json contains all 5/2/5 existing reference trees in source-file
order. Each record has name, edges, root, n_tips, n_nodes, and edge_lengths.
Tip node i is always names[i]; internal IDs are consecutive starting at n_tips.
The designated computational root is n_tips. Each record is connected/acyclic;
all original taxa occur exactly once. Redundant degree-two internal nodes are
suppressed, then every edge is assigned length 1.0. Original files contain no
branch lengths. Unit lengths are a simulation convention, not estimated branch
lengths or durations. After suppression all supplied trees are binary; no
arbitrary resolution of polytomies is required.

An outcome-independent default would use the first OLD source tree only:
  tetrapods O1: 95 tips, 187 edges, 93 degree-three internal nodes.
  dinosaurs baron: 74 tips, 145 edges, 72 degree-three internal nodes.
  ichthyosaurs moon_source_0001: 101 tips, 199 edges, 99 internal nodes.
The remaining topologies are retained for audit, not newly selected by fit.
Later tetrapod R1/R2/R3 and dinosaur langer topologies must not accidentally be
called source trees. Selection of a particular generating tree belongs to the
separate frozen design, not this extraction.

Runnable paired-mask design interface
------------------------------------
Load labels and tree from the files above. For each frozen response subset,
generate one complete binary Mk matrix X on the fixed source topology. Original
arm: X_visible = X with labels != 0 masked; evaluate only labels == 2. Permuted
arm: independently permute each full column of the THREE-class label array,
including predictor-only columns, keeping X, topology and generating rate
fixed. This preserves per-column observed, background-unknown and target counts
while disrupting their taxon placement and repeated cross-character placement.
Permutation seeds must be frozen before results. This is a compound geometry
intervention: it cannot isolate phylogenetic clustering from repeated taxa or
row completeness. Fit only frozen response columns; retaining other predictor
columns does not require predicting or completing all unknown background cells.
Downstream D must compare source against TARGET-ONLY completion. Non-target
unavailable cells remain unknown in the source, true-target, and predicted-target
matrices. Filling every missing cell would change the empirical estimand.

Runtime risks known before fitting
---------------------------------
The full 367 dinosaur response fits are much larger than Round 16's 32-column
fit. A uniformly sampled, seeded 32-response-column subset per panel, frozen
before new outcomes and shared by both arms, is a defensible scoped test if
engineering timing demands it. Such a test is not all 1,874 historical targets.
It must record the selected IDs and selected target counts. No response may be
dropped because a fitted method performs poorly. For all eligible response
columns, minimum observed training rows are 24/15/19, respectively, hence no
empty training response occurs under within-column label permutations.
Zero overlaps in predictor comparisons and single generated training states
remain possible and must use the already frozen fallback rules.

Observed descriptive signal, without fitted conclusions
-------------------------------------------------------
Recommended-template unavailable fractions are 30.151%, 58.011%, 47.578%.
All-target taxon concentration is substantial and unequal: the five taxa with
most historical endpoints hold 36.676%, 29.182%, 71.354% of each panel's targets.
The largest single-taxon counts are 33/349, 112/1333 and 56/192 respectively.
These are descriptive summaries of the full endpoints, not performance results
or evidence that any imputation method will fail. Per-taxon/character rows and
concentration diagnostics are archived so a design can be checked against them.

Rebuild and verification
------------------------
Run: python new_project_search/round17/empirical_bridge/build_templates.py
Dependencies: Python, numpy. Each panel's SOURCE_MANIFEST.json hashes the exact
Round 14 panel, cells, reference trees, smoke check, full training input and frozen
target file. OUTPUT_MANIFEST.json hashes all portable outputs and the extractor.
The extractor asserts simple/target disjointness, literal '?' for every target,
nonduplicated endpoint positions, exact common source tokens, unique source
tree leaves, and tree edge/node degree conditions. Independent audit is separate.
