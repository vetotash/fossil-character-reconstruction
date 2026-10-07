EXACT TOPOLOGY EXTENSION

This extension scores the saved six-tip completion draws from the Round 15
controlled simulation. It generates no new character matrices or random draws.
Dependencies: Python 3 and NumPy. No PAUP binary or external topology package.
Run commands from the round15 directory:

python -B topology/prepare_manifest.py --simulation-results simulation/results --output topology/topology_input_manifest.json
python -B topology/run_topology.py --manifest topology/topology_input_manifest.json --output topology/results

The first command rebuilds relative input paths and hashes from the simulation's
scenario registry. It expects all 24 six-tip scenarios and DESIGN_FREEZE.json.
The second enumerates all 105 labelled unrooted binary trees and scores each of
729 binary/missing tip patterns. It scores 2,400 independent matrices and all
3 x 64 saved completion draws per matrix. Cost-table creation and full scoring
are inexpensive and deterministic. Running over the same outputs replaces
reproducible output files; source simulation matrices are never modified.

Inputs
Each simulation/results/sXX_matrices.npz has complete pattern IDs (bit j is tip
j), missing-tip bit masks, common marginal cellwise-MAP IDs for three conditions,
and draw IDs [100, 3, 64, 32]. Scheme order is oracle_joint, product_marginals,
sharpened_product. Draws are exactly those saved by the simulation. All observed
cells are verified unchanged. Generating-tree D, complete-truth D, and MAP D
must match the independently computed values in the main simulation.

Outputs
- results/registry/six_tip_tree_registry.json: all tree edges, indices and
  canonical nontrivial bipartitions; internal-node labels have no significance.
- results/registry/six_tip_cost_registry.npz: costs[105,729]; ternary pattern 2
  admits either state; full_to_ternary maps binary pattern IDs to this registry.
- results/arrays/sXX_topology.npz: full/source/map/draw metrics; all optimal-tree
  membership bitsets; and complete/source/MAP tree scores. MPT bits are packed
  little-endian along the final axis, 14 bytes for 105 possible trees.
- topology_matrix_summary.csv: 7,200 matrix-by-condition rows. Means over the
  64 completion draws are within-matrix summaries, not independent samples.
- topology_scenario_summary.csv: all 72 scenario-by-condition rows. Means and
  Monte Carlo standard errors use the 100 independent matrices. For conditional
  quantities undefined when there are no D matches, n_valid is explicit. Pooled
  count ratios are descriptive and have no draw-level confidence interval.
- topology_paired_contrasts.csv: all three condition pairs in each scenario;
  differences are left minus right and their MCSE uses paired matrices.
- TOPOLOGY_CHECKS.json records input/output hashes and automated gates. The
  separate independent_review files record independent verification.

Topology definitions
Every most-parsimonious tree is retained. Mean normalized RF averages the
symmetric difference between each optimal tree's split set and the generating
tree's split set, divided by six. Truth membership and MPT count are separate
outcomes: truth in a large unresolved set is not the same as unique recovery.
No closest tied tree is chosen. The known complete character matrix can itself
fail to recover the generating topology because characters are finite.

D and paired complete-matrix baseline
D is computed on the generating tree and is independent of selecting MPTs.
D_exact_match_rf_worse_than_full counts draws with D identical to the actual
complete-data D AND mean all-ties RF exceeding that same complete matrix's RF
by more than 1e-12. Both its joint frequency among all draws and its conditional
frequency among D matches are retained, alongside the unconditional rate of
RF exceeding the complete-matrix baseline. This is an endpoint-identifiability
diagnostic, not a causal comparison of matched and unmatched draws.
The added paired-full-baseline description was requested after initial result
inspection; SUMMARY_REFRESH.json makes that timing explicit. All scenarios
were retained and score arrays remained unchanged. refresh_summaries.py can
reconstruct descriptive CSVs from the saved score arrays without rescoring.

Interpretive limits
Generating topology, edge lengths and rates are treated as known when the
probabilities are computed. They are extra information supplied to the oracle.
Consequently a cellwise-MAP completion can appear closer to the generating
tree than the original complete finite-character matrix. This must not be
interpreted as a practicable method outperforming anatomical truth. None of
these controlled results changes the fixed-tree nature of the Round 14 fossil
analysis or validates the fossil scoring anatomically.
