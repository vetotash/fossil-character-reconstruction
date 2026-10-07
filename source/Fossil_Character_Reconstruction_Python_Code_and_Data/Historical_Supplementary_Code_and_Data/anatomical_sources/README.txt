DOCUMENTARY SOURCE AUDIT AND COMPLETED INDEPENDENT HUMAN VERIFICATION

Purpose
This folder records an initially AI-assisted retrospective check of published
evidence for historical prediction endpoints. The authors have independently
verified the source-based judgements and scientific interpretations. See
../AUTHOR_VERIFICATION_STATEMENT.txt for the completed author verification and
AI-use declaration. The original audit records and classifications are preserved.
This review does not add independent anatomical-expert decisions, new specimen
observations, or corrected historical labels.

This distributed folder excludes internal SI-editing drafts and external article
PDFs/full-text extracts. The original audit receipts are retained as provenance.
local_source_verification_report.json records the source files available during
the original audit; it is not an offline verification claim for this package.
PACKAGED_MATRICES.json and packaged_matrix_source_map.json identify the ten
included primary matrices. Run `python -B verify_packaged_matrices.py` from this
folder to verify them against their recorded source hashes.

Selection
REVIEW_PLAN.json and PLAN_FREEZE.json specify the rule. The original prediction
frame contains 2254 endpoints (350 tetrapods, 1333 dinosaurs, 196 ichthyosaurs,
212 gnathostomes and 163 squamates). Twenty endpoints per history were selected
by the specified SHA256 ordering. No replacement is allowed. These are prediction
endpoints; downstream eligibility is a separate property. SAMPLE_FREEZE.json
records the sample. The analyst's prior aggregate-result and source exposure is
disclosed in the plan. The audit is retrospective, not preregistered or blind.

Completed documentary audit
The five *_documentary_reviews.json files contain the source-specific records.
assemble_source_audit.py validates the selected keys and frozen labels, checks
source-file hashes, and creates:
  completed_documentary_sample.json / .csv: the selected 100 records;
  endpoint_register_completed.json: all 2254 endpoints, retaining unreviewed rows;
  documentary_audit_counts.json: separate, nonexclusive evidence dimensions;
  source_evidence_registry.json: source URLs, file hashes, recorded licences;
  assembly_manifest.json: input and output hashes.
Run from any working directory with Python 3. No third-party package is needed
for this assembly step, but assembly expects the original workspace-relative
source paths and validates the source files used during the audit. Per-history
source-audit scripts document how the manually
read source observations were attached to the fixed sample; running a script is
not a substitute for anatomical judgment.

Portable verification
Run `python verify_documentary_register.py --audit-dir PATH_TO_THIS_FOLDER` to
check the authored files, frozen selection, labels and blank expert records.
This standalone command does not need the original workspace, source articles,
internet access or third-party packages. It explicitly reports source files as
NOT_CHECKED. source_index.json maps scientific evidence to DOI/URL, hash and
page/figure/character locators; original local paths are optional provenance.
To verify separately retrieved sources, add `--verify-source-files --source-root
PATH`, optionally `--source-map mapping.json`. The mapping is a JSON object from
source IDs (for example S001) or original paths to local paths. Missing external
documents are reported as unavailable and do not become verified by implication.
Neither mode establishes anatomical truth or independent expert review.

Interpretation
An exact score-modification entry verifies the authors' coding decision. It does
not itself show that a specimen was independently examined or that the trait,
OTU or specimen set is unchanged. Published anatomical descriptions are separated
from bare scoring assertions. Unresolved documentation does not prove a coding
error. Counts overlap across dimensions and the equally allocated sample does
not estimate the prevalence of evidence categories in the pooled population.
NOT_REVIEWED refers only to the new fixed 100-endpoint audit and does not deny
earlier targeted checks. tetrapod_prior_review_overlap.json compares the earlier
58 records with the new tetrapod 20 after selection: four overlap and the union
contains 74 endpoints, with different review scopes and no pooled evidence grade.

Additive correspondence and OTU follow-up
definition_followup/ contains a subsequent AI-assisted documentary review of the
twenty sampled dinosaur entries, corrected precise-page locators, and archived
evidence for the history-specific OTU admission and exclusion decisions. It does
not alter this folder's original fixed 100-entry audit, its classifications or
counts, the frozen endpoint labels, or any model input or numerical result.
The revised SI reports the definition follow-up in S16.2 and the Inherited
column of Table S30, and the OTU decisions in S16.5. Nineteen dinosaur entries
have an additional documented inheritance route; this is distinct from a
two-version definition comparison or independent anatomical validation. The
original Def classification remains 78/100 overall and 1/20 for dinosaurs.

Run `python -B definition_followup/verify_followup.py` from this folder to check
the original audit hashes, the twenty endpoint identities, native archived
matrix tokens, evidence-route counts and recovered merger-exclusion accounting.
The follow-up README.txt states the evidence limits and explains the supplied
JSON/CSV records and local manifest. Publisher PDFs and derived full texts are
not included. Original audit receipts and ../PACKAGE_VALIDATION.json retain
their historical scope; they do not represent new full-analysis validation.

Archived materials for a separate anatomical-expert assessment
The completed independent human verification described above is distinct from
the uncompleted anatomical-expert workflow documented in this archive.
INDEPENDENT_EXPERT_REVIEW.txt is the archived protocol. expert_source_locator_packet.json
withholds AI judgments and model outcomes. expert_blank_register.json / .csv
contains unfilled decisions and signatures. expert_later_state_reveal.json is a
separate file. Original articles or matrices can reveal the later state, so the
reviewer must record actual label exposure. Source locations are supplied rather
than a claim that all early anatomical definitions and specimen inventories have
been recovered. The archived protocol records that no anatomical expert had
been appointed, no packet had been sent, and no such expert review had been
performed at the time of those records. These historical records are not evidence
against the authors' subsequent independent human verification of source-based
judgements and scientific interpretation. The completed AI-assisted audit should
not be included in a packet intended to withhold the analyst's judgements from
future anatomical-expert reviewers.

Source-triggered issues
The gnathostome duplicate448 register enumerates every original-cohort endpoint
of a publication-declared duplicate character, independently of model results.
The dinosaur conflict register preserves the frozen Laquintasaura character366
label while documenting a disagreement in the later publication. Such findings
can motivate separately labelled post hoc sensitivities; they do not retrospectively
alter the frozen random sample or establish anatomical truth.

Redistribution
Authentic public matrix files should retain their original attribution and the
licence recorded by the acquisition manifest. Full-text article PDFs and derived
full-text extracts were downloaded only to inspect the evidence. Prefer source
URLs, hashes and page/figure locators in a code release where redistribution
permission is not confirmed. The source registry retains the licence information
actually recovered; public access alone is not treated as a new licence.
