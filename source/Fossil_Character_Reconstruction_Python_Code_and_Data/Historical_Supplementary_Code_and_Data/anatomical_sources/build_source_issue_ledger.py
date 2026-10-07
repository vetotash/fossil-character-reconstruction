#!/usr/bin/env python3
"""List distinct source issues without editing labels or selecting by outcomes."""
import csv
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent


def main():
    sample = {r['endpoint_id']: r for r in json.loads((BASE / 'completed_documentary_sample.json').read_text())}
    rows = []
    for r in csv.DictReader((BASE / 'gnathostome_duplicate448_verified_keys.csv').open()):
        rows.append({
            'endpoint_id': r['endpoint_id'], 'history': 'gnathostomes',
            'taxon': r['original_taxon_label'], 'old_character_one_based': 448,
            'frozen_later_state': r['later_state'],
            'issue_type': 'PUBLICATION_DECLARED_DUPLICATE_TRAIT',
            'evidence': 'Original character450 is a publication-declared duplicate of448; its earlier singleton equals this later448 state. PDF and NEXUS state definitions also differ.',
            'source_doi': '10.1038/s41467-023-43557-9',
            'source_locator': 'Results and discussion phylogenetic-analysis paragraph; both published character lists at448; gnathostome_duplicate448_rule.json',
            'sensitivity_action': 'Separately labelled, source-triggered exclusion of all13 original-cohort targets; no label correction.',
            'selected_in_fixed_sample': r['endpoint_id'] in sample,
        })
    for r in json.loads((BASE / 'dinosaur_source_conflicts.json').read_text()):
        rows.append({
            'endpoint_id': r['endpoint_id'], 'history': 'dinosaurs', 'taxon': r['taxon'],
            'old_character_one_based': 366, 'frozen_later_state': r['frozen_later_state'],
            'issue_type': 'LATER_MATRIX_VS_MODIFICATION_LIST',
            'evidence': 'Later matrix and embedded PDF matrix assign0; modification list prints a change from? to2. The original definition lists only states0/1.',
            'source_doi': r['doi'], 'source_locator': 'Later supplement PDFp7; embedded matrix pp18–19; original supplement PDFp50.',
            'sensitivity_action': 'Separately labelled, source-triggered exclusion of this endpoint; preserve frozen0.',
            'selected_in_fixed_sample': True,
        })
    for r in sample.values():
        if r['original_primary_token'] == '-':
            rows.append({
                'endpoint_id': r['endpoint_id'], 'history': r['history'], 'taxon': r['original_taxon_label'],
                'old_character_one_based': r['old_character_index_zero_based'] + 1,
                'frozen_later_state': r['later_state'], 'issue_type': 'PREVIOUSLY_DOCUMENTED_MISSINGNESS_NORMALISATION',
                'evidence': 'Primary earlier cell is a dash; the historical analysis had already normalised it to?. This is not a newly inferred coding error.',
                'source_doi': '10.1038/nature21700', 'source_locator': 'Original NEXUS row and column; detailed locators in sampled review.',
                'sensitivity_action': 'Keep distinct from new publication conflicts; refer to the existing strict-original-question-mark analysis.',
                'selected_in_fixed_sample': True,
            })
    for endpoint, issue, locator in [
        ('squ_0aef2f9cfcc10b90', 'CHANGE_NARRATIVE_VS_EARLIER_NATIVE_TOKEN', 'Later supplement PDFp17 narrative andp27 description; old NEXUS character212.'),
        ('squ_0bf8ee3bd1d3a0a0', 'DESCRIPTIVE_TENSION_REQUIRING_ANATOMICAL_REVIEW', 'Old supplement PDFp82; later supplement PDFp27; character166.'),
    ]:
        r = sample[endpoint]
        rows.append({
            'endpoint_id': endpoint, 'history': r['history'], 'taxon': r['original_taxon_label'],
            'old_character_one_based': r['old_character_index_zero_based'] + 1,
            'frozen_later_state': r['later_state'], 'issue_type': issue,
            'evidence': r['source_reason'], 'source_doi': '10.1126/sciadv.aax5833',
            'source_locator': locator,
            'sensitivity_action': 'No automatic exclusion, recoding or assertion of definition change. Retain for expert assessment.',
            'selected_in_fixed_sample': True,
        })
    assert len(rows) == 18 and len({r['endpoint_id'] for r in rows}) == 18
    result = {
        'purpose': 'Separate source-defined duplicate information, publication discrepancies, previously documented normalisation and anatomical interpretive tensions. These are not18 established coding errors.',
        'labels_changed': 0,
        'outcome_based_selection_used': False,
        'sample_selection_unchanged': True,
        'records': rows,
    }
    (BASE / 'source_issue_ledger.json').write_text(json.dumps(result, indent=2) + '\n')
    with (BASE / 'source_issue_ledger.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print('Wrote18 distinct source-issue records; no label changes.')


if __name__ == '__main__':
    main()
