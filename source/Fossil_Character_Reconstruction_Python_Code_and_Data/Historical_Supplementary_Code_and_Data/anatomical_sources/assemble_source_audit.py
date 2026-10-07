#!/usr/bin/env python3
"""Assemble frozen documentary reviews and blank expert-review materials.

No model predictions, scores or outcomes are read. This is an assembly and
integrity-checking program, not an automated anatomical judgment program.
"""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'extension_work/anatomy'
HISTORIES = ['tetrapods', 'dinosaurs', 'ichthyosaurs', 'gnathostomes', 'squamates']
FILES = ['tetrapod', 'dinosaur', 'ichthyosaur', 'gnathostome', 'squamates']
GATES = ['token_history', 'character_correspondence', 'taxon_scope',
         'specimen_continuity', 'later_state_support']
IMMUTABLE = ['endpoint_id', 'history', 'original_taxon_label',
             'old_character_index_zero_based', 'new_character_index_zero_based',
             'later_state', 'selection_digest', 'canonical_selection_string']
SUPPORTED = 'SUPPORTED_BY_EXPLICIT_SOURCE'
CONFLICT = 'DOCUMENTED_CHANGE_OR_CONFLICT'


def read(name):
    return json.loads((BASE / name).read_text())


def write(name, obj):
    (BASE / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_write(path, fields, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def original_token(r):
    return r.get('original_token', r.get('original_missing_token', r.get('old_primary_token')))


def later_token(r):
    return r.get('source_later_token', r.get('checked_later_token', r.get('later_primary_token')))


def licence_index():
    """Reuse explicit acquisition metadata without assigning new licences."""
    result = {}
    manifests = [
        'public_validation/data/raw/tetrapods/provenance.json',
        'public_validation/data/raw/dinosaurs/acquisition_manifest.json',
        'public_validation/data/raw/ichthyosaurs/manifest.json',
        'public_validation/data/raw/gnathostomes/acquisition_manifest.json',
        'public_validation/data/raw/squamates/acquisition_manifest.json',
        'extension_work/anatomy/langer2017_supplement.provenance.json',
        'extension_work/anatomy/ichthyosaur_documentary_manifest.json',
        'extension_work/anatomy/gnathostome_review_manifest.json',
    ]
    def visit(obj, origin, inherited=None):
        if isinstance(obj, list):
            for v in obj:
                visit(v, origin, inherited)
        elif isinstance(obj, dict):
            licence = obj.get('license', obj.get('licence', inherited))
            if obj.get('sha256') and licence:
                result[obj['sha256']] = {'licence_as_recorded': licence, 'licence_metadata_source': origin}
            for v in obj.values():
                if isinstance(v, (dict, list)):
                    visit(v, origin, licence)
    for filename in manifests:
        if (ROOT / filename).exists():
            visit(json.loads((ROOT / filename).read_text()), filename)
    return result


def main():
    selected = read('selected_endpoints.json')
    selected_by_id = {r['endpoint_id']: r for r in selected}
    assert len(selected) == len(selected_by_id) == 100
    reviews = []
    input_hashes = {}
    allowed = set(read('REVIEW_PLAN.json')['gate_values'])
    for history, stem in zip(HISTORIES, FILES):
        name = stem + '_documentary_reviews.json'
        if not (BASE / name).exists() and history == 'squamates':
            name = 'squamate_documentary_reviews.json'
        data = read(name)
        assert len(data) == 20, (name, len(data))
        input_hashes[name] = sha(BASE / name)
        for r in data:
            assert r['history'] == history
            assert r['endpoint_id'] in selected_by_id
            expected = selected_by_id[r['endpoint_id']]
            for field in IMMUTABLE:
                assert r[field] == expected[field], (r['endpoint_id'], field)
            for field in GATES:
                assert r[field] in allowed, (r['endpoint_id'], field, r[field])
            assert r['expert_review_status'] == 'NOT_PERFORMED'
            r['source_review_classification_as_supplied'] = r['documentary_review_status']
            r['documentary_review_status'] = 'REVIEWED'
            r['original_primary_token'] = original_token(r)
            r['later_primary_token'] = later_token(r)
            assert r['original_primary_token'] in ['?', '-'], r['endpoint_id']
            assert r['later_primary_token'] == r['later_state'], r['endpoint_id']
            r['exact_anatomical_state_reobserved_by_this_audit'] = False
            r['complete_documentary_chain_supported'] = all(r[g] == SUPPORTED for g in GATES)
            # Exact state assertions are distinguished from descriptions of anatomy.
            r['explicit_published_specimen_description_support'] = (
                r['later_state_support'] == SUPPORTED and r['support_mode'] in
                ['EXPLICIT_SPECIMEN_DESCRIPTION', 'EXPLICIT_FIGURE_INTERPRETATION'])
            reviews.append(r)
    assert {r['endpoint_id'] for r in reviews} == set(selected_by_id)
    assert len(reviews) == 100
    by_id = {r['endpoint_id']: r for r in reviews}
    reviews = [by_id[r['endpoint_id']] for r in selected]
    register = read('endpoint_register_initial.json')
    assert len(register) == 2254
    for i, r in enumerate(register):
        if r['endpoint_id'] in by_id:
            register[i] = by_id[r['endpoint_id']]
    assert Counter(r['documentary_review_status'] for r in register) == {
        'REVIEWED': 100, 'NOT_REVIEWED': 2154}
    write('completed_documentary_sample.json', reviews)
    write('endpoint_register_completed.json', register)
    fields = ['endpoint_id', 'history', 'original_taxon_label',
              'old_character_index_zero_based', 'new_character_index_zero_based',
              'original_primary_token', 'later_primary_token'] + GATES + [
              'support_mode', 'explicit_published_specimen_description_support',
              'complete_documentary_chain_supported', 'old_specimens',
              'later_specimens', 'source_reason', 'expert_review_status']
    csv_write(BASE / 'completed_documentary_sample.csv', fields, reviews)

    counts = {}
    for history in HISTORIES + ['all']:
        rows = reviews if history == 'all' else [r for r in reviews if r['history'] == history]
        counts[history] = {
            'sample_n': len(rows),
            'original_frame_n': sum(r['history'] == history for r in register) if history != 'all' else len(register),
            **{g: dict(Counter(r[g] for r in rows)) for g in GATES},
            'support_mode': dict(Counter(r['support_mode'] for r in rows)),
            'strict_original_question_mark_n': sum(r['original_primary_token'] == '?' for r in rows),
            'original_dash_n': sum(r['original_primary_token'] == '-' for r in rows),
            'explicit_published_specimen_description_support_n': sum(r['explicit_published_specimen_description_support'] for r in rows),
            'specimen_continuity_and_later_support_n': sum(r['specimen_continuity'] == SUPPORTED and r['later_state_support'] == SUPPORTED for r in rows),
            'specimen_continuity_and_description_support_n': sum(r['specimen_continuity'] == SUPPORTED and r['explicit_published_specimen_description_support'] for r in rows),
            'complete_documentary_chain_supported_n': sum(r['complete_documentary_chain_supported'] for r in rows),
            'expert_reviews_completed_n': 0,
        }
    write('documentary_audit_counts.json', {
        'review_type': 'AI-assisted retrospective documentary source audit; no expert review',
        'nonexclusive_dimensions': True,
        'not_reviewed_scope': 'NOT_REVIEWED means outside this fixed 100-endpoint audit; it does not negate earlier targeted source checks.',
        'sampling_caution': 'Equal allocation across histories; pooled counts are not a population prevalence estimate.',
        'counts': counts,
    })

    sources = {}
    licences = licence_index()
    for r in reviews:
        for loc in r['source_locators']:
            filename = loc.get('local_file')
            if not filename:
                continue
            path = ROOT / filename
            assert path.is_file(), filename
            digest = sha(path)
            if loc.get('sha256'):
                assert loc['sha256'] == digest, filename
            if filename not in sources:
                sources[filename] = {
                    'source_id': 'S%03d' % (len(sources) + 1),
                    'local_file': filename, 'sha256': digest,
                    'bytes': path.stat().st_size,
                    'urls': [], 'dois': [], 'review_endpoint_ids': [],
                    'bundle_recommendation': 'Include authentic public matrix input with provenance and applicable licence.' if path.suffix.lower() in ['.nex', '.nexus', '.nxs', '.tnt', '.txt'] else 'Provide URL, hash and source locators; do not bundle full text without confirmed redistribution permission.',
                    **licences.get(digest, {'licence_as_recorded': 'No explicit file-specific redistribution licence recovered in the available acquisition metadata.'}),
                }
            s = sources[filename]
            for field, value in [('urls', loc.get('url')), ('dois', loc.get('doi')), ('review_endpoint_ids', r['endpoint_id'])]:
                if value and value not in s[field]:
                    s[field].append(value)
    write('source_evidence_registry.json', list(sources.values()))
    portable_sources = []
    for filename, source in sources.items():
        locators = []
        for r in reviews:
            for loc in r['source_locators']:
                if loc.get('local_file') == filename:
                    concise = {k: v for k, v in loc.items() if k not in ['local_file', 'sha256', 'url', 'doi', 'checked_scope'] and v is not None}
                    concise['endpoint_id'] = r['endpoint_id']
                    if concise not in locators:
                        locators.append(concise)
        portable_sources.append({
            'source_id': source['source_id'], 'dois': source['dois'], 'urls': source['urls'],
            'sha256': source['sha256'], 'bytes': source['bytes'],
            'optional_original_local_file': filename,
            'licence_as_recorded': source['licence_as_recorded'],
            'locators': locators,
            'offline_fulltext_verification_claimed_for_distributed_package': False,
        })
    write('source_index.json', {
        'scope': 'Portable evidence index. Source full texts need not be included in the distributed package; hashes identify the documents actually inspected.',
        'sources': portable_sources,
    })

    packet = []
    reveal = []
    blank = []
    for r in reviews:
        # Deliberately exclude AI judgments, source_reason, model information and
        # checked_scope prose, which may explicitly disclose the later label.
        source_refs = []
        for loc in r['source_locators']:
            ref = {k: v for k, v in loc.items() if k in [
                'doi', 'url', 'local_file', 'sha256', 'pdf_page', 'printed_page',
                'figure', 'character', 'character_one_based', 'docx_paragraph_zero_based',
                'line', 'locator', 'matrix_row', 'taxon_row'] and v is not None}
            if loc.get('local_file') in sources:
                ref['source_id'] = sources[loc['local_file']]['source_id']
            source_refs.append(ref)
        packet.append({
            'endpoint_id': r['endpoint_id'], 'history': r['history'],
            'taxon_label': r['original_taxon_label'],
            'old_character_one_based': r['old_character_index_zero_based'] + 1,
            'later_character_one_based': r['new_character_index_zero_based'] + 1,
            'old_primary_token': r['original_primary_token'],
            'source_locators': source_refs,
            'source_label_exposure_warning': 'Original source documents or matrices may disclose the later label. Record actual exposure; label blinding is not guaranteed.',
            'review_status': 'NOT_PERFORMED',
        })
        reveal.append({'endpoint_id': r['endpoint_id'], 'later_published_state': r['later_state']})
        blank.append({k: '' for k in [
            'endpoint_id', 'reviewer_name', 'reviewer_qualification', 'conflict_of_interest',
            'review_date', 'prior_model_result_exposure', 'later_label_exposure_before_provisional_state',
            'old_missingness_class', 'token_history', 'character_correspondence',
            'taxon_scope', 'specimen_continuity', 'provisional_anatomical_state',
            'provisional_confidence', 'later_state_support', 'evidence_type',
            'old_specimens', 'later_specimens', 'source_doi_url', 'page_figure_character',
            'reason', 'signature', 'review_status']})
        blank[-1].update(endpoint_id=r['endpoint_id'], review_status='NOT_PERFORMED')
    write('expert_source_locator_packet.json', packet)
    write('expert_later_state_reveal.json', reveal)
    write('expert_blank_register.json', blank)
    csv_write(BASE / 'expert_blank_register.csv', list(blank[0]), blank)
    write('expert_packet_manifest.json', {
        'status': 'MATERIALS_PREPARED_REVIEW_NOT_PERFORMED',
        'endpoint_n': 100,
        'reviewers_appointed': 0,
        'packets_sent': 0,
        'instructions': 'INDEPENDENT_EXPERT_REVIEW.txt',
        'source_locator_packet': 'expert_source_locator_packet.json',
        'separate_label_reveal': 'expert_later_state_reveal.json',
        'blank_register': ['expert_blank_register.json', 'expert_blank_register.csv'],
        'limits': [
            'A locator packet, not a claim that every original anatomical definition or specimen inventory has been recovered.',
            'Actual source files must be supplied or retrieved before review; original source material may reveal published states.',
            'The packet withholds AI documentary judgments and all model predictions; the custodian must record each reviewer\'s actual exposure.',
            'No expert decisions, signatures or anatomical reobservations have been supplied.',
        ],
    })
    outputs = ['completed_documentary_sample.json', 'endpoint_register_completed.json',
               'completed_documentary_sample.csv', 'documentary_audit_counts.json',
               'source_evidence_registry.json', 'source_index.json', 'expert_source_locator_packet.json',
               'expert_later_state_reveal.json', 'expert_blank_register.json',
               'expert_blank_register.csv', 'expert_packet_manifest.json']
    write('assembly_manifest.json', {
        'status': 'PASS', 'sample_n': 100, 'population_n': 2254,
        'selection_unchanged': True, 'later_labels_unchanged': True,
        'source_hashes_verified': len(sources),
        'inputs': input_hashes,
        'frozen_plan_sha256': sha(BASE / 'REVIEW_PLAN.json'),
        'frozen_sample_sha256': sha(BASE / 'selected_endpoints.json'),
        'outputs': {f: sha(BASE / f) for f in outputs},
        'assembly_script_sha256': sha(Path(__file__)),
    })
    print(json.dumps({'status': 'PASS', 'sample_n': 100, 'population_n': 2254,
                      'source_files': len(sources), 'counts': counts}, indent=2))


if __name__ == '__main__':
    main()
