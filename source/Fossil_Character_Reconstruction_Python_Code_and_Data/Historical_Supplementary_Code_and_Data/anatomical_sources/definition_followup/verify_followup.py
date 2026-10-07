"""Source-only consistency checks; no fitting and no anatomical truth claims."""
import csv
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read_matrix(path):
    text = path.read_text()
    block = re.search(r'\bMATRIX\s*(.*?)\s*;', text, re.I|re.S).group(1)
    records = {}
    for line in block.splitlines():
        if not line.strip():
            continue
        taxon, seq = line.split(maxsplit=1)
        cells = re.findall(r'\([^)]*\)|\{[^}]*\}|[0-9A-Z?\-]', seq)
        assert len(cells) == 457
        records[taxon] = cells
    return records

def main():
    frozen=json.loads((HERE/'original_audit_hashes.json').read_text())
    for record in frozen['files']:
        assert digest(ROOT/record['path']) == record['sha256'], record['path']
    source=json.loads((ROOT/'anatomical_sources/dinosaur_documentary_reviews.json').read_text())
    follow=json.loads((HERE/'dinosaur_correspondence_followup.json').read_text())
    rows=follow['records']
    assert len(rows)==20 and {r['endpoint_id'] for r in rows}=={r['endpoint_id'] for r in source}
    src={r['endpoint_id']:r for r in source}
    raw=ROOT/'source_data/original/new_project_search/round8/replication/raw'
    old=read_matrix(raw/'baron_original.nex');new=read_matrix(raw/'langer_original.nex')
    for r in rows:
        s=src[r['endpoint_id']];i=r['old_character_one_based']-1
        assert i==s['old_character_index_zero_based']==s['new_character_index_zero_based']
        assert r['original_audit_character_correspondence']==s['character_correspondence']
        assert old[s['source_old_taxon']][i]=='?'
        assert new[s['source_new_taxon']][i]==s['later_state']==r['archived_matrix_check']['later_token']
        assert not r['changes_original_record']
    assert sum(r['later_definition_reprinted'] for r in rows)==1
    assert sum('DECLARED_MATRIX_INHERITANCE' in r['followup_documentary_correspondence'] for r in rows)==19
    assert sum(bool(r['remaining_issue']) for r in rows)==1
    flat=list(csv.DictReader((HERE/'dinosaur_correspondence_followup.csv').open()))
    assert len(flat)==20 and {r['endpoint_id'] for r in flat}==set(src)
    candidates=list(csv.DictReader((ROOT/'source_data/original/new_project_search/round10/ichthyosaur_reaudit/all_candidate_cells.csv').open()))
    merge=[r for r in candidates if r['taxon']=='Suevoleviathan_integer']
    sole=[r for r in merge if r['exclusion_reason']=='2019 explicit merger of old S. disinteger scores']
    assert len(candidates)==469 and sum(r['eligible'].lower()=='true' for r in candidates)==196
    assert len(merge)==34 and len(sole)==25 and all(r['eligible'].lower()=='false' for r in merge)
    assert all(r['source_symbol']=='?' and int(r['old_singletons'])>=15 for r in sole)
    result={'status':'passed','scope':'unchanged original audit hashes; endpoint identity; native archived matrix tokens; evidence-route counts; merger exclusion accounting',
        'original_audit_files_checked':len(frozen['files']),
        'dinosaur_endpoints_checked':20,'original_supported_dinosaur_definitions':1,
        'additional_inheritance_routes':19,'state_conflicts_retained':1,
        'ichthyosaur_candidate_rows':469,'ichthyosaur_eligible_targets':196,
        'Suevoleviathan_candidates_excluded':34,'otherwise_eligible_merger_exclusions':25,
        'anatomical_validation':False,'new_model_fits':0,'original_audit_records_changed':False,
        'limitation':'This checks stored documentary records and numerical tokens. It does not repeat source reading, validate anatomy, or extend the original full analysis validation.'}
    (HERE/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))

if __name__=='__main__':
    main()
