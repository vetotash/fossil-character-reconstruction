"""Project original target identifiers; never use model outcomes for selection."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
ARCHIVE = ROOT/'extension_inputs/empirical14/round14'
PUBLIC = ROOT/'public_validation/data/processed'
SOURCES = {
    'tetrapods': ARCHIVE/'old_refits/inputs/tetrapods/frozen_targets.csv',
    'dinosaurs': ARCHIVE/'old_refits/inputs/dinosaurs/frozen_targets.csv',
    'ichthyosaurs': ARCHIVE/'ichthyosaur_refit/frozen_targets.csv',
    'gnathostomes': ROOT/'extension_inputs/original10/new_project_search/round10/gnathostome_shared_fit/frozen_targets.csv',
    'squamates': ROOT/'extension_inputs/original10/new_project_search/round10/najash_shared_fit/frozen_targets.csv',
}
EXPECTED = {'tetrapods':350,'dinosaurs':1333,'ichthyosaurs':196,'gnathostomes':212,'squamates':163}

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    rows=[];frames=[]
    for history,p in SOURCES.items():
        public=json.loads((PUBLIC/(history+'.json')).read_text())
        correspondence={x['old_index']:x['new_index'] for x in public['character_map']}
        if history=='gnathostomes':
            mapping=ROOT/'extension_inputs/original10/new_project_search/round10/other_candidates/fish_frozen/accepted_character_mapping.csv'
            correspondence={int(r['old_character'])-1:int(r['new_character'])-1 for r in csv.DictReader(mapping.open()) if r['admitted']=='True'}
        unique={}
        with p.open() as handle:
            for original in csv.DictReader(handle):
                # Only this explicit allow-list enters registration or selection.
                r={k:original[k] for k in ['scenario','taxon','character','new_char','true'] if k in original}
                if r['scenario'] not in {'later_publication','later_target'}:continue
                oldidx=int(r['character'])-1
                if r.get('new_char'):
                    newidx=int(r['new_char'])-1
                elif oldidx in correspondence:
                    newidx=correspondence[oldidx]
                else:
                    # Some original frozen endpoints were excluded by the separate
                    # conservative public-data map. Retain them in this audit:
                    # identical source labels locate the column but do not certify
                    # anatomical correspondence, which is reviewed separately.
                    label=public['old']['characters'][oldidx]
                    assert public['new']['characters'].count(label)==1,(history,oldidx)
                    newidx=public['new']['characters'].index(label)
                key=(r['taxon'],oldidx,newidx)
                if key in unique:
                    assert unique[key]['later_state']==r['true']
                    continue
                canonical='anatomical-source-audit-v1\n'+history+'\n'+r['taxon']+'\n'+str(oldidx)+'\n'+str(newidx)
                digest=hashlib.sha256(canonical.encode('utf-8')).hexdigest()
                unique[key]={'endpoint_id':history[:3]+'_'+digest[:16], 'history':history,
                    'original_taxon_label':r['taxon'],'old_character_index_zero_based':oldidx,
                    'new_character_index_zero_based':newidx,'later_state':r['true'],
                    'selection_digest':digest,'canonical_selection_string':canonical,
                    'original_target_file':str(p.relative_to(ROOT)),'original_target_file_sha256':sha(p),
                    'documentary_review_status':'NOT_REVIEWED','token_history':'NOT_REVIEWED',
                    'character_correspondence':'NOT_REVIEWED','taxon_scope':'NOT_REVIEWED',
                    'specimen_continuity':'NOT_REVIEWED','later_state_support':'NOT_REVIEWED',
                    'support_mode':'NOT_REVIEWED','old_specimens':'','later_specimens':'',
                    'source_locators':[],'source_reason':'','expert_review_status':'NOT_PERFORMED'}
        assert len(unique)==EXPECTED[history],(history,len(unique))
        ordered=sorted(unique.values(),key=lambda x:(x['selection_digest'],x['canonical_selection_string']))
        selected={x['endpoint_id'] for x in ordered[:20]}
        for row in ordered:row['selected_for_documentary_review']=row['endpoint_id'] in selected
        rows.extend(ordered)
        frames.append({'history':history,'n_original_endpoints':len(ordered),'n_selected':len(selected),
                       'input_file':str(p.relative_to(ROOT)),'sha256':sha(p)})
    (OUT/'endpoint_register_initial.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False)+'\n')
    sample=[x for x in rows if x['selected_for_documentary_review']]
    (OUT/'selected_endpoints.json').write_text(json.dumps(sample,indent=2,ensure_ascii=False)+'\n')
    fields=['endpoint_id','history','original_taxon_label','old_character_index_zero_based','new_character_index_zero_based',
            'later_state','selection_digest']
    with (OUT/'selected_endpoint_keys.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:r[k] for k in fields} for r in sample)
    freeze={'plan_sha256':sha(OUT/'REVIEW_PLAN.json'),'frames':frames,
            'missing_original_frames':[h for h in EXPECTED if h not in SOURCES],
            'sample_file_sha256':sha(OUT/'selected_endpoints.json'),
            'selection_uses_predictions':False,'note':'Historical endpoint keys only; source review follows this frozen selection. No replacements for unavailable evidence.'}
    (OUT/'SAMPLE_FREEZE.json').write_text(json.dumps(freeze,indent=2)+'\n')
    print(json.dumps(freeze,indent=2))

if __name__=='__main__':main()
