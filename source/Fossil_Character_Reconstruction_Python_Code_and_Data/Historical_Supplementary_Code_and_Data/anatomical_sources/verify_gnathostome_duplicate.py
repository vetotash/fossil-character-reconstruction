#!/usr/bin/env python3
"""Outcome-independent source-declared duplicate-character check for original cohort."""
from pathlib import Path
import csv,hashlib,json,re
from lxml import etree

ROOT=Path(__file__).resolve().parents[2]
A=ROOT/'extension_work/anatomy'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
registry=A/'endpoint_register_initial.json'
allrows=json.loads(registry.read_text())
frame=[r for r in allrows if r['history']=='gnathostomes']
assert len(frame)==212
selected=[r for r in frame if r['old_character_index_zero_based']==447]
source=ROOT/'public_validation/data/processed/gnathostomes.json'
data=json.loads(source.read_text())
old_source=ROOT/'public_validation'/data['old']['source_file']
new_source=ROOT/'public_validation'/data['new']['source_file']
assert sha(old_source)==data['old']['source_sha256']
assert sha(new_source)==data['new']['source_sha256']
xml=ROOT/'public_validation/data/raw/gnathostomes/cui2023_europepmc.xml'
paras=[' '.join(''.join(p.itertext()).split()) for p in etree.parse(str(xml)).xpath('//body//p')]
evidence=[p for p in paras if 'duplicated character' in p and '450' in p and '448' in p]
assert len(evidence)==1
rows=[]
for r in selected:
 taxon=r['original_taxon_label'];i=data['old']['taxa'].index(taxon);j=data['new']['taxa'].index(taxon)
 missing=data['old']['matrix'][i][447];duplicate=data['old']['matrix'][i][449]
 later=data['new']['matrix'][j][r['new_character_index_zero_based']]
 assert missing=='?' and duplicate in list('0123456789') and later==r['later_state']
 assert duplicate==later
 rows.append({**{k:r[k] for k in ('endpoint_id','original_taxon_label','old_character_index_zero_based','new_character_index_zero_based','later_state')},'old_response_token':missing,'old_duplicate_character_index_zero_based':449,'old_duplicate_token':duplicate,'later_primary_source_token':later,'duplicate_is_singleton':True,'duplicate_equals_later_state':True})
out=A/'gnathostome_duplicate448_verified_keys.csv'
with out.open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
meta={'status':'PASS','original_cohort_n':212,'source_declared_duplicate_response_one_based':448,'source_declared_duplicate_predictor_one_based':450,'verified_endpoints':len(rows),'all_old_response_tokens_question_mark':True,'all_old_duplicate_tokens_singleton':True,'all_old_duplicate_tokens_equal_later_label':True,'rule':'Enumerate every original-cohort response448 endpoint because Cui et al. 2023 explicitly identifies original450 as duplicate448; use no fitted predictions, confidence or error measures. Verify source tokens after enumeration.','source_declaration':{'doi':'10.1038/s41467-023-43557-9','url':'https://www.nature.com/articles/s41467-023-43557-9','local_file':str(xml.relative_to(ROOT)),'sha256':sha(xml),'locator':'Results and discussion, phylogenetic-analysis paragraph beginning To investigate the impact; explicit deletion of original Character450 duplicated with Character448'},'interpretation':'These were missing entries in column448 but already had the source-declared duplicate trait scored in the earlier matrix. Removing them is a source-triggered historical-information sensitivity, not an anatomical-accuracy subset. Equal tokens do not prove correctness or constant specimen scope.','source_pdf_nexus_ambiguity':'Both published PDF character448 definitions print absent/present, whereas both machine-readable Nexus state labels distinguish anterior/ventrolateral from posterior orbital positions. Preserve this documentary discrepancy; do not resolve it by model performance.','hashes':{'registry':sha(registry),'processed_source':sha(source),'old_primary_matrix':sha(old_source),'new_primary_matrix':sha(new_source),'verified_keys_csv':sha(out),'script':sha(Path(__file__))},'not_used':['predictions','probabilities','agreement flags','model ranks','D contributions']}
(A/'gnathostome_duplicate448_rule.json').write_text(json.dumps(meta,indent=2)+'\n')
print(json.dumps({'status':'PASS','original_cohort':212,'duplicate_endpoints':len(rows),'all_tokens_match':True}))
