from pathlib import Path
import json,hashlib,datetime
import numpy as np
import pandas as pd

P=Path(__file__).resolve().parent
assert not (P/'CURATION_FREEZE.json').exists(),'Do not overwrite an existing freeze'
d=np.load(P/'raw_parsed.npz')
mapping=pd.read_csv(P/'character_audit.csv')
chars=mapping.loc[mapping.accepted,'character'].to_numpy(dtype=int)
targets=pd.read_csv(P/'eligible_cells_provisional.csv',dtype={'source_symbol':str,'later_published_state':str})
targets=targets.sort_values(['character','taxon'],kind='stable').reset_index(drop=True)
targets['true']=targets.later_published_state
targets['scenario']='later_target'
targets.to_csv(P/'frozen_future_targets.csv',index=False)
np.savez_compressed(P/'source_model_inputs.npz',A=d['old_tokens'][:,chars-1],old_names=d['old_names'],old_char_numbers=chars)
assert len(targets)==196 and len(chars)==242
assert set(targets.source_symbol)=={'?'}
targets.groupby('taxon').size().rename('eligible_targets').reset_index().sort_values(['eligible_targets','taxon'],ascending=[False,True]).to_csv(P/'target_taxon_counts.csv',index=False)
files=['CURATION_PROTOCOL.txt','audit_inputs.py','freeze_curation.py','download_manifest.json','character_audit.csv','all_candidate_cells.csv','row_identity_audit.csv','raw_parsed.npz','source_model_inputs.npz','frozen_future_targets.csv','target_taxon_counts.csv','previous195_not_currently_eligible.csv','INPUT_AUDIT.json']
files += ['raw/'+x for x in ['tjsp_a_1394922_sm0622.nex','tjsp_a_1394922_sm0623.pdf','Sachicarum_APPENDIX3.nex','Supplementary_Data_R1.docx','moon2017_definitions.txt','maxwell2019_supplement.txt']]
out=dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),new_fit_or_propagation_seen=False,source_taxa=115,source_feature_columns=242,future_targets=196,future_taxa=30,future_characters=130,old195_retained=192,old195_excluded=3,additional_targets=4,controls_not_yet_generated=True,tree_score_scope_not_yet_frozen=True,hashes={x:hashlib.sha256((P/x).read_bytes()).hexdigest() for x in files})
(P/'CURATION_FREEZE.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
