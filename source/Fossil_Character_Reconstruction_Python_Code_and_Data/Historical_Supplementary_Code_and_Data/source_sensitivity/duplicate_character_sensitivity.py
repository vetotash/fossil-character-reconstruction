#!/usr/bin/env python3
"""Exclude source-declared duplicate-character targets without refitting."""
from pathlib import Path
import argparse,hashlib,json,math
import pandas as pd

def run(predictions,keys,output):
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    k=pd.read_csv(keys,keep_default_na=False,dtype={'later_state':str})
    assert len(k)==13 and (k.old_character_index_zero_based==447).all()
    excluded={(r.original_taxon_label,int(r.old_character_index_zero_based)+1):r.later_state for r in k.itertuples()}
    data=pd.read_csv(predictions,keep_default_na=False,dtype={'true':str,'pred':str})
    data=data[data.scenario=='later_target'].copy();rows=[];cells=[]
    for model,group in data.groupby('model'):
        assert len(group)==212 and not group.duplicated(['taxon','character']).any()
        keep=[]
        for r in group.itertuples():
            key=(r.taxon,int(r.character));flag=key in excluded
            if flag:assert r.true==excluded[key]
            p=json.loads(r.state_probabilities);ptrue=p.get(r.true,0.)
            b=sum((p.get(s,0.)-int(s==r.true))**2 for s in set(p)|{r.true})
            loss=-math.log(max(ptrue,1e-12))
            assert abs(b-r.brier)<1e-10 and abs(loss-r.log_loss)<1e-10
            keep.append(not flag)
            cells.append(dict(model=model,taxon=r.taxon,character=int(r.character),excluded_as_source_declared_duplicate=flag,true=r.true,pred=r.pred,brier=b,log_loss=loss))
        assert sum(keep)==199
        for cohort,part in [('original212',group),('exclude_duplicate199',group[keep])]:
            rows.append(dict(model=model,cohort=cohort,n_targets=len(part),agreement_count=int((part.pred==part.true).sum()),agreement=float((part.pred==part.true).mean()),brier=float(part.brier.mean()),log_loss=float(part.log_loss.mean())))
    pd.DataFrame(rows).to_csv(output/'state_exclusion_summary.csv',index=False)
    pd.DataFrame(cells).to_csv(output/'cell_scores.csv',index=False)
    receipt={'status':'PASS','source_triggered_rule':'Exclude all13 original character448 endpoints already singleton-scored at source-declared duplicate450, independently of model errors','original_targets':212,'sensitivity_targets':199,'model_fits':0,'later_scores_unchanged':True,'sha256':{str(p.name):hashlib.sha256(p.read_bytes()).hexdigest() for p in [predictions,keys,Path(__file__)]}}
    (output/'analysis_manifest.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(pd.DataFrame(rows).to_string(index=False))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--predictions',type=Path,required=True);p.add_argument('--keys',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.predictions,a.keys,a.output)
