from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from urllib.request import Request, urlopen
from urllib.parse import quote
import json,hashlib
P=Path(__file__).resolve().parent
DOIS=['10.1017/S1477201906002008','10.7717/peerj.5565','10.1038/nature21700','10.1038/nature24011','10.1038/s41598-023-35784-3','10.1111/2041-210X.14339','10.1007/s11692-021-09534-0','10.1671/0272-4634(2003)023[0275:MERDAA]2.0.CO;2','10.1111/cla.12581','10.1038/s41467-025-61036-1','10.1093/sysbio/syae055','10.1111/2041-210X.70258','10.1093/sysbio/syag063','10.1038/nature22966']
def fetch(doi):
 url='https://api.crossref.org/works/'+quote(doi,safe='')
 try:
  req=Request(url,headers={'User-Agent':'FossilCharacterResearch/1.0','Accept':'application/json'})
  with urlopen(req,timeout=35) as response:raw=response.read()
  m=json.loads(raw)['message']
  fields=['DOI','title','author','container-title','volume','issue','page','article-number','published','published-print','published-online','URL']
  out={k:m[k] for k in fields if k in m};out['metadata_source']=url;out['response_sha256']=hashlib.sha256(raw).hexdigest()
  return {'requested_doi':doi,'status':'ok','metadata':out}
 except Exception as e:return {'requested_doi':doi,'status':'failed','error':str(e)}
if __name__=='__main__':
 with ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(fetch,DOIS))
 (P/'crossref_metadata.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
 for r in rows:
  m=r.get('metadata',{});print(r['status'],r['requested_doi'],m.get('title'),m.get('published'),m.get('page'))
