from pathlib import Path
import urllib.request,hashlib,json
from datetime import datetime,timezone
R=Path(__file__).resolve().parent
urls={
'zenodo3584386.json':'https://zenodo.org/api/records/3584386',
'maxwell2019_article.pdf':'https://stri-apps.si.edu/docs/publications/pdfs/A_new_specimen_of_Platypterygius_sachicarum_Reptilia_Ichthyosauria.pdf'
}
rows=[]
for name,url in urls.items():
 try:
  r=urllib.request.urlopen(url,timeout=45);content=r.read()
  p=R/'source_cache'/name;p.write_bytes(content)
  rows.append(dict(name=name,url=url,bytes=len(content),sha256=hashlib.sha256(content).hexdigest(),md5=hashlib.md5(content).hexdigest(),status=r.status,downloaded_utc=datetime.now(timezone.utc).isoformat()))
 except Exception as e:rows.append(dict(name=name,url=url,error=str(e)))
 print(rows[-1],flush=True)
(R/'DOWNLOAD_MANIFEST.json').write_text(json.dumps(rows,indent=2))
