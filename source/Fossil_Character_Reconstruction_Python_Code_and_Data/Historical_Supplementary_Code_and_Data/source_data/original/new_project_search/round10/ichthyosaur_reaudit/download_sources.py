"""Download only public publisher-linked inputs; verify supplied file checksums."""
from pathlib import Path
import json, urllib.request, hashlib, datetime

P = Path(__file__).resolve().parent
(P / 'raw').mkdir(exist_ok=True)
manifest = []
for article in [5688949, 7987313]:
    url = f'https://api.figshare.com/v2/articles/{article}'
    body = urllib.request.urlopen(url, timeout=35).read()
    data = json.loads(body)
    (P/'raw'/f'figshare{article}.json').write_bytes(body)
    print(article, data['version'], data['published_date'], flush=True)
    for f in data['files']:
        if article == 5688949 and f['id'] not in [9960361,9960364,9960391,9960394]:
            continue
        body = urllib.request.urlopen(f['download_url'], timeout=35).read()
        assert hashlib.md5(body).hexdigest() == f['computed_md5']
        (P/'raw'/f['name']).write_bytes(body)
        manifest.append(dict(article=article,id=f['id'],name=f['name'],url=f['download_url'],bytes=len(body),md5=f['computed_md5'],sha256=hashlib.sha256(body).hexdigest(),retrieved=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        print(f['name'], len(body), flush=True)
(P/'download_manifest.json').write_text(json.dumps(manifest,indent=2))
