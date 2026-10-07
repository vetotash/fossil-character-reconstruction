"""Retrieve official PAUP and two private runtime libraries, with pinned hashes.

No system files are changed. Review the author's current download terms at
https://phylosolutions.com/paup-test/ before using this reproducibility recipe.
This script grants no redistribution rights to third-party executables.
"""
from pathlib import Path
import gzip
import hashlib
import json
import subprocess
import urllib.request

OUT=Path(__file__).resolve().parent/'tools'
DOWNLOADS=[
 ('https://phylosolutions.com/paup-test/paup4a169_ubuntu64.gz',
  'a30dc437b9a3d439d397bbc5e6dea010d588de4667e97d247cadc5a33035040c'),
 ('https://archive.ubuntu.com/ubuntu/pool/main/g/gcc-7/libgfortran4_7.5.0-3ubuntu1~18.04_amd64.deb',
  'f89b2473bc251ba0c8eb9479ad31d5f96019e2ea1f8be466beea3a68b11b19fc'),
 ('https://archive.ubuntu.com/ubuntu/pool/universe/p/python2.7/libpython2.7_2.7.18~rc1-2_amd64.deb',
  '77d67e841b4812bdc32274219cc86c316d357c00e8589d54be9987cb060db832'),
]

if __name__=='__main__':
 OUT.mkdir(exist_ok=True)
 records=[]
 for url,sha in DOWNLOADS:
  filename=OUT/url.rsplit('/',1)[-1]
  data=filename.read_bytes() if filename.exists() else urllib.request.urlopen(url,timeout=60).read()
  assert hashlib.sha256(data).hexdigest()==sha,(url,'checksum changed; inspect new official version before use')
  filename.write_bytes(data)
  if filename.suffix=='.gz':
   binary=filename.with_suffix('');binary.write_bytes(gzip.decompress(data));binary.chmod(0o755)
   assert hashlib.sha256(binary.read_bytes()).hexdigest()=='0e0b7a7d98c8d7a5664f1bfe29d0fe873743127d4d0b9ecc20a30bcb848517e3'
  else:
   subprocess.run(['dpkg-deb','-x',str(filename),str(OUT/'runtime')],check=True)
  records.append(dict(url=url,sha256=sha,bytes=len(data)))
 (OUT/'reconstruction_manifest.json').write_text(json.dumps(records,indent=2))
 print('Ready. Set LD_LIBRARY_PATH to tools/runtime/usr/lib/x86_64-linux-gnu before running PAUP.')
