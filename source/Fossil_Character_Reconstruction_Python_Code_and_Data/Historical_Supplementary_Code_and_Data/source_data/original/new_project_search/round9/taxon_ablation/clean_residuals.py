"""Preserve evidence for noncanonical raw residues; leave canonical gzip untouched."""
from pathlib import Path
import datetime,gzip,hashlib,json

P=Path(__file__).resolve().parent
def sha(data):return hashlib.sha256(data).hexdigest()
def main():
    rows=[]
    folders=[P/'cases']+[q for q in P.iterdir() if q.is_dir() and q.name.startswith('validation')]
    for folder in folders:
        for path in sorted(folder.rglob('*')):
            if not path.is_file() or path.name not in ['diagnose.nex','native.log','native_scores.tsv','stdout.txt','lower_bounds.csv']:continue
            canonical=path.with_suffix(path.suffix+'.gz');assert canonical.exists(),str(path)
            original=path.read_bytes();full=gzip.decompress(canonical.read_bytes())
            kind='equal_to_canonical' if original==full else 'proper_prefix_of_canonical' if full.startswith(original) else 'different_from_canonical'
            record=dict(file=str(path.relative_to(P)),bytes=len(original),sha256=sha(original),canonical_gzip=str(canonical.relative_to(P)),canonical_uncompressed_bytes=len(full),canonical_uncompressed_sha256=sha(full),relation=kind)
            if kind!='equal_to_canonical':
                preserved=path.with_name(path.name+'.residual.gz');assert not preserved.exists();preserved.write_bytes(gzip.compress(original,compresslevel=6,mtime=0));assert gzip.decompress(preserved.read_bytes())==original
                record['residual_preserved_as']=str(preserved.relative_to(P));record['residual_gzip_sha256']=sha(preserved.read_bytes())
            # Only after exact preservation in canonical or extra gzip is verified.
            path.unlink();rows.append(record)
    report=dict(time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),raw_files_processed=len(rows),canonical_gzip_unchanged=True,score_arrays_and_summaries_unchanged=True,mechanism_of_residual_reappearance_not_established=True,records=rows)
    (P/'RESIDUAL_CLEANUP.json').write_text(json.dumps(report,indent=2));print(dict(files=len(rows),relations={k:sum(r['relation']==k for r in rows) for k in ['equal_to_canonical','proper_prefix_of_canonical','different_from_canonical']}))
if __name__=='__main__':main()
