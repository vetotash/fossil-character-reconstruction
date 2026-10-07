"""Verify complete native gzip evidence and remove stale uncompressed duplicates."""
from pathlib import Path
import datetime,gzip,hashlib,json
HERE=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()

def main():
    assert (HERE/'RUN_MANIFEST.json').exists()
    records=[];duplicates=[]
    evidence=sorted(HERE.rglob('native_evidence.json'))
    assert len(evidence)==605,len(evidence)
    for meta in evidence:
        info=json.loads(meta.read_text());folder=meta.parent
        for name,spec in info['evidence'].items():
            compressed=folder/(name+'.gz');payload=compressed.read_bytes();raw=gzip.decompress(payload)
            assert sha(payload)==spec['gzip_sha256'],str(compressed)
            assert sha(raw)==spec['raw_sha256'] and len(raw)==spec['bytes'],str(compressed)
            original=folder/name
            record=dict(path=str(compressed.relative_to(HERE)),bytes=spec['bytes'],gzip_sha256=sha(payload),uncompressed_sha256=sha(raw))
            if original.exists():
                old=original.read_bytes()
                assert raw.startswith(old),(str(original),'not identical or a prefix of verified complete gzip')
                duplicates.append(dict(path=str(original.relative_to(HERE)),bytes=len(old),sha256=sha(old),
                    canonical_complete_bytes=len(raw),status='identical_copy' if old==raw else 'stale_prefix',
                    canonical_gzip=str(compressed.relative_to(HERE)),action='removed after canonical byte and hash verification'))
                original.unlink()
            records.append(record)
    result=dict(completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),native_batches=len(evidence),
        compressed_files=len(records),all_gzip_and_uncompressed_hashes_match=True,uncompressed_duplicates_removed=len(duplicates),
        stale_prefix_count=sum(d['status']=='stale_prefix' for d in duplicates),
        explanation='The scorer removes original files after compression. Later inspection nevertheless found old uncompressed duplicates and prefixes with earlier modification times. No analysis step recreates these files; the cause is undetermined. The complete compressed files matched every recorded compressed and uncompressed hash. Duplicate/prefix provenance is retained here and partial originals were removed to avoid presenting incomplete logs as complete evidence.',
        duplicates=duplicates,canonical_files=records)
    (HERE/'EVIDENCE_VALIDATION.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ['duplicates','canonical_files']},indent=2))

if __name__=='__main__':main()
