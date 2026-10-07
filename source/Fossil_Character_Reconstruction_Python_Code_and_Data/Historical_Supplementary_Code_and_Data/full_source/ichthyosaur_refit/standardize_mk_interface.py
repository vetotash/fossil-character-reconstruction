"""Add read-only aliases to saved fitted Mk records for joint draw consumer.

No trees or rates are fitted. Aliases expose existing indices and source states.
"""
from pathlib import Path
import hashlib
import json
import numpy as np

OUT = Path(__file__).resolve().parent
p = OUT / 'mk_models.json'
models = json.loads(p.read_text())
before = hashlib.sha256(p.read_bytes()).hexdigest()
assert len(models) == 130
for r in models:
    r['train_indices'] = r['train']
    r['target_indices'] = r['test']
    r['train_states'] = [r['observed'][i] for i in r['train']]
    assert len(r['train_indices']) == len(r['train_states'])
    assert all(r['observed'][i] == '?' for i in r['target_indices'])
p.write_text(json.dumps(models, indent=2, allow_nan=False))
(OUT / 'MK_INTERFACE_CHECK.json').write_text(json.dumps({
    'models': 130, 'refits': 0, 'added_aliases': ['train_indices', 'train_states', 'target_indices'],
    'input_mk_models_sha256': before,
    'output_mk_models_sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
    'aliases_equal_original_indices_and_states': True,
}, indent=2))
manifest = []
for f in sorted(OUT.iterdir()):
    if f.is_file() and f.name != 'OUTPUT_MANIFEST.json':
        manifest.append({'path': str(f.resolve()), 'bytes': f.stat().st_size,
                         'sha256': hashlib.sha256(f.read_bytes()).hexdigest()})
(OUT / 'OUTPUT_MANIFEST.json').write_text(json.dumps(manifest, indent=2))
print('Joint consumer aliases added; no fits or scientific values changed.')
