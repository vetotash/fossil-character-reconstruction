"""Read actual package members for documentation; never execute analysis code."""
from pathlib import Path
import csv,gzip,hashlib,json,re
import numpy as np

GENERATED={'DATA_DICTIONARY.tsv','FILE_CATALOG.tsv','ARRAY_LAYOUTS.tsv','DOCUMENTATION_COVERAGE.json'}

def json_fields(obj):
    result={}
    def add(path,value,container=False):
        kind='object' if isinstance(value,dict) else 'array' if isinstance(value,list) else 'null' if value is None else type(value).__name__
        row=result.setdefault(path,{'types':set(),'examples':[],'shape':set()})
        row['types'].add(kind)
        if container:row['shape'].add(len(value))
        elif len(row['examples'])<2:
            val=str(value)
            if val not in row['examples']:row['examples'].append(val[:800])
    def visit(value,path):
        if isinstance(value,dict):
            add(path,value,True);keys=list(value)
            dynamic=bool(keys) and (all(re.fullmatch(r'[-+]?\d+(?:\.\d+)?',str(k)) for k in keys) or (len(keys)>5 and all('/' in str(k) or str(k).endswith(('.py','.csv','.json','.npz','.nex','.gz')) for k in keys)) or (len(keys)>20 and sum(not re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*',str(k)) for k in keys)>len(keys)*.7))
            for key,child in value.items():
                token='.*' if dynamic else '.'+str(key) if re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*',str(key)) else '['+json.dumps(str(key))+']'
                visit(child,path+token)
        elif isinstance(value,list):
            add(path,value,True)
            for child in value:visit(child,path+'[]')
        else:add(path,value)
    visit(obj,'$')
    return {k:{'types':sorted(v['types']),'examples':v['examples'],'container_sizes':sorted(v['shape'])[:30]} for k,v in result.items()}

def inspect_inputs(root):
    root=Path(root);tables={};jsons={};afiles={};agroups={}
    for path in sorted(root.rglob('*')):
        if not path.is_file():continue
        rel=path.relative_to(root).as_posix()
        if rel in GENERATED:continue
        if any(path.name.endswith(s) for s in ('.csv','.tsv','.csv.gz','.tsv.gz')):
            opener=gzip.open if path.suffix=='.gz' else open
            with opener(path,'rt',encoding='utf-8-sig',newline='') as handle:
                reader=csv.reader(handle,delimiter='\t' if '.tsv' in path.name else ',')
                fields=next(reader,[]);samples=[]
                for i,row in enumerate(reader):
                    if i>=2:break
                    samples.append(row)
            sid='T'+hashlib.sha256(json.dumps(fields).encode()).hexdigest()[:10]
            tables.setdefault(sid,{'fields':fields,'samples':samples,'files':[]})['files'].append(rel)
        elif path.suffix=='.json':
            fields=json_fields(json.loads(path.read_text(encoding='utf-8-sig')))
            sid='J'+hashlib.sha256(json.dumps(sorted(fields)).encode()).hexdigest()[:10]
            jsons.setdefault(sid,{'fields':fields,'files':[]})['files'].append(rel)
        elif path.suffix in ('.npz','.npy'):
            fields={}
            if path.suffix=='.npz':
                with np.load(path,allow_pickle=False) as arrays:
                    for key in arrays.files:
                        arr=arrays[key]
                        fields[key]={'dtype':str(arr.dtype),'shape':list(arr.shape),'sample':[str(v)[:120] for v in arr.flat[:3]]}
            else:
                arr=np.load(path,allow_pickle=False,mmap_mode='r')
                fields['array']={'dtype':str(arr.dtype),'shape':list(arr.shape),'sample':[str(v)[:120] for v in arr.flat[:3]]}
            sid='A'+hashlib.sha256(json.dumps(sorted((k,v['dtype'][0],len(v['shape'])) for k,v in fields.items())).encode()).hexdigest()[:10]
            afiles[rel]={'schema_id':sid,'fields':fields}
            agroups.setdefault(sid,{'fields':fields,'files':[]})['files'].append(rel)
    return tables,jsons,{'files':afiles,'schemas':agroups}
