"""Export PAUP native per-character lengths for existing trees, with no tree search.

Example, from any directory:
  python native_character_scores.py --matrix ../inputs/source.nex \
      --trees published_O1_O2.tre --out original_O1_O2

The same command accepts native PAUP best.tre files containing TRANSLATE blocks.
PAUP reads those trees itself. Python never implements parsimony or interprets
polymorphic/uncertain cells. The native mstaxa=variable option retains the source
distinction between () polymorphism and {} uncertainty.
"""
from pathlib import Path
import argparse, csv, hashlib, json, os, re, subprocess
from collections import Counter
from decimal import Decimal
from fractions import Fraction

HERE=Path(__file__).resolve().parent
WORK=HERE.parent
PROJECT=WORK.parents[1]

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def quote(path):return "'"+str(path).replace("'","''")+"'"

def identity_audit(matrix_text,tree_text):
    """Check only leaf identities in these NEXUS files, never parsimony states."""
    label_block=re.search(r'\bTAXLABELS\s+(.*?);',matrix_text,re.I|re.S)
    tokens=re.findall(r"'(?:[^']|'')*'|\S+",label_block[1])
    expected={x.strip("'").replace("''","'") for x in tokens}
    ntax=int(re.search(r'\bNTAX\s*=\s*(\d+)',matrix_text,re.I)[1])
    assert len(expected)==ntax==102
    clean=re.sub(r'\[[^\]]*\]','',tree_text,flags=re.S)
    translation={}
    match=re.search(r'\bTRANSLATE\s+(.*?);',clean,re.I|re.S)
    if match:
        for part in match[1].split(','):
            key,value=part.strip().split(None,1)
            translation[key]=value.strip().strip("'").replace("''","'")
    matches=list(re.finditer(r'^\s*(?:TREE|UTREE)\s+(?:\*\s*)?(\S+)\s*=\s*(.*?);',clean,re.I|re.M|re.S))
    for m in matches:
        leaves=re.findall(r"(?<=[(,])\s*('(?:[^']|'')*'|[^\s():,;]+)",m[2])
        leaves=[x.strip("'").replace("''","'") for x in leaves]
        names=[translation.get(x,x) for x in leaves]
        if len(names)!=ntax or len(set(names))!=ntax or set(names)!=expected:
            raise ValueError(f'Tree {m[1]} has invalid source-leaf identity: n={len(names)}, unexpected={set(names)-expected}, missing={expected-set(names)}')
    if not matches:raise ValueError('Expected existing named NEXUS trees')
    return len(matches),{'all_trees_exact_source_taxa':True,'taxa_per_tree':ntax,'all_leaf_names_unique':True,'translation_keys_resolved':bool(translation)}

def read_scorefile(path,nchar):
    """Parse PAUP's scoreFile table, retaining exact native decimal arithmetic."""
    rows=list(csv.reader(path.open(),delimiter='\t'))
    if rows[0]!=['Tree','Length','Character','Length']:
        raise ValueError(f'Unexpected PAUP header: {rows[0]}')
    totals={};scores={};current=None
    for row in rows[1:]:
        if not any(row):continue
        if row[0]:
            current=int(row[0]);totals[current]=Decimal(row[1]);scores[current]={}
        else:
            if current is None or len(row)<4:raise ValueError(row)
            ch=int(row[2]);score=Decimal(row[3])
            if ch in scores[current]:raise ValueError(f'Duplicate character {ch}')
            scores[current][ch]=score
    assert totals, 'No native tree scores were exported'
    for tree,values in scores.items():
        assert set(values)==set(range(1,nchar+1)),(tree,len(values),nchar)
        assert sum(values.values())==totals[tree],(tree,sum(values.values()),totals[tree])
    return totals,scores

def write_character_table(scores,labels,target_counts,path):
    """Write atomically and reread every exported cell against native scores."""
    temporary=path.with_suffix(path.suffix+'.part')
    fields=['tree_index','character','source_character_label','native_steps','frozen_target_cells_in_character']
    with temporary.open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for tree,values in scores.items():
            for ch,value in values.items():
                writer.writerow(dict(tree_index=tree,character=ch,source_character_label=labels.get(ch,''),native_steps=str(value),frozen_target_cells_in_character=target_counts[ch]))
        handle.flush();os.fsync(handle.fileno())
    observed=Counter()
    with temporary.open() as handle:
        for row in csv.DictReader(handle):
            tree=int(row['tree_index']);ch=int(row['character'])
            assert Decimal(row['native_steps'])==scores[tree][ch]
            observed[tree]+=1
    expected={tree:len(values) for tree,values in scores.items()}
    assert dict(observed)==expected,(dict(observed),expected)
    temporary.replace(path)
    return {'rows_verified':sum(observed.values()),'trees_verified':len(observed),'native_cells_reread_equal':True,'sha256':sha(path)}

def native_export(matrix,trees,out,binary,runtime):
    matrix=matrix.resolve();trees=trees.resolve();out=out.resolve()
    out.mkdir(parents=True,exist_ok=True)
    source=matrix.read_text()
    # Only the matrix/character assumptions are executed. Any pre-existing PAUP
    # command or TREES block is removed from this NEW diagnostic copy.
    matrix_only=re.sub(r'BEGIN\s+(?:PAUP|TREES)\s*;.*?END\s*;','',source,flags=re.I|re.S)
    nchar=int(re.search(r'\bNCHAR\s*=\s*(\d+)',matrix_only,re.I)[1])
    n_input_trees,leaf_audit=identity_audit(matrix_only,trees.read_text())
    if n_input_trees>5000:raise ValueError(f'{n_input_trees} trees: report this size to the parent before running; no subset will be selected')
    # Remove only this tool's own previous outputs, so a PAUP failure cannot be
    # followed by accidental parsing of a successful stale score table.
    for name in ['native_scores.tsv','native_scores.log','stdout.txt','character_lengths.csv','character_ensemble.csv','validation.json','tree2_minus_tree1_character_contrast.csv']:
        (out/name).unlink(missing_ok=True)
    commands=f'''\nBEGIN PAUP;
set autoclose=yes increase=no maxtrees={max(100,n_input_trees)};
outgroup Eusthenopteron / only;
pset mstaxa=variable opt=deltran;
gettrees file={quote(trees)};
pscores all / single=all tl=yes scorefile=native_scores.tsv replace=yes;
quit;
END;
'''
    driver=out/'score_existing_trees.nex'
    driver.write_text(matrix_only+commands)
    env=os.environ.copy();env['LD_LIBRARY_PATH']=str(runtime.resolve());env['OMP_NUM_THREADS']='1'
    cmd=[str(binary.resolve()),'-n','-u','-L','native_scores.log','-r',driver.name]
    with (out/'stdout.txt').open('w') as handle:
        run=subprocess.run(cmd,cwd=out,env=env,stdout=handle,stderr=subprocess.STDOUT,timeout=300)
    if run.returncode:raise RuntimeError(f'PAUP exit={run.returncode}; inspect {out}/stdout.txt')
    log=(out/'native_scores.log').read_text()
    if re.search(r'\berrors?\b|execution terminated|processing.*terminated',log,re.I):
        raise RuntimeError(f'Native PAUP log reports an error despite exit code{run.returncode}; inspect {out}/native_scores.log')
    assert 'Multistate taxa interpretation depends on "{}" versus "()" designation' in log
    totals,scores=read_scorefile(out/'native_scores.tsv',nchar)
    assert len(totals)==n_input_trees,(len(totals),n_input_trees)
    # Preserve the complete native bytes atomically after process exit; no score
    # content is transformed. Its hash makes later file-integrity checks explicit.
    raw_native=(out/'native_scores.tsv').read_bytes()
    with (out/'native_scores.tsv.part').open('wb') as handle:
        handle.write(raw_native);handle.flush();os.fsync(handle.fileno())
    (out/'native_scores.tsv.part').replace(out/'native_scores.tsv')
    label_block=re.search(r'CHARSTATELABELS\s+(.*?);',matrix_only,re.I|re.S)
    labels={int(m[1]):m[2].split('/')[0].strip() for m in re.finditer(r'(?:^|,)\s*(\d+)\s+([^,]+)',label_block[1])} if label_block else {}
    frozen=PROJECT/'round2/inputs/tetrapods_frozen_targets.csv'
    cells={(r['taxon'],int(r['character'])) for r in csv.DictReader(frozen.open()) if r['scenario']=='later_publication' and r['model']=='mode'}
    assert len(cells)==350
    target_counts=Counter(ch for taxon,ch in cells)
    character_csv_check=write_character_table(scores,labels,target_counts,out/'character_lengths.csv')
    report=dict(matrix=str(matrix),tree_input=str(trees),nchar=nchar,n_trees=len(scores),
                native_option='pset mstaxa=variable opt=deltran; pscores all / single=all tl=yes',
                native_command=cmd,no_topology_search=True,
                leaf_identity_audit=leaf_audit,no_native_error_in_log=True,
                character_csv_check=character_csv_check,
                exact_sum_checks={str(t):{'sum_character_steps':str(sum(scores[t].values())),'native_total':str(totals[t]),'equal':True} for t in totals},
                hashes={'matrix':sha(matrix),'trees':sha(trees),'binary':sha(binary),'diagnostic_driver':sha(driver),'native_tsv':sha(out/'native_scores.tsv')},
                target_index_source=str(frozen),n_frozen_target_cells=len(cells),n_target_characters=len(target_counts))
    # Summarize every provided tree, not a selected representative. Exact rational
    # means guarantee that per-character means add to the exact mean tree length.
    mean_steps={j:sum(Fraction(scores[t][j]) for t in scores)/len(scores) for j in range(1,nchar+1)}
    mean_total=sum(Fraction(x) for x in totals.values())/len(totals)
    assert sum(mean_steps.values())==mean_total
    with (out/'character_ensemble.csv').open('w') as handle:
        fields=['character','source_character_label','n_trees','mean_steps','exact_mean_fraction','min_steps','max_steps','frozen_target_cells_in_character']
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for j in range(1,nchar+1):
            values=[scores[t][j] for t in scores]
            writer.writerow(dict(character=j,source_character_label=labels.get(j,''),n_trees=len(scores),mean_steps=float(mean_steps[j]),exact_mean_fraction=str(mean_steps[j]),min_steps=str(min(values)),max_steps=str(max(values)),frozen_target_cells_in_character=target_counts[j]))
    report['ensemble']={'all_input_trees_scored':True,'all_tree_totals_equal':len(set(totals.values()))==1,'exact_mean_tree_length':str(mean_total),'exact_sum_character_means':str(sum(mean_steps.values())),'interpretation':'descriptive summary of all supplied existing trees, not a posterior or exhaustive optimal-tree distribution','best_tree_claim_made':False}
    if len(scores)==2:
        keys=sorted(scores);left,right=keys;diff={j:scores[right][j]-scores[left][j] for j in scores[left]}
        assert sum(diff.values())==totals[right]-totals[left]
        with (out/'tree2_minus_tree1_character_contrast.csv').open('w') as handle:
            fields=['character','source_character_label','tree1_steps','tree2_steps','tree2_minus_tree1','frozen_target_cells_in_character']
            writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
            for j in sorted(diff):writer.writerow(dict(character=j,source_character_label=labels.get(j,''),tree1_steps=str(scores[left][j]),tree2_steps=str(scores[right][j]),tree2_minus_tree1=str(diff[j]),frozen_target_cells_in_character=target_counts[j]))
        report['two_tree_contrast']={'right_minus_left':str(sum(diff.values())),'n_positive':sum(v>0 for v in diff.values()),'n_negative':sum(v<0 for v in diff.values()),'n_zero':sum(v==0 for v in diff.values()),'target_character_difference_sum':str(sum(v for j,v in diff.items() if target_counts[j])),'other_character_difference_sum':str(sum(v for j,v in diff.items() if not target_counts[j]))}
    (out/'validation.json').write_text(json.dumps(report,indent=2))
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix',type=Path,default=WORK/'inputs/source.nex')
    parser.add_argument('--trees',type=Path,default=HERE/'published_O1_O2.tre')
    parser.add_argument('--out',type=Path,default=HERE/'original_O1_O2')
    parser.add_argument('--binary',type=Path,default=WORK/'tools/paup4a169_ubuntu64')
    parser.add_argument('--runtime',type=Path,default=WORK/'tools/runtime/usr/lib/x86_64-linux-gnu')
    args=parser.parse_args()
    print(json.dumps(native_export(args.matrix,args.trees,args.out,args.binary,args.runtime),indent=2))
