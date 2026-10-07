"""Freeze the Zhu 2022 -> Cui 2023 historical panel without model outcomes.

The raw line parser is checked against an independent command/token parser.
Only exact OTU labels and exact character/state labels after removal of
documented source-number prefixes are joined. Source PDFs are read-only audit
materials, not redistributable outputs of this script.
"""
from pathlib import Path
import re, json, csv, hashlib, collections
import numpy as np
from pypdf import PdfReader
from screen_placoderm import parse

P = Path(__file__).resolve().parent
OUT = P / 'fish_frozen'
OUT.mkdir(exist_ok=True)
OLD = P / 'placoderm2022_matrix.nex'
NEW = P / 'placoderm2023/Supplementary Data 1.nex'
SINGLE = re.compile(r'^[0-9A-Za-z]$')
PREFIX = re.compile(r'^(?:(?:\[?[A-Z]+\d+\]?)(?:,?\s+))+')

def independent_parse(path):
    raw=path.read_bytes()
    try: text=raw.decode('utf8')
    except UnicodeDecodeError: text=raw.decode('cp1252')
    # Independent full-file lexical scan. Comments, quoted atoms, and
    # punctuation are tokenized before locating the MATRIX command.
    tokens=[]; i=0
    while i<len(text):
        c=text[i]
        if c.isspace(): i+=1; continue
        if c=='[':
            level=1; i+=1
            while level:
                if text[i]=='[': level+=1
                elif text[i]==']': level-=1
                i+=1
            continue
        if c=="'":
            i+=1; s=''
            while i<len(text):
                if text[i]=="'":
                    if i+1<len(text) and text[i+1]=="'": s+="'"; i+=2; continue
                    i+=1; break
                s+=text[i];i+=1
            tokens.append(s); continue
        if c in ';=(),{}': tokens.append(c); i+=1;continue
        j=i
        while i<len(text) and not text[i].isspace() and text[i] not in "[]';=(),{}": i+=1
        tokens.append(text[j:i])
    upper=[t.upper() for t in tokens]
    nt=int(tokens[upper.index('NTAX')+2]); nc=int(tokens[upper.index('NCHAR')+2])
    tl=upper.index('TAXLABELS')+1; labels=[]
    while tokens[tl]!=';':labels.append(tokens[tl]);tl+=1
    assert len(labels)==nt
    q=upper.index('MATRIX')+1; rows=[]; actual=[]
    for row in range(nt):
        name=tokens[q];q+=1;cells=[];actual.append(name)
        while len(cells)<nc:
            z=tokens[q];q+=1
            if z in ('(', '{'):
                close=')' if z=='(' else '}';cell=z
                while tokens[q]!=close:cell+=tokens[q];q+=1
                cells.append(cell+close);q+=1
            else:cells.extend(z)
        assert len(cells)==nc
        rows.append(cells)
    assert tokens[q]==';' and actual==labels
    return actual,np.array(rows)

def write_csv(name, rows):
    assert rows
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def declarations(year):
    first,last=(16,90) if year==2022 else (21,72)
    r=PdfReader(P/f'placoderm{year}_supp.pdf');d={};cur=0
    for ix in range(first,last):
        for line in r.pages[ix].extract_text().splitlines():
            if re.fullmatch(r'\s*\d+\s*',line): continue
            m=re.match(r'^\s*(\d+)\.\s+(.*)',line)
            # Only sequential character headings. A reference line may end
            # with "Character 346." and must never replace character 346.
            if m and int(m[1])==cur+1:
                cur+=1;d[cur]={'page':ix+1,'text':m[2]}
            elif cur:d[cur]['text']+=' '+line.strip()
    assert cur==(691 if year==2022 else 694),cur
    for v in d.values():
        text=v['text']
        # Only character declaration before source citations/coding notes.
        text=re.split(r'\b(?:Choo|King|Castiello|Coates|Dearden|Long|Cui|Zhu|Giles|Burrow|Dupret|Brazeau|Davis|Friedman|Janvier|Lu|Maisey|Qiao|Zhao|Hanke|Gardiner|Cloutier|Goujet|Sallan)\s+(?:et al\.|\(\d)',text)[0]
        text=re.split(r'\b(?:The coding|The original|Chondrichthyan|Supplementary Figures|Hyoid and gill arches)',text)[0]
        # First declaration period after state zero; leave an unpunctuated
        # final state intact (several original headings omit final full stop).
        if '(0)' in text:
            start=text.index('(0)');end=text.find('.',start)
            if end>=0:text=text[:end+1]
        v['declaration']=text.strip()
    return d

def canon(x):return re.sub(r'[^a-z0-9]','',x.lower())
def normalized(d):return (PREFIX.sub('',d[0]),*d[1:])

def main():
    old,A,ad=parse(OLD);later,B,bd=parse(NEW)
    for path,names,matrix in [(OLD,old,A),(NEW,later,B)]:
        names2,matrix2=independent_parse(path)
        assert names==names2 and np.array_equal(matrix,matrix2),path
    ap,bp=declarations(2022),declarations(2023)
    inv=collections.defaultdict(list)
    for j,d in bd.items():inv[normalized(d)].append(j)
    pairs=[(i,inv[normalized(d)][0]) for i,d in ad.items() if len(inv[normalized(d)])==1]
    # Policy exclusions fixed from source notes, before fitting any model.
    exclusions={37:'2023 Note 5 explicitly changes absent versus inapplicable coding for chondrichthyan scales',
                46:'2023 Note 5 explicitly changes absent versus inapplicable coding for chondrichthyan scales',
                47:'2023 Note 5 explicitly changes absent versus inapplicable coding for chondrichthyan scales and selected osteichthyans',
                48:'2023 Note 5 explicitly changes absent versus inapplicable coding for chondrichthyan scales',
                49:'2023 applicability change plus source contradiction: state 0 is present in definition but called absent in coding note',
                50:'2023 Note 5 explicitly changes absent versus inapplicable coding for chondrichthyan scales',
                593:'Both source PDFs reverse state meanings relative to their own Nexus labels; unresolved source-internal semantics',
                604:'Both source PDFs reverse state meanings relative to their own Nexus labels; unresolved source-internal semantics'}
    # Remaining nonliteral PDF differences are inspected and reported. Do not
    # infer equivalence from observed states or prediction performance.
    differ=[]
    for i,j in pairs:
        if canon(ap[i+1]['declaration'])!=canon(bp[j+1]['declaration']):
            differ.append({'old_character':i+1,'new_character':j+1,'old_declaration':ap[i+1]['declaration'],'later_declaration':bp[j+1]['declaration']})
    if differ:
        write_csv('pdf_declaration_differences.csv',differ)
        raise RuntimeError('Unresolved PDF declaration differences require source inspection')
    charrows=[];accepted=[]
    for i in range(A.shape[1]):
        match=[j for ii,j in pairs if ii==i]
        j=match[0] if match else None
        reason=exclusions.get(i+1,'') if j is not None else 'No unique exact normalized title plus complete state-label sequence'
        if j is not None and not reason:accepted.append((i,j))
        charrows.append(dict(old_character=i+1,new_character='' if j is None else j+1,admitted=bool(j is not None and not reason),removed_source_prefix=(PREFIX.match(ad[i][0])[0] if PREFIX.match(ad[i][0]) else ''),old_title=ad[i][0],later_title='' if j is None else bd[j][0],old_states=' | '.join(ad[i][2:]),later_states='' if j is None else ' | '.join(bd[j][2:]),old_pdf_page=ap[i+1]['page'],later_pdf_page='' if j is None else bp[j+1]['page'],exclusion_reason=reason))
    write_csv('all_character_audit.csv',charrows)
    write_csv('accepted_character_mapping.csv',[x for x in charrows if x['admitted']])
    write_csv('old_raw_otu_names.csv',[dict(old_index=i,old_otu=n) for i,n in enumerate(old)])
    write_csv('later_raw_otu_names.csv',[dict(later_index=i,later_otu=n) for i,n in enumerate(later)])
    write_csv('otu_join_audit.csv',[dict(old_index=i,old_otu=n,exact_later_index=later.index(n) if n in later else '',admitted=n in later,scope_class='composite_or_higher_taxon' if ('spp' in n or n in ['Osteostraci','Galeaspida','Chimaeroidei']) else 'named_species' if '_' in n else 'genus_label',rule='literal raw OTU equality only; renamed or narrowed composites are not joined') for i,n in enumerate(old)])
    rows=[]
    for n in old:
        if n not in later:continue
        a=old.index(n);b=later.index(n)
        for k,(i,j) in enumerate(accepted):
            count=sum(bool(SINGLE.fullmatch(s)) for s in A[:,i])
            if A[a,i]=='?' and SINGLE.fullmatch(B[b,j]) and count>=15:
                rows.append(dict(target_id=f'fish_{a}_{i+1}',taxon=n,old_index=a,later_index=b,admitted_character_index=k,old_character=i+1,later_character=j+1,old_state='?',later_state=B[b,j],old_singleton_count=count,scope_class='composite_or_higher_taxon' if ('spp' in n or n in ['Osteostraci','Galeaspida','Chimaeroidei']) else 'named_species'))
    write_csv('future_targets.csv',rows)
    np.savez(OUT/'source_model_inputs.npz',A=A[:,[i for i,j in accepted]],old_names=np.array(old),old_char_numbers=np.array([i+1 for i,j in accepted]),old_characters=np.array([i+1 for i,j in accepted]))
    np.savez(OUT/'full_raw_matrices.npz',old_A=A,later_B=B,old_names=np.array(old),later_names=np.array(later))
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    freeze=dict(chain='Zhu et al. 2022 Nature -> Cui et al. 2023 Nature Communications',old_shape=list(A.shape),later_shape=list(B.shape),old_doi='10.1038/s41586-022-05136-8',later_doi='10.1038/s41467-023-43557-9',old_sha256=sha(OLD),later_sha256=sha(NEW),independent_raw_parsers_agree=True,raw_symbol_counts_old=dict(collections.Counter(A.flat)),raw_symbol_counts_later=dict(collections.Counter(B.flat)),exact_shared_otus=len(set(old)&set(later)),normalized_character_candidates=len(pairs),admitted_characters=len(accepted),character_exclusions=exclusions,targets=len(rows),target_otus=len(set(r['taxon'] for r in rows)),target_characters=len(set(r['old_character'] for r in rows)),target_taxon_counts=dict(collections.Counter(r['taxon'] for r in rows)),target_character_counts=dict(collections.Counter(str(r['old_character']) for r in rows)),composite_target_cells=sum(r['scope_class']=='composite_or_higher_taxon' for r in rows),criterion='old literal ?; later exactly one alphanumeric state; >=15 literal old singleton observations in same character over original159 old OTUs; exact raw OTU label; exact normalized Nexus character and ordered state labels; identical source PDF declarations; eight source-semantic exclusions; no model-dependent selection',treatment='All original polymorphic, uncertain, gap and missing tokens retained in raw files. Predictive models must document treatment separately. Both Nexus files declare all characters unordered.',tree_status=('Original2022 published strict-consensus157-tip vector figure graph transcribed and audited; see published_tree_transcription_audit.json. Original machine tree retrieval blocked403/401.' if (OUT/'published_tree_transcription_audit.json').exists() else 'Original2022 publisher strict consensus157-tip PDF downloaded. Original machine tree Dryad public web/API retrieval blocked403/401. Graph transcription not yet frozen.'),interpretation='Endpoint is later published coding at OTU level, not verified fixed-specimen ground truth or necessarily a newly discovered observation.',outcome_blind='No candidate-specific predictive model or downstream result has been calculated during admission.')
    (OUT/'freeze.json').write_text(json.dumps(freeze,indent=2)+'\n')
    print(json.dumps({k:freeze[k] for k in ['admitted_characters','targets','target_otus','target_characters','composite_target_cells']},indent=2))

if __name__=='__main__':main()
