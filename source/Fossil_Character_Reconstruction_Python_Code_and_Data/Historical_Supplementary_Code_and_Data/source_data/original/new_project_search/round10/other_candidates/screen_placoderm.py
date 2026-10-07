from pathlib import Path
import re,json,csv,collections,numpy as np
P=Path(__file__).resolve().parent
def parse(f):
 b=f.read_bytes()
 try:t=b.decode('utf8')
 except UnicodeDecodeError:t=b.decode('cp1252')
 nc=int(re.search(r'NCHAR\s*=\s*(\d+)',t,re.I)[1]);nt=int(re.search(r'NTAX\s*=\s*(\d+)',t,re.I)[1]);cl=re.search(r'CHARSTATELABELS\s+(.*?);',t,re.I|re.S)[1]
 ds={}
 for e in re.split(r',\s*(?=\d+\s)',cl.strip()):
  m=re.match(r'(\d+)\s+(.*)',e,re.S);tokens=re.findall(r"'(?:[^']|'')*'|[^\s]+",m[2]); ds[int(m[1])-1]=tuple(z.strip("'").replace("''", "'").replace('_',' ').strip() for z in tokens)
 b=re.search(r'^\s*MATRIX\s*\n(.*?);',t,re.I|re.S|re.M)[1];n=[];rr=[]
 for line in b.strip().splitlines():
  line=re.sub(r'\[.*?\]','',line).strip()
  if not line:continue
  m=re.match(r"('(?:[^']|'')*'|\S+)\s+(.+)",line);n.append(m[1].strip("'"));cells=re.findall(r'\([^)]*\)|\{[^}]*\}|.',re.sub(r'\s+','',m[2]));assert len(cells)==nc,(n[-1],len(cells),nc);rr.append(cells)
 assert len(n)==nt and len(ds)==nc
 return n,np.array(rr),ds

def main():
 old,A,ad=parse(P/'placoderm2022_matrix.nex');new,B,bd=parse(P/'placoderm2023/Supplementary Data 1.nex');ri=collections.defaultdict(list)
 for j,d in bd.items():ri[d].append(j)
 pairs=[(i,ri[d][0]) for i,d in ad.items() if len(ri[d])==1]
 rows=[]
 for n in sorted(set(old)&set(new)):
  a=old.index(n);b=new.index(n)
  for i,j in pairs:
   if A[a,i]=='?' and re.fullmatch('[0-9A-Za-z]',B[b,j]):rows.append(dict(taxon=n,old_index=a,new_index=b,old_character=i+1,new_character=j+1,later_state=B[b,j],old_singleton_count=sum(bool(re.fullmatch('[0-9A-Za-z]',s)) for s in A[:,i])))
 elig=[r for r in rows if r['old_singleton_count']>=15]
 out=dict(old_shape=A.shape,new_shape=B.shape,exact_character_definition_matches=len(pairs),exact_shared_taxa=len(set(old)&set(new)),question_to_singleton_cells=len(rows),eligible_cells=len(elig),eligible_taxa=dict(collections.Counter(r['taxon'] for r in elig)),unmapped_old_characters=[i+1 for i in ad if i not in dict(pairs)],status='Screen only. Character definition and applicability changes require audit before admission.')
 (P/'placoderm_screen.json').write_text(json.dumps(out,indent=2)+'\n')
 with (P/'placoderm_candidate_targets.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 with (P/'placoderm_exact_character_map.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['old_character','new_character','definition_and_states']);w.writeheader();w.writerows(dict(old_character=i+1,new_character=j+1,definition_and_states=' | '.join(ad[i])) for i,j in pairs)
 np.savez(P/'placoderm_parsed.npz',A=A,B=B,old_names=old,new_names=new,character_pairs=np.array(pairs))
 print(json.dumps(out,indent=2))

if __name__=='__main__':main()
