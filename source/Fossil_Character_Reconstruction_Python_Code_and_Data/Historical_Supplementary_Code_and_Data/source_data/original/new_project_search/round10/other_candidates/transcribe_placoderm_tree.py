"""Transcribe the original2022 supplementary strict-consensus vector figure.

This is a figure transcription, not an original authors' machine tree. All
multifurcations are retained. No branch lengths or binary resolutions inferred.
"""
from pathlib import Path
import fitz, json, collections, re, csv, hashlib
from screen_placoderm import parse
P=Path(__file__).resolve().parent;OUT=P/'fish_frozen'

def main():
    file=P/'placoderm2022_tree.pdf';page=fitz.open(file)[0]
    horizontal=[];vertical=collections.defaultdict(list)
    for path in page.get_drawings():
        if path['type']!='s':continue
        for shape in path['items']:
            assert shape[0]=='l'
            _,a,b=shape
            if abs(a.x-b.x)<.003 and abs(a.y-b.y)<.003:continue
            if abs(a.y-b.y)<.003:
                horizontal.append((round(min(a.x,b.x),3),round(max(a.x,b.x),3),round(a.y,3)))
            elif abs(a.x-b.x)<.003:
                vertical[round(a.x,3)].append((min(a.y,b.y),max(a.y,b.y)))
            else:raise ValueError('Nonorthogonal segment')
    components=[]
    for x,segments in vertical.items():
        parts=[]
        for lo,hi in sorted(segments):
            if parts and lo<=parts[-1][1]+.003:parts[-1][1]=max(parts[-1][1],hi)
            else:parts.append([lo,hi])
        components.extend((x,lo,hi) for lo,hi in parts)
    def node_at(x,y):
        hit=[i for i,(xx,lo,hi) in enumerate(components) if abs(xx-x)<.003 and lo-.003<=y<=hi+.003]
        assert len(hit)<=1,(x,y,hit)
        return ('node',hit[0]) if hit else ('point',x,y)
    children=collections.defaultdict(list);parents={}
    for a,b,y in horizontal:
        parent,child=node_at(a,y),node_at(b,y)
        assert child not in parents,(parent,child,parents.get(child),a,b,y)
        children[parent].append(child);parents[child]=parent
    roots=set(children)-set(parents);assert len(roots)==1
    root=roots.pop()
    while len(children[root])==1:root=children[root][0]
    leaves=[x for x in parents if x not in children]
    assert len(leaves)==157
    spans=[s for b in page.get_text('dict')['blocks'] if b['type']==0 for l in b['lines'] for s in l['spans'] if s['origin'][1]>20]
    assert len(spans)==157
    old,_,_=parse(P/'placoderm2022_matrix.nex')
    normalized=lambda n:re.sub(r'[^a-z0-9]','',n.lower())
    labels={};audit=[];used=set()
    for leaf in leaves:
        _,x,y=leaf
        scores=sorted((abs(s['origin'][1]-y-3.386)+abs(s['origin'][0]-x-5.64),k,s) for k,s in enumerate(spans))
        score,k,s=scores[0]
        assert score<.02 and scores[1][0]>8 and k not in used,(leaf,scores[:2])
        used.add(k);printed=s['text'];nn=normalized(printed)
        exact=[n for n in old if normalized(n)==nn]
        matches=exact if exact else [n for n in old if normalized(n).startswith(nn)]
        assert len(matches)==1,(printed,matches)
        labels[leaf]=matches[0]
        audit.append(dict(printed_label=printed,old_raw_otu=matches[0],match_rule='unique normalized exact' if exact else 'unique normalized truncated prefix',leaf_x=x,leaf_y=y,text_x=s['origin'][0],text_baseline_y=s['origin'][1],geometry_residual=score))
    assert len(set(labels.values()))==157 and set(old)-set(labels.values())=={'Xiushanosteus','Shenacanthus'}
    def clade(n):return {labels[n]} if n in labels else set().union(*(clade(z) for z in children[n]))
    visited=set()
    def visit(n):
        assert n not in visited
        visited.add(n)
        for z in children.get(n,[]):visit(z)
    visit(root)
    assert len(visited)==249 and len([n for n in visited if n in children])==92
    # Manually checked against enlarged top/middle/bottom areas of the PDF,
    # including true multifurcations rather than only obvious sister pairs.
    checked_groups=[['Entelognathus_primordialis','Qilinyu_rostrata'],
                    ['Campbellodus_decipiens','Rhamphodopsis_threiplandi','Austroptyctodus_gardineri'],
                    ['Kansasiella','Lawrenciella','Osorioichthys_marginis','Moythomasia_durgaringa','Mimipiscis_toombsi','Kentuckia_deani_','Howqualepis_rostridens','Raynerius'],
                    ['Triodus','Diplodoselache','Orthacanthus_spp'],
                    ['Squalus','Synechodus']]
    for group in checked_groups:
        hits=[n for n in visited if n in children and clade(n)==set(group)]
        assert len(hits)==1 and len(children[hits[0]])==len(group)
    def newick(n):return "'"+labels[n].replace("'","''")+"'" if n in labels else '('+','.join(newick(z) for z in sorted(children[n],key=lambda x:min(clade(x))))+')'
    text=newick(root)+';\n'
    (OUT/'published2022_strict_consensus_graph_transcription.nwk').write_text(text)
    with (OUT/'published_tree_label_geometry_audit.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(audit[0]));w.writeheader();w.writerows(sorted(audit,key=lambda r:r['leaf_y']))
    clades=[sorted(clade(n)) for n in children if n!=root and len(clade(n))>1]
    (OUT/'published_tree_clades.json').write_text(json.dumps(clades,indent=2)+'\n')
    info=dict(source_doi='10.1038/s41586-022-05136-8',source_url='https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-022-05136-8/MediaObjects/41586_2022_5136_MOESM5_ESM.pdf',source_pdf_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),description='Original Supplementary Data 3, strict consensus of 46 MPTs. The analysis excluded Xiushanosteus and Shenacanthus before running; figure header says 0 taxa excluded relative to that reduced input.',provenance='Geometric transcription of original published vector figure, not original author machine tree; no 2026 topology used',tips=len(leaves),horizontal_segments=len(horizontal),vertical_segments=sum(len(x) for x in vertical.values()),vertical_components=len(components),internal_nodes_after_root_stem_removed=92,degree_counts=dict(collections.Counter(len(children[n]) for n in visited if n in children)),maximum_children=max(map(len,children.values())),all159old_minus_two_tips_confirmed=True,graph_checks='All horizontal connections have unique parent; exactly one rooted connected tree; 157 distinct tips; unique geometric text matches under residual 0.02; unique exact or truncated-prefix old-name mapping',biological_limit='A strict consensus retains soft polytomies. Do not interpret a score on this multifurcation as the minimum over its binary refinements. No numerical branch lengths are provided.',manual_checks_pending=False,manually_checked_direct_child_groups=checked_groups)
    (OUT/'published_tree_transcription_audit.json').write_text(json.dumps(info,indent=2)+'\n')
    print(json.dumps(info,indent=2))

if __name__=='__main__':main()
