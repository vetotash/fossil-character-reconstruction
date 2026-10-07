#!/usr/bin/env python3
"""Fixed-sample documentary audit; no predictions are read and no expert review is claimed."""
from pathlib import Path
import copy,hashlib,json,re,collections
import fitz

ROOT=Path(__file__).resolve().parents[2];A=ROOT/'extension_work/anatomy';P=ROOT/'public_validation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
source=json.loads((P/'data/processed/gnathostomes.json').read_text())
selected=[r for r in json.loads((A/'selected_endpoints.json').read_text()) if r['history']=='gnathostomes']
assert len(selected)==20
old_pdf=P/'data/raw/gnathostomes/41586_2022_5136_MOESM1_ESM.pdf'
new_pdf=P/'data/raw/gnathostomes/41467_2023_43557_MOESM1_ESM.pdf'
xml=P/'data/raw/gnathostomes/cui2023_europepmc.xml'
old_doi='10.1038/s41586-022-05136-8';new_doi='10.1038/s41467-023-43557-9'
old_doc=fitz.open(old_pdf);new_doc=fitz.open(new_pdf)
provenance={str(P/r['path']):r for r in source['provenance'] if 'path' in r}
def locator(path,doi,kind,**extra):
    p=Path(path);item=provenance.get(str(p),{})
    return {'kind':kind,'doi':doi,'url':item.get('url',item.get('source_url','https://doi.org/'+doi if doi else '')),'local_file':str(p.relative_to(ROOT)),'sha256':sha(p),**extra}

def raw_matrix(side):
    info=source[side];path=P/info['source_file'];assert sha(path)==info['source_sha256']
    text=path.read_text(encoding=info['source_text_encoding'])
    start=re.search(r'(?im)^\s*MATRIX\s*$',text)
    assert start
    block=text[start.end():].split(';',1)[0]
    records={}
    lines={}
    for line in block.splitlines():
        if not line.strip():continue
        match=re.match(r"\s*('(?:[^']|'')*'|\S+)\s+(.+)",line)
        assert match,line[:60]
        label=match[1].strip("'").replace("''", "'")
        tokens=re.findall(r'\{[^}]*\}|\([^)]*\)|[^\s]',match[2])
        tokens=[''.join(t.split()) for t in tokens]
        assert label not in records
        records[label]=tokens
        lines[label]=text[:start.end()].count('\n')+block[:block.index(line)].count('\n')+1
    expected=len(info['characters'])
    assert all(len(v)==expected for v in records.values())
    return path,records,lines
old_mat,old_rows,old_lines=raw_matrix('old');new_mat,new_rows,new_lines=raw_matrix('new')
# Character meanings were read directly in both PDF lists; summaries are paraphrases.
characters={
13:(19,22,'Enameloid occurrence on dermal skeleton: absent/present.'),
19:(19,23,'Extensive pore canals: absent/present.'),
32:(21,23,'Scute-like ridge scales (basal fulcra): absent/present.'),
33:(21,23,'Flank-scale arrangement: vertical, oblique/rhombic, or disorganised.'),
34:(21,23,'Scale category: macromeric versus micromeric.'),
41:(22,24,'Flattened scale bases: absent/present.'),
44:(22,25,'Basal scale canals or open vascular spaces: absent/present.'),
45:(22,25,'Scale neck canal: absent/present.'),
52:(23,26,'Body sensory canal: below scales, over/enclosed by scales, or traversing scales.'),
263:(45,41,'Strict-sense maxilla/premaxilla lacking a palatal lamina: absent/present.'),
448:(65,55,'PDFs describe absence/presence; Nexus labels instead describe position relative to the orbit.'),
645:(85,69,'Pelvic-fin basal organisation: monobasal versus polybasal; renumbered to644.')}
taxon_pages={'Bothriolepis_spp':(10,14),'Campbellodus_decipiens':(11,15),'Cheirolepis_canadensis':(11,15),'Entelognathus_primordialis':(12,16),'Gogonasus_andrewsae':(13,17),'Homalacanthus_concinnus':(13,17),'Ischnacanthus_gracilis_':(13,17),'Jagorina_pandora':(13,17),'Ligulalepis_toombsi':(14,18),'Lupopsyrus_pygmaeus_':(14,18),'Mesacanthus_mitchelli':(14,18),'Osorioichthys_marginis':(15,19),'Psarolepis_romeri_':(15,19),'Pterichthyodes_milleri':(15,19),'Romundina_stellina_':(16,20)}
source_notes={
'Bothriolepis_spp':'Both taxon lists cite Long and Trinajstic2010. The OTU explicitly pools species; neither matrix names the contributing specimen set.',
'Campbellodus_decipiens':'Both taxon lists cite Long1997. No cell-specific source or unchanged specimen inventory is given for the scale-category update.',
'Cheirolepis_canadensis':'Both taxon lists cite Arratia and Cloutier1996 and Pearson and Westoll1979. The revised list identifies no specimen-specific explanation for these three selected updates.',
'Entelognathus_primordialis':'The old taxon list cites Zhu et al.2013. Cui2023 presents new articulated post-thoracic and isolated scale/plate material; the shared name does not establish an unchanged specimen basis.',
'Gogonasus_andrewsae':'Old and revised taxon lists cite multiple anatomical works. The later list references88–92; neither list assigns an individual specimen to the selected scale-alignment state.',
'Homalacanthus_concinnus':'Both lists cite Russell1951. No specimen-specific source explanation was located for the selected later hyomandibular state.',
'Ischnacanthus_gracilis_':'The old list cites Miles1973 and Watson1937. The later list adds Burrow et al.2018 (reference110). This documents added literature, but does not identify the specimen underlying this particular recoding.',
'Jagorina_pandora':'Both lists cite Stensio1969 and Young1986. The later character list supplies no taxon-specific scale-alignment rationale or specimen inventory.',
'Ligulalepis_toombsi':'The old list cites Schultze1968, Burrow1994, Basden et al.2000 and Basden and Young2001. The later list adds Burrow, Young and Lu2023 (reference131). Exact specimen attribution of the new scale-category code remains unreported.',
'Lupopsyrus_pygmaeus_':'Both lists cite Hanke and Davis2012 (later reference135). The DOI resolved to an access/challenge response during this audit; a complete primary anatomical text was not recovered, so flattened-base anatomy is not independently verified here.',
'Mesacanthus_mitchelli':'The old list cites Miles1973 and Watson1937. The later list adds Burrow et al.2022 (reference142). The recovered DOI page is a resolver, not the article; the exact neck-canal assertion and contributing specimens remain unresolved.',
'Osorioichthys_marginis':'Both lists cite Taverne1997 (later reference159). That primary paper explicitly describes one holotype, IRSNB P1.340, and identifies its upper-jaw bones, but the selected strict-sense state also requires the palatal-lamina qualification.',
'Psarolepis_romeri_':'Both lists point to multiple older anatomical studies. No taxon-specific rationale in the revised character list identifies the specimen basis of this scale-category update.',
'Pterichthyodes_milleri':'Both lists cite Hemmings1978 (later reference143). No cell-specific scale-category explanation or specimen linkage is given in the retrieved 2023 supplement.',
'Romundina_stellina_':'Both lists cite Dupret et al.2014 and Orvig1975. Neither retrieved character list assigns a specimen, section or figure to this selected neck-canal code.'}
rows=[]
for original in selected:
    row=copy.deepcopy(original);taxon=row['original_taxon_label'];o=row['old_character_index_zero_based'];n=row['new_character_index_zero_based']
    assert old_rows[taxon][o]=='?' and new_rows[taxon][n]==row['later_state']
    op,np,definition=characters[o+1];ot,nt=taxon_pages[taxon]
    # Independent primary text declaration check, not just a machine-readable title match.
    assert re.search(r'\b'+str(o+1)+r'\.',old_doc[op-1].get_text())
    assert re.search(r'\b'+str(n+1)+r'\.',new_doc[np-1].get_text())
    row.update(documentary_review_status='UNRESOLVED',token_history='SUPPORTED_BY_EXPLICIT_SOURCE',
               character_correspondence='SUPPORTED_BY_EXPLICIT_SOURCE',taxon_scope='UNRESOLVED',
               specimen_continuity='UNRESOLVED',later_state_support='UNRESOLVED',
               support_mode='MATRIX_ENTRY_ONLY',old_specimens='',later_specimens='',
               expert_review_status='NOT_PERFORMED',old_primary_token='?',later_primary_token=new_rows[taxon][n],
               old_otu_label=taxon,later_otu_label=taxon,character_definition_summary=definition,
               source_reason=source_notes[taxon]+' Exact primary tokens and character correspondence were checked; label identity alone is not specimen continuity or anatomical support.',
               label_exposure='Later published state was visible to this documentary reviewer; no blinded anatomical assessment was performed.',
               taxon_scope_note='Exact machine-readable label is shared; no claim of invariant biological/specimen scope.',
               source_locators=[locator(old_mat,old_doi,'old_primary_matrix',matrix_row=taxon,character_one_based=o+1,line=old_lines[taxon]),
                                locator(new_mat,new_doi,'later_primary_matrix',matrix_row=taxon,character_one_based=n+1,line=new_lines[taxon]),
                                locator(old_pdf,old_doi,'old_character_definition',pdf_page=op,printed_page=op-1,character_one_based=o+1),
                                locator(new_pdf,new_doi,'later_character_definition',pdf_page=np,printed_page=np,character_one_based=n+1),
                                locator(old_pdf,old_doi,'old_taxon_sources',pdf_page=ot,printed_page=ot-1),
                                locator(new_pdf,new_doi,'later_taxon_sources',pdf_page=nt,printed_page=nt)])
    if o==447:
        duplicate=old_rows[taxon][449];assert duplicate==row['later_state'] and duplicate in '0123456789'
        row.update(documentary_review_status='DOCUMENTED_CHANGE_OR_CONFLICT',character_correspondence='DOCUMENTED_CHANGE_OR_CONFLICT',later_state_support='DOCUMENTED_CHANGE_OR_CONFLICT',support_mode='INHERITED_MATRIX_WITHOUT_CELL_RATIONALE',old_duplicate_character_one_based=450,old_duplicate_token=duplicate)
        row['source_reason']+=' Original450 was already scored with the same singleton and is explicitly declared a duplicate of448 by Cui2023. This endpoint is not independent newly available trait information. Both PDF lists use absent/present for448 while both Nexus labels use anatomical position; that within-source discrepancy is not silently resolved.'
        row['source_locators'].extend([locator(old_mat,old_doi,'already_observed_duplicate',matrix_row=taxon,character_one_based=450,line=old_lines[taxon]),locator(xml,new_doi,'explicit_duplicate_declaration',section='Results and discussion: phylogenetic-analysis paragraph beginning To investigate the impact',character_one_based=[448,450])])
        if taxon=='Bothriolepis_spp':row['taxon_scope_note']='Explicit multi-species OTU; contributing species and specimen inventories remain unresolved.'
    if taxon=='Entelognathus_primordialis':
        row.update(documentary_review_status='DOCUMENTED_CHANGE_OR_CONFLICT',specimen_continuity='DOCUMENTED_CHANGE_OR_CONFLICT',later_specimens='IVPP V32322; V32323.1–V32323.5 (Cui2023 study inventory; individual character linkage below)',old_specimens='Earlier taxon source Zhu et al.2013; original specimen inventory not recovered in this audit.')
        row['source_locators'].extend([locator(xml,new_doi,'new_material_inventory',section_id='Sec9',section='Methods/Materials'),locator(xml,new_doi,'scanned_specimens',section_id='Sec10',specimen='IVPP V32322; V32323.5')])
        if o==51:
            row.update(later_state_support='SUPPORTED_BY_EXPLICIT_SOURCE',support_mode='EXPLICIT_SPECIMEN_DESCRIPTION',later_specimens='IVPP V32322, articulated lateral-line scales')
            row['source_reason']+=' The authors describe a groove across the scale crown ending at a posterior notch, supporting the later over-scale/open-groove state. This is their published observation on added material, not new expert inspection by this audit.'
            row['source_locators'].extend([locator(xml,new_doi,'direct_state_description',section_id='Sec3',locator='Paragraph beginning The lateral line scales',figure='Fig.1a,b; Supplementary Fig.1',specimen='IVPP V32322'),locator(new_pdf,new_doi,'lateral_line_description',pdf_page=3,printed_page=3,figure='Fig.4a',specimen='IVPP V32322, Ll02')])
        elif o==43:
            row.update(later_state_support='SUPPORTED_BY_EXPLICIT_SOURCE',support_mode='EXPLICIT_SPECIMEN_DESCRIPTION',later_specimens='IVPP V32323.5, CT-scanned isolated scale')
            row['source_reason']+=' The authors reconstruct genuine vascular canals traversing the superficial and basal scale layers, supporting the basal-canal occurrence state. The enlarged specimen basis is explicit; no old-versus-new specimen equivalence is inferred.'
            row['source_locators'].append(locator(xml,new_doi,'direct_state_description',section_id='Sec4',locator='Paragraph beginning The 3D virtual model of the scale',figure='Fig.3c–h',specimen='IVPP V32323.5'))
        elif o==31:
            row['source_reason']+=' The new description records postmedian dorsal/ridge scales, but does not explicitly justify absence of homologous scute-like basal fulcra. Those structures are not equated here; the exact absent state remains unresolved.'
            row['source_locators'].append(locator(xml,new_doi,'relevant_but_insufficient_description',section_id='Sec3',locator='Paragraph beginning There are three postmedian dorsal plates/scales',figure='Fig.2f–i; Fig.1b',specimen='IVPP V32322'))
        elif o==644:
            row['source_reason']+=' The new material includes dermal pelvic plates, but no explicit description of multiple pelvic-fin basals was located in the article or supplementary text. A dermal plate or a phylogenetic expectation is insufficient to verify polybasal articulation.'
            row['source_locators'].append(locator(xml,new_doi,'relevant_but_insufficient_pelvic_material',section='Results and discussion; Figure4 legend',figure='Fig.4a–c',specimen='IVPP V32322'))
    if taxon=='Osorioichthys_marginis':
        extra=A/'gnathostome_sources/taverne1997.pdf'
        row.update(old_specimens='IRSNB P1.340, sole holotype in the shared cited source Taverne1997.',later_specimens='IRSNB P1.340 in the shared cited source; no cell-specific later specimen record.')
        row['source_reason']+=' Taverne describes the preserved maxilla/premaxillary region, but the retrieved passage does not explicitly exclude a palatal lamina; that stricter state requirement remains unresolved.'
        item=locator(extra,'','primary_anatomical_description',pdf_page=8,printed_page=64,figure='Figs.3,4,6,7,9,10',specimen='IRSNB P1.340');item['url']='https://biblio.naturalsciences.be/rbins-publications/bulletin-of-the-royal-belgian-institute-of-natural-sciences-earth-sciences/67-1997/irscnb_p4087_00eebfp_67_bulletin-4-red.pdf';row['source_locators'].append(item)
    rows.append(row)
(A/'gnathostome_documentary_reviews.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False)+'\n')
counts=dict(collections.Counter(r['documentary_review_status'] for r in rows));direct=[r['endpoint_id'] for r in rows if r['later_state_support']=='SUPPORTED_BY_EXPLICIT_SOURCE']
manifest={'status':'COMPLETED_DOCUMENTARY_REVIEW','selected':20,'counts':counts,'explicit_later_state_support_count':len(direct),'explicit_later_state_support_ids':direct,'fully_supported_all_gates':0,'independent_expert_reviews':0,'n_primary_matrix_token_pairs_independently_reparsed':20,'limitations':['Retrospective AI-assisted source reading with later labels exposed; no independent anatomical expert or specimen inspection.','Most cell-specific specimen continuity and scoring rationale remain unresolved.','Source additions were not selected by prediction performance; none of the fixed20 was replaced.','Support for a later statement does not prove biological correctness or unchanged endpoint scope.','An access/challenge page and a DOI resolver were retained as unsuccessful full-text retrievals, not article evidence.'],'hashes':{'selection':sha(A/'selected_endpoints.json'),'plan':sha(A/'REVIEW_PLAN.json'),'reviews':sha(A/'gnathostome_documentary_reviews.json'),'script':sha(Path(__file__))}}
(A/'gnathostome_review_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest,indent=2))
