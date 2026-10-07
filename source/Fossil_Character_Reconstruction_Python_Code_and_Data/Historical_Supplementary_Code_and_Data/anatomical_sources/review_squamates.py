#!/usr/bin/env python3
"""Rebuild a fixed-sample documentary audit without loading model outputs.

Primary evidence is the native publisher NEXUS and supplementary descriptions.
This is an AI-assisted documentary review, not independent anatomical expertise.
"""
from pathlib import Path
import collections, hashlib, json, re, subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'extension_work/anatomy'
RAW = ROOT / 'public_validation/data/raw/squamates'
OLD = RAW / '41586_2018_93_MOESM3_ESM.txt'
NEW = RAW / 'garberoglio2019_supplementaryFiles/aax5833_Data_file_S4.nxs'
OLDPDF = RAW / '41586_2018_93_MOESM1_ESM.pdf'
NEWPDF = RAW / 'garberoglio2019_supplementaryFiles/aax5833_SM.pdf'
XML = RAW / 'garberoglio2019_europepmc.xml'
OLDURL = 'https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-018-0093-3/MediaObjects/'
NEWURL = 'https://www.ebi.ac.uk/europepmc/webservices/rest/PMC6867888/supplementaryFiles'
YES='SUPPORTED_BY_EXPLICIT_SOURCE'; NO='UNRESOLVED'; CHANGE='DOCUMENTED_CHANGE_OR_CONFLICT'

# Character number: (support gate, mode, short source-based assessment,
#                    later PDF pages, specimen, anatomy section, state definition summary)
ASSESSMENTS = {
95:(NO,'MATRIX_ENTRY_ONLY','The vomer description identifies a dorsal excavation and partial enclosure, but does not explicitly establish the coded ventral convexity in cross-section. Figure S1 J–K is not a section through the scored region; the exact state remains unresolved.',[19,30],'MPCA 500','Vomer; Fig. S1 J–K','Vomer cross-section: flat 0; ventrally convex 1.'),
199:(YES,'EXPLICIT_SPECIMEN_DESCRIPTION','The coronoid description identifies a long anterior process overlapping the dentary medially, providing documentary support for presence of the anteromedial process. Anatomical homology has not been independently validated.',[27],'MPCA 500; MPCA 591','Coronoid','Coronoid anteromedial process: present 0; absent 1.'),
142:(NO,'MATRIX_ENTRY_ONLY','A dorsal alar process and its parietal contact are described. The source does not explicitly equate that process with the thin alar crest defined in the older character; the exact character-level identification remains unresolved.',[23,24,30],'MPCA 500; MPCA 418; MPCA 591','Prootic; Fig. S1 H–I','Prootic alar crest: absent 0; present 1; older remarks specify an anterior bony flange.'),
204:(NO,'MATRIX_ENTRY_ONLY','The described teeth are curved and lack carinae, but absence of mesiodistal accessory cusps is not explicitly documented for the scored posterior dentition. Lack of carinae is not treated as proof of absence of cusps.',[27],'MPCA 419; MPCA 591','Dentary','Posterior tooth mesiodistal accessory cusps: absent 0; present 1.'),
57:(NO,'MATRIX_ENTRY_ONLY','The postfrontal is described as triradiate and illustrated separately, but the specific distal process used in the old coding is not expressly identified. The figure alone was not used to infer anatomical homology.',[21,30],'MPCA 500','Postfrontal; Fig. S1 L–N','Postfrontal distal process: present 0; absent 1.'),
22:(NO,'MATRIX_ENTRY_ONLY','Only the right nasal is described and illustrated. This is compatible with paired nasals, but no explicit fusion assessment of both nasals is provided; the coded condition remains unresolved.',[18,30],'MPCA 500','Nasals; Fig. S1 J–K','Nasals: paired 0; fused 1.'),
33:(NO,'TENTATIVE_OR_CONTRADICTORY','The matrix codes crest absence, whereas the description discusses a thick preorbital ridge near the prefrontal. Whether this is the dorsal orbital crest defined by the old character requires anatomical review; no equivalence or coding error is asserted.',[19,30],'MPCA 500','Prefrontal; Fig. S1 L–N','Prefrontal dorsal orbital crest: absent 0; present 1.'),
113:(YES,'EXPLICIT_SPECIMEN_DESCRIPTION','A separate ectopterygoid is explicitly described in MPCA 500, with its maxillary and pterygoid contacts, supporting presence.',[20,30],'MPCA 500','Ectopterygoid; Fig. S1 A–B and L–N','Ectopterygoid: absent 0; present 1.'),
151:(YES,'EXPLICIT_SPECIMEN_DESCRIPTION','The otooccipital is explicitly identified as fused exoccipital plus opisthotic, while the prootic is described as a separate ossification. These descriptions support state 1 rather than the state including prootic fusion.',[23,25],'MPCA 500 and other described skulls','Prootic; Otooccipitals','Exoccipital fusion: unfused 0; opisthotic only 1; opisthotic and prootic 2. Older remarks restrict assessment to adults.'),
137:(YES,'EXPLICIT_SPECIMEN_DESCRIPTION','The CT-based description explicitly identifies a low dorsum sellae behind the sella turcica in MPCA 500, supporting presence.',[23],'MPCA 500','Parabasisphenoid','Basisphenoid dorsum sellae: absent 0; present 1.'),
212:(YES,'EXPLICIT_SPECIMEN_DESCRIPTION','MPCA 500 dentary alveoli are explicitly described as separated on three sides by interdental ridges, supporting state 1. The supplement also calls this a correction from 2 to 1, but the archived publisher old Najash cell is actually ?. That narrative discrepancy is retained without changing the frozen endpoint or inferring a definition change.',[17,18,27],'MPCA 500','Data Set 2; Dentary','Posterior dentary tooth enclosure: labial wall only 0; three sides 1; four sides 2; lingual and labial walls 3; inapplicable for apical placement.'),
143:(NO,'MATRIX_ENTRY_ONLY','The prootic description details the trigeminal margins and alar process, but does not explicitly assess the separate supratrigeminal process required by this character. Absence is not inferred from silence.',[23,24,30],'MPCA 500; MPCA 418; MPCA 591','Prootic; Fig. S1 H–I','Prootic supratrigeminal process: absent 0; present 1.'),
166:(CHANGE,'TENTATIVE_OR_CONTRADICTORY','The matrix codes a flat symphysial area (0), whereas the description of MPCA 500 and 380 identifies a rounded symphysial process and probable mobile joint. This is a documentary tension with the older flat-versus-convex definition, not a proven coding error or a new recoding.',[27],'MPCA 500; MPCA 380','Dentary','Dentary symphysial area: flat 0; convex 1; older remarks associate rounded surfaces with mobility.'),
202:(NO,'MATRIX_ENTRY_ONLY','The tooth description discusses curvature and lack of carinae, not labial apical striations. The particular absence state has only matrix-level documentation in the material reviewed.',[27],'MPCA 419; MPCA 591','Dentary','Labial tooth-crown apical striations: absent 0; present 1.'),
224:(YES,'EXPLICIT_SPECIMEN_DESCRIPTION','The cervical description explicitly identifies two unfused axis intercentra and their articular locations, supporting intercentrum presence.',[29],'MPCA 480; MPCA 500; MPCA 581; MPCA 564; MPCA 591','Cervical vertebrae','Axis intercentrum: present 0; absent 1.'),
63:(NO,'MATRIX_ENTRY_ONLY','The supratemporal description rules out a free posterior extension. The old character instead specifies a lateral temporal process; absence of one direction of extension is not used to establish absence of the other.',[25,26],'MPCA 480; MPCA 581; MPCA 500','Supratemporal','Supratemporal lateral temporal process: absent 0; present 1.'),
172:(NO,'TENTATIVE_OR_CONTRADICTORY','The later description names a dentary coronoid process but also describes the dorsal posterior process as tooth-bearing. The old character specifically concerns an edentulous posterior projection, so exact applicability and process identity remain unresolved.',[27,28],'MPCA 500; MPCA 591','Dentary; Compound bone','Dentary coronoid process: absent 0; present 1; older remarks specify an edentulous posterior projection.'),
59:(NO,'MATRIX_ENTRY_ONLY','The postfrontal is described at the frontoparietal suture. Its particular medial-margin position relative to the parietal is not explicitly stated using the four old alternatives; the lateral state remains unresolved.',[21,30],'MPCA 500','Postfrontal; Fig. S1 L–N','Postfrontal medial margin relative to parietal: ventral 0; dorsal 1; lateral 2; anterior 3.'),
2:(YES,'EXPLICIT_SPECIMEN_DESCRIPTION','A fragment of the premaxillary nasal process is explicitly described in MPCA 500 and identified in the CT reconstruction, supporting presence.',[18,30],'MPCA 500','Premaxilla; Fig. S1 J–K','Premaxillary nasal process: present 0; absent 1.'),
17:(YES,'EXPLICIT_SPECIMEN_DESCRIPTION','The maxilla description explicitly identifies a developed facial/ascending process and its prefrontal articulation; the old definition identifies this as the nasal/facial process, supporting presence.',[20,30],'MPCA 500','Maxilla; Fig. S1 L–N','Maxillary nasal/facial process: absent 0; present 1.'),
}

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def native(path):
    s=path.read_text()
    match=re.search(r'(?mi)^\s*Najash_rionegrina\s+([^\n]+)',s)
    assert match, path
    tokens=re.findall(r'\{[^}]*\}|\([^)]*\)|[^\s]',match.group(1))
    assert len(tokens)==347, (path,len(tokens))
    chars=re.search(r'CHARSTATELABELS\s+(.*?);',s,re.I|re.S).group(1)
    labels={int(k):v.strip() for k,v in re.findall(r'(\d+)\s+(.*?)(?=,\s*\d+\s+|$)',chars,re.S)}
    assert len(labels)==347
    return tokens, labels

def locator(path, doi, url, scope, character=None, page=None, figure=None, specimen=None):
    return {'doi':doi,'url':url,'local_file':str(path.relative_to(ROOT)), 'sha256':sha(path),
        'pdf_page':page,'printed_page':str(page-1) if path==OLDPDF and page else None,
        'character':character,'figure':figure,'specimen':specimen,'checked_scope':scope}

def main():
    selection=json.loads((OUT/'selected_endpoints.json').read_text())
    rows=[r for r in selection if r['history']=='squamates']
    assert len(rows)==20 and {r['old_character_index_zero_based']+1 for r in rows}==set(ASSESSMENTS)
    old,oldlabels=native(OLD);new,newlabels=native(NEW)
    # Local text files preserve page breaks; no model-output files are opened.
    oldtext=subprocess.check_output(['pdftotext','-layout',str(OLDPDF),'-']).decode()
    oldpages=oldtext.split('\f')
    scopes=("The old supplement explicitly restricts scored Najash to holotype MPCA 390–398, 400, although referred material including MPCA 500 was observed. The later paper incorporates additional cranial specimens conservatively assigned to Najash sp. pending taxonomic review; the reused NEXUS species label therefore does not establish unchanged OTU or specimen scope.")
    result=[]
    for original in rows:
        r=dict(original);c=r['old_character_index_zero_based']+1
        assert r['original_taxon_label']=='Najash_rionegrina' and c==r['new_character_index_zero_based']+1
        assert old[c-1]=='?' and new[c-1]==r['later_state']
        assert oldlabels[c]==newlabels[c]
        target=ROOT/r['original_target_file'];assert sha(target)==r['original_target_file_sha256']
        gate,mode,reason,newpages,specimens,section,definition=ASSESSMENTS[c]
        matches=[i+1 for i,p in enumerate(oldpages) if re.search(rf'(?m)^[ \t]*{c}\.\s+',p)]
        assert len(matches)==1,(c,matches)
        loc=[locator(OLD,'10.1038/s41586-018-0093-3',OLDURL+OLD.name,'Native MATRIX row Najash_rionegrina; exact old token ?.',c),
             locator(NEW,'10.1126/sciadv.aax5833',NEWURL,'Data file S4 native MATRIX row; exact later singleton '+new[c-1]+'.',c),
             locator(OLDPDF,'10.1038/s41586-018-0093-3',OLDURL+OLDPDF.name,'Old character definition, state meanings and applicability.',c,matches[0]),
             locator(OLDPDF,'10.1038/s41586-018-0093-3',OLDURL+OLDPDF.name,'Najash holotype inventory (continues to the explicit scoring restriction on PDF page 49).',c,48,specimen='MPCA 390–398, 400'),
             locator(OLDPDF,'10.1038/s41586-018-0093-3',OLDURL+OLDPDF.name,'Explicit restriction to holotype scoring despite observed referred material.',c,49,specimen='MPCA 390–398, 400'),
             locator(NEWPDF,'10.1126/sciadv.aax5833',NEWURL,'Data Set 2 explicitly adopts Simões character construction and list; continues onto PDF page 18.',c,17),
             locator(XML,'10.1126/sciadv.aax5833','https://www.ebi.ac.uk/europepmc/webservices/rest/PMC6867888/fullTextXML','RESULTS: new specimens assigned to Najash sp. pending taxonomic revision, plus introduction reporting eight new skulls and three postcrania.',c,specimen='Additional Najash sp. specimens including MPCA 500')]
        for p in newpages:
            loc.append(locator(NEWPDF,'10.1126/sciadv.aax5833',NEWURL,section+'; inspected for the exact state, not merely matrix availability.',c,p,'S1' if p==30 else None,specimens))
        r.update(documentary_review_status='REVIEWED',token_history=YES,
          character_correspondence=YES,taxon_scope=CHANGE,specimen_continuity=CHANGE,
          later_state_support=gate,support_mode=mode,
          old_specimens='Scoring explicitly restricted to holotype MPCA 390–398, 400. Referred specimens observed but excluded from old scoring.',
          later_specimens=specimens+'; expanded Najash sp. sample, not demonstrated to be the unchanged holotype OTU.',
          source_locators=loc,source_reason=reason+' Expanded specimen and taxonomic scope applies independently of this state-support assessment.',
          expert_review_status='NOT_PERFORMED',original_missing_token=old[c-1],checked_later_token=new[c-1],
          character_number_one_based=c,character_label=oldlabels[c].strip("'"),state_definition_summary=definition,
          character_correspondence_note='The native labels agree and the later supplement explicitly adopts the old character list. No selected definition or state-number redefinition is established. Matching definitions do not establish biological continuity.',
          taxon_scope_note=scopes,specimen_continuity_note=scopes,
          review_label_exposure='Later published states visible; no model predictions, probabilities, errors or costs accessed.',
          notes=['AI-assisted documentary review; no independent anatomical expert assessment.',
          'UNRESOLVED records insufficient exact character-level evidence in the inspected material, not a finding that the state is wrong.',
          'PDF page numbers include the cover; old printed page numbers are one lower. Later supplement has no printed pagination.'])
        if c==212:
            r['source_narrative_discrepancy']='Later supplement reports 2→1 for Najash, but published old native token is ?. Both raw matrices and frozen endpoint are preserved; this is not evidence of a state-definition change.'
        result.append(r)
    path=OUT/'squamate_documentary_reviews.json';path.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    manifest={'history':'squamates','selected_count':20,'reviewed_count':20,
       'later_state_support_counts':dict(collections.Counter(r['later_state_support'] for r in result)),
       'support_mode_counts':dict(collections.Counter(r['support_mode'] for r in result)),
       'old_native_missing_tokens':{'?':20},'taxon_scope_change_count':20,'specimen_scope_change_count':20,
       'expert_review_status':'NOT_PERFORMED','label_exposure':'Later published states visible; model outputs not read.',
       'raw_old_nchar':347,'raw_new_nchar':347,'same_selected_labels_count':20,
       'sources':[{'path':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in [OLD,NEW,OLDPDF,NEWPDF,XML]],
       'output_path':str(path.relative_to(ROOT)),'output_sha256':sha(path),
       'major_findings':['Old scoring restricted explicitly to holotype; later Najash sp. sample expands taxon/specimen scope.',
        'Character 212 narrative 2→1 conflicts with actual published old token ?, without demonstrated definition change.',
        'Character 166 matrix flat state and rounded symphysial description are in documentary tension; expert resolution needed.'],
       'limitations':['Only selected 20 of original163 endpoints reviewed. All are Najash; no generalisation beyond this fixed sample.',
        'Documentary support is not specimen-level expert validation; no fossil or CT reanalysis performed.',
        'No later state recoding or frozen endpoint removal performed.'],
       'visual_checks':['Old supplement PDF49 explicit scope restriction; later supplement PDF30 Fig.S1 read visually. Figures did not substitute for unresolved exact state definitions.']}
    (OUT/'squamate_documentary_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k.endswith('counts') or k.endswith('count')},indent=2))

if __name__=='__main__':main()
