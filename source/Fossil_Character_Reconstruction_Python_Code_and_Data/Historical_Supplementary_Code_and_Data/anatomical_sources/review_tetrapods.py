"""Materialise manually read Appendix S1 findings for the frozen 20-cell sample."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
RAW=ROOT/'public_validation/data/raw/tetrapods'
DATA=json.loads((ROOT/'public_validation/data/processed/tetrapods.json').read_text())

# These are source-reading notes, not model outputs or specimen reobservations.
NOTES={
'Leptorophus': ('SOURCE_SCORING_RATIONALE','The authors explicitly assign state 1 and cite Werneburg (2007a); no specimen number is supplied in this character section.','','Werneburg (2007a)'),
'Whatcheeria': ('SOURCE_SCORING_RATIONALE','State 0 is explicitly justified by a premaxillary contribution to the choanal margin, citing Lombard and Bolt (1995).','','Lombard and Bolt (1995)'),
'Doleserpeton': ('EXPLICIT_SPECIMEN_DESCRIPTION','State 1 is explicitly reported from the authors\' observation of AMNH 29470 and Sigurdsen and Bolt (2010). The present audit has not inspected that specimen.','AMNH 29470','Sigurdsen and Bolt (2010)'),
'Edops': ('REINTERPRETATION','The posterior crest is counted as state 1 by extension of the authors\' interpretation of Archegosaurus. MCZ 7162 and MCZ 7259 are identified as observed Edops specimens.','MCZ 7162; MCZ 7259','Witzmann and Schoch (2006b), fig. 9B, is the comparative Archegosaurus reference'),
'Triadobatrachus': ('SOURCE_SCORING_RATIONALE','The authors explicitly list state 1 and cite Ascarrunz et al. (2016), 3D model 1. That model has not been independently reinterpreted in this audit.','','Ascarrunz et al. (2016), 3D model 1'),
'Lethiscus': ('SOURCE_SCORING_RATIONALE','State 0 is explicitly listed with Anderson, Carroll and Rowe (2003) and Pardo et al. (2017). No endpoint-specific specimen identifier is given in this section.','','Anderson, Carroll and Rowe (2003); Pardo et al. (2017)'),
'Panderichthys': ('SOURCE_SCORING_RATIONALE','State 0 is explicitly listed and attributed to Boisvert (2005).','','Boisvert (2005)'),
'Ventastega': ('SOURCE_SCORING_RATIONALE','State 0 is explicitly listed and attributed to Ahlberg et al. (2008).','','Ahlberg et al. (2008)'),
'Solenodonsaurus': ('SOURCE_SCORING_RATIONALE','The dentary is explicitly scored as toothed, state 0, citing Danto, Witzmann and Müller (2012).','','Danto, Witzmann and Müller (2012)'),
'Cardiocephalus': ('SOURCE_SCORING_RATIONALE','The authors explicitly assign state 1 and locate the cited evidence in Carroll and Gaskill (1978), figure 31. The older illustration has not been independently reinterpreted.','','Carroll and Gaskill (1978), fig. 31'),
'Batropetes': ('SOURCE_SCORING_RATIONALE','State 1 is explicitly listed with Glienke (2013). The article interprets most OTUs at genus level and discusses Batropetes plesiomorphic morphotypes; exact earlier-versus-later specimen continuity remains unresolved.','','Glienke (2013); article section Treatment of OTUs'),
'Hyloplesion': ('SOURCE_SCORING_RATIONALE','The authors explain that complete tails preserve state 1 and cite Carroll and Gaskill (1978). No individual specimen number is supplied here.','','Carroll and Gaskill (1978)'),
'Isodectes': ('EXPLICIT_SPECIMEN_DESCRIPTION','The authors explicitly state that they observed state 0 in CM 81512. The older matrix label has a numerical prefix, 332_tru_ver_30, whereas the revision uses tru_ver_30.','CM 81512','Appendix S1 character 275'),
'Schoenfelderpeton': ('TENTATIVE_OR_CONTRADICTORY','The source qualifies the state-1 assignment as apparent and cites Boy (1986), page 154, figure 15b. This is qualified documentary support, not an unambiguous anatomical confirmation.','','Boy (1986), p. 154, fig. 15b'),
'Tseajaia': ('SOURCE_SCORING_RATIONALE','State 1 is explicitly listed for the absence of the specified ectopterygoid tooth row, citing Moss (1972), page 12.','','Moss (1972), p. 12'),
'Oestocephalus': ('MATRIX_ENTRY_ONLY','The deposited revised matrix assigns state 0. Appendix S1 character 255 gives the definition and discusses other taxa but supplies no Oestocephalus-specific scoring rationale. Its anatomical support remains unresolved in the sources reviewed.','','Appendix S1 character 255; Supplement 3 matrix row Oestocephalus'),
'Cochleosaurus': ('SOURCE_SCORING_RATIONALE','State 0 is explicitly listed and attributed to Sequeira (2009).','','Sequeira (2009)'),
'Ossinodus': ('SOURCE_SCORING_RATIONALE','The authors explicitly assign state 0 for the humeral shaft and cite Bishop (2014). No specimen number or figure is supplied in this character section.','','Bishop (2014)'),
'Platyrhinops': ('EXPLICIT_SPECIMEN_DESCRIPTION','State 1 is explicitly attributed to the authors\' observation of AMNH 2002. The present audit has not inspected that specimen.','AMNH 2002','Appendix S1 character 242'),
'Acheloma': ('SOURCE_SCORING_RATIONALE','State 1 is explicitly listed and attributed to Polley and Reisz (2011). The species/specimen scope of the original endpoint is not established by the shared genus label.','','Polley and Reisz (2011)')
}

def main():
    samples=[r for r in json.loads((OUT/'selected_endpoints.json').read_text()) if r['history']=='tetrapods']
    pages=(RAW/'appendix_s1.txt').read_text().split('\f')
    pdf=RAW/'peerj-07-5565-s001.pdf'
    pdfhash=hashlib.sha256(pdf.read_bytes()).hexdigest()
    for r in samples:
        t=r['original_taxon_label'];i=r['old_character_index_zero_based'];j=r['new_character_index_zero_based']
        a=DATA['old']['matrix'][DATA['old']['taxa'].index(t)][i]
        b=DATA['new']['matrix'][DATA['new']['taxa'].index(t)][j]
        assert a=='?' and b==r['later_state'],(r,a,b)
        p=next(k+1 for k,tx in enumerate(pages) if re.search(r'^\s*\d+\s+'+str(j+1)+r'\.\s+[A-Z]',tx,re.M))
        mode,note,spec,cited=NOTES[t]
        r.update(documentary_review_status='REVIEWED',token_history='SUPPORTED_BY_EXPLICIT_SOURCE',
                 character_correspondence='SUPPORTED_BY_EXPLICIT_SOURCE',
                 taxon_scope='UNRESOLVED',specimen_continuity='UNRESOLVED',
                 later_state_support=('UNRESOLVED' if mode in {'TENTATIVE_OR_CONTRADICTORY','MATRIX_ENTRY_ONLY'} else 'SUPPORTED_BY_EXPLICIT_SOURCE'),
                 support_mode=mode,later_specimens=spec,old_specimens='Not established from the available original-source record',
                 original_token=a,source_later_token=b,
                 old_character_label=DATA['old']['characters'][i],new_character_label=DATA['new']['characters'][j],
                 definition_check_depth='Revision-authors\' preservation statement on Appendix S1 p. 1 plus the sampled character section. Original RC07 full anatomical definition was not independently retrieved here; this is documentary support from the revising authors, not a fresh anatomical equivalence assessment.',
                 source_reason=note+' Exact old/new matrix tokens were checked. Shared taxon names do not establish biological or specimen continuity. No specimens or cited underlying anatomical illustrations were independently assessed.',
                 expert_review_status='NOT_PERFORMED')
        r['source_locators']=[{'doi':'10.7717/peerj.5565/supp-1',
            'url':'https://pmc.ncbi.nlm.nih.gov/articles/instance/6322490/bin/peerj-07-5565-s001.pdf',
            'local_file':str(pdf.relative_to(ROOT)),'sha256':pdfhash,'pdf_page':p,
            'printed_page':p,'character':j+1,'figure':cited,'specimen':spec,
            'checked_scope':'Read the published character definition and complete sampled character discussion; adjoining deleted-character discussions excluded.'}]
        for side,filename,doisuff,idx in [('old','peerj-07-5565-s007.nex','7',i),('new','peerj-07-5565-s003.nex','3',j)]:
            f=RAW/filename
            r['source_locators'].append({'doi':'10.7717/peerj.5565/supp-'+doisuff,
                'url':'https://www.ebi.ac.uk/europepmc/webservices/rest/PMC6322490/supplementaryFiles',
                'local_file':str(f.relative_to(ROOT)),'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),
                'taxon_row':t,'character':idx+1,'checked_scope':side+' matrix token and character label'})
    (OUT/'tetrapod_documentary_reviews.json').write_text(json.dumps(samples,indent=2,ensure_ascii=False)+'\n')
    from collections import Counter
    print(dict(Counter(r['later_state_support'] for r in samples)))
if __name__=='__main__':main()
