"""Record a documentary reading of the frozen dinosaur sample, not anatomical validation."""
import csv,hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
PUB=ROOT/'public_validation/data/raw/dinosaurs'
OLDPDF=PUB/'baron2017_publisher_supplement.pdf'
NEWPDF=OUT/'langer2017_supplement.pdf'
OLDURL='https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fnature21700/MediaObjects/41586_2017_BFnature21700_MOESM140_ESM.pdf'
NEWURL=json.loads((OUT/'langer2017_supplement.provenance.json').read_text())['url']
SPEC={
 'Eodromaeus':('PVSJ 560–562','PVSJ 560 (holotype)',15,4,'The later named specimen is within the explicitly listed earlier specimen set.'),
 'Sanjuansaurus':('PVSJ 605','PVSJ 605 (holotype)',16,11,'The same individual specimen is explicitly listed in both source tables.'),
 'Syntarsus':('MCZ 9175; USNM 442404; Rowe (1989)','MNA V2623 (holotype)',16,13,'Listed catalogue numbers differ. Whether older references or casts cover the same original individual was not independently established; no specimen expansion or continuity is assumed.'),
 'Panphagia':('PVSJ 874','PVSJ 874 (holotype)',16,10,'The same individual specimen is explicitly listed in both source tables.'),
 'Scelidosaurus':('NHMUK PV R1111, R6704; BRSMG CE12785','NHMUK PV 1111 (lectotype), R6704',16,12,'The lists overlap, but the R1111/1111 identifier discrepancy and per-character specimen basis were not resolved.'),
 'Asilisaurus':('Nesbitt et al. (2010); no specimen numbers in the earlier source table','NMT RB 9, 10–16, 18–21, 34, 101, 110–121, 123–130, 132–133, 135–136, 138–146, 817',15,1,'The later list is explicit; the earlier cited publication\'s complete material basis was not independently reconstructed.'),
 'Euparkeria':('Ewer (1965); Gower and Weber (1998); no specimen numbers in earlier source table','GPIT 1681; SAM-PK 5867, 5883, 6047–6050, 7659, 7696, 7700, 7704, 7706, 7707, 7709, 7710, 7712, 10671, 13664–13667; SAM-PK-K8050, K8051, K8309; UMZC T692',15,5,'The earlier material basis cannot be inferred from citation names alone.'),
 'Sacisaurus':('Ferigolo and Langer (2007); no specimen numbers in earlier table','All specimens mentioned in Langer and Ferigolo (2013), according to the revising authors',16,11,'The two references differ; their complete specimen lists were not independently compared.'),
 'Pampadromaeus':('Cabreira et al. (2011); no specimen numbers in earlier table','ULBRA-PVT016 (holotype)',16,9,'The old table cites a description but does not itself name an individual specimen.'),
 'Silesaurus':('ZPAL Ab III/361; Dzik (2003)','ZPAL Ab111/361, 362, 363, 364, 403, 404, 406, 411, 413, 415, 423, 432, 452, 457, 460, 461, 837, 907, 1216, 1218, 1228, 1271, 1272, 1884 and further material as listed in the source',16,13,'The newer explicit list is broader than the single identifier printed earlier, but older cited literature may include more material; full specimen continuity is unresolved.'),
 'Lophostropheus':('Ezcurra and Cuny (2007); no specimen number in earlier table','Unnumbered University of Caen specimen',15,8,'An unnumbered specimen and literature citation do not by themselves prove specimen continuity.'),
 'Hexinlusaurus':('ZDM 6001–2','ZDM T6001, holotype, scored from photographs',15,6,'The source identifiers appear related but the T prefix difference and exact material equivalence were not independently resolved.'),
 'Eoraptor':('PVSJ 512','PVSJ 512 (holotype)',15,4,'The same individual specimen is explicitly listed in both source tables.'),
 'Laquintasaura':('Multiple MBLUZ specimens; Barrett et al. (2014)','MBLUZ P1434, P1515 and various unnumbered IVIC specimens',15,7,'The older list is not individually enumerated and the later list includes unnumbered IVIC material; unchanged specimen composition is unresolved.'),
 'Efraasia':('Galton (1973); Yates (2003); no specimen numbers in earlier table','SMNS 12684, 12667',15,4,'The earlier literature was not independently used to reconstruct the complete specimen set.')
}
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    data=json.loads((ROOT/'public_validation/data/processed/dinosaurs.json').read_text())
    maps={x['new_taxon']:x for x in data['taxon_map']}
    oldpages=(PUB/'baron2017_publisher_supplement.txt').read_text().split('\f')
    newtext=(OUT/'langer2017_supplement.txt').read_text()
    rows=[r for r in json.loads((OUT/'selected_endpoints.json').read_text()) if r['history']=='dinosaurs']
    conflicts=[]
    for r in rows:
        t=r['original_taxon_label'];nt='Syntarsus_kayentakatae' if t=='Syntarsus' else t
        ot=maps[nt]['old_taxon'];i=r['old_character_index_zero_based'];j=r['new_character_index_zero_based']
        a=data['old']['matrix'][data['old']['taxa'].index(ot)][i]
        b=data['new']['matrix'][data['new']['taxa'].index(nt)][j]
        assert a in {'?','-'} and b==r['later_state'],(r,a,b)
        odpage=next(k+1 for k,p in enumerate(oldpages) if k+1>=16 and re.search(r'^\s*'+str(i+1)+r'\.\s+',p,re.M))
        os,ns,op,np,scopenote=SPEC[t]
        special=(t=='Laquintasaura' and i==365)
        r.update(documentary_review_status='REVIEWED',original_token=a,source_later_token=b,
            source_old_taxon=ot,source_new_taxon=nt,taxon_alias_evidence=maps[nt]['evidence'],
            token_history=('DOCUMENTED_CHANGE_OR_CONFLICT' if a=='-' else 'SUPPORTED_BY_EXPLICIT_SOURCE'),
            character_correspondence=('SUPPORTED_BY_EXPLICIT_SOURCE' if i==96 else 'UNRESOLVED'),
            taxon_scope='UNRESOLVED',
            specimen_continuity=('SUPPORTED_BY_EXPLICIT_SOURCE' if t in {'Eodromaeus','Sanjuansaurus','Panphagia','Eoraptor'} else 'UNRESOLVED'),
            old_specimens=os,later_specimens=ns,
            later_state_support=('DOCUMENTED_CHANGE_OR_CONFLICT' if special else 'SUPPORTED_BY_EXPLICIT_SOURCE'),
            support_mode=('TENTATIVE_OR_CONTRADICTORY' if special else 'SOURCE_SCORING_RATIONALE'),
            exact_anatomical_state_reobserved=False,
            definition_check_depth=('Original and later printed principal state clauses agree.' if i==96 else 'Read the original printed state definition. The later supplement identifies these as score modifications to the original matrix but does not reprint this sampled definition; a complete independent semantic check remains unresolved.'),
            expert_review_status='NOT_PERFORMED')
        note='The later taxon-specific modification list explicitly records the published state '+b+' for this character and identifies a specimen or literature basis. This is a published scoring assertion, not independent observation of that anatomical state. '+scopenote
        if a=='-':note+=' The original printed token is a dash, normalised to unknown in the historical archive; its biological applicability has not been inferred from the symbol.'
        if special:
            note='The later matrix retains state 0, but the published modification list on p. 7 gives state 2 for character 366. The earlier definition on p. 50 defines only states 0 and 1. The embedded later PDF matrix on pp. 18–19 also gives state 0. This is an unresolved source inconsistency; retain frozen label 0 and do not infer an alternative anatomical state or its cause. '+scopenote
            conflicts.append({'endpoint_id':r['endpoint_id'],'history':'dinosaurs','taxon':t,'character_one_based':366,'old_token':a,'frozen_later_state':b,'modification_list_state':'2','doi':'10.1038/nature24011','pdf_page':7,'short_quote':'366: ? >2','rule':'Exclude this documented source-conflict endpoint in a separately labelled post hoc sensitivity; no label correction.'})
        r['source_reason']=note
        r['source_locators']=[
            {'doi':'10.1038/nature21700','url':OLDURL,'local_file':str(OLDPDF.relative_to(ROOT)),'sha256':digest(OLDPDF),'pdf_page':odpage,'character':i+1,'checked_scope':'Original principal state definition'},
            {'doi':'10.1038/nature21700','url':OLDURL,'local_file':str(OLDPDF.relative_to(ROOT)),'sha256':digest(OLDPDF),'pdf_page':op,'specimen':os,'checked_scope':'Original taxon source table'},
            {'doi':'10.1038/nature24011','url':NEWURL,'local_file':str(NEWPDF.relative_to(ROOT)),'sha256':digest(NEWPDF),'pdf_page':np,'character':j+1,'specimen':ns,'checked_scope':'Later exact taxon-specific score-modification entry and its stated material basis'}]
        if i==96:
            dp=next(k+1 for k,p in enumerate(newtext.split('\f')) if re.search(r'^\s*97\.\s+Supraoccipital',p,re.M))
            r['source_locators'].append({'doi':'10.1038/nature24011','url':NEWURL,'local_file':str(NEWPDF.relative_to(ROOT)),'sha256':digest(NEWPDF),'pdf_page':dp,'character':97,'checked_scope':'Later printed state clauses; no Hexinlusaurus-specific anatomical description in this character discussion'})
        for side in ['old','new']:
            filename='baron_original.nex' if side=='old' else 'langer_original.nex'
            f=PUB/filename
            r['source_locators'].append({'doi':'10.5281/zenodo.7979718','url':'https://zenodo.org/api/records/7979718/files/'+filename+'/content','local_file':str(f.relative_to(ROOT)),'sha256':digest(f),'character':i+1,'checked_scope':side+' archived numerical row; old printed dash restored from publisher PDF by verified public parser'})
    (OUT/'dinosaur_documentary_reviews.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
    (OUT/'dinosaur_source_conflicts.json').write_text(json.dumps(conflicts,ensure_ascii=False,indent=2)+'\n')
    with (OUT/'dinosaur_source_conflict_keys.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(conflicts[0]));w.writeheader();w.writerows(conflicts)
    print('Reviewed',len(rows),'source conflicts',len(conflicts))
if __name__=='__main__':main()
