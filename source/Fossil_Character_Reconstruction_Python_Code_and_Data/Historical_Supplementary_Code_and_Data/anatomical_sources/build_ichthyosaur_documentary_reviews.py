"""Assemble a source-only review of the 20 frozen ichthyosaur endpoints.

No prediction, probability, loss, cost or method-performance file is read.
Interpretive notes below were written after reading the named source passages.
"""
import copy
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

from docx import Document

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "public_validation/src"))
from acquire_ichthyosaurs import parse_nexus

OUT = ROOT / "extension_work/anatomy"
RAW = ROOT / "public_validation/data/raw/ichthyosaurs"
OLD_MATRIX = RAW / "tjsp_a_1394922_sm0622.nex"
NEW_MATRIX = RAW / "Sachicarum_APPENDIX3.nex"
OLD_DOC = RAW / "tjsp_a_1394922_sm0623.pdf"
NEW_DOC = RAW / "Supplementary_Data_R1.docx"
ARTICLE = ROOT / "ichthyosaur_source_evidence/maxwell2019_article.pdf"
HENDERSON = OUT / "ichthyosaur_sources/binall15.pdf"
OLD_DOI = "10.1080/14772019.2017.1394922"
NEW_DOI = "10.1080/02724634.2019.1577875"
S = "SUPPORTED_BY_EXPLICIT_SOURCE"
U = "UNRESOLVED"
C = "DOCUMENTED_CHANGE_OR_CONFLICT"

URLS = {
    OLD_MATRIX: "https://ndownloader.figshare.com/files/9960364",
    NEW_MATRIX: "https://ndownloader.figshare.com/files/14877068",
    OLD_DOC: "https://ndownloader.figshare.com/files/9960361",
    NEW_DOC: "https://ndownloader.figshare.com/files/14877071",
    ARTICLE: "https://doi.org/" + NEW_DOI,
    HENDERSON: "https://oops.uni-oldenburg.de/2571/1/binall15.pdf",
}
HASHES = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in URLS}

CHAR_PAGES = {5: 17, 14: 18, 40: 21, 41: 22, 54: 23, 73: 25,
              105: 28, 108: 28, 140: 31, 157: 33, 183: 35, 186: 35,
              197: 36, 211: 37, 216: 38, 223: 39, 224: 39, 241: 40,
              256: 42}
CHAR_PARAS = {5: 296, 14: 315, 40: 371, 41: 373, 54: 400, 73: 438,
              105: 502, 108: 508, 140: 573, 157: 610, 183: 664,
              186: 669, 197: 691, 211: 719, 216: 729, 223: 743,
              224: 745, 241: 780, 256: 811}

TAXA = {
    "Macgowania_janiceps": {
        "old_page": 13, "later_para": 188,
        "old_specimens": "McGowan (1996a); ROM 41992 (Henderson 2015 comparison); complete old inventory unresolved.",
        "later_specimens": "TMP 2009.121.1; skull cast and published postcranial description.",
        "scope": C,
        "reason": "The later scoring uses a different documented specimen, TMP 2009.121.1.",
    },
    "Mixosaurus_kuhnschneyderi": {
        "old_page": 14, "later_para": 166,
        "old_specimens": "Moon S1 cites Brinkmann (1998a) and Chen and Cheng (2009); no complete specimen inventory in the checked row.",
        "later_specimens": "Brinkmann (2004) is cited for character 157; specimen identifiers are not supplied in this scoring note.",
        "reason": "Appendix 1 links character 157 to Brinkmann (2004), but does not state the rib-facet condition. That underlying description was not directly checked here.",
    },
    "Leptonectes_solei": {
        "old_page": 13, "later_para": 218,
        "old_specimens": "McGowan (1993) and personal observation of BRSMG Ce 9856 are listed by Moon.",
        "later_specimens": "The referred specimen figured in McGowan (1993); its number is not supplied in the checked scoring note.",
        "reason": "Appendix 1 identifies a referred specimen as the basis of character 241 but gives no explicit metacarpal-row count. Whether this adds a specimen to the earlier scoring set is unresolved.",
    },
    "Leptonectes_moorei": {
        "old_page": 13, "later_para": 220,
        "old_specimens": "McGowan and Milner (1999) is the listed source; the checked table does not enumerate specimens.",
        "later_specimens": "Holotype, personally observed by the revision authors; no identifier in this scoring note.",
        "reason": "The revision explicitly associates character 5 with observation of the holotype, but does not state the subnarial-process proportion. The original description was not directly re-examined here.",
    },
    "Caypullisaurus_bonapartei": {
        "old_page": 12, "later_para": 246,
        "old_specimens": "Fernandez (1997, 1998, 2001, 2007a) are listed by Moon; no complete specimen inventory in that row.",
        "later_specimens": "Character 256 follows the interpretation of Delsett et al. (2017); no specimen identifier is given in the scoring note.",
        "mode": "REINTERPRETATION",
        "reason": "Appendix 1 attributes characters 256–257 to a revised interpretation, without stating the precise fusion condition or specimen for this entry. The later matrix alone supplies state 1.",
    },
    "Mixosaurus_cornalianus": {
        "old_page": 14, "later_para": 164,
        "old_specimens": "Moon lists Repossi, Wiman, Huene, Motani, Callaway, and Maisch and Matzke; no complete specimen inventory in the checked row.",
        "later_specimens": "Maisch et al. (2006) and personal observations; no specimen identifier for character 108 in the scoring note.",
        "reason": "Appendix 1 names character 108 and its source, but does not explicitly describe the paraoccipital-process shape for this taxon. Direct anatomical support for the exact state remains unresolved.",
    },
    "Eurhinosaurus_longirostris": {
        "old_page": 12, "later_para": 216,
        "old_specimens": "Moon lists literature sources, MFNB MB.R. 4004 and specimens in SMNS.",
        "later_specimens": "Specimens at SMNS and the Urweltmuseum Fischer; individual identifiers absent from this scoring note.",
        "reason": "The revision identifies character 140 among the palatal scores based on museum specimens, but does not explicitly describe presence or absence of pterygoidal teeth. Overlap of old and new specimen sets is unresolved.",
    },
    "Stenopterygius_uniter": {
        "old_page": 15, "later_para": 232,
        "old_specimens": "Moon lists McGowan (1979), Maisch (2008), MFNB MB.R. 1746 and specimens in MHH and SMNS.",
        "later_specimens": "Specimens at Urweltmuseum Hauff, GPIT and PMU; individual identifiers absent from this scoring note.",
        "reason": "Appendix 1 names the selected character among specimen-based scoring changes, but does not state the exact anatomical condition. The complete earlier and later specimen sets cannot be equated from these collection lists.",
    },
    "Platypterygius_hauthali": {
        "old_page": 14, "later_para": 278,
        "old_specimens": "Moon cites Huene (1927) and Fernandez and Aguirre-Urreta (2005), without a per-character specimen inventory.",
        "later_specimens": "Pardo-Perez et al. (2012) is added for forelimb scoring; the exact specimen for character 224 is not named in the revision note.",
        "reason": "The revision identifies a source for additional forelimb scores, but supplies neither a character-224-specific specimen nor its proximal/distal ulna-width relation. The article also recognises Argentine and Chilean material; exact specimen continuity remains unresolved.",
    },
    "Ichthyosaurus_conybeari": {
        "old_page": 13, "later_para": 208,
        "old_specimens": "McGowan (1974b) and Massare and Lomax (2016) are listed; no complete specimen inventory in the checked row.",
        "later_specimens": "SMNS 58275 for basicranial characters.",
        "reason": "Appendix 1 identifies SMNS 58275 as the basis of basicranial scores, but does not describe the cultriform-process base corresponding to character 105. The exact later state is explicit only in the matrix checked here.",
    },
    "Platypterygius_sachicarum": {
        "old_page": 15,
        "old_specimens": "Moon cites Paramo (1997); Maxwell identifies that material as the skull holotype DON-19671.",
        "later_specimens": "Referred specimen CIP-GA-01042014; the later account distinguishes it from holotype DON-19671.",
        "scope": C,
        "reason": "The later article explicitly adds CIP-GA-01042014 and describes the first postcranial material assigned to the species.",
    },
}


def locator(path, char, checked, pdf_page=None, printed_page=None,
            figure=None, specimen=None, paragraph=None):
    doi = OLD_DOI if path in (OLD_MATRIX, OLD_DOC) else NEW_DOI
    if path == HENDERSON:
        doi = None
    return {"doi": doi, "url": URLS[path],
            "local_file": str(path.relative_to(ROOT)), "sha256": HASHES[path],
            "pdf_page": pdf_page, "printed_page": printed_page,
            "character": char, "figure": figure, "specimen": specimen,
            "docx_paragraph_zero_based": paragraph, "checked_scope": checked}


def main():
    selected = [copy.deepcopy(x) for x in json.loads((OUT / "selected_endpoints.json").read_text())
                if x["history"] == "ichthyosaurs"]
    assert len(selected) == 20
    old, new = parse_nexus(OLD_MATRIX), parse_nexus(NEW_MATRIX)
    doc = Document(NEW_DOC)
    for row in selected:
        tax = row["original_taxon_label"]
        j, k = row["old_character_index_zero_based"], row["new_character_index_zero_based"]
        char = j + 1
        ot = old["matrix"][old["taxa"].index(tax)][j]
        nt = new["matrix"][new["taxa"].index(tax)][k]
        assert ot == "?" and nt == row["later_state"]
        assert j == k
        assert doc.paragraphs[CHAR_PARAS[char]].text.startswith(f"Character {char} ")
        meta = TAXA[tax]
        row.update({
            "documentary_review_status": "REVIEWED",
            "token_history": S,
            "character_correspondence": S,
            "taxon_scope": meta.get("scope", U),
            "specimen_continuity": meta.get("scope", U),
            "later_state_support": U,
            "support_mode": meta.get("mode", "SOURCE_SCORING_RATIONALE"),
            "old_specimens": meta["old_specimens"],
            "later_specimens": meta["later_specimens"],
            "original_missing_token": ot,
            "checked_later_token": nt,
            "character_number_one_based": char,
            "character_correspondence_note": "The anatomical definition and numbered state clauses agree in Moon S1 and Maxwell Appendix 2 for this character; the correspondence is not based solely on a matching number.",
            "taxon_scope_note": meta["reason"] if "scope" in meta else "The matrix labels match, but the checked sources do not establish an unchanged operational specimen set.",
            "expert_review_status": "NOT_PERFORMED",
            "review_label_exposure": "Later published state visible; no model predictions, errors, probabilities or costs used in this source review.",
        })
        row["source_locators"] = [
            locator(OLD_MATRIX, char, f"Native MATRIX row {tax}: exact original token '?' verified by parsing compound symbols as single cells."),
            locator(NEW_MATRIX, char, f"Appendix 3 MATRIX row {tax}: exact later singleton {nt} verified."),
            locator(OLD_DOC, char, "Original character definition and state clauses.", CHAR_PAGES[char], str(CHAR_PAGES[char])),
            locator(NEW_DOC, char, "Appendix 2 character definition and numbered states.", paragraph=CHAR_PARAS[char]),
            locator(OLD_DOC, char, "Table S1 taxon-level scoring sources; not a complete per-cell specimen inventory.", meta["old_page"], str(meta["old_page"])),
        ]
        reason = meta["reason"]
        if "later_para" in meta:
            row["source_locators"].append(locator(NEW_DOC, char,
                "Appendix 1 taxon-specific scoring rationale; source citation alone is not a direct description of the selected state.",
                paragraph=meta["later_para"]))
        if tax == "Macgowania_janiceps":
            row["support_mode"] = "EXPANDED_SPECIMEN_SCOPE"
            row["source_locators"].append(locator(HENDERSON, char,
                "Table 2 identifies the distinct comparison specimens.", 128, "128",
                specimen="ROM 41992; TMP 2009.121.1"))
            if char in (183, 186):
                row["later_state_support"] = S
                row["support_mode"] = "EXPLICIT_SPECIMEN_DESCRIPTION"
                reason += (" Coracoid dimensions support anteroposterior elongation." if char == 183
                           else " A posterior coracoid notch is explicitly described.")
                row["source_locators"].append(locator(HENDERSON, char,
                    "Coracoid description supports the selected state; no new specimen inspection.",
                    138, "138", specimen="TMP 2009.121.1"))
            elif char == 41:
                reason += " Frontal boundaries are described as indiscernible; the later cast-based score remains unresolved here."
                row["source_locators"].append(locator(HENDERSON, char,
                    "Frontal preservation limits.", 132, "132", specimen="TMP 2009.121.1"))
                row["support_mode"] = "TENTATIVE_OR_CONTRADICTORY"
            else:
                reason += " The selected state lacks an explicit matching description in the checked passages."
        if tax == "Platypterygius_sachicarum":
            row["support_mode"] = "EXPANDED_SPECIMEN_SCOPE"
            row["source_locators"].extend([
                locator(ARTICLE, char, "Material and first documented postcranial material of the species.",
                        3, "e1577875-2", specimen="CIP-GA-01042014; DON-19671"),
                locator(ARTICLE, char, "Diagnosis and explicit distinction between holotype and referred specimen.",
                        4, "e1577875-3", specimen="CIP-GA-01042014; DON-19671"),
            ])
            if char == 40:
                row["later_state_support"] = S
                row["support_mode"] = "EXPLICIT_SPECIMEN_DESCRIPTION"
                reason += " The frontal description explicitly places its widening posterior to the nasal contact, supporting the posterior widest-exposure state."
                row["source_locators"].append(locator(ARTICLE, char,
                    "Frontal description; author-reported morphology, not an independent interpretation of the figure.",
                    4, "e1577875-3", figure="3D,E (referred to by authors)", specimen="CIP-GA-01042014"))
            elif char == 216:
                reason += " Measured metacarpal thickness supports the metapodial component, but the combined epipodial/metapodial state is not fully established from the checked prose."
                row["support_mode"] = "EXPLICIT_SPECIMEN_DESCRIPTION"
                row["source_locators"].append(locator(ARTICLE, char,
                    "Forelimb description, including metacarpal IV thickness; partial support only for this combined character.",
                    8, "e1577875-7", specimen="CIP-GA-01042014"))
            elif char == 211:
                reason += " The prose describes distal humeral facets but does not explicitly resolve the relative facing directions required by character 211; state 0 remains supported only by the matrix entry here."
                row["support_mode"] = "MATRIX_ENTRY_ONLY"
                row["source_locators"].append(locator(ARTICLE, char,
                    "Forelimb description checked for facet orientation; exact state unresolved without additional anatomical interpretation.",
                    7, "e1577875-6", figure="4C,D (not independently anatomically interpreted)", specimen="CIP-GA-01042014"))
            elif char == 223:
                reason += " The ulna is described as pentagonal and lacking visible notches seen in another specimen. The exact straight/convex posterior-margin state is not explicitly distinguished from all alternatives in the checked prose."
                row["support_mode"] = "MATRIX_ENTRY_ONLY"
                row["source_locators"].append(locator(ARTICLE, char,
                    "Comparison with CIP-FCG-CBP-87; absence of its notches does not by itself establish the complete selected state clause.",
                    8, "e1577875-7", specimen="CIP-GA-01042014; comparison CIP-FCG-CBP-87"))
            elif char == 224:
                reason += " The description compares the size of the radius and ulna, but does not give the ulna's proximal-versus-distal width relation. The selected state is explicit only in the matrix checked here."
                row["support_mode"] = "MATRIX_ENTRY_ONLY"
                row["source_locators"].append(locator(ARTICLE, char,
                    "Forelimb description; no explicit within-ulna width comparison located.",
                    7, "e1577875-6", figure="4C,D (not independently anatomically interpreted)", specimen="CIP-GA-01042014"))
        row["source_reason"] = reason + " This is documentary support assessment, not independent anatomical expert validation."
        row["notes"] = [
            "Matching definitions establish documentary correspondence, not homology validation.",
            "UNRESOLVED means the exact support or continuity could not be established in the checked material; it does not assert that the published state is wrong.",
        ]
    path = OUT / "ichthyosaur_documentary_reviews.json"
    path.write_text(json.dumps(selected, ensure_ascii=False, indent=2) + "\n")
    manifest = {
        "reviewed_endpoints": len(selected),
        "token_history": dict(Counter(r["token_history"] for r in selected)),
        "character_correspondence": dict(Counter(r["character_correspondence"] for r in selected)),
        "later_state_support": dict(Counter(r["later_state_support"] for r in selected)),
        "specimen_continuity": dict(Counter(r["specimen_continuity"] for r in selected)),
        "expert_review_performed": False,
        "source_files": [{"path": str(p.relative_to(ROOT)), "url": URLS[p], "sha256": HASHES[p]} for p in URLS],
        "sampling": "Unchanged frozen 20-endpoint ichthyosaur sample; no replacement.",
        "result_inputs": "No model prediction, probability, cost or performance file read.",
        "source_reproduction": "Source PDFs are retained locally for inspection; their presence is not permission to redistribute them in the supplementary code archive.",
    }
    (OUT / "ichthyosaur_documentary_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "source_files"}, indent=2))


if __name__ == "__main__":
    main()
