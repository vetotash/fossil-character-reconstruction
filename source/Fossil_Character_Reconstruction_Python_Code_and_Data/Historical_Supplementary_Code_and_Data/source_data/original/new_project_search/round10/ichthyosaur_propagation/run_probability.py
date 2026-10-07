"""Six fixed-scheme ichthyosaur completions scored with native PAUP minima."""
from pathlib import Path
from decimal import Decimal
from datetime import datetime, timezone
import argparse, gzip, hashlib, json, os, re, subprocess, time
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PANEL = HERE.parent / "ichthyosaur_reaudit/score_inputs"
FIT = HERE.parent / "ichthyosaur_shared_fit"
JOINT = HERE.parent / "ichthyosaur_joint"
TOOLS = ROOT / "morphology_workspace/morphology_project/round5/tetrapod_consequence/tools"
MODELS = ["mode", "knn5", "logistic", "extra_trees", "nj_mk"]
SCHEMES = MODELS + ["nj_mk_joint"]
TREATMENTS = ["source", "later_codes"] + MODELS
B, SEED, BATCH = 1000, 2026100310, 50
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
now = lambda: datetime.now(timezone.utc).isoformat()


def load_panel():
    z = np.load(PANEL / "panel.npz", allow_pickle=False)
    names, raw = z["names"].tolist(), z["A"]
    ordered = set(map(int, z["ordered_characters"]))
    cells = pd.read_csv(PANEL / "target_cells.csv", keep_default_na=False, dtype={"true": str})
    cells = cells.sort_values(["character", "taxon"]).reset_index(drop=True)
    tree_names = re.findall(r"^TREE\s+(\S+)\s*=", (PANEL / "reference_trees.nex").read_text(), re.M)
    assert raw.shape == (101, 287) and len(cells) == 192 and len(tree_names) == 5
    assert ordered == {99, 203, 204, 218}
    for r in cells.itertuples():
        assert raw[names.index(r.taxon), int(r.character)-1] == "?"
        assert r.true in "0123" and len(r.true) == 1
    return names, raw, ordered, cells, tree_names


def generate():
    assert not (HERE / "FREEZE.json").exists(), "Do not overwrite frozen draws"
    scope = json.loads((PANEL / "PRESCORE_GATE.json").read_text())
    for path, h in scope["hashes"].items():
        assert sha(PANEL.parent / path) == h, path
    names, raw, ordered, cells, trees = load_panel()
    p = pd.read_csv(FIT / "predictions.csv", keep_default_na=False, dtype={"true": str, "pred": str})
    p = p[p.scenario.eq("later_target")]
    full = p[p.model.eq("mode")].sort_values(["character", "taxon"]).reset_index(drop=True)
    assert len(full) == 196 and not full.duplicated(["character", "taxon"]).any()
    key = ["character", "taxon"]
    lookup = {(int(r.character), r.taxon): i for i, r in enumerate(full.itertuples())}
    ix = np.array([lookup[(int(r.character), r.taxon)] for r in cells.itertuples()])
    assert full.iloc[ix].true.tolist() == cells.true.tolist()
    u = np.random.default_rng(SEED).random((B, len(full)))
    np.save(HERE / "common_uniforms_full196.npy", u)
    full.to_csv(HERE / "full_future_cells.csv", index=False)
    cells["full_prediction_column"] = ix
    cells.to_csv(HERE / "cells.csv", index=False)
    hard = [np.repeat("?", len(cells)), cells.true.to_numpy(dtype="U1")]
    for model in MODELS:
        sub = p[p.model.eq(model)].sort_values(key).reset_index(drop=True)
        assert sub[key+['true']].equals(full[key+['true']])
        sample = np.empty((B, len(full)), dtype="U1")
        for i, r in enumerate(sub.itertuples()):
            probs = json.loads(r.state_probabilities); states = sorted(probs)
            values = np.array([probs[s] for s in states])
            assert all(len(s)==1 and s in "0123" for s in states)
            assert np.isfinite(values).all() and (values >= 0).all() and abs(values.sum()-1)<1e-12
            cdf = values.cumsum(); cdf[-1] = 1.0
            sample[:, i] = np.array(states)[np.searchsorted(cdf, u[:, i], side="right")]
        np.save(HERE / f"{model}_draws.npy", sample[:, ix], allow_pickle=False)
        hard.append(sub.iloc[ix].pred.to_numpy(dtype="U1"))
    jc = pd.read_csv(JOINT / "future_cells.csv", keep_default_na=False)
    jm = {(int(r.character), r.taxon): int(r.draw_column) for r in jc.itertuples()}
    ji = np.array([jm[(int(r.character), r.taxon)] for r in cells.itertuples()])
    joint = np.load(JOINT / "future_joint_draws.npy", allow_pickle=False)[:, ji]
    assert joint.shape == (B, 192) and np.isin(joint, list("0123")).all()
    np.save(HERE / "nj_mk_joint_draws.npy", joint, allow_pickle=False)
    np.save(HERE / "hard_assignments.npy", np.stack(hard), allow_pickle=False)
    paths = [Path(__file__), HERE / "PROTOCOL.txt", PANEL / "PRESCORE_GATE.json", PANEL / "panel.npz",
             PANEL / "reference_trees.nex", PANEL / "target_cells.csv", FIT / "predictions.csv",
             FIT / "FIT_FREEZE.json", FIT / "FIT_COMPLETED.json", JOINT / "FREEZE.json",
             JOINT / "future_joint_draws.npy", JOINT / "future_cells.csv", JOINT / "DRAW_CHECK.json",
             HERE / "common_uniforms_full196.npy", HERE / "cells.csv", HERE / "full_future_cells.csv",
             HERE / "hard_assignments.npy", TOOLS / "paup4a169_ubuntu64"]
    paths += [HERE / f"{s}_draws.npy" for s in SCHEMES]
    freeze = {"created_utc": now(), "seed": SEED, "joint_seed": 2026100311,
              "replicates": B, "batch_size": BATCH, "scoring_taxa": len(names), "targets": len(cells),
              "target_characters": int(cells.character.nunique()), "ordered_characters": sorted(ordered),
              "trees": trees, "schemes": SCHEMES, "native_scores_seen": False,
              "files": [{"path": str(p.resolve()), "sha256": sha(p), "bytes": p.stat().st_size} for p in paths]}
    (HERE / "FREEZE.json").write_text(json.dumps(freeze, indent=2))
    print(json.dumps({k:v for k,v in freeze.items() if k!='files'}, indent=2))


def check_freeze():
    freeze = json.loads((HERE / "FREEZE.json").read_text())
    for r in freeze["files"]:
        assert sha(r["path"]) == r["sha256"], r["path"]
    return freeze


def build_matrix(raw, names, chars, cells, draws, ordered):
    """Original character IDs stay explicit when independent blocks repeat."""
    nr = 1 if draws is None else len(draws)
    m = np.tile(raw[:, np.array(chars)-1], (1, nr))
    lookup = {int(c): j for j, c in enumerate(chars)}
    if draws is not None:
        assert draws.shape[1] == len(cells)
        for i, r in enumerate(cells.itertuples()):
            if int(r.character) not in lookup:
                continue
            ri = names.index(r.taxon); ci = lookup[int(r.character)]
            assert raw[ri, int(r.character)-1] == "?"
            m[ri, ci+np.arange(nr)*len(chars)] = draws[:, i]
    nc = m.shape[1]
    oc = [k*len(chars)+j+1 for k in range(nr) for j,ch in enumerate(chars) if ch in ordered]
    uc = [k*len(chars)+j+1 for k in range(nr) for j,ch in enumerate(chars) if ch not in ordered]
    specs = []
    if oc: specs.append("ord: " + " ".join(map(str, oc)))
    if uc: specs.append("unord: " + " ".join(map(str, uc)))
    text = "#NEXUS\nBEGIN DATA;\n" + f"DIMENSIONS NTAX={len(names)} NCHAR={nc};\n"
    text += 'FORMAT DATATYPE=STANDARD SYMBOLS="0123" GAP=- MISSING=?;\nMATRIX\n'
    text += "".join(n+" "+"".join(row)+"\n" for n,row in zip(names,m)) + ";\nEND;\n"
    text += "BEGIN ASSUMPTIONS; OPTIONS DEFTYPE=UNORD POLYTCOUNT=MINSTEPS;\nTYPESET * repeated=" + ", ".join(specs) + ";\nEND;\n"
    return m, text, len(oc)


def parse_scores(path, nc, ntree):
    tab = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    assert tab.columns.tolist() == ['Tree','Length','Character','Length.1']
    scores = np.zeros((ntree, nc), dtype=np.int16); totals = {}; seen = [set() for _ in range(ntree)]
    tree = None
    for r in tab.itertuples(index=False, name=None):
        if r[0]:
            tree = int(r[0])-1; total = Decimal(r[1]); assert total == int(total)
            assert tree not in totals and 0 <= tree < ntree; totals[tree] = int(total)
        elif r[2]:
            ch = int(r[2])-1; value = Decimal(r[3]); assert value == int(value)
            assert tree is not None and 0 <= ch < nc and ch not in seen[tree]
            scores[tree,ch] = int(value); seen[tree].add(ch)
    assert set(totals) == set(range(ntree))
    assert all(s == set(range(nc)) for s in seen)
    assert all(int(scores[i].sum()) == totals[i] for i in range(ntree))
    return scores, totals


def score(raw, names, chars, cells, draws, ordered, dest, tree_file=None, tree_names=None):
    """General native wrapper also supports later changed-order character slices."""
    dest = Path(dest); assert not dest.exists(); dest.mkdir(parents=True)
    tree_file = PANEL / "reference_trees.nex" if tree_file is None else Path(tree_file)
    if tree_names is None:
        tree_names = re.findall(r"^TREE\s+(\S+)\s*=", tree_file.read_text(), re.M)
    matrix, text, nordered = build_matrix(raw, names, chars, cells, draws, ordered)
    nc, ntree = matrix.shape[1], len(tree_names); nr = nc//len(chars)
    driver = dest / "diagnose.nex"
    driver.write_text(text + f'''\nBEGIN PAUP;
set autoclose=yes increase=no maxtrees=100;
outgroup Hupehsuchus_nanchangensis / only;
pset mstaxa=variable minforfit=minlength opt=deltran;
gettrees file='{tree_file.resolve()}';
pscores all / single=all tl=yes scorefile=native_scores.tsv replace=yes;
describe 1 / plot=none diag=yes;
quit;
END;\n''')
    env = os.environ.copy(); env["LD_LIBRARY_PATH"] = str((TOOLS / "runtime/usr/lib/x86_64-linux-gnu").resolve()); env['OMP_NUM_THREADS']='1'
    started = time.monotonic()
    with (dest / "stdout.txt").open("w") as f:
        process = subprocess.run([str((TOOLS / "paup4a169_ubuntu64").resolve()),'-n','-u','-L','native.log','-r',driver.name],
                                  cwd=dest, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=120)
    assert process.returncode == 0
    log = (dest / "native.log").read_text()
    assert not re.search(r'\berrors?\b|execution terminated|processing.*terminated', log, re.I)
    if nordered:
        assert re.search(r'\b'+str(nordered)+r" character(?:s)? (?:is|are) of type 'ord'", log), ('ordered count', nordered)
    else:
        assert not re.search(r"\b[1-9]\d* character(?:s)? (?:is|are) of type 'ord'", log)
    has_multi = any(s.startswith(('(', '{')) for s in matrix.ravel())
    if has_multi:
        assert 'Multistate taxa interpretation depends on "{}" versus "()" designation' in log
    scores, totals = parse_scores(dest / "native_scores.tsv", nc, ntree)
    block = log.split('Character diagnostics:\n',1)[1]
    assert re.search(r'Character\s+Range\s+steps\s+steps\s+steps', block)
    minima = {}; diagnostic = []
    for line in block.splitlines():
        match = re.match(r'^\s*(\d+)\s+(.*)$', line)
        if not match: continue
        vals = match[2].split()[-9:]
        if len(vals) < 9: continue
        index = int(match[1])-1
        ran, minimum, assigned, maximum = map(Decimal, vals[:4])
        assert minimum == int(minimum) and assigned == int(assigned)
        assert index not in minima and 0 <= index < nc and minimum <= assigned <= maximum
        assert int(assigned) == int(scores[0,index])
        minima[index] = int(minimum)
        diagnostic.append({'column':index+1,'state_range':str(ran),'native_minimum_steps':int(minimum),
                           'native_assigned_steps':int(assigned),'native_maximum_steps':str(maximum),'omitted_constant':False})
    absent = set(range(nc)) - set(minima)
    constant = re.search(r'\((\d+) constant character(?:s)? not shown\)', block)
    omitted = int(constant[1]) if constant else 0
    assert len(absent) == omitted, (len(absent), omitted)
    for index in absent:
        tokens = set(matrix[:,index]) - {'?','-'}
        assert len(tokens) <= 1 and all(len(s)==1 and s in '0123' for s in tokens)
        assert np.all(scores[:,index] == 0)
        minima[index] = 0
        diagnostic.append({'column':index+1,'state_range':'0','native_minimum_steps':0,
                           'native_assigned_steps':0,'native_maximum_steps':'0','omitted_constant':True})
    minimum = np.array([minima[i] for i in range(nc)], dtype=np.int16)
    assert np.all(scores >= minimum[None,:])
    pd.DataFrame(diagnostic).sort_values('column').to_csv(dest/'lower_bounds.csv',index=False)
    evidence = {}
    for name in ['diagnose.nex','native.log','native_scores.tsv','stdout.txt','lower_bounds.csv']:
        path = dest/name; payload = path.read_bytes(); compressed = path.with_suffix(path.suffix+'.gz')
        compressed.write_bytes(gzip.compress(payload, compresslevel=6, mtime=0))
        evidence[name] = {'bytes':len(payload),'raw_sha256':sha(path),'gzip_sha256':sha(compressed)}
        path.unlink()
    verification = {'seconds':time.monotonic()-started,'columns':nc,'replicates':nr,'tree_names':tree_names,
                    'ordered_columns':nordered,'multistate_tokens_present':has_multi,'omitted_constants':omitted,
                    'native_totals':totals,'all_sums_verified':True,'all_constants_verified':True,'evidence':evidence}
    (dest/'native_evidence.json').write_text(json.dumps(verification,indent=2))
    return scores.reshape(ntree,nr,len(chars)), minimum.reshape(nr,len(chars))


def validate():
    check_freeze(); names, raw, ordered, cells, trees = load_panel()
    chars = sorted(cells.character.unique().tolist()); assert len(chars)==128
    hard = np.load(HERE/'hard_assignments.npy',allow_pickle=False)
    slices, minima = [], []
    for i,treatment in enumerate(TREATMENTS):
        s,m = score(raw,names,list(range(1,288)),cells,None if i==0 else hard[i:i+1],ordered,
                    HERE/'validation'/treatment)
        slices.append(s); minima.append(m)
    fulls = np.concatenate(slices,axis=1); fullm = np.concatenate(minima,axis=0)
    nontarget = np.array([c-1 for c in range(1,288) if c not in chars])
    assert np.all(fulls[:,:,nontarget] == fulls[:,0:1,nontarget])
    assert np.all(fullm[:,nontarget] == fullm[0:1,nontarget])
    cs,cm = score(raw,names,chars,cells,hard,ordered,HERE/'validation'/'hard_concatenated')
    indices=np.array(chars)-1
    assert np.array_equal(cs,fulls[:,:,indices]) and np.array_equal(cm,fullm[:,indices])
    np.savez_compressed(HERE/'hard_full_scores.npz',steps=fulls,minima=fullm,characters=np.arange(1,288),trees=trees,treatments=TREATMENTS)
    np.savez_compressed(HERE/'hard_scores.npz',steps=cs,minima=cm,characters=chars,trees=trees,treatments=TREATMENTS)
    rows=[]
    for ti,tree in enumerate(trees):
        for k,treatment in enumerate(TREATMENTS):
            ds=int(fulls[ti,k].sum()-fulls[ti,0].sum()); dm=int(fullm[k].sum()-fullm[0].sum())
            rows.append({'reference':tree,'treatment':treatment,'full_tree_steps':int(fulls[ti,k].sum()),
                         'full_minimum_steps':int(fullm[k].sum()),'added_steps':ds,'added_minimum_steps':dm,'added_excess_steps':ds-dm})
    pd.DataFrame(rows).to_csv(HERE/'fixed_tree_totals.csv',index=False)
    check={'completed_utc':now(),'source_and_six_hard_treatments_separately_scored':True,
           'target_concatenation_scores_equal':True,'target_concatenation_minima_equal':True,
           'nontarget_score_and_minimum_changes_zero':True,'all_polymorphic_tokens_preserved':True,
           'source_taxa':101,'source_characters':287,'target_characters':128,'targets':192,
           'character_tree_checks':int(cs.size),'minimum_checks':int(cm.size),'hard_scores_sha256':sha(HERE/'hard_scores.npz')}
    (HERE/'VALIDATION.json').write_text(json.dumps(check,indent=2));print(json.dumps(check,indent=2))


def run():
    check_freeze(); validation=json.loads((HERE/'VALIDATION.json').read_text())
    assert sha(HERE/'hard_scores.npz')==validation['hard_scores_sha256']
    names,raw,ordered,cells,trees=load_panel();chars=sorted(cells.character.unique().tolist())
    hard=np.load(HERE/'hard_scores.npz',allow_pickle=False);hs,hm=hard['steps'],hard['minima']
    bs,bm=hs[:,0,:],hm[0]
    observed=hs[:,1,:].sum(axis=1)-bs.sum(axis=1)-(hm[1].sum()-bm.sum())
    rows=[];all_replicates=[];inventory={};started=time.monotonic()
    for scheme in SCHEMES:
        drawpath=HERE/f'{scheme}_draws.npy';draws=np.load(drawpath,allow_pickle=False)
        assert draws.shape==(B,len(cells))
        ss=[];mm=[]
        for start in range(0,B,BATCH):
            stop=min(B,start+BATCH);dest=HERE/'native'/scheme/f'batch_{start:04d}_{stop:04d}'
            scores,mins=score(raw,names,chars,cells,draws[start:stop],ordered,dest)
            ss.append(scores);mm.append(mins)
            (HERE/'progress.json').write_text(json.dumps({'scheme':scheme,'completed_in_scheme':stop,
                'previous_schemes':list(inventory),'seconds':time.monotonic()-started},indent=2))
        scores=np.concatenate(ss,axis=1);mins=np.concatenate(mm,axis=0)
        np.savez_compressed(HERE/f'{scheme}_scores.npz',steps=scores,minima=mins,characters=chars,trees=trees)
        ds=scores.sum(axis=2)-bs.sum(axis=1)[:,None];dm=mins.sum(axis=1)-bm.sum();extra=ds-dm[None,:]
        assert np.all(ds>=0) and np.all(dm>=0)
        model='nj_mk' if scheme=='nj_mk_joint' else scheme;k=TREATMENTS.index(model)
        point=hs[:,k,:].sum(axis=1)-bs.sum(axis=1)-(hm[k].sum()-bm.sum())
        for ti,tree in enumerate(trees):
            v=extra[ti];q=np.quantile(v,[.025,.975])
            rows.append({'model':scheme,'reference':tree,'replicates':B,'reference_excess_steps':int(observed[ti]),
                'point_excess_steps':int(point[ti]),'sim_mean_excess_steps':float(v.mean()),'sim_sd_excess_steps':float(v.std(ddof=1)),
                'sim_q025':float(q[0]),'sim_q975':float(q[1]),'fraction_at_least_reference':float((v>=observed[ti]).mean()),
                'fraction_at_most_reference':float((v<=observed[ti]).mean()),'reference_in_central95':bool(q[0]<=observed[ti]<=q[1]),
                'mean_added_steps':float(ds[ti].mean()),'mean_added_minimum_steps':float(dm.mean())})
            all_replicates.extend({'model':scheme,'reference':tree,'replicate':b,'added_steps':int(ds[ti,b]),
                'added_minimum_steps':int(dm[b]),'added_excess_steps':int(v[b])} for b in range(B))
        pd.DataFrame(rows).to_csv(HERE/'summary.csv',index=False)
        pd.DataFrame(all_replicates).to_csv(HERE/'replicate_totals.csv',index=False)
        inventory[scheme]={'completed_replicates':B,'scores_sha256':sha(HERE/f'{scheme}_scores.npz'),'draws_sha256':sha(drawpath)}
        print(json.dumps({'model':scheme,'completed':B,'seconds':round(time.monotonic()-started,2)}),flush=True)
    check={'completed_utc':now(),'seconds':time.monotonic()-started,'models_refitted':0,'tree_searches':0,
           'replicates_per_scheme':B,'n_schemes':6,'n_references':5,'summary_rows':len(rows),'replicate_tree_totals':len(all_replicates),
           'main_ordered_characters':sorted(ordered),'all_native_minima_used':True,'inventory':inventory}
    assert len(rows)==30 and len(all_replicates)==30000
    (HERE/'RUN_MANIFEST.json').write_text(json.dumps(check,indent=2));print(json.dumps(check,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('operation',choices=['generate','validate','score']);args=parser.parse_args()
    {'generate':generate,'validate':validate,'score':run}[args.operation]()
