"""Extract observed geometry only. No simulation, fitted model, or ranking.

All Round 14 files are read-only inputs. The portable output excludes later
labels from its fitting template. Historical targets are positional labels.
"""
from pathlib import Path
import csv, hashlib, json, re, shutil
import numpy as np

HERE = Path(__file__).resolve().parent
R14 = HERE.parents[1] / 'round14'
PANELS = ('tetrapods', 'dinosaurs', 'ichthyosaurs')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def known(a):
    return np.vectorize(lambda s: bool(re.fullmatch('[0-9A-Z]', str(s))))(a)


def write_csv(p, rows):
    with p.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def distribution(v):
    v = np.asarray(v, float)
    total = float(v.sum())
    hhi = float(np.sum((v / total) ** 2)) if total else None
    return dict(total=total, minimum=float(v.min()), q25=float(np.quantile(v, .25)),
                median=float(np.median(v)), q75=float(np.quantile(v, .75)),
                maximum=float(v.max()), mean=float(v.mean()),
                sd=float(v.std(ddof=0)), nonzero=int((v > 0).sum()),
                top_five_share=float(np.sort(v)[-5:].sum()/total) if total else None,
                concentration_hhi=hhi, effective_units=1/hhi if hhi else None)


def parse_tree(s):
    s = re.sub(r'\[.*?\]', '', s).strip().rstrip(';')
    assert ':' not in s, 'Do not silently discard supplied branch lengths'
    tokens = re.findall(r'[(),]|[^(),\s]+', s)
    i = 0
    def rec():
        nonlocal i
        token = tokens[i]; i += 1
        if token != '(':
            assert token not in (',', ')')
            return token.strip("'")
        children = [rec()]
        while tokens[i] == ',':
            i += 1; children.append(rec())
        assert tokens[i] == ')'; i += 1
        return children
    tree = rec(); assert i == len(tokens)
    return tree


def tree_edges(tree, names):
    """Suppress degree-two internal nodes, then assign unit edges separately.

    No original branch lengths are present. This makes a portable topology;
    unit lengths are an explicit simulation convention, not fossil estimates.
    """
    label = {v: i for i, v in enumerate(names)}
    edges = []; next_id = len(names); seen = []
    def rec(t):
        nonlocal next_id
        if isinstance(t, str):
            seen.append(t); return label[t]
        node = next_id; next_id += 1
        for child in t:
            edges.append((node, rec(child)))
        return node
    rec(tree)
    assert len(seen) == len(set(seen)) == len(names) and set(seen) == set(names)
    adjacency = {n:set() for n in range(next_id)}
    for a,b in edges:
        adjacency[a].add(b); adjacency[b].add(a)
    suppressed = []
    while True:
        candidates = [n for n in sorted(adjacency) if n >= len(names) and len(adjacency[n]) == 2]
        if not candidates: break
        node = candidates[0]; a,b = sorted(adjacency[node]); suppressed.append(node)
        adjacency[a].remove(node); adjacency[b].remove(node)
        adjacency[a].add(b); adjacency[b].add(a); del adjacency[node]
    assert all(len(v) >= 3 for n,v in adjacency.items() if n >= len(names))
    assert all(len(adjacency[n]) == 1 for n in range(len(names)))
    renumber = {n:i for i,n in enumerate(sorted(adjacency))}
    out = sorted((renumber[a],renumber[b]) for a,bs in adjacency.items() for b in bs if a < b)
    assert len(out) == len(adjacency)-1
    return dict(edges=out, n_nodes=len(adjacency), n_tips=len(names),
                root=len(names), edge_lengths=[1.0]*len(out),
                suppressed_degree_two_nodes=len(suppressed),
                internal_degree_counts={str(d):sum(len(v)==d for n,v in adjacency.items() if n>=len(names))
                                        for d in sorted(set(len(v) for n,v in adjacency.items() if n>=len(names)))},
                original_branch_lengths_present=False,
                edge_length_interpretation='Unit length assigned after suppression; not fossil branch-length estimates')


def build():
    summaries=[]; all_inputs=[]
    for panel in PANELS:
        src=R14/'downstream'/panel; dest=HERE/panel; dest.mkdir(exist_ok=True)
        sourcefiles=[src/'panel.npz',src/'cells.csv',src/'reference_trees.nex',src/'SMOKE_CHECK.json']
        trainpath=R14/('ichthyosaur_refit/source_training.npz' if panel=='ichthyosaurs'
                      else f'old_refits/inputs/{panel}/source_training.npz')
        frozenpath=trainpath.with_name('frozen_targets.csv')
        sourcefiles += [trainpath,frozenpath]
        z=np.load(src/'panel.npz',allow_pickle=False)
        raw=z['raw']; names=z['names'].tolist(); simple=known(raw)
        with (src/'cells.csv').open() as f: rows=list(csv.DictReader(f))
        target=np.zeros(raw.shape,bool); positions=[]
        for r in rows:
            i=names.index(r['taxon']); j=int(r['character'])-1
            assert raw[i,j]=='?' and not target[i,j]
            target[i,j]=True
            positions.append(dict(row_zero_based=i,column_zero_based=j,taxon=names[i],character_one_based=j+1))
        assert not np.any(target & simple)
        # 0 simple original state, 1 other unavailable, 2 historical target.
        labels=np.where(simple,0,np.where(target,2,1)).astype(np.int8)
        columns=np.flatnonzero(target.any(0))
        t=np.load(trainpath,allow_pickle=False); tn=t['old_names'].tolist(); tc=t['old_char_numbers']
        assert set(columns+1) <= set(tc)
        np.savez_compressed(dest/'template.npz',names=np.asarray(names),characters=np.arange(1,raw.shape[1]+1),
                            labels=labels,observed_mask=simple,target_mask=target,
                            source_question_mask=raw=='?',nonsingleton_or_gap_mask=(~simple)&(raw!='?'),
                            target_columns=columns,original_ordered_characters=z['ordered'].astype(int),
                            admitted_predictor_columns=tc-1,
                            admitted_predictor_characters=tc)
        # Keep both raw geometry and the exact admitted historical predictor projection.
        np.savez_compressed(dest/'admitted_template.npz',names=np.asarray(names),characters=tc,
                            labels=labels[:,tc-1],observed_mask=simple[:,tc-1],target_mask=target[:,tc-1],
                            source_question_mask=(raw=='?')[:,tc-1],
                            nonsingleton_or_gap_mask=((~simple)&(raw!='?'))[:,tc-1],
                            target_columns=np.flatnonzero(target[:,tc-1].any(0)),
                            original_column_indices=tc-1,
                            original_ordered_characters=z['ordered'].astype(int))
        # Raw source tokens are provenance only, not supplied as states to any binary fit.
        shutil.copyfile(src/'panel.npz',dest/'source_panel.npz')
        shutil.copyfile(src/'reference_trees.nex',dest/'reference_trees.nex')
        write_csv(dest/'target_positions.csv',positions)
        records=[]
        for name,newick in re.findall(r'^\s*TREE\s+(\S+)\s*=\s*(.*?;)',(src/'reference_trees.nex').read_text(),re.M|re.I):
            records.append(dict(name=name,**tree_edges(parse_tree(newick),names)))
        (dest/'tree_topologies.json').write_text(json.dumps(records,indent=2)+'\n')
        bytax=[]
        for i,name in enumerate(names):
            bytax.append(dict(row_zero_based=i,taxon=name,characters=raw.shape[1],simple_observed=int(simple[i].sum()),
                              unavailable=int((~simple[i]).sum()),question=int((raw[i]=='?').sum()),
                              nonsingleton_or_gap=int(((~simple[i])&(raw[i]!='?')).sum()),
                              historical_targets=int(target[i].sum()),missing_fraction=float((~simple[i]).mean()),
                              historical_target_fraction=float(target[i].mean())))
        bychar=[]
        for j in range(raw.shape[1]):
            bychar.append(dict(column_zero_based=j,character_one_based=j+1,taxa=len(names),simple_observed=int(simple[:,j].sum()),
                               unavailable=int((~simple[:,j]).sum()),question=int((raw[:,j]=='?').sum()),
                               nonsingleton_or_gap=int(((~simple[:,j])&(raw[:,j]!='?')).sum()),
                               historical_targets=int(target[:,j].sum()),is_target_character=bool(target[:,j].any()),
                               missing_fraction=float((~simple[:,j]).mean()),original_ordered=bool(j+1 in z['ordered'])))
        write_csv(dest/'by_taxon.csv',bytax);write_csv(dest/'by_character.csv',bychar)
        projected=t['A'][[tn.index(n) for n in names]]
        assert np.array_equal(projected,raw[:,tc-1])
        with frozenpath.open() as f: frozen=list(csv.DictReader(f))
        future=[r for r in frozen if r['scenario'].startswith('later')]
        frozenkeys={(r['taxon'],int(r['character'])) for r in future}
        keys={(r['taxon'],r['character_one_based']) for r in positions}
        assert keys <= frozenkeys
        symbols={str(k):int(v) for k,v in zip(*np.unique(raw,return_counts=True))}
        stats=dict(panel=panel,n_taxa=len(names),n_characters=raw.shape[1],n_cells=raw.size,
                   simple_observed=int(simple.sum()),unavailable=int((~simple).sum()),
                   unavailable_fraction=float((~simple).mean()),question=int((raw=='?').sum()),
                   nonsingleton_or_gap=int(((~simple)&(raw!='?')).sum()),
                   historical_targets=int(target.sum()),target_characters=len(columns),
                   target_taxa=int(target.any(1).sum()),complete_anchor_characters=int(simple.all(0).sum()),
                   singleton_complete_characters=int(simple.all(0).sum()),
                   characters_without_literal_question_mark=int((raw!='?').all(0).sum()),
                   characters_without_literal_question_or_gap=int((~np.isin(raw,['?','-'])).all(0).sum()),
                   target_fraction_of_unavailable=float(target.sum()/(~simple).sum()),
                   min_simple_observed_in_target_character=int(simple[:,columns].sum(0).min()),
                   source_symbol_inventory=symbols,
                   unavailable_by_taxon=distribution((~simple).sum(1)),
                   unavailable_by_character=distribution((~simple).sum(0)),
                   historical_targets_by_taxon=distribution(target.sum(1)),
                   historical_targets_by_character=distribution(target.sum(0)),
                   full_source_training_shape=list(t['A'].shape),full_source_future_targets=len(frozenkeys),
                   full_source_future_prediction_rows=len(future),
                   downstream_subset_source_tokens_exact=True,
                   full_training_characters_retained=tc.tolist(),
                   admitted_predictor_unavailable_fraction=float((~simple[:,tc-1]).mean()),
                   admitted_predictor_complete_anchor_characters=int(simple[:,tc-1].all(0).sum()),
                   admitted_predictor_unavailable_by_taxon=distribution((~simple[:,tc-1]).sum(1)),
                   admitted_predictor_unavailable_by_character=distribution((~simple[:,tc-1]).sum(0)),
                   admitted_predictor_label_counts={str(i):int((labels[:,tc-1]==i).sum()) for i in range(3)},
                   excluded_from_admitted_predictor_characters=sorted(set(range(1,raw.shape[1]+1))-set(map(int,tc))),
                   frozen_targets_excluded_by_downstream_scope=[dict(taxon=n,character=c) for n,c in sorted(frozenkeys-keys)],
                   reference_tree_count=len(records),tree_names=[r['name'] for r in records])
        (dest/'description.json').write_text(json.dumps(stats,indent=2)+'\n')
        input_records=[dict(path=str(p.resolve()),sha256=sha(p),bytes=p.stat().st_size) for p in sourcefiles]
        all_inputs.extend(input_records)
        (dest/'SOURCE_MANIFEST.json').write_text(json.dumps(input_records,indent=2)+'\n')
        summaries.append(stats)
    (HERE/'SUMMARY.json').write_text(json.dumps(summaries,indent=2)+'\n')
    manifest=dict(script_sha256=sha(__file__),inputs=all_inputs,outputs=[])
    for p in sorted(HERE.rglob('*')):
        if p.is_file() and p.name!='OUTPUT_MANIFEST.json':
            manifest['outputs'].append(dict(path=str(p.relative_to(HERE)),sha256=sha(p),bytes=p.stat().st_size))
    (HERE/'OUTPUT_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps([{k:s[k] for k in ('panel','n_taxa','n_characters','unavailable_fraction','historical_targets','target_characters','complete_anchor_characters','reference_tree_count')} for s in summaries],indent=2))


if __name__ == '__main__':
    build()
