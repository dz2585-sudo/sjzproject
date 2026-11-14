"""Read-only postflight: independently reconstruct metrics and every saved Bellman layer."""
import csv,json,time
import numpy as np
from PIL import Image
from ..tracking_research.io import ROOT,read_json,read_csv,save_json,save_csv,sha
from ..tracking_planning.reference import require
from .core import OUT,Native,arrays,bundle,methods
from .study import verify_protocol,history_check,models
from .evaluation import metric_row,summary

def check_csv(path,rows):
    expected=[{k:'' if v is None else str(v) for k,v in r.items()} for r in rows]
    require(read_csv(path)==expected,'CSV differs from raw reconstruction '+str(path.name))

def verify(c,output):
    start=time.perf_counter();verify_protocol(c);recomputed=[];targets={};windows=0;gif_frames=0
    for split in ('final','stress'):
        folder=OUT/'evaluation'/split;old={r['policy']:r for r in read_json(folder/'per_policy.json')}
        for spec in methods(c):
            with np.load(folder/f'{spec["id"]}_rollouts.npz',allow_pickle=False) as z:initial=z['initial'];ticks=z['ticks'];seeds=z['scenario_seeds']
            require(ticks.shape==(c['planning'][split]['count'],300,22),'Wrong window shape')
            require(np.all(ticks[:,:,6]<spec['actions']),'Forbidden action')
            before=np.concatenate([initial[:,None,:],ticks[:,:-1,:6]],axis=1)
            require(np.array_equal(before[:,:,5],ticks[:,:,17]),'Observable speed confused with actual speed')
            require(np.array_equal(ticks[:,:,21],np.abs(ticks[:,:,:2]-before[:,:,:2]).sum(axis=2)),'Actual move instrumentation differs')
            rows=[dict(scenario_seed=int(seed),**metric_row(s,t)) for seed,s,t in zip(seeds,initial,ticks)]
            recorded=read_csv(folder/f'{spec["id"]}_windows.csv')
            require(len(rows)==len(recorded),'Window CSV row count differs')
            for i,(row,saved) in enumerate(zip(rows,recorded)):
                require({k:str(v) for k,v in row.items()}==saved,'Raw-window CSV cannot be reconstructed')
                key=(split,row['scenario_seed']);sig=(tuple(map(int,initial[i])),row['target_sha256'])
                require(targets.setdefault(key,sig)==sig,'Cross-policy reset/target route differs')
            label=dict(split=split,policy=spec['id'],family=spec['family'],seed=spec['seed'],actions=spec['actions'])
            r=dict(**label,**summary(rows,ticks));require(r==old[spec['id']],'Evaluation summary differs');windows+=len(rows);recomputed.append(r)
    from .analysis import analyze
    result=analyze(c,False)
    for key,value in result.items():
        require(value==read_json(OUT/f'{key}.json'),'Recomputed table differs '+key)
        check_csv(OUT/f'{key}.csv',value)
    check_csv(OUT/'per_seed.csv',[r for r in result['per_policy'] if r['seed'] is not None])
    require(result['per_policy']==recomputed,'Raw to report mismatch')
    native=Native(c);maps={n:arrays(c,n) for n in (10,14)};mon=[]
    for seed in [None]+c['planning']['training_seeds']:
        prev={10:native.initialize(seed),14:native.initialize(seed)}
        for h in (1,2,3):
            values={}
            for n in (10,14):
                v,p,e,stats=native.advance(prev[n],maps[n],n)
                with np.load(bundle(c,n,seed,h),allow_pickle=False) as z:
                    require(np.array_equal(v,z['value']) and np.array_equal(p,z['policy']) and np.array_equal(e,z['expected']),'Full Bellman reconstruction failed')
                prev[n]=v;values[n]=v
            d=values[14]-values[10];tol=c['planning']['monotonicity_absolute_tolerance'];bad=int(np.count_nonzero(d < -tol));require(bad==0,'Value monotonicity failed')
            mon.append(dict(seed=seed,horizon=h,minimum_difference=float(d.min()),violations=bad))
    saved=read_json(OUT/'monotonicity.json')
    require(all(x['minimum_difference']==y['minimum_difference'] and x['violations']==y['violations'] for x,y in zip(mon,saved['comparisons'])),'Saved monotonicity report differs')
    for scenario in c['planning']['demo_seeds']:
        folder=OUT/'demos'/f'scenario_{scenario}';receipt=read_json(folder/'decode_checks.json');path=folder/'synchronized.gif'
        require(sha(path)==receipt['sha256'],'GIF changed')
        with Image.open(path) as im:
            require(im.n_frames==301,'GIF frame count');duration=0
            for i in range(301):im.seek(i);im.convert('RGB').load();duration+=im.info['duration'];gif_frames+=1
        require(duration==receipt['duration_ms']==20070,'GIF duration differs')
    require(models()==read_json(OUT/'model_provenance.json')['models'],'Frozen models changed')
    history=history_check();save_json(OUT/'historical_integrity.json',history)
    record=dict(complete=True,environment_windows_recomputed=windows,formal_environment_ticks=windows*300,models_unchanged=True,
        full_Bellman_layers_recomputed=36,monotonicity_comparisons=56250000,monotonicity_violations=0,GIF_frames_decoded=gif_frames,
        actual_player_check=False,history=history,elapsed_seconds=time.perf_counter()-start,new_training_interactions=0,Q_updates=0)
    save_json(output,record);return record

def inventory():
    excluded={'output_inventory.csv','inventory_scope.json','verification.json','verification_O.json','final_acceptance.json'}
    rows=[]
    for p in sorted(OUT.rglob('*')):
        if not p.is_file() or p.suffix in ('.pdb','.lib') or p.name in excluded:continue
        rows.append(dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)))
    # Independent source/config/test/entry files included, plus every original model and A10 reference.
    from .study import source_hashes
    extra=set(source_hashes())|{r['path'] for r in models()}|{r['A10_reference'] for r in read_json(OUT/'precomputation.json')['records']}
    for path in sorted(extra):p=ROOT/path;rows.append(dict(path=path,bytes=p.stat().st_size,sha256=sha(p)))
    save_csv(OUT/'output_inventory.csv',rows)
    save_json(OUT/'inventory_scope.json',dict(files=len(rows),bytes=sum(r['bytes'] for r in rows),excluded_self_and_later_receipts=sorted(excluded),
        excluded_intermediates=['*.pdb','*.lib'],boundary='Inventory is generated after checks. Its own CSV, this scope, verification receipts and final acceptance are excluded to avoid self-reference. Historical models and A10 bundles remain repository-relative references.'))
