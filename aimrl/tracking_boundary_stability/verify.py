"""Artifact-based checks; every requirement uses explicit exceptions under -O."""
import json
import math
import sys

import numpy as np
from PIL import Image

from ..tracking_research.io import ROOT, read_csv, read_json, save_csv, save_json, sha
from ..tracking_tradeoff.core import summarize, window_metrics
from .analysis import aggregate, contrasts, thresholds
from .core import CONDITIONS, OUT, load_model, require


def excluded(path):
    name=path.relative_to(OUT).as_posix()
    return (name in ('output_inventory.csv','inventory_scope.json','verification_python.json','verification_optimized.json','postflight.json')
            or name.startswith('execution/postflight_') or name.startswith('checks/postflight'))


def inventory():
    rows=[]
    for path in sorted(OUT.rglob('*')):
        if path.is_file() and not excluded(path):
            rows.append(dict(path=path.relative_to(OUT).as_posix(),bytes=path.stat().st_size,sha256=sha(path)))
    save_csv(OUT/'output_inventory.csv',rows)
    save_json(OUT/'inventory_scope.json',dict(files=len(rows),bytes=sum(r['bytes'] for r in rows),
              excludes=['output_inventory.csv','inventory_scope.json','verification_python.json','verification_optimized.json',
                        'postflight.json','execution/postflight_*','checks/postflight*'],
              reason='Avoid circular verification receipts; frozen protocol/models and history remain individually checked.'))
    return rows


def equal_records(actual,expected,label):
    require(len(actual)==len(expected),f'{label}: record count differs')
    for a,b in zip(actual,expected):
        require(a.keys()==b.keys(),f'{label}: columns differ')
        for k in a:
            if isinstance(a[k],float):
                require(math.isclose(a[k],b[k],rel_tol=1e-12,abs_tol=1e-12),f'{label}: {k} differs')
            else:require(a[k]==b[k],f'{label}: {k} differs')


def verify(c):
    from .study import check_protocol,history_check
    check_protocol(c);history=history_check()
    formal=read_json(OUT/'training_complete.json');pilot=read_json(OUT/'pilot_complete.json')
    require(formal['complete'] and len(formal['runs'])==20 and formal['interactions']==100000000,'Formal completion missing')
    require(pilot['complete'] and len(pilot['runs'])==4 and pilot['interactions']==4000000,'Pilot completion missing')
    checkpoints=0;final_updates=0;formal_windows=0
    for r in formal['runs']:
        condition,seed=r['condition'],r['seed'];root=OUT/'runs'/condition/f'seed_{seed}'
        recorded=read_json(root/'metadata.json');require(recorded==r,'Run metadata differs')
        rows=read_csv(root/'training_windows.csv')
        require(sum(int(x['ticks']) for x in rows)==5000000,'Logged real interaction count differs')
        require(len(rows)==16667 and sum(x['budget_tail']=='True' for x in rows)==1 and int(rows[-1]['ticks'])==200,'300-tick windows / budget tail differ')
        require(r['full_windows']==16666 and r['partial_windows']==1 and r['tail_ticks']==200,'Run window accounting differs')
        formal_windows+=len(rows)
        require(int(sum(r['alpha']['actual_update_histogram']))==5000000,'Actual alpha histogram count differs')
        dev=read_csv(root/'development.csv')
        require([int(x['step']) for x in dev]==list(range(0,5000001,500000)),'Checkpoint schedule differs')
        for step in range(0,5000001,500000):
            path=root/'checkpoints'/f'step_{step:07d}.npz'
            agent,m=load_model(path,c)
            require(m['step']==step and m['training_seed']==seed and not m['pilot'],'Checkpoint identity differs')
            require(sha(path)==dev[step//500000]['checkpoint_sha256'],'Checkpoint file hash differs')
            checkpoints+=1
            if step==5000000:
                final_updates+=int(agent.counts.sum(dtype=np.uint64))
                require(sha(path)==r['model_sha256'],'Final model hash differs')
                # Recover the exact multiset of real pre-update counts from monotone counters.
                freq=np.bincount(agent.counts.ravel());survival=np.cumsum(freq[::-1],dtype=np.uint64)[::-1]
                n=np.arange(len(freq)-1);weights=survival[1:]
                alpha=.15*(1+n/1000)**(-.6) if condition.endswith('Decay') else np.full(len(n),.15)
                hist=np.histogram(alpha,bins=np.array(r['alpha']['bins']),weights=weights)[0]
                require(np.array_equal(hist,np.array(r['alpha']['actual_update_histogram'],np.uint64)),'Actual alpha histogram differs from update counters')
            dev_windows=read_csv(root/'development'/f'windows_{step:07d}.csv')
            require(len(dev_windows)==200 and [int(x['scenario_seed']) for x in dev_windows]==list(range(5100000,5100200)),
                    'Development scenario schedule differs')
            del agent
    require(final_updates==100000000 and checkpoints==220,'Final updates/checkpoints differ')
    for seed in c['study']['training_seeds']:
        paired=[r for r in formal['runs'] if r['seed']==seed]
        require(len(paired)==4 and len({r['target_stream_sha256'] for r in paired})==1,'Training target pairing differs')
    # Recompute primary metrics directly from complete saved physical rollouts.
    saved=read_json(OUT/'per_seed.json');recomputed=[];scenario_windows=0
    for split,count,start in (('final',500,5200000),('stress',200,5300000)):
        seen={}
        for seed in c['study']['training_seeds']:
            for condition in CONDITIONS:
                folder=OUT/'evaluation'/split;stem=f'{condition}_seed{seed}'
                with np.load(folder/f'{stem}_rollouts.npz',allow_pickle=False) as z:
                    initial,ticks,actions,seeds=(z[k] for k in ('initial','ticks','raw_actions','scenario_seeds'))
                    require(ticks.shape==(count,300,16) and list(seeds)==list(range(start,start+count)),'Rollout budget differs')
                    raw=ticks.copy();raw[:,:,6]=actions;raw[:,:,12]=((actions==5)&(raw[:,:,7]>0))
                    for variant,trace in (('original',raw),('guard',ticks)):
                        rows=[]
                        for i in range(count):
                            m=window_metrics(initial[i],trace[i],True)
                            require(seen.setdefault(int(seeds[i]),m['target_sha256'])==m['target_sha256'],'Saved targets are not paired')
                            rows.append(dict(scenario_seed=int(seeds[i]),**m))
                        summary=summarize(rows);expected=next(r for r in saved if (r['split'],r['condition'],r['seed'],r['variant'])==(split,condition,seed,variant))
                        for k,v in summary.items():
                            require(v==expected[k] or isinstance(v,float) and math.isclose(v,expected[k],abs_tol=1e-12),f'Saved metric differs: {split}/{stem}/{variant}/{k}')
                        recorded=read_csv(folder/f'{stem}_{variant}_windows.csv')
                        require(len(recorded)==count,'Window CSV count differs')
                        for a,b in zip(rows,recorded):
                            for k in ('scenario_seed','distance_sum','close_ticks','hits','shots','misses','target_sha256'):
                                require(str(a[k])==b[k],f'Window CSV mismatch: {k}')
                        scenario_windows+=count
        require(len(seen)==count,'Scenario pairing count differs')
    equal_records(aggregate(saved),read_json(OUT/'comparison.json'),'comparison')
    equal_records(thresholds(c,saved,aggregate(saved)),read_json(OUT/'thresholds.json'),'thresholds')
    gif_frames=0
    for receipt in read_json(OUT/'demos/decode_checks.json'):
        path=OUT/receipt['path'];require(sha(path)==receipt['sha256'],'GIF hash differs')
        duration=0
        with Image.open(path) as im:
            require(im.n_frames==301,'GIF incomplete')
            for i in range(301):im.seek(i);im.load();duration+=im.info['duration'];gif_frames+=1
        require(duration==20070 and receipt['decoded_all_pixels_exact'],'GIF duration / decoded receipt differs')
    inventory_count=0
    if (OUT/'output_inventory.csv').exists():
        entries=read_csv(OUT/'output_inventory.csv')
        for r in entries:
            path=OUT/r['path'];require(path.is_file() and path.stat().st_size==int(r['bytes']) and sha(path)==r['sha256'],'Inventory mismatch '+r['path'])
        require({r['path'] for r in entries}=={p.relative_to(OUT).as_posix() for p in OUT.rglob('*') if p.is_file() and not excluded(p)},'Inventory file set differs')
        inventory_count=len(entries)
    result=dict(complete=True,optimized=not __debug__,formal_interactions=100000000,pilot_interactions=4000000,
                final_updates=final_updates,checkpoints=checkpoints,formal_windows=formal_windows,
                evaluation_windows=scenario_windows,gif_frames_decoded=gif_frames,history=history,
                inventory_files_verified=inventory_count,training_interactions_this_check=0)
    save_json(OUT/('verification_optimized.json' if not __debug__ else 'verification_python.json'),result)
    return result
