"""Narrow release adapter: original evidence/hashes retained, receipts redirected.

No training/evaluation entry point is exposed. Documentation changes require both
archived baseline bytes and reviewed current bytes; models cannot be omitted.
"""
import argparse,hashlib,json,os,subprocess,sys,time,unittest,zipfile
from pathlib import Path
import numpy as np
from .tracking_research.io import ROOT,read_json,read_csv,save_json,save_csv,sha
from .tracking_planning.reference import require

RELEASE=ROOT/'provenance/action_resolution_release'
DOCUMENTS={'README.md','.gitignore','docs/research_index.md','docs/tracking_planning.md','docs/reproduce_latest_studies.md'}

def contract():return read_json(RELEASE/'release.json')

def verify_doc(relative,expected,root=ROOT,release=None):
    require(relative in DOCUMENTS,'Unapproved release document scope '+relative)
    c=contract() if release is None else release;entry=c['documentation_changes'].get(relative)
    require(entry is not None and expected==entry['original_sha256'],'Unapproved release document change '+relative)
    require(sha(root/relative)==entry['published_sha256'],'Reviewed release document changed '+relative)
    archive=root/c['original_documents_archive'];require(sha(archive)==c['original_documents_archive_sha256'],'Release document archive changed')
    with zipfile.ZipFile(archive) as z:raw=z.read(relative)
    require(hashlib.sha256(raw).hexdigest()==expected,'Archived baseline document mismatch')
    return entry

def omission(relative,expected,root=ROOT,release=None):
    c=contract() if release is None else release;e=c['omissions'].get(relative)
    require(relative=='results/tracking_action_resolution/delivery_state.json' and e is not None and e['kind']=='private_identity_snapshot','Unapproved omission '+relative)
    require(e['original_sha256']==expected,'Original omitted digest mismatch')
    require(sha(root/e['replacement_path'])==e['replacement_sha256'],'Redacted copy changed')
    return e

def protected(mapping):
    counts=dict(unchanged_files=0,documents=[],omissions=[])
    for p,h in mapping.items():
        f=ROOT/p
        if f.exists() and sha(f)==h:counts['unchanged_files']+=1
        elif f.exists():verify_doc(p,h);counts['documents'].append(p)
        else:omission(p,h);counts['omissions'].append(p)
    return counts

def install():
    """Install only runtime adapters. Historical source and contracts are unmodified."""
    from . import latest_publication as old
    from .tracking_action_resolution import study
    old_document=old.verify_document
    def chained(relative,expected,root=ROOT,release=None):
        if Path(root).resolve()!=ROOT or (release is not None and release!=old.contract()) or relative not in contract()['documentation_changes']:
            return old_document(relative,expected,root,release)
        new=contract()['documentation_changes'][relative]
        verify_doc(relative,new['original_sha256'])
        if expected==new['original_sha256']:return new
        previous=old.contract();entry=previous['documentation_changes'].get(relative)
        require(entry is not None and entry['original_sha256']==expected and entry['published_sha256']==new['original_sha256'],'Unapproved historical document chain '+relative)
        zpath=ROOT/previous['original_documents_archive'];require(sha(zpath)==previous['original_documents_archive_sha256'],'Historical original archive changed')
        with zipfile.ZipFile(zpath) as z:require(hashlib.sha256(z.read(relative)).hexdigest()==expected,'Historical original bytes differ')
        return dict(**new,historical_original_sha256=expected,chain_verified=True)
    old.verify_document=chained
    def history():return protected(read_json(study.OUT/'historical_sha256_before.json'))
    study.history_check=history
    return old,study

def raw_audit(output):
    """Raw tick sums, independent from printed report values; no environment."""
    from .tracking_action_resolution.core import OUT,config,methods,arrays
    from .tracking_action_resolution.analysis import aggregate
    c=config();policies=[];denominators=[];total=0
    for split in ('final','stress'):
        old={r['policy']:r for r in read_json(OUT/'per_policy.json') if r['split']==split}
        usage=read_csv(OUT/'action_usage.csv')
        for spec in methods(c):
            with np.load(OUT/'evaluation'/split/f'{spec["id"]}_rollouts.npz',allow_pickle=False) as z:initial=z['initial'];ticks=z['ticks']
            count=len(ticks);flat=ticks.reshape(-1,22);shots=int(np.count_nonzero(flat[:,6]==5));hits=int(flat[:,11].sum(dtype=np.int64))
            r=dict(split=split,policy=spec['id'],family=spec['family'],seed=spec['seed'],windows=count,ticks=len(flat),
                distance_sum=int(flat[:,9].sum(dtype=np.int64)),close_ticks=int(np.count_nonzero(flat[:,9]<=1)),hits=hits,shots=shots,
                D=float(flat[:,9].sum(dtype=np.int64)/len(flat)),C=float(np.count_nonzero(flat[:,9]<=1)/len(flat)),H=hits/count,A=hits/shots if shots else None)
            require(all(r[k]==old[spec['id']][k] for k in ('D','C','H','A')),'Independent raw counter discrepancy')
            policies.append(r);total+=count
            before=np.concatenate([initial[:,None,:],ticks[:,:-1,:6]],axis=1);a=ticks[:,:,6]
            boundary=np.any(np.minimum(before[:,:,:4],24-before[:,:,:4])<=3,axis=2)
            for group,mask in [('all',np.ones(a.shape,bool)),('pre_speed_1',before[:,:,5]==1),('pre_speed_2',before[:,:,5]==2),('boundary',boundary),('interior',~boundary)]:
                nt=int(mask.sum());medium=int(np.count_nonzero(mask&(a>=10)));stay=int(np.count_nonzero(mask&(a==4)))
                u=next(x for x in usage if x['split']==split and x['policy']==spec['id'] and x['group']==group)
                actual=[int(np.count_nonzero(mask&(a>=10)&(ticks[:,:,21]==v))) for v in (0,1,2)]
                require(nt==int(u['ticks']) and medium/nt==float(u['medium_fraction']) and stay/nt==float(u['stay_fraction']),'Action denominator mismatch')
                require(actual==[int(u[f'medium_actual_{v}']) for v in (0,1,2)],'Actual displacement count differs')
                denominators.append(dict(split=split,policy=spec['id'],group=group,ticks=nt,medium=medium,stay=stay,medium_actual_0=actual[0],medium_actual_1=actual[1],medium_actual_2=actual[2]))
    require(total==8400,'Formal window count differs')
    means=[];differences=[]
    for split in ('final','stress'):
        for terminal in ('Plan','Hybrid'):
            groups={n:[r for r in policies if r['split']==split and r['family']==f'{terminal}-H3-A{n}'] for n in (10,14)}
            m={n:{k:float(np.mean([r[k] for r in groups[n]])) for k in ('D','C','H','A')} for n in groups}
            for n in groups:means.append(dict(split=split,family=f'{terminal}-H3-A{n}',instances=len(groups[n]),**m[n],**{k+'_sd':float(np.std([r[k] for r in groups[n]],ddof=1)) if len(groups[n])>1 else None for k in ('D','C','H','A')}))
            byseed={r['seed']:r for r in groups[10]};ratios=[r['H']/byseed[r['seed']]['H']-1 for r in groups[14]]
            d=dict(split=split,terminal=terminal,**{'delta_'+k:m[14][k]-m[10][k] for k in ('D','C','H','A')},
                H_percent_ratio_of_means=100*(m[14]['H']/m[10]['H']-1),H_percent_mean_of_paired_ratios=100*float(np.mean(ratios)),A14_C=m[14]['C'])
            differences.append(d)
    save_json(output/'raw_recomputed.json',dict(policies=policies,means=means,differences=differences,windows=total,environment_ticks=total*300,
        H_percent_definition='100*(mean_source_models(H_A14)/mean_source_models(H_A10)-1). Per-source ratios reported separately, never substituted for this aggregate.',
        bootstrap='Frozen analysis replays 2000 scene/model paired resamples, common scene indices across all model pairs; no ticks resampled.'))
    save_csv(output/'raw_recomputed.csv',policies);save_csv(output/'paired_aggregate_effects.csv',differences);save_csv(output/'action_denominators.csv',denominators)
    pre=read_json(OUT/'precomputation.json');counts=arrays(c,14)[2]
    expected=dict(backups=3125000*36,chance_terms=int(counts.sum())*625*36,action_lookups=3125000*18*(10+14))
    require(expected==dict(backups=pre['A10_value_backups']+pre['A14_value_backups'],chance_terms=pre['chance_terms'],action_lookups=pre['action_lookups']),'Precomputation accounting mismatch')
    save_json(output/'cost_definition_check.json',dict(**expected,new_training_interactions=0,Q_updates=0,
        definition='One backup per cursor625 × target5000 state and horizon. Six terminals × 3 depths × 2 action sets. Chance expectation is shared over action preimages; action lookups enumerate 10 or 14.'))
    return differences

def verify_new(output,native=None):
    old,study=install()
    from .tracking_action_resolution import verify,core
    if native:core.LIBRARY=Path(native).resolve()
    verify.history_check=study.history_check
    original_save=verify.save_json
    def receipt(path,value):
        p=Path(path)
        if p.parent==core.OUT:
            require(p.name=='historical_integrity.json','Unexpected frozen output write');p=output/'historical_integrity.json'
        original_save(p,value)
    verify.save_json=receipt
    result=verify.verify(core.config(),output/'verification.json');raw_audit(output)
    return result

def verify_gifs(output):
    """Render from saved traces instead of executing a controller/environment."""
    from PIL import Image
    from .tracking_action_resolution import visuals,core
    c=core.config();records=[]
    for seed in c['planning']['demo_seeds']:
        source=core.OUT/'demos'/f'scenario_{seed}'
        with np.load(source/'rollouts.npz',allow_pickle=False) as z:initial=z['initial'];ticks=z['ticks'];ids=z['policy_ids'];times=z['decision_ns']
        class SavedEnv:
            def __init__(self,*args):pass
            def reset(self,scenario):require(scenario==seed,'Unexpected reset');return tuple(map(int,initial[0])),{}
        class SavedNative:
            def __init__(self,*args):pass
            def run(self,env,start,p):
                j=list(ids).index(p.spec['id']);require(tuple(start)==tuple(initial[j]),'Saved demo initial differs')
                return ticks[j].copy(),times[j].copy()
        visuals.TrackingEnv=SavedEnv;visuals.Native=SavedNative
        rendered=output/f'scenario_{seed}';receipt=visuals.render_demo(c,seed,rendered)
        with Image.open(source/'synchronized.gif') as original,Image.open(rendered/'synchronized.gif') as reconstructed:
            require(original.n_frames==reconstructed.n_frames==301,'GIF frames differ');duration=0
            for i in range(301):
                original.seek(i);reconstructed.seek(i)
                require(original.convert('RGB').tobytes()==reconstructed.convert('RGB').tobytes(),'Saved-trace rendered pixels differ')
                require(original.info['duration']==reconstructed.info['duration'],'GIF timing differs');duration+=original.info['duration']
            require(duration==20070,'GIF duration differs')
        records.append(dict(seed=seed,frames=301,duration_ms=duration,all_pixels_equal=True,environment_ticks=0,actual_player=False))
    save_json(output/'saved_trace_gif_check.json',records)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['tests','tests-O','history','boundary','planning','action','gifs','inputs']);p.add_argument('--output',type=Path,required=True);p.add_argument('--native-directory',type=Path)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);old,study=install()
    if a.phase in ('tests','tests-O'):
        suite=unittest.defaultTestLoader.discover('tests',pattern='test_tracking_action_resolution.py' if a.phase=='tests-O' else 'test*.py')
        r=unittest.TextTestRunner(verbosity=2).run(suite);save_json(out/'tests.json',dict(tests=r.testsRun,failures=len(r.failures),errors=len(r.errors),skipped=len(r.skipped)));sys.exit(not r.wasSuccessful())
    elif a.phase=='history':old.historical_studies_verify(out/'historical_studies')
    elif a.phase=='boundary':old.boundary_verify(out/'boundary')
    elif a.phase=='planning':old.planning_verify(out/'planning',a.native_directory/'planning.dll' if a.native_directory else None)
    elif a.phase=='action':verify_new(out,a.native_directory/'action_resolution.dll' if a.native_directory else None)
    elif a.phase=='gifs':verify_gifs(out)
    else:
        save_json(out/'historical.json',protected(read_json(study.OUT/'historical_sha256_before.json')))
        save_json(out/'research.json',protected(read_json(RELEASE/'research_input_sha256.json')))
        for e in contract()['omissions'].values():require(sha(ROOT/e['replacement_path'])==e['replacement_sha256'],'Replacement changed')
        for r in read_csv(study.OUT/'output_inventory.csv'):
            f=ROOT/r['path']
            if f.exists():require(f.stat().st_size==int(r['bytes']) and sha(f)==r['sha256'],'Frozen delivery manifest mismatch '+r['path'])
            else:omission(r['path'],r['sha256'])
        save_json(out/'old_release_inputs.json',old.verify_release_inputs());save_json(out/'old_links.json',old.check_links())

if __name__=='__main__':main()
