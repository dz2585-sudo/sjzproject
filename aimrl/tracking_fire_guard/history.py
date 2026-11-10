"""Recompute historical evidence and freeze new, unused scenario ranges."""
import csv
import math
import statistics

from ..tracking_research.io import ROOT, read_json, save_csv, save_json, sha
from .replay import ratio, require

PRIOR = ROOT / 'results/tracking_shooting_tradeoff'
SPLITS = {'final':(4500000,500),'stress':(4600000,200)}
LOCAL_SETUP_LOG = 'execution/setup.log'
RELEASE_METADATA = ROOT / 'provenance/tracking_studies_release.json'
METRICS = ('D','C','H','A','shots','misses','base_return','longest_far_streak',
           'first_close_tick','acquired_fraction','alignment_opportunity_fraction',
           'fire_given_alignment','hit_nonclose_fraction')


def rows(path):
    with path.open(newline='',encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def stat(values):
    defined = [x for x in values if x is not None]
    return (statistics.mean(defined) if defined else None,
            statistics.stdev(defined) if len(defined)>1 else None, len(defined))


def values(windows):
    def count(key):
        return sum(int(w[key]) for w in windows)

    t,n,h,s,m,o = (count('ticks'),len(windows),count('hits'),count('shots'),
                   count('misses'),count('alignment_opportunities'))
    require(h+m==s, 'Shot count identity differs')
    return dict(D=count('distance_sum')/t, C=count('close_ticks')/t, H=h/n, A=ratio(h,s),
                shots=s/n, misses=m/n, base_return=(-.05*t+10*h-m)/n,
                longest_far_streak=count('longest_far_streak')/n,first_close_tick=count('first_close_tick')/n,
                acquired_fraction=count('acquired')/n,alignment_opportunity_fraction=o/t,
                fire_given_alignment=ratio(count('aligned_shots'),o),hit_nonclose_fraction=ratio(count('hit_nonclose'),h),
                windows=n,ticks=t,total_hits=h,total_shots=s,total_misses=m,
                total_close_ticks=count('close_ticks'),total_hit_nonclose=count('hit_nonclose'),
                total_alignment_opportunities=o,total_aligned_shots=count('aligned_shots'),
                total_unaligned_shots=count('unaligned_shots'))


def group(rows):
    summary = {}
    for key in METRICS:
        summary[key],summary[key+'_sd'],summary[key+'_defined_n']=stat([r[key] for r in rows])
    summary['total_hits']=sum(r['total_hits'] for r in rows)
    summary['total_shots']=sum(r['total_shots'] for r in rows)
    summary['A_pooled']=ratio(summary['total_hits'],summary['total_shots'])
    return summary


def number(raw):
    return None if raw in ('','NA',None) else float(raw)


def agrees(actual, recorded, name):
    expected=number(recorded)
    require(actual is expected if actual is None or expected is None else
            math.isclose(actual,expected,rel_tol=0,abs_tol=1e-10), f'Reconciliation mismatch: {name}')


def check_setup_omission(entry):
    """Validate the one documented, machine-local historical transcript omission."""
    omission=read_json(RELEASE_METADATA)['redacted_tradeoff_setup_log']
    require(entry['path']==omission['historical_path']==LOCAL_SETUP_LOG
            and entry['sha256']==omission['historical_sha256']
            and int(entry['bytes'])==omission['historical_bytes'],
            'Release omission does not match the historical inventory')
    replacement=ROOT/'provenance/tracking_tradeoff_setup_public.txt'
    require(omission['replacement_path']==replacement.relative_to(ROOT).as_posix()
            and sha(replacement)==omission['replacement_sha256'],
            'Redacted setup transcript hash differs')
    return omission


def historical_hashes(prior=PRIOR):
    evidence = rows(prior/'output_inventory.csv')
    result = {r['path']:r['sha256'] for r in evidence}
    if not (prior/LOCAL_SETUP_LOG).is_file():
        check_setup_omission(next(r for r in evidence if r['path']==LOCAL_SETUP_LOG))
        result.pop(LOCAL_SETUP_LOG)
    for name in ('output_inventory.csv','execution_checks.json'):
        result[name]=sha(prior/name)
    return result


def verify_history(expected, prior=PRIOR):
    for path,digest in expected.items():
        require(sha(prior/path)==digest,f'Historical file hash mismatch: {path}')


def previous(output, expected):
    verify_history(expected)
    protocol=read_json(PRIOR/'protocol.json')
    config=protocol['config']
    execution=read_json(PRIOR/'execution_checks.json')
    training=read_json(PRIOR/'training_complete.json')
    commands=read_json(PRIOR/'execution/commands.json')
    recorded=rows(PRIOR/'per_seed.csv')
    groups=rows(PRIOR/'comparison.csv')
    paired=rows(PRIOR/'paired_differences.csv')
    raw={}
    for r in recorded:
        split,c,s=r['split'],r['condition'],r['seed']
        name=f'{c}_seed{s}' if s else c
        windows=rows(PRIOR/'evaluation'/split/f'{name}_windows.csv')
        first=config['tradeoff'][split]['start']
        count=config['tradeoff'][split]['count']
        require([int(w['scenario_seed']) for w in windows]==list(range(first,first+count))
                and all(int(w['ticks'])==300 for w in windows), 'Historical window keys differ')
        v=values(windows)
        for k,x in v.items():
            agrees(x,r[k],(split,c,s,k))
        raw[split,c,s]=v
    rebuilt_groups=[]
    for r in groups:
        split,c=r['split'],r['condition']
        rebuilt=group([v for (sp,co,_),v in raw.items() if (sp,co)==(split,c)])
        for k,x in rebuilt.items():
            if k!='A_pooled':
                agrees(x,r[k],(split,c,k))
        rebuilt_groups.append({'split':split,'condition':c,**rebuilt})
    rebuilt_pairs=[]
    for r in paired:
        split,c,s=(r[k] for k in ('split','condition','seed'))
        a,b=raw[split,c,s],raw[split,'reference',s]
        delta={k:a[k]-b[k] if a[k] is not None and b[k] is not None else None for k in METRICS}
        for k,x in delta.items():
            agrees(x,r[k],(split,c,s,k))
        rebuilt_pairs.append({'split':split,'condition':c,'seed':s,**delta})
    save_csv(output/'historical_per_seed.csv',[dict(split=sp,condition=c,seed=s,**v) for (sp,c,s),v in raw.items()])
    save_csv(output/'historical_comparison.csv',rebuilt_groups)
    save_csv(output/'historical_paired_differences.csv',rebuilt_pairs)
    interactions,full,partial,partial_ticks=0,0,0,0
    require(len(training['runs'])==21 and len({(r['condition'],r['seed']) for r in training['runs']})==21,
            'Historical model matrix differs')
    for run in training['runs']:
        c,s=run['condition'],run['seed']
        path=PRIOR/'runs'/c/f'seed_{s}'
        log=rows(path/'training_windows.csv')
        end=0
        require(len(log)==16667,'Historical window count differs')
        for i,r in enumerate(log):
            n=int(r['ticks'])
            end+=n
            require(end==int(r['end_step']) and (n,r['partial'],r['truncated'])==
                    ((300,'False','True') if i<16666 else (200,'True','False')),
                    'Historical training window accounting differs')
            full+=i<16666
            partial+=i==16666
            partial_ticks+=n if i==16666 else 0
        require(end==run['interactions']==5000000,'Historical interaction budget differs')
        interactions+=end
    require((interactions,full,partial,partial_ticks)==(105000000,349986,21,4200)
            and execution['training_interactions']==interactions and execution['exact_checkpoint_reloads']==231
            and execution['development_windows']==46200 and execution['postflight']['final_tests_exit_code']==0,
            'Historical execution records differ')
    log=(PRIOR/'execution/targeted_tests.log').read_text(encoding='utf-8')
    require('test_zero_weight_matches_legacy_1001_updates' in log and
            'test_all_weighted_updates_match_python_in_greedy_regime' in log and log.strip().endswith('OK'),
            'Historical native training parity record missing')
    for path,digest in protocol['source_sha256'].items():
        require(sha(ROOT/path)==digest,f'Historical source hash mismatch: {path}')
    record={'previous_per_seed_rows':len(raw),'previous_comparison_rows':len(rebuilt_groups),
            'previous_paired_rows':len(rebuilt_pairs),'prior_hashes_verified':len(expected),
            'prior_publication_omissions':[] if LOCAL_SETUP_LOG in expected else [LOCAL_SETUP_LOG],
            'training_interactions':interactions,'full_windows':full,'partial_windows':partial,
            'partial_ticks':partial_ticks,'checkpoints':231,'development_windows':46200,
            'inner_training_seconds':training['elapsed_seconds'],
            'sum_per_run_seconds':sum(r['elapsed_seconds'] for r in training['runs']),
            'phase_seconds':execution['training_stage']['elapsed_seconds'],
            'outer_command_seconds':next(r['elapsed_seconds'] for r in commands if r['stage']=='train'),
            'backend':'Zig 0.15.2 C11 Windows library called from Python via ctypes. Python owns resets, NumPy PCG64, checkpoints, logging and evaluation orchestration.',
            'historical_python_training_parity':'Exact 1,001-step zero-weight Python parity and each weight in a greedy-heavy regime, including Q, counts and RNG states. These historical tests were audited, not rerun during this inference-only study.',
            'previous_command_exit_codes':{r['stage']:r['exit_code'] for r in commands},
            'kernel_sha256':protocol['kernel_sha256']}
    save_json(output/'historical_reconciliation.json',record)
    return config,record


def seed_audit(config):
    from scripts.check_tracking_tradeoff_seeds import audit, integers
    wanted={s for start,n in SPLITS.values() for s in range(start,start+n)}
    roots=[(ROOT,[])]
    if ((ROOT.parent.parent/'.git').is_dir()
            and (ROOT.parent.parent/'configs/tracking_research.json').is_file()):
        roots.append((ROOT.parent.parent,['--others','--exclude-standard']))
    found=audit(roots,wanted,replicating_guard=True)
    require(found['conflict_count']==0,f'Historical seed conflicts: {found["conflicts"]}')
    prior=0
    for path in sorted(PRIOR.rglob('*.csv')):
        with path.open(newline='',encoding='utf-8-sig') as stream:
            reader=csv.DictReader(stream)
            if 'scenario_seed' not in (reader.fieldnames or []):
                continue
            for row in reader:
                prior+=1
                require(int(row['scenario_seed']) not in wanted, f'Prior study seed collision: {path.name}')
    for interval in integers(config):
        require(not any(s in interval for s in wanted),'Prior declared study scenario seed collision')
    return {**found,'prior_study_scenario_rows_checked':prior,'ranges':SPLITS}
