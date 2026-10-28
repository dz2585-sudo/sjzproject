"""Final accounting, preserved-history checks, and reproducible verification evidence."""
import json
import math
import re
import subprocess
import sys
import time

import numpy as np

from .agent import ResearchAgent
from .io import ROOT, git_state, peak_memory_bytes, read_csv, read_json, save_csv, save_json, sha
from .workflow import METHODS, pair_rows, seeds, verify_protocol, verify_run, verify_stage_files


def validate_windows(rows, expected):
    keys = [int(r['scenario_seed']) for r in rows]
    if len(keys)!=len(set(keys)) or set(keys)!=set(expected): raise ValueError('Missing or duplicate scenario records')
    for row in rows:
        if int(row['steps'])!=300: raise ValueError('Incomplete evaluation window')
        if int(row['shots'])!=int(row['hit_count'])+int(row['miss_count']): raise ValueError('Shot identity failed')
        if not math.isclose(float(row['base_return']),10*int(row['hit_count'])-int(row['miss_count'])-15,abs_tol=1e-7): raise ValueError('Evaluation return identity failed')
        hist = json.loads(row['distance_histogram'])
        if len(hist)!=49 or sum(hist)!=300 or sum(i*n for i,n in enumerate(hist))!=int(row['distance_sum']): raise ValueError('Exact histogram accounting failed')
        if sum(json.loads(row['action_counts']))!=300: raise ValueError('Action count mismatch')


def validate_summary(summary):
    slices = summary['slices']; n = summary['ticks']
    if slices['all']['ticks']!=n or sum(summary['distance_histogram'])!=n: raise ValueError('Summary tick count mismatch')
    for left,right in [('boundary_near','interior'),('target_near','target_interior'),('crosshair_near','crosshair_interior'),('speed_1','speed_2')]:
        if sum(slices.get(k,{}).get('ticks',0) for k in (left,right))!=n: raise ValueError('Slice denominator mismatch')
    for item in slices.values():
        if sum(item['distance_histogram'])!=item['ticks']: raise ValueError('Slice histogram mismatch')


def audit(config,output):
    protocol = verify_protocol(config,output,True)
    runs = []; pairing = {s:{} for s in (10,11,12)}
    dev_pairing = {}; dev_windows = dev_summaries = 0
    for method in METHODS:
        for seed in (10,11,12):
            run = output/'runs'/method/f'seed_{seed}'
            meta = verify_run(config,method,seed,run); runs.append(meta)
            for row in read_csv(run/'training_windows.csv'):
                step = int(row['end_step']); digest = row['target_trajectory_sha256']
                if step in pairing[seed] and pairing[seed][step]!=digest: raise ValueError('Training paths not paired')
                pairing[seed][step] = digest
            for item in read_json(run/'validation.json'):
                t = item['global_step']; directory = output/'evaluation/development'/method/f'seed_{seed}'
                if read_json(directory/f'checkpoint_{t:07d}.json')!=item: raise ValueError('Checkpoint summary mismatch')
                rows = read_csv(directory/f'windows_{t:07d}.csv')
                validate_windows(rows,seeds(config,'development')); pair_rows(dev_pairing,rows); validate_summary(item['metrics'])
                dev_windows += len(rows); dev_summaries += 1
    if len(runs)!=12 or sum(r['interactions'] for r in runs)!=60000000: raise ValueError('Formal matrix incomplete')
    if (dev_summaries,dev_windows)!=(132,26400): raise ValueError('Development budget differs')
    split_checks = {}
    for split,expected in [('final',8000),('stress',3200)]:
        directory = output/'evaluation'/split; stage = verify_stage_files(directory)
        summaries = read_json(directory/'summary.json'); target_pairing = {}; total = 0
        if len(summaries)!=16: raise ValueError('Reference replication or missing policy')
        for key,summary in summaries.items():
            rows = read_csv(directory/f'{key}_windows.csv'); validate_windows(rows,seeds(config,split))
            pair_rows(target_pairing,rows); validate_summary(summary); total += len(rows)
        if total!=expected: raise ValueError('Independent split budget differs')
        split_checks[split] = {'windows':total,'paired_scenarios':len(target_pairing),'all_target_hashes_match':True}
    verify_stage_files(output/'diagnostics')
    pilots = [verify_run(config,m,99,output/'pilots'/m,100000) for m in METHODS]
    changed = []
    for name,digest in protocol['starting_snapshot']['historical_sha256'].items():
        if sha(ROOT/name)!=digest:
            if name=='README.md': changed.append(name)
            else: raise ValueError('Historical bytes changed: '+name)
    for model in protocol['historical_models'].values():
        if sha(model['path'])!=model['package_sha256']: raise ValueError('Archived package changed')
    if git_state()['identity']!=protocol['starting_snapshot']['git_identity']: raise ValueError('Git identity changed')
    if git_state()['head']!=protocol['starting_snapshot']['local_head']: raise ValueError('Git HEAD changed')
    from PIL import Image
    clips = read_json(output/'render/render_checks.json')['clips']
    if len(clips)!=4: raise ValueError('Missing full-length GIF')
    for clip in clips:
        durations = []
        with Image.open(ROOT/clip['path']) as gif:
            if gif.n_frames!=301: raise ValueError('Incorrect GIF frame count')
            for i in range(gif.n_frames): gif.seek(i); gif.load(); durations.append(gif.info.get('duration',0))
        if sum(durations)!=20070 or set(durations)!={60,70}: raise ValueError('Incorrect GIF delays')
        clip['sha256'] = sha(ROOT/clip['path'])
    commands = []
    for args,label in [(['scripts/check_setup.py'],'setup'),(['-m','unittest','discover','-s','tests','-p','test_*.py','-v'],'tests')]:
        started = time.perf_counter()
        result = subprocess.run([sys.executable,*args],cwd=ROOT,capture_output=True,text=True)
        log = output/'stage_execution'/f'final_{label}.txt'; log.write_text(result.stdout+result.stderr,encoding='utf-8')
        commands.append({'command':[sys.executable,*args],'exit_code':result.returncode,'elapsed_seconds':time.perf_counter()-started,'log':str(log)})
        if result.returncode: raise RuntimeError('Final verification command failed: '+label)
    # One bounded saved-model replay, without learning updates or global RNG changes.
    from ..tracking25.env import TrackingEnv
    a = ResearchAgent(config,'relative'); a.load_package(output/'runs/relative_potential/seed_10/q_table.npz',True)
    before = a.snapshot(); env = TrackingEnv(config); state,_ = env.reset(3601000); demo = []
    for t in range(30):
        action = a.greedy(state); state,reward,term,trunc,info = env.step(action)
        demo.append({'tick':t+1,'action':action,'distance':info['distance'],'hit':info['hit'],'base_reward':reward})
    if a.snapshot()!=before: raise ValueError('Saved-model demo mutated learner')
    save_csv(output/'render/bounded_saved_model_demo.csv',demo); del a
    records = [read_json(p) for p in (output/'stage_execution').glob('*.json')]
    checks = {'complete':True,'formal_runs':12,'formal_interactions':60000000,'pilot_interactions':sum(p['interactions'] for p in pilots),
              'runs':runs,'formal_elapsed_seconds':sum(r['elapsed_seconds'] for r in runs),
              'development_summaries':dev_summaries,'development_windows':dev_windows,'development_ticks':dev_windows*300,
              'final_windows':split_checks['final']['windows'],'stress_windows':split_checks['stress']['windows'],
              'training_target_paths_paired':{str(k):len(v) for k,v in pairing.items()},'evaluation_pairing':split_checks,
              'historical_files_verified':len(protocol['starting_snapshot']['historical_sha256'])-len(changed),
              'intentional_existing_file_edits':changed,'initial_free_disk_bytes':protocol['free_disk_bytes'],
              'peak_working_set_bytes':max([peak_memory_bytes()]+[r['peak_working_set_bytes'] for r in records]),
              'commands':commands,'stages':records,'clips':clips,'bounded_demo_ticks':30,'bounded_demo_hits':sum(r['hit'] for r in demo),
              'gui_verified':False,'git_head_and_identity_unchanged':True,'no_remote_publication':True}
    from .report import write_report
    write_report(config,output,checks)
    links = []
    for path in [ROOT/'README.md',ROOT/'docs/tracking_improvement.md',output/'results_summary.md']:
        for target in re.findall(r'\]\(([^)]+)\)',path.read_text(encoding='utf-8')):
            if target.startswith(('http:','https:','#','mailto:')): continue
            resolved = path.parent/target.split('#')[0]
            if not resolved.exists(): raise ValueError('Broken documentation link: '+str(path)+' -> '+target)
            links.append({'source':str(path),'target':target})
    checks['local_links_verified'] = links
    new_files = list((ROOT/'aimrl/tracking_research').glob('*.py'))+[ROOT/'configs/tracking_research.json',ROOT/'tests/test_tracking_research.py',ROOT/'docs/tracking_improvement.md']
    new_files += [p for p in output.rglob('*') if p.is_file() and p.suffix in ('.md','.json','.csv','.txt')]
    chinese = [str(p) for p in new_files if re.search(r'[\u3400-\u9fff]',p.read_text(encoding='utf-8'))]
    if chinese: raise ValueError('Chinese text in new study content: '+str(chinese))
    checks['english_text_files_scanned'] = len(new_files)
    checks['study_files_sha256'] = {p.relative_to(output).as_posix():sha(p) for p in output.rglob('*') if p.is_file() and p.name!='execution_checks.json'}
    save_json(output/'execution_checks.json',checks)
    print('Final audit passed: 60M formal updates, 400k pilots,132 checkpoints,26400 development windows,8000 final,3200 stress, four full GIFs.',flush=True)
