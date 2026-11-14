"""Generate publication navigation from unrounded window counts; preserve evidence."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from aimrl.tracking_research.io import read_json,save_json,save_csv,sha

PUB=ROOT/'provenance/latest_studies_release'
B=ROOT/'results/tracking_boundary_stability';P=ROOT/'results/tracking_planning'
BORDER=('R-Fixed','R-Decay','B-Fixed','B-Decay')
PORDER=('Rule','R-Fixed','R-Decay','Plan-H1','Plan-H2','Plan-H3','Plan-H3-Zero','Hybrid-H3')


def require(value,message):
    if not value:raise ValueError(message)


def window_values(path):
    with path.open(newline='',encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    def total(k):return sum(int(r[k]) for r in rows)
    ticks,hits,shots=total('ticks'),total('hits'),total('shots')
    return dict(D=total('distance_sum')/ticks,C=total('close_ticks')/ticks,H=hits/len(rows),A=hits/shots if shots else None,
                windows=len(rows),ticks=ticks,hits=hits,shots=shots)


def recompute():
    boundary=[];planning=[]
    bs=read_json(B/'per_seed.json');ps=read_json(P/'per_policy.json')
    for split in ('final','stress'):
        for condition in BORDER:
            for seed in range(60,65):
                values=window_values(B/'evaluation'/split/f'{condition}_seed{seed}_guard_windows.csv')
                old=next(r for r in bs if (r['split'],r['condition'],r['seed'],r['variant'])==(split,condition,seed,'guard'))
                for k in ('D','C','H','A'):require(values[k]==old[k],f'Boundary raw-window reconciliation failed: {k}')
                boundary.append(dict(split=split,family=condition,policy=f'{condition}_s{seed}',seed=seed,**values))
        for old in [r for r in ps if r['split']==split]:
            values=window_values(P/'evaluation'/split/f'{old["policy"]}_windows.csv')
            for k in ('D','C','H','A'):require(values[k]==old[k],f'Planning raw-window reconciliation failed: {k}')
            planning.append(dict(split=split,family=old['family'],policy=old['policy'],seed=old['seed'],**values))
    def aggregate(rows):
        result=[]
        for split in ('final','stress'):
            for family in dict.fromkeys(r['family'] for r in rows):
                rs=[r for r in rows if r['split']==split and r['family']==family]
                item=dict(split=split,family=family,instances=len(rs))
                for k in ('D','C','H','A'):
                    values=[r[k] for r in rs if r[k] is not None]
                    item[k]=statistics.mean(values) if values else None
                    item[k+'_sd']=statistics.stdev(values) if len(values)>1 else None
                    item[k+'_defined']=len(values)
                result.append(item)
        return result
    return dict(boundary_instances=boundary,planning_instances=planning,boundary=aggregate(boundary),planning=aggregate(planning))


def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows])


def main_table(data,order):
    rows=[]
    for split in ('final','stress'):
        for family in order:
            r=next(r for r in data if r['split']==split and r['family']==family)
            def value(k,scale=1,digits=4):
                if r[k] is None:return 'NA'
                text=f'{scale*r[k]:.{digits}f}'
                if r[k+'_sd'] is not None:text+=f' ± {scale*r[k+"_sd"]:.{digits}f}'
                return text
            rows.append([split.title(),family,r['instances'],value('D'),value('C',100,3),value('H'),value('A',100,2)])
    return table(['Split','Method','Instances','D ↓','C (%) ↑','H / 300 ticks ↑','A (%)'],rows)


def gate(candidate,baseline):
    accuracy=candidate['A'] is not None and candidate['A']>=.98
    if 'instances' in candidate:accuracy=accuracy and candidate['A_defined']==candidate['instances']
    common=candidate['D']<=baseline['D'] and candidate['C']>=baseline['C']+.05 and accuracy
    retained=candidate['H']>=.98*baseline['H']
    return dict(practical=common and retained,strong=common and candidate['H']>=1.02*baseline['H'],
                absolute_C75=candidate['C']>=.75,C75_with_hits=candidate['C']>=.75 and retained,
                D_delta=candidate['D']-baseline['D'],C_delta=candidate['C']-baseline['C'],H_ratio=candidate['H']/baseline['H'])


def paired_tables(data):
    out=[];display=[]
    for study,rows in (('boundary',data['boundary_instances']),('planning',data['planning_instances'])):
        reference='R-Fixed' if study=='boundary' else 'R-Decay'
        for split in ('final','stress'):
            refs={r['seed']:r for r in rows if r['family']==reference and r['split']==split}
            candidates=[r for r in rows if r['split']==split and (study=='boundary' and r['family']=='R-Decay' or study=='planning' and r['family']=='Hybrid-H3')]
            for r in candidates:
                b=refs[r['seed']];g=gate(r,b)
                record=dict(study=study,split=split,seed=r['seed'],candidate=r['policy'],baseline=b['policy'],**g)
                out.append(record)
                display.append([study,split,r['seed'],f'{g["D_delta"]:+.5f}',f'{100*g["C_delta"]:+.4f}',f'{r["H"]-b["H"]:+.4f}',g['practical'],g['strong'],g['C75_with_hits']])
    return out,table(['Study','Split','Seed','ΔD','ΔC (pp)','ΔH','Relative gate','Stronger gate','C≥75% + hits'],display)


def redact(text):
    for prefix,label in ((str(ROOT),'<REPOSITORY_ROOT>'),(str(ROOT).replace('\\','/'),'<REPOSITORY_ROOT>'),
                         (str(ROOT.parents[1]),'<RESEARCH_PARENT>'),(str(Path.home()),'<USER_HOME>')):
        text=text.replace(prefix,label)
    return re.sub(r'[A-Za-z]:[\\/]+Users[\\/]+[^\\/\r\n" ]+', '<USER_HOME>',text)


def prepare():
    require((PUB/'original_documents.zip').is_file(),'Archive original documentation before publication edits')
    data=recompute();save_json(PUB/'recomputed_results.json',data)
    save_csv(PUB/'boundary_recomputed.csv',data['boundary']);save_csv(PUB/'planning_recomputed.csv',data['planning'])
    paired,pair_table=paired_tables(data);save_csv(PUB/'paired_improvements.csv',paired)
    omissions={}
    failed=['results/tracking_boundary_stability/execution/'+name for name in
            ('preflight_studies.txt','preflight_studies_optimized.txt','postflight_new_verify.txt','postflight_new_verify_optimized.txt')]
    failed+=['results/tracking_boundary_stability/native/build_attempt_1.json','results/tracking_boundary_stability/native/build_attempt_2.json']
    for number,relative in enumerate(failed,1):
        p=ROOT/relative;require(p.is_file(),'Expected original failure transcript missing')
        replacement=PUB/'public_transcripts'/f'failed_{number:02d}.txt'
        replacement.parent.mkdir(parents=True,exist_ok=True)
        header=f'PUBLICATION COPY: machine-local paths redacted.\nOriginal: {relative}\nOriginal SHA-256: {sha(p)}\nOriginal status/messages retained; this is not the original byte stream.\n\n'
        replacement.write_text(header+redact(p.read_text(encoding='utf-8')),encoding='utf-8')
        omissions[relative]=dict(kind='redacted_failure_transcript',original_sha256=sha(p),original_bytes=p.stat().st_size,
            reason='Machine-local failed command transcript; original retained locally, clearly labelled path-redacted publication copy supplied.',
            replacement_path=replacement.relative_to(ROOT).as_posix(),replacement_sha256=sha(replacement))
    for folder in (B/'native',P/'native'):
        for p in folder.iterdir():
            if p.suffix in ('.pdb','.lib'):
                omissions[p.relative_to(ROOT).as_posix()]=dict(kind='native_intermediate',original_sha256=sha(p),original_bytes=p.stat().st_size,
                   reason='Compiler debug/import intermediate; not needed to load the frozen DLL or rebuild from included C11 source.')
    bt=main_table(data['boundary'],BORDER);pt=main_table(data['planning'],PORDER)
    pm={(r['split'],r['family']):r for r in data['planning']};bm={(r['split'],r['family']):r for r in data['boundary']}
    h=pm['final','Hybrid-H3'];r=pm['final','R-Decay'];stress=pm['stress','Hybrid-H3']
    boundary_intro='''## Boundary information and learning-rate stability

**Question.** Does omitted boundary context or constant update size limit sustained tracking? Four conditions cross the original relative observation (already including heading and speed) or boundary-group augmentation with fixed alpha=0.15 or per-pair decay `0.15*(1+n/1000)^(-0.6)`. Rewards, actions, exploration and 5M budget are held fixed. Five paired training seeds (60–64) give **100,000,000 formal interactions**, plus an independent **4,000,000-interaction pilot**.

R-Decay improves close tracking and cross-seed C stability. The much larger boundary table performs worse at the same budget; those negative results remain included. Neither the relative joint improvement gate nor the absolute C≥75% target is reached in this study. Guarded results below are recalculated from raw windows; entries are mean ± sample SD across five trained seeds.

'''+bt+'''

![Boundary stability: paired training seeds](results/tracking_boundary_stability/figures/paired_seeds.png)

![Boundary stability: all 301 frames, fixed scenario 5400000](results/tracking_boundary_stability/demos/scenario_5400000.gif)

Each GIF has initial state plus all 300 ticks (301 frames, 20.07 s). Programmatic decoding is verified; actual-player inspection was not performed. [Study navigation](docs/tracking_boundary_stability.md) · [report](results/tracking_boundary_stability/results_summary.md) · [all data](results/tracking_boundary_stability/) · [other GIFs](results/tracking_boundary_stability/demos/) · [publication reproduction commands](docs/reproduce_latest_studies.md).

'''
    planning_intro='''## Short-horizon planning with learned terminal values

**Question.** How much can stochastic feedback planning improve the existing ten-action task, and what comes from horizon versus terminal value? The study reuses frozen models: **zero new training interactions and zero Q updates**. The **20 instances** are Rule (1), R-Fixed (5), R-Decay (5), Phi-terminal Plan-H1/H2/H3 (3), zero-terminal Plan-H3-Zero (1), and learned-terminal Hybrid-H3 (5). Final has **10,000 windows**; Stress has **4,000**, totaling **4,200,000 environment ticks**.

Planners observe full coordinates and a known stochastic model; R uses relative state. These information and computation advantages accompany cross-family differences. H1→H2→H3 within the same Phi family isolates horizon. Zero removes the distance terminal; Hybrid uses `max(Q_shaped)+Phi`, where `Phi=-distance` and task value = shaped value + Phi. In Stress the real turn/toggle probabilities are **0.10**, while every planning model remains **0.05**.

'''+pt+f'''

On this new Final, Hybrid-H3 has D={h['D']:.4f}, C={100*h['C']:.2f}%, H={h['H']:.3f}; H is **{100*(h['H']/r['H']-1):.2f}%** above the same-scene R-Decay mean. All five Hybrid instances reach C≥75% and retain at least 98% of their paired baseline hits. Stress Hybrid C is {100*stress['C']:.2f}%, below 75%. The prespecified relative gate requiring **C gain ≥5 percentage points** still fails. H1's high close score comes with substantial hit loss and is not joint improvement. Pure Rule/planners are single instances and have no invented training-seed SD.

**Different holdouts:** boundary-study R-Decay Final C is {100*bm['final','R-Decay']['C']:.2f}%; planning-study R-Decay Final C is {100*r['C']:.2f}%. Differences are computed within each study from unrounded counts, never across these tables.

Planning preparation is separately charged in the precomputation record: state backups, chance terms and action-value lookups are distinct from real environment ticks. Formal online use is 1,890,000 cached planning queries, not training updates. Full measured preparation/evaluation costs and timer-resolution caveats are in the report.

![Planning: horizon response with fixed state, model and terminal](results/tracking_planning/figures/horizon_response.png)

![Planning: all 301 frames, fixed scenario 6400000](results/tracking_planning/demos/scenario_6400000/synchronized.gif)

Programmatic decoding verified all 301 frames and their timing; actual-player inspection was not performed and no desktop/browser permission was requested. [Study navigation](docs/tracking_planning.md) · [report](results/tracking_planning/results_summary.md) · [paired results](results/tracking_planning/paired_differences.csv) · [other GIFs](results/tracking_planning/demos/) · [reproduce](docs/reproduce_latest_studies.md).

**Next work:** a controlled 2-cell movement-action experiment is proposed, not completed. Preserve the current-action planning baseline and test C and H jointly; these results do not establish a theoretical ceiling.

'''
    with zipfile.ZipFile(PUB/'original_documents.zip') as z:original=z.read('README.md').decode('utf-8')
    status='**Current status:** the boundary-stability study completed 100M formal + 4M pilot interactions. The zero-training short-horizon planning study completed 20 policy instances on new Final/Stress scenarios; Hybrid reaches the 75% Final close target while retaining hits, but the relative +5pp joint gate remains unmet. A 2-cell action experiment remains future work.'
    original=re.sub(r'\*\*Current status:\*\*[^\n]+',status,original,count=1)
    original=original.replace('[Representation benchmark](#representation-benchmark)', '[Research index](docs/research_index.md) · [Boundary stability](#boundary-information-and-learning-rate-stability) · [Short-horizon planning](#short-horizon-planning-with-learned-terminal-values) · [Representation benchmark](#representation-benchmark)',1)
    historical='![Full continuous-tracking comparison: archived Potential, Relative Base, Relative Potential, and predictive rule](results/tracking_improvement/render/seed3601000_main_comparison.gif)'
    original=original.replace(historical+'\n','',1)
    opening='Earlier full representation-study rollout (historical reference; its different budget is described below):\n\n'+historical+'\n\n'
    readme=original.replace('## Representation benchmark',opening+boundary_intro+planning_intro+'## Representation benchmark',1)
    (ROOT/'README.md').write_text(readme,encoding='utf-8',newline='\n')
    (ROOT/'docs/tracking_boundary_stability.md').write_text('# Boundary information and learning-rate stability\n\n'+boundary_intro.replace('## Boundary information and learning-rate stability\n\n','').replace('](results/','](../results/').replace('](docs/','](')+
        '\n## Paired checks from unrounded records\n\n'+pair_table+'\n\n[Protocol](../results/tracking_boundary_stability/protocol.json) · [source](../aimrl/tracking_boundary_stability/) · [config](../configs/tracking_boundary_stability.json) · [tests](../tests/test_tracking_boundary_stability.py) · [full results](../results/tracking_boundary_stability/comparison.csv)\n',encoding='utf-8',newline='\n')
    (ROOT/'docs/tracking_planning.md').write_text('# Short-horizon planning with learned terminal values\n\n'+planning_intro.replace('## Short-horizon planning with learned terminal values\n\n','').replace('](results/','](../results/').replace('](docs/','](')+
        '\n## Same-model paired improvements\n\n'+pair_table+'\n\n[Protocol](../results/tracking_planning/protocol.json) · [source](../aimrl/tracking_planning/) · [config](../configs/tracking_planning.json) · [tests](../tests/test_tracking_planning.py) · [per-policy data](../results/tracking_planning/per_policy.csv) · [thresholds](../results/tracking_planning/thresholds.csv)\n',encoding='utf-8',newline='\n')
    (ROOT/'docs/research_index.md').write_text('''# AimRL research index

| Research | Question and outcome | Evidence |
| --- | --- | --- |
| [Boundary information and learning-rate stability](tracking_boundary_stability.md) | Does boundary context or update decay help? R decay improves tracking/stability; the larger B table is worse at fixed budget. The declared gates remain unmet. | [Report](../results/tracking_boundary_stability/results_summary.md), [data](../results/tracking_boundary_stability/), [GIFs](../results/tracking_boundary_stability/demos/) |
| [Short-horizon planning with learned terminal values](tracking_planning.md) | Is there room with the existing actions? Hybrid improves all five model pairs and reaches C≥75% with retained hits on Final. Relative +5pp and Stress 75% remain unmet. | [Report](../results/tracking_planning/results_summary.md), [data](../results/tracking_planning/), [GIFs](../results/tracking_planning/demos/) |
| [Tracking–shooting reward trade-offs](tracking_shooting_tradeoff.md) | Same-action reward comparison across 21 trained models. | [Report](../results/tracking_shooting_tradeoff/results_summary.md) |
| [Firing guard](tracking_fire_guard.md) | Remove empty shots without changing physical tracking. | [Report](../results/tracking_fire_guard/results_summary.md) |
| [Representation study](tracking_improvement.md) | Compare absolute and relative observations under matched budgets. | [Report](../results/tracking_improvement/results_summary.md) |

[Reproduction entry points](reproduce_latest_studies.md) distinguish saved-record analysis, frozen-model reevaluation, native rebuilding and optional exact training replication. [Publication evidence](../provenance/latest_studies_release/README.md) records all actual payload hashes, permitted documentation changes, omissions and checks. The 2-cell action study is future work.
''',encoding='utf-8',newline='\n')
    with zipfile.ZipFile(PUB/'original_documents.zip') as z:ignore=z.read('.gitignore').decode('utf-8')
    ignore+='\n# Local native intermediates from the two latest studies; frozen DLLs are published.\nresults/tracking_boundary_stability/native/*.pdb\nresults/tracking_boundary_stability/native/*.lib\nresults/tracking_planning/native/*.pdb\nresults/tracking_planning/native/*.lib\n'
    ignore+='\n# Original machine-local failed transcripts; labelled publication copies are in provenance.\n'+'\n'.join(failed)+'\n'
    (ROOT/'.gitignore').write_text(ignore,encoding='utf-8',newline='\n')
    with zipfile.ZipFile(PUB/'original_documents.zip') as z:
        changes={name:dict(original_sha256=hashlib.sha256(z.read(name)).hexdigest(),published_sha256=sha(ROOT/name),
                          scope='Publication navigation / exclusion rules only; original bytes archived, no experiment core change.') for name in ('README.md','.gitignore')}
    release=dict(schema='aimrl_latest_studies_publication_v1',base_commit=read_json(PUB/'start.json')['head'],
       original_documents_archive='provenance/latest_studies_release/original_documents.zip',original_documents_archive_sha256=sha(PUB/'original_documents.zip'),
       documentation_changes=changes,omissions=omissions,identity=dict(name='dz2585-sudo',email='232613424+dz2585-sudo@users.noreply.github.com'),
       core_policy='No frozen environment, action, learning, model, protocol or expected experiment hash changes.',
       privacy='No credential/token/private-key matches; only public noreply author email. Frozen protocol/runtime metadata retains research-related workspace/toolchain paths to preserve evidence bytes. Verbose failed transcripts get labelled path-redacted public copies. No unrelated personal files are published.',
       native='Original frozen boundary.dll and planning.dll are retained with hashes. Only PDB/LIB intermediates are omitted. Rebuild C11 without fast-math or FP contraction; read-only parity can use rebuilt engines.',
       hash_scope='Publication payload manifest excludes only itself and manifest_scope.json. Post-push receipts are local and cannot be included in their own commit.')
    save_json(PUB/'release.json',release)
    print(f'Recomputed 40 boundary and 40 planning instances from windows; documented {len(omissions)} non-core omissions.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--analyze-only',type=Path);args=parser.parse_args()
    if args.analyze_only:
        args.analyze_only.mkdir(parents=True,exist_ok=False);save_json(args.analyze_only/'recomputed_results.json',recompute())
    else:prepare()
