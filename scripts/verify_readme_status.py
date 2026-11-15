"""Audit the scoped README revision; no scientific code is imported or executed."""
import argparse,hashlib,json,re,subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE='80c25e452e38e46ee18a04e6e2fb668bb93d2367'
RECORD=ROOT/'provenance/readme_status_clarification'

def require(ok,message):
    if not ok:raise ValueError(message)

def read(path):return json.loads((ROOT/path).read_text(encoding='utf-8'))
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT)

def verify():
    revision=read('provenance/readme_status_clarification/revision.json')
    require(revision['baseline_commit']==BASE,'Unexpected baseline')
    require(hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==revision['verifier_sha256'],'Revision verifier changed after review')
    original=git('show',BASE+':README.md');current=(ROOT/'README.md').read_bytes()
    require(hashlib.sha256(original).hexdigest()==revision['original_readme_sha256'],'Old README object differs')
    previous=read('provenance/action_resolution_release/release.json')
    require(previous['documentation_changes']['README.md']['published_sha256']==revision['original_readme_sha256'],'Old expected README digest was changed')
    rebuilt=original.decode('utf-8');ops=revision['operations']
    require(len(ops)==4 and all(o['before'].startswith('- ') for o in ops[:3]) and ops[3]['before'].startswith('## Installation and verification'),'Edit scope differs')
    for o in ops:
        require(rebuilt.count(o['before'])==1,'Ambiguous historical text');rebuilt=rebuilt.replace(o['before'],o['after'],1)
    require(rebuilt.encode('utf-8')==current and hashlib.sha256(current).hexdigest()==revision['revised_readme_sha256'],'README differs beyond the declared substitutions')
    rows=[]
    for item in git('ls-tree','-r','-z',BASE).split(b'\0'):
        if item:
            meta,path=item.split(b'\t',1);mode,kind,digest=meta.decode().split();p=path.decode()
            if p!='README.md':rows.append((p,digest))
    actual=subprocess.check_output(['git','hash-object','--stdin-paths'],input=('\n'.join(p for p,_ in rows)+'\n').encode(),cwd=ROOT)
    hashes=actual.decode().splitlines();require(len(hashes)==len(rows),'Baseline file count differs')
    changed=[p for (p,expected),actual in zip(rows,hashes) if expected!=actual]
    require(not changed,'Protected baseline files changed: '+str(changed[:5]))
    configs=[('configs/tracking_research.json','study',[10,11,12]),('configs/tracking_shooting_tradeoff.json','tradeoff',[20,21,22]),('configs/tracking_boundary_stability.json','study',[60,61,62,63,64])]
    for file,key,seeds in configs:require(read(file)[key]['training_seeds']==seeds,'Seed-count statement inconsistent')
    rep=read('results/tracking_improvement/evaluation/cross_seed_summary.json')
    require(rep['aspirational_targets']['close_fraction_at_least_75_percent'] is False,'Early close-target statement differs')
    planning=read('results/tracking_planning/per_policy.json');action=read('results/tracking_action_resolution/per_policy.json')
    for split in ('final','stress'):
        hybrid=[r for r in planning if r['split']==split and r['family']=='Hybrid-H3']
        require(sorted(r['seed'] for r in hybrid)==[60,61,62,63,64],'Planning source-model count differs')
        baseline={r['seed']:r for r in planning if r['split']==split and r['family']=='R-Decay'}
        require(all((r['C']>=.75 if split=='final' else r['C']<.75) for r in hybrid),'Planning target statement differs')
        if split=='final':require(all(r['H']>=.98*baseline[r['seed']]['H'] for r in hybrid),'Planning hit-retention statement differs')
        for family in ('Plan-H3-A14','Hybrid-H3-A14'):
            records=[r for r in action if r['split']==split and r['family']==family]
            require(len(records)==(1 if family.startswith('Plan') else 5),'Controller count differs')
            require(all((r['C']>=.75 if split=='final' else r['C']<.75) for r in records),'Action target statement differs')
    pgates=[r for r in read('results/tracking_planning/thresholds.json') if r['candidate']=='Hybrid-H3' and r['level']=='family_mean']
    agates=read('results/tracking_action_resolution/thresholds.json')
    require(len(pgates)==2 and all(not r['C_plus_5pp'] and not r['practical_pass'] and not r['simultaneous_pass'] for r in pgates+agates),'Relative-gate statement differs')
    require(all(r['H_retains_98pct'] and r['absolute_C75_and_hits_retained'] for r in agates if r['split']=='final'),'Action hit-retention statement differs')
    links=0
    for p in (ROOT/'README.md',RECORD/'README.md'):
        text=p.read_text(encoding='utf-8')
        for raw in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',text):
            if raw.startswith(('http:','https:','mailto:')):continue
            relative,_,anchor=raw.partition('#');target=(p.parent/relative).resolve() if relative else p
            require(target.is_relative_to(ROOT) and target.exists(),'Broken relative link '+raw)
            if anchor and target.suffix=='.md':
                headings=re.findall(r'^#+\s+(.+)$',target.read_text(encoding='utf-8'),re.M)
                slugs={re.sub(r'[^\w -]','',h.lower()).replace(' ','-') for h in headings}
                require(anchor in slugs,'Broken anchor '+raw)
            links+=1
    return dict(complete=True,baseline_commit=BASE,exact_substitutions=4,protected_baseline_files=len(rows),protected_files_changed=[],
        old_contract_unchanged=True,saved_record_statement_checks=True,relative_links_checked=links,training_interactions=0,Q_updates=0,environment_rollout_ticks=0,
        historical_test_suites_executed=False,check_scope='Documentation scope, saved record consistency, all other baseline blob hashes, relative links and anchors.')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    require(not a.output.exists(),'Use a new receipt path');result=verify();a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
