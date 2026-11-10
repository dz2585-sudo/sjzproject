"""Reconcile the two follow-up studies from saved records without training or replay."""
import argparse
import hashlib
import re
import shutil
import sys
from pathlib import Path
from urllib.parse import unquote
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image

from aimrl.tracking_fire_guard.analysis import compile_metrics, diagnostics
from aimrl.tracking_fire_guard.history import (
    LOCAL_SETUP_LOG, PRIOR, agrees, check_setup_omission, historical_hashes, previous, rows, values,
)
from aimrl.tracking_fire_guard.replay import require
from aimrl.tracking_fire_guard.report import evaluation_audit
from aimrl.tracking_fire_guard.study import identity
from aimrl.tracking_research.io import read_json, save_json, sha
from aimrl.tracking_tradeoff.analysis import audit as audit_tradeoff

GUARD = ROOT / 'results/tracking_fire_guard'
RELEASE = ROOT / 'provenance/tracking_studies_release.json'


def verify_inventory(directory, public_tree):
    inventory = rows(directory / 'output_inventory.csv')
    require(len(inventory) == len({r['path'] for r in inventory}), f'Duplicate inventory paths: {directory.name}')
    verified = 0
    omitted = []
    for row in inventory:
        relative = row['path']
        path = directory / relative
        require(path.resolve().is_relative_to(directory), f'Inventory path outside study: {relative}')
        if directory == PRIOR and relative == LOCAL_SETUP_LOG:
            check_setup_omission(row)
            if public_tree:
                require(not path.exists(), 'Machine-local setup log must be absent in the public tree')
            if not path.exists():
                omitted.append(relative)
                continue
        require(path.stat().st_size == int(row['bytes']) and sha(path) == row['sha256'],
                f'File size or digest mismatch: {directory.name}/{relative}')
        verified += 1
    expected = {r['path'] for r in inventory} | {'output_inventory.csv','execution_checks.json'}
    require({p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_file()} == expected-set(omitted),
            f'Unlisted or missing output: {directory.name}')
    return {'verified_payload_files':verified,'documented_omissions':omitted}


def verify_guard_sources(release, protocol):
    adapter=release['guard_checkout_adapter']
    path=ROOT/adapter['original_sources_path']
    require(sha(path)==adapter['original_sources_zip_sha256'],'Original source archive hash differs')
    original=adapter['original_source_sha256']
    current=adapter['checkout_source_sha256']
    require(set(original)==set(current)=={'__main__.py','history.py','study.py'},'Unexpected checkout adapter scope')
    with ZipFile(path) as archive:
        require(set(archive.namelist())==set(original),'Unexpected original source archive entries')
        for name,digest in protocol['source_sha256'].items():
            checkout=sha(ROOT/'aimrl/tracking_fire_guard'/name)
            if name in original:
                require(digest==original[name]==hashlib.sha256(archive.read(name)).hexdigest()
                        and checkout==current[name],f'Frozen or checkout adapter source differs: {name}')
            else:
                require(checkout==digest,f'Unadapted guard source differs: {name}')
    for name,digest in read_json(GUARD/'execution_checks.json')['report_sources_sha256'].items():
        require(sha(ROOT/'aimrl/tracking_fire_guard'/name)==digest,f'Guard reporting source differs: {name}')


def verify_recomputed(scratch):
    config,_=previous(scratch,historical_hashes())
    for filename in ('historical_per_seed.csv','historical_comparison.csv','historical_paired_differences.csv'):
        require(sha(scratch/filename)==sha(GUARD/filename),f'Historical recomputation differs: {filename}')
    tradeoff=audit_tradeoff(config,PRIOR)
    protocol=read_json(GUARD/'protocol.json')
    archive=read_json(GUARD/'evaluation/per_seed.json')
    require(len(archive)==86,'Incomplete guard result matrix')
    for r in archive:
        name=identity(r)
        evaluated=rows(GUARD/'evaluation'/r['split']/f'{name}_{r["variant"]}_windows.csv')
        for key,value in values(evaluated).items():
            agrees(value,r[key],f'{r["split"]}/{name}/{r["variant"]}/{key}')
    (scratch/'evaluation').mkdir()
    shutil.copyfile(GUARD/'evaluation/per_seed.json',scratch/'evaluation/per_seed.json')
    _,within,screens=compile_metrics(scratch,protocol)
    for filename in ('comparison.csv','paired_differences.csv','paired_delta_summary.csv',
                     'paired_vs_reference.csv','thresholds.csv'):
        require(sha(scratch/filename)==sha(GUARD/filename),f'Guard recomputation differs: {filename}')
    previous_screens={r['condition']:r for r in screens if r['comparison']=='previous_original'}
    for record in rows(PRIOR/'screening.csv'):
        actual=previous_screens[record['condition']]
        for key in ('delta_D','delta_C','delta_H','relative_H'):
            agrees(actual[key],record[key],f'Prior reward threshold delta: {key}')
        for key in ('simultaneous','retention'):
            require(str(actual[key])==record[key],f'Prior reward threshold differs: {key}')
    shutil.copytree(GUARD/'q_events',scratch/'q_events')
    diagnostics(scratch,protocol,within)
    for filename in ('q_diagnostics_summary.json','q_diagnostics_distinct_states.csv',
                     'q_diagnostics_by_condition.csv','q_diagnostics_per_seed.csv'):
        require(sha(scratch/filename)==sha(GUARD/filename),f'Q diagnostic recomputation differs: {filename}')
    return tradeoff,evaluation_audit(GUARD,protocol,within)


def verify_gifs():
    clips=[(PRIOR,r) for r in read_json(PRIOR/'execution_checks.json')['gifs']]
    clips.append((GUARD,read_json(GUARD/'execution_checks.json')['demo']))
    checked=[]
    for directory,record in clips:
        path=directory/record['path']
        require(sha(path)==record['sha256'],f'GIF digest differs: {path.name}')
        with Image.open(path) as gif:
            require(gif.n_frames==record['frames']==301,f'GIF frame count differs: {path.name}')
            milliseconds=0
            for frame in range(gif.n_frames):
                gif.seek(frame)
                gif.load()
                milliseconds+=gif.info['duration']
            require(milliseconds==record['duration_ms']==20070,f'GIF timing differs: {path.name}')
        checked.append({'path':path.relative_to(ROOT).as_posix(),'frames':301,'duration_ms':milliseconds})
    return checked


def verify_links():
    files=[ROOT/'README.md',ROOT/'docs/tracking_shooting_tradeoff.md',ROOT/'docs/tracking_fire_guard.md',
           PRIOR/'results_summary.md',PRIOR/'paper_addendum.md',
           GUARD/'results_summary.md',GUARD/'paper_addendum.md',GUARD/'REPRODUCE.md']
    link_count=0
    for path in files:
        text=path.read_text(encoding='utf-8')
        for raw in re.findall(r'\]\(([^)\s]+)\)',text):
            if raw.startswith(('https:','http:','mailto:')):
                continue
            relative,_,fragment=unquote(raw).partition('#')
            target=(path.parent/relative).resolve() if relative else path
            require(target.is_relative_to(ROOT) and target.exists(),
                    f'Broken or non-repository link in {path.relative_to(ROOT)}: {raw}')
            if fragment and target.suffix=='.md':
                headings=re.findall(r'^#+\s+(.+)$',target.read_text(encoding='utf-8'),re.M)
                slugs={re.sub(r'[^\w -]','',h.lower()).replace(' ','-') for h in headings}
                require(fragment in slugs,f'Broken heading link in {path.relative_to(ROOT)}: {raw}')
            if target.suffix=='.png':
                with Image.open(target) as picture:
                    picture.load()
            link_count+=1
    readme=files[0].read_text(encoding='utf-8')
    for suffix in ('tracking_shooting_tradeoff/demos/scenario_4400000.gif','tracking_fire_guard/demo/synchronized_fire_guard.gif',
                   'tracking_shooting_tradeoff/figures/close_hits_tradeoff.png','tracking_fire_guard/figures/objective_tradeoffs.png'):
        require(re.search(r'!\[[^\]]*\]\(results/'+re.escape(suffix)+r'\)',readme),f'README embed missing: {suffix}')
    return link_count


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--public-tree',action='store_true',help='Require the local-only transcript to be absent')
    args=parser.parse_args()
    output=args.output.resolve()
    require(not any(output.is_relative_to(p) for p in (PRIOR,GUARD)),'Keep verification output outside frozen results')
    output.mkdir(parents=True,exist_ok=False)
    release=read_json(RELEASE)
    inventories={p.name:verify_inventory(p,args.public_tree) for p in (PRIOR,GUARD)}
    verify_guard_sources(release,read_json(GUARD/'protocol.json'))
    tradeoff,guard=verify_recomputed(output)
    gifs=verify_gifs()
    links=verify_links()
    require(not list(GUARD.rglob('*.npz')) and not list(GUARD.rglob('*.npy')),'Guard must reuse frozen model packages')
    result={'complete':True,'inventories':inventories,'reward_study':tradeoff,'guard_study':guard,
            'decoded_gifs':gifs,'relative_links_checked':links,'training_interactions_this_check':0}
    save_json(output/'verification.json',result)
    print(f'Verified {sum(x["verified_payload_files"] for x in inventories.values())} payload files, '
          f'{links} links and {len(gifs)} complete GIFs.',flush=True)


if __name__=='__main__':
    main()
