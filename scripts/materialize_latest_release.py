"""List exact publishable paths and construct a clean read-only audit view."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from aimrl.latest_publication import RELEASE,contract,verify_release_inputs
from aimrl.tracking_research.io import save_json,sha


def selected():
    release=contract();paths={'README.md','.gitignore','aimrl/latest_publication.py'}
    for folder in ('aimrl/tracking_boundary_stability','aimrl/tracking_planning','results/tracking_boundary_stability','results/tracking_planning','provenance/latest_studies_release'):
        for p in (ROOT/folder).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc','.pyo'):
                relative=p.relative_to(ROOT).as_posix()
                if relative not in release['omissions']:paths.add(relative)
    paths.update(['configs/tracking_boundary_stability.json','configs/tracking_planning.json',
        'tests/test_tracking_boundary_stability.py','tests/test_tracking_planning.py','tests/test_latest_publication.py',
        'scripts/run_boundary_stability_checks.py','scripts/check_tracking_planning.py','scripts/prepare_latest_publication.py',
        'scripts/reproduce_latest_studies.py','scripts/check_latest_release.py','scripts/materialize_latest_release.py',
        'docs/research_index.md','docs/tracking_boundary_stability.md','docs/tracking_planning.md','docs/reproduce_latest_studies.md'])
    return sorted(paths)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--view',type=Path);parser.add_argument('--paths-file',type=Path,required=True)
    args=parser.parse_args();verify_release_inputs();paths=selected()
    for relative in paths:
        p=ROOT/relative
        if not p.is_file():raise ValueError('Missing selected file '+relative)
        if p.stat().st_size>=100*1024**2:raise ValueError('Oversized GitHub file '+relative)
    ignored=subprocess.run(['git','check-ignore','--stdin'],input='\n'.join(paths)+'\n',cwd=ROOT,text=True,capture_output=True)
    if ignored.returncode not in (0,1) or ignored.stdout.strip():raise ValueError('Necessary files ignored: '+ignored.stdout+ignored.stderr)
    args.paths_file.parent.mkdir(parents=True,exist_ok=True)
    args.paths_file.write_bytes(b'\0'.join(p.encode('utf-8') for p in paths)+b'\0')
    result=dict(selected_files=len(paths),selected_bytes=sum((ROOT/p).stat().st_size for p in paths),
                maximum_file_bytes=max((ROOT/p).stat().st_size for p in paths),omitted_paths=sorted(contract()['omissions']))
    if args.view:
        view=args.view.resolve()
        if not view.is_relative_to(ROOT/'.tmp'):raise ValueError('Audit view must be in repository .tmp')
        view.mkdir(parents=True,exist_ok=False)
        tracked={p for p in subprocess.check_output(['git','ls-files','-z'],cwd=ROOT,text=True).split('\0') if p}
        all_paths=sorted(tracked|set(paths));linked=copied=0
        for relative in all_paths:
            src=ROOT/relative;dst=view/relative;dst.parent.mkdir(parents=True,exist_ok=True)
            try:os.link(src,dst);linked+=1
            except OSError:shutil.copyfile(src,dst);copied+=1
        result.update(view_files=len(all_paths),hardlinked=linked,copied=copied)
        # No protected linked file is written by the checker. Verification
        # receipts and test temporaries go into fresh .tmp directories only.
        save_json(ROOT/'.tmp/latest-release-view.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
