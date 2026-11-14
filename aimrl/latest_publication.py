"""Explicit publication adapters; frozen experiment sources/hashes stay intact.

Only archived documentation changes and named non-core omissions are accepted.
All transferred research bytes, models, protocols and mathematical checks retain
their original expected hashes. Verification receipts go to a fresh directory.
"""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

from .tracking_research.io import ROOT,read_json,read_csv,save_json,sha

RELEASE=ROOT/'provenance/latest_studies_release'


def require(ok,message):
    if not ok:raise ValueError(message)


def contract():return read_json(RELEASE/'release.json')


def verify_omission(relative,expected,root=ROOT,release=None):
    release=contract() if release is None else release
    entry=release['omissions'].get(relative)
    require(entry is not None and entry['original_sha256']==expected,'Undocumented missing evidence: '+relative)
    # Only machine-local failed transcripts or native intermediate files may be omitted.
    require(entry['kind'] in ('native_intermediate','redacted_failure_transcript'),'Invalid omission class')
    if entry['kind']=='native_intermediate':
        require(Path(relative).suffix in ('.lib','.pdb') and '/native/' in relative,'Omission is not a native intermediate')
    else:
        require('/execution/' in relative or '/native/build_attempt_' in relative,'Not a failed transcript')
        replacement=root/entry['replacement_path']
        require(replacement.is_file() and sha(replacement)==entry['replacement_sha256'],'Publication transcript replacement differs')
    return entry


def verify_document(relative,expected,root=ROOT,release=None):
    release=contract() if release is None else release
    entry=release['documentation_changes'].get(relative)
    require(entry is not None and expected==entry['original_sha256'],'Unapproved protected-file change: '+relative)
    require(sha(root/relative)==entry['published_sha256'],'Published document changed after review: '+relative)
    archive=root/release['original_documents_archive']
    require(sha(archive)==release['original_documents_archive_sha256'],'Original document archive differs')
    with zipfile.ZipFile(archive) as z:
        original=z.read(relative)
    require(hashlib.sha256(original).hexdigest()==expected,'Archived original document hash differs')
    return entry


def verify_protected(mapping,root=ROOT,release=None):
    release=contract() if release is None else release
    verified=0;documents=[];omitted=[]
    for relative,expected in mapping.items():
        path=root/relative
        if path.is_file() and sha(path)==expected:verified+=1;continue
        if path.is_file():verify_document(relative,expected,root,release);documents.append(relative)
        else:verify_omission(relative,expected,root,release);omitted.append(relative)
    return dict(unchanged_files_verified=verified,archived_document_originals_verified=len(documents),
                allowed_publication_document_changes=documents,documented_noncore_omissions=omitted,
                changed_experiment_core_files=[],expected_historical_hashes_modified=False)


def inventory_reader(original):
    def checked(path):
        rows=original(path)
        path=Path(path).resolve()
        if path in (ROOT/'results/tracking_boundary_stability/output_inventory.csv',ROOT/'results/tracking_planning/output_inventory.csv'):
            kept=[]
            for row in rows:
                relative=(path.parent/row['path']).relative_to(ROOT).as_posix()
                target=ROOT/relative
                if not target.exists():
                    entry=verify_omission(relative,row['sha256'])
                    require(int(row['bytes'])==entry['original_bytes'],'Omitted inventory size differs')
                else:kept.append(row)
            return kept
        return rows
    return checked


def boundary_verify(output):
    from .tracking_boundary_stability import verify_artifacts as canonical
    from .tracking_boundary_stability import verify as frozen
    from .tracking_boundary_stability import study
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    def history(out=study.OUT):
        result=verify_protected(read_json(out/'historical_sha256_before.json'))
        result['head']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        return result
    def receipt(path,value):
        require(Path(path).name in ('verification_python.json','verification_optimized.json'),'Unexpected boundary receipt')
        save_json(output/Path(path).name,value)
    study.history_check=history
    # Frozen fingerprint is the archived Windows artifact on every platform;
    # boundary artifact checking itself does not execute this DLL.
    study.LIBRARY=ROOT/'results/tracking_boundary_stability/native/boundary.dll'
    frozen.read_csv=inventory_reader(frozen.read_csv)
    frozen.save_json=receipt;canonical.save_json=receipt
    canonical.main()
    result=read_json(output/('verification_python.json' if __debug__ else 'verification_optimized.json'))
    result['publication_adapter']='Archived document originals and explicitly inventoried non-core omissions; all original numerical checks execute.'
    save_json(output/'result.json',result)
    return result


def planning_verify(output,library=None):
    from .tracking_planning import core,study,verify
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    def protected():return verify_protected(read_json(study.OUT/'protected_sha256_before.json'))
    def receipt(path,value):
        require(Path(path).name in ('verification_python.json','verification_optimized.json'),'Unexpected planning receipt')
        save_json(output/Path(path).name,value)
    study.protect=protected
    study.LIBRARY=ROOT/'results/tracking_planning/native/planning.dll'
    verify.read_csv=inventory_reader(verify.read_csv)
    verify.save_json=receipt
    if library is not None:
        # study.LIBRARY still identifies the original archived DLL for protocol
        # provenance; only the explicit mathematical audit engine is replaced.
        core.LIBRARY=Path(library).resolve()
    result=verify.verify(core.config())
    result['publication_adapter']='Archived documentation and named non-core omissions; complete DP reconstruction and all saved proposals still checked.'
    if library is not None:result['rebuilt_math_engine']=dict(path=Path(library).name,sha256=sha(library),source_sha256=sha(ROOT/'aimrl/tracking_planning/kernel.c'))
    save_json(output/'result.json',result)
    print(json.dumps(result,indent=2))
    return result


def historical_studies_verify(output):
    import sys
    from .tracking_tradeoff import analysis
    import scripts.verify_tracking_studies as original
    output=Path(output).resolve()
    reader=analysis.read_json
    documented=[]
    def historical_read(path):
        data=reader(path)
        if Path(path).resolve()==ROOT/'results/tracking_shooting_tradeoff/historical_sha256_before.json':
            data=dict(data)
            expected=data.get('.gitignore')
            if expected is not None and sha(ROOT/'.gitignore')!=expected:
                # Verify both real byte streams before acknowledging this one
                # documentation exception. No expected digest is substituted.
                verify_document('.gitignore',expected)
                data.pop('.gitignore');documented.append('.gitignore')
        return data
    analysis.read_json=historical_read
    previous=sys.argv
    try:
        sys.argv=['scripts/verify_tracking_studies.py','--public-tree','--output',str(output)]
        original.main()
    finally:sys.argv=previous
    result=read_json(output/'verification.json')
    result['publication_document_exceptions_verified']=sorted(set(documented))
    result['old_expected_hashes_modified']=False
    save_json(output/'publication_verification.json',result)
    return result


def verify_release_inputs():
    release=contract()
    for path,entry in release['documentation_changes'].items():verify_document(path,entry['original_sha256'])
    baseline=read_json(RELEASE/'experiment_input_sha256.json')
    result=verify_protected(baseline)
    for path,entry in release['omissions'].items():verify_omission(path,entry['original_sha256'])
    for item in read_json(ROOT/'results/tracking_planning/model_provenance.json')['models']:
        require(not Path(item['path']).is_absolute() and (ROOT/item['path']).is_file(),'Repository-local model reference is missing')
        require(sha(ROOT/item['path'])==item['file_sha256'],'Referenced model hash differs')
    return result


def check_links():
    import re
    paths=[ROOT/'README.md',RELEASE/'README.md',ROOT/'docs/research_index.md',ROOT/'docs/tracking_boundary_stability.md',ROOT/'docs/tracking_planning.md',
           ROOT/'docs/reproduce_latest_studies.md',ROOT/'results/tracking_boundary_stability/results_summary.md',
           ROOT/'results/tracking_boundary_stability/REPRODUCE.md',ROOT/'results/tracking_planning/results_summary.md',
           ROOT/'results/tracking_planning/REPRODUCE.md']
    n=0
    for path in paths:
        text=path.read_text(encoding='utf-8')
        for raw in re.findall(r'\]\(([^)]+)\)',text):
            if raw.startswith(('http:','https:','mailto:')):continue
            relative,_,fragment=raw.partition('#');target=(path.parent/relative).resolve() if relative else path
            require(target.is_relative_to(ROOT) and target.exists(),'Broken publication link: '+str(path.relative_to(ROOT))+' -> '+raw)
            if fragment and target.suffix=='.md':
                headings=re.findall(r'^#+\s+(.+)$',target.read_text(encoding='utf-8'),re.M)
                slugs={re.sub(r'[^\w -]','',h.lower()).replace(' ','-') for h in headings}
                require(fragment in slugs,'Broken document anchor: '+raw)
            n+=1
    return dict(relative_links_checked=n)
