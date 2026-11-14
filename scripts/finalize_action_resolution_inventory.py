"""Finalize delivery hashes with exact root-only self-reference exclusions.

Metadata-only post-analysis adapter; never modifies a frozen experiment source,
protocol, numeric result or historical receipt. Nested historical verification
receipts are payload, even when their basename matches a root receipt.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from aimrl.tracking_action_resolution.core import OUT
from aimrl.tracking_action_resolution.verify import inventory
from aimrl.tracking_research.io import read_csv,read_json,save_csv,save_json,sha

def finalize_inventory():
    inventory();rows={r['path']:r for r in read_csv(OUT/'output_inventory.csv')}
    excluded={OUT/name for name in ('output_inventory.csv','inventory_scope.json','verification.json','verification_O.json','final_acceptance.json')}
    for p in list(OUT.rglob('*'))+[ROOT/'scripts/interpret_tracking_action_resolution.py',Path(__file__)]:
        if not p.is_file() or p in excluded or p.suffix in ('.pdb','.lib'):continue
        rows[p.relative_to(ROOT).as_posix()]=dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p))
    rows=[rows[k] for k in sorted(rows)];save_csv(OUT/'output_inventory.csv',rows)
    scope=read_json(OUT/'inventory_scope.json');scope.update(files=len(rows),bytes=sum(int(r['bytes']) for r in rows),
        excluded_self_and_later_receipts=sorted(p.relative_to(ROOT).as_posix() for p in excluded),
        post_final_metadata_scripts=['scripts/interpret_tracking_action_resolution.py','scripts/finalize_action_resolution_inventory.py'],
        boundary='Only the five explicit root paths above are excluded. All nested verification receipts are hashed payload. The frozen generic inventory helper is followed by this metadata-only path adapter; no scientific source or historical hash changes.')
    save_json(OUT/'inventory_scope.json',scope);return rows

if __name__=='__main__':
    rows=finalize_inventory()
    for row in rows:
        if sha(ROOT/row['path'])!=row['sha256']:raise RuntimeError('Inventory integrity failure '+row['path'])
    print('Verified delivery payload:',len(rows))
