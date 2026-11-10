"""Check historical declarations and CSV seed columns for scenario overlap."""
import argparse
import csv
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def integers(node, seed_context=False):
    if isinstance(node, dict):
        start, count = node.get('start'), node.get('count')
        if type(start) is int and type(count) is int:
            yield range(start, start + count)
        for key, value in node.items():
            yield from integers(value, 'seed' in key.lower())
    elif isinstance(node, list):
        for value in node:
            yield from integers(value, seed_context)
    elif seed_context and type(node) is int:
        yield range(node, node + 1)


def audit(roots, scenarios, replicating_guard=False):
    counts = {'json_files':0,'csv_seed_files':0,'csv_seed_values':0}
    conflicts = []
    for root, args in roots:
        paths = subprocess.check_output(['git','ls-files',*args,'-z'],cwd=root).decode().split('\0')
        for relative in filter(None,paths):
            path = root / relative
            if relative.startswith(('results/tracking_shooting_tradeoff/','results/tracking_fire_guard/')) or relative == 'configs/tracking_shooting_tradeoff.json':
                continue
            if (path.suffix == '.json' and relative.startswith(('configs/','outputs/','results/'))
                    and re.search(r'protocol|config|metadata|checks|seed|export', path.name)):
                counts['json_files'] += 1
                data = json.loads(path.read_text(encoding='utf-8-sig'))
                for declared in integers(data):
                    matches = sorted(s for s in scenarios if s in declared)
                    if matches:
                        conflicts.append((relative,matches))
            elif path.suffix == '.csv':
                with path.open(newline='',encoding='utf-8-sig') as stream:
                    reader = csv.reader(stream)
                    columns = [i for i,k in enumerate(next(reader,[])) if 'seed' in k.lower()]
                    if not columns:
                        continue
                    counts['csv_seed_files'] += 1
                    for row in reader:
                        for i in columns:
                            if i < len(row) and row[i].isdigit():
                                counts['csv_seed_values'] += 1
                                if int(row[i]) in scenarios:
                                    conflicts.append((relative,[int(row[i])]))
    # The guard's gate and demo intentionally reuse earlier routes. Only the guard's
    # own holdout is reserved by that study. Exact guard replications reuse its reservation.
    guard_protocol=ROOT/'results/tracking_fire_guard/protocol.json'
    if guard_protocol.is_file() and not replicating_guard:
        for start,count in json.loads(guard_protocol.read_text(encoding='utf-8'))['holdout'].values():
            matched=sorted(s for s in scenarios if start<=s<start+count)
            if matched:
                conflicts.append(('results/tracking_fire_guard/protocol.json',matched))
    return {'checked':counts,'conflict_count':len(conflicts),'conflicts':conflicts[:20]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', type=Path, help='Optional original workspace to audit as well')
    args = parser.parse_args()
    design = json.loads((ROOT/'configs/tracking_shooting_tradeoff.json').read_text(encoding='utf-8'))['tradeoff']
    scenarios = set(design['demo_seeds'])
    for split in ('development','final','stress'):
        start, count = (design[split][k] for k in ('start','count'))
        scenarios.update(range(start,start+count))
    roots = [(ROOT, [])]
    if args.original:
        roots.append((args.original, ['--others','--exclude-standard']))
    record = audit(roots,scenarios)
    print(json.dumps(record,indent=2))
    return int(record['conflict_count'] > 0)


if __name__ == '__main__':
    raise SystemExit(main())
