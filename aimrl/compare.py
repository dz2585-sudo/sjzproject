"""Compare two Day 4 validation runs and check the Day 3 base regression."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .mpl_setup import configure_mpl


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as file:
        return list(csv.DictReader(file))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True, type=Path)
    parser.add_argument('--potential', required=True, type=Path)
    parser.add_argument('--day3-base', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f'Nonempty output directory: {args.output}')
    dirs = {'base': args.base, 'potential': args.potential}
    configs = {k: json.loads((d / 'run_config.json').read_text(encoding='utf-8')) for k, d in dirs.items()}
    metadata = {k: json.loads((d / 'metadata.json').read_text(encoding='utf-8')) for k, d in dirs.items()}
    if configs['base'] != configs['potential'] or any(metadata[k]['reward_mode'] != k for k in dirs):
        raise ValueError('Run configurations or reward modes do not match')
    if metadata['base']['root_seed'] != metadata['potential']['root_seed']:
        raise ValueError('Training seeds do not match')
    rows = {k: read_csv(d / 'validation.csv') for k, d in dirs.items()}
    if [r['global_step'] for r in rows['base']] != [r['global_step'] for r in rows['potential']]:
        raise ValueError('Validation checkpoints do not match')
    old_config = json.loads((args.day3_base / 'run_config.json').read_text(encoding='utf-8'))
    old_meta = json.loads((args.day3_base / 'metadata.json').read_text(encoding='utf-8'))
    old_rows = read_csv(args.day3_base / 'validation.csv')
    q_match = np.array_equal(np.load(args.day3_base / 'q_table.npy', allow_pickle=False),
                             np.load(args.base / 'q_table.npy', allow_pickle=False))
    compared_meta = ['root_seed', 'random_streams', 'environment_initialization_seed', 'global_steps', 'numpy', 'matplotlib', 'python']
    regression = {'config_equal': old_config == configs['base'], 'validation_equal': old_rows == rows['base'],
                  'q_table_equal': bool(q_match), 'metadata_fields_equal': all(old_meta[k] == metadata['base'][k] for k in compared_meta)}
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'comparison.csv').open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['reward_mode', *rows['base'][0].keys()])
        writer.writeheader()
        for mode in dirs:
            writer.writerows({'reward_mode': mode, **r} for r in rows[mode])
    efficiency = {}
    for mode, entries in rows.items():
        efficiency[mode] = next((int(a['global_step']) for a, b in zip(entries, entries[1:])
                                 if float(a['success_rate']) >= .9 and float(b['success_rate']) >= .9), None)
    summary = {'scope': 'seed=0 single-seed development comparison; not a statistical conclusion',
               'sources': {k: str(v.resolve()) for k, v in dirs.items()},
               'day3_base_source': str(args.day3_base.resolve()), 'day3_regression': regression,
               'first_of_two_consecutive_validation_checkpoints_at_least_90_percent': efficiency,
               'checkpoint_resolution_steps': configs['base']['experiments']['validation']['interval_steps'],
               'key_checkpoints': {k: [r for r in entries if int(r['global_step']) in (0, 10000, 30000, 100000)] for k, entries in rows.items()}}
    (args.output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    (args.output / 'README.md').write_text('This single-seed development comparison uses seed 0, equal training budgets, and the same validation scenarios. It is not a multi-seed formal result or evidence of statistical significance. Curves are derived from validation.csv for each run; reward-return scales are not compared across methods.\n', encoding='utf-8')
    configure_mpl()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    for mode, entries in rows.items():
        x = [int(r['global_step']) for r in entries]
        axes[0].plot(x, [100 * float(r['success_rate']) for r in entries], marker='o', label=mode)
        axes[1].plot(x, [float(r['mean_steps']) for r in entries], marker='o', label=mode)
    axes[0].set_ylabel('Success rate (%)')
    axes[1].set_ylabel('Mean steps')
    axes[1].set_xlabel('Training interaction steps')
    for ax in axes:
        ax.grid(alpha=.3)
        ax.legend()
    fig.suptitle('Base vs Potential Q-learning | seed 0 | Validation')
    fig.tight_layout()
    fig.savefig(args.output / 'comparison.png', dpi=150)
    plt.close(fig)
    print('Day 3 regression:', regression, '; first consecutive >=90% checkpoints:', efficiency)


if __name__ == '__main__':
    main()
