"""Offline English plots and full-length real-trajectory four-policy GIFs."""
import json
import math
from pathlib import Path

import numpy as np

from ..mpl_setup import configure_mpl
from .env import ACTION_NAMES
from .workflow import read_csv, read_json, save_json, verify_protocol

LABELS = {'base': 'Base Q-learning', 'potential': 'Potential Q-learning', 'rule': 'Predictive rule', 'random': 'Random'}


def optional(value):
    return float(value) if value not in ('', None) else float('nan')


def plots(output, destination):
    import matplotlib.pyplot as plt
    learning = read_csv(output / 'evaluation/validation.csv')
    comparison = read_csv(output / 'evaluation/comparison.csv')
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for mode in ('base', 'potential'):
        rows = [r for r in learning if r['method'] == mode]
        x = [int(r['global_step']) / 1000000 for r in rows]
        for ax, field, label, scale in zip(axes.flat, ('mean_distance', 'close_fraction', 'hits_per_episode', 'pooled_shot_accuracy'),
                                          ('Mean post-motion distance', 'Close tracking (distance <=1, %)', 'Hits per 300-tick episode', 'Pooled shot accuracy (%)'), (1, 100, 1, 100)):
            ax.plot(x, [optional(r[field]) * scale for r in rows], marker='o', label=LABELS[mode])
            ax.set_ylabel(label); ax.set_xlabel('Training interactions (millions)'); ax.grid(alpha=.25); ax.legend()
    fig.suptitle('Tracking25 V2 | seed 0 | 200 development scenarios | no across-seed error bands')
    fig.tight_layout(); fig.savefig(destination / 'learning_curves.png', dpi=140); plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, field, label, scale in zip(axes.flat, ('mean_distance', 'close_fraction', 'hits_per_episode', 'pooled_shot_accuracy', 'recovery_fraction', 'mean_capped_recovery_lag'),
                                      ('Mean post-motion distance', 'Close tracking (%)', 'Hits per 300-tick episode', 'Pooled shot accuracy (%)', 'Observed recovery fraction (%)', 'Mean capped recovery lag (ticks)'), (1, 100, 1, 100, 100, 1)):
        values = [optional(r[field]) * scale for r in comparison]
        bars = ax.bar([r['method'].title() for r in comparison], values, color=['#377eb8', '#ff7f00', '#4daf4a', '#999999'])
        ax.bar_label(bars, fmt='%.2f', fontsize=8, padding=2)
        ax.set_ylabel(label); ax.grid(axis='y', alpha=.2); ax.margins(y=.2)
    fig.suptitle('Final development comparison | shared target paths | recovery windows may overlap')
    fig.tight_layout(); fig.savefig(destination / 'policy_comparison.png', dpi=140); plt.close(fig)
    fig, axes = plt.subplots(1, 4, figsize=(13, 4), sharey=True)
    for ax, row in zip(axes, comparison):
        fast, slow = [], []
        for key in ('0_3', '4_11', '12_plus'):
            a, b = int(row['slow_bin_' + key]), int(row['fast_bin_' + key])
            slow.append(a / (a + b) if a + b else float('nan')); fast.append(b / (a + b) if a + b else float('nan'))
        ax.bar(['0–3', '4–11', '>=12'], slow, label='Slow', color='#377eb8')
        ax.bar(['0–3', '4–11', '>=12'], fast, bottom=slow, label='Fast', color='#ff7f00')
        ax.set_title(LABELS[row['method']]); ax.set_xlabel('Pre-action distance'); ax.set_ylim(0, 1.1)
    axes[0].set_ylabel('Proportion among movement actions only'); axes[-1].legend()
    fig.suptitle('Movement-speed choices (stay and fire excluded; empty bins remain undefined)')
    fig.tight_layout(); fig.savefig(destination / 'speed_choices.png', dpi=140); plt.close(fig)


def clip(seed, traces, destination):
    import matplotlib.pyplot as plt
    from PIL import Image
    methods = ('base', 'potential', 'rule', 'random')
    rows = {mode: [r for r in traces if r['method'] == mode and int(r['scenario_seed']) == seed] for mode in methods}
    if any([int(r['tick']) for r in series] != list(range(301)) for series in rows.values()):
        raise ValueError('Each demo must include the initial state and all 300 ticks')
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=100)
    fig.subplots_adjust(left=.06, right=.98, bottom=.07, top=.86, wspace=.20, hspace=.5)
    artists = []
    for ax, mode in zip(axes.flat, methods):
        ax.set_xlim(0, 25); ax.set_ylim(25, 0); ax.set_aspect('equal')
        labels = [0, 6, 12, 18, 24]
        ax.set_xticks([x + .5 for x in labels], labels=labels); ax.set_yticks([x + .5 for x in labels], labels=labels)
        ax.set_xticks(range(26), minor=True); ax.set_yticks(range(26), minor=True)
        ax.grid(which='minor', color='#dde1e6', linewidth=.4); ax.tick_params(which='minor', length=0)
        ax.set_xlabel('x (right)'); ax.set_ylabel('y (down)')
        target_trail, = ax.plot([], [], color='#f2a398', linewidth=1)
        cross_trail, = ax.plot([], [], color='#a5c8ee', linewidth=1)
        target, = ax.plot([], [], 'o', markerfacecolor='none', markeredgecolor='#d94f39', markersize=8, markeredgewidth=2)
        cross, = ax.plot([], [], 'x', color='#1766b1', markersize=7, markeredgewidth=2)
        flash, = ax.plot([], [], '*', color='#db9b00', markersize=12)
        title = ax.set_title('', fontsize=9)
        artists.append((target_trail, cross_trail, target, cross, flash, title))
    fig.suptitle(f'Tracking25 V2 | coverage-selected scenario {seed} | 15 FPS playback\nTarget: red circle | Crosshair: blue cross | Hit: gold star at actual shot location', fontsize=11)
    frames = []
    start_seconds = __import__('time').perf_counter()
    try:
        for tick in range(301):
            for mode, art in zip(methods, artists):
                row = rows[mode][tick]; tail = rows[mode][max(0, tick - 8):tick + 1]
                target_trail, cross_trail, target, cross, flash, title = art
                target_trail.set_data([int(r['xt']) + .5 for r in tail], [int(r['yt']) + .5 for r in tail])
                cross_trail.set_data([int(r['xc']) + .5 for r in tail], [int(r['yc']) + .5 for r in tail])
                target.set_data([int(row['xt']) + .5], [int(row['yt']) + .5]); cross.set_data([int(row['xc']) + .5], [int(row['yc']) + .5])
                is_hit = row['hit'] == 'True'
                flash.set_data([int(row['shot_x']) + .5] if is_hit else [], [int(row['shot_y']) + .5] if is_hit else [])
                action = 'reset' if int(row['action']) < 0 else ACTION_NAMES[int(row['action'])]
                title.set_text(f"{LABELS[mode]} | tick {tick}/300 | target speed {row['target_speed']}\n{action} ({row['movement_mode']}) | distance {row['distance']}\nShots {row['shots']} | Hits {row['hits']}")
            fig.canvas.draw()
            image = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3]).copy()
            if tick in (0, 150, 300): image.save(destination / f'seed{seed}_source_{tick:03d}.png')
            if tick == 150 and not (destination / 'demo_preview.png').exists(): image.save(destination / 'demo_preview.png')
            frames.append(image.quantize(colors=256, method=Image.Quantize.FASTOCTREE))
            image.close()
        durations = [(round((i + 1) * 100 / 15) - round(i * 100 / 15)) * 10 for i in range(301)]
        path = destination / f'seed{seed}_comparison.gif'
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=durations, loop=0, disposal=2, optimize=False)
    finally:
        plt.close(fig)
        for image in frames: image.close()
    with Image.open(path) as gif:
        if gif.n_frames != 301: raise RuntimeError('GIF did not retain all 301 frames')
        for tick in (0, 150, 300):
            gif.seek(tick); gif.convert('RGB').save(destination / f'seed{seed}_decoded_{tick:03d}.png')
    result = {'scenario_seed': seed, 'frames': 301, 'dimensions': [1000, 800], 'duration_ms': sum(durations),
              'render_seconds': __import__('time').perf_counter() - start_seconds, 'writer': 'Pillow',
              'post_hit_continuation': {m: {'hit_ticks': [int(r['tick']) for r in rows[m] if r['hit'] == 'True'],
                                           'last_tick': int(rows[m][-1]['tick']), 'final_hits': int(rows[m][-1]['hits'])} for m in methods}}
    print('Exported full-length demo:', result['scenario_seed'], '301 frames;', f"{result['render_seconds']:.2f}s", flush=True)
    return result


def render_stage(config, output):
    protocol = verify_protocol(config, output)
    destination = output / 'render'
    if (destination / 'render_checks.json').exists():
        print('Completed renders already exist; no exports overwritten.'); return
    destination.mkdir(exist_ok=True)
    configure_mpl()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plots(output, destination)
    traces = read_csv(output / 'evaluation/demo_traces.csv')
    first_seed = protocol['selected_demos'][0]['scenario_seed']
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for mode in ('base', 'potential', 'rule', 'random'):
        selected = [r for r in traces if r['method'] == mode and int(r['scenario_seed']) == first_seed and int(r['tick']) > 0]
        axes[1].plot([int(r['tick']) for r in selected], [int(r['displacement']) for r in selected], alpha=.65, linewidth=.8, label=LABELS[mode])
    axes[0].step([int(r['tick']) for r in selected], [int(r['target_speed']) for r in selected], where='post', label='Target nominal speed')
    axes[0].set_ylabel('Target cells/tick'); axes[1].set_ylabel('Actual crosshair cells/tick'); axes[1].set_xlabel('Simulation tick')
    for ax in axes: ax.legend(ncol=2); ax.grid(alpha=.2)
    fig.suptitle(f'Representative coverage-selected scenario {first_seed}: nominal speed and actual displacement')
    fig.tight_layout(); fig.savefig(destination / 'time_trace.png', dpi=140); plt.close(fig)
    clips = []
    for selection in protocol['selected_demos']:
        seed = selection['scenario_seed']
        meta_path = destination / f'seed{seed}_export.json'
        if meta_path.exists(): clips.append(read_json(meta_path)); continue
        meta = clip(seed, traces, destination)
        save_json(meta_path, meta); clips.append(meta)
    save_json(destination / 'render_checks.json', {'clips': clips, 'gui_verified': False, 'added_dependencies': [], 'physics_independent_of_playback': True})
    write_report(config, output, protocol, clips)


def write_report(config, output, protocol, clips):
    rows = read_csv(output / 'evaluation/comparison.csv')
    checks = read_json(output / 'evaluation/checks.json')
    lines = ['# Tracking25 V2: Continuing-Tracking Development Results', '',
             'This is a new fully observed continuing-tracking simulation. A hit does not terminate or freeze the target. Every evaluation window contains all 300 ticks, and target motion is paired across policies. Each learned policy chooses its own slow/fast actions; no heuristic override is applied.',
             'Both methods use training seed 0, zero float32 initialization and 20,000,000 interactions. Validation scenarios are 3600000–3600199; no independent final test or multi-seed formal experiment is used. The earlier hit-once trial measures a different task.', '',
             '| Policy | Mean distance | Close fraction | Hits/window | Shots/window | Misses/window | Base return | Pooled accuracy | First acquisition (capped) | Acquired fraction |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        accuracy = 'null (no shots)' if r['pooled_shot_accuracy'] == '' else f"{100*float(r['pooled_shot_accuracy']):.3f}%"
        lines.append(f"| {r['method']} | {float(r['mean_distance']):.4f} | {100*float(r['close_fraction']):.3f}% | {float(r['hits_per_episode']):.4f} | {float(r['shots_per_episode']):.3f} | {float(r['misses_per_episode']):.3f} | {float(r['mean_base_return']):.4f} | {accuracy} | {float(r['mean_first_acquisition_tick']):.3f} | {100*float(r['acquired_fraction']):.3f}% |")
    lines += ['', '## Speed Choice and Observed Recovery', '', '| Policy | Slow movement share | Fast movement share | Recovery fraction | Mean capped recovery lag | Actual displacement/tick | Target nominal speed/tick | Target path/tick | Target net displacement/tick |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        fmt = lambda k: 'null' if r[k] == '' else f'{float(r[k]):.4f}'
        lines.append('| ' + r['method'] + ' | ' + ' | '.join(fmt(k) for k in ('slow_movement_fraction', 'fast_movement_fraction', 'recovery_fraction', 'mean_capped_recovery_lag', 'mean_crosshair_displacement', 'mean_target_nominal_speed', 'mean_target_path_distance', 'mean_target_net_displacement')) + ' |')
    lines += ['', 'Movement shares exclude stay/fire and are null when there are no movement actions. Accuracy is pooled from hit/shot counts, not an average of per-window percentages. Distance and proximity include every post-motion tick. First acquisition uses the first post-step distance <=1; non-acquisition is capped at 300 with a separate flag.',
              'Eligible recovery events are deduplicated ticks with a random turn, speed toggle or reflection at tick <=280. Recovery searches lags 0–20; failures receive capped lag 20 and recovered=false. Later events can overlap the window: this is observed recovery around changes, not an isolated causal reaction time.', '', '## Execution and Mechanism Checks', '']
    for mode, meta in checks['run_metadata'].items():
        lines.append(f"- {mode}: {meta['global_steps']:,} interactions; {meta['complete_windows']:,} full windows and a {meta['partial_window']['steps']}-tick partial window; elapsed {meta['elapsed_seconds']:.3f} seconds; exact final-model reload verified.")
    lines += ['', 'Timing includes training, checkpoint validation, streamed logs, final save and chunked reload verification. Evaluation is independent of training environment and exploration randomness. No eleven-table checkpoint archive is saved. Both runs begin from zero. Configurations, seed derivation, original source hashes and demo selection were recorded before training in protocol.json.',
              f"Validation records: {checks['validation_summaries']} summaries, {checks['validation_episode_records']} windows, {checks['validation_interactions']:,} ticks. Baselines: {checks['baseline_episode_records']} windows, {checks['baseline_interactions']:,} ticks. Target path hashes match across all checkpoints and policies for all 200 scenarios.", '', '## Demonstrations', '',
              'The first three qualifying seeds were selected before training using target motion alone: x/y span >=16 and at least 50 ticks at each speed. These coverage-selected illustrations are not the unbiased evaluation set. All four panels continue through 300 ticks after hits. A hit star marks the actual shot coordinate before target motion. Playback is about 15 FPS and does not change physics.']
    for selection, exported in zip(protocol['selected_demos'], clips):
        lines.append(f"- Seed {selection['scenario_seed']}: x/y spans {selection['x_span']}/{selection['y_span']}, speed-1/2 occupancy {selection['speed_1_ticks']}/{selection['speed_2_ticks']} ticks; {exported['frames']} GIF frames, {exported['duration_ms']/1000:.2f} seconds playback.")
    lines += ['', '## Limitations and Reproduction', '',
              'Mechanics are checked separately from learned skills. This single training seed and finite development set do not establish statistical significance or real-game generalization. Sustained tracking, hit counts, speed choices and misses must be interpreted together. A policy can maintain proximity without firing accurately. The predictive rule is a velocity-aware heuristic, not a proven optimum. No training budget or physics was adjusted after observing results. GUI behavior remains unverified; exported GIFs are checked separately.',
              'Run from the project root:', '', '```powershell',
              '.\\.venv\\Scripts\\python.exe -m aimrl.tracking25 --config configs/tracking25_v2.json --stage check',
              '.\\.venv\\Scripts\\python.exe -m aimrl.tracking25 --config configs/tracking25_v2.json --stage train',
              '.\\.venv\\Scripts\\python.exe -m aimrl.tracking25 --config configs/tracking25_v2.json --stage evaluate',
              '.\\.venv\\Scripts\\python.exe -m aimrl.tracking25 --config configs/tracking25_v2.json --stage render', '```', '',
              'Completed training stages are reused only after configuration, model hash and interaction accounting checks. Interrupted training is not resumable: no full RNG/training state is saved. Use a clearly named new session directory for incomplete or incompatible work. Historical experiments and Git identity settings are preserved; no GitHub push is performed.']
    (output / 'results_summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
