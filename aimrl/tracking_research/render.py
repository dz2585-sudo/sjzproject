"""English figures and complete saved-model animations; no training in rendering."""
import math
import time
from pathlib import Path

import numpy as np

from .agent import ResearchAgent
from .evaluate import evaluate
from .io import ROOT, historical_agent, read_json, save_csv, save_json
from .workflow import METHODS, pair_rows, verify_protocol

LABELS = {'absolute_base': 'Absolute Base (5M)', 'absolute_potential': 'Absolute Potential (5M)',
          'relative_base': 'Relative Base (5M)', 'relative_potential': 'Relative Potential (5M)',
          'historical_base': 'Archived Absolute Base (20M, seed 0)',
          'historical_potential': 'Archived Absolute Potential (20M, seed 0)',
          'rule': 'Predictive rule (untrained)', 'random': 'Random (untrained negative control)'}
COLORS = ('#377eb8','#984ea3','#4daf4a','#ff7f00')


def seed_values(summary, method, field):
    return np.array([summary[f'{method}_seed{s}'][field] for s in (10,11,12)],dtype=float)


def plots(config, output):
    from ..mpl_setup import configure_mpl
    configure_mpl()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    destination = output / 'render'; destination.mkdir(parents=True,exist_ok=True)
    final = read_json(output / 'evaluation/final/summary.json')
    fig,axes = plt.subplots(2,3,figsize=(15,8))
    fields = ('mean_distance','close_fraction','far_fraction','hits_per_episode','pooled_shot_accuracy','mean_base_return')
    units = ('Mean L1 distance (cells)','Close fraction (d <= 1)','Far fraction (d > 5)','Hits / 300 ticks','Pooled accuracy','Unshaped base return / window')
    for method,color in zip(METHODS,COLORS):
        validations = [read_json(output/'runs'/method/f'seed_{s}'/'validation.json') for s in (10,11,12)]
        x = np.array([row['global_step'] for row in validations[0]])/1e6
        for ax,field,unit in zip(axes.flat,fields,units):
            y = np.array([[r['metrics'][field] if r['metrics'][field] is not None else np.nan for r in run] for run in validations])
            mean,std = np.mean(y,axis=0),np.std(y,axis=0,ddof=1)
            ax.plot(x,mean,color=color,label=LABELS[method]); ax.fill_between(x,mean-std,mean+std,color=color,alpha=.12)
            ax.set_xlabel('Training interactions (millions)'); ax.set_ylabel(unit); ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Matched 5M study: development curves | bands = sample SD across three training seeds')
    fig.tight_layout(); fig.savefig(destination/'learning_curves.png',dpi=150); plt.close(fig)
    fig,axes = plt.subplots(2,3,figsize=(15,8))
    for ax,field,unit in zip(axes.flat,fields,units):
        values = [seed_values(final,m,field) for m in METHODS]
        ax.bar(range(4),[v.mean() for v in values],yerr=[v.std(ddof=1) for v in values],capsize=3,color=COLORS)
        for i,v in enumerate(values): ax.scatter([i]*3,v,color='black',s=12,zorder=3)
        for ref,style in [('historical_potential','--'),('rule',':')]:
            ax.axhline(final[ref][field],linestyle=style,color='#555555' if ref=='rule' else '#a35c00',label=LABELS[ref])
        ax.set_xticks(range(4),['Abs\nBase','Abs\nPotential','Rel\nBase','Rel\nPotential']); ax.set_ylabel(unit); ax.grid(axis='y',alpha=.2)
    axes[0,0].legend(fontsize=7)
    fig.suptitle('Independent final test: 500 paired windows | bars = seed means; error bars = sample SD; dots = all seeds')
    fig.tight_layout(); fig.savefig(destination/'final_comparison.png',dpi=150); plt.close(fig)
    fig,axes = plt.subplots(1,3,figsize=(14,4))
    for method,color in zip(METHODS,COLORS):
        hist = np.mean([final[f'{method}_seed{s}']['distance_histogram'] for s in (10,11,12)],axis=0)
        p = hist/hist.sum(); x = np.arange(49)
        axes[0].plot(x,p,label=LABELS[method],color=color)
        axes[1].step(x,1-np.cumsum(p),where='post',color=color)
        axes[2].bar(list(METHODS).index(method),np.mean([final[f'{method}_seed{s}']['mean_longest_far_streak'] for s in (10,11,12)]),color=color)
    axes[0].set(xlabel='Post-motion L1 distance',ylabel='Pooled tick probability'); axes[0].legend(fontsize=7)
    axes[1].set(xlabel='Distance threshold (cells)',ylabel='Fraction exceeding threshold'); axes[1].set_yscale('symlog',linthresh=1e-4)
    axes[2].set(ylabel='Mean longest far streak (ticks / window)',xticks=range(4),xticklabels=['Abs B','Abs P','Rel B','Rel P'])
    for ax in axes: ax.grid(alpha=.2)
    fig.suptitle('Distance tails | equal-weight training seeds; initial frames excluded')
    fig.tight_layout(); fig.savefig(destination/'distance_tails.png',dpi=150); plt.close(fig)
    contexts = ('boundary_near','interior','target_near','target_interior','crosshair_near','crosshair_interior','speed_1','speed_2')
    fig,axes = plt.subplots(1,3,figsize=(16,5))
    for method,color in zip(METHODS,COLORS):
        for ax,field in zip(axes,('mean_distance','close_fraction','far_fraction')):
            y = [np.mean([final[f'{method}_seed{s}']['slices'].get(c,{}).get(field,np.nan) for s in (10,11,12)]) for c in contexts]
            ax.plot(range(len(contexts)),y,'o-',color=color,label=LABELS[method]); ax.set_ylabel(field.replace('_',' '))
            ax.set_xticks(range(len(contexts)),[c.replace('_','\n') for c in contexts],fontsize=7); ax.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    fig.suptitle('Pre-action diagnostic slices | policy-dependent denominators in JSON; not randomized subgroups')
    fig.tight_layout(); fig.savefig(destination/'boundary_speed_diagnostics.png',dpi=150); plt.close(fig)
    fig,axes = plt.subplots(1,3,figsize=(14,4))
    for method,color in zip(METHODS,COLORS):
        v = [read_json(output/'runs'/method/f'seed_{s}'/'validation.json') for s in (10,11,12)]
        x = [r['global_step']/1e6 for r in v[0]]
        for ax,field in zip(axes[:2],('visited_row_fraction','visited_pair_fraction')):
            ax.plot(x,np.mean([[r['coverage'][field] for r in run] for run in v],axis=0),label=LABELS[method],color=color)
            ax.set_ylabel(field.replace('_',' ')); ax.set_xlabel('Training interactions (millions)')
        axes[2].plot(x,np.mean([[r['metrics']['slices']['all']['zero_visit_row_fraction'] for r in run] for run in v],axis=0),color=color)
    axes[2].set(xlabel='Training interactions (millions)',ylabel='Development ticks at zero-visit rows')
    axes[0].legend(fontsize=7)
    for ax in axes: ax.grid(alpha=.2)
    fig.suptitle('Genuine update coverage | relative aggregation merges physical states; percentages are not equivalent')
    fig.tight_layout(); fig.savefig(destination/'coverage.png',dpi=150); plt.close(fig)
    fig,axes = plt.subplots(1,4,figsize=(14,4),sharey=True)
    for ax,method in zip(axes,METHODS):
        vals = []
        for bin_name in ('0_3','4_11','12_plus'):
            fractions = []
            for s in (10,11,12):
                row = final[f'{method}_seed{s}']; slow,fast = row['slow_bin_'+bin_name],row['fast_bin_'+bin_name]
                fractions.append(fast/(slow+fast) if slow+fast else np.nan)
            vals.append(np.nanmean(fractions))
        ax.bar(['0–3','4–11','>=12'],1-np.array(vals),label='Slow',color='#377eb8')
        ax.bar(['0–3','4–11','>=12'],vals,bottom=1-np.array(vals),label='Fast',color='#ff7f00')
        ax.set_title(LABELS[method],fontsize=9); ax.set_xlabel('Pre-action distance (cells)')
    axes[0].set_ylabel('Proportion among movement actions only'); axes[-1].legend()
    fig.suptitle('Final-test speed choices | mean of three seed-level proportions; empty bins undefined')
    fig.tight_layout(); fig.savefig(destination/'speed_choices.png',dpi=150); plt.close(fig)


def animation(seed, policy_rows, destination, basename):
    from PIL import Image, ImageDraw, ImageFont
    started = time.perf_counter()
    policies = list(policy_rows); columns = 2 if len(policies)==4 else 3
    width,height = columns*540,990
    font_path = Path('C:/Windows/Fonts/arial.ttf')
    font = ImageFont.truetype(str(font_path),16)
    small = ImageFont.truetype(str(font_path),14)
    heading_font = ImageFont.truetype(str(font_path),20)
    frames = []; summaries = {}
    target_paths = [[(r['xt'],r['yt'],r['heading'],r['target_speed']) for r in rows] for rows in policy_rows.values()]
    if not all(path==target_paths[0] and len(path)==301 for path in target_paths): raise ValueError('Demo pairing/frame count failed')
    for tick in range(301):
        image = Image.new('RGB',(width,height),'white'); draw = ImageDraw.Draw(image)
        draw.text((22,8),f'Tracking representation study | scenario {seed} | tick {tick}/300 | ~15 FPS',font=heading_font,fill='#222222')
        draw.text((22,35),'Red circle: post-motion target | Blue cross: crosshair | Gold star: pre-motion hit coordinate',font=font,fill='#333333')
        for i,policy in enumerate(policies):
            rows = policy_rows[policy]; row = rows[tick]
            origin_x,origin_y = (i%columns)*540,65+(i//columns)*460
            label = LABELS[policy]
            if policy in METHODS: label += ' | training seed 10'
            draw.text((origin_x+18,origin_y),label,font=font,fill='#111111')
            from ..tracking25.env import ACTION_NAMES
            action = 'reset' if int(row['action'])<0 else ACTION_NAMES[int(row['action'])]
            draw.text((origin_x+18,origin_y+24),f"Target speed {row['target_speed']} | {action} ({row['movement_mode']})",font=small,fill='#222222')
            draw.text((origin_x+18,origin_y+44),f"Post-motion distance {row['distance']} | Shots {row['shots']} | Hits {row['hits']}",font=small,fill='#222222')
            left,top,cell = origin_x+78,origin_y+85,14
            def point(x,y): return left+(float(x)+.5)*cell,top+(float(y)+.5)*cell
            for j in range(26):
                draw.line((left+j*cell,top,left+j*cell,top+25*cell),fill='#e0e3e8')
                draw.line((left,top+j*cell,left+25*cell,top+j*cell),fill='#e0e3e8')
            for j in (0,6,12,18,24):
                px,py = point(j,j)
                draw.text((px-7,top+25*cell+4),str(j),font=small,fill='#555555')
                draw.text((left-25,py-8),str(j),font=small,fill='#555555')
            tail = rows[max(0,tick-8):tick+1]
            if len(tail)>1:
                draw.line([point(r['xt'],r['yt']) for r in tail],fill='#eab0a8',width=2)
                draw.line([point(r['xc'],r['yc']) for r in tail],fill='#a2bedf',width=2)
            tx,ty = point(row['xt'],row['yt']); cx,cy = point(row['xc'],row['yc'])
            draw.ellipse((tx-5,ty-5,tx+5,ty+5),outline='#d54a34',width=3)
            draw.line((cx-5,cy-5,cx+5,cy+5),fill='#1766b1',width=3)
            draw.line((cx-5,cy+5,cx+5,cy-5),fill='#1766b1',width=3)
            if tick:
                previous = rows[tick-1]; px,py = point(previous['xc'],previous['yc'])
                actual = abs(row['xc']-previous['xc'])+abs(row['yc']-previous['yc'])
                if actual != int(row['displacement']): raise ValueError('Actual-displacement rendering mismatch')
                if actual:
                    draw.line((px,py,cx,cy),fill='#144a86',width=2)
                    angle = math.atan2(cy-py,cx-px)
                    draw.polygon([(cx,cy),(cx-8*math.cos(angle-.45),cy-8*math.sin(angle-.45)),
                                  (cx-8*math.cos(angle+.45),cy-8*math.sin(angle+.45))],fill='#144a86')
            if row['hit'] in (True,'True'):
                sx,sy = point(row['shot_x'],row['shot_y'])
                star = [(sx+(9 if k%2==0 else 4)*math.cos(k*math.pi/5-math.pi/2),sy+(9 if k%2==0 else 4)*math.sin(k*math.pi/5-math.pi/2)) for k in range(10)]
                draw.polygon(star,fill='#e1a000')
            summaries[policy] = {'last_tick':tick,'hits':int(row['hits']), 'shots':int(row['shots'])}
        if tick in (0,150,300): image.save(destination/f'{basename}_source_{tick:03d}.png')
        if tick==150: image.save(destination/f'{basename}_preview.png')
        frames.append(image.quantize(colors=256,method=Image.Quantize.FASTOCTREE)); image.close()
    durations = [(round((i+1)*100/15)-round(i*100/15))*10 for i in range(301)]
    path = destination/f'{basename}.gif'
    frames[0].save(path,save_all=True,append_images=frames[1:],duration=durations,loop=0,disposal=2,optimize=False)
    for frame in frames: frame.close()
    actual_durations = []
    with Image.open(path) as gif:
        if gif.n_frames!=301: raise ValueError('GIF missing frames')
        for tick in range(gif.n_frames):
            gif.seek(tick); gif.load(); actual_durations.append(gif.info.get('duration',0))
            if tick in (0,150,300): gif.convert('RGB').save(destination/f'{basename}_decoded_{tick:03d}.png')
    if sum(actual_durations)!=20070: raise ValueError('Incorrect decoded GIF duration')
    return {'path':path.relative_to(ROOT).as_posix(),'scenario_seed':seed,'frames':301,'duration_ms':sum(actual_durations),
            'actual_frame_durations_ms':sorted(set(actual_durations)),'dimensions':[width,height],
            'elapsed_seconds':time.perf_counter()-started,'policies':summaries,'all_targets_paired':True,
            'actual_displacement_arrows_checked':True, 'gui_verified':False}


def render(config, output):
    verify_protocol(config,output,True)
    destination = output/'render'; destination.mkdir(parents=True,exist_ok=True)
    plots(config,output)
    agents = {}
    traces = {s:{} for s in config['study']['demo_seeds']}
    # Load one table at a time; retain only small, exact 301-frame traces.
    for policy in ('historical_potential',*METHODS,'rule','random'):
        agent = None
        if policy=='historical_potential': agent,_ = historical_agent(config,'potential')
        elif policy in METHODS:
            agent = ResearchAgent(config,policy.split('_')[0]); agent.load_package(output/'runs'/policy/'seed_10/q_table.npz',True)
        _,records,rows = evaluate(config,policy,config['study']['demo_seeds'],agent,True)
        for s in traces: traces[s][policy] = [r for r in rows if r['scenario_seed']==s]
        del agent
    clips = []
    for s in traces:
        main = {p:traces[s][p] for p in ('historical_potential','relative_base','relative_potential','rule')}
        name = f'seed{s}_main_comparison'
        save_csv(destination/f'{name}_traces.csv',[r for rows in main.values() for r in rows])
        clips.append(animation(s,main,destination,name))
        print('Rendered',name,'301 frames; decoded20.07 seconds',flush=True)
    s = 3601000
    six = {p:traces[s][p] for p in (*METHODS,'rule','random')}
    name = f'seed{s}_six_policy_comparison'
    save_csv(destination/f'{name}_traces.csv',[r for rows in six.values() for r in rows])
    clips.append(animation(s,six,destination,name))
    save_json(destination/'render_checks.json',{'clips':clips,'gui_verified':False,'all_frames_decoded':True})
    print('All four complete GIFs exported and decoded.',flush=True)
