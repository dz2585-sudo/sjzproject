"""Publication figures and fixed-seed decoded GIF checks."""
import hashlib

import numpy as np

from ..tracking25.env import ACTION_NAMES
from ..tracking_research.io import read_json, save_csv, save_json, sha
from .core import COLUMNS, evaluate, load_model
from .study import require, run_dir

LABEL = {'reference':'Reference','distance_0p1':'Distance 0.1','distance_0p3':'Distance 0.3',
         'distance_1p0':'Distance 1.0','close_0p5':'Close 0.5','close_2p0':'Close 2.0',
         'close_5p0':'Close 5.0','rule':'Rule'}
COLORS = {'reference':'#2e3644','distance_0p1':'#579cce','distance_0p3':'#247caf',
          'distance_1p0':'#10486e','close_0p5':'#ed9f58','close_2p0':'#c76c24',
          'close_5p0':'#873708','rule':'#51835d'}


def figures(config, output, final, per_seed, frontiers, actions):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    dest = output / 'figures'
    dest.mkdir(exist_ok=False)
    plt.rcParams.update({'axes.spines.top':False, 'axes.spines.right':False,
                         'axes.labelcolor':'#283345','text.color':'#202a39','font.size':10,
                         'axes.titleweight':'bold','figure.facecolor':'#ffffff','axes.axisbelow':True})
    by_id = {r['condition']:r for r in final}
    ref = by_id['reference']
    threshold = config['tradeoff']['thresholds']
    for metric, filename, title, xlabel in (
        ('D','distance_hits_tradeoff.png','Distance versus hits','Mean post-motion distance, cells (lower is better)'),
        ('C','close_hits_tradeoff.png','Close tracking versus hits','Post-motion ticks within one cell (higher is better)'),
    ):
        fig, ax = plt.subplots(figsize=(10.5,6), layout='constrained')
        front = [by_id[c] for c in frontiers[metric]]
        ax.plot([r[metric] for r in front],[r['H'] for r in front],color='#9da5af',lw=1.5,
                zorder=1,label='Empirical non-dominated means')
        if metric == 'C':
            ax.axvline(ref['C']+threshold['close_gain'], color='#7b6e8c', ls='--', lw=1, label='Close gain gate (+5 pp)')
            ax.axhline(ref['H']*threshold['simultaneous_hits_ratio'], color='#40725c', ls='--', lw=1, label='Hit gain gate (+2%)')
            ax.axhline(ref['H']*threshold['retention_hits_ratio'], color='#ab8051', ls=':', lw=1.4, label='Hit retention gate (−2%)')
            ax.xaxis.set_major_formatter(PercentFormatter(1))
            ax.text(.99,.025,'Full screens also require D ≤ reference and accuracy ≥ 98%',
                    transform=ax.transAxes,ha='right',va='bottom',fontsize=8,color='#465265')
        for index, row in enumerate(final):
            c = row['condition']
            seed_rows = [r for r in per_seed if r['split']=='final' and r['condition']==c and r['seed'] is not None]
            if seed_rows:
                ax.scatter([r[metric] for r in seed_rows],[r['H'] for r in seed_rows],
                           color=COLORS[c],s=18,alpha=.30,zorder=2)
            ax.errorbar(row[metric],row['H'],xerr=row[metric+'_sd'],yerr=row['H_sd'],
                        color=COLORS[c],fmt='D' if c=='rule' else 'o',markersize=6,capsize=3,lw=1.1,zorder=3)
            offsets = [(6,12),(-5,18),(-7,-19),(8,-12),(5,-20),(7,17),(-7,15),(8,8)]
            dx,dy = offsets[index]
            ax.annotate(LABEL[c],(row[metric],row['H']),xytext=(dx,dy),textcoords='offset points',
                        color=COLORS[c],fontsize=8,ha='left' if dx>0 else 'right')
        ax.set(title=title,xlabel=xlabel,ylabel='Hits per 300-tick window')
        ax.grid(alpha=.23)
        ax.margins(x=.16,y=.18)
        ax.legend(loc='lower left',frameon=False,fontsize=8)
        fig.suptitle('Final set · 500 paired scenarios · mean ± sample SD across three training seeds',fontsize=10)
        fig.savefig(dest/filename,dpi=170)
        plt.close(fig)

    fig, axes = plt.subplots(4,2,figsize=(10,9.4),layout='constrained')
    for col, (family, weight_index) in enumerate((('distance',0),('close',1))):
        members = ['reference',*(c for c in by_id if c.startswith(family+'_'))]
        weights = [config['tradeoff']['conditions'][c][weight_index] for c in members]
        for row, (metric, label) in enumerate((('D','Distance (cells)'),('C','Close ticks'),('H','Hits / window'),('A','Accuracy'))):
            ax=axes[row,col]
            ax.errorbar(weights,[by_id[c][metric] if by_id[c][metric] is not None else np.nan for c in members],
                        yerr=[by_id[c][metric+'_sd'] or 0 for c in members],
                        color=COLORS[members[-1]],marker='o',capsize=3)
            ax.set_ylabel(label)
            ax.grid(alpha=.2)
            if metric in ('C','A'):
                ax.yaxis.set_major_formatter(PercentFormatter(1))
            if row == 0:
                ax.set_title('Distance penalty' if col==0 else 'Close bonus')
            if row == 3:
                ax.set_xlabel('Distance coefficient λ' if col==0 else 'Close bonus coefficient β')
    fig.suptitle('Final-set weight response · three seed means and sample SD')
    fig.savefig(dest/'weight_response.png',dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(3,2,figsize=(12,8),sharex=True,layout='constrained')
    for col,family in enumerate(('distance','close')):
        members = ['reference',*(c for c in by_id if c.startswith(family+'_'))]
        detail = axes[0,col].inset_axes([.30,.17,.66,.44])
        detail.set_title('Distance after the first checkpoint',fontsize=7,pad=3)
        detail.tick_params(labelsize=7)
        detail.grid(alpha=.2)
        for condition in members:
            data=[read_json(run_dir(output,condition,s)/'development_summary.json') for s in config['tradeoff']['training_seeds']]
            steps=np.array([r['step'] for r in data[0]])/1e6
            for i,(metric,label) in enumerate((('D','Distance (cells)'),('C','Close ticks'),('H','Hits / window'))):
                ax=axes[i,col]
                values=[[r[metric] for r in seed] for seed in data]
                for values_for_seed in values:
                    ax.plot(steps,values_for_seed,color=COLORS[condition],alpha=.24,lw=.85)
                    if metric == 'D':
                        detail.plot(steps[1:],values_for_seed[1:],color=COLORS[condition],alpha=.25,lw=.65)
                ax.plot(steps,np.mean(values,axis=0),color=COLORS[condition],label=LABEL[condition],lw=1.65)
                if metric == 'D':
                    detail.plot(steps[1:],np.mean(values,axis=0)[1:],color=COLORS[condition],lw=1.15)
                ax.set_ylabel(label)
                ax.grid(alpha=.2)
                if metric=='C':
                    ax.yaxis.set_major_formatter(PercentFormatter(1))
        axes[0,col].set_title('Distance penalty' if col==0 else 'Close bonus')
        axes[0,col].legend(frameon=False,ncol=2,fontsize=8)
        axes[-1,col].set_xlabel('Training interactions (millions)')
    fig.suptitle('Development learning curves · each faint line is one training seed')
    fig.savefig(dest/'learning_curves.png',dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2,2,figsize=(13,8),layout='constrained')
    ix=np.arange(len(final))
    axes[0,0].bar(ix,[r['alignment_opportunity_fraction'] for r in final],color=[COLORS[r['condition']] for r in final])
    axes[0,0].set_title('Pre-action alignment opportunities')
    axes[0,1].bar(ix,[r['fire_given_alignment'] or 0 for r in final],color=[COLORS[r['condition']] for r in final])
    axes[0,1].set_title('Probability of firing when aligned')
    for j,(metric,color) in enumerate((('shots','#48648d'),('H','#548566'),('misses','#bc6e56'))):
        axes[1,0].bar(ix+(j-1)*.25,[r[metric] for r in final],width=.25,label={'H':'Hits'}.get(metric,metric.title()),color=color)
    axes[1,0].set_title('Shots, hits and misses per window')
    axes[1,0].legend(frameon=False)
    bottom=np.zeros(len(final))
    for mode,color in zip(('slow','fast','stay','fire'),('#7bb3c6','#42648c','#ccb995','#b86d51')):
        fractions=[]
        for r in final:
            subset=[a for a in actions if a['split']=='final' and a['condition']==r['condition']]
            fractions.append(sum(a[mode+'_count'] for a in subset)/sum(a['ticks'] for a in subset))
        axes[1,1].bar(ix,fractions,bottom=bottom,color=color,label=mode.title())
        bottom+=fractions
    axes[1,1].set_title('Action mix')
    axes[1,1].legend(frameon=False,ncol=4,loc='upper center')
    for ax in (axes[0,0],axes[0,1],axes[1,1]):
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    for ax in axes.flat:
        ax.set_xticks(ix,[LABEL[r['condition']] for r in final],rotation=26,ha='right')
        ax.grid(axis='y',alpha=.18)
    fig.suptitle('Final-set mechanism diagnostics')
    fig.savefig(dest/'alignment_and_shooting.png',dpi=160)
    plt.close(fig)


def trace_rows(condition, seed, scenario, initial, steps):
    d0=abs(initial[0]-initial[2])+abs(initial[1]-initial[3])
    start=(*initial,-1,d0,d0,d0,0,0,0,0,0,0)
    records=[]
    hits=shots=close=0
    previous=initial
    for tick,row in enumerate([start,*steps]):
        if tick:
            hits+=int(row[11])
            shots+=int(row[6]==5)
            close+=int(row[9]<=1)
        records.append({'condition':condition,'training_seed':seed,'scenario_seed':scenario,'tick':tick,
                        **dict(zip(COLUMNS,map(int,row))),
                        **{f'pre_{k}':int(v) for k,v in zip(COLUMNS[:6],previous)},
                        'action_name':ACTION_NAMES[row[6]] if tick else 'reset',
                        'cumulative_hits':hits,'cumulative_shots':shots,'cumulative_close_ticks':close,
                        'cumulative_close_fraction':close/tick if tick else None})
        previous=row[:6]
    return records


def diagnostic_fragments(config,kernel,output):
    design=config['tradeoff']
    seed=design['display_training_seed']
    scenario=design['final']['start']
    all_rows, fragments=[],[]
    for c in (*design['conditions'],'rule'):
        agent=None if c=='rule' else load_model(run_dir(output,c,seed)/'checkpoints/step_5000000.npz',config,c,seed,5000000)
        _,trace=evaluate(config,kernel,agent,[scenario],keep_traces=True)
        rows=trace_rows(c,seed if agent else None,scenario,*trace[scenario])
        all_rows.extend(rows)
        def crossing(row, x, y):
            target_x, target_y = row[f'pre_{x[0]}t'], row[f'pre_{y[0]}t']
            return (row['pre_'+y]==row[y]==target_y
                    and (row['pre_'+x]-target_x)*(row[x]-target_x)<0)
        overshoot=next((r['tick'] for r in rows[1:] if crossing(r,'xc','yc') or crossing(r,'yc','xc')),None)
        stay=next((r['tick'] for r in rows[1:] if r['action']==4),None)
        offset=next((i for i in range(1,299) if all(r['d_post']==1 for r in rows[i:i+3])),None)
        for label,index in (('crossed_pre_motion_target',overshoot),('stay',stay),('three_ticks_one_cell_offset',offset)):
            fragments.append({'condition':c,'scenario_seed':scenario,'event':label,'first_tick':index if index is not None else 'NA',
                              'fragment_start':max(0,index-2) if index is not None else 'NA',
                              'fragment_end':min(300,index+3) if index is not None else 'NA'})
    save_csv(output/'diagnostics'/'fixed_final_trace.csv',all_rows)
    save_csv(output/'diagnostics'/'first_fragments.csv',fragments)


def gif_frame(rows, tick, scenario, labels, colors, fonts):
    from PIL import Image, ImageDraw
    size=(840,806)
    palette, ink, muted, grid, cross, target, hit, header, paper, *extra=colors
    accents, target_tail, crosshair_tail=extra[:4],extra[4],extra[5]
    im=Image.new('P',size,paper)
    im.putpalette(palette)
    draw=ImageDraw.Draw(im)
    draw.fontmode='1'
    large, small, bold=fonts
    draw.rectangle((0,0,size[0],63),fill=header)
    draw.text((16,9),f'Tracking and shooting  |  Scenario {scenario}  |  Tick {tick:03d} / 300',font=large,fill=paper)
    draw.text((17,38),'Blue cross: crosshair   Red ring: post-motion target   Gold dot: hit location before target movement',
              font=small,fill=paper)
    for index,(condition, records) in enumerate(rows.items()):
        row=records[tick]
        x=12+420*(index%2)
        y=75+363*(index//2)
        draw.text((x+4,y),labels[condition],font=bold,fill=accents[index])
        close='NA' if not tick else f'{row["cumulative_close_fraction"]:.1%}'
        draw.text((x+4,y+25),f'Distance {row["d_post"]}  |  Hits {row["cumulative_hits"]}  |  Shots {row["cumulative_shots"]}  |  Close {close}',
                  font=small,fill=ink)
        draw.text((x+4,y+47),f'Action {row["action_name"]}   |   Target speed {row["speed"]}',font=small,fill=muted)
        left,top,cell=x+61,y+68,11
        def point(px,py):
            return left+px*cell+5,top+py*cell+5
        for k in range(26):
            draw.line((left+k*cell,top,left+k*cell,top+275),fill=grid)
            draw.line((left,top+k*cell,left+275,top+k*cell),fill=grid)
        tail=records[max(0,tick-8):tick+1]
        if len(tail)>1:
            draw.line([point(r['xt'],r['yt']) for r in tail],fill=target_tail)
            draw.line([point(r['xc'],r['yc']) for r in tail],fill=crosshair_tail)
        cx,cy=point(row['xc'],row['yc'])
        tx,ty=point(row['xt'],row['yt'])
        draw.ellipse((tx-4,ty-4,tx+4,ty+4),outline=target,width=2)
        draw.line((cx-4,cy-4,cx+4,cy+4),fill=cross,width=2)
        draw.line((cx-4,cy+4,cx+4,cy-4),fill=cross,width=2)
        if row['hit']:
            draw.ellipse((cx-3,cy-3,cx+3,cy+3),fill=hit)
    return im


def demos(config,kernel,output):
    from PIL import Image, ImageFont
    design=config['tradeoff']
    selection=read_json(output/'selection.json')['selected']
    conditions=['reference',selection['distance']['condition'],selection['close']['condition'],'rule']
    seed=design['display_training_seed']
    scenarios=design['demo_seeds']
    destination=output/'demos'
    destination.mkdir(exist_ok=False)
    by_scenario={s:{} for s in scenarios}
    labels={}
    for c in conditions:
        agent=None if c=='rule' else load_model(run_dir(output,c,seed)/'checkpoints/step_5000000.npz',config,c,seed,5000000)
        _,traces=evaluate(config,kernel,agent,scenarios,keep_traces=True)
        if c=='rule':
            labels[c]='Rule  |  Fixed policy'
        else:
            lam,beta=design['conditions'][c]
            labels[c]=f'{LABEL[c]}  |  lambda={lam:g}  beta={beta:g}  |  Seed {seed}'
        for s,(initial,steps) in traces.items():
            by_scenario[s][c]=trace_rows(c,seed if agent else None,s,initial,steps)
    palette=[32,42,57,99,111,129,224,229,237,28,101,178,196,63,55,241,168,22,33,46,66,255,255,255,
             46,54,68,25,103,151,155,73,20,54,113,70,228,173,168,144,182,218]+[0]*(768-42)
    colors=(palette,*range(14))
    fonts=tuple(ImageFont.load_default(size=n) for n in (22,15,18))
    checks=[]
    for s, rows in by_scenario.items():
        target_paths=[[(r['xt'],r['yt'],r['heading'],r['speed']) for r in trace] for trace in rows.values()]
        require(all(path==target_paths[0] for path in target_paths), 'Demo target routes differ')
        aim_paths=[[(r['xc'],r['yc']) for r in trace] for trace in rows.values()]
        path=destination/f'scenario_{s}.gif'
        frames=[gif_frame(rows,t,s,labels,colors,fonts) for t in range(301)]
        timing=[(round((i+1)*100/15)-round(i*100/15))*10 for i in range(301)]
        frames[0].save(path,save_all=True,append_images=frames[1:],duration=timing,loop=0,disposal=2,optimize=False)
        sample_checks={}
        with Image.open(path) as decoded:
            require(decoded.n_frames==301,'GIF frame count differs')
            durations=[]
            for i in range(decoded.n_frames):
                decoded.seek(i)
                rgb=decoded.convert('RGB')
                require(rgb.tobytes()==frames[i].convert('RGB').tobytes(),f'Decoded GIF pixels differ at tick {i}')
                durations.append(decoded.info['duration'])
                if i in (0,150,300):
                    rgb.save(destination/f'scenario_{s}_tick_{i:03d}.png')
                    sample_checks[str(i)]=hashlib.sha256(rgb.tobytes()).hexdigest()
            require(durations==timing and sum(durations)==20070,'GIF timing differs')
        for frame in frames:
            frame.close()
        save_csv(destination/f'scenario_{s}_trace.csv',[r for trace in rows.values() for r in trace])
        checks.append({'scenario_seed':s,'path':path.relative_to(output).as_posix(),'sha256':sha(path),
                       'frames':301,'duration_ms':20070,'target_paths_paired':True,
                       'distinct_crosshair_paths':len({tuple(p) for p in aim_paths}),
                       'decoded_pixels_match_every_frame':True,'sample_rgb_sha256':sample_checks,
                       'panels':{c:{'hits':trace[-1]['cumulative_hits'],'shots':trace[-1]['cumulative_shots'],
                                    'close_fraction':trace[-1]['cumulative_close_fraction']} for c,trace in rows.items()}})
    save_json(destination/'render_checks.json',checks)
    return checks
