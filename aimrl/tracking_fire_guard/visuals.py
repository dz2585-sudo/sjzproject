"""Objective plots and a frame-verified side-by-side replay."""
import hashlib

from ..tracking_research.io import sha
from ..tracking_tradeoff.visuals import COLORS, LABEL
from .analysis import export
from .replay import replay_pair, require
from .study import load_frozen


def figures(output,summaries,screens):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    import numpy as np

    destination=output/'figures'
    destination.mkdir(exist_ok=True)
    original={r['condition']:r for r in summaries if r['split']=='final' and r['variant']=='original'}
    guarded={r['condition']:r for r in summaries if r['split']=='final' and r['variant']=='guard'}
    reference=original['reference']
    fig,axes=plt.subplots(1,2,figsize=(13,5.8),layout='constrained')
    frontiers={}
    for ax,key,label,direction in zip(axes,('D','C'),('Mean post-motion distance (cells)','Post-motion ticks within one cell'),(-1,1)):
        def dominates(a,b):
            d,h=direction*(a[key]-b[key]),a['H']-b['H']
            return d>=0 and h>=0 and (d>0 or h>0)
        candidates=list(original.values())
        front=sorted([r for r in candidates if not any(dominates(a,r) for a in candidates)],key=lambda r:r[key])
        frontiers[key]=[r['condition'] for r in front]
        ax.plot([r[key] for r in front],[r['H'] for r in front],color='#8798a7',lw=1.7,label='Observed non-dominated means')
        for idx,(condition,row) in enumerate(original.items()):
            if condition in guarded:
                require((row[key],row['H'])==(guarded[condition][key],guarded[condition]['H']),'Objective points do not overlap')
                ax.scatter(row[key],row['H'],s=105,edgecolors=COLORS[condition],facecolors='none',lw=1.2,zorder=3)
            ax.errorbar(row[key],row['H'],xerr=row[key+'_sd'],yerr=row['H_sd'],color=COLORS[condition],
                        fmt='D' if condition=='rule' else 'o',markersize=4,lw=.9,capsize=2,zorder=4)
            offsets=(
                [(6,12),(-7,15),(-7,-20),(8,-14),(-6,-12),(7,15),(-6,9),(7,6)]
                if key=='D' else
                [(6,10),(-7,15),(5,-18),(7,-11),(-7,-19),(7,15),(-6,9),(7,6)]
            )
            dx,dy=offsets[idx]
            ax.annotate(LABEL[condition],(row[key],row['H']),xytext=(dx,dy),textcoords='offset points',
                        ha='left' if dx>0 else 'right',color=COLORS[condition],fontsize=8,
                        bbox={'facecolor':'white','alpha':.72,'edgecolor':'none','pad':.6},zorder=5)
        ax.axhline(1.02*reference['H'],ls='--',lw=.8,color='#68836a',label='+2% hit gate')
        ax.axhline(.98*reference['H'],ls=':',lw=1,color='#9d7054',label='−2% hit gate')
        if key=='C':
            ax.axvline(reference['C']+.05,ls='--',lw=.9,color='#7a679b',label='+5 pp close gate')
            ax.xaxis.set_major_formatter(PercentFormatter(1))
        else:
            ax.axvline(reference['D'],ls='--',lw=.9,color='#7a679b',label='Reference distance')
        ax.set(xlabel=label,ylabel='Hits per 300-tick window')
        ax.grid(alpha=.18)
        ax.margins(.18,.13)
        ax.legend(loc='lower right' if key=='D' else 'lower left',fontsize=7,frameon=False)
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Firing guard · objective coordinates are unchanged\n'
                 'Filled point: original · ring: guarded at the identical point · bars: sample SD across seeds')
    fig.savefig(destination/'objective_tradeoffs.png',dpi=170)
    plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(11,7),layout='constrained',sharex=True)
    condition=list(original)
    x=np.arange(len(condition))
    for shift,values,label,color in ((-.16,original,'Original','#56799e'),(.16,guarded,'Firing guard','#459679')):
        for ax,key in zip(axes,('A','misses')):
            y=[values[c][key] if c in values and values[c][key] is not None else np.nan for c in condition]
            ax.bar(x+shift,y,width=.30,color=color,label=label)
    axes[0].axhline(.98,color='#a66747',ls='--',lw=1,label='98% accuracy gate')
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].set_ylabel('Mean seed accuracy')
    axes[0].set_ylim(0,1.15)
    axes[0].legend(frameon=False,loc='upper center',ncol=3)
    axes[1].set_ylabel('Misses per window')
    axes[1].set_xticks(x,[LABEL[c] for c in condition])
    for ax in axes:
        ax.grid(axis='y',alpha=.18)
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Held-out final set · shot precision and discarded misses')
    fig.savefig(destination/'accuracy_and_misses.png',dpi=170)
    plt.close(fig)
    return frontiers


def demo(output,protocol):
    from PIL import Image,ImageDraw,ImageFont
    preset=protocol['demo']
    condition,seed,scenario=(preset[k] for k in ('condition','training_seed','scenario_seed'))
    model=next(m for m in protocol['models'] if (m['condition'],m['training_seed'])==(condition,seed))
    agent=load_frozen(protocol['config'],condition,seed,model['sha256'])
    before=agent.snapshot()
    initial,_,_,a,b,_,trace=replay_pair(protocol['config'],scenario,agent.greedy,agent,strict_rng=True,keep_trace=True)
    require(agent.snapshot()==before,'Demo changed frozen learner')
    initial_row={k:v for k,v in zip(('xc','yc','xt','yt','heading','speed'),initial)}
    initial_row.update(scenario_seed=scenario,tick=0,d_pre=abs(initial[0]-initial[2])+abs(initial[1]-initial[3]),
                       proposed_action='reset',guard_executed_action='reset',hit=0,original_miss=0,guard_miss=0,filtered=0)
    initial_row['d_post']=initial_row['d_pre']
    display=[initial_row,*trace]
    hits=misses=raw_shots=guard_shots=0
    for row in display:
        hits+=row['hit']
        misses+=row['filtered']
        raw_shots+=row['proposed_action']=='fire'
        guard_shots+=row['guard_executed_action']=='fire'
        row.update(cumulative_hits=hits,cumulative_removed=misses,
                   cumulative_original_shots=raw_shots,cumulative_guard_shots=guard_shots)
    require((hits,misses,raw_shots,guard_shots)==(a['hits'],a['misses'],a['shots'],b['shots']),
            'Demo counters do not match executed actions')
    folder=output/'demo'
    folder.mkdir(exist_ok=True)
    export(folder,'paired_ticks',trace)
    palette=[31,43,61,95,109,123,218,226,234,32,110,174,197,64,59,232,156,32,255,255,255,
             48,69,96,34,115,91,213,169,164,160,193,220]+[0]*(768-33)
    large,medium,small=(ImageFont.load_default(size=s) for s in (23,19,15))

    def frame(index):
        row=display[index]
        im=Image.new('P',(880,501),6)
        im.putpalette(palette)
        draw=ImageDraw.Draw(im)
        draw.fontmode='1'
        draw.rectangle((0,0,879,62),fill=0)
        draw.text((19,9),f'Firing guard  |  Scenario {scenario}  |  Tick {index:03d} / 300',font=large,fill=6)
        draw.text((20,39),'Same frozen policy and target route. Blue: crosshair  Red: target  Gold: valid hit',font=small,fill=6)
        for panel,(title,action,shots,accent) in enumerate((
            ('Original',row['proposed_action'],row['cumulative_original_shots'],7),
            ('Q-learning + firing guard',row['guard_executed_action'],row['cumulative_guard_shots'],8),
        )):
            x=17+panel*440
            draw.text((x,77),title,font=medium,fill=accent)
            draw.text((x,104),f'Executed {action}  |  D pre/post {row["d_pre"]}/{row["d_post"]}  |  Hits {row["cumulative_hits"]}  |  Shots {shots}',
                      font=small,fill=0)
            if row['filtered']:
                label='Unaligned Fire replaced with Stay' if panel else 'Unaligned Fire (miss)'
            else:
                label='Actions match' if panel else 'Original greedy action'
            draw.text((x,129),label,font=small,fill=4 if row['filtered'] else 1)
            left,top,cell=x+56,153,12
            def point(px,py):
                return left+px*cell+6,top+py*cell+6
            for k in range(26):
                draw.line((left+k*cell,top,left+k*cell,top+300),fill=2)
                draw.line((left,top+k*cell,left+300,top+k*cell),fill=2)
            tail=display[max(0,index-7):index+1]
            if len(tail)>1:
                draw.line([point(r['xt'],r['yt']) for r in tail],fill=9)
                draw.line([point(r['xc'],r['yc']) for r in tail],fill=10)
            xc,yc=point(row['xc'],row['yc'])
            xt,yt=point(row['xt'],row['yt'])
            draw.ellipse((xt-4,yt-4,xt+4,yt+4),outline=4,width=2)
            draw.line((xc-4,yc-4,xc+4,yc+4),fill=3,width=2)
            draw.line((xc-4,yc+4,xc+4,yc-4),fill=3,width=2)
            if row['hit']:
                draw.ellipse((xc-3,yc-3,xc+3,yc+3),fill=5)
        draw.text((19,472),f'Frozen Close 5.0, training seed {seed}  |  Removed misses: {row["cumulative_removed"]}',
                  font=small,fill=0)
        return im

    frames=[frame(i) for i in range(301)]
    timing=[(round((i+1)*100/15)-round(i*100/15))*10 for i in range(301)]
    gif=folder/'synchronized_fire_guard.gif'
    frames[0].save(gif,save_all=True,append_images=frames[1:],duration=timing,loop=0,disposal=2,optimize=False)
    samples={}
    with Image.open(gif) as image:
        require(image.n_frames==301,'Unexpected animation frame count')
        for i in range(301):
            image.seek(i)
            rgb=image.convert('RGB')
            require(rgb.tobytes()==frames[i].convert('RGB').tobytes()
                    and image.info['duration']==timing[i],f'Animation frame mismatch: {i}')
            if i in (0,150,300) or i==next((r['tick'] for r in trace if r['filtered']),-1):
                rgb.save(folder/f'tick_{i:03d}.png')
                samples[str(i)]=hashlib.sha256(rgb.tobytes()).hexdigest()
    for image in frames:
        image.close()
    return {'path':gif.relative_to(output).as_posix(),'sha256':sha(gif),'frames':301,
            'duration_ms':sum(timing),'samples':samples,'removed_misses':a['misses'],
            'hits_each':a['hits'],'decoded_pixels_and_timing_exact':True}
