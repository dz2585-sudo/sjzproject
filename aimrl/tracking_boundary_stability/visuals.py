"""Static research figures and complete, pixel-verified synchronized rollouts."""
import hashlib

import numpy as np

from ..tracking_research.io import read_csv, save_csv, save_json, sha
from ..tracking_tradeoff.core import target_hash
from .core import CONDITIONS, OUT, load_model, require

COLORS=dict(zip(CONDITIONS,('#476b97','#43a086','#d59342','#9a69ac')))


def plots(c,comparison):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    folder=OUT/'figures';folder.mkdir(exist_ok=True)
    seeds=c['study']['training_seeds']
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for condition in CONDITIONS:
        curves=[read_csv(OUT/'runs'/condition/f'seed_{s}/development.csv') for s in seeds]
        x=[int(r['step'])/1e6 for r in curves[0]]
        for ax,k in zip(axes,('D','C','H')):
            y=np.array([[float(r[k]) for r in curve] for curve in curves]);mean=y.mean(axis=0);sd=y.std(axis=0,ddof=1)
            ax.plot(x,mean,label=condition,color=COLORS[condition]);ax.fill_between(x,mean-sd,mean+sd,color=COLORS[condition],alpha=.12)
            ax.set(xlabel='Training interactions (millions)',ylabel=k);ax.grid(alpha=.2)
    axes[1].yaxis.set_major_formatter(PercentFormatter(1));axes[0].legend(frameon=False)
    fig.suptitle('Development, guarded policies | mean and sample SD across five training seeds')
    fig.savefig(folder/'learning_curves.png',dpi=170);plt.close(fig)

    from ..tracking_research.io import read_json
    per_seed=read_json(OUT/'per_seed.json')
    fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for row,split in enumerate(('final','stress')):
        for ax,k in zip(axes[row],('D','C','H')):
            for seed in seeds:
                values=[next(r[k] for r in per_seed if r['split']==split and r['variant']=='guard' and r['condition']==cond and r['seed']==seed) for cond in CONDITIONS]
                ax.plot(range(4),values,marker='o',label=f'Seed {seed}',alpha=.8)
            ax.set_xticks(range(4),CONDITIONS,rotation=15);ax.set_ylabel(f'{split.title()} {k}');ax.grid(alpha=.2)
            if k=='C':ax.yaxis.set_major_formatter(PercentFormatter(1))
    axes[0,0].legend(fontsize=8,frameon=False)
    fig.suptitle('Paired training seeds | each line is one independently trained seed')
    fig.savefig(folder/'paired_seeds.png',dpi=170);plt.close(fig)

    fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    for row,split in enumerate(('final','stress')):
        rows=[r for r in comparison if r['split']==split and r['variant']=='guard']
        base=next(r for r in rows if r['condition']=='R-Fixed')
        for ax,k in zip(axes[row],('D','C')):
            for r in rows:
                ax.errorbar(r[k],r['H'],xerr=r[k+'_sd'],yerr=r['H_sd'],fmt='o',color=COLORS[r['condition']],capsize=3,label=r['condition'])
            ax.axhline(.98*base['H'],ls=':',color='#666666',lw=.9,label='98% hit retention')
            ax.axhline(1.02*base['H'],ls='--',color='#999999',lw=.8,label='+2% hits')
            ax.axvline(base[k]+(.05 if k=='C' else 0),ls='--',color='#777777',lw=.8)
            ax.set(xlabel=k,ylabel=f'{split.title()} H');ax.grid(alpha=.18)
            if k=='C':ax.xaxis.set_major_formatter(PercentFormatter(1))
    axes[0,0].legend(fontsize=8,frameon=False)
    fig.suptitle('Distance / close-time / hit tradeoffs | error bars are training-seed sample SD')
    fig.savefig(folder/'D_C_H_tradeoff.png',dpi=170);plt.close(fig)

    events=read_csv(OUT/'event_curves.csv')
    fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for row,split in enumerate(('final','stress')):
        for ax,kind in zip(axes[row],('bounce','turn','toggle')):
            for condition in CONDITIONS:
                x=list(range(-5,16));ys=[]
                for off in x:
                    vals=[float(r['distance_mean']) for r in events if r['split']==split and r['condition']==condition and r['kind']==kind and int(r['offset'])==off]
                    ys.append(float(np.mean(vals)) if vals else np.nan)
                ax.plot(x,ys,label=condition,color=COLORS[condition])
            ax.axvline(0,color='#777777',lw=.7);ax.set(xlabel=f'Ticks from {kind}',ylabel=f'{split.title()} distance');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8,frameon=False)
    fig.suptitle('Policy-specific trajectories | descriptive event-conditioned means, equal training-seed weight')
    fig.savefig(folder/'boundary_dynamic_events.png',dpi=170);plt.close(fig)

    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for condition in CONDITIONS:
        traces=[read_csv(OUT/'runs'/condition/f'seed_{s}/q_diagnostics.csv') for s in seeds]
        for ax,key in zip(axes,('wrong_ranking_fraction','fire_bellman_mae','q_change_mae')):
            curves=[]
            for trace in traces:
                rs=[r for r in trace if r['group']=='all' and r['weighting']=='state' and (r[key] or '')!='']
                curves.append(([int(r['step'])/1e6 for r in rs],[float(r[key]) for r in rs]))
            ax.plot(curves[0][0],np.mean([v[1] for v in curves],axis=0),label=condition,color=COLORS[condition])
            ax.set(xlabel='Interactions (millions)',ylabel=key.replace('_',' '));ax.grid(alpha=.2)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1));axes[0].legend(frameon=False,fontsize=8)
    fig.suptitle('Frozen common physical states | state-equal diagnostics, mean across five seeds')
    fig.savefig(folder/'q_stability.png',dpi=170);plt.close(fig)


def demos(c,kernel):
    from PIL import Image,ImageDraw,ImageFont
    from ..tracking25.env import ACTION_NAMES,TrackingEnv
    seed=c['study']['display_training_seed'];folder=OUT/'demos';folder.mkdir(exist_ok=True)
    agents={cond:load_model(OUT/'runs'/cond/f'seed_{seed}/checkpoints/step_5000000.npz',c)[0] for cond in CONDITIONS}
    palette=[255,255,255,31,43,61,218,226,234,32,110,174,197,64,59,232,156,32,
             95,109,123,230,240,249,229,241,234,243,237,226,239,232,245]+[0]*(768-33)
    large,medium,small=(ImageFont.load_default(size=s) for s in (21,18,14))
    checks=[]
    for scenario in c['study']['demo_seeds']:
        trajectories={};target=None;records=[]
        for cond,agent in agents.items():
            before=agent.snapshot();env=TrackingEnv(c);initial,_=env.reset(scenario)
            ticks=kernel.run(env,initial,300,agent,guard=True)
            digest=target_hash(initial,ticks)
            require(target is None or digest==target,'Demo target routes differ');target=digest
            require(agent.snapshot()==before,'Demo changed model')
            states=np.vstack([initial,ticks[:,:6]])
            hits=np.r_[0,np.cumsum(ticks[:,11])];shots=np.r_[0,np.cumsum(ticks[:,6]==5)]
            trajectories[cond]=(states,ticks,hits,shots)
            for i in range(301):
                records.append(dict(condition=cond,tick=i,**dict(zip(('xc','yc','xt','yt','heading','speed'),map(int,states[i]))),
                                    action='reset' if i==0 else ACTION_NAMES[int(ticks[i-1,6])],
                                    d_post=int(abs(states[i,0]-states[i,2])+abs(states[i,1]-states[i,3])),
                                    cumulative_hits=int(hits[i]),cumulative_shots=int(shots[i])))
        save_csv(folder/f'scenario_{scenario}_ticks.csv',records)
        def frame(i):
            im=Image.new('P',(1140,448),0);im.putpalette(palette);d=ImageDraw.Draw(im);d.fontmode='1'
            d.rectangle((0,0,1139,69),fill=1)
            d.text((18,9),f'AimRL | Scenario {scenario} | Tick {i:03d} / 300 | Training seed {seed}',font=large,fill=0)
            d.text((18,39),'Frozen 5M models + firing guard | same target route | blue crosshair, red target, gold hit',font=small,fill=0)
            for panel,cond in enumerate(CONDITIONS):
                states,ticks,hits,shots=trajectories[cond];s=states[i];x=12+panel*284
                d.rectangle((x,78,x+273,109),fill=7+panel)
                d.text((x+8,83),cond,font=medium,fill=1)
                left,top,cell=x+11,120,10
                def point(px,py):return left+int(px)*cell+5,top+int(py)*cell+5
                for k in range(26):
                    d.line((left+k*cell,top,left+k*cell,top+250),fill=2)
                    d.line((left,top+k*cell,left+250,top+k*cell),fill=2)
                tail=states[max(0,i-7):i+1]
                if len(tail)>1:
                    d.line([point(t[2],t[3]) for t in tail],fill=6)
                xc,yc=point(s[0],s[1]);xt,yt=point(s[2],s[3])
                d.ellipse((xt-4,yt-4,xt+4,yt+4),outline=4,width=2)
                d.line((xc-4,yc,xc+4,yc),fill=3,width=2);d.line((xc,yc-4,xc,yc+4),fill=3,width=2)
                if i and ticks[i-1,11]:d.ellipse((xc-3,yc-3,xc+3,yc+3),fill=5)
                action='reset' if not i else ACTION_NAMES[int(ticks[i-1,6])]
                dist=abs(int(s[0])-int(s[2]))+abs(int(s[1])-int(s[3]))
                d.text((x+8,380),f'Action: {action} | D: {dist}',font=small,fill=1)
                d.text((x+8,402),f'Hits: {int(hits[i])} | Shots: {int(shots[i])}',font=small,fill=1)
            d.text((18,430),'Complete rollout: initial frame + all 300 environment ticks; 15 frames per second.',font=small,fill=6)
            return im
        frames=[frame(i) for i in range(301)]
        durations=[(round((i+1)*100/15)-round(i*100/15))*10 for i in range(301)]
        gif=folder/f'scenario_{scenario}.gif'
        frames[0].save(gif,save_all=True,append_images=frames[1:],duration=durations,loop=0,disposal=2,optimize=False)
        sample_hashes={}
        with Image.open(gif) as decoded:
            require(decoded.n_frames==301,'GIF frame count differs')
            for i in range(301):
                decoded.seek(i);rgb=decoded.convert('RGB')
                require(rgb.tobytes()==frames[i].convert('RGB').tobytes() and decoded.info['duration']==durations[i],f'GIF decoded frame mismatch {i}')
                if i in (0,150,300):
                    rgb.save(folder/f'scenario_{scenario}_tick_{i:03d}.png')
                    sample_hashes[str(i)]=hashlib.sha256(rgb.tobytes()).hexdigest()
        for im in frames:im.close()
        checks.append(dict(scenario=scenario,path=gif.relative_to(OUT).as_posix(),sha256=sha(gif),frames=301,
                           duration_ms=sum(durations),decoded_all_pixels_exact=True,decoded_timing_exact=True,
                           target_sha256=target,samples=sample_hashes,actual_player_check='See player_check.json; decoding alone is not a player check.'))
    save_json(folder/'decode_checks.json',checks)
    return checks
