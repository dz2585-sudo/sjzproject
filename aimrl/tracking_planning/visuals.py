"""Research plots and complete decoded, timed 301-frame synchronized demos."""
import hashlib

import numpy as np

from ..tracking25.env import ACTION_NAMES,TrackingEnv
from ..tracking_research.io import read_csv,read_json,save_csv,save_json,sha
from .core import OUT,Native,Policy,methods
from .evaluation import canonical_metrics,summary
from .reference import require

ORDER=('Rule','R-Fixed','R-Decay','Plan-H1','Plan-H2','Plan-H3','Plan-H3-Zero','Hybrid-H3')
COLORS=dict(zip(ORDER,('#8e979d','#5076a3','#409a83','#c89948','#db774b','#ba596a','#817a92','#8664ac')))


def render_demo(c,scenario,folder,development=False):
    from PIL import Image,ImageDraw,ImageFont
    folder.mkdir(parents=True,exist_ok=False)
    seed=c['planning']['display_training_seed'];ids=('Rule',f'R-Decay_s{seed}','Plan-H3',f'Hybrid-H3_s{seed}')
    native=Native(c);specs={m['id']:m for m in methods(c)};traces={};route=None;records=[];metrics=[];times=[]
    for key in ids:
        p=Policy(c,specs[key]);env=TrackingEnv(c);initial,_=env.reset(scenario)
        ticks,ns=native.run(env,initial,p)
        m=canonical_metrics(initial,ticks)
        require(route is None or route==m['target_sha256'],'Demo target route differs');route=m['target_sha256']
        metrics.append(dict(policy=key,**summary([m],ticks[None])))
        states=np.vstack([initial,ticks[:,:6]]);hits=np.r_[0,np.cumsum(ticks[:,11])];shots=np.r_[0,np.cumsum(ticks[:,6]==5)]
        traces[key]=(states,ticks,hits,shots);times.append(ns)
        for i in range(301):records.append(dict(policy=key,tick=i,**dict(zip(('xc','yc','xt','yt','heading','speed'),map(int,states[i]))),
                     action='reset' if i==0 else ACTION_NAMES[int(ticks[i-1,6])],hits=int(hits[i]),shots=int(shots[i]),
                     d_post=int(abs(states[i,0]-states[i,2])+abs(states[i,1]-states[i,3]))))
    save_csv(folder/'ticks.csv',records);save_json(folder/'metrics.json',metrics)
    np.savez_compressed(folder/'rollouts.npz',initial=np.array([traces[k][0][0] for k in ids],np.int16),
                        ticks=np.array([traces[k][1] for k in ids],np.int16),policy_ids=np.array(ids),decision_ns=np.array(times))
    palette=[255,255,255,31,43,61,218,226,234,32,110,174,197,64,59,232,156,32,95,109,123,
             230,240,249,229,241,234,243,237,226,239,232,245]+[0]*(768-33)
    large,medium,small=(ImageFont.load_default(size=s) for s in (21,18,14))
    def frame(i):
        im=Image.new('P',(1140,470),0);im.putpalette(palette);d=ImageDraw.Draw(im);d.fontmode='1'
        d.rectangle((0,0,1139,69),fill=1)
        d.text((18,9),f'AimRL stochastic planning | Scenario {scenario} | Tick {i:03d}/300',font=large,fill=0)
        d.text((18,39),f'Same target route | Frozen R-Decay / Hybrid seed {seed} | all actions guarded | blue crosshair, red target',font=small,fill=0)
        for panel,(key,title) in enumerate(zip(ids,('Rule','R-Decay','Plan-H3 (Phi)','Hybrid-H3 (Q + Phi)'))):
            states,ticks,hits,shots=traces[key];s=states[i];x=12+panel*284
            d.rectangle((x,78,x+273,109),fill=7+panel);d.text((x+8,83),title,font=medium,fill=1)
            left,top,cell=x+11,120,10
            def point(px,py):return left+int(px)*cell+5,top+int(py)*cell+5
            for k in range(26):
                d.line((left+k*cell,top,left+k*cell,top+250),fill=2);d.line((left,top+k*cell,left+250,top+k*cell),fill=2)
            tail=states[max(0,i-7):i+1]
            if len(tail)>1:d.line([point(t[2],t[3]) for t in tail],fill=6)
            xc,yc=point(s[0],s[1]);xt,yt=point(s[2],s[3])
            d.ellipse((xt-4,yt-4,xt+4,yt+4),outline=4,width=2)
            d.line((xc-4,yc,xc+4,yc),fill=3,width=2);d.line((xc,yc-4,xc,yc+4),fill=3,width=2)
            if i and ticks[i-1,11]:d.ellipse((xc-3,yc-3,xc+3,yc+3),fill=5)
            action='reset' if i==0 else ACTION_NAMES[int(ticks[i-1,6])]
            distance=abs(int(s[0])-int(s[2]))+abs(int(s[1])-int(s[3]))
            d.text((x+8,380),f'{action} | D: {distance}',font=small,fill=1)
            d.text((x+8,402),f'Hits: {int(hits[i])} | Shots: {int(shots[i])}',font=small,fill=1)
            speed='reset' if i==0 else f'{int(ticks[i-1,17])} / {int(ticks[i-1,5])}'
            d.text((x+8,424),f'Speed before / motion: {speed}',font=small,fill=1)
        d.text((18,451),'Initial frame + every environment tick | 15 fps | planners use the same horizon at tick 300.',font=small,fill=6)
        return im
    frames=[frame(i) for i in range(301)]
    duration=[(round((i+1)*100/15)-round(i*100/15))*10 for i in range(301)]
    path=folder/'synchronized.gif';frames[0].save(path,save_all=True,append_images=frames[1:],duration=duration,loop=0,disposal=2,optimize=False)
    samples={}
    with Image.open(path) as im:
        require(im.n_frames==301,'GIF frame count differs')
        for i in range(301):
            im.seek(i);rgb=im.convert('RGB')
            require(rgb.tobytes()==frames[i].convert('RGB').tobytes() and im.info['duration']==duration[i],f'GIF decoded pixels/timing differ at frame {i}')
            if i in (0,150,300):rgb.save(folder/f'tick_{i:03d}.png');samples[str(i)]=hashlib.sha256(rgb.tobytes()).hexdigest()
    for f in frames:f.close()
    receipt=dict(scenario_seed=scenario,development=development,policies=list(ids),frames=301,duration_ms=sum(duration),
                 decoded_all_frame_pixels_and_timing=True,sha256=sha(path),target_sha256=route,samples=samples,
                 environment_ticks=1200,planning_lookups=600,new_training_interactions=0,
                 actual_player_check='Not performed, as requested; no desktop/browser permissions requested.')
    save_json(folder/'decode_checks.json',receipt)
    return receipt


def plots(c,comparison,paired):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    folder=OUT/'figures';folder.mkdir(exist_ok=True)
    def rows(split):return {r['family']:r for r in comparison if r['split']==split}
    fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for rid,split in enumerate(('final','stress')):
        rs=rows(split)
        for ax,metric in zip(axes[rid],('D','C','H')):
            ids=('Plan-H1','Plan-H2','Plan-H3')
            ax.plot((1,2,3),[rs[k][metric] for k in ids],marker='o',color=COLORS['Plan-H3'],label='Phi terminal (one controller each)')
            ax.axhline(rs['R-Decay'][metric],color=COLORS['R-Decay'],ls='--',label='R-Decay mean of five models')
            ax.set(xticks=(1,2,3),xlabel='Feedback planning horizon',ylabel=f'{split.title()} {metric}');ax.grid(alpha=.2)
            if metric=='C':ax.yaxis.set_major_formatter(PercentFormatter(1))
    axes[0,0].legend(fontsize=8,frameon=False);fig.suptitle('Horizon response within the same full-state, fixed-model planner family')
    fig.savefig(folder/'horizon_response.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    terminal_order=('Plan-H3-Zero','Plan-H3','Hybrid-H3')
    for rid,split in enumerate(('final','stress')):
        rs=rows(split)
        for ax,metric in zip(axes[rid],('D','C','H')):
            values=[rs[k][metric] for k in terminal_order];errors=[rs[k][metric+'_sd'] or 0 for k in terminal_order]
            ax.bar(range(3),values,color=[COLORS[k] for k in terminal_order],yerr=errors,capsize=3)
            ax.set_xticks(range(3),('Zero','Phi','Q + Phi'));ax.set_ylabel(f'{split.title()} {metric}');ax.grid(axis='y',alpha=.2)
            if metric=='C':ax.yaxis.set_major_formatter(PercentFormatter(1))
    fig.suptitle('Same horizon H=3 | bars show training-model sample SD only for Hybrid')
    fig.savefig(folder/'terminal_values.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for rid,split in enumerate(('final','stress')):
        rs=rows(split);base=rs['R-Decay']
        for ax,metric in zip(axes[rid],('D','C')):
            for family in ORDER:
                r=rs[family];ax.errorbar(r[metric],r['H'],xerr=r[metric+'_sd'] or 0,yerr=r['H_sd'] or 0,
                       fmt='s' if r['instances']==1 else 'o',color=COLORS[family],capsize=2,label=family)
            ax.axhline(.98*base['H'],color='#777777',ls=':',lw=.9);ax.axhline(1.02*base['H'],color='#aaaaaa',ls='--',lw=.8)
            ax.axvline(base[metric]+(.05 if metric=='C' else 0),color='#777777',ls='--',lw=.8)
            ax.set(xlabel=metric,ylabel=f'{split.title()} H');ax.grid(alpha=.2)
            if metric=='C':ax.xaxis.set_major_formatter(PercentFormatter(1))
    axes[0,0].legend(fontsize=7,frameon=False,ncol=2)
    fig.suptitle('D / C / H tradeoffs | squares: single pure controller; circles/bars: five frozen models')
    fig.savefig(folder/'D_C_H.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    per=[]
    for split in ('final','stress'):per+=read_json(OUT/'evaluation'/split/'per_policy.json')
    for rid,split in enumerate(('final','stress')):
        for ax,metric in zip(axes[rid],('D','C','H')):
            for seed in c['planning']['training_seeds']:
                rr=next(r for r in per if r['split']==split and r['family']=='R-Decay' and r['seed']==seed)
                hh=next(r for r in per if r['split']==split and r['family']=='Hybrid-H3' and r['seed']==seed)
                ax.plot((0,1),(rr[metric],hh[metric]),marker='o',label=f'Seed {seed}')
            ax.set(xticks=(0,1),xticklabels=('R-Decay','Hybrid-H3'),ylabel=f'{split.title()} {metric}');ax.grid(alpha=.2)
            if metric=='C':ax.yaxis.set_major_formatter(PercentFormatter(1))
    axes[0,0].legend(fontsize=8,frameon=False);fig.suptitle('Hybrid versus its own frozen learning model | five paired model identities')
    fig.savefig(folder/'paired_hybrid.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for rid,split in enumerate(('final','stress')):
        ev=read_csv(OUT/'evaluation'/split/'event_curves.csv')
        for ax,kind in zip(axes[rid],('bounce','turn','toggle')):
            for family in ('R-Decay','Plan-H1','Plan-H2','Plan-H3','Hybrid-H3'):
                xs=list(range(-5,16));ys=[]
                for off in xs:
                    values=[float(r['D']) for r in ev if r['family']==family and r['kind']==kind and int(r['offset'])==off]
                    ys.append(float(np.mean(values)) if values else np.nan)
                ax.plot(xs,ys,label=family,color=COLORS[family])
            ax.axvline(0,color='#888888',lw=.7);ax.set(xlabel=f'Ticks from {kind}',ylabel=f'{split.title()} D');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8,frameon=False);fig.suptitle('Descriptive policy-specific event slices | model instances averaged equally within family')
    fig.savefig(folder/'dynamic_events.png',dpi=170);plt.close(fig)
