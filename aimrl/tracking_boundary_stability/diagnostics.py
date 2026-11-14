"""Mechanism evidence, exact-backend gates and common-state diagnostics."""
import copy
import csv
import json
import math

import numpy as np

from ..tracking25.env import TrackingEnv, advance_target
from ..tracking25.policies import rule_policy
from ..tracking_research.io import save_csv, save_json
from .core import Agent, CONDITIONS, Encoder, boundary, reference_run, require, reward_parts


def mechanisms(c, output):
    deterministic = copy.deepcopy(c)
    deterministic['environment'].update(turn_probability=0, speed_toggle_probability=0)
    evidence = []
    for state in ((10,10,11,10,0,2), (23,10,24,10,0,2)):
        env = TrackingEnv(deterministic); env.reset(17); env.state = state
        nxt, base, _, _, info = env.step(4)
        evidence.append(dict(state=list(state), relative=[state[2]-state[0],state[3]-state[1],state[4],state[5]],
                             next_state=list(nxt), next_relative=[nxt[2]-nxt[0],nxt[3]-nxt[1],nxt[4],nxt[5]],
                             R_row=Encoder('R').index(state), B_row=Encoder('B').index(state)))
    require(evidence[0]['R_row'] == evidence[1]['R_row'] and evidence[0]['B_row'] != evidence[1]['B_row']
            and evidence[0]['next_relative'] != evidence[1]['next_relative'], 'Boundary alias counterexample failed')
    agent = Agent(c, 'R-Fixed')
    agent.q[:] = np.random.default_rng(9876).normal(size=agent.q.shape).astype(np.float32)
    pairs = []
    rng = np.random.default_rng(876)
    for i in range(128):
        env0, env1 = TrackingEnv(c), TrackingEnv(c)
        env0.reset(7700000+i); env1.reset(7700000+i)
        s = tuple(map(int, (*rng.integers(25,size=4), rng.integers(4), rng.integers(1,3))))
        if s[:2] == s[2:4]: continue
        env0.state = env1.state = s
        n0,b0,_,_,i0 = env0.step(4); n1,b1,_,_,i1 = env1.step(5)
        p0=reward_parts(b0,i0['pre_distance'],i0['distance'],c)
        p1=reward_parts(b1,i1['pre_distance'],i1['distance'],c)
        t0=p0[-1]+c['learning']['gamma']*float(agent.q[agent.index(n0)].max())
        t1=p1[-1]+c['learning']['gamma']*float(agent.q[agent.index(n1)].max())
        require(n0==n1 and env0.rng.bit_generator.state==env1.rng.bit_generator.state,
                'Unaligned Fire and Stay have different dynamics')
        require(p0[1:3]==p1[1:3] and math.isclose(p0[-1]-p1[-1],1,abs_tol=1e-13)
                and math.isclose(t0-t1,1,abs_tol=1e-13), 'Reward / frozen TD gap is not one')
        pairs.append(dict(pair=i,state=json.dumps(s),next_state=json.dumps(n0),stay_base=b0,fire_base=b1,
                          distance_term=0,close_term=p0[1],potential=p0[2],stay_reward=p0[-1],fire_reward=p1[-1],
                          stay_td_target=t0,fire_td_target=t1,reward_gap=p0[-1]-p1[-1],td_gap=t0-t1))
    save_csv(output/'diagnostics/fire_stay_pairs.csv',pairs)
    save_json(output/'diagnostics/mechanisms.json',dict(boundary_alias=evidence,paired_states=len(pairs),
              frozen_table_q_hash=agent.snapshot()[0],physical_rng_equal=True,reward_and_td_gap=1,
              interpretation='Immediate reward and frozen one-step target identities do not imply convergence of learned action values.'))


def parity(c, kernel, output):
    records, all_audit = [], []
    for condition in CONDITIONS:
        for case, offset, forced_mode in (('exploration',0,False),('late_epsilon',3999800,False),('forced',4999700,True)):
            pc=copy.deepcopy(c)
            if forced_mode: pc['environment'].update(turn_probability=.35,speed_toggle_probability=.4)
            streams=np.random.SeedSequence(88001).spawn(2)
            a=Agent(c,condition,np.random.default_rng(streams[1]))
            b=Agent(c,condition,np.random.default_rng(streams[1]))
            # Exercise late, frequently updated pairs, independently of the selected epsilon.
            a.counts[:,4:6]=b.counts[:,4:6]=1234
            e0,e1=TrackingEnv(pc),TrackingEnv(pc)
            seed=int(streams[0].generate_state(1,dtype=np.uint64)[0])
            e0.reset(seed);e1.reset(seed)
            step=0;events=np.zeros(7,dtype=np.int64);seen=set()
            for window,length in enumerate((300,300,300,101)):
                if forced_mode:
                    e0.state=e1.state=((24,10,24,10,0,2) if window%2==0 else (0,0,0,0,2,1))
                initial=e0.state
                forced=np.resize(np.array([5,4,0,1,2,3,6,7,8,9],np.int16),length) if forced_mode else None
                native,nd=kernel.run(e0,initial,length,a,offset+step,train=True,forced=forced,debug=True)
                python,pd=reference_run(e1,initial,length,b,offset+step,train=True,forced=forced,c=c)
                require(np.array_equal(native,python),f'Trajectory mismatch {condition}/{case}')
                require(np.allclose(nd,pd,rtol=0,atol=2e-13),f'Per-step update mismatch {condition}/{case}: {np.max(abs(nd-pd))}')
                require(a.snapshot()==b.snapshot(),f'Q/count/action RNG parity mismatch {condition}/{case}')
                require(e0.rng.bit_generator.state==e1.rng.bit_generator.state,'Target RNG parity failed')
                for i in range(length):
                    all_audit.append(dict(condition=condition,case=case,step=offset+step+i,
                          state=json.dumps(list(initial) if i==0 else native[i-1,:6].tolist()),
                          next_state=json.dumps(native[i,:6].tolist()),action=int(native[i,6]),
                          **{k:float(v) for k,v in zip(('base','close','potential','reward','alpha','td_target','q_before','q_after','count_before','row'),nd[i])},
                          python_native_exact_q=nd[i,7]==pd[i,7]))
                events+=np.array([native[:,13].sum(),native[:,14].sum(),native[:,15].sum(),
                                 (native[:,6]==5).sum(),(native[:,6]==4).sum(),
                                 (native[:,6]>=6).sum(),(native[:,6]<=3).sum()])
                seen.update(map(int,native[:,6]));step+=length
                if length==300: require(e0.reset()==e1.reset(),'Reset parity failed')
            records.append(dict(condition=condition,case=case,updates=step,actions=sorted(seen),
                                events=events.tolist(),q_counts_rng_exact=True,tail_ticks=101,
                                max_debug_tolerance=2e-13,window_truncation_bootstraps=True))
            del a,b
    require(all(set(range(10))==set(r['actions']) for r in records),'Missing action coverage')
    save_csv(output/'diagnostics/python_native_steps.csv',all_audit)
    save_json(output/'diagnostics/parity.json',dict(complete=True,runs=records,total_real_updates=12012))
    return records


def encoder_checks(output):
    records=[]
    for kind in ('R','B'):
        enc=Encoder(kind)
        keys={}
        for c in range(25):
            for t in range(25):
                k=(t-c,) if kind=='R' else (t-c,int(boundary(c)),int(boundary(t)))
                idx=int(enc.axis[c,t]);require(keys.setdefault(k,idx)==idx,'Key encoding differs')
        require(len(set(keys.values()))==len(keys)==enc.width,'Encoding is not one-to-one')
        # Exhaustively verify all 3,125,000 physical states by vectorized batches.
        xy=np.indices((25,25,25),dtype=np.int32).reshape(3,-1).T
        checked=0
        for xc in range(25):
            for h in range(4):
                for speed in (1,2):
                    states=np.column_stack((np.full(len(xy),xc),xy,np.full(len(xy),h),np.full(len(xy),speed)))
                    ids=enc.indices(states)
                    require(ids.min()>=0 and ids.max()<enc.rows,'Index out of bounds')
                    checked+=len(states)
        records.append(dict(**enc.record(),physical_states_checked=checked,one_to_one_keys=True))
    save_json(output/'diagnostics/encoding.json',records)
    return records


def common_states(c, output):
    """A frozen, policy-independent-of-learned-models visitation sample."""
    events=[];spec=c['study']['diagnostic']
    for seed in range(spec['start'],spec['start']+spec['count']):
        for policy in ('rule','random'):
            env=TrackingEnv(c);s,_=env.reset(seed)
            rng=np.random.default_rng(seed+100000)
            for i in range(300):
                events.append(s)
                action=rule_policy(s,25) if policy=='rule' else int(rng.integers(10))
                s,*_=env.step(action)
    states,frequency=np.unique(np.array(events,dtype=np.int16),axis=0,return_counts=True)
    path=output/'diagnostics/common_states.npz'
    np.savez_compressed(path,states=states,frequency=frequency)
    save_json(output/'diagnostics/common_states.json',dict(distinct=len(states),events=len(events),
              seeds=list(range(spec['start'],spec['start']+spec['count'])),
              construction='Pre-action physical states from Rule and independent uniform-random rollouts, 300 ticks each. Frozen before pilot and Final access.',
              weighting='state: one per distinct physical state; event: frequency in these fixed rollouts.'))


def expected_stay(c, agent, states):
    """Exact six-outcome conditional expected target; Fire is exactly target-1 off alignment."""
    states=np.asarray(states,dtype=np.int32)
    dpre=abs(states[:,0]-states[:,2])+abs(states[:,1]-states[:,3])
    target=np.zeros(len(states),dtype=np.float64)
    tp=c['environment']['turn_probability'];sp=c['environment']['speed_toggle_probability']
    hx=np.array([1,0,-1,0]);hy=np.array([0,1,0,-1])
    for turn,pt in ((0,1-tp),(1,tp/2),(-1,tp/2)):
        for toggle,ps in ((0,1-sp),(1,sp)):
            s=states.copy();s[:,4]=(s[:,4]+turn)%4
            if toggle:s[:,5]=3-s[:,5]
            for sub in range(2):
                active=s[:,5]>sub
                x=s[:,2]+hx[s[:,4]];y=s[:,3]+hy[s[:,4]]
                bounce=active&((x<0)|(x>=25)|(y<0)|(y>=25))
                s[bounce,4]=(s[bounce,4]+2)%4
                s[active,2]+=hx[s[active,4]];s[active,3]+=hy[s[active,4]]
            d=abs(s[:,0]-s[:,2])+abs(s[:,1]-s[:,3])
            r=reward_parts(c['rewards']['step'],dpre,d,c)[-1]
            target+=pt*ps*(r+c['learning']['gamma']*agent.q[agent.encoder.indices(s)].max(axis=1).astype(np.float64))
    return target


def diagnose(c, agent, states, frequency, step, previous=None, distinct_path=None):
    rows=agent.encoder.indices(states);q=agent.q[rows];counts=agent.counts[rows]
    d=abs(states[:,0]-states[:,2])+abs(states[:,1]-states[:,3])
    unaligned=d>0;target=expected_stay(c,agent,states)
    gap=q[:,5].astype(float)-q[:,4].astype(float)
    residual_stay=target-q[:,4];residual_fire=target-1-q[:,5]
    boundary=np.any(np.minimum(states[:,:4],24-states[:,:4])<=3,axis=1)
    greedy=q.argmax(axis=1);frequent=frequency>=c['study']['frequent_state_min_events']
    groups={'all':np.ones(len(states),bool),'boundary':boundary,'interior':~boundary,
            'speed_1':states[:,5]==1,'speed_2':states[:,5]==2,
            **{f'd_pre_{k}':d==k for k in range(4)},'d_pre_4plus':d>=4,'frequent':frequent}
    result=[]
    for group,mask in groups.items():
        m=mask&unaligned
        if not m.any():continue
        for weighting,w in (('state',np.ones(len(states))),('event',frequency)):
            def avg(x):return float(np.average(x[m],weights=w[m]))
            result.append(dict(condition=agent.condition,step=step,group=group,weighting=weighting,
                         distinct_states=int(m.sum()),events=int(frequency[m].sum()),
                         q_gap_mean=avg(gap),q_gap_p05=float(np.quantile(gap[m],.05)),q_gap_median=float(np.median(gap[m])),
                         q_gap_p95=float(np.quantile(gap[m],.95)),quantiles_weighting='distinct states',
                         wrong_ranking_fraction=avg(gap>0),greedy_unaligned_fire_fraction=avg(greedy==5),
                         mean_fire_updates=avg(counts[:,5]),mean_stay_updates=avg(counts[:,4]),
                         fire_unvisited_fraction=avg(counts[:,5]==0),stay_unvisited_fraction=avg(counts[:,4]==0),
                         stay_bellman_mae=avg(abs(residual_stay)),fire_bellman_mae=avg(abs(residual_fire)),
                         stay_bellman_signed=avg(residual_stay),fire_bellman_signed=avg(residual_fire),
                         q_change_mae=avg(np.mean(abs(q.astype(float)-previous),axis=1)) if previous is not None else None,
                         greedy_change_fraction=avg(greedy!=previous.argmax(axis=1)) if previous is not None else None))
    if distinct_path:
        save_csv(distinct_path,[dict(zip(('xc','yc','xt','yt','heading','speed'),map(int,s)),frequency=int(frequency[i]),
                    row=int(rows[i]),d_pre=int(d[i]),q_gap=float(gap[i]),q_fire=float(q[i,5]),q_stay=float(q[i,4]),
                    fire_updates=int(counts[i,5]),stay_updates=int(counts[i,4]),greedy=int(greedy[i]),
                    stay_bellman_residual=float(residual_stay[i]),fire_bellman_residual=float(residual_fire[i]))
                    for i,s in enumerate(states) if unaligned[i]])
    return result,q.copy()
