"""Mechanism, Bellman, shaping, backend parity and no-hidden-future gates."""
import copy
import json
import time

import numpy as np

from ..tracking25.env import TrackingEnv
from ..tracking_research.io import save_csv,save_json
from .core import OUT,Native,Policy,config,frozen_agent,methods,state_id,target_state,transition_arrays
from .reference import ReferencePlanner,phi,require,target_outcomes,transitions


class FixedDraws:
    def __init__(self,draws):self.draws=np.array(draws,np.float64)
    def random(self,n):
        require(n==3,'Unexpected environment random draw shape')
        return self.draws.copy()


def brute(c,state,h,terminal,agent=None,shaped=False):
    if h==0:
        value=0.0 if terminal=='Zero' else float(phi(state))
        if terminal=='Q+Phi':value+=float(agent.q[agent.index(state)].max())
        return value-phi(state) if shaped else value
    return max(sum(p*(reward+(c['gamma']*phi(nxt)-phi(state) if shaped else 0)+c['gamma']*brute(c,nxt,h-1,terminal,agent,shaped))
                   for p,nxt,base,reward in transitions(c,state,a)) for a in range(10))


def mechanism_checks(c):
    sums=[];duplicates=0
    for t in range(5000):
        outcomes=target_outcomes(target_state(t),.05,.05)
        sums.append(sum(p for p,_ in outcomes));duplicates+=6-len(outcomes)
    require(max(abs(p-1) for p in sums)<1e-14 and duplicates>0,'Probability / duplicate merging failed')
    states=[(10,10,11,10,0,2),(23,10,24,10,0,2),(0,0,0,0,2,1),(24,24,24,24,1,2),(10,10,10,10,0,2)]
    rng=np.random.default_rng(481)
    states += [tuple(map(int,(*rng.integers(25,size=4),rng.integers(4),rng.integers(1,3)))) for _ in range(20)]
    comparisons=0
    for s in states:
        for a in range(10):
            enum=transitions(c,s,a)
            probability_by_state={n:p for p,n,base,reward in enum}
            empirical={}
            for turn,pt,ut,usign in ((0,.95,.8,.2),(1,.025,.01,.2),(-1,.025,.01,.8)):
                for toggle,ps,uspeed in ((0,.95,.8),(1,.05,.01)):
                    env=TrackingEnv(c);env.state=s;env.rng=FixedDraws((ut,usign,uspeed))
                    nxt,base,done,trunc,info=env.step(a)
                    require(not done and not trunc,'Unexpected true termination / truncation')
                    match=next(r for r in enum if r[1]==nxt)
                    require(base==match[2] and match[3]==base+.5*(info['distance']<=1),'Pre-hit/post-close reward order differs')
                    empirical[nxt]=empirical.get(nxt,0)+pt*ps
                    comparisons+=1
            require(empirical==probability_by_state,'Enumerated probabilities differ from original mechanics')
    deterministic=copy.deepcopy(c);deterministic['planning'].update(turn_probability=0,speed_toggle_probability=0)
    deterministic['environment'].update(turn_probability=0,speed_toggle_probability=0)
    env=TrackingEnv(deterministic);env.reset(121)
    for i in range(300):
        a=i%10;s=env.state;expected=transitions(deterministic,s,a)
        nxt,base,_,_,info=env.step(a)
        require(len(expected)==1 and expected[0][0]==1 and nxt==expected[0][1] and base==expected[0][2], 'Deterministic step mismatch')
    return dict(target_states=5000,probability_sum_max_error=max(abs(p-1) for p in sums),merged_duplicate_branches=duplicates,
                original_environment_steps=comparisons+300,deterministic_steps=300)


def validate(c,output=None):
    folder=OUT/'validation' if output is None else output;folder.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter();mechanism=mechanism_checks(c)
    sample=[(10,10,11,10,0,2),(23,10,24,10,0,2),(0,0,0,0,2,1),(12,12,12,12,0,2),
            (12,10,10,10,0,2),(24,23,24,24,1,1),(10,10,13,11,3,2),(0,24,24,0,0,1)]
    records=[];max_error=0;shaping_error=0
    for spec in methods(c):
        if spec['kind']!='plan':continue
        policy=Policy(c,spec,load_expected=True)
        agent=frozen_agent('R-Decay',spec['seed']) if spec['terminal']=='Q+Phi' else None
        reference=ReferencePlanner(c,spec['terminal'],agent)
        shaped=ReferencePlanner(c,spec['terminal'],agent,shaped=True)
        before=agent.snapshot() if agent is not None else None
        for s in sample:
            v,a,q=reference.solve(s,spec['horizon']);native=policy.q_values(s)
            error=float(np.max(abs(native-np.array(q))));max_error=max(max_error,error)
            require(error<1e-9 and policy.propose(s)==a,'Native/reference action or Q-value mismatch '+spec['id'])
            sv,sa,sq=shaped.solve(s,spec['horizon'])
            se=float(np.max(abs(np.array(q)-np.array(sq)-phi(s))));shaping_error=max(shaping_error,se)
            require(se<1e-9 and sa==a,'Task/shaped terminal conversion equivalence failed')
            if phi(s)!=0:require(abs(native[4]-native[5]-1)<1e-10 and a!=5,'Planner proposed strictly dominated Fire')
            records.append(dict(method=spec['id'],state=json.dumps(s),horizon=spec['horizon'],action=a,
                                max_native_error=error,task_minus_shaped=float(v-sv),expected_constant=phi(s),shaping_error=se))
        require(before is None or agent.snapshot()==before,'Planning changed a frozen model')
        save_json(folder/f'{spec["id"]}_reference_cache.json',reference.statistics())
        del reference,shaped,policy,agent
    # Independent exhaustive stochastic tree, no memoization, not only a second
    # invocation of the same shared-subproblem implementation.
    brute_records=[]
    for terminal in ('Phi','Zero'):
        for h in (1,2,3):
            s=(23,10,24,10,0,2);ref=ReferencePlanner(c,terminal)
            direct=brute(c,s,h,terminal)
            require(abs(ref.solve(s,h)[0]-direct)<1e-10,'Feedback exhaustive-tree value mismatch')
            brute_records.append(dict(terminal=terminal,horizon=h,value=direct))
    # Full-action enumeration and the optional dominance prune must agree.
    for s in sample:
        full=ReferencePlanner(c);pruned=ReferencePlanner(c,prune_fire=True)
        a=full.solve(s,3);b=pruned.solve(s,3)
        require(abs(a[0]-b[0])<1e-10 and a[1]==b[1],'Dominance-pruning equivalence failed')
    # Same observed state under unrelated hidden target streams yields identical actions.
    native=Native(c);rollout_steps=0
    for spec in methods(c):
        policy=Policy(c,spec)
        for probability in (.05,.10):
            cc=copy.deepcopy(c);cc['environment'].update(turn_probability=probability,speed_toggle_probability=probability)
            e0,e1=TrackingEnv(cc),TrackingEnv(cc);initial,_=e0.reset(6100000);e1.reset(6100000)
            prior_rng=copy.deepcopy(e0.rng.bit_generator.state)
            for hidden_seed in (16,724,981):
                unrelated=TrackingEnv(cc);unrelated.reset(hidden_seed);unrelated.state=initial
                rng=copy.deepcopy(unrelated.rng.bit_generator.state)
                require(policy.act(initial)==policy.act(unrelated.state),'Hidden RNG changed a planning action')
                require(unrelated.rng.bit_generator.state==rng,'Planning call consumed environment RNG')
            require(e0.rng.bit_generator.state==prior_rng,'Planning changed true rollout RNG')
            before=policy.agent.snapshot() if policy.agent is not None else None
            out,timing=native.run(e0,initial,policy,300)
            for i,row in enumerate(out):
                state=e1.state;proposed=policy.propose(state);a=policy.act(state)
                nxt,base,done,trunc,info=e1.step(a)
                expected=(*nxt,a,info['pre_distance'],abs(nxt[0]-state[2])+abs(nxt[1]-state[3]),info['distance'],
                          info['target_net_displacement'],info['hit'],info['miss'],info['reflections'],info['random_turn'],info['speed_toggle'],proposed,state[5])
                require(tuple(row)==expected,'Native evaluation / original environment step mismatch '+spec['id'])
                require(not done and trunc==(i==299),'Continuing window truncation differs')
            require(e0.rng.bit_generator.state==e1.rng.bit_generator.state,'Native rollout target RNG differs')
            require(before is None or policy.agent.snapshot()==before,'Evaluation changed a frozen learner')
            rollout_steps+=600
    save_csv(folder/'reference_native_shaping.csv',records)
    save_json(folder/'checks.json',dict(complete=True,mechanisms=mechanism,max_native_value_error=max_error,
              max_shaping_constant_error=shaping_error,brute_force=brute_records,original_python_native_environment_steps=rollout_steps,
              no_hidden_future=True,all_twenty_methods=True,training_interactions=0,elapsed_seconds=time.perf_counter()-start))
    print(f'Validated all planners and 20 rollout methods in {time.perf_counter()-start:.2f}s',flush=True)
    return readback(folder/'checks.json')


def readback(path):
    from ..tracking_research.io import read_json
    return read_json(path)
