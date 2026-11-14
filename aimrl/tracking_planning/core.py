"""Dense shared-subproblem DP and inference-only adapters for frozen policies."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from ..tracking25.env import TrackingEnv
from ..tracking25.policies import rule_policy
from ..tracking_research.agent import array_hash
from ..tracking_research.io import ROOT,read_json,save_json,sha
from ..tracking_boundary_stability.core import load_model
from .reference import require,target_outcomes

OUT=(ROOT/os.environ.get('AIMRL_PLANNING_OUTPUT','results/tracking_planning')).resolve()
CONFIG=ROOT/'configs/tracking_planning.json'
PREVIOUS=ROOT/'results/tracking_boundary_stability'
N_TARGET=5000
N_STATE=3125000
LIBRARY=OUT/'native/planning.dll' if os.name=='nt' else OUT/'native/planning.so'


def config():
    c=read_json(CONFIG);old=read_json(ROOT/'configs/tracking_boundary_stability.json')
    for k in ('environment','actions','rewards'):require(c[k]==old[k],'Frozen environment changed: '+k)
    require(c['gamma']==old['learning']['gamma']==.99 and old['learning']['shaping_scale']==1,'Shaping / discount differs')
    require(c['close_weight']==old['study']['close_weight']==.5,'Task reward differs')
    require(c['planning']['turn_probability']==c['planning']['speed_toggle_probability']==.05,'Main model must remain 0.05')
    return c


def state_id(s):
    s=tuple(s)
    require(len(s)==6 and all(isinstance(x,(int,np.integer)) and not isinstance(x,(bool,np.bool_)) for x in s)
            and all(0<=x<25 for x in s[:4]) and s[4] in range(4) and s[5] in (1,2),'Invalid physical state')
    return ((s[0]*25+s[1])*625+s[2]*25+s[3])*8+s[4]*2+s[5]-1


def target_id(t):return (t[0]*25+t[1])*8+t[2]*2+t[3]-1


def target_state(i):return (i//8//25,i//8%25,i//2%4,i%2+1)


def model_path(condition,seed):return PREVIOUS/'runs'/condition/f'seed_{seed}/checkpoints/step_5000000.npz'


def frozen_agent(condition,seed):
    a,m=load_model(model_path(condition,seed),read_json(ROOT/'configs/tracking_boundary_stability.json'))
    require(m['step']==5000000 and m['training_seed']==seed and m['condition']==condition,'Frozen model identity differs')
    a.q.flags.writeable=False;a.counts.flags.writeable=False
    return a


def methods(c):
    return ([dict(id='Rule',family='Rule',seed=None,kind='rule',horizon=0,terminal=None)]
      +[dict(id=f'{condition}_s{seed}',family=condition,seed=seed,kind='learned',horizon=0,terminal=None)
        for condition in ('R-Fixed','R-Decay') for seed in c['planning']['training_seeds']]
      +[dict(id=f'Plan-H{h}',family=f'Plan-H{h}',seed=None,kind='plan',horizon=h,terminal='Phi') for h in (1,2,3)]
      +[dict(id='Plan-H3-Zero',family='Plan-H3-Zero',seed=None,kind='plan',horizon=3,terminal='Zero')]
      +[dict(id=f'Hybrid-H3_s{seed}',family='Hybrid-H3',seed=seed,kind='plan',horizon=3,terminal='Q+Phi') for seed in c['planning']['training_seeds']])


def cache_key(c,terminal,seed=None):
    identity=dict(grid=25,actions=c['actions'],rewards=c['rewards'],gamma=c['gamma'],close_weight=c['close_weight'],
                  turn_probability=c['planning']['turn_probability'],toggle_probability=c['planning']['speed_toggle_probability'],
                  tie_tolerance=c['planning']['tie_tolerance'],terminal=terminal,
                  terminal_q_sha256=array_hash(frozen_agent('R-Decay',seed).q) if terminal=='Q+Phi' else None,
                  representation='physical_cursor625_target5000_v1',precision='float64',maximum_horizon=3)
    return hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest(),identity


def bundle_path(c,terminal,seed,horizon):
    key,_=cache_key(c,terminal,seed)
    return OUT/'cache'/key/f'h{horizon}.npz'


def transition_arrays(c):
    nxt=np.zeros((5000,6),np.int32);prob=np.zeros((5000,6),np.float64);counts=np.zeros(5000,np.uint8)
    for t in range(5000):
        outcomes=target_outcomes(target_state(t),c['planning']['turn_probability'],c['planning']['speed_toggle_probability'])
        counts[t]=len(outcomes)
        for i,(p,target) in enumerate(outcomes):nxt[t,i]=target_id(target);prob[t,i]=p
    require(np.allclose(prob.sum(axis=1),1,rtol=0,atol=1e-14),'Native model probability sum differs')
    from ..tracking25.env import clipped_position
    moves=np.array([[clipped_position(x,y,a,25)[0]*25+clipped_position(x,y,a,25)[1] for a in range(10)]
                    for x in range(25) for y in range(25)],dtype=np.int16)
    return nxt,prob,counts,moves


class Native:
    def __init__(self,c):
        self.c=c;self.lib=ctypes.CDLL(str(LIBRARY.resolve()));P=ctypes.c_void_p;I=ctypes.c_int
        self.init=self.lib.initialize;self.init.argtypes=[I,P,P];self.init.restype=None
        self.sweep=self.lib.sweep;self.sweep.argtypes=[P]*10;self.sweep.restype=None
        self.rollout=self.lib.rollout;self.rollout.argtypes=[P,P,I,I,P,P,P,P,P,P];self.rollout.restype=None
        r=c['rewards'];self.parameters=np.array([c['gamma'],c['close_weight'],r['step'],r['hit'],r['miss'],c['planning']['tie_tolerance']],np.float64)

    def initialize(self,terminal,agent=None):
        a=np.empty(N_STATE,np.float64)
        self.init(('Phi','Zero','Q+Phi').index(terminal),agent.q.ctypes.data if agent is not None else None,a.ctypes.data)
        return a

    def advance(self,previous,arrays):
        nxt,prob,counts,moves=arrays
        expected=np.empty_like(previous);value=np.empty_like(previous);policy=np.empty(N_STATE,np.uint8);stats=np.zeros(4,np.uint64)
        self.sweep(previous.ctypes.data,nxt.ctypes.data,prob.ctypes.data,counts.ctypes.data,moves.ctypes.data,
                   self.parameters.ctypes.data,expected.ctypes.data,value.ctypes.data,policy.ctypes.data,stats.ctypes.data)
        return value,policy,expected,stats

    def run(self,env,initial,policy,n=300):
        require(n>0 and env.elapsed+n<=env.cap,'Rollout crosses reset boundary')
        state=np.asarray(initial,np.int32);draws=env.rng.random((n,3))
        out=np.empty((n,18),np.int16);timing=np.empty(n,np.int64)
        e=env;parameters=np.array([e.turn_probability,e.toggle_probability],np.float64)
        mode=0 if policy.spec['kind']=='rule' else 1 if policy.spec['kind']=='learned' else 2
        data=policy.agent.q if mode==1 else policy.table if mode==2 else None
        self.rollout(state.ctypes.data,draws.ctypes.data,n,mode,data.ctypes.data if data is not None else None,
                     self.parameters.ctypes.data,parameters.ctypes.data,out.ctypes.data,timing.ctypes.data,None)
        env.state=tuple(map(int,out[-1,:6]));env.elapsed+=n;env.finished=env.elapsed>=env.cap
        return out,timing


def precompute(c):
    native=Native(c);arrays=transition_arrays(c);(OUT/'cache').mkdir(exist_ok=True)
    np.savez_compressed(OUT/'transition_model.npz',next_target=arrays[0],probability=arrays[1],branches=arrays[2],moves=arrays[3])
    families=[('Phi',None),('Zero',None)]+[('Q+Phi',s) for s in c['planning']['training_seeds']]
    records=[];total_start=time.perf_counter()
    for terminal,seed in families:
        key,identity=cache_key(c,terminal,seed);folder=OUT/'cache'/key;folder.mkdir(exist_ok=False)
        agent=frozen_agent('R-Decay',seed) if seed is not None else None
        started=time.perf_counter();value=native.initialize(terminal,agent)
        for h in (1,2,3):
            t=time.perf_counter();value,policy,expected,stats=native.advance(value,arrays)
            compute=time.perf_counter()-t
            require(np.isfinite(value).all() and np.isfinite(expected).all(),'Nonfinite Bellman values')
            meta=dict(cache_key=key,identity=identity,horizon=h,terminal=terminal,seed=seed,compute_seconds=compute,
                      value_backups=int(stats[0]),chance_transition_terms=int(stats[1]),action_value_lookups=int(stats[2]),
                      shared_expectation_cache_hits=int(stats[3]),policy_sha256=array_hash(policy),value_sha256=array_hash(value),expected_sha256=array_hash(expected))
            path=folder/f'h{h}.npz';np.savez_compressed(path,value=value,policy=policy,expected=expected,metadata=json.dumps(meta,sort_keys=True))
            records.append(dict(**meta,bundle=path.relative_to(OUT).as_posix(),bundle_sha256=sha(path),bundle_bytes=path.stat().st_size))
            del policy,expected
        print(f'Precomputed {terminal} seed {seed}: {time.perf_counter()-started:.2f}s',flush=True)
    from ..reproduce import memory_record
    result=dict(complete=True,records=records,elapsed_seconds=time.perf_counter()-total_start,memory=memory_record(),
                value_backups=sum(r['value_backups'] for r in records),chance_transition_terms=sum(r['chance_transition_terms'] for r in records),
                action_value_lookups=sum(r['action_value_lookups'] for r in records),shared_expectation_cache_hits=sum(r['shared_expectation_cache_hits'] for r in records),
                working_arrays_bytes=N_STATE*(8*3+1),transition_map_bytes=sum(a.nbytes for a in arrays),new_training_interactions=0)
    save_json(OUT/'precomputation.json',result)


class Policy:
    """Only a current observation enters act(); no seed, RNG or trajectory argument."""
    def __init__(self,c,spec,load_expected=False):
        self.spec=spec;self.c=c;self.agent=None;self.table=None;self.expected=None;self.metadata=None
        start=time.perf_counter()
        if spec['kind']=='learned':self.agent=frozen_agent(spec['family'],spec['seed'])
        elif spec['kind']=='plan':
            path=bundle_path(c,spec['terminal'],spec['seed'],spec['horizon'])
            with np.load(path,allow_pickle=False) as z:
                self.table=z['policy'];self.metadata=json.loads(str(z['metadata']))
                key,identity=cache_key(c,spec['terminal'],spec['seed'])
                require(self.metadata['cache_key']==key and self.metadata['identity']==identity and self.metadata['horizon']==spec['horizon'],'Cache identity differs')
                require(self.table.shape==(N_STATE,) and self.table.dtype==np.uint8 and int(self.table.max())<10
                        and array_hash(self.table)==self.metadata['policy_sha256'],'Invalid cached policy')
                self.table.flags.writeable=False
                if load_expected:self.expected=z['expected'];self.expected.flags.writeable=False
        self.load_seconds=time.perf_counter()-start

    def propose(self,state):
        if self.spec['kind']=='rule':return rule_policy(tuple(map(int,state)),25)
        if self.spec['kind']=='learned':return self.agent.greedy(state)
        return int(self.table[state_id(state)])

    def act(self,state):
        proposed=self.propose(state)
        return 4 if proposed==5 and (state[0]!=state[2] or state[1]!=state[3]) else proposed

    def q_values(self,state):
        require(self.expected is not None,'Expected-value table not loaded')
        state=tuple(map(int,state))
        from ..tracking25.env import clipped_position
        target=target_id(state[2:]);values=[];r=self.c['rewards']
        for a in range(10):
            x,y=clipped_position(state[0],state[1],a,25)
            base=r['step']+(r['hit'] if state[:2]==state[2:4] else r['miss']) if a==5 else r['step']
            values.append(base+float(self.expected[(x*25+y)*5000+target]))
        return np.array(values)
