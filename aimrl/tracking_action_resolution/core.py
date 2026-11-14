"""Distinct versioned A14 cache; original A10 bundles are referenced, not duplicated."""
import ctypes,hashlib,json,os,time
import numpy as np
from ..tracking_research.io import ROOT,read_json,save_json,sha
from ..tracking_research.agent import array_hash
from ..tracking_planning.core import N_STATE,state_id as historical_state_id,target_id,transition_arrays as old_arrays,frozen_agent,Native as OldNative,bundle_path as old_bundle
from ..tracking_planning.reference import require
from .env import ACTION_NAMES,clipped_position

OUT=(ROOT/os.environ.get('AIMRL_ACTION_OUTPUT','results/tracking_action_resolution')).resolve()
CONFIG=ROOT/'configs/tracking_action_resolution.json'
LIBRARY=OUT/'native'/('action_resolution.dll' if os.name=='nt' else 'action_resolution.so')

def state_id(s):
    # Widen int16 saved observations before products. Do not alter the historical helper.
    require(all(isinstance(x,(int,np.integer)) and not isinstance(x,(bool,np.bool_)) for x in s),'Non-integer state')
    return historical_state_id(tuple(map(int,s)))

def config():
    c=read_json(CONFIG);old=read_json(ROOT/'configs/tracking_planning.json')
    for k in ('environment','rewards','gamma','close_weight'):require(c[k]==old[k],'Historical setting changed '+k)
    require(tuple(c['actions'])==ACTION_NAMES,'Bad action mapping')
    for k in ('turn_probability','speed_toggle_probability','tie_tolerance','training_seeds','event_window','thresholds'):
        require(c['planning'][k]==old['planning'][k],'Historical planning setting changed '+k)
    return c

def methods(c):
    return [dict(id=f'{family}-H3-A{n}'+(f'_s{s}' if s is not None else ''),family=f'{family}-H3-A{n}',
        terminal='Phi' if s is None else 'Q+Phi',seed=s,actions=n,horizon=3)
        for family,seeds in [('Plan',[None]),('Hybrid',c['planning']['training_seeds'])] for n in (10,14) for s in seeds]

def cache_key(c,n,seed):
    identity=dict(actions=c['actions'][:n],action_count=n,horizon=3,grid=25,gamma=c['gamma'],rewards=c['rewards'],
        close_weight=c['close_weight'],turn=c['planning']['turn_probability'],toggle=c['planning']['speed_toggle_probability'],
        terminal_version=c['planning']['terminal_version'],terminal='Phi' if seed is None else 'Q+Phi',
        terminal_q_sha256=array_hash(frozen_agent('R-Decay',seed).q) if seed is not None else None,
        precision='float64',tie_tolerance=c['planning']['tie_tolerance'],state_index='cursor625_target5000_v1')
    return hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest(),identity

def bundle(c,n,seed,h):
    if n==10:return old_bundle(read_json(ROOT/'configs/tracking_planning.json'),'Phi' if seed is None else 'Q+Phi',seed,h)
    return OUT/'cache'/cache_key(c,n,seed)[0]/f'h{h}.npz'

def arrays(c,n):
    nxt,prob,count,_=old_arrays(c)
    moves=np.array([[clipped_position(x,y,a)[0]*25+clipped_position(x,y,a)[1] for a in range(n)] for x in range(25) for y in range(25)],np.int16)
    return nxt,prob,count,moves

class Native:
    def __init__(self,c):
        self.c=c;self.lib=ctypes.CDLL(str(LIBRARY));P=ctypes.c_void_p;I=ctypes.c_int
        self.init=self.lib.initialize;self.init.argtypes=[I,P,P];self.init.restype=None
        self.sweep=self.lib.sweep;self.sweep.argtypes=[I]+[P]*10;self.sweep.restype=None
        self.rollout=self.lib.rollout;self.rollout.argtypes=[P,P,I,P,P,P,P,P];self.rollout.restype=None
        r=c['rewards'];self.parameters=np.array([c['gamma'],c['close_weight'],r['step'],r['hit'],r['miss'],c['planning']['tie_tolerance']],np.float64)
    def initialize(self,seed):
        a=frozen_agent('R-Decay',seed) if seed is not None else None;v=np.empty(N_STATE,np.float64)
        self.init(2 if a else 0,a.q.ctypes.data if a else None,v.ctypes.data);return v
    def advance(self,previous,maps,n):
        exp=np.empty_like(previous);v=np.empty_like(previous);policy=np.empty(N_STATE,np.uint8);stats=np.zeros(4,np.uint64)
        self.sweep(n,previous.ctypes.data,*[a.ctypes.data for a in maps],self.parameters.ctypes.data,exp.ctypes.data,v.ctypes.data,policy.ctypes.data,stats.ctypes.data)
        return v,policy,exp,stats
    def run(self,env,initial,policy,n=300):
        require(n>0 and env.elapsed+n<=env.cap,'Rollout crosses reset')
        state=np.asarray(initial,np.int32);draws=env.rng.random((n,3));out=np.empty((n,22),np.int16);ns=np.empty(n,np.int64)
        e=np.array([env.turn_probability,env.toggle_probability],np.float64)
        self.rollout(state.ctypes.data,draws.ctypes.data,n,policy.table.ctypes.data,self.parameters.ctypes.data,e.ctypes.data,out.ctypes.data,ns.ctypes.data)
        require(int(out[:,6].max())<policy.spec['actions'],'A10 used new action')
        env.state=tuple(map(int,out[-1,:6]));env.elapsed+=n;env.finished=env.elapsed>=env.cap
        return out,ns

class Policy:
    def __init__(self,c,spec,expected=False):
        self.spec=spec;self.c=c;self.expected=None
        path=bundle(c,spec['actions'],spec['seed'],3)
        with np.load(path,allow_pickle=False) as z:
            self.table=z['policy'];meta=json.loads(str(z['metadata']))
            if spec['actions']==14:
                key,identity=cache_key(c,14,spec['seed']);require(meta['cache_key']==key and meta['identity']==identity,'Wrong A14 cache')
            else:
                from ..tracking_planning.core import cache_key as old_key
                key,identity=old_key(read_json(ROOT/'configs/tracking_planning.json'),spec['terminal'],spec['seed'])
                require(meta['cache_key']==key and meta['identity']==identity,'Wrong A10 cache')
            require(self.table.shape==(N_STATE,) and self.table.dtype==np.uint8 and int(self.table.max())<spec['actions'],'Bad policy')
            require(array_hash(self.table)==meta['policy_sha256'],'Policy digest differs')
            if expected:self.expected=z['expected']
        self.table.flags.writeable=False
    def propose(self,state):return int(self.table[state_id(state)])
    def act(self,state):
        a=self.propose(state);return 4 if a==5 and tuple(state[:2])!=tuple(state[2:4]) else a
    def q_values(self,state):
        require(self.expected is not None,'Expected array not loaded')
        c=self.c;t=target_id(state[2:]);r=c['rewards'];q=[]
        for a in range(self.spec['actions']):
            x,y=clipped_position(*state[:2],a);base=r['step']+(r['hit'] if tuple(state[:2])==tuple(state[2:4]) else r['miss']) if a==5 else r['step']
            q.append(base+self.expected[(x*25+y)*5000+t])
        return np.asarray(q)

def precompute(c):
    native=Native(c);maps={n:arrays(c,n) for n in (10,14)};records=[];invariants=[];started=time.perf_counter()
    for seed in [None]+c['planning']['training_seeds']:
        key,identity=cache_key(c,14,seed);folder=OUT/'cache'/key;folder.mkdir(parents=True,exist_ok=False)
        a=native.initialize(seed);b=a.copy()
        for h in (1,2,3):
            start=time.perf_counter();a,pa,ea,sa=native.advance(a,maps[10],10);b,pb,eb,sb=native.advance(b,maps[14],14)
            path10=bundle(c,10,seed,h)
            with np.load(path10,allow_pickle=False) as z:
                require(np.array_equal(a,z['value']) and np.array_equal(pa,z['policy']) and np.array_equal(ea,z['expected']),'A10 full-layer reproduction failed')
            delta=b-a;tol=c['planning']['monotonicity_absolute_tolerance'];violations=int(np.count_nonzero(delta < -tol))
            require(violations==0,'Subset monotonicity failed')
            invariants.append(dict(seed=seed,horizon=h,states=N_STATE,minimum_difference=float(delta.min()),maximum_difference=float(delta.max()),
                violations=violations,tolerance=tol,exact_A10_value_policy_expected=True,negative_roundoff_count=int(np.count_nonzero(delta<0))))
            meta=dict(cache_key=key,identity=identity,horizon=h,policy_sha256=array_hash(pb),value_sha256=array_hash(b),expected_sha256=array_hash(eb))
            path=folder/f'h{h}.npz';np.savez_compressed(path,value=b,policy=pb,expected=eb,metadata=json.dumps(meta,sort_keys=True))
            records.append(dict(seed=seed,horizon=h,bundle=path.relative_to(ROOT).as_posix(),sha256=sha(path),
                A10_reference=path10.relative_to(ROOT).as_posix(),A10_sha256=sha(path10),compute_and_storage_seconds=time.perf_counter()-start,
                A10_stats=list(map(int,sa)),A14_stats=list(map(int,sb))))
            del pa,pb,ea,eb,delta
        print('Precomputed and compared all A10/A14 layers, terminal',seed,flush=True)
    save_json(OUT/'monotonicity.json',dict(complete=True,comparisons=invariants,total_state_horizon_terminal_comparisons=18*N_STATE,
        tolerance=tol,violations=sum(r['violations'] for r in invariants),value_definition='Exact maximum stored; policy lowest ID within 1e-10 of maximum. Finite-horizon dominance does not imply rollout metric dominance.'))
    from ..reproduce import memory_record
    save_json(OUT/'precomputation.json',dict(records=records,elapsed_seconds=time.perf_counter()-started,memory=memory_record(),
        A10_value_backups=sum(r['A10_stats'][0] for r in records),A14_value_backups=sum(r['A14_stats'][0] for r in records),
        chance_terms=sum(r['A10_stats'][1]+r['A14_stats'][1] for r in records),action_lookups=sum(r['A10_stats'][2]+r['A14_stats'][2] for r in records),
        shared_expectation_reuses=sum(r['A10_stats'][3]+r['A14_stats'][3] for r in records),training_interactions=0,Q_updates=0))
