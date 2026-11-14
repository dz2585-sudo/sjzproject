"""Readable finite-horizon feedback Bellman recursion, with no environment input."""
from functools import lru_cache

import numpy as np

from ..tracking25.env import advance_target,clipped_position
from ..tracking_research.agent import array_hash


def require(value,message):
    if not value:raise ValueError(message)


def phi(state):
    return -abs(state[0]-state[2])-abs(state[1]-state[3])


@lru_cache(maxsize=None)
def target_outcomes(target,turn_probability,toggle_probability):
    """Turn, then toggle, then unit-substep reflection; merge complete target states."""
    xt,yt,heading,speed=target
    merged={}
    for turn,pt in ((0,1-turn_probability),(1,turn_probability/2),(-1,turn_probability/2)):
        for toggle,ps in ((0,1-toggle_probability),(1,toggle_probability)):
            probability=pt*ps
            if probability==0:continue
            h=(heading+turn)%4;s=3-speed if toggle else speed
            x,y,h,_=advance_target(xt,yt,h,s,25)
            key=(x,y,h,s)
            merged[key]=merged.get(key,0.0)+probability
    require(abs(sum(merged.values())-1)<1e-14,'Transition probabilities do not sum to one')
    return tuple((p,s) for s,p in merged.items())


def transitions(c,state,action,model=None):
    model=c['planning'] if model is None else model
    x,y=clipped_position(state[0],state[1],action,25)
    hit=action==5 and (x,y)==state[2:4]
    miss=action==5 and not hit
    r=c['rewards'];base=r['step']+r['hit']*hit+r['miss']*miss
    result=[]
    for p,target in target_outcomes(tuple(state[2:]),model['turn_probability'],model['speed_toggle_probability']):
        nxt=(x,y,*target)
        result.append((p,nxt,base,base+c['close_weight']*(phi(nxt)>=-1)))
    return tuple(result)


class ReferencePlanner:
    """Each child chooses its own best action; this is not open-loop sequence search."""
    def __init__(self,c,terminal='Phi',agent=None,shaped=False,prune_fire=False):
        require(terminal in ('Phi','Zero','Q+Phi'),'Unknown terminal value')
        require(terminal!='Q+Phi' or agent is not None,'Frozen Q terminal requires a model')
        self.c=c;self.terminal=terminal;self.agent=agent;self.shaped=shaped;self.prune_fire=prune_fire
        self.identity=(c['gamma'],c['close_weight'],c['planning']['turn_probability'],c['planning']['speed_toggle_probability'],
                       tuple(sorted(c['rewards'].items())),terminal,array_hash(agent.q) if agent is not None else None,shaped,prune_fire)
        self.backups=0;self.transition_terms=0

        @lru_cache(maxsize=None)
        def backup(state,h):
            if h==0:
                task=0.0 if terminal=='Zero' else float(phi(state))
                if terminal=='Q+Phi':task+=float(agent.q[agent.index(state)].max())
                return (task-phi(state) if shaped else task),-1,()
            self.backups+=1
            values=[]
            for a in range(10):
                if prune_fire and a==5 and phi(state)!=0:
                    values.append(-float('inf'));continue
                expected=0.0
                for p,nxt,base,reward in transitions(c,state,a):
                    self.transition_terms+=1
                    if shaped:reward+=c['gamma']*phi(nxt)-phi(state)
                    expected+=p*(reward+c['gamma']*backup(nxt,h-1)[0])
                values.append(expected)
            best=max(values)
            action=next(a for a,v in enumerate(values) if v>=best-c['planning']['tie_tolerance'])
            return best,action,tuple(values)
        self.backup=backup

    def solve(self,state,horizon):
        state=tuple(map(int,state))
        require(len(state)==6 and all(0<=v<25 for v in state[:4]) and state[4] in range(4) and state[5] in (1,2),'Invalid observation')
        require(type(horizon) is int and 1<=horizon<=3,'Horizon must be 1, 2 or 3')
        return self.backup(state,horizon)

    def act(self,state,horizon):return self.solve(state,horizon)[1]

    def statistics(self):
        info=self.backup.cache_info()
        return dict(value_backups=self.backups,transition_terms=self.transition_terms,cache_hits=info.hits,
                    cache_misses=info.misses,cache_entries=info.currsize)
