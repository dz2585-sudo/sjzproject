"""Python reference first: feedback recursion over A10/A14 at every depth."""
from functools import lru_cache
from ..tracking_planning.reference import target_outcomes,phi,require
from ..tracking_research.agent import array_hash
from .env import clipped_position

def transitions(c,state,action):
    x,y=clipped_position(*state[:2],action)
    hit=action==5 and (x,y)==tuple(state[2:4]);r=c['rewards']
    base=r['step']+(r['hit'] if hit else r['miss']) if action==5 else r['step']
    return tuple((p,(x,y,*t),base,base+c['close_weight']*(abs(x-t[0])+abs(y-t[1])<=1))
        for p,t in target_outcomes(tuple(state[2:]),c['planning']['turn_probability'],c['planning']['speed_toggle_probability']))

class ReferencePlanner:
    def __init__(self,c,action_count=14,agent=None,shaped=False):
        require(action_count in (10,14),'Invalid action set')
        self.c=c;self.action_count=action_count;self.agent=agent
        self.identity=(tuple(c['actions'][:action_count]),3,c['gamma'],c['close_weight'],tuple(sorted(c['rewards'].items())),
            c['planning']['turn_probability'],c['planning']['speed_toggle_probability'],c['planning']['tie_tolerance'],
            c['planning']['terminal_version'],array_hash(agent.q) if agent else 'Phi',shaped)
        self.backups=0;self.terms=0
        @lru_cache(maxsize=None)
        def backup(state,h):
            if h==0:
                v=float(phi(state))+(float(agent.q[agent.index(state)].max()) if agent else 0)
                return v-phi(state) if shaped else v,-1,()
            self.backups+=1;values=[]
            for a in range(action_count):
                e=0.
                for p,nxt,base,r in transitions(c,state,a):
                    self.terms+=1
                    if shaped:r+=c['gamma']*phi(nxt)-phi(state)
                    e+=p*(r+c['gamma']*backup(nxt,h-1)[0])
                values.append(e)
            best=max(values);chosen=next(a for a,v in enumerate(values) if v>=best-c['planning']['tie_tolerance'])
            return best,chosen,tuple(values)
        self.backup=backup
    def solve(self,state,h=3):
        from ..tracking_planning.core import state_id
        state_id(state);require(type(h) is int and 0<=h<=3,'Bad horizon')
        return self.backup(tuple(map(int,state)),h)
    def act(self,state,h=3):return self.solve(state,h)[1]
