"""Independent one-tick A14 extension; no mutation of historical action constants."""
import numbers
import numpy as np
from ..tracking25.env import TrackingEnv, MOVES as OLD_MOVES, ACTION_NAMES as OLD_NAMES, advance_target

MOVES=OLD_MOVES+((-2,0),(2,0),(0,-2),(0,2))
ACTION_NAMES=OLD_NAMES+('left_medium','right_medium','up_medium','down_medium')

def clipped_position(x,y,action,size=25):
    dx,dy=MOVES[action]
    return max(0,min(size-1,x+dx)),max(0,min(size-1,y+dy))

def mode(a):
    return 'slow' if a<4 else 'stay' if a==4 else 'fire' if a==5 else 'fast' if a<10 else 'medium'

class ActionEnv(TrackingEnv):
    def __init__(self,c,action_count=14):
        if action_count not in (10,14):raise ValueError('A10 or A14 required')
        if tuple(c['actions'][:10])!=OLD_NAMES or tuple(c['actions'])!=ACTION_NAMES:raise ValueError('Action schema differs')
        super().__init__({**c,'actions':list(OLD_NAMES)})
        self.action_count=action_count

    def step(self,action):
        if self.state is None or self.finished:raise RuntimeError('Reset before stepping')
        if isinstance(action,(bool,np.bool_)) or not isinstance(action,numbers.Integral) or not 0<=action<self.action_count:
            raise ValueError('Invalid action for this action set')
        action=int(action);xc,yc,xt,yt,h,s=self.state;ox,oy=xc,yc
        pre=abs(xc-xt)+abs(yc-yt);xc,yc=clipped_position(xc,yc,action,self.size)
        hit=action==5 and (xc,yc)==(xt,yt);miss=action==5 and not hit
        ut,us,uv=self.rng.random(3);turn=ut<self.turn_probability;toggle=uv<self.toggle_probability
        if turn:h=(h+(1 if us<.5 else -1))%4
        if toggle:s=3-s
        tx,ty=xt,yt;xt,yt,h,bounce=advance_target(xt,yt,h,s,self.size)
        self.state=(xc,yc,xt,yt,h,s);self.elapsed+=1;self.finished=self.elapsed>=self.cap
        reward=self.rewards['step']+self.rewards['hit']*hit+self.rewards['miss']*miss
        info=dict(elapsed_ticks=self.elapsed,hit=hit,miss=miss,shot=(xc,yc) if action==5 else None,base_reward=reward,
            pre_distance=pre,distance=abs(xc-xt)+abs(yc-yt),movement_mode=mode(action),actual_displacement=abs(xc-ox)+abs(yc-oy),
            random_turn=bool(turn),speed_toggle=bool(toggle),reflections=bounce,target_nominal_speed=s,target_path_distance=s,
            target_net_displacement=abs(xt-tx)+abs(yt-ty),nominal_displacement=sum(map(abs,MOVES[action])),
            actual_dx=xc-ox,actual_dy=yc-oy)
        return self.state,float(reward),False,self.finished,info
