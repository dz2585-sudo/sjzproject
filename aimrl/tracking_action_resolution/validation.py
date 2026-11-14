"""Fail-closed correctness gates; counted separately from formal evaluation."""
import copy,json,time
import numpy as np
from ..tracking25.env import TrackingEnv
from ..tracking_planning.core import state_id,frozen_agent,Policy as OldPolicy,methods as old_methods
from ..tracking_planning.reference import phi,require
from ..tracking_planning.validation import FixedDraws
from ..tracking_research.io import ROOT,read_json,save_json,save_csv
from .env import ActionEnv,clipped_position,MOVES
from .reference import ReferencePlanner,transitions
from .core import OUT,Native,Policy,methods,bundle

def python_checks(c):
    old=read_json(ROOT/'configs/tracking_planning.json');steps=0;branch_steps=0
    for seed in (381,519,1001,8889):
        a=TrackingEnv(old);b=ActionEnv(c);require(a.reset(seed)==b.reset(seed),'Reset differs')
        for t in range(300):
            aa=a.step(t%10);bb=b.step(t%10)
            require(aa[:4]==bb[:4] and all(bb[4][k]==v for k,v in aa[4].items()),'Original action mechanism differs')
            require(a.rng.bit_generator.state==b.rng.bit_generator.state,'Original RNG differs');steps+=2
    for state in [(0,0,0,0,2,1),(24,24,24,24,0,2),(1,23,23,1,1,2),(12,12,13,12,0,2)]:
        for a in range(14):
            outcomes=transitions(c,state,a);emp={}
            for prob,draw in [(p*q,(u,v,w)) for p,u,v in [(.95,.8,.2),(.025,.01,.2),(.025,.01,.8)] for q,w in [(.95,.8),(.05,.01)]]:
                env=ActionEnv(c);env.state=state;env.rng=FixedDraws(draw);nxt,base,_,_,info=env.step(a);branch_steps+=1
                entry=next(x for x in outcomes if x[1]==nxt)
                require(base==entry[2] and base+.5*(info['distance']<=1)==entry[3],'Python transition/reward mismatch')
                require(info['nominal_displacement']==sum(map(abs,MOVES[a])) and info['actual_displacement']==abs(nxt[0]-state[0])+abs(nxt[1]-state[1]),'Movement instrumentation differs')
                emp[nxt]=emp.get(nxt,0)+prob
            require(emp=={s:p for p,s,_,_ in outcomes},'Turn/toggle transition law differs')
    # Independent action streams consume exactly the same target draws, including medium moves.
    a=ActionEnv(c);b=ActionEnv(c);a.reset(99007);b.reset(99007)
    for t in range(300):
        aa=a.step(t%14);bb=b.step((13-t)%14);steps+=2
        require(aa[0][2:]==bb[0][2:] and a.rng.bit_generator.state==b.rng.bit_generator.state,'Policy-dependent target route')
    # Clipping cases include genuine 2, 1 and 0 displacement, one tick each.
    for pos,a,expected in [((3,4),10,(1,4)),((1,4),10,(0,4)),((0,4),10,(0,4)),((23,24),11,(24,24)),((4,1),12,(4,0)),((4,23),13,(4,24))]:
        require(clipped_position(*pos,a)==expected,'Medium clipping failed')
    e=ActionEnv(c,10);e.reset(1)
    try:e.step(10)
    except ValueError:pass
    else:raise ValueError('A10 accepted a medium action')
    r=ReferencePlanner(c,10);oldref=__import__('aimrl.tracking_planning.reference',fromlist=['ReferencePlanner']).ReferencePlanner(old)
    s=(23,12,24,12,0,2)
    for h in (1,2,3):require(r.solve(s,h)==oldref.solve(s,h),'Disabling medium did not restore A10')
    return dict(complete=True,environment_steps=steps+branch_steps,branch_steps=branch_steps,original_actions_RNG_exact=True,
        target_policy_independent=True,medium_clipping_2_1_0=True,disabled_medium_restores_A10=True)

def validate(c,folder):
    folder.mkdir(parents=True,exist_ok=False);start=time.perf_counter();checks=python_checks(c);rows=[];max_error=0.;shape_error=0.;steps=0
    states=[(23,12,24,12,0,2),(0,1,0,0,2,1),(12,12,12,12,1,2)]
    # Full-array A10 equality and monotonicity are checked in precompute; here independent Python recursion.
    for spec in methods(c):
        a=frozen_agent('R-Decay',spec['seed']) if spec['seed'] else None;before=a.snapshot() if a else None
        ref=ReferencePlanner(c,spec['actions'],a);shaped=ReferencePlanner(c,spec['actions'],a,True)
        for h in (1,2,3):
            with np.load(bundle(c,spec['actions'],spec['seed'],h),allow_pickle=False) as z:v=z['value'];p=z['policy'];expected=z['expected']
            for s in states:
                rv,ra,rq=ref.solve(s,h);sv,sa,sq=shaped.solve(s,h);idx=state_id(s)
                err=abs(float(v[idx])-rv);se=max(abs(np.asarray(rq)-np.asarray(sq)-phi(s)))
                require(err<1e-9 and int(p[idx])==ra,'Reference/native Bellman differs')
                require(se<1e-9 and ra==sa,'Task/shaped terminal conversion failed')
                # Check every action's native expected-value reconstruction, not merely the maximum.
                nativeq=[];from .env import clipped_position
                t=idx%5000
                for act in range(spec['actions']):
                    x,y=clipped_position(*s[:2],act);r=c['rewards'];base=r['step']+(r['hit'] if s[:2]==s[2:4] else r['miss']) if act==5 else r['step']
                    nativeq.append(base+expected[(x*25+y)*5000+t])
                qe=float(max(abs(np.asarray(nativeq)-rq)));require(qe<1e-9,'Per-action transition/value mismatch')
                max_error=max(max_error,err,qe);shape_error=max(shape_error,se)
                rows.append(dict(policy=spec['id'],horizon=h,state=json.dumps(s),value_error=err,all_action_value_error=qe,shaping_error=float(se),action=ra))
            del v,p,expected
        require(a is None or a.snapshot()==before,'Frozen learner mutated')
        # Full 300-tick native/Python parity, both real dynamics; RNG must end exactly equal.
        policy=Policy(c,spec);native=Native(c)
        for probability in (.05,.10):
            cc=copy.deepcopy(c);cc['environment'].update(turn_probability=probability,speed_toggle_probability=probability)
            e0,e1=ActionEnv(cc,spec['actions']),ActionEnv(cc,spec['actions']);initial,_=e0.reset(48211);e1.reset(48211)
            out,_=native.run(e0,initial,policy)
            for row in out:
                s=e1.state;proposed=policy.propose(s);rng=copy.deepcopy(e1.rng.bit_generator.state)
                require(policy.act(s)==policy.act(tuple(s)) and rng==e1.rng.bit_generator.state,'Planner consumed RNG')
                nxt,r,done,trunc,info=e1.step(policy.act(s));steps+=2
                expected=(*nxt,policy.act(s),info['pre_distance'],abs(nxt[0]-s[2])+abs(nxt[1]-s[3]),info['distance'],info['target_net_displacement'],
                    info['hit'],info['miss'],info['reflections'],info['random_turn'],info['speed_toggle'],proposed,s[5],info['nominal_displacement'],info['actual_dx'],info['actual_dy'],info['actual_displacement'])
                require(tuple(row)==expected and not done,'Native rollout differs from reference environment')
            require(e0.rng.bit_generator.state==e1.rng.bit_generator.state and trunc,'Native RNG/truncation differs')
        del a,ref,shaped,policy
    # Saved historical trajectory states, only observations, no historical complete reevaluation.
    old=read_json(ROOT/'configs/tracking_planning.json');historical=[]
    for spec in [m for m in methods(c) if m['actions']==10]:
        oldid='Plan-H3' if spec['seed'] is None else f'Hybrid-H3_s{spec["seed"]}'
        with np.load(ROOT/'results/tracking_planning/evaluation/final'/f'{oldid}_rollouts.npz',allow_pickle=False) as z:
            samples=np.vstack([z['initial'][0],z['ticks'][0,::30,:6]])
        p=Policy(c,spec);op=OldPolicy(old,next(m for m in old_methods(old) if m['id']==oldid))
        for s in samples:
            s=tuple(map(int,s));require(p.act(s)==op.act(s),'Historical saved-state A10 disagreement')
        historical.append(dict(policy=spec['id'],saved_states=len(samples),all_actions_equal=True))
    save_csv(folder/'python_native_shaping.csv',rows)
    result=dict(complete=True,python=checks,max_value_error=max_error,max_shaping_error=float(shape_error),historical_A10=historical,
        native_python_environment_steps=steps,total_environment_steps=steps+checks['environment_steps'],
        elapsed_seconds=time.perf_counter()-start,new_training_interactions=0,Q_updates=0)
    save_json(folder/'checks.json',result);return result
