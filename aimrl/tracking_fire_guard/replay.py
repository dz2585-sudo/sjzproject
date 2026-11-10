"""Pair independent Python environment rollouts with an inference-only guard."""
import math

import numpy as np

from ..tracking25.env import ACTION_NAMES, TrackingEnv
from ..tracking_tradeoff.core import COLUMNS, window_metrics

FIRE = ACTION_NAMES.index('fire')
STAY = ACTION_NAMES.index('stay')


def aligned(state):
    return state[0] == state[2] and state[1] == state[3]


class FiringGuard:
    def __init__(self, policy):
        self.policy = policy
        self.proposed_action = None

    def __call__(self, state):
        self.proposed_action = self.policy(state)
        return STAY if self.proposed_action == FIRE and not aligned(state) else self.proposed_action


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def require(condition, message):
    if not condition:
        raise ValueError(message)


def event_q(agent, state):
    index = agent.index(state)
    values, counts = agent.q[index], agent.counts[index]
    fire_q, stay_q = float(values[FIRE]), float(values[STAY])
    return {'q_fire': fire_q, 'q_stay': stay_q, 'q_gap': fire_q - stay_q,
            'fire_updates': int(counts[FIRE]), 'stay_updates': int(counts[STAY]),
            'fire_stay_tied': fire_q == stay_q,
            'max_tie_count': int(np.count_nonzero(values == fire_q)),
            'tied_max_action_ids': ' '.join(str(a) for a in np.flatnonzero(values == fire_q)),
            'relative_dx': state[2]-state[0], 'relative_dy': state[3]-state[1],
            'heading': state[4], 'speed': state[5]}


def replay_pair(config, seed, policy, agent=None, strict_rng=False, keep_trace=False):
    original_env, filtered_env = TrackingEnv(config), TrackingEnv(config)
    initial, _ = original_env.reset(seed)
    guarded_state, _ = filtered_env.reset(seed)
    require(initial == guarded_state, 'Initial physical states differ')
    guard = FiringGuard(policy)
    original_state = initial
    raw = np.empty((300,len(COLUMNS)), dtype=np.int16)
    filtered = np.empty_like(raw)
    events, trace = [], []
    original_return = guarded_return = 0.0
    if strict_rng:
        require(original_env.rng.bit_generator.state == filtered_env.rng.bit_generator.state,
                'Initial target RNG state differs')
    for i in range(300):
        raw_action = policy(original_state)
        executed = guard(guarded_state)
        require(guard.proposed_action == raw_action, 'Original proposals differ for paired states')
        changed = raw_action != executed
        require(changed == (raw_action == FIRE and not aligned(original_state))
                and (not changed or executed == STAY), 'Unexpected change to the executed action')
        before = original_state
        original_state, r0, done0, trunc0, info0 = original_env.step(raw_action)
        guarded_state, r1, done1, trunc1, info1 = filtered_env.step(executed)
        require(original_state == guarded_state and (done0, done1, trunc0, trunc1) ==
                (False, False, i==299, i==299), f'Physical or lifecycle divergence: {seed}/{i+1}')
        require(info0['distance'] == info1['distance'] and info0['hit'] == info1['hit']
                and info0['random_turn'] == info1['random_turn'] and info0['speed_toggle'] == info1['speed_toggle']
                and info0['reflections'] == info1['reflections'], f'Event divergence: {seed}/{i+1}')
        require(bool(info0['miss']) == changed and not info1['miss']
                and math.isclose(r1-r0, int(changed), abs_tol=1e-12), f'Miss or reward mismatch: {seed}/{i+1}')
        if strict_rng:
            require(original_env.rng.bit_generator.state == filtered_env.rng.bit_generator.state,
                    f'Target RNG state differs: {seed}/{i+1}')
        xc, yc, xt, yt, _, _ = original_state
        pre = abs(before[0]-before[2])+abs(before[1]-before[3])
        mid = abs(xc-before[2])+abs(yc-before[3])
        for row,action,info in ((raw[i],raw_action,info0),(filtered[i],executed,info1)):
            row[:] = (*original_state, action, pre, mid, info['distance'], info['target_net_displacement'],
                      info['hit'], info['miss'], info['reflections'], info['random_turn'], info['speed_toggle'])
        if changed and agent is not None:
            events.append({'scenario_seed':seed,'tick':i+1,'d_pre':pre,**event_q(agent,before)})
        original_return += r0
        guarded_return += r1
        if keep_trace:
            trace.append({'scenario_seed':seed,'tick':i+1,**dict(zip(COLUMNS[:6],original_state)),
                          'pre_xc':before[0],'pre_yc':before[1],'pre_xt':before[2],'pre_yt':before[3],
                          'd_pre':pre,'d_post':info0['distance'],'proposed_action':ACTION_NAMES[raw_action],
                          'guard_executed_action':ACTION_NAMES[executed],'hit':int(info0['hit']),
                          'original_miss':int(info0['miss']),'guard_miss':int(info1['miss']),
                          'filtered':int(changed),'original_reward':r0,'guard_reward':r1,
                          'target_turn':int(info0['random_turn']),'speed_toggle':int(info0['speed_toggle']),
                          'reflections':info0['reflections']})
    require(original_env.rng.bit_generator.state == filtered_env.rng.bit_generator.state, 'Final target RNG states differ')
    a,b = window_metrics(initial, raw),window_metrics(initial,filtered)
    for k in ('ticks','distance_sum','close_ticks','hits','target_sha256','hit_nonclose'):
        require(a[k] == b[k], f'Invariance failed: {k}')
    require(a['shots']-b['shots']==a['misses'] and b['misses']==0 and
            math.isclose(guarded_return-original_return,a['misses'],abs_tol=1e-9), 'Window accounting mismatch')
    for row, total in ((a,original_return),(b,guarded_return)):
        require(math.isclose(total,row['base_return'],abs_tol=1e-9), 'Actual and recomputed base returns differ')
        row['accuracy'] = ratio(row['hits'],row['shots'])
    return initial, raw, filtered, a, b, events, trace
