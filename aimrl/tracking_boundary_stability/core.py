"""Readable float32 reference learner and compact, physical-state encoders."""
import copy
import ctypes
import json
import os
from pathlib import Path

import numpy as np

from ..tracking25.agent import epsilon
from ..tracking25.env import TrackingEnv
from ..tracking_research.agent import array_hash
from ..tracking_research.io import ROOT, read_json

CONDITIONS = ('R-Fixed', 'R-Decay', 'B-Fixed', 'B-Decay')
CONFIG = ROOT / 'configs/tracking_boundary_stability.json'
OUT = (ROOT / os.environ.get('AIMRL_BOUNDARY_OUTPUT','results/tracking_boundary_stability')).resolve()


def require(value, message):
    if not value:
        raise ValueError(message)


def config():
    c = read_json(CONFIG)
    old = read_json(ROOT / 'configs/tracking_shooting_tradeoff.json')
    for key in ('environment', 'actions', 'rewards', 'learning'):
        require(c[key] == old[key], f'Published parameter changed: {key}')
    require(c['environment']['grid_size'] == 25, 'This encoder is defined for 25 cells')
    require(c['study']['conditions'] == list(CONDITIONS), 'Invalid factorial design')
    require(c['study']['close_weight']==.5 and c['study']['decay']==dict(scale=1000,power=.6),
            'Prespecified reward / learning-rate formula differs')
    return c


def boundary(x):
    x = np.asarray(x)
    return np.where(x < 4, x, np.where(x <= 20, 4, x - 16))


class Encoder:
    """Enumerate valid one-axis keys; their product contains only valid 2-D keys.

    Physical x/y coordinate pairs can be combined independently. Heading and
    speed also have no physical constraints, so this product is exact and compact.
    """
    def __init__(self, kind):
        require(kind in ('R', 'B'), 'Unknown encoder')
        self.kind = kind
        def key(c, t):
            return (t-c,) if kind == 'R' else (t-c, int(boundary(c)), int(boundary(t)))
        self.keys = sorted({key(c, t) for c in range(25) for t in range(25)})
        lookup = {k: i for i, k in enumerate(self.keys)}
        self.axis = np.array([[lookup[key(c, t)] for t in range(25)] for c in range(25)], dtype=np.int32)
        self.width = len(self.keys)
        self.rows = self.width ** 2 * 8

    def indices(self, states):
        s = np.asarray(states, dtype=np.int32)
        return ((self.axis[s[..., 0], s[..., 2]] * self.width + self.axis[s[..., 1], s[..., 3]]) * 4 + s[..., 4]) * 2 + s[..., 5] - 1

    def index(self, state):
        s = np.asarray(state)
        require(s.shape == (6,) and s.dtype.kind in 'iu' and np.all((s[:4] >= 0) & (s[:4] < 25))
                and s[4] in range(4) and s[5] in (1, 2), 'Invalid physical observation')
        return int(self.indices(s))

    def record(self):
        return dict(representation=self.kind, axis_keys=self.width, rows=self.rows,
                    state_action_pairs=self.rows*10, q_bytes=self.rows*40,
                    counts_bytes=self.rows*40, index_map_bytes=self.axis.nbytes,
                    index_sha256=array_hash(self.axis))


class Agent:
    def __init__(self, c, condition, rng=None):
        require(condition in CONDITIONS, 'Unknown condition')
        self.condition = condition
        self.encoder = Encoder(condition[0])
        self.q = np.full((self.encoder.rows, 10), c['learning']['initial_q'], dtype=np.float32)
        self.counts = np.zeros(self.q.shape, dtype=np.uint32)
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.learning = c['learning']
        self.decay = condition.endswith('Decay')
        self.rule = c['study']['decay']

    def index(self, state):
        return self.encoder.index(state)

    def greedy(self, state):
        return int(np.argmax(self.q[self.index(state)]))

    def select(self, state, step):
        return int(self.rng.integers(10)) if self.rng.random() < epsilon(step, self.learning) else self.greedy(state)

    def alpha(self, n):
        return self.learning['alpha'] * (1.0 + n / self.rule['scale']) ** (-self.rule['power']) if self.decay else self.learning['alpha']

    def update(self, state, action, reward, next_state):
        row = self.index(state)
        n = int(self.counts[row, action])
        alpha = self.alpha(n)
        target = reward + self.learning['gamma'] * float(self.q[self.index(next_state)].max())
        before = float(self.q[row, action])
        # Explicit float32 increment and float32 addition reproduce the published update.
        self.q[row, action] = np.float32(self.q[row, action] + np.float32(alpha * (target - before)))
        self.counts[row, action] += 1
        return alpha, target, before, float(self.q[row, action]), n, row

    def snapshot(self):
        return array_hash(self.q), array_hash(self.counts), copy.deepcopy(self.rng.bit_generator.state)


def reward_parts(base, pre, post, c):
    close = c['study']['close_weight'] * (post <= 1)
    potential = c['learning']['gamma'] * -post + pre
    return base, close, potential, (base + close) + potential


def reference_run(env, state, n, agent, offset=0, train=True, guard=False, forced=None, c=None):
    """One real env.step and one real Q update per recorded interaction."""
    env.state = tuple(state)
    rows, debug = [], []
    for i in range(n):
        a = int(forced[i]) if forced is not None else agent.select(state, offset+i) if train else agent.greedy(state)
        if guard and a == 5 and state[:2] != state[2:4]:
            a = 4
        nxt, base, terminated, truncated, info = env.step(a)
        parts = reward_parts(base, info['pre_distance'], info['distance'], c)
        update = agent.update(state, a, parts[-1], nxt) if train else (0,)*6
        debug.append((*parts, *update))
        rows.append((*nxt, a, info['pre_distance'], abs(nxt[0]-state[2])+abs(nxt[1]-state[3]),
                     info['distance'], info['target_net_displacement'], info['hit'], info['miss'],
                     info['reflections'], info['random_turn'], info['speed_toggle']))
        require(not terminated, 'Unexpected true termination')
        state = nxt
    return np.asarray(rows, dtype=np.int16), np.asarray(debug, dtype=np.float64)


class Kernel:
    def __init__(self, library, c):
        self.lib = ctypes.CDLL(str(Path(library).resolve()))
        self.call = self.lib.rollout
        P, I = ctypes.c_void_p, ctypes.c_int
        self.call.argtypes = [P, P, P, I, P, P, I, I, I, I, P, P, P, P, P, P, P]
        self.call.restype = None
        l, e, r, s = c['learning'], c['environment'], c['rewards'], c['study']
        self.parameters = np.array([l['alpha'], l['gamma'], l['epsilon_start'], l['epsilon_end'],
                                    l['epsilon_decay_steps'], s['decay']['scale'], s['decay']['power'],
                                    r['step'], r['hit'], r['miss'], s['close_weight'],
                                    e['turn_probability'], e['speed_toggle_probability']], dtype=np.float64)

    def run(self, env, state, n, agent, offset=0, train=False, guard=False, forced=None, debug=False):
        require(n > 0 and env.elapsed+n <= env.cap, 'Native call crosses a reset boundary')
        initial = np.asarray(state, dtype=np.int32)
        draws = env.rng.random((n, 3))
        out = np.empty((n, 16), dtype=np.int16)
        audit = np.empty((n, 10), dtype=np.float64) if debug else None
        force = np.ascontiguousarray(forced, dtype=np.int16) if forced is not None else None
        require(force is None or (force.shape == (n,) and np.all((force >= 0) & (force < 10))), 'Invalid forced actions')
        rng = agent.rng.bit_generator.ctypes
        params = self.parameters.copy()
        params[-2:] = env.turn_probability, env.toggle_probability
        self.call(agent.q.ctypes.data, agent.counts.ctypes.data, agent.encoder.axis.ctypes.data,
                  agent.encoder.width, initial.ctypes.data, draws.ctypes.data, n, offset,
                  int(train), int(agent.decay) + 2*int(guard), rng.state,
                  ctypes.cast(rng.next_double, ctypes.c_void_p), ctypes.cast(rng.next_uint32, ctypes.c_void_p),
                  params.ctypes.data, force.ctypes.data if force is not None else None,
                  out.ctypes.data, audit.ctypes.data if audit is not None else None)
        env.state = tuple(map(int, out[-1, :6]))
        env.elapsed += n
        env.finished = env.elapsed >= env.cap
        return (out, audit) if debug else out


def save_model(path, agent, step, c, **extra):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    require(int(agent.counts.sum(dtype=np.uint64)) == step, 'Update sum differs from interactions')
    metadata = dict(condition=agent.condition, step=step, config=c, q_sha256=array_hash(agent.q),
                    counts_sha256=array_hash(agent.counts), encoder=agent.encoder.record(), **extra)
    np.savez_compressed(path, q=agent.q, counts=agent.counts, metadata=json.dumps(metadata, sort_keys=True))


def load_model(path, c):
    with np.load(path, allow_pickle=False) as z:
        m = json.loads(str(z['metadata']))
        require(m['config'] == c, 'Model config differs')
        a = Agent(c, m['condition'])
        require(m['encoder'] == a.encoder.record(), 'Model encoding differs')
        for k in ('q', 'counts'):
            data = z[k]
            require(data.shape == a.q.shape and data.dtype == getattr(a, k).dtype and np.isfinite(data).all()
                    and array_hash(data) == m[k+'_sha256'], 'Invalid model array '+k)
            getattr(a, k)[:] = data
        require(int(a.counts.sum(dtype=np.uint64)) == m['step'], 'Model counter mismatch')
        return a, m
