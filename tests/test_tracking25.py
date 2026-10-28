import copy
import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

from aimrl.tracking25.agent import TrackingAgent, epsilon, exact_reload_check, learning_reward, potential
from aimrl.tracking25.env import TrackingEnv, advance_target
from aimrl.tracking25.evaluate import evaluate
from aimrl.tracking25.metrics import EpisodeMetrics, aggregate, recovery
from aimrl.tracking25.policies import rule_policy
from aimrl.tracking25.workflow import read_json, train_run


class FixedDraws:
    def __init__(self, values): self.values, self.calls = values, 0
    def random(self, count):
        assert count == 3
        self.calls += 1
        return np.array(self.values)


class TrackingTests(unittest.TestCase):
    def setUp(self):
        self.config = read_json(Path(__file__).resolve().parent.parent / 'configs/tracking25_v2.json')
        self.small = copy.deepcopy(self.config)
        self.small['environment'].update(grid_size=5, minimum_reset_distance=2, episode_ticks=6)
        Path('.tmp').mkdir(exist_ok=True)

    def test_reset_feasibility_and_indexing(self):
        env = TrackingEnv(self.small)
        for seed in range(20):
            state, _ = env.reset(seed=seed)
            self.assertGreaterEqual(abs(state[0] - state[2]) + abs(state[1] - state[3]), 2)
            self.assertIn(state[4], range(4)); self.assertIn(state[5], (1, 2))
        agent = TrackingAgent(self.small)
        self.assertEqual(agent.index((1, 2, 3, 4, 0, 1)), (1, 2, 3, 4, 0, 0))
        self.assertEqual(agent.index((1, 2, 3, 4, 3, 2)), (1, 2, 3, 4, 3, 1))
        bad = copy.deepcopy(self.small); bad['environment']['minimum_reset_distance'] = 9
        with self.assertRaises(ValueError): TrackingEnv(bad)

    def test_slow_fast_clipping_and_invalid_immutability(self):
        env = TrackingEnv(self.config); env.reset(seed=0)
        for action, expected in [(0, (9, 10)), (1, (11, 10)), (2, (10, 9)), (3, (10, 11)),
                                 (6, (7, 10)), (7, (13, 10)), (8, (10, 7)), (9, (10, 13))]:
            env.state = (10, 10, 20, 20, 0, 1)
            state, _, _, _, info = env.step(action)
            self.assertEqual(state[:2], expected)
            self.assertEqual(info['actual_displacement'], 3 if action >= 6 else 1)
        env.state = (0, 0, 10, 10, 0, 1)
        self.assertEqual(env.step(6)[4]['actual_displacement'], 0)
        before = copy.deepcopy(env.rng.bit_generator.state); state = env.state; elapsed = env.elapsed
        for bad in [True, np.bool_(False), 1.2, -1, 10, None]:
            with self.assertRaises(ValueError): env.step(bad)
            self.assertEqual(env.state, state); self.assertEqual(env.elapsed, elapsed)
            self.assertEqual(env.rng.bit_generator.state, before)

    def test_perpendicular_turn_speed_toggle_and_unit_reflection(self):
        env = TrackingEnv(self.config); env.reset(seed=0)
        env.state = (0, 0, 10, 10, 0, 1); env.rng = FixedDraws((0, .1, 0))
        state, _, _, _, info = env.step(4)
        self.assertEqual(state[2:], (10, 12, 1, 2))
        self.assertTrue(info['random_turn'] and info['speed_toggle']); self.assertEqual(env.rng.calls, 1)
        env.state = (0, 0, 10, 10, 0, 1); env.rng = FixedDraws((0, .9, 1))
        self.assertEqual(env.step(4)[0][2:], (10, 9, 3, 1))
        self.assertEqual(advance_target(3, 2, 0, 2, 5), (3, 2, 2, 1))
        env.state = (0, 0, 23, 2, 0, 2); env.rng = FixedDraws((1, 1, 1))
        _, _, _, _, info = env.step(4)
        self.assertEqual(info['target_path_distance'], 2); self.assertEqual(info['target_net_displacement'], 0)

    def test_hit_moves_target_and_play_continues(self):
        env = TrackingEnv(self.config); env.reset(seed=0)
        env.state = (10, 10, 10, 10, 0, 1); env.rng = FixedDraws((1, 1, 1))
        state, reward, term, trunc, info = env.step(5)
        self.assertTrue(info['hit']); self.assertEqual(info['shot'], (10, 10))
        self.assertEqual(state[2:4], (11, 10)); self.assertAlmostEqual(reward, 9.95)
        self.assertFalse(term or trunc)
        state, reward, term, trunc, info = env.step(5)
        self.assertTrue(info['miss']); self.assertEqual(state[2:4], (12, 10)); self.assertAlmostEqual(reward, -1.05)
        self.assertEqual(env.rng.calls, 2)

    def test_exact_300_ticks_and_hit_at_truncation(self):
        env = TrackingEnv(self.config); env.reset(seed=0)
        for tick in range(300):
            _, _, term, trunc, _ = env.step(4)
            self.assertFalse(term); self.assertEqual(trunc, tick == 299)
        with self.assertRaises(RuntimeError): env.step(4)
        env.reset(seed=0); env.elapsed = 299; env.state = (10, 10, 10, 10, 0, 1)
        _, _, term, trunc, info = env.step(5)
        self.assertTrue(info['hit'] and trunc); self.assertFalse(term)

    def test_shaping_once_and_truncation_bootstrap(self):
        agent = TrackingAgent(self.small)
        state, nxt = (0, 0, 3, 0, 0, 1), (1, 0, 3, 0, 0, 1)
        reward, shaped = learning_reward(-.05, state, nxt, 'potential', .99)
        self.assertAlmostEqual(shaped, 1.02); self.assertAlmostEqual(reward, .97)
        agent.q[agent.index(nxt) + (4,)] = 4
        agent.update(state, 1, reward, nxt)
        self.assertAlmostEqual(agent.q[agent.index(state) + (1,)], .15 * (.97 + .99 * 4), places=5)
        hit_state, hit_next = (2, 2, 2, 2, 0, 1), (2, 2, 3, 2, 0, 1)
        learned, shaped = learning_reward(9.95, hit_state, hit_next, 'potential', .99)
        self.assertAlmostEqual(shaped, -.99)
        self.assertAlmostEqual(learned, 8.96)

    def test_target_paths_independent_of_actions_and_global_rng(self):
        a, b = TrackingEnv(self.config), TrackingEnv(self.config)
        self.assertEqual(a.reset(seed=99), b.reset(seed=99))
        for i in range(300):
            left = a.step(4); right = b.step(i % 10)
            self.assertEqual(left[0][2:], right[0][2:])
            self.assertEqual(left[4]['random_turn'], right[4]['random_turn'])
        self.assertEqual(a.reset(), b.reset())

    def test_null_accuracy_and_recovery_edge_cases(self):
        result, rows, _ = evaluate(self.small, 'base', TrackingAgent(self.small), seeds=[3600000])
        self.assertIsNone(result['pooled_shot_accuracy'])
        self.assertEqual(recovery([8] * 300, [True] + [False] * 299), (1, 0, 20))
        distances = [8] * 300; distances[20] = 1
        self.assertEqual(recovery(distances, [True] + [False] * 299), (1, 1, 20))
        self.assertEqual(recovery([0] * 300, [False] * 300), (0, 0, 0))
        env = TrackingEnv(self.small); state, _ = env.reset(seed=0); metrics = EpisodeMetrics(state)
        for i in range(6):
            state, reward, _, _, info = env.step(4)
            info.update(random_turn=True, speed_toggle=True, reflections=1)
            metrics.add(4, state, reward, info)
        self.assertEqual(len(metrics.events), 6)
        self.assertTrue(all(metrics.events))
        metrics = EpisodeMetrics((0, 0, 1, 1, 0, 1))
        for tick in range(300):
            info.update(distance=0, pre_distance=0, hit=False, miss=False, random_turn=tick == 0,
                        speed_toggle=tick == 0, reflections=int(tick == 0))
            metrics.add(4, (0, 0, 1, 1, 0, 1), -.05, info)
        self.assertEqual(metrics.finish()['eligible_events'], 1)
        self.assertEqual(metrics.finish()['recovered_events'], 1)

    def test_predictive_rule_clipped_ties_and_no_future_rng(self):
        self.assertEqual(rule_policy((2, 2, 2, 2, 0, 2), 5), 5)
        self.assertEqual(rule_policy((0, 2, 3, 2, 0, 2), 5), 7)
        self.assertIn(rule_policy((0, 0, 1, 1, 0, 1), 5), range(10))

    def test_greedy_ties_epsilon_and_model_schema(self):
        agent = TrackingAgent(self.small, np.random.default_rng(1)); state = (0, 0, 1, 1, 0, 1)
        self.assertEqual(agent.select(state, 0), 0)
        agent.q[agent.index(state) + (7,)] = 3
        self.assertEqual(agent.select(state, 0), 7)
        self.assertEqual(epsilon(0, self.config['learning']), 1)
        self.assertAlmostEqual(epsilon(16000000, self.config['learning']), .05)
        with tempfile.TemporaryDirectory(dir=Path('.tmp').resolve()) as directory:
            path = Path(directory) / 'q.npy'; agent.save(path)
            other = TrackingAgent(self.small); other.load(path)
            self.assertTrue(exact_reload_check(agent, path)); np.testing.assert_array_equal(agent.q, other.q)
            np.save(path, np.zeros((5, 5, 5, 5, 2, 6), np.float32))
            with self.assertRaisesRegex(ValueError, 'Incompatible'): other.load(path)

    def test_short_training_repeatability_and_validation_isolation(self):
        config = copy.deepcopy(self.small)
        config['learning'].update(training_steps=20, epsilon_decay_steps=16)
        config['experiments'].update(validation_interval=10, validation_episodes=2)
        with tempfile.TemporaryDirectory(dir=Path('.tmp').resolve()) as directory:
            root = Path(directory)
            a = train_run(config, 'potential', root / 'a')
            b = train_run(config, 'potential', root / 'b')
            c = train_run(config, 'potential', root / 'c', validations=False)
            np.testing.assert_array_equal(np.load(root / 'a/q_table.npy'), np.load(root / 'b/q_table.npy'))
            np.testing.assert_array_equal(np.load(root / 'a/q_table.npy'), np.load(root / 'c/q_table.npy'))
            self.assertEqual(a['global_steps'], 20); self.assertEqual(a['partial_window']['steps'], 2)
            self.assertEqual(a['complete_windows'], 3); self.assertFalse(a['partial_window']['truncated'])
            self.assertFalse(a['partial_window']['terminated'])

    def test_bounded_full_grid_table_budget(self):
        agent = TrackingAgent(self.config)
        self.assertEqual(agent.q.shape, (25, 25, 25, 25, 4, 2, 10))
        self.assertEqual(agent.q.nbytes, 125000000)
        self.assertFalse(np.any(agent.q))


if __name__ == '__main__': unittest.main()
