"""Reward and exact rollout parity for the independent trade-off study."""
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from aimrl.tracking25.agent import epsilon, learning_reward
from aimrl.tracking25.env import TrackingEnv
from aimrl.tracking25.policies import rule_policy
from aimrl.tracking_research.agent import ResearchAgent
from aimrl.tracking_research.io import ROOT
from aimrl.tracking_tradeoff.core import Kernel, evaluate, load_model, reward_parts, save_model, window_metrics
from aimrl.tracking_tradeoff.study import config_checked


class TradeoffRewardTests(unittest.TestCase):
    def test_reward_terms_use_post_motion_and_preserve_base(self):
        base, distance, close, potential, train = reward_parts(9.95, 0, 2, (0.3, 2))
        self.assertEqual((base, distance, close, potential), (9.95, -0.6, 0, -1.98))
        self.assertAlmostEqual(train, 7.37)
        self.assertEqual(reward_parts(-0.05, 2, 1, (0, 5))[1:4], (0, 5, 1.01))

    def test_fire_before_motion_and_speed_two_reflection(self):
        config = config_checked()
        env = TrackingEnv(config)
        for position, net, reflected in ((10, 2, 0), (23, 0, 1)):
            with self.subTest(position=position):
                env.reset(1)
                env.state = (position, 10, position, 10, 0, 2)
                env.elapsed = 299
                _, reward, terminated, truncated, info = env.step(5)
                self.assertEqual(reward, 9.95)
                self.assertFalse(terminated)
                self.assertTrue(truncated)
                self.assertTrue(info['hit'])
                self.assertEqual((info['distance'], info['target_net_displacement'], info['reflections']),
                                 (net, net, reflected))


@unittest.skipUnless(os.environ.get('AIMRL_TRADEOFF_KERNEL'), 'Set AIMRL_TRADEOFF_KERNEL to the compiled library')
class TradeoffKernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = config_checked()
        cls.kernel = Kernel(os.environ['AIMRL_TRADEOFF_KERNEL'])

    def compare_training(self, weights, seed, offset):
        env_sequence, action_sequence = np.random.SeedSequence(seed).spawn(2)
        start = int(env_sequence.generate_state(1, dtype=np.uint64)[0])
        native_env, python_env = TrackingEnv(self.config), TrackingEnv(self.config)
        native_env.reset(start)
        python_env.reset(start)
        native = ResearchAgent(self.config, 'relative', np.random.default_rng(action_sequence))
        python = ResearchAgent(self.config, 'relative', np.random.default_rng(action_sequence))
        step = 0
        for length in (300, 300, 300, 101):
            state = python_env.state
            initial = native_env.state
            observed = self.kernel.run(native_env, initial, length, native, offset + step, weights, train=True)
            for row in observed:
                action = python.select(state, epsilon(offset + step, self.config['learning']))
                next_state, base, terminated, truncated, info = python_env.step(action)
                signal = reward_parts(base, info['pre_distance'], info['distance'], weights)[-1]
                if weights == (0, 0):
                    self.assertEqual(signal, learning_reward(base, state, next_state, 'potential', 0.99)[0])
                self.assertFalse(terminated)
                self.assertEqual(truncated, (step + 1) % 300 == 0)
                self.assertEqual(tuple(row[:6]), next_state)
                self.assertEqual(tuple(row[[6, 7, 9, 10, 11, 12, 13, 14, 15]]),
                                 (action, info['pre_distance'], info['distance'], info['target_net_displacement'],
                                  info['hit'], info['miss'], info['reflections'], info['random_turn'], info['speed_toggle']))
                python.update(state, action, signal, next_state)
                state = next_state
                step += 1
            window_metrics(initial, observed, detailed=True)
            if length == 300:
                self.assertEqual(native_env.reset(), python_env.reset())
        self.assertEqual(step, 1001)
        self.assertEqual(native.snapshot(), python.snapshot())
        self.assertEqual(native_env.rng.bit_generator.state, python_env.rng.bit_generator.state)
        self.assertEqual(int(native.counts.sum()), 1001)
        return native

    def test_zero_weight_matches_legacy_1001_updates(self):
        self.compare_training((0, 0), 99, 0)

    def test_all_weighted_updates_match_python_in_greedy_regime(self):
        for condition, weights in self.config['tradeoff']['conditions'].items():
            with self.subTest(condition=condition):
                self.compare_training(tuple(weights), 99, 3999800)

    def test_reload_identity_and_evaluation_isolation(self):
        agent = self.compare_training((0.3, 0), 99, 0)
        workspace = ROOT / '.tmp'
        workspace.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=workspace) as directory:
            path = Path(directory) / 'model.npz'
            save_model(path, agent, self.config, 'distance_0p3', 99, 1001)
            loaded = load_model(path, self.config, 'distance_0p3', 99, 1001)
            np.testing.assert_array_equal(loaded.q, agent.q)
            np.testing.assert_array_equal(loaded.counts, agent.counts)
            with self.assertRaisesRegex(ValueError, 'identity'):
                load_model(path, self.config, 'reference', 99, 1001)
        before = agent.snapshot()
        learned, _ = evaluate(self.config, self.kernel, agent, [4100000, 4100001])
        rule, traces = evaluate(self.config, self.kernel, None, [4100000, 4100001], keep_traces=True)
        self.assertEqual(before, agent.snapshot())
        self.assertEqual([row['target_sha256'] for row in learned], [row['target_sha256'] for row in rule])
        env = TrackingEnv(self.config)
        for seed, (initial, rows) in traces.items():
            state, _ = env.reset(seed)
            self.assertEqual(state, initial)
            for row in rows:
                action = rule_policy(state, 25)
                state, _, _, _, _ = env.step(action)
                self.assertEqual(tuple(row[:7]), (*state, action))
