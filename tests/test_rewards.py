import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from aimrl.agent import QLearningAgent
from aimrl.config import load_config
from aimrl.env import AimEnv
from aimrl.rewards import learning_reward, potential
from aimrl.train import train


class RewardTests(unittest.TestCase):
    def setUp(self):
        Path('.tmp').mkdir(exist_ok=True)
        self.config = load_config()
        self.s3 = (0, 0, 3, 0, 1)
        self.s2 = (1, 0, 3, 0, 1)
        self.s1 = (2, 0, 3, 0, 1)
        self.s0 = (3, 0, 3, 0, 1)

    def test_potential(self):
        self.assertEqual(potential((0, 1, 3, 3, -1)), -5)
        self.assertEqual(potential(self.s0), 0)

    def test_hand_calculated_transitions(self):
        for state, nxt, expected_shaping, expected_reward in [(self.s3, self.s2, 1.1, 1.05),
                (self.s2, self.s3, -0.85, -0.90), (self.s2, self.s2, 0.10, 0.05)]:
            with self.subTest(state=state, nxt=nxt):
                result, shaping = learning_reward(-0.05, state, nxt, False, 'potential', 0.95, 1)
                self.assertAlmostEqual(shaping, expected_shaping)
                self.assertAlmostEqual(result, expected_reward)

    def test_termination_ignores_next_potential_and_bootstrap(self):
        learned, shape = learning_reward(9.95, self.s2, self.s3, True, 'potential', .95, 1)
        self.assertAlmostEqual(shape, 2)
        self.assertAlmostEqual(learned, 11.95)
        a = QLearningAgent(self.config)
        a.q[a.state_index(self.s3) + (0,)] = 100
        a.update(self.s2, 0, learned, self.s3, True)
        self.assertAlmostEqual(a.q[a.state_index(self.s2) + (0,)], .15 * 11.95, places=5)

    def test_truncation_retains_tail_and_bootstraps(self):
        learned, shape = learning_reward(-.05, self.s3, self.s2, False, 'potential', .95, 1)
        self.assertAlmostEqual(shape, 1.1)
        a = QLearningAgent(self.config)
        a.q[a.state_index(self.s2) + (0,)] = 4
        a.update(self.s3, 0, learned, self.s2, False, True)
        self.assertAlmostEqual(a.q[a.state_index(self.s3) + (0,)], .15 * (1.05 + .95 * 4), places=5)

    def test_base_and_zero_scale(self):
        for mode, scale in [('base', 1), ('potential', 0)]:
            self.assertEqual(learning_reward(-.05, self.s3, self.s2, False, mode, .95, scale), (-.05, 0))
        with self.assertRaises(ValueError):
            learning_reward(0, self.s3, self.s2, False, 'unknown', .95, 1)

    def test_fixed_transition_single_update(self):
        a = QLearningAgent(self.config)
        r, shape = learning_reward(-.05, self.s3, self.s2, False, 'potential', .95, 1)
        a.update(self.s3, 1, r, self.s2, False)
        self.assertAlmostEqual(a.q[a.state_index(self.s3) + (1,)], .15 * 1.05, places=6)

    def test_discounted_telescoping_terminal_and_truncated(self):
        for states, terminal in [([self.s3, self.s2, self.s2, self.s1, self.s0], True),
                                 ([self.s3, self.s2, self.s2, self.s1], False)]:
            terms = [learning_reward(0, states[i], states[i + 1], terminal and i == len(states) - 2,
                                     'potential', .95, 1)[1] for i in range(len(states) - 1)]
            lhs = sum(.95 ** i * x for i, x in enumerate(terms))
            tail = 0 if terminal else potential(states[-1])
            self.assertAlmostEqual(lhs, -potential(states[0]) + .95 ** len(terms) * tail)

    def test_transformation_pure_and_environment_identical(self):
        a, b = AimEnv(self.config), AimEnv(self.config)
        agent = QLearningAgent(self.config, np.random.default_rng(123))
        s1, _ = a.reset(seed=77)
        s2, _ = b.reset(seed=77)
        for action in [0, 1, 2, 3, 4, 5] * 3:
            old = s1
            env_before = copy.deepcopy(a.rng.bit_generator.state)
            action_before = copy.deepcopy(agent.rng.bit_generator.state)
            learning_reward(-.05, s1, s2, False, 'potential', .95, 1)
            self.assertEqual(a.rng.bit_generator.state, env_before)
            self.assertEqual(agent.rng.bit_generator.state, action_before)
            self.assertEqual(s1, old)
            left, right = a.step(action), b.step(action)
            self.assertEqual(left, right)
            s1, s2 = left[0], right[0]
            learning_reward(left[1], old, s1, left[2], 'potential', .95, 1)
            if left[2] or left[3]:
                break

    def test_training_zero_scale_and_potential_reproducibility(self):
        config = copy.deepcopy(self.config)
        config['learning'].update(training_steps=300, epsilon_decay_steps=200, shaping_scale=0)
        config['experiments']['validation'].update(interval_steps=100, episodes=4)
        with tempfile.TemporaryDirectory(dir=Path('.tmp').resolve()) as directory:
            root = Path(directory)
            base, _ = train(config, 7, root / 'base', reward_mode='base')
            zero, _ = train(config, 7, root / 'zero', reward_mode='potential')
            np.testing.assert_array_equal(base.q, zero.q)
            config['learning']['shaping_scale'] = 1
            first, _ = train(config, 7, root / 'first', reward_mode='potential')
            second, _ = train(config, 7, root / 'second', reward_mode='potential')
            no_mid, _ = train(config, 7, root / 'no_mid', validate_mid=False, reward_mode='potential')
            np.testing.assert_array_equal(first.q, second.q)
            np.testing.assert_array_equal(first.q, no_mid.q)


if __name__ == '__main__':
    unittest.main()
