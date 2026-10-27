import copy
import unittest

import numpy as np

from aimrl.config import load_config
from aimrl.env import AimEnv
from aimrl.policies import random_policy, rule_policy


class EnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.config = copy.deepcopy(load_config())
        self.config["environment"].update(target_turn_probability=0, target_move_probability=0)
        self.env = AimEnv(self.config)
        self.env.reset(seed=10)

    def state(self, state, steps=0):
        # Explicit test-only setup; normal reset always uses random non-overlapping positions.
        self.env.state = state
        self.env.step_count = steps
        self.env.finished = False

    def test_reset_and_snapshots(self):
        obs, info = self.env.reset(seed=42)
        original_info = info.copy()
        self.assertTrue(all(0 <= x < 5 for x in obs[:4]))
        self.assertNotEqual(obs[:2], obs[2:4])
        self.assertIn(obs[4], (-1, 1))
        self.assertEqual(info["step_count"], 0)
        self.env.step(4)
        self.assertEqual(info, original_info)
        self.assertEqual(obs, self.env.reset(seed=42)[0])
        self.assertNotEqual(id(info), id(self.env.step(4)[4]))

    def test_all_movement_and_boundaries(self):
        for action, expected in [(0, (1, 2)), (1, (3, 2)), (2, (2, 1)), (3, (2, 3)), (4, (2, 2))]:
            with self.subTest(action=action):
                self.state((2, 2, 4, 4, 1))
                obs, reward, terminated, truncated, _ = self.env.step(action)
                self.assertEqual(obs[:2], expected)
                self.assertAlmostEqual(reward, -0.05)
                self.assertFalse(terminated or truncated)
        for action, position in [(0, (0, 2)), (1, (4, 2)), (2, (2, 0)), (3, (2, 4))]:
            with self.subTest(boundary=action):
                self.state((*position, 1, 1, 1))
                self.assertEqual(self.env.step(action)[0][:2], position)
        self.state((1, 1, 2, 1, 1))
        self.assertFalse(self.env.step(1)[2])  # Moving onto a target is not firing.

    def test_hit_before_target_motion(self):
        self.env.move_probability = self.env.turn_probability = 1
        self.state((2, 2, 2, 2, 1))
        obs, reward, term, trunc, info = self.env.step(5)
        self.assertEqual(obs, (2, 2, 2, 2, 1))
        self.assertAlmostEqual(reward, 9.95)
        self.assertTrue(term and not trunc and info["hit"] and not info["miss"])
        self.assertEqual(info["base_reward"], reward)

    def test_miss_then_target_enters_crosshair(self):
        self.env.move_probability = 1
        self.state((2, 2, 1, 2, 1))
        obs, reward, term, trunc, info = self.env.step(5)
        self.assertEqual(obs, (2, 2, 2, 2, 1))
        self.assertAlmostEqual(reward, -1.05)
        self.assertFalse(term or trunc)
        self.assertTrue(info["miss"] and not info["hit"])

    def test_target_bounces_at_both_edges(self):
        self.env.move_probability = 1
        for xt, direction, expected in [(4, 1, (3, -1)), (0, -1, (1, 1))]:
            with self.subTest(xt=xt):
                self.state((2, 2, xt, 1, direction))
                obs = self.env.step(4)[0]
                self.assertEqual((obs[2], obs[4]), expected)

    def test_direction_can_change_without_motion(self):
        self.env.turn_probability = 1
        self.state((2, 2, 1, 1, 1))
        self.assertEqual(self.env.step(4)[0], (2, 2, 1, 1, -1))

    def test_limit_and_hit_priority(self):
        self.state((0, 0, 4, 4, 1), 48)
        self.assertEqual(self.env.step(4)[2:4], (False, False))
        obs, reward, term, trunc, info = self.env.step(4)
        self.assertEqual((term, trunc, info["step_count"]), (False, True, 50))
        self.assertAlmostEqual(reward, -0.05)
        self.state((2, 2, 2, 2, 1), 49)
        self.assertEqual(self.env.step(5)[2:4], (True, False))

    def test_lifecycle_and_invalid_actions_do_not_consume_randomness(self):
        other = AimEnv(self.config)
        with self.assertRaisesRegex(RuntimeError, "reset"):
            other.step(0)
        self.env.reset(seed=123)
        other.reset(seed=123)
        before = self.env.state
        rng_before = copy.deepcopy(self.env.rng.bit_generator.state)
        for bad in (True, np.bool_(False), 1.0, "1", -1, 6, None):
            with self.subTest(action=bad), self.assertRaises(ValueError):
                self.env.step(bad)
            self.assertEqual(self.env.state, before)
            self.assertEqual(self.env.step_count, 0)
            self.assertEqual(self.env.rng.bit_generator.state, rng_before)
        self.assertEqual(self.env.step(np.int64(4)), other.step(4))
        self.state((1, 1, 1, 1, 1))
        self.env.step(5)
        with self.assertRaisesRegex(RuntimeError, "reset"):
            self.env.step(4)
        self.state((0, 0, 4, 4, 1), 49)
        self.env.step(4)
        with self.assertRaisesRegex(RuntimeError, "reset"):
            self.env.step(4)

    def test_reproducibility_and_rng_isolation(self):
        config = load_config()
        a, b = AimEnv(config), AimEnv(config)
        self.assertEqual(a.reset(seed=1234), b.reset(seed=1234))
        policy_rng = np.random.default_rng(44)
        for action in [0, 1, 2, 3, 4, 5] * 3:
            for _ in range(10):
                random_policy(policy_rng)
            left, right = a.step(action), b.step(action)
            self.assertEqual(left, right)
            if left[2] or left[3]:
                break
        # reset(None) continues the stream, and remains reproducible across environments.
        self.assertEqual(a.reset(), b.reset())
        self.assertEqual(a.reset(), b.reset())

    def test_rule_policy(self):
        for obs, action in [((2, 2, 2, 2, 1), 5), ((2, 2, 1, 0, 1), 0),
                            ((2, 2, 3, 0, 1), 1), ((2, 2, 2, 1, 1), 2),
                            ((2, 2, 2, 3, 1), 3)]:
            with self.subTest(obs=obs):
                self.assertEqual(rule_policy(obs), action)


if __name__ == "__main__":
    unittest.main()
