import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from aimrl.agent import QLearningAgent
from aimrl.config import load_config
from aimrl.evaluate import evaluate
from aimrl.train import epsilon_at, train


class LearningTests(unittest.TestCase):
    def setUp(self):
        Path('.tmp').mkdir(exist_ok=True)
        self.config = load_config()
        self.agent = QLearningAgent(self.config, np.random.default_rng(7))
        self.s = (0, 1, 2, 3, -1)
        self.n = (0, 1, 2, 3, 1)

    def test_direction_mapping(self):
        self.assertEqual(self.agent.state_index(self.s)[-1], 0)
        self.assertEqual(self.agent.state_index(self.n)[-1], 1)
        self.agent.q[self.agent.state_index(self.s) + (3,)] = 4
        self.assertEqual(self.agent.q[self.agent.state_index(self.n) + (3,)], 0)

    def test_action_selection_and_greedy_rng_isolation(self):
        q = self.agent.q[self.agent.state_index(self.s)]
        q[1], q[3] = 4, 4
        before = copy.deepcopy(self.agent.rng.bit_generator.state)
        self.assertEqual(self.agent.greedy_action(self.s), 1)
        self.assertEqual(self.agent.rng.bit_generator.state, before)
        chosen = {self.agent.select_action(self.s, 0) for _ in range(100)}
        self.assertEqual(chosen, {1, 3})
        q[3] = 5
        self.assertEqual(self.agent.select_action(self.s, 0), 3)

    def test_epsilon_global_boundaries(self):
        learn = self.config["learning"]
        self.assertEqual(epsilon_at(0, learn), 1)
        self.assertAlmostEqual(epsilon_at(80000, learn), 0.05)
        self.assertAlmostEqual(epsilon_at(90000, learn), 0.05)
        self.assertAlmostEqual(epsilon_at(40000, learn), 0.525)

    def test_update_and_termination_vs_truncation(self):
        for terminated, truncated, reward, expected in [(False, False, -0.05, 2.2625),
                                                         (True, False, 9.95, 3.1925),
                                                         (False, True, -0.05, 2.2625)]:
            with self.subTest(terminated=terminated, truncated=truncated):
                self.agent.q.fill(0)
                self.agent.q[self.agent.state_index(self.s) + (0,)] = 2
                self.agent.q[self.agent.state_index(self.n) + (2,)] = 100 if terminated else 4
                self.agent.update(self.s, 0, reward, self.n, terminated, truncated)
                self.assertAlmostEqual(self.agent.q[self.agent.state_index(self.s) + (0,)], expected, places=5)

    def test_save_load_and_rejections(self):
        with tempfile.TemporaryDirectory(dir=Path('.tmp').resolve()) as directory:
            path = Path(directory) / 'q.npy'
            self.agent.q[self.agent.state_index(self.s) + (3,)] = 5
            self.agent.save(path)
            other = QLearningAgent(self.config)
            other.load(path)
            np.testing.assert_array_equal(other.q, self.agent.q)
            self.assertEqual(other.greedy_action(self.s), 3)
            np.save(path, np.zeros((2, 2), dtype=np.float32))
            with self.assertRaisesRegex(ValueError, 'Incompatible'):
                other.load(path)
            np.save(path, np.full(self.agent.shape, np.nan, dtype=np.float32))
            with self.assertRaisesRegex(ValueError, 'non-finite'):
                other.load(path)
            bad = copy.deepcopy(self.config)
            bad['actions'][0:2] = reversed(bad['actions'][0:2])
            with self.assertRaisesRegex(ValueError, 'action mapping'):
                QLearningAgent(bad)

    def test_validation_freezes_q_and_rng(self):
        before = self.agent.q.copy()
        rng_before = copy.deepcopy(self.agent.rng.bit_generator.state)
        evaluate(self.config, self.agent.greedy_action, 100000, 3)
        np.testing.assert_array_equal(self.agent.q, before)
        self.assertEqual(self.agent.rng.bit_generator.state, rng_before)

    def test_short_training_reproducible_and_validation_isolated(self):
        config = copy.deepcopy(self.config)
        config['learning'].update(training_steps=21, epsilon_decay_steps=20)
        config['experiments']['validation'].update(interval_steps=5, episodes=3)
        with tempfile.TemporaryDirectory(dir=Path('.tmp').resolve()) as directory:
            root = Path(directory)
            a, meta_a = train(config, 3, root / 'a')
            b, meta_b = train(config, 3, root / 'b')
            c, meta_c = train(config, 3, root / 'c', validate_mid=False)
            np.testing.assert_array_equal(a.q, b.q)
            np.testing.assert_array_equal(a.q, c.q)
            self.assertEqual([meta_a['global_steps'], meta_b['global_steps'], meta_c['global_steps']], [21] * 3)
            self.assertEqual(meta_a['complete_episode_steps'] + meta_a['unfinished_episode']['steps'], 21)
            with (root / 'a' / 'validation.csv').open(newline='') as file:
                self.assertEqual([int(r['global_step']) for r in csv.DictReader(file)], [0, 5, 10, 15, 20])
            self.assertEqual(len(list((root / 'a' / 'checkpoints').glob('*.npy'))), 5)
            with self.assertRaises(FileExistsError):
                train(config, 3, root / 'a')

    def test_budget_end_is_not_environment_termination(self):
        config = copy.deepcopy(self.config)
        config['learning'].update(training_steps=1, epsilon_decay_steps=1)
        config['experiments']['validation'].update(interval_steps=10, episodes=2)
        with tempfile.TemporaryDirectory(dir=Path('.tmp').resolve()) as directory:
            _, meta = train(config, 2, Path(directory) / 'run')
            self.assertEqual(meta['unfinished_episode']['steps'], 1)
            self.assertEqual(meta['complete_episode_steps'], 0)
            with (Path(directory) / 'run' / 'episodes.csv').open(newline='') as file:
                self.assertEqual(list(csv.DictReader(file)), [])


if __name__ == '__main__':
    unittest.main()
