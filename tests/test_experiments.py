import copy
import unittest

import numpy as np

from aimrl.agent import QLearningAgent
from aimrl.config import load_config
from aimrl.experiments import (aggregate_episodes, evaluate_policy, evaluation_config,
                               mean_std, require_keys, threshold, verify_episodes)


class ExperimentTests(unittest.TestCase):
    def test_sample_standard_deviation(self):
        mean, std = mean_std([.8, .9, 1.0])
        self.assertAlmostEqual(mean, .9)
        self.assertAlmostEqual(std, .1)
        with self.assertRaises(ValueError): mean_std([.8, .9])

    def test_consecutive_threshold(self):
        def rows(values): return [{'global_step': i * 10000, 'success_rate': value} for i, value in enumerate(values)]
        self.assertEqual(threshold(rows([.2, .9, .91])), (True, 10000, 20000))
        self.assertEqual(threshold(rows([.9, .5, .9, .91])), (True, 20000, 30000))
        self.assertEqual(threshold(rows([.1, .3, .9])), (False, None, None))
        self.assertEqual(threshold(rows([.1, .2, .3])), (False, None, None))

    def test_missing_duplicate_and_unexpected_keys(self):
        expected = {('0', '1'), ('1', '1')}
        require_keys([{'seed': 0, 'scenario': 1}, {'seed': 1, 'scenario': 1}], ('seed', 'scenario'), expected)
        for rows in ([{'seed': 0, 'scenario': 1}], [{'seed': 0, 'scenario': 1}] * 2,
                     [{'seed': 0, 'scenario': 1}, {'seed': 2, 'scenario': 1}]):
            with self.assertRaises(ValueError): require_keys(rows, ('seed', 'scenario'), expected)

    def test_evaluation_override_and_frozen_agent(self):
        config = load_config(); original = copy.deepcopy(config)
        changed = evaluation_config(config, .5)
        self.assertEqual(changed['environment']['target_move_probability'], .5)
        self.assertEqual(changed['environment']['target_turn_probability'], .1)
        self.assertEqual(config, original)
        agent = QLearningAgent(config); before = agent.q.copy()
        rows = evaluate_policy(config, 'base', 0, .5, agent, 300000, 5)
        verify_episodes(rows)
        np.testing.assert_array_equal(before, agent.q)
        self.assertEqual(config, original)

    def test_baselines_have_no_training_seed(self):
        config = load_config()
        for mode in ('random', 'rule'):
            rows = evaluate_policy(config, mode, None, .25, None, 300000, 5)
            verify_episodes(rows)
            self.assertTrue(all(r['training_seed'] == '' and r['model_path'] == '' for r in rows))
            self.assertIn('success_rate', aggregate_episodes(rows))
        with self.assertRaises(ValueError):
            verify_episodes([{'success': 1, 'steps': 2, 'misses': 0, 'terminated': 0, 'truncated': 1, 'base_return': 9.9}])


if __name__ == '__main__': unittest.main()
