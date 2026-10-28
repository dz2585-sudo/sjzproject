import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from aimrl.agent import QLearningAgent
from aimrl.env import AimEnv
from aimrl.trial25 import load_agent, read_json, trajectory, validate_config, verify_episode


class Trial25Tests(unittest.TestCase):
    def setUp(self):
        self.config = read_json(Path(__file__).resolve().parent.parent / 'configs/trial25.json')
        Path('.tmp').mkdir(exist_ok=True)

    def test_requested_configuration_and_q_size(self):
        validate_config(self.config)
        agent = QLearningAgent(self.config)
        self.assertEqual(agent.q.shape, (25, 25, 25, 25, 2, 6))
        self.assertEqual(agent.q.nbytes, 18750000)
        self.assertEqual(agent.q.dtype, np.float32)
        self.assertFalse(np.any(agent.q))

    def test_25x25_boundaries_and_target_bounce(self):
        config = copy.deepcopy(self.config)
        config['environment'].update(target_turn_probability=0, target_move_probability=1)
        env = AimEnv(config); env.reset(seed=1)
        for action, state, expected in [(0, (0, 12, 24, 1, 1), (0, 12, 23, 1, -1)),
                                        (1, (24, 12, 0, 1, -1), (24, 12, 1, 1, 1)),
                                        (2, (12, 0, 24, 1, 1), (12, 0, 23, 1, -1)),
                                        (3, (12, 24, 0, 1, -1), (12, 24, 1, 1, 1))]:
            with self.subTest(action=action):
                env.state = state
                self.assertEqual(env.step(action)[0], expected)

    def test_exact_hit_before_motion_and_step_100_priority(self):
        env = AimEnv(self.config); env.reset(seed=2)
        env.state = (24, 24, 24, 24, 1); env.step_count = 99
        state, reward, terminated, truncated, _ = env.step(5)
        self.assertEqual(state, (24, 24, 24, 24, 1))
        self.assertAlmostEqual(reward, 9.95)
        self.assertTrue(terminated); self.assertFalse(truncated)
        env.reset(seed=2); env.state = (23, 24, 24, 24, 1)
        self.assertFalse(env.step(5)[2])

    def test_cap_and_saved_trajectory_final_state(self):
        rows = trajectory(self.config, lambda state: 4, 'Stay diagnostic')
        self.assertEqual(len(rows), 101)
        self.assertEqual(rows[-1]['status'], 'Timeout')
        self.assertEqual(rows[-1]['step_count'], 100)
        self.assertFalse(rows[99]['truncated'])
        verify_episode(rows[-1], 'step_count', 'terminated')
        self.assertEqual(rows[0]['scenario_seed'], 3501000)

    def test_loader_uses_saved_25x25_configuration(self):
        with tempfile.TemporaryDirectory(dir=Path('.tmp').resolve()) as directory:
            path = Path(directory)
            (path / 'run_config.json').write_text(json.dumps(self.config), encoding='utf-8')
            # Fixture metadata is not an experimental result.
            (path / 'metadata.json').write_text(json.dumps({'reward_mode': 'base', 'root_seed': 0, 'global_steps': 5000000}), encoding='utf-8')
            original = QLearningAgent(self.config)
            original.q[24, 24, 23, 24, 1, 5] = 3
            original.save(path / 'q_table.npy')
            loaded = load_agent(path, self.config, 'base')
            np.testing.assert_array_equal(original.q, loaded.q)
            self.assertEqual(loaded.greedy_action((24, 24, 23, 24, 1)), 5)
            with self.assertRaisesRegex(ValueError, 'metadata'):
                load_agent(path, self.config, 'potential')
            np.save(path / 'q_table.npy', np.zeros((5, 5, 5, 5, 2, 6), dtype=np.float32))
            with self.assertRaisesRegex(ValueError, 'Incompatible'):
                load_agent(path, self.config, 'base')

    def test_configuration_and_record_errors_are_rejected(self):
        config = copy.deepcopy(self.config)
        config['learning']['gamma'] = .95
        with self.assertRaises(ValueError): validate_config(config)
        bad = {'hit': True, 'steps': 100, 'misses': 0, 'base_return': 5, 'terminated': True, 'truncated': True}
        with self.assertRaises(ValueError): verify_episode(bad)


if __name__ == '__main__':
    unittest.main()
