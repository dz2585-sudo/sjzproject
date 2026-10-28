"""Research checks for aggregation limits, matched learning, and honest diagnostics."""
import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from aimrl.tracking25.agent import TrackingAgent, learning_reward, epsilon
from aimrl.tracking25.env import TrackingEnv
from aimrl.tracking_research.agent import ResearchAgent, coverage
from aimrl.tracking_research.evaluate import contexts, distance_p95, evaluate, longest_far, slice_summary, FIELDS
from aimrl.tracking_research.io import ROOT, read_json


class ResearchTests(unittest.TestCase):
    def setUp(self):
        self.config = read_json(ROOT / 'configs/tracking_research.json')

    def test_relative_index_sign_range_shape_dtype(self):
        a = ResearchAgent(self.config, 'relative')
        self.assertEqual(a.shape, (49,49,4,2,10))
        self.assertEqual(a.q.nbytes, 768320)
        self.assertEqual(a.q.dtype, np.float32)
        self.assertEqual(a.counts.dtype, np.uint32)
        self.assertEqual(a.index((24,24,0,0,3,2)), (0,0,3,1))
        self.assertEqual(a.index((0,0,24,24,0,1)), (48,48,0,0))
        self.assertEqual(a.index((10,10,9,12,1,1)), (23,26,1,0))
        for state in ((25,0,0,0,0,1), (0,0,0,0,0,0), (True,0,0,0,0,1), (0.,0,0,0,0,1)):
            with self.assertRaises(ValueError): a.index(state)

    def test_boundary_clipping_alias(self):
        config = copy.deepcopy(self.config)
        config['environment'].update(turn_probability=0, speed_toggle_probability=0)
        env = TrackingEnv(config); agent = ResearchAgent(config, 'relative')
        first, second = (0,10,4,10,0,1), (10,10,14,10,0,1)
        self.assertEqual(agent.index(first), agent.index(second))
        outputs = []
        for state in (first,second):
            env.reset(1); env.state = state
            outputs.append(agent.index(env.step(6)[0]))
        self.assertNotEqual(*outputs)
        self.assertEqual((outputs[0][0],outputs[1][0]), (29,32))

    def test_boundary_reflection_alias(self):
        config = copy.deepcopy(self.config)
        config['environment'].update(turn_probability=0, speed_toggle_probability=0)
        env = TrackingEnv(config); agent = ResearchAgent(config, 'relative')
        first, second = (18,12,24,12,0,2), (8,12,14,12,0,2)
        self.assertEqual(agent.index(first), agent.index(second))
        outputs = []
        for state in (first,second):
            env.reset(1); env.state = state
            outputs.append(agent.index(env.step(4)[0]))
        self.assertNotEqual(*outputs)
        self.assertEqual((outputs[0][0],outputs[1][0]), (28,32))

    def test_absolute_fixture_matches_v2_actions_updates_rng(self):
        sequence = np.random.SeedSequence(10).spawn(2)
        seed = int(sequence[0].generate_state(1,dtype=np.uint64)[0])
        left = TrackingAgent(self.config, np.random.default_rng(sequence[1]))
        right = ResearchAgent(self.config, 'absolute', np.random.default_rng(sequence[1]))
        envs = [TrackingEnv(self.config),TrackingEnv(self.config)]
        states = [e.reset(seed)[0] for e in envs]
        for t in range(320):
            actions = [a.select(s, epsilon(t,self.config['learning'])) for a,s in zip((left,right),states)]
            self.assertEqual(*actions)
            transitions = [e.step(a) for e,a in zip(envs,actions)]
            self.assertEqual(*transitions)
            for i,a in enumerate((left,right)):
                nxt,reward,_,truncated,_ = transitions[i]
                learned,_ = learning_reward(reward,states[i],nxt,'potential',a.gamma)
                a.update(states[i],actions[i],learned,nxt)
                states[i] = envs[i].reset()[0] if truncated else nxt
        self.assertTrue(np.array_equal(left.q,right.q))
        self.assertEqual(left.rng.bit_generator.state,right.rng.bit_generator.state)
        self.assertEqual(coverage(right)['update_sum'],320)

    def test_hits_truncation_and_partial_keep_actual_next_bootstrap(self):
        a = ResearchAgent(self.config,'relative')
        env = TrackingEnv(self.config); env.reset(4)
        env.turn_probability = env.toggle_probability = 0
        env.state = (10,10,10,10,0,1); env.elapsed = 299
        state = env.state
        nxt,reward,terminated,truncated,info = env.step(5)
        self.assertTrue(info['hit']); self.assertFalse(terminated); self.assertTrue(truncated)
        a.q[a.index(nxt)] = 2
        learned,shaping = learning_reward(reward,state,nxt,'potential',a.gamma)
        self.assertAlmostEqual(shaping,-.99)
        a.update(state,5,learned,nxt)
        self.assertAlmostEqual(float(a.q[a.index(state)+(5,)]), .15*(9.95-.99+.99*2), places=6)
        # A budget stop receives exactly the same continuing update, with no reset bootstrap.
        b = ResearchAgent(self.config,'relative'); b.q[b.index(nxt)] = 2
        b.update(state,5,learned,nxt)
        self.assertTrue(np.array_equal(a.q,b.q))
        self.assertEqual(int(a.counts.sum()),1)

    def test_model_provenance_reload_and_rejection(self):
        a = ResearchAgent(self.config,'relative')
        a.q[0,0,0,0,2] = 1.25; a.counts[0,0,0,0,2] = 3
        with tempfile.TemporaryDirectory(dir=ROOT / '.tmp') as d:
            p = Path(d)/'q_table.npz'; a.save_package(p)
            b = ResearchAgent(self.config,'relative'); b.load_package(p,True)
            self.assertTrue(np.array_equal(a.q,b.q)); self.assertTrue(np.array_equal(a.counts,b.counts))
            with self.assertRaises(ValueError): ResearchAgent(self.config,'absolute',count_updates=False).load_package(p)
            np.savez_compressed(p,q_table=a.q,representation=np.array('relative'),indexing_version=np.array('unverified'))
            with self.assertRaises(ValueError): b.load_package(p)

    def test_evaluation_isolation_pairing_and_empty_diagnostics(self):
        a = ResearchAgent(self.config,'relative',np.random.default_rng(20))
        before = a.snapshot()
        summary,rows,_ = evaluate(self.config,'relative_base',[3700000],a)
        rule,rules,_ = evaluate(self.config,'rule',[3700000])
        self.assertEqual(before,a.snapshot())
        self.assertEqual(rows[0]['target_trajectory_sha256'],rules[0]['target_trajectory_sha256'])
        all_ticks = summary['slices']['all']
        self.assertEqual(all_ticks['all_zero_q_rows'],300)
        self.assertEqual(all_ticks['exact_tied_maxima'],300)
        self.assertEqual(all_ticks['zero_visit_rows'],300)
        self.assertEqual(all_ticks['unvisited_greedy_actions'],300)
        self.assertIsNone(summary['pooled_shot_accuracy'])
        self.assertGreater(all_ticks['zero_displacement_moves'],0)
        for left,right in [('boundary_near','interior'),('target_near','target_interior'),('crosshair_near','crosshair_interior'),('speed_1','speed_2')]:
            self.assertEqual(sum(summary['slices'].get(k,{}).get('ticks',0) for k in (left,right)),300)
        self.assertEqual(rows[0]['first_acquisition_tick'],300 if not rows[0]['acquired'] else rows[0]['first_acquisition_tick'])
        self.assertAlmostEqual(rows[0]['base_return'],-15)

    def test_synthetic_histogram_streak_and_denominators(self):
        distances = [6,7,1,8,8,8,0,2,2,2]
        hist = [distances.count(i) for i in range(49)]
        self.assertEqual(distance_p95(hist),8)
        self.assertEqual(longest_far(distances),3)
        self.assertEqual(longest_far([9]*300),300)
        self.assertIsNone(distance_p95([0]*49))
        empty = slice_summary([0]*len(FIELDS),[0]*49)
        self.assertIsNone(empty['clipped_move_fraction'])
        self.assertIsNone(empty['mean_distance'])
        self.assertIn('boundary_near',contexts((3,12,10,10,0,1)))
        self.assertIn('interior',contexts((4,12,10,10,0,1)))


if __name__ == '__main__': unittest.main()
