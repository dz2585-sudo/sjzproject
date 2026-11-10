"""Focused inference-only firing-guard tests."""
import unittest

from aimrl.tracking_fire_guard.history import PRIOR
from aimrl.tracking_fire_guard.replay import FIRE, STAY, FiringGuard, replay_pair
from aimrl.tracking_research.io import read_json


class FiringGuardTests(unittest.TestCase):
    def test_calls_policy_once_and_only_changes_unaligned_fire(self):
        aligned=(2,3,2,3,0,1)
        unaligned=(2,3,2,4,0,1)
        for state in (aligned,unaligned):
            for action in range(10):
                with self.subTest(state=state, action=action):
                    calls=[]
                    def original(observation):
                        calls.append(observation)
                        return action
                    guard=FiringGuard(original)
                    self.assertEqual(guard(state),STAY if state==unaligned and action==FIRE else action)
                    self.assertEqual(guard.proposed_action,action)
                    self.assertEqual(calls,[state])

    def test_zero_shots_have_no_accuracy(self):
        config=read_json(PRIOR/'protocol.json')['config']
        *_, a,b,events,trace=replay_pair(config,4200000,lambda _:STAY,strict_rng=True)
        self.assertEqual(a,b)
        self.assertIsNone(a['accuracy'])
        self.assertEqual(a['shots'],0)
        self.assertEqual(events,[])
        self.assertEqual(trace,[])

    def test_fire_miss_replacement_matches_actual_rewards(self):
        config=read_json(PRIOR/'protocol.json')['config']
        _,_,_,a,b,_,trace=replay_pair(config,4200000,lambda _:FIRE,strict_rng=True,keep_trace=True)
        removed=[t for t in trace if t['filtered']]
        self.assertGreater(len(removed),0)
        self.assertEqual(len(removed),a['misses'])
        self.assertEqual(b['misses'],0)
        self.assertEqual(b['shots'],b['hits'])
        self.assertAlmostEqual(b['base_return']-a['base_return'],len(removed))
        self.assertTrue(all(t['proposed_action']=='fire' and t['guard_executed_action']=='stay'
                            and t['original_miss']==1 and t['guard_miss']==0 for t in removed))
