"""Mathematical and native gates for the independent zero-training planning study."""
import copy
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from aimrl.tracking_planning.core import config,state_id,target_id,target_state,methods,frozen_agent,Policy
from aimrl.tracking_planning.reference import ReferencePlanner,phi,target_outcomes,transitions
from aimrl.tracking_planning.validation import brute,mechanism_checks


class PlanningTests(unittest.TestCase):
    def test_state_index_and_method_replication_units(self):
        self.assertEqual(state_id((0,0,0,0,0,1)),0)
        self.assertEqual(state_id((24,24,24,24,3,2)),3124999)
        for i in range(5000):self.assertEqual(target_id(target_state(i)),i)
        specs=methods(config());self.assertEqual(len(specs),20)
        self.assertEqual(sum(s['seed'] is None for s in specs),5)
        self.assertEqual(sum(s['family']=='Hybrid-H3' for s in specs),5)

    def test_original_transition_mechanics_exhaustive_probabilities(self):
        self.assertEqual(mechanism_checks(config())['target_states'],5000)

    def test_h1_manual_enumeration_and_unit_empty_fire_gap(self):
        c=config();s=(23,10,24,10,0,2);ref=ReferencePlanner(c)
        manual=[sum(p*(r+c['gamma']*phi(n)) for p,n,b,r in transitions(c,s,a)) for a in range(10)]
        np.testing.assert_allclose(ref.solve(s,1)[2],manual,rtol=0,atol=1e-12)
        self.assertAlmostEqual(manual[4]-manual[5],1,12)

    def test_feedback_h2_h3_exhaustive_tree(self):
        c=config();s=(12,10,10,10,0,2)
        for h in (2,3):self.assertAlmostEqual(ReferencePlanner(c).solve(s,h)[0],brute(c,s,h,'Phi'),10)

    def test_shaping_conversion_and_guard_dominance(self):
        c=config();s=(23,10,24,10,0,2)
        for terminal,agent in (('Phi',None),('Q+Phi',frozen_agent('R-Decay',60))):
            for h in (1,2,3):
                ref=ReferencePlanner(c,terminal,agent);shaped=ReferencePlanner(c,terminal,agent,shaped=True)
                v,a,q=ref.solve(s,h);sv,sa,sq=shaped.solve(s,h)
                np.testing.assert_allclose(np.array(q)-np.array(sq),phi(s),rtol=0,atol=1e-10)
                self.assertEqual(a,sa);self.assertNotEqual(a,5)
        self.assertEqual(ReferencePlanner(c).solve(s,3)[1],ReferencePlanner(c,prune_fire=True).solve(s,3)[1])

    def test_stress_does_not_change_planning_parameters(self):
        c=config();cc=copy.deepcopy(c);cc['environment'].update(turn_probability=.1,speed_toggle_probability=.1)
        s=(10,10,10,10,0,2)
        self.assertEqual(ReferencePlanner(c).solve(s,3),ReferencePlanner(cc).solve(s,3))

    @unittest.skipUnless(os.environ.get('AIMRL_PLANNING_NATIVE'),'Set native test gate')
    def test_native_root_value_and_actions(self):
        c=config()
        for spec in methods(c):
            if spec['kind']!='plan':continue
            p=Policy(c,spec,True);a=frozen_agent('R-Decay',spec['seed']) if spec['seed'] is not None else None
            s=(23,10,24,10,0,2);ref=ReferencePlanner(c,spec['terminal'],a)
            np.testing.assert_allclose(p.q_values(s),ref.solve(s,spec['horizon'])[2],rtol=0,atol=1e-9)
            self.assertEqual(p.propose(s),ref.act(s,spec['horizon']))


if __name__=='__main__':unittest.main()
