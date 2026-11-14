import copy,unittest
import numpy as np
from aimrl.tracking_action_resolution.env import ActionEnv,ACTION_NAMES,MOVES,clipped_position,mode
from aimrl.tracking_action_resolution.core import config,cache_key,methods,frozen_agent,state_id
from aimrl.tracking_action_resolution.reference import ReferencePlanner,transitions
from aimrl.tracking_action_resolution.validation import python_checks
from aimrl.tracking25.env import ACTION_NAMES as OLD_NAMES,MOVES as OLD_MOVES

class ActionResolutionTests(unittest.TestCase):
    def test_python_mechanics_and_rng(self):self.assertTrue(python_checks(config())['complete'])
    def test_schema_is_append_only(self):
        self.assertEqual(ACTION_NAMES[:10],OLD_NAMES);self.assertEqual(MOVES[:10],OLD_MOVES)
        self.assertEqual(MOVES[10:],((-2,0),(2,0),(0,-2),(0,2)))
        self.assertEqual([mode(a) for a in (0,4,5,6,10)],['slow','stay','fire','fast','medium'])
    def test_invalid_actions_fail_closed(self):
        for count in (10,14):
            e=ActionEnv(config(),count);e.reset(12)
            for a in (-1,count,True,1.1):
                with self.assertRaises(ValueError):e.step(a)
    def test_one_tick_medium(self):
        e=ActionEnv(config());e.reset(12);e.state=(10,10,20,10,0,2)
        class Draws:
            calls=0
            def random(self,n):self.calls+=1;self.n=n;return np.ones(3)*.8
        e.rng=Draws();s,*_=e.step(11)
        self.assertEqual(s,(12,10,22,10,0,2));self.assertEqual(e.elapsed,1);self.assertEqual(e.rng.calls,1);self.assertEqual(e.rng.n,3)
    def test_cache_separation(self):
        c=config();keys=[cache_key(c,n,s)[0] for n in (10,14) for s in [None,60,61]]
        self.assertEqual(len(set(keys)),6)
        for key in ('gamma','close_weight'):
            changed=copy.deepcopy(c);changed[key]+=.01;self.assertNotEqual(cache_key(c,14,60)[0],cache_key(changed,14,60)[0])
        changed=copy.deepcopy(c);changed['planning']['terminal_version']+='x';self.assertNotEqual(cache_key(c,14,60)[0],cache_key(changed,14,60)[0])
    def test_terminal_is_original_ten_max(self):
        c=config();a=frozen_agent('R-Decay',60);snap=a.snapshot();s=(12,12,14,12,0,2)
        x=ReferencePlanner(c,10,a).solve(s,0)[0];y=ReferencePlanner(c,14,a).solve(s,0)[0]
        self.assertEqual(x,y);self.assertEqual(x,float(a.q[a.index(s)].max())-2);self.assertEqual(a.snapshot(),snap);self.assertEqual(a.q.shape[-1],10)
    def test_reference_subset_and_shaping(self):
        c=config();s=(24,12,23,12,2,2);a=ReferencePlanner(c,10);b=ReferencePlanner(c,14);sh=ReferencePlanner(c,14,shaped=True)
        for h in (1,2,3):
            self.assertGreaterEqual(b.solve(s,h)[0]+1e-9,a.solve(s,h)[0]);self.assertAlmostEqual(b.solve(s,h)[0]-sh.solve(s,h)[0],-1,places=9)
    def test_twelve_instances(self):
        ms=methods(config());self.assertEqual(len(ms),12);self.assertEqual(sum(m['seed'] is None for m in ms),2)
    def test_saved_int16_state_index_is_widened(self):
        s=(24,24,23,22,3,2)
        self.assertEqual(state_id(s),state_id(np.asarray(s,np.int16)))
        self.assertGreater(state_id(s),3000000)

if __name__=='__main__':unittest.main()
