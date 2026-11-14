"""Independent study gates remain active under python -O."""
import copy
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from aimrl.tracking25.env import TrackingEnv
from aimrl.tracking_boundary_stability.core import Agent, Encoder, Kernel, config, load_model, require, save_model
from aimrl.tracking_boundary_stability.diagnostics import expected_stay, mechanisms, parity


class BoundaryTests(unittest.TestCase):
    def test_compact_rows_and_alias(self):
        r,b=Encoder('R'),Encoder('B')
        self.assertEqual((r.rows,b.rows),(19208,1089288))
        s,t=(10,10,11,10,0,2),(23,10,24,10,0,2)
        self.assertEqual(r.index(s),r.index(t));self.assertNotEqual(b.index(s),b.index(t))
        with self.assertRaises(ValueError):r.index((25,0,0,0,0,1))

    def test_learning_rate_uses_preupdate_count_and_bootstraps(self):
        c=config();a=Agent(c,'R-Decay');s=(10,10,11,10,0,2);n=(10,10,13,10,0,2)
        a.counts[a.index(s),4]=1000;a.q[a.index(n),0]=10
        alpha,target,before,after,count,row=a.update(s,4,-1,n)
        self.assertAlmostEqual(alpha,.15*2**(-.6),15)
        self.assertEqual(count,1000);self.assertEqual(target,8.9)
        self.assertEqual(int(a.counts[row,4]),1001)
        self.assertEqual(after,float(np.float32(np.float32(alpha*8.9))))

    def test_mechanism_and_nonassert_require(self):
        with tempfile.TemporaryDirectory() as d:mechanisms(config(),Path(d))
        with self.assertRaises(ValueError):require(False,'active under optimization')

    def test_exact_expected_target_has_unit_fire_gap(self):
        c=config();a=Agent(c,'R-Fixed');s=np.array([[10,10,11,10,0,2],[23,10,24,10,0,2]])
        self.assertTrue(np.isfinite(expected_stay(c,a,s)).all())
        # Independently enumerate the original environment with deterministic turn/toggle cases.
        from aimrl.tracking25.env import advance_target
        expected=[]
        for state in s:
            result=0
            for turn,p in ((0,.95),(1,.025),(-1,.025)):
                for toggle,p2 in ((0,.95),(1,.05)):
                    x,y,h,b=advance_target(int(state[2]),int(state[3]),(int(state[4])+turn)%4,
                                          3-int(state[5]) if toggle else int(state[5]),25)
                    d=abs(int(state[0])-x)+abs(int(state[1])-y)
                    result+=p*p2*((-.05+.5*(d<=1))+.99*-d+1)
            expected.append(result)
        np.testing.assert_allclose(expected_stay(c,a,s),expected,rtol=0,atol=1e-14)

    @unittest.skipUnless(os.environ.get('AIMRL_BOUNDARY_KERNEL'),'Native library not supplied')
    def test_native_stepwise_full_gate(self):
        c=config();kernel=Kernel(os.environ['AIMRL_BOUNDARY_KERNEL'],c)
        with tempfile.TemporaryDirectory() as d:parity(c,kernel,Path(d))

    @unittest.skipUnless(os.environ.get('AIMRL_BOUNDARY_KERNEL') and os.environ.get('AIMRL_TRADEOFF_KERNEL'),'Both native libraries required')
    def test_fixed_relative_exactly_matches_published_close_0p5(self):
        from aimrl.tracking_research.agent import ResearchAgent
        from aimrl.tracking_tradeoff.core import Kernel as PublishedKernel
        c=config();new=Kernel(os.environ['AIMRL_BOUNDARY_KERNEL'],c);old=PublishedKernel(os.environ['AIMRL_TRADEOFF_KERNEL'])
        for offset in (0,3999800):
            stream=np.random.SeedSequence(1010).spawn(2)
            a=Agent(c,'R-Fixed',np.random.default_rng(stream[1]));b=ResearchAgent(c,'relative',np.random.default_rng(stream[1]))
            e0,e1=TrackingEnv(c),TrackingEnv(c);s0,_=e0.reset(1414);s1,_=e1.reset(1414)
            n0=new.run(e0,s0,300,a,offset,train=True);n1=old.run(e1,s1,300,b,offset,(0,.5),train=True)
            self.assertTrue(np.array_equal(n0,n1));self.assertTrue(np.array_equal(a.q,b.q.reshape(-1,10)))
            self.assertTrue(np.array_equal(a.counts,b.counts.reshape(-1,10)))
            self.assertEqual(a.rng.bit_generator.state,b.rng.bit_generator.state)

    def test_zero_shot_accuracy_is_na(self):
        from aimrl.tracking_tradeoff.core import summarize,window_metrics
        env=TrackingEnv(config());s,_=env.reset(101)
        ticks,_=__import__('aimrl.tracking_boundary_stability.core',fromlist=['reference_run']).reference_run(
            env,s,300,Agent(config(),'R-Fixed'),train=False,forced=np.full(300,4),c=config())
        self.assertIsNone(summarize([window_metrics(s,ticks)])['A'])


if __name__=='__main__':unittest.main()
