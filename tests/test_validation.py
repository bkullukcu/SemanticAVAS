import unittest
import numpy as np
import pandas as pd
from semanticavas import SemanticAtlas, RidgeSurrogate

class SensitivityTests(unittest.TestCase):
    def test_joint_hull_rejects_box_inside_hull_outside(self):
        x=np.array([[0.,0.],[1.,0.],[0.,1.]])
        y=np.arange(3.)[:,None]
        model=RidgeSurrogate().fit(x,y)
        self.assertEqual(model.predict([[.7,.7]]).shape,(1,1))
        with self.assertRaisesRegex(ValueError,"joint"):
            model.predict([[.7,.7]],support="convex_hull")
        self.assertEqual(model.predict([[.2,.2]],support="convex_hull").shape,(1,1))
    def test_rank_bootstrap_reproducible(self):
        data=pd.DataFrame([{"participant":p,"stimulus":s,"truck":v}
            for p in range(1,6) for s,v in [("a",3),("b",1)]])
        atlas=SemanticAtlas(data,["truck"])
        x=atlas.bootstrap_target_choice({"truck":3},n_boot=55,seed=10)
        self.assertEqual(x["a"],1.)
        self.assertEqual(x["b"],0.)
    def test_incomplete_listener_panel_rejected(self):
        data=pd.DataFrame([{"participant":p,"stimulus":s,"truck":v}
            for p in range(1,4) for s,v in [("a",3),("b",1)] if not(p==3 and s=="a")])
        with self.assertRaisesRegex(ValueError,"complete"):
            SemanticAtlas(data,["truck"]).bootstrap_target_choice({"truck":3})
    def test_invalid_support_rejected(self):
        m=RidgeSurrogate().fit(np.array([[0],[1],[2.]]),np.array([[1],[2],[3.]]))
        with self.assertRaises(ValueError):m.predict([[1]],support="guess")
