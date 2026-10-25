import unittest
import numpy as np
import pandas as pd
from semanticavas import *

class TestSemantic(unittest.TestCase):
 def setUp(self):self.frame=pd.DataFrame({'participant':[1,1,2,2], 'stimulus':['a','b','a','b'],'truck':[10,80,20,90],'annoying':[5,60,15,70]})
 def test_participant_balanced_repeats(self):
  f=pd.concat([self.frame,self.frame.iloc[[0]]]);a=SemanticAtlas(f,['truck']);self.assertEqual(a.means.loc['a','truck'],15)
 def test_rank_target(self):self.assertEqual(SemanticAtlas(self.frame,['truck']).rank({'truck':90}).index[0],'b')
 def test_bootstrap_reproducibility(self):
  a=SemanticAtlas(self.frame,['truck']);lo,hi=a.intervals(30);l2,h2=a.intervals(30);np.testing.assert_equal(lo.values,l2.values);self.assertTrue((lo<=hi).all().all())
 def test_incomplete_bootstrap(self):
  with self.assertRaises(ValueError):SemanticAtlas(self.frame.iloc[:-1],['truck']).intervals(10)
 def test_pareto(self):np.testing.assert_array_equal(pareto_front([[1,2],[2,1],[3,3],[1,2]]),[True,True,False,True])
 def test_surrogate_constant(self):
  x=np.arange(10.)[:,None];m=RidgeSurrogate().fit(x,np.ones(10)*4);np.testing.assert_allclose(m.predict(x),4,atol=1e-9)
 def test_extrapolation(self):
  m=RidgeSurrogate().fit(np.arange(5.)[:,None],np.arange(5.))
  with self.assertRaises(ValueError):m.predict([[20]])
 def test_synthesis_rms(self):
  y=synthesize();self.assertLess(max(abs(y)),1);self.assertAlmostEqual(20*np.log10(np.sqrt(np.mean(y*y))),-24,places=4)
 def test_nyquist_rejected(self):
  with self.assertRaises(ValueError):synthesize(cutoff=30000)
 def test_groups(self):
  for a,b in group_holdout(['a','a','b','b']):self.assertFalse(set(a)&set(b))
 def test_bad_target(self):
  with self.assertRaises(ValueError):SemanticAtlas(self.frame,['truck']).rank({'truck':np.nan})
if __name__=='__main__':unittest.main()
