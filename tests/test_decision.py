import numpy as np
import pandas as pd
import pytest
from semanticavas.decision import admissible_catalog,weight_sensitivity

def test_constraints_before_ranking():
    p=pd.DataFrame({'x':[0.,1.]},index=['a','b']);m=pd.DataFrame({'class':['car','truck']},index=p.index)
    assert list(admissible_catalog(p,m,{'class':['truck']}).index)==['b']
    with pytest.raises(ValueError):admissible_catalog(p,m.iloc[:1],{'class':['truck']})

def test_ties_and_weight_scaling():
    p=pd.DataFrame({'x':[0.,1.],'y':[1.,0.]},index=['a','b'])
    a=weight_sensitivity(p,{'x':0,'y':0},{'x':1,'y':1},[[1,1],[1,0]])
    np.testing.assert_allclose(a['winner_share'],[.75,.25])
    b=weight_sensitivity(p,{'x':0,'y':0},{'x':1,'y':1},[[10,10],[10,0]])
    np.testing.assert_allclose(a['losses'],b['losses'])
    with pytest.raises(ValueError):weight_sensitivity(p,{'x':0},{'x':1},[[0]])
