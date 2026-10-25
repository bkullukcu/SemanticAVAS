import numpy as np
import pytest
from semanticavas.synthesis35 import _decode_trial_matrix, published_design_table, SCALES

class FakeParticipant:
    def __init__(self,order,results):
        self.expOrder=np.asarray(order)
        self.results=np.asarray(results,dtype=float)

def test_decode_trial_rows_using_randomized_stimulus_order():
    # trial i -> original sound no. order[i]. Ensure no accidental direct row use.
    order=list(range(35,0,-1))
    trial=np.column_stack([np.array(order,dtype=float)+k for k in (0,10,20)])
    f=FakeParticipant(order,trial)
    decoded=_decode_trial_matrix(f,3)
    assert np.allclose(decoded[:,0],np.arange(1,36))
    assert np.allclose(decoded[:,2],np.arange(21,56))

def test_missing_rating_preserved():
    order=list(range(35,0,-1))
    a=np.ones((35,3))*50;a[6,1]=np.nan
    d=_decode_trial_matrix(FakeParticipant(order,a),3)
    assert np.isnan(d[order[6]-1,1])
    assert np.isfinite(d).sum()==104

def test_invalid_stimulus_order_rejected():
    order=list(range(1,36));order[2]=order[1]
    with pytest.raises(ValueError):
        _decode_trial_matrix(FakeParticipant(order,np.ones((35,3))),3)

def test_out_of_range_rejected():
    v=np.ones((35,3))*50;v[10,0]=105
    with pytest.raises(ValueError):
        _decode_trial_matrix(FakeParticipant(range(1,36),v),3)

def test_published_design_does_not_impute_unknown_candidates():
    d=published_design_table()
    assert len(d)==18 and 'S18' in d.stimulus.tolist()
    assert 'S19' not in d.stimulus.tolist()
    assert len(SCALES)==7
