import sys,types
import numpy as np,pandas as pd, pytest
from semanticavas.interop import calibrated_pressure,mosqito_stationary_loudness,to_pyfar_signal
from semanticavas.validation import match_synthesis_ratings,score_family_holdout

def test_calibration_required_and_units():
    x=np.array([0.5,-.5])
    assert np.allclose(calibrated_pressure(x,pa_per_full_scale=0.1),[.05,-.05])
    with pytest.raises(ValueError):mosqito_stationary_loudness(x,48000)
    with pytest.raises(ValueError):calibrated_pressure(x,pa_per_full_scale=0)

def test_optional_adapters_mocked(monkeypatch):
    fake=types.SimpleNamespace(Signal=lambda s,fs:(s,fs))
    monkeypatch.setitem(sys.modules,'pyfar',fake)
    y=to_pyfar_signal([0.,1.],48000)
    assert y[1]==48000
    sq=types.SimpleNamespace(loudness_zwst=lambda arr,fs,field_type:(np.array([1.2]),None,None),
                             loudness_ecma=lambda arr,fs:(np.array([1.5]),None,None,None,None))
    monkeypatch.setitem(sys.modules,'mosqito',types.SimpleNamespace(sq_metrics=sq))
    assert mosqito_stationary_loudness(np.zeros(128),48000,pa_per_full_scale=.1)['units']=='sone'
    assert mosqito_stationary_loudness(np.zeros(128),48000,pa_per_full_scale=.1,model='ecma')['units']=='sone_HMS'
    with pytest.raises(ValueError):mosqito_stationary_loudness(np.zeros(128),44100,pa_per_full_scale=.1,model='ecma')

def test_join_rejects_mismatched_and_cv_runs():
    p=pd.DataFrame({'stimulus':[f's{i}' for i in range(9)],'sound_family':[f'f{i//3}' for i in range(9)],
                    'f0':np.linspace(40,140,9),'cutoff':np.linspace(1200,5200,9)})
    r=pd.DataFrame({'stimulus':[f's{i}' for i in range(9) for _ in range(3)],
                    'participant':[j for i in range(9) for j in range(3)],
                    'rating':[i*2+j for i in range(9) for j in range(3)]})
    tab=match_synthesis_ratings(p,r,parameter_columns=['f0','cutoff'],rating_columns=['rating'])
    result=score_family_holdout(tab,parameter_columns=['f0','cutoff'],rating_columns=['rating'])
    assert len(result)==9 and result.model_mae.notna().all()
    with pytest.raises(ValueError):match_synthesis_ratings(p.iloc[:-1],r,parameter_columns=['f0'],rating_columns=['rating'])
    with pytest.raises(ValueError):match_synthesis_ratings(pd.concat([p,p.iloc[[0]]]),r,parameter_columns=['f0'],rating_columns=['rating'])
