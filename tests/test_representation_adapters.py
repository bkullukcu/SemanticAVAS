"""Synthetic input-contract tests for E3 and E4 feature adapters."""
from hashlib import sha256
import numpy as np
import pytest
from scipy.io import wavfile
from semanticavas.representation_adapters import (
    calibrated_mosqito_features,fixed_clap_embeddings,
    checked_embedding_matrix,nested_representation_test,E3_NAMES)


def test_calibration_must_be_known_before_signal_processing(tmp_path):
    with pytest.raises(ValueError,match='pa_per_fullscale'):
        calibrated_mosqito_features(tmp_path/'missing.wav',pa_per_fullscale=None)


def test_physical_pressure_and_resampling_with_stub(tmp_path):
    fs=16000
    t=np.arange(fs)/fs
    wavfile.write(tmp_path/'tone.wav',fs,(.1*np.sin(2*np.pi*440*t)).astype('float32'))
    observed=[]
    def loud(x,sr):
        observed.append((len(x),sr))
        return 5.,None,None
    adapters=(loud,lambda x,sr:1.2,lambda x,sr:(np.array([.2,.4]),None,None,None),
              lambda x,sr:(8.,None,None,None))
    f=calibrated_mosqito_features(tmp_path/'tone.wav',pa_per_fullscale=.3,metrics=adapters)
    assert set(f)==set(E3_NAMES) and f['roughness_asper']==pytest.approx(.3)
    assert observed==[(48000,48000)]


class _ClapStub:
    def get_audio_embedding_from_filelist(self,x,use_tensor=False):
        return np.ones((len(x),12))


def test_clap_checkpoint_and_wav_integrity(tmp_path):
    p=tmp_path/'test.wav'
    wavfile.write(p,16000,np.zeros(16000,dtype=np.float32))
    weights=tmp_path/'weights.pt'
    weights.write_bytes(b'fixture')
    frame,meta=fixed_clap_embeddings([p],['s1'],checkpoint_path=weights,model=_ClapStub())
    h=sha256(p.read_bytes()).hexdigest()
    assert checked_embedding_matrix(frame,['s1'],[h]).shape==(1,12)
    assert len(meta['checkpoint_sha256'])==64
    with pytest.raises(ValueError,match='checksum'):
        checked_embedding_matrix(frame,['s1'],['0'*64])


def test_train_only_pca_on_held_out_sources():
    rng=np.random.default_rng(8)
    y=40+rng.normal(size=(12,2))
    grp=np.repeat(['a','b','c','d'],3)
    records,summary=nested_representation_test(
        {'E0':rng.normal(size=(12,10)),'E4':rng.normal(size=(12,24))},
        y,grp,alphas=(.1,1.),pca_components=(2,))
    assert len(records)==24 and len(summary)==2
    assert set(summary.representation)=={'E0','E4'}
    assert np.isfinite(summary.delta_mae).all()
