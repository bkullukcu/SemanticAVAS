"""Regression contracts for the SoftwareX 01551 review revision.

Fixtures are synthetic. They do not establish perceptual validity.
"""
import numpy as np
import pandas as pd
import pytest
from scipy.io import wavfile

from semanticavas.core import SemanticAtlas
from semanticavas.empirical import rank_existing_sounds
from semanticavas.prediction import extract_wav, prepare_training, train_wav_predictor, save_predictor, load_predictor

def test_single_ranking_loss_matches_public_apis():
    frame=pd.DataFrame({'participant':['a','a','b','b'],
      'stimulus':['S01','S02','S01','S02'],
      'truck_like':[65,35,70,25], 'annoying':[15,20,20,25]})
    atlas=SemanticAtlas(frame,['truck_like','annoying'])
    target={'truck_like':70,'annoying':20}
    ranked=atlas.rank(target,weights={'truck_like':2.})
    direct=rank_existing_sounds(atlas.means,target,weights={'truck_like':2.})
    np.testing.assert_allclose(ranked.target_loss.to_numpy(),direct.target_loss.to_numpy())
    assert ranked.index.tolist()==direct.stimulus.tolist()

def test_full_recording_option_uses_later_audio(tmp_path):
    fs=16000
    first=.1*np.sin(2*np.pi*200*np.arange(fs*5)/fs)
    later=.1*np.sin(2*np.pi*800*np.arange(fs*3)/fs)
    path=tmp_path/'test.wav'
    wavfile.write(path,fs,np.r_[first,later].astype('float32'))
    f,fmeta=extract_wav(path,max_seconds=5)
    a,ameta=extract_wav(path,max_seconds=None)
    assert fmeta['duration_s']==5 and ameta['duration_s']==8
    assert a[1]>f[1]+50

def test_scale_bounds_missing_policy_and_archive(tmp_path):
    fs=16000; audio=tmp_path/'audio'; audio.mkdir()
    m=[];ratings=[]
    for i in range(10):
        name=f'S{i:02d}'
        wave=.1*np.sin(2*np.pi*(200+i*20)*np.arange(fs)/fs)
        path=audio/f'{name}.wav'
        wavfile.write(path,fs,wave.astype('float32'))
        m.append({'stimulus':name,'wav':str(path)})
        for participant in ['a','b','c']:
            ratings.append({'stimulus':name,'participant':participant,'pleasant':1.+(i%5)/2.})
    mp=tmp_path/'manifest.csv';rp=tmp_path/'ratings.csv'
    pd.DataFrame(m).to_csv(mp,index=False)
    df=pd.DataFrame(ratings);df.loc[0,'pleasant']=np.nan
    df.to_csv(rp,index=False)
    with pytest.raises(ValueError,match='missing ratings'):
        prepare_training(mp,rp,['pleasant'],scale_bounds={'pleasant':[1,5]})
    model,_,_=train_wav_predictor(mp,rp,['pleasant'],
        scale_bounds={'pleasant':[1,5]},missing_policy='stimulus_mean',max_seconds=None)
    save_predictor(model,tmp_path/'model.npz')
    loaded=load_predictor(tmp_path/'model.npz')
    assert loaded.max_seconds is None
    assert loaded.scale_bounds=={'pleasant':(1,5)}
    assert 1<=loaded.predict_wav(audio/'S00.wav')['pleasant']<=5
