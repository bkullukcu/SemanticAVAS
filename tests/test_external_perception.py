import numpy as np
import pandas as pd
import pytest
from zipfile import ZipFile
from io import BytesIO
from scipy.io import wavfile
from semanticavas.external_perception import (load_chalmers_localization,localization_descriptives,load_araus_responses,
 araus_leave_fold_out_baseline,araus_foldwise_ridge,inspect_evat_wav_archive)

def test_chalmers_valid(tmp_path):
    p=tmp_path/'processedData.xlsx';pd.DataFrame({'participant':['a','b'],'le_mn_m':[12,10],'lt_mn_m':[3,4],'fl_ccc':[0,25]}).to_excel(p,index=False)
    data=load_chalmers_localization(p)
    assert len(data)==6
    assert set(data.endpoint)=={'localization_error_deg','localization_time_s','failure_rate_percent'}
    assert len(localization_descriptives(data))==3

def test_chalmers_missing_endpoint_rejected(tmp_path):
    p=tmp_path/'processedData.xlsx';pd.DataFrame({'score':[1]}).to_excel(p,index=False)
    with pytest.raises(ValueError):load_chalmers_localization(p)

def _araus_csv(tmp_path):
    p=tmp_path/'responses.csv'; rows=[]
    for i in range(20):
        rows.append({'participant':f'p{i}','fold_r':i%4,'soundscape':'s1','masker':'m1',
        'pleasant':1+(i%5),'annoying':1+(i%5),'is_attention':int(i==0),
        'Leq_L_r':45+i,'Leq_R_r':48+i,'Navg_r':2+i/5})
    pd.DataFrame(rows).to_csv(p,index=False);return p

def test_araus_fold_separation(tmp_path):
    data=load_araus_responses(_araus_csv(tmp_path))
    assert len(data)==19 and data['domain'].eq('urban_soundscape_not_individual_AVAS').all()
    assert len(araus_leave_fold_out_baseline(data))==4
    assert len(araus_foldwise_ridge(data))==4

def test_araus_invalid_scale(tmp_path):
    p=_araus_csv(tmp_path);d=pd.read_csv(p);d.loc[2,'annoying']=99;d.to_csv(p,index=False)
    with pytest.raises(ValueError,match='1-5'):load_araus_responses(p)

def test_evat_audio_zip(tmp_path):
    bio=BytesIO();t=np.arange(22050)/22050;wavfile.write(bio,22050,np.int16(np.sin(2*np.pi*300*t)*10000))
    p=tmp_path/'audio_examples.zip'
    with ZipFile(p,'w') as z:z.writestr('avas/example.wav',bio.getvalue())
    res=inspect_evat_wav_archive(p)
    assert len(res)==1 and 'spectral_centroid_hz' in res
