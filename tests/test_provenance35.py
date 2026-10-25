import numpy as np
import pandas as pd
import pytest
from semanticavas.provenance35 import nearest_published_descriptor_neighbors,group_cv_summary,audio_inventory

def test_neighbor_not_treated_as_synthesis_inverse():
    x=pd.DataFrame({'stimulus':[f'S{i:02d}' for i in range(1,27)],'f':np.arange(26)*1.0,'g':np.arange(26)[::-1]*.3})
    r=nearest_published_descriptor_neighbors(x,['f','g'])
    assert len(r)==8
    assert all('NOT a synthesis recipe' in a for a in r['interpretation'])

def test_35_assets_uniqueness_guard(tmp_path):
    with pytest.raises(ValueError):
        audio_inventory(tmp_path/'not_exists.zip',pd.DataFrame({'stimulus':['S01'],'stimulus_filename':['1.wav']}))

def test_group_cv_zero_and_positive_delta():
    df=pd.DataFrame({'group':['a','a','b'],'scale_measured':[1,1,2],
      'scale_group_prediction':[1,1,3], 'scale_group_baseline':[2,2,2]})
    g=group_cv_summary(df,['scale'])
    assert g.set_index('group').loc['a','model_minus_baseline']==-1
    assert g.set_index('group').loc['b','model_minus_baseline']==1


def test_inventory_includes_assets_without_optional_hdf(tmp_path):
    from zipfile import ZipFile
    from semanticavas.provenance35 import AUDIO_ROOT
    mapping=pd.DataFrame({'stimulus':[f'S{i:02d}' for i in range(1,36)],
        'stimulus_filename':[f'{i}.wav' if i<=26 else f'reference_{i}.wav' for i in range(1,36)]})
    p=tmp_path/'assets.zip'
    with ZipFile(p,'w') as z:
        for i,name in enumerate(mapping.stimulus_filename):z.writestr(AUDIO_ROOT+name,f'unique asset bytes {i}')
    result=audio_inventory(p,mapping)
    assert len(result)==35 and not result.file_has_hdf.any()
    assert not result.full_recipe_verified.any()
