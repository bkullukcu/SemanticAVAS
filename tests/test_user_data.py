import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from semanticavas.user_data import convert, InputContractError
from semanticavas.core import SemanticAtlas

FIX = Path(__file__).resolve().parents[1] / 'examples' / 'step3_user_data'


def test_wide_csv_can_feed_atlas(tmp_path):
    r=convert(FIX/'ratings_wide.csv',FIX/'schema.json',tmp_path/'out')
    assert (r['rows_after_policy'],r['participants'],r['stimuli'])==(12,4,3)
    atlas=SemanticAtlas(pd.read_csv(tmp_path/'out/ratings.csv'),list(r['scales']))
    assert len(atlas.rank({'truck_like': 60},scale_ranges={'truck_like':100}))==3
    assert (tmp_path/'out/conversion_manifest.json').exists()


def test_npz_scale_mapping_order_independent(tmp_path):
    d=pd.read_csv(FIX/'ratings_wide.csv')
    np.savez(tmp_path/'ratings.npz',ratings=d[['annoying_rating','heavy_rating']].to_numpy(float),
             participants=d.listener.astype(str).to_numpy(dtype=str),stimuli=d.sound_id.astype(str).to_numpy(dtype=str),
             scales=['annoying','truck_like'])
    r=convert(tmp_path/'ratings.npz',FIX/'schema.json',tmp_path/'out')
    assert r['stimuli']==3
    out=pd.read_csv(tmp_path/'out/ratings.csv')
    assert np.isclose(out[out.stimulus=='AVAS_A']['truck_like'].mean(),32.5)


def test_long_ratings_pivot(tmp_path):
    df=pd.read_csv(FIX/'ratings_wide.csv')
    x=df.melt(id_vars=['listener','sound_id'],value_vars=['heavy_rating','annoying_rating'],var_name='question',value_name='score')
    x['question']=x['question'].map({'heavy_rating':'truck_like','annoying_rating':'annoying'})
    path=tmp_path/'long.csv';x.to_csv(path,index=False)
    cfg=json.loads((FIX/'schema.json').read_text());cfg['layout']='long';cfg['columns']={'participant':'listener','stimulus':'sound_id','scale':'question','rating':'score'}
    sch=tmp_path/'cfg.json';sch.write_text(json.dumps(cfg))
    r=convert(path,sch,tmp_path/'out')
    assert r['rows_after_policy']==12


def test_repeated_respondent_rejected_without_trial(tmp_path):
    d=pd.read_csv(FIX/'ratings_wide.csv');d=pd.concat([d,d.iloc[[0]]])
    path=tmp_path/'dup.csv';d.to_csv(path,index=False)
    with pytest.raises(InputContractError,match='duplicate'):
        convert(path,FIX/'schema.json',tmp_path/'out')


def test_out_of_range_rejected(tmp_path):
    d=pd.read_csv(FIX/'ratings_wide.csv');d.loc[0,'heavy_rating']=500
    path=tmp_path/'bad.csv';d.to_csv(path,index=False)
    with pytest.raises(InputContractError,match='range'):
        convert(path,FIX/'schema.json',tmp_path/'out')


def test_missing_policy_discloses_drops(tmp_path):
    d=pd.read_csv(FIX/'ratings_wide.csv');d.loc[0,'heavy_rating']=np.nan
    path=tmp_path/'missing.csv';d.to_csv(path,index=False)
    with pytest.raises(InputContractError,match='missing rating'):
        convert(path,FIX/'schema.json',tmp_path/'out1')
    cfg=json.loads((FIX/'schema.json').read_text());cfg['missing_policy']='drop_incomplete'
    sch=tmp_path/'cfg.json';sch.write_text(json.dumps(cfg))
    assert convert(path,sch,tmp_path/'out2')['rows_dropped']==1


def test_metadata_key_checked(tmp_path):
    cfg=json.loads((FIX/'schema.json').read_text());cfg['metadata_csv']=str(FIX/'metadata.csv')
    sch=tmp_path/'cfg.json';sch.write_text(json.dumps(cfg))
    assert convert(FIX/'ratings_wide.csv',sch,tmp_path/'out')['metadata_included']
    assert (tmp_path/'out/stimulus_metadata.csv').exists()


def test_npz_object_dtype_rejected_without_pickle(tmp_path):
    p=tmp_path/'bad.npz';np.savez(p,ratings=np.ones((2,2)),participants=np.array(['P1','P2'],dtype=object),
         stimuli=np.array(['A','B'],dtype=str),scales=np.array(['truck_like','annoying'],dtype=str))
    with pytest.raises(InputContractError,match='not object/pickle'):
        convert(p,FIX/'schema.json',tmp_path/'out')


def test_long_duplicate_scale_cell_detected(tmp_path):
    d=pd.read_csv(FIX/'ratings_wide.csv')
    x=d.melt(id_vars=['listener','sound_id'],value_vars=['heavy_rating','annoying_rating'],var_name='question',value_name='score')
    x['question']=x['question'].map({'heavy_rating':'truck_like','annoying_rating':'annoying'})
    x=pd.concat([x,x.iloc[[0]]],ignore_index=True)
    src=tmp_path/'long.csv';x.to_csv(src,index=False)
    cfg=json.loads((FIX/'schema.json').read_text());cfg['layout']='long';cfg['columns']={'participant':'listener','stimulus':'sound_id','scale':'question','rating':'score'}
    sch=tmp_path/'cfg.json';sch.write_text(json.dumps(cfg))
    with pytest.raises(InputContractError,match='duplicate'):
        convert(src,sch,tmp_path/'out')
