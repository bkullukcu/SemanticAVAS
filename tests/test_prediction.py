"""End-to-end and failure-mode checks for the WAV predictor."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.io import wavfile

from semanticavas.prediction import (
    train_wav_predictor, save_predictor, load_predictor, extract_wav,
    prepare_training,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'examples'))
from prediction_demo import create_demo


@pytest.fixture
def sample(tmp_path):
    return create_demo(tmp_path)


def test_train_validation_and_unseen_wav(sample):
    model, oof, audio = train_wav_predictor(sample/'manifest.csv', sample/'ratings.csv',
                                            ['truck_like', 'annoying'])
    assert model.validation['strategy'] == 'nested_leave_one_family_out'
    assert model.validation['sounds'] == 16
    assert model.validation['families'] == 4
    assert len(oof) == 16 and len(audio) == 16
    assert oof[['predicted_truck_like','predicted_annoying']].apply(np.isfinite).all().all()
    prediction = model.predict_wav(sample/'unseen.wav')
    assert 0 <= prediction['truck_like'] <= 100
    assert 0 <= prediction['annoying'] <= 100
    assert isinstance(prediction['outside_training_feature_ranges'], bool)
    assert prediction['audio']['sample_rate_hz'] == 16000
    save_predictor(model, sample/'model.npz')
    loaded = load_predictor(sample/'model.npz')
    assert loaded.predict_wav(sample/'unseen.wav')['truck_like'] == pytest.approx(prediction['truck_like'])


def test_ungrouped_cv(sample):
    manifest = pd.read_csv(sample/'manifest.csv').drop(columns='sound_family')
    manifest.to_csv(sample/'manifest.csv', index=False)
    model, oof, _ = train_wav_predictor(sample/'manifest.csv', sample/'ratings.csv', ['truck_like'])
    assert model.validation['strategy'] == 'nested_leave_one_sound_out'
    assert len(oof) == 16


def test_duplicate_stimulus_and_sound_content(sample):
    m = pd.read_csv(sample/'manifest.csv')
    m.loc[1, 'stimulus'] = m.loc[0, 'stimulus']
    m.to_csv(sample/'manifest.csv', index=False)
    with pytest.raises(ValueError, match='unique'):
        prepare_training(sample/'manifest.csv', sample/'ratings.csv', ['truck_like'])
    m.loc[1, 'stimulus'] = 'S01'
    m.loc[1, 'wav'] = m.loc[0, 'wav']
    m.to_csv(sample/'manifest.csv', index=False)
    with pytest.raises(ValueError, match='same WAV content'):
        prepare_training(sample/'manifest.csv', sample/'ratings.csv', ['truck_like'])


def test_missing_rating_and_range(sample):
    ratings = pd.read_csv(sample/'ratings.csv')
    ratings = ratings[ratings.stimulus != 'S15']
    ratings.to_csv(sample/'ratings.csv', index=False)
    with pytest.raises(ValueError, match='Exact correspondence'):
        prepare_training(sample/'manifest.csv', sample/'ratings.csv', ['truck_like'])
    ratings.loc[len(ratings)] = dict(participant='P1', stimulus='S15', truck_like=150, annoying=50)
    ratings.to_csv(sample/'ratings.csv', index=False)
    with pytest.raises(ValueError, match='0–100'):
        prepare_training(sample/'manifest.csv', sample/'ratings.csv', ['truck_like'])


def test_family_labels_required_when_column_present(sample):
    m = pd.read_csv(sample/'manifest.csv')
    m.loc[0, 'sound_family'] = None
    m.to_csv(sample/'manifest.csv', index=False)
    with pytest.raises(ValueError, match='sound_family'):
        prepare_training(sample/'manifest.csv', sample/'ratings.csv', ['truck_like'])


def test_model_archive_rejects_invalid_schema(sample):
    model, _, _ = train_wav_predictor(sample/'manifest.csv',sample/'ratings.csv',['truck_like'])
    save_predictor(model,sample/'model.npz')
    with np.load(sample/'model.npz', allow_pickle=False) as data:
        fields={k:data[k] for k in data.files}
    metadata=json.loads(str(fields['metadata'].item()))
    metadata['feature_names'][0]='changed_descriptor'
    fields['metadata']=np.asarray(json.dumps(metadata))
    np.savez_compressed(sample/'incompatible.npz',**fields)
    with pytest.raises(ValueError, match='Incompatible'):
        load_predictor(sample/'incompatible.npz')


def test_unseen_feature_flag(sample):
    model, _, _ = train_wav_predictor(sample/'manifest.csv',sample/'ratings.csv',['truck_like'])
    probe=model.x_mean.copy()
    probe[0]=model.max_features[0]+100.0
    p=model.predict_features(probe[None,:])[0]
    assert p['outside_training_feature_ranges'] is True
    assert 0 <= p['truck_like'] <= 100


def test_invalid_audio(sample):
    wavfile.write(sample/'bad.wav', 16000, np.zeros(8000,dtype=np.float32))
    with pytest.raises(ValueError, match='no non-silent'):
        extract_wav(sample/'bad.wav')


def test_cli_train_predict(sample):
    train = subprocess.run([sys.executable,'-m','semanticavas.cli','train-predictor',
        '--manifest',str(sample/'manifest.csv'),'--ratings',str(sample/'ratings.csv'),
        '--scales','truck_like','annoying','--output-dir',str(sample/'model_output')],
        capture_output=True,text=True,timeout=110)
    assert train.returncode == 0, train.stderr
    predict=subprocess.run([sys.executable,'-m','semanticavas.cli','predict-wav',
        '--model',str(sample/'model_output/semantic_model.npz'),str(sample/'unseen.wav'),
        '--output',str(sample/'pred.csv')],capture_output=True,text=True,timeout=110)
    assert predict.returncode == 0, predict.stderr
    frame=pd.read_csv(sample/'pred.csv')
    assert len(frame)==1 and {'truck_like','annoying'}<=set(frame.columns)


def test_8bit_unsigned_pcm_is_centered(tmp_path):
    fs=16000
    t=np.arange(fs)/fs
    signed=.40*np.sin(2*np.pi*310*t)
    data=np.clip(np.round(signed*128+128),0,255).astype(np.uint8)
    wavfile.write(tmp_path/'unsigned.wav', fs, data)
    wavfile.write(tmp_path/'float.wav', fs, signed.astype(np.float32))
    x_uint,_=extract_wav(tmp_path/'unsigned.wav')
    x_float,_=extract_wav(tmp_path/'float.wav')
    # Quantization introduces a small measurement difference but no 8-bit DC offset.
    assert abs(x_uint[0]-x_float[0]) < .2
    assert abs(x_uint[1]-x_float[1]) < 50


def test_recorded_training_provenance(sample):
    train = subprocess.run([sys.executable,'-m','semanticavas.cli','train-predictor',
        '--manifest',str(sample/'manifest.csv'),'--ratings',str(sample/'ratings.csv'),
        '--scales','truck_like','annoying','--output-dir',str(sample/'trained')],
        capture_output=True,text=True,timeout=110)
    assert train.returncode == 0,train.stderr
    report=json.loads((sample/'trained/validation.json').read_text())
    assert report['software_version'] == '0.9.0'
    assert len(report['ratings_sha256']) == 64 and len(report['manifest_sha256']) == 64


def test_prediction_figure_script(sample):
    model, table, _ = train_wav_predictor(sample/'manifest.csv',sample/'ratings.csv',
                                           ['truck_like','annoying'])
    table.to_csv(sample/'heldout.csv',index=False)
    run=subprocess.run([sys.executable,str(ROOT/'examples/plot_prediction_validation.py'),
         '--table',str(sample/'heldout.csv'),'--output',str(sample/'figure.svg')],
         capture_output=True,text=True,timeout=75)
    assert run.returncode == 0,run.stderr
    text=(sample/'figure.svg').read_text(encoding='utf-8')
    assert '<svg' in text and 'Truck like' in text
