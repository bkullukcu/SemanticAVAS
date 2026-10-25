import numpy as np
import pytest
import io
from zipfile import ZipFile
from scipy.io import wavfile
from semanticavas.evat_fidelity import evat_manifest,benchmark_evat_archive,best_envelope_correlation


def _raw(sr=24000,freq=300,duration=1.2):
    t=np.arange(round(sr*duration))/sr
    io_=io.BytesIO();wavfile.write(io_,sr,np.int16(10000*np.sin(2*np.pi*freq*t)))
    return io_.getvalue()

def _make_zip(path):
    with ZipFile(path,'w') as z:
        for category,name,left,right in [
           ('avas','vehicleA_avas_forward','measured','synthesized'),
           ('passby','vehicleA_passBy_forwards','measured','generated'),
           ('tire','tire','measured','synthesized')]:
            z.writestr(f'{category}/{name}_{left}.wav',_raw())
            z.writestr(f'{category}/{name}_{right}.wav',_raw(freq=340))

def test_manifest_pairs_expected(tmp_path):
    p=tmp_path/'x.zip';_make_zip(p)
    m=evat_manifest(p)
    assert len(m)==3 and set(m.category)=={'avas','passby','tire'}

def test_identity_metrics(tmp_path):
    pytest.importorskip('soundfile')  # optional [evat] extra
    p=tmp_path/'x.zip';_make_zip(p)
    _,s,b=benchmark_evat_archive(p)
    assert len(s)==3 and len(b)>10
    assert s.third_octave_shape_js_distance_100_8000.between(0,1).all()
    assert s[s.category=='passby'].envelope_best_pearson_r.isna().all()  # stationary tones have undefined envelope correlation

def test_unpaired_rejected(tmp_path):
    p=tmp_path/'bad.zip'
    with ZipFile(p,'w') as z:z.writestr('avas/vehicleA_measured.wav',_raw())
    with pytest.raises(ValueError,match='Unpaired'):evat_manifest(p,require_all_categories=False)

def test_envelope_alignment():
    x=np.sin(np.arange(100)/10)
    r,lag,n=best_envelope_correlation(x[5:],x[:-5],max_shift_frames=10)
    assert r>.99 and n>=85
