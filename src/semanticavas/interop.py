"""Optional, explicitly calibrated interoperation with pyfar and MoSQITo.

Digital waveform samples (dBFS) are NOT acoustic pressure (Pa). Standardised
loudness estimation is intentionally blocked until a physical calibration is supplied.
"""
from __future__ import annotations
import numpy as np


def to_pyfar_signal(samples, sampling_rate=44100):
    """Convert a mono digital waveform to a pyfar Signal without SPL claims."""
    y=np.asarray(samples, float)
    if y.ndim!=1 or not len(y) or not np.isfinite(y).all() or not np.isfinite(sampling_rate) or sampling_rate<=0:
        raise ValueError('expected finite mono waveform and positive sampling rate')
    try:
        import pyfar as pf
    except ImportError as e:
        raise ImportError('pyfar is optional: pip install pyfar') from e
    return pf.Signal(y, sampling_rate)


def calibrated_pressure(samples, *, pa_per_full_scale):
    """Explicit microphone/playback chain conversion: digital FS -> pressure Pa.

    `pa_per_full_scale` MUST come from an independently documented calibration.
    No conversion from digital RMS/dBFS to acoustic Pa is inferred.
    """
    y=np.asarray(samples,float)
    if y.ndim!=1 or len(y)==0 or not np.isfinite(y).all():
        raise ValueError('finite 1D samples required')
    if not np.isfinite(pa_per_full_scale) or pa_per_full_scale<=0:
        raise ValueError('positive physical calibration pa_per_full_scale is mandatory')
    return y*float(pa_per_full_scale)


def mosqito_stationary_loudness(samples, sampling_rate, *, pa_per_full_scale=None,
                                model='zwicker', field_type='free'):
    """Calibrated MoSQITo loudness; never infer Pa from digital full scale.

    `model='zwicker'`: ISO 532-1-based loudness_zwst, returns sones.
    `model='ecma'`: ECMA-418-2-based loudness_ecma at exactly 48 kHz,
    returns sone_HMS. The units are deliberately not merged.
    """
    if pa_per_full_scale is None:
        raise ValueError('measured acoustic calibration (Pa per digital full scale) required')
    if not np.isfinite(sampling_rate) or sampling_rate<=0:
        raise ValueError('invalid sampling rate')
    pressure=calibrated_pressure(samples, pa_per_full_scale=pa_per_full_scale)
    try:
        from mosqito import sq_metrics
    except ImportError as e:
        raise ImportError('MoSQITo is optional: pip install mosqito') from e
    if model=='zwicker':
        result=sq_metrics.loudness_zwst(pressure,int(sampling_rate),field_type=field_type)
        return {'loudness':float(np.asarray(result[0]).mean()),'units':'sone',
                'model':'ISO 532-1 stationary (MoSQITo)', 'sampling_rate_hz':float(sampling_rate)}
    if model=='ecma':
        if int(sampling_rate)!=48000:
            raise ValueError('ECMA loudness calculation needs 48000 Hz input')
        result=sq_metrics.loudness_ecma(pressure,int(sampling_rate))
        return {'loudness':float(np.asarray(result[0]).mean()),'units':'sone_HMS',
                'model':'ECMA-418-2 (MoSQITo)', 'sampling_rate_hz':float(sampling_rate)}
    raise ValueError('model must be zwicker or ecma')
