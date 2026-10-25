"""Create a self-contained WAV training set, train a predictor, then test a new sound.

The generated audio and questionnaire scores are illustrative, deterministic data.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

from semanticavas import synthesize, write_wav
from semanticavas.prediction import train_wav_predictor, save_predictor, load_predictor


def create_demo(destination):
    dest = Path(destination)
    dest.mkdir(parents=True, exist_ok=True)
    manifest, ratings = [], []
    for k in range(16):
        tone = 62.0 + k * 8.0
        mix = 0.10 + 0.035 * (k % 5)
        sig = synthesize(f0=tone, cutoff=1750+80*k,
                         noise_center=230+27*k, noise_width=105,
                         noise_mix=mix, fs=16000, duration=.65,
                         rms_dbfs=-25, seed=100+k)
        name = f'S{k:02d}'
        write_wav(dest / f'{name}.wav', sig, fs=16000)
        manifest.append({'stimulus': name, 'wav': f'{name}.wav', 'sound_family': f'F{k//4}'})
        for p in range(5):
            ratings.append({'participant': f'P{p+1}', 'stimulus': name,
                            'truck_like': float(np.clip(75-0.36*(tone-62)+1.6*(p-2), 0, 100)),
                            'annoying': float(np.clip(22+72*mix+2.0*(p-2), 0, 100))})
    pd.DataFrame(manifest).to_csv(dest / 'manifest.csv', index=False)
    pd.DataFrame(ratings).to_csv(dest / 'ratings.csv', index=False)
    unseen = synthesize(f0=118, cutoff=2200, noise_center=440, noise_width=105,
                        noise_mix=.18, duration=.65, fs=16000, seed=999, rms_dbfs=-25)
    write_wav(dest / 'unseen.wav', unseen, fs=16000)
    return dest


def main():
    p = argparse.ArgumentParser(description='Generated-audio prediction demonstration')
    p.add_argument('--output-dir', default='prediction_demo_output')
    a = p.parse_args()
    dest = create_demo(a.output_dir)
    model, folds, audio = train_wav_predictor(dest / 'manifest.csv', dest / 'ratings.csv',
                                            ['truck_like', 'annoying'])
    save_predictor(model, dest / 'semantic_model.npz')
    folds.to_csv(dest / 'heldout_predictions.csv', index=False)
    audio.to_csv(dest / 'audio_summary.csv', index=False)
    predicted = load_predictor(dest / 'semantic_model.npz').predict_wav(dest / 'unseen.wav')
    print('Validation:', model.validation['strategy'], 'mean MAE', round(model.validation['mean_cv_mae'], 2))
    print('Unseen synthetic WAV predictions:', {s: round(predicted[s], 2) for s in model.scales})
    print('Training-feature range flag:', predicted['outside_training_feature_ranges'])
    print('Files written to:', dest)


if __name__ == '__main__':
    main()
