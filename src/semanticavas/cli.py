"""Command-line interface for semantic ratings, WAV models and sound synthesis."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from . import __version__
from .core import SemanticAtlas, synthesize, write_wav


def main(argv=None):
    p = argparse.ArgumentParser(
        description='SemanticAVAS: analyse listener ratings and predict semantics from WAV files')
    p.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
    commands = p.add_subparsers(dest='command', required=True)

    convert_parser = commands.add_parser('convert', help='Normalize participant ratings from CSV or NPZ')
    convert_parser.add_argument('input')
    convert_parser.add_argument('--schema', required=True)
    convert_parser.add_argument('--output-dir', default='converted_ratings')

    rank = commands.add_parser('rank', help='Rank already evaluated sounds against a target')
    rank.add_argument('csv')
    rank.add_argument('--target', required=True, help='JSON mapping of scales to target ratings')
    rank.add_argument('--output', default='ranking.csv')
    rank.add_argument('--weights', help='JSON mapping of scale weights')
    rank.add_argument('--scale-ranges', help='JSON mapping of scale widths')

    synth = commands.add_parser('synthesize', help='Generate a digital prototype waveform')
    synth.add_argument('--f0', type=float, default=80)
    synth.add_argument('--cutoff', type=float, default=2000)
    synth.add_argument('--output', default='prototype.wav')

    train = commands.add_parser('train-predictor', help='Fit a WAV-to-semantic ridge predictor')
    train.add_argument('--manifest', required=True, help='CSV: stimulus,wav[,sound_family]')
    train.add_argument('--ratings', required=True, help='CSV: participant,stimulus,<rating scales>')
    train.add_argument('--scales', nargs='+', required=True, help='0–100 semantic rating columns')
    train.add_argument('--output-dir', required=True)
    train.add_argument('--max-seconds', type=float, default=5.0)

    predict = commands.add_parser('predict-wav', help='Apply a trained model to new WAV files')
    predict.add_argument('--model', required=True)
    predict.add_argument('wav', nargs='+')
    predict.add_argument('--output', default='wav_predictions.csv')

    a = p.parse_args(argv)
    if a.command == 'convert':
        from .user_data import convert
        print(json.dumps(convert(a.input, a.schema, a.output_dir), indent=2))
    elif a.command == 'rank':
        target = json.loads(a.target)
        atlas = SemanticAtlas(pd.read_csv(a.csv), list(target))
        atlas.rank(target,
            weights=None if a.weights is None else json.loads(a.weights),
            scale_ranges=None if a.scale_ranges is None else json.loads(a.scale_ranges),
        ).to_csv(a.output)
    elif a.command == 'synthesize':
        write_wav(a.output, synthesize(f0=a.f0, cutoff=a.cutoff))
    elif a.command == 'train-predictor':
        from .prediction import train_wav_predictor, save_predictor
        model, heldout, audio = train_wav_predictor(
            a.manifest, a.ratings, a.scales, max_seconds=a.max_seconds)
        out = Path(a.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        save_predictor(model, out / 'semantic_model.npz')
        heldout.to_csv(out / 'heldout_predictions.csv', index=False)
        audio.to_csv(out / 'audio_summary.csv', index=False)
        (out / 'validation.json').write_text(json.dumps({
            **model.validation,
            'software_version': __version__,
            'manifest_sha256': hashlib.sha256(Path(a.manifest).read_bytes()).hexdigest(),
            'ratings_sha256': hashlib.sha256(Path(a.ratings).read_bytes()).hexdigest(),
            'full_fit_alpha': model.alpha,
            'feature_extractor_max_seconds': model.max_seconds,
        }, indent=2), encoding='utf-8')
        print(json.dumps({'model': str(out / 'semantic_model.npz'),
            'validation': str(out / 'validation.json'),
            'mean_cv_mae': model.validation['mean_cv_mae'],
            'mean_baseline_cv_mae': model.validation['mean_baseline_cv_mae']}, indent=2))
    elif a.command == 'predict-wav':
        from .prediction import load_predictor
        model = load_predictor(a.model)
        predictions = [model.predict_wav(path) for path in a.wav]
        pd.DataFrame([{k: v for k, v in result.items()
                       if k not in ('audio', 'cv_abs_error_p90_by_scale')}
                      for result in predictions]).to_csv(a.output, index=False)
        print(json.dumps({'output': a.output, 'predictions': predictions}, indent=2))


if __name__ == '__main__':
    main()
