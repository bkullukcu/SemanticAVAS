# Multi-source AVAS scene-design model

SemanticAVAS provides a separate, optional predictor for experimental design
parameters. The model predicts **listener-mean scores** for unseen scenes using
declared design factors. It does not read audio and it does not replace the
acoustic-to-semantic WAV predictor.

## Input

Supply one row per response to `prepare_scene_ratings`, with these columns:

- `participant_id`, `scene_id`, `design_family`
- `scene_type`: `single`, `pair`, or `identity_pair`
- `detuning_cents`: a nonnegative value, or missing when not defined
- `offset_s`: a nonnegative time offset
- `geometry`: `G0_LEFT_RIGHT` or `G45_DIAGONAL`
- `spatial_variant`: `A`, `B`, or `single`
- the named 0–100 listener-rating columns

The seven design descriptors are log-transformed detuning, a detuning-missing
indicator, offset, diagonal geometry, single-source and identity-pair
indicators, and the B route variant. They are experimental parameters rather
than psychoacoustic measurements. Repeated presentations are first averaged
within each listener/scene; scenes then receive one mean per scale.

```python
import pandas as pd
import numpy as np
from semanticavas.scene_model import prepare_scene_ratings, nested_scene_cv

trials = pd.read_csv('pilot_trials.csv')
trials['design_family'] = np.select(
    [trials.scene_type.eq('identity_pair'),
     trials.scene_type.eq('single'),
     trials.block.eq('GEOM')],
    ['IDENTITY', 'SINGLE', 'GEOMETRY'], default='CORE')
x, y, scene_ids, families, n_listeners = prepare_scene_ratings(
    trials, ['intrusiveness_0_100', 'chaoticness_0_100'])
by_scene, mean_reference, alphas = nested_scene_cv(x, y)
by_family, family_reference, _ = nested_scene_cv(x, y, groups=families)
print('Scene holdout MAE:', float(np.abs(by_scene-y).mean()))
print('Family holdout MAE:', float(np.abs(by_family-y).mean()))
```

Each outer fold trains its own standardization and tunes ridge regularization
through an inner holdout. The comparator is the mean score among training
scenes. Scene-level results quantify held-out scenes from an existing design;
family-level results test transfer across declared design groups. Neither
protocol measures prediction on new listeners.

## Connecting a WAV to ratings

The WAV training manifest optionally accepts a `wav_sha256` column:

```text
stimulus,wav,sound_family,wav_sha256
```

Every supplied digest must match the actual audio bytes before training.
Checksums catch accidental pairing of ratings with a re-rendered WAV that has
the same filename. When a study used multiple versions, recover the audio
actually presented before fitting an acoustic model. Digital-audio metrics
from later versions are not interchangeable with the original playback files.

## Evidence boundary

The exploratory nine-person Multi-AVAS pilot included 290 Exp. 2
presentations of 36 distinct scenes. Scene-parameter ridge predictions
improved on a training-mean reference within the recorded design (mean
MAE 5.54 against 6.93 on the five 0–100 scales), while complete design-family
holdout remained close to the mean reference (7.48 against 7.35).
These results come from author-controlled pilot data, not the synthetic
repository fixtures. The original WAV versions have not yet been recovered
and verified for each listening session. No pilot ratings or WAV files are
committed here. These findings therefore support a scene-design baseline,
not an improvement in acoustic WAV prediction or independent-study transfer.
