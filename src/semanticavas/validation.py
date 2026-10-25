"""Explicit per-stimulus parameter/response joins and family-blocked scoring."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .core import RidgeSurrogate, group_holdout


def match_synthesis_ratings(parameters,ratings,*,parameter_columns,rating_columns,
                            stimulus='stimulus',participant='participant',family='sound_family'):
    """Fail closed on unmatched stimulus codes, missing parameters, or non-finite ratings.

    `parameters`: one row per physical stimulus with explicit family and features.
    `ratings`: participant x stimulus responses, repetitions may be averaged.
    Returns participant-balanced stimulus means merged with parameters.
    """
    parameter_columns=list(parameter_columns);rating_columns=list(rating_columns)
    if not parameter_columns or not rating_columns:
        raise ValueError('numeric parameter/rating column lists required')
    needp=[stimulus,family]+parameter_columns
    needr=[stimulus,participant]+rating_columns
    if not set(needp)<=set(parameters.columns) or not set(needr)<=set(ratings.columns):
        raise ValueError('missing parameter, family, participant or rating columns')
    p=parameters[needp].copy(); r=ratings[needr].copy()
    if p[stimulus].isna().any() or p[stimulus].duplicated().any() or p[family].isna().any():
        raise ValueError('parameter table requires unique stimulus and defined family')
    if r[[stimulus,participant]].isna().any().any():
        raise ValueError('ratings require nonmissing stimulus and participant IDs')
    if set(r[stimulus])!=set(p[stimulus]):
        raise ValueError('unmatched stimulus codes: exact join required')
    for x in [p[parameter_columns],r[rating_columns]]:
        try: arr=x.to_numpy(dtype=float)
        except (TypeError,ValueError) as e:raise ValueError('numeric values required') from e
        if not np.isfinite(arr).all():raise ValueError('all parameters and ratings must be finite')
    per_listener=r.groupby([stimulus,participant],as_index=False)[rating_columns].mean()
    means=per_listener.groupby(stimulus,as_index=False)[rating_columns].mean()
    df=p.merge(means,on=stimulus,validate='one_to_one',how='inner')
    return df.sort_values(stimulus).reset_index(drop=True)


def score_family_holdout(table,*,parameter_columns,rating_columns,family='sound_family',alpha=1.0):
    """LOFO validation for matched REAL data once available; safe for synthetic demonstrations.

    The quadratic model is refit independently within each held-out family.
    Prediction of held-out families requires `allow_extrapolation=True` solely
    for *scoring*; this is not permission for unguarded sound design use.
    """
    cols=list(parameter_columns);targets=list(rating_columns)
    need=cols+targets+[family]
    if not set(need)<=set(table.columns):raise ValueError('missing columns')
    x=table[cols].to_numpy(float);y=table[targets].to_numpy(float)
    if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('non-finite input')
    splits=list(group_holdout(table[family].to_numpy()))
    output=[]
    for tr,te in splits:
        if len(tr)<3:raise ValueError('at least 3 training sounds per fold')
        model=RidgeSurrogate(alpha).fit(x[tr],y[tr]); pred=model.predict(x[te],allow_extrapolation=True)
        baseline=np.broadcast_to(y[tr].mean(axis=0),(len(te),len(targets)))
        bounds=np.all((x[te]>=model.lower-1e-10)&(x[te]<=model.upper+1e-10),axis=1)
        for k,j in enumerate(te):
            output.append({'sound_family':str(table.iloc[j][family]),'row_index':int(j),
                           'inside_training_marginal_bounds':bool(bounds[k]),
                           'model_mae':float(np.mean(abs(pred[k]-y[j]))),
                           'train_mean_baseline_mae':float(np.mean(abs(baseline[k]-y[j])))})
    return pd.DataFrame(output)
