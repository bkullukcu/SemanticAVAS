"""Admissibility-first catalog decisions and transparent weight sensitivity."""
import numpy as np
import pandas as pd

def admissible_catalog(profiles, metadata, allowed):
    """Filter a unique stimulus-indexed catalog by explicit metadata values.

    Missing metadata fail closed. allowed maps columns to permitted value lists.
    No calibration or regulatory suitability is inferred from audio descriptors.
    """
    if not profiles.index.is_unique or not metadata.index.is_unique:raise ValueError('unique stimulus indices required')
    if not profiles.index.isin(metadata.index).all():raise ValueError('missing stimulus metadata')
    m=metadata.reindex(profiles.index);keep=np.ones(len(m),bool)
    for key,values in allowed.items():
        if key not in m or m[key].isna().any():raise ValueError('missing admissibility metadata')
        if isinstance(values,str):raise ValueError('allowed values must be a collection, not a string')
        keep &= m[key].isin(values).to_numpy()
    if not keep.any():raise ValueError('no admissible candidates')
    return profiles.loc[keep].copy()

def weight_sensitivity(profiles, target, scale_ranges, weight_scenarios):
    """Return every loss/rank and tie-aware winner share across supplied scenarios.

    Scenario frequencies describe chosen preferences, never population probability.
    Input profiles are measured catalog means, not predictions of new sounds.
    """
    cols=list(target);x=profiles[cols].to_numpy(float)
    t=np.array([target[c] for c in cols],float);r=np.array([scale_ranges[c] for c in cols],float)
    w=np.asarray(weight_scenarios,float)
    if not profiles.index.is_unique or len(profiles)==0 or w.ndim!=2 or w.shape[1]!=len(cols) or len(w)==0:
        raise ValueError('unique nonempty catalog and scenario-by-scale weights required')
    if not all(np.isfinite(v).all() for v in [x,t,r,w]) or np.any(r<=0) or np.any(w<0) or np.any(w.sum(1)<=0):
        raise ValueError('finite values, positive widths and nonnegative weights required')
    losses=(((x-t)/r)**2)@ (w/w.sum(1)[:,None]).T
    winners=np.isclose(losses,losses.min(0),rtol=1e-10,atol=1e-12)
    shares=(winners/winners.sum(0)).mean(1)
    details=pd.DataFrame(losses,index=profiles.index,columns=range(len(w)))
    return {'losses':details,'ranks':details.rank(method='min'),
            'winner_share':pd.Series(shares,index=profiles.index,name='scenario_winner_share')}
