"""Source-aware domain warnings, active listening design and optional intervals.

All decisions are derived from training or independently held-out calibration
material. Scores are diagnostics, not probabilities of correct perception.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd


def _two_dimensional(x, label):
    x=np.asarray(x,float)
    if x.ndim!=2 or not np.isfinite(x).all():
        raise ValueError(f'{label} must be a finite numeric matrix')
    return x


@dataclass
class FeatureDomain:
    mean: np.ndarray
    scale: np.ndarray
    training_z: np.ndarray
    threshold: float
    quantile: float
    calibration_unit: str

    def assess(self, features):
        """Detect recordings distant from previously observed feature vectors."""
        x=_two_dimensional(features,'probe features')
        if x.shape[1]!=len(self.mean):
            raise ValueError('Feature dimensions do not match training')
        z=(x-self.mean)/self.scale
        squared=np.sum((z[:,None,:]-self.training_z[None,:,:])**2,axis=2)
        distances=np.sqrt(squared.min(axis=1))
        return pd.DataFrame({'nearest_training_distance':distances,
            'reference_distance_threshold':self.threshold,
            'out_of_domain_warning':distances>self.threshold,
            'distance_over_threshold':distances/max(self.threshold,1e-12)})


def fit_feature_domain(features, *, groups=None, quantile=.95):
    """Calibrate nearest-sound distance with leave-one-source-out references.

    The reference is entirely in-sample; this is a heuristic OOD warning,
    not a calibrated probability of validity.
    """
    x=_two_dimensional(features,'training features')
    if len(x)<6 or x.shape[1]<1:
        raise ValueError('At least six labeled training recordings needed')
    if not 0.5<=quantile<1.:
        raise ValueError('quantile must be in [0.5,1)')
    mean=x.mean(axis=0);scale=x.std(axis=0)
    scale[scale<1e-12]=1.
    z=(x-mean)/scale
    if groups is None:
        distinct=np.arange(len(x))
        unit='sound'
    else:
        distinct=np.asarray(groups,str)
        if len(distinct)!=len(x) or len(np.unique(distinct))<3:
            raise ValueError('At least three source groups required')
        unit='source_group'
    held=[]
    for i in range(len(z)):
        valid=np.flatnonzero(distinct!=distinct[i])
        if not len(valid):
            raise ValueError('No independent training references')
        held.append(float(np.min(np.linalg.norm(z[valid]-z[i],axis=1))))
    threshold=float(np.quantile(held,quantile))
    return FeatureDomain(mean,scale,z,threshold,float(quantile),unit)


def propose_listening_batch(candidate_features, candidate_ids, labeled_features, *,
                            n_select=5, candidate_groups=None,
                            max_per_group=None, reserved_groups=(),
                            estimated_uncertainty=None, uncertainty_weight=.25):
    """Greedy novelty + supplied uncertainty ranking with source constraints.

    Only candidate training-pool sounds may be selected; all reserved holdout
    sources are excluded, even if their candidate WAV files are available.
    """
    candidates=_two_dimensional(candidate_features,'candidate features')
    known=_two_dimensional(labeled_features,'training features')
    ids=list(map(str,candidate_ids))
    if (len(candidates)!=len(ids) or len(set(ids))!=len(ids)
            or len(known)<3 or candidates.shape[1]!=known.shape[1]):
        raise ValueError('Aligned, distinct candidates and >=3 training sounds required')
    if not isinstance(n_select,int) or n_select<1:
        raise ValueError('n_select must be positive')
    if not 0<=uncertainty_weight<=1:
        raise ValueError('uncertainty_weight must be between 0 and 1')
    if max_per_group is not None and (not isinstance(max_per_group,int) or max_per_group<1):
        raise ValueError('max_per_group must be positive')
    if candidate_groups is None:
        groups=['unknown']*len(ids)
    else:
        groups=list(map(str,candidate_groups))
        if len(groups)!=len(ids):
            raise ValueError('candidate_groups length mismatch')
    reserved=set(map(str,reserved_groups))
    allowed=np.array([i for i,g in enumerate(groups) if g not in reserved])
    if not len(allowed) or n_select>len(allowed):
        raise ValueError('No feasible candidates after holdout exclusions')
    if estimated_uncertainty is None:
        uncertainty=np.zeros(len(ids),float)
    else:
        uncertainty=np.asarray(estimated_uncertainty,float)
        if (uncertainty.shape!=(len(ids),) or not np.isfinite(uncertainty).all()
                or (uncertainty<0).any()):
            raise ValueError('Provide finite non-negative uncertainties per candidate')
    center=known.mean(axis=0);scale=known.std(axis=0)
    scale[scale<1e-12]=1
    zc=(candidates-center)/scale
    known_z=(known-center)/scale
    uncertainty=uncertainty/(uncertainty.max() if uncertainty.max()>0 else 1.)
    available=set(map(int,allowed))
    selected=[]
    counts={}
    anchors=known_z.copy()
    for rank in range(min(n_select,len(available))):
        if not available:
            break
        remaining=sorted(i for i in available
                         if max_per_group is None or counts.get(groups[i],0)<max_per_group)
        if not remaining:
            break
        distances=np.array([np.linalg.norm(anchors-zc[i],axis=1).min()
                            for i in remaining])
        novelty=distances/(distances.max() if distances.max()>0 else 1.)
        score=(1.-uncertainty_weight)*novelty+uncertainty_weight*uncertainty[remaining]
        winner=int(np.argmax(score))
        i=remaining[winner]
        selected.append({'rank':rank+1,'stimulus':ids[i],'source_group':groups[i],
                         'novelty_distance':float(distances[winner]),
                         'declared_uncertainty':float(uncertainty[i]),
                         'acquisition_score':float(score[winner])})
        anchors=np.vstack((anchors,zc[i]))
        counts[groups[i]]=counts.get(groups[i],0)+1
        available.remove(i)
    if len(selected)<n_select:
        raise ValueError('Source quotas cannot fulfill requested batch size')
    return pd.DataFrame(selected)


def group_conformal_radius(observed, predictions, groups, *, alpha=.10):
    """Conservative max-residual group quantile on independent calibration data.

    The caller must guarantee independent, exchangeable calibration groups
    not used for training or model selection. No finite radius is returned
    if the number of groups is too small for the requested miscoverage.
    """
    observed=_two_dimensional(observed,'observed calibration')
    predictions=_two_dimensional(predictions,'predicted calibration')
    g=np.asarray(groups,str)
    if (observed.shape!=predictions.shape or len(g)!=len(observed)
            or pd.isna(np.asarray(groups, dtype=object)).any()):
        raise ValueError('Calibration arrays must be aligned')
    if not 0<alpha<1:
        raise ValueError('alpha must be in (0,1)')
    units=np.unique(g)
    if len(units)<3 or len(g)!=len(observed):
        raise ValueError('Independent calibration source groups required')
    residuals=np.abs(observed-predictions)
    by_group=np.array([residuals[g==label].max(axis=0) for label in units])
    quantile_index=int(np.ceil((len(units)+1)*(1-alpha)))-1
    if quantile_index>=len(units):
        raise ValueError('Too few independent groups for finite conformal radius')
    radius=np.sort(by_group,axis=0)[quantile_index]
    return radius, {'n_calibration_groups':len(units),'alpha':alpha,
                    'unit':'source_group','coverage_assumption':'exchangeable calibration groups'}


def sample_weight_from_sem(sem, *, floor=2., max_ratio=10.):
    """Numerical weights for an exploratory noise-aware regression.

    These weights alone do not establish a better perceptual prediction model.
    """
    values=np.asarray(sem,float)
    if values.ndim!=1 or not np.isfinite(values).all() or np.any(values<0):
        raise ValueError('SEM must be non-negative and finite')
    if floor<=0 or max_ratio<1:
        raise ValueError('Invalid weight regularization')
    raw=1./(values**2+floor**2)
    raw=np.minimum(raw,raw.min()*max_ratio)
    return raw/raw.mean()


@dataclass
class ListenerWeightedRidge:
    """Ridge fitted with training-only participant SEM weights."""
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: np.ndarray
    coefficients: np.ndarray

    def predict(self, features):
        x=_two_dimensional(features,'weighted ridge features')
        if x.shape[1]!=len(self.x_mean):
            raise ValueError('Feature dimension mismatch')
        return ((x-self.x_mean)/self.x_scale)@self.coefficients+self.y_mean


def fit_listener_weighted_ridge(features, responses, sem, *, alpha=10., floor=2.):
    """Fit ridge with regularized inverse-SEM weights derived from training listeners."""
    x=_two_dimensional(features,'features')
    y=_two_dimensional(responses,'responses')
    if len(x)!=len(y) or len(x)<3 or not np.isfinite(alpha) or alpha<=0:
        raise ValueError('At least three aligned rows and positive alpha required')
    w=sample_weight_from_sem(sem,floor=floor)
    if len(w)!=len(x):
        raise ValueError('One observed SEM per training sound required')
    mean=np.average(x,axis=0,weights=w)
    centered=x-mean
    scale=np.sqrt(np.average(centered**2,axis=0,weights=w))
    scale[scale<1e-12]=1.
    z=centered/scale
    ym=np.average(y,axis=0,weights=w)
    beta=np.linalg.solve((z.T*w)@z+float(alpha)*np.eye(z.shape[1]),
                         (z.T*w)@(y-ym))
    return ListenerWeightedRidge(mean,scale,ym,beta)


def evaluate_listener_weighting(features, responses, sem, groups, *,
                                alphas=(.1,1.,10.,100.), floor=2.,
                                bounds=(0.,100.)):
    """Source-group-nested comparison of weighted ridge versus training mean.

    Inner and outer test-response SEM values are never used for fitting.
    """
    x=_two_dimensional(features,'features')
    y=_two_dimensional(responses,'responses')
    errors=np.asarray(sem,float)
    g=np.asarray(groups,str)
    if (len(y)!=len(x) or errors.shape!=(len(x),)
            or len(g)!=len(x) or len(np.unique(g))<3):
        raise ValueError('Aligned sounds/SEM and at least three groups required')
    lo=np.broadcast_to(np.asarray(bounds[0],float),(y.shape[1],))
    hi=np.broadcast_to(np.asarray(bounds[1],float),(y.shape[1],))
    all_rows=np.arange(len(x))
    rows=[]
    for withheld in np.unique(g):
        test=all_rows[g==withheld]
        train=all_rows[g!=withheld]
        choices=[]
        for alpha in alphas:
            losses=[]
            for inner_source in np.unique(g[train]):
                inner_test=train[g[train]==inner_source]
                fit=train[g[train]!=inner_source]
                if len(fit)<3:raise ValueError('Insufficient independent inner training sounds')
                mdl=fit_listener_weighted_ridge(x[fit],y[fit],errors[fit],
                                                alpha=alpha,floor=floor)
                pred=np.clip(mdl.predict(x[inner_test]),lo,hi)
                losses.append(float(np.abs(pred-y[inner_test]).mean()))
            choices.append(float(np.mean(losses)))
        chosen=float(alphas[int(np.argmin(choices))])
        model=fit_listener_weighted_ridge(x[train],y[train],errors[train],
                                          alpha=chosen,floor=floor)
        predicted=np.clip(model.predict(x[test]),lo,hi)
        baseline=np.repeat(y[train].mean(0)[None,:],len(test),axis=0)
        for j,i in enumerate(test):
            rows.append({'sample_index':int(i),'source_group':str(withheld),
                         'selected_alpha':chosen,
                         'mae':float(np.abs(predicted[j]-y[i]).mean()),
                         'baseline_mae':float(np.abs(baseline[j]-y[i]).mean())})
    return pd.DataFrame(rows)
