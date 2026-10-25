"""Empirical participant-balanced semantic aggregation with explicit missingness.

Only computes rankings among evaluated sounds; a new synthesized sound requires
independently paired synthesis parameters and listener ratings.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def validate_semantic_ratings(frame: pd.DataFrame, scales, *, participant='participant', stimulus='stimulus',
                              min_participants=2, require_complete_grid=True):
    """Audit finite 0-100 data without silently imputing missing scale ratings."""
    scales=list(scales)
    if not scales or len(set(scales))!=len(scales):
        raise ValueError('nonempty unique scales required')
    req=[participant,stimulus]+scales
    if not set(req).issubset(frame.columns):raise ValueError('missing columns')
    f=frame[req].copy()
    if f[[participant,stimulus]].isna().any().any():raise ValueError('missing IDs')
    if f.duplicated([participant,stimulus]).any():raise ValueError('duplicate subject-stimulus row')
    for s in scales:
        f[s]=pd.to_numeric(f[s],errors='raise').astype(float)
        vals=f[s].dropna().values
        if not np.isfinite(vals).all() or np.any((vals<0)|(vals>100)):
            raise ValueError('ratings outside 0-100 or nonfinite')
    pp=sorted(f[participant].unique(),key=str)
    ss=sorted(f[stimulus].unique(),key=str)
    if len(pp)<min_participants or len(ss)<2:raise ValueError('too few participants/stimuli')
    if require_complete_grid and len(f)!=len(pp)*len(ss):
        raise ValueError('incomplete participant x stimulus grid')
    return f


def aggregate_semantics(frame,scales):
    """Mean per sound+scale; denominators follow observed (not imputed) cells."""
    f=validate_semantic_ratings(frame,scales)
    means=f.groupby('stimulus',sort=True)[list(scales)].mean()
    coverage=f.groupby('stimulus',sort=True)[list(scales)].count()
    return means,coverage


def bootstrap_profiles(frame,scales,n_boot=2000,seed=2026):
    """Participant cluster bootstrap, skipping missing values within each replicate."""
    f=validate_semantic_ratings(frame,scales)
    pp=sorted(f.participant.unique(),key=str)
    ss=sorted(f.stimulus.unique(),key=str)
    cube=np.stack([f.loc[f.participant==p].set_index('stimulus').reindex(ss)[list(scales)].to_numpy(float) for p in pp])
    if not (isinstance(n_boot,int) and n_boot>=2):raise ValueError('n_boot>=2')
    rng=np.random.default_rng(seed)
    draws=rng.integers(0,len(pp),size=(n_boot,len(pp)))
    data=cube[draws]
    with np.errstate(invalid='ignore',divide='ignore'):
        numer=np.nansum(data,axis=1)
        denom=np.sum(np.isfinite(data),axis=1)
        bmean=np.divide(numer,denom,out=np.full_like(numer,np.nan),where=denom>0)
    # A bootstrap resample may contain no non-missing observation for a rare cell.
    # Exclude that draw for that cell; never substitute an invented rating.
    good=np.isfinite(bmean).sum(axis=0)
    if np.any(good < max(2,n_boot//2)):
        raise ValueError('insufficient valid bootstrap replicates for at least one sound/scale')
    low,high=np.nanquantile(bmean,[.025,.975],axis=0)
    return (pd.DataFrame(low,index=ss,columns=scales),pd.DataFrame(high,index=ss,columns=scales))


def rank_existing_sounds(means,target,*,weights=None,scale_ranges=None):
    cols=list(target)
    if not cols or not set(cols).issubset(means.columns):raise ValueError('unknown scales')
    w=np.asarray([1 if weights is None else weights[c] for c in cols],float)
    width=np.asarray([100 if scale_ranges is None else scale_ranges[c] for c in cols],float)
    t=np.asarray([target[c] for c in cols],float)
    if not np.isfinite(t).all() or not np.isfinite(w).all() or not np.isfinite(width).all() or np.any(w<0) or w.sum()<=0 or np.any(width<=0):
        raise ValueError('invalid target/weights/widths')
    mm=means[cols].to_numpy(float)
    if not np.isfinite(mm).all():raise ValueError('nonfinite means')
    score=np.sum(w*((mm-t)/width)**2,axis=1)/w.sum()
    return pd.DataFrame({'stimulus':means.index,'target_loss':score}).sort_values(['target_loss','stimulus'],kind='stable').reset_index(drop=True)


def bootstrap_target(frame,target,n_boot=2000,seed=2026):
    cols=list(target)
    f=validate_semantic_ratings(frame,cols)
    pp=sorted(f.participant.unique(),key=str); ss=sorted(f.stimulus.unique(),key=str)
    cube=np.stack([f.loc[f.participant==p].set_index('stimulus').reindex(ss)[cols].to_numpy(float) for p in pp]);
    rng=np.random.default_rng(seed)
    tally=np.zeros(len(ss),int)
    t=np.array([target[k] for k in cols],float)
    for ids in rng.integers(len(pp),size=(n_boot,len(pp))):
        a=cube[ids]
        den=np.isfinite(a).sum(0)
        mean=np.divide(np.nansum(a,axis=0),den,out=np.full((len(ss),len(cols)),np.nan),where=den>0)
        if not np.isfinite(mean).all():raise ValueError('unobserved bootstrap profile')
        cost=((mean-t)**2).mean(1)
        tally[np.argmin(cost)]+=1
    return pd.DataFrame({'stimulus':ss,'bootstrap_selection_fraction':tally/n_boot}).sort_values('bootstrap_selection_fraction',ascending=False).reset_index(drop=True)


def leave_one_participant_out(frame,target):
    cols=list(target)
    f=validate_semantic_ratings(frame,cols)
    out=[]
    for p in sorted(f.participant.unique(),key=str):
        sub=f.loc[f.participant!=p]
        means,_=aggregate_semantics(sub,cols)
        rank=rank_existing_sounds(means,target)
        out.append({'left_out_participant':p,'selected_stimulus':rank.iloc[0]['stimulus'],'target_loss':float(rank.iloc[0]['target_loss'])})
    return pd.DataFrame(out)


def pca_stimulus_profiles(means,scales):
    """Descriptive PCA on measured stimulus means; NOT inferential participant PCA."""
    x=means[list(scales)].to_numpy(float)
    if not np.isfinite(x).all() or x.shape[0]<3:raise ValueError('finite means and >=3 sounds required')
    mean=x.mean(0);std=x.std(0,ddof=1);std[std<=1e-12]=1
    z=(x-mean)/std
    u,s,vt=np.linalg.svd(z,full_matrices=False)
    scores=u*s
    explained=(s*s)/(s*s).sum()
    for k in range(vt.shape[0]):
        j=np.argmax(abs(vt[k]));
        if vt[k,j]<0:scores[:,k]*=-1;vt[k]*=-1
    return pd.DataFrame(scores[:,:2],index=means.index,columns=['PC1','PC2']),pd.DataFrame(vt[:2].T,index=scales,columns=['PC1','PC2']),explained


def _ridge_fit_predict(train_x,train_y,test_x,alpha):
    mean=train_x.mean(0);sd=train_x.std(0);sd[sd==0]=1
    tx=(train_x-mean)/sd;vx=(test_x-mean)/sd
    ym=train_y.mean(0)
    beta=np.linalg.solve(tx.T@tx+alpha*np.eye(tx.shape[1]),tx.T@(train_y-ym))
    return vx@beta+ym


def nested_leave_one_sound_out(features,targets,alpha_grid=(.1,1.,10.,100.,1000.),inner_splits=4):
    """True outer sound-level LOOCV; features standardized and alpha tuned inside folds."""
    x=np.asarray(features,float);y=np.asarray(targets,float)
    if x.ndim!=2 or y.ndim!=2 or len(x)!=len(y) or len(x)<10 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('>=10 aligned finite stimuli required')
    out=np.zeros_like(y);baseline=np.zeros_like(y);alphas=[]
    for heldout in range(len(x)):
        tr=np.array([j for j in range(len(x)) if j!=heldout]); folds=np.array_split(tr,inner_splits)
        loss=[]
        for alpha in alpha_grid:
            errors=[]
            for va in folds:
                it=np.setdiff1d(tr,va)
                pred=_ridge_fit_predict(x[it],y[it],x[va],float(alpha))
                errors.append(abs(y[va]-pred).mean())
            loss.append(float(np.mean(errors)))
        a=float(alpha_grid[int(np.argmin(loss))]);alphas.append(a)
        out[heldout]=_ridge_fit_predict(x[tr],y[tr],x[[heldout]],a)[0]
        baseline[heldout]=y[tr].mean(0)
    return out,baseline,np.array(alphas)


def nested_leave_one_group_out(features,targets,groups,alpha_grid=(.1,1.,10.,100.,1000.)):
    """Source-blocked outer CV and source-blocked alpha tuning.

    The caller must provide reliable source group identifiers; unknown siblings
    of any recording can cause optimistic error estimates.
    """
    x=np.asarray(features,float);y=np.asarray(targets,float);g=np.asarray(groups)
    if x.ndim!=2 or y.ndim!=2 or len(x)!=len(y) or len(g)!=len(x) or len(np.unique(g))<3:
        raise ValueError('need matched data and >=3 independent source groups')
    if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('nonfinite values')
    pred=np.zeros_like(y);base=np.zeros_like(y);selected=np.empty(len(x),float)
    for held in np.unique(g):
        te=np.where(g==held)[0];tr=np.where(g!=held)[0]
        if len(tr)<5:raise ValueError('too few remaining training sounds')
        inner_groups=np.unique(g[tr])
        scores=[]
        for alpha in alpha_grid:
            fold_losses=[]
            for val_group in inner_groups:
                it=tr[g[tr]!=val_group];iv=tr[g[tr]==val_group]
                if len(it)<3:raise ValueError('too few inner training sounds')
                estimate=_ridge_fit_predict(x[it],y[it],x[iv],float(alpha))
                fold_losses.append(np.mean(abs(estimate-y[iv])))
            scores.append(float(np.mean(fold_losses)))
        a=float(alpha_grid[int(np.argmin(scores))]);selected[te]=a
        pred[te]=_ridge_fit_predict(x[tr],y[tr],x[te],a)
        base[te]=y[tr].mean(0)
    return pred,base,selected
