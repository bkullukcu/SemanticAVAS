"""Semantic target search and explicit, bounded synthesis-surrogate support."""
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.optimize import linprog

__all__=['SemanticAtlas','RidgeSurrogate','pareto_front','synthesize','write_wav','group_holdout']


class SemanticAtlas:
    """Participant-balanced stimulus profiles with cluster bootstrap uncertainty.

    Rows are participant x stimulus. Multiple repetitions are averaged before
    participants. Scales must be explicitly named; the API does not infer polarity.
    """
    def __init__(self,frame,scales,participant='participant',stimulus='stimulus'):
        self.scales=list(scales); self.participant=participant; self.stimulus=stimulus
        if not self.scales or len(set(self.scales))!=len(self.scales):raise ValueError('unique scales required')
        req=[participant,stimulus]+self.scales
        if not set(req)<=set(frame.columns):raise ValueError('missing columns')
        if frame[req].isna().any().any():raise ValueError('missing identifiers or ratings')
        f=frame[req].copy();f[self.scales]=f[self.scales].astype(float)
        if not np.isfinite(f[self.scales]).all().all():raise ValueError('nonfinite ratings')
        self.frame=f.groupby([participant,stimulus],as_index=False)[self.scales].mean()
        self.means=self.frame.groupby(stimulus)[self.scales].mean().sort_index()

    def rank(self,target,weights=None,scale_ranges=None):
        """Weighted squared distance to explicit numeric targets in original units.

        scale_ranges supplies external scale widths (e.g. 100 for 0--100).
        It defaults to one, so equal units should be used or explicit ranges set.
        """
        cols=list(target)
        if not cols or not set(cols)<=set(self.scales):raise ValueError('unknown/empty target scales')
        t=np.array([target[c] for c in cols],float)
        w=np.array([1. if weights is None else weights.get(c,1.) for c in cols])
        r=np.array([1. if scale_ranges is None else scale_ranges[c] for c in cols])
        if not np.isfinite(t).all() or not np.isfinite(w).all() or np.any(w<0) or w.sum()==0 or not np.isfinite(r).all() or np.any(r<=0):
            raise ValueError('invalid target, weights or scale widths')
        out=self.means.copy();out['target_loss']=np.sum(w*((out[cols].values-t)/r)**2,axis=1)/w.sum()
        return out.sort_values('target_loss',kind='stable')

    def intervals(self,n_boot=1000,seed=2026):
        """Pointwise percentile intervals, resampling whole participant profiles.

        Requires complete participant x stimulus data so bootstrap weights are
        identical across stimuli. Not simultaneous intervals or new-sound bounds.
        """
        if not isinstance(n_boot,(int,np.integer)) or n_boot<2:raise ValueError('n_boot >=2 required')
        ids=self.frame[self.participant].unique(); stimuli=self.means.index
        if len(ids)<2:raise ValueError('at least two participants required')
        cube=np.stack([self.frame[self.frame[self.participant]==i].set_index(self.stimulus).reindex(stimuli)[self.scales].values for i in ids])
        if not np.isfinite(cube).all():raise ValueError('complete repeated-measures design required for intervals')
        rng=np.random.default_rng(seed); boot=cube[rng.integers(len(ids),size=(n_boot,len(ids)))].mean(axis=1)
        lo,hi=np.quantile(boot,[.025,.975],axis=0)
        return pd.DataFrame(lo,index=stimuli,columns=self.scales),pd.DataFrame(hi,index=stimuli,columns=self.scales)


    def bootstrap_target_choice(self,target,weights=None,scale_ranges=None,
                                n_boot=500,seed=2026):
        """Rank stability among *measured sounds*, resampling whole listeners.

        Requires all participant-stimulus observations for all used scales.
        Choice fractions are sample sensitivity, not probability of a new sound's
        success and not inferential certainty of a participant population.
        """
        if not isinstance(n_boot,(int,np.integer)) or n_boot < 2:
            raise ValueError("n_boot >= 2 required")
        ids=self.frame[self.participant].unique()
        stimuli=self.means.index
        if len(ids)<2:raise ValueError("at least two participants required")
        scales=self.scales
        cube=np.stack([self.frame[self.frame[self.participant]==i]
            .set_index(self.stimulus).reindex(stimuli)[scales].to_numpy(float) for i in ids])
        if not np.isfinite(cube).all():raise ValueError("complete repeated-measures design required")
        cols=list(target)
        if not cols or not set(cols)<=set(scales):raise ValueError("unknown/empty target scales")
        t=np.asarray([target[c] for c in cols],float)
        w=np.asarray([1. if weights is None else weights.get(c,1.) for c in cols],float)
        r=np.asarray([1. if scale_ranges is None else scale_ranges[c] for c in cols],float)
        if not (np.isfinite(t).all() and np.isfinite(w).all() and np.isfinite(r).all()) or np.any(w<0) or w.sum()<=0 or np.any(r<=0):
            raise ValueError("invalid targets, weights, scale widths")
        indices=[scales.index(c) for c in cols]
        rng=np.random.default_rng(seed)
        tally=np.zeros(len(stimuli),dtype=int)
        for ids_sample in rng.integers(len(ids),size=(n_boot,len(ids))):
            avg=cube[ids_sample].mean(axis=0)[:,indices]
            loss=np.sum(w*((avg-t)/r)**2,axis=1)/w.sum()
            tally[int(np.argmin(loss))]+=1
        return pd.Series(tally/n_boot,index=stimuli,name="bootstrap_selection_fraction")


class RidgeSurrogate:
    """Small-data quadratic ridge model for numeric synthesis parameters.

    This model is not pretrained. Fit only with matched parameter/rating records.
    Range checks reject marginal extrapolation; they do not prove joint support.
    """
    def __init__(self,alpha=1.):
        if alpha<=0:raise ValueError('alpha must be positive')
        self.alpha=float(alpha)
    def _features(self,x):
        z=(x-self.mean)/self.scale
        return np.column_stack([np.ones(len(z)),z]+[z[:,i]*z[:,j] for i in range(z.shape[1]) for j in range(i,z.shape[1])])
    def fit(self,x,y):
        x=np.asarray(x,float);y=np.asarray(y,float)
        if y.ndim==1:y=y[:,None]
        if x.ndim!=2 or y.ndim!=2 or len(x)!=len(y) or len(x)<3 or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('finite aligned matrices with >=3 rows required')
        self.mean=x.mean(0);self.scale=x.std(0);self.scale[self.scale==0]=1
        self.training_x=x.copy();self.lower=x.min(0);self.upper=x.max(0);f=self._features(x)
        penalty=np.eye(f.shape[1])*self.alpha;penalty[0,0]=0
        self.coef=np.linalg.solve(f.T@f+penalty,f.T@y)
        return self
    def predict(self,x,allow_extrapolation=False, support="bounds"):
        if not hasattr(self,'coef'):raise ValueError('fit first')
        x=np.asarray(x,float)
        if x.ndim!=2 or x.shape[1]!=len(self.mean) or not np.isfinite(x).all():raise ValueError('invalid parameters')
        if support not in {"bounds", "convex_hull"}:raise ValueError("support must be bounds or convex_hull")
        if not allow_extrapolation:
            if np.any((x<self.lower-1e-10)|(x>self.upper+1e-10)):
                raise ValueError('outside training parameter bounds')
            if support == "convex_hull":
                # This check is geometric. A convex hull is NOT evidence of a
                # perceptually valid interpolation, especially for categories.
                aeq=np.vstack([self.training_x.T,np.ones(len(self.training_x))])
                for row in x:
                    ok=linprog(np.zeros(len(self.training_x)),A_eq=aeq,
                               b_eq=np.r_[row,1.0],bounds=(0,None),method="highs").success
                    if not ok:raise ValueError("outside joint convex training support")
        return self._features(x)@self.coef


def group_holdout(groups):
    """Leave-one-stimulus/family-out; fit preprocessing inside every split."""
    g=np.asarray(groups)
    if g.ndim!=1 or len(np.unique(g))<2:raise ValueError('need >=2 groups')
    for key in np.unique(g):yield np.where(g!=key)[0],np.where(g==key)[0]


def pareto_front(costs):
    """Boolean nondominated mask; every column is minimized. Ties are retained."""
    c=np.asarray(costs,float)
    if c.ndim!=2 or not np.isfinite(c).all():raise ValueError('finite cost matrix required')
    return np.array([not np.any(np.all(c<=r,axis=1)&np.any(c<r,axis=1)) for r in c])


def synthesize(f0=80.,cutoff=2000.,noise_center=300.,noise_width=100.,noise_mix=.2,
               duration=1.5,fs=44100,seed=2026,rms_dbfs=-24.):
    """Periodic band-limited harmonic/noise prototype, not calibrated AVAS output.

    All components are restricted below Nyquist. Broadband noise is filtered by
    an FFT mask. Output is normalised to RMS and raises on digital clipping.
    """
    if not isinstance(fs,(int,np.integer)) or fs<1000 or not np.isfinite([f0,cutoff,noise_center,noise_width,noise_mix,duration,rms_dbfs]).all():raise ValueError('invalid synthesis arguments')
    if not 0<f0<=cutoff<fs/2 or not 0<=noise_mix<=1 or duration<=0 or noise_width<=0 or noise_center-noise_width/2<0 or noise_center+noise_width/2>=fs/2:raise ValueError('invalid frequency band, mix or duration')
    n=int(round(fs*duration))
    if n<32:raise ValueError('duration too short')
    t=np.arange(n)/fs;y=np.zeros(n)
    for k in range(1,int(cutoff//f0)+1):y+=np.sin(2*np.pi*k*f0*t)/k
    y/=np.sqrt(np.mean(y*y))
    rng=np.random.default_rng(seed);noise=rng.normal(size=n);f=np.fft.rfftfreq(n,1/fs)
    mask=(abs(f-noise_center)<=noise_width/2)&(f<=cutoff)
    if not mask.any():raise ValueError('noise band has no FFT bins below cutoff')
    noise=np.fft.irfft(np.fft.rfft(noise)*mask,n=n);noise/=np.sqrt(np.mean(noise*noise))
    y=(1-noise_mix)*y+noise_mix*noise
    ramp=min(n//4,int(.02*fs));env=np.ones(n);env[:ramp]=np.sin(np.linspace(0,np.pi/2,ramp))**2;env[-ramp:]=env[:ramp][::-1]
    y*=env;y*=10**(rms_dbfs/20)/np.sqrt(np.mean(y*y))
    if max(abs(y))>=1:raise ValueError('requested RMS clips; lower rms_dbfs')
    return y.astype(np.float32)


def write_wav(path,signal,fs=44100):
    a=np.asarray(signal,float)
    if a.ndim!=1 or len(a)==0 or not np.isfinite(a).all() or np.max(abs(a))>=1:raise ValueError('finite unclipped mono waveform required')
    wavfile.write(path,fs,a.astype(np.float32))
