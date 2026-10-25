"""Self-contained synthesis and surrogate example, with explicitly synthetic targets."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from semanticavas import *

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',default='examples/output');a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 x=np.array([(f,c) for f in [50,80,120,160] for c in [1000,2000,3000,4000]],float)
 # Arbitrary analytic response used ONLY to verify software, not human ratings.
 y=np.column_stack([100-0.2*x[:,0]-.007*x[:,1],10+.005*x[:,1]+.00001*(x[:,0]-80)**2])
 m=RidgeSurrogate(.1).fit(x,y);pred=m.predict(x);mask=pareto_front(np.column_stack([-pred[:,0],pred[:,1]]))
 fig,ax=plt.subplots(1,2,figsize=(7.2,3.1),layout='constrained')
 for f0,c in [(50,1000),(120,3000)]:
  s=synthesize(f0=f0,cutoff=c);write_wav(out/f'prototype_f{f0}_c{c}.wav',s);freq=np.fft.rfftfreq(len(s),1/44100);amp=20*np.log10(np.maximum(abs(np.fft.rfft(s))/len(s),1e-9));ax[0].plot(freq,amp,label=f'{f0} Hz / {c} Hz')
 ax[0].set(xlim=(0,4500),ylim=(-100,-20),xlabel='Frequency (Hz)',ylabel='FFT amplitude (dBFS)',title='A  Digital prototypes');ax[0].legend(fontsize=7)
 ax[1].scatter(pred[:,0],pred[:,1],c=x[:,1],cmap='viridis');ax[1].scatter(pred[mask,0],pred[mask,1],s=100,facecolors='none',edgecolors='#D55E00',label='Nondominated');ax[1].set(xlabel='Synthetic objective 1 (maximize)',ylabel='Synthetic objective 2 (minimize)',title='B  Surrogate API verification');ax[1].legend(fontsize=7)
 fig.suptitle('SYNTHETIC VERIFICATION - not predicted human judgments',fontsize=10)
 for ext in ['pdf','svg','png']:fig.savefig(out/('synthetic_design.'+ext),dpi=400,bbox_inches='tight')
 (out/'synthetic_results.json').write_text(json.dumps({'training_rmse':float(np.sqrt(np.mean((y-pred)**2))),'n_parameters':len(x),'note':'training fit to analytic synthetic targets; not validation of human perception'},indent=2));plt.close(fig)
if __name__=='__main__':main()
