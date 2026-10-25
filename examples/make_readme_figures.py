"""Recreate README figures using generated (not measured) listener ratings.

    python examples/make_readme_figures.py

Use --latex for labels typeset with a locally installed LaTeX distribution.
The default requires only the package dependencies.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from semanticavas import SemanticAtlas

BLUE='#245778'; ORANGE='#ce7950'; TEAL='#0d8278'; GOLD='#ae842c'
DARK='#23313f'; MUTED='#61707b'
SOUNDS=['A','B','C','D']
COLORS=[BLUE,TEAL,ORANGE,GOLD]
SCALES=['truck_like','annoying','tonal']
DISPLAY=['Truck-like','Annoying','Tonal']
MEANS=np.array([[82,23,38],[55,44,72],[70,31,28],[47,62,85]],dtype=float)
TARGET={'truck_like':78.,'annoying':22.}


def configure(latex):
    plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'], 'font.size':10,
        'axes.titlesize':12,'axes.labelsize':10,'xtick.labelsize':9,'ytick.labelsize':9,
        'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#66727b',
        'axes.labelcolor':DARK,'text.color':DARK,'xtick.color':MUTED,'ytick.color':MUTED,
        'svg.fonttype':'path','text.usetex':latex,'figure.facecolor':'white','axes.facecolor':'white'})


def demo_atlas():
    rng=np.random.default_rng(42)
    rows=[]
    for p in range(18):
        shared=rng.normal(0,3,size=3)
        for s,sound in enumerate(SOUNDS):
            sample=np.clip(MEANS[s]+shared+rng.normal(0,5,size=3),0,100)
            rows.append({'participant':f'SYN{p+1:02d}','stimulus':sound,**dict(zip(SCALES,sample))})
    atlas=SemanticAtlas(pd.DataFrame(rows),SCALES)
    lo,hi=atlas.intervals(n_boot=600,seed=9)
    ranking=atlas.rank(TARGET,scale_ranges={'truck_like':100,'annoying':100})
    return atlas,lo,hi,ranking


def render(out):
    atlas,lo,hi,ranking=demo_atlas()
    fig,axs=plt.subplots(1,3,figsize=(9.8,3.6),sharey=True)
    for ax,k,title in zip(axs,SCALES,DISPLAY):
        for idx,sound in enumerate(SOUNDS):
            mean=atlas.means.loc[sound,k]
            error=np.array([[mean-lo.loc[sound,k]],[hi.loc[sound,k]-mean]])
            ax.errorbar(mean,idx,xerr=error,fmt='o',ms=7,capsize=3,
                        elinewidth=1.55,color=COLORS[idx],ecolor=COLORS[idx],zorder=3)
        ax.set(xlim=(0,100),xlabel='Rating (0–100)',title=title)
        ax.set_xticks([0,25,50,75,100]);ax.grid(axis='x',lw=.7,color='#e1e6eb'); ax.invert_yaxis()
    axs[0].set_yticks(range(4),[f'Sound {s}' for s in SOUNDS])
    axs[0].set_ylabel('Evaluated sound')
    fig.suptitle('Participant-balanced semantic profiles',x=.075,ha='left',fontsize=14,color=DARK)
    fig.subplots_adjust(left=.13,right=.98,top=.77,bottom=.25,wspace=.24)
    fig.text(.13,.10,'Dots: synthetic means; whiskers: pointwise 95% listener-bootstrap intervals (18 simulated listeners)',fontsize=8.4,color=MUTED)
    fig.savefig(out/'semantic_profiles.svg',format='svg',metadata={'Date':None,'Creator':'SemanticAVAS deterministic example'})
    plt.close(fig)

    fig,(ax,ax2)=plt.subplots(1,2,figsize=(9.6,4.45),gridspec_kw={'width_ratios':[1.65,1]})
    ax.scatter(TARGET['truck_like'],TARGET['annoying'],marker='*',s=265,c='#172b39',edgecolors='white',linewidths=.8,zorder=5,label='Target')
    for idx,sound in enumerate(SOUNDS):
        row=atlas.means.loc[sound]
        ax.plot([TARGET['truck_like'],row['truck_like']],[TARGET['annoying'],row['annoying']],lw=1.1,ls=':',color=COLORS[idx],alpha=.7)
        ax.scatter(row['truck_like'],row['annoying'],s=87,color=COLORS[idx],edgecolor='white',lw=1,zorder=4)
        ax.annotate(sound,(row['truck_like'],row['annoying']),xytext=(7,-8 if sound in ['A','D'] else 9),textcoords='offset points',fontsize=10,color=DARK,weight='bold')
    ax.set(xlim=(30,100),ylim=(5,80),xlabel='Truck-like rating (0–100)',ylabel='Annoying rating (0–100)')
    ax.grid(color='#e1e6eb',linewidth=.65)
    ax.legend(frameon=False,loc='upper right')
    ranked=list(ranking.index)
    vals=ranking.loc[ranked,'target_loss'].to_numpy()
    ax2.barh(np.arange(len(ranked)),vals,color=[COLORS[SOUNDS.index(s)] for s in ranked],height=.58)
    ax2.set_yticks(np.arange(len(ranked)),[f'Sound {s}' for s in ranked]);ax2.invert_yaxis()
    ax2.set(xlabel='Weighted normalized squared distance',title='Target ranking')
    ax2.grid(axis='x',color='#e1e6eb',linewidth=.65)
    ax2.set_axisbelow(True)
    ax2.set_xlim(0,max(vals)*1.17)
    fig.suptitle('Target-based selection among evaluated sounds',x=.065,ha='left',fontsize=14,color=DARK)
    fig.subplots_adjust(left=.10,right=.98,top=.82,bottom=.19,wspace=.42)
    fig.text(.99,.032,'Synthetic ratings  |  equal weights  |  smaller distance = closer target',ha='right',fontsize=8.2,color=MUTED)
    fig.savefig(out/'target_ranking.svg',format='svg',metadata={'Date':None,'Creator':'SemanticAVAS deterministic example'})
    plt.close(fig)
    print(f'Wrote two SVG figures; synthetic top-ranked sound: {ranking.index[0]}')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parents[1]/'docs'/'figures')
    p.add_argument('--latex',action='store_true')
    args=p.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)
    configure(args.latex); render(args.output_dir)

if __name__=='__main__':main()
