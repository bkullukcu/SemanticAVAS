"""Generate a scientific held-out prediction plot from the synthetic tutorial."""
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description='Plot synthetic held-out validation results')
    parser.add_argument('--table', default='prediction_demo_output/heldout_predictions.csv')
    parser.add_argument('--output', default='docs/figures/prediction_validation.svg')
    args = parser.parse_args()
    table = pd.read_csv(args.table)
    scales = [c[len('observed_'):] for c in table if c.startswith('observed_')]
    if not scales:
        raise ValueError('No observed scale columns in the validation table')
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':9, 'svg.fonttype':'none',
                         'axes.spines.top':False, 'axes.spines.right':False})
    fig, ax = plt.subplots(1, len(scales), figsize=(7.3,3.0), squeeze=False, constrained_layout=True)
    for i, scale in enumerate(scales):
        a=ax[0,i]
        obs = table[f'observed_{scale}'].to_numpy(float)
        pred = table[f'predicted_{scale}'].to_numpy(float)
        base = table[f'baseline_{scale}'].to_numpy(float)
        mae = float(np.mean(abs(obs - pred)))
        bmae = float(np.mean(abs(obs - base)))
        a.plot([0,100],[0,100],color='#6b7280',alpha=.65,linewidth=1,linestyle='--',label='Ideal')
        a.scatter(obs,base,s=24,marker='x',color='#a6a6a6',alpha=.8,label='Train-mean reference')
        a.scatter(obs,pred,s=37,edgecolor='#173349',linewidth=.5,color='#236d9e',alpha=.9,label='Ridge predictor')
        a.set(xlim=(0,100),ylim=(0,100),xlabel='Observed listener mean / 100',
              ylabel='Held-out prediction / 100',title=scale.replace('_',' ').capitalize())
        a.text(.04,.97,f'MAE {mae:.1f} · reference {bmae:.1f}',transform=a.transAxes,
               ha='left',va='top',fontsize=8)
        a.set_aspect('equal', adjustable='box')
    handles, labels=ax[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(0.5, -0.04),ncol=3,frameon=False,fontsize=8)
    dest=Path(args.output);dest.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(dest,format='svg',bbox_inches='tight')
    plt.close(fig)
    print(dest)

if __name__ == '__main__':
    main()
