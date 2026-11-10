#!/usr/bin/env python3
# %% Imports and paths
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
VALIDATION_DIR=Path('/mnt/d/Rocca/MieScattering/outputs/numerical_validation')
HOLLOWING_DIR=Path('/mnt/d/Rocca/MieScattering/outputs/hollowing_60L_nH1p5')
FIGURE_DIR=HOLLOWING_DIR/'analysis_figures'; FIGURE_DIR.mkdir(parents=True,exist_ok=True)
# %% Validation overview
v=pd.read_csv(VALIDATION_DIR/'validation_metrics.csv')
fig,ax=plt.subplots(figsize=(8,4.5),constrained_layout=True)
for (nh,k),g in v.groupby(['n_high','k']):
 q=g.groupby('n_layers').valid_fraction.min(); ax.plot(q.index,q.values,marker='o',label=f'nH={nh:g}, k={k:g}')
ax.set(xlabel='Number of layers',ylabel='Worst valid fraction',ylim=(-.02,1.02),title='Onset of raw near-field instability'); ax.legend(fontsize=6,ncol=2); fig.savefig(FIGURE_DIR/'validation_onset.png',dpi=300); plt.show()
# %% Homogeneous-control error from unity
h=v[(np.isclose(v.n_high,1))&(np.isclose(v.k,0))]
fig,ax=plt.subplots(figsize=(8,4.5),constrained_layout=True)
for npts,g in h.groupby('npts'): ax.semilogy(g.n_layers,g.raw_max,marker='o',label=f'{npts} points')
ax.axhline(1,color='k',ls=':'); ax.set(xlabel='Number of artificial identical layers',ylabel='Maximum raw |E|/|E0|',title='Homogeneous control'); ax.legend(); fig.savefig(FIGURE_DIR/'homogeneous_control.png',dpi=300); plt.show()
# %% Hollowing spectra
s=pd.read_pickle(HOLLOWING_DIR/'hollowing_spectra.pkl')
for series in ['single_layer','complete_bilayer']:
 d=s[s.series==series]; fig,ax=plt.subplots(figsize=(9,5),constrained_layout=True)
 norm=plt.Normalize(d.n_air_layers.min(),d.n_air_layers.max()); cmap=plt.get_cmap('viridis')
 for n,g in d.groupby('n_air_layers'): ax.plot(g.wavelength_nm,g.qbk,color=cmap(norm(n)),lw=.8)
 sm=plt.cm.ScalarMappable(norm=norm,cmap=cmap); fig.colorbar(sm,ax=ax,label='Inner layers replaced by air'); ax.set(xlabel='Wavelength [nm]',ylabel='Qbk',title=f'60-layer hollowing: {series}'); fig.savefig(FIGURE_DIR/f'hollowing_{series}_qbk.png',dpi=300); plt.show()
# %% Peak evolution
summary=pd.read_csv(HOLLOWING_DIR/'hollowing_summary.csv'); fig,axes=plt.subplots(2,1,figsize=(8,7),sharex=True,constrained_layout=True)
for series,g in summary[summary.series!='all_air_control'].groupby('series'):
 axes[0].plot(g.inner_air_radius_nm,g.peak_qbk,marker='o',label=series); axes[1].plot(g.inner_air_radius_nm,g.peak_qbk_wavelength_nm,marker='o',label=series)
axes[0].set_ylabel('Peak Qbk'); axes[1].set(xlabel='Inner air radius [nm]',ylabel='Peak wavelength [nm]'); axes[0].legend(); fig.savefig(FIGURE_DIR/'hollowing_peak_evolution.png',dpi=300); plt.show()
