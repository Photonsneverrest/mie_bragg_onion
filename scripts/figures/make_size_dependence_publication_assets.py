#!/usr/bin/env python3
"""Create traceable publication/SI assets for the fixed-outer-shell size sweep.

Inputs are processed copies from the completed sweep. Raw calculator output is never
modified. Numerical points already flagged in the audit remain represented only in
that audit; this script does not add new screening rules.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--metrics', type=Path, required=True)
    p.add_argument('--spectra', type=Path, required=True)
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--style', type=Path)
    return p.parse_args()


def main():
    a=parse_args()
    if a.style:
        if not a.style.exists(): raise FileNotFoundError(a.style)
        plt.style.use(a.style)
    for p in (a.metrics,a.spectra,a.audit):
        if not p.exists(): raise FileNotFoundError(p)
    out=a.output_dir; out.mkdir(parents=True,exist_ok=True)
    m=pd.read_csv(a.metrics).sort_values('diameter_um')
    s=pd.read_csv(a.spectra)
    audit=pd.read_csv(a.audit)
    # one spectrum per diameter and direction, already cleaned by upstream analysis
    diameters=np.sort(m.diameter_um.unique())
    cmap=plt.get_cmap('viridis'); norm=Normalize(diameters.min(),diameters.max())
    fig,ax=plt.subplots(2,2,figsize=(7.2,5.8),constrained_layout=True)
    for d in diameters:
        c=cmap(norm(d))
        for j,direction in enumerate(('backward','forward')):
            q=s[(np.isclose(s.diameter_um,d))&(s.direction==direction)].sort_values('wavelength_nm')
            ax[0,j].plot(q.wavelength_nm,q.q_collected_geom,color=c,lw=0.85,alpha=0.9)
    ax[0,0].set(title='Backward-collected spectra',ylabel='Collected efficiency',xlim=(380,780))
    ax[0,1].set(title='Forward-collected spectra',xlim=(380,780))
    for x in ax[0]: x.set_xlabel('Wavelength [nm]')
    # Integrated redistribution
    ax[1,0].plot(m.diameter_um,m.backward_integral_clean,'o-',label='Backward')
    ax[1,0].plot(m.diameter_um,m.forward_integral,'s-',label='Forward')
    ax[1,0].plot(m.diameter_um,m.total_integral_clean,'^-',label='Total')
    ax[1,0].set(xlabel='Outer diameter [µm]',ylabel='Spectral integral [nm]',title='Integrated scattering')
    ax[1,0].legend(frameon=False)
    # colour and directionality stability
    ax2=ax[1,1]
    ax2.plot(m.diameter_um,m.C,'o-',color='#31688e',label=r'$C^*_{ab}$')
    ax2.plot(m.diameter_um,m.L,'s-',color='#35b779',label=r'$L^*$')
    ax2.set(xlabel='Outer diameter [µm]',ylabel='CIELAB value',title='Backward-colour stability')
    ax2b=ax2.twinx(); ax2b.plot(m.diameter_um,m.g_visible_mean,'^-',color='#d55e00',label=r'$\langle g\rangle$')
    ax2b.set_ylabel(r'Visible mean $g$')
    h1,l1=ax2.get_legend_handles_labels(); h2,l2=ax2b.get_legend_handles_labels(); ax2.legend(h1+h2,l1+l2,frameon=False,loc='best')
    sm=ScalarMappable(norm=norm,cmap=cmap); sm.set_array([]); fig.colorbar(sm,ax=ax[0,:],label='Outer diameter [µm]',fraction=.03,pad=.02)
    for x in ax.flat: x.grid(alpha=.15)
    fig.savefig(out/'Figure_size_dependence_fixed_outer_shell.pdf',bbox_inches='tight')
    fig.savefig(out/'Figure_size_dependence_fixed_outer_shell.png',dpi=600,bbox_inches='tight')
    plt.close(fig)
    # Evidence table
    evidence=m[['diameter_um','backward_integral_clean','forward_integral','total_integral_clean','L','C','hue_deg','eta_C','eta_L','hex','g_visible_mean','g_470nm']].copy()
    evidence['flagged_wavelength_count']=evidence.diameter_um.map(audit.groupby('diameter_um').wavelength_nm.nunique()).fillna(0).astype(int)
    evidence.to_csv(out/'Table_size_dependence_evidence.csv',index=False)
    # concise tex table
    cols=['diameter_um','backward_integral_clean','forward_integral','L','C','hue_deg','g_visible_mean']
    evidence[cols].to_latex(out/'Table_size_dependence_evidence.tex',index=False,float_format='%.3f',escape=False,
        caption='Fixed ten-domain outer reflector: size-dependent integrated scattering and backward-colour metrics.',
        label='tab:size_dependence_evidence')
    manifest={
      'script':'make_size_dependence_publication_assets.py',
      'inputs':{str(p):sha256(p) for p in (a.metrics,a.spectra,a.audit)},
      'screening':'No new screening. Uses upstream cleaned plotting spectra and preserves the supplied audit.',
      'diameters_um':diameters.tolist(),
      'outputs':[p.name for p in sorted(out.iterdir())]
    }
    (out/'size_dependence_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(f'Saved {out}')

if __name__=='__main__': main()
