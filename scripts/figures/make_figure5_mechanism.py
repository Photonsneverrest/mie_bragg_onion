#!/usr/bin/env python3
"""Create the publication mechanism figure with conservative spike handling.

Only nominally lossless hollow-onion wavelengths that violate passivity
(|Qabs| > tolerance) or contain nonfinite values are masked. Only isolated,
bounded one-sample gaps are interpolated in a plotting copy. Original data and an
audit CSV are retained. No magnitude, derivative, or percentile filter is used.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SELECTED=[
 ('hollow_bragg_onion','none',0.0,'Hollow Bragg onion','#17324d','-'),
 ('homogeneous','bruggeman',0.0,'Homogeneous, $k=0$','#64748b','--'),
 ('homogeneous','bruggeman',0.01,'Homogeneous, $k=0.01$','#2a9d8f','-'),
 ('homogeneous','bruggeman',0.10,'Homogeneous, $k=0.1$','#e9c46a','-'),
 ('homogeneous','bruggeman',1.0,'Homogeneous, $k=1$','#e76f51','-'),
 ('homogeneous','bruggeman',3.0,'Homogeneous, $k=3$','#8e3b46','-')]

def clean_for_plot(df,tol):
 out=df.copy(); h=(out.model=='hollow_bragg_onion')&np.isclose(out.k,0)
 metrics=['qext','qsca','qforward','qbackward','g']
 invalid=h&((out.qabs.abs()>tol)|(~np.isfinite(out[['qext','qsca','qforward','qbackward']]).all(axis=1)))
 audit=out.loc[invalid,['model','mixing','k','wavelength_nm','qext','qsca','qabs','qforward','qbackward','lossless_residual']].copy()
 audit['reason']=f'lossless passivity violation: |Qabs| > {tol:g} or nonfinite value'
 out.loc[invalid,metrics]=np.nan
 for _,idx in out.groupby(['model','mixing','n_real','k'],dropna=False).groups.items():
  part=out.loc[idx].sort_values('wavelength_nm')
  for col in metrics:
   vals=part[col].copy(); bad=vals.isna().to_numpy()
   for i in np.where(bad)[0]:
    if i>0 and i<len(vals)-1 and not bad[i-1] and not bad[i+1]:
     vals.iloc[i]=np.interp(part.wavelength_nm.iloc[i],part.wavelength_nm.iloc[[i-1,i+1]],vals.iloc[[i-1,i+1]])
   out.loc[part.index,col]=vals.values
 return out,audit

def main():
 p=argparse.ArgumentParser();p.add_argument('--spectra',type=Path,required=True);p.add_argument('--polar',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--style',type=Path);p.add_argument('--passivity-tolerance',type=float,default=1e-6);a=p.parse_args()
 a.output_dir.mkdir(parents=True,exist_ok=True)
 if a.style: plt.style.use(a.style)
 spec=pd.read_csv(a.spectra); polar=pd.read_csv(a.polar); plot,audit=clean_for_plot(spec,a.passivity_tolerance)
 audit.to_csv(a.output_dir/'mechanism_spike_audit.csv',index=False);plot.to_csv(a.output_dir/'homogeneous_reference_spectra_v3_plot_clean.csv',index=False)
 fig,axes=plt.subplots(2,2,figsize=(13.2,8.6),constrained_layout=True); A,B,C,D=axes.ravel()
 for model,mix,k,label,color,ls in SELECTED:
  x=plot[(plot.model==model)&(plot.mixing==mix)&np.isclose(plot.k,k)].sort_values('wavelength_nm')
  A.plot(x.wavelength_nm,x.qsca,label=label,color=color,ls=ls,lw=1.4)
  B.plot(x.wavelength_nm,x.qforward,color=color,ls=ls,lw=1.4)
  B.plot(x.wavelength_nm,x.qbackward,color=color,ls=':' if ls=='-' else '-.',lw=1.15)
  if model != 'hollow_bragg_onion':
   q=polar[(polar.model==model)&(polar.mixing==mix)&np.isclose(polar.k,k)].sort_values('theta_deg')
   C.semilogy(q.theta_deg,np.maximum(q.dQdOmega,1e-12),color=color,ls=ls,lw=1.3)
 for ax in [A,B]: ax.set(xlim=(380,780),xlabel='Wavelength [nm]');ax.grid(alpha=.18)
 A.set(ylabel=r'$Q_{\rm sca}$',title='(a) Total scattering');A.legend(frameon=False,fontsize=8,ncol=2)
 B.set(ylabel='Hemisphere efficiency',title='(b) Forward (solid) and backward (dotted)')
 C.set(xlim=(0,180),xlabel=r'Scattering angle $\theta$ [deg]',ylabel=r'$\mathrm{d}Q_{\rm sca}/\mathrm{d}\Omega$ [sr$^{-1}$]',title='(c) Homogeneous controls at 470 nm');C.grid(alpha=.18,which='both')
 C.text(.98,.04,'Hollow onion omitted at 470 nm: passivity failure',transform=C.transAxes,ha='right',va='bottom',fontsize=7,color='0.35')
 x=spec[(spec.model=='homogeneous')&(spec.mixing=='bruggeman')&np.isclose(spec.wavelength_nm,470)&(spec.k>0)].sort_values('k')
 D.semilogx(x.k,x.qforward,'o-',label='Forward hemisphere',color='#264653');D.semilogx(x.k,x.qbackward,'s-',label='Backward hemisphere',color='#e76f51');D.semilogx(x.k,x.qsca,'^-',label='Total scattering',color='#2a9d8f')
 D2=D.twinx();D2.semilogx(x.k,x.g,'d-',label='$g$',color='#6a4c93');D.set(xlabel=r'Homogeneous-sphere extinction $k$',ylabel='Scattering efficiency',title='(d) Absorption cannot remove forward diffraction');D2.set_ylabel(r'Asymmetry parameter $g$',color='#6a4c93');D.grid(alpha=.18);h1,l1=D.get_legend_handles_labels();h2,l2=D2.get_legend_handles_labels();D.legend(h1+h2,l1+l2,frameon=False,fontsize=8)
 for ext in ['pdf','png']:
  fig.savefig(a.output_dir/f'Figure5_mechanism_homogeneous_controls.{ext}',dpi=600 if ext=='png' else None,bbox_inches='tight')
 plt.close(fig)
 manifest={'inputs':[str(a.spectra),str(a.polar)],'passivity_tolerance':a.passivity_tolerance,'masked_wavelength_nm':audit.wavelength_nm.tolist()}
 (a.output_dir/'figure5_manifest.json').write_text(json.dumps(manifest,indent=2))
 print('Saved',a.output_dir,'masked:',audit.wavelength_nm.tolist())
if __name__=='__main__':main()
