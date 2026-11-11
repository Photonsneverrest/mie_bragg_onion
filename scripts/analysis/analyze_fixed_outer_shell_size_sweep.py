#!/usr/bin/env python3
"""Analysis of the fixed-ten-domain outer-reflector size sweep."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np,pandas as pd,matplotlib.pyplot as plt

def isolated(w,y,bad):
 z=np.asarray(y,float).copy();z[bad]=np.nan
 for i in np.where(~np.isfinite(z))[0]:
  if i>0 and i<len(z)-1 and np.isfinite(z[i-1]) and np.isfinite(z[i+1]):z[i]=np.interp(w[i],[w[i-1],w[i+1]],[z[i-1],z[i+1]])
 return z

def main():
 p=argparse.ArgumentParser();p.add_argument('--input-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,default=Path('outputs/figures/main/size_sweep'));p.add_argument('--style',type=Path,default=Path('styles/publication.mplstyle'));p.add_argument('--passivity-tolerance',type=float,default=1e-6);a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True);plt.style.use(a.style)
 summary=pd.read_csv(a.input_dir/'fixed_outer_shell_size_summary.csv');spectra=pd.read_csv(a.input_dir/'fixed_outer_shell_size_spectra.csv')
 back=summary[summary.collection_direction=='backward'].sort_values('diameter_um').copy();forward=summary[summary.collection_direction=='forward'][['diameter_um','spectrum_integral']].rename(columns={'spectrum_integral':'forward_integral'});back=back.merge(forward,on='diameter_um')
 records=[];audit=[];clean=[]
 for (D,direction),x in spectra.groupby(['diameter_um','direction']):
  x=x.sort_values('wavelength_nm').copy();bad=(~np.isfinite(x.q_collected_geom))|(~np.isfinite(x.qsca))|(x.qabs.abs()>a.passivity_tolerance)
  for w in x.loc[bad,'wavelength_nm']:audit.append({'diameter_um':D,'direction':direction,'wavelength_nm':w,'reason':f'lossless |Qabs|>{a.passivity_tolerance:g} or nonfinite'})
  for c in ['q_collected_geom','qsca','asymmetry_g']:x[c]=isolated(x.wavelength_nm.to_numpy(),x[c].to_numpy(),bad.to_numpy())
  clean.append(x)
  if direction=='backward':records.append({'diameter_um':D,'backward_integral_clean':np.trapezoid(x.q_collected_geom,x.wavelength_nm),'total_integral_clean':np.trapezoid(x.qsca,x.wavelength_nm),'g_visible_mean':np.nanmean(x.asymmetry_g),'g_470nm':np.interp(470,x.wavelength_nm,x.asymmetry_g)})
 metrics=back.merge(pd.DataFrame(records),on='diameter_um');metrics.to_csv(a.output_dir/'fixed_outer_shell_size_metrics.csv',index=False);pd.DataFrame(audit).to_csv(a.output_dir/'fixed_outer_shell_size_spike_audit.csv',index=False);clean=pd.concat(clean,ignore_index=True);clean.to_csv(a.output_dir/'fixed_outer_shell_size_spectra_plot_clean.csv',index=False)
 fig,ax=plt.subplots(2,2,figsize=(7.2,6),constrained_layout=True);A,B,C,D=ax.ravel()
 A.plot(metrics.diameter_um,metrics.q_collected_geom_max,'o-',label='Backward peak');A.set(xlabel='Outer diameter [µm]',ylabel='Backward-collected efficiency',title='(a) Backward response');A2=A.twinx();A2.plot(metrics.diameter_um,metrics.spectrum_integral,'s--',color='#D55E00',label='Backward integral');A2.set_ylabel('Backward spectral integral',color='#D55E00')
 B.plot(metrics.diameter_um,metrics.forward_integral,'o-',label='Forward integral');B.plot(metrics.diameter_um,metrics.total_integral_clean,'s-',label='Total integral');B.set(xlabel='Outer diameter [µm]',ylabel='Spectral integral',title='(b) Forward and total scattering');B.legend()
 C.plot(metrics.diameter_um,metrics.g_visible_mean,'o-',label='Visible mean');C.plot(metrics.diameter_um,metrics.g_470nm,'s--',label='At 470 nm');C.set(xlabel='Outer diameter [µm]',ylabel='Asymmetry parameter $g$',title='(c) Directionality');C.legend()
 for Dm in [5,10,20,40]:
  x=clean[(clean.direction=='backward')&np.isclose(clean.diameter_um,Dm)];
  if len(x):D.plot(x.wavelength_nm,x.q_collected_geom,label=f'{Dm:g} µm')
 D.set(xlim=(380,780),xlabel='Wavelength [nm]',ylabel='Backward-collected efficiency',title='(d) Size-dependent spectra');D.legend()
 for ext in ['pdf','png']:fig.savefig(a.output_dir/f'Figure_size_dependence_fixed_outer_shell.{ext}',dpi=600 if ext=='png' else None)
 plt.close(fig)
 # colour evidence table from solver-produced backward colour metrics
 cols=[c for c in ['diameter_um','L','a','b','C','hue_deg','eta_C','eta_L','hex','q_collected_geom_max','spectrum_integral'] if c in metrics.columns]
 metrics[cols].to_csv(a.output_dir/'Table_size_sweep_colour_metrics.csv',index=False)
 manifest={'inputs':[str(a.input_dir/'fixed_outer_shell_size_summary.csv'),str(a.input_dir/'fixed_outer_shell_size_spectra.csv')],'passivity_tolerance':a.passivity_tolerance,'outputs':{}}
 for f in a.output_dir.iterdir():
  if f.is_file():manifest['outputs'][f.name]=hashlib.sha256(f.read_bytes()).hexdigest()
 (a.output_dir/'size_sweep_manifest.json').write_text(json.dumps(manifest,indent=2));print('Saved',a.output_dir)
if __name__=='__main__':main()
