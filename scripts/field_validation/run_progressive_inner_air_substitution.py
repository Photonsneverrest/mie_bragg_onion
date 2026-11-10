#!/usr/bin/env python3
"""60-layer nL=1.0, nH=1.5, k=0 progressive inner-air substitution.
Preserves the original radii and outer radius. Generates single-layer and bilayer series.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--repo-root',type=Path,default=Path.cwd()); ap.add_argument('--output-dir',type=Path,required=True); ap.add_argument('--wavelength-min-nm',type=float,default=300); ap.add_argument('--wavelength-max-nm',type=float,default=900); ap.add_argument('--wavelength-step-nm',type=float,default=1); ap.add_argument('--theta-points',type=int,default=181); ap.add_argument('--collection-na',type=float,default=1.0); args=ap.parse_args(); sys.path.insert(0,str(args.repo_root/'src'))
 from bragg_onion.materials import ConstantDispersion
 from bragg_onion.geometry import build_bragg_onion_from_peak_wavelength, resolve_layer_stack, ResolvedLayerStack
 from bragg_onion.solver import run_scattnlay_spectrum
 A=ConstantDispersion.from_nk('A_n1.5',n=1.5,k=0); B=ConstantDispersion.from_nk('B_air',n=1.0,k=0); air=ConstantDispersion.from_nk('substituted_air',n=1.0,k=0)
 geom=build_bragg_onion_from_peak_wavelength(material_a=A,material_b=B,peak_wavelength_m=470e-9,n_layers=60,outer_layer='A',core_thickness_factor=1.0); base=resolve_layer_stack(geometry=geom,material_a=A,material_b=B)
 wl=np.arange(args.wavelength_min_nm,args.wavelength_max_nm+0.5*args.wavelength_step_nm,args.wavelength_step_nm); theta=np.linspace(0,np.pi,args.theta_points)
 # Single layers 0..59 preserves outer A. Bilayers 0,2,..,58. Include 60 only as all-air control.
 tasks=[('single_layer',n) for n in range(60)]+[('complete_bilayer',n) for n in range(0,60,2)]+[('all_air_control',60)]
 rows=[]; summary=[]
 for mode,n_air in tasks:
  disps=tuple(air if i<n_air else d for i,d in enumerate(base.layer_dispersions)); names=tuple('substituted_air' if i<n_air else n for i,n in enumerate(base.layer_material_names)); stack=ResolvedLayerStack(geometry=geom,layer_material_names=names,layer_dispersions=disps)
  res=run_scattnlay_spectrum(stack=stack,wavelengths_m=wl*1e-9,n_medium=1.0,theta_rad=theta)
  # backward hemisphere raw angular mean proxy plus exact qbk retained
  s1=np.asarray(res.s1); s2=np.asarray(res.s2); back=theta>=np.pi/2; intensity=(abs(s1)**2+abs(s2)**2)/2; collected=np.trapz(intensity[:,back]*np.sin(theta[back]),theta[back],axis=1)
  inner_air_radius_nm=(geom.outer_radii_m[n_air-1]*1e9) if n_air else 0.0
  for i,w in enumerate(wl): rows.append(dict(series=mode,n_air_layers=n_air,active_layers=60-n_air,inner_air_radius_nm=inner_air_radius_nm,wavelength_nm=w,qext=res.qext[i],qsca=res.qsca[i],qabs=res.qabs[i],qbk=res.qbk[i],backward_collected_proxy=collected[i],terms=int(np.asarray(res.terms)[i])))
  summary.append(dict(series=mode,n_air_layers=n_air,active_layers=60-n_air,inner_air_radius_nm=inner_air_radius_nm,outer_radius_nm=geom.outer_radius_m*1e9,peak_qbk_wavelength_nm=float(wl[np.nanargmax(res.qbk)]),peak_qbk=float(np.nanmax(res.qbk))))
  print(mode,n_air,flush=True)
 out=args.output_dir; out.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_pickle(out/'hollowing_spectra.pkl'); pd.DataFrame(rows).to_csv(out/'hollowing_spectra.csv',index=False); pd.DataFrame(summary).to_csv(out/'hollowing_summary.csv',index=False); (out/'hollowing_metadata.json').write_text(json.dumps(vars(args)|{'repo_root':str(args.repo_root),'outer_radius_nm':geom.outer_radius_m*1e9},indent=2,default=str))
if __name__=='__main__': main()
