#!/usr/bin/env python3
"""Pointed validation sweep for multilayer-sphere near-field stability.
Creates homogeneous/index-matched controls and weak/high-contrast comparisons.
Does not overwrite project sources.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--repo-root',type=Path,default=Path.cwd())
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--npts',type=int,nargs='+',default=[201,401,801])
    ap.add_argument('--wavelengths-nm',type=float,nargs='+',default=[416,418,420,422,424,470])
    ap.add_argument('--max-layers',type=int,default=15)
    ap.add_argument('--use-parquet',action='store_true')
    args=ap.parse_args(); sys.path.insert(0,str(args.repo_root/'src'))
    from bragg_onion.materials import ConstantDispersion
    from bragg_onion.geometry import build_bragg_onion_from_peak_wavelength, resolve_layer_stack, ExtinctionModifier
    from bragg_onion.fields import compute_field_map
    from bragg_onion.solver import run_scattnlay_spectrum

    out=args.output_dir; out.mkdir(parents=True,exist_ok=True)
    rows=[]; profiles=[]
    cases=[]
    for nl in range(1,args.max_layers+1):
        for nh in (1.0,1.001,1.01,1.5,2.5):
            for k in (0.0,0.01):
                cases.append((nl,nh,k))
    for ci,(nl,nh,k) in enumerate(cases):
        A=ConstantDispersion.from_nk(f'A_n{nh:g}_k{k:g}',n=nh,k=k)
        B=ConstantDispersion.from_nk(f'B_n1_k{k:g}',n=1.0,k=k)
        geom=build_bragg_onion_from_peak_wavelength(material_a=A,material_b=B,peak_wavelength_m=470e-9,n_layers=nl,outer_layer='A',core_thickness_factor=1.0)
        stack=resolve_layer_stack(geometry=geom,material_a=A,material_b=B)
        # Confirm whether identical interfaces remain separate.
        nk=stack.refractive_indices_at_wavelength(470e-9)
        identical_adjacent=int(np.sum(np.isclose(nk[1:],nk[:-1])))
        spec=run_scattnlay_spectrum(stack=stack,wavelengths_m=np.asarray(args.wavelengths_nm)*1e-9,n_medium=1.0,theta_rad=np.array([0.0,np.pi]))
        for npts in args.npts:
            for wi,w in enumerate(args.wavelengths_nm):
                fm=compute_field_map(stack=stack,wavelength_m=w*1e-9,n_medium=1.0,plane='xz',npts=npts,extent_outer_radius_factor=1.001,incident_e0=1+0j)
                j=int(np.argmin(abs(fm.coord1_m))); z=fm.coord2_m/geom.outer_radius_m; y=np.asarray(fm.e_total_mag[:,j],float)
                finite=np.isfinite(y); good=finite&(y>=0)&(y<=1e6)
                logy=np.log10(np.where(finite&(y>0),y,np.nan)); jumps=np.abs(np.diff(logy))
                rows.append(dict(case_index=ci,n_layers=nl,n_high=nh,k=k,wavelength_nm=w,npts=npts,outer_radius_nm=geom.outer_radius_m*1e9,terms=int(np.asarray(spec.terms)[wi]),qext=float(spec.qext[wi]),qsca=float(spec.qsca[wi]),qabs=float(spec.qabs[wi]),energy_residual=float(abs(spec.qext[wi]-spec.qsca[wi]-spec.qabs[wi])),identical_adjacent_layers=identical_adjacent,finite_fraction=float(finite.mean()),valid_fraction=float(good.mean()),raw_max=float(np.nanmax(y)),max_log10_jump=float(np.nanmax(jumps)) if np.isfinite(jumps).any() else np.nan,zR_at_max=float(z[np.nanargmax(y)])))
                profiles.append(pd.DataFrame({'case_index':ci,'n_layers':nl,'n_high':nh,'k':k,'wavelength_nm':w,'npts':npts,'sample_index':np.arange(npts),'z_over_R':z,'E_magnitude':y}))
                print(f'[{ci+1}/{len(cases)}] L={nl} nH={nh:g} k={k:g} w={w:g} npts={npts} max={np.nanmax(y):.3g}',flush=True)
    metrics=pd.DataFrame(rows); prof=pd.concat(profiles,ignore_index=True)
    metrics.to_csv(out/'validation_metrics.csv',index=False); prof.to_pickle(out/'validation_axis_profiles.pkl')
    if args.use_parquet:
        metrics.to_parquet(out/'validation_metrics.parquet',index=False); prof.to_parquet(out/'validation_axis_profiles.parquet',index=False)
    (out/'validation_metadata.json').write_text(json.dumps(vars(args)|{'repo_root':str(args.repo_root),'output_dir':str(out)},indent=2,default=str))
if __name__=='__main__': main()
