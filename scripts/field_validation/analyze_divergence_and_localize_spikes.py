#!/usr/bin/env python3
"""Automated raw-profile divergence and spike localization diagnostic."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd

def load_table(path):
    return pd.read_pickle(path) if path.suffix in {'.pkl','.pickle'} else pd.read_parquet(path)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--axis-profiles',type=Path,required=True); ap.add_argument('--output-dir',type=Path,required=True); ap.add_argument('--absolute-limit',type=float,default=1e6); ap.add_argument('--jump-decades',type=float,default=4.0); args=ap.parse_args()
    out=args.output_dir; out.mkdir(parents=True,exist_ok=True); a=load_table(args.axis_profiles)
    records=[]; spikes=[]
    for keys,g in a.groupby(['case_label','wavelength_nm'],sort=False):
        g=g.sort_values('z_nm').reset_index(drop=True); y=pd.to_numeric(g.E_magnitude_enhancement,errors='coerce').to_numpy(float); z=g.z_over_outer_radius.to_numpy(float); finite=np.isfinite(y); pos=finite&(y>0); log=np.log10(np.where(pos,y,np.nan)); jump=np.abs(np.diff(log)); imax=int(np.nanargmax(y)) if finite.any() else -1
        bad=(~finite)|(np.abs(y)>args.absolute_limit); jump_idx=np.where(jump>args.jump_decades)[0]
        records.append(dict(case_label=keys[0],wavelength_nm=keys[1],samples=len(g),finite_fraction=finite.mean(),divergent_fraction=bad.mean(),raw_max=np.nanmax(y) if finite.any() else np.nan,zR_at_max=z[imax] if imax>=0 else np.nan,index_at_max=imax,max_log10_jump=np.nanmax(jump) if np.isfinite(jump).any() else np.nan,n_large_jumps=len(jump_idx),spike_ratio=(np.nanmax(y)/np.nanpercentile(y[finite],95)) if finite.any() else np.nan))
        candidate=np.unique(np.r_[np.where(bad)[0],jump_idx,jump_idx+1])
        for i in candidate:
            if i<0 or i>=len(g): continue
            r=float(g.radius_nm.iloc[i]); shell=g.shell_number_one_based.iloc[i] if 'shell_number_one_based' in g else np.nan
            # nearest boundary inferred from changes of shell index along the sampled axis
            boundary=np.where(g.shell_number_one_based.to_numpy()[1:]!=g.shell_number_one_based.to_numpy()[:-1])[0] if 'shell_number_one_based' in g else np.array([],int)
            db=int(np.min(abs(boundary-i))) if len(boundary) else -1
            spikes.append(dict(case_label=keys[0],wavelength_nm=keys[1],sample_index=int(i),z_over_R=z[i],radius_nm=r,shell_number_one_based=shell,E_magnitude=y[i],log10_E=log[i],distance_to_nearest_sampled_shell_transition=db,is_origin_sample=bool(np.isclose(z[i],0,atol=1e-12))))
    pd.DataFrame(records).to_csv(out/'profile_quality_and_spike_metrics.csv',index=False); pd.DataFrame(spikes).to_csv(out/'localized_spikes.csv',index=False)
    print('Wrote',out)
if __name__=='__main__': main()
