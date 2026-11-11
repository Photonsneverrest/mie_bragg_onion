#!/usr/bin/env python3
"""Fixed ten-domain outer reflector with a variable-radius air core.

Verified against scripts/run_hollow_absorption_sweep.py interface:
- uses run_bragg_onion_sweep directly with the same accepted keywords;
- uses LayerSubstitution(layer_indices=(0,), ...) for an air core;
- uses geometry_mode='peak_wavelength' and core_thickness_factor;
- does not forward any unverified wrapper-only CLI flags.

Geometry: n_layers=11, A outermost. Domain 0 is replaced by air and enlarged;
domains 1..10 are the fixed quarter-wave outer reflector (five A and five B).
The script verifies the returned diameter against the requested diameter and fails
rather than silently accepting different geometry semantics.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ.setdefault(key,'1')
import argparse,hashlib,inspect,json,pickle,sys,traceback
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np,pandas as pd
REPO_ROOT=Path(__file__).resolve().parents[2];SRC=REPO_ROOT/'src';sys.path.insert(0,str(SRC))
from bragg_onion.materials import ConstantDispersion
from bragg_onion.geometry import LayerSubstitution
from bragg_onion.sweep import run_bragg_onion_sweep,make_plotting_aliases

def csvnums(s,cast=float):return [cast(x.strip()) for x in s.split(',') if x.strip()]
def qwave(n,w):return w/(4*n)
def hsh(d):return hashlib.sha256(json.dumps(d,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:20]
def replacement(indices,material,name):return LayerSubstitution(layer_indices=tuple(indices),material=material,name=name)
def dump(obj,p):
 t=p.with_suffix(p.suffix+'.tmp')
 with t.open('wb') as f:pickle.dump(obj,f,pickle.HIGHEST_PROTOCOL);f.flush();os.fsync(f.fileno())
 os.replace(t,p)
def extract_arrays(sweep,direction,diameter):
 c=sweep.case_results[0];sc=c.scattering_result;wl=np.asarray(sc.wavelengths_m)*1e9
 integ=getattr(c,'integrated_result',None) or getattr(c,'collection_result',None)
 qcol=np.asarray(integ.q_collected_geom,float)
 g=np.asarray(getattr(sc,'g',np.full_like(wl,np.nan)),float)
 return pd.DataFrame({'diameter_um':diameter,'direction':direction,'wavelength_nm':wl,'q_collected_geom':qcol,'qsca':np.asarray(sc.qsca,float),'qext':np.asarray(sc.qext,float),'qabs':np.asarray(sc.qabs,float),'asymmetry_g':g})
def task_for(D,args):
 tA=qwave(args.n_A,args.design_wavelength_nm);tB=qwave(args.n_B,args.design_wavelength_nm)
 fixed_radial=5*(tA+tB);core_radius=1000*D/2-fixed_radial
 if core_radius<=0:raise ValueError(f'Diameter {D} um is <= fixed-shell diameter {2*fixed_radial/1000:.6f} um')
 d={'diameter_um':D,'core_radius_nm':core_radius,'core_thickness_factor':core_radius/tA,'design_wavelength_nm':args.design_wavelength_nm,'n_A':args.n_A,'n_B':args.n_B,'n_medium':args.n_medium,'outer_domains':10,'n_layers_model':11,'collection_na':args.collection_na}
 d['task_hash']=hsh(d);return d
def run_one(task,out,wl,theta):
 out=Path(out);stem=f"D{task['diameter_um']:07.3f}um_{task['task_hash']}";fp=out/'checkpoints'/f'{stem}.pkl'
 if fp.exists():return 'loaded',task['diameter_um']
 A=ConstantDispersion.from_nk('A',n=task['n_A'],k=0);B=ConstantDispersion.from_nk('B',n=task['n_B'],k=0);air=ConstantDispersion.from_nk('air_core',n=1,k=0)
 kwargs=dict(geometry_mode='peak_wavelength',parameter_grid={'peak_wavelength_m':[task['design_wavelength_nm']*1e-9],'n_layers':[11],'outer_layer':['A'],'core_thickness_factor':[task['core_thickness_factor']]},material_a=A,material_b=B,n_medium=task['n_medium'],wavelengths_m=np.asarray(wl)*1e-9,theta_rad=np.deg2rad(theta),collection_na=task['collection_na'],integration_quantity_for_colour='q_collected_geom',colour_wavelength_min_nm=None,colour_wavelength_max_nm=None,colour_normalize_input=False,colour_normalization='max',layer_substitutions=[replacement((0,),air,'variable_air_core')],store_full_results=True,progress=False)
 back=run_bragg_onion_sweep(collection_direction='backward',**kwargs);fwd=run_bragg_onion_sweep(collection_direction='forward',**kwargs)
 db=make_plotting_aliases(back.dataframe).copy();df=make_plotting_aliases(fwd.dataframe).copy()
 returned=float(db.iloc[0].diameter_nm)/1000
 if abs(returned-task['diameter_um'])>0.002:raise RuntimeError(f'Returned diameter {returned:.6f} um differs from target {task["diameter_um"]:.6f} um; core_thickness_factor semantics not accepted.')
 for frame,direction in [(db,'backward'),(df,'forward')]:
  frame['collection_direction']=direction
  for k,v in task.items():frame[k]=v
 spectra=pd.concat([extract_arrays(back,'backward',task['diameter_um']),extract_arrays(fwd,'forward',task['diameter_um'])],ignore_index=True)
 compact=pd.concat([db,df],ignore_index=True);dump({'task':task,'backward':back,'forward':fwd,'compact':compact,'spectra':spectra},fp)
 return 'calculated',task['diameter_um']
def main():
 p=argparse.ArgumentParser()
 p.add_argument('--output-dir',type=Path,default=Path('outputs/fixed_outer_shell_size_sweep_470nm'))
 p.add_argument('--diameters-um',default='5,7.5,10,12.5,15,20,25,30,40')
 p.add_argument('--design-wavelength-nm',type=float,default=470);p.add_argument('--n-A',type=float,default=1.55);p.add_argument('--n-B',type=float,default=1.0);p.add_argument('--n-medium',type=float,default=1.0)
 p.add_argument('--wavelength-min-nm',type=float,default=380);p.add_argument('--wavelength-max-nm',type=float,default=780);p.add_argument('--wavelength-step-nm',type=float,default=1);p.add_argument('--theta-points',type=int,default=721);p.add_argument('--collection-na',type=float,default=1.0);p.add_argument('--workers',type=int,default=8);p.add_argument('--manifest-only',action='store_true')
 a=p.parse_args();sig=inspect.signature(run_bragg_onion_sweep)
 required={'geometry_mode','parameter_grid','layer_substitutions','collection_direction','theta_rad','collection_na'};missing=sorted(required-set(sig.parameters))
 if missing:raise RuntimeError(f'Underlying run_bragg_onion_sweep lacks required parameters: {missing}')
 tasks=[task_for(D,a) for D in csvnums(a.diameters_um)]
 a.output_dir.mkdir(parents=True,exist_ok=True);(a.output_dir/'checkpoints').mkdir(exist_ok=True);(a.output_dir/'failures').mkdir(exist_ok=True)
 pd.DataFrame(tasks).to_csv(a.output_dir/'task_manifest_all.csv',index=False)
 (a.output_dir/'run_metadata.json').write_text(json.dumps(vars(a)|{'output_dir':str(a.output_dir),'verified_run_bragg_signature':str(sig)},indent=2,default=str))
 tA=qwave(a.n_A,a.design_wavelength_nm);tB=qwave(a.n_B,a.design_wavelength_nm)
 print(f'Fixed shell: 10 domains = 5 A + 5 B; radial thickness={5*(tA+tB):.6f} nm; minimum diameter={2*5*(tA+tB)/1000:.6f} um')
 print(pd.DataFrame(tasks).to_string(index=False));print('Manifest only:',a.manifest_only)
 if a.manifest_only:return
 wl=np.arange(a.wavelength_min_nm,a.wavelength_max_nm+.5*a.wavelength_step_nm,a.wavelength_step_nm);theta=np.linspace(0,180,a.theta_points)
 failures=[]
 with ProcessPoolExecutor(max_workers=min(a.workers,len(tasks))) as ex:
  fs={ex.submit(run_one,t,str(a.output_dir),wl,theta):t for t in tasks}
  for f in as_completed(fs):
   t=fs[f]
   try:print(*f.result(),flush=True)
   except Exception as e:failures.append(t|{'error_type':type(e).__name__,'error_message':str(e),'traceback':traceback.format_exc()})
 if failures:pd.DataFrame(failures).to_csv(a.output_dir/'failures'/'failed_tasks.csv',index=False);raise RuntimeError(f'{len(failures)} tasks failed')
 payloads=[]
 for fp in sorted((a.output_dir/'checkpoints').glob('*.pkl')):
  with fp.open('rb') as f:payloads.append(pickle.load(f))
 pd.concat([x['compact'] for x in payloads],ignore_index=True).to_csv(a.output_dir/'fixed_outer_shell_size_summary.csv',index=False)
 pd.concat([x['spectra'] for x in payloads],ignore_index=True).to_csv(a.output_dir/'fixed_outer_shell_size_spectra.csv',index=False)
 print('Saved',a.output_dir)
if __name__=='__main__':main()
