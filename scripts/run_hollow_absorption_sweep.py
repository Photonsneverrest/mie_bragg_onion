#!/usr/bin/env python3
from __future__ import annotations
"""Two-stage hollow/absorbing Bragg-onion sweep for forward-scattering suppression.

Requires bragg_onion to expose LayerSubstitution from bragg_onion.geometry and
accept layer_substitutions=... in run_bragg_onion_sweep. Layer indices are
0-based from the core. See the preflight error for the required source API.
"""
import os
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(key,"1")
import argparse, hashlib, inspect, json, pickle, sys, traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd

REPO_ROOT=Path(__file__).resolve().parents[1]
SRC_DIR=REPO_ROOT/'src'
if str(SRC_DIR) not in sys.path: sys.path.insert(0,str(SRC_DIR))
from bragg_onion.materials import ConstantDispersion
from bragg_onion.sweep import run_bragg_onion_sweep, make_plotting_aliases
try:
    from bragg_onion.geometry import LayerSubstitution
except ImportError as exc:
    raise ImportError(
        "This sweep needs LayerSubstitution in bragg_onion.geometry. Upload "
        "src/bragg_onion/geometry.py and src/bragg_onion/sweep.py so the small "
        "source adaptation can be matched exactly to your current repository."
    ) from exc

CACHE_VERSION='hollow_absorption_v1'

def csv_numbers(text, cast=float):
    return [cast(x.strip()) for x in text.split(',') if x.strip()]

def stable_hash(d):
    return hashlib.sha256(json.dumps(d,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:20]

def qwave(n, wavelength_nm): return wavelength_nm/(4.0*n)

def expected_diameter_nm(n_layers, t_a, t_b, outer='A'):
    # With even N and A outermost, the core is B and there are N/2 of each.
    if n_layers%2 or outer!='A': raise ValueError('Reference requires even N and A outermost.')
    return 2.0*(n_layers//2)*(t_a+t_b)

def replacement(indices, material, name):
    """Construct against the intended minimal source API."""
    return LayerSubstitution(layer_indices=tuple(int(i) for i in indices), material=material, name=name)

def make_tasks(args):
    hollow_counts=csv_numbers(args.hollow_layers,int)
    if args.include_all_hollow: hollow_counts=sorted(set(hollow_counts+[0,args.n_layers]))
    if any(x<0 or x>args.n_layers for x in hollow_counts): raise ValueError('hollow count outside 0..N')
    absorber_counts=csv_numbers(args.absorber_layer_counts,int)
    k_values=csv_numbers(args.absorber_k,float)
    tasks=[]
    for h in hollow_counts:
        tasks.append(dict(stage='hollowing',hollow_layers=h,absorber_mode='none',absorber_layers=0,absorber_n=0.0,absorber_k=0.0))
    if args.selected_hollow_layers is not None:
        h=args.selected_hollow_layers
        if not 0<=h<=args.n_layers: raise ValueError('selected hollow count outside 0..N')
        for k in k_values:
            tasks.append(dict(stage='absorption',hollow_layers=h,absorber_mode='fill_hollow',absorber_layers=0,absorber_n=args.absorber_n,absorber_k=k))
            for m in absorber_counts:
                if m<0 or h+m>args.n_layers: raise ValueError(f'absorber layer count {m} invalid for hollow count {h}')
                tasks.append(dict(stage='absorption',hollow_layers=h,absorber_mode='solid_layers_outward',absorber_layers=m,absorber_n=args.absorber_n,absorber_k=k))
    common=dict(cache_version=CACHE_VERSION,design_wavelength_nm=args.design_wavelength_nm,n_A=args.n_A,n_B=args.n_B,
                n_layers=args.n_layers,outer_layer='A',n_medium=args.n_medium,
                t_a_nm=qwave(args.n_A,args.design_wavelength_nm),t_b_nm=qwave(args.n_B,args.design_wavelength_nm))
    for t in tasks:
        t.update(common); t['task_hash']=stable_hash(t)
    tasks=sorted(tasks,key=lambda x:(x['stage'],x['hollow_layers'],x['absorber_mode'],x['absorber_layers'],x['absorber_k']))
    for i,t in enumerate(tasks): t['task_index']=i
    return tasks

def substitutions_for(task):
    h,N=task['hollow_layers'],task['n_layers']
    air=ConstantDispersion.from_nk('hollow_air',n=1.0,k=0.0)
    subs=[]
    if h:
        hollow_material=air
        if task['absorber_mode']=='fill_hollow':
            hollow_material=ConstantDispersion.from_nk('absorbing_hollow_fill',n=task['absorber_n'],k=task['absorber_k'])
        subs.append(replacement(range(0,h),hollow_material,'hollow_volume'))
    if task['absorber_mode']=='solid_layers_outward' and task['absorber_layers']:
        absorber=ConstantDispersion.from_nk('absorbing_solid_layers',n=task['absorber_n'],k=task['absorber_k'])
        subs.append(replacement(range(h,h+task['absorber_layers']),absorber,'absorber_from_interface'))
    return subs or None

def compact(sweep,task,direction):
    df=make_plotting_aliases(sweep.dataframe).copy()
    if len(df)!=1: raise RuntimeError(f'Expected one row, got {len(df)}')
    for k,v in task.items(): df[k]=v
    df['collection_direction']=direction
    df['hollow_fraction_layers']=task['hollow_layers']/task['n_layers']
    return df

def dump_atomic(obj,path):
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('wb') as f: pickle.dump(obj,f,pickle.HIGHEST_PROTOCOL); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)

def run_task(task,out,wavelengths_nm,theta_deg,na):
    out=Path(out); stem=f"task_{task['task_index']:04d}_{task['task_hash']}"
    full=out/'full_checkpoints'/f'{stem}.pkl'; small=out/'compact_checkpoints'/f'{stem}.pkl'
    if full.exists() and small.exists(): return 'loaded'
    A=ConstantDispersion.from_nk('A',n=task['n_A'],k=0.0); B=ConstantDispersion.from_nk('B',n=task['n_B'],k=0.0)
    kwargs=dict(geometry_mode='peak_wavelength',parameter_grid={'peak_wavelength_m':[task['design_wavelength_nm']*1e-9],
        'n_layers':[task['n_layers']],'outer_layer':['A'],'core_thickness_factor':[1.0]},material_a=A,material_b=B,
        n_medium=task['n_medium'],wavelengths_m=np.asarray(wavelengths_nm)*1e-9,theta_rad=np.deg2rad(theta_deg),
        collection_na=na,integration_quantity_for_colour='q_collected_geom',colour_wavelength_min_nm=None,
        colour_wavelength_max_nm=None,colour_normalize_input=False,colour_normalization='max',
        layer_substitutions=substitutions_for(task),store_full_results=True,progress=False)
    back=run_bragg_onion_sweep(collection_direction='backward',**kwargs)
    fwd=run_bragg_onion_sweep(collection_direction='forward',**kwargs)
    df=pd.concat([compact(back,task,'backward'),compact(fwd,task,'forward')],ignore_index=True)
    dump_atomic({'task':task,'backward':back,'forward':fwd,'compact':df},full); df.to_pickle(small)
    return 'calculated'

def combine(out):
    parts=sorted((out/'compact_checkpoints').glob('task_*.pkl'))
    if not parts: raise FileNotFoundError('No compact checkpoints found')
    df=pd.concat([pd.read_pickle(p) for p in parts],ignore_index=True)
    df=df.sort_values(['stage','hollow_layers','absorber_mode','absorber_layers','absorber_k','collection_direction']).reset_index(drop=True)
    df.to_pickle(out/'hollow_absorption_summary.pkl'); df.to_csv(out/'hollow_absorption_summary.csv',index=False)
    try: df.to_parquet(out/'hollow_absorption_summary.parquet',index=False)
    except Exception: pass
    return df

def preflight(args,tasks):
    sig=inspect.signature(run_bragg_onion_sweep)
    if 'layer_substitutions' not in sig.parameters:
        raise RuntimeError('run_bragg_onion_sweep lacks layer_substitutions. Upload geometry.py and sweep.py for an exact patch.')
    tA=qwave(args.n_A,args.design_wavelength_nm); tB=qwave(args.n_B,args.design_wavelength_nm)
    D=expected_diameter_nm(args.n_layers,tA,tB)
    print(f'Quarter-wave thicknesses: A={tA:.6f} nm, B={tB:.6f} nm')
    print(f'Predicted outer diameter: {D/1000:.6f} um ({D:.3f} nm)')
    print(f'Tasks: {len(tasks)}; LayerSubstitution={inspect.signature(LayerSubstitution)}')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output-dir',type=Path,default=Path('/mnt/d/Rocca/MieScattering/outputs/hollow_absorption_470nm'))
    p.add_argument('--design-wavelength-nm',type=float,default=470.0); p.add_argument('--n-A',type=float,default=1.55); p.add_argument('--n-B',type=float,default=1.0)
    p.add_argument('--n-layers',type=int,default=120); p.add_argument('--n-medium',type=float,default=1.0)
    p.add_argument('--hollow-layers',default='0,10,20,30,40,50,60,70,80,90,100,110,120'); p.add_argument('--include-all-hollow',action='store_true')
    p.add_argument('--selected-hollow-layers',type=int); p.add_argument('--absorber-layer-counts',default='1,2,4,8,12,16,20')
    p.add_argument('--absorber-n',type=float,default=1.55); p.add_argument('--absorber-k',default='0.0001,0.0003,0.001,0.003,0.01,0.03,0.1')
    p.add_argument('--workers',type=int,default=8); p.add_argument('--wavelength-min-nm',type=float,default=380); p.add_argument('--wavelength-max-nm',type=float,default=780)
    p.add_argument('--wavelength-step-nm',type=float,default=1); p.add_argument('--theta-points',type=int,default=721); p.add_argument('--collection-na',type=float,default=1.0)
    p.add_argument('--task-mod',type=int,default=1); p.add_argument('--task-rem',type=int,default=0); p.add_argument('--manifest-only',action='store_true')
    a=p.parse_args()
    if a.n_layers<1 or a.workers<1 or a.theta_points<3 or a.wavelength_step_nm<=0: raise ValueError('Invalid positive numerical setting')
    if a.task_mod<1 or not 0<=a.task_rem<a.task_mod: raise ValueError('Require task_mod>=1 and 0<=task_rem<task_mod')
    tasks_all=make_tasks(a); preflight(a,tasks_all)
    out=a.output_dir
    for d in (out,out/'full_checkpoints',out/'compact_checkpoints',out/'failures'): d.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(tasks_all).to_csv(out/'task_manifest_all.csv',index=False)
    tasks=[t for t in tasks_all if t['task_index']%a.task_mod==a.task_rem]
    pd.DataFrame(tasks).to_csv(out/f'task_manifest_mod{a.task_mod}_rem{a.task_rem}.csv',index=False)
    (out/'run_metadata.json').write_text(json.dumps(vars(a)|{'output_dir':str(out),'cache_version':CACHE_VERSION},indent=2,default=str))
    if a.manifest_only: return
    wl=np.arange(a.wavelength_min_nm,a.wavelength_max_nm+0.5*a.wavelength_step_nm,a.wavelength_step_nm).tolist(); th=np.linspace(0,180,a.theta_points).tolist()
    failures=[]; start=perf_counter()
    with ProcessPoolExecutor(max_workers=min(a.workers,max(1,len(tasks)))) as ex:
        fs={ex.submit(run_task,t,str(out),wl,th,a.collection_na):t for t in tasks}
        for f in as_completed(fs):
            t=fs[f]
            try: print(t['task_index'],f.result())
            except Exception as e: failures.append(t|{'error_type':type(e).__name__,'error_message':str(e),'traceback':traceback.format_exc()})
    if failures: pd.DataFrame(failures).to_csv(out/'failures'/'failed_tasks.csv',index=False)
    df=combine(out); print(f'Rows={len(df)}; hours={(perf_counter()-start)/3600:.2f}')
    if failures: raise RuntimeError(f'{len(failures)} tasks failed; inspect failures/failed_tasks.csv')
if __name__=='__main__': main()
