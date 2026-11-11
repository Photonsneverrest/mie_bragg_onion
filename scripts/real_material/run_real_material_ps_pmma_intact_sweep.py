from __future__ import annotations
import argparse, os, platform, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np, pandas as pd
from real_material_common import *

def parse_csv_numbers(text, cast=float): return tuple(cast(x) for x in text.split(',') if x.strip())
def argparser():
 p=argparse.ArgumentParser(description='Dispersive PS/PMMA intact Bragg-onion sweep.')
 p.add_argument('--ps-file',required=True); p.add_argument('--pmma-file',required=True); p.add_argument('--output-dir',required=True)
 p.add_argument('--design-wavelengths-nm',default=','.join(map(str,DESIGNS_NM)))
 p.add_argument('--layer-counts',default=','.join(map(str,LAYER_COUNTS)))
 p.add_argument('--wavelength-min-nm',type=float,default=400); p.add_argument('--wavelength-max-nm',type=float,default=900)
 p.add_argument('--wavelength-step-nm',type=float,default=1); p.add_argument('--theta-points',type=int,default=721)
 p.add_argument('--collection-na',type=float,default=1.0); p.add_argument('--workers',type=int,default=8)
 p.add_argument('--manifest-only',action='store_true'); return p

def worker(t):
 out=Path(t['output_dir']); ck=out/'checkpoints'/f"task_{t['task_index']:04d}_{t['task_hash']}.pkl"
 if ck.exists(): return load_pair_payload(ck)['summary']
 wl=np.arange(t['wl_min'],t['wl_max']+0.5*t['wl_step'],t['wl_step'])
 kw=dict(ps_path=t['ps_file'],pmma_path=t['pmma_file'],design_nm=t['design_nm'],n_layers=t['n_layers'],wavelength_nm=wl,theta_points=t['theta_points'],collection_na=t['collection_na'])
 b=run_case(direction='backward',**kw); f=run_case(direction='forward',**kw)
 summary=summarize_pair(t,b,f); atomic_pickle(ck,{'task':t,'summary':summary,'backward':b,'forward':f}); return summary

def main():
 a=argparser().parse_args(); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
 designs=parse_csv_numbers(a.design_wavelengths_nm,float); layers=parse_csv_numbers(a.layer_counts,int)
 if min(designs)<400: raise ValueError('Material-realistic designs below 400 nm are intentionally excluded.')
 ps=parse_ri_info(a.ps_file,'PS'); pmma=parse_ri_info(a.pmma_file,'PMMA')
 wl=np.arange(a.wavelength_min_nm,a.wavelength_max_nm+0.5*a.wavelength_step_nm,a.wavelength_step_nm); validate_range((ps,pmma),wl)
 tasks=[]
 for i,(d,n) in enumerate((x for x in __import__('itertools').product(designs,layers))):
  base={'stage':'intact','design_nm':d,'n_layers':n,'ps_file':str(Path(a.ps_file).resolve()),'pmma_file':str(Path(a.pmma_file).resolve()),'output_dir':str(out.resolve()),'wl_min':a.wavelength_min_nm,'wl_max':a.wavelength_max_nm,'wl_step':a.wavelength_step_nm,'theta_points':a.theta_points,'collection_na':a.collection_na}
  base['task_index']=i; base['task_hash']=sha256_text(json.dumps(base,sort_keys=True)); tasks.append(base)
 pd.DataFrame(tasks).to_csv(out/'task_manifest_all.csv',index=False)
 json_dump(out/'run_metadata.json',{'argv':sys.argv,'python':sys.version,'platform':platform.platform(),'tasks':len(tasks),'workers':a.workers,'material_range_validated':True,'extrapolation':'error'})
 if a.manifest_only: print(f'Manifest only: {len(tasks)} tasks'); return
 rows=[]
 with ProcessPoolExecutor(max_workers=a.workers) as ex:
  futs={ex.submit(worker,t):t for t in tasks}
  for j,fut in enumerate(as_completed(futs),1): rows.append(fut.result()); print(f'{j}/{len(tasks)} complete',flush=True)
 df=pd.DataFrame(rows).sort_values(['design_nm','n_layers']); df.to_csv(out/'intact_summary.csv',index=False)
 try: df.to_parquet(out/'intact_summary.parquet',index=False)
 except Exception as e: print(f'Parquet skipped: {e}')
 print(f'Saved {out}')
if __name__=='__main__': main()
