from __future__ import annotations
import argparse, json, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np, pandas as pd
from real_material_common import *
def nums(s): return [float(x) for x in s.split(',') if x.strip()]
def worker(t):
 out=Path(t['output_dir']); ck=out/'checkpoints'/f"task_{t['task_index']:04d}_{t['task_hash']}.pkl"
 if ck.exists(): return load_pair_payload(ck)['summary']
 fill=ConstantDispersion.from_nk('melanin_like_fill',t['fill_n'],t['fill_k']); sub=LayerSubstitution(tuple(range(t['hollow_layers'])),fill,'melanin_like_fill') if t['hollow_layers'] else None
 wl=np.arange(t['wl_min'],t['wl_max']+.5*t['wl_step'],t['wl_step']); kw=dict(ps_path=t['ps_file'],pmma_path=t['pmma_file'],design_nm=t['design_nm'],n_layers=t['n_layers'],wavelength_nm=wl,theta_points=t['theta_points'],collection_na=t['collection_na'],substitutions=sub)
 b=run_case(direction='backward',**kw); f=run_case(direction='forward',**kw); s=summarize_pair(t,b,f); atomic_pickle(ck,{'task':t,'summary':s,'backward':b,'forward':f}); return s
def main():
 p=argparse.ArgumentParser(description='Melanin-like fill sweep for selected 470 and 580 nm PS/PMMA hollow geometries.')
 p.add_argument('--selected-hollow-geometries',required=True); p.add_argument('--output-dir',required=True); p.add_argument('--fill-n',type=float,default=1.75); p.add_argument('--fill-k',default='0,0.001,0.003,0.01,0.03,0.1,0.3')
 p.add_argument('--wavelength-min-nm',type=float,default=400); p.add_argument('--wavelength-max-nm',type=float,default=900); p.add_argument('--wavelength-step-nm',type=float,default=1); p.add_argument('--theta-points',type=int,default=721); p.add_argument('--collection-na',type=float,default=1); p.add_argument('--workers',type=int,default=8); p.add_argument('--manifest-only',action='store_true'); a=p.parse_args()
 sel=pd.read_csv(a.selected_hollow_geometries); sel=sel[np.isclose(sel.design_nm,470)|np.isclose(sel.design_nm,580)]; out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); tasks=[]
 for _,r in sel.iterrows():
  for k in nums(a.fill_k):
   base={'stage':'melanin_fill','design_nm':float(r.design_nm),'n_layers':int(r.n_layers),'hollow_layers':int(r.hollow_layers),'fill_n':a.fill_n,'fill_k':k,'ps_file':r.ps_file,'pmma_file':r.pmma_file,'output_dir':str(out.resolve()),'wl_min':a.wavelength_min_nm,'wl_max':a.wavelength_max_nm,'wl_step':a.wavelength_step_nm,'theta_points':a.theta_points,'collection_na':a.collection_na}
   base['task_index']=len(tasks); base['task_hash']=sha256_text(json.dumps(base,sort_keys=True)); tasks.append(base)
 pd.DataFrame(tasks).to_csv(out/'task_manifest_all.csv',index=False); json_dump(out/'run_metadata.json',{'argv':sys.argv,'tasks':len(tasks),'fill_n':a.fill_n,'citation_keys':['Stavenga2015MelaninRefractiveIndex','Li2020SyntheticMelaninOpticalConstants']})
 if a.manifest_only: print(f'Manifest only: {len(tasks)} tasks'); return
 rows=[]
 with ProcessPoolExecutor(max_workers=a.workers) as ex:
  fs={ex.submit(worker,t):t for t in tasks}
  for j,f in enumerate(as_completed(fs),1): rows.append(f.result()); print(f'{j}/{len(tasks)} complete',flush=True)
 pd.DataFrame(rows).sort_values(['design_nm','fill_k']).to_csv(out/'melanin_fill_summary.csv',index=False); print('Saved',out)
if __name__=='__main__': main()
