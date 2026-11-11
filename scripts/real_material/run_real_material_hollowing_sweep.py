from __future__ import annotations
import argparse, json, platform, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np, pandas as pd
from real_material_common import *
def argparser():
 p=argparse.ArgumentParser(description='Progressive inner-air substitution for selected PS/PMMA optima.')
 p.add_argument('--optima',required=True); p.add_argument('--output-dir',required=True); p.add_argument('--workers',type=int,default=8)
 p.add_argument('--wavelength-min-nm',type=float,default=400); p.add_argument('--wavelength-max-nm',type=float,default=900); p.add_argument('--wavelength-step-nm',type=float,default=1)
 p.add_argument('--theta-points',type=int,default=721); p.add_argument('--collection-na',type=float,default=1); p.add_argument('--manifest-only',action='store_true'); return p

def worker(t):
 out=Path(t['output_dir']); ck=out/'checkpoints'/f"task_{t['task_index']:05d}_{t['task_hash']}.pkl"
 if ck.exists(): return load_pair_payload(ck)['summary']
 air=ConstantDispersion.from_nk('air',1,0); sub=None
 if t['hollow_layers']>0: sub=LayerSubstitution(tuple(range(t['hollow_layers'])),air,'inner_air')
 wl=np.arange(t['wl_min'],t['wl_max']+0.5*t['wl_step'],t['wl_step'])
 kw=dict(ps_path=t['ps_file'],pmma_path=t['pmma_file'],design_nm=t['design_nm'],n_layers=t['n_layers'],wavelength_nm=wl,theta_points=t['theta_points'],collection_na=t['collection_na'],substitutions=sub)
 b=run_case(direction='backward',**kw); f=run_case(direction='forward',**kw); s=summarize_pair(t,b,f)
 atomic_pickle(ck,{'task':t,'summary':s,'backward':b,'forward':f}); return s

def main():
 a=argparser().parse_args(); opt=pd.read_csv(a.optima); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); tasks=[]
 for _,r in opt.iterrows():
  n=int(r.n_layers)
  for h in range(0,n+1):
   base={'stage':'hollowing','design_nm':float(r.design_nm),'n_layers':n,'hollow_layers':h,'remaining_layers':n-h,'ps_file':r.ps_file,'pmma_file':r.pmma_file,'output_dir':str(out.resolve()),'wl_min':a.wavelength_min_nm,'wl_max':a.wavelength_max_nm,'wl_step':a.wavelength_step_nm,'theta_points':a.theta_points,'collection_na':a.collection_na}
   base['task_index']=len(tasks); base['task_hash']=sha256_text(json.dumps(base,sort_keys=True)); tasks.append(base)
 pd.DataFrame(tasks).to_csv(out/'task_manifest_all.csv',index=False); json_dump(out/'run_metadata.json',{'argv':sys.argv,'tasks':len(tasks),'workers':a.workers})
 if a.manifest_only: print(f'Manifest only: {len(tasks)} tasks'); return
 rows=[]
 with ProcessPoolExecutor(max_workers=a.workers) as ex:
  futs={ex.submit(worker,t):t for t in tasks}
  for j,f in enumerate(as_completed(futs),1): rows.append(f.result()); print(f'{j}/{len(tasks)} complete',flush=True)
 df=pd.DataFrame(rows).sort_values(['design_nm','hollow_layers']); df.to_csv(out/'hollowing_summary.csv',index=False)
 # Compute eligibility against h=0 reference. Spectral correlation and deltaE from checkpoints.
 metrics=[]
 for design,g in df.groupby('design_nm'):
  ref=g[g.hollow_layers==0].iloc[0]; refp=load_pair_payload(out/'checkpoints'/f"task_{int(ref.task_index):05d}_{ref.task_hash}.pkl"); refs=payload_spectra(refp)
  for _,r in g.iterrows():
   pay=load_pair_payload(out/'checkpoints'/f"task_{int(r.task_index):05d}_{r.task_hash}.pkl"); sp=payload_spectra(pay)
   corr=float(np.corrcoef(refs.backward,sp.backward)[0,1]); de=delta_e76(r,ref); cp=(float(r.eta_C)*float(r.eta_L))/(float(ref.eta_C)*float(ref.eta_L))
   metrics.append({**r.to_dict(),'backward_integral_retention':r.backward_integral/ref.backward_integral,'backward_peak_retention':r.backward_peak/ref.backward_peak,'spectral_correlation':corr,'delta_E76':de,'colour_performance_retention':cp})
 met=pd.DataFrame(metrics); met['eligible']=(met.backward_integral_retention>=.90)&(met.spectral_correlation>=.98)&(met.delta_E76<=5)&(met.colour_performance_retention>=.90)
 met.to_csv(out/'hollowing_metrics.csv',index=False)
 selected=[]
 for design,g in met.groupby('design_nm'):
  q=g[g.eligible].sort_values(['remaining_layers','hollow_layers'],ascending=[True,False]); selected.append((q.iloc[0] if len(q) else g[g.hollow_layers==0].iloc[0]))
 pd.DataFrame(selected).to_csv(out/'selected_hollow_geometries.csv',index=False)
 print('Saved',out)
if __name__=='__main__': main()
