#!/usr/bin/env python3
"""Analyze 580 nm low-contrast hollowing with backward, forward and total scattering."""
from __future__ import annotations
import argparse,pickle,json
from pathlib import Path
import numpy as np,pandas as pd,matplotlib.pyplot as plt

def load_summary(d):
 for name in ['hollow_absorption_summary.pkl','hollow_absorption_summary.csv']:
  p=d/name
  if p.exists(): return pd.read_pickle(p) if p.suffix=='.pkl' else pd.read_csv(p)
 raise FileNotFoundError(d)
def payload(d,h):
 fs=list((d/'full_checkpoints').glob(f'task_*_{h}.pkl'))
 if not fs: raise FileNotFoundError(h)
 with fs[0].open('rb') as f:return pickle.load(f)
def arrays(sweep):
 c=sweep.case_results[0];sc=c.scattering_result;wl=np.asarray(sc.wavelengths_m)*1e9
 integ=getattr(c,'integrated_result',None) or getattr(c,'collection_result',None)
 return wl,np.asarray(integ.q_collected_geom,float),np.asarray(sc.qsca,float),np.asarray(sc.qabs,float)
def isolated_clean(wl,y,bad):
 z=y.copy();z[bad]=np.nan
 for i in np.where(np.isnan(z))[0]:
  if i>0 and i<len(z)-1 and np.isfinite(z[i-1]) and np.isfinite(z[i+1]):z[i]=np.interp(wl[i],[wl[i-1],wl[i+1]],[z[i-1],z[i+1]])
 return z

def main():
 p=argparse.ArgumentParser();p.add_argument('--input-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--style',type=Path);p.add_argument('--passivity-tolerance',type=float,default=1e-6);a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
 if a.style:plt.style.use(a.style)
 df=load_summary(a.input_dir);tasks=df[df.stage.eq('hollowing')].drop_duplicates('task_hash').sort_values('hollow_layers')
 records=[];spectra={};audit=[]
 for _,r in tasks.iterrows():
  h=int(r.hollow_layers);pay=payload(a.input_dir,r.task_hash);wb,qb,qsb,qab=arrays(pay['backward']);wf,qf,qsf,qaf=arrays(pay['forward'])
  bad=(np.abs(qab)>a.passivity_tolerance)|(~np.isfinite(qsb))|(~np.isfinite(qb))|(~np.isfinite(qf));
  for w in wb[bad]:audit.append({'hollow_layers':h,'wavelength_nm':w,'reason':'lossless passivity or nonfinite'})
  qb=isolated_clean(wb,qb,bad);qf=isolated_clean(wb,qf,bad);qs=isolated_clean(wb,qsb,bad);spectra[h]=(wb,qb,qf,qs)
  records.append({'hollow_layers':h,'remaining_A_layers':(120-h)//2,'backward_peak':np.nanmax(qb),'backward_peak_nm':wb[np.nanargmax(qb)],'backward_integral':np.trapezoid(qb,wb),'forward_integral':np.trapezoid(qf,wb),'total_integral':np.trapezoid(qs,wb)})
 m=pd.DataFrame(records).sort_values('hollow_layers');ref=m.iloc[0];m['backward_peak_retention']=m.backward_peak/ref.backward_peak;m['backward_integral_retention']=m.backward_integral/ref.backward_integral;m['forward_integral_retention']=m.forward_integral/ref.forward_integral;m['total_integral_retention']=m.total_integral/ref.total_integral
 m.to_csv(a.output_dir/'hollowing_580nm_metrics.csv',index=False);pd.DataFrame(audit).to_csv(a.output_dir/'hollowing_580nm_spike_audit.csv',index=False)
 fig,ax=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
 ax[0].plot(m.remaining_A_layers,m.backward_peak_retention,'o-',label='Backward peak');ax[0].plot(m.remaining_A_layers,m.backward_integral_retention,'s-',label='Backward integral');ax[0].plot(m.remaining_A_layers,m.total_integral_retention,'^-',label='Total integral');ax[0].set(xlabel='Remaining high-index A domains',ylabel='Retention relative to intact sphere',title='580 nm low-contrast hollowing');ax[0].legend(frameon=False);ax[0].grid(alpha=.2)
 chosen=[120,110,100,80,0]
 for h in chosen:
  if h not in spectra:continue
  wl,qb,qf,qs=spectra[h];ax[1].plot(wl,qb,label=f'{(120-h)//2} A domains: backward');ax[1].plot(wl,qs,ls='--',alpha=.8,label=f'{(120-h)//2} A domains: total')
 ax[1].set(xlim=(380,780),xlabel='Wavelength [nm]',ylabel='Scattering efficiency',title='Backward and total scattering');ax[1].legend(frameon=False,fontsize=7,ncol=2);ax[1].grid(alpha=.2)
 fig.savefig(a.output_dir/'Figure_580nm_hollowing_generality.pdf',bbox_inches='tight');fig.savefig(a.output_dir/'Figure_580nm_hollowing_generality.png',dpi=600,bbox_inches='tight');plt.close(fig)
 # thresholds based on backward integral
 thresholds=[]
 for target in [.9,.8,.5]:
  ok=m[m.backward_integral_retention>=target]
  thresholds.append({'retention_target':target,'minimum_remaining_A_layers':int(ok.remaining_A_layers.min()) if len(ok) else np.nan})
 pd.DataFrame(thresholds).to_csv(a.output_dir/'hollowing_580nm_retention_thresholds.csv',index=False)
 print(m.to_string(index=False));print('Saved',a.output_dir)
if __name__=='__main__':main()
