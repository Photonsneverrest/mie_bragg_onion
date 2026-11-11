#!/usr/bin/env python3
"""Create Figure 4: contrast-dependent hollowing and outer-shell dominance.

Inputs:
- 470 nm compact summary plus its full_checkpoint directory
- 580 nm metrics plus its full_checkpoint directory

Plot screening is deliberately conservative: only nonfinite data or nominally
lossless points with |Qabs| above the supplied tolerance are masked. Only bounded,
isolated one-sample gaps are interpolated. Original checkpoints are never changed.
"""
from __future__ import annotations
import argparse, hashlib, json, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_payload(root: Path, task_hash: str):
    files=list((root/'full_checkpoints').glob(f'task_*_{task_hash}.pkl'))
    if len(files)!=1: raise FileNotFoundError(f'Expected one checkpoint for {task_hash}; found {len(files)}')
    with files[0].open('rb') as f:return pickle.load(f)

def arrays(sweep):
    case=sweep.case_results[0]; sc=case.scattering_result
    wl=np.asarray(sc.wavelengths_m,float)*1e9
    integ=getattr(case,'integrated_result',None) or getattr(case,'collection_result',None)
    qcol=np.asarray(integ.q_collected_geom,float)
    qsca=np.asarray(sc.qsca,float); qabs=np.asarray(sc.qabs,float)
    return wl,qcol,qsca,qabs

def clean_isolated(wl,y,bad):
    z=np.asarray(y,float).copy();z[bad]=np.nan
    for i in np.where(~np.isfinite(z))[0]:
        if i>0 and i<len(z)-1 and np.isfinite(z[i-1]) and np.isfinite(z[i+1]):
            z[i]=np.interp(wl[i],[wl[i-1],wl[i+1]],[z[i-1],z[i+1]])
    return z

def robust_threshold(frame,col,target):
    """Smallest A count reached continuously from 60 without crossing below target."""
    x=frame.sort_values('remaining_A_layers',ascending=False)
    answer=int(x.iloc[0].remaining_A_layers)
    for _,row in x.iterrows():
        if row[col]>=target: answer=int(row.remaining_A_layers)
        else: break
    return answer

def prepare_summary(path):
    df=pd.read_csv(path)
    b=df[(df.stage=='hollowing')&(df.collection_direction=='backward')&(df.hollow_layers%2==0)].copy()
    f=df[(df.stage=='hollowing')&(df.collection_direction=='forward')&(df.hollow_layers%2==0)].copy()
    keep=['task_hash','hollow_layers','spectrum_integral','q_collected_geom_max','eta_C','eta_L','L','C','a','b','hex']
    b=b[keep].rename(columns={c:f'{c}_backward' for c in keep if c not in ['task_hash','hollow_layers']})
    f=f[['task_hash','hollow_layers','spectrum_integral']].rename(columns={'spectrum_integral':'spectrum_integral_forward'})
    x=b.merge(f,on=['task_hash','hollow_layers']);x['remaining_A_layers']=(120-x.hollow_layers)//2
    ref=x[x.remaining_A_layers==60].iloc[0]
    for c in ['spectrum_integral_backward','spectrum_integral_forward','q_collected_geom_max_backward','eta_C_backward','eta_L_backward']:
        x[c+'_retention']=x[c]/ref[c]
    x['delta_E76']=np.sqrt((x.L_backward-ref.L_backward)**2+(x.a_backward-ref.a_backward)**2+(x.b_backward-ref.b_backward)**2)
    return x.sort_values('remaining_A_layers')

def selected_spectra(root,summary,counts,tol,audit,design):
    out={}
    for A in counts:
        row=summary.iloc[(summary.remaining_A_layers-A).abs().argsort()[:1]].iloc[0]
        pay=load_payload(root,row.task_hash)
        wb,qb,qs,qabs=arrays(pay['backward']); wf,qf,_,_=arrays(pay['forward'])
        bad=(~np.isfinite(qb))|(~np.isfinite(qf))|(~np.isfinite(qs))|(np.abs(qabs)>tol)
        for w in wb[bad]: audit.append({'design_nm':design,'remaining_A_layers':int(row.remaining_A_layers),'wavelength_nm':float(w),'reason':f'nonfinite or |Qabs|>{tol:g}'})
        out[int(row.remaining_A_layers)]=(wb,clean_isolated(wb,qb,bad),clean_isolated(wb,qf,bad),clean_isolated(wb,qs,bad))
    return out

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--summary-470',type=Path,required=True);p.add_argument('--checkpoints-470',type=Path,required=True)
    p.add_argument('--summary-580',type=Path,required=True);p.add_argument('--checkpoints-580',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,default=Path('outputs/figures/main/Figure4'))
    p.add_argument('--style',type=Path,default=Path('styles/publication.mplstyle'));p.add_argument('--passivity-tolerance',type=float,default=1e-6)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    if not a.style.exists(): raise FileNotFoundError(a.style)
    plt.style.use(a.style)
    m470=prepare_summary(a.summary_470);m580=prepare_summary(a.summary_580)
    audit=[]
    s470=selected_spectra(a.checkpoints_470,m470,[60,13,5,0],a.passivity_tolerance,audit,470)
    # choose exact physically available 580 cases: intact, 90% threshold, five A, all-air
    s580=selected_spectra(a.checkpoints_580,m580,[60,31,5,0],a.passivity_tolerance,audit,580)
    colors={'470':'#0072B2','580':'#D55E00'}
    fig,ax=plt.subplots(2,2,figsize=(7.2,6.0),constrained_layout=True)
    A,B,C,D=ax.ravel()
    A.plot(m470.remaining_A_layers,m470.spectrum_integral_backward_retention,'o-',color=colors['470'],label=r'470 nm, $n_A/n_B=1.55$')
    A.plot(m580.remaining_A_layers,m580.spectrum_integral_backward_retention,'s-',color=colors['580'],label=r'580 nm, $n_A/n_B=1.129$')
    for y in [.9,.8,.5]:A.axhline(y,color='0.65',lw=.7,ls=':')
    A.set(xlabel='Remaining high-index A domains',ylabel='Backward-integral retention',title='(a) Contrast-dependent shell requirement',ylim=(0,1.08));A.legend()
    B.plot(m470.remaining_A_layers,m470.spectrum_integral_forward_retention,'o-',color=colors['470'],label='470 nm forward')
    B.plot(m580.remaining_A_layers,m580.spectrum_integral_forward_retention,'s-',color=colors['580'],label='580 nm forward')
    B.plot(m470.remaining_A_layers,m470.spectrum_integral_backward_retention,'o--',color=colors['470'],alpha=.6,label='470 nm backward')
    B.plot(m580.remaining_A_layers,m580.spectrum_integral_backward_retention,'s--',color=colors['580'],alpha=.6,label='580 nm backward')
    B.axhline(1,color='0.45',lw=.8,ls=':');B.set(xlabel='Remaining high-index A domains',ylabel='Integral relative to intact sphere',title='(b) Forward redistribution');B.legend(ncol=2,fontsize=7)
    for AA,(wl,qb,qf,qs) in s470.items(): C.plot(wl,qb,label=f'{AA} A domains')
    C.set(xlim=(380,780),xlabel='Wavelength [nm]',ylabel='Backward-collected efficiency',title=r'(c) 470 nm design, $n_A/n_B=1.55$');C.legend(fontsize=7)
    for AA,(wl,qb,qf,qs) in s580.items(): D.plot(wl,qb,label=f'{AA} A domains')
    D.set(xlim=(380,780),xlabel='Wavelength [nm]',ylabel='Backward-collected efficiency',title=r'(d) 580 nm design, $n_A/n_B=1.129$');D.legend(fontsize=7)
    for ext in ['pdf','png']:
        fig.savefig(a.output_dir/f'Figure4_hollowing_outer_shell_dominance.{ext}',dpi=600 if ext=='png' else None)
    plt.close(fig)
    pd.DataFrame(audit).to_csv(a.output_dir/'Figure4_spike_audit.csv',index=False)
    rows=[]
    for design,m in [(470,m470),(580,m580)]:
        for t in [.9,.8,.5]: rows.append({'design_nm':design,'retention_metric':'backward_spectrum_integral','target':t,'minimum_contiguous_remaining_A_layers':robust_threshold(m,'spectrum_integral_backward_retention' if design==470 else 'spectrum_integral_backward_retention',t)})
    pd.DataFrame(rows).to_csv(a.output_dir/'Figure4_retention_thresholds.csv',index=False)
    manifest={'inputs':{k:str(v) for k,v in vars(a).items() if isinstance(v,Path)},'passivity_tolerance':a.passivity_tolerance,'selected_A_domains':{'470':[60,13,5,0],'580':[60,31,5,0]},'threshold_definition':'smallest A count reached continuously from intact sphere without dropping below target','sha256':{}}
    for f in a.output_dir.iterdir():
        if f.is_file():manifest['sha256'][f.name]=hashlib.sha256(f.read_bytes()).hexdigest()
    (a.output_dir/'Figure4_manifest.json').write_text(json.dumps(manifest,indent=2))
    print('Saved',a.output_dir)
if __name__=='__main__':main()
