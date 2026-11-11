from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from real_material_common import *
def main():
 p=argparse.ArgumentParser(description='Analyze PS/PMMA intact, hollowing and melanin-like fill studies.')
 p.add_argument('--intact-dir',required=True); p.add_argument('--optima-dir',required=True); p.add_argument('--hollowing-dir',required=True); p.add_argument('--melanin-dir',required=True); p.add_argument('--output-dir',required=True); p.add_argument('--style'); p.add_argument('--ideal-summary'); a=p.parse_args()
 if a.style: plt.style.use(a.style)
 out=Path(a.output_dir); (out/'figures').mkdir(parents=True,exist_ok=True); (out/'tables').mkdir(parents=True,exist_ok=True); (out/'latex').mkdir(parents=True,exist_ok=True)
 intact=pd.read_csv(Path(a.intact_dir)/'intact_summary.csv'); opt=pd.read_csv(Path(a.optima_dir)/'selected_intact_optima.csv'); hol=pd.read_csv(Path(a.hollowing_dir)/'hollowing_metrics.csv'); sel=pd.read_csv(Path(a.hollowing_dir)/'selected_hollow_geometries.csv'); mel=pd.read_csv(Path(a.melanin_dir)/'melanin_fill_summary.csv')
 # Melanin trade-off relative to k=0 for each design.
 rows=[]
 for d,g in mel.groupby('design_nm'):
  ref=g[np.isclose(g.fill_k,0)].iloc[0]
  for _,r in g.iterrows():
   rows.append({**r.to_dict(),'forward_integral_retention':r.forward_integral/ref.forward_integral,'backward_integral_retention':r.backward_integral/ref.backward_integral,'forward_attenuation_db':-10*np.log10(r.forward_integral/ref.forward_integral),'delta_E76':delta_e76(r,ref),'spectral_score':(r.eta_C*r.eta_L)/(ref.eta_C*ref.eta_L)})
 trade=pd.DataFrame(rows); trade.to_csv(out/'tables'/'melanin_fill_tradeoff.csv',index=False)
 opt.to_csv(out/'tables'/'Table_real_material_intact_optima_full.csv',index=False); sel.to_csv(out/'tables'/'Table_real_material_hollowing_selection_full.csv',index=False)
 # Main compact figure.
 fig,ax=plt.subplots(2,2,figsize=(7.2,6.0),constrained_layout=True)
 for d,g in intact.groupby('design_nm'):
  ax[0,0].plot(g.n_layers,g.eta_C*g.eta_L,marker='o',ms=2,label=f'{d:.0f} nm')
 ax[0,0].set(xlabel='Radial domains',ylabel=r'$\eta_C\eta_L$',title='a  Intact PS/PMMA optimisation'); ax[0,0].legend(ncol=2,fontsize=6)
 for d,g in hol.groupby('design_nm'):
  ax[0,1].plot(g.remaining_layers,g.backward_integral_retention,label=f'{d:.0f} nm')
 ax[0,1].axhline(.9,color='.3',ls=':'); ax[0,1].set(xlabel='Remaining radial domains',ylabel='Backward integral retention',title='b  Hollowing tolerance')
 ax[1,0].scatter(opt.a,opt.b,c=[x if isinstance(x,str) else '#777777' for x in opt.hex],edgecolor='k',s=35)
 for _,r in opt.iterrows(): ax[1,0].annotate(f'{r.design_nm:.0f}',(r.a,r.b),xytext=(3,3),textcoords='offset points',fontsize=6)
 ax[1,0].set(xlabel=r'$a^*$',ylabel=r'$b^*$',title='c  Calculated backward colours')
 for d,g in trade.groupby('design_nm'):
  ax[1,1].plot(g.backward_integral_retention,g.forward_attenuation_db,marker='o',label=f'{d:.0f} nm')
 ax[1,1].set(xlabel='Backward integral retention',ylabel='Forward attenuation [dB]',title='d  Melanin-like fill trade-off'); ax[1,1].legend()
 for x in ax.flat: x.grid(alpha=.2)
 fig.savefig(out/'figures'/'Figure7_real_material_PS_PMMA.pdf'); fig.savefig(out/'figures'/'Figure7_real_material_PS_PMMA.png',dpi=600)
 # SI table and evidence map.
 alltab=opt.merge(sel[['design_nm','hollow_layers','remaining_layers','backward_integral_retention','spectral_correlation','delta_E76','colour_performance_retention']],on='design_nm',suffixes=('_intact','_hollow'))
 alltab.to_csv(out/'tables'/'Table_SI_real_material_complete.csv',index=False); (out/'tables'/'Table_SI_real_material_complete.tex').write_text(alltab.to_latex(index=False,float_format=lambda x:f'{x:.4g}',escape=False),encoding='utf-8')
 evidence=pd.DataFrame([
  {'claim':'Intact PS/PMMA optimum for each design','evidence':'Figure7a; Table_real_material_intact_optima_full.csv'},
  {'claim':'Minimum hollow geometry retaining defined colour performance','evidence':'Figure7b; Table_real_material_hollowing_selection_full.csv'},
  {'claim':'Calculated realistic colour coordinates','evidence':'Figure7c; Table_SI_real_material_complete.csv'},
  {'claim':'Melanin-like fill trade-off at 470 and 580 nm','evidence':'Figure7d; melanin_fill_tradeoff.csv'}])
 evidence.to_csv(out/'tables'/'evidence_map.csv',index=False)
 manifest={'inputs':vars(a),'outputs':[str(p.relative_to(out)) for p in out.rglob('*') if p.is_file()]}; json_dump(out/'analysis_manifest.json',manifest); print('Saved',out)
if __name__=='__main__': main()
