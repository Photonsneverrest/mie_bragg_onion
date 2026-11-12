from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from real_material_common import *
def main():
 p=argparse.ArgumentParser(description='Select one intact PS/PMMA optimum per design wavelength.')
 p.add_argument('--input-dir',required=True); p.add_argument('--output-dir',required=True); a=p.parse_args()
 inp=Path(a.input_dir); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
 df=pd.read_csv(inp/'intact_summary.csv')
 df = df[pd.to_numeric(df['n_layers'], errors='coerce') > 10].copy()
 df['colour_score']=pd.to_numeric(df.eta_C,errors='coerce').fillna(0)*pd.to_numeric(df.eta_L,errors='coerce').fillna(0)
 # Conservative tie-breaks: colour score, spectral integral, fewer layers.
 chosen=[]
 for design,g in df.groupby('design_nm'):
  q=g.replace([np.inf,-np.inf],np.nan).dropna(subset=['colour_score','backward_integral']).copy()
  q['score_norm']=q.colour_score/(q.colour_score.max() or 1)
  q['integral_norm']=q.backward_integral/(q.backward_integral.max() or 1)
  q['selection_score']=0.75*q.score_norm+0.25*q.integral_norm
  chosen.append(q.sort_values(['selection_score','n_layers'],ascending=[False,True]).iloc[0])
 opt=pd.DataFrame(chosen).reset_index(drop=True)
 ps=parse_ri_info(df.ps_file.iloc[0],'PS'); pmma=parse_ri_info(df.pmma_file.iloc[0],'PMMA')
 for label,nm in [('design','design_nm'),('dline',None)]:
  wl=(opt.design_nm.to_numpy() if nm else np.full(len(opt),587.56))*1e-9
  opt[f'n_PS_{label}']=[ps([x])[0].real for x in wl]; opt[f'n_PMMA_{label}']=[pmma([x])[0].real for x in wl]
  opt[f'delta_n_{label}']=opt[f'n_PS_{label}']-opt[f'n_PMMA_{label}']; opt[f'index_ratio_{label}']=opt[f'n_PS_{label}']/opt[f'n_PMMA_{label}']
 opt.to_csv(out/'selected_intact_optima.csv',index=False)
 cols=['design_nm','n_layers','diameter_nm','colour_score','eta_C','eta_L','backward_integral','forward_integral','L','a','b','C','hue_deg','hex','n_PS_design','n_PMMA_design','delta_n_design','n_PS_dline','n_PMMA_dline','delta_n_dline']
 opt[[c for c in cols if c in opt]].to_csv(out/'Table_real_material_intact_optima.csv',index=False)
 latex=opt[[c for c in cols if c in opt]].to_latex(index=False,float_format=lambda x:f'{x:.4g}',escape=False)
 (out/'Table_real_material_intact_optima.tex').write_text(latex,encoding='utf-8')
 json_dump(out/'selection_manifest.json',{'method':'0.75 normalized eta_C*eta_L + 0.25 normalized backward integral; ties prefer fewer layers','filter':'n_layers > 10','source':str(inp/'intact_summary.csv')})
 print(opt[['design_nm','n_layers','diameter_nm','selection_score']].to_string(index=False))
if __name__=='__main__': main()
