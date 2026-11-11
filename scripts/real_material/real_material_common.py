from __future__ import annotations
import hashlib, json, pickle, re, sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd

from bragg_onion.materials import TabulatedDispersion, ConstantDispersion
from bragg_onion.geometry import LayerSubstitution
from bragg_onion.sweep import run_bragg_onion_sweep, make_plotting_aliases

DESIGNS_NM = (470, 540, 580, 620, 680, 760, 850)
LAYER_COUNTS = (2,4,6,8,10,12,14,16,18,20,24,28,32,36,40,48,56,64,72,80,90,100,110,120)

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:20]

def json_dump(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding='utf-8')

def atomic_pickle(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('wb') as f: pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)

def parse_ri_info(path: str|Path, name: str) -> TabulatedDispersion:
    path=Path(path)
    lines=path.read_text(encoding='utf-8', errors='replace').splitlines()
    in_data=False; rows=[]
    for line in lines:
        if 'data:' in line and '|' in line:
            in_data=True; continue
        if not in_data: continue
        s=line.strip()
        m=re.fullmatch(r'([0-9.+\-Ee]+)\s+([0-9.+\-Ee]+)\s+([0-9.+\-Ee]+)', s)
        if m:
            rows.append(tuple(map(float,m.groups())))
        elif rows and s and not s.startswith('#'):
            break
    if len(rows)<2: raise ValueError(f'Could not parse tabulated nk block from {path}')
    arr=np.asarray(rows,float); order=np.argsort(arr[:,0]); arr=arr[order]
    wl_m=arr[:,0]*1e-6; nk=arr[:,1]+1j*arr[:,2]
    return TabulatedDispersion(name=name, source=str(path), wavelengths_m=wl_m, nk_values=nk, extrapolation='error')

def validate_range(materials, wavelength_nm):
    grid=np.asarray(wavelength_nm,float)*1e-9
    for m in materials: m(grid)

def run_case(*, ps_path, pmma_path, design_nm, n_layers, direction, wavelength_nm,
             theta_points=721, collection_na=1.0, substitutions=None, store_full=True):
    ps=parse_ri_info(ps_path,'PS'); pmma=parse_ri_info(pmma_path,'PMMA')
    validate_range((ps,pmma), wavelength_nm)
    result=run_bragg_onion_sweep(
        geometry_mode='peak_wavelength',
        parameter_grid={'peak_wavelength_m':[float(design_nm)*1e-9], 'n_layers':[int(n_layers)],
                        'outer_layer':['A'], 'core_thickness_factor':[1.0]},
        material_a=ps, material_b=pmma, n_medium=1.0,
        wavelengths_m=np.asarray(wavelength_nm,float)*1e-9,
        theta_rad=np.linspace(0,np.pi,int(theta_points)),
        collection_na=float(collection_na), collection_direction=direction,
        integration_quantity_for_colour='q_collected_geom',
        colour_wavelength_min_nm=400.0, colour_wavelength_max_nm=700.0,
        colour_normalize_input=True, colour_normalization='max',
        layer_substitutions=substitutions, store_full_results=store_full, progress=False)
    return result

def summarize_pair(task, backward, forward):
    b=make_plotting_aliases(backward.dataframe).iloc[0].to_dict()
    f=make_plotting_aliases(forward.dataframe).iloc[0].to_dict()
    cb=backward.case_results[0]; cf=forward.case_results[0]
    wl=cb.scattering_result.wavelengths_m*1e9
    qb=np.asarray(cb.integrated_result.q_collected_geom,float)
    qf=np.asarray(cf.integrated_result.q_collected_geom,float)
    qsca=np.asarray(cb.scattering_result.qsca,float)
    qabs=np.asarray(cb.scattering_result.qabs,float)
    trap=np.trapezoid if hasattr(np,'trapezoid') else np.trapz
    row=dict(task)
    row.update({
      'diameter_nm':b.get('diameter_nm'), 'outer_radius_nm':b.get('outer_radius_nm'),
      'backward_peak':float(np.nanmax(qb)), 'backward_peak_nm':float(wl[np.nanargmax(qb)]),
      'backward_integral':float(trap(qb,wl)), 'forward_peak':float(np.nanmax(qf)),
      'forward_peak_nm':float(wl[np.nanargmax(qf)]), 'forward_integral':float(trap(qf,wl)),
      'qsca_integral':float(trap(qsca,wl)), 'qsca_max':float(np.nanmax(qsca)),
      'qabs_abs_max':float(np.nanmax(np.abs(qabs))),
    })
    for key in ('eta_C','eta_L','eta_Y','L','a','b','C','hue_deg','hex'):
        row[key]=b.get(key,np.nan)
    return row

def load_pair_payload(path: Path):
    with path.open('rb') as f: return pickle.load(f)

def payload_spectra(payload):
    b=payload['backward'].case_results[0]; f=payload['forward'].case_results[0]
    wl=b.scattering_result.wavelengths_m*1e9
    return pd.DataFrame({'wavelength_nm':wl,
        'backward':np.asarray(b.integrated_result.q_collected_geom,float),
        'forward':np.asarray(f.integrated_result.q_collected_geom,float),
        'qsca':np.asarray(b.scattering_result.qsca,float),
        'qabs':np.asarray(b.scattering_result.qabs,float),
        'g':np.asarray(b.scattering_result.g,float)})

def delta_e76(row, ref):
    return float(np.sqrt(sum((float(row[c])-float(ref[c]))**2 for c in ('L','a','b'))))
