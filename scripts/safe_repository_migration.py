#!/usr/bin/env python3
"""Hash-aware, non-destructive repository organization. Dry-run is the default."""
from __future__ import annotations
import argparse,hashlib,json,shutil
from datetime import datetime,timezone
from pathlib import Path

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--downloads',type=Path,required=True);p.add_argument('--apply',action='store_true',help='Actually create/copy; otherwise dry-run');a=p.parse_args()
 mappings={
 'REPRODUCIBILITY_LAYOUT.md':'docs/reproducibility/REPRODUCIBILITY_LAYOUT.md',
 'Table_homogeneous_controls_470nm.csv':'outputs/tables/Table_homogeneous_controls_470nm.csv',
 'Table_selected_hollow_fill_candidates.csv':'outputs/tables/Table_selected_hollow_fill_candidates.csv',
 'figure5_manifest.json':'outputs/manifests/Figure5_manifest.json',
 'mechanism_spike_audit.csv':'data/audits/mechanism_spike_audit.csv',
 'homogeneous_reference_spectra_v3_plot_clean.csv':'data/processed/homogeneous_reference_spectra_v3_plot_clean.csv',
 'NEXT_ACTIONS.md':'docs/development/NEXT_ACTIONS_2026-11-10.md',
 'manuscript_patch_hollowing_directionality_v2.tex':'manuscript/snippets/archive/manuscript_patch_hollowing_directionality_v2.tex'}
 log=[]
 for srcname,rel in mappings.items():
  src=a.downloads/srcname;dst=a.repo/rel
  entry={'source':str(src),'destination':str(dst),'action':'missing-source'}
  if src.exists():
   sh=sha(src);entry['source_sha256']=sh
   if dst.exists():
    dh=sha(dst);entry['destination_sha256']=dh
    entry['action']='already-identical' if sh==dh else 'conflict-not-overwritten'
   else:
    entry['action']='would-copy' if not a.apply else 'copied'
    if a.apply:dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
  log.append(entry)
 stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
 manifest=a.repo/'outputs'/'manifests'/f'repository_migration_{stamp}.json'
 payload={'dry_run':not a.apply,'created_utc':stamp,'entries':log}
 if a.apply:manifest.parent.mkdir(parents=True,exist_ok=True);manifest.write_text(json.dumps(payload,indent=2))
 print(json.dumps(payload,indent=2));print('Manifest:',manifest if a.apply else 'not written in dry-run')
if __name__=='__main__':main()
