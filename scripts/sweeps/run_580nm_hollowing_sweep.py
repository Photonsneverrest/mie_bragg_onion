#!/usr/bin/env python3
"""Launch the 580 nm, nA=1.129, nB=1.0 even-depth hollowing sweep."""
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--base-script',type=Path,default=Path('scripts/run_hollow_absorption_sweep.py'));p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--workers',type=int,default=8);p.add_argument('--manifest-only',action='store_true');a=p.parse_args()
 hollow=','.join(map(str,range(0,121,2)))
 cmd=[sys.executable,str(a.base_script),'--output-dir',str(a.output_dir),'--design-wavelength-nm','580','--n-A','1.129','--n-B','1.0','--n-layers','120','--n-medium','1.0','--hollow-layers',hollow,'--wavelength-min-nm','380','--wavelength-max-nm','780','--wavelength-step-nm','1','--theta-points','721','--collection-na','1.0','--workers',str(a.workers)]
 if a.manifest_only: cmd.append('--manifest-only')
 print(' '.join(cmd),flush=True);subprocess.run(cmd,check=True)
if __name__=='__main__':main()
