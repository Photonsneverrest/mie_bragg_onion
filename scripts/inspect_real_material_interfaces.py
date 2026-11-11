#!/usr/bin/env python3
"""Generate a non-invasive API/CLI report before real-material sweep authoring."""
from __future__ import annotations
import argparse, ast, hashlib, importlib, inspect, json, subprocess, sys
from pathlib import Path

def digest(p):
    h=hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()

def argparse_flags(path):
    tree=ast.parse(path.read_text(encoding='utf-8'))
    flags=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='add_argument':
            vals=[]
            for a in node.args:
                if isinstance(a,ast.Constant) and isinstance(a.value,str): vals.append(a.value)
            flags.extend(vals)
    return sorted(set(flags))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--repo',type=Path,default=Path.cwd())
    p.add_argument('--output',type=Path,default=Path('outputs/manifests/real_material_interface_report.json'))
    a=p.parse_args(); root=a.repo.resolve()
    targets=[
      root/'src/bragg_onion/materials.py', root/'src/bragg_onion/geometry.py',
      root/'src/bragg_onion/sweep.py', root/'scripts/run_hollow_absorption_sweep.py',
      root/'examples/analyse_bragg_onions.py'
    ]
    missing=[str(x) for x in targets if not x.exists()]
    report={'repo':str(root),'python':sys.version,'missing':missing,'files':{}}
    for x in targets:
        if x.exists(): report['files'][str(x.relative_to(root))]={'sha256':digest(x),'argparse_flags':argparse_flags(x)}
    sys.path.insert(0,str(root/'src'))
    for modname in ['bragg_onion.materials','bragg_onion.geometry','bragg_onion.sweep']:
        try:
            mod=importlib.import_module(modname)
            entries={}
            for name in ['MaterialFileSpec','load_tabulated_material','TabulatedDispersion','LayerSubstitution','run_bragg_onion_sweep']:
                if hasattr(mod,name):
                    obj=getattr(mod,name)
                    try: entries[name]=str(inspect.signature(obj))
                    except Exception: entries[name]='signature unavailable'
            report[modname]=entries
        except Exception as e: report[modname]={'import_error':repr(e)}
    # Exact --help execution for the only current wrapper involved in later hollowing.
    wrapper=root/'scripts/run_hollow_absorption_sweep.py'
    if wrapper.exists():
        cp=subprocess.run([sys.executable,str(wrapper),'--help'],text=True,capture_output=True)
        report['wrapper_help']={'returncode':cp.returncode,'stdout':cp.stdout,'stderr':cp.stderr}
    out=(root/a.output).resolve() if not a.output.is_absolute() else a.output
    out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(out)
    if missing: raise SystemExit(2)
if __name__=='__main__': main()
