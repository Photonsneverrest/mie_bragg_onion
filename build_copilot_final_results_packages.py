#!/usr/bin/env python3
"""Build exactly three lossless, upload-supported packages for the finalized
Bragg-onion validation, spike-diagnostic, hollowing, and analysis results.

The generated .py files are transport containers. Each contains a common
manifest, readable inventories, checksum-verified zlib+Base85 payloads, and
self-extraction functions. Original relative paths are preserved on extraction.
"""
from __future__ import annotations

import argparse
import ast
import base64
import csv
import hashlib
import json
import os
import pickle
import shutil
import tempfile
import zlib
from datetime import datetime, timezone
from pathlib import Path

# These are the non-duplicate, scientifically useful originals requested for
# follow-up analysis. CSV/JSON provide readable metrics; PKL retains full arrays.
# Existing 140-case source data are intentionally not repeated because they were
# already packaged previously. The two new validation runs are kept separately.
SPECS = [
    # Final analysis notebook and plots
    ("notebooks/Hollowing_and_NumericalValidation_Analysis.ipynb",
     "notebooks/Hollowing_and_NumericalValidation_Analysis.ipynb"),

    # Numerical validation: full run
    ("outputs/numerical_validation/validation_axis_profiles.pkl",
     "outputs/numerical_validation/validation_axis_profiles.pkl"),
    ("outputs/numerical_validation/validation_metrics.csv",
     "outputs/numerical_validation/validation_metrics.csv"),
    ("outputs/numerical_validation/validation_metadata.json",
     "outputs/numerical_validation/validation_metadata.json"),

    # Numerical validation: max-layers-12 run
    ("outputs/numerical_validation_max-layers12/validation_axis_profiles.pkl",
     "outputs/numerical_validation_max-layers12/validation_axis_profiles.pkl"),
    ("outputs/numerical_validation_max-layers12/validation_metrics.csv",
     "outputs/numerical_validation_max-layers12/validation_metrics.csv"),
    ("outputs/numerical_validation_max-layers12/validation_metadata.json",
     "outputs/numerical_validation_max-layers12/validation_metadata.json"),

    # Automated diagnostics on the original 140-case sweep
    ("outputs/field_enhancement_sweep/spike_diagnostics/localized_spikes.csv",
     "outputs/field_enhancement_sweep/spike_diagnostics/localized_spikes.csv"),
    ("outputs/field_enhancement_sweep/spike_diagnostics/profile_quality_and_spike_metrics.csv",
     "outputs/field_enhancement_sweep/spike_diagnostics/profile_quality_and_spike_metrics.csv"),

    # Progressive hollowing study
    ("outputs/hollowing_60L_nH1p5/hollowing_spectra.pkl",
     "outputs/hollowing_60L_nH1p5/hollowing_spectra.pkl"),
    ("outputs/hollowing_60L_nH1p5/hollowing_summary.csv",
     "outputs/hollowing_60L_nH1p5/hollowing_summary.csv"),
    ("outputs/hollowing_60L_nH1p5/hollowing_metadata.json",
     "outputs/hollowing_60L_nH1p5/hollowing_metadata.json"),

    # Final notebook figures
    ("outputs/hollowing_60L_nH1p5/analysis_figures/hollowing_complete_bilayer_qbk.png",
     "outputs/hollowing_60L_nH1p5/analysis_figures/hollowing_complete_bilayer_qbk.png"),
    ("outputs/hollowing_60L_nH1p5/analysis_figures/hollowing_single_layer_qbk.png",
     "outputs/hollowing_60L_nH1p5/analysis_figures/hollowing_single_layer_qbk.png"),
    ("outputs/hollowing_60L_nH1p5/analysis_figures/hollowing_peak_evolution.png",
     "outputs/hollowing_60L_nH1p5/analysis_figures/hollowing_peak_evolution.png"),
    ("outputs/hollowing_60L_nH1p5/analysis_figures/homogeneous_control.png",
     "outputs/hollowing_60L_nH1p5/analysis_figures/homogeneous_control.png"),
    ("outputs/hollowing_60L_nH1p5/analysis_figures/validation_onset.png",
     "outputs/hollowing_60L_nH1p5/analysis_figures/validation_onset.png"),
]


def sha256_file(path: Path, block: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(root: Path, relative: str) -> Path:
    direct = root / Path(relative)
    if direct.is_file():
        return direct
    # Permit running from a parent folder, but require a unique suffix match.
    suffix = Path(relative).parts
    hits = [
        p for p in root.rglob(Path(relative).name)
        if p.is_file() and tuple(p.parts[-len(suffix):]) == suffix
    ]
    if not hits:
        raise FileNotFoundError(f"Required file not found: {relative}")
    if len(hits) > 1:
        raise RuntimeError(
            f"Ambiguous relative path {relative}:\n" + "\n".join(map(str, hits))
        )
    return hits[0]


def summarize(path: Path, archive_path: str, trusted_pickles: bool) -> str:
    item = {
        "archive_path": archive_path,
        "source_path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "suffix": path.suffix.lower(),
    }
    try:
        suffix = path.suffix.lower()
        if suffix == ".py":
            text = path.read_text(encoding="utf-8")
            tree = ast.parse(text)
            item.update(
                type="python",
                lines=len(text.splitlines()),
                definitions=[
                    node.name for node in ast.walk(tree)
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                ],
            )
        elif suffix == ".csv":
            with path.open(encoding="utf-8-sig", newline="") as handle:
                reader = csv.reader(handle)
                columns = next(reader, [])
                rows = sum(1 for _ in reader)
            item.update(type="csv", columns=columns, rows=rows)
        elif suffix == ".json":
            item.update(type="json", content=json.loads(path.read_text(encoding="utf-8")))
        elif suffix == ".ipynb":
            notebook = json.loads(path.read_text(encoding="utf-8"))
            cells = notebook.get("cells", [])
            item.update(
                type="jupyter_notebook",
                nbformat=notebook.get("nbformat"),
                cells=len(cells),
                code_cells=sum(c.get("cell_type") == "code" for c in cells),
                markdown_cells=sum(c.get("cell_type") == "markdown" for c in cells),
            )
        elif suffix == ".pkl":
            item["type"] = "pickle"
            if trusted_pickles:
                with path.open("rb") as handle:
                    value = pickle.load(handle)
                item["object_type"] = type(value).__name__
                if hasattr(value, "shape"):
                    item["shape"] = list(value.shape)
                if hasattr(value, "columns"):
                    item["columns"] = [str(c) for c in value.columns]
                if isinstance(value, dict):
                    item["keys"] = [str(k) for k in value]
            else:
                item["note"] = "Not deserialized; use --trusted-pickles for your own files."
        elif suffix in {".png", ".jpg", ".jpeg"}:
            item["type"] = "image"
            try:
                from PIL import Image
                with Image.open(path) as image:
                    item["pixel_size"] = list(image.size)
                    item["mode"] = image.mode
            except Exception as exc:
                item["image_inventory_error"] = f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        item["inventory_error"] = f"{type(exc).__name__}: {exc}"
    return json.dumps(item, ensure_ascii=False, indent=2, default=str)


def encode_to_temp(path: Path, temp_dir: Path) -> tuple[Path, int]:
    """Stream-compress and Base85-encode one file without loading it all in RAM."""
    encoded_path = temp_dir / (hashlib.sha256(str(path).encode()).hexdigest() + ".b85")
    compressor = zlib.compressobj(level=9)
    carry = b""
    written = 0
    with path.open("rb") as src, encoded_path.open("wb") as dst:
        while chunk := src.read(4 * 1024 * 1024):
            compressed = carry + compressor.compress(chunk)
            usable = len(compressed) - (len(compressed) % 4)
            if usable:
                text = base64.b85encode(compressed[:usable])
                dst.write(text)
                written += len(text)
            carry = compressed[usable:]
        compressed = carry + compressor.flush()
        if compressed:
            text = base64.b85encode(compressed)
            dst.write(text)
            written += len(text)
    return encoded_path, written


RUNTIME = r'''
def original_bytes(archive_path):
    entry = PAYLOADS[archive_path]
    raw = zlib.decompress(base64.b85decode(entry["data"].encode("ascii")))
    if len(raw) != entry["size"]:
        raise RuntimeError("Size failure: " + archive_path)
    if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
        raise RuntimeError("Checksum failure: " + archive_path)
    return raw

def verify():
    for archive_path in PAYLOADS:
        original_bytes(archive_path)
    print("Verified package %d/%d: %d originals" % (
        PACKAGE_INDEX, PACKAGE_TOTAL, len(PAYLOADS)
    ))

def extract(folder):
    root = Path(folder)
    root.mkdir(parents=True, exist_ok=True)
    for archive_path in PAYLOADS:
        target = root / archive_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(original_bytes(archive_path))
    print("Extracted", len(PAYLOADS), "originals to", root)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--manifest", action="store_true")
    args = parser.parse_args()
    if args.extract:
        extract(args.extract)
    elif args.list:
        for key, value in PAYLOADS.items():
            print(key, value["size"], value["sha256"])
    elif args.manifest:
        print(json.dumps(MASTER_MANIFEST, indent=2))
    else:
        verify()
if __name__ == "__main__":
    main()
'''


def append_encoded_chunks(handle, encoded_path: Path, chunk_chars: int = 12000) -> None:
    with encoded_path.open("r", encoding="ascii") as encoded:
        while text := encoded.read(chunk_chars):
            handle.write(" " + repr(text) + "\n")


def write_package(out: Path, index: int, total: int, group: list[dict], manifest: dict) -> Path:
    target = out / f"copilot_final_results_package_{index:02d}_of_{total:02d}.py"
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("#!/usr/bin/env python3\n")
        handle.write("# Lossless Copilot transport package. Do not edit encoded payloads.\n")
        handle.write("import argparse, base64, hashlib, json, zlib\n")
        handle.write("from pathlib import Path\n")
        handle.write(f"PACKAGE_INDEX={index}\nPACKAGE_TOTAL={total}\n")
        handle.write("MASTER_MANIFEST=" + repr(manifest) + "\n")
        handle.write("PAYLOADS={\n")
        for item in group:
            handle.write(repr(item["archive_path"]) + ":{")
            handle.write('"size":' + str(item["size"]) + ",")
            handle.write('"sha256":' + repr(item["sha256"]) + ",")
            handle.write('"data":(\n')
            append_encoded_chunks(handle, item["encoded_path"])
            handle.write(")},\n")
        handle.write("}\nINVENTORIES={\n")
        for item in group:
            handle.write(repr(item["archive_path"]) + ":" + repr(item["inventory"]) + ",\n")
        handle.write("}\n")
        handle.write(RUNTIME)
    compile(target.read_text(encoding="utf-8"), str(target), "exec")
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path.cwd(),
        help="Project root containing outputs/ and notebooks/",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("copilot_final_upload"),
        help="Folder for the three generated .py packages",
    )
    parser.add_argument("--trusted-pickles", action="store_true")
    parser.add_argument(
        "--keep-temp", action="store_true",
        help="Keep temporary Base85 streams for debugging",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix="copilot_pkg_", dir=out))

    try:
        items = []
        manifest_files = []
        for relative, archive_path in SPECS:
            path = resolve_path(root, relative)
            size = path.stat().st_size
            checksum = sha256_file(path)
            encoded_path, encoded_size = encode_to_temp(path, temp_dir)
            inventory = summarize(path, archive_path, args.trusted_pickles)
            items.append({
                "source": path,
                "archive_path": archive_path,
                "size": size,
                "sha256": checksum,
                "encoded_path": encoded_path,
                "encoded_size": encoded_size,
                "inventory": inventory,
            })
            manifest_files.append({
                "archive_path": archive_path,
                "source_path": str(path.relative_to(root)).replace(os.sep, "/"),
                "bytes": size,
                "sha256": checksum,
            })
            print(f"Prepared {archive_path}: {size / 1048576:.2f} MiB")

        # Greedy compressed-size balancing into exactly three upload files.
        groups = [[], [], []]
        loads = [0, 0, 0]
        for item in sorted(items, key=lambda x: x["encoded_size"], reverse=True):
            slot = min(range(3), key=loads.__getitem__)
            groups[slot].append(item)
            loads[slot] += item["encoded_size"]

        master = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "lossless": True,
            "encoding": "zlib+base85",
            "package_count": 3,
            "selection_note": (
                "Final numerical-validation arrays and metrics; 140-case spike diagnostics; "
                "60-layer hollowing arrays, summaries and figures; final analysis notebook. "
                "Duplicate CSV/Parquet/PKL representations were omitted where a full PKL plus "
                "readable CSV metrics preserve the needed information."
            ),
            "files": manifest_files,
        }

        made = [write_package(out, i + 1, 3, groups[i], master) for i in range(3)]
        # Syntax and checksum verification. This can be memory intensive for the
        # largest package but guarantees the uploaded containers are usable.
        for package in made:
            namespace = {"__name__": "package_check"}
            exec(compile(package.read_text(encoding="utf-8"), str(package), "exec"), namespace)
            namespace["verify"]()

        print("\nUpload exactly these three files:")
        for package in made:
            print(f"{package}  {package.stat().st_size / 1048576:.2f} MiB")
        print("\nPackage encoded payload loads [MiB]:", [round(v / 1048576, 2) for v in loads])
    finally:
        if args.keep_temp:
            print("Temporary encoded streams retained in", temp_dir)
        else:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
