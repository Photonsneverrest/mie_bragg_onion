#!/usr/bin/env python3
from __future__ import annotations

"""Refractive-index and sphere-size sensitivity of electric-field enhancement in Bragg onions.

Parameter matrix: n_low=1.0; n_high in 10 specified values; 5, 10, 15, 20, 40, 60, or 80
radial layers; k=0 or k=0.01 in both materials; outermost layer is material A.

Outputs
-------
1. Backward-collected spectra and robust local Bragg-peak estimates.
2. Central propagation-axis profiles of |E|/|E0| and |E|^2/|E0|^2.
3. Cross-sectional (XZ-plane) shell averages for every radial layer.
4. Scalar colour results returned by run_bragg_onion_sweep, when available.

The local peak is selected from a smoothed backward spectrum inside a user-defined
wavelength window. This avoids selecting unrelated edge/Mie maxima far from the
470 nm Bragg design wavelength.
"""

import os
for _name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")

import argparse
import hashlib
import json
import pickle
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(iterable=None, total=None, desc=None):
        return iterable if iterable is not None else []

from bragg_onion.materials import ConstantDispersion
from bragg_onion.geometry import ExtinctionModifier
from bragg_onion.sweep import run_bragg_onion_sweep
from bragg_onion.fields import compute_field_map
from bragg_onion._notebook_tools.field_analysis import rebuild_stack_from_row

N_LOW = 1.0
N_HIGH_VALUES = (1.0, 1.001, 1.01, 1.05, 1.1, 1.2, 1.3, 1.5, 2.0, 2.5)
ABSORPTION_K = 0.01
DESIGN_WAVELENGTH_NM = 470.0
N_LAYER_VALUES = (5, 10, 15, 20, 40, 60, 80)
OUTER_LAYER = "A"
CORE_THICKNESS_FACTOR = 1.0
CACHE_VERSION = "core_field_enhancement_v4_small_spheres"


def stable_hash(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()[:20]


def atomic_pickle_dump(obj, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def parquet_available() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except Exception:
        try:
            import fastparquet  # noqa: F401
            return True
        except Exception:
            return False


def iter_nested(obj, path="root", depth=0, max_depth=8, visited=None):
    if visited is None:
        visited = set()
    if depth > max_depth or id(obj) in visited:
        return
    visited.add(id(obj))
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield from iter_nested(value, f"{path}[{key!r}]", depth + 1, max_depth, visited)
    elif isinstance(obj, pd.DataFrame):
        for column in obj.columns:
            for index, value in obj[column].items():
                if isinstance(value, np.ndarray):
                    yield f"{path}.dataframe[{column!r}][{index!r}]", value
    elif isinstance(obj, pd.Series):
        for index, value in obj.items():
            if isinstance(value, np.ndarray):
                yield f"{path}[{index!r}]", value
    elif isinstance(obj, np.ndarray):
        yield path, obj
    elif isinstance(obj, (list, tuple)):
        for index, value in enumerate(obj):
            yield from iter_nested(value, f"{path}[{index}]", depth + 1, max_depth, visited)
    elif hasattr(obj, "__dict__"):
        for attribute, value in vars(obj).items():
            if not attribute.startswith("__"):
                yield from iter_nested(value, f"{path}.{attribute}", depth + 1, max_depth, visited)


def as_real_1d(array):
    array = np.squeeze(np.asarray(array))
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        return None
    return np.asarray(array, dtype=float)


def extract_backward_spectrum(sweep_result) -> tuple[np.ndarray, np.ndarray]:
    arrays = list(iter_nested(sweep_result, path="backward"))
    wavelength_candidates = []
    for path, raw in arrays:
        array = as_real_1d(raw)
        if array is None or array.size < 10:
            continue
        finite = array[np.isfinite(array)]
        if finite.size == 0:
            continue
        score = 10 if any(token in path.lower() for token in ("wavelength", "lambda")) else 0
        if finite.min() > 0 and finite.max() < 1e-3:
            wavelength_nm = array * 1e9
            score += 5
        elif finite.min() >= 100 and finite.max() <= 5000:
            wavelength_nm = array
            score += 5
        else:
            continue
        wavelength_candidates.append((score, path, wavelength_nm))
    if not wavelength_candidates:
        raise KeyError("Could not identify wavelength array")
    wavelength_candidates.sort(key=lambda item: (item[0], len(item[2])), reverse=True)
    wavelength_nm = wavelength_candidates[0][2]

    spectrum_candidates = []
    for path, raw in arrays:
        array = as_real_1d(raw)
        if array is None or len(array) != len(wavelength_nm):
            continue
        text = path.lower()
        if any(token in text for token in ("wavelength", "theta", "angle", "field", "coefficient")):
            continue
        finite = array[np.isfinite(array)]
        if finite.size == 0:
            continue
        score = sum(points for token, points in (
            ("q_collected_geom", 30), ("collected_geom", 20),
            ("collected", 10), ("spectrum", 3), ("efficiency", 2),
        ) if token in text)
        spectrum_candidates.append((score, path, array))
    if not spectrum_candidates:
        raise KeyError("Could not identify backward-collected spectrum")
    spectrum_candidates.sort(key=lambda item: item[0], reverse=True)
    if spectrum_candidates[0][0] <= 0:
        raise KeyError(f"Untrustworthy spectrum candidate: {spectrum_candidates[0][1]}")
    spectrum = np.asarray(spectrum_candidates[0][2], dtype=float)
    order = np.argsort(wavelength_nm)
    return wavelength_nm[order], spectrum[order]


def extract_scalar_sweep_summary(sweep_result) -> dict:
    """Preserve scalar colour/geometry columns returned by the sweep."""
    dataframe = getattr(sweep_result, "dataframe", None)
    if not isinstance(dataframe, pd.DataFrame) or dataframe.empty:
        return {}
    row = dataframe.iloc[0]
    output = {}
    for column, value in row.items():
        if isinstance(value, (str, bool, int, float, np.integer, np.floating)) or value is None:
            output[f"sweep_{column}"] = value
    return output


def smooth_spectrum(wavelength_nm: np.ndarray, values: np.ndarray, smoothing_nm: float) -> np.ndarray:
    if smoothing_nm <= 0:
        return np.asarray(values, dtype=float).copy()
    spacing = float(np.nanmedian(np.diff(wavelength_nm)))
    points = max(1, int(round(smoothing_nm / spacing)))
    if points % 2 == 0:
        points += 1
    kernel = np.ones(points, dtype=float) / points
    return np.convolve(np.asarray(values, dtype=float), kernel, mode="same")


def select_local_peak(
    wavelength_nm: np.ndarray,
    values: np.ndarray,
    *,
    search_min_nm: float,
    search_max_nm: float,
    smoothing_nm: float,
) -> tuple[float, np.ndarray, float]:
    smoothed = smooth_spectrum(wavelength_nm, values, smoothing_nm)
    mask = (
        np.isfinite(wavelength_nm) & np.isfinite(smoothed)
        & (wavelength_nm >= search_min_nm) & (wavelength_nm <= search_max_nm)
    )
    if not np.any(mask):
        raise ValueError("Peak-search window contains no spectral samples")
    indices = np.flatnonzero(mask)
    peak_index = int(indices[np.nanargmax(smoothed[mask])])
    local_values = smoothed[mask]
    contrast = float(
        (np.nanmax(local_values) - np.nanmedian(local_values))
        / max(abs(float(np.nanmedian(local_values))), np.finfo(float).tiny)
    )
    return float(wavelength_nm[peak_index]), smoothed, contrast


def make_tasks() -> list[dict]:
    """Build all 7 x 10 x 2 = 140 layer/index/absorption cases."""
    tasks = []
    for n_layers in N_LAYER_VALUES:
        for absorbing in (False, True):
            for n_high in N_HIGH_VALUES:
                task = {
                    "cache_version": CACHE_VERSION,
                    "n_low": N_LOW,
                    "n_high": float(n_high),
                    "k_all_layers": ABSORPTION_K if absorbing else 0.0,
                    "absorbing": absorbing,
                    "design_wavelength_nm": DESIGN_WAVELENGTH_NM,
                    "n_layers": int(n_layers),
                    "outer_layer": OUTER_LAYER,
                    "core_thickness_factor": CORE_THICKNESS_FACTOR,
                }
                task["task_hash"] = stable_hash(task)
                tasks.append(task)
    tasks.sort(key=lambda task: (task["n_layers"], task["absorbing"], task["n_high"]))
    for index, task in enumerate(tasks):
        task["task_index"] = index
        suffix = "lossless" if not task["absorbing"] else f"k_{task['k_all_layers']:.3f}"
        task["case_label"] = (
            f"L_{task['n_layers']:03d}_nH_{task['n_high']:.3f}_{suffix}"
        )
    return tasks


def make_materials_and_modifier(task: dict):
    high = ConstantDispersion.from_nk("A_high", n=float(task["n_high"]), k=0.0)
    low = ConstantDispersion.from_nk("B_low", n=float(task["n_low"]), k=0.0)
    modifiers = None
    if task["absorbing"]:
        modifiers = [ExtinctionModifier(
            target="both_materials",
            mode="substitute",
            profile=float(task["k_all_layers"]),
            name=f"all_layers_k_{task['k_all_layers']:.6f}",
        )]
    return high, low, modifiers


def rebuild_case_stack(task: dict):
    high, low, modifiers = make_materials_and_modifier(task)
    row = pd.Series({
        "peak_wavelength_m": float(task["design_wavelength_nm"]) * 1e-9,
        "n_layers": int(task["n_layers"]),
        "outer_layer": str(task["outer_layer"]),
        "core_thickness_factor": float(task["core_thickness_factor"]),
    })
    geometry, stack = rebuild_stack_from_row(
        row, material_a=high, material_b=low, fixed_geometry_kwargs={},
        extinction_modifiers=modifiers, extra_outer_shell=None,
    )
    return geometry, stack


def run_backward_spectrum(task, wavelengths_nm, theta_deg, collection_na):
    high, low, modifiers = make_materials_and_modifier(task)
    sweep = run_bragg_onion_sweep(
        geometry_mode="peak_wavelength",
        parameter_grid={
            "peak_wavelength_m": [float(task["design_wavelength_nm"]) * 1e-9],
            "n_layers": [int(task["n_layers"])],
            "outer_layer": [str(task["outer_layer"])],
            "core_thickness_factor": [float(task["core_thickness_factor"])],
        },
        material_a=high, material_b=low, n_medium=1.0,
        wavelengths_m=np.asarray(wavelengths_nm) * 1e-9,
        theta_rad=np.deg2rad(np.asarray(theta_deg)),
        collection_na=float(collection_na), collection_direction="backward",
        integration_quantity_for_colour="q_collected_geom",
        colour_wavelength_min_nm=None, colour_wavelength_max_nm=None,
        colour_normalize_input=False, colour_normalization="max",
        extinction_modifiers=modifiers, extra_outer_shell=None,
        store_full_results=True, progress=False,
    )
    wavelength_nm, spectrum = extract_backward_spectrum(sweep)
    return wavelength_nm, spectrum, extract_scalar_sweep_summary(sweep)


def nearest_unique(values, candidates):
    return np.unique([candidates[int(np.argmin(np.abs(candidates - value)))] for value in values])


def safe_square(values):
    values = np.asarray(values, dtype=float)
    output = np.full_like(values, np.nan)
    with np.errstate(over="ignore", invalid="ignore"):
        np.square(values, out=output, where=np.isfinite(values))
    output[~np.isfinite(output)] = np.nan
    return output


def extract_profiles_and_shells(field_map, task, wavelength_nm, design_nm, peak_nm):
    x_nm = np.asarray(field_map.coord1_m, dtype=float) * 1e9
    z_nm = np.asarray(field_map.coord2_m, dtype=float) * 1e9
    X_nm, Z_nm = np.meshgrid(x_nm, z_nm, indexing="xy")
    radius_nm_grid = np.hypot(X_nm, Z_nm)
    radii_nm = np.asarray(field_map.radii_m, dtype=float) * 1e9
    e_map = np.asarray(field_map.e_total_mag, dtype=float)
    intensity_map = safe_square(e_map)

    ix = int(np.argmin(np.abs(x_nm)))
    e_axis = e_map[:, ix]
    intensity_axis = intensity_map[:, ix]
    radial_axis_nm = np.abs(z_nm)
    shell_axis = np.searchsorted(radii_nm, radial_axis_nm, side="left")
    inside_axis = radial_axis_nm <= radii_nm[-1]

    common = {
        "task_index": int(task["task_index"]), "task_hash": task["task_hash"],
        "case_label": task["case_label"], "n_low": float(task["n_low"]),
        "n_high": float(task["n_high"]), "delta_n": float(task["n_high"] - task["n_low"]),
        "k_all_layers": float(task["k_all_layers"]), "absorbing": bool(task["absorbing"]),
        "wavelength_nm": float(wavelength_nm),
        "delta_design_nm": float(wavelength_nm - design_nm),
        "delta_peak_nm": float(wavelength_nm - peak_nm),
    }
    profile = pd.DataFrame({
        **common,
        "z_nm": z_nm,
        "z_over_outer_radius": z_nm / radii_nm[-1],
        "radius_nm": radial_axis_nm,
        "inside_sphere": inside_axis,
        "shell_index_zero_based": np.where(inside_axis, shell_axis, -1),
        "shell_number_one_based": np.where(inside_axis, shell_axis + 1, 0),
        "E_magnitude_enhancement": e_axis,
        "E_intensity_enhancement": intensity_axis,
    })

    shell_records = []
    for shell_zero, outer_radius_nm in enumerate(radii_nm):
        inner_radius_nm = 0.0 if shell_zero == 0 else float(radii_nm[shell_zero - 1])
        plane_selection = (
            (radius_nm_grid > inner_radius_nm if shell_zero > 0 else radius_nm_grid >= 0.0)
            & (radius_nm_grid <= outer_radius_nm)
        )
        axis_selection = inside_axis & (shell_axis == shell_zero)
        plane_e = e_map[plane_selection & np.isfinite(e_map)]
        plane_i = intensity_map[plane_selection & np.isfinite(intensity_map)]
        axis_e = e_axis[axis_selection & np.isfinite(e_axis)]
        axis_i = intensity_axis[axis_selection & np.isfinite(intensity_axis)]
        shell_records.append({
            **common,
            "shell_index_zero_based": shell_zero,
            "shell_number_one_based": shell_zero + 1,
            "shell_label": "core" if shell_zero == 0 else f"shell_{shell_zero + 1:03d}",
            "inner_radius_nm": inner_radius_nm,
            "outer_radius_nm": float(outer_radius_nm),
            "plane_sample_count": int(plane_e.size),
            "axis_sample_count": int(axis_e.size),
            "mean_E_magnitude_plane": float(np.nanmean(plane_e)) if plane_e.size else np.nan,
            "median_E_magnitude_plane": float(np.nanmedian(plane_e)) if plane_e.size else np.nan,
            "max_E_magnitude_plane": float(np.nanmax(plane_e)) if plane_e.size else np.nan,
            "mean_E_intensity_plane": float(np.nanmean(plane_i)) if plane_i.size else np.nan,
            "mean_E_magnitude_axis": float(np.nanmean(axis_e)) if axis_e.size else np.nan,
            "mean_E_intensity_axis": float(np.nanmean(axis_i)) if axis_i.size else np.nan,
        })
    return profile, pd.DataFrame(shell_records), radii_nm


def run_task(task, args_dict, wavelengths_nm, theta_deg):
    args = argparse.Namespace(**args_dict)
    output_dir = Path(args.output_dir)
    checkpoint_path = output_dir / "case_checkpoints" / f"case_{task['task_index']:03d}_{task['task_hash']}.pkl"
    if checkpoint_path.exists() and not args.overwrite:
        return {"status": "loaded", "case_label": task["case_label"]}

    wavelengths_nm = np.asarray(wavelengths_nm, dtype=float)
    spectrum_nm, spectrum, scalar_summary = run_backward_spectrum(
        task, wavelengths_nm, np.asarray(theta_deg), args.collection_na
    )
    peak_nm, smoothed_spectrum, peak_contrast = select_local_peak(
        spectrum_nm, spectrum,
        search_min_nm=args.peak_search_min_nm,
        search_max_nm=args.peak_search_max_nm,
        smoothing_nm=args.peak_smoothing_nm,
    )

    requested = np.arange(
        peak_nm - args.field_offset_nm,
        peak_nm + args.field_offset_nm + 0.5 * args.field_step_nm,
        args.field_step_nm,
    )
    requested = np.append(requested, [DESIGN_WAVELENGTH_NM, peak_nm])
    requested = requested[(requested >= wavelengths_nm.min()) & (requested <= wavelengths_nm.max())]
    field_wavelengths_nm = nearest_unique(requested, wavelengths_nm)

    geometry, stack = rebuild_case_stack(task)
    profiles, shells = [], []
    radii_nm = None
    for index, wavelength_nm in enumerate(field_wavelengths_nm, start=1):
        print(f"[{task['case_label']}] field {index}/{len(field_wavelengths_nm)} at {wavelength_nm:.1f} nm", flush=True)
        field_map = compute_field_map(
            stack=stack, wavelength_m=float(wavelength_nm) * 1e-9, n_medium=1.0,
            plane="xz", npts=int(args.field_npts),
            extent_outer_radius_factor=float(args.field_extent_factor),
            incident_e0=1.0 + 0.0j,
        )
        profile, shell_table, radii_nm = extract_profiles_and_shells(
            field_map, task, float(wavelength_nm), DESIGN_WAVELENGTH_NM, peak_nm
        )
        profiles.append(profile)
        shells.append(shell_table)

    payload = {
        "cache_version": CACHE_VERSION, "task": task,
        "spectrum_wavelength_nm": spectrum_nm,
        "backward_spectrum": spectrum,
        "backward_spectrum_smoothed": smoothed_spectrum,
        "backward_peak_wavelength_nm": peak_nm,
        "peak_search_min_nm": args.peak_search_min_nm,
        "peak_search_max_nm": args.peak_search_max_nm,
        "peak_smoothing_nm": args.peak_smoothing_nm,
        "peak_contrast": peak_contrast,
        "field_wavelengths_nm": field_wavelengths_nm,
        "outer_radius_nm": float(geometry.outer_radius_m * 1e9),
        "radii_nm": radii_nm,
        "profiles": pd.concat(profiles, ignore_index=True),
        "shell_plane_averages": pd.concat(shells, ignore_index=True),
        "sweep_scalar_summary": scalar_summary,
        "method_note": (
            "E_magnitude_enhancement=|E_total|/|E0| and E_intensity_enhancement=|E_total|^2/|E0|^2, E0=1. "
            "Shell plane averages are area averages over annuli in the central XZ cross-section."
        ),
    }
    atomic_pickle_dump(payload, checkpoint_path)
    return {"status": "calculated", "case_label": task["case_label"], "peak_nm": peak_nm}


def combine_results(output_dir: Path):
    paths = sorted((output_dir / "case_checkpoints").glob("case_*.pkl"))
    if not paths:
        raise FileNotFoundError("No case checkpoints are available")
    spectra, profiles, shells, summaries = [], [], [], []
    for path in paths:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        task = payload["task"]
        spectra.append(pd.DataFrame({
            "task_index": int(task["task_index"]), "task_hash": task["task_hash"],
            "case_label": task["case_label"], "n_low": float(task["n_low"]),
            "n_high": float(task["n_high"]), "delta_n": float(task["n_high"] - task["n_low"]),
            "k_all_layers": float(task["k_all_layers"]), "absorbing": bool(task["absorbing"]),
            "wavelength_nm": payload["spectrum_wavelength_nm"],
            "backward_collected_spectrum": payload["backward_spectrum"],
            "backward_collected_spectrum_smoothed": payload["backward_spectrum_smoothed"],
            "backward_peak_wavelength_nm": float(payload["backward_peak_wavelength_nm"]),
        }))
        profiles.append(payload["profiles"])
        shells.append(payload["shell_plane_averages"])
        summaries.append({
            **task,
            "delta_n": float(task["n_high"] - task["n_low"]),
            "backward_peak_wavelength_nm": float(payload["backward_peak_wavelength_nm"]),
            "peak_shift_from_design_nm": float(payload["backward_peak_wavelength_nm"] - DESIGN_WAVELENGTH_NM),
            "peak_contrast": float(payload["peak_contrast"]),
            "peak_search_min_nm": float(payload["peak_search_min_nm"]),
            "peak_search_max_nm": float(payload["peak_search_max_nm"]),
            "peak_smoothing_nm": float(payload["peak_smoothing_nm"]),
            "outer_radius_nm": float(payload["outer_radius_nm"]),
            "field_wavelength_count": len(payload["field_wavelengths_nm"]),
            **payload.get("sweep_scalar_summary", {}),
        })
    tables = {
        "case_summary": pd.DataFrame(summaries).sort_values(["n_layers", "absorbing", "n_high"]).reset_index(drop=True),
        "backward_spectra": pd.concat(spectra, ignore_index=True),
        "axis_profiles": pd.concat(profiles, ignore_index=True),
        "shell_plane_averages": pd.concat(shells, ignore_index=True),
    }
    for stem, frame in tables.items():
        frame.to_pickle(output_dir / f"{stem}.pkl")
        frame.to_csv(output_dir / f"{stem}.csv", index=False)
        if parquet_available():
            frame.to_parquet(output_dir / f"{stem}.parquet", index=False)
        print(stem, frame.shape)


def main():
    parser = argparse.ArgumentParser(description="Electric-field enhancement versus refractive-index contrast and sphere size")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "core_field_enhancement_v4_small_spheres")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--wavelength-min-nm", type=float, default=300.0)
    parser.add_argument("--wavelength-max-nm", type=float, default=900.0)
    parser.add_argument("--wavelength-step-nm", type=float, default=1.0)
    parser.add_argument("--theta-points", type=int, default=181)
    parser.add_argument("--collection-na", type=float, default=1.0)
    parser.add_argument("--peak-search-min-nm", type=float, default=350.0)
    parser.add_argument("--peak-search-max-nm", type=float, default=600.0)
    parser.add_argument("--peak-smoothing-nm", type=float, default=21.0)
    parser.add_argument("--field-offset-nm", type=float, default=150.0)
    parser.add_argument("--field-step-nm", type=float, default=10.0)
    parser.add_argument("--field-npts", type=int, default=401)
    parser.add_argument("--field-extent-factor", type=float, default=1.001)
    parser.add_argument("--task-mod", type=int, default=1)
    parser.add_argument("--task-rem", type=int, default=0)
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--combine-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if not np.any(np.isclose(args.wavelength_step_nm, [1.0, 5.0, 10.0, 20.0], rtol=0, atol=1e-12)):
        raise ValueError("--wavelength-step-nm must be 1, 5, 10, or 20 nm because the sweep performs ASTM E308 conversion")
    if args.field_npts < 121 or args.field_npts % 2 == 0:
        raise ValueError("--field-npts must be odd and >=121; use >=401 for shell-resolved analysis; 801 is recommended for central-mode extent")
    if args.field_extent_factor < 1.0:
        raise ValueError("--field-extent-factor must be >=1")
    if args.peak_search_min_nm >= args.peak_search_max_nm:
        raise ValueError("Invalid peak-search window")

    output_dir = args.output_dir.resolve()
    checkpoint_dir = output_dir / "case_checkpoints"
    failure_dir = output_dir / "failures"
    for directory in (output_dir, checkpoint_dir, failure_dir):
        directory.mkdir(parents=True, exist_ok=True)

    tasks_all = make_tasks()
    tasks = [task for task in tasks_all if task["task_index"] % args.task_mod == args.task_rem]
    pd.DataFrame(tasks_all).to_csv(output_dir / "task_manifest_all.csv", index=False)
    metadata = {"cache_version": CACHE_VERSION, "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if args.manifest_only:
        return
    if args.combine_only:
        combine_results(output_dir)
        return

    wavelengths_nm = np.arange(args.wavelength_min_nm, args.wavelength_max_nm + 0.5 * args.wavelength_step_nm, args.wavelength_step_nm)
    theta_deg = np.linspace(0.0, 180.0, args.theta_points)
    pending = [task for task in tasks if args.overwrite or not (checkpoint_dir / f"case_{task['task_index']:03d}_{task['task_hash']}.pkl").exists()]
    print(f"Cases: {len(tasks)}; completed: {len(tasks)-len(pending)}; pending: {len(pending)}")
    args_dict = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    args_dict["output_dir"] = str(output_dir)
    failures = []
    start = perf_counter()
    if pending:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(pending))) as executor:
            futures = {executor.submit(run_task, task, args_dict, wavelengths_nm.tolist(), theta_deg.tolist()): task for task in pending}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Cases"):
                task = futures[future]
                try:
                    print(future.result(), flush=True)
                except Exception as exc:
                    failures.append({**task, "error_type": type(exc).__name__, "error_message": str(exc), "traceback": traceback.format_exc()})
                    print("FAILED", task["case_label"], exc, flush=True)
    failure_path = failure_dir / "failed_tasks.csv"
    if failures:
        pd.DataFrame(failures).to_csv(failure_path, index=False)
    elif failure_path.exists():
        failure_path.unlink()
    if list(checkpoint_dir.glob("case_*.pkl")):
        combine_results(output_dir)
    print(f"Finished in {(perf_counter()-start)/3600:.2f} h")
    if failures:
        raise RuntimeError(f"{len(failures)} case(s) failed; inspect {failure_path}")


if __name__ == "__main__":
    main()
