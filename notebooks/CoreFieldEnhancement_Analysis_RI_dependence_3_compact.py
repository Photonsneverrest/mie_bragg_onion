#!/usr/bin/env python3
# %% [markdown]
# # Compact analysis of the core-field-enhancement sweep
#
# Goals:
# - separate spectra and axis profiles by number of layers;
# - switch between lossless, absorbing, or both;
# - inspect raw profiles without masking;
# - map where numerical divergence occurs;
# - study wavelength dependence only for fully finite profiles;
# - optionally recompute one focused XZ field map around the center.

# %% Imports and settings
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, Normalize
from IPython.display import display

OUTPUT_DIR = Path(os.environ.get(
    "CORE_FIELD_OUTPUT_DIR",
    "/mnt/d/Rocca/MieScattering/outputs/field_enhancement_sweep",
))
FIGURE_DIR = OUTPUT_DIR / "analysis_figures_compact"
TABLE_DIR = OUTPUT_DIR / "analysis_tables_compact"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

DESIGN_WAVELENGTH_NM = 470.0
PNG_DPI = 300

# "lossless", "absorbing", or "both"
ABSORPTION_VIEW = "lossless"

# Layer panels to show. None means all available layer counts.
LAYER_COUNTS = None

# Spectrum controls.
SPECTRUM_COLUMN = "backward_collected_spectrum"  # raw by default
SPECTRUM_XLIM = (300.0, 900.0)
SPECTRUM_YLIM = None

# Raw-profile controls. Add/remove exact case labels as needed.
RAW_PROFILE_CASES = [
    "L_005_nH_1.001_lossless",
    "L_005_nH_2.500_lossless",
    "L_005_nH_1.001_k_0.010",
    "L_005_nH_2.500_k_0.010",
    "L_080_nH_2.500_lossless",
    "L_080_nH_2.500_k_0.010",
]
RAW_PROFILE_TARGET_NM = DESIGN_WAVELENGTH_NM
RAW_PROFILE_MAX_ABS = None  # None = no masking and no clipping
RAW_PROFILE_LOG_Y = False

# Numerical-divergence criterion used only for diagnostics.
DIVERGENCE_LIMIT = 1e6

# Finite-profile wavelength study.
FINITE_CASE_LABEL = "L_020_nH_2.000_lossless"
FINITE_REQUIRE_ALL_SAMPLES = True
FINITE_MIN_VALID_FRACTION = 0.999

# Focused raw center inspection.
CENTER_CASE_LABEL = "L_020_nH_2.000_lossless"
CENTER_WAVELENGTH_NM = DESIGN_WAVELENGTH_NM
CENTER_HALF_WIDTH_Z_OVER_R = 0.60

# Optional XZ recomputation. This runs only when explicitly enabled.
RUN_FOCUSED_FIELD_RECOMPUTE = False
RECOMPUTE_CASE_LABEL = CENTER_CASE_LABEL
RECOMPUTE_WAVELENGTH_NM = CENTER_WAVELENGTH_NM
RECOMPUTE_NPTS = 801
RECOMPUTE_EXTENT_FACTOR = 0.60  # focused square: +/- 0.60 R if supported
REPO_ROOT = Path(os.environ.get("BRAGG_ONION_REPO", Path.cwd()))


def save_figure(fig, stem):
    fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=PNG_DPI, bbox_inches="tight")
    fig.savefig(FIGURE_DIR / f"{stem}.pdf", bbox_inches="tight")


def load_preferred(stem):
    for suffix, loader in (
        (".parquet", pd.read_parquet),
        (".pkl", pd.read_pickle),
        (".csv", pd.read_csv),
    ):
        path = OUTPUT_DIR / f"{stem}{suffix}"
        if path.exists():
            frame = loader(path)
            print(f"Loaded {path.name}: {frame.shape}")
            return frame
    raise FileNotFoundError(f"No {stem} table found in {OUTPUT_DIR}")


def absorption_mask(frame, view=ABSORPTION_VIEW):
    absorbing = frame["absorbing"].astype(bool)
    if view == "lossless":
        return ~absorbing
    if view == "absorbing":
        return absorbing
    if view == "both":
        return pd.Series(True, index=frame.index)
    raise ValueError("ABSORPTION_VIEW must be 'lossless', 'absorbing', or 'both'")


def nearest(values, target):
    values = np.asarray(pd.Series(values).dropna().unique(), dtype=float)
    return np.nan if values.size == 0 else float(values[np.argmin(abs(values - target))])


# %% Load and normalize
case_summary = load_preferred("case_summary")
backward_spectra = load_preferred("backward_spectra")
axis_profiles = load_preferred("axis_profiles")
shell_plane_averages = load_preferred("shell_plane_averages")

for frame in (case_summary, backward_spectra, axis_profiles, shell_plane_averages):
    if "absorbing" in frame:
        frame["absorbing"] = frame["absorbing"].astype(str).str.lower().map(
            {"true": True, "false": False, "1": True, "0": False}
        ).fillna(frame["absorbing"]).astype(bool)

numeric_columns = [
    "n_layers", "n_low", "n_high", "k_all_layers", "wavelength_nm",
    "z_nm", "z_over_outer_radius", "E_magnitude_enhancement",
    "outer_radius_nm", "backward_collected_spectrum",
]
for frame in (case_summary, backward_spectra, axis_profiles, shell_plane_averages):
    for column in numeric_columns:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

if "n_layers" not in backward_spectra:
    backward_spectra = backward_spectra.merge(
        case_summary[["case_label", "n_layers"]], on="case_label", how="left", validate="many_to_one"
    )
if "n_layers" not in axis_profiles:
    axis_profiles = axis_profiles.merge(
        case_summary[["case_label", "n_layers"]], on="case_label", how="left", validate="many_to_one"
    )

available_layers = sorted(case_summary["n_layers"].dropna().astype(int).unique())
layer_counts = available_layers if LAYER_COUNTS is None else [n for n in LAYER_COUNTS if n in available_layers]
print("Available layer counts:", available_layers)
print("Displayed layer counts:", layer_counts)

# %% Data-quality map: where does numerical divergence occur?
raw = pd.to_numeric(axis_profiles["E_magnitude_enhancement"], errors="coerce")
axis_profiles["field_finite"] = np.isfinite(raw)
axis_profiles["field_divergent"] = (~np.isfinite(raw)) | (np.abs(raw) > DIVERGENCE_LIMIT)
axis_profiles["field_valid_under_limit"] = np.isfinite(raw) & (np.abs(raw) <= DIVERGENCE_LIMIT)

quality = (
    axis_profiles.groupby(
        ["case_label", "n_layers", "n_high", "k_all_layers", "absorbing", "wavelength_nm"],
        as_index=False,
    )
    .agg(
        samples=("E_magnitude_enhancement", "size"),
        finite_fraction=("field_finite", "mean"),
        divergent_fraction=("field_divergent", "mean"),
        valid_fraction=("field_valid_under_limit", "mean"),
        raw_max=("E_magnitude_enhancement", lambda x: np.nanmax(np.abs(pd.to_numeric(x, errors="coerce")))),
    )
)
quality.to_csv(TABLE_DIR / "axis_profile_numerical_quality.csv", index=False)

display(
    quality.groupby(["n_layers", "absorbing", "n_high"], as_index=False)
    .agg(
        median_valid_fraction=("valid_fraction", "median"),
        worst_valid_fraction=("valid_fraction", "min"),
        maximum_raw_field=("raw_max", "max"),
    )
    .sort_values(["n_layers", "absorbing", "n_high"])
)

# %% Backward spectra in one panel per layer count
spectra = backward_spectra.loc[absorption_mask(backward_spectra)].copy()
if SPECTRUM_COLUMN not in spectra:
    raise KeyError(f"Missing spectrum column: {SPECTRUM_COLUMN}")

ncols = min(3, len(layer_counts))
nrows = int(np.ceil(len(layer_counts) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 3.4 * nrows), sharex=True, sharey=True, constrained_layout=True)
axes = np.atleast_1d(axes).ravel()
norm_n = Normalize(case_summary["n_high"].min(), case_summary["n_high"].max())
cmap_n = plt.get_cmap("viridis")

for ax, n_layers in zip(axes, layer_counts):
    panel = spectra.loc[spectra["n_layers"] == n_layers]
    for case_label, data in panel.groupby("case_label"):
        data = data.sort_values("wavelength_nm")
        meta = data.iloc[0]
        ax.plot(
            data["wavelength_nm"], data[SPECTRUM_COLUMN],
            color=cmap_n(norm_n(meta["n_high"])),
            linestyle="--" if bool(meta["absorbing"]) else "-",
            linewidth=0.9,
        )
    ax.axvline(DESIGN_WAVELENGTH_NM, color="black", linestyle=":", linewidth=0.8)
    ax.set_title(f"{n_layers} layers")
    ax.set_xlim(*SPECTRUM_XLIM)
    if SPECTRUM_YLIM is not None:
        ax.set_ylim(*SPECTRUM_YLIM)
    ax.set_xlabel("Wavelength [nm]")
    ax.set_ylabel("Backward-collected efficiency")

for ax in axes[len(layer_counts):]:
    ax.remove()
sm = ScalarMappable(norm=norm_n, cmap=cmap_n); sm.set_array([])
fig.colorbar(sm, ax=axes[:len(layer_counts)].tolist(), label=r"$n_H$")
fig.suptitle(f"Raw backward spectra: {ABSORPTION_VIEW}")
save_figure(fig, f"backward_spectra_by_layers_{ABSORPTION_VIEW}_{SPECTRUM_COLUMN}")
plt.show()

# %% Axis profiles in one panel per layer count, raw values
profiles_view = axis_profiles.loc[absorption_mask(axis_profiles)].copy()
fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 3.4 * nrows), sharex=True, constrained_layout=True)
axes = np.atleast_1d(axes).ravel()

for ax, n_layers in zip(axes, layer_counts):
    panel_cases = case_summary.loc[(case_summary["n_layers"] == n_layers) & absorption_mask(case_summary)]
    for _, case in panel_cases.iterrows():
        available = profiles_view.loc[profiles_view["case_label"] == case["case_label"], "wavelength_nm"]
        selected_nm = nearest(available, DESIGN_WAVELENGTH_NM)
        data = profiles_view.loc[
            (profiles_view["case_label"] == case["case_label"])
            & np.isclose(profiles_view["wavelength_nm"], selected_nm)
            & profiles_view["inside_sphere"].astype(bool)
        ].sort_values("z_over_outer_radius")
        y = pd.to_numeric(data["E_magnitude_enhancement"], errors="coerce")
        # Raw plot: only nonfinite values are naturally absent. No numerical ceiling is imposed.
        ax.plot(data["z_over_outer_radius"], y, color=cmap_n(norm_n(case["n_high"])),
                linestyle="--" if case["absorbing"] else "-", linewidth=0.75)
    ax.set_title(f"{n_layers} layers")
    ax.set_xlim(-1, 1)
    ax.set_xlabel(r"$z/R$")
    ax.set_ylabel(r"Raw $|E|/|E_0|$")

for ax in axes[len(layer_counts):]:
    ax.remove()
fig.colorbar(sm, ax=axes[:len(layer_counts)].tolist(), label=r"$n_H$")
fig.suptitle(f"Raw central-axis profiles near {DESIGN_WAVELENGTH_NM:.0f} nm: {ABSORPTION_VIEW}")
save_figure(fig, f"raw_axis_profiles_by_layers_{ABSORPTION_VIEW}")
plt.show()

# %% Explicit raw-profile inspector
fig, axes = plt.subplots(len(RAW_PROFILE_CASES), 1, figsize=(9, max(2.4, 2.2 * len(RAW_PROFILE_CASES))), sharex=True, constrained_layout=True)
axes = np.atleast_1d(axes)
raw_profile_records = []

for ax, case_label in zip(axes, RAW_PROFILE_CASES):
    available = axis_profiles.loc[axis_profiles["case_label"] == case_label, "wavelength_nm"]
    selected_nm = nearest(available, RAW_PROFILE_TARGET_NM)
    data = axis_profiles.loc[
        (axis_profiles["case_label"] == case_label)
        & np.isclose(axis_profiles["wavelength_nm"], selected_nm)
        & axis_profiles["inside_sphere"].astype(bool)
    ].sort_values("z_over_outer_radius")
    y = pd.to_numeric(data["E_magnitude_enhancement"], errors="coerce")
    if RAW_PROFILE_MAX_ABS is not None:
        y = y.where(np.abs(y) <= RAW_PROFILE_MAX_ABS)
    ax.plot(data["z_over_outer_radius"], y, linewidth=0.9)
    if RAW_PROFILE_LOG_Y:
        ax.set_yscale("log")
    ax.set_title(f"{case_label}, {selected_nm:.1f} nm")
    ax.set_ylabel(r"Raw $|E|/|E_0|$")
    raw_profile_records.append({
        "case_label": case_label,
        "selected_wavelength_nm": selected_nm,
        "finite_fraction": float(np.isfinite(y).mean()) if len(y) else np.nan,
        "maximum_finite": float(np.nanmax(y)) if np.isfinite(y).any() else np.nan,
    })
axes[-1].set_xlabel(r"Propagation coordinate $z/R$")
for ax in axes:
    ax.set_xlim(-1, 1)
save_figure(fig, "selected_raw_axis_profiles")
plt.show()
display(pd.DataFrame(raw_profile_records))

# %% Wavelength dependence for finite profiles only
case_data = axis_profiles.loc[
    (axis_profiles["case_label"] == FINITE_CASE_LABEL)
    & axis_profiles["inside_sphere"].astype(bool)
].copy()
finite_quality = (
    case_data.groupby("wavelength_nm", as_index=False)
    .agg(valid_fraction=("field_valid_under_limit", "mean"), raw_max=("E_magnitude_enhancement", "max"))
)
finite_wavelengths = finite_quality.loc[
    finite_quality["valid_fraction"] >= FINITE_MIN_VALID_FRACTION,
    "wavelength_nm",
].to_numpy(float)
if FINITE_REQUIRE_ALL_SAMPLES:
    finite_wavelengths = finite_quality.loc[np.isclose(finite_quality["valid_fraction"], 1.0), "wavelength_nm"].to_numpy(float)

print(f"{FINITE_CASE_LABEL}: {len(finite_wavelengths)} fully/acceptably finite wavelengths")
if finite_wavelengths.size:
    wave_norm = Normalize(finite_wavelengths.min(), finite_wavelengths.max())
    wave_cmap = plt.get_cmap("plasma")
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    metrics = []
    for wavelength_nm in finite_wavelengths:
        data = case_data.loc[np.isclose(case_data["wavelength_nm"], wavelength_nm)].sort_values("z_over_outer_radius")
        y = pd.to_numeric(data["E_magnitude_enhancement"], errors="coerce")
        z = data["z_over_outer_radius"].to_numpy(float)
        ax.plot(z, y, color=wave_cmap(wave_norm(wavelength_nm)), linewidth=0.8)
        center = y[np.abs(z) <= 0.5]
        metrics.append({
            "wavelength_nm": wavelength_nm,
            "center_median": float(np.nanmedian(center)),
            "center_p95": float(np.nanpercentile(center, 95)),
            "center_max": float(np.nanmax(center)),
            "whole_max": float(np.nanmax(y)),
        })
    ax.set(xlim=(-1, 1), xlabel=r"$z/R$", ylabel=r"Raw finite $|E|/|E_0|$",
           title=f"Finite raw profiles: {FINITE_CASE_LABEL}")
    sw = ScalarMappable(norm=wave_norm, cmap=wave_cmap); sw.set_array([])
    fig.colorbar(sw, ax=ax, label="Wavelength [nm]")
    save_figure(fig, f"finite_wavelength_profiles_{FINITE_CASE_LABEL}")
    plt.show()
    finite_metrics = pd.DataFrame(metrics)
    finite_metrics.to_csv(TABLE_DIR / f"finite_wavelength_metrics_{FINITE_CASE_LABEL}.csv", index=False)
    display(finite_metrics)
else:
    print("No wavelength satisfies the finite-profile criterion. Inspect the quality table or relax FINITE_MIN_VALID_FRACTION.")

# %% Raw center-region inspection: no masking and no clipping
available = axis_profiles.loc[axis_profiles["case_label"] == CENTER_CASE_LABEL, "wavelength_nm"]
center_nm = nearest(available, CENTER_WAVELENGTH_NM)
center_data = axis_profiles.loc[
    (axis_profiles["case_label"] == CENTER_CASE_LABEL)
    & np.isclose(axis_profiles["wavelength_nm"], center_nm)
    & axis_profiles["inside_sphere"].astype(bool)
].sort_values("z_over_outer_radius").copy()
center_data = center_data.loc[np.abs(center_data["z_over_outer_radius"]) <= CENTER_HALF_WIDTH_Z_OVER_R]
center_data["raw_field"] = pd.to_numeric(center_data["E_magnitude_enhancement"], errors="coerce")
center_data["log10_raw_field"] = np.log10(center_data["raw_field"].where(center_data["raw_field"] > 0))

fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True, constrained_layout=True)
axes[0].plot(center_data["z_over_outer_radius"], center_data["raw_field"], linewidth=0.9)
axes[0].set_ylabel(r"Raw $|E|/|E_0|$")
axes[1].plot(center_data["z_over_outer_radius"], center_data["log10_raw_field"], linewidth=0.9)
axes[1].set(xlabel=r"$z/R$", ylabel=r"$\log_{10}(|E|/|E_0|)$")
fig.suptitle(f"Unmodified center region: {CENTER_CASE_LABEL}, {center_nm:.1f} nm")
save_figure(fig, f"raw_center_region_{CENTER_CASE_LABEL}_{center_nm:.0f}nm")
plt.show()
center_data.to_csv(TABLE_DIR / f"raw_center_region_{CENTER_CASE_LABEL}_{center_nm:.0f}nm.csv", index=False)
print(center_data[["raw_field", "log10_raw_field"]].describe())

# %% Optional focused field-map recomputation
# This block recomputes an XZ map for one case. It does not change the multipole
# truncation because the current public call used by the sweep exposes no lmax/nmax
# argument. It is useful for checking whether the extreme region has coherent 2D
# spatial structure. Leave RUN_FOCUSED_FIELD_RECOMPUTE=False for ordinary analysis.
if RUN_FOCUSED_FIELD_RECOMPUTE:
    src_dir = REPO_ROOT / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from bragg_onion.materials import ConstantDispersion
    from bragg_onion.geometry import ExtinctionModifier
    from bragg_onion.fields import compute_field_map
    from bragg_onion._notebook_tools.field_analysis import rebuild_stack_from_row

    case = case_summary.loc[case_summary["case_label"] == RECOMPUTE_CASE_LABEL].iloc[0]
    high = ConstantDispersion.from_nk("A_high", n=float(case["n_high"]), k=0.0)
    low = ConstantDispersion.from_nk("B_low", n=float(case["n_low"]), k=0.0)
    modifiers = None
    if bool(case["absorbing"]):
        modifiers = [ExtinctionModifier(
            target="both_materials", mode="substitute", profile=float(case["k_all_layers"]), name="focused_test"
        )]
    row = pd.Series({
        "peak_wavelength_m": DESIGN_WAVELENGTH_NM * 1e-9,
        "n_layers": int(case["n_layers"]),
        "outer_layer": str(case.get("outer_layer", "A")),
        "core_thickness_factor": float(case.get("core_thickness_factor", 1.0)),
    })
    geometry, stack = rebuild_stack_from_row(
        row, material_a=high, material_b=low, fixed_geometry_kwargs={},
        extinction_modifiers=modifiers, extra_outer_shell=None,
    )
    # compute_field_map normally expects extent >= 1.0. If a focused extent below
    # 1.0 is unsupported in your installed version, set RECOMPUTE_EXTENT_FACTOR=1.0
    # and crop the returned map afterward.
    fmap = compute_field_map(
        stack=stack,
        wavelength_m=RECOMPUTE_WAVELENGTH_NM * 1e-9,
        n_medium=1.0,
        plane="xz",
        npts=RECOMPUTE_NPTS,
        extent_outer_radius_factor=max(1.0, RECOMPUTE_EXTENT_FACTOR),
        incident_e0=1.0 + 0.0j,
    )
    x = np.asarray(fmap.coord1_m) / geometry.outer_radius_m
    z = np.asarray(fmap.coord2_m) / geometry.outer_radius_m
    field = np.asarray(fmap.e_total_mag, dtype=float)
    crop_x = np.abs(x) <= RECOMPUTE_EXTENT_FACTOR
    crop_z = np.abs(z) <= RECOMPUTE_EXTENT_FACTOR
    field_crop = field[np.ix_(crop_z, crop_x)]
    finite_positive = field_crop[np.isfinite(field_crop) & (field_crop > 0)]
    vmax = np.nanpercentile(finite_positive, 99.5) if finite_positive.size else 1.0

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    im0 = axes[0].imshow(field_crop, origin="lower", extent=[x[crop_x].min(), x[crop_x].max(), z[crop_z].min(), z[crop_z].max()],
                         aspect="equal", vmin=0, vmax=vmax, cmap="magma")
    fig.colorbar(im0, ax=axes[0], label=r"$|E|/|E_0|$")
    axes[0].set_title("Linear, 99.5th-percentile scale")
    positive_floor = max(np.nanpercentile(finite_positive, 1), 1e-12) if finite_positive.size else 1e-12
    im1 = axes[1].imshow(field_crop, origin="lower", extent=[x[crop_x].min(), x[crop_x].max(), z[crop_z].min(), z[crop_z].max()],
                         aspect="equal", norm=LogNorm(vmin=positive_floor, vmax=max(vmax, positive_floor * 10)), cmap="magma")
    fig.colorbar(im1, ax=axes[1], label=r"$|E|/|E_0|$")
    axes[1].set_title("Logarithmic")
    for ax in axes:
        ax.set(xlabel=r"$x/R$", ylabel=r"$z/R$")
    fig.suptitle(f"Focused recomputation: {RECOMPUTE_CASE_LABEL}, {RECOMPUTE_WAVELENGTH_NM:.1f} nm")
    save_figure(fig, f"focused_recompute_{RECOMPUTE_CASE_LABEL}_{RECOMPUTE_WAVELENGTH_NM:.0f}nm")
    plt.show()

# %% Summary
print("Analysis complete.")
print("Figures:", FIGURE_DIR)
print("Tables:", TABLE_DIR)
print("Important: raw plots contain no finite-value ceiling; screened values are used only in the quality diagnostics.")
