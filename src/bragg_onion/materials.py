# dispersion models / interpolation
"""
# materials.py
#
# Utilities for loading, storing, and evaluating wavelength-dependent
# refractive indices n(λ) + i k(λ) for scattering simulations.
#
# Key features:
# - load material data from text files with custom column names / skipped rows
# - support files containing only n or both n and k
# - handle wavelength units explicitly ("nm", "um", "m")
# - interpolate onto arbitrary wavelength grids
# - configurable extrapolation: "error", "hold", "nan", or "extrapolate"
#
# Main classes:
# - Dispersion: generic callable material object
# - ConstantDispersion: wavelength-independent material
# - TabulatedDispersion: interpolated material from tabulated data
# - MaterialFileSpec: per-file loading specification
#
# Main functions:
# - load_tabulated_material(spec): load one material from file
# - load_materials(specs): load multiple materials from file specs
# - load_and_interpolate_legacy(...): compatibility helper for older scripts
#
# Typical usage:
#   spec = MaterialFileSpec(
#       name="PS",
#       path="PS.txt",
#       wavelength_unit="nm",
#       skiprows=2,
#       names=["Wavelength", "RefractiveIndex", "k"],
#       k_column="k",
#   )
#   ps = load_tabulated_material(spec)
#   nk = ps(wavelengths_m)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

WavelengthUnit = Literal["m", "um", "nm"]
ExtrapolationMode = Literal["error", "hold", "extrapolate", "nan"]


__all__ = [
    "Dispersion",
    "ConstantDispersion",
    "TabulatedDispersion",
    "MaterialFileSpec",
    "load_tabulated_material",
    "load_materials",
    "load_and_interpolate_legacy",
]


# ============================================================
# Helper functions
# ============================================================

def _as_1d_float_array(values: float | Iterable[float] | np.ndarray) -> FloatArray:
    """Convert scalar or iterable input to a 1D float64 NumPy array."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    elif arr.ndim != 1:
        raise ValueError("Input must be a scalar or a 1D array.")
    return arr.astype(np.float64, copy=False)


def _convert_wavelengths(values: FloatArray, from_unit: WavelengthUnit, to_unit: WavelengthUnit) -> FloatArray:
    """Convert wavelengths between m, um, and nm."""
    to_m = {
        "m": 1.0,
        "um": 1e-6,
        "nm": 1e-9,
    }
    if from_unit not in to_m:
        raise ValueError(f"Unsupported source unit: {from_unit}")
    if to_unit not in to_m:
        raise ValueError(f"Unsupported target unit: {to_unit}")

    values_m = values * to_m[from_unit]
    return values_m / to_m[to_unit]


def _interp_with_extrapolation(
    x_query: FloatArray,
    x_data: FloatArray,
    y_data: FloatArray,
    mode: ExtrapolationMode = "error",
) -> FloatArray:
    """
    Linear interpolation with configurable extrapolation.

    Parameters
    ----------
    x_query : array
        Query points.
    x_data : array
        Known x points (must be strictly increasing).
    y_data : array
        Known y values.
    mode : {"error", "hold", "extrapolate", "nan"}
        Extrapolation behavior outside x_data range.

    Returns
    -------
    array
        Interpolated values.
    """
    if x_data.ndim != 1 or y_data.ndim != 1:
        raise ValueError("x_data and y_data must be 1D.")
    if x_data.shape != y_data.shape:
        raise ValueError("x_data and y_data must have the same shape.")
    if np.any(np.diff(x_data) <= 0):
        raise ValueError("x_data must be strictly increasing.")

    x_min = x_data[0]
    x_max = x_data[-1]

    inside = (x_query >= x_min) & (x_query <= x_max)
    out = np.empty_like(x_query, dtype=float)

    # Standard interpolation inside the data range
    out[inside] = np.interp(x_query[inside], x_data, y_data)

    if np.all(inside):
        return out

    outside = ~inside

    if mode == "error":
        bad = x_query[outside]
        raise ValueError(
            f"Requested wavelength(s) outside tabulated range "
            f"[{x_min:.6g}, {x_max:.6g}]. "
            f"First offending value: {bad[0]:.6g}"
        )

    elif mode == "hold":
        out[x_query < x_min] = y_data[0]
        out[x_query > x_max] = y_data[-1]
        return out

    elif mode == "nan":
        out[outside] = np.nan
        return out

    elif mode == "extrapolate":
        # Linear extrapolation using edge slopes
        left_mask = x_query < x_min
        right_mask = x_query > x_max

        if x_data.size < 2:
            raise ValueError("Need at least two points for extrapolation.")

        left_slope = (y_data[1] - y_data[0]) / (x_data[1] - x_data[0])
        right_slope = (y_data[-1] - y_data[-2]) / (x_data[-1] - x_data[-2])

        out[left_mask] = y_data[0] + left_slope * (x_query[left_mask] - x_data[0])
        out[right_mask] = y_data[-1] + right_slope * (x_query[right_mask] - x_data[-1])
        return out

    else:
        raise ValueError(f"Unsupported extrapolation mode: {mode}")


# ============================================================
# Core dispersion classes
# ============================================================

@dataclass(frozen=True)
class Dispersion:
    """
    Generic material dispersion object.

    Parameters
    ----------
    name :
        Human-readable material name.
    nk :
        Callable that takes wavelengths in meters and returns complex n(λ) = n + i k.
    source :
        Optional source description (e.g. file path, citation, notes).
    """
    name: str
    nk: Callable[[FloatArray], ComplexArray]
    source: str | None = field(default=None, kw_only=True)

    def __call__(self, wavelengths_m: float | Iterable[float] | np.ndarray) -> ComplexArray:
        wl = _as_1d_float_array(wavelengths_m)
        nk = np.asarray(self.nk(wl), dtype=np.complex128)
        if nk.shape != wl.shape:
            raise ValueError(
                f"Dispersion '{self.name}' returned shape {nk.shape}, expected {wl.shape}."
            )
        return nk

    def n(self, wavelengths_m: float | Iterable[float] | np.ndarray) -> FloatArray:
        """Return the real refractive index n(λ)."""
        return np.real(self(wavelengths_m)).astype(np.float64)

    def k(self, wavelengths_m: float | Iterable[float] | np.ndarray) -> FloatArray:
        """Return the extinction coefficient k(λ)."""
        return np.imag(self(wavelengths_m)).astype(np.float64)


@dataclass(frozen=True)
class ConstantDispersion(Dispersion):
    """Wavelength-independent material."""

    @classmethod
    def from_nk(
        cls,
        name: str,
        n: float,
        k: float = 0.0,
        source: str | None = None,
    ) -> "ConstantDispersion":
        nk_const = complex(n, k)

        def _nk(wavelengths_m: FloatArray) -> ComplexArray:
            return np.full(wavelengths_m.shape, nk_const, dtype=np.complex128)

        return cls(name=name, nk=_nk, source=source)


@dataclass(frozen=True)
class TabulatedDispersion(Dispersion):
    """
    Dispersion defined by tabulated wavelength-dependent data.

    Attributes
    ----------
    wavelengths_m :
        Wavelength grid in meters.
    nk_values :
        Complex refractive index values on that grid.
    extrapolation :
        Extrapolation mode used for interpolation.
    """
    wavelengths_m: FloatArray
    nk_values: ComplexArray
    extrapolation: ExtrapolationMode = "error"
    nk: Callable[[FloatArray], ComplexArray] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        wl = np.asarray(self.wavelengths_m, dtype=float)
        nk = np.asarray(self.nk_values, dtype=np.complex128)

        if wl.ndim != 1:
            raise ValueError("wavelengths_m must be 1D.")
        if nk.ndim != 1:
            raise ValueError("nk_values must be 1D.")
        if wl.shape != nk.shape:
            raise ValueError("wavelengths_m and nk_values must have the same shape.")
        if wl.size < 2:
            raise ValueError("At least two wavelength points are required.")
        if np.any(wl <= 0):
            raise ValueError("Wavelengths must be positive.")
        if np.any(np.diff(wl) <= 0):
            raise ValueError("Wavelengths must be strictly increasing.")

        def _nk_interp(query_wavelengths_m: FloatArray) -> ComplexArray:
            n_vals = _interp_with_extrapolation(
                query_wavelengths_m,
                wl,
                np.real(nk).astype(float),
                mode=self.extrapolation,
            )
            k_vals = _interp_with_extrapolation(
                query_wavelengths_m,
                wl,
                np.imag(nk).astype(float),
                mode=self.extrapolation,
            )
            return n_vals + 1j * k_vals

        object.__setattr__(self, "nk", _nk_interp)


# ============================================================
# File-based material specification
# ============================================================

@dataclass(frozen=True)
class MaterialFileSpec:
    """
    Specification for loading one material file.

    Parameters
    ----------
    name :
        Material name, e.g. "TiO2".
    path :
        Path to the data file.
    wavelength_unit :
        Unit used in the file for the wavelength column ("nm", "um", or "m").
    skiprows :
        Number of initial lines to skip before reading numeric data.
        This is safer and clearer than the old `header_lines` usage.
    names :
        Column names to assign manually.
        Examples:
            ["Wavelength", "RefractiveIndex"]
            ["Wavelength", "RefractiveIndex", "k"]
    wavelength_column :
        Name of the wavelength column after loading.
    n_column :
        Name of the real refractive index column after loading.
    k_column :
        Name of the extinction column after loading, or None if absent.
    comment :
        Comment marker.
    extrapolation :
        How to behave outside the tabulated wavelength range.
    """
    name: str
    path: str | Path
    wavelength_unit: WavelengthUnit = "nm"
    skiprows: int = 0
    names: list[str] | tuple[str, ...] = ("Wavelength", "RefractiveIndex")
    wavelength_column: str = "Wavelength"
    n_column: str = "RefractiveIndex"
    k_column: str | None = None
    comment: str = "#"
    extrapolation: ExtrapolationMode = "extrapolate"


# ============================================================
# Public loaders
# ============================================================

def load_tabulated_material(spec: MaterialFileSpec) -> TabulatedDispersion:
    """
    Load a material from file according to a MaterialFileSpec.
    """
    path = Path(spec.path)
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(
        path,
        sep=r"\s+",
        comment=spec.comment,
        header=None,
        skiprows=spec.skiprows,
        names=list(spec.names),
        engine="python",
    )

    if df.empty:
        raise ValueError(f"No data read from file: {path}")

    required = [spec.wavelength_column, spec.n_column]
    for col in required:
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' not found in {path}. "
                f"Available columns: {list(df.columns)}"
            )

    if spec.k_column is not None and spec.k_column not in df.columns:
        raise ValueError(
            f"Column '{spec.k_column}' not found in {path}. "
            f"Available columns: {list(df.columns)}"
        )

    wavelengths = pd.to_numeric(df[spec.wavelength_column], errors="coerce").to_numpy(dtype=float)
    n_vals = pd.to_numeric(df[spec.n_column], errors="coerce").to_numpy(dtype=float)

    if spec.k_column is None:
        k_vals = np.zeros_like(n_vals, dtype=float)
    else:
        k_vals = pd.to_numeric(df[spec.k_column], errors="coerce").to_numpy(dtype=float)

    # Remove rows with NaN
    valid = np.isfinite(wavelengths) & np.isfinite(n_vals) & np.isfinite(k_vals)
    wavelengths = wavelengths[valid]
    n_vals = n_vals[valid]
    k_vals = k_vals[valid]

    if wavelengths.size < 2:
        raise ValueError(f"Not enough valid data rows in {path}")

    wavelengths_m = _convert_wavelengths(wavelengths, spec.wavelength_unit, "m")
    nk_values = n_vals + 1j * k_vals

    # Sort by wavelength
    order = np.argsort(wavelengths_m)
    wavelengths_m = wavelengths_m[order]
    nk_values = nk_values[order]

    # Remove duplicate wavelengths by keeping the first occurrence
    unique_mask = np.ones_like(wavelengths_m, dtype=bool)
    unique_mask[1:] = np.diff(wavelengths_m) > 0
    wavelengths_m = wavelengths_m[unique_mask]
    nk_values = nk_values[unique_mask]

    return TabulatedDispersion(
        name=spec.name,
        source=str(path),
        wavelengths_m=wavelengths_m,
        nk_values=nk_values,
        extrapolation=spec.extrapolation,
    )


def load_materials(specs: dict[str, MaterialFileSpec]) -> dict[str, TabulatedDispersion]:
    """
    Load several materials from a dictionary of MaterialFileSpec objects.

    Parameters
    ----------
    specs :
        Dictionary like {"TiO2": MaterialFileSpec(...), ...}

    Returns
    -------
    dict
        Dictionary of loaded Dispersion objects.
    """
    return {name: load_tabulated_material(spec) for name, spec in specs.items()}


# ============================================================
# Legacy-compatible helper
# ============================================================

def load_and_interpolate_legacy(
    file_path: str | Path,
    wavelength_range_nm: float | Iterable[float] | np.ndarray,
    wavelengths_in_nm: bool = True,
    header_lines: int = 0,
    names: list[str] | tuple[str, ...] = ("Wavelength", "RefractiveIndex"),
) -> ComplexArray:
    """
    Legacy-style helper compatible with your previous workflow.

    Parameters
    ----------
    file_path :
        Path to the optical constants file.
    wavelength_range_nm :
        Target wavelength grid in nm.
    wavelengths_in_nm :
        If False, file wavelengths are assumed to be in um and converted to nm.
    header_lines :
        Number of rows to skip at the top of the file.
    names :
        Assigned column names.
        If 'k' is included in names, complex n + i k is returned.

    Returns
    -------
    np.ndarray
        Interpolated real or complex refractive index evaluated on wavelength_range_nm.
    """
    names = list(names)

    k_column = "k" if "k" in names else None
    wavelength_unit: WavelengthUnit = "nm" if wavelengths_in_nm else "um"

    spec = MaterialFileSpec(
        name=Path(file_path).stem,
        path=file_path,
        wavelength_unit=wavelength_unit,
        skiprows=header_lines,
        names=names,
        wavelength_column="Wavelength",
        n_column="RefractiveIndex",
        k_column=k_column,
        extrapolation="extrapolate",
    )

    material = load_tabulated_material(spec)

    wavelength_range_nm = _as_1d_float_array(wavelength_range_nm)
    wavelength_range_m = _convert_wavelengths(wavelength_range_nm, "nm", "m")
    return material(wavelength_range_m)