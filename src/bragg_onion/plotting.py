from __future__ import annotations

# plotting.py
#
# Plot scattering results from solver.py and integration.py.
#
# Key features:
# - plot differential scattering vs wavelength for selected angles
# - plot integrated scattering vs wavelength
# - plot polar scattering diagram at a selected wavelength
# - choose between linear scale and logarithmic dB scale (10*log10)
#
# Supported quantities:
# From ScatteringResult:
# - "dcs_m2_sr"              : differential scattering cross-section [m^2/sr]
# - "dcs_geom_norm_sr_inv"   : differential scattering normalized by geometric area [1/sr]
# - "phase_function_sr_inv"  : differential scattering normalized by total scattering [1/sr]
# - "qsca", "qext", "qabs"   : efficiencies [-]
# - "csca_m2", "cext_m2", "cabs_m2" : cross-sections [m^2]
#
# From IntegratedScatteringResult:
# - "c_collected_m2"         : collected scattering cross-section [m^2]
# - "fraction_collected"     : collected fraction [-]
# - "c_collected_geom_norm"  : collected scattering normalized by geometric area [-]
#
# Conventions:
# - wavelengths are plotted in nm
# - theta is given in degrees for user-facing inputs, internally in radians
# - dB scale uses 10*log10(value), appropriate for power-like quantities

from typing import Iterable, Literal, TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .solver import ScatteringResult
    from .integration import IntegratedScatteringResult


FloatArray = NDArray[np.float64]
ScaleMode = Literal["linear", "db"]


__all__ = [
    "plot_differential_scattering_vs_wavelength",
    "plot_integrated_scattering",
    "plot_scattering_polar",
    "plot_efficiency_vs_wavelength",
]


# ============================================================
# Helpers
# ============================================================

def _as_1d_float_array(values: float | Iterable[float] | np.ndarray, name: str) -> FloatArray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    elif arr.ndim != 1:
        raise ValueError(f"{name} must be a scalar or a 1D array.")
    return arr.astype(np.float64, copy=False)


def _nearest_index(values: FloatArray, target: float) -> int:
    return int(np.argmin(np.abs(values - target)))


def _apply_scale(values: np.ndarray, scale: ScaleMode, floor: float) -> np.ndarray:
    """
    Apply linear or dB scaling.

    dB scaling uses:
        10 * log10(value)

    A positive floor is applied before the logarithm.
    """
    arr = np.asarray(values, dtype=float)
    if scale == "linear":
        return arr
    if scale == "db":
        return 10.0 * np.log10(np.maximum(arr, floor))
    raise ValueError(f"Unknown scale: {scale!r}")


def _quantity_label_scattering(quantity: str, scale: ScaleMode) -> str:
    base = {
        "dcs_m2_sr": r"$d\sigma/d\Omega$ [m$^2$/sr]",
        "dcs_geom_norm_sr_inv": r"$(d\sigma/d\Omega)/(\pi R^2)$ [1/sr]",
        "phase_function_sr_inv": r"$(d\sigma/d\Omega)/C_{\mathrm{sca}}$ [1/sr]",
        "qsca": r"$Q_{\mathrm{sca}}$ [-]",
        "qext": r"$Q_{\mathrm{ext}}$ [-]",
        "qabs": r"$Q_{\mathrm{abs}}$ [-]",
        "csca_m2": r"$C_{\mathrm{sca}}$ [m$^2$]",
        "cext_m2": r"$C_{\mathrm{ext}}$ [m$^2$]",
        "cabs_m2": r"$C_{\mathrm{abs}}$ [m$^2$]",
    }.get(quantity, quantity)

    if scale == "db":
        return f"{base} [dB]"
    return base


def _quantity_label_integrated(quantity: str, scale: ScaleMode) -> str:
    base = {
        "c_collected_m2": r"$C_{\mathrm{collected}}$ [m$^2$]",
        "fraction_collected": r"$C_{\mathrm{collected}}/C_{\mathrm{sca}}$ [-]",
        "c_collected_geom_norm": r"$C_{\mathrm{collected}}/(\pi R^2)$ [-]",
    }.get(quantity, quantity)

    if scale == "db":
        return f"{base} [dB]"
    return base


def _get_scattering_quantity(result: ScatteringResult, quantity: str) -> np.ndarray:
    if not hasattr(result, quantity):
        raise ValueError(f"ScatteringResult has no quantity '{quantity}'.")
    return np.asarray(getattr(result, quantity))


def _get_integrated_quantity(result: IntegratedScatteringResult, quantity: str) -> np.ndarray:
    if not hasattr(result, quantity):
        raise ValueError(f"IntegratedScatteringResult has no quantity '{quantity}'.")
    return np.asarray(getattr(result, quantity))


# ============================================================
# Plot: differential scattering vs wavelength
# ============================================================

def plot_differential_scattering_vs_wavelength(
    result: ScatteringResult,
    angles_deg: Iterable[float],
    *,
    quantity: Literal["dcs_m2_sr", "dcs_geom_norm_sr_inv", "phase_function_sr_inv"] = "dcs_m2_sr",
    scale: ScaleMode = "linear",
    floor: float = 1e-30,
    cmap: str = "viridis",
    ax: plt.Axes | None = None,
):
    """
    Plot angle-resolved differential scattering versus wavelength.

    Parameters
    ----------
    result :
        Output from solver.run_scattnlay_spectrum(...)
    angles_deg :
        Angles to plot, in degrees
    quantity :
        One of:
        - "dcs_m2_sr"
        - "dcs_geom_norm_sr_inv"
        - "phase_function_sr_inv"
    scale :
        "linear" or "db"
    floor :
        Positive floor used before log10 when scale="db"
    cmap :
        Matplotlib colormap name
    ax :
        Optional matplotlib axes

    Returns
    -------
    ax
    """
    allowed = {"dcs_m2_sr", "dcs_geom_norm_sr_inv", "phase_function_sr_inv"}
    if quantity not in allowed:
        raise ValueError(f"quantity must be one of {allowed}.")

    wavelengths_nm = result.wavelengths_m * 1e9
    theta_deg_grid = np.rad2deg(result.theta_rad)
    data = _get_scattering_quantity(result, quantity)

    angles_deg = list(float(a) for a in angles_deg)
    if len(angles_deg) == 0:
        raise ValueError("angles_deg must contain at least one angle.")

    if ax is None:
        fig, ax = plt.subplots()

    cmap_obj = plt.get_cmap(cmap)
    colors = cmap_obj(np.linspace(0, 1, len(angles_deg)))

    for color, angle_deg in zip(colors, angles_deg):
        idx = _nearest_index(theta_deg_grid, angle_deg)
        y = _apply_scale(data[:, idx], scale=scale, floor=floor)
        ax.plot(
            wavelengths_nm,
            y,
            color=color,
            label=fr"{theta_deg_grid[idx]:.1f}°",
        )

    ax.set_xlabel("Wavelength [nm]")
    ax.set_ylabel(_quantity_label_scattering(quantity, scale))
    ax.set_title("Differential scattering vs wavelength")
    ax.legend(title="Scattering angle")
    return ax


# ============================================================
# Plot: integrated scattering vs wavelength
# ============================================================

def plot_integrated_scattering(
    result: IntegratedScatteringResult,
    *,
    quantity: Literal["c_collected_m2", "fraction_collected", "c_collected_geom_norm"] = "c_collected_m2",
    scale: ScaleMode = "linear",
    floor: float = 1e-30,
    ax: plt.Axes | None = None,
):
    """
    Plot integrated scattering versus wavelength.

    Parameters
    ----------
    result :
        Output from integration.integrate_theta_range(...) or integrate_collection_na(...)
    quantity :
        One of:
        - "c_collected_m2"
        - "fraction_collected"
        - "c_collected_geom_norm"
    scale :
        "linear" or "db"
    floor :
        Positive floor used before log10 when scale="db"
    ax :
        Optional matplotlib axes

    Returns
    -------
    ax
    """
    allowed = {"c_collected_m2", "fraction_collected", "c_collected_geom_norm"}
    if quantity not in allowed:
        raise ValueError(f"quantity must be one of {allowed}.")

    wavelengths_nm = result.wavelengths_m * 1e9
    y_raw = _get_integrated_quantity(result, quantity)
    y = _apply_scale(y_raw, scale=scale, floor=floor)

    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(wavelengths_nm, y)
    ax.set_xlabel("Wavelength [nm]")
    ax.set_ylabel(_quantity_label_integrated(quantity, scale))
    ax.set_title(f"Integrated scattering vs wavelength ({result.direction})")
    return ax


# ============================================================
# Plot: scattering polar diagram
# ============================================================

def plot_scattering_polar(
    result: ScatteringResult,
    wavelength_m: float,
    *,
    quantity: Literal["dcs_m2_sr", "dcs_geom_norm_sr_inv", "phase_function_sr_inv"] = "dcs_m2_sr",
    scale: ScaleMode = "linear",
    floor: float = 1e-30,
    mirror: bool = True,
    ax: plt.Axes | None = None,
):
    """
    Plot angular scattering at one wavelength as a polar diagram.

    Parameters
    ----------
    result :
        Output from solver.run_scattnlay_spectrum(...)
    wavelength_m :
        Wavelength to plot [m]
    quantity :
        One of:
        - "dcs_m2_sr"
        - "dcs_geom_norm_sr_inv"
        - "phase_function_sr_inv"
    scale :
        "linear" or "db"
    floor :
        Positive floor used before log10 when scale="db"
    mirror :
        If True, mirror the 0..π curve into π..2π for a full polar display
    ax :
        Optional polar axes

    Returns
    -------
    ax
    """
    allowed = {"dcs_m2_sr", "dcs_geom_norm_sr_inv", "phase_function_sr_inv"}
    if quantity not in allowed:
        raise ValueError(f"quantity must be one of {allowed}.")

    idx = _nearest_index(result.wavelengths_m, wavelength_m)
    theta = np.asarray(result.theta_rad, dtype=float)
    values = _get_scattering_quantity(result, quantity)[idx]
    values_plot = _apply_scale(values, scale=scale, floor=floor)

    if mirror:
        theta_plot = np.concatenate([theta, 2.0 * np.pi - theta[::-1]])
        values_plot = np.concatenate([values_plot, values_plot[::-1]])
    else:
        theta_plot = theta

    if ax is None:
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"})

    ax.plot(theta_plot, values_plot)
    ax.set_title(
        f"Polar scattering at {result.wavelengths_m[idx]*1e9:.1f} nm"
    )
    return ax


# ============================================================
# Plot: scalar efficiencies / cross-sections vs wavelength
# ============================================================

def plot_efficiency_vs_wavelength(
    result: ScatteringResult,
    *,
    quantity: Literal["qsca", "qext", "qabs", "csca_m2", "cext_m2", "cabs_m2"] = "qsca",
    scale: ScaleMode = "linear",
    floor: float = 1e-30,
    ax: plt.Axes | None = None,
):
    """
    Plot scalar efficiency or cross-section versus wavelength.

    Parameters
    ----------
    result :
        Output from solver.run_scattnlay_spectrum(...)
    quantity :
        One of:
        - "qsca", "qext", "qabs"
        - "csca_m2", "cext_m2", "cabs_m2"
    scale :
        "linear" or "db"
    floor :
        Positive floor used before log10 when scale="db"
    ax :
        Optional matplotlib axes

    Returns
    -------
    ax
    """
    allowed = {"qsca", "qext", "qabs", "csca_m2", "cext_m2", "cabs_m2"}
    if quantity not in allowed:
        raise ValueError(f"quantity must be one of {allowed}.")

    wavelengths_nm = result.wavelengths_m * 1e9
    y_raw = _get_scattering_quantity(result, quantity)
    y = _apply_scale(y_raw, scale=scale, floor=floor)

    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(wavelengths_nm, y)
    ax.set_xlabel("Wavelength [nm]")
    ax.set_ylabel(_quantity_label_scattering(quantity, scale))
    ax.set_title(f"{quantity} vs wavelength")
    return ax