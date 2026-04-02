from __future__ import annotations

# integration.py
#
# Integrate angle-resolved scattering over solid angle.
#
# Key features:
# - convert collection NA into a maximum collection angle
# - integrate differential scattering over a cone or a theta range
# - support forward and backward collection
# - return collected scattering cross-section [m^2]
# - return collected fraction relative to total scattering [-]
# - return geometric-area-normalized collected scattering [-]
#
# Conventions:
# - theta is the polar scattering angle in radians
# - theta = 0   : forward direction
# - theta = π   : backward direction
# - assumes azimuthal symmetry around the incident axis
# - integration uses:
#       dΩ = 2π sin(theta) dtheta
#
# Note:
# If you later turn this into a package, you may want to replace:
#     from solver import ScatteringResult
# with:
#     from .solver import ScatteringResult

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
from numpy.typing import NDArray

from solver import ScatteringResult


FloatArray = NDArray[np.float64]
CollectionDirection = Literal["forward", "backward"]


__all__ = [
    "IntegratedScatteringResult",
    "na_to_theta_max",
    "integrate_theta_range",
    "integrate_collection_na",
]


# ============================================================
# Dataclass
# ============================================================

@dataclass(frozen=True)
class IntegratedScatteringResult:
    """
    Integrated scattering results over a selected solid-angle region.

    Attributes
    ----------
    wavelengths_m :
        Wavelength grid [m], shape (n_wavelengths,)
    theta_min_rad, theta_max_rad :
        Polar angular limits of the integration region [rad]
    direction :
        "forward", "backward", or "custom"
    collection_na :
        Numerical aperture used for defining the region, or None
    c_collected_m2 :
        Collected scattering cross-section [m^2], shape (n_wavelengths,)
    fraction_collected :
        Collected fraction relative to total scattering cross-section [-],
        shape (n_wavelengths,)
    c_collected_geom_norm :
        Collected scattering normalized by geometric cross-sectional area [-],
        shape (n_wavelengths,)
    solid_angle_sr :
        Solid angle of the integrated region [sr], shape (n_wavelengths,)
    """
    wavelengths_m: FloatArray
    theta_min_rad: FloatArray
    theta_max_rad: FloatArray
    direction: str
    collection_na: float | None

    c_collected_m2: FloatArray
    fraction_collected: FloatArray
    c_collected_geom_norm: FloatArray
    solid_angle_sr: FloatArray


# ============================================================
# Helpers
# ============================================================

def _as_1d_float_array(values: float | Iterable[float] | np.ndarray, name: str) -> FloatArray:
    """Convert scalar or iterable to a 1D float64 NumPy array."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    elif arr.ndim != 1:
        raise ValueError(f"{name} must be a scalar or a 1D array.")
    return arr.astype(np.float64, copy=False)


def _solid_angle_of_cone(theta_max_rad: FloatArray) -> FloatArray:
    """
    Solid angle of a cone with half-angle theta_max.

    Ω = 2π (1 - cos(theta_max))
    """
    return 2.0 * np.pi * (1.0 - np.cos(theta_max_rad))


def _solid_angle_of_theta_band(theta_min_rad: FloatArray, theta_max_rad: FloatArray) -> FloatArray:
    """
    Solid angle of a polar band from theta_min to theta_max.

    Ω = 2π (cos(theta_min) - cos(theta_max))
    """
    return 2.0 * np.pi * (np.cos(theta_min_rad) - np.cos(theta_max_rad))


def _validate_theta_range(theta_min_rad: FloatArray, theta_max_rad: FloatArray) -> None:
    if theta_min_rad.shape != theta_max_rad.shape:
        raise ValueError("theta_min_rad and theta_max_rad must have the same shape.")
    if np.any(theta_min_rad < 0) or np.any(theta_max_rad > np.pi):
        raise ValueError("theta limits must satisfy 0 <= theta <= π.")
    if np.any(theta_max_rad < theta_min_rad):
        raise ValueError("theta_max_rad must be >= theta_min_rad.")


def na_to_theta_max(
    collection_na: float,
    n_medium: float | complex | FloatArray | NDArray[np.complex128],
) -> FloatArray:
    """
    Convert numerical aperture to collection half-angle in the surrounding medium.

    Uses:
        theta_max = arcsin(NA / Re(n_medium))

    Parameters
    ----------
    collection_na :
        Numerical aperture
    n_medium :
        Real/complex refractive index or wavelength-dependent array

    Returns
    -------
    np.ndarray
        Theta max in radians
    """
    na = float(collection_na)
    if na < 0:
        raise ValueError("collection_na must be non-negative.")

    n_med = np.asarray(n_medium)
    n_med_real = np.real(n_med).astype(float)

    if np.any(n_med_real <= 0):
        raise ValueError("Real part of n_medium must be positive.")

    ratio = na / n_med_real
    if np.any(ratio > 1.0):
        raise ValueError(
            "collection_na exceeds Re(n_medium) for at least one wavelength. "
            "NA must satisfy NA <= Re(n_medium)."
        )

    return np.arcsin(ratio)


def _integrate_dcs_over_theta_mask(
    theta_rad: FloatArray,
    dcs_row_m2_sr: FloatArray,
    mask: NDArray[np.bool_],
) -> float:
    """
    Integrate one wavelength row of dσ/dΩ over the selected theta region.

    Uses:
        C_collected = ∫ (dσ/dΩ) dΩ
                    = ∫ (dσ/dΩ) 2π sin(theta) dtheta
    """
    theta_sel = theta_rad[mask]
    dcs_sel = dcs_row_m2_sr[mask]

    if theta_sel.size < 2:
        return 0.0

    integrand = dcs_sel * 2.0 * np.pi * np.sin(theta_sel)
    return float(np.trapz(integrand, theta_sel))


# ============================================================
# Public integration functions
# ============================================================

def integrate_theta_range(
    result: ScatteringResult,
    theta_min_rad: float | Iterable[float] | np.ndarray,
    theta_max_rad: float | Iterable[float] | np.ndarray,
    *,
    direction: str = "custom",
    collection_na: float | None = None,
) -> IntegratedScatteringResult:
    """
    Integrate scattering over an explicit theta range.

    Parameters
    ----------
    result :
        Output from solver.run_scattnlay_spectrum(...)
    theta_min_rad, theta_max_rad :
        Lower and upper polar angle limits [rad]
        Can be scalars or arrays of shape (n_wavelengths,)
    direction :
        Label stored in the result metadata
    collection_na :
        Optional NA value stored in the result metadata

    Returns
    -------
    IntegratedScatteringResult
    """
    wl = np.asarray(result.wavelengths_m, dtype=float)
    theta = np.asarray(result.theta_rad, dtype=float)
    dcs = np.asarray(result.dcs_m2_sr, dtype=float)

    theta_min = _as_1d_float_array(theta_min_rad, "theta_min_rad")
    theta_max = _as_1d_float_array(theta_max_rad, "theta_max_rad")

    if theta_min.size == 1:
        theta_min = np.full(wl.shape, float(theta_min[0]), dtype=float)
    if theta_max.size == 1:
        theta_max = np.full(wl.shape, float(theta_max[0]), dtype=float)

    if theta_min.shape != wl.shape or theta_max.shape != wl.shape:
        raise ValueError(
            "theta_min_rad and theta_max_rad must be scalars or arrays with shape (n_wavelengths,)."
        )

    _validate_theta_range(theta_min, theta_max)

    c_collected = np.empty_like(wl, dtype=float)

    for i in range(wl.size):
        mask = (theta >= theta_min[i]) & (theta <= theta_max[i])
        c_collected[i] = _integrate_dcs_over_theta_mask(theta, dcs[i], mask)

    with np.errstate(divide="ignore", invalid="ignore"):
        fraction = np.where(result.csca_m2 > 0, c_collected / result.csca_m2, np.nan)
        geom_norm = np.where(result.c_geo_m2 > 0, c_collected / result.c_geo_m2, np.nan)

    solid_angle = _solid_angle_of_theta_band(theta_min, theta_max)

    return IntegratedScatteringResult(
        wavelengths_m=wl,
        theta_min_rad=theta_min,
        theta_max_rad=theta_max,
        direction=direction,
        collection_na=collection_na,
        c_collected_m2=c_collected,
        fraction_collected=fraction,
        c_collected_geom_norm=geom_norm,
        solid_angle_sr=solid_angle,
    )


def integrate_collection_na(
    result: ScatteringResult,
    collection_na: float,
    *,
    direction: CollectionDirection = "forward",
) -> IntegratedScatteringResult:
    """
    Integrate scattering collected by a numerical-aperture cone.

    Parameters
    ----------
    result :
        Output from solver.run_scattnlay_spectrum(...)
    collection_na :
        Collection numerical aperture
    direction :
        - "forward"  : cone centered at theta = 0
        - "backward" : cone centered at theta = π

    Returns
    -------
    IntegratedScatteringResult
    """
    theta_max = na_to_theta_max(collection_na, result.n_medium)

    if direction == "forward":
        theta_min = np.zeros_like(theta_max)
        theta_upper = theta_max
    elif direction == "backward":
        theta_min = np.pi - theta_max
        theta_upper = np.full_like(theta_max, np.pi)
    else:
        raise ValueError("direction must be 'forward' or 'backward'.")

    return integrate_theta_range(
        result=result,
        theta_min_rad=theta_min,
        theta_max_rad=theta_upper,
        direction=direction,
        collection_na=collection_na,
    )