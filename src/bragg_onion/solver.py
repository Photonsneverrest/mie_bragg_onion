# scattnlay wrapper
from __future__ import annotations

# solver.py
#
# Run scattnlay for wavelength-dependent multilayer Bragg onion spheres.
#
# Key features:
# - converts a resolved layer stack into scattnlay inputs x and m
# - supports one wavelength or a wavelength grid
# - computes efficiencies (Qext, Qsca, Qabs, Qbk, Qpr, g, Albedo)
# - computes differential scattering cross-section:
#       dσ/dΩ = (|S1|^2 + |S2|^2) / (2 k^2)      [m^2/sr]
# - computes geometric-area-normalized differential scattering:
#       (dσ/dΩ) / (π R^2)                        [1/sr]
# - computes phase function normalized by total scattering cross-section:
#       (dσ/dΩ) / Csca                           [1/sr]
#
# Conventions:
# - wavelengths are in meters
# - angles theta are in radians
# - radii are cumulative outer radii [m]
# - x has shape (n_wavelengths, n_layers) for spectrum calculations
# - m has shape (n_wavelengths, n_layers) for spectrum calculations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

try:
    from scattnlay import scattnlay  # type: ignore
except ImportError as exc:
    raise ImportError(
        "scattnlay is required for bragg_onion.solver. "
        "Install the optional dependency with: pip install 'bragg-onion[scattnlay]'"
    ) from exc

from .materials import Dispersion
from .geometry import ResolvedLayerStack


FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
IntArray = NDArray[np.int_]


__all__ = [
    "ScatteringResult",
    "evaluate_medium_index",
    "build_scattnlay_inputs",
    "build_scattnlay_inputs_single_wavelength",
    "run_scattnlay_spectrum",
]

def _is_dispersion_like(obj: object) -> bool:
    """
    Return True for dispersion-like objects.

    This intentionally uses duck typing instead of strict isinstance(...)
    checks so that the code remains robust when modules are reloaded in
    notebooks (which can otherwise break class identity).
    """
    scalar_types = (int, float, complex, np.number)
    return callable(obj) and not isinstance(obj, scalar_types)


# ============================================================
# Dataclass for results
# ============================================================

@dataclass(frozen=True)
class ScatteringResult:
    """
    Results of a wavelength-dependent scattnlay simulation.

    Attributes
    ----------
    wavelengths_m :
        Wavelength grid [m], shape (n_wavelengths,)
    theta_rad :
        Scattering angle grid [rad], shape (n_theta,)
    radii_m :
        Cumulative outer radii of the layers [m], shape (n_layers,)
    n_medium :
        Surrounding medium refractive index on the wavelength grid,
        shape (n_wavelengths,)
    k_medium_m_inv :
        Medium wavevector k = 2π n_medium / λ [1/m], shape (n_wavelengths,)
    x :
        scattnlay size-parameter array, shape (n_wavelengths, n_layers)
    m :
        Relative refractive-index array, shape (n_wavelengths, n_layers)

    terms :
        Number of Mie terms used by scattnlay, shape (n_wavelengths,)
    qext, qsca, qabs, qbk, qpr, g, albedo :
        Efficiency / asymmetry outputs from scattnlay, shape (n_wavelengths,)
    c_geo_m2 :
        Geometric cross-sectional area π R² [m²], shape (n_wavelengths,)
        (constant for a fixed geometry, but stored as array for convenience)
    cext_m2, csca_m2, cabs_m2 :
        Absolute extinction / scattering / absorption cross-sections [m²],
        shape (n_wavelengths,)

    s1, s2 :
        Scattering amplitude functions from scattnlay,
        shape (n_wavelengths, n_theta)
    dcs_m2_sr :
        Differential scattering cross-section dσ/dΩ [m²/sr],
        shape (n_wavelengths, n_theta)
    dcs_geom_norm_sr_inv :
        Geometric-area-normalized differential scattering [1/sr],
        shape (n_wavelengths, n_theta)
    phase_function_sr_inv :
        Differential scattering normalized by total scattering cross-section [1/sr],
        shape (n_wavelengths, n_theta)
    """
    wavelengths_m: FloatArray
    theta_rad: FloatArray
    radii_m: FloatArray
    n_medium: ComplexArray
    k_medium_m_inv: ComplexArray
    x: FloatArray
    m: ComplexArray

    terms: IntArray
    qext: FloatArray
    qsca: FloatArray
    qabs: FloatArray
    qbk: FloatArray
    qpr: FloatArray
    g: FloatArray
    albedo: FloatArray

    c_geo_m2: FloatArray
    cext_m2: FloatArray
    csca_m2: FloatArray
    cabs_m2: FloatArray

    s1: ComplexArray
    s2: ComplexArray
    dcs_m2_sr: FloatArray
    dcs_geom_norm_sr_inv: FloatArray
    phase_function_sr_inv: FloatArray

    @property
    def outer_radius_m(self) -> float:
        """Particle outer radius [m]."""
        return float(self.radii_m[-1])

    @property
    def diameter_m(self) -> float:
        """Particle diameter [m]."""
        return 2.0 * self.outer_radius_m

    @property
    def n_wavelengths(self) -> int:
        return int(self.wavelengths_m.size)

    @property
    def n_theta(self) -> int:
        return int(self.theta_rad.size)

    @property
    def n_layers(self) -> int:
        return int(self.radii_m.size)


# ============================================================
# Helpers
# ============================================================

def _as_1d_float_array(values: float | Iterable[float] | np.ndarray, name: str) -> FloatArray:
    """Convert a scalar or iterable into a 1D float64 array."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    elif arr.ndim != 1:
        raise ValueError(f"{name} must be a scalar or a 1D array.")
    return arr.astype(np.float64, copy=False)


def _as_1d_theta_array(theta_rad: float | Iterable[float] | np.ndarray) -> FloatArray:
    """Validate and return scattering angle array in radians."""
    theta = _as_1d_float_array(theta_rad, "theta_rad")
    if np.any(theta < 0) or np.any(theta > np.pi):
        raise ValueError("theta_rad must lie in the interval [0, π].")
    return theta


def evaluate_medium_index(
    wavelengths_m: float | Iterable[float] | np.ndarray,
    n_medium: float | complex | Dispersion,
) -> ComplexArray:
    """
    Evaluate surrounding-medium refractive index on a wavelength grid.

    Parameters
    ----------
    wavelengths_m :
        Wavelengths [m]
    n_medium :
        Can be:
        - constant real number
        - constant complex number
        - dispersion-like callable object

    Returns
    -------
    np.ndarray
        Shape (n_wavelengths,), dtype complex128
    """
    wl = _as_1d_float_array(wavelengths_m, "wavelengths_m")

    if _is_dispersion_like(n_medium):
        n_med = np.asarray(n_medium(wl), dtype=np.complex128)
        if n_med.shape != wl.shape:
            raise ValueError(
                f"Dispersion-like n_medium returned shape {n_med.shape}, expected {wl.shape}."
            )
    else:
        n_med = np.full(wl.shape, complex(n_medium), dtype=np.complex128)

    if np.any(n_med == 0):
        raise ValueError("n_medium must be non-zero at all wavelengths.")

    return n_med


def build_scattnlay_inputs(
    stack: ResolvedLayerStack,
    wavelengths_m: float | Iterable[float] | np.ndarray,
    n_medium: float | complex | Dispersion,
) -> tuple[FloatArray, ComplexArray, FloatArray, ComplexArray, FloatArray]:
    """
    Build wavelength-dependent scattnlay inputs x and m.

    Parameters
    ----------
    stack :
        Resolved layer stack from geometry.resolve_layer_stack(...)
    wavelengths_m :
        Wavelength grid [m]
    n_medium :
        Surrounding medium refractive index (constant or Dispersion)

    Returns
    -------
    x :
        Size-parameter array, shape (n_wavelengths, n_layers), real-valued
    m :
        Relative refractive-index array, shape (n_wavelengths, n_layers)
    radii_m :
        Cumulative outer radii [m], shape (n_layers,)
    n_medium_eval :
        Surrounding medium refractive index, shape (n_wavelengths,)
    k_medium_m_inv :
        Medium wavevector used for x = 2π Re(n_medium) / λ [1/m], shape (n_wavelengths,)
    """
    wl = _as_1d_float_array(wavelengths_m, "wavelengths_m")
    radii_m = np.asarray(stack.radii_m, dtype=np.float64)

    if radii_m.ndim != 1:
        raise ValueError("stack.radii_m must be a 1D array.")
    if np.any(radii_m <= 0):
        raise ValueError("All layer radii must be positive.")
    if np.any(np.diff(radii_m) <= 0):
        raise ValueError("Layer radii must be strictly increasing.")
    if np.any(wl <= 0):
        raise ValueError("All wavelengths must be positive.")

    n_med = evaluate_medium_index(wl, n_medium)

    # scattnlay expects x to be real-valued
    n_med_real = np.real(n_med).astype(np.float64)
    if np.any(n_med_real <= 0):
        raise ValueError("Real part of n_medium must be positive for x construction.")

    # Optional warning if the medium has a non-negligible imaginary part
    if np.any(np.abs(np.imag(n_med)) > 1e-12):
        print(
            "Warning: n_medium has a non-zero imaginary part. "
            "solver.py currently uses Re(n_medium) to construct x for scattnlay."
        )

    k_medium = 2.0 * np.pi * n_med_real / wl   # real-valued
    x = k_medium[:, None] * radii_m[None, :]
    m = stack.m_spectrum(wl, n_medium=n_medium)

    return (
        np.asarray(x, dtype=np.float64),
        np.asarray(m, dtype=np.complex128),
        radii_m,
        np.asarray(n_med, dtype=np.complex128),
        np.asarray(k_medium, dtype=np.float64),
    )


def build_scattnlay_inputs_single_wavelength(
    stack: ResolvedLayerStack,
    wavelength_m: float,
    n_medium: float | complex | Dispersion,
) -> tuple[FloatArray, ComplexArray, FloatArray, complex, complex]:
    """
    Build scattnlay inputs x and m for one wavelength.

    Returns
    -------
    x :
        Shape (n_layers,), dtype float64
    m :
        Shape (n_layers,), dtype complex128
    radii_m :
        Shape (n_layers,), dtype float64
    n_medium_eval :
        Complex refractive index of surrounding medium at this wavelength
    k_medium_m_inv :
        Medium wavevector 2π n_medium / λ [1/m]
    """
    wl = float(wavelength_m)
    x_all, m_all, radii_m, n_med, k_medium = build_scattnlay_inputs(
        stack=stack,
        wavelengths_m=np.array([wl], dtype=float),
        n_medium=n_medium,
    )
    return x_all[0], m_all[0], radii_m, complex(n_med[0]), complex(k_medium[0])


def _compute_cross_sections(
    s1_row: ComplexArray,
    s2_row: ComplexArray,
    k_medium: float,
    c_geo_m2: float,
    csca_m2: float,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """
    Compute differential scattering outputs for one wavelength.

    Returns
    -------
    dcs_m2_sr :
        Differential scattering cross-section [m^2/sr]
    dcs_geom_norm_sr_inv :
        Differential scattering normalized by geometric area [1/sr]
    phase_function_sr_inv :
        Differential scattering normalized by total scattering cross-section [1/sr]
    """
    k_sq = float(k_medium) ** 2
    if k_sq == 0:
        raise ValueError("Medium wavevector k must be non-zero.")

    dcs = (np.abs(s1_row) ** 2 + np.abs(s2_row) ** 2) / (2.0 * k_sq)
    dcs = np.asarray(dcs, dtype=np.float64)

    dcs_geom_norm = dcs / c_geo_m2

    if csca_m2 > 0:
        phase_fn = dcs / csca_m2
    else:
        phase_fn = np.full_like(dcs, np.nan, dtype=np.float64)

    return dcs, dcs_geom_norm, phase_fn


# ============================================================
# Main solver
# ============================================================

def run_scattnlay_spectrum(
    stack: ResolvedLayerStack,
    wavelengths_m: float | Iterable[float] | np.ndarray,
    theta_rad: float | Iterable[float] | np.ndarray,
    n_medium: float | complex | Dispersion,
) -> ScatteringResult:
    """
    Run scattnlay on a wavelength grid for a multilayer sphere.

    Parameters
    ----------
    stack :
        Resolved layer stack created with geometry.resolve_layer_stack(...)
    wavelengths_m :
        Wavelength grid [m]
    theta_rad :
        Scattering angles [rad], 0 <= theta <= π
    n_medium :
        Surrounding medium refractive index (constant or Dispersion)

    Returns
    -------
    ScatteringResult

    Notes
    -----
    This function loops over wavelengths and calls:

        scattnlay(x_i, m_i, theta_rad)

    where:
    - x_i has shape (n_layers,)
    - m_i has shape (n_layers,)
    - theta_rad has shape (n_theta,)

    Differential scattering outputs are computed as:

        dσ/dΩ = (|S1|^2 + |S2|^2) / (2 k^2)

    with k = 2π n_medium / λ.

    Two normalized angular outputs are also provided:
    - dcs_geom_norm_sr_inv = (dσ/dΩ) / (π R^2)
    - phase_function_sr_inv = (dσ/dΩ) / Csca
    """
    wl = _as_1d_float_array(wavelengths_m, "wavelengths_m")
    theta = _as_1d_theta_array(theta_rad)

    x, m, radii_m, n_med, k_medium = build_scattnlay_inputs(
        stack=stack,
        wavelengths_m=wl,
        n_medium=n_medium,
    )

    n_wl = wl.size
    n_theta = theta.size
    outer_radius_m = float(radii_m[-1])
    c_geo_scalar = np.pi * outer_radius_m ** 2

    terms = np.empty(n_wl, dtype=int)
    qext = np.empty(n_wl, dtype=float)
    qsca = np.empty(n_wl, dtype=float)
    qabs = np.empty(n_wl, dtype=float)
    qbk = np.empty(n_wl, dtype=float)
    qpr = np.empty(n_wl, dtype=float)
    g = np.empty(n_wl, dtype=float)
    albedo = np.empty(n_wl, dtype=float)

    s1 = np.empty((n_wl, n_theta), dtype=np.complex128)
    s2 = np.empty((n_wl, n_theta), dtype=np.complex128)
    dcs_m2_sr = np.empty((n_wl, n_theta), dtype=np.float64)
    dcs_geom_norm_sr_inv = np.empty((n_wl, n_theta), dtype=np.float64)
    phase_function_sr_inv = np.empty((n_wl, n_theta), dtype=np.float64)

    c_geo_m2 = np.full(n_wl, c_geo_scalar, dtype=np.float64)
    cext_m2 = np.empty(n_wl, dtype=float)
    csca_m2 = np.empty(n_wl, dtype=float)
    cabs_m2 = np.empty(n_wl, dtype=float)

    for i in range(n_wl):
        x_i = np.asarray(x[i], dtype=np.float64)
        m_i = np.asarray(m[i], dtype=np.complex128)

        (
            terms_i,
            qext_i,
            qsca_i,
            qabs_i,
            qbk_i,
            qpr_i,
            g_i,
            albedo_i,
            s1_i,
            s2_i,
        ) = scattnlay(x_i, m_i, theta)

        s1_i = np.asarray(s1_i, dtype=np.complex128).reshape(-1)
        s2_i = np.asarray(s2_i, dtype=np.complex128).reshape(-1)

        if s1_i.size != n_theta or s2_i.size != n_theta:
            raise ValueError(
                "Unexpected S1/S2 shape returned by scattnlay. "
                f"Expected {n_theta} values, got {s1_i.size} and {s2_i.size}."
            )

        terms[i] = int(np.asarray(terms_i).reshape(()))
        qext[i] = float(np.asarray(qext_i).reshape(()))
        qsca[i] = float(np.asarray(qsca_i).reshape(()))
        qabs[i] = float(np.asarray(qabs_i).reshape(()))
        qbk[i] = float(np.asarray(qbk_i).reshape(()))
        qpr[i] = float(np.asarray(qpr_i).reshape(()))
        g[i] = float(np.asarray(g_i).reshape(()))
        albedo[i] = float(np.asarray(albedo_i).reshape(()))

        cext_m2[i] = qext[i] * c_geo_scalar
        csca_m2[i] = qsca[i] * c_geo_scalar
        cabs_m2[i] = qabs[i] * c_geo_scalar

        s1[i, :] = s1_i
        s2[i, :] = s2_i

        dcs_i, dcs_geom_i, phase_i = _compute_cross_sections(
            s1_row=s1_i,
            s2_row=s2_i,
            k_medium=k_medium[i],
            c_geo_m2=c_geo_scalar,
            csca_m2=csca_m2[i],
        )

        dcs_m2_sr[i, :] = dcs_i
        dcs_geom_norm_sr_inv[i, :] = dcs_geom_i
        phase_function_sr_inv[i, :] = phase_i

    return ScatteringResult(
        wavelengths_m=wl,
        theta_rad=theta,
        radii_m=radii_m,
        n_medium=np.asarray(n_med, dtype=np.complex128),
        k_medium_m_inv=np.asarray(k_medium, dtype=np.complex128),
        x=np.asarray(x, dtype=np.float64),
        m=np.asarray(m, dtype=np.complex128),
        terms=terms,
        qext=qext,
        qsca=qsca,
        qabs=qabs,
        qbk=qbk,
        qpr=qpr,
        g=g,
        albedo=albedo,
        c_geo_m2=c_geo_m2,
        cext_m2=cext_m2,
        csca_m2=csca_m2,
        cabs_m2=cabs_m2,
        s1=s1,
        s2=s2,
        dcs_m2_sr=dcs_m2_sr,
        dcs_geom_norm_sr_inv=dcs_geom_norm_sr_inv,
        phase_function_sr_inv=phase_function_sr_inv,
    )