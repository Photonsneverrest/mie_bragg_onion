# layer construction
from __future__ import annotations

# geometry.py
#
# Build Bragg onion geometries consisting of alternating A/B layers.
#
# Key features:
# - explicit shell thicknesses or quarter-wave thicknesses from a design wavelength
# - choose either the number of Bragg layers (including core) or a maximum total diameter
# - choose the outer Bragg material ("A" or "B")
# - adjustable core thickness factor (e.g. 2.0 or 0.5)
# - optional extinction-coefficient manipulation in:
#   * core
#   * material A
#   * material B
#   * both materials
# - optional additional outermost shell with full wavelength-dependent dispersion
# - outputs scattnlay-style cumulative radii and relative refractive index arrays m
#
# Conventions:
# - layers are stored from core -> outermost layer
# - Bragg layers use labels "A" and "B"
# - optional extra outer shell uses label "O"
# - all lengths are stored internally in meters
#
# Note:
# If you later turn this into a package, you may want to replace:
#     from materials import Dispersion
# with:
#     from .materials import Dispersion

from dataclasses import dataclass
from typing import Callable, Iterable, Literal

import numpy as np
from numpy.typing import NDArray

from materials import Dispersion


FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

BraggLayerLabel = Literal["A", "B"]
LayerLabel = Literal["A", "B", "O"]
ExtinctionMode = Literal["add", "substitute"]
ExtinctionTarget = Literal["core", "material_A", "material_B", "both_materials"]


__all__ = [
    "LayerThicknesses",
    "ExtraOuterShellSpec",
    "ExtinctionModifier",
    "BraggOnionGeometry",
    "ResolvedLayerStack",
    "alternating_layer_labels",
    "quarter_wave_thicknesses",
    "build_bragg_onion_from_thicknesses",
    "build_bragg_onion_from_peak_wavelength",
    "resolve_layer_stack",
]

def _is_dispersion_like(obj: object) -> bool:
    """
    Return True for dispersion-like objects.

    Uses duck typing instead of strict isinstance(...) checks so that
    notebook reloads do not break class detection.
    """
    scalar_types = (int, float, complex, np.number)
    return callable(obj) and not isinstance(obj, scalar_types)

# ============================================================
# Dataclasses
# ============================================================

@dataclass(frozen=True)
class LayerThicknesses:
    """
    Reference shell thicknesses for the alternating Bragg materials.
    The core thickness is derived later using core_thickness_factor.
    """
    t_a_m: float
    t_b_m: float
    design_peak_wavelength_m: float | None = None
    core_thickness_factor: float = 1.0

    def for_material(self, label: BraggLayerLabel) -> float:
        if label == "A":
            return self.t_a_m
        if label == "B":
            return self.t_b_m
        raise ValueError(f"Unknown material label: {label!r}")


@dataclass(frozen=True)
class ExtraOuterShellSpec:
    """
    Specification for one additional outermost shell.

    Parameters
    ----------
    thickness_m :
        Thickness of the extra outer shell [m]
    material :
        Full wavelength-dependent dispersion for the extra shell
    name :
        Optional label for debugging / metadata
    """
    thickness_m: float
    material: Dispersion
    name: str = "outer_shell"


@dataclass(frozen=True)
class ExtinctionModifier:
    """
    Manipulate the extinction coefficient k(λ) in selected regions.

    Parameters
    ----------
    target :
        Where to apply the modification:
        - "core"
        - "material_A"
        - "material_B"
        - "both_materials"
    mode :
        - "add"        : k_eff = k_base + k_mod
        - "substitute" : k_eff = k_mod
    profile :
        Extinction profile to apply. Supported:
        - float
        - Dispersion (imaginary part is used)
        - callable(wavelengths_m) -> array
    name :
        Optional label for metadata / debugging
    """
    target: ExtinctionTarget
    mode: ExtinctionMode
    profile: float | Dispersion | Callable[[FloatArray], FloatArray | ComplexArray]
    name: str = "k_modifier"


@dataclass(frozen=True)
class BraggOnionGeometry:
    """
    Fully defined Bragg onion geometry.

    Attributes
    ----------
    layer_labels :
        Labels from core -> outermost layer
    layer_thicknesses_m :
        Thickness of each layer [m]
    outer_radii_m :
        Cumulative outer radius of each layer [m]
    n_layers :
        Number of Bragg layers including core (A/B only)
    n_layers_total :
        Total number of layers including optional outer shell
    outer_layer :
        Chosen outer Bragg layer ("A" or "B")
    t_a_m, t_b_m :
        Reference A/B shell thicknesses [m]
    outer_radius_m :
        Total particle radius [m]
    diameter_m :
        Total particle diameter [m]
    design_peak_wavelength_m :
        Quarter-wave design wavelength [m], or None
    core_thickness_factor :
        Factor applied to the core thickness relative to its material shell thickness
    extra_outer_shell :
        Optional extra outermost shell
    """
    layer_labels: tuple[LayerLabel, ...]
    layer_thicknesses_m: FloatArray
    outer_radii_m: FloatArray
    n_layers: int
    n_layers_total: int
    outer_layer: BraggLayerLabel
    t_a_m: float
    t_b_m: float
    outer_radius_m: float
    diameter_m: float
    design_peak_wavelength_m: float | None = None
    core_thickness_factor: float = 1.0
    extra_outer_shell: ExtraOuterShellSpec | None = None

    @property
    def core_layer(self) -> BraggLayerLabel:
        label = self.layer_labels[0]
        if label not in ("A", "B"):
            raise RuntimeError("Core layer must be 'A' or 'B'.")
        return label

    @property
    def has_extra_outer_shell(self) -> bool:
        return self.extra_outer_shell is not None

    @property
    def actual_outer_layer(self) -> LayerLabel:
        return self.layer_labels[-1]

    @property
    def bragg_layer_labels(self) -> tuple[BraggLayerLabel, ...]:
        labels = self.layer_labels[:self.n_layers]
        return tuple(label for label in labels if label in ("A", "B"))

    @property
    def bragg_outer_radii_m(self) -> FloatArray:
        return self.outer_radii_m[:self.n_layers]

    @property
    def core_radius_m(self) -> float:
        return float(self.outer_radii_m[0])

    @property
    def radii_m(self) -> FloatArray:
        """
        scattnlay-style cumulative outer radii [m].
        Same as outer_radii_m.
        """
        return self.outer_radii_m


@dataclass(frozen=True)
class ResolvedLayerStack:
    """
    Geometry plus fully resolved per-layer dispersions.

    Attributes
    ----------
    geometry :
        Structural geometry
    layer_material_names :
        Human-readable material name for each layer
    layer_dispersions :
        Effective dispersion assigned to each layer
    """
    geometry: BraggOnionGeometry
    layer_material_names: tuple[str, ...]
    layer_dispersions: tuple[Dispersion, ...]

    @property
    def layer_labels(self) -> tuple[LayerLabel, ...]:
        return self.geometry.layer_labels

    @property
    def layer_thicknesses_m(self) -> FloatArray:
        return self.geometry.layer_thicknesses_m

    @property
    def outer_radii_m(self) -> FloatArray:
        return self.geometry.outer_radii_m

    @property
    def radii_m(self) -> FloatArray:
        return self.geometry.radii_m

    def refractive_indices_at_wavelength(self, wavelength_m: float) -> ComplexArray:
        """
        Complex refractive indices of all layers at one wavelength.

        Returns
        -------
        np.ndarray
            Shape (n_layers_total,), dtype complex128
        """
        wl = np.array([float(wavelength_m)], dtype=float)
        return np.asarray(
            [disp(wl)[0] for disp in self.layer_dispersions],
            dtype=np.complex128,
        )

    def refractive_indices_spectrum(self, wavelengths_m: FloatArray) -> ComplexArray:
        """
        Complex refractive indices of all layers on a wavelength grid.

        Returns
        -------
        np.ndarray
            Shape (n_wavelengths, n_layers_total)
        """
        wl = np.asarray(wavelengths_m, dtype=float)
        if wl.ndim != 1:
            raise ValueError("wavelengths_m must be a 1D array.")

        return np.column_stack([disp(wl) for disp in self.layer_dispersions]).astype(
            np.complex128
        )

    def m_at_wavelength(
        self,
        wavelength_m: float,
        n_medium: float | complex | Dispersion,
    ) -> ComplexArray:
        """
        Relative refractive index array m at one wavelength.

        m_i = n_layer_i / n_medium

        Returns
        -------
        np.ndarray
            Shape (n_layers_total,), dtype complex128
        """
        wl = np.array([float(wavelength_m)], dtype=float)

        if _is_dispersion_like(n_medium):
            n_med_arr = np.asarray(n_medium(wl), dtype=np.complex128)
            if n_med_arr.shape != wl.shape:
                raise ValueError(
                    f"Dispersion-like n_medium returned shape {n_med_arr.shape}, expected {wl.shape}."
                )
            n_med = complex(n_med_arr[0])
        else:
            n_med = complex(n_medium)

        if n_med == 0:
            raise ValueError("n_medium must be non-zero.")

        nk_layers = self.refractive_indices_at_wavelength(wavelength_m)
        return np.asarray(nk_layers / n_med, dtype=np.complex128)


    def m_spectrum(
        self,
        wavelengths_m: FloatArray,
        n_medium: float | complex | Dispersion,
    ) -> ComplexArray:
        """
        Relative refractive index array m on a wavelength grid.

        Returns
        -------
        np.ndarray
            Shape (n_wavelengths, n_layers_total)
        """
        wl = np.asarray(wavelengths_m, dtype=float)
        if wl.ndim != 1:
            raise ValueError("wavelengths_m must be 1D.")

        nk_layers = self.refractive_indices_spectrum(wl)

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

        return nk_layers / n_med[:, None]


# ============================================================
# Validation helpers
# ============================================================

def _validate_positive(name: str, value: float) -> None:
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number, got {value!r}.")


def _other_label(label: BraggLayerLabel) -> BraggLayerLabel:
    if label == "A":
        return "B"
    if label == "B":
        return "A"
    raise ValueError(f"Label must be 'A' or 'B', got {label!r}.")


def _validate_layer_choice(n_layers: int | None, diameter_m: float | None) -> None:
    if (n_layers is None and diameter_m is None) or (n_layers is not None and diameter_m is not None):
        raise ValueError("Exactly one of 'n_layers' or 'diameter_m' must be provided.")


# ============================================================
# Layer sequence logic
# ============================================================

def alternating_layer_labels(
    n_layers: int,
    outer_layer: BraggLayerLabel,
) -> tuple[BraggLayerLabel, ...]:
    """
    Return alternating Bragg layer labels from core -> outer.

    Example
    -------
    n_layers=5, outer_layer="A" -> ("A", "B", "A", "B", "A")
    n_layers=4, outer_layer="A" -> ("B", "A", "B", "A")
    """
    if not isinstance(n_layers, int) or n_layers < 1:
        raise ValueError(f"n_layers must be an integer >= 1, got {n_layers!r}.")
    if outer_layer not in ("A", "B"):
        raise ValueError(f"outer_layer must be 'A' or 'B', got {outer_layer!r}.")

    labels_outer_to_core = tuple(
        outer_layer if i % 2 == 0 else _other_label(outer_layer)
        for i in range(n_layers)
    )
    return tuple(reversed(labels_outer_to_core))


def _layer_thicknesses_from_labels(
    labels_core_to_outer: tuple[BraggLayerLabel, ...],
    thicknesses: LayerThicknesses,
) -> FloatArray:
    """
    Compute Bragg layer thicknesses from labels.

    Rule:
    - shell thickness = t_material
    - core thickness = core_thickness_factor * t_material(core)
    """
    values = np.empty(len(labels_core_to_outer), dtype=float)

    for i, label in enumerate(labels_core_to_outer):
        t = thicknesses.for_material(label)
        values[i] = thicknesses.core_thickness_factor * t if i == 0 else t

    return values


def _build_geometry_from_labels(
    bragg_labels_core_to_outer: tuple[BraggLayerLabel, ...],
    thicknesses: LayerThicknesses,
    outer_layer: BraggLayerLabel,
    extra_outer_shell: ExtraOuterShellSpec | None = None,
) -> BraggOnionGeometry:
    """
    Build a BraggOnionGeometry from an already determined Bragg sequence.
    """
    bragg_thicknesses_m = _layer_thicknesses_from_labels(
        bragg_labels_core_to_outer,
        thicknesses,
    )

    if extra_outer_shell is None:
        layer_labels: tuple[LayerLabel, ...] = bragg_labels_core_to_outer
        layer_thicknesses_m = bragg_thicknesses_m
    else:
        _validate_positive("extra_outer_shell.thickness_m", extra_outer_shell.thickness_m)
        layer_labels = bragg_labels_core_to_outer + ("O",)
        layer_thicknesses_m = np.concatenate(
            [bragg_thicknesses_m, np.array([extra_outer_shell.thickness_m], dtype=float)]
        )

    outer_radii_m = np.cumsum(layer_thicknesses_m, dtype=float)
    outer_radius_m = float(outer_radii_m[-1])
    diameter_m = 2.0 * outer_radius_m

    return BraggOnionGeometry(
        layer_labels=layer_labels,
        layer_thicknesses_m=layer_thicknesses_m,
        outer_radii_m=outer_radii_m,
        n_layers=len(bragg_labels_core_to_outer),
        n_layers_total=len(layer_labels),
        outer_layer=outer_layer,
        t_a_m=float(thicknesses.t_a_m),
        t_b_m=float(thicknesses.t_b_m),
        outer_radius_m=outer_radius_m,
        diameter_m=diameter_m,
        design_peak_wavelength_m=thicknesses.design_peak_wavelength_m,
        core_thickness_factor=thicknesses.core_thickness_factor,
        extra_outer_shell=extra_outer_shell,
    )


# ============================================================
# Thickness construction
# ============================================================

def quarter_wave_thicknesses(
    material_a: Dispersion,
    material_b: Dispersion,
    peak_wavelength_m: float,
    core_thickness_factor: float = 1.0,
) -> LayerThicknesses:
    """
    Compute quarter-wave shell thicknesses from a design wavelength.

    Uses:
        t_i = lambda_peak / (4 * Re[n_i(lambda_peak)])

    Notes
    -----
    - Only the real part of the refractive index is used.
    - The core thickness is handled separately via core_thickness_factor.
    """
    _validate_positive("peak_wavelength_m", peak_wavelength_m)
    _validate_positive("core_thickness_factor", core_thickness_factor)

    wl = np.array([peak_wavelength_m], dtype=float)
    n_a = float(np.real(material_a(wl)[0]))
    n_b = float(np.real(material_b(wl)[0]))

    _validate_positive("Re[n_A(peak_wavelength)]", n_a)
    _validate_positive("Re[n_B(peak_wavelength)]", n_b)

    t_a_m = peak_wavelength_m / (4.0 * n_a)
    t_b_m = peak_wavelength_m / (4.0 * n_b)

    return LayerThicknesses(
        t_a_m=t_a_m,
        t_b_m=t_b_m,
        design_peak_wavelength_m=peak_wavelength_m,
        core_thickness_factor=core_thickness_factor,
    )


# ============================================================
# Number-of-layers / diameter logic
# ============================================================

def _geometry_for_n_layers(
    n_layers: int,
    outer_layer: BraggLayerLabel,
    thicknesses: LayerThicknesses,
    extra_outer_shell: ExtraOuterShellSpec | None = None,
) -> BraggOnionGeometry:
    labels = alternating_layer_labels(n_layers=n_layers, outer_layer=outer_layer)
    return _build_geometry_from_labels(
        bragg_labels_core_to_outer=labels,
        thicknesses=thicknesses,
        outer_layer=outer_layer,
        extra_outer_shell=extra_outer_shell,
    )


def _infer_max_layers_from_diameter(
    diameter_m: float,
    outer_layer: BraggLayerLabel,
    thicknesses: LayerThicknesses,
    extra_outer_shell: ExtraOuterShellSpec | None = None,
) -> int:
    """
    Find the largest number of full Bragg layers that fits within the requested
    total diameter. If an extra outer shell is present, it is included in the
    diameter budget.
    """
    _validate_positive("diameter_m", diameter_m)

    n_layers = 1
    best_n_layers = None

    while True:
        geom = _geometry_for_n_layers(
            n_layers=n_layers,
            outer_layer=outer_layer,
            thicknesses=thicknesses,
            extra_outer_shell=extra_outer_shell,
        )
        if geom.diameter_m <= diameter_m + 1e-18:
            best_n_layers = n_layers
            n_layers += 1
        else:
            break

    if best_n_layers is None:
        smallest = _geometry_for_n_layers(
            n_layers=1,
            outer_layer=outer_layer,
            thicknesses=thicknesses,
            extra_outer_shell=extra_outer_shell,
        )
        raise ValueError(
            "Requested diameter is too small even for the smallest possible particle. "
            f"Minimum achievable diameter is {smallest.diameter_m:.6e} m."
        )

    return best_n_layers


# ============================================================
# Extinction manipulation helpers
# ============================================================

def _evaluate_k_profile(
    profile: float | Dispersion | Callable[[FloatArray], FloatArray | ComplexArray],
    wavelengths_m: FloatArray,
) -> FloatArray:
    """
    Evaluate a k-profile on a wavelength grid.

    Supported profile types:
    - float
    - dispersion-like callable object (imaginary part is used)
    - callable returning real or complex values
    """
    wl = np.asarray(wavelengths_m, dtype=float)

    if np.isscalar(profile):
        return np.full(wl.shape, float(profile), dtype=float)

    if _is_dispersion_like(profile):
        values = np.asarray(profile(wl))
        if values.shape != wl.shape:
            raise ValueError(
                f"Dispersion-like k-profile returned shape {values.shape}, expected {wl.shape}."
            )
        if np.iscomplexobj(values):
            return np.imag(values).astype(float)
        return np.asarray(values, dtype=float)

    values = np.asarray(profile(wl))
    if values.ndim == 0:
        return np.full(wl.shape, float(values), dtype=float)
    if values.shape != wl.shape:
        raise ValueError(
            f"k-profile returned shape {values.shape}, expected {wl.shape}."
        )
    if np.iscomplexobj(values):
        return np.imag(values).astype(float)
    return np.asarray(values, dtype=float)


def _modifier_applies(
    modifier: ExtinctionModifier,
    layer_label: LayerLabel,
    layer_index: int,
) -> bool:
    """
    Check whether an extinction modifier applies to a given layer.

    Notes
    -----
    - Modifiers do not apply to the optional extra outer shell ("O").
      If you want custom optical constants there, define them directly
      through ExtraOuterShellSpec.material.
    """
    if layer_label == "O":
        return False

    if modifier.target == "core":
        return layer_index == 0
    if modifier.target == "material_A":
        return layer_label == "A"
    if modifier.target == "material_B":
        return layer_label == "B"
    if modifier.target == "both_materials":
        return layer_label in ("A", "B")

    raise ValueError(f"Unknown extinction target: {modifier.target!r}")


def _make_modified_dispersion(
    base_material: Dispersion,
    modifiers: list[ExtinctionModifier],
    derived_name: str,
) -> Dispersion:
    """
    Create a new Dispersion with modified extinction coefficient.
    Modifiers are applied sequentially in the given order.
    """
    if not modifiers:
        return base_material

    def _nk(wavelengths_m: FloatArray) -> ComplexArray:
        base_nk = np.asarray(base_material(wavelengths_m), dtype=np.complex128)
        n_vals = np.real(base_nk).astype(float)
        k_vals = np.imag(base_nk).astype(float)

        for modifier in modifiers:
            k_mod = _evaluate_k_profile(modifier.profile, wavelengths_m)
            if modifier.mode == "add":
                k_vals = k_vals + k_mod
            elif modifier.mode == "substitute":
                k_vals = k_mod
            else:
                raise ValueError(f"Unknown extinction mode: {modifier.mode!r}")

        return n_vals + 1j * k_vals

    return Dispersion(
        name=derived_name,
        nk=_nk,
        source=f"derived from {base_material.name}",
    )


# ============================================================
# Public builders
# ============================================================

def build_bragg_onion_from_thicknesses(
    *,
    t_a_m: float,
    t_b_m: float,
    outer_layer: BraggLayerLabel,
    n_layers: int | None = None,
    diameter_m: float | None = None,
    extra_outer_shell: ExtraOuterShellSpec | None = None,
    core_thickness_factor: float = 1.0,
) -> BraggOnionGeometry:
    """
    Build a Bragg onion from explicit shell thicknesses.

    Exactly one of n_layers or diameter_m must be provided.

    Parameters
    ----------
    t_a_m, t_b_m :
        Shell thicknesses of materials A and B [m]
    outer_layer :
        Outermost Bragg material ("A" or "B")
    n_layers :
        Number of Bragg layers including core
    diameter_m :
        Maximum total particle diameter [m]
    extra_outer_shell :
        Optional extra outermost shell
    core_thickness_factor :
        Core thickness relative to the shell thickness of the core material
    """
    _validate_positive("t_a_m", t_a_m)
    _validate_positive("t_b_m", t_b_m)
    _validate_positive("core_thickness_factor", core_thickness_factor)
    _validate_layer_choice(n_layers, diameter_m)

    thicknesses = LayerThicknesses(
        t_a_m=t_a_m,
        t_b_m=t_b_m,
        core_thickness_factor=core_thickness_factor,
    )

    if n_layers is not None:
        return _geometry_for_n_layers(
            n_layers=n_layers,
            outer_layer=outer_layer,
            thicknesses=thicknesses,
            extra_outer_shell=extra_outer_shell,
        )

    inferred_n_layers = _infer_max_layers_from_diameter(
        diameter_m=diameter_m,
        outer_layer=outer_layer,
        thicknesses=thicknesses,
        extra_outer_shell=extra_outer_shell,
    )
    return _geometry_for_n_layers(
        n_layers=inferred_n_layers,
        outer_layer=outer_layer,
        thicknesses=thicknesses,
        extra_outer_shell=extra_outer_shell,
    )


def build_bragg_onion_from_peak_wavelength(
    *,
    material_a: Dispersion,
    material_b: Dispersion,
    peak_wavelength_m: float,
    outer_layer: BraggLayerLabel,
    n_layers: int | None = None,
    diameter_m: float | None = None,
    extra_outer_shell: ExtraOuterShellSpec | None = None,
    core_thickness_factor: float = 1.0,
) -> BraggOnionGeometry:
    """
    Build a Bragg onion from quarter-wave shell thicknesses.

    Uses:
        t_i = lambda_peak / (4 * Re[n_i(lambda_peak)])

    Exactly one of n_layers or diameter_m must be provided.
    """
    _validate_layer_choice(n_layers, diameter_m)

    thicknesses = quarter_wave_thicknesses(
        material_a=material_a,
        material_b=material_b,
        peak_wavelength_m=peak_wavelength_m,
        core_thickness_factor=core_thickness_factor,
    )

    if n_layers is not None:
        return _geometry_for_n_layers(
            n_layers=n_layers,
            outer_layer=outer_layer,
            thicknesses=thicknesses,
            extra_outer_shell=extra_outer_shell,
        )

    inferred_n_layers = _infer_max_layers_from_diameter(
        diameter_m=diameter_m,
        outer_layer=outer_layer,
        thicknesses=thicknesses,
        extra_outer_shell=extra_outer_shell,
    )
    return _geometry_for_n_layers(
        n_layers=inferred_n_layers,
        outer_layer=outer_layer,
        thicknesses=thicknesses,
        extra_outer_shell=extra_outer_shell,
    )


# ============================================================
# Resolve effective per-layer materials
# ============================================================

def resolve_layer_stack(
    geometry: BraggOnionGeometry,
    material_a: Dispersion,
    material_b: Dispersion,
    extinction_modifiers: ExtinctionModifier | Iterable[ExtinctionModifier] | None = None,
) -> ResolvedLayerStack:
    """
    Resolve the effective material assigned to each layer.

    Parameters
    ----------
    geometry :
        Geometry created by one of the build_* functions
    material_a, material_b :
        Base Bragg materials
    extinction_modifiers :
        Optional extinction modifiers. Can be:
        - one ExtinctionModifier
        - an iterable of ExtinctionModifier
        - None

    Returns
    -------
    ResolvedLayerStack
        Geometry plus one effective Dispersion per layer

    Notes
    -----
    - The optional extra outer shell uses geometry.extra_outer_shell.material directly
    - Extinction modifiers do not affect the extra outer shell
    """
    if extinction_modifiers is None:
        modifier_list: list[ExtinctionModifier] = []
    elif isinstance(extinction_modifiers, ExtinctionModifier):
        modifier_list = [extinction_modifiers]
    else:
        modifier_list = list(extinction_modifiers)

    layer_material_names: list[str] = []
    layer_dispersions: list[Dispersion] = []

    for idx, label in enumerate(geometry.layer_labels):
        if label == "A":
            base = material_a
            base_name = material_a.name
        elif label == "B":
            base = material_b
            base_name = material_b.name
        elif label == "O":
            if geometry.extra_outer_shell is None:
                raise RuntimeError(
                    "Geometry contains 'O' layer but no extra_outer_shell is stored."
                )
            base = geometry.extra_outer_shell.material
            base_name = geometry.extra_outer_shell.name
        else:
            raise ValueError(f"Unknown layer label: {label!r}")

        applicable_modifiers = [
            mod for mod in modifier_list
            if _modifier_applies(mod, layer_label=label, layer_index=idx)
        ]

        if applicable_modifiers:
            effective = _make_modified_dispersion(
                base_material=base,
                modifiers=applicable_modifiers,
                derived_name=f"{base_name}_modified_layer{idx}",
            )
            layer_material_names.append(effective.name)
            layer_dispersions.append(effective)
        else:
            layer_material_names.append(base_name)
            layer_dispersions.append(base)

    return ResolvedLayerStack(
        geometry=geometry,
        layer_material_names=tuple(layer_material_names),
        layer_dispersions=tuple(layer_dispersions),
    )