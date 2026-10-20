# field_analysis.py

import numpy as np
import matplotlib.pyplot as plt

from bragg_onion.geometry import (
    build_bragg_onion_from_peak_wavelength,
    resolve_layer_stack,
)

from bragg_onion.fields import (
    compute_field_map,
    plot_field_magnitude,
    plot_poynting_streamlines,
)


def rebuild_stack_from_row(
    row,
    *,
    material_a,
    material_b,
    fixed_geometry_kwargs=None,
    extinction_modifiers=None,
    extra_outer_shell=None,
):
    """
    Reconstruct geometry + resolved layer stack (exactly like your working notebook).
    """

    fixed_geometry_kwargs = dict(fixed_geometry_kwargs or {})

    geom_kwargs = dict(fixed_geometry_kwargs)
    geom_kwargs.update(
        {
            "peak_wavelength_m": float(row["peak_wavelength_m"]),
            "n_layers": int(row["n_layers"]),
            "outer_layer": str(row["outer_layer"]),
            "core_thickness_factor": float(row["core_thickness_factor"]),
        }
    )

    geometry = build_bragg_onion_from_peak_wavelength(
        material_a=material_a,
        material_b=material_b,
        extra_outer_shell=extra_outer_shell,
        **geom_kwargs,
    )

    stack = resolve_layer_stack(
        geometry=geometry,
        material_a=material_a,
        material_b=material_b,
        extinction_modifiers=extinction_modifiers,
    )

    return geometry, stack
