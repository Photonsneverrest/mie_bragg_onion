from bragg_onion.materials import ConstantDispersion
from bragg_onion.geometry import build_bragg_onion_from_peak_wavelength, resolve_layer_stack

def main():
    mat_A = ConstantDispersion.from_nk("A", n=1.59, k=0.0)
    mat_B = ConstantDispersion.from_nk("B", n=1.49, k=0.0)
    medium = ConstantDispersion.from_nk("medium", n=1.33, k=0.0)

    geom = build_bragg_onion_from_peak_wavelength(
        material_a=mat_A,
        material_b=mat_B,
        peak_wavelength_m=550e-9,
        outer_layer="A",
        n_layers=7,
        core_thickness_factor=0.5,
    )

    stack = resolve_layer_stack(
        geometry=geom,
        material_a=mat_A,
        material_b=mat_B,
    )

    print("Core smoke test successful.")
    print("Layer labels:", stack.layer_labels)
    print("Radii [nm]:", stack.radii_m * 1e9)
    print("Relative indices at 550 nm:", stack.m_at_wavelength(550e-9, n_medium=medium))

if __name__ == "__main__":
    main()