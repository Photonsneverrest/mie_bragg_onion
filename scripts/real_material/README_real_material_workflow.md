# PS/PMMA production package
Place `real_material_common.py` beside the five executable scripts, or ensure the scripts directory is on `PYTHONPATH`.

## Verified project interfaces
- `run_bragg_onion_sweep(..., layer_substitutions=...)` exists.
- `LayerSubstitution(layer_indices, material, name)` exists.
- Angles are supplied in radians.
- No extrapolation is used.
- No wrapper-only flags are assumed.

## Sequence
1. Run intact sweep.
2. Select intact optima.
3. Run hollowing sweep and select minimum-material eligible geometry.
4. Run melanin-like fill sweep for 470 and 580 nm.
5. Run analysis.
