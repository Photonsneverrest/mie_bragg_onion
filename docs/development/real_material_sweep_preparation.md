# Real-material PS/PMMA Bragg-onion sweep: frozen specification

## Purpose
This is the final remaining Mie-scattering parameter study before publication-figure consolidation. It will use the full dispersive complex refractive indices of PS and PMMA.

## Stages
1. **Layer-count optimization** for design wavelengths 380, 470, 540, 580, 620, 680, 760, and 850 nm, up to 120 radial domains.
2. **Hollowing optimization** for every design, using only physically distinct replacements and selecting the minimum remaining active PS domains that satisfies the frozen 90% colour-performance rule.
3. **Absorber optimization** only for 470 and 580 nm, using the selected hollow geometry and an absorber in the hollow region.

## Proposed frozen performance rule
A hollow geometry is eligible only when all conditions hold relative to the optimum non-hollow PS/PMMA sphere for the same design wavelength:
- backward visible spectral-integral retention >= 0.90;
- visible backward-spectrum correlation >= 0.98;
- CIELAB DeltaE76 <= 5;
- colour-performance score `(eta_C * eta_L)` retention >= 0.90.

The selected geometry is the eligible case with the fewest remaining active PS domains. Peak retention is reported but is not the primary selection metric because narrow Mie peaks can be numerically sensitive.

## Material-data constraint
The supplied Zhang et al. PS and PMMA files begin at 0.40 um. A 380 nm design is outside the tabulated range. The production script must not extrapolate silently. One of the following must be approved:
1. provide optical constants covering at least 360-400 nm;
2. explicitly authorize a documented extrapolation rule;
3. omit the 380 nm material-realistic design.

## Required source inspection before script generation
Upload the current authoritative files:
- `src/bragg_onion/materials.py`;
- the current production sweep used for the ideal/material studies (preferably `scripts/run_primary_sweep.py`, or state the actual filename);
- one combined text file containing the complete PS and PMMA source files with clear separators.

The final sweep script will be generated only after verifying the actual material-loader API, sweep CLI, automatic output behavior, and colour-calculation path.
