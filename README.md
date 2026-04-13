# mie_bragg_onion

**This project is currently under active research development.**  
APIs may change without notice.

Scattnlay-based Python tools for multilayer **Bragg onion spheres**:
wavelength-dependent materials, multilayer geometry generation, scattering simulation,
NA-based collection integration, colour analysis, parameter sweeps, and near-field / Poynting-flow visualization.

---

## Features

- **Material handling**
  - load dispersive refractive index data from files
  - interpolate `n(λ) + i k(λ)`
  - support wavelength units in nm / µm / m

- **Bragg onion geometry**
  - alternating A/B multilayer spheres
  - explicit thickness mode
  - quarter-wave design mode from peak wavelength
  - selectable outer layer
  - optional extra outer shell
  - optional extinction-coefficient modifications

- **Scattering** *(optional simulation stack)*
  - wrapper around `scattnlay`
  - differential scattering cross-sections
  - efficiencies and absolute cross-sections
  - angle-resolved scattering spectra

- **Collection / reflectance-like metrics** *(optional simulation stack)*
  - integrate scattering over collection numerical aperture (NA)
  - forward / backward collection
  - collected fraction and collected cross-section

- **Colour analysis**
  - XYZ, xyY, CIELAB, sRGB, HSV, Hex
  - Rosch–MacAdam colour-solid based performance metrics

- **Sweep / screening** *(optional simulation stack)*
  - scan design wavelength, number of layers, outer layer, and more
  - collect metrics into a tidy `pandas.DataFrame`
  - plot sweep trends and resulting colours

- **Near field** *(optional simulation stack)*
  - field maps in selected planes
  - total / scattered-like / delta-flow visualizations
  - Poynting streamlines

---

## Installation

### Core package

Install the core package in editable mode from the repository root:

```bash
python -m pip install -e .