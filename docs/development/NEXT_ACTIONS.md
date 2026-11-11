# Sequential next actions

1. Run `run_homogeneous_reference_sweep_v3.py`. This corrects the polar-angle unit passed to `scattnlay`.
2. Upload only:
   - `homogeneous_reference_summary_v3.csv`
   - `homogeneous_reference_spectra_v3.csv`
   - `homogeneous_reference_polar_470nm_v3.csv`
3. After those files are reviewed, freeze the 470 nm mechanism figure and LaTeX snippet.
4. Then start the 580 nm low-contrast hollowing generality sweep.

Do not use v1 or v2 forward/backward hemisphere values in the manuscript. The v2 full-sphere closure is internally normalized, but the amplitudes were evaluated with a degree grid passed to an API used elsewhere in the project with radians.
