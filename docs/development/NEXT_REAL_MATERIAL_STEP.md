# Next production step

The real-material study is frozen to PS/PMMA designs at 470, 540, 580, 620, 680, 760, and 850 nm. The idealized 380 nm result remains in the manuscript, but no material-realistic 380 nm calculation will be claimed because the selected Zhang et al. polymer data start at 400 nm.

The realistic absorber is represented as an **idealized melanin-like fill** with constant real index n=1.75 and a swept extinction coefficient k. This value is supported as a visible-range melanin reference by Stavenga et al. (melanin n approximately 1.7-1.8), while Li et al. supplies wavelength-dependent complex optical constants for synthetic melanin and motivates treating k as chemistry- and wavelength-dependent.

Before production scripts are generated, run `inspect_real_material_interfaces.py` and upload its JSON report together with the current `src/bragg_onion/geometry.py` and `src/bragg_onion/sweep.py`. This preserves the repository rule that source interfaces, wrapper CLI arguments, and output behavior must be verified before code changes or execution commands are issued.
