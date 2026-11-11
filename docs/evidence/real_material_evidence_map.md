# Evidence map for material-realistic PS/PMMA section

| Manuscript statement | Required evidence |
|---|---|
| One intact optimum is selected per design wavelength | Figure 7a; `Table_real_material_intact_optima_full.csv` |
| The selected hollow geometry retains at least 90% backward integral, 98% spectral correlation, 90% colour performance and $\Delta E_{76}\le5$ | Figure 7b; `Table_real_material_hollowing_selection_full.csv` |
| PS/PMMA optical constants are dispersive and not extrapolated | `run_metadata.json`; material-range validation; `Zhang2020PolymerComplexRefractiveIndices` |
| The realistic 380 nm case is omitted because the data begin at 400 nm | Methods text; frozen config |
| The melanin-like fill uses $n=1.75$ as a representative value, not a universal constant | Methods text; `Stavenga2015MelaninRefractiveIndex`; `Li2020SyntheticMelaninOpticalConstants` |
| Forward attenuation is balanced against backward retention | Figure 7d; `melanin_fill_tradeoff.csv` |
