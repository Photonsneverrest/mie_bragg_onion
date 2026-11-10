#!/usr/bin/env python
# coding: utf-8

# <strong># OVERVIEW OF THE STUDY</strong>
# <br><br>
# <strong>## Parameters of the absorber-location sweep</strong>
# <br><br>
# <strong>Material A refractive index</strong><br>
# n<sub>A</sub> ∈ [1.01, 1.9] (20 steps)
# <br><br>
# <strong>Material B refractive index</strong><br>
# n<sub>B</sub> = 1.0
# <br><br>
# <strong>Absorber extinction coefficient</strong><br>
# k ∈ {0.01, 0.10} (10 steps)
# <br><br>
# <strong>Absorber location</strong><br>
# (i) Core only<br>
# (ii) Low-index layers only (material B)<br>
# (iii) All Bragg layers (materials A and B)<br>
# (iv) Additional outer absorbing shell
# <br><br>
# <strong>Design wavelength</strong><br>
# λ<sub>design</sub> ∈ {480, 560, 640} nm
# <br><br>
# <strong>Number of Bragg layers</strong><br>
# N = 60
# <br><br>
# <strong>Outer Bragg layer</strong><br>
# Material A
# <br><br>
# <strong>Collection geometry</strong><br>
# Backward hemisphere collection
# <br><br>
# <strong>Collection numerical aperture</strong><br>
# NA = 1.0
# <br><br>
# <strong>Outer shell thicknesses</strong><br>
# d<sub>shell</sub> ∈ {5, 10, 20, 40} nm
# <br><br>
# <strong>Core thickness Factors</strong><br>
# d<sub>shell</sub> ∈ {0.5, 1.0, 2.0, 4.0}
# <br><br>
# <strong>Objective</strong><br>
# Quantify how absorber placement influences the optical response, colour quality, scattering efficiency, and collection efficiency of Bragg onion photonic pigments while keeping the absorber strength, geometry, and collection conditions fixed.
#
#
# ## Quantities to analyze
#
#
# ### 1. Colour performance
#
# Rosch–MacAdam colour solid
#
# Optimal colour contour
#
# CIELAB a*–b*
#
# Colour strips
#
# η_C, η_L heatmaps
#
# ### 2. Scattering
#
# Differential scattering vs θ (dB)
#
# Integrated efficiencies vs λ
#
# Asymmetry g, albedo
#
# Forward / side / backward (dB)
#
# Pie chart at α = 70.5°
#
# ### 3. Near fields
#
# |E| (dB), xz / yz / xy
#
# Poynting flow + streamlines
#
# Hotspots
#
# ### 4. Resonances
#
# Mie l‑decomposition
#
# l‑resolved spectra
#
# ### 5. Sensitivity
#
# η_C, η_L vs n_layers, Δn, k

# In[185]:


# CELL 0 — IMPORTS & GLOBAL SETTINGS

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from bragg_onion._notebook_tools.io import load_sweep_outputs
from bragg_onion._notebook_tools.colour import add_display_normalized_colour_columns
from bragg_onion._notebook_tools.strip_plots import (
    select_best_etaC_by_wavelength,
    plot_best_etaC_colour_strip,
    select_strip_grid,
    plot_colour_strip_grid,
    plot_sweep_heatmap_local,
    prepare_strip_grid,
    summarize_strip_grid_cases,
    plot_colour_strip_grid,
)
from bragg_onion._notebook_tools.rosch_plots import (
    make_rosch_macadam_figures,
    add_highlight_cielab_points,
)

plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 600

print("Notebook helper modules imported successfully.")

# In[186]:


# CELL 3 — OUTPUT SOURCE / NOTEBOOK SETTINGS

# Windows path to the copied workstation output directory.
# This can stay in Windows form even when the notebook runs in WSL:
# load_sweep_outputs(...) will convert it automatically if needed.
SOURCE_OUTPUT_DIR_WINDOWS = (
    r"C:\Users\SchwarzN\OneDrive - Université de Fribourg"
    r"\Institution\P1_BraggSpericalPigments\Simulation\MIE_Computation"
    r"\Scattnlay_BraggOnion\outputs\rerun_primary_sweep_01"
)

# Default colour view for object-colour-style plotting:
DEFAULT_COLOUR_VIEW = "display"   # "raw", "display", or "both"

print("SOURCE_OUTPUT_DIR_WINDOWS:")
print(SOURCE_OUTPUT_DIR_WINDOWS)
print("DEFAULT_COLOUR_VIEW:", DEFAULT_COLOUR_VIEW)

# In[187]:


# CELL 4 — LOAD SWEEP OUTPUTS & ADD DISPLAY-NORMALIZED COLOURS

df_all, OUTPUT_DIR, USE_PARQUET = load_sweep_outputs(
    SOURCE_OUTPUT_DIR_WINDOWS,
    combine_if_needed=True,
    verbose=True,
)

print("\nLoaded dataframe summary:")
print("OUTPUT_DIR   :", OUTPUT_DIR)
print("USE_PARQUET  :", USE_PARQUET)
print("df_all rows  :", len(df_all))

# In[188]:


# FLAG CASES CONSIDERED NUMERICALLY OR PHYSICALLY SUSPICIOUS
#
# df_all remains unchanged in content, except for two added audit columns:
#   excluded_from_analysis
#   exclusion_reason
#
# df_analysis contains only the retained cases.

import numpy as np
import pandas as pd


# ============================================================
# EXCLUSION DEFINITIONS
# ============================================================
#
# Missing parameters act as wildcards.
#
# Example:
# {
#     "design_peak_wavelength_nm": 580.0,
#     "absorber_location": "low_index",
#     "n_A": 1.51,
#     "n_layers": 100,
# }
#
# flags every matching k value.

exclusion_cases = [
    {
        "filters": {
            "design_peak_wavelength_nm": 540.0,
            "absorber_location": "low_index",
            "n_A": 1.17666667,
            "k": 0.001,
            "n_layers": 40,
        },
        "reason": "Suspicious hue or colour result: low-index absorber, 540 nm, nA=1.1767, k=0.001, 40 layers",
    },
    {
        "filters": {
            "design_peak_wavelength_nm": 540.0,
            "absorber_location": "low_index",
            "n_A": 1.17666667,
            "k": 0.00316228,
            "n_layers": 40,
        },
        "reason": "Suspicious hue or colour result: low-index absorber, 540 nm, nA=1.1767, k=0.00316, 40 layers",
    },
    {
        "filters": {
            "design_peak_wavelength_nm": 540.0,
            "absorber_location": "low_index",
            "n_A": 1.17666667,
            "k": 0.01,
            "n_layers": 40,
        },
        "reason": "Suspicious hue or colour result: low-index absorber, 540 nm, nA=1.1767, k=0.01, 40 layers",
    },
    {
        "filters": {
            "design_peak_wavelength_nm": 540.0,
            "absorber_location": "low_index",
            "n_A": 1.17666667,
            "k": 0.1,
            "n_layers": 63,
        },
        "reason": "Suspicious hue or colour result: low-index absorber, 540 nm, nA=1.1767, k=0.1, 63 layers",
    },
    {
        "filters": {
            "design_peak_wavelength_nm": 540.0,
            "absorber_location": "low_index",
            "n_A": 1.17666667,
            "k": 0.001,
            "n_layers": 100,
        },
        "reason": "Suspicious hue or colour result: low-index absorber, 540 nm, nA=1.1767, k=0.001, 100 layers",
    },
    {
        "filters": {
            "design_peak_wavelength_nm": 580.0,
            "absorber_location": "low_index",
            "n_A": 1.51,
            "n_layers": 100,
        },
        "reason": "Suspicious hue or colour result across all k: low-index absorber, 580 nm, nA=1.51, 100 layers",
    },
    {
        "filters": {
            "design_peak_wavelength_nm": 620.0,
            "absorber_location": "low_index",
            "n_A": 1.51,
            "n_layers": 100,
        },
        "reason": "Suspicious hue or colour result across all k: low-index absorber, 620 nm, nA=1.51, 100 layers",
    },
    {
        "filters": {
            "design_peak_wavelength_nm": 680.0,
            "absorber_location": "none",
            "n_A": 1.51,
            "n_layers": 63,
        },
        "reason": "Suspicious hue or colour result: nonabsorbing case, 680 nm, nA=1.51, 63 layers",
    },
    {
        "filters": {
            "design_peak_wavelength_nm": 680.0,
            "absorber_location": "none",
            "n_A": 2.51,
            "n_layers": 40,
        },
        "reason": "Suspicious hue or colour result: nonabsorbing case, 680 nm, nA=2.51, 40 layers",
    },
    {
        "filters": {
            "design_peak_wavelength_nm": 680.0,
            "absorber_location": "none",
            "n_A": 2.51,
            "n_layers": 63,
        },
        "reason": "Suspicious hue or colour result: nonabsorbing case, 680 nm, nA=2.51, 63 layers",
    },
    {
        "filters": {
            "design_peak_wavelength_nm": 380.0,
            "absorber_location": "none",
            "n_A": 1.51,
            "n_layers": 25,
        },
        "reason": "Suspicious hue or colour result: nonabsorbing case, 380 nm, nA=1.51, 25 layers",
    },
    {
        "filters": {
            "design_peak_wavelength_nm": 380.0,
            "absorber_location": "none",
            "n_A": 1.51,
            "n_layers": 63,
        },
        "reason": "Suspicious hue or colour result: nonabsorbing case, 380 nm, nA=1.51, 63 layers",
    },
    {
        "filters": {
            "design_peak_wavelength_nm": 580.0,
            "absorber_location": "none",
            "n_A": 1.51,
            "n_layers": 100,
        },
        "reason": "Suspicious hue or colour result: nonabsorbing case, 580 nm, nA=1.51, 100 layers",
    },
    {
        "filters": {
            "design_peak_wavelength_nm": 620.0,
            "absorber_location": "none",
            "n_A": 1.51,
            "n_layers": 100,
        },
        "reason": "Suspicious hue or colour result: nonabsorbing case, 620 nm, nA=1.51, 100 layers",
    },
    {
        "filters": {
            "design_peak_wavelength_nm": 680.0,
            "absorber_location": "low_index",
            "n_A": 1.51,
            "k": 1.0,
            "n_layers": 10,
        },
        "reason": "Suspicious hue or colour result: strongly absorbing low-index case, 680 nm, k=1, 10 layers",
    },
    {
        "filters": {
            "design_peak_wavelength_nm": 760.0,
            "absorber_location": "low_index",
            "n_A": 1.51,
            "k": 1.0,
            "n_layers": 10,
        },
        "reason": "Suspicious hue or colour result: strongly absorbing low-index case, 760 nm, k=1, 10 layers",
    },
]


# ============================================================
# HELPER FUNCTION
# ============================================================

def flag_parameter_cases(
    df: pd.DataFrame,
    exclusion_cases: list[dict],
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    reset_existing_flags: bool = True,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Flag rows that match any exclusion definition.

    Numeric values are compared using np.isclose.
    Text and categorical values are compared exactly.

    Parameters omitted from an exclusion dictionary act as wildcards.

    Returns
    -------
    df_flagged
        Full dataframe with exclusion flags and reasons.
    df_excluded
        Only the flagged rows.
    """

    df_flagged = df.copy()

    if (
        reset_existing_flags
        or "excluded_from_analysis" not in df_flagged.columns
    ):
        df_flagged["excluded_from_analysis"] = False

    if (
        reset_existing_flags
        or "exclusion_reason" not in df_flagged.columns
    ):
        df_flagged["exclusion_reason"] = ""

    for exclusion_number, exclusion in enumerate(
        exclusion_cases,
        start=1,
    ):
        filters = exclusion["filters"]
        reason = exclusion.get(
            "reason",
            f"Manual exclusion {exclusion_number}",
        )

        case_mask = np.ones(
            len(df_flagged),
            dtype=bool,
        )

        for column, requested_value in filters.items():
            if column not in df_flagged.columns:
                raise KeyError(
                    f"Exclusion {exclusion_number} uses missing "
                    f"column {column!r}."
                )

            column_values = df_flagged[column]

            if (
                pd.api.types.is_numeric_dtype(column_values)
                and isinstance(
                    requested_value,
                    (int, float, np.integer, np.floating),
                )
            ):
                numeric_values = pd.to_numeric(
                    column_values,
                    errors="coerce",
                ).to_numpy(dtype=float)

                case_mask &= np.isclose(
                    numeric_values,
                    float(requested_value),
                    rtol=rtol,
                    atol=atol,
                    equal_nan=False,
                )

            else:
                case_mask &= (
                    column_values.astype(str).to_numpy()
                    == str(requested_value)
                )

        number_matching = int(case_mask.sum())

        if verbose:
            print(
                f"Exclusion {exclusion_number:02d}: "
                f"{number_matching} matching row(s)"
            )
            print("  Filters:", filters)

        if number_matching == 0:
            continue

        # Add rather than overwrite reasons if a row matches
        # more than one exclusion definition.
        existing_reasons = df_flagged.loc[
            case_mask,
            "exclusion_reason",
        ].astype(str)

        df_flagged.loc[
            case_mask,
            "exclusion_reason",
        ] = np.where(
            existing_reasons.str.len() > 0,
            existing_reasons + " | " + reason,
            reason,
        )

        df_flagged.loc[
            case_mask,
            "excluded_from_analysis",
        ] = True

    df_excluded = df_flagged.loc[
        df_flagged["excluded_from_analysis"]
    ].copy()

    if verbose:
        print("\nSummary")
        print("-------")
        print("Total rows       :", len(df_flagged))
        print("Flagged rows     :", len(df_excluded))
        print(
            "Retained rows    :",
            int(
                (
                    ~df_flagged["excluded_from_analysis"]
                ).sum()
            ),
        )

    return df_flagged, df_excluded


# ============================================================
# APPLY FLAGS
# ============================================================

df_all_flagged, df_excluded = flag_parameter_cases(
    df_all,
    exclusion_cases,
    rtol=1e-5,
    atol=1e-8,
    reset_existing_flags=True,
    verbose=True,
)

# Preserve every row and its traceability.
df_all = df_all_flagged

# Use this dataframe for subsequent scientific analysis.
df_analysis = df_all.loc[
    ~df_all["excluded_from_analysis"]
].copy()


# ============================================================
# INSPECT FLAGGED CASES
# ============================================================

columns_to_show = [
    "design_peak_wavelength_nm",
    "absorber_location",
    "n_A",
    "k",
    "n_layers",
    "eta_C",
    "eta_L",
    "C",
    "hue_deg",
    "hex",
    "has_warnings",
    "Warnings_messages",
    "exclusion_reason",
]

columns_to_show = [
    column
    for column in columns_to_show
    if column in df_excluded.columns
]

display(
    df_excluded[
        columns_to_show
    ].sort_values(
        [
            "design_peak_wavelength_nm",
            "absorber_location",
            "n_A",
            "n_layers",
            "k",
        ]
    )
)


# ============================================================
# OPTIONAL SUMMARY OF THE FLAGGED PARAMETER PATTERN
# ============================================================

print("\nFlagged cases by absorber location:")
display(
    df_excluded["absorber_location"]
    .value_counts(dropna=False)
    .rename_axis("absorber_location")
    .to_frame("number_flagged")
)

print("\nFlagged cases by design wavelength:")
display(
    df_excluded["design_peak_wavelength_nm"]
    .value_counts(dropna=False)
    .sort_index()
    .rename_axis("design_peak_wavelength_nm")
    .to_frame("number_flagged")
)

print("\nFlagged cases by refractive index:")
display(
    df_excluded["n_A"]
    .value_counts(dropna=False)
    .sort_index()
    .rename_axis("n_A")
    .to_frame("number_flagged")
)

print("\nFlagged cases by number of layers:")
display(
    df_excluded["n_layers"]
    .value_counts(dropna=False)
    .sort_index()
    .rename_axis("n_layers")
    .to_frame("number_flagged")
)

# In[189]:



# # Filter out bad cases (outliers in form of unreasonable spectra)
# df_all = df_all.loc[df_all["n_A"] != 2.5].copy()
# Probably also filter out NA = 0.22 cases!

# Working dataframe for plotting/analysis
df_plot = add_display_normalized_colour_columns(
    df_analysis,
    inplace=False,
)
print("df_plot rows :", len(df_plot))

display(df_plot.head())

# In[190]:


test_columns = [
    "absorber_location",
    "k",
    "core_thickness_factor",
    "n_layers",
    "n_layers_total",
    "core_radius_nm",
    "diameter_nm",
    "target_total_diameter_nm",
    "outer_shell_thickness_nm",
    "backward_forward_band_db",
    "g_band_weighted",
    "albedo_band_weighted",
    "g_off_band_rms",
    "albedo_off_band_rms",
]

display(
    df_analysis[
        [
            column
            for column in test_columns
            if column in df_analysis.columns
        ]
    ]
    .drop_duplicates()
    .sort_values(
        [
            "absorber_location",
            "core_thickness_factor",
        ]
    )
)

# In[213]:


# IDENTIFY THE BEST OVERALL COLOUR CASE FOR EACH TARGET WAVELENGTH
#
# Selection metric:
#
#   d_opt = sqrt[(1 - min(eta_C, 1))^2 + (1 - min(eta_L, 1))^2]
#
# Equal weighting is used for eta_C and eta_L.
# Values above 1.0 receive neither a penalty nor an additional reward.
# Smaller d_opt is better.

import numpy as np
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

TARGET_WAVELENGTH_COLUMN = "design_peak_wavelength_nm"

ETA_C_COLUMN = "eta_C"
ETA_L_COLUMN = "eta_L"

BEST_CASE_METRIC_COLUMN = "d_opt"

# Numerical tolerance used to identify tied best cases.
TIE_RTOL = 1e-10
TIE_ATOL = 1e-12


# ============================================================
# VALIDATE INPUT DATA
# ============================================================

required_columns = {
    TARGET_WAVELENGTH_COLUMN,
    ETA_C_COLUMN,
    ETA_L_COLUMN,
}

missing_columns = required_columns - set(df_analysis.columns)

if missing_columns:
    raise KeyError(
        "df_analysis is missing required columns: "
        f"{sorted(missing_columns)}"
    )


# Work on a copy so that df_analysis remains available unchanged.
df_colour_selection = df_analysis.copy()

df_colour_selection[ETA_C_COLUMN] = pd.to_numeric(
    df_colour_selection[ETA_C_COLUMN],
    errors="coerce",
)

df_colour_selection[ETA_L_COLUMN] = pd.to_numeric(
    df_colour_selection[ETA_L_COLUMN],
    errors="coerce",
)

df_colour_selection[TARGET_WAVELENGTH_COLUMN] = pd.to_numeric(
    df_colour_selection[TARGET_WAVELENGTH_COLUMN],
    errors="coerce",
)

# Exclude rows for which the selection metric cannot be calculated.
valid_metric_mask = (
    np.isfinite(df_colour_selection[ETA_C_COLUMN])
    & np.isfinite(df_colour_selection[ETA_L_COLUMN])
    & np.isfinite(
        df_colour_selection[TARGET_WAVELENGTH_COLUMN]
    )
)

number_invalid = int((~valid_metric_mask).sum())

if number_invalid > 0:
    print(
        f"Ignoring {number_invalid} row(s) with missing or "
        "nonfinite eta_C, eta_L, or target wavelength."
    )

df_colour_selection = df_colour_selection.loc[
    valid_metric_mask
].copy()

if df_colour_selection.empty:
    raise ValueError(
        "No valid rows remain for best-case selection."
    )


# ============================================================
# CALCULATE THE DISTANCE TO THE OPTIMAL COLOUR
# ============================================================

# Clip only the upper side:
#   eta > 1  -> treated as 1, therefore no penalty
#   eta < 0  -> retained and receives the corresponding deficit
df_colour_selection["eta_C_for_d_opt"] = (
    df_colour_selection[ETA_C_COLUMN]
    .clip(upper=1.0)
)

df_colour_selection["eta_L_for_d_opt"] = (
    df_colour_selection[ETA_L_COLUMN]
    .clip(upper=1.0)
)

df_colour_selection["eta_C_deficit"] = (
    1.0
    - df_colour_selection["eta_C_for_d_opt"]
)

df_colour_selection["eta_L_deficit"] = (
    1.0
    - df_colour_selection["eta_L_for_d_opt"]
)

df_colour_selection[BEST_CASE_METRIC_COLUMN] = np.hypot(
    df_colour_selection["eta_C_deficit"],
    df_colour_selection["eta_L_deficit"],
)


# ============================================================
# IDENTIFY ALL TIED BEST CASES FOR EACH TARGET WAVELENGTH
# ============================================================

minimum_distance_by_wavelength = (
    df_colour_selection
    .groupby(
        TARGET_WAVELENGTH_COLUMN,
        dropna=False,
    )[BEST_CASE_METRIC_COLUMN]
    .transform("min")
)

best_case_mask = np.isclose(
    df_colour_selection[BEST_CASE_METRIC_COLUMN],
    minimum_distance_by_wavelength,
    rtol=TIE_RTOL,
    atol=TIE_ATOL,
)

# Contains every tied optimum, if several cases have equal d_opt.
df_best_cases_all = (
    df_colour_selection.loc[best_case_mask]
    .copy()
    .sort_values(
        [
            TARGET_WAVELENGTH_COLUMN,
            BEST_CASE_METRIC_COLUMN,
            ETA_C_COLUMN,
            ETA_L_COLUMN,
        ],
        ascending=[
            True,
            True,
            False,
            False,
        ],
    )
    .reset_index(drop=True)
)


# ============================================================
# CREATE ONE DETERMINISTIC WINNER PER TARGET WAVELENGTH
# ============================================================
#
# If several cases have the same d_opt:
#   1. prefer higher raw eta_C;
#   2. then prefer higher raw eta_L;
#   3. then prefer lower k, if available;
#   4. then prefer fewer layers, if available.
#
# df_best_cases_all remains the authoritative table containing
# all mathematically tied best cases.

tie_break_columns = [
    TARGET_WAVELENGTH_COLUMN,
    BEST_CASE_METRIC_COLUMN,
    ETA_C_COLUMN,
    ETA_L_COLUMN,
]

tie_break_ascending = [
    True,   # wavelength
    True,   # lower d_opt
    False,  # higher eta_C
    False,  # higher eta_L
]

if "k" in df_colour_selection.columns:
    tie_break_columns.append("k")
    tie_break_ascending.append(True)

if "n_layers" in df_colour_selection.columns:
    tie_break_columns.append("n_layers")
    tie_break_ascending.append(True)

df_best_cases = (
    df_colour_selection
    .sort_values(
        tie_break_columns,
        ascending=tie_break_ascending,
    )
    .groupby(
        TARGET_WAVELENGTH_COLUMN,
        as_index=False,
        dropna=False,
    )
    .first()
    .sort_values(TARGET_WAVELENGTH_COLUMN)
    .reset_index(drop=True)
)


# ============================================================
# ADD IDENTIFIERS FOR LATER NOTEBOOK CELLS
# ============================================================

df_colour_selection["is_best_d_opt_case"] = best_case_mask

best_original_indices = set(
    df_best_cases_all.index
)

# Dictionary for convenient access:
# best_cases_by_wavelength[540.0] returns the selected row.
best_cases_by_wavelength = {
    float(row[TARGET_WAVELENGTH_COLUMN]): row
    for _, row in df_best_cases.iterrows()
}

# List of target wavelengths available in subsequent cells.
best_case_target_wavelengths = sorted(
    best_cases_by_wavelength
)


# ============================================================
# DISPLAY THE RESULTS
# ============================================================

columns_to_show = [
    TARGET_WAVELENGTH_COLUMN,
    BEST_CASE_METRIC_COLUMN,
    ETA_C_COLUMN,
    ETA_L_COLUMN,
    "eta_Y",
    "L",
    "a",
    "b",
    "C",
    "hue_deg",
    "n_A",
    "n_B",
    "k",
    "n_layers",
    "diameter_nm",
    "core_thickness_factor",
    "absorber_location",
    "q_collected_geom_peak_wavelength_nm",
    "hex",
    "hex_display",
]

columns_to_show = [
    column
    for column in columns_to_show
    if column in df_best_cases.columns
]

print(
    "One selected best case per target wavelength "
    "(minimum d_opt):"
)

display(
    df_best_cases[columns_to_show]
)


# Report mathematical ties separately.
tie_counts = (
    df_best_cases_all
    .groupby(TARGET_WAVELENGTH_COLUMN)
    .size()
    .rename("number_of_tied_best_cases")
    .reset_index()
)

print("\nNumber of tied optimum cases per target wavelength:")

display(tie_counts)

if (tie_counts["number_of_tied_best_cases"] > 1).any():
    print(
        "\nSome wavelengths have multiple cases with the same "
        "minimum d_opt. All tied cases are stored in "
        "`df_best_cases_all`; one deterministic representative "
        "per wavelength is stored in `df_best_cases`."
    )


# ============================================================
# SANITY CHECK
# ============================================================

expected_wavelengths = sorted(
    df_colour_selection[
        TARGET_WAVELENGTH_COLUMN
    ].unique()
)

selected_wavelengths = sorted(
    df_best_cases[
        TARGET_WAVELENGTH_COLUMN
    ].unique()
)

if expected_wavelengths != selected_wavelengths:
    raise RuntimeError(
        "Best-case selection did not produce one representative "
        "for every target wavelength."
    )

print("\nReusable notebook objects created:")
print("  df_colour_selection")
print("  df_best_cases_all")
print("  df_best_cases")
print("  best_cases_by_wavelength")
print("  best_case_target_wavelengths")

# In[191]:


import numpy as np
import matplotlib.pyplot as plt

# Parameter values
k_values = np.sort(df_analysis['k'].unique())
nA_values = np.sort(df_analysis['n_A'].unique())
layer_values = np.sort(df_analysis['n_layers'].unique())

# Marker for each absorber location
absorber_locations = np.sort(df_analysis["absorber_location"].unique())

marker_dict = {
    "core": "o",
    "low_index": "s",
    "all_layers": "^",
    "outer_shell": "D",
    "none": "x"
}

# Small x-offset for each n_A
offsets = np.linspace(-15, 15, len(absorber_locations))
offset_dict = dict(zip(absorber_locations, offsets))

# Marker size for each k value
sizes = np.linspace(60, 180, len(k_values))[::-1]
size_dict = dict(zip(k_values, sizes))

plt.figure(figsize=(10, 6))

for k in k_values:
    # if k != 0.05:
    #     continue

    for nA in nA_values:
        # if nA == 1.5:
        #     continue

        for absorber_location in absorber_locations:
            # if absorber_location != "outer_shell": # 'core' 'low_index' 'all_layers' 'outer_shell'
            #     continue

            subset = df_analysis[
                (df_analysis['collection_na'] == 1.0) &
                (df_analysis['k'] == k) &
                (df_analysis['n_A'] == nA) &
                (df_analysis['absorber_location'] == absorber_location)
            ]

            if subset.empty:
                continue

            hue = np.mod(np.degrees(np.arctan2(subset['b'], subset['a'])), 360)
            # Shift hsv colour map
            hue_shift = 290  # degrees

            hue_display = (hue + hue_shift) % 360

            plt.scatter(
                subset['design_peak_wavelength_nm'] + offset_dict[absorber_location],
                hue,
                # c=hue_display,
                c=subset['hex'],
                # cmap='hsv',
                # marker=marker_dict[subset['absorber_location'].iloc[0]],
                marker=marker_dict[absorber_location],
                s=size_dict[k],
                edgecolor=None if marker_dict[absorber_location] == 'x' else 'k',
                label=f'k={k}, nA={nA}, absorber_location={absorber_location}',
                alpha=1
            )

import matplotlib as mpl

ax = plt.gca()      # or better: fig, ax = plt.subplots(...)

norm = mpl.colors.Normalize(vmin=0, vmax=360)
sm = mpl.cm.ScalarMappable(norm=norm, cmap='hsv')
sm.set_array([])
ticks = np.linspace(0, 360, 7)
cbar = plt.colorbar(sm, ax=ax, ticks=ticks)
cbar.set_label("Hue angle [degrees] (hsv colormap)")

plt.xticks(df_analysis['design_peak_wavelength_nm'].unique())
plt.xlabel('Design Peak Wavelength (nm)')
plt.yticks(np.arange(0, 361, 60))
plt.ylabel('Hue Angle [degrees] (CIE a*b* plane)')
plt.title('Hue Angle vs Design Peak Wavelength')
from matplotlib.lines import Line2D

size_handles = []

for k in k_values:
    size_handles.append(
        Line2D(
            [0], [0],
            marker='o',
            color='w',
            markerfacecolor='gray',
            markeredgecolor='k',
            markersize=np.sqrt(size_dict[k]),
            label=f'k = {k:.4f}'
        )
    )

shape_handles = []

for loc in absorber_locations:
    shape_handles.append(
        Line2D(
            [0], [0],
            marker=marker_dict.get(loc, 'x'),
            linestyle='None',
            color='k',
            markersize=8,
            label=loc.replace('_', ' ')
        )
    )

leg1 = plt.legend(
    handles=size_handles,
    title='Extinction k',
    loc='upper right'
)

plt.gca().add_artist(leg1)

plt.legend(
    handles=shape_handles,
    title='Absorber location',
    loc='lower right'
)

plt.text(
    0.02, 0.2,
    "Horizontal offset indicates absorber location\n"
    "Marker size indicates $k$\n"
    "Marker shape indicates absorber location",
    transform=plt.gca().transAxes,
    va='top',
    fontsize=10,
    bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray')
)

# possible target hues
target_hues = np.array([30, 60, 90, 150, 210, 270, 315])
# corresponding target wavelengths
target_wavelengths = np.array([760, 680, 620, 540, 500, 440, 380])
for target_hue in target_hues:
    plt.axhline(
        y=target_hue,
        color='gray',
        linestyle='--',
        linewidth=0.8,
        alpha=0.5
    )

plt.show()

# In[205]:


# CELL 5 — MAIN PLOT
# PART A — ROSCH–MACADAM COLOUR SOLID
# ============================================================

ROSCH_COLOUR_VIEW = "both"   # "raw", "display", or "both"

generated_figures = make_rosch_macadam_figures(
    df_plot,
    colour_view=ROSCH_COLOUR_VIEW,
    max_overlay_points=2000,
    fixed_marker_size=4.0,
    marker_symbol_variable="peak_wavelength_nm",   # change freely
    marker_symbol_style="solid_and_line",  # "solid_only" or "solid_and_line"
    marker_edge_color=None,           # or None
    marker_edge_width=0.4,
    optimal_marker_size=4.8,
    optimal_marker_edge_color=None,
    optimal_marker_edge_width=0.7,
    save_html=True,
    output_dir=OUTPUT_DIR,
    show_figures=True,
)

print("Generated Rosch–MacAdam figure keys:", list(generated_figures.keys()))

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

group_cols = [
    "absorber_location",
    "design_peak_wavelength_nm",
]

idx = (
    df_analysis
    .groupby(group_cols)["eta_C"]
    .idxmax()
)

df_best = df_analysis.loc[idx].copy()

locations = df_best["absorber_location"].unique()

wavelengths = sorted(df_best["design_peak_wavelength_nm"].unique())

fig, ax = plt.subplots(
    figsize=(3.0 * len(wavelengths),
             2.3 * len(locations))
)

for i, loc in enumerate(locations):
    for j, wl in enumerate(wavelengths):

        row = df_best[
            (df_best["absorber_location"] == loc)
            & (df_best["design_peak_wavelength_nm"] == wl)
        ]

        if row.empty:
            continue

        row = row.iloc[0]

        rect = Rectangle(
            (j, len(locations)-1-i),
            1,
            1,
            facecolor=row["hex"],
            edgecolor="black",
        )

        ax.add_patch(rect)

        if loc == "core":
            txt = (
                f"ηC={100*row['eta_C']:.2f}%\n"
                f"ηL={100*row['eta_L']:.2f}%\n"
                f"nA={row['n_A']:.2f}\n"
                f"k={row['k']:.3f}\n"
                f"d core factor={row['core_thickness_factor']:.1f}\n\n"
                f"n_layers={int(row['n_layers'])}"
            )
        elif loc == "outer_shell":
            txt = (
                f"ηC={100*row['eta_C']:.2f}%\n"
                f"ηL={100*row['eta_L']:.2f}%\n"
                f"nA={row['n_A']:.2f}\n"
                f"k={row['k']:.3f}\n"
                f"d shell={row['outer_shell_thickness_nm']:.0f} nm\n"
                f"n_layers={int(row['n_layers'])}"
            )
        else:
            txt = (
                f"ηC={100*row['eta_C']:.2f}%\n"
                f"ηL={100*row['eta_L']:.2f}%\n"
                f"nA={row['n_A']:.2f}\n"
                f"k={row['k']:.3f}\n"
                f"n_layers={int(row['n_layers'])}\n"
            )

        ax.text(
            j + 0.5,
            len(locations)-1-i + 0.5,
            txt,
            ha="center",
            va="center",
            fontsize=9,
            color="black",
        )

ax.set_xlim(0, len(wavelengths))
ax.set_ylim(0, len(locations))

ax.set_xticks(np.arange(len(wavelengths)) + 0.5)
ax.set_xticklabels([f"{w:.0f} nm" for w in wavelengths])

ax.set_yticks(np.arange(len(locations)) + 0.5)
ax.set_yticklabels(locations[::-1])

ax.set_title("Best ηC case for each absorber location and target wavelength")

plt.tight_layout()
plt.show()

# In[194]:


import seaborn as sns
import matplotlib.pyplot as plt

heatmap_data = (
    df_best
    .pivot(
        index="absorber_location",
        columns="design_peak_wavelength_nm",
        values="eta_C",
    )
)

plt.figure(figsize=(7,4))

sns.heatmap(
    100 * heatmap_data,
    annot=True,
    fmt=".0f",
    cmap="viridis",
    cbar_kws={"label": "ηC [%]"},
)

plt.xlabel("Design wavelength (nm)")
plt.ylabel("Absorber location")
plt.title("Maximum chroma achieved by each absorber location")

plt.tight_layout()
plt.show()


# In[214]:


# CHROMA-LIGHTNESS TRADE-OFF WITH BEST d_opt CASES HIGHLIGHTED

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ============================================================
# PLOT SETTINGS
# ============================================================

marker_dict = {
    "core": "o",
    "low_index": "s",
    "all_layers": "^",
    "outer_shell": "D",
    "none": "x",
}

location_labels = {
    "core": "Core",
    "low_index": "Low-index layers",
    "all_layers": "All layers",
    "outer_shell": "Outer shell",
    "none": "No absorber",
}

POINT_ALPHA = 0.70
POINT_EDGE_WIDTH = 0.35

BEST_OUTER_SIZE = 300
BEST_INNER_SIZE = 190
BEST_STAR_SIZE = 85


# ============================================================
# VALIDATE INPUTS
# ============================================================

required_analysis_columns = {
    "design_peak_wavelength_nm",
    "absorber_location",
    "eta_C",
    "eta_L",
    "k",
    "hex",
}

required_best_columns = {
    "design_peak_wavelength_nm",
    "absorber_location",
    "eta_C",
    "eta_L",
    "hex",
    "d_opt",
}

missing_analysis = (
    required_analysis_columns - set(df_analysis.columns)
)

missing_best = (
    required_best_columns - set(df_best_cases.columns)
)

if missing_analysis:
    raise KeyError(
        "df_analysis is missing required columns: "
        f"{sorted(missing_analysis)}"
    )

if missing_best:
    raise KeyError(
        "df_best_cases is missing required columns: "
        f"{sorted(missing_best)}"
    )


# ============================================================
# PREPARE MARKER SIZES FOR k
# ============================================================

k_values = np.sort(
    pd.to_numeric(
        df_analysis["k"],
        errors="coerce",
    )
    .dropna()
    .unique()
)

if len(k_values) == 0:
    raise ValueError("No valid k values are available.")

if len(k_values) == 1:
    k_sizes = np.array([100.0])
else:
    # Marker area varies with log(k) for absorbing cases.
    positive_k = k_values[k_values > 0]

    if len(positive_k) > 1:
        log_min = np.log10(positive_k.min())
        log_max = np.log10(positive_k.max())

        k_sizes = []

        for k_value in k_values:
            if k_value <= 0:
                size = 45.0
            else:
                fraction = (
                    np.log10(k_value) - log_min
                ) / (
                    log_max - log_min
                )

                size = 55.0 + fraction * 195.0

            k_sizes.append(size)

        k_sizes = np.asarray(k_sizes)

    else:
        k_sizes = np.linspace(
            50.0,
            250.0,
            len(k_values),
        )

size_dict = dict(
    zip(
        k_values,
        k_sizes,
    )
)


def marker_size_for_k(k_value):
    """
    Return the marker area for the nearest available k value.
    This avoids floating-point dictionary lookup problems.
    """
    k_value = float(k_value)

    nearest_index = int(
        np.argmin(
            np.abs(k_values - k_value)
        )
    )

    return float(
        k_sizes[nearest_index]
    )


# ============================================================
# CREATE ONE PANEL PER TARGET WAVELENGTH
# ============================================================

target_wavelengths = sorted(
    pd.to_numeric(
        df_analysis["design_peak_wavelength_nm"],
        errors="coerce",
    )
    .dropna()
    .unique()
)

n_panels = len(target_wavelengths)

if n_panels == 0:
    raise ValueError(
        "No target wavelengths are available."
    )

fig, axes = plt.subplots(
    1,
    n_panels,
    figsize=(3.2 * n_panels, 5.2),
    sharex=True,
    sharey=True,
)

axes = np.atleast_1d(axes)


# ============================================================
# PLOT ALL CASES AND HIGHLIGHT THE SELECTED CASE
# ============================================================

for ax, wavelength_nm in zip(
    axes,
    target_wavelengths,
):

    subset_wavelength = df_analysis.loc[
        np.isclose(
            pd.to_numeric(
                df_analysis[
                    "design_peak_wavelength_nm"
                ],
                errors="coerce",
            ),
            float(wavelength_nm),
        )
    ].copy()

    # --------------------------------------------------------
    # Plot all sweep cases
    # --------------------------------------------------------
    for absorber_location in marker_dict:

        subset_location = subset_wavelength.loc[
            subset_wavelength[
                "absorber_location"
            ] == absorber_location
        ]

        if subset_location.empty:
            continue

        marker = marker_dict[absorber_location]

        marker_sizes = [
            marker_size_for_k(k_value)
            for k_value in subset_location["k"]
        ]

        scatter_kwargs = {
            "x": 100.0 * subset_location["eta_L"],
            "y": 100.0 * subset_location["eta_C"],
            "c": subset_location["hex"],
            "s": marker_sizes,
            "marker": marker,
            "alpha": POINT_ALPHA,
            "zorder": 2,
        }

        if marker == "x":
            scatter_kwargs.update(
                {
                    "linewidths": 1.0,
                }
            )
        else:
            scatter_kwargs.update(
                {
                    "edgecolors": "black",
                    "linewidths": POINT_EDGE_WIDTH,
                }
            )

        ax.scatter(**scatter_kwargs)

    # --------------------------------------------------------
    # Find the selected minimum-d_opt case
    # --------------------------------------------------------
    selected_case = df_best_cases.loc[
        np.isclose(
            pd.to_numeric(
                df_best_cases[
                    "design_peak_wavelength_nm"
                ],
                errors="coerce",
            ),
            float(wavelength_nm),
        )
    ]

    if len(selected_case) != 1:
        raise ValueError(
            f"Expected exactly one selected best case for "
            f"{wavelength_nm:.0f} nm, but found "
            f"{len(selected_case)}."
        )

    selected_case = selected_case.iloc[0]

    best_x = 100.0 * float(
        selected_case["eta_L"]
    )

    best_y = 100.0 * float(
        selected_case["eta_C"]
    )

    best_hex = str(
        selected_case["hex"]
    )

    best_location = str(
        selected_case["absorber_location"]
    )

    best_marker = marker_dict.get(
        best_location,
        "o",
    )

    # Large white halo
    ax.scatter(
        [best_x],
        [best_y],
        s=BEST_OUTER_SIZE,
        marker=best_marker,
        facecolors="none",
        edgecolors="white",
        linewidths=5.0,
        zorder=7,
    )

    # Black outline
    ax.scatter(
        [best_x],
        [best_y],
        s=BEST_INNER_SIZE,
        marker=best_marker,
        facecolors=best_hex,
        edgecolors="black",
        linewidths=2.2,
        zorder=8,
    )

    # Central star indicates the final selected case
    ax.scatter(
        [best_x],
        [best_y],
        s=BEST_STAR_SIZE,
        marker="*",
        facecolors="white",
        edgecolors="black",
        linewidths=0.8,
        zorder=9,
    )

    # Annotation
    annotation_lines = [
        f"dopt={float(selected_case['d_opt']):.3f}",
        f"ηC={best_y:.1f}%",
        f"ηL={best_x:.1f}%",
        location_labels.get(
            best_location,
            best_location,
        ),
    ]

    if "n_A" in selected_case.index:
        annotation_lines.append(
            f"nA={float(selected_case['n_A']):.2f}"
        )

    if "k" in selected_case.index:
        annotation_lines.append(
            f"k={float(selected_case['k']):.3g}"
        )

    if "n_layers" in selected_case.index:
        annotation_lines.append(
            f"N={int(selected_case['n_layers'])}"
        )

    ax.annotate(
        "\n".join(annotation_lines),
        xy=(best_x, best_y),
        xytext=(8, -12),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=7.5,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "black",
            "alpha": 0.88,
        },
        zorder=10,
    )

    # Ideal normalized point
    ax.scatter(
        [100.0],
        [100.0],
        marker="+",
        s=110,
        color="red",
        linewidths=1.8,
        zorder=6,
    )

    ax.set_title(
        f"{wavelength_nm:.0f} nm"
    )

    ax.set_xlabel("ηL [%]")
    ax.grid(alpha=0.3)


axes[0].set_ylabel("ηC [%]")


# ============================================================
# AXIS RANGES
# ============================================================

all_eta_L_pct = (
    100.0
    * pd.to_numeric(
        df_analysis["eta_L"],
        errors="coerce",
    )
)

all_eta_C_pct = (
    100.0
    * pd.to_numeric(
        df_analysis["eta_C"],
        errors="coerce",
    )
)

x_max = max(
    105.0,
    float(
        np.nanpercentile(
            all_eta_L_pct,
            99.8,
        )
    )
    * 1.05,
)

y_max = max(
    105.0,
    float(
        np.nanpercentile(
            all_eta_C_pct,
            99.8,
        )
    )
    * 1.05,
)

for ax in axes:
    ax.set_xlim(0.0, x_max)
    ax.set_ylim(0.0, y_max)


# ============================================================
# LEGENDS
# ============================================================

location_handles = []

for location, marker in marker_dict.items():

    if location not in set(
        df_analysis["absorber_location"]
    ):
        continue

    if marker == "x":
        handle = Line2D(
            [0],
            [0],
            marker=marker,
            color="black",
            linestyle="None",
            markersize=8,
            label=location_labels.get(
                location,
                location,
            ),
        )
    else:
        handle = Line2D(
            [0],
            [0],
            marker=marker,
            color="black",
            markerfacecolor="gray",
            markeredgecolor="black",
            linestyle="None",
            markersize=8,
            label=location_labels.get(
                location,
                location,
            ),
        )

    location_handles.append(handle)


best_case_handle = Line2D(
    [0],
    [0],
    marker="*",
    color="black",
    markerfacecolor="white",
    markeredgecolor="black",
    linestyle="None",
    markersize=11,
    label="Selected minimum-dopt case",
)

ideal_handle = Line2D(
    [0],
    [0],
    marker="+",
    color="red",
    linestyle="None",
    markersize=10,
    markeredgewidth=1.8,
    label="Ideal ηL=ηC=100%",
)

fig.legend(
    handles=(
        location_handles
        + [
            best_case_handle,
            ideal_handle,
        ]
    ),
    loc="lower center",
    bbox_to_anchor=(0.5, -0.06),
    ncol=3,
    title="Absorber placement and selected cases",
)


fig.suptitle(
    "Chroma-lightness trade-off with minimum-dopt cases highlighted",
    y=1.02,
)

fig.tight_layout()
plt.show()

# In[196]:


# ============================================================
# PART B — COLOUR STRIP OPTION 1
# Best η_C case for each design wavelength
# ============================================================

fixed_filters_best_strip = {
    # "collection_angle_deg": 90.0,
    # "n_layers": 100,
    # "n_A": 2.1,
    # "k": 0.05,
}

df_best_strip = select_best_etaC_by_wavelength(
    df_plot,   # use df_plot if you want hex_display
    fixed_filters=fixed_filters_best_strip,
    x="design_peak_wavelength_nm",
    score_col="eta_C",
)

plot_best_etaC_colour_strip(
    df_best_strip,
    x="design_peak_wavelength_nm",
    colour_hex_col="hex",   # or "hex"
    annotate_cols=["eta_C", "eta_L"],
    title="Best η_C case for each design wavelength\n(fixed k = 0.05)",
)

plt.show()

# Print properties of the best strip dataframe for reference:
df_best_strip_print = df_best_strip.copy().sort_values("design_peak_wavelength_nm")

if "eta_C" in df_best_strip_print.columns:
    df_best_strip_print["eta_C_pct"] = (100 * df_best_strip_print["eta_C"]).round(0).astype(int)

if "eta_L" in df_best_strip_print.columns:
    df_best_strip_print["eta_L_pct"] = (100 * df_best_strip_print["eta_L"]).round(0).astype(int)

cols_to_show = [
    "design_peak_wavelength_nm",
    "case_index",
    "eta_C_pct",
    "eta_L_pct",
    "n_layers",
    "n_A",
    "k",
    "collection_angle_deg",
    "outer_layer",
    "hex_display",
]

cols_to_show = [c for c in cols_to_show if c in df_best_strip_print.columns]

display(df_best_strip_print[cols_to_show])

# In[274]:


print(df_analysis.columns)
# print(df_plot["k"].unique())
# print(df_plot["collection_na"].unique())
# print(df_plot["n_A"].unique())
# print(df_plot["design_peak_wavelength_nm"].unique())

# In[198]:


# # Define cases considered invalid
# wrong_case_1 = (
#     (df_all["design_peak_wavelength_nm"] == 540.0)
#     & (df_all["absorber_location"] == "low_index")
#     & np.isclose(df_all["n_A"].astype(float), 1.17666667)
#     & np.isclose(df_all["k"].astype(float), 0.001)
#     & (df_all["n_layers"] == 40)
# )

# wrong_case_2 = (
#     (df_all["design_peak_wavelength_nm"] == 540.0)
#     & (df_all["absorber_location"] == "low_index")
#     & np.isclose(df_all["n_A"].astype(float), 1.17666667)
#     & np.isclose(df_all["k"].astype(float), 0.00316228)
#     & (df_all["n_layers"] == 40)
# )

# wrong_mask = wrong_case_1 | wrong_case_2

# display(df_all.loc[wrong_mask])

# df_all_clean = df_all.loc[~wrong_mask].copy()

# In[ ]:


# ============================================================
# PART C — COLOUR STRIP OPTION 2
# Multi-row strip grid:
# rows = full range of one sweep parameter
# columns = design wavelength
# ============================================================


# ------------------------------------------------------------
# User choices
# ------------------------------------------------------------
row_variable = "absorber_location"   # e.g. "k", "n_layers", "collection_angle_deg", "n_A"
colour_hex_col = "hex"                  # or "hex_display"

fixed_filters_grid = {
    # "collection_angle_deg": 90,   # removed automatically if row_variable matches
    # "n_layers": 10,
    # "n_A": 1.51,
    # "k": 0.01,
    # "requested_diameter_nm": 15000.0,
    # "absorber_location": "low_index",  # or "outer_shell", "core", "low_index", "all_layers", "none"
    # "outer_shell_thickness_nm": 50.0,
    # "core_thickness_factor": 5.0
    "collection_na": 1.0
}

# ------------------------------------------------------------
# Prepare data + auto-generated metadata
# ------------------------------------------------------------
df_grid, grid_meta = prepare_strip_grid(
    df_analysis,
    df_plot,
    row_variable=row_variable,
    fixed_filters=fixed_filters_grid,
    colour_hex_col=colour_hex_col,
    x="design_peak_wavelength_nm",
    score_col="eta_C",
    select_best_within_cell=True,
)

# Optional info messages
if grid_meta["removed_filter"] is not None:
    print(
        f"Removed '{row_variable}={grid_meta['removed_filter']}' from fixed_filters "
        f"because the row variable must vary across rows."
    )

print("Fixed filters used:")
print(grid_meta["fixed_filters_clean"])
print(f"Row variable : {grid_meta['row_variable']}")
print(f"Colour column: {grid_meta['colour_hex_col']}")
print(f"Source df    : {grid_meta['df_source_name']}")
print(f"Selected tile cases: {len(df_grid)}")

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
# fig, ax = plot_colour_strip_grid(
#     df_grid,
#     row_variable=grid_meta["row_variable"],
#     x="design_peak_wavelength_nm",
#     colour_hex_col=grid_meta["colour_hex_col"],
#     annotate_cols=["eta_C", "eta_L"],
#     row_label_fmt=grid_meta["row_label_fmt"],
#     title=grid_meta["title"],
#     overlay_line_cols=["eta_C", "eta_L"],
#     overlay_line_colors={"eta_C": "black", "eta_L": "white"},
#     overlay_line_scale="global_shared",
#     overlay_reference_values=[0.0, 1.0],
# )
fig, ax = plot_colour_strip_grid(
    df_grid,
    row_variable=grid_meta["row_variable"],
    x="design_peak_wavelength_nm",
    colour_hex_col=grid_meta["colour_hex_col"],
    annotate_cols=["eta_C", "eta_L"],
    row_label_fmt=grid_meta["row_label_fmt"],
    title=grid_meta["title"],
    overlay_line_cols=None,
    overlay_line_colors=None,
    overlay_line_scale="global_shared",
    overlay_reference_values=[0.0, 1.0],
)

# Set figure size to 33.2 cm × 14.8 cm
fig.set_size_inches(33.2 / 2.54, 14.8 / 2.54)

import os

path = "/mnt/c/Users/SchwarzN/OneDrive - Université de Fribourg/Institution/P1_BraggSpericalPigments/Simulation/MIE_Computation/Scattnlay_BraggOnion/outputs/rerun_primary_sweep_01"
plt.savefig(os.path.join(path, "ColourStrip_absorber_location.png"), dpi = 600, transparent = True)
plt.show()

# ------------------------------------------------------------
# Compact summary table of the plotted tiles
# ------------------------------------------------------------
df_grid_print = summarize_strip_grid_cases(
    df_grid,
    row_variable=grid_meta["row_variable"],
    fixed_filters=grid_meta["fixed_filters_clean"],
    colour_hex_col=grid_meta["colour_hex_col"],
    x="design_peak_wavelength_nm",
)

display(df_grid_print.sort_values("design_peak_wavelength_nm"))

# In[200]:


from matplotlib.patches import Circle

marker_dict = {
    "380.0": "o",
    "540.0": "s",
    "760.0": "^"
}
print(df_analysis["design_peak_wavelength_nm"].unique())
# (1.2) CIELAB a*–b* plane
plt.figure(figsize=(6,6))
for design_peak_wavelength in df_analysis["design_peak_wavelength_nm"].unique():
    subset = df_analysis[df_analysis["design_peak_wavelength_nm"] == design_peak_wavelength]
    plt.scatter(
        subset["a"],
        subset["b"],
        c=subset["hex"],
        s=30,
        marker=marker_dict.get(str(design_peak_wavelength), 'x'),
        edgecolor=None if marker_dict.get(str(design_peak_wavelength), 'x') == 'x' else 'k',
        label=design_peak_wavelength
    )
# plt.scatter(df_analysis["a"], df_analysis["b"], c=df_analysis["hex"], s=15, marker=marker_dict.get(df_analysis["absorber_location"].iloc[0], 'x'))
plt.axhline(0, color="gray")
plt.axvline(0, color="gray")
plt.xlabel("a*")
plt.ylabel("b*")
plt.title("CIELAB a*–b* plane")
plt.axis("equal")
# add radial rings to indicate hue angles and chroma levels
hue_angles = np.arange(0, 360, 30)
chroma_levels = np.arange(0.0, 75.0, 20.0)
for hue in hue_angles:
    plt.plot(
        chroma_levels * np.cos(np.radians(hue)),
        chroma_levels * np.sin(np.radians(hue)),
        color="k",
        linestyle="--",
        linewidth=0.5,
        alpha=0.5,
    )
# add rings for chroma levels
for chroma in chroma_levels:
    circle = Circle((0, 0), float(chroma), color="gray", fill=False, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.gca().add_artist(circle)

# plt.xlim(-100, 100)
# plt.ylim(-100, 100)
plt.show()

# Case acceptance base on hue difference to best case

# In[219]:


# CIELAB a*–b* PLANE WITH HUE SECTORS CENTERED ON df_best_cases
#
# The center of each hue-acceptance sector is defined by the selected
# minimum-d_opt case in df_best_cases for that target wavelength.
#
# All sweep cases come from df_analysis.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from matplotlib.patches import Circle, Wedge
from matplotlib.lines import Line2D


# ============================================================
# USER OPTIONS
# ============================================================

HUE_HALF_WIDTH_DEG = 20.0

# Hue is unstable for nearly neutral colours.
MIN_CHROMA_FOR_HUE_CHECK = 5.0

# overlap fraction = overlap width / complete sector width
MAX_ALLOWED_OVERLAP_FRACTION = 0.25

# If True, overlapping targets are removed automatically.
# The target with the smaller d_opt is retained.
AUTO_OMIT_OVERLAPPING_TARGETS = False

# Raw or display-normalized colour for plotting.
COLOUR_HEX_COL = "hex"  # alternatively "hex_display"

# Plot appearance
ACCEPTED_ALPHA = 0.75
ACCEPTED_SIZE = 32
OUTLIER_SIZE = 65
LOW_CHROMA_SIZE = 24

BEST_OUTER_SIZE = 220
BEST_INNER_SIZE = 145
BEST_STAR_SIZE = 65

SECTOR_ALPHA = 0.12
SECTOR_EDGE_ALPHA = 0.75


# ============================================================
# VALIDATE INPUT DATA
# ============================================================

required_analysis_columns = {
    "design_peak_wavelength_nm",
    "a",
    "b",
    COLOUR_HEX_COL,
}

required_best_columns = {
    "design_peak_wavelength_nm",
    "a",
    "b",
    "d_opt",
    COLOUR_HEX_COL,
}

missing_analysis_columns = (
    required_analysis_columns
    - set(df_analysis.columns)
)

missing_best_columns = (
    required_best_columns
    - set(df_best_cases.columns)
)

if missing_analysis_columns:
    raise KeyError(
        "df_analysis is missing required columns: "
        f"{sorted(missing_analysis_columns)}"
    )

if missing_best_columns:
    raise KeyError(
        "df_best_cases is missing required columns: "
        f"{sorted(missing_best_columns)}"
    )


# ============================================================
# PREPARE THE COMPLETE SWEEP DATA
# ============================================================

df_hue = df_analysis.copy()

for column in [
    "design_peak_wavelength_nm",
    "a",
    "b",
]:
    df_hue[column] = pd.to_numeric(
        df_hue[column],
        errors="coerce",
    )

df_hue = df_hue.dropna(
    subset=[
        "design_peak_wavelength_nm",
        "a",
        "b",
    ]
).copy()

df_hue["C_calculated"] = np.hypot(
    df_hue["a"],
    df_hue["b"],
)

df_hue["hue_calculated_deg"] = np.mod(
    np.degrees(
        np.arctan2(
            df_hue["b"],
            df_hue["a"],
        )
    ),
    360.0,
)


# ============================================================
# PREPARE THE SELECTED BEST CASES
# ============================================================

df_best_hue = df_best_cases.copy()

for column in [
    "design_peak_wavelength_nm",
    "a",
    "b",
    "d_opt",
]:
    df_best_hue[column] = pd.to_numeric(
        df_best_hue[column],
        errors="coerce",
    )

df_best_hue = df_best_hue.dropna(
    subset=[
        "design_peak_wavelength_nm",
        "a",
        "b",
        "d_opt",
    ]
).copy()

df_best_hue["C_calculated"] = np.hypot(
    df_best_hue["a"],
    df_best_hue["b"],
)

df_best_hue["hue_calculated_deg"] = np.mod(
    np.degrees(
        np.arctan2(
            df_best_hue["b"],
            df_best_hue["a"],
        )
    ),
    360.0,
)

duplicate_best_wavelengths = (
    df_best_hue["design_peak_wavelength_nm"]
    .duplicated(keep=False)
)

if duplicate_best_wavelengths.any():
    duplicated_values = sorted(
        df_best_hue.loc[
            duplicate_best_wavelengths,
            "design_peak_wavelength_nm",
        ].unique()
    )

    raise ValueError(
        "df_best_cases must contain exactly one selected case "
        "per target wavelength. Duplicate wavelengths: "
        f"{duplicated_values}"
    )


# ============================================================
# CIRCULAR-ANGLE HELPERS
# ============================================================

def signed_circular_difference_deg(
    angle_deg,
    center_deg,
):
    """
    Signed shortest angular difference in [-180°, 180°).
    Both inputs may be scalars or array-like.
    """
    angle = np.asarray(
        angle_deg,
        dtype=float,
    )

    center = np.asarray(
        center_deg,
        dtype=float,
    )

    return (
        angle - center + 180.0
    ) % 360.0 - 180.0


def circular_distance_deg(
    angle_1_deg,
    angle_2_deg,
):
    """
    Unsigned shortest angular separation in [0°, 180°].
    """
    return abs(
        float(
            signed_circular_difference_deg(
                angle_1_deg,
                angle_2_deg,
            )
        )
    )


# ============================================================
# BUILD HUE ANCHORS FROM df_best_cases
# ============================================================

anchor_records = []

for _, best_row in df_best_hue.iterrows():

    anchor_records.append(
        {
            "design_peak_wavelength_nm": float(
                best_row[
                    "design_peak_wavelength_nm"
                ]
            ),
            "anchor_hue_deg": float(
                best_row["hue_calculated_deg"]
            ),
            "anchor_chroma": float(
                best_row["C_calculated"]
            ),
            "anchor_d_opt": float(
                best_row["d_opt"]
            ),
            "anchor_eta_C": (
                float(best_row["eta_C"])
                if "eta_C" in best_row.index
                else np.nan
            ),
            "anchor_eta_L": (
                float(best_row["eta_L"])
                if "eta_L" in best_row.index
                else np.nan
            ),
            "anchor_a": float(best_row["a"]),
            "anchor_b": float(best_row["b"]),
            "anchor_hex": str(
                best_row[COLOUR_HEX_COL]
            ),
            "anchor_absorber_location": str(
                best_row["absorber_location"]
            )
            if "absorber_location" in best_row.index
            else "",
            "anchor_n_A": (
                float(best_row["n_A"])
                if "n_A" in best_row.index
                else np.nan
            ),
            "anchor_k": (
                float(best_row["k"])
                if "k" in best_row.index
                else np.nan
            ),
            "anchor_n_layers": (
                int(best_row["n_layers"])
                if "n_layers" in best_row.index
                else pd.NA
            ),
        }
    )

df_hue_anchors = (
    pd.DataFrame(anchor_records)
    .sort_values("design_peak_wavelength_nm")
    .reset_index(drop=True)
)

if df_hue_anchors.empty:
    raise ValueError(
        "No valid hue anchors could be generated "
        "from df_best_cases."
    )


anchor_hue_by_wavelength = dict(
    zip(
        df_hue_anchors[
            "design_peak_wavelength_nm"
        ],
        df_hue_anchors[
            "anchor_hue_deg"
        ],
    )
)

anchor_d_opt_by_wavelength = dict(
    zip(
        df_hue_anchors[
            "design_peak_wavelength_nm"
        ],
        df_hue_anchors[
            "anchor_d_opt"
        ],
    )
)


# ============================================================
# CONFIRM THAT ALL ANCHOR WAVELENGTHS EXIST IN df_analysis
# ============================================================

available_analysis_wavelengths = set(
    df_hue[
        "design_peak_wavelength_nm"
    ].unique()
)

anchor_wavelengths = set(
    df_hue_anchors[
        "design_peak_wavelength_nm"
    ].unique()
)

missing_wavelengths = (
    anchor_wavelengths
    - available_analysis_wavelengths
)

if missing_wavelengths:
    raise ValueError(
        "The following df_best_cases wavelengths are not "
        "present in df_analysis: "
        f"{sorted(missing_wavelengths)}"
    )


# ============================================================
# DETECT OVERLAPPING TARGET SECTORS
# ============================================================

sector_full_width_deg = (
    2.0 * HUE_HALF_WIDTH_DEG
)

target_wavelengths = (
    df_hue_anchors[
        "design_peak_wavelength_nm"
    ]
    .to_numpy(dtype=float)
)

overlap_records = []

for i in range(len(target_wavelengths)):
    for j in range(
        i + 1,
        len(target_wavelengths),
    ):

        wavelength_1 = float(
            target_wavelengths[i]
        )

        wavelength_2 = float(
            target_wavelengths[j]
        )

        center_1 = (
            anchor_hue_by_wavelength[
                wavelength_1
            ]
        )

        center_2 = (
            anchor_hue_by_wavelength[
                wavelength_2
            ]
        )

        center_distance = (
            circular_distance_deg(
                center_1,
                center_2,
            )
        )

        overlap_width = max(
            0.0,
            sector_full_width_deg
            - center_distance,
        )

        overlap_fraction = (
            overlap_width
            / sector_full_width_deg
            if sector_full_width_deg > 0
            else 0.0
        )

        overlap_records.append(
            {
                "target_wavelength_1_nm": wavelength_1,
                "target_wavelength_2_nm": wavelength_2,
                "center_hue_1_deg": center_1,
                "center_hue_2_deg": center_2,
                "center_distance_deg": center_distance,
                "overlap_width_deg": overlap_width,
                "overlap_fraction": overlap_fraction,
                "excessive_overlap": (
                    overlap_fraction
                    > MAX_ALLOWED_OVERLAP_FRACTION
                ),
            }
        )

df_sector_overlaps = pd.DataFrame(
    overlap_records
)


# ============================================================
# OPTIONAL AUTOMATIC REMOVAL OF OVERLAPPING TARGETS
# ============================================================

selected_target_wavelengths = sorted(
    target_wavelengths.tolist()
)

omitted_target_wavelengths = []

if AUTO_OMIT_OVERLAPPING_TARGETS:

    # Smaller d_opt has priority.
    candidate_order = sorted(
        selected_target_wavelengths,
        key=lambda wavelength: (
            anchor_d_opt_by_wavelength[
                wavelength
            ]
        ),
    )

    retained = []

    for candidate in candidate_order:

        candidate_center = (
            anchor_hue_by_wavelength[
                candidate
            ]
        )

        has_conflict = False

        for retained_wavelength in retained:

            retained_center = (
                anchor_hue_by_wavelength[
                    retained_wavelength
                ]
            )

            separation = circular_distance_deg(
                candidate_center,
                retained_center,
            )

            overlap_width = max(
                0.0,
                sector_full_width_deg
                - separation,
            )

            overlap_fraction = (
                overlap_width
                / sector_full_width_deg
            )

            if (
                overlap_fraction
                > MAX_ALLOWED_OVERLAP_FRACTION
            ):
                has_conflict = True
                break

        if has_conflict:
            omitted_target_wavelengths.append(
                candidate
            )
        else:
            retained.append(candidate)

    selected_target_wavelengths = sorted(
        retained
    )


# ============================================================
# CLASSIFY EVERY SWEEP POINT RELATIVE TO df_best_cases
# ============================================================

df_hue["anchor_hue_deg"] = (
    df_hue[
        "design_peak_wavelength_nm"
    ].map(
        anchor_hue_by_wavelength
    )
)

df_hue = df_hue.dropna(
    subset=["anchor_hue_deg"]
).copy()

df_hue["hue_difference_deg"] = (
    signed_circular_difference_deg(
        df_hue[
            "hue_calculated_deg"
        ],
        df_hue[
            "anchor_hue_deg"
        ],
    )
)

df_hue["hue_check_applicable"] = (
    df_hue["C_calculated"]
    >= MIN_CHROMA_FOR_HUE_CHECK
)

df_hue["within_hue_acceptance"] = (
    np.abs(
        df_hue["hue_difference_deg"]
    )
    <= HUE_HALF_WIDTH_DEG
)

df_hue["potential_hue_outlier"] = (
    df_hue["hue_check_applicable"]
    & ~df_hue[
        "within_hue_acceptance"
    ]
)

df_hue["target_retained"] = (
    df_hue[
        "design_peak_wavelength_nm"
    ].isin(
        selected_target_wavelengths
    )
)


# ============================================================
# MARKERS AND SECTOR COLOURS
# ============================================================

marker_cycle = [
    "o",
    "s",
    "^",
    "D",
    "v",
    "P",
    "X",
    "*",
]

marker_dict = {
    wavelength: marker_cycle[
        index % len(marker_cycle)
    ]
    for index, wavelength in enumerate(
        selected_target_wavelengths
    )
}

sector_cmap = mpl.colormaps["turbo"]

if len(selected_target_wavelengths) > 1:

    sector_norm = mpl.colors.Normalize(
        vmin=min(
            selected_target_wavelengths
        ),
        vmax=max(
            selected_target_wavelengths
        ),
    )

else:

    only_wavelength = (
        selected_target_wavelengths[0]
    )

    sector_norm = mpl.colors.Normalize(
        vmin=only_wavelength - 1.0,
        vmax=only_wavelength + 1.0,
    )


# ============================================================
# DETERMINE PLOT RADIUS
# ============================================================

finite_chroma = df_hue.loc[
    np.isfinite(
        df_hue["C_calculated"]
    ),
    "C_calculated",
].to_numpy(dtype=float)

anchor_chroma_values = (
    df_hue_anchors[
        "anchor_chroma"
    ].to_numpy(dtype=float)
)

combined_chroma = np.concatenate(
    [
        finite_chroma,
        anchor_chroma_values,
    ]
)

if combined_chroma.size == 0:
    plot_radius = 80.0
else:
    plot_radius = max(
        80.0,
        1.10
        * np.nanpercentile(
            combined_chroma,
            99.5,
        ),
    )

chroma_levels = np.arange(
    20.0,
    plot_radius + 20.0,
    20.0,
)


# ============================================================
# CREATE PLOT
# ============================================================

fig, ax = plt.subplots(
    figsize=(9, 9)
)


# ------------------------------------------------------------
# Draw target hue sectors centered on df_best_cases
# ------------------------------------------------------------

for wavelength in selected_target_wavelengths:

    center_hue = (
        anchor_hue_by_wavelength[
            wavelength
        ]
    )

    sector_colour = sector_cmap(
        sector_norm(wavelength)
    )

    wedge = Wedge(
        center=(0.0, 0.0),
        r=plot_radius,
        theta1=(
            center_hue
            - HUE_HALF_WIDTH_DEG
        ),
        theta2=(
            center_hue
            + HUE_HALF_WIDTH_DEG
        ),
        facecolor=sector_colour,
        edgecolor=sector_colour,
        linewidth=1.5,
        alpha=SECTOR_ALPHA,
        zorder=0,
    )

    ax.add_patch(wedge)

    center_radii = np.array(
        [0.0, plot_radius],
        dtype=float,
    )

    ax.plot(
        center_radii
        * np.cos(
            np.deg2rad(center_hue)
        ),
        center_radii
        * np.sin(
            np.deg2rad(center_hue)
        ),
        color=sector_colour,
        linewidth=1.5,
        alpha=SECTOR_EDGE_ALPHA,
        zorder=1,
    )


# ------------------------------------------------------------
# Draw hue spokes and chroma rings
# ------------------------------------------------------------

for hue_deg in np.arange(
    0.0,
    360.0,
    30.0,
):

    radii = np.array(
        [0.0, plot_radius],
        dtype=float,
    )

    ax.plot(
        radii
        * np.cos(
            np.deg2rad(hue_deg)
        ),
        radii
        * np.sin(
            np.deg2rad(hue_deg)
        ),
        color="gray",
        linestyle="--",
        linewidth=0.5,
        alpha=0.45,
        zorder=0,
    )

    label_radius = (
        1.03 * plot_radius
    )

    ax.text(
        label_radius
        * np.cos(
            np.deg2rad(hue_deg)
        ),
        label_radius
        * np.sin(
            np.deg2rad(hue_deg)
        ),
        f"{hue_deg:.0f}°",
        ha="center",
        va="center",
        fontsize=8,
        color="dimgray",
    )

for chroma in chroma_levels:

    circle = Circle(
        (0.0, 0.0),
        radius=float(chroma),
        fill=False,
        edgecolor="gray",
        linestyle="--",
        linewidth=0.5,
        alpha=0.45,
        zorder=0,
    )

    ax.add_patch(circle)

    ax.text(
        float(chroma),
        0.0,
        f"C*={chroma:.0f}",
        fontsize=7,
        color="dimgray",
        ha="left",
        va="bottom",
    )


# ------------------------------------------------------------
# Draw all sweep points
# ------------------------------------------------------------

for wavelength in selected_target_wavelengths:

    subset = df_hue.loc[
        np.isclose(
            df_hue[
                "design_peak_wavelength_nm"
            ],
            wavelength,
        )
    ]

    marker = marker_dict[wavelength]

    accepted = subset.loc[
        subset[
            "hue_check_applicable"
        ]
        & subset[
            "within_hue_acceptance"
        ]
    ]

    outliers = subset.loc[
        subset[
            "potential_hue_outlier"
        ]
    ]

    low_chroma = subset.loc[
        ~subset[
            "hue_check_applicable"
        ]
    ]

    if not accepted.empty:
        ax.scatter(
            accepted["a"],
            accepted["b"],
            c=accepted[
                COLOUR_HEX_COL
            ],
            s=ACCEPTED_SIZE,
            marker=marker,
            edgecolors="black",
            linewidths=0.35,
            alpha=ACCEPTED_ALPHA,
            zorder=3,
        )

    if not outliers.empty:
        ax.scatter(
            outliers["a"],
            outliers["b"],
            c=outliers[
                COLOUR_HEX_COL
            ],
            s=OUTLIER_SIZE,
            marker=marker,
            edgecolors="red",
            linewidths=1.4,
            alpha=1.0,
            zorder=5,
        )

    if not low_chroma.empty:
        ax.scatter(
            low_chroma["a"],
            low_chroma["b"],
            facecolors="none",
            edgecolors="gray",
            s=LOW_CHROMA_SIZE,
            marker=marker,
            linewidths=0.7,
            alpha=0.45,
            zorder=2,
        )


# ------------------------------------------------------------
# Highlight df_best_cases anchors
# ------------------------------------------------------------

for _, anchor in df_hue_anchors.iterrows():

    wavelength = float(
        anchor[
            "design_peak_wavelength_nm"
        ]
    )

    if (
        wavelength
        not in selected_target_wavelengths
    ):
        continue

    marker = marker_dict[wavelength]

    anchor_a = float(
        anchor["anchor_a"]
    )

    anchor_b = float(
        anchor["anchor_b"]
    )

    anchor_hex = str(
        anchor["anchor_hex"]
    )

    # Large white halo
    ax.scatter(
        [anchor_a],
        [anchor_b],
        facecolors="none",
        edgecolors="white",
        marker=marker,
        s=BEST_OUTER_SIZE,
        linewidths=5.0,
        zorder=7,
    )

    # Black-outlined target marker
    ax.scatter(
        [anchor_a],
        [anchor_b],
        facecolors=anchor_hex,
        edgecolors="black",
        marker=marker,
        s=BEST_INNER_SIZE,
        linewidths=2.2,
        zorder=8,
    )

    # Central star
    ax.scatter(
        [anchor_a],
        [anchor_b],
        facecolors="white",
        edgecolors="black",
        marker="*",
        s=BEST_STAR_SIZE,
        linewidths=0.8,
        zorder=9,
    )

    annotation_lines = [
        f"{wavelength:.0f} nm",
        f"dopt={float(anchor['anchor_d_opt']):.3f}",
        f"ηC={100 * float(anchor['anchor_eta_C']):.1f}%",
        f"ηL={100 * float(anchor['anchor_eta_L']):.1f}%",
    ]

    if anchor["anchor_absorber_location"]:
        annotation_lines.append(
            str(
                anchor[
                    "anchor_absorber_location"
                ]
            )
        )

    ax.annotate(
        "\n".join(annotation_lines),
        xy=(anchor_a, anchor_b),
        xytext=(8, 8),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=7.5,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "black",
            "alpha": 0.88,
        },
        zorder=10,
    )


# ============================================================
# AXIS FORMATTING
# ============================================================

ax.axhline(
    0.0,
    color="gray",
    linewidth=0.8,
    zorder=1,
)

ax.axvline(
    0.0,
    color="gray",
    linewidth=0.8,
    zorder=1,
)

ax.set_xlim(
    -1.12 * plot_radius,
    1.12 * plot_radius,
)

ax.set_ylim(
    -1.12 * plot_radius,
    1.12 * plot_radius,
)

ax.set_aspect("equal")

ax.set_xlabel("CIELAB a*")
ax.set_ylabel("CIELAB b*")

ax.set_title(
    "CIELAB hue-acceptance sectors by design wavelength\n"
    "Sector centers defined by minimum-dopt cases from df_best_cases; "
    f"acceptance = ±{HUE_HALF_WIDTH_DEG:.1f}°"
)


# ============================================================
# LEGEND
# ============================================================

target_handles = []

for wavelength in selected_target_wavelengths:

    center_hue = anchor_hue_by_wavelength[wavelength]
    anchor_d_opt = anchor_d_opt_by_wavelength[wavelength]

    handle = Line2D(
        [0],
        [0],
        marker=marker_dict[wavelength],
        linestyle="None",
        markerfacecolor=sector_cmap(
            sector_norm(wavelength)
        ),
        markeredgecolor="black",
        markersize=9,
        label=(
            f"{wavelength:.0f} nm; "
            f"center={center_hue:.1f} degrees; "
            f"d_opt={anchor_d_opt:.3f}"
        ),
    )

    target_handles.append(handle)

diagnostic_handles = [
    Line2D(
        [0],
        [0],
        marker="*",
        linestyle="None",
        markerfacecolor="white",
        markeredgecolor="black",
        markersize=11,
        label="Selected minimum-dopt case",
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="None",
        markerfacecolor="white",
        markeredgecolor="red",
        markeredgewidth=1.5,
        markersize=8,
        label="Outside accepted hue sector",
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="None",
        markerfacecolor="none",
        markeredgecolor="gray",
        markersize=8,
        label=(
            f"C* < "
            f"{MIN_CHROMA_FOR_HUE_CHECK:g}; "
            "hue not assessed"
        ),
    ),
]

ax.legend(
    handles=(
        target_handles
        + diagnostic_handles
    ),
    loc="upper left",
    bbox_to_anchor=(1.02, 1.0),
    title=(
        "Target wavelength / "
        "minimum-dopt anchor"
    ),
)

fig.tight_layout()
plt.show()


# ============================================================
# DIAGNOSTIC TABLES
# ============================================================

summary_records = []

for wavelength in target_wavelengths:

    subset = df_hue.loc[
        np.isclose(
            df_hue[
                "design_peak_wavelength_nm"
            ],
            wavelength,
        )
    ]

    applicable = subset.loc[
        subset[
            "hue_check_applicable"
        ]
    ]

    n_total = len(subset)
    n_applicable = len(applicable)

    n_outliers = int(
        applicable[
            "potential_hue_outlier"
        ].sum()
    )

    summary_records.append(
        {
            "design_peak_wavelength_nm": wavelength,
            "anchor_hue_deg": (
                anchor_hue_by_wavelength[
                    wavelength
                ]
            ),
            "anchor_d_opt": (
                anchor_d_opt_by_wavelength[
                    wavelength
                ]
            ),
            "acceptance_half_width_deg": (
                HUE_HALF_WIDTH_DEG
            ),
            "number_of_cases": n_total,
            "number_hue_assessed": n_applicable,
            "number_potential_outliers": n_outliers,
            "potential_outlier_fraction": (
                n_outliers / n_applicable
                if n_applicable > 0
                else np.nan
            ),
            "target_retained": (
                wavelength
                in selected_target_wavelengths
            ),
        }
    )

df_hue_acceptance_summary = (
    pd.DataFrame(summary_records)
)

print(
    "\nHue anchors from df_best_cases:"
)

display(df_hue_anchors)

print("\nHue-acceptance summary:")

display(df_hue_acceptance_summary)

if not df_sector_overlaps.empty:

    print(
        "\nTarget-sector overlap assessment:"
    )

    display(
        df_sector_overlaps.sort_values(
            "overlap_fraction",
            ascending=False,
        )
    )

if omitted_target_wavelengths:

    print(
        "\nAutomatically omitted target wavelengths:",
        omitted_target_wavelengths,
    )


# ============================================================
# POTENTIAL OUTLIER TABLE
# ============================================================

outlier_columns = [
    "design_peak_wavelength_nm",
    "a",
    "b",
    "C_calculated",
    "hue_calculated_deg",
    "anchor_hue_deg",
    "hue_difference_deg",
    "n_A",
    "k",
    "n_layers",
    "absorber_location",
    "eta_C",
    "eta_L",
    COLOUR_HEX_COL,
]

outlier_columns = [
    column
    for column in outlier_columns
    if column in df_hue.columns
]

df_potential_hue_outliers = (
    df_hue.loc[
        df_hue[
            "potential_hue_outlier"
        ],
        outlier_columns,
    ]
    .sort_values(
        [
            "design_peak_wavelength_nm",
            "hue_difference_deg",
        ]
    )
    .reset_index(drop=True)
)

print(
    "\nPotential hue outliers:",
    len(df_potential_hue_outliers),
)

display(
    df_potential_hue_outliers.head(50)
)

# Analysis extended to forward scattering

# In[202]:


from bragg_onion.geometry import (
    ExtinctionModifier,
    ExtraOuterShellSpec,
)
from bragg_onion.materials import ConstantDispersion


def build_absorber_objects(row):
    """
    Reconstruct extinction_modifiers / extra_outer_shell
    from one dataframe row.
    """

    k = float(row["k"])
    absorber_location = row["absorber_location"]

    extinction_modifiers = None
    extra_outer_shell = None

    if absorber_location == "core":

        extinction_modifiers = [
            ExtinctionModifier(
                target="core",
                mode="substitute",
                profile=k,
                name=f"core_k_{k:.3f}",
            )
        ]

    elif absorber_location == "low_index":

        extinction_modifiers = [
            ExtinctionModifier(
                target="material_B",
                mode="substitute",
                profile=k,
                name=f"low_index_k_{k:.3f}",
            )
        ]

    elif absorber_location == "all_layers":

        extinction_modifiers = [
            ExtinctionModifier(
                target="both_materials",
                mode="substitute",
                profile=k,
                name=f"all_layers_k_{k:.3f}",
            )
        ]

    elif absorber_location == "outer_shell":

        absorber_material = ConstantDispersion.from_nk(
            "outer_absorber",
            n=1.0,
            k=k,
        )

        extra_outer_shell = ExtraOuterShellSpec(
            thickness_m=float(row["outer_shell_thickness_nm"]) * 1e-9,
            material=absorber_material,
            name=f"outer_shell_k_{k:.3f}",
        )

    else:
        raise ValueError(
            f"Unknown absorber location: {absorber_location}"
        )

    return extinction_modifiers, extra_outer_shell

# In[203]:


import numpy as np

from bragg_onion.materials import ConstantDispersion
from bragg_onion.sweep import run_bragg_onion_sweep
from bragg_onion._notebook_tools.scattering_extract import (
    extract_case_arrays,
)

analysis_wavelengths_nm = np.arange(380.0, 781.0, 1.0)
analysis_theta_deg = np.linspace(0.0, 180.0, 721)

best_cases = {}

for _, row_case in df_best.iterrows():

    # --------------------------------------------------------
    # Reconstruct the base, nonabsorbing Bragg materials
    # Absorption is added separately according to location.
    # --------------------------------------------------------
    material_a = ConstantDispersion.from_nk(
        "A",
        n=float(row_case["n_A"]),
        k=0.0,
    )

    material_b = ConstantDispersion.from_nk(
        "B",
        n=float(row_case["n_B"]),
        k=0.0,
    )

    # --------------------------------------------------------
    # Reconstruct absorber placement
    # --------------------------------------------------------
    extinction_modifiers, extra_outer_shell = (
        build_absorber_objects(row_case)
    )

    # --------------------------------------------------------
    # Reconstruct exactly one geometry case
    # --------------------------------------------------------
    parameter_grid = {
        "peak_wavelength_m": [
            float(row_case["peak_wavelength_m"])
        ],
        "n_layers": [
            int(row_case["n_layers"])
        ],
        "outer_layer": [
            str(row_case["outer_layer"])
        ],
        "core_thickness_factor": [
            float(row_case["core_thickness_factor"])
        ],
    }

    # --------------------------------------------------------
    # Detailed rerun
    # The backward result reproduces the original sweep setup.
    # --------------------------------------------------------
    sweep_case = run_bragg_onion_sweep(
        geometry_mode="peak_wavelength",
        parameter_grid=parameter_grid,
        material_a=material_a,
        material_b=material_b,
        n_medium=1.0,
        wavelengths_m=analysis_wavelengths_nm * 1e-9,
        theta_rad=np.deg2rad(analysis_theta_deg),
        collection_na=float(row_case["collection_na"]),
        collection_direction="backward",
        integration_quantity_for_colour="q_collected_geom",
        colour_wavelength_min_nm=None,
        colour_wavelength_max_nm=None,
        colour_normalize_input=False,
        colour_normalization="max",
        extinction_modifiers=extinction_modifiers,
        extra_outer_shell=extra_outer_shell,
        store_full_results=True,
        progress=False,
    )

    data = extract_case_arrays(sweep_case)

    key = (
        str(row_case["absorber_location"]),
        int(round(float(row_case["design_peak_wavelength_nm"]))),
    )

    if key in best_cases:
        raise ValueError(
            f"Duplicate best-case key encountered: {key}"
        )

    best_cases[key] = {
        "row": row_case.copy(),
        "sweep_case": sweep_case,
        "case_result": sweep_case.case_results[0],
        "data": data,
    }

    print(
        f"Finished {key}: "
        f"n_A={float(row_case['n_A']):.3f}, "
        f"k={float(row_case['k']):.3f}"
    )

print(f"\nStored {len(best_cases)} detailed rerun cases.")
print("Keys:")
print(list(best_cases.keys()))

# In[ ]:


# RERUN THE SAME 12 BEST CASES WITH FORWARD COLLECTION

import numpy as np

from bragg_onion.materials import ConstantDispersion
from bragg_onion.sweep import run_bragg_onion_sweep
from bragg_onion._notebook_tools.scattering_extract import (
    extract_case_arrays,
)

best_cases_forward = {}

for _, row_case in df_best.iterrows():

    absorber_location = str(row_case["absorber_location"])
    design_wavelength_nm = int(
        round(float(row_case["design_peak_wavelength_nm"]))
    )

    key = (
        absorber_location,
        design_wavelength_nm,
    )

    print(
        f"Running forward collection: "
        f"location={absorber_location}, "
        f"design λ={design_wavelength_nm} nm"
    )

    # --------------------------------------------------------
    # Reconstruct the nonabsorbing base materials.
    # Absorption is introduced by build_absorber_objects().
    # --------------------------------------------------------
    material_a = ConstantDispersion.from_nk(
        "A",
        n=float(row_case["n_A"]),
        k=0.0,
    )

    material_b = ConstantDispersion.from_nk(
        "B",
        n=float(row_case["n_B"]),
        k=0.0,
    )

    # --------------------------------------------------------
    # Reconstruct absorber placement and strength.
    # --------------------------------------------------------
    extinction_modifiers, extra_outer_shell = (
        build_absorber_objects(row_case)
    )

    # --------------------------------------------------------
    # Reconstruct exactly one geometry.
    # Explicit casts are important for values read from pandas.
    # --------------------------------------------------------
    parameter_grid = {
        "peak_wavelength_m": [
            float(row_case["peak_wavelength_m"])
        ],
        "n_layers": [
            int(row_case["n_layers"])
        ],
        "outer_layer": [
            str(row_case["outer_layer"])
        ],
        "core_thickness_factor": [
            float(row_case["core_thickness_factor"])
        ],
    }

    # --------------------------------------------------------
    # Forward-collected rerun.
    # --------------------------------------------------------
    sweep_case_forward = run_bragg_onion_sweep(
        geometry_mode="peak_wavelength",
        parameter_grid=parameter_grid,
        material_a=material_a,
        material_b=material_b,
        n_medium=1.0,
        wavelengths_m=np.asarray(
            analysis_wavelengths_nm,
            dtype=float,
        ) * 1e-9,
        theta_rad=np.deg2rad(
            np.asarray(analysis_theta_deg, dtype=float)
        ),
        collection_na=float(row_case["collection_na"]),
        collection_direction="forward",
        integration_quantity_for_colour="q_collected_geom",
        colour_wavelength_min_nm=None,
        colour_wavelength_max_nm=None,
        colour_normalize_input=False,
        colour_normalization="max",
        extinction_modifiers=extinction_modifiers,
        extra_outer_shell=extra_outer_shell,
        store_full_results=True,
        progress=False,
    )

    data_forward = extract_case_arrays(sweep_case_forward)

    best_cases_forward[key] = {
        "row": row_case.copy(),
        "sweep_case": sweep_case_forward,
        "case_result": sweep_case_forward.case_results[0],
        "data": data_forward,
    }

    print(f"  Finished {key}")

print(
    f"\nStored {len(best_cases_forward)} "
    "forward-collected detailed cases."
)

print("Forward-case keys:")
print(list(best_cases_forward.keys()))

# In[ ]:


# COMMON SETUP FOR FORWARD/BACKWARD COMPARISON PLOTS

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from bragg_onion.sweep import make_plotting_aliases


location_order = [
    "core",
    "low_index",
    "all_layers",
    "outer_shell",
]

location_labels = {
    "core": "Core",
    "low_index": "Low-index layers",
    "all_layers": "All Bragg layers",
    "outer_shell": "Outer shell",
}

location_styles = {
    "core": "-",
    "low_index": "--",
    "all_layers": "-.",
    "outer_shell": ":",
}

design_wavelengths = sorted(
    {
        int(round(float(key[1])))
        for key in best_cases.keys()
    }
)

# Confirm corresponding forward and backward results exist.
backward_keys = set(best_cases.keys())
forward_keys = set(best_cases_forward.keys())

if backward_keys != forward_keys:
    raise ValueError(
        "Forward and backward case dictionaries do not contain identical cases.\n"
        f"Missing from forward: {sorted(backward_keys - forward_keys)}\n"
        f"Missing from backward: {sorted(forward_keys - backward_keys)}"
    )

print("Absorber locations:", location_order)
print("Design wavelengths:", design_wavelengths)
print("Number of corresponding cases:", len(backward_keys))

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from matplotlib.patches import Rectangle
from bragg_onion.sweep import make_plotting_aliases


def readable_text_colour(hex_colour):
    """Choose black or white annotation text from tile luminance."""
    rgb = np.asarray(
        mpl.colors.to_rgb(hex_colour),
        dtype=float,
    )

    luminance = (
        0.2126 * rgb[0]
        + 0.7152 * rgb[1]
        + 0.0722 * rgb[2]
    )

    return "black" if luminance > 0.52 else "white"


# ------------------------------------------------------------
# Extract and validate the forward-collected colour results
# ------------------------------------------------------------
forward_summary_rows = {}

for key, result in best_cases_forward.items():

    sweep_case_forward = result["sweep_case"]
    case_result_forward = result["case_result"]
    data_forward = result["data"]

    # IMPORTANT:
    # make_plotting_aliases creates the convenient columns
    # "hex", "eta_C", "eta_L", etc.
    forward_df = make_plotting_aliases(
        sweep_case_forward.dataframe
    )

    if len(forward_df) != 1:
        raise ValueError(
            f"Expected exactly one forward dataframe row for {key}, "
            f"but found {len(forward_df)}."
        )

    if "hex" not in forward_df.columns:
        raise KeyError(
            f"The forward dataframe for {key} still has no 'hex' alias.\n"
            f"Available columns:\n{forward_df.columns.tolist()}"
        )

    forward_row = forward_df.iloc[0]

    # Validate that the forward colour spectrum is the same quantity
    # as the forward-collected q_collected_geom spectrum.
    colour_result = case_result_forward.colour_result

    spectrum_used = np.asarray(
        colour_result.spectrum_used,
        dtype=float,
    ).ravel()

    q_forward = np.asarray(
        data_forward["q_collected_geom"],
        dtype=float,
    ).ravel()

    wavelengths_colour_nm = np.asarray(
        colour_result.wavelengths_nm,
        dtype=float,
    ).ravel()

    wavelengths_integrated_nm = np.asarray(
        data_forward["wavelengths_int_nm"],
        dtype=float,
    ).ravel()

    if not np.allclose(
        wavelengths_colour_nm,
        wavelengths_integrated_nm,
    ):
        raise ValueError(
            f"Colour and integrated wavelength grids differ for {key}."
        )

    if not np.allclose(
        spectrum_used,
        q_forward,
        equal_nan=True,
    ):
        raise ValueError(
            f"The forward spectrum_used does not match "
            f"q_collected_geom for {key}."
        )

    # This is the hex colour calculated from the forward spectrum.
    tile_hex = str(forward_row["hex_display"])

    if not mpl.colors.is_color_like(tile_hex):
        raise ValueError(
            f"Invalid forward hex colour for {key}: {tile_hex!r}"
        )

    forward_summary_rows[key] = {
        "summary_row": forward_row,
        "tile_hex": tile_hex,
        "spectrum_used": spectrum_used,
        "wavelengths_nm": wavelengths_colour_nm,
    }


print(
    f"Validated forward spectra and colours for "
    f"{len(forward_summary_rows)} cases."
)

# In[ ]:


# PANEL 1 — FORWARD-COLLECTED COLOUR-STRIP MATRIX
#
# Each tile corresponds to the same parameter combination that maximized
# backward eta_C for that absorber location and design wavelength.
# The displayed colour and metrics are recalculated for forward collection.

def readable_text_colour(hex_colour):
    """
    Choose black or white text based on displayed tile luminance.
    """
    try:
        rgb = np.asarray(
            mpl.colors.to_rgb(hex_colour),
            dtype=float,
        )
        luminance = (
            0.2126 * rgb[0]
            + 0.7152 * rgb[1]
            + 0.0722 * rgb[2]
        )
        return "black" if luminance > 0.52 else "white"
    except Exception:
        return "black"


# Extract one forward summary row for every corresponding case.
forward_summary_rows = {}

for key, result in best_cases_forward.items():

    # Add short plotting aliases such as eta_C, eta_L and hex.
    forward_df = make_plotting_aliases(
        result["sweep_case"].dataframe
    )

    # Add display-normalized CIELAB, XYZ and hex columns.
    forward_df = add_display_normalized_colour_columns(
        forward_df,
        inplace=False,
    )

    if len(forward_df) != 1:
        raise ValueError(
            f"Expected one forward dataframe row for {key}, "
            f"but found {len(forward_df)}."
        )

    required_columns = [
        "eta_C",
        "eta_L",
        "hex",
        "hex_display",
        "L_display",
        "a_display",
        "b_display",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in forward_df.columns
    ]

    if missing_columns:
        raise KeyError(
            f"Forward dataframe for {key} is missing columns: "
            f"{missing_columns}"
        )

    forward_summary_rows[key] = forward_df.iloc[0]


fig, ax = plt.subplots(
    figsize=(
        3.2 * len(design_wavelengths),
        2.5 * len(location_order),
    )
)

n_locations = len(location_order)

for i, location in enumerate(location_order):
    y = n_locations - 1 - i

    for j, design_wl in enumerate(design_wavelengths):
        key = (location, design_wl)

        if key not in best_cases_forward:
            continue

        original_row = best_cases_forward[key]["row"]
        forward_row = forward_summary_rows[key]

        tile_hex = str(forward_row["hex_display"])

        rect = Rectangle(
            (j, y),
            1,
            1,
            facecolor=tile_hex,
            edgecolor="black",
            linewidth=1.0,
        )
        ax.add_patch(rect)

        text_lines = [
            f"ηC,fwd={100 * float(forward_row['eta_C']):.0f}%",
            f"ηL,fwd={100 * float(forward_row['eta_L']):.0f}%",
            f"nA={float(original_row['n_A']):.2f}",
            f"k={float(original_row['k']):.3f}",
        ]

        if location == "core":
            text_lines.append(
                "core factor="
                f"{float(original_row['core_thickness_factor']):.2f}"
            )

        elif location == "outer_shell":
            if "outer_shell_thickness_nm" in original_row.index:
                text_lines.append(
                    "shell="
                    f"{float(original_row['outer_shell_thickness_nm']):.0f} nm"
                )

        ax.text(
            j + 0.5,
            y + 0.5,
            "\n".join(text_lines),
            ha="center",
            va="center",
            fontsize=9,
            color=readable_text_colour(tile_hex),
        )


ax.set_xlim(0, len(design_wavelengths))
ax.set_ylim(0, len(location_order))
ax.set_aspect("equal")

ax.set_xticks(
    np.arange(len(design_wavelengths)) + 0.5
)
ax.set_xticklabels(
    [f"{wl:.0f} nm" for wl in design_wavelengths]
)

ax.set_yticks(
    np.arange(len(location_order)) + 0.5
)
ax.set_yticklabels(
    [
        location_labels[loc]
        for loc in location_order[::-1]
    ]
)

ax.set_xlabel("Design wavelength")
ax.set_ylabel("Absorber location")
ax.set_title(
    "Forward-collected colours for the cases with best backward ηC\n"
    "(NA = 1; identical physical cases used for forward and backward comparison)"
)

fig.tight_layout()
plt.show()

# In[ ]:


# PANEL 2 — FORWARD AND BACKWARD COLLECTED SPECTRA

NORMALIZE_SPECTRA = False
# False: compare absolute q_collected_geom values.
# True: normalize each individual spectrum to its own maximum,
#       emphasizing shape rather than magnitude.

forward_colour = "tab:blue"
backward_colour = "tab:red"


def normalize_spectrum_if_requested(values, normalize):
    values = np.asarray(values, dtype=float).ravel()

    if not normalize:
        return values

    maximum = np.nanmax(values)
    return values / maximum if maximum > 0 else values


fig, axes = plt.subplots(
    1,
    len(design_wavelengths),
    figsize=(5.2 * len(design_wavelengths), 4.8),
    sharex=True,
    sharey=NORMALIZE_SPECTRA,
)

axes = np.atleast_1d(axes)

for ax, design_wl in zip(axes, design_wavelengths):

    for location in location_order:
        key = (location, design_wl)

        backward_data = best_cases[key]["data"]
        forward_data = best_cases_forward[key]["data"]

        wl_backward = np.asarray(
            backward_data["wavelengths_int_nm"],
            dtype=float,
        ).ravel()

        wl_forward = np.asarray(
            forward_data["wavelengths_int_nm"],
            dtype=float,
        ).ravel()

        q_backward = normalize_spectrum_if_requested(
            backward_data["q_collected_geom"],
            NORMALIZE_SPECTRA,
        )

        q_forward = normalize_spectrum_if_requested(
            forward_data["q_collected_geom"],
            NORMALIZE_SPECTRA,
        )

        linestyle = location_styles[location]

        ax.plot(
            wl_backward,
            q_backward,
            color=backward_colour,
            linestyle=linestyle,
            linewidth=2.0,
        )

        ax.plot(
            wl_forward,
            q_forward,
            color=forward_colour,
            linestyle=linestyle,
            linewidth=2.0,
        )

    ax.axvline(
        design_wl,
        color="gray",
        linestyle="--",
        linewidth=1.0,
        alpha=0.8,
    )

    ax.set_title(f"Design λ = {design_wl:.0f} nm")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_xlim(
        min(wl_backward.min(), wl_forward.min()),
        max(wl_backward.max(), wl_forward.max()),
    )
    ax.grid(True, alpha=0.3)

    if NORMALIZE_SPECTRA:
        ax.set_ylim(0, 1.05)

if NORMALIZE_SPECTRA:
    axes[0].set_ylabel(
        "Collected spectrum\n(normalized independently)"
    )
else:
    axes[0].set_ylabel(
        r"Collected scattering efficiency $Q_{\mathrm{collected,geom}}$"
    )


# Legend part 1: collection direction
direction_handles = [
    Line2D(
        [0], [0],
        color=forward_colour,
        linewidth=2.5,
        label="Forward collection",
    ),
    Line2D(
        [0], [0],
        color=backward_colour,
        linewidth=2.5,
        label="Backward collection",
    ),
]

# Legend part 2: absorber location
location_handles = [
    Line2D(
        [0], [0],
        color="black",
        linestyle=location_styles[location],
        linewidth=2.0,
        label=location_labels[location],
    )
    for location in location_order
]

fig.legend(
    handles=direction_handles + location_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.08),
    ncol=3,
    frameon=True,
)

fig.suptitle(
    "Forward- and backward-collected spectra of the same best-ηC cases",
    y=1.16,
)

fig.tight_layout()
plt.show()

# In[ ]:


# PANEL 3 — ASYMMETRY g AND ALBEDO

location_colours = {
    "core": "tab:blue",
    "low_index": "tab:orange",
    "all_layers": "tab:green",
    "outer_shell": "tab:red",
}

# ------------------------------------------------------------
# Verify that g and albedo are independent of collection direction.
# ------------------------------------------------------------
for key in best_cases:
    backward_data = best_cases[key]["data"]
    forward_data = best_cases_forward[key]["data"]

    if not np.allclose(
        backward_data["g"],
        forward_data["g"],
        equal_nan=True,
    ):
        raise ValueError(
            f"Forward/backward g mismatch for {key}."
        )

    if not np.allclose(
        backward_data["albedo"],
        forward_data["albedo"],
        equal_nan=True,
    ):
        raise ValueError(
            f"Forward/backward albedo mismatch for {key}."
        )

print("Verified: g and albedo agree between forward and backward reruns.")


# ------------------------------------------------------------
# Plot.
# ------------------------------------------------------------
fig, axes = plt.subplots(
    1,
    len(design_wavelengths),
    figsize=(5.4 * len(design_wavelengths), 4.8),
    sharex=True,
    sharey=True,
)

axes = np.atleast_1d(axes)
right_axes = []

for ax, design_wl in zip(axes, design_wavelengths):
    ax_right = ax.twinx()
    right_axes.append(ax_right)

    for location in location_order:
        key = (location, design_wl)
        data_case = best_cases[key]["data"]

        wavelength_nm = np.asarray(
            data_case["wavelengths_nm"],
            dtype=float,
        ).ravel()

        g_values = np.asarray(
            data_case["g"],
            dtype=float,
        ).ravel()

        albedo_values = np.asarray(
            data_case["albedo"],
            dtype=float,
        ).ravel()

        colour = location_colours[location]

        # g on left axis
        ax.plot(
            wavelength_nm,
            g_values,
            color=colour,
            linestyle="-",
            linewidth=2.0,
        )

        # albedo on right axis
        ax_right.plot(
            wavelength_nm,
            albedo_values,
            color=colour,
            linestyle="--",
            linewidth=1.8,
            alpha=0.9,
        )

    ax.axvline(
        design_wl,
        color="gray",
        linestyle=":",
        linewidth=1.0,
    )

    ax.set_title(f"Design λ = {design_wl:.0f} nm")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_xlim(wavelength_nm[0], wavelength_nm[-1])
    ax.grid(True, alpha=0.3)

    # Physical axis ranges
    ax.set_ylim(-1.0, 1.0)
    ax_right.set_ylim(0.0, 1.05)

# Label only outer axes to reduce clutter
axes[0].set_ylabel("Asymmetry parameter g")
right_axes[-1].set_ylabel("Single-scattering albedo")

# Location legend: colour encoding
location_handles = [
    Line2D(
        [0], [0],
        color=location_colours[location],
        linewidth=2.5,
        label=location_labels[location],
    )
    for location in location_order
]

# Quantity legend: line-style encoding
quantity_handles = [
    Line2D(
        [0], [0],
        color="black",
        linestyle="-",
        linewidth=2.0,
        label="Asymmetry g",
    ),
    Line2D(
        [0], [0],
        color="black",
        linestyle="--",
        linewidth=2.0,
        label="Albedo",
    ),
]

fig.legend(
    handles=location_handles + quantity_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.08),
    ncol=3,
    frameon=True,
)

fig.suptitle(
    "Asymmetry and albedo of the best backward-ηC cases",
    y=1.16,
)

fig.tight_layout()
plt.show()

# In[ ]:


# PARAMETER SUMMARY FOR THE 12 COMPARED CASES

parameter_columns = [
    "absorber_location",
    "design_peak_wavelength_nm",
    "n_A",
    "n_B",
    "k",
    "n_layers",
    "core_thickness_factor",
    "outer_shell_thickness_nm",
    "eta_C",
    "eta_L",
]

parameter_columns = [
    column
    for column in parameter_columns
    if column in df_best.columns
]

df_case_summary = (
    df_best[parameter_columns]
    .sort_values(
        [
            "design_peak_wavelength_nm",
            "absorber_location",
        ]
    )
    .reset_index(drop=True)
)

display(df_case_summary)

# In[ ]:


# Parameter values used by the previously identified best-chroma cases

print(
    "N_A_VALS = np.array(",
    sorted(df_best["n_A"].dropna().unique().tolist()),
    ")",
)

print(
    "K_BEST_VALS = np.array(",
    sorted(df_best["k"].dropna().unique().tolist()),
    ")",
)

print(
    "CORE_THICKNESS_FACTORS = np.array(",
    sorted(
        df_best["core_thickness_factor"]
        .dropna()
        .unique()
        .tolist()
    ),
    ")",
)

if "outer_shell_thickness_nm" in df_best.columns:
    print(
        "OUTER_SHELL_THICKNESS_NM_VALS = np.array(",
        sorted(
            df_best.loc[
                df_best["absorber_location"] == "outer_shell",
                "outer_shell_thickness_nm",
            ]
            .dropna()
            .unique()
            .tolist()
        ),
        ")",
    )
