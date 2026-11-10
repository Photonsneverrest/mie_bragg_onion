#!/usr/bin/env python
# coding: utf-8

# # Sensitivity of optimized Bragg-onion colours
#
# This notebook analyzes the local sensitivity of four optimized Bragg-onion reference designs with quarter-wave design wavelengths
#
# \[
# \lambda_0 \in \{380,\,580,\,620,\,760\}\ \mathrm{nm}.
# \]
#
# Each reference design is the minimum-\(d_{\mathrm{opt}}\) case previously selected from the refined parameter sweep. Unless explicitly stated otherwise, one parameter is varied at a time while the remaining parameters are fixed at their reference values.
#
# ## Sensitivity studies
#
# ### Number of shells
#
# The total number of radial A/B domains, including the core, is denoted by \(N_{\mathrm{shell}}\). The investigated shell-number factors are
#
# \[
# \frac{N_{\mathrm{shell}}}{N_{\mathrm{shell},0}}
# \in
# \{
# 0.10,\,
# 0.12,\,
# 0.17,\,
# 0.24,\,
# 0.34,\,
# 0.50,\,
# 0.70,\,
# 0.80,\,
# 0.90,\,
# 1.00,\,
# 1.05,\,
# 1.10,\,
# 1.15
# \}.
# \]
#
# The resulting shell numbers are rounded to the nearest even integer. The quarter-wave shell thicknesses remain unchanged, so changing \(N_{\mathrm{shell}}\) also changes the particle diameter.
#
# ### Refractive-index ratio with quarter-wave reconstruction
#
# The refractive-index ratio is varied according to
#
# \[
# \frac{(n_A/n_B)}{(n_A/n_B)_0}
# \in
# \{
# 0.80,\,
# 0.875,\,
# 0.925,\,
# 0.975,\,
# 1.00,\,
# 1.025,\,
# 1.075,\,
# 1.125,\,
# 1.20
# \}.
# \]
#
# Cases for which \(n_A<1\) are omitted. The quarter-wave condition is maintained,
#
# \[
# n_A d_A=n_B d_B=\frac{\lambda_0}{4},
# \]
#
# so varying \(n_A/n_B\) also changes the high-index shell thickness \(d_A\).
#
# A separate control calculation for the \(580\ \mathrm{nm}\) reference design varies \(n_A/n_B\) while retaining the original physical shell radii.
#
# ### Surrounding-medium index
#
# The surrounding refractive index is varied over
#
# \[
# n_{\mathrm{m}}
# \in
# \{
# 1.00,\,
# 1.05,\,
# 1.10,\,
# 1.20,\,
# 1.33,\,
# 1.40,\,
# 1.50
# \}.
# \]
#
# The reciprocal angular integration covers a constant \(90^\circ\) half-angle for every surrounding index, corresponding to diffuse illumination over the complete front hemisphere in the original configuration.
#
# ### Illumination angular range
#
# The reciprocal illumination half-angle is varied over
#
# \[
# \theta_{\mathrm{ill,max}}
# \in
# \{
# 5^\circ,\,
# 12.7^\circ,\,
# 20^\circ,\,
# 30^\circ,\,
# 45^\circ,\,
# 60^\circ,\,
# 90^\circ
# \}.
# \]
#
# The \(90^\circ\) case represents illumination over the complete front hemisphere.
#
# ### Absorption
#
# For an absorbing reference design, the absorber remains at its optimized location and its extinction coefficient is varied according to
#
# \[
# \frac{k}{k_0}
# \in
# \{
# 0,\,
# 0.25,\,
# 0.50,\,
# 0.75,\,
# 1.00,\,
# 1.25,\,
# 1.50,\,
# 2.00,\,
# 3.00
# \}.
# \]
#
# For a nonabsorbing reference design, absorption is introduced uniformly in all shells using
#
# \[
# k
# \in
# \{
# 0,\,
# 10^{-4},\,
# 2.5\times10^{-4},\,
# 5\times10^{-4},\,
# 10^{-3},\,
# 2.5\times10^{-3},\,
# 5\times10^{-3},\,
# 10^{-2},\,
# 2\times10^{-2}
# \}.
# \]
#
# Because \(k/k_0\) is undefined for a nonabsorbing reference with \(k_0=0\), normalized absorption-sensitivity curves include only reference designs with \(k_0>0\). The absolute-\(k\) responses of initially nonabsorbing cases are analyzed separately.
#
# ## Colour-response metrics
#
# Changes are calculated relative to the corresponding optimized reference:
#
# \[
# \Delta C_{ab}^*
# =
# C_{ab}^*-C_{ab,0}^*,
# \]
#
# \[
# \Delta L^*
# =
# L^*-L_0^*,
# \]
#
# and
#
# \[
# \Delta h_{ab}
# =
# \operatorname{wrap}_{[-180^\circ,180^\circ)}
# \left(
# h_{ab}-h_{ab,0}
# \right).
# \]
#
# The wrapped hue difference prevents artificial discontinuities at \(0^\circ/360^\circ\).
#
# ## Planned figure panels
#
# - **Panel A:** local one-parameter sensitivity curves for \(\Delta C_{ab}^*\), \(\Delta L^*\), and \(\Delta h_{ab}\).
# - **Panel B:** joint shell-number/refractive-index sensitivity maps. This panel requires an additional two-parameter interaction sweep and cannot be inferred rigorously from the one-parameter sweeps.
# - **Panel C:** operational robustness limits, defined as the smallest relative parameter perturbation producing
#   \[
#   |\Delta C_{ab}^*|\geq5,\qquad
#   |\Delta L^*|\geq5,\qquad
#   |\Delta h_{ab}|\geq5^\circ.
#   \]
#   These thresholds are comparative criteria for this study and are not presented as universal perceptual thresholds.

# In[1]:


# ============================================================
# IMPORTS AND GENERAL PLOTTING CONFIGURATION
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from matplotlib.lines import Line2D
from matplotlib.colors import TwoSlopeNorm
from scipy.interpolate import griddata


# ------------------------------------------------------------
# Optional paper style
# ------------------------------------------------------------

mpl.rcdefaults()

STYLE_PATH = Path(
    "/mnt/c/Users/SchwarzN/"
    "OneDrive - Université de Fribourg/"
    "Institution/Other/Scripts_and_HowTo/"
    "science_paper.mplstyle"
)

if STYLE_PATH.exists():
    plt.style.use(STYLE_PATH)
else:
    print(
        "Warning: paper style not found. "
        "Using Matplotlib defaults."
    )


# ------------------------------------------------------------
# Data location
# ------------------------------------------------------------

SENSITIVITY_DIRECTORY = Path(
    "/mnt/c/Users/SchwarzN/"
    "OneDrive - Université de Fribourg/"
    "Institution/P1_BraggSpericalPigments/"
    "Simulation/MIE_Computation/"
    "Scattnlay_BraggOnion/outputs/"
    "sensitivity_analysis"
)


# ------------------------------------------------------------
# Reference-design appearance
# Consistent with Figure 2
# ------------------------------------------------------------

REFERENCE_WAVELENGTHS_NM = [
    380,
    580,
    620,
    760,
]

REFERENCE_MARKERS = {
    380: "o",
    580: "^",
    620: "v",
    760: "h",
}

REFERENCE_COLOURS = {
    380: "#7B2CBF",
    580: "#2C7FA0",
    620: "#2CA02C",
    760: "#D62728",
}

REFERENCE_LABELS = {
    wavelength_nm: f"{wavelength_nm} nm"
    for wavelength_nm in REFERENCE_WAVELENGTHS_NM
}


# ------------------------------------------------------------
# Plot appearance
# ------------------------------------------------------------

LINE_WIDTH = 1.2
MARKER_SIZE = 4.5
MARKER_EDGE_WIDTH = 0.55
ZERO_LINE_COLOUR = "0.55"
REFERENCE_LINE_COLOUR = "0.25"

FIGURE_OUTPUT_DIRECTORY = (
    SENSITIVITY_DIRECTORY
    / "figures"
)

FIGURE_OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)

PNG_DPI = 600

# In[2]:


# ============================================================
# LOAD COMPUTED SENSITIVITY RESULTS
# ============================================================

STUDY_NAMES = [
    "shells",
    "index_quarterwave",
    "medium",
    "illumination",
    "absorption",
]


def load_combined_study(
    root_directory,
    study_name,
):
    """
    Load the combined result of one sensitivity study.

    Preference:
      1. Parquet
      2. Pickle
      3. CSV
    """

    study_directory = (
        root_directory
        / study_name
    )

    candidate_paths = [
        study_directory
        / f"sensitivity_{study_name}.parquet",

        study_directory
        / f"sensitivity_{study_name}.pkl",

        study_directory
        / f"sensitivity_{study_name}.csv",
    ]

    for path in candidate_paths:

        if not path.exists():
            continue

        if path.suffix == ".parquet":
            dataframe = pd.read_parquet(path)

        elif path.suffix == ".pkl":
            dataframe = pd.read_pickle(path)

        else:
            dataframe = pd.read_csv(path)

        print(
            f"Loaded {study_name}: "
            f"{len(dataframe)} rows from {path.name}"
        )

        return dataframe

    raise FileNotFoundError(
        f"No combined result file was found for "
        f"{study_name!r} in:\n{study_directory}"
    )


sensitivity_results = {
    study_name: load_combined_study(
        SENSITIVITY_DIRECTORY,
        study_name,
    )
    for study_name in STUDY_NAMES
}


# ------------------------------------------------------------
# Load fixed-geometry index control
# ------------------------------------------------------------

FIXED_GEOMETRY_INDEX_PATH = (
    SENSITIVITY_DIRECTORY
    / "index_fixed_geometry_580.csv"
)

if FIXED_GEOMETRY_INDEX_PATH.exists():

    df_index_fixed_580 = pd.read_csv(
        FIXED_GEOMETRY_INDEX_PATH
    )

    print(
        "Loaded fixed-geometry index control:",
        len(df_index_fixed_580),
        "rows",
    )

else:

    df_index_fixed_580 = pd.DataFrame()

    print(
        "Warning: fixed-geometry 580 nm index file "
        "was not found."
    )


# ------------------------------------------------------------
# Assign convenient names
# ------------------------------------------------------------

df_shells = sensitivity_results["shells"].copy()
df_index_quarterwave = (
    sensitivity_results[
        "index_quarterwave"
    ].copy()
)
df_medium = sensitivity_results["medium"].copy()
df_illumination = (
    sensitivity_results[
        "illumination"
    ].copy()
)
df_absorption = (
    sensitivity_results[
        "absorption"
    ].copy()
)


# ------------------------------------------------------------
# Standardize numeric columns
# ------------------------------------------------------------

numeric_columns = [
    "reference_design_wavelength_nm",
    "parameter_value",
    "parameter_reference_value",
    "variation_factor",
    "reference_n_shells",
    "reference_n_A",
    "reference_n_B",
    "reference_k",
    "L",
    "a",
    "b",
    "C",
    "hue_deg",
    "eta_C",
    "eta_L",
    "d_opt",
    "delta_C",
    "delta_L",
    "delta_hue_deg",
    "relative_delta_C",
    "relative_delta_L",
    "delta_E_ab",
]

for dataframe in sensitivity_results.values():

    for column in numeric_columns:

        if column in dataframe.columns:

            dataframe[column] = pd.to_numeric(
                dataframe[column],
                errors="coerce",
            )


# Reassign after numeric conversion.
df_shells = sensitivity_results["shells"]
df_index_quarterwave = sensitivity_results[
    "index_quarterwave"
]
df_medium = sensitivity_results["medium"]
df_illumination = sensitivity_results[
    "illumination"
]
df_absorption = sensitivity_results[
    "absorption"
]


# ------------------------------------------------------------
# Basic validation
# ------------------------------------------------------------

required_analysis_columns = {
    "reference_design_wavelength_nm",
    "parameter_value",
    "variation_factor",
    "delta_C",
    "delta_L",
    "delta_hue_deg",
}

for study_name, dataframe in (
    sensitivity_results.items()
):

    missing = (
        required_analysis_columns
        - set(dataframe.columns)
    )

    if missing:
        raise KeyError(
            f"{study_name} is missing columns: "
            f"{sorted(missing)}"
        )


# ------------------------------------------------------------
# Dataset summary
# ------------------------------------------------------------

df_loading_summary = pd.DataFrame(
    [
        {
            "study": study_name,
            "rows": len(dataframe),
            "reference_designs": sorted(
                dataframe[
                    "reference_design_wavelength_nm"
                ]
                .dropna()
                .astype(int)
                .unique()
                .tolist()
            ),
            "parameter_min": dataframe[
                "parameter_value"
            ].min(),
            "parameter_max": dataframe[
                "parameter_value"
            ].max(),
        }
        for study_name, dataframe
        in sensitivity_results.items()
    ]
)

display(df_loading_summary)

# In[3]:


# ============================================================
# RECALCULATE LOCAL DIFFERENCES FROM EACH STUDY'S RERUN BASELINE
# ============================================================

def circular_difference_deg(
    values_deg,
    reference_deg,
):
    """
    Circular difference in the interval [-180, 180).
    """

    return (
        (
            np.asarray(
                values_deg,
                dtype=float,
            )
            - float(reference_deg)
            + 180.0
        )
        % 360.0
        - 180.0
    )


def add_rerun_baseline_differences(
    dataframe,
    *,
    baseline_selector,
):
    """
    Recalculate delta C, delta L, and wrapped delta hue relative
    to one explicitly identified rerun point per reference design.
    """

    output = dataframe.copy()

    output["delta_C_rerun"] = np.nan
    output["delta_L_rerun"] = np.nan
    output["delta_hue_deg_rerun"] = np.nan
    output["delta_d_opt_rerun"] = np.nan

    for reference_wavelength_nm, group in output.groupby(
        "reference_design_wavelength_nm"
    ):

        baseline_mask = baseline_selector(
            group
        )

        baseline_candidates = group.loc[
            baseline_mask
        ]

        if len(baseline_candidates) != 1:

            print(
                f"Warning: expected one rerun baseline for "
                f"{reference_wavelength_nm:.0f} nm, found "
                f"{len(baseline_candidates)}."
            )

            continue

        baseline = baseline_candidates.iloc[0]

        indices = group.index

        output.loc[
            indices,
            "delta_C_rerun",
        ] = (
            group["C"].to_numpy(dtype=float)
            - float(baseline["C"])
        )

        output.loc[
            indices,
            "delta_L_rerun",
        ] = (
            group["L"].to_numpy(dtype=float)
            - float(baseline["L"])
        )

        output.loc[
            indices,
            "delta_hue_deg_rerun",
        ] = circular_difference_deg(
            group["hue_deg"],
            baseline["hue_deg"],
        )

        if "d_opt" in group.columns:

            output.loc[
                indices,
                "delta_d_opt_rerun",
            ] = (
                group["d_opt"].to_numpy(
                    dtype=float
                )
                - float(baseline["d_opt"])
            )

    return output


# Shell, quarter-wave index, and medium studies use factor/value 1.
df_shells = add_rerun_baseline_differences(
    df_shells,
    baseline_selector=lambda group: np.isclose(
        group["variation_factor"],
        1.0,
    ),
)

df_index_quarterwave = (
    add_rerun_baseline_differences(
        df_index_quarterwave,
        baseline_selector=lambda group: np.isclose(
            group["variation_factor"],
            1.0,
        ),
    )
)

df_medium = add_rerun_baseline_differences(
    df_medium,
    baseline_selector=lambda group: np.isclose(
        group["parameter_value"],
        1.0,
    ),
)

# Complete hemispherical illumination is the 90 degree baseline.
df_illumination = add_rerun_baseline_differences(
    df_illumination,
    baseline_selector=lambda group: np.isclose(
        group["parameter_value"],
        90.0,
    ),
)

# Only absorbing references have a defined k/k0 = 1 baseline.
df_absorption_normalized = df_absorption.loc[
    pd.to_numeric(
        df_absorption[
            "reference_k"
        ],
        errors="coerce",
    )
    > 0.0
].copy()

df_absorption_normalized = (
    add_rerun_baseline_differences(
        df_absorption_normalized,
        baseline_selector=lambda group: np.isclose(
            group["variation_factor"],
            1.0,
        ),
    )
)


# Update dictionary with baseline-corrected copies.
sensitivity_results_corrected = {
    "shells": df_shells,
    "index_quarterwave": (
        df_index_quarterwave
    ),
    "medium": df_medium,
    "illumination": df_illumination,
    "absorption": (
        df_absorption_normalized
    ),
}

# Panel A: Local sensitivity curves

# In[4]:


# ============================================================
# PANEL A: LOCAL ONE-PARAMETER SENSITIVITY CURVES
#
# Rows:
#   Delta C*
#   Delta L*
#   Delta hue
#
# Columns:
#   N_shell / N_shell,0
#   (n_A/n_B) / (n_A/n_B)_0
#   n_medium / n_medium,0
#   k / k_0 for absorbing references
# ============================================================

PANEL_A_SAVE = True
PANEL_A_WIDTH_CM = 17.5
PANEL_A_HEIGHT_CM = 11.5

panel_a_columns = [
    {
        "name": "shells",
        "data": df_shells,
        "x_column": "variation_factor",
        "x_label": (
            r"$N_{\mathrm{shell}}/"
            r"N_{\mathrm{shell},0}$"
        ),
    },
    {
        "name": "index_quarterwave",
        "data": df_index_quarterwave,
        "x_column": "variation_factor",
        "x_label": (
            r"$(n_A/n_B)/(n_A/n_B)_0$"
        ),
    },
    {
        "name": "medium",
        "data": df_medium,
        "x_column": "variation_factor",
        "x_label": (
            r"$n_{\mathrm{m}}/n_{\mathrm{m},0}$"
        ),
    },
    {
        "name": "absorption",
        "data": df_absorption_normalized,
        "x_column": "variation_factor",
        "x_label": r"$k/k_0$",
    },
]

panel_a_rows = [
    {
        "column": "delta_C_rerun",
        "label": r"$\Delta C_{ab}^*$",
    },
    {
        "column": "delta_L_rerun",
        "label": r"$\Delta L^*$",
    },
    {
        "column": "delta_hue_deg_rerun",
        "label": r"$\Delta h_{ab}$ [$^\circ$]",
    },
]


cm_to_inches = 1.0 / 2.54

fig, axes = plt.subplots(
    3,
    4,
    figsize=(
        PANEL_A_WIDTH_CM * cm_to_inches,
        PANEL_A_HEIGHT_CM * cm_to_inches,
    ),
    sharex="col",
    constrained_layout=True,
)


for column_index, column_spec in enumerate(
    panel_a_columns
):

    dataframe = column_spec["data"]

    for row_index, row_spec in enumerate(
        panel_a_rows
    ):

        ax = axes[
            row_index,
            column_index,
        ]

        # Zero response
        ax.axhline(
            0.0,
            color=ZERO_LINE_COLOUR,
            linewidth=0.7,
            linestyle="-",
            zorder=0,
        )

        # Reference parameter value
        ax.axvline(
            1.0,
            color=REFERENCE_LINE_COLOUR,
            linewidth=0.8,
            linestyle="--",
            zorder=0,
        )

        for reference_wavelength_nm in (
            REFERENCE_WAVELENGTHS_NM
        ):

            group = dataframe.loc[
                np.isclose(
                    dataframe[
                        "reference_design_wavelength_nm"
                    ],
                    reference_wavelength_nm,
                )
            ].copy()

            if group.empty:
                continue

            group = (
                group
                .dropna(
                    subset=[
                        column_spec["x_column"],
                        row_spec["column"],
                    ]
                )
                .sort_values(
                    column_spec["x_column"]
                )
            )

            if group.empty:
                continue

            ax.plot(
                group[
                    column_spec["x_column"]
                ],
                group[
                    row_spec["column"]
                ],
                color=REFERENCE_COLOURS[
                    reference_wavelength_nm
                ],
                marker=REFERENCE_MARKERS[
                    reference_wavelength_nm
                ],
                markersize=MARKER_SIZE,
                markeredgecolor="black",
                markeredgewidth=(
                    MARKER_EDGE_WIDTH
                ),
                linewidth=LINE_WIDTH,
                label=REFERENCE_LABELS[
                    reference_wavelength_nm
                ],
                zorder=3,
            )

        if column_index == 0:
            ax.set_ylabel(
                row_spec["label"]
            )

        if row_index == 2:
            ax.set_xlabel(
                column_spec["x_label"]
            )

        ax.grid(
            False,
            which="both",
            axis="both",
        )


# ------------------------------------------------------------
# Column headers
# ------------------------------------------------------------

column_titles = [
    "Shell number",
    "Index ratio\nquarter-wave geometry",
    "Surrounding medium",
    "Absorption\nabsorbing references only",
]

for column_index, title in enumerate(
    column_titles
):
    axes[0, column_index].set_title(
        title,
        fontsize=9,
    )


# ------------------------------------------------------------
# Figure-level legend
# ------------------------------------------------------------

reference_handles = [
    Line2D(
        [0],
        [0],
        color=REFERENCE_COLOURS[
            wavelength_nm
        ],
        marker=REFERENCE_MARKERS[
            wavelength_nm
        ],
        markeredgecolor="black",
        markeredgewidth=(
            MARKER_EDGE_WIDTH
        ),
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE,
        label=REFERENCE_LABELS[
            wavelength_nm
        ],
    )
    for wavelength_nm
    in REFERENCE_WAVELENGTHS_NM
]

fig.legend(
    handles=reference_handles,
    loc="outside upper center",
    ncol=4,
    frameon=False,
    title=r"Reference $\lambda_0$",
    fontsize=8,
    title_fontsize=8,
    handlelength=1.4,
    columnspacing=0.9,
)


if PANEL_A_SAVE:

    figure_path_png = (
        FIGURE_OUTPUT_DIRECTORY
        / "Figure3A_local_sensitivity.png"
    )

    figure_path_pdf = (
        FIGURE_OUTPUT_DIRECTORY
        / "Figure3A_local_sensitivity.pdf"
    )

    fig.savefig(
        figure_path_png,
        dpi=PNG_DPI,
        transparent=True,
        bbox_inches=None,
    )

    fig.savefig(
        figure_path_pdf,
        transparent=True,
        bbox_inches=None,
    )

    print("Saved:", figure_path_png)
    print("Saved:", figure_path_pdf)


plt.show()

# Code cell 5: Optional illumination sensitivity panel

# In[5]:


# ============================================================
# OPTIONAL PANEL: ILLUMINATION HALF-ANGLE SENSITIVITY
# ============================================================

ILLUMINATION_PANEL_SAVE = True
ILLUMINATION_WIDTH_CM = 8.0
ILLUMINATION_HEIGHT_CM = 10.0

fig, axes = plt.subplots(
    3,
    1,
    figsize=(
        ILLUMINATION_WIDTH_CM
        * cm_to_inches,
        ILLUMINATION_HEIGHT_CM
        * cm_to_inches,
    ),
    sharex=True,
    constrained_layout=True,
)

illumination_rows = [
    (
        "delta_C_rerun",
        r"$\Delta C_{ab}^*$",
    ),
    (
        "delta_L_rerun",
        r"$\Delta L^*$",
    ),
    (
        "delta_hue_deg_rerun",
        r"$\Delta h_{ab}$ [$^\circ$]",
    ),
]

for ax, (
    response_column,
    y_label,
) in zip(
    axes,
    illumination_rows,
):

    ax.axhline(
        0.0,
        color=ZERO_LINE_COLOUR,
        linewidth=0.7,
    )

    ax.axvline(
        90.0,
        color=REFERENCE_LINE_COLOUR,
        linewidth=0.8,
        linestyle="--",
    )

    for wavelength_nm in (
        REFERENCE_WAVELENGTHS_NM
    ):

        group = df_illumination.loc[
            np.isclose(
                df_illumination[
                    "reference_design_wavelength_nm"
                ],
                wavelength_nm,
            )
        ].sort_values(
            "parameter_value"
        )

        ax.plot(
            group["parameter_value"],
            group[response_column],
            color=REFERENCE_COLOURS[
                wavelength_nm
            ],
            marker=REFERENCE_MARKERS[
                wavelength_nm
            ],
            markeredgecolor="black",
            markeredgewidth=(
                MARKER_EDGE_WIDTH
            ),
            markersize=MARKER_SIZE,
            linewidth=LINE_WIDTH,
            label=REFERENCE_LABELS[
                wavelength_nm
            ],
        )

    ax.set_ylabel(y_label)
    ax.grid(False)

axes[-1].set_xlabel(
    r"Illumination half-angle [$^\circ$]"
)

fig.legend(
    handles=reference_handles,
    loc="outside upper center",
    ncol=2,
    frameon=False,
    fontsize=8,
    title=r"Reference $\lambda_0$",
    title_fontsize=8,
)


if ILLUMINATION_PANEL_SAVE:

    path_png = (
        FIGURE_OUTPUT_DIRECTORY
        / "Figure3_illumination_sensitivity.png"
    )

    path_pdf = (
        FIGURE_OUTPUT_DIRECTORY
        / "Figure3_illumination_sensitivity.pdf"
    )

    fig.savefig(
        path_png,
        dpi=PNG_DPI,
        transparent=True,
        bbox_inches=None,
    )

    fig.savefig(
        path_pdf,
        transparent=True,
        bbox_inches=None,
    )

    print("Saved:", path_png)
    print("Saved:", path_pdf)


plt.show()

# In[6]:


# ============================================================
# EXTRACT PARAMETER CHANGES REQUIRED TO REACH COLOUR THRESHOLDS
#
# Thresholds:
#   |Delta C_ab*| = 5
#   |Delta L*|    = 5
#   |Delta h_ab|  = 5 degrees
#
# Parameter-change units:
#   shells                 : percent
#   index ratio            : percent
#   surrounding index      : absolute Delta n_m
#   existing absorption   : percent change in k/k0
#   introduced absorption : absolute 10^3 k
#   illumination          : absolute angle change in degrees
#
# Decreasing and increasing parameter branches are analyzed
# independently.
# ============================================================

import numpy as np
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

THRESHOLDS = {
    "C": 5.0,
    "L": 5.0,
    "h": 5.0,
}

THRESHOLD_LABELS = {
    "C": r"$|\Delta C_{ab}^*|=5$",
    "L": r"$|\Delta L^*|=5$",
    "h": r"$|\Delta h_{ab}|=5^\circ$",
}


# ============================================================
# HELPERS
# ============================================================

def circular_difference_deg(values_deg, reference_deg):
    """
    Calculate circular angular differences in degrees.

    The result is wrapped into the interval [-180, 180).
    """
    values_deg = np.asarray(values_deg, dtype=float)
    reference_deg = float(reference_deg)

    return (
        (values_deg - reference_deg + 180.0) % 360.0
        - 180.0
    )


def average_duplicate_coordinates(parameter, response):
    """
    Average response values occurring at identical parameter
    coordinates.

    NaN values in either parameter or response are removed.
    The returned arrays are sorted by parameter.
    """
    dataframe = pd.DataFrame(
        {
            "parameter": np.asarray(parameter, dtype=float),
            "response": np.asarray(response, dtype=float),
        }
    ).dropna()

    if dataframe.empty:
        return (
            np.array([], dtype=float),
            np.array([], dtype=float),
        )

    dataframe = (
        dataframe
        .groupby("parameter", as_index=False)["response"]
        .mean()
        .sort_values("parameter")
    )

    return (
        dataframe["parameter"].to_numpy(dtype=float),
        dataframe["response"].to_numpy(dtype=float),
    )


def determine_actual_baseline(
    dataframe,
    parameter_column,
    reference_parameter,
):
    """
    Return the calculated row whose parameter value is closest
    to the nominal reference parameter.

    Rows with invalid parameter values are ignored.
    """
    parameter = pd.to_numeric(
        dataframe[parameter_column],
        errors="coerce",
    ).to_numpy(dtype=float)

    valid_mask = np.isfinite(parameter)

    if not np.any(valid_mask):
        raise ValueError(
            f"No finite values found in parameter column "
            f"'{parameter_column}'."
        )

    valid_indices = np.flatnonzero(valid_mask)

    baseline_index = valid_indices[
        np.argmin(
            np.abs(
                parameter[valid_mask]
                - float(reference_parameter)
            )
        )
    ]

    return dataframe.iloc[int(baseline_index)]


def prepare_full_response_data(
    study_key,
    dataframe,
    wavelength_nm,
):
    """
    Prepare the full calculated response curve for one study and
    one reference wavelength.

    Returns
    -------
    dict or None
        Dictionary containing the prepared group, parameter
        information, baseline coordinate, display conversion,
        and study metadata.

        None is returned if the requested wavelength is absent
        or if the study is not applicable to the reference
        absorption state.
    """

    if dataframe is None or dataframe.empty:
        return None

    if "reference_design_wavelength_nm" not in dataframe.columns:
        raise KeyError(
            "The dataframe must contain the column "
            "'reference_design_wavelength_nm'."
        )

    group = dataframe.loc[
        np.isclose(
            pd.to_numeric(
                dataframe["reference_design_wavelength_nm"],
                errors="coerce",
            ),
            float(wavelength_nm),
        )
    ].copy()

    if group.empty:
        return None

    # --------------------------------------------------------
    # Determine reference absorption coefficient
    # --------------------------------------------------------

    if "reference_k" not in group.columns:
        reference_k = 0.0
    else:
        reference_k_values = pd.to_numeric(
            group["reference_k"],
            errors="coerce",
        ).dropna()

        if reference_k_values.empty:
            reference_k = 0.0
        else:
            reference_k = float(reference_k_values.iloc[0])

    # --------------------------------------------------------
    # Define parameter coordinate and reference
    # --------------------------------------------------------

    parameter_column_internal = "_analysis_parameter"

    if study_key == "shells":

        required_column = "variation_factor"

        if required_column not in group.columns:
            raise KeyError(
                f"Study '{study_key}' requires column "
                f"'{required_column}'."
            )

        group[parameter_column_internal] = pd.to_numeric(
            group[required_column],
            errors="coerce",
        )

        reference_parameter = 1.0

        display_unit = "%"
        display_factor = 100.0

        parameter_label = (
            r"$N_{\mathrm{shell}}/"
            r"N_{\mathrm{shell},0}$"
        )

        discrete = True
        applicability = "all"

    elif study_key == "index":

        required_column = "variation_factor"

        if required_column not in group.columns:
            raise KeyError(
                f"Study '{study_key}' requires column "
                f"'{required_column}'."
            )

        group[parameter_column_internal] = pd.to_numeric(
            group[required_column],
            errors="coerce",
        )

        reference_parameter = 1.0

        display_unit = "%"
        display_factor = 100.0

        parameter_label = (
            r"$(n_A/n_B)/(n_A/n_B)_0$"
        )

        discrete = False
        applicability = "all"

    elif study_key == "medium":

        required_column = "parameter_value"

        if required_column not in group.columns:
            raise KeyError(
                f"Study '{study_key}' requires column "
                f"'{required_column}'."
            )

        group[parameter_column_internal] = pd.to_numeric(
            group[required_column],
            errors="coerce",
        )

        # The surrounding-medium index is an absolute quantity,
        # so the nominal reference is n_m = 1.0 in the original
        # analysis.
        reference_parameter = 1.0

        display_unit = r"$\Delta n_{\mathrm{m}}$"
        display_factor = 1.0

        parameter_label = r"$n_{\mathrm{m}}$"

        discrete = False
        applicability = "all"

    elif study_key == "illumination":

        required_column = "parameter_value"

        if required_column not in group.columns:
            raise KeyError(
                f"Study '{study_key}' requires column "
                f"'{required_column}'."
            )

        group[parameter_column_internal] = pd.to_numeric(
            group[required_column],
            errors="coerce",
        )

        reference_parameter = 90.0

        display_unit = r"$^\circ$"
        display_factor = 1.0

        parameter_label = "Illumination half-angle"

        discrete = False
        applicability = "all"

    elif study_key == "absorption_absolute":

        k_values = pd.to_numeric(
            group["parameter_value"],
            errors="coerce",
        ).to_numpy(dtype=float)

        # Use the physical extinction coefficient itself as the
        # analysis coordinate. The reference can be zero or nonzero.
        group[
            "_analysis_parameter"
        ] = k_values

        parameter_column_internal = (
            "_analysis_parameter"
        )

        reference_parameter = reference_k

        # Display the required absolute change as 10^3 |Delta k|.
        display_unit = r"$10^3|\Delta k|$"
        display_factor = 1e3

        parameter_label = r"$k$"

        discrete = False
        applicability = "all"

    # elif study_key == "absorption_existing":

    #     # This study only applies when the reference system
    #     # already has absorption.
    #     if not np.isfinite(reference_k) or reference_k <= 0.0:
    #         return None

    #     required_column = "parameter_value"

    #     if required_column not in group.columns:
    #         raise KeyError(
    #             f"Study '{study_key}' requires column "
    #             f"'{required_column}'."
    #         )

    #     k_values = pd.to_numeric(
    #         group[required_column],
    #         errors="coerce",
    #     )

    #     group[parameter_column_internal] = (
    #         k_values / reference_k
    #     )

    #     reference_parameter = 1.0

    #     display_unit = "%"
    #     display_factor = 100.0

    #     parameter_label = r"$k/k_0$"

    #     discrete = False
    #     applicability = "absorbing"

    # elif study_key == "absorption_introduced":

    #     # This study only applies when the reference system
    #     # is non-absorbing.
    #     if not np.isfinite(reference_k) or reference_k > 0.0:
    #         return None

    #     required_column = "parameter_value"

    #     if required_column not in group.columns:
    #         raise KeyError(
    #             f"Study '{study_key}' requires column "
    #             f"'{required_column}'."
    #         )

    #     k_values = pd.to_numeric(
    #         group[required_column],
    #         errors="coerce",
    #     )

    #     group[parameter_column_internal] = k_values

    #     reference_parameter = 0.0

    #     display_unit = r"$10^3 k$"
    #     display_factor = 1e3

    #     parameter_label = "Introduced absorption"

    #     discrete = False
    #     applicability = "nonabsorbing"

    else:

        raise ValueError(
            f"Unknown study key: {study_key}"
        )

    # --------------------------------------------------------
    # Remove rows with invalid parameter values
    # --------------------------------------------------------

    group = group.loc[
        np.isfinite(
            pd.to_numeric(
                group[parameter_column_internal],
                errors="coerce",
            )
        )
    ].copy()

    if group.empty:
        return None

    # --------------------------------------------------------
    # Determine the actual calculated baseline
    #
    # Important:
    # The baseline is not necessarily exactly the nominal
    # reference parameter. It is the calculated data point
    # closest to that reference.
    # --------------------------------------------------------

    baseline = determine_actual_baseline(
        group,
        parameter_column_internal,
        reference_parameter,
    )

    # --------------------------------------------------------
    # Validate colour columns
    # --------------------------------------------------------

    required_colour_columns = [
        "C",
        "L",
        "hue_deg",
    ]

    missing_colour_columns = [
        column
        for column in required_colour_columns
        if column not in group.columns
    ]

    if missing_colour_columns:
        raise KeyError(
            "Missing required colour column(s): "
            + ", ".join(missing_colour_columns)
        )

    baseline_C = pd.to_numeric(
        pd.Series([baseline["C"]]),
        errors="coerce",
    ).iloc[0]

    baseline_L = pd.to_numeric(
        pd.Series([baseline["L"]]),
        errors="coerce",
    ).iloc[0]

    baseline_h = pd.to_numeric(
        pd.Series([baseline["hue_deg"]]),
        errors="coerce",
    ).iloc[0]

    if not (
        np.isfinite(baseline_C)
        and np.isfinite(baseline_L)
        and np.isfinite(baseline_h)
    ):
        raise ValueError(
            f"Invalid baseline colour values for study "
            f"'{study_key}', wavelength {wavelength_nm} nm."
        )

    # --------------------------------------------------------
    # Calculate colour changes relative to the actual baseline
    # --------------------------------------------------------

    group["_delta_C"] = (
        pd.to_numeric(
            group["C"],
            errors="coerce",
        )
        - float(baseline_C)
    )

    group["_delta_L"] = (
        pd.to_numeric(
            group["L"],
            errors="coerce",
        )
        - float(baseline_L)
    )

    group["_delta_h"] = circular_difference_deg(
        pd.to_numeric(
            group["hue_deg"],
            errors="coerce",
        ),
        float(baseline_h),
    )

    # --------------------------------------------------------
    # Sort by parameter
    # --------------------------------------------------------

    group = group.sort_values(
        parameter_column_internal
    ).reset_index(drop=True)

    return {
        "group": group,
        "parameter_column": parameter_column_internal,
        "reference_parameter": reference_parameter,
        "display_unit": display_unit,
        "display_factor": display_factor,
        "parameter_label": parameter_label,
        "discrete": discrete,
        "applicability": applicability,
    }


def threshold_crossing_on_branch(
    parameter,
    response,
    *,
    reference_parameter,
    threshold,
    direction,
    discrete=False,
):
    """
    Find the first parameter point, moving away from the
    reference parameter in the requested direction, at which
    |response| reaches the specified threshold.

    Parameters
    ----------
    parameter : array-like
        Parameter coordinates.

    response : array-like
        Signed response values, e.g. Delta C, Delta L, or
        circular Delta h.

    reference_parameter : float
        Nominal parameter value defining the baseline.

    threshold : float
        Absolute response threshold.

    direction : {"decrease", "increase"}
        Parameter branch to analyze.

    discrete : bool, default=False
        If True, no interpolation is performed. The first
        physically calculated point crossing the threshold
        is retained.

    Returns
    -------
    dict
        Contains:
            threshold_reached
            crossing_parameter
            parameter_change
            response_sign
            maximum_tested_change
    """

    threshold = float(threshold)
    reference_parameter = float(reference_parameter)

    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError(
            "threshold must be a finite non-negative number."
        )

    # --------------------------------------------------------
    # Clean and average duplicate coordinates
    # --------------------------------------------------------

    parameter, response = average_duplicate_coordinates(
        parameter,
        response,
    )

    if parameter.size == 0:
        return {
            "threshold_reached": False,
            "crossing_parameter": np.nan,
            "parameter_change": np.nan,
            "response_sign": np.nan,
            "maximum_tested_change": np.nan,
        }

    # --------------------------------------------------------
    # Select requested branch
    # --------------------------------------------------------

    if direction == "decrease":

        branch_mask = (
            parameter
            <= reference_parameter + 1e-12
        )

    elif direction == "increase":

        branch_mask = (
            parameter
            >= reference_parameter - 1e-12
        )

    else:

        raise ValueError(
            "direction must be 'decrease' or 'increase'."
        )

    branch_parameter = parameter[branch_mask]
    branch_response = response[branch_mask]

    if branch_parameter.size == 0:
        return {
            "threshold_reached": False,
            "crossing_parameter": np.nan,
            "parameter_change": np.nan,
            "response_sign": np.nan,
            "maximum_tested_change": np.nan,
        }

    # --------------------------------------------------------
    # Sort by increasing distance from the reference
    #
    # This ensures that the first threshold crossing found is
    # the first one encountered while moving away from the
    # baseline.
    # --------------------------------------------------------

    distance = np.abs(
        branch_parameter
        - reference_parameter
    )

    order = np.argsort(distance)

    branch_parameter = branch_parameter[order]
    branch_response = branch_response[order]
    distance = distance[order]

    maximum_tested_change = float(
        np.nanmax(distance)
    )

    # --------------------------------------------------------
    # Determine whether the threshold is reached
    # --------------------------------------------------------

    absolute_response = np.abs(branch_response)

    crossing_indices = np.flatnonzero(
        np.isfinite(absolute_response)
        & (
            absolute_response
            >= threshold
        )
    )

    if len(crossing_indices) == 0:

        # Sign of the response at the furthest tested parameter value.
        final_response = float(
            branch_response[-1]
        )

        return {
            "threshold_reached": False,
            "crossing_parameter": np.nan,
            "parameter_change": np.nan,
            "response_sign": float(
                np.sign(final_response)
            ),
            "maximum_tested_change": (
                maximum_tested_change
            ),
            "maximum_tested_response": (
                final_response
            ),
        }

    crossing_index = int(
        crossing_indices[0]
    )

    # --------------------------------------------------------
    # Discrete parameters:
    #
    # Use the first physically calculated point crossing the
    # threshold. No interpolation is performed.
    # --------------------------------------------------------

    if discrete or crossing_index == 0:

        crossing_parameter = float(
            branch_parameter[crossing_index]
        )

        crossing_response = float(
            branch_response[crossing_index]
        )

    # --------------------------------------------------------
    # Continuous parameters:
    #
    # Linearly interpolate between the last point below the
    # threshold and the first point above the threshold.
    # --------------------------------------------------------

    else:

        parameter_0 = float(
            branch_parameter[
                crossing_index - 1
            ]
        )

        parameter_1 = float(
            branch_parameter[
                crossing_index
            ]
        )

        response_0 = float(
            absolute_response[
                crossing_index - 1
            ]
        )

        response_1 = float(
            absolute_response[
                crossing_index
            ]
        )

        if np.isclose(
            response_1,
            response_0,
        ):
            interpolation_fraction = 1.0

        else:
            interpolation_fraction = (
                (threshold - response_0)
                / (response_1 - response_0)
            )

            interpolation_fraction = float(
                np.clip(
                    interpolation_fraction,
                    0.0,
                    1.0,
                )
            )

        crossing_parameter = (
            parameter_0
            + interpolation_fraction
            * (
                parameter_1
                - parameter_0
            )
        )

        # Interpolate the signed response as well, so that the
        # sign reported in the output corresponds to the
        # interpolated crossing.
        signed_response_0 = float(
            branch_response[
                crossing_index - 1
            ]
        )

        signed_response_1 = float(
            branch_response[
                crossing_index
            ]
        )

        crossing_response = (
            signed_response_0
            + interpolation_fraction
            * (
                signed_response_1
                - signed_response_0
            )
        )

    # --------------------------------------------------------
    # Return crossing information
    # --------------------------------------------------------

    return {
        "threshold_reached": True,
        "crossing_parameter": float(
            crossing_parameter
        ),
        "parameter_change": float(
            abs(
                crossing_parameter
                - reference_parameter
            )
        ),
        "response_sign": float(
            np.sign(crossing_response)
        ),
        "maximum_tested_change": (
            maximum_tested_change
        ),
        "maximum_tested_response": np.nan,
    }


# ============================================================
# STUDY DEFINITIONS
# ============================================================

robustness_studies = [
    {
        "key": "shells",
        "label": (
            r"$N_{\mathrm{shell}}/"
            r"N_{\mathrm{shell},0}$"
        ),
        "data": df_shells,
    },
    {
        "key": "index",
        "label": (
            r"$(n_A/n_B)/(n_A/n_B)_0$"
        ),
        "data": df_index_quarterwave,
    },
    {
        "key": "medium",
        "label": r"$n_{\mathrm{m}}$",
        "data": df_medium,
    },
    {
        "key": "absorption_absolute",
        "label": r"Absorption $k$",
        "data": df_absorption,
    },
    {
        "key": "illumination",
        "label": (
            "Illumination\n"
            "half-angle"
        ),
        "data": df_illumination,
    },
]


# ============================================================
# CALCULATE THRESHOLD TABLE
# ============================================================

threshold_records = []

for study in robustness_studies:

    for wavelength_nm in REFERENCE_WAVELENGTHS_NM:

        prepared = prepare_full_response_data(
            study_key=study["key"],
            dataframe=study["data"],
            wavelength_nm=wavelength_nm,
        )

        if prepared is None:
            continue

        group = prepared["group"]

        parameter = pd.to_numeric(
            group[
                prepared["parameter_column"]
            ],
            errors="coerce",
        ).to_numpy(dtype=float)

        for response_key, threshold in THRESHOLDS.items():

            response_column = {
                "C": "_delta_C",
                "L": "_delta_L",
                "h": "_delta_h",
            }[response_key]

            response = pd.to_numeric(
                group[response_column],
                errors="coerce",
            ).to_numpy(dtype=float)

            for direction in (
                "decrease",
                "increase",
            ):
                # A nonabsorbing reference cannot undergo a physical decrease in k.
                if (
                    study["key"] == "absorption_absolute"
                    and np.isclose(
                        prepared["reference_parameter"],
                        0.0,
                    )
                    and direction == "decrease"
                ):
                    continue

                crossing = threshold_crossing_on_branch(
                    parameter,
                    response,
                    reference_parameter=(
                        prepared[
                            "reference_parameter"
                        ]
                    ),
                    threshold=threshold,
                    direction=direction,
                    discrete=prepared[
                        "discrete"
                    ],
                )

                # ------------------------------------------------
                # Convert the parameter change to the requested
                # display units.
                #
                # IMPORTANT:
                # display_factor converts the absolute parameter
                # change, not the parameter itself.
                # ------------------------------------------------

                if crossing["threshold_reached"]:

                    display_change = (
                        crossing["parameter_change"]
                        * prepared["display_factor"]
                    )

                else:

                    display_change = (
                        crossing["maximum_tested_change"]
                        * prepared["display_factor"]
                    )

                threshold_records.append(
                    {
                        "study": study["key"],

                        "study_label": (
                            study["label"]
                        ),

                        "reference_design_wavelength_nm": (
                            wavelength_nm
                        ),

                        "response": response_key,

                        "threshold": threshold,

                        "direction": direction,

                        "threshold_reached": (
                            crossing[
                                "threshold_reached"
                            ]
                        ),

                        "crossing_parameter": (
                            crossing[
                                "crossing_parameter"
                            ]
                        ),

                        "parameter_change": (
                            crossing[
                                "parameter_change"
                            ]
                        ),

                        "display_change": (
                            display_change
                        ),

                        "display_unit": (
                            prepared[
                                "display_unit"
                            ]
                        ),

                        "response_sign": (
                            crossing[
                                "response_sign"
                            ]
                        ),

                        "maximum_tested_change": (
                            crossing[
                                "maximum_tested_change"
                            ]
                        ),

                        "discrete_parameter": (
                            prepared["discrete"]
                        ),
                        "reference_parameter": (
                            prepared[
                                "reference_parameter"
                            ]
                        ),
                    }
                )


# ============================================================
# CREATE AND DISPLAY RESULT TABLE
# ============================================================

df_threshold_robustness = pd.DataFrame(
    threshold_records
)

if df_threshold_robustness.empty:

    print(
        "No threshold results were generated. "
        "Check the input dataframes and "
        "REFERENCE_WAVELENGTHS_NM."
    )

else:

    df_threshold_robustness = (
        df_threshold_robustness
        .sort_values(
            [
                "study",
                "response",
                "reference_design_wavelength_nm",
                "direction",
            ]
        )
        .reset_index(drop=True)
    )

    display(
        df_threshold_robustness
    )

# In[10]:


# ============================================================
# PANEL A: THRESHOLD-BASED ROBUSTNESS BARS
#
# Bar length:
#   parameter change required to reach the response threshold
#
# Blue:
#   parameter decrease
#
# Orange:
#   parameter increase
#
# Open endpoint:
#   threshold not reached in the investigated parameter range
#
# Sign inside bar:
#   sign of Delta C*, Delta L*, or Delta h at the crossing
# ============================================================

from matplotlib.lines import Line2D
from matplotlib.patches import Patch


# ============================================================
# SETTINGS
# ============================================================

PANEL_A_WIDTH_CM = 15.92
PANEL_A_HEIGHT_CM = 15.92

PANEL_A_SAVE = True

DECREASE_COLOUR = "#3182BD"
INCREASE_COLOUR = "#F28E2B"

BAR_HEIGHT = 0.55
WAVELENGTH_SPACING = 0.72
ROW_GROUP_GAP = 0.75

ANNOTATE_VALUES = True
VALUE_FONT_SIZE = 6.5

PANEL_A_BASENAME = (
    "Figure3A_threshold_robustness"
)

# ============================================================
# ADDITIONAL DISPLAY SETTINGS
# ============================================================

# Show the coloured reference-wavelength markers at x = 0.
SHOW_REFERENCE_WAVELENGTH_MARKERS = False

# Show the reference-wavelength legend below the figure.
SHOW_REFERENCE_WAVELENGTH_LEGEND = False

# Annotate only short bars, for which reading the value directly
# from the axis is difficult.
ANNOTATE_ONLY_SMALL_BARS = True

# The threshold is interpreted in the displayed unit of each row:
#   shells and index ratio       -> percent
#   surrounding medium          -> Delta n_m
#   existing absorption         -> percent
#   introduced absorption       -> 10^3 k
#   illumination                -> degrees
SMALL_BAR_ANNOTATION_THRESHOLDS = {
    "shells": 50,
    "index": 15.0,
    "medium": 0.2,
    # Display units are 10^3 |Delta k|.
    "absorption_absolute": 15.0,
    # "absorption_introduced": 2.0,
    "illumination": 15.0,
}

# If False, all bars are annotated regardless of their size.
ANNOTATE_BAR_VALUES = True

# Distance between the parameter-group label and the y tick labels.
PARAMETER_LABEL_PAD = 10

# Ignore branches with no actual available parameter change.
ZERO_BRANCH_TOLERANCE = 1e-12

# Annotation spacing in points rather than data-axis units.
# This remains visually consistent across differently scaled rows.
ANNOTATION_OFFSET_POINTS = 3.0

# ============================================================
# ROW DEFINITIONS
# ============================================================

row_definitions = [
    {
        "study": "shells",
        "label": (
            r"$N_{\mathrm{shell}}/"
            r"N_{\mathrm{shell},0}$"
            "\n[%]"
        ),
    },
    {
        "study": "index",
        "label": (
            r"$(n_A/n_B)/(n_A/n_B)_0$"
            "\n[%]"
        ),
    },
    {
        "study": "medium",
        "label": (
            r"$\Delta n_{\mathrm{m}}$"
        ),
    },
    {
        "study": "absorption_absolute",
        "label": (
            r"Extinction"
            "\n"
            r"coefficient"
            "\n"
            r"$10^3|\Delta k|$"
        ),
    },
    {
        "study": "illumination",
        "label": (
            r"$|\Delta\theta_{\mathrm{ill}}|$"
            "\n[$^\circ$]"
        ),
    },
]


# ============================================================
# ROW POSITIONS
# ============================================================

row_positions = {}
group_centres = {}

current_y = 0.0

for row_index, row_definition in enumerate(
    row_definitions
):

    positions = []

    for wavelength_nm in REFERENCE_WAVELENGTHS_NM:

        row_positions[
            (
                row_definition["study"],
                wavelength_nm,
            )
        ] = current_y

        positions.append(current_y)

        current_y += WAVELENGTH_SPACING

    group_centres[
        row_definition["study"]
    ] = float(
        np.mean(positions)
    )

    if row_index < len(row_definitions) - 1:
        current_y += ROW_GROUP_GAP


# ============================================================
# FIGURE
# ============================================================

cm_to_inches = 1.0 / 2.54

fig, axes = plt.subplots(
    len(row_definitions),
    3,
    figsize=(
        PANEL_A_WIDTH_CM * cm_to_inches,
        PANEL_A_HEIGHT_CM * cm_to_inches,
    ),
    constrained_layout=True,
)


column_responses = [
    "C",
    "L",
    "h",
]


# ============================================================
# DRAW BARS
# ============================================================

def add_branch_annotation(
    ax,
    endpoint_x,
    y_position,
    magnitude,
    direction,
    response_sign,
    colour,
    show_value,
    threshold_reached,
    near_zero,
):
    """Add exactly one value/response annotation per branch."""

    if not np.isfinite(response_sign):
        arrow = "?"
    elif response_sign > 0:
        arrow = "\u2191"
    elif response_sign < 0:
        arrow = "\u2193"
    else:
        arrow = "\u2192"

    if show_value:
        prefix = "" if threshold_reached else ">"
        text = f"{prefix}{magnitude:.2g} {arrow}"
    else:
        text = arrow

    # Position text outside the branch endpoint.
    if direction == "decrease":
        dx = -4.0
        ha = "right"
    else:
        dx = 4.0
        ha = "left"

    # Separate opposing annotations when both endpoints are close
    # to the central x = 0 line.
    if near_zero:
        # Place both annotations slightly above the endpoint.
        # Horizontal alignment already separates decrease and increase.
        dy = 5.0
    else:
        dy = 0.0

    ax.annotate(
        text,
        xy=(endpoint_x, y_position),
        xytext=(dx, dy),
        textcoords="offset points",
        ha=ha,
        va="center",
        fontsize=VALUE_FONT_SIZE,
        color=colour,
        annotation_clip=False,
        zorder=8,
    )


for plot_row, row_definition in enumerate(
    row_definitions
):

    study_name = row_definition["study"]

    # --------------------------------------------------------
    # Select this study from the threshold-result dataframe
    # --------------------------------------------------------

    study_table = df_threshold_robustness.loc[
        df_threshold_robustness["study"]
        == study_name
    ].copy()

    # --------------------------------------------------------
    # Draw the three response columns
    # --------------------------------------------------------

    for plot_column, response_key in enumerate(
        column_responses
    ):

        ax = axes[
            plot_row,
            plot_column,
        ]

        # ----------------------------------------------------
        # Reference line
        # ----------------------------------------------------

        ax.axvline(
            0.0,
            color="black",
            linewidth=0.8,
            zorder=1,
        )

        # ----------------------------------------------------
        # Select this colour-response column
        # ----------------------------------------------------

        response_table = study_table.loc[
            study_table["response"]
            == response_key
        ]

        # ----------------------------------------------------
        # Determine x-axis limit from finite display changes
        #
        # Both reached and unreached thresholds are included.
        # ----------------------------------------------------

        if response_table.empty:

            axis_limit = 1.0

        else:

            finite_display_values = (
                pd.to_numeric(
                    response_table[
                        "display_change"
                    ],
                    errors="coerce",
                )
                .replace(
                    [np.inf, -np.inf],
                    np.nan,
                )
                .dropna()
            )

            if finite_display_values.empty:
                axis_limit = 1.0
            else:
                axis_limit = max(
                    float(finite_display_values.abs().max()) * 1.35,
                    1e-6,
                )

        # ----------------------------------------------------
        # Draw one pair of bars for every wavelength
        # ----------------------------------------------------

        for wavelength_index, wavelength_nm in enumerate(
            REFERENCE_WAVELENGTHS_NM
        ):

            y_position = (
                row_positions[
                    (
                        study_name,
                        wavelength_nm,
                    )
                ]
            )

            wavelength_table = response_table.loc[
                np.isclose(
                    pd.to_numeric(
                        response_table[
                            "reference_design_wavelength_nm"
                        ],
                        errors="coerce",
                    ),
                    float(wavelength_nm),
                )
            ]

            # ------------------------------------------------
            # Draw decrease and increase branches
            # ------------------------------------------------

            for direction in (
                "decrease",
                "increase",
            ):

                row = wavelength_table.loc[
                    wavelength_table["direction"]
                    == direction
                ]

                if row.empty:
                    continue

                # There should normally be exactly one row for
                # each wavelength/direction combination. Use the
                # first one if duplicates are present.
                result = row.iloc[0]

                magnitude = pd.to_numeric(
                    pd.Series(
                        [result["display_change"]]
                    ),
                    errors="coerce",
                ).iloc[0]

                if not np.isfinite(magnitude):
                    continue

                magnitude = float(
                    abs(magnitude)
                )
                # Do not draw or annotate an unavailable branch represented by
                # zero parameter change.
                if magnitude <= ZERO_BRANCH_TOLERANCE:
                    continue

                # ------------------------------------------------
                # Convert the magnitude to a signed x coordinate
                # ------------------------------------------------

                if direction == "decrease":

                    signed_width = -magnitude
                    bar_colour = DECREASE_COLOUR

                else:

                    signed_width = magnitude
                    bar_colour = INCREASE_COLOUR

                reached = bool(result["threshold_reached"])
                endpoint_x = signed_width

                response_sign = pd.to_numeric(
                    pd.Series([result["response_sign"]]),
                    errors="coerce",
                ).iloc[0]

                annotation_threshold = (
                    SMALL_BAR_ANNOTATION_THRESHOLDS.get(
                        study_name,
                        np.inf,
                    )
                )

                show_this_value = (
                    ANNOTATE_BAR_VALUES
                    and (
                        not ANNOTATE_ONLY_SMALL_BARS
                        or magnitude <= annotation_threshold
                    )
                )

                near_zero = (
                    abs(endpoint_x)
                    <= 0.1 * axis_limit
                )

                if reached:
                    # Solid bar ending at the threshold crossing.
                    reference_parameter = pd.to_numeric(
                        pd.Series(
                            [result.get("reference_parameter", np.nan)]
                        ),
                        errors="coerce",
                    ).iloc[0]

                    bar_hatch = (
                        "///"
                        if (
                            study_name == "absorption_absolute"
                            and np.isfinite(reference_parameter)
                            and np.isclose(reference_parameter, 0.0)
                        )
                        else None
                    )

                    ax.barh(
                        y_position,
                        signed_width,
                        height=BAR_HEIGHT,
                        color=bar_colour,
                        edgecolor="black",
                        linewidth=0.5,
                        hatch=bar_hatch,
                        zorder=3,
                    )

                else:
                    # Dotted branch ending at the maximum investigated change.
                    ax.plot(
                        [0.0, endpoint_x],
                        [y_position, y_position],
                        color=bar_colour,
                        linestyle=":",
                        linewidth=1.2,
                        zorder=3,
                    )

                    # Open endpoint means threshold not reached.
                    ax.plot(
                        endpoint_x,
                        y_position,
                        marker="o",
                        markersize=4.0,
                        markerfacecolor="white",
                        markeredgecolor=bar_colour,
                        markeredgewidth=0.9,
                        linestyle="None",
                        zorder=4,
                    )

                # One annotation call only, for both reached and unreached branches.
                add_branch_annotation(
                    ax=ax,
                    endpoint_x=endpoint_x,
                    y_position=y_position,
                    magnitude=magnitude,
                    direction=direction,
                    response_sign=response_sign,
                    colour=bar_colour,
                    show_value=show_this_value,
                    threshold_reached=reached,
                    near_zero=near_zero,
                )

                # ------------------------------------------------
                # Threshold NOT reached:
                #
                # Draw a dotted line to the maximum tested
                # parameter change and an open marker at the
                # endpoint.
                # ------------------------------------------------


            # ----------------------------------------------------
            # Reference marker at zero
            #
            # This is drawn after the bars so it remains visible.
            # ----------------------------------------------------

            # --------------------------------------------------------
            # Optional marker at the reference position
            # --------------------------------------------------------

            if SHOW_REFERENCE_WAVELENGTH_MARKERS:

                ax.plot(
                    0.0,
                    y_position,
                    marker=REFERENCE_MARKERS[
                        wavelength_nm
                    ],
                    markersize=4.5,
                    markerfacecolor=(
                        REFERENCE_COLOURS[
                            wavelength_nm
                        ]
                    ),
                    markeredgecolor="black",
                    markeredgewidth=0.6,
                    linestyle="None",
                    zorder=6,
                )

        # ----------------------------------------------------
        # X-axis
        # ----------------------------------------------------

        ax.set_xlim(
            -axis_limit,
            axis_limit,
        )

        # ----------------------------------------------------
        # Y-axis
        # ----------------------------------------------------

        ax.set_yticks(
            [
                row_positions[
                    (
                        study_name,
                        wavelength_nm,
                    )
                ]
                for wavelength_nm in (
                    REFERENCE_WAVELENGTHS_NM
                )
            ]
        )

        if plot_column == 0:

            ax.set_yticklabels(
                [
                    f"{wavelength_nm}"
                    for wavelength_nm in (
                        REFERENCE_WAVELENGTHS_NM
                    )
                ],
                fontsize=7,
            )

            for tick_label, wavelength_nm in zip(
                ax.get_yticklabels(),
                REFERENCE_WAVELENGTHS_NM,
            ):

                tick_label.set_color(
                    REFERENCE_COLOURS[
                        wavelength_nm
                    ]
                )

                tick_label.set_fontweight(
                    "bold"
                )

            ax.set_ylabel(
                row_definition["label"],
                rotation=0,
                ha="right",
                va="center",
                labelpad=PARAMETER_LABEL_PAD,
                fontsize=8,
            )

        else:

            ax.set_yticklabels([])

        ax.invert_yaxis()

        # ----------------------------------------------------
        # Grid
        # ----------------------------------------------------

        ax.grid(
            True,
            axis="x",
            color="0.88",
            linewidth=0.6,
        )

        ax.grid(
            False,
            axis="y",
        )


# ============================================================
# TITLES AND X LABELS
# ============================================================

axes[0, 0].set_title(
    THRESHOLD_LABELS["C"],
    fontsize=9,
)

axes[0, 1].set_title(
    THRESHOLD_LABELS["L"],
    fontsize=9,
)

axes[0, 2].set_title(
    THRESHOLD_LABELS["h"],
    fontsize=9,
)

# ============================================================
# X-AXIS LABELS
# ============================================================

for ax in axes[-1, :]:
    ax.set_xlabel(
        "Required parameter change",
        fontsize=8,
    )

# ============================================================
# LEGEND
# ============================================================

direction_handles = [
    Patch(
        facecolor=DECREASE_COLOUR,
        edgecolor="black",
        linewidth=0.5,
        label="Parameter decrease",
    ),
    Patch(
        facecolor=INCREASE_COLOUR,
        edgecolor="black",
        linewidth=0.5,
        label="Parameter increase",
    ),
    Patch(
        facecolor="white",
        edgecolor="black",
        hatch="///",
        label=(
            "Absorption introduced from "
            r"$k_0=0$"
        ),
    ),
    Line2D(
        [0],
        [0],
        color="none",
        marker="",
        linestyle="None",
        label=(
            r"$\uparrow$: positive colour response"
        ),
    ),
    Line2D(
        [0],
        [0],
        color="none",
        marker="",
        linestyle="None",
        label=(
            r"$\downarrow$: negative colour response"
        ),
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        linestyle=":",
        color="black",
        markerfacecolor="white",
        markeredgecolor="black",
        label="Threshold not reached",
    ),
]

wavelength_handles = [
    Line2D(
        [0],
        [0],
        marker=REFERENCE_MARKERS[
            wavelength_nm
        ],
        markerfacecolor=(
            REFERENCE_COLOURS[
                wavelength_nm
            ]
        ),
        markeredgecolor="black",
        linestyle="None",
        label=f"{wavelength_nm} nm",
    )
    for wavelength_nm in (
        REFERENCE_WAVELENGTHS_NM
    )
]

# ============================================================
# LEGENDS
# ============================================================

direction_legend = fig.legend(
    handles=direction_handles,
    loc="outside upper center",
    ncol=3,
    frameon=True,
    fontsize=7,
    borderpad=0.25,
    handletextpad=0.35,
    columnspacing=0.8,
)

wavelength_legend = None

if SHOW_REFERENCE_WAVELENGTH_LEGEND:

    wavelength_legend = fig.legend(
        handles=wavelength_handles,
        loc="outside lower center",
        ncol=4,
        frameon=True,
        fontsize=8,
        title=r"Reference $\lambda_0$",
        title_fontsize=8,
        handletextpad=0.3,
        columnspacing=0.8,
    )


# ============================================================
# SAVE
# ============================================================

if PANEL_A_SAVE:

    FIGURE_OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    panel_a_png = (
        FIGURE_OUTPUT_DIRECTORY
        / f"{PANEL_A_BASENAME}.png"
    )

    panel_a_pdf = (
        FIGURE_OUTPUT_DIRECTORY
        / f"{PANEL_A_BASENAME}.pdf"
    )

    # Complete all layout calculations before saving.
    fig.canvas.draw()

    fig.savefig(
        panel_a_png,
        dpi=PNG_DPI,
        transparent=True,
        bbox_inches="tight",
        pad_inches=0.03,
    )

    fig.savefig(
        panel_a_pdf,
        transparent=True,
        bbox_inches="tight",
        pad_inches=0.03,
    )

    print("Saved:", panel_a_png)
    print("Saved:", panel_a_pdf)


plt.show()

# In[ ]:


# ============================================================
# PANEL B: REPRESENTATIVE FULL SENSITIVITY CURVES
#
# Default references:
#   580 nm and 620 nm
#
# Rows:
#   Delta C_ab*
#   Delta L*
#   Delta h_ab
#
# Columns:
#   shell number
#   index ratio
#   surrounding-medium index
#   absorption
#   illumination half-angle
#
# Solid curves:
#   calculated responses over the complete investigated range
#
# Dashed line near reference:
#   local linear fit
#
# R^2:
#   coefficient of determination of the local linear fit
# ============================================================

from matplotlib.lines import Line2D


# ============================================================
# SETTINGS
# ============================================================

REPRESENTATIVE_WAVELENGTHS_NM = [
    380,
    620,
    760
]

PANEL_B_WIDTH_CM = 15.92
PANEL_B_HEIGHT_CM = 15.92

PANEL_B_SAVE = True

PANEL_B_BASENAME = (
    "SI_representative_sensitivity_curves"
)

LOCAL_RELATIVE_FIT_LIMIT = 0.10
LOCAL_MEDIUM_FIT_MAX = 1.1
LOCAL_ILLUMINATION_FIT_MIN_DEG = 60.0
LOCAL_ABSOLUTE_K_FIT_MAX = 2.5e-3

SHOW_LOCAL_FIT_REGION = False
SHOW_R_SQUARED = False


# ============================================================
# PANEL B STUDY DEFINITIONS
# ============================================================

panel_b_studies = [
    {
        "key": "shells",
        "label": (
            r"$N_{\mathrm{shell}}/"
            r"N_{\mathrm{shell},0}$"
        ),
        "data": df_shells,
    },
    {
        "key": "index",
        "label": (
            r"$(n_A/n_B)/(n_A/n_B)_0$"
        ),
        "data": df_index_quarterwave,
    },
    {
        "key": "medium",
        "label": r"$n_{\mathrm{m}}$",
        "data": df_medium,
    },
    {
        "key": "absorption",
        "label": (
            r"$k/k_0$ or $10^3k$"
        ),
        "data": df_absorption,
    },
    {
        "key": "illumination",
        "label": (
            r"$\theta_{\mathrm{ill,max}}$ "
            r"[$^\circ$]"
        ),
        "data": df_illumination,
    },
]

panel_b_responses = [
    {
        "key": "C",
        "column": "_delta_C",
        "label": r"$\Delta C_{ab}^*$",
    },
    {
        "key": "L",
        "column": "_delta_L",
        "label": r"$\Delta L^*$",
    },
    {
        "key": "h",
        "column": "_delta_h",
        "label": (
            r"$\Delta h_{ab}$ [$^\circ$]"
        ),
    },
]


# ============================================================
# PREPARE FULL CURVE FOR PANEL B
# ============================================================

def prepare_panel_b_curve(
    study_key,
    dataframe,
    wavelength_nm,
):
    """Prepare one complete sensitivity curve."""

    if study_key != "absorption":

        prepared = prepare_full_response_data(
            study_key=study_key,
            dataframe=dataframe,
            wavelength_nm=wavelength_nm,
        )

        if prepared is None:
            return None

        group = prepared["group"].copy()

        parameter_column = (
            prepared["parameter_column"]
        )

        group["_plot_parameter"] = (
            pd.to_numeric(
                group[parameter_column],
                errors="coerce",
            )
        )

        reference_parameter = float(
            prepared[
                "reference_parameter"
            ]
        )

        if study_key in [
            "shells",
            "index",
        ]:

            local_mask = (
                np.abs(
                    group[
                        "_plot_parameter"
                    ]
                    - reference_parameter
                )
                <= LOCAL_RELATIVE_FIT_LIMIT
                + 1e-12
            )

        elif study_key == "medium":

            local_mask = (
                group["_plot_parameter"]
                <= LOCAL_MEDIUM_FIT_MAX
                + 1e-12
            )

        elif study_key == "illumination":

            local_mask = (
                group["_plot_parameter"]
                >= LOCAL_ILLUMINATION_FIT_MIN_DEG
                - 1e-12
            )

        else:

            local_mask = np.ones(
                len(group),
                dtype=bool,
            )

        return {
            "group": group,
            "reference_parameter": (
                reference_parameter
            ),
            "local_mask": np.asarray(
                local_mask,
                dtype=bool,
            ),
            "x_label": {
                "shells": (
                    r"$N_{\mathrm{shell}}/"
                    r"N_{\mathrm{shell},0}$"
                ),
                "index": (
                    r"$(n_A/n_B)/(n_A/n_B)_0$"
                ),
                "medium": r"$n_{\mathrm{m}}$",
                "illumination": (
                    r"$\theta_{\mathrm{ill,max}}$ "
                    r"[$^\circ$]"
                ),
            }[study_key],
        }

    # --------------------------------------------------------
    # Absorption requires separate handling
    # --------------------------------------------------------

    raw_group = dataframe.loc[
        np.isclose(
            pd.to_numeric(
                dataframe[
                    "reference_design_wavelength_nm"
                ],
                errors="coerce",
            ),
            wavelength_nm,
        )
    ].copy()

    if raw_group.empty:
        return None

    reference_k = float(
        pd.to_numeric(
            raw_group["reference_k"],
            errors="coerce",
        ).iloc[0]
    )

    k_values = pd.to_numeric(
        raw_group["parameter_value"],
        errors="coerce",
    ).to_numpy(dtype=float)

    if reference_k > 0.0:

        raw_group[
            "_analysis_parameter"
        ] = k_values / reference_k

        reference_parameter = 1.0

        x_label = r"$k/k_0$"

        local_mask = (
            np.abs(
                raw_group[
                    "_analysis_parameter"
                ]
                - 1.0
            )
            <= 0.50 + 1e-12
        )

    else:

        raw_group[
            "_analysis_parameter"
        ] = 1e3 * k_values

        reference_parameter = 0.0

        x_label = r"$10^3 k$"

        local_mask = (
            k_values
            <= LOCAL_ABSOLUTE_K_FIT_MAX
            + 1e-15
        )

    baseline = determine_actual_baseline(
        raw_group,
        "_analysis_parameter",
        reference_parameter,
    )

    raw_group["_delta_C"] = (
        pd.to_numeric(
            raw_group["C"],
            errors="coerce",
        )
        - float(baseline["C"])
    )

    raw_group["_delta_L"] = (
        pd.to_numeric(
            raw_group["L"],
            errors="coerce",
        )
        - float(baseline["L"])
    )

    raw_group["_delta_h"] = (
        circular_difference_deg(
            raw_group["hue_deg"],
            baseline["hue_deg"],
        )
    )

    raw_group["_plot_parameter"] = (
        raw_group[
            "_analysis_parameter"
        ]
    )

    raw_group = raw_group.sort_values(
        "_plot_parameter"
    )

    local_mask = np.asarray(
        local_mask,
        dtype=bool,
    )[
        np.argsort(
            pd.to_numeric(
                raw_group[
                    "_plot_parameter"
                ],
                errors="coerce",
            ).to_numpy(dtype=float)
        )
    ] if False else (
        (
            np.abs(
                raw_group[
                    "_plot_parameter"
                ]
                - reference_parameter
            )
            <= (
                0.50
                if reference_k > 0.0
                else 1e3
                * LOCAL_ABSOLUTE_K_FIT_MAX
            )
            + 1e-12
        ).to_numpy(dtype=bool)
    )

    return {
        "group": raw_group,
        "reference_parameter": (
            reference_parameter
        ),
        "local_mask": local_mask,
        "x_label": x_label,
    }


# ============================================================
# LOCAL LINEAR FIT
# ============================================================

def fit_local_linear_response(
    x,
    y,
    local_mask,
    reference_x,
):
    """
    Fit y = intercept + slope * (x - reference_x) in the selected
    local region and return slope, R^2, and fitted values.
    """

    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    local_mask = np.asarray(
        local_mask,
        dtype=bool,
    )

    finite = (
        np.isfinite(x)
        & np.isfinite(y)
        & local_mask
    )

    x_fit = x[finite]
    y_fit = y[finite]

    if len(x_fit) < 2:

        return {
            "slope": np.nan,
            "r_squared": np.nan,
            "x_fit": np.array([]),
            "y_predicted": np.array([]),
            "n_points": len(x_fit),
        }

    if np.isclose(
        np.nanmax(x_fit),
        np.nanmin(x_fit),
    ):

        return {
            "slope": np.nan,
            "r_squared": np.nan,
            "x_fit": np.array([]),
            "y_predicted": np.array([]),
            "n_points": len(x_fit),
        }

    centered_x = (
        x_fit - reference_x
    )

    slope, intercept = np.polyfit(
        centered_x,
        y_fit,
        deg=1,
    )

    order = np.argsort(x_fit)

    x_fit_sorted = x_fit[order]

    predicted_sorted = (
        slope
        * (
            x_fit_sorted
            - reference_x
        )
        + intercept
    )

    predicted_original_order = (
        slope * centered_x
        + intercept
    )

    residual_sum = np.sum(
        (
            y_fit
            - predicted_original_order
        )
        ** 2
    )

    total_sum = np.sum(
        (
            y_fit
            - np.mean(y_fit)
        )
        ** 2
    )

    if np.isclose(total_sum, 0.0):

        r_squared = 1.0

    else:

        r_squared = (
            1.0
            - residual_sum / total_sum
        )

    return {
        "slope": float(slope),
        "r_squared": float(r_squared),
        "x_fit": x_fit_sorted,
        "y_predicted": predicted_sorted,
        "n_points": len(x_fit),
    }


# ============================================================
# CREATE FIGURE
# ============================================================

fig, axes = plt.subplots(
    3,
    5,
    figsize=(
        PANEL_B_WIDTH_CM * cm_to_inches,
        PANEL_B_HEIGHT_CM * cm_to_inches,
    ),
    sharex="col",
    constrained_layout=True,
)


# ============================================================
# DRAW FULL CURVES AND LOCAL FITS
# ============================================================

panel_b_fit_records = []

for column_index, study in enumerate(
    panel_b_studies
):

    for wavelength_nm in (
        REPRESENTATIVE_WAVELENGTHS_NM
    ):

        prepared = prepare_panel_b_curve(
            study_key=study["key"],
            dataframe=study["data"],
            wavelength_nm=wavelength_nm,
        )

        if prepared is None:
            continue

        group = prepared["group"]

        x = pd.to_numeric(
            group["_plot_parameter"],
            errors="coerce",
        ).to_numpy(dtype=float)

        for row_index, response in enumerate(
            panel_b_responses
        ):

            ax = axes[
                row_index,
                column_index,
            ]

            y = pd.to_numeric(
                group[response["column"]],
                errors="coerce",
            ).to_numpy(dtype=float)

            finite = (
                np.isfinite(x)
                & np.isfinite(y)
            )

            ax.plot(
                x[finite],
                y[finite],
                color=REFERENCE_COLOURS[
                    wavelength_nm
                ],
                marker=REFERENCE_MARKERS[
                    wavelength_nm
                ],
                markersize=3.8,
                markeredgecolor="black",
                markeredgewidth=0.45,
                linewidth=1.1,
                label=f"{wavelength_nm} nm",
                zorder=3,
            )

            fit_result = (
                fit_local_linear_response(
                    x,
                    y,
                    prepared["local_mask"],
                    prepared[
                        "reference_parameter"
                    ],
                )
            )

            if (
                len(
                    fit_result["x_fit"]
                )
                >= 2
            ):

                ax.plot(
                    fit_result["x_fit"],
                    fit_result[
                        "y_predicted"
                    ],
                    color=REFERENCE_COLOURS[
                        wavelength_nm
                    ],
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.9,
                    zorder=4,
                )

            panel_b_fit_records.append(
                {
                    "study": study["key"],
                    "reference_design_wavelength_nm": (
                        wavelength_nm
                    ),
                    "response": response["key"],
                    "slope": fit_result["slope"],
                    "R_squared": (
                        fit_result["r_squared"]
                    ),
                    "fit_points": (
                        fit_result["n_points"]
                    ),
                }
            )

    # --------------------------------------------------------
    # Common formatting for the column
    # --------------------------------------------------------

    for row_index, response in enumerate(
        panel_b_responses
    ):

        ax = axes[
            row_index,
            column_index,
        ]

        ax.axhline(
            0.0,
            color="0.5",
            linewidth=0.7,
            zorder=0,
        )

        # Reference line differs for absolute-k cases, but the
        # plotted representative cases may use different absorption
        # coordinates. Therefore no common absorption reference line
        # is drawn in that column.
        if study["key"] != "absorption":

            reference_x = {
                "shells": 1.0,
                "index": 1.0,
                "medium": 1.0,
                "illumination": 90.0,
            }[study["key"]]

            ax.axvline(
                reference_x,
                color="0.35",
                linestyle=":",
                linewidth=0.7,
                zorder=0,
            )

        ax.grid(
            False,
            which="both",
            axis="both",
        )

        if column_index == 0:

            ax.set_ylabel(
                response["label"]
            )


# ============================================================
# R-SQUARED ANNOTATIONS
# ============================================================

df_panel_b_fits = pd.DataFrame(
    panel_b_fit_records
)

if SHOW_R_SQUARED:

    for column_index, study in enumerate(
        panel_b_studies
    ):

        for row_index, response in enumerate(
            panel_b_responses
        ):

            ax = axes[
                row_index,
                column_index,
            ]

            annotation_lines = []

            subset = df_panel_b_fits.loc[
                (
                    df_panel_b_fits[
                        "study"
                    ]
                    == study["key"]
                )
                & (
                    df_panel_b_fits[
                        "response"
                    ]
                    == response["key"]
                )
            ]

            for wavelength_nm in (
                REPRESENTATIVE_WAVELENGTHS_NM
            ):

                row = subset.loc[
                    np.isclose(
                        subset[
                            "reference_design_wavelength_nm"
                        ],
                        wavelength_nm,
                    )
                ]

                if row.empty:
                    continue

                r_squared = float(
                    row.iloc[0][
                        "R_squared"
                    ]
                )

                if np.isfinite(r_squared):

                    annotation_lines.append(
                        (
                            wavelength_nm,
                            rf"$R^2={r_squared:.2f}$",
                        )
                    )

            for line_index, (
                wavelength_nm,
                annotation,
            ) in enumerate(
                annotation_lines
            ):

                ax.text(
                    0.03,
                    0.96
                    - line_index * 0.13,
                    annotation,
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=6.5,
                    color=REFERENCE_COLOURS[
                        wavelength_nm
                    ],
                )


# ============================================================
# TITLES AND X LABELS
# ============================================================

column_titles = [
    "Shell number",
    "Index ratio",
    "Surrounding medium",
    "Absorption",
    "Illumination",
]

for column_index, title in enumerate(
    column_titles
):

    axes[
        0,
        column_index,
    ].set_title(
        title,
        fontsize=9,
    )


for column_index, study in enumerate(
    panel_b_studies
):

    # Retrieve the x label from the first available representative.
    x_label = study["label"]

    axes[
        -1,
        column_index,
    ].set_xlabel(
        x_label,
        fontsize=8,
    )


# ============================================================
# LEGEND
# ============================================================

representative_handles = [
    Line2D(
        [0],
        [0],
        color=REFERENCE_COLOURS[
            wavelength_nm
        ],
        marker=REFERENCE_MARKERS[
            wavelength_nm
        ],
        markeredgecolor="black",
        markeredgewidth=0.45,
        linewidth=1.1,
        markersize=4.5,
        label=f"{wavelength_nm} nm",
    )
    for wavelength_nm in (
        REPRESENTATIVE_WAVELENGTHS_NM
    )
]

representative_handles.append(
    Line2D(
        [0],
        [0],
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="Reference",
    )
)

fig.legend(
    handles=representative_handles,
    loc="outside upper center",
    ncol=4,
    frameon=False,
    fontsize=8,
)

from matplotlib.patches import Rectangle

# Make sure the constrained layout has been calculated
fig.canvas.draw()

# Get the bounding boxes of the five bottom axes
bottom_axes = axes[-1, :]

left = min(
    ax.get_position().x0
    for ax in bottom_axes
) - 0.09

right = max(
    ax.get_position().x1
    for ax in bottom_axes
)

bottom = min(
    ax.get_position().y0
    for ax in bottom_axes
)

top = max(
    ax.get_position().y1
    for ax in bottom_axes
)

# Add semitransparent rectangle
rectangle = Rectangle(
    (left, bottom),
    right - left,
    top - bottom,
    transform=fig.transFigure,
    facecolor="grey",
    edgecolor="none",
    alpha=0.20,
    zorder=10,
)

fig.add_artist(rectangle)

# ============================================================
# SAVE AND DISPLAY
# ============================================================

if PANEL_B_SAVE:

    FIGURE_OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    panel_b_png = (
        FIGURE_OUTPUT_DIRECTORY
        / f"{PANEL_B_BASENAME}.png"
    )

    panel_b_pdf = (
        FIGURE_OUTPUT_DIRECTORY
        / f"{PANEL_B_BASENAME}.pdf"
    )

    fig.savefig(
        panel_b_png,
        dpi=PNG_DPI,
        transparent=True,
        bbox_inches=None,
    )

    fig.savefig(
        panel_b_pdf,
        transparent=True,
        bbox_inches=None,
    )

    print("Saved:", panel_b_png)
    print("Saved:", panel_b_pdf)


plt.show()


# ============================================================
# DISPLAY AND SAVE LOCAL-LINEARITY TABLE
# ============================================================

display(
    df_panel_b_fits.sort_values(
        [
            "study",
            "response",
            "reference_design_wavelength_nm",
        ]
    ).style.format(
        {
            "slope": "{:+.4f}",
            "R_squared": "{:.3f}",
        }
    )
)

PANEL_B_FIT_TABLE_PATH = (
    SENSITIVITY_DIRECTORY
    / "representative_local_linearity.csv"
)

df_panel_b_fits.to_csv(
    PANEL_B_FIT_TABLE_PATH,
    index=False,
)

print(
    "Saved:",
    PANEL_B_FIT_TABLE_PATH,
)

# Panel B: Shell-number versus index-ratio interaction maps

# In[ ]:


# ============================================================
# PANEL B INPUT: LOAD JOINT SHELL/INDEX INTERACTION SWEEP
# ============================================================

INTERACTION_DIRECTORY = (
    SENSITIVITY_DIRECTORY
    / "shell_index_interaction"
)

interaction_candidates = [
    INTERACTION_DIRECTORY
    / "sensitivity_shell_index_interaction.parquet",

    INTERACTION_DIRECTORY
    / "sensitivity_shell_index_interaction.pkl",

    INTERACTION_DIRECTORY
    / "sensitivity_shell_index_interaction.csv",
]

interaction_path = next(
    (
        path
        for path in interaction_candidates
        if path.exists()
    ),
    None,
)

if interaction_path is None:

    df_shell_index_interaction = (
        pd.DataFrame()
    )

    print(
        "Panel B cannot yet be generated."
    )

    print(
        "A joint N_shell x (n_A/n_B) interaction "
        "sweep is required."
    )

    print(
        "Separate one-parameter sweeps must not be "
        "interpolated into a two-dimensional map."
    )

else:

    if interaction_path.suffix == ".parquet":

        df_shell_index_interaction = (
            pd.read_parquet(
                interaction_path
            )
        )

    elif interaction_path.suffix == ".pkl":

        df_shell_index_interaction = (
            pd.read_pickle(
                interaction_path
            )
        )

    else:

        df_shell_index_interaction = (
            pd.read_csv(
                interaction_path
            )
        )

    required_interaction_columns = {
        "reference_design_wavelength_nm",
        "n_shells",
        "n_A",
        "n_B",
        "C",
        "L",
        "hue_deg",
    }

    missing = (
        required_interaction_columns
        - set(
            df_shell_index_interaction.columns
        )
    )

    if missing:
        raise KeyError(
            "Interaction data are missing columns: "
            f"{sorted(missing)}"
        )

    print(
        "Loaded interaction sweep:",
        len(df_shell_index_interaction),
        "rows",
    )

# In[ ]:


# ============================================================
# PANEL B: SHELL-NUMBER / INDEX-RATIO INTERACTION MAPS
#
# Map colour:
#     Delta C*
#
# White solid contours:
#     Delta L*
#
# Black contours:
#     |Delta hue| = 5, 10, and 20 degrees
#
# Star:
#     selected reference design
# ============================================================

if df_shell_index_interaction.empty:

    print(
        "Skipping Panel B because no joint interaction "
        "dataset is available."
    )

else:

    PANEL_B_SAVE = True
    PANEL_B_WIDTH_CM = 15.5
    PANEL_B_HEIGHT_CM = 11.5

    interaction = (
        df_shell_index_interaction.copy()
    )

    interaction["index_ratio"] = (
        pd.to_numeric(
            interaction["n_A"],
            errors="coerce",
        )
        / pd.to_numeric(
            interaction["n_B"],
            errors="coerce",
        )
    )

    interaction["shell_ratio"] = (
        pd.to_numeric(
            interaction["n_shells"],
            errors="coerce",
        )
        / pd.to_numeric(
            interaction[
                "reference_n_shells"
            ],
            errors="coerce",
        )
    )

    interaction["index_ratio_normalized"] = (
        interaction["index_ratio"]
        / (
            pd.to_numeric(
                interaction[
                    "reference_n_A"
                ],
                errors="coerce",
            )
            / pd.to_numeric(
                interaction[
                    "reference_n_B"
                ],
                errors="coerce",
            )
        )
    )

    interaction["delta_C_map"] = (
        interaction["C"]
        - interaction["reference_C"]
    )

    interaction["delta_L_map"] = (
        interaction["L"]
        - interaction["reference_L"]
    )

    interaction[
        "delta_hue_map_deg"
    ] = circular_difference_deg(
        interaction["hue_deg"],
        interaction[
            "reference_hue_deg"
        ],
    )


    # Global symmetric colour scale
    colour_limit = float(
        np.nanmax(
            np.abs(
                interaction[
                    "delta_C_map"
                ]
            )
        )
    )

    colour_norm = TwoSlopeNorm(
        vmin=-colour_limit,
        vcenter=0.0,
        vmax=colour_limit,
    )


    fig, axes = plt.subplots(
        2,
        2,
        figsize=(
            PANEL_B_WIDTH_CM * cm_to_inches,
            PANEL_B_HEIGHT_CM * cm_to_inches,
        ),
        constrained_layout=True,
    )

    mesh_for_colourbar = None


    for ax, wavelength_nm in zip(
        axes.ravel(),
        REFERENCE_WAVELENGTHS_NM,
    ):

        group = interaction.loc[
            np.isclose(
                interaction[
                    "reference_design_wavelength_nm"
                ],
                wavelength_nm,
            )
        ].copy()

        shell_values = np.sort(
            group["shell_ratio"].unique()
        )

        index_values = np.sort(
            group[
                "index_ratio_normalized"
            ].unique()
        )

        expected_size = (
            len(shell_values)
            * len(index_values)
        )

        if len(group) != expected_size:
            raise ValueError(
                f"The {wavelength_nm} nm interaction "
                "grid is incomplete. Expected "
                f"{expected_size} rows, found {len(group)}."
            )

        C_grid = (
            group.pivot(
                index="index_ratio_normalized",
                columns="shell_ratio",
                values="delta_C_map",
            )
            .reindex(
                index=index_values,
                columns=shell_values,
            )
        )

        L_grid = (
            group.pivot(
                index="index_ratio_normalized",
                columns="shell_ratio",
                values="delta_L_map",
            )
            .reindex(
                index=index_values,
                columns=shell_values,
            )
        )

        hue_grid = (
            group.pivot(
                index="index_ratio_normalized",
                columns="shell_ratio",
                values="delta_hue_map_deg",
            )
            .reindex(
                index=index_values,
                columns=shell_values,
            )
        )

        X, Y = np.meshgrid(
            shell_values,
            index_values,
        )

        mesh_for_colourbar = ax.pcolormesh(
            X,
            Y,
            C_grid.to_numpy(dtype=float),
            shading="auto",
            cmap="coolwarm",
            norm=colour_norm,
        )

        # Lightness contours
        finite_L = np.abs(
            L_grid.to_numpy(dtype=float)
        )

        maximum_L = np.nanmax(finite_L)

        if maximum_L > 0:

            lightness_levels = np.linspace(
                -maximum_L,
                maximum_L,
                7,
            )

            lightness_levels = (
                lightness_levels[
                    ~np.isclose(
                        lightness_levels,
                        0.0,
                    )
                ]
            )

            ax.contour(
                X,
                Y,
                L_grid.to_numpy(dtype=float),
                levels=lightness_levels,
                colors="white",
                linewidths=0.65,
            )

        # Absolute hue-drift contours
        hue_contours = ax.contour(
            X,
            Y,
            np.abs(
                hue_grid.to_numpy(
                    dtype=float
                )
            ),
            levels=[5.0, 10.0, 20.0],
            colors="black",
            linewidths=[1.4, 0.9, 0.7],
        )

        ax.clabel(
            hue_contours,
            inline=True,
            fontsize=7,
            fmt=lambda value: (
                f"{value:.0f}°"
            ),
        )

        ax.scatter(
            [1.0],
            [1.0],
            marker="*",
            s=85,
            facecolor="white",
            edgecolor="black",
            linewidth=0.8,
            zorder=6,
        )

        ax.set_title(
            f"{wavelength_nm} nm"
        )

        ax.set_xlabel(
            r"$N_{\mathrm{shell}}/"
            r"N_{\mathrm{shell},0}$"
        )

        ax.set_ylabel(
            r"$(n_A/n_B)/(n_A/n_B)_0$"
        )


    colourbar = fig.colorbar(
        mesh_for_colourbar,
        ax=axes,
        location="right",
        shrink=0.88,
        pad=0.02,
    )

    colourbar.set_label(
        r"$\Delta C_{ab}^*$"
    )


    if PANEL_B_SAVE:

        path_png = (
            FIGURE_OUTPUT_DIRECTORY
            / "Figure3B_shell_index_maps.png"
        )

        path_pdf = (
            FIGURE_OUTPUT_DIRECTORY
            / "Figure3B_shell_index_maps.pdf"
        )

        fig.savefig(
            path_png,
            dpi=PNG_DPI,
            transparent=True,
            bbox_inches=None,
        )

        fig.savefig(
            path_pdf,
            transparent=True,
            bbox_inches=None,
        )

        print("Saved:", path_png)
        print("Saved:", path_pdf)


    plt.show()

# Panel C: Robustness summary

# In[ ]:


# ============================================================
# ROBUSTNESS-THRESHOLD FUNCTIONS
# ============================================================

ROBUSTNESS_THRESHOLDS = {
    "delta_C_rerun": 5.0,
    "delta_L_rerun": 5.0,
    "delta_hue_deg_rerun": 5.0,
}

ROBUSTNESS_LABELS = {
    "delta_C_rerun": (
        r"$|\Delta C_{ab}^*|=5$"
    ),
    "delta_L_rerun": (
        r"$|\Delta L^*|=5$"
    ),
    "delta_hue_deg_rerun": (
        r"$|\Delta h_{ab}|=5^\circ$"
    ),
}


def first_threshold_crossing(
    parameter_ratio,
    response,
    threshold,
):
    """
    Return the smallest absolute fractional parameter change at
    which |response| reaches the threshold.

    Linear interpolation is applied only between adjacent sampled
    parameter points on the same side of the reference value 1.

    Returns NaN if the threshold is never crossed.
    """

    x = np.asarray(
        parameter_ratio,
        dtype=float,
    )

    y = np.abs(
        np.asarray(
            response,
            dtype=float,
        )
    )

    finite = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    x = x[finite]
    y = y[finite]

    if len(x) == 0:
        return np.nan

    candidate_crossings = []

    # Analyze each side of the reference independently.
    for side_mask in [
        x <= 1.0,
        x >= 1.0,
    ]:

        x_side = x[side_mask]
        y_side = y[side_mask]

        if len(x_side) == 0:
            continue

        # Increasing distance from the reference.
        order = np.argsort(
            np.abs(
                x_side - 1.0
            )
        )

        x_side = x_side[order]
        y_side = y_side[order]

        distances = np.abs(
            x_side - 1.0
        )

        # Threshold already exceeded at a sampled point.
        above = np.flatnonzero(
            y_side >= threshold
        )

        if len(above) == 0:
            continue

        crossing_index = int(
            above[0]
        )

        if crossing_index == 0:

            candidate_crossings.append(
                distances[0]
            )

            continue

        x0 = distances[
            crossing_index - 1
        ]

        x1 = distances[
            crossing_index
        ]

        y0 = y_side[
            crossing_index - 1
        ]

        y1 = y_side[
            crossing_index
        ]

        if np.isclose(y1, y0):

            crossing_distance = x1

        else:

            fraction = (
                threshold - y0
            ) / (
                y1 - y0
            )

            fraction = np.clip(
                fraction,
                0.0,
                1.0,
            )

            crossing_distance = (
                x0
                + fraction
                * (x1 - x0)
            )

        candidate_crossings.append(
            crossing_distance
        )

    if not candidate_crossings:
        return np.nan

    return float(
        np.nanmin(
            candidate_crossings
        )
    )


def build_robustness_summary(
    studies,
):
    """
    Build one row for every parameter, reference design, and
    response threshold.
    """

    records = []

    for study_name, dataframe in (
        studies.items()
    ):

        # The normalized absorption dataset includes only cases
        # for which k/k0 is mathematically defined.
        for wavelength_nm in (
            REFERENCE_WAVELENGTHS_NM
        ):

            group = dataframe.loc[
                np.isclose(
                    dataframe[
                        "reference_design_wavelength_nm"
                    ],
                    wavelength_nm,
                )
            ].copy()

            if group.empty:
                continue

            for response_column, threshold in (
                ROBUSTNESS_THRESHOLDS.items()
            ):

                if response_column not in group:
                    continue

                crossing = (
                    first_threshold_crossing(
                        group[
                            "variation_factor"
                        ],
                        group[
                            response_column
                        ],
                        threshold,
                    )
                )

                records.append(
                    {
                        "study": study_name,
                        "reference_design_wavelength_nm": (
                            wavelength_nm
                        ),
                        "response": (
                            response_column
                        ),
                        "threshold": threshold,
                        "minimum_fractional_parameter_change": (
                            crossing
                        ),
                        "minimum_percent_parameter_change": (
                            100.0 * crossing
                            if np.isfinite(crossing)
                            else np.nan
                        ),
                        "threshold_reached": (
                            np.isfinite(crossing)
                        ),
                    }
                )

    return pd.DataFrame(records)

# Code cell 9: Calculate and inspect robustness limits

# In[ ]:


# ============================================================
# CALCULATE ROBUSTNESS SUMMARY
# ============================================================

# Illumination is omitted from the principal normalized robustness
# panel because its natural coordinate is angular rather than a
# conventional material or structural parameter ratio.
robustness_studies = {
    "Shell number": df_shells,
    "Index ratio": (
        df_index_quarterwave
    ),
    "Medium index": df_medium,
    "Absorption": (
        df_absorption_normalized
    ),
}

df_robustness = build_robustness_summary(
    robustness_studies
)

display(
    df_robustness.sort_values(
        [
            "study",
            "response",
            "reference_design_wavelength_nm",
        ]
    )
)

# In[ ]:


# ============================================================
# PANEL C: OPERATIONAL ROBUSTNESS SUMMARY
#
# Horizontal coordinate:
#   smallest absolute parameter variation required to reach
#   the specified response threshold
#
# Open symbols at the right boundary:
#   threshold not reached in the investigated range
# ============================================================

PANEL_C_SAVE = True
PANEL_C_WIDTH_CM = 16.0
PANEL_C_HEIGHT_CM = 10.5

response_order = [
    "delta_C_rerun",
    "delta_L_rerun",
    "delta_hue_deg_rerun",
]

study_order = [
    "Shell number",
    "Index ratio",
    "Medium index",
    "Absorption",
]


# Determine maximum finite value for consistent axes.
finite_robustness = df_robustness[
    "minimum_percent_parameter_change"
].dropna()

if finite_robustness.empty:
    raise ValueError(
        "No response threshold was reached in any study."
    )

robustness_x_max = max(
    25.0,
    1.15
    * float(
        finite_robustness.max()
    ),
)


fig, axes = plt.subplots(
    1,
    3,
    figsize=(
        PANEL_C_WIDTH_CM * cm_to_inches,
        PANEL_C_HEIGHT_CM * cm_to_inches,
    ),
    sharey=True,
    constrained_layout=True,
)


for ax, response_column in zip(
    axes,
    response_order,
):

    subset = df_robustness.loc[
        df_robustness["response"]
        == response_column
    ].copy()

    for study_index, study_name in enumerate(
        study_order
    ):

        study_data = subset.loc[
            subset["study"] == study_name
        ]

        for wavelength_offset, wavelength_nm in enumerate(
            REFERENCE_WAVELENGTHS_NM
        ):

            row = study_data.loc[
                np.isclose(
                    study_data[
                        "reference_design_wavelength_nm"
                    ],
                    wavelength_nm,
                )
            ]

            if row.empty:
                continue

            row = row.iloc[0]

            y_position = (
                study_index
                + (
                    wavelength_offset - 1.5
                )
                * 0.10
            )

            threshold_reached = bool(
                row["threshold_reached"]
            )

            if threshold_reached:

                x_value = float(
                    row[
                        "minimum_percent_parameter_change"
                    ]
                )

                facecolour = (
                    REFERENCE_COLOURS[
                        wavelength_nm
                    ]
                )

            else:

                # Place an open marker at the displayed boundary.
                x_value = robustness_x_max

                facecolour = "none"

            ax.scatter(
                [x_value],
                [y_position],
                marker=REFERENCE_MARKERS[
                    wavelength_nm
                ],
                s=38,
                facecolor=facecolour,
                edgecolor=REFERENCE_COLOURS[
                    wavelength_nm
                ],
                linewidth=0.9,
                zorder=3,
            )

    ax.set_xlim(
        0.0,
        robustness_x_max,
    )

    ax.set_xlabel(
        "Minimum parameter\nchange [%]"
    )

    ax.set_title(
        ROBUSTNESS_LABELS[
            response_column
        ],
        fontsize=9,
    )

    ax.grid(
        False,
        which="both",
        axis="both",
    )


axes[0].set_yticks(
    np.arange(
        len(study_order)
    )
)

axes[0].set_yticklabels(
    study_order
)

axes[0].invert_yaxis()


# Reference-design legend
fig.legend(
    handles=reference_handles,
    loc="outside upper center",
    ncol=4,
    frameon=False,
    fontsize=8,
    title=r"Reference $\lambda_0$",
    title_fontsize=8,
)


# Explanation for open symbols
open_handle = Line2D(
    [0],
    [0],
    marker="o",
    linestyle="None",
    markerfacecolor="none",
    markeredgecolor="black",
    label="Threshold not reached",
)

fig.legend(
    handles=[open_handle],
    loc="outside lower center",
    frameon=False,
    fontsize=8,
)


if PANEL_C_SAVE:

    path_png = (
        FIGURE_OUTPUT_DIRECTORY
        / "Figure3C_robustness_summary.png"
    )

    path_pdf = (
        FIGURE_OUTPUT_DIRECTORY
        / "Figure3C_robustness_summary.pdf"
    )

    fig.savefig(
        path_png,
        dpi=PNG_DPI,
        transparent=True,
        bbox_inches=None,
    )

    fig.savefig(
        path_pdf,
        transparent=True,
        bbox_inches=None,
    )

    print("Saved:", path_png)
    print("Saved:", path_pdf)


plt.show()

# Code cell 11: Compare fixed and quarter-wave geometry at 580 nm

# In[ ]:


# ============================================================
# 580 nm INDEX SENSITIVITY:
# QUARTER-WAVE RECONSTRUCTION VS FIXED GEOMETRY
# ============================================================

if df_index_fixed_580.empty:

    print(
        "Fixed-geometry comparison unavailable."
    )

else:

    fixed = df_index_fixed_580.copy()

    for column in [
        "variation_factor",
        "delta_C",
        "delta_L",
        "delta_hue_deg",
    ]:
        fixed[column] = pd.to_numeric(
            fixed[column],
            errors="coerce",
        )

    quarterwave_580 = (
        df_index_quarterwave.loc[
            np.isclose(
                df_index_quarterwave[
                    "reference_design_wavelength_nm"
                ],
                580.0,
            )
        ]
        .sort_values(
            "variation_factor"
        )
    )

    fixed = fixed.sort_values(
        "variation_factor"
    )

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(
            8.0 * cm_to_inches,
            10.0 * cm_to_inches,
        ),
        sharex=True,
        constrained_layout=True,
    )

    comparison_rows = [
        (
            "delta_C_rerun",
            "delta_C",
            r"$\Delta C_{ab}^*$",
        ),
        (
            "delta_L_rerun",
            "delta_L",
            r"$\Delta L^*$",
        ),
        (
            "delta_hue_deg_rerun",
            "delta_hue_deg",
            r"$\Delta h_{ab}$ [$^\circ$]",
        ),
    ]

    for ax, (
        quarterwave_column,
        fixed_column,
        y_label,
    ) in zip(
        axes,
        comparison_rows,
    ):

        ax.axhline(
            0.0,
            color=ZERO_LINE_COLOUR,
            linewidth=0.7,
        )

        ax.axvline(
            1.0,
            color=REFERENCE_LINE_COLOUR,
            linewidth=0.8,
            linestyle="--",
        )

        ax.plot(
            quarterwave_580[
                "variation_factor"
            ],
            quarterwave_580[
                quarterwave_column
            ],
            color=REFERENCE_COLOURS[580],
            marker=REFERENCE_MARKERS[580],
            linewidth=1.3,
            markersize=4.5,
            label="Quarter-wave geometry",
        )

        ax.plot(
            fixed["variation_factor"],
            fixed[fixed_column],
            color="black",
            marker="s",
            linestyle="--",
            linewidth=1.1,
            markersize=4.0,
            label="Fixed geometry",
        )

        ax.set_ylabel(y_label)
        ax.grid(False)

    axes[-1].set_xlabel(
        r"$(n_A/n_B)/(n_A/n_B)_0$"
    )

    axes[0].legend(
        frameon=False,
        loc="best",
        fontsize=8,
    )

    plt.show()
