#!/usr/bin/env python
# coding: utf-8

# # Core-field enhancement versus refractive-index contrast
#
# Primary outputs:
# - backward-collected spectral overview;
# - field-magnitude line profiles at 470 nm;
# - field-magnitude line profiles at each case's local Bragg peak;
# - wavelength series for one selected case;
# - shell-by-shell cross-sectional averages.
#
# Shell averages are XZ-plane annular averages, not 3D volume averages.

# In[4]:


# Cell 1: imports, paths, and robust plotting helpers
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from IPython.display import display, HTML

OUTPUT_DIR = Path("/mnt/c/Users/SchwarzN/OneDrive - Université de Fribourg/Institution/" \
"P1_BraggSpericalPigments/Simulation/MIE_Computation/Scattnlay_BraggOnion/outputs/field_enhancement_sweep")
FIGURE_DIR = OUTPUT_DIR / "analysis_figures"
TABLE_DIR = OUTPUT_DIR / "analysis_tables"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)
PNG_DPI = 500
DESIGN_WAVELENGTH_NM = 470.0

STYLE_PATH = Path(
    "/mnt/c/Users/SchwarzN/OneDrive - Université de Fribourg/"
    "Institution/Other/Scripts_and_HowTo/science_paper.mplstyle"
)
mpl.rcdefaults()
if STYLE_PATH.exists():
    plt.style.use(STYLE_PATH)
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": PNG_DPI, "pdf.fonttype": 42, "ps.fonttype": 42})

def save_figure(fig, stem):
    for suffix in ("png", "pdf"):
        fig.savefig(
            FIGURE_DIR / f"{stem}.{suffix}",
            dpi=PNG_DPI if suffix == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )

def finite_positive(values):
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values) & (values > 0)]

def robust_log10(values, floor=1e-12, ceiling_percentile=99.8):
    values = np.asarray(values, dtype=float)
    positive = finite_positive(values)
    if positive.size == 0:
        return np.full_like(values, np.nan), (-12.0, 0.0)
    ceiling = np.nanpercentile(positive, ceiling_percentile)
    clipped = np.clip(values, floor, ceiling)
    logged = np.log10(clipped)
    return logged, (float(np.log10(floor)), float(np.log10(ceiling)))

# In[5]:


# ============================================================
# CELL 2: LOAD DATA AND CREATE ROBUST ANALYSIS COLUMNS
#
# Spectra:
#   raw -> spike-masked -> interpolated -> broad envelope
#
# Fields:
#   raw -> catastrophic values masked -> short gaps interpolated
#
# Shells:
#   raw mean/median -> catastrophic values masked
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.ndimage import median_filter
from scipy.signal import savgol_filter

from matplotlib.colors import Normalize
from IPython.display import display


# ============================================================
# SETTINGS
# ============================================================

# Spectral processing.
SPECTRUM_MEDIAN_WINDOW = 9
SPECTRUM_SAVGOL_WINDOW = 31
SPECTRUM_SAVGOL_ORDER = 3

SPECTRUM_SPIKE_ROBUST_Z = 5.0
SPECTRUM_SPIKE_MINIMUM_RATIO = 1.5
SPECTRUM_MASK_NEGATIVE = True

# Exploratory field screening.
FIELD_CATASTROPHIC_LIMIT = 1e4
SHELL_CATASTROPHIC_LIMIT = 1e4

# A profile is accepted when at least 45% of its samples survive.
FIELD_MINIMUM_VALID_FRACTION = 0.45

# Interpolate only short internal gaps for visualization.
FIELD_INTERPOLATION_MAXIMUM_GAP_POINTS = 4


# ============================================================
# LOAD TABLES
# ============================================================

def load_preferred(stem):

    for suffix, loader in (
        (".parquet", pd.read_parquet),
        (".pkl", pd.read_pickle),
        (".csv", pd.read_csv),
    ):

        path = OUTPUT_DIR / f"{stem}{suffix}"

        if path.exists():

            frame = loader(path)

            print(
                f"Loaded: {path}  shape={frame.shape}"
            )

            return frame

    raise FileNotFoundError(
        f"No table found for {stem!r} in {OUTPUT_DIR}."
    )


case_summary = load_preferred(
    "case_summary"
)

backward_spectra = load_preferred(
    "backward_spectra"
)

axis_profiles = load_preferred(
    "axis_profiles"
)

shell_plane_averages = load_preferred(
    "shell_plane_averages"
)


# ============================================================
# VALIDATE AND NORMALIZE TYPES
# ============================================================

required_columns = {
    "case_summary": {
        "case_label",
        "n_low",
        "n_high",
        "k_all_layers",
        "absorbing",
    },
    "backward_spectra": {
        "case_label",
        "wavelength_nm",
        "backward_collected_spectrum",
    },
    "axis_profiles": {
        "case_label",
        "wavelength_nm",
        "z_nm",
        "inside_sphere",
        "E_magnitude_enhancement",
        "E_intensity_enhancement",
    },
    "shell_plane_averages": {
        "case_label",
        "wavelength_nm",
        "shell_number_one_based",
        "mean_E_magnitude_plane",
    },
}

frames = {
    "case_summary": case_summary,
    "backward_spectra": backward_spectra,
    "axis_profiles": axis_profiles,
    "shell_plane_averages": shell_plane_averages,
}

for name, required in required_columns.items():

    missing = required - set(
        frames[name].columns
    )

    if missing:

        raise KeyError(
            f"{name} is missing columns: {sorted(missing)}"
        )


numeric_columns = (
    "n_low",
    "n_high",
    "k_all_layers",
    "wavelength_nm",
    "design_wavelength_nm",
    "backward_peak_wavelength_nm",
    "z_nm",
    "z_over_outer_radius",
    "radius_nm",
    "shell_number_one_based",
    "E_magnitude_enhancement",
    "E_intensity_enhancement",
    "mean_E_magnitude_plane",
    "median_E_magnitude_plane",
    "max_E_magnitude_plane",
    "mean_E_intensity_plane",
)

for frame in frames.values():

    for column in numeric_columns:

        if column in frame.columns:

            frame[column] = pd.to_numeric(
                frame[column],
                errors="coerce",
            )


def normalize_boolean(series):

    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)

    mapping = {
        "true": True,
        "1": True,
        "yes": True,
        "false": False,
        "0": False,
        "no": False,
    }

    normalized = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(mapping)
    )

    if normalized.isna().any():

        invalid = sorted(
            series.loc[
                normalized.isna()
            ].astype(str).unique()
        )

        raise ValueError(
            f"Unrecognized boolean values: {invalid}"
        )

    return normalized.astype(bool)


for frame in frames.values():

    if "absorbing" in frame.columns:

        frame["absorbing"] = normalize_boolean(
            frame["absorbing"]
        )


axis_profiles["inside_sphere"] = normalize_boolean(
    axis_profiles["inside_sphere"]
)


for frame in frames.values():

    if {
        "n_low",
        "n_high",
    }.issubset(frame.columns):

        frame["delta_n"] = (
            frame["n_high"]
            - frame["n_low"]
        )


def odd_window(
    requested,
    array_length,
    *,
    minimum=3,
):

    largest = (
        array_length
        if array_length % 2 == 1
        else array_length - 1
    )

    window = int(requested)

    if window % 2 == 0:
        window += 1

    return max(
        minimum,
        min(window, largest),
    )


# ============================================================
# PROCESS BACKWARD SPECTRA
# ============================================================

backward_spectra = backward_spectra.copy()

backward_spectra[
    "backward_collected_spectrum_raw"
] = pd.to_numeric(
    backward_spectra[
        "backward_collected_spectrum"
    ],
    errors="coerce",
)

for column in (
    "spectrum_local_median",
    "spectrum_local_scale",
    "spectrum_spike_residual",
    "backward_collected_spectrum_spike_masked",
    "backward_collected_spectrum_interpolated",
    "backward_collected_spectrum_broad_envelope",
):

    backward_spectra[column] = np.nan

backward_spectra[
    "spectrum_numerical_outlier"
] = False


for case_label, indices in (
    backward_spectra
    .groupby("case_label")
    .groups
    .items()
):

    group = (
        backward_spectra.loc[
            list(indices)
        ]
        .sort_values("wavelength_nm")
    )

    wavelengths = group[
        "wavelength_nm"
    ].to_numpy(float)

    values = group[
        "backward_collected_spectrum_raw"
    ].to_numpy(float)

    finite = np.isfinite(values)

    if not finite.any():
        continue

    temporary = (
        pd.Series(
            values,
            index=wavelengths,
        )
        .interpolate(
            method="index",
            limit_direction="both",
        )
        .to_numpy(float)
    )

    median_window = odd_window(
        SPECTRUM_MEDIAN_WINDOW,
        len(temporary),
    )

    local_median = median_filter(
        temporary,
        size=median_window,
        mode="nearest",
    )

    residual = (
        temporary
        - local_median
    )

    absolute_residual = np.abs(
        residual
    )

    local_mad = median_filter(
        absolute_residual,
        size=median_window,
        mode="nearest",
    )

    finite_residuals = absolute_residual[
        np.isfinite(absolute_residual)
    ]

    global_mad = (
        float(
            np.nanmedian(
                finite_residuals
            )
        )
        if finite_residuals.size
        else 0.0
    )

    fallback_scale = max(
        1.4826 * global_mad,
        np.finfo(float).eps,
    )

    local_scale = np.maximum(
        1.4826 * local_mad,
        fallback_scale,
    )

    robust_z = (
        residual
        / local_scale
    )

    finite_temporary = temporary[
        np.isfinite(temporary)
    ]

    global_reference = (
        float(
            np.nanmedian(
                np.abs(
                    finite_temporary
                )
            )
        )
        if finite_temporary.size
        else 1.0
    )

    local_reference = np.maximum(
        np.abs(local_median),
        global_reference * 1e-6,
    )

    positive_residual_ratio = (
        np.maximum(
            residual,
            0.0,
        )
        / np.maximum(
            local_reference,
            np.finfo(float).tiny,
        )
    )

    spike_mask = (
        ~finite
        | (
            (
                robust_z
                > SPECTRUM_SPIKE_ROBUST_Z
            )
            & (
                positive_residual_ratio
                > SPECTRUM_SPIKE_MINIMUM_RATIO
            )
        )
    )

    if SPECTRUM_MASK_NEGATIVE:

        spike_mask |= (
            values < 0.0
        )

    spike_masked = values.copy()

    spike_masked[
        spike_mask
    ] = np.nan

    interpolated = (
        pd.Series(
            spike_masked,
            index=wavelengths,
        )
        .interpolate(
            method="index",
            limit_direction="both",
        )
        .to_numpy(float)
    )

    envelope_median = median_filter(
        interpolated,
        size=median_window,
        mode="nearest",
    )

    savgol_window = odd_window(
        SPECTRUM_SAVGOL_WINDOW,
        len(envelope_median),
        minimum=(
            SPECTRUM_SAVGOL_ORDER
            + 2
        ),
    )

    if (
        savgol_window
        > SPECTRUM_SAVGOL_ORDER
    ):

        broad_envelope = savgol_filter(
            envelope_median,
            window_length=savgol_window,
            polyorder=SPECTRUM_SAVGOL_ORDER,
            mode="interp",
        )

    else:

        broad_envelope = (
            envelope_median
        )

    broad_envelope = np.maximum(
        broad_envelope,
        0.0,
    )

    target_index = group.index

    backward_spectra.loc[
        target_index,
        "spectrum_local_median",
    ] = local_median

    backward_spectra.loc[
        target_index,
        "spectrum_local_scale",
    ] = local_scale

    backward_spectra.loc[
        target_index,
        "spectrum_spike_residual",
    ] = residual

    backward_spectra.loc[
        target_index,
        "spectrum_numerical_outlier",
    ] = spike_mask

    backward_spectra.loc[
        target_index,
        "backward_collected_spectrum_spike_masked",
    ] = spike_masked

    backward_spectra.loc[
        target_index,
        "backward_collected_spectrum_interpolated",
    ] = interpolated

    backward_spectra.loc[
        target_index,
        "backward_collected_spectrum_broad_envelope",
    ] = broad_envelope


backward_spectra[
    "backward_collected_spectrum_clean"
] = backward_spectra[
    "backward_collected_spectrum_spike_masked"
]


# ============================================================
# SCREEN FIELD VALUES
# ============================================================

axis_profiles = axis_profiles.copy()

axis_profiles[
    "E_magnitude_enhancement_raw"
] = pd.to_numeric(
    axis_profiles[
        "E_magnitude_enhancement"
    ],
    errors="coerce",
)

axis_profiles[
    "E_intensity_enhancement_raw"
] = pd.to_numeric(
    axis_profiles[
        "E_intensity_enhancement"
    ],
    errors="coerce",
)

axis_profiles[
    "field_numerical_outlier"
] = (
    ~np.isfinite(
        axis_profiles[
            "E_magnitude_enhancement_raw"
        ]
    )
    | (
        axis_profiles[
            "E_magnitude_enhancement_raw"
        ]
        < 0.0
    )
    | (
        axis_profiles[
            "E_magnitude_enhancement_raw"
        ]
        > FIELD_CATASTROPHIC_LIMIT
    )
)

axis_profiles[
    "E_magnitude_enhancement_clean"
] = axis_profiles[
    "E_magnitude_enhancement_raw"
].mask(
    axis_profiles[
        "field_numerical_outlier"
    ]
)

axis_profiles[
    "E_intensity_enhancement_clean"
] = axis_profiles[
    "E_intensity_enhancement_raw"
].mask(
    axis_profiles[
        "field_numerical_outlier"
    ]
)


# ============================================================
# CREATE SHORT-GAP-INTERPOLATED PLOTTING PROFILES
# ============================================================

axis_profiles[
    "E_magnitude_enhancement_plot"
] = np.nan

axis_profiles[
    "E_intensity_enhancement_plot"
] = np.nan


for group_key, indices in (
    axis_profiles
    .groupby(
        [
            "case_label",
            "wavelength_nm",
        ]
    )
    .groups
    .items()
):

    group = (
        axis_profiles.loc[
            list(indices)
        ]
        .sort_values("z_nm")
    )

    z_values = group[
        "z_nm"
    ].to_numpy(float)

    clean_values = group[
        "E_magnitude_enhancement_clean"
    ].to_numpy(float)

    clean_series = pd.Series(
        clean_values,
        index=z_values,
        dtype=float,
    )

    plot_series = (
        clean_series
        .interpolate(
            method="index",
            limit=(
                FIELD_INTERPOLATION_MAXIMUM_GAP_POINTS
            ),
            limit_area="inside",
            limit_direction="both",
        )
    )

    plot_series = plot_series.where(
        np.isfinite(plot_series)
        & (plot_series >= 0.0)
        & (
            plot_series
            <= FIELD_CATASTROPHIC_LIMIT
        )
    )

    plot_values = plot_series.to_numpy(
        dtype=float
    )

    with np.errstate(
        over="ignore",
        invalid="ignore",
    ):

        plot_intensity = (
            plot_values
            * plot_values
        )

    plot_intensity[
        ~np.isfinite(plot_intensity)
    ] = np.nan

    axis_profiles.loc[
        group.index,
        "E_magnitude_enhancement_plot",
    ] = plot_values

    axis_profiles.loc[
        group.index,
        "E_intensity_enhancement_plot",
    ] = plot_intensity


# ============================================================
# CALCULATE PROFILE QUALITY AFTER MASKING AND INTERPOLATION
# ============================================================

profile_quality = (
    axis_profiles
    .groupby(
        [
            "case_label",
            "wavelength_nm",
        ],
        as_index=False,
    )
    .agg(
        profile_sample_count=(
            "E_magnitude_enhancement_raw",
            "size",
        ),
        profile_clean_count=(
            "E_magnitude_enhancement_clean",
            "count",
        ),
        profile_plot_count=(
            "E_magnitude_enhancement_plot",
            "count",
        ),
        profile_raw_max=(
            "E_magnitude_enhancement_raw",
            "max",
        ),
        profile_clean_max=(
            "E_magnitude_enhancement_clean",
            "max",
        ),
        profile_plot_max=(
            "E_magnitude_enhancement_plot",
            "max",
        ),
    )
)

profile_quality[
    "profile_valid_fraction"
] = (
    profile_quality[
        "profile_clean_count"
    ]
    / profile_quality[
        "profile_sample_count"
    ]
)

profile_quality[
    "plot_profile_valid_fraction"
] = (
    profile_quality[
        "profile_plot_count"
    ]
    / profile_quality[
        "profile_sample_count"
    ]
)

profile_quality[
    "profile_is_usable"
] = (
    profile_quality[
        "profile_valid_fraction"
    ]
    >= FIELD_MINIMUM_VALID_FRACTION
)


# Make this cell safely rerunnable.
for column in (
    "profile_valid_fraction",
    "plot_profile_valid_fraction",
    "profile_is_usable",
):

    if column in axis_profiles.columns:

        axis_profiles = (
            axis_profiles.drop(
                columns=column
            )
        )


axis_profiles = axis_profiles.merge(
    profile_quality[
        [
            "case_label",
            "wavelength_nm",
            "profile_valid_fraction",
            "plot_profile_valid_fraction",
            "profile_is_usable",
        ]
    ],
    on=[
        "case_label",
        "wavelength_nm",
    ],
    how="left",
    validate="many_to_one",
)


# ============================================================
# SCREEN SHELL AVERAGES
# ============================================================

shell_plane_averages = (
    shell_plane_averages.copy()
)

shell_plane_averages[
    "mean_E_magnitude_plane_raw"
] = pd.to_numeric(
    shell_plane_averages[
        "mean_E_magnitude_plane"
    ],
    errors="coerce",
)

shell_plane_averages[
    "shell_numerical_outlier"
] = (
    ~np.isfinite(
        shell_plane_averages[
            "mean_E_magnitude_plane_raw"
        ]
    )
    | (
        shell_plane_averages[
            "mean_E_magnitude_plane_raw"
        ]
        < 0.0
    )
    | (
        shell_plane_averages[
            "mean_E_magnitude_plane_raw"
        ]
        > SHELL_CATASTROPHIC_LIMIT
    )
)

shell_plane_averages[
    "mean_E_magnitude_plane_clean"
] = shell_plane_averages[
    "mean_E_magnitude_plane_raw"
].mask(
    shell_plane_averages[
        "shell_numerical_outlier"
    ]
)

if (
    "median_E_magnitude_plane"
    in shell_plane_averages.columns
):

    median_values = pd.to_numeric(
        shell_plane_averages[
            "median_E_magnitude_plane"
        ],
        errors="coerce",
    )

    shell_plane_averages[
        "median_E_magnitude_plane_clean"
    ] = median_values.where(
        np.isfinite(median_values)
        & (median_values >= 0.0)
        & (
            median_values
            <= SHELL_CATASTROPHIC_LIMIT
        )
    )


# ============================================================
# RECREATE CASE GROUPS
# ============================================================

lossless_cases = (
    case_summary.loc[
        ~case_summary["absorbing"]
    ]
    .sort_values("n_high")
    .copy()
)

absorbing_cases = (
    case_summary.loc[
        case_summary["absorbing"]
    ]
    .sort_values("n_high")
    .copy()
)

norm = Normalize(
    vmin=float(
        lossless_cases["n_high"].min()
    ),
    vmax=float(
        lossless_cases["n_high"].max()
    ),
)

cmap = plt.get_cmap(
    "viridis"
)


# ============================================================
# COMPACT DIAGNOSTICS
# ============================================================

spectrum_diagnostics = (
    backward_spectra
    .groupby(
        [
            "case_label",
            "n_high",
            "k_all_layers",
            "absorbing",
        ],
        as_index=False,
    )
    .agg(
        samples=(
            "wavelength_nm",
            "size",
        ),
        removed_spikes=(
            "spectrum_numerical_outlier",
            "sum",
        ),
        raw_max=(
            "backward_collected_spectrum_raw",
            "max",
        ),
        envelope_max=(
            "backward_collected_spectrum_broad_envelope",
            "max",
        ),
    )
)

spectrum_diagnostics[
    "removed_fraction"
] = (
    spectrum_diagnostics[
        "removed_spikes"
    ]
    / spectrum_diagnostics[
        "samples"
    ]
)

# In[20]:


# ============================================================
# CELL 3: BROAD BACKWARD-SPECTRAL ENVELOPES
#
# Plot:
#   broad spectral envelope only
#
# Peak detection:
#   maximum of the broad envelope in the selected interval
#
# Raw calculated spectra remain available in:
#   backward_collected_spectrum_raw
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.cm import ScalarMappable


# ============================================================
# SETTINGS
# ============================================================

ANALYSIS_PEAK_MIN_NM = 380.0
ANALYSIS_PEAK_MAX_NM = 550.0
ANALYSIS_PEAK_EDGE_MARGIN_NM = 10.0
ANALYSIS_PEAK_MINIMUM_PROMINENCE_RATIO = 0.10

SHOW_PEAK_MARKERS = True

SPECTRUM_X_LIMITS = (
    300.0,
    900.0,
)

SPECTRUM_Y_LIMITS = (
    0.0,
    1.21,
)

SPECTRUM_FIGURE_WIDTH_CM = 17.5
SPECTRUM_FIGURE_HEIGHT_CM = 8.0

SAVE_BACKWARD_SPECTRA_FIGURE = True

PLOT_COLUMN = (
    "backward_collected_spectrum_broad_envelope"
)


# ============================================================
# VALIDATION
# ============================================================

required_columns = {
    "case_label",
    "wavelength_nm",
    "n_high",
    "k_all_layers",
    "absorbing",
    PLOT_COLUMN,
}

missing = (
    required_columns
    - set(
        backward_spectra.columns
    )
)

if missing:

    raise KeyError(
        "Run the updated Cell 2 first. Missing columns: "
        f"{sorted(missing)}"
    )


# ============================================================
# DETERMINE PEAKS FROM BROAD ENVELOPES
# ============================================================

peak_records = []


for case_label, data in (
    backward_spectra.groupby(
        "case_label",
        sort=False,
    )
):

    data = data.sort_values(
        "wavelength_nm"
    )

    wavelength_nm = data[
        "wavelength_nm"
    ].to_numpy(float)

    envelope = data[
        PLOT_COLUMN
    ].to_numpy(float)

    search_mask = (
        np.isfinite(wavelength_nm)
        & np.isfinite(envelope)
        & (
            wavelength_nm
            >= ANALYSIS_PEAK_MIN_NM
        )
        & (
            wavelength_nm
            <= ANALYSIS_PEAK_MAX_NM
        )
    )

    metadata = (
        case_summary.loc[
            case_summary["case_label"]
            == case_label
        ]
        .iloc[0]
    )

    if not np.any(search_mask):

        peak_records.append(
            {
                "case_label": case_label,
                "analysis_peak_wavelength_nm": np.nan,
                "analysis_peak_value": np.nan,
                "analysis_peak_baseline": np.nan,
                "analysis_peak_prominence_ratio": np.nan,
                "analysis_peak_at_window_edge": True,
                "analysis_peak_is_reliable": False,
            }
        )

        continue

    search_indices = np.flatnonzero(
        search_mask
    )

    peak_index = int(
        search_indices[
            np.nanargmax(
                envelope[
                    search_mask
                ]
            )
        ]
    )

    peak_wavelength = float(
        wavelength_nm[
            peak_index
        ]
    )

    peak_value = float(
        envelope[
            peak_index
        ]
    )

    baseline = float(
        np.nanmedian(
            envelope[
                search_mask
            ]
        )
    )

    prominence_ratio = (
        (
            peak_value
            - baseline
        )
        / max(
            abs(baseline),
            np.finfo(float).tiny,
        )
    )

    at_edge = bool(
        peak_wavelength
        <= (
            ANALYSIS_PEAK_MIN_NM
            + ANALYSIS_PEAK_EDGE_MARGIN_NM
        )
        or peak_wavelength
        >= (
            ANALYSIS_PEAK_MAX_NM
            - ANALYSIS_PEAK_EDGE_MARGIN_NM
        )
    )

    reliable = bool(
        np.isfinite(prominence_ratio)
        and (
            prominence_ratio
            >= ANALYSIS_PEAK_MINIMUM_PROMINENCE_RATIO
        )
        and not at_edge
    )

    peak_records.append(
        {
            "case_label": case_label,
            "analysis_peak_wavelength_nm": peak_wavelength,
            "analysis_peak_value": peak_value,
            "analysis_peak_baseline": baseline,
            "analysis_peak_prominence_ratio": prominence_ratio,
            "analysis_peak_at_window_edge": at_edge,
            "analysis_peak_is_reliable": reliable,
        }
    )


analysis_peak_table = pd.DataFrame(
    peak_records
)


analysis_peak_columns = [
    "analysis_peak_wavelength_nm",
    "analysis_peak_value",
    "analysis_peak_baseline",
    "analysis_peak_prominence_ratio",
    "analysis_peak_at_window_edge",
    "analysis_peak_is_reliable",
]

case_summary = case_summary.drop(
    columns=[
        column
        for column in analysis_peak_columns
        if column in case_summary.columns
    ],
    errors="ignore",
)

case_summary = case_summary.merge(
    analysis_peak_table,
    on="case_label",
    how="left",
    validate="one_to_one",
)


lossless_cases = (
    case_summary.loc[
        ~case_summary["absorbing"]
    ]
    .sort_values("n_high")
    .copy()
)

absorbing_cases = (
    case_summary.loc[
        case_summary["absorbing"]
    ]
    .sort_values("n_high")
    .copy()
)


# ============================================================
# DISPLAY CLEAN PEAK TABLE
# ============================================================

peak_display_columns = [
    "case_label",
    "n_high",
    "k_all_layers",
    "absorbing",
    "analysis_peak_wavelength_nm",
    "analysis_peak_value",
    "analysis_peak_prominence_ratio",
    "analysis_peak_at_window_edge",
    "analysis_peak_is_reliable",
]

peak_display_table = (
    case_summary
    .sort_values(
        [
            "absorbing",
            "n_high",
        ]
    )
    .loc[
        :,
        peak_display_columns,
    ]
    .reset_index(
        drop=True
    )
)

display(
    peak_display_table.style.format(
        {
            "n_high": "{:.3f}",
            "k_all_layers": "{:.3f}",
            # "analysis_peak_at_window_edge": "{:.0f}",
            "analysis_peak_wavelength_nm": "{:.1f}",
            "analysis_peak_value": "{:.5g}",
            "analysis_peak_prominence_ratio": "{:.3f}",
        },
        na_rep="not available",
    )
)


analysis_peak_table.to_csv(
    TABLE_DIR
    / "broad_envelope_peak_table.csv",
    index=False,
)


# ============================================================
# PLOT
# ============================================================

cm_to_inches = 1.0 / 2.54

fig, ax = plt.subplots(
    figsize=(
        SPECTRUM_FIGURE_WIDTH_CM
        * cm_to_inches,
        SPECTRUM_FIGURE_HEIGHT_CM
        * cm_to_inches,
    ),
    constrained_layout=True,
)


# Lossless series.
for _, case in lossless_cases.iterrows():

    data = (
        backward_spectra.loc[
            backward_spectra["case_label"]
            == case["case_label"]
        ]
        .sort_values("wavelength_nm")
    )

    colour = cmap(
        norm(
            float(
                case["n_high"]
            )
        )
    )

    ax.plot(
        data["wavelength_nm"],
        data[PLOT_COLUMN],
        color=colour,
        linewidth=1.05,
        alpha=0.95,
        zorder=2,
    )

    if (
        SHOW_PEAK_MARKERS
        and bool(
            case[
                "analysis_peak_is_reliable"
            ]
        )
    ):

        ax.scatter(
            case[
                "analysis_peak_wavelength_nm"
            ],
            case[
                "analysis_peak_value"
            ],
            s=13,
            color=colour,
            edgecolor="black",
            linewidth=0.35,
            zorder=5,
        )


# Absorbing comparison cases.
for _, case in absorbing_cases.iterrows():

    data = (
        backward_spectra.loc[
            backward_spectra["case_label"]
            == case["case_label"]
        ]
        .sort_values("wavelength_nm")
    )

    colour = cmap(
        norm(
            float(
                case["n_high"]
            )
        )
    )

    ax.plot(
        data["wavelength_nm"],
        data[PLOT_COLUMN],
        color=colour,
        linestyle="--",
        linewidth=1.35,
        label=(
            rf"$n_h={case['n_high']:.2f}$, "
            rf"$k={case['k_all_layers']:.3f}$"
        ),
        zorder=3,
    )

    if (
        SHOW_PEAK_MARKERS
        and bool(
            case[
                "analysis_peak_is_reliable"
            ]
        )
    ):

        ax.scatter(
            case[
                "analysis_peak_wavelength_nm"
            ],
            case[
                "analysis_peak_value"
            ],
            s=18,
            marker="s",
            color=colour,
            edgecolor="black",
            linewidth=0.4,
            zorder=5,
        )


ax.axvline(
    DESIGN_WAVELENGTH_NM,
    color="black",
    linestyle=":",
    linewidth=1.0,
    label=(
        rf"Design $\lambda_0="
        f"{DESIGN_WAVELENGTH_NM:.0f}"
        r"\,\mathrm{nm}$"
    ),
)


ax.axvspan(
    ANALYSIS_PEAK_MIN_NM,
    ANALYSIS_PEAK_MAX_NM,
    color="0.5",
    alpha=0.035,
    linewidth=0.0,
)


ax.set_xlim(
    *SPECTRUM_X_LIMITS
)

ax.set_ylim(
    *SPECTRUM_Y_LIMITS
)

ax.set_xlabel(
    "Wavelength [nm]"
)

ax.set_ylabel(
    "Backward-collected efficiency"
)

ax.set_title(
    "Broad backward-collected spectral envelopes"
)

# ax.legend(
#     fontsize=6.5,
#     ncol=5,
#     frameon=False,
#     loc="upper right",
# )


scalar = ScalarMappable(
    norm=norm,
    cmap=cmap,
)

scalar.set_array([])

colorbar = fig.colorbar(
    scalar,
    ax=ax,
    pad=0.02,
    fraction=0.045,
)

colorbar.set_label(
    r"Lossless-series $n_h$"
)


if SAVE_BACKWARD_SPECTRA_FIGURE:

    save_figure(
        fig,
        "backward_collected_broad_envelopes",
    )


plt.show()

plt.close(fig)

# In[ ]:


# ============================================================
# CELL 4: COMPACT COLOUR-SWATCH COMPARISON TABLE
#
# Rows:
#   high refractive index n_h
#
# Columns:
#   lossless, k = 0
#   absorbing, k = 0.045
#
# Important:
#   These colours were calculated by the original sweep from the
#   original spectra. They have not been recalculated from the
#   cleaned broad spectral envelopes.
# ============================================================

import re

import numpy as np
import pandas as pd

from IPython.display import HTML, display


# ============================================================
# SETTINGS
# ============================================================

SHOW_COLOUR_NUMERICAL_TABLE = True

SAVE_COLOUR_TABLES = True

COLOUR_TABLE_HTML_NAME = (
    "calculated_colour_swatches_raw_spectra.html"
)

COLOUR_TABLE_CSV_NAME = (
    "calculated_colour_metrics_raw_spectra.csv"
)


# ============================================================
# IDENTIFY REQUIRED COLOUR COLUMNS EXPLICITLY
# ============================================================

colour_column_map = {
    "hex": "sweep_Hex_hex",
    "L_star": "sweep_CIELAB_L",
    "a_star": "sweep_CIELAB_a",
    "b_star": "sweep_CIELAB_b",
    "chroma": "sweep_CIELAB_C",
    "hue_deg": "sweep_CIELAB_hue_deg",
    "sRGB_r": "sweep_sRGB_r",
    "sRGB_g": "sweep_sRGB_g",
    "sRGB_b": "sweep_sRGB_b",
    "sRGB_r_clipped": (
        "sweep_sRGB_r_clipped"
    ),
    "sRGB_g_clipped": (
        "sweep_sRGB_g_clipped"
    ),
    "sRGB_b_clipped": (
        "sweep_sRGB_b_clipped"
    ),
}


missing_colour_columns = [
    column
    for column in colour_column_map.values()
    if column not in case_summary.columns
]


if missing_colour_columns:

    print(
        "Colour table not created because these columns "
        "are unavailable:"
    )

    for column in missing_colour_columns:
        print("  ", column)

else:

    # --------------------------------------------------------
    # Prepare a clean colour-data table
    # --------------------------------------------------------

    colour_table = case_summary[
        [
            "case_label",
            "n_high",
            "k_all_layers",
            "absorbing",
            *colour_column_map.values(),
        ]
    ].copy()


    colour_table = colour_table.rename(
        columns={
            source: destination
            for destination, source
            in colour_column_map.items()
        }
    )


    # Normalize hexadecimal strings.
    def normalize_hex_colour(
        value,
    ):
        """
        Return a valid #RRGGBB colour or neutral grey.
        """

        text = str(value).strip()

        if re.fullmatch(
            r"#[0-9A-Fa-f]{6}",
            text,
        ):
            return text.upper()

        return "#808080"


    colour_table["hex"] = (
        colour_table["hex"]
        .map(
            normalize_hex_colour
        )
    )


    # Add spectrum-cleaning information when available.
    if (
        "spectrum_diagnostics"
        in globals()
    ):

        available_diagnostic_columns = [
            column
            for column in [
                "case_label",
                "removed_spikes",
                "removed_fraction",
                "spectrum_outlier_count",
                "spectrum_outlier_fraction",
            ]
            if column
            in spectrum_diagnostics.columns
        ]

        if (
            "case_label"
            in available_diagnostic_columns
        ):

            colour_table = (
                colour_table.merge(
                    spectrum_diagnostics[
                        available_diagnostic_columns
                    ],
                    on="case_label",
                    how="left",
                    validate="one_to_one",
                )
            )


    colour_table = (
        colour_table
        .sort_values(
            [
                "n_high",
                "k_all_layers",
            ]
        )
        .reset_index(
            drop=True
        )
    )


    # --------------------------------------------------------
    # Build the swatch-matrix table
    # --------------------------------------------------------

    all_n_high = np.sort(
        colour_table[
            "n_high"
        ].dropna().unique()
    )


    def select_colour_case(
        n_high,
        k_value,
    ):
        """
        Select one colour calculation for n_high and k.
        """

        selected = colour_table.loc[
            np.isclose(
                colour_table["n_high"],
                float(n_high),
            )
            & np.isclose(
                colour_table[
                    "k_all_layers"
                ],
                float(k_value),
            )
        ]

        if selected.empty:
            return None

        return selected.iloc[0]


    def make_colour_cell(
        row,
    ):
        """
        Generate an HTML swatch and concise colour metrics.
        """

        if row is None:

            return (
                '<div class="missing-case">'
                "not calculated"
                "</div>"
            )

        hex_colour = row["hex"]

        lightness = float(
            row["L_star"]
        )

        chroma = float(
            row["chroma"]
        )

        hue = float(
            row["hue_deg"]
        )

        return f"""
        <div class="colour-case">
            <div
                class="colour-swatch"
                style="background-color:{hex_colour};"
                title="{hex_colour}">
            </div>

            <div class="colour-values">
                <strong>{hex_colour}</strong><br>
                L* = {lightness:.1f}<br>
                C* = {chroma:.1f}<br>
                h = {hue:.1f}°
            </div>
        </div>
        """


    html_rows = []


    for n_high in all_n_high:

        lossless_row = select_colour_case(
            n_high,
            0.0,
        )

        absorbing_row = select_colour_case(
            n_high,
            0.045,
        )

        html_rows.append(
            f"""
            <tr>
                <td class="index-value">
                    {n_high:.3f}
                </td>

                <td>
                    {make_colour_cell(lossless_row)}
                </td>

                <td>
                    {make_colour_cell(absorbing_row)}
                </td>
            </tr>
            """
        )


    html_document = f"""
    <style>
        .colour-comparison-table {{
            border-collapse: collapse;
            font-family: sans-serif;
            font-size: 13px;
            margin-top: 8px;
        }}

        .colour-comparison-table th,
        .colour-comparison-table td {{
            border: 1px solid #B0B0B0;
            padding: 7px 9px;
            text-align: center;
            vertical-align: middle;
        }}

        .colour-comparison-table th {{
            background: #F0F0F0;
            font-weight: 600;
        }}

        .index-value {{
            font-variant-numeric: tabular-nums;
            font-weight: 600;
            min-width: 60px;
        }}

        .colour-case {{
            display: flex;
            align-items: center;
            gap: 9px;
            min-width: 155px;
            text-align: left;
        }}

        .colour-swatch {{
            width: 58px;
            height: 36px;
            border: 1px solid #404040;
            box-sizing: border-box;
            flex-shrink: 0;
        }}

        .colour-values {{
            line-height: 1.35;
            font-size: 11.5px;
            font-variant-numeric: tabular-nums;
        }}

        .missing-case {{
            color: #777777;
            font-style: italic;
            min-width: 155px;
        }}

        .colour-note {{
            max-width: 720px;
            margin-top: 9px;
            font-size: 12px;
            line-height: 1.4;
            color: #444444;
        }}
    </style>

    <table class="colour-comparison-table">
        <thead>
            <tr>
                <th>
                    High index n<sub>h</sub>
                </th>

                <th>
                    Lossless<br>
                    k = 0
                </th>

                <th>
                    Absorbing<br>
                    k = 0.045
                </th>
            </tr>
        </thead>

        <tbody>
            {''.join(html_rows)}
        </tbody>
    </table>

    <div class="colour-note">
        <strong>Note:</strong>
        displayed colours and CIELAB coordinates were calculated
        from the original simulated spectra. The numerical-spike
        cleaning and broad-envelope smoothing used in the spectral
        figure have not been applied to the colour calculation.
    </div>
    """


    display(
        HTML(
            html_document
        )
    )


    # --------------------------------------------------------
    # Optional numerical colour table
    # --------------------------------------------------------

    if SHOW_COLOUR_NUMERICAL_TABLE:

        numerical_columns = [
            "n_high",
            "k_all_layers",
            "absorbing",
            "hex",
            "L_star",
            "a_star",
            "b_star",
            "chroma",
            "hue_deg",
            "sRGB_r_clipped",
            "sRGB_g_clipped",
            "sRGB_b_clipped",
        ]

        for optional_column in [
            "removed_spikes",
            "removed_fraction",
            "spectrum_outlier_count",
            "spectrum_outlier_fraction",
        ]:

            if (
                optional_column
                in colour_table.columns
            ):

                numerical_columns.append(
                    optional_column
                )


        colour_numerical_table = (
            colour_table[
                numerical_columns
            ]
            .sort_values(
                [
                    "absorbing",
                    "n_high",
                ]
            )
            .reset_index(
                drop=True
            )
        )


        display(
            colour_numerical_table.style.format(
                {
                    "n_high": "{:.3f}",
                    "k_all_layers": "{:.3f}",
                    "L_star": "{:.2f}",
                    "a_star": "{:.2f}",
                    "b_star": "{:.2f}",
                    "chroma": "{:.2f}",
                    "hue_deg": "{:.1f}",
                    "sRGB_r_clipped": "{:.3f}",
                    "sRGB_g_clipped": "{:.3f}",
                    "sRGB_b_clipped": "{:.3f}",
                    "removed_spikes": "{:.0f}",
                    "removed_fraction": "{:.3%}",
                    "spectrum_outlier_count": "{:.0f}",
                    "spectrum_outlier_fraction": "{:.3%}",
                },
                na_rep="not available",
            )
        )


    # --------------------------------------------------------
    # Save tables
    # --------------------------------------------------------

    if SAVE_COLOUR_TABLES:

        html_path = (
            TABLE_DIR
            / COLOUR_TABLE_HTML_NAME
        )

        csv_path = (
            TABLE_DIR
            / COLOUR_TABLE_CSV_NAME
        )


        html_path.write_text(
            html_document,
            encoding="utf-8",
        )


        colour_table.to_csv(
            csv_path,
            index=False,
        )


        print(
            "Saved:",
            html_path,
        )

        print(
            "Saved:",
            csv_path,
        )

# In[ ]:


# ============================================================
# CELL 5: SELECT EXPLORATORY DESIGN- AND PEAK-WAVELENGTH PROFILES
#
# Profiles use:
#   E_magnitude_enhancement_plot
#
# This contains:
#   - catastrophic values masked;
#   - short internal gaps interpolated;
#   - long invalid regions retained as NaN.
# ============================================================

import numpy as np
import pandas as pd

from IPython.display import display


# ============================================================
# SETTINGS
# ============================================================

PROFILE_MAXIMUM_WAVELENGTH_MISMATCH_NM = 10.0

PROFILE_REQUIRE_USABLE = True
PROFILE_RAISE_IF_UNAVAILABLE = False

PROFILE_FIELD_COLUMN = (
    "E_magnitude_enhancement_plot"
)


# ============================================================
# VALIDATION
# ============================================================

required_axis_columns = {
    "case_label",
    "wavelength_nm",
    "z_nm",
    "z_over_outer_radius",
    "inside_sphere",
    PROFILE_FIELD_COLUMN,
    "profile_valid_fraction",
    "plot_profile_valid_fraction",
    "profile_is_usable",
}

missing = (
    required_axis_columns
    - set(axis_profiles.columns)
)

if missing:

    raise KeyError(
        "Run the updated Cell 2 first. "
        f"Missing columns: {sorted(missing)}"
    )

if (
    "analysis_peak_wavelength_nm"
    not in case_summary.columns
):

    raise KeyError(
        "Run Cell 3 before Cell 5."
    )


# ============================================================
# HELPERS
# ============================================================

def available_profile_wavelengths(
    case_label,
    *,
    require_usable=PROFILE_REQUIRE_USABLE,
):

    quality = (
        axis_profiles.loc[
            axis_profiles["case_label"]
            == case_label,
            [
                "wavelength_nm",
                "profile_is_usable",
                "profile_valid_fraction",
                "plot_profile_valid_fraction",
            ],
        ]
        .drop_duplicates(
            "wavelength_nm"
        )
        .sort_values(
            "wavelength_nm"
        )
    )

    if require_usable:

        quality = quality.loc[
            quality[
                "profile_is_usable"
            ].astype(bool)
        ]

    return quality[
        "wavelength_nm"
    ].dropna().to_numpy(float)


def nearest_profile_wavelength(
    case_label,
    target_nm,
    *,
    require_usable=PROFILE_REQUIRE_USABLE,
    maximum_mismatch_nm=(
        PROFILE_MAXIMUM_WAVELENGTH_MISMATCH_NM
    ),
):

    available = available_profile_wavelengths(
        case_label,
        require_usable=require_usable,
    )

    if available.size == 0:

        return (
            np.nan,
            np.nan,
        )

    selected = float(
        available[
            np.argmin(
                np.abs(
                    available
                    - float(target_nm)
                )
            )
        ]
    )

    mismatch = abs(
        selected
        - float(target_nm)
    )

    if (
        maximum_mismatch_nm is not None
        and mismatch
        > maximum_mismatch_nm
    ):

        return (
            np.nan,
            float(mismatch),
        )

    return (
        selected,
        float(mismatch),
    )


def unavailable_result(
    case,
    mode,
    target_nm,
    reason,
    *,
    mismatch_nm=np.nan,
):

    diagnostics = {
        "case_label": case["case_label"],
        "mode": mode,
        "target_wavelength_nm": target_nm,
        "selected_wavelength_nm": np.nan,
        "wavelength_mismatch_nm": mismatch_nm,
        "profile_valid_fraction": np.nan,
        "plot_profile_valid_fraction": np.nan,
        "profile_is_usable": False,
        "inside_sphere_sample_count": 0,
        "finite_plot_sample_count": 0,
        "selection_successful": False,
        "reason": reason,
    }

    if PROFILE_RAISE_IF_UNAVAILABLE:

        raise ValueError(
            f"{case['case_label']}: {reason}"
        )

    return (
        np.nan,
        pd.DataFrame(),
        diagnostics,
    )


def profile_at(
    case,
    mode,
    *,
    require_usable=PROFILE_REQUIRE_USABLE,
    maximum_mismatch_nm=(
        PROFILE_MAXIMUM_WAVELENGTH_MISMATCH_NM
    ),
):

    if mode == "design":

        target_nm = float(
            DESIGN_WAVELENGTH_NM
        )

    elif mode == "peak":

        target_nm = float(
            case[
                "analysis_peak_wavelength_nm"
            ]
        )

        if not np.isfinite(target_nm):

            return unavailable_result(
                case,
                mode,
                np.nan,
                "No finite broad-envelope peak.",
            )

        if (
            "analysis_peak_is_reliable"
            in case.index
            and not bool(
                case[
                    "analysis_peak_is_reliable"
                ]
            )
        ):

            return unavailable_result(
                case,
                mode,
                target_nm,
                "Broad-envelope peak failed the reliability criterion.",
            )

    else:

        raise ValueError(
            "mode must be 'design' or 'peak'."
        )


    selected_nm, mismatch_nm = (
        nearest_profile_wavelength(
            case["case_label"],
            target_nm,
            require_usable=require_usable,
            maximum_mismatch_nm=(
                maximum_mismatch_nm
            ),
        )
    )


    if not np.isfinite(selected_nm):

        available_count = len(
            available_profile_wavelengths(
                case["case_label"],
                require_usable=(
                    require_usable
                ),
            )
        )

        reason = (
            "No acceptable profile; "
            f"usable wavelength count={available_count}."
        )

        if np.isfinite(mismatch_nm):

            reason += (
                f" Nearest mismatch={mismatch_nm:.1f} nm."
            )

        return unavailable_result(
            case,
            mode,
            target_nm,
            reason,
            mismatch_nm=mismatch_nm,
        )


    profile = (
        axis_profiles.loc[
            (
                axis_profiles[
                    "case_label"
                ]
                == case[
                    "case_label"
                ]
            )
            & np.isclose(
                axis_profiles[
                    "wavelength_nm"
                ],
                selected_nm,
            )
            & axis_profiles[
                "inside_sphere"
            ].astype(bool)
        ]
        .sort_values("z_nm")
        .copy()
    )


    if profile.empty:

        return unavailable_result(
            case,
            mode,
            target_nm,
            "Selected profile has no inside-sphere samples.",
            mismatch_nm=mismatch_nm,
        )


    profile_valid_fraction = float(
        profile[
            "profile_valid_fraction"
        ].iloc[0]
    )

    plot_valid_fraction = float(
        profile[
            "plot_profile_valid_fraction"
        ].iloc[0]
    )

    profile_is_usable = bool(
        profile[
            "profile_is_usable"
        ].iloc[0]
    )

    finite_plot_samples = int(
        np.count_nonzero(
            np.isfinite(
                profile[
                    PROFILE_FIELD_COLUMN
                ]
            )
        )
    )

    diagnostics = {
        "case_label": case["case_label"],
        "mode": mode,
        "target_wavelength_nm": target_nm,
        "selected_wavelength_nm": selected_nm,
        "wavelength_mismatch_nm": mismatch_nm,
        "profile_valid_fraction": (
            profile_valid_fraction
        ),
        "plot_profile_valid_fraction": (
            plot_valid_fraction
        ),
        "profile_is_usable": (
            profile_is_usable
        ),
        "inside_sphere_sample_count": int(
            len(profile)
        ),
        "finite_plot_sample_count": (
            finite_plot_samples
        ),
        "selection_successful": bool(
            finite_plot_samples > 0
        ),
        "reason": (
            "selected"
            if finite_plot_samples > 0
            else "No finite screened plotting samples."
        ),
    }

    return (
        selected_nm,
        profile,
        diagnostics,
    )


# ============================================================
# AUDIT DESIGN AND PEAK SELECTIONS
# ============================================================

records = []

for mode in (
    "design",
    "peak",
):

    for _, case in (
        case_summary
        .sort_values(
            [
                "absorbing",
                "n_high",
            ]
        )
        .iterrows()
    ):

        _, _, diagnostics = profile_at(
            case,
            mode,
        )

        diagnostics.update(
            {
                "n_high": float(
                    case["n_high"]
                ),
                "k_all_layers": float(
                    case["k_all_layers"]
                ),
                "absorbing": bool(
                    case["absorbing"]
                ),
            }
        )

        records.append(
            diagnostics
        )


profile_selection_table = pd.DataFrame(
    records
)


display_columns = [
    "case_label",
    "n_high",
    "k_all_layers",
    "absorbing",
    "mode",
    "target_wavelength_nm",
    "selected_wavelength_nm",
    "wavelength_mismatch_nm",
    "profile_valid_fraction",
    "plot_profile_valid_fraction",
    "profile_is_usable",
    "finite_plot_sample_count",
    "selection_successful",
    "reason",
]


display(
    profile_selection_table[
        display_columns
    ]
    .sort_values(
        [
            "mode",
            "absorbing",
            "n_high",
        ]
    )
    .reset_index(
        drop=True
    )
    .style.format(
        {
            "n_high": "{:.3f}",
            "k_all_layers": "{:.3f}",
            "target_wavelength_nm": "{:.1f}",
            "selected_wavelength_nm": "{:.1f}",
            "wavelength_mismatch_nm": "{:.1f}",
            "profile_valid_fraction": "{:.2%}",
            "plot_profile_valid_fraction": "{:.2%}",
            "finite_plot_sample_count": "{:.0f}",
        },
        na_rep="not available",
    )
)


profile_selection_table.to_csv(
    TABLE_DIR
    / "field_profile_selection_diagnostics.csv",
    index=False,
)

# In[ ]:


# ============================================================
# CELL 6: SCREENED CENTRAL-AXIS FIELD MAGNITUDE NEAR 470 nm
#
# Exploratory visualization:
#   - catastrophic values were masked in Cell 2;
#   - only short internal gaps were interpolated;
#   - long invalid regions remain as visible gaps;
#   - curves may use the nearest acceptable stored wavelength.
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.cm import ScalarMappable
from IPython.display import display


# ============================================================
# SETTINGS
# ============================================================

DESIGN_PROFILE_MODE = "design"

DESIGN_PROFILE_FIELD_COLUMN = (
    "E_magnitude_enhancement_plot"
)

DESIGN_PROFILE_FIGURE_WIDTH_CM = 17.5
DESIGN_PROFILE_FIGURE_HEIGHT_CM = 8.0

# Robust common limit across all plotted curves.
DESIGN_PROFILE_UPPER_PERCENTILE = 99.0

# Prevent the remaining values close to the catastrophic threshold
# from dominating the exploratory visualization.
DESIGN_PROFILE_MAXIMUM_DISPLAY_VALUE = 100.0

DESIGN_PROFILE_Y_MIN = 0.0
DESIGN_PROFILE_REQUIRE_USABLE = True

SAVE_DESIGN_PROFILE_FIGURE = True


# ============================================================
# VALIDATION
# ============================================================

required_columns = {
    "case_label",
    "wavelength_nm",
    "z_nm",
    "z_over_outer_radius",
    "inside_sphere",
    DESIGN_PROFILE_FIELD_COLUMN,
    "profile_valid_fraction",
    "plot_profile_valid_fraction",
    "profile_is_usable",
}

missing = (
    required_columns
    - set(axis_profiles.columns)
)

if missing:

    raise KeyError(
        "Run the updated Cells 2 and 5 first. "
        f"Missing columns: {sorted(missing)}"
    )


# ============================================================
# SELECT PROFILES
# ============================================================

selected_profiles = []
selection_records = []

ordered_cases = (
    case_summary
    .sort_values(
        [
            "absorbing",
            "n_high",
        ]
    )
    .reset_index(
        drop=True
    )
)


for _, case in ordered_cases.iterrows():

    (
        selected_wavelength_nm,
        profile_data,
        diagnostics,
    ) = profile_at(
        case,
        DESIGN_PROFILE_MODE,
        require_usable=(
            DESIGN_PROFILE_REQUIRE_USABLE
        ),
    )

    diagnostics = {
        **diagnostics,
        "n_high": float(
            case["n_high"]
        ),
        "k_all_layers": float(
            case["k_all_layers"]
        ),
        "absorbing": bool(
            case["absorbing"]
        ),
    }

    selection_records.append(
        diagnostics
    )

    if (
        profile_data.empty
        or not diagnostics[
            "selection_successful"
        ]
    ):
        continue

    if (
        DESIGN_PROFILE_REQUIRE_USABLE
        and not diagnostics[
            "profile_is_usable"
        ]
    ):
        continue

    profile_for_plot = (
        profile_data.copy()
    )

    profile_for_plot[
        DESIGN_PROFILE_FIELD_COLUMN
    ] = pd.to_numeric(
        profile_for_plot[
            DESIGN_PROFILE_FIELD_COLUMN
        ],
        errors="coerce",
    )

    # Mask remaining extreme but sub-threshold values for display.
    #
    # Do not drop these rows. NaN values intentionally break the line
    # instead of connecting unrelated valid regions.
    profile_for_plot.loc[
        (
            profile_for_plot[
                DESIGN_PROFILE_FIELD_COLUMN
            ]
            < 0.0
        )
        | (
            profile_for_plot[
                DESIGN_PROFILE_FIELD_COLUMN
            ]
            > (
                DESIGN_PROFILE_MAXIMUM_DISPLAY_VALUE
            )
        ),
        DESIGN_PROFILE_FIELD_COLUMN,
    ] = np.nan

    finite_count = int(
        np.count_nonzero(
            np.isfinite(
                profile_for_plot[
                    DESIGN_PROFILE_FIELD_COLUMN
                ]
            )
        )
    )

    if finite_count == 0:
        continue

    selected_profiles.append(
        {
            "case": case,
            "selected_wavelength_nm": (
                selected_wavelength_nm
            ),
            "data": profile_for_plot,
            "diagnostics": diagnostics,
        }
    )


design_profile_selection_table = pd.DataFrame(
    selection_records
)


# ============================================================
# DISPLAY COMPACT DIAGNOSTICS
# ============================================================

diagnostic_columns = [
    "case_label",
    "n_high",
    "k_all_layers",
    "absorbing",
    "target_wavelength_nm",
    "selected_wavelength_nm",
    "wavelength_mismatch_nm",
    "profile_valid_fraction",
    "plot_profile_valid_fraction",
    "selection_successful",
    "reason",
]

diagnostic_columns = [
    column
    for column in diagnostic_columns
    if column
    in design_profile_selection_table.columns
]


display(
    design_profile_selection_table[
        diagnostic_columns
    ]
    .sort_values(
        [
            "absorbing",
            "n_high",
        ]
    )
    .reset_index(
        drop=True
    )
    .style.format(
        {
            "n_high": "{:.3f}",
            "k_all_layers": "{:.3f}",
            "target_wavelength_nm": "{:.1f}",
            "selected_wavelength_nm": "{:.1f}",
            "wavelength_mismatch_nm": "{:.1f}",
            "profile_valid_fraction": "{:.2%}",
            "plot_profile_valid_fraction": "{:.2%}",
        },
        na_rep="not available",
    )
)


design_profile_selection_table.to_csv(
    TABLE_DIR
    / "design_wavelength_profile_plot_diagnostics.csv",
    index=False,
)


# ============================================================
# ROBUST COMMON Y LIMIT
# ============================================================

if selected_profiles:

    pooled_values = np.concatenate(
        [
            item["data"][
                DESIGN_PROFILE_FIELD_COLUMN
            ].to_numpy(
                dtype=float
            )
            for item in selected_profiles
        ]
    )

    pooled_values = pooled_values[
        np.isfinite(
            pooled_values
        )
        & (
            pooled_values >= 0.0
        )
    ]

    if pooled_values.size:

        design_profile_y_max = float(
            np.nanpercentile(
                pooled_values,
                DESIGN_PROFILE_UPPER_PERCENTILE,
            )
        )

        design_profile_y_max = min(
            design_profile_y_max,
            DESIGN_PROFILE_MAXIMUM_DISPLAY_VALUE,
        )

        design_profile_y_max *= 1.05

    else:

        design_profile_y_max = 1.0

else:

    design_profile_y_max = 1.0


# ============================================================
# PLOT
# ============================================================

cm_to_inches = 1.0 / 2.54

fig, ax = plt.subplots(
    figsize=(
        DESIGN_PROFILE_FIGURE_WIDTH_CM
        * cm_to_inches,
        DESIGN_PROFILE_FIGURE_HEIGHT_CM
        * cm_to_inches,
    ),
    constrained_layout=True,
)


for item in selected_profiles:

    case = item["case"]
    data = item["data"]

    colour = cmap(
        norm(
            float(
                case["n_high"]
            )
        )
    )

    selected_nm = float(
        item[
            "selected_wavelength_nm"
        ]
    )

    if bool(
        case["absorbing"]
    ):

        linestyle = "--"
        linewidth = 1.20

        # label = (
        #     rf"$n_h={case['n_high']:.2f}$, "
        #     rf"$k={case['k_all_layers']:.3f}$, "
        #     rf"$\lambda={selected_nm:.0f}$ nm"
        # )

    else:

        linestyle = "-"
        linewidth = 0.90
        label = None

    ax.plot(
        data[
            "z_over_outer_radius"
        ],
        data[
            DESIGN_PROFILE_FIELD_COLUMN
        ],
        color=colour,
        linestyle=linestyle,
        linewidth=linewidth,
        alpha=0.95,
        label=label,
    )


if selected_profiles:

    ax.set_xlim(
        -1.0,
        1.0,
    )

    ax.set_ylim(
        DESIGN_PROFILE_Y_MIN,
        design_profile_y_max,
    )

    ax.set_xlabel(
        r"Propagation coordinate $z/R$"
    )

    ax.set_ylabel(
        r"Screened field-magnitude enhancement "
        r"$|E|/|E_0|$"
    )

    ax.set_title(
        rf"Screened central-axis field near "
        rf"$\lambda_0="
        f"{DESIGN_WAVELENGTH_NM:.0f}"
        r"\,\mathrm{nm}$"
    )

    if any(
        item["case"]["absorbing"]
        for item in selected_profiles
    ):

        ax.legend(
            fontsize=6.2,
            frameon=False,
            loc="upper right",
        )

    scalar = ScalarMappable(
        norm=norm,
        cmap=cmap,
    )

    scalar.set_array([])

    colorbar = fig.colorbar(
        scalar,
        ax=ax,
        pad=0.02,
        fraction=0.045,
    )

    colorbar.set_label(
        r"Lossless-series $n_h$"
    )

    ax.text(
        0.01,
        0.02,
        (
            "Exploratory screening: catastrophic values masked; "
            "short gaps interpolated; long gaps omitted."
        ),
        transform=ax.transAxes,
        fontsize=5.8,
        color="0.35",
        ha="left",
        va="bottom",
    )

else:

    ax.set_axis_off()

    ax.text(
        0.5,
        0.55,
        "No screened profiles could be plotted.",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=11,
    )


if SAVE_DESIGN_PROFILE_FIGURE:

    save_figure(
        fig,
        "axis_field_magnitude_design_470nm_exploratory_screened",
    )


plt.show()
plt.close(fig)


print(
    "Number of plotted profiles:",
    len(selected_profiles),
    "/",
    len(ordered_cases),
)

print(
    "Actual plotted wavelengths:",
    {
        item["case"]["case_label"]:
        item["selected_wavelength_nm"]
        for item in selected_profiles
    },
)

# In[15]:


# ============================================================
# CELL 7: SCREENED CENTRAL-AXIS FIELD AT THE REFLECTION PEAK
#
# For each case:
#   target wavelength = broad-envelope reflection peak
#   plotted wavelength = nearest usable stored field wavelength
#
# Exploratory processing:
#   - catastrophic values masked;
#   - short gaps interpolated;
#   - long invalid regions remain missing;
#   - visually extreme residual values are omitted.
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.cm import ScalarMappable
from IPython.display import display


# ============================================================
# SETTINGS
# ============================================================

PEAK_PROFILE_FIELD_COLUMN = (
    "E_magnitude_enhancement_plot"
)

PEAK_PROFILE_FIGURE_WIDTH_CM = 17.5
PEAK_PROFILE_FIGURE_HEIGHT_CM = 8.0

PEAK_PROFILE_UPPER_PERCENTILE = 99.0
PEAK_PROFILE_MAXIMUM_DISPLAY_VALUE = 100.0

# n_h = 1.50 contains no refractive-index contrast and therefore
# has no meaningful Bragg reflection peak.
PEAK_PROFILE_MINIMUM_N_HIGH = 1.51

PEAK_PROFILE_REQUIRE_USABLE = True
SAVE_PEAK_PROFILE_FIGURE = True


# ============================================================
# VALIDATION
# ============================================================

required_columns = {
    "case_label",
    "wavelength_nm",
    "z_nm",
    "z_over_outer_radius",
    "inside_sphere",
    PEAK_PROFILE_FIELD_COLUMN,
    "profile_valid_fraction",
    "plot_profile_valid_fraction",
    "profile_is_usable",
}

missing = (
    required_columns
    - set(axis_profiles.columns)
)

if missing:

    raise KeyError(
        "Run updated Cells 2, 3, and 5 first. "
        f"Missing columns: {sorted(missing)}"
    )


# ============================================================
# SELECT PEAK-WAVELENGTH PROFILES
# ============================================================

selected_peak_profiles = []
peak_selection_records = []

peak_cases = (
    case_summary.loc[
        case_summary["n_high"]
        >= PEAK_PROFILE_MINIMUM_N_HIGH
    ]
    .sort_values(
        [
            "absorbing",
            "n_high",
        ]
    )
    .reset_index(
        drop=True
    )
)


for _, case in peak_cases.iterrows():

    (
        selected_wavelength_nm,
        profile_data,
        diagnostics,
    ) = profile_at(
        case,
        "peak",
        require_usable=(
            PEAK_PROFILE_REQUIRE_USABLE
        ),
    )

    diagnostics = {
        **diagnostics,
        "n_high": float(
            case["n_high"]
        ),
        "k_all_layers": float(
            case["k_all_layers"]
        ),
        "absorbing": bool(
            case["absorbing"]
        ),
    }

    peak_selection_records.append(
        diagnostics
    )

    if (
        profile_data.empty
        or not diagnostics[
            "selection_successful"
        ]
    ):
        continue

    if (
        PEAK_PROFILE_REQUIRE_USABLE
        and not diagnostics[
            "profile_is_usable"
        ]
    ):
        continue

    profile_for_plot = (
        profile_data.copy()
    )

    values = pd.to_numeric(
        profile_for_plot[
            PEAK_PROFILE_FIELD_COLUMN
        ],
        errors="coerce",
    )

    # Preserve rows and replace suspicious display values with NaN,
    # ensuring that Matplotlib draws gaps.
    profile_for_plot[
        PEAK_PROFILE_FIELD_COLUMN
    ] = values.where(
        np.isfinite(values)
        & (values >= 0.0)
        & (
            values
            <= PEAK_PROFILE_MAXIMUM_DISPLAY_VALUE
        )
    )

    finite_count = int(
        profile_for_plot[
            PEAK_PROFILE_FIELD_COLUMN
        ].notna().sum()
    )

    if finite_count == 0:
        continue

    selected_peak_profiles.append(
        {
            "case": case,
            "selected_wavelength_nm": (
                selected_wavelength_nm
            ),
            "data": profile_for_plot,
            "diagnostics": diagnostics,
        }
    )


peak_profile_selection_table = (
    pd.DataFrame(
        peak_selection_records
    )
)


# ============================================================
# DISPLAY COMPACT DIAGNOSTICS
# ============================================================

diagnostic_columns = [
    "case_label",
    "n_high",
    "k_all_layers",
    "absorbing",
    "target_wavelength_nm",
    "selected_wavelength_nm",
    "wavelength_mismatch_nm",
    "profile_valid_fraction",
    "plot_profile_valid_fraction",
    "selection_successful",
    "reason",
]

diagnostic_columns = [
    column
    for column in diagnostic_columns
    if column
    in peak_profile_selection_table.columns
]


display(
    peak_profile_selection_table[
        diagnostic_columns
    ]
    .sort_values(
        [
            "absorbing",
            "n_high",
        ]
    )
    .reset_index(
        drop=True
    )
    .style.format(
        {
            "n_high": "{:.3f}",
            "k_all_layers": "{:.3f}",
            "target_wavelength_nm": "{:.1f}",
            "selected_wavelength_nm": "{:.1f}",
            "wavelength_mismatch_nm": "{:.1f}",
            "profile_valid_fraction": "{:.2%}",
            "plot_profile_valid_fraction": "{:.2%}",
        },
        na_rep="not available",
    )
)


peak_profile_selection_table.to_csv(
    TABLE_DIR
    / "peak_wavelength_profile_plot_diagnostics.csv",
    index=False,
)


# ============================================================
# ROBUST COMMON Y LIMIT
# ============================================================

if selected_peak_profiles:

    pooled_values = np.concatenate(
        [
            item["data"][
                PEAK_PROFILE_FIELD_COLUMN
            ].to_numpy(
                dtype=float
            )
            for item
            in selected_peak_profiles
        ]
    )

    pooled_values = pooled_values[
        np.isfinite(pooled_values)
        & (pooled_values >= 0.0)
    ]

    if pooled_values.size:

        peak_profile_y_max = float(
            np.nanpercentile(
                pooled_values,
                PEAK_PROFILE_UPPER_PERCENTILE,
            )
        )

        peak_profile_y_max = min(
            peak_profile_y_max,
            PEAK_PROFILE_MAXIMUM_DISPLAY_VALUE,
        )

        peak_profile_y_max *= 1.05

    else:

        peak_profile_y_max = 1.0

else:

    peak_profile_y_max = 1.0


# ============================================================
# PLOT
# ============================================================

cm_to_inches = 1.0 / 2.54

fig, ax = plt.subplots(
    figsize=(
        PEAK_PROFILE_FIGURE_WIDTH_CM
        * cm_to_inches,
        PEAK_PROFILE_FIGURE_HEIGHT_CM
        * cm_to_inches,
    ),
    constrained_layout=True,
)


for item in selected_peak_profiles:

    case = item["case"]
    data = item["data"]

    selected_nm = float(
        item["selected_wavelength_nm"]
    )

    target_nm = float(
        case[
            "analysis_peak_wavelength_nm"
        ]
    )

    colour = cmap(
        norm(
            float(
                case["n_high"]
            )
        )
    )

    if bool(
        case["absorbing"]
    ):

        linestyle = "--"
        linewidth = 1.20

        label = (
            rf"$n_h={case['n_high']:.2f}$, "
            rf"$k={case['k_all_layers']:.3f}$, "
            rf"$\lambda={selected_nm:.0f}$ nm"
        )

    else:

        linestyle = "-"
        linewidth = 0.90
        label = None

    ax.plot(
        data[
            "z_over_outer_radius"
        ],
        data[
            PEAK_PROFILE_FIELD_COLUMN
        ],
        color=colour,
        linestyle=linestyle,
        linewidth=linewidth,
        alpha=0.95,
        label=label,
    )


if selected_peak_profiles:

    ax.set_xlim(
        -1.0,
        1.0,
    )

    ax.set_ylim(
        0.0,
        peak_profile_y_max,
    )

    ax.set_xlabel(
        r"Propagation coordinate $z/R$"
    )

    ax.set_ylabel(
        r"Screened field-magnitude enhancement "
        r"$|E|/|E_0|$"
    )

    ax.set_title(
        "Screened central-axis field near each "
        "broad-envelope reflection peak"
    )

    if any(
        item["case"]["absorbing"]
        for item in selected_peak_profiles
    ):

        ax.legend(
            fontsize=6.2,
            frameon=False,
            loc="upper right",
        )

    scalar = ScalarMappable(
        norm=norm,
        cmap=cmap,
    )

    scalar.set_array([])

    colorbar = fig.colorbar(
        scalar,
        ax=ax,
        pad=0.02,
        fraction=0.045,
    )

    colorbar.set_label(
        r"Lossless-series $n_h$"
    )

    ax.text(
        0.01,
        0.02,
        (
            "Exploratory screening: catastrophic values masked; "
            "short gaps interpolated; long gaps omitted."
        ),
        transform=ax.transAxes,
        fontsize=5.8,
        color="0.35",
        ha="left",
        va="bottom",
    )

else:

    ax.set_axis_off()

    ax.text(
        0.5,
        0.55,
        "No screened peak-wavelength profiles could be plotted.",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=11,
    )


if SAVE_PEAK_PROFILE_FIGURE:

    save_figure(
        fig,
        "axis_field_magnitude_peak_exploratory_screened",
    )


plt.show()
plt.close(fig)


print(
    "Number of plotted peak profiles:",
    len(selected_peak_profiles),
    "/",
    len(peak_cases),
)

print(
    "Actual plotted peak-profile wavelengths:",
    {
        item["case"]["case_label"]:
        item["selected_wavelength_nm"]
        for item in selected_peak_profiles
    },
)

# In[16]:


# ============================================================
# CELL 8: WAVELENGTH-DEPENDENT SCREENED AXIS PROFILES
#
# One selected case is shown at multiple wavelengths.
#
# Processing:
#   - uses the screened plotting field from Cell 2;
#   - preserves potentially physical central enhancement;
#   - removes only nonfinite/catastrophic values;
#   - interpolates only the short gaps already treated in Cell 2;
#   - retains long gaps as NaN;
#   - provides a linear and logarithmic panel;
#   - quantifies central-region enhancement versus wavelength.
#
# Important:
#   The central feature is not automatically discarded.
#   Values above the display limit are hidden from the line plot but
#   remain available in the raw data.
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from IPython.display import display


# ============================================================
# SETTINGS
# ============================================================

SELECTED_CASE_LABEL = (
    "nH_1.520_lossless"
)

# Approximate wavelength interval between displayed curves.
WAVELENGTH_PLOT_STEP_NM = 20.0

# Optional wavelength range.
#
# Set either value to None to use the complete calculated range.
WAVELENGTH_PLOT_MIN_NM = None
WAVELENGTH_PLOT_MAX_NM = None

# Screened plotting column created in Cell 2.
WAVELENGTH_PROFILE_FIELD_COLUMN = (
    "E_magnitude_enhancement_plot"
)

# Plot only wavelength profiles accepted by Cell 2.
WAVELENGTH_PROFILE_REQUIRE_USABLE = True

# The current catastrophic threshold in Cell 2 is 1e4.
# This lower limit affects visualization only.
#
# A value of 100 retains very strong enhancement while preventing
# residual extreme values from flattening the useful curves.
WAVELENGTH_PROFILE_DISPLAY_MAXIMUM = 100.0

# Robust common upper limit for the linear panel.
WAVELENGTH_PROFILE_UPPER_PERCENTILE = 99.0

# Lower limit for the logarithmic panel.
WAVELENGTH_PROFILE_LOG_MINIMUM = 1e-2

# Region interpreted as the particle center.
CENTRAL_REGION_HALF_WIDTH_Z_OVER_R = 0.05

# Reference regions away from the exact center.
OUTER_REGION_MINIMUM_ABS_Z_OVER_R = 0.70

WAVELENGTH_PROFILE_FIGURE_WIDTH_CM = 18.0
WAVELENGTH_PROFILE_FIGURE_HEIGHT_CM = 12.0

SAVE_WAVELENGTH_PROFILE_FIGURE = True


# ============================================================
# VALIDATION
# ============================================================

required_columns = {
    "case_label",
    "wavelength_nm",
    "z_nm",
    "z_over_outer_radius",
    "inside_sphere",
    WAVELENGTH_PROFILE_FIELD_COLUMN,
    "profile_valid_fraction",
    "plot_profile_valid_fraction",
    "profile_is_usable",
}

missing = (
    required_columns
    - set(axis_profiles.columns)
)

if missing:

    raise KeyError(
        "Run the updated Cell 2 before Cell 8. "
        f"Missing columns: {sorted(missing)}"
    )


if (
    SELECTED_CASE_LABEL
    not in axis_profiles[
        "case_label"
    ].unique()
):

    raise ValueError(
        f"Unknown case: {SELECTED_CASE_LABEL!r}"
    )


# ============================================================
# SELECT CASE AND ACCEPTABLE WAVELENGTHS
# ============================================================

selected_case_profiles = (
    axis_profiles.loc[
        (
            axis_profiles[
                "case_label"
            ]
            == SELECTED_CASE_LABEL
        )
        & axis_profiles[
            "inside_sphere"
        ].astype(bool)
    ]
    .copy()
)


wavelength_quality = (
    selected_case_profiles[
        [
            "wavelength_nm",
            "profile_valid_fraction",
            "plot_profile_valid_fraction",
            "profile_is_usable",
        ]
    ]
    .drop_duplicates(
        subset="wavelength_nm"
    )
    .sort_values(
        "wavelength_nm"
    )
)


if WAVELENGTH_PROFILE_REQUIRE_USABLE:

    wavelength_quality = (
        wavelength_quality.loc[
            wavelength_quality[
                "profile_is_usable"
            ].astype(bool)
        ]
    )


if (
    WAVELENGTH_PLOT_MIN_NM
    is not None
):

    wavelength_quality = (
        wavelength_quality.loc[
            wavelength_quality[
                "wavelength_nm"
            ]
            >= float(
                WAVELENGTH_PLOT_MIN_NM
            )
        ]
    )


if (
    WAVELENGTH_PLOT_MAX_NM
    is not None
):

    wavelength_quality = (
        wavelength_quality.loc[
            wavelength_quality[
                "wavelength_nm"
            ]
            <= float(
                WAVELENGTH_PLOT_MAX_NM
            )
        ]
    )


available_wavelengths = (
    wavelength_quality[
        "wavelength_nm"
    ]
    .dropna()
    .to_numpy(
        dtype=float
    )
)


if available_wavelengths.size == 0:

    raise ValueError(
        "No acceptable wavelength profiles remain for "
        f"{SELECTED_CASE_LABEL}."
    )


# Select wavelengths approximately separated by the requested step.
requested_wavelengths = np.arange(
    available_wavelengths.min(),
    (
        available_wavelengths.max()
        + 0.5
        * WAVELENGTH_PLOT_STEP_NM
    ),
    WAVELENGTH_PLOT_STEP_NM,
)


selected_wavelengths = np.unique(
    [
        available_wavelengths[
            np.argmin(
                np.abs(
                    available_wavelengths
                    - requested
                )
            )
        ]
        for requested
        in requested_wavelengths
    ]
)


# Explicitly include the design wavelength and the cleaned
# broad-envelope peak when acceptable profiles exist nearby.
case_row = case_summary.loc[
    case_summary[
        "case_label"
    ]
    == SELECTED_CASE_LABEL
].iloc[0]


special_wavelengths = [
    float(
        DESIGN_WAVELENGTH_NM
    ),
]


if np.isfinite(
    case_row.get(
        "analysis_peak_wavelength_nm",
        np.nan,
    )
):

    special_wavelengths.append(
        float(
            case_row[
                "analysis_peak_wavelength_nm"
            ]
        )
    )


for requested in special_wavelengths:

    nearest = available_wavelengths[
        np.argmin(
            np.abs(
                available_wavelengths
                - requested
            )
        )
    ]

    selected_wavelengths = np.append(
        selected_wavelengths,
        nearest,
    )


selected_wavelengths = np.unique(
    selected_wavelengths
)


# ============================================================
# PREPARE PROFILES AND CENTRAL-REGION METRICS
# ============================================================

prepared_profiles = []
wavelength_metric_records = []


for wavelength_nm in selected_wavelengths:

    data = (
        selected_case_profiles.loc[
            np.isclose(
                selected_case_profiles[
                    "wavelength_nm"
                ],
                wavelength_nm,
            )
        ]
        .sort_values(
            "z_over_outer_radius"
        )
        .copy()
    )


    values = pd.to_numeric(
        data[
            WAVELENGTH_PROFILE_FIELD_COLUMN
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )


    z_over_r = data[
        "z_over_outer_radius"
    ].to_numpy(
        dtype=float
    )


    # Preserve the screened values for metrics.
    metric_values = values.copy()

    metric_values[
        ~np.isfinite(
            metric_values
        )
        | (
            metric_values < 0.0
        )
        | (
            metric_values
            > FIELD_CATASTROPHIC_LIMIT
        )
    ] = np.nan


    # Apply a separate display-only ceiling.
    plot_values = metric_values.copy()

    plot_values[
        plot_values
        > WAVELENGTH_PROFILE_DISPLAY_MAXIMUM
    ] = np.nan


    data[
        "field_for_plot"
    ] = plot_values


    finite_plot_count = int(
        np.count_nonzero(
            np.isfinite(
                plot_values
            )
        )
    )


    if finite_plot_count == 0:
        continue


    prepared_profiles.append(
        {
            "wavelength_nm": float(
                wavelength_nm
            ),
            "data": data,
        }
    )


    central_mask = (
        np.abs(
            z_over_r
        )
        <= CENTRAL_REGION_HALF_WIDTH_Z_OVER_R
    )

    outer_mask = (
        np.abs(
            z_over_r
        )
        >= OUTER_REGION_MINIMUM_ABS_Z_OVER_R
    )


    central_values = metric_values[
        central_mask
        & np.isfinite(
            metric_values
        )
    ]

    outer_values = metric_values[
        outer_mask
        & np.isfinite(
            metric_values
        )
    ]

    whole_values = metric_values[
        np.isfinite(
            metric_values
        )
    ]


    central_median = (
        float(
            np.nanmedian(
                central_values
            )
        )
        if central_values.size
        else np.nan
    )

    central_p95 = (
        float(
            np.nanpercentile(
                central_values,
                95.0,
            )
        )
        if central_values.size
        else np.nan
    )

    central_maximum = (
        float(
            np.nanmax(
                central_values
            )
        )
        if central_values.size
        else np.nan
    )

    outer_median = (
        float(
            np.nanmedian(
                outer_values
            )
        )
        if outer_values.size
        else np.nan
    )

    center_to_outer_ratio = (
        central_median
        / outer_median
        if (
            np.isfinite(
                central_median
            )
            and np.isfinite(
                outer_median
            )
            and outer_median > 0.0
        )
        else np.nan
    )


    quality_row = (
        wavelength_quality.loc[
            np.isclose(
                wavelength_quality[
                    "wavelength_nm"
                ],
                wavelength_nm,
            )
        ]
    )


    wavelength_metric_records.append(
        {
            "case_label": (
                SELECTED_CASE_LABEL
            ),
            "wavelength_nm": float(
                wavelength_nm
            ),
            "profile_valid_fraction": (
                float(
                    quality_row[
                        "profile_valid_fraction"
                    ].iloc[0]
                )
                if not quality_row.empty
                else np.nan
            ),
            "plot_profile_valid_fraction": (
                float(
                    quality_row[
                        "plot_profile_valid_fraction"
                    ].iloc[0]
                )
                if not quality_row.empty
                else np.nan
            ),
            "finite_metric_sample_count": int(
                whole_values.size
            ),
            "central_sample_count": int(
                central_values.size
            ),
            "central_median_enhancement": (
                central_median
            ),
            "central_p95_enhancement": (
                central_p95
            ),
            "central_maximum_below_catastrophic_limit": (
                central_maximum
            ),
            "outer_median_enhancement": (
                outer_median
            ),
            "central_to_outer_median_ratio": (
                center_to_outer_ratio
            ),
        }
    )


wavelength_profile_metrics = pd.DataFrame(
    wavelength_metric_records
)


if not prepared_profiles:

    raise ValueError(
        "No finite screened profiles are available "
        "for plotting."
    )


# ============================================================
# ROBUST DISPLAY LIMIT
# ============================================================

pooled_plot_values = np.concatenate(
    [
        item["data"][
            "field_for_plot"
        ].to_numpy(
            dtype=float
        )
        for item
        in prepared_profiles
    ]
)


pooled_plot_values = pooled_plot_values[
    np.isfinite(
        pooled_plot_values
    )
    & (
        pooled_plot_values >= 0.0
    )
]
# ============================================================
# COMPLETE ROBUST DISPLAY LIMIT
# ============================================================

if pooled_plot_values.size > 0:

    linear_y_max = float(
        np.nanpercentile(
            pooled_plot_values,
            WAVELENGTH_PROFILE_UPPER_PERCENTILE,
        )
    )

    linear_y_max = min(
        linear_y_max,
        WAVELENGTH_PROFILE_DISPLAY_MAXIMUM,
    )

    linear_y_max = max(
        linear_y_max * 1.05,
        1.0,
    )

else:

    linear_y_max = 1.0


# ============================================================
# WAVELENGTH COLOUR NORMALIZATION
# ============================================================

displayed_wavelengths = np.asarray(
    [
        item["wavelength_nm"]
        for item in prepared_profiles
    ],
    dtype=float,
)


if displayed_wavelengths.size == 1:

    # Avoid a zero-width colour normalization.
    wave_norm = Normalize(
        vmin=(
            displayed_wavelengths[0]
            - 0.5
        ),
        vmax=(
            displayed_wavelengths[0]
            + 0.5
        ),
    )

else:

    wave_norm = Normalize(
        vmin=float(
            displayed_wavelengths.min()
        ),
        vmax=float(
            displayed_wavelengths.max()
        ),
    )


wave_cmap = plt.get_cmap(
    "plasma"
)


# ============================================================
# CREATE LINEAR AND LOGARITHMIC PANELS
# ============================================================

cm_to_inches = 1.0 / 2.54


fig, axes = plt.subplots(
    nrows=2,
    ncols=1,
    figsize=(
        WAVELENGTH_PROFILE_FIGURE_WIDTH_CM
        * cm_to_inches,
        WAVELENGTH_PROFILE_FIGURE_HEIGHT_CM
        * cm_to_inches,
    ),
    sharex=True,
    constrained_layout=True,
)


for item in prepared_profiles:

    wavelength_nm = float(
        item["wavelength_nm"]
    )

    data = item["data"]

    colour = wave_cmap(
        wave_norm(
            wavelength_nm
        )
    )


    # Linear panel.
    axes[0].plot(
        data[
            "z_over_outer_radius"
        ],
        data[
            "field_for_plot"
        ],
        color=colour,
        linewidth=0.9,
        alpha=0.92,
    )


    # For the logarithmic panel, zero and negative values must be
    # omitted. NaN values retain gaps in the plotted curve.
    log_values = pd.to_numeric(
        data[
            "field_for_plot"
        ],
        errors="coerce",
    ).where(
        lambda values:
        np.isfinite(values)
        & (
            values
            >= WAVELENGTH_PROFILE_LOG_MINIMUM
        )
    )


    axes[1].plot(
        data[
            "z_over_outer_radius"
        ],
        log_values,
        color=colour,
        linewidth=0.9,
        alpha=0.92,
    )


# ============================================================
# FORMAT LINEAR PANEL
# ============================================================

axes[0].set_ylim(
    0.0,
    linear_y_max,
)

axes[0].set_ylabel(
    r"Screened field magnitude "
    r"$|E|/|E_0|$"
)

axes[0].set_title(
    (
        "Wavelength-dependent screened axis profiles: "
        f"{SELECTED_CASE_LABEL}"
    )
)


# ============================================================
# FORMAT LOGARITHMIC PANEL
# ============================================================

axes[1].set_yscale(
    "log"
)

axes[1].set_ylim(
    WAVELENGTH_PROFILE_LOG_MINIMUM,
    WAVELENGTH_PROFILE_DISPLAY_MAXIMUM,
)

axes[1].set_ylabel(
    r"Screened field magnitude "
    r"$|E|/|E_0|$"
)

axes[1].set_xlabel(
    r"Propagation coordinate $z/R$"
)


# ============================================================
# SHARED AXIS ELEMENTS
# ============================================================

for ax in axes:

    ax.set_xlim(
        -1.0,
        1.0,
    )

    # Highlight the central region used for calculating the
    # wavelength-dependent enhancement metrics.
    ax.axvspan(
        -CENTRAL_REGION_HALF_WIDTH_Z_OVER_R,
        CENTRAL_REGION_HALF_WIDTH_Z_OVER_R,
        color="0.5",
        alpha=0.08,
        linewidth=0.0,
        zorder=0,
    )

    # Mark the exact sphere center.
    ax.axvline(
        0.0,
        color="0.25",
        linestyle=":",
        linewidth=0.6,
        alpha=0.7,
        zorder=0,
    )


# ============================================================
# WAVELENGTH COLOUR BAR
# ============================================================

scalar = ScalarMappable(
    norm=wave_norm,
    cmap=wave_cmap,
)

scalar.set_array([])


colour_bar = fig.colorbar(
    scalar,
    ax=axes,
    pad=0.02,
    fraction=0.035,
)

colour_bar.set_label(
    "Wavelength [nm]"
)


# ============================================================
# EXPLORATORY-SCREENING NOTE
# ============================================================

fig.text(
    0.01,
    0.005,
    (
        "Exploratory screening: nonfinite and catastrophic values "
        "masked; short gaps interpolated; central enhancement retained."
    ),
    fontsize=6.0,
    color="0.35",
)


# ============================================================
# SAVE AND DISPLAY FIGURE
# ============================================================

if SAVE_WAVELENGTH_PROFILE_FIGURE:

    save_figure(
        fig,
        (
            "axis_field_magnitude_wavelength_series_"
            f"{SELECTED_CASE_LABEL}_screened"
        ),
    )


plt.show()

plt.close(fig)


# ============================================================
# DISPLAY AND SAVE CENTRAL-REGION METRICS
# ============================================================

if not wavelength_profile_metrics.empty:

    wavelength_profile_metrics = (
        wavelength_profile_metrics
        .sort_values(
            "wavelength_nm"
        )
        .reset_index(
            drop=True
        )
    )


    display(
        wavelength_profile_metrics.style.format(
            {
                "wavelength_nm": "{:.1f}",
                "profile_valid_fraction": "{:.2%}",
                "plot_profile_valid_fraction": "{:.2%}",
                "finite_metric_sample_count": "{:.0f}",
                "central_sample_count": "{:.0f}",
                "central_median_enhancement": "{:.4g}",
                "central_p95_enhancement": "{:.4g}",
                "central_maximum_below_catastrophic_limit": (
                    "{:.4g}"
                ),
                "outer_median_enhancement": "{:.4g}",
                "central_to_outer_median_ratio": "{:.4g}",
            },
            na_rep="not available",
        )
    )


    wavelength_profile_metrics.to_csv(
        TABLE_DIR
        / (
            "wavelength_dependent_central_enhancement_"
            f"{SELECTED_CASE_LABEL}.csv"
        ),
        index=False,
    )


    # --------------------------------------------------------
    # Plot wavelength dependence of central enhancement metrics
    # --------------------------------------------------------

    fig_metrics, ax_metrics = plt.subplots(
        figsize=(
            17.5 / 2.54,
            7.0 / 2.54,
        ),
        constrained_layout=True,
    )


    ax_metrics.plot(
        wavelength_profile_metrics[
            "wavelength_nm"
        ],
        wavelength_profile_metrics[
            "central_median_enhancement"
        ],
        marker="o",
        markersize=3.5,
        linewidth=1.0,
        label="Central median",
    )


    ax_metrics.plot(
        wavelength_profile_metrics[
            "wavelength_nm"
        ],
        wavelength_profile_metrics[
            "central_p95_enhancement"
        ],
        marker="s",
        markersize=3.2,
        linewidth=1.0,
        label="Central 95th percentile",
    )

# In[17]:


# ============================================================
# CELL 9: SHELL-RESOLVED FIELD MAGNITUDE NEAR 470 nm
#
# Exploratory shell analysis:
#   - uses median shell magnitude when available;
#   - masks nonfinite and catastrophically large shell values;
#   - excludes n_h = 1.50 from the principal comparison;
#   - uses the same selected wavelengths as the screened profiles;
#   - reports actual wavelengths and mismatches;
#   - preserves missing shells as gaps.
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.cm import ScalarMappable
from matplotlib.ticker import MaxNLocator
from IPython.display import display


# ============================================================
# SETTINGS
# ============================================================

SHELL_PROFILE_MINIMUM_N_HIGH = 1.51

SHELL_PROFILE_REQUIRE_USABLE = True

# Prefer the median because the shell mean can be dominated by a
# small number of numerically extreme pixels.
if (
    "median_E_magnitude_plane_clean"
    in shell_plane_averages.columns
):

    SHELL_PROFILE_FIELD_COLUMN = (
        "median_E_magnitude_plane_clean"
    )

    SHELL_PROFILE_STATISTIC_LABEL = (
        "XZ-plane shell median"
    )

else:

    SHELL_PROFILE_FIELD_COLUMN = (
        "mean_E_magnitude_plane_clean"
    )

    SHELL_PROFILE_STATISTIC_LABEL = (
        "XZ-plane shell mean"
    )


# Values above this limit are omitted from the displayed line.
# This does not modify the stored data.
SHELL_PROFILE_DISPLAY_MAXIMUM = 100.0

SHELL_PROFILE_UPPER_PERCENTILE = 99.0

SHELL_PROFILE_FIGURE_WIDTH_CM = 17.5
SHELL_PROFILE_FIGURE_HEIGHT_CM = 8.0

SAVE_SHELL_PROFILE_FIGURE = True


# ============================================================
# VALIDATION
# ============================================================

required_shell_columns = {
    "case_label",
    "wavelength_nm",
    "shell_number_one_based",
    SHELL_PROFILE_FIELD_COLUMN,
}

missing_shell_columns = (
    required_shell_columns
    - set(shell_plane_averages.columns)
)

if missing_shell_columns:

    raise KeyError(
        "Run the updated Cell 2 first. "
        "Missing shell columns: "
        f"{sorted(missing_shell_columns)}"
    )


# ============================================================
# SELECT SHELL PROFILES
# ============================================================

selected_shell_profiles = []
shell_selection_records = []


shell_cases = (
    lossless_cases.loc[
        lossless_cases["n_high"]
        >= SHELL_PROFILE_MINIMUM_N_HIGH
    ]
    .sort_values("n_high")
    .reset_index(drop=True)
)


for _, case in shell_cases.iterrows():

    # Function returns selected wavelength and mismatch.
    (
        selected_wavelength_nm,
        mismatch_nm,
    ) = nearest_profile_wavelength(
        case["case_label"],
        DESIGN_WAVELENGTH_NM,
        require_usable=(
            SHELL_PROFILE_REQUIRE_USABLE
        ),
    )

    if not np.isfinite(
        selected_wavelength_nm
    ):

        shell_selection_records.append(
            {
                "case_label": case["case_label"],
                "n_high": float(case["n_high"]),
                "target_wavelength_nm": float(
                    DESIGN_WAVELENGTH_NM
                ),
                "selected_wavelength_nm": np.nan,
                "wavelength_mismatch_nm": mismatch_nm,
                "finite_shell_count": 0,
                "selection_successful": False,
                "reason": (
                    "No acceptable wavelength profile."
                ),
            }
        )

        continue


    data = (
        shell_plane_averages.loc[
            (
                shell_plane_averages[
                    "case_label"
                ]
                == case["case_label"]
            )
            & np.isclose(
                shell_plane_averages[
                    "wavelength_nm"
                ],
                selected_wavelength_nm,
            )
        ]
        .sort_values(
            "shell_number_one_based"
        )
        .copy()
    )


    if data.empty:

        shell_selection_records.append(
            {
                "case_label": case["case_label"],
                "n_high": float(case["n_high"]),
                "target_wavelength_nm": float(
                    DESIGN_WAVELENGTH_NM
                ),
                "selected_wavelength_nm": (
                    selected_wavelength_nm
                ),
                "wavelength_mismatch_nm": mismatch_nm,
                "finite_shell_count": 0,
                "selection_successful": False,
                "reason": (
                    "No shell table at selected wavelength."
                ),
            }
        )

        continue


    values = pd.to_numeric(
        data[
            SHELL_PROFILE_FIELD_COLUMN
        ],
        errors="coerce",
    )


    # Preserve shell positions but mask suspicious display values.
    data[
        "shell_field_for_plot"
    ] = values.where(
        np.isfinite(values)
        & (values >= 0.0)
        & (
            values
            <= SHELL_PROFILE_DISPLAY_MAXIMUM
        )
    )


    finite_shell_count = int(
        data[
            "shell_field_for_plot"
        ].notna().sum()
    )


    shell_selection_records.append(
        {
            "case_label": case["case_label"],
            "n_high": float(case["n_high"]),
            "target_wavelength_nm": float(
                DESIGN_WAVELENGTH_NM
            ),
            "selected_wavelength_nm": (
                selected_wavelength_nm
            ),
            "wavelength_mismatch_nm": mismatch_nm,
            "finite_shell_count": finite_shell_count,
            "total_shell_count": int(len(data)),
            "finite_shell_fraction": (
                finite_shell_count
                / len(data)
                if len(data)
                else np.nan
            ),
            "selection_successful": (
                finite_shell_count > 0
            ),
            "reason": (
                "selected"
                if finite_shell_count > 0
                else "No finite screened shell values."
            ),
        }
    )


    if finite_shell_count == 0:
        continue


    selected_shell_profiles.append(
        {
            "case": case,
            "selected_wavelength_nm": (
                selected_wavelength_nm
            ),
            "data": data,
        }
    )


shell_selection_table = pd.DataFrame(
    shell_selection_records
)


# ============================================================
# DISPLAY DIAGNOSTICS
# ============================================================

display(
    shell_selection_table.style.format(
        {
            "n_high": "{:.3f}",
            "target_wavelength_nm": "{:.1f}",
            "selected_wavelength_nm": "{:.1f}",
            "wavelength_mismatch_nm": "{:.1f}",
            "finite_shell_count": "{:.0f}",
            "total_shell_count": "{:.0f}",
            "finite_shell_fraction": "{:.2%}",
        },
        na_rep="not available",
    )
)


shell_selection_table.to_csv(
    TABLE_DIR
    / "shell_profile_design_wavelength_diagnostics.csv",
    index=False,
)


# ============================================================
# ROBUST COMMON Y LIMIT
# ============================================================

if selected_shell_profiles:

    pooled_shell_values = np.concatenate(
        [
            item["data"][
                "shell_field_for_plot"
            ].to_numpy(float)
            for item
            in selected_shell_profiles
        ]
    )

    pooled_shell_values = (
        pooled_shell_values[
            np.isfinite(
                pooled_shell_values
            )
            & (
                pooled_shell_values >= 0.0
            )
        ]
    )

    if pooled_shell_values.size:

        shell_y_max = float(
            np.nanpercentile(
                pooled_shell_values,
                SHELL_PROFILE_UPPER_PERCENTILE,
            )
        )

        shell_y_max = min(
            shell_y_max,
            SHELL_PROFILE_DISPLAY_MAXIMUM,
        )

        shell_y_max = max(
            1.05 * shell_y_max,
            1.0,
        )

    else:

        shell_y_max = 1.0

else:

    shell_y_max = 1.0


# ============================================================
# PLOT
# ============================================================

cm_to_inches = 1.0 / 2.54


fig, ax = plt.subplots(
    figsize=(
        SHELL_PROFILE_FIGURE_WIDTH_CM
        * cm_to_inches,
        SHELL_PROFILE_FIGURE_HEIGHT_CM
        * cm_to_inches,
    ),
    constrained_layout=True,
)


for item in selected_shell_profiles:

    case = item["case"]
    data = item["data"]
    selected_wavelength_nm = item["selected_wavelength_nm"]
    colour = cmap(
    norm(
        float(
            case["n_high"]
        )
    )
)

ax.plot(
    data[
        "shell_number_one_based"
    ],
    data[
        "shell_field_for_plot"
    ],
    color=colour,
    linewidth=0.95,
    alpha=0.95,
)


# ============================================================
# FORMAT PLOT
# ============================================================

if selected_shell_profiles:

    maximum_shell_number = max(
        int(
            item["data"][
                "shell_number_one_based"
            ].max()
        )
        for item in selected_shell_profiles
    )

    ax.set_xlim(
        1,
        maximum_shell_number,
    )

    ax.set_ylim(
        0.0,
        shell_y_max,
    )

    ax.set_xlabel(
        "Radial layer number "
        "(1 = core, outermost number = outer layer)"
    )

    ax.set_ylabel(
        rf"{SHELL_PROFILE_STATISTIC_LABEL} "
        r"$|E|/|E_0|$"
    )

    ax.set_title(
        rf"Screened shell-resolved field near "
        rf"$\lambda_0="
        f"{DESIGN_WAVELENGTH_NM:.0f}"
        r"\,\mathrm{nm}$"
    )

    ax.xaxis.set_major_locator(
        MaxNLocator(
            integer=True,
            nbins=12,
        )
    )

    # Mark the core explicitly.
    ax.axvspan(
        0.5,
        1.5,
        color="0.5",
        alpha=0.10,
        linewidth=0.0,
        zorder=0,
    )

    ax.text(
        1.0,
        0.98,
        "core",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=6.5,
        color="0.35",
    )


    scalar = ScalarMappable(
        norm=norm,
        cmap=cmap,
    )

    scalar.set_array([])

    colorbar = fig.colorbar(
        scalar,
        ax=ax,
        pad=0.02,
        fraction=0.045,
    )

    colorbar.set_label(
        r"Lossless-series $n_h$"
    )


    ax.text(
        0.01,
        0.02,
        (
            "Exploratory screening; shell median used where "
            "available. Missing or extreme values are omitted."
        ),
        transform=ax.transAxes,
        fontsize=5.8,
        color="0.35",
        ha="left",
        va="bottom",
    )

else:

    ax.set_axis_off()

    ax.text(
        0.5,
        0.55,
        "No screened shell profiles could be plotted.",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=11,
    )


# =======================================================


# In[18]:


# ============================================================
# CELL 10: GROUPED SHELL BAR PLOTS
#
# Compares lossless and absorbing systems:
#   k = 0
#   k = 0.045
#
# Two figures are generated:
#   1. Near the 470 nm design wavelength
#   2. Near each case's cleaned broad-envelope reflection peak
#
# Exploratory processing:
#   - uses shell median where available;
#   - masks nonfinite and extreme values;
#   - uses nearest usable stored wavelength;
#   - reports the actual wavelengths used.
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.ticker import MaxNLocator
from IPython.display import display


# ============================================================
# SETTINGS
# ============================================================

BAR_N_HIGH = [
    1.52,
    1.57,
    2.00,
]

BAR_MODES = (
    "design",
    "peak",
)

BAR_REQUIRE_USABLE_PROFILE = True

# Prefer robust shell median.
if (
    "median_E_magnitude_plane_clean"
    in shell_plane_averages.columns
):

    BAR_SHELL_FIELD_COLUMN = (
        "median_E_magnitude_plane_clean"
    )

    BAR_STATISTIC_LABEL = (
        "Shell median"
    )

else:

    BAR_SHELL_FIELD_COLUMN = (
        "mean_E_magnitude_plane_clean"
    )

    BAR_STATISTIC_LABEL = (
        "Shell mean"
    )


# Display-only limit. Stored values are not modified.
BAR_DISPLAY_MAXIMUM = 100.0

# Width of each individual bar.
BAR_WIDTH = 0.42

BAR_FIGURE_WIDTH_CM = 17.5
BAR_FIGURE_HEIGHT_CM = 13.5

SAVE_BAR_FIGURES = True


# ============================================================
# VALIDATION
# ============================================================

required_columns = {
    "case_label",
    "n_high",
    "k_all_layers",
    "absorbing",
    "wavelength_nm",
    "shell_number_one_based",
    BAR_SHELL_FIELD_COLUMN,
}

missing = (
    required_columns
    - (
        set(case_summary.columns)
        | set(shell_plane_averages.columns)
    )
)

if missing:

    raise KeyError(
        "Run updated Cells 2, 3, and 5 first. "
        f"Missing columns: {sorted(missing)}"
    )


if (
    "analysis_peak_wavelength_nm"
    not in case_summary.columns
):

    raise KeyError(
        "Run Cell 3 first. "
        "'analysis_peak_wavelength_nm' is unavailable."
    )


# ============================================================
# BUILD FIGURES
# ============================================================

bar_selection_records = []

cm_to_inches = 1.0 / 2.54


for mode in BAR_MODES:

    fig, axes = plt.subplots(
        nrows=len(BAR_N_HIGH),
        ncols=1,
        figsize=(
            BAR_FIGURE_WIDTH_CM
            * cm_to_inches,
            BAR_FIGURE_HEIGHT_CM
            * cm_to_inches,
        ),
        sharex=True,
        constrained_layout=True,
    )

    axes = np.atleast_1d(
        axes
    )

    mode_values_for_limit = []


    for ax, n_high in zip(
        axes,
        BAR_N_HIGH,
    ):

        panel_has_data = False

        panel_wavelength_labels = []


        for (
            absorbing,
            colour,
            alpha,
            legend_label,
            offset_sign,
        ) in (
            (
                False,
                "#31688E",
                0.88,
                r"$k=0$",
                -1.0,
            ),
            (
                True,
                "#D55E00",
                0.68,
                r"$k=0.045$",
                +1.0,
            ),
        ):

            matching_cases = (
                case_summary.loc[
                    np.isclose(
                        case_summary[
                            "n_high"
                        ],
                        n_high,
                    )
                    & (
                        case_summary[
                            "absorbing"
                        ].astype(bool)
                        == absorbing
                    )
                ]
            )


            if matching_cases.empty:

                bar_selection_records.append(
                    {
                        "mode": mode,
                        "n_high": n_high,
                        "absorbing": absorbing,
                        "k_all_layers": (
                            0.045
                            if absorbing
                            else 0.0
                        ),
                        "case_label": np.nan,
                        "target_wavelength_nm": np.nan,
                        "selected_wavelength_nm": np.nan,
                        "wavelength_mismatch_nm": np.nan,
                        "finite_shell_count": 0,
                        "selection_successful": False,
                        "reason": "Case not calculated.",
                    }
                )

                continue


            case = matching_cases.iloc[0]


            if mode == "design":

                target_nm = float(
                    DESIGN_WAVELENGTH_NM
                )

            elif mode == "peak":

                target_nm = float(
                    case[
                        "analysis_peak_wavelength_nm"
                    ]
                )

                peak_reliable = bool(
                    case.get(
                        "analysis_peak_is_reliable",
                        True,
                    )
                )

                if (
                    not np.isfinite(
                        target_nm
                    )
                    or not peak_reliable
                ):

                    bar_selection_records.append(
                        {
                            "mode": mode,
                            "n_high": n_high,
                            "absorbing": absorbing,
                            "k_all_layers": float(
                                case[
                                    "k_all_layers"
                                ]
                            ),
                            "case_label": case[
                                "case_label"
                            ],
                            "target_wavelength_nm": (
                                target_nm
                            ),
                            "selected_wavelength_nm": (
                                np.nan
                            ),
                            "wavelength_mismatch_nm": (
                                np.nan
                            ),
                            "finite_shell_count": 0,
                            "selection_successful": False,
                            "reason": (
                                "No reliable cleaned "
                                "broad-envelope peak."
                            ),
                        }
                    )

                    continue

            else:

                raise ValueError(
                    f"Unknown mode: {mode!r}"
                )


            # The function returns two values.
            (
                selected_nm,
                mismatch_nm,
            ) = nearest_profile_wavelength(
                case["case_label"],
                target_nm,
                require_usable=(
                    BAR_REQUIRE_USABLE_PROFILE
                ),
            )


            if not np.isfinite(
                selected_nm
            ):

                bar_selection_records.append(
                    {
                        "mode": mode,
                        "n_high": n_high,
                        "absorbing": absorbing,
                        "k_all_layers": float(
                            case[
                                "k_all_layers"
                            ]
                        ),
                        "case_label": case[
                            "case_label"
                        ],
                        "target_wavelength_nm": (
                            target_nm
                        ),
                        "selected_wavelength_nm": (
                            np.nan
                        ),
                        "wavelength_mismatch_nm": (
                            mismatch_nm
                        ),
                        "finite_shell_count": 0,
                        "selection_successful": False,
                        "reason": (
                            "No acceptable stored "
                            "field wavelength."
                        ),
                    }
                )

                continue


            data = (
                shell_plane_averages.loc[
                    (
                        shell_plane_averages[
                            "case_label"
                        ]
                        == case[
                            "case_label"
                        ]
                    )
                    & np.isclose(
                        shell_plane_averages[
                            "wavelength_nm"
                        ],
                        selected_nm,
                    )
                ]
                .sort_values(
                    "shell_number_one_based"
                )
                .copy()
            )


            if data.empty:

                bar_selection_records.append(
                    {
                        "mode": mode,
                        "n_high": n_high,
                        "absorbing": absorbing,
                        "k_all_layers": float(
                            case[
                                "k_all_layers"
                            ]
                        ),
                        "case_label": case[
                            "case_label"
                        ],
                        "target_wavelength_nm": (
                            target_nm
                        ),
                        "selected_wavelength_nm": (
                            selected_nm
                        ),
                        "wavelength_mismatch_nm": (
                            mismatch_nm
                        ),
                        "finite_shell_count": 0,
                        "selection_successful": False,
                        "reason": (
                            "No shell data at selected "
                            "wavelength."
                        ),
                    }
                )

                continue


            shell_values = pd.to_numeric(
                data[
                    BAR_SHELL_FIELD_COLUMN
                ],
                errors="coerce",
            )


            # Mask values only for display. Do not clip.
            data[
                "bar_field_value"
            ] = shell_values.where(
                np.isfinite(
                    shell_values
                )
                & (
                    shell_values >= 0.0
                )
                & (
                    shell_values
                    <= BAR_DISPLAY_MAXIMUM
                )
            )


            finite_shell_count = int(
                data[
                    "bar_field_value"
                ].notna().sum()
            )


            bar_selection_records.append(
                {
                    "mode": mode,
                    "n_high": n_high,
                    "absorbing": absorbing,
                    "k_all_layers": float(
                        case[
                            "k_all_layers"
                        ]
                    ),
                    "case_label": case[
                        "case_label"
                    ],
                    "target_wavelength_nm": (
                        target_nm
                    ),
                    "selected_wavelength_nm": (
                        selected_nm
                    ),
                    "wavelength_mismatch_nm": (
                        mismatch_nm
                    ),
                    "finite_shell_count": (
                        finite_shell_count
                    ),
                    "total_shell_count": int(
                        len(data)
                    ),
                    "finite_shell_fraction": (
                        finite_shell_count
                        / len(data)
                        if len(data)
                        else np.nan
                    ),
                    "selection_successful": (
                        finite_shell_count > 0
                    ),
                    "reason": (
                        "selected"
                        if finite_shell_count > 0
                        else (
                            "No finite screened "
                            "shell values."
                        )
                    ),
                }
            )


            if finite_shell_count == 0:
                continue


            panel_has_data = True

            shell_numbers = data[
                "shell_number_one_based"
            ].to_numpy(
                dtype=float
            )

            offset = (
                offset_sign
                * BAR_WIDTH
                / 2.0
            )


            ax.bar(
                shell_numbers
                + offset,
                data[
                    "bar_field_value"
                ],
                width=BAR_WIDTH,
                color=colour,
                alpha=alpha,
                linewidth=0.0,
                label=legend_label,
            )


            finite_values = data[
                "bar_field_value"
            ].dropna().to_numpy(
                dtype=float
            )

            if finite_values.size:

                mode_values_for_limit.append(
                    finite_values
                )


            panel_wavelength_labels.append(
                (
                    rf"$k="
                    f"{case['k_all_layers']:.3f}"
                    rf"$: {selected_nm:.0f} nm"
                )
            )


        # ----------------------------------------------------
        # Format this refractive-index panel
        # ----------------------------------------------------

        ax.set_ylabel(
            rf"{BAR_STATISTIC_LABEL} "
            r"$|E|/|E_0|$"
        )

        subtitle = (
            "; ".join(
                panel_wavelength_labels
            )
            if panel_wavelength_labels
            else "no acceptable shell data"
        )

        ax.set_title(
            (
                rf"$n_h={n_high:.2f}$"
                "\n"
                + subtitle
            ),
            fontsize=8.5,
        )


        # Identify the core.
        ax.axvspan(
            0.5,
            1.5,
            color="0.5",
            alpha=0.09,
            linewidth=0.0,
            zorder=0,
        )


        if not panel_has_data:

            ax.text(
                0.5,
                0.5,
                "No screened data",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="0.4",
                fontsize=8,
            )


    # ========================================================
    # COMMON Y LIMIT
    # ========================================================

    if mode_values_for_limit:

        pooled_values = np.concatenate(
            mode_values_for_limit
        )

        pooled_values = pooled_values[
            np.isfinite(
                pooled_values
            )
            & (
                pooled_values >= 0.0
            )
        ]

        if pooled_values.size:

            common_y_max = float(
                np.nanpercentile(
                    pooled_values,
                    99.0,
                )
            )

            common_y_max = min(
                common_y_max,
                BAR_DISPLAY_MAXIMUM,
            )

            common_y_max = max(
                common_y_max * 1.05,
                1.0,
            )

        else:

            common_y_max = 1.0

    else:

        common_y_max = 1.0


    for ax in axes:

        ax.set_ylim(
            0.0,
            common_y_max,
        )

        ax.xaxis.set_major_locator(
            MaxNLocator(
                integer=True,
                nbins=12,
            )
        )


    maximum_shell_number = int(
        pd.to_numeric(
            shell_plane_averages[
                "shell_number_one_based"
            ],
            errors="coerce",
        ).max()
    )


    axes[-1].set_xlim(
        0.5,
        maximum_shell_number + 0.5,
    )

    axes[-1].set_xlabel(
        "Radial layer number "
        "(1 = core, outermost number = outer layer)"
    )


    # Avoid duplicate legend entries.
    handles, labels = (
        axes[0].get_legend_handles_labels()
    )

    unique_legend = dict(
        zip(
            labels,
            handles,
        )
    )

    if unique_legend:

        axes[0].legend(
            unique_legend.values(),
            unique_legend.keys(),
            frameon=False,
            fontsize=7,
            ncol=2,
            loc="upper right",
        )


    mode_title = (
        "near the 470 nm design wavelength"
        if mode == "design"
        else (
            "near each cleaned broad-envelope "
            "reflection peak"
        )
    )


    fig.suptitle(
        (
            "Screened shell-resolved field comparison "
            + mode_title
        ),
        fontsize=10,
    )


    fig.text(
        0.01,
        0.005,
        (
            "Exploratory screening; shell median used "
            "where available. Extreme or missing values omitted."
        ),
        fontsize=5.8,
        color="0.35",
    )


    if SAVE_BAR_FIGURES:

        save_figure(
            fig,
            (
                "shell_bar_field_magnitude_"
                f"{mode}_exploratory_screened"
            ),
        )


    plt.show()

    plt.close(fig)


# ============================================================
# DIAGNOSTICS
# ============================================================

bar_selection_table = pd.DataFrame(
    bar_selection_records
)


display_columns = [
    "mode",
    "n_high",
    "k_all_layers",
    "absorbing",
    "case_label",
    "target_wavelength_nm",
    "selected_wavelength_nm",
    "wavelength_mismatch_nm",
    "finite_shell_count",
    "total_shell_count",
    "finite_shell_fraction",
    "selection_successful",
    "reason",
]

display_columns = [
    column
    for column in display_columns
    if column
    in bar_selection_table.columns
]


display(
    bar_selection_table[
        display_columns
    ]
    .sort_values(
        [
            "mode",
            "n_high",
            "absorbing",
        ]
    )
    .reset_index(
        drop=True
    )
    .style.format(
        {
            "n_high": "{:.3f}",
            "k_all_layers": "{:.3f}",
            "target_wavelength_nm": "{:.1f}",
            "selected_wavelength_nm": "{:.1f}",
            "wavelength_mismatch_nm": "{:.1f}",
            "finite_shell_count": "{:.0f}",
            "total_shell_count": "{:.0f}",
            "finite_shell_fraction": "{:.2%}",
        },
        na_rep="not available",
    )
)


bar_selection_table.to_csv(
    TABLE_DIR
    / "grouped_shell_bar_plot_diagnostics.csv",
    index=False,
)


print(
    "Shell statistic used:",
    BAR_SHELL_FIELD_COLUMN,
)

# In[59]:


# ============================================================
# CELL 11: EXPLORATORY MATERIAL-SYSTEM RANKING
#
# Uses:
#   - screened axis profiles;
#   - cleaned broad-envelope reflection peaks;
#   - robust median and 95th-percentile field metrics;
#   - screened core-shell median where available;
#   - explicit profile-quality and wavelength-mismatch information.
#
# These rankings are exploratory because approximately half of the
# original field samples required numerical screening.
# ============================================================

import numpy as np
import pandas as pd

from IPython.display import display


# ============================================================
# SETTINGS
# ============================================================

RANKING_FIELD_COLUMN = (
    "E_magnitude_enhancement_plot"
)

# Prefer the shell median because shell means are highly sensitive
# to numerically extreme pixels.
if (
    "median_E_magnitude_plane_clean"
    in shell_plane_averages.columns
):

    RANKING_SHELL_COLUMN = (
        "median_E_magnitude_plane_clean"
    )

else:

    RANKING_SHELL_COLUMN = (
        "mean_E_magnitude_plane_clean"
    )


# Do not let residual extreme values dominate ranking metrics.
RANKING_DISPLAY_MAXIMUM = 100.0

# Robust high-field statistic.
RANKING_HIGH_PERCENTILE = 95.0

# Central region used for cavity-like enhancement metrics.
RANKING_CENTRAL_HALF_WIDTH_Z_OVER_R = 0.05

# Outer reference region.
RANKING_OUTER_MINIMUM_ABS_Z_OVER_R = 0.70

# n_h = n_low = 1.50 is retained as a reference but excluded from
# the principal ranking order.
RANKING_MINIMUM_N_HIGH = 1.51


# ============================================================
# HELPERS
# ============================================================

def finite_screened_values(
    data,
    column,
    *,
    maximum=RANKING_DISPLAY_MAXIMUM,
):
    """
    Return finite, nonnegative, screened values.
    """

    if data.empty or column not in data.columns:

        return np.array(
            [],
            dtype=float,
        )

    values = pd.to_numeric(
        data[column],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    return values[
        np.isfinite(values)
        & (values >= 0.0)
        & (values <= maximum)
    ]


def robust_profile_metrics(
    profile,
):
    """
    Calculate robust whole-profile, central, and outer metrics.
    """

    empty_result = {
        "profile_median": np.nan,
        "profile_p95": np.nan,
        "profile_maximum_screened": np.nan,
        "central_median": np.nan,
        "central_p95": np.nan,
        "outer_median": np.nan,
        "central_to_outer_ratio": np.nan,
        "finite_profile_sample_count": 0,
        "finite_central_sample_count": 0,
    }

    if (
        profile.empty
        or RANKING_FIELD_COLUMN
        not in profile.columns
        or "z_over_outer_radius"
        not in profile.columns
    ):

        return empty_result


    field_values = pd.to_numeric(
        profile[
            RANKING_FIELD_COLUMN
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    z_over_r = pd.to_numeric(
        profile[
            "z_over_outer_radius"
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )


    valid = (
        np.isfinite(field_values)
        & np.isfinite(z_over_r)
        & (field_values >= 0.0)
        & (
            field_values
            <= RANKING_DISPLAY_MAXIMUM
        )
    )

    whole_values = field_values[
        valid
    ]

    central_values = field_values[
        valid
        & (
            np.abs(z_over_r)
            <= RANKING_CENTRAL_HALF_WIDTH_Z_OVER_R
        )
    ]

    outer_values = field_values[
        valid
        & (
            np.abs(z_over_r)
            >= RANKING_OUTER_MINIMUM_ABS_Z_OVER_R
        )
    ]


    profile_median = (
        float(
            np.nanmedian(
                whole_values
            )
        )
        if whole_values.size
        else np.nan
    )

    profile_p95 = (
        float(
            np.nanpercentile(
                whole_values,
                RANKING_HIGH_PERCENTILE,
            )
        )
        if whole_values.size
        else np.nan
    )

    profile_maximum = (
        float(
            np.nanmax(
                whole_values
            )
        )
        if whole_values.size
        else np.nan
    )

    central_median = (
        float(
            np.nanmedian(
                central_values
            )
        )
        if central_values.size
        else np.nan
    )

    central_p95 = (
        float(
            np.nanpercentile(
                central_values,
                RANKING_HIGH_PERCENTILE,
            )
        )
        if central_values.size
        else np.nan
    )

    outer_median = (
        float(
            np.nanmedian(
                outer_values
            )
        )
        if outer_values.size
        else np.nan
    )

    central_to_outer_ratio = (
        central_median
        / outer_median
        if (
            np.isfinite(
                central_median
            )
            and np.isfinite(
                outer_median
            )
            and outer_median > 0.0
        )
        else np.nan
    )


    return {
        "profile_median": profile_median,
        "profile_p95": profile_p95,
        "profile_maximum_screened": profile_maximum,
        "central_median": central_median,
        "central_p95": central_p95,
        "outer_median": outer_median,
        "central_to_outer_ratio": (
            central_to_outer_ratio
        ),
        "finite_profile_sample_count": int(
            whole_values.size
        ),
        "finite_central_sample_count": int(
            central_values.size
        ),
    }


def core_shell_value(
    case_label,
    wavelength_nm,
):
    """
    Return the screened field statistic for radial layer 1, the core.
    """

    if not np.isfinite(
        wavelength_nm
    ):

        return np.nan


    core_rows = (
        shell_plane_averages.loc[
            (
                shell_plane_averages[
                    "case_label"
                ]
                == case_label
            )
            & np.isclose(
                shell_plane_averages[
                    "wavelength_nm"
                ],
                wavelength_nm,
            )
            & (
                shell_plane_averages[
                    "shell_number_one_based"
                ]
                == 1
            ),
            RANKING_SHELL_COLUMN,
        ]
    )


    core_values = pd.to_numeric(
        core_rows,
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    core_values = core_values[
        np.isfinite(core_values)
        & (core_values >= 0.0)
        & (
            core_values
            <= RANKING_DISPLAY_MAXIMUM
        )
    ]


    return (
        float(
            np.nanmedian(
                core_values
            )
        )
        if core_values.size
        else np.nan
    )


# ============================================================
# BUILD RANKING TABLE
# ============================================================

records = []


for _, case in (
    case_summary
    .sort_values(
        [
            "absorbing",
            "n_high",
        ]
    )
    .iterrows()
):

    (
        design_selected_nm,
        design_data,
        design_diagnostics,
    ) = profile_at(
        case,
        "design",
        require_usable=True,
    )


    (
        peak_selected_nm,
        peak_data,
        peak_diagnostics,
    ) = profile_at(
        case,
        "peak",
        require_usable=True,
    )


    design_metrics = robust_profile_metrics(
        design_data
    )

    peak_metrics = robust_profile_metrics(
        peak_data
    )


    design_core_value = core_shell_value(
        case["case_label"],
        design_selected_nm,
    )

    peak_core_value = core_shell_value(
        case["case_label"],
        peak_selected_nm,
    )


    records.append(
        {
            # -----------------------------------------------
            # System definition
            # -----------------------------------------------
            "case_label": case[
                "case_label"
            ],
            "n_low": float(
                case["n_low"]
            ),
            "n_high": float(
                case["n_high"]
            ),
            "delta_n": float(
                case["n_high"]
                - case["n_low"]
            ),
            "k_all_layers": float(
                case["k_all_layers"]
            ),
            "absorbing": bool(
                case["absorbing"]
            ),
            "reference_case": bool(
                case["n_high"]
                < RANKING_MINIMUM_N_HIGH
            ),

            # -----------------------------------------------
            # Spectral information
            # -----------------------------------------------
            "analysis_peak_wavelength_nm": float(
                case.get(
                    "analysis_peak_wavelength_nm",
                    np.nan,
                )
            ),
            "analysis_peak_value": float(
                case.get(
                    "analysis_peak_value",
                    np.nan,
                )
            ),
            "analysis_peak_prominence_ratio": float(
                case.get(
                    "analysis_peak_prominence_ratio",
                    np.nan,
                )
            ),
            "analysis_peak_is_reliable": bool(
                case.get(
                    "analysis_peak_is_reliable",
                    False,
                )
            ),

            # -----------------------------------------------
            # Wavelengths actually used
            # -----------------------------------------------
            "design_profile_wavelength_nm": (
                design_selected_nm
            ),
            "design_wavelength_mismatch_nm": (
                design_diagnostics.get(
                    "wavelength_mismatch_nm",
                    np.nan,
                )
            ),
            "peak_profile_wavelength_nm": (
                peak_selected_nm
            ),
            "peak_wavelength_mismatch_nm": (
                peak_diagnostics.get(
                    "wavelength_mismatch_nm",
                    np.nan,
                )
            ),

            # -----------------------------------------------
            # Profile quality
            # -----------------------------------------------
            "design_profile_valid_fraction": (
                design_diagnostics.get(
                    "profile_valid_fraction",
                    np.nan,
                )
            ),
            "design_plot_valid_fraction": (
                design_diagnostics.get(
                    "plot_profile_valid_fraction",
                    np.nan,
                )
            ),
            "peak_profile_valid_fraction": (
                peak_diagnostics.get(
                    "profile_valid_fraction",
                    np.nan,
                )
            ),
            "peak_plot_valid_fraction": (
                peak_diagnostics.get(
                    "plot_profile_valid_fraction",
                    np.nan,
                )
            ),
            "design_selection_successful": bool(
                design_diagnostics.get(
                    "selection_successful",
                    False,
                )
            ),
            "peak_selection_successful": bool(
                peak_diagnostics.get(
                    "selection_successful",
                    False,
                )
            ),

            # -----------------------------------------------
            # Design-wavelength profile metrics
            # -----------------------------------------------
            "axis_median_design": (
                design_metrics[
                    "profile_median"
                ]
            ),
            "axis_p95_design": (
                design_metrics[
                    "profile_p95"
                ]
            ),
            "central_median_design": (
                design_metrics[
                    "central_median"
                ]
            ),
            "central_p95_design": (
                design_metrics[
                    "central_p95"
                ]
            ),
            "outer_median_design": (
                design_metrics[
                    "outer_median"
                ]
            ),
            "central_to_outer_ratio_design": (
                design_metrics[
                    "central_to_outer_ratio"
                ]
            ),
            "core_shell_field_design": (
                design_core_value
            ),

            # -----------------------------------------------
            # Peak-wavelength profile metrics
            # -----------------------------------------------
            "axis_median_peak": (
                peak_metrics[
                    "profile_median"
                ]
            ),
            "axis_p95_peak": (
                peak_metrics[
                    "profile_p95"
                ]
            ),
            "central_median_peak": (
                peak_metrics[
                    "central_median"
                ]
            ),
            "central_p95_peak": (
                peak_metrics[
                    "central_p95"
                ]
            ),
            "outer_median_peak": (
                peak_metrics[
                    "outer_median"
                ]
            ),
            "central_to_outer_ratio_peak": (
                peak_metrics[
                    "central_to_outer_ratio"
                ]
            ),
            "core_shell_field_peak": (
                peak_core_value
            ),

            # -----------------------------------------------
            # Sample counts
            # -----------------------------------------------
            "finite_axis_samples_design": (
                design_metrics[
                    "finite_profile_sample_count"
                ]
            ),
            "finite_axis_samples_peak": (
                peak_metrics[
                    "finite_profile_sample_count"
                ]
            ),
            "finite_central_samples_design": (
                design_metrics[
                    "finite_central_sample_count"
                ]
            ),
            "finite_central_samples_peak": (
                peak_metrics[
                    "finite_central_sample_count"
                ]
            ),
        }
    )


ranking = (
    pd.DataFrame(
        records
    )
    .replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )
)


# ============================================================
# CREATE EXPLORATORY RANKING SCORES
#
# Percentile ranks are used because the metrics have different
# numerical scales. Higher values receive higher scores.
# ============================================================

principal_mask = (
    ~ranking[
        "reference_case"
    ]
)


def percentile_score(
    series,
):
    """
    Rank finite values from zero to one.
    """

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    return numeric.rank(
        method="average",
        pct=True,
        na_option="keep",
    )


ranking[
    "score_central_localization_design"
] = percentile_score(
    ranking[
        "central_to_outer_ratio_design"
    ]
)

ranking[
    "score_central_localization_peak"
] = percentile_score(
    ranking[
        "central_to_outer_ratio_peak"
    ]
)

ranking[
    "score_central_strength_design"
] = percentile_score(
    ranking[
        "central_p95_design"
    ]
)

ranking[
    "score_central_strength_peak"
] = percentile_score(
    ranking[
        "central_p95_peak"
    ]
)


score_columns = [
    "score_central_localization_design",
    "score_central_localization_peak",
    "score_central_strength_design",
    "score_central_strength_peak",
]


ranking[
    "exploratory_cavity_score"
] = ranking[
    score_columns
].mean(
    axis=1,
    skipna=True,
)


ranking.loc[
    ranking[
        "reference_case"
    ],
    "exploratory_cavity_score",
] = np.nan


ranking[
    "exploratory_rank"
] = ranking[
    "exploratory_cavity_score"
].rank(
    method="min",
    ascending=False,
    na_option="bottom",
)


# ============================================================
# SAVE COMPLETE RANKING
# ============================================================

ranking.to_csv(
    TABLE_DIR
    / "candidate_system_ranking.csv",
    index=False,
)


# ============================================================
# DISPLAY COMPACT RANKING
# ============================================================

compact_columns = [
    "exploratory_rank",
    "case_label",
    "n_high",
    "delta_n",
    "k_all_layers",
    "analysis_peak_wavelength_nm",
    "design_profile_wavelength_nm",
    "peak_profile_wavelength_nm",
    "design_plot_valid_fraction",
    "peak_plot_valid_fraction",
    "central_median_design",
    "central_p95_design",
    "central_to_outer_ratio_design",
    "central_median_peak",
    "central_p95_peak",
    "central_to_outer_ratio_peak",
    "core_shell_field_design",
    "core_shell_field_peak",
    "exploratory_cavity_score",
]


compact_ranking = (
    ranking.loc[
        principal_mask,
        compact_columns,
    ]
    .sort_values(
        [
            "exploratory_cavity_score",
            "central_to_outer_ratio_peak",
        ],
        ascending=[
            False,
            False,
        ],
        na_position="last",
    )
    .reset_index(
        drop=True
    )
)


display(
    compact_ranking.style.format(
        {
            "exploratory_rank": "{:.0f}",
            "n_high": "{:.3f}",
            "delta_n": "{:.3f}",
            "k_all_layers": "{:.3f}",
            "analysis_peak_wavelength_nm": (
                "{:.1f}"
            ),
            "design_profile_wavelength_nm": (
                "{:.1f}"
            ),
            "peak_profile_wavelength_nm": (
                "{:.1f}"
            ),
            "design_plot_valid_fraction": (
                "{:.2%}"
            ),
            "peak_plot_valid_fraction": (
                "{:.2%}"
            ),
            "central_median_design": "{:.4g}",
            "central_p95_design": "{:.4g}",
            "central_to_outer_ratio_design": (
                "{:.4g}"
            ),
            "central_median_peak": "{:.4g}",
            "central_p95_peak": "{:.4g}",
            "central_to_outer_ratio_peak": (
                "{:.4g}"
            ),
            "core_shell_field_design": "{:.4g}",
            "core_shell_field_peak": "{:.4g}",
            "exploratory_cavity_score": "{:.3f}",
        },
        na_rep="not available",
    )
)


# ============================================================
# DISPLAY REFERENCE CASE SEPARATELY
# ============================================================

reference_ranking = ranking.loc[
    ranking[
        "reference_case"
    ],
    [
        "case_label",
        "n_high",
        "k_all_layers",
        "design_profile_wavelength_nm",
        "central_median_design",
        "central_p95_design",
        "central_to_outer_ratio_design",
        "design_plot_valid_fraction",
    ],
]


if not reference_ranking.empty:

    print(
        "\nReference case, excluded from ranking:"
    )

    display(
        reference_ranking.style.format(
            {
                "n_high": "{:.3f}",
                "k_all_layers": "{:.3f}",
                "design_profile_wavelength_nm": (
                    "{:.1f}"
                ),
                "central_median_design": (
                    "{:.4g}"
                ),
                "central_p95_design": "{:.4g}",
                "central_to_outer_ratio_design": (
                    "{:.4g}"
                ),
                "design_plot_valid_fraction": (
                    "{:.2%}"
                ),
            },
            na_rep="not available",
        )
    )


print(
    "Ranking field column:",
    RANKING_FIELD_COLUMN,
)

print(
    "Ranking shell column:",
    RANKING_SHELL_COLUMN,
)

print(
    "Ranking interpretation: higher exploratory cavity score "
    "means stronger and more centrally localized screened field."
)

# In[ ]:
