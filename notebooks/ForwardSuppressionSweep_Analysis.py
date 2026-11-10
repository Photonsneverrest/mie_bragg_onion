#!/usr/bin/env python
# coding: utf-8

# # Forward-scattering suppression sweep: colour and directional-scattering analysis
#
# This notebook analyses all calculated cases in the forward-scattering
# suppression sweep.
#
# The objectives are:
#
# 1. verify the completeness and numerical consistency of the sweep;
# 2. quantify colour performance using CIELAB and the normalized
#    Rösch-MacAdam metrics;
# 3. quantify forward-scattering suppression using directional,
#    band-integrated, and spectral metrics;
# 4. identify trade-offs between colour quality and directional scattering;
# 5. identify Pareto-optimal and application-relevant designs;
# 6. determine which physical parameters control the observed performance.
#
# No case is excluded silently. Every exclusion or warning is retained in
# an auditable table.

# Code cell 1: Imports and paths

# In[62]:


# ============================================================
# IMPORTS, STYLE, AND PATHS
# ============================================================

from pathlib import Path
import json
import os
import re
import warnings

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from IPython.display import display

from bragg_onion._notebook_tools.io import load_sweep_outputs
from bragg_onion._notebook_tools.colour import (
    add_display_normalized_colour_columns,
)

# ------------------------------------------------------------
# Matplotlib configuration
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
    print("Loaded plotting style:", STYLE_PATH)
else:
    warnings.warn(
        "The paper plotting style was not found. "
        "Matplotlib defaults will be used."
    )

plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 600

# ------------------------------------------------------------
# Sweep output directory
# ------------------------------------------------------------

# Native WSL version of:
#
# C:\Users\SchwarzN\OneDrive - Université de Fribourg\
# Institution\P1_BraggSpericalPigments\Simulation\
# MIE_Computation\Scattnlay_BraggOnion\
# outputs\forward_suppression_sweep

OUTPUT_DIR = Path(
    "/mnt/c/Users/SchwarzN/"
    "OneDrive - Université de Fribourg/"
    "Institution/P1_BraggSpericalPigments/"
    "Simulation/MIE_Computation/"
    "Scattnlay_BraggOnion/outputs/"
    "forward_suppression_sweep"
)

FIGURE_DIR = OUTPUT_DIR / "analysis_figures"
TABLE_DIR = OUTPUT_DIR / "analysis_tables"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

PNG_DPI = 600

print("OUTPUT_DIR :", OUTPUT_DIR)
print("Exists     :", OUTPUT_DIR.exists())
print("FIGURE_DIR :", FIGURE_DIR)
print("TABLE_DIR  :", TABLE_DIR)

if not OUTPUT_DIR.exists():
    raise FileNotFoundError(
        "The forward-suppression sweep directory was not found:\n"
        f"{OUTPUT_DIR}\n\n"
        "Check the spelling and confirm that the OneDrive directory "
        "is accessible from WSL."
    )

# Code cell 2: Inspect everything stored in the directory
#
# This cell is intentionally independent of the project loader. It gives us an inventory of all result and metadata files.

# In[63]:


# ============================================================
# INVENTORY OF THE SWEEP OUTPUT DIRECTORY
# ============================================================

def human_readable_size(number_of_bytes):
    """Return a compact human-readable file size."""
    number_of_bytes = float(number_of_bytes)

    for unit in ["B", "kB", "MB", "GB", "TB"]:
        if number_of_bytes < 1024.0 or unit == "TB":
            return f"{number_of_bytes:.2f} {unit}"
        number_of_bytes /= 1024.0


file_records = []

for path in sorted(OUTPUT_DIR.rglob("*")):
    if not path.is_file():
        continue

    relative_path = path.relative_to(OUTPUT_DIR)

    file_records.append(
        {
            "relative_path": str(relative_path),
            "name": path.name,
            "suffix": path.suffix.lower(),
            "parent": str(relative_path.parent),
            "size_bytes": path.stat().st_size,
            "size": human_readable_size(path.stat().st_size),
        }
    )

df_file_inventory = pd.DataFrame(file_records)

if df_file_inventory.empty:
    raise FileNotFoundError(
        f"No files were found in {OUTPUT_DIR}"
    )

print("Number of files:", len(df_file_inventory))
print(
    "Total size:",
    human_readable_size(df_file_inventory["size_bytes"].sum()),
)

print("\nFiles by extension:")
display(
    df_file_inventory
    .groupby("suffix", dropna=False)
    .agg(
        number_of_files=("relative_path", "size"),
        total_bytes=("size_bytes", "sum"),
    )
    .sort_values("total_bytes", ascending=False)
)

print("\nComplete file inventory:")
display(
    df_file_inventory[
        [
            "relative_path",
            "suffix",
            "size",
        ]
    ]
)

# Code cell 3: Load the combined sweep
#
# Your previous notebooks pass a Windows path to load_sweep_outputs, but here we use the explicit WSL path to avoid ambiguity

# In[64]:


# ============================================================
# LOAD THE COMPLETE FORWARD-SUPPRESSION SWEEP
# ============================================================

SUMMARY_CANDIDATES = [
    OUTPUT_DIR
    / "forward_suppression_summary.parquet",

    OUTPUT_DIR
    / "forward_suppression_summary.pkl",

    OUTPUT_DIR
    / "forward_suppression_summary.csv",
]

summary_path = next(
    (
        path
        for path in SUMMARY_CANDIDATES
        if path.exists()
    ),
    None,
)

if summary_path is None:
    raise FileNotFoundError(
        "No combined forward-suppression summary was found.\n"
        "Checked:\n"
        + "\n".join(
            f"  - {path}"
            for path in SUMMARY_CANDIDATES
        )
    )


# ------------------------------------------------------------
# Load the preferred available format
# ------------------------------------------------------------

if summary_path.suffix == ".parquet":

    df_all = pd.read_parquet(
        summary_path
    )

    USE_PARQUET = True

elif summary_path.suffix == ".pkl":

    df_all = pd.read_pickle(
        summary_path
    )

    USE_PARQUET = False

elif summary_path.suffix == ".csv":

    df_all = pd.read_csv(
        summary_path
    )

    USE_PARQUET = False

else:
    raise ValueError(
        f"Unsupported summary format: {summary_path}"
    )


OUTPUT_DIR_LOADED = OUTPUT_DIR


# ------------------------------------------------------------
# Validate the loaded result
# ------------------------------------------------------------

if df_all.empty:
    raise ValueError(
        "The loaded forward-suppression summary is empty."
    )

if df_all.columns.duplicated().any():

    duplicated_columns = (
        df_all.columns[
            df_all.columns.duplicated()
        ]
        .tolist()
    )

    raise ValueError(
        "The loaded dataframe contains duplicated columns: "
        f"{duplicated_columns}"
    )


# Preserve an untouched copy.
df_raw = df_all.copy(deep=True)


# ------------------------------------------------------------
# Report the loaded dataset
# ------------------------------------------------------------

print("Loaded forward-suppression sweep")
print("--------------------------------")
print("File              :", summary_path)
print("Format            :", summary_path.suffix)
print("Number of rows    :", len(df_raw))
print("Number of columns :", len(df_raw.columns))
print("Memory usage      :", human_readable_size(
    df_raw.memory_usage(
        index=True,
        deep=True,
    ).sum()
))

print("\nFirst rows:")
display(df_raw.head())

# In[65]:


# ============================================================
# BASIC SWEEP-COMPLETENESS CHECK
# ============================================================

print(
    "Collection directions:",
    sorted(
        df_raw[
            "collection_direction"
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    if "collection_direction" in df_raw.columns
    else "column unavailable",
)

print(
    "Reference wavelengths:",
    sorted(
        pd.to_numeric(
            df_raw[
                "reference_design_wavelength_nm"
            ],
            errors="coerce",
        )
        .dropna()
        .unique()
        .tolist()
    )
    if (
        "reference_design_wavelength_nm"
        in df_raw.columns
    )
    else "column unavailable",
)

print(
    "Case families:",
    sorted(
        df_raw[
            "case_family"
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    if "case_family" in df_raw.columns
    else "column unavailable",
)

print(
    "Scenarios:",
    sorted(
        df_raw[
            "scenario"
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    if "scenario" in df_raw.columns
    else "column unavailable",
)

# Code cell 4: Column schema and candidate metric detection
#
# This is the most important first diagnostic. It identifies likely colour, forward, backward, band, spectral, warning, and geometry columns automatically.

# In[66]:


# ============================================================
# AUDIT THE AVAILABLE COLUMNS
#
# Array-safe version:
# Some columns contain wavelength-dependent or angular arrays.
# Such arrays cannot be passed directly to Series.nunique().
# ============================================================

def classify_column(column_name):
    """
    Assign broad scientific categories based on the column name.
    """

    name = str(column_name).lower()
    categories = []

    keyword_groups = {
        "colour": [
            "eta_c",
            "eta_l",
            "eta_y",
            "cielab",
            "hue",
            "chroma",
            "hex",
            "colour",
            "color",
            "xyz",
            "lab",
            "d_opt",
        ],
        "forward": [
            "forward",
            "fwd",
        ],
        "backward": [
            "backward",
            "back",
            "bwd",
        ],
        "directionality": [
            "asymmetry",
            "direction",
            "forward",
            "backward",
            "side",
            "g_",
            "_g",
        ],
        "scattering": [
            "qsca",
            "qext",
            "qabs",
            "qbk",
            "qpr",
            "scatt",
            "albedo",
            "cross_section",
            "collected",
        ],
        "spectral": [
            "wavelength",
            "peak",
            "band",
            "spectrum",
            "spectral",
            "fwhm",
        ],
        "geometry": [
            "diameter",
            "radius",
            "layer",
            "shell",
            "thickness",
            "core",
            "outer",
        ],
        "material": [
            "n_a",
            "n_b",
            "index",
            "medium",
            "extinction",
            "absorber",
            "material",
        ],
        "quality_control": [
            "warning",
            "error",
            "valid",
            "passiv",
            "converg",
            "failed",
            "status",
        ],
    }

    for category, keywords in (
        keyword_groups.items()
    ):
        if any(
            keyword in name
            for keyword in keywords
        ):
            categories.append(category)

    if not categories:
        categories.append("other")

    return ", ".join(categories)


def value_type_name(value):
    """
    Return a concise description of one stored value.
    """

    if isinstance(value, np.ndarray):
        return (
            f"ndarray"
            f"[shape={value.shape}, "
            f"dtype={value.dtype}]"
        )

    if isinstance(value, list):
        return f"list[length={len(value)}]"

    if isinstance(value, tuple):
        return f"tuple[length={len(value)}]"

    if isinstance(value, dict):
        return f"dict[length={len(value)}]"

    return type(value).__name__


def make_hashable_summary(value):
    """
    Convert scalar or array-like values into hashable summaries.

    The conversion is used only for schema inspection. It does not
    modify the original dataframe.
    """

    if isinstance(value, np.ndarray):

        array = np.asarray(value)

        if array.size == 0:
            return (
                "ndarray",
                tuple(array.shape),
                str(array.dtype),
                "empty",
            )

        if np.issubdtype(
            array.dtype,
            np.number,
        ):

            finite = array[
                np.isfinite(array)
            ]

            if finite.size == 0:

                return (
                    "ndarray",
                    tuple(array.shape),
                    str(array.dtype),
                    "no_finite_values",
                )

            return (
                "ndarray",
                tuple(array.shape),
                str(array.dtype),
                float(np.nanmin(array)),
                float(np.nanmax(array)),
                float(np.nanmean(array)),
            )

        return (
            "ndarray",
            tuple(array.shape),
            str(array.dtype),
            repr(array.ravel()[:3].tolist()),
        )

    if isinstance(value, list):

        return (
            "list",
            len(value),
            repr(value[:3]),
        )

    if isinstance(value, tuple):

        return (
            "tuple",
            len(value),
            repr(value[:3]),
        )

    if isinstance(value, dict):

        return (
            "dict",
            tuple(
                sorted(
                    map(str, value.keys())
                )
            ),
        )

    try:
        hash(value)
        return value

    except TypeError:
        return repr(value)


def safe_number_unique(series):
    """
    Count unique scalar values or unique array summaries without
    raising an exception for unhashable types.
    """

    if series.empty:
        return 0

    try:
        return series.nunique(dropna=True)

    except TypeError:

        hashable_summaries = (
            series
            .apply(make_hashable_summary)
            .dropna()
        )

        return hashable_summaries.nunique(dropna=True)

# In[67]:


# ============================================================
# IDENTIFY ARRAY-VALUED COLUMNS
# ============================================================

array_column_records = []

for column in df_raw.columns:

    series = df_raw[column]

    array_values = [
        value
        for value in series
        if isinstance(
            value,
            np.ndarray,
        )
    ]

    if not array_values:
        continue

    shapes = sorted(
        {
            tuple(
                np.asarray(
                    value
                ).shape
            )
            for value in array_values
        }
    )

    dtypes = sorted(
        {
            str(
                np.asarray(
                    value
                ).dtype
            )
            for value in array_values
        }
    )

    lengths = sorted(
        {
            int(
                np.asarray(
                    value
                ).size
            )
            for value in array_values
        }
    )

    array_column_records.append(
        {
            "column": column,
            "number_of_array_rows": (
                len(array_values)
            ),
            "shapes": shapes,
            "dtypes": dtypes,
            "array_sizes": lengths,
        }
    )


df_array_columns = pd.DataFrame(
    array_column_records
)

print(
    "Number of array-valued columns:",
    len(df_array_columns),
)

display(df_array_columns)


df_array_columns.to_csv(
    TABLE_DIR
    / "array_valued_columns.csv",
    index=False,
)

# Code cell 5: Print scientifically relevant column groups

# In[68]:


# ============================================================
# SCIENTIFICALLY RELEVANT COLUMNS AND DIRECTION VALUES
# ============================================================

def columns_containing(
    *keywords,
):
    keywords = [
        keyword.lower()
        for keyword in keywords
    ]

    return [
        column
        for column in df_raw.columns
        if any(
            keyword
            in str(column).lower()
            for keyword in keywords
        )
    ]


column_groups = {
    "colour": columns_containing(
        "eta_c",
        "eta_l",
        "eta_y",
        "hue",
        "chroma",
        "hex",
        "colour",
        "color",
        "d_opt",
    ),
    "collection": columns_containing(
        "collected",
        "collection",
        "fraction",
    ),
    "scattering": columns_containing(
        "qsca",
        "qext",
        "qabs",
        "qbk",
        "qpr",
        "albedo",
        "asymmetry",
        "g",
    ),
    "geometry": columns_containing(
        "diameter",
        "radius",
        "layer",
        "shell",
        "thickness",
        "core",
    ),
    "spectral_arrays": columns_containing(
        "wavelength",
        "spectrum",
        "theta",
        "angle",
    ),
    "quality_control": columns_containing(
        "warning",
        "error",
        "valid",
        "passiv",
        "status",
        "failed",
    ),
}


for group_name, columns in (
    column_groups.items()
):

    print(
        f"\n{group_name.upper()}"
    )

    print(
        "-" * len(group_name)
    )

    if columns:

        for column in columns:
            print(column)

    else:

        print(
            "No matching columns found."
        )


# ------------------------------------------------------------
# Direction is stored as categorical values, not column names.
# ------------------------------------------------------------

if "collection_direction" in df_raw.columns:

    print(
        "\nCOLLECTION DIRECTION VALUES"
    )

    print(
        "---------------------------"
    )

    print(
        sorted(
            df_raw[
                "collection_direction"
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
    )

    display(
        df_raw[
            "collection_direction"
        ]
        .value_counts(
            dropna=False
        )
        .rename_axis(
            "collection_direction"
        )
        .reset_index(
            name="number_of_rows"
        )
    )

else:

    print(
        "\nWarning: collection_direction "
        "column was not found."
    )

# Code cell 6: Detect parameters and dimensions of the sweep
#
# This helps separate input parameters from result metrics.

# In[69]:


# ============================================================
# DETECT LIKELY SWEEP PARAMETERS
#
# Array-valued columns are excluded because they contain spectra
# or angularly resolved quantities rather than scalar parameters.
# ============================================================

known_result_patterns = [
    "eta_",
    "qsca",
    "qext",
    "qabs",
    "qbk",
    "qpr",
    "collected",
    "forward",
    "backward",
    "albedo",
    "asymmetry",
    "hue",
    "hex",
    "peak",
    "maximum",
    "mean",
    "rms",
    "warning",
    "error",
    "spectrum",
    "wavelengths",
    "theta",
    "angles",
]


def contains_unhashable_values(
    series,
    maximum_values_to_check=50,
):
    """
    Return True if the series contains arrays, lists, dictionaries,
    sets, or other unhashable objects.
    """

    checked = 0

    for value in series:

        if value is None:
            continue

        if (
            isinstance(value, float)
            and np.isnan(value)
        ):
            continue

        if isinstance(
            value,
            (
                np.ndarray,
                list,
                dict,
                set,
            ),
        ):
            return True

        try:
            hash(value)

        except TypeError:
            return True

        checked += 1

        if checked >= maximum_values_to_check:
            break

    return False


def safe_unique_scalar_values(
    series,
):
    """
    Return unique scalar values.

    Array-valued or otherwise unhashable columns return None.
    """

    if contains_unhashable_values(series):
        return None

    try:

        return series.dropna().unique()

    except TypeError:

        return None


likely_parameter_columns = []
excluded_array_columns = []

for column in df_raw.columns:

    name = str(column).lower()
    series = df_raw[column]

    unique_values = (
        safe_unique_scalar_values(
            series
        )
    )

    if unique_values is None:

        excluded_array_columns.append(
            column
        )

        continue

    number_unique = len(
        unique_values
    )

    if number_unique <= 1:
        continue

    if any(
        pattern in name
        for pattern in known_result_patterns
    ):
        continue

    # A parameter usually has considerably fewer unique values
    # than the number of calculated result rows.
    maximum_parameter_values = max(
        100,
        int(
            0.10
            * len(df_raw)
        ),
    )

    if number_unique <= maximum_parameter_values:

        likely_parameter_columns.append(
            column
        )


# ============================================================
# BUILD PARAMETER SUMMARY
# ============================================================

parameter_summary_records = []

for column in likely_parameter_columns:

    series = df_raw[column]

    unique_values = (
        safe_unique_scalar_values(
            series
        )
    )

    if unique_values is None:
        continue

    number_unique = len(
        unique_values
    )

    if number_unique <= 20:

        try:

            values_preview = sorted(
                unique_values.tolist()
            )

        except TypeError:

            values_preview = (
                unique_values.tolist()
            )

    else:

        values_preview = (
            unique_values[:20]
            .tolist()
        )

        values_preview.append(
            "..."
        )

    parameter_summary_records.append(
        {
            "parameter": column,
            "dtype": str(
                series.dtype
            ),
            "number_of_values": (
                number_unique
            ),
            "values_preview": (
                values_preview
            ),
        }
    )


df_parameter_summary = pd.DataFrame(
    parameter_summary_records
)


if df_parameter_summary.empty:

    print(
        "No likely scalar sweep parameters "
        "were detected automatically."
    )

else:

    df_parameter_summary = (
        df_parameter_summary
        .sort_values(
            [
                "number_of_values",
                "parameter",
            ]
        )
        .reset_index(drop=True)
    )

    print(
        "Likely scalar sweep parameters:"
    )

    display(
        df_parameter_summary
    )


print(
    "\nArray-valued columns excluded from "
    "parameter detection:"
)

for column in excluded_array_columns:
    print("  ", column)


parameter_summary_path = (
    TABLE_DIR
    / "likely_sweep_parameters.csv"
)

df_parameter_summary.to_csv(
    parameter_summary_path,
    index=False,
)

print(
    "\nSaved:",
    parameter_summary_path,
)

# In[70]:


# ============================================================
# EXPLICIT SUMMARY OF THE FORWARD-SUPPRESSION SWEEP
# ============================================================

expected_parameter_columns = [
    "reference_design_wavelength_nm",
    "case_family",
    "scenario",
    "absorber_location",
    "absorber_k",
    "core_thickness_factor",
    "n_shells",
    "outer_shell_thickness_nm",
    "outer_shell_n",
    "collection_direction",
]

explicit_parameter_records = []

for column in expected_parameter_columns:

    if column not in df_raw.columns:
        continue

    series = df_raw[column]

    unique_values = (
        safe_unique_scalar_values(
            series
        )
    )

    if unique_values is None:

        values = [
            "unhashable values"
        ]

    else:

        try:

            values = sorted(
                unique_values.tolist()
            )

        except TypeError:

            values = (
                unique_values.tolist()
            )

    explicit_parameter_records.append(
        {
            "parameter": column,
            "number_of_values": (
                len(unique_values)
                if unique_values
                is not None
                else np.nan
            ),
            "values": values,
        }
    )


df_explicit_parameter_summary = (
    pd.DataFrame(
        explicit_parameter_records
    )
)

display(
    df_explicit_parameter_summary
)

# Code cell 7: Basic duplicate and missing-data audit

# In[71]:


# ============================================================
# ARRAY-SAFE DATA-INTEGRITY AUDIT
# ============================================================

# ------------------------------------------------------------
# Select scalar columns that uniquely describe one result row
# ------------------------------------------------------------

candidate_identifier_columns = [
    "task_index",
    "task_hash",
    "reference_design_wavelength_nm",
    "case_family",
    "scenario",
    "absorber_location",
    "absorber_k",
    "core_thickness_factor",
    "n_shells",
    "outer_shell_thickness_nm",
    "outer_shell_n",
    "collection_direction",
]

identifier_columns = [
    column
    for column in candidate_identifier_columns
    if column in df_raw.columns
]


if not identifier_columns:
    raise KeyError(
        "No suitable scalar identifier columns were found."
    )


print(
    "Columns used for duplicate detection:"
)

for column in identifier_columns:
    print("  ", column)


# ------------------------------------------------------------
# Duplicate checks
# ------------------------------------------------------------

duplicate_identifier_mask = (
    df_raw.duplicated(
        subset=identifier_columns,
        keep=False,
    )
)

number_duplicated_rows = int(
    duplicate_identifier_mask.sum()
)

number_duplicated_groups = int(
    df_raw.loc[
        duplicate_identifier_mask,
        identifier_columns,
    ]
    .drop_duplicates()
    .shape[0]
)


# A task should usually have exactly two compact result rows:
# one backward and one forward.
task_group_columns = [
    column
    for column in [
        "task_index",
        "task_hash",
    ]
    if column in df_raw.columns
]

if task_group_columns:

    df_rows_per_task = (
        df_raw
        .groupby(
            task_group_columns,
            dropna=False,
        )
        .size()
        .rename(
            "rows_per_task"
        )
        .reset_index()
    )

else:

    df_rows_per_task = pd.DataFrame()


# ------------------------------------------------------------
# Missingness audit
# ------------------------------------------------------------

columns_fully_missing = [
    column
    for column in df_raw.columns
    if df_raw[column].isna().all()
]

rows_with_missing_values = int(
    df_raw.isna().any(axis=1).sum()
)


audit_records = [
    {
        "check": "number_of_rows",
        "value": len(df_raw),
    },
    {
        "check": "number_of_columns",
        "value": len(df_raw.columns),
    },
    {
        "check": (
            "rows_in_duplicated_identifier_groups"
        ),
        "value": number_duplicated_rows,
    },
    {
        "check": (
            "duplicated_identifier_groups"
        ),
        "value": number_duplicated_groups,
    },
    {
        "check": (
            "columns_containing_only_missing_values"
        ),
        "value": len(
            columns_fully_missing
        ),
    },
    {
        "check": (
            "rows_containing_at_least_one_missing_value"
        ),
        "value": rows_with_missing_values,
    },
]


df_basic_audit = pd.DataFrame(
    audit_records
)

display(
    df_basic_audit
)


# ------------------------------------------------------------
# Inspect duplicated identifiers
# ------------------------------------------------------------

if number_duplicated_rows > 0:

    print(
        "\nDuplicated identifier combinations:"
    )

    display(
        df_raw.loc[
            duplicate_identifier_mask,
            identifier_columns,
        ]
        .sort_values(
            identifier_columns
        )
    )

else:

    print(
        "\nNo duplicated result identifiers were found."
    )


# ------------------------------------------------------------
# Verify rows per physical task
# ------------------------------------------------------------

if not df_rows_per_task.empty:

    print(
        "\nNumber of compact result rows per physical task:"
    )

    display(
        df_rows_per_task[
            "rows_per_task"
        ]
        .value_counts()
        .sort_index()
        .rename_axis(
            "rows_per_task"
        )
        .reset_index(
            name="number_of_tasks"
        )
    )

    unexpected_task_sizes = (
        df_rows_per_task.loc[
            df_rows_per_task[
                "rows_per_task"
            ]
            != 2
        ]
    )

    if unexpected_task_sizes.empty:

        print(
            "Every physical task has exactly two rows: "
            "one backward and one forward result."
        )

    else:

        print(
            "Warning: some tasks do not have exactly "
            "two compact result rows."
        )

        display(
            unexpected_task_sizes
        )


# ------------------------------------------------------------
# Columns with missing values
# ------------------------------------------------------------

missingness_records = []

for column in df_raw.columns:

    series = df_raw[column]

    number_missing = int(
        series.isna().sum()
    )

    if number_missing == 0:
        continue

    missingness_records.append(
        {
            "column": column,
            "dtype": str(
                series.dtype
            ),
            "missing": number_missing,
            "missing_fraction": (
                number_missing
                / len(series)
            ),
        }
    )


df_missingness = pd.DataFrame(
    missingness_records
)


if df_missingness.empty:

    print(
        "\nNo missing values were found."
    )

else:

    df_missingness = (
        df_missingness
        .sort_values(
            "missing_fraction",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    print(
        "\nColumns containing missing values:"
    )

    display(
        df_missingness.style.format(
            {
                "missing_fraction": (
                    "{:.2%}"
                ),
            }
        )
    )


# ------------------------------------------------------------
# Save audit tables
# ------------------------------------------------------------

df_basic_audit.to_csv(
    TABLE_DIR
    / "basic_integrity_audit.csv",
    index=False,
)

df_rows_per_task.to_csv(
    TABLE_DIR
    / "rows_per_task.csv",
    index=False,
)

df_missingness.to_csv(
    TABLE_DIR
    / "missingness_audit.csv",
    index=False,
)

print(
    "\nSaved integrity-audit tables to:",
    TABLE_DIR,
)

# Code cell 8: Inspect possible forward-suppression metrics numerically

# In[72]:


# ============================================================
# PAIR FORWARD AND BACKWARD RESULTS
# AND CALCULATE PROVISIONAL SCALAR METRICS
#
# Important:
# The compact summary stores forward and backward results as
# separate rows identified by collection_direction.
#
# This cell converts them into one row per physical case and
# calculates:
#
#   - backward/forward peak-efficiency ratio
#   - directional contrast in dB
#   - forward suppression relative to the nonabsorbing reference
#   - backward-scattering retention relative to the reference
#   - changes in backward-colour CIELAB coordinates
#
# Peak-based metrics are useful for initial screening but are not
# final spectral figures because forward and backward maxima can
# occur at different wavelengths. Spectrally matched metrics will
# later be calculated from the full checkpoints.
# ============================================================

import numpy as np
import pandas as pd

from IPython.display import display


# ============================================================
# SETTINGS
# ============================================================

DIRECTION_COLUMN = "collection_direction"

DIRECTION_VALUES_REQUIRED = {
    "backward",
    "forward",
}

# Small positive number used only to prevent division by zero.
# The resulting ratios should still be inspected for cases with
# negligibly small absolute scattering.
RATIO_EPSILON = 1e-15


# ============================================================
# VALIDATE DIRECTION ENCODING
# ============================================================

if DIRECTION_COLUMN not in df_raw.columns:
    raise KeyError(
        f"Required column {DIRECTION_COLUMN!r} was not found."
    )

direction_values = set(
    df_raw[
        DIRECTION_COLUMN
    ]
    .dropna()
    .astype(str)
    .str.lower()
    .unique()
)

missing_directions = (
    DIRECTION_VALUES_REQUIRED
    - direction_values
)

if missing_directions:
    raise ValueError(
        "The compact summary does not contain all required "
        "collection directions.\n"
        f"Available: {sorted(direction_values)}\n"
        f"Missing: {sorted(missing_directions)}"
    )

print(
    "Collection directions:",
    sorted(direction_values),
)


# Standardize direction names.
df_directional = df_raw.copy()

df_directional[
    DIRECTION_COLUMN
] = (
    df_directional[
        DIRECTION_COLUMN
    ]
    .astype(str)
    .str.lower()
)


# ============================================================
# DEFINE PHYSICAL-CASE IDENTIFIERS
# ============================================================

# task_hash should uniquely identify one physical case.
# task_index is retained for traceability but is not required
# for matching if task_hash is available.

if "task_hash" in df_directional.columns:

    case_identifier_columns = [
        "task_hash",
    ]

    if "task_index" in df_directional.columns:
        case_identifier_columns.insert(
            0,
            "task_index",
        )

else:

    # Fallback if task_hash is ever absent.
    fallback_identifiers = [
        "reference_design_wavelength_nm",
        "case_family",
        "scenario",
        "absorber_location",
        "absorber_k",
        "core_thickness_factor",
        "n_shells",
        "outer_shell_thickness_nm",
        "outer_shell_n",
    ]

    case_identifier_columns = [
        column
        for column in fallback_identifiers
        if column in df_directional.columns
    ]

    if not case_identifier_columns:
        raise KeyError(
            "No physical-case identifier columns were found."
        )


print(
    "Physical-case identifiers:",
    case_identifier_columns,
)


# ============================================================
# VERIFY ONE ROW PER DIRECTION AND PHYSICAL CASE
# ============================================================

direction_count_table = (
    df_directional
    .groupby(
        case_identifier_columns,
        dropna=False,
    )[DIRECTION_COLUMN]
    .agg(
        number_of_rows="size",
        number_of_directions="nunique",
    )
    .reset_index()
)

invalid_direction_groups = (
    direction_count_table.loc[
        (
            direction_count_table[
                "number_of_rows"
            ]
            != 2
        )
        | (
            direction_count_table[
                "number_of_directions"
            ]
            != 2
        )
    ]
)

if not invalid_direction_groups.empty:

    print(
        "Warning: some physical cases do not contain exactly "
        "one forward and one backward row."
    )

    display(
        invalid_direction_groups
    )

    raise ValueError(
        "Resolve incomplete or duplicated direction pairs before "
        "continuing with the analysis."
    )

print(
    "All physical cases contain exactly one forward and "
    "one backward result row."
)


# ============================================================
# SELECT SCALAR RESULT COLUMNS FOR PAIRING
# ============================================================

candidate_directional_result_columns = [
    # Collected scattering
    "q_collected_geom_max",
    "q_collected_geom_peak_wavelength_nm",
    "Ccollected_max_m2",
    "fraction_collected_max",
    "fraction_collected_mean",

    # Colour coordinates
    "L",
    "a",
    "b",
    "C",
    "hue_deg",
    "eta_C",
    "eta_L",
    "eta_Y",
    "X",
    "Y",
    "Z",
    "hex",
    "sRGB_r_clipped",
    "sRGB_g_clipped",
    "sRGB_b_clipped",
    "sRGB_was_clipped",

    # Scalar spectral summaries
    "spectrum_peak_wavelength_nm",
    "spectrum_peak_value",
    "spectrum_integral",
    "spectrum_centroid_nm",

    # Solver summaries
    "Qsca_max",
    "Qsca_peak_wavelength_nm",
    "Qext_max",
    "Qabs_max",
]

directional_result_columns = [
    column
    for column in candidate_directional_result_columns
    if column in df_directional.columns
]

if not directional_result_columns:
    raise KeyError(
        "No scalar directional result columns were found."
    )

print(
    "\nDirectional scalar columns to pair:"
)

for column in directional_result_columns:
    print("  ", column)


# ============================================================
# PIVOT FORWARD AND BACKWARD RESULTS
# ============================================================

directional_wide = (
    df_directional[
        case_identifier_columns
        + [DIRECTION_COLUMN]
        + directional_result_columns
    ]
    .set_index(
        case_identifier_columns
        + [DIRECTION_COLUMN]
    )
    .unstack(
        DIRECTION_COLUMN
    )
)


# Flatten the two-level column index.
directional_wide.columns = [
    f"{result_column}_{direction}"
    for result_column, direction
    in directional_wide.columns
]

directional_wide = (
    directional_wide
    .reset_index()
)


# ============================================================
# ADD CASE METADATA ONCE PER PHYSICAL CASE
# ============================================================

metadata_candidates = [
    "reference_design_wavelength_nm",
    "reference_id",
    "case_family",
    "scenario",
    "absorber_location",
    "absorber_k",
    "core_thickness_factor",
    "n_shells",
    "outer_shell_thickness_nm",
    "outer_shell_n",
    "n_A",
    "n_B",
    "outer_layer",
    "reference_n_shells",
    "reference_diameter_nm",
    "reference_core_thickness_factor",
    "original_reference_k",
    "original_reference_absorber_location",
    "predicted_diameter_nm",
    "predicted_diameter_difference_nm",
    "reference_absorber_removed",
]

metadata_columns = [
    column
    for column in metadata_candidates
    if column in df_directional.columns
    and column not in case_identifier_columns
]

df_case_metadata = (
    df_directional[
        case_identifier_columns
        + metadata_columns
    ]
    .drop_duplicates(
        subset=case_identifier_columns
    )
)


df_paired = directional_wide.merge(
    df_case_metadata,
    on=case_identifier_columns,
    how="left",
    validate="one_to_one",
)


print(
    "\nPaired dataframe shape:",
    df_paired.shape,
)

print(
    "Physical cases:",
    len(df_paired),
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_ratio(
    numerator,
    denominator,
    epsilon=RATIO_EPSILON,
):
    """
    Calculate a nonnegative ratio while avoiding division by zero.
    """

    numerator = pd.to_numeric(
        numerator,
        errors="coerce",
    )

    denominator = pd.to_numeric(
        denominator,
        errors="coerce",
    )

    ratio = (
        numerator
        / (
            denominator
            + epsilon
        )
    )

    ratio.loc[
        (
            numerator < 0.0
        )
        | (
            denominator < 0.0
        )
    ] = np.nan

    return ratio


def circular_difference_deg(
    values_deg,
    reference_deg,
):
    """
    Wrapped circular difference in the interval [-180, 180).
    """

    return (
        (
            np.asarray(
                values_deg,
                dtype=float,
            )
            - np.asarray(
                reference_deg,
                dtype=float,
            )
            + 180.0
        )
        % 360.0
        - 180.0
    )


# ============================================================
# DIRECTIONAL SCATTERING METRICS
# ============================================================

required_peak_columns = {
    "q_collected_geom_max_backward",
    "q_collected_geom_max_forward",
}

if required_peak_columns.issubset(
    df_paired.columns
):

    q_back_peak = pd.to_numeric(
        df_paired[
            "q_collected_geom_max_backward"
        ],
        errors="coerce",
    )

    q_forward_peak = pd.to_numeric(
        df_paired[
            "q_collected_geom_max_forward"
        ],
        errors="coerce",
    )

    # Ratio above one indicates stronger backward than forward
    # collected peak efficiency.
    df_paired[
        "backward_to_forward_peak_ratio"
    ] = safe_ratio(
        q_back_peak,
        q_forward_peak,
    )

    # Positive values indicate backward-dominant peak scattering.
    df_paired[
        "backward_to_forward_peak_db"
    ] = (
        10.0
        * np.log10(
            df_paired[
                "backward_to_forward_peak_ratio"
            ]
        )
    )

    # Fraction of the two hemispherical peak metrics associated
    # with backward collection.
    #
    # This remains provisional because the two maxima can occur at
    # different wavelengths.
    df_paired[
        "backward_peak_share"
    ] = safe_ratio(
        q_back_peak,
        q_back_peak + q_forward_peak,
    )

else:

    print(
        "Warning: q_collected_geom_max was unavailable in one or "
        "both collection directions."
    )


# ============================================================
# IDENTIFY THE NEW NONABSORBING REFERENCE CASES
# ============================================================

reference_mask = pd.Series(
    False,
    index=df_paired.index,
)

if "case_family" in df_paired.columns:

    reference_mask |= (
        df_paired["case_family"]
        .astype(str)
        .str.lower()
        .eq("reference")
    )

if (
    "absorber_location"
    in df_paired.columns
    and "absorber_k"
    in df_paired.columns
):

    reference_mask |= (
        df_paired[
            "absorber_location"
        ]
        .astype(str)
        .str.lower()
        .eq("none")
        & np.isclose(
            pd.to_numeric(
                df_paired[
                    "absorber_k"
                ],
                errors="coerce",
            ),
            0.0,
        )
    )


df_reference_cases = (
    df_paired.loc[
        reference_mask
    ]
    .copy()
)


if "reference_design_wavelength_nm" not in df_reference_cases.columns:
    raise KeyError(
        "reference_design_wavelength_nm is required to pair each "
        "absorbing case with its nonabsorbing reference."
    )


reference_counts = (
    df_reference_cases[
        "reference_design_wavelength_nm"
    ]
    .value_counts(
        dropna=False
    )
)

print(
    "\nNonabsorbing reference cases per design wavelength:"
)

display(
    reference_counts
    .rename_axis(
        "reference_design_wavelength_nm"
    )
    .reset_index(
        name="number_of_reference_cases"
    )
)


if not (
    reference_counts == 1
).all():

    display(
        df_reference_cases[
            [
                column
                for column in [
                    "reference_design_wavelength_nm",
                    "task_index",
                    "task_hash",
                    "case_family",
                    "scenario",
                    "absorber_location",
                    "absorber_k",
                ]
                if column
                in df_reference_cases.columns
            ]
        ]
    )

    raise ValueError(
        "Expected exactly one new nonabsorbing reference for each "
        "design wavelength."
    )


# ============================================================
# MERGE REFERENCE METRICS INTO EVERY CASE
# ============================================================

reference_metric_candidates = [
    "q_collected_geom_max_backward",
    "q_collected_geom_max_forward",
    "spectrum_integral_backward",
    "spectrum_integral_forward",
    "L_backward",
    "a_backward",
    "b_backward",
    "C_backward",
    "hue_deg_backward",
    "eta_C_backward",
    "eta_L_backward",
    "eta_Y_backward",
    "hex_backward",
]

reference_metric_columns = [
    column
    for column in reference_metric_candidates
    if column in df_reference_cases.columns
]


df_reference_metrics = (
    df_reference_cases[
        [
            "reference_design_wavelength_nm",
        ]
        + reference_metric_columns
    ]
    .copy()
)


df_reference_metrics = (
    df_reference_metrics.rename(
        columns={
            column: (
                f"{column}_reference"
            )
            for column
            in reference_metric_columns
        }
    )
)


df_paired = df_paired.merge(
    df_reference_metrics,
    on="reference_design_wavelength_nm",
    how="left",
    validate="many_to_one",
)


# ============================================================
# FORWARD SUPPRESSION AND BACKWARD RETENTION
# ============================================================

if {
    "q_collected_geom_max_forward",
    "q_collected_geom_max_forward_reference",
}.issubset(
    df_paired.columns
):

    df_paired[
        "forward_peak_retention"
    ] = safe_ratio(
        df_paired[
            "q_collected_geom_max_forward"
        ],
        df_paired[
            "q_collected_geom_max_forward_reference"
        ],
    )

    df_paired[
        "forward_peak_suppression_fraction"
    ] = (
        1.0
        - df_paired[
            "forward_peak_retention"
        ]
    )

    # Positive dB values mean that forward peak scattering has
    # been suppressed relative to the nonabsorbing reference.
    df_paired[
        "forward_peak_suppression_db"
    ] = (
        -10.0
        * np.log10(
            df_paired[
                "forward_peak_retention"
            ]
        )
    )


if {
    "q_collected_geom_max_backward",
    "q_collected_geom_max_backward_reference",
}.issubset(
    df_paired.columns
):

    df_paired[
        "backward_peak_retention"
    ] = safe_ratio(
        df_paired[
            "q_collected_geom_max_backward"
        ],
        df_paired[
            "q_collected_geom_max_backward_reference"
        ],
    )

    df_paired[
        "backward_peak_change_fraction"
    ] = (
        df_paired[
            "backward_peak_retention"
        ]
        - 1.0
    )


# Spectral-integral metrics are preferable to independent maxima
# if spectrum_integral exists in the compact result table.
if {
    "spectrum_integral_forward",
    "spectrum_integral_forward_reference",
}.issubset(
    df_paired.columns
):

    df_paired[
        "forward_integral_retention"
    ] = safe_ratio(
        df_paired[
            "spectrum_integral_forward"
        ],
        df_paired[
            "spectrum_integral_forward_reference"
        ],
    )

    df_paired[
        "forward_integral_suppression_fraction"
    ] = (
        1.0
        - df_paired[
            "forward_integral_retention"
        ]
    )

    df_paired[
        "forward_integral_suppression_db"
    ] = (
        -10.0
        * np.log10(
            df_paired[
                "forward_integral_retention"
            ]
        )
    )


if {
    "spectrum_integral_backward",
    "spectrum_integral_backward_reference",
}.issubset(
    df_paired.columns
):

    df_paired[
        "backward_integral_retention"
    ] = safe_ratio(
        df_paired[
            "spectrum_integral_backward"
        ],
        df_paired[
            "spectrum_integral_backward_reference"
        ],
    )

    df_paired[
        "backward_to_forward_integral_ratio"
    ] = safe_ratio(
        df_paired[
            "spectrum_integral_backward"
        ],
        df_paired[
            "spectrum_integral_forward"
        ],
    )

    df_paired[
        "backward_to_forward_integral_db"
    ] = (
        10.0
        * np.log10(
            df_paired[
                "backward_to_forward_integral_ratio"
            ]
        )
    )


# ============================================================
# BACKWARD-COLOUR PERFORMANCE RELATIVE TO REFERENCE
# ============================================================

if {
    "C_backward",
    "C_backward_reference",
}.issubset(
    df_paired.columns
):

    df_paired[
        "delta_C_backward"
    ] = (
        pd.to_numeric(
            df_paired[
                "C_backward"
            ],
            errors="coerce",
        )
        - pd.to_numeric(
            df_paired[
                "C_backward_reference"
            ],
            errors="coerce",
        )
    )


if {
    "L_backward",
    "L_backward_reference",
}.issubset(
    df_paired.columns
):

    df_paired[
        "delta_L_backward"
    ] = (
        pd.to_numeric(
            df_paired[
                "L_backward"
            ],
            errors="coerce",
        )
        - pd.to_numeric(
            df_paired[
                "L_backward_reference"
            ],
            errors="coerce",
        )
    )


if {
    "hue_deg_backward",
    "hue_deg_backward_reference",
}.issubset(
    df_paired.columns
):

    df_paired[
        "delta_hue_backward_deg"
    ] = circular_difference_deg(
        df_paired[
            "hue_deg_backward"
        ],
        df_paired[
            "hue_deg_backward_reference"
        ],
    )


if {
    "L_backward",
    "a_backward",
    "b_backward",
    "L_backward_reference",
    "a_backward_reference",
    "b_backward_reference",
}.issubset(
    df_paired.columns
):

    df_paired[
        "delta_E_ab_backward"
    ] = np.sqrt(
        (
            pd.to_numeric(
                df_paired[
                    "L_backward"
                ],
                errors="coerce",
            )
            - pd.to_numeric(
                df_paired[
                    "L_backward_reference"
                ],
                errors="coerce",
            )
        )
        ** 2
        + (
            pd.to_numeric(
                df_paired[
                    "a_backward"
                ],
                errors="coerce",
            )
            - pd.to_numeric(
                df_paired[
                    "a_backward_reference"
                ],
                errors="coerce",
            )
        )
        ** 2
        + (
            pd.to_numeric(
                df_paired[
                    "b_backward"
                ],
                errors="coerce",
            )
            - pd.to_numeric(
                df_paired[
                    "b_backward_reference"
                ],
                errors="coerce",
            )
        )
        ** 2
    )


# Existing normalized colour objective.
if {
    "eta_C_backward",
    "eta_L_backward",
}.issubset(
    df_paired.columns
):

    eta_C_capped = pd.to_numeric(
        df_paired[
            "eta_C_backward"
        ],
        errors="coerce",
    ).clip(
        upper=1.0
    )

    eta_L_capped = pd.to_numeric(
        df_paired[
            "eta_L_backward"
        ],
        errors="coerce",
    ).clip(
        upper=1.0
    )

    df_paired[
        "d_opt_colour_backward"
    ] = np.hypot(
        1.0 - eta_C_capped,
        1.0 - eta_L_capped,
    )


# ============================================================
# REPORT AVAILABLE DERIVED METRICS
# ============================================================

derived_metric_candidates = [
    "backward_to_forward_peak_ratio",
    "backward_to_forward_peak_db",
    "backward_peak_share",
    "forward_peak_retention",
    "forward_peak_suppression_fraction",
    "forward_peak_suppression_db",
    "backward_peak_retention",
    "backward_peak_change_fraction",
    "forward_integral_retention",
    "forward_integral_suppression_fraction",
    "forward_integral_suppression_db",
    "backward_integral_retention",
    "backward_to_forward_integral_ratio",
    "backward_to_forward_integral_db",
    "delta_C_backward",
    "delta_L_backward",
    "delta_hue_backward_deg",
    "delta_E_ab_backward",
    "d_opt_colour_backward",
]

derived_metric_columns = [
    column
    for column in derived_metric_candidates
    if column in df_paired.columns
]


derived_summary_records = []

for column in derived_metric_columns:

    values = pd.to_numeric(
        df_paired[column],
        errors="coerce",
    )

    finite_values = values.loc[
        np.isfinite(values)
    ]

    if finite_values.empty:
        continue

    derived_summary_records.append(
        {
            "metric": column,
            "finite_values": len(
                finite_values
            ),
            "minimum": float(
                finite_values.min()
            ),
            "median": float(
                finite_values.median()
            ),
            "mean": float(
                finite_values.mean()
            ),
            "maximum": float(
                finite_values.max()
            ),
        }
    )


df_derived_metric_summary = pd.DataFrame(
    derived_summary_records
)


print(
    "\nDerived scalar metrics:"
)

display(
    df_derived_metric_summary.style.format(
        {
            "minimum": "{:.5g}",
            "median": "{:.5g}",
            "mean": "{:.5g}",
            "maximum": "{:.5g}",
        }
    )
)


# ============================================================
# DISPLAY THE NEW NONABSORBING REFERENCES
# ============================================================

reference_display_columns = [
    column
    for column in [
        "reference_design_wavelength_nm",
        "n_A",
        "n_B",
        "reference_n_shells",
        "reference_diameter_nm",
        "q_collected_geom_max_backward",
        "q_collected_geom_max_forward",
        "spectrum_integral_backward",
        "spectrum_integral_forward",
        "L_backward",
        "C_backward",
        "hue_deg_backward",
        "eta_C_backward",
        "eta_L_backward",
        "d_opt_colour_backward",
        "hex_backward",
    ]
    if column in df_reference_cases.columns
    or column in df_paired.columns
]


print(
    "\nNew nonabsorbing reference cases:"
)

display(
    df_paired.loc[
        reference_mask,
        reference_display_columns,
    ].sort_values(
        "reference_design_wavelength_nm"
    )
)


# ============================================================
# SAVE THE PAIRED ANALYSIS TABLE
# ============================================================

PAIRED_PARQUET_PATH = (
    TABLE_DIR
    / "forward_backward_paired_cases.parquet"
)

PAIRED_CSV_PATH = (
    TABLE_DIR
    / "forward_backward_paired_cases.csv"
)

df_paired.to_parquet(
    PAIRED_PARQUET_PATH,
    index=False,
)

df_paired.to_csv(
    PAIRED_CSV_PATH,
    index=False,
)

df_derived_metric_summary.to_csv(
    TABLE_DIR
    / "derived_scalar_metric_summary.csv",
    index=False,
)


print(
    "\nSaved paired analysis table:"
)

print(
    "  ",
    PAIRED_PARQUET_PATH,
)

print(
    "  ",
    PAIRED_CSV_PATH,
)

# In[73]:


print(df_paired.shape)
print(df_paired.columns.tolist())
display(df_derived_metric_summary)

# In[74]:


# ============================================================
# REFERENCE-RELATIVE DIRECTIONALITY METRICS
#
# Primary quantities:
#
#   Directional ratio:
#       R_BF = Integral(Q_back) / Integral(Q_forward)
#
#   Directionality improvement:
#       G_BF = R_BF / R_BF,reference
#
#   Directionality improvement in dB:
#       Delta S_BF = 10 log10(G_BF)
#
# Positive Delta S_BF means that the backward-to-forward balance
# improved relative to the nonabsorbing reference.
#
# This does not imply that scattering became backward dominant.
# ============================================================

required_columns = {
    "reference_design_wavelength_nm",
    "spectrum_integral_backward",
    "spectrum_integral_forward",
    "backward_integral_retention",
    "forward_integral_retention",
}

missing_columns = (
    required_columns
    - set(df_paired.columns)
)

if missing_columns:
    raise KeyError(
        "Missing columns required for integrated directionality "
        f"analysis: {sorted(missing_columns)}"
    )


# ------------------------------------------------------------
# Directional ratio for every physical case
# ------------------------------------------------------------

df_paired[
    "backward_to_forward_integral_ratio"
] = safe_ratio(
    df_paired[
        "spectrum_integral_backward"
    ],
    df_paired[
        "spectrum_integral_forward"
    ],
)


# ------------------------------------------------------------
# Extract one ratio for each nonabsorbing reference
# ------------------------------------------------------------

reference_ratio_table = (
    df_paired.loc[
        reference_mask,
        [
            "reference_design_wavelength_nm",
            "backward_to_forward_integral_ratio",
        ],
    ]
    .rename(
        columns={
            "backward_to_forward_integral_ratio":
                "backward_to_forward_integral_ratio_reference",
        }
    )
    .copy()
)

reference_ratio_counts = (
    reference_ratio_table[
        "reference_design_wavelength_nm"
    ]
    .value_counts()
)

if not (
    reference_ratio_counts == 1
).all():
    raise ValueError(
        "Expected exactly one nonabsorbing directional-ratio "
        "reference for each design wavelength."
    )


# Remove an old merged reference column if this cell is rerun.
column_to_replace = (
    "backward_to_forward_integral_ratio_reference"
)

if column_to_replace in df_paired.columns:
    df_paired = df_paired.drop(
        columns=[
            column_to_replace,
        ]
    )


df_paired = df_paired.merge(
    reference_ratio_table,
    on="reference_design_wavelength_nm",
    how="left",
    validate="many_to_one",
)


# ------------------------------------------------------------
# Improvement relative to the nonabsorbing reference
# ------------------------------------------------------------

df_paired[
    "directionality_improvement_factor"
] = safe_ratio(
    df_paired[
        "backward_to_forward_integral_ratio"
    ],
    df_paired[
        "backward_to_forward_integral_ratio_reference"
    ],
)

df_paired[
    "directionality_improvement_percent"
] = (
    100.0
    * (
        df_paired[
            "directionality_improvement_factor"
        ]
        - 1.0
    )
)

df_paired[
    "directionality_improvement_db"
] = (
    10.0
    * np.log10(
        df_paired[
            "directionality_improvement_factor"
        ]
    )
)


# ------------------------------------------------------------
# Directionality change can also be calculated directly from
# retentions:
#
#    G_BF = backward retention / forward retention
#
# This should agree numerically with the ratio-based definition.
# ------------------------------------------------------------

df_paired[
    "directionality_improvement_from_retentions"
] = safe_ratio(
    df_paired[
        "backward_integral_retention"
    ],
    df_paired[
        "forward_integral_retention"
    ],
)

df_paired[
    "directionality_improvement_consistency_error"
] = (
    df_paired[
        "directionality_improvement_factor"
    ]
    - df_paired[
        "directionality_improvement_from_retentions"
    ]
)


maximum_consistency_error = float(
    np.nanmax(
        np.abs(
            df_paired[
                "directionality_improvement_consistency_error"
            ]
        )
    )
)

print(
    "Maximum consistency difference:",
    maximum_consistency_error,
)

if maximum_consistency_error > 1e-8:
    warnings.warn(
        "The two directionality-improvement definitions differ "
        "by more than expected. Inspect the reference pairing."
    )


# ------------------------------------------------------------
# Convenient percentage metrics
# ------------------------------------------------------------

df_paired[
    "forward_integral_suppression_percent"
] = (
    100.0
    * (
        1.0
        - df_paired[
            "forward_integral_retention"
        ]
    )
)

df_paired[
    "backward_integral_retention_percent"
] = (
    100.0
    * df_paired[
        "backward_integral_retention"
    ]
)

df_paired[
    "backward_integral_loss_percent"
] = (
    100.0
    * (
        1.0
        - df_paired[
            "backward_integral_retention"
        ]
    )
)


# ------------------------------------------------------------
# Inspect reference directionality
# ------------------------------------------------------------

reference_directionality_columns = [
    "reference_design_wavelength_nm",
    "spectrum_integral_backward",
    "spectrum_integral_forward",
    "backward_to_forward_integral_ratio",
    "backward_to_forward_integral_db",
    "L_backward",
    "C_backward",
    "hue_deg_backward",
    "eta_C_backward",
    "eta_L_backward",
    "d_opt_colour_backward",
]

reference_directionality_columns = [
    column
    for column in reference_directionality_columns
    if column in df_paired.columns
]

print(
    "\nNonabsorbing reference directionality:"
)

display(
    df_paired.loc[
        reference_mask,
        reference_directionality_columns,
    ]
    .sort_values(
        "reference_design_wavelength_nm"
    )
    .style.format(
        {
            "spectrum_integral_backward": "{:.4g}",
            "spectrum_integral_forward": "{:.4g}",
            "backward_to_forward_integral_ratio": "{:.4f}",
            "backward_to_forward_integral_db": "{:.2f}",
            "L_backward": "{:.2f}",
            "C_backward": "{:.2f}",
            "hue_deg_backward": "{:.2f}",
            "eta_C_backward": "{:.3f}",
            "eta_L_backward": "{:.3f}",
            "d_opt_colour_backward": "{:.3f}",
        }
    )
)

# In[75]:


# ============================================================
# INTERPRETABLE PERFORMANCE CONSTRAINTS
#
# These are analysis thresholds, not universal perceptual limits.
# Adjust them after inspecting the trade-off distributions.
# ============================================================

MIN_FORWARD_SUPPRESSION_PERCENT = 10.0
MIN_BACKWARD_RETENTION_PERCENT = 80.0

MAX_ABS_DELTA_C = 10.0
MAX_ABS_DELTA_L = 10.0
MAX_ABS_DELTA_HUE_DEG = 10.0

# Optional colour-solid performance constraints.
MIN_ETA_C = 0.25
MIN_ETA_L = 0.60


# ------------------------------------------------------------
# Primary flags
# ------------------------------------------------------------

df_paired[
    "improves_directionality"
] = (
    df_paired[
        "directionality_improvement_factor"
    ]
    > 1.0
)

df_paired[
    "suppresses_forward_integral"
] = (
    df_paired[
        "forward_integral_suppression_percent"
    ]
    >= MIN_FORWARD_SUPPRESSION_PERCENT
)

df_paired[
    "retains_backward_integral"
] = (
    df_paired[
        "backward_integral_retention_percent"
    ]
    >= MIN_BACKWARD_RETENTION_PERCENT
)

df_paired[
    "preserves_chroma"
] = (
    df_paired[
        "delta_C_backward"
    ].abs()
    <= MAX_ABS_DELTA_C
)

df_paired[
    "preserves_lightness"
] = (
    df_paired[
        "delta_L_backward"
    ].abs()
    <= MAX_ABS_DELTA_L
)

df_paired[
    "preserves_hue"
] = (
    df_paired[
        "delta_hue_backward_deg"
    ].abs()
    <= MAX_ABS_DELTA_HUE_DEG
)


# ------------------------------------------------------------
# Absolute colour-performance flags
# ------------------------------------------------------------

df_paired[
    "acceptable_eta_C"
] = (
    pd.to_numeric(
        df_paired[
            "eta_C_backward"
        ],
        errors="coerce",
    )
    >= MIN_ETA_C
)

df_paired[
    "acceptable_eta_L"
] = (
    pd.to_numeric(
        df_paired[
            "eta_L_backward"
        ],
        errors="coerce",
    )
    >= MIN_ETA_L
)


# ------------------------------------------------------------
# Combined transparent selection
# ------------------------------------------------------------

df_paired[
    "passes_primary_tradeoff_constraints"
] = (
    df_paired[
        "improves_directionality"
    ]
    & df_paired[
        "suppresses_forward_integral"
    ]
    & df_paired[
        "retains_backward_integral"
    ]
    & df_paired[
        "preserves_chroma"
    ]
    & df_paired[
        "preserves_lightness"
    ]
    & df_paired[
        "preserves_hue"
    ]
)


constraint_summary = pd.DataFrame(
    {
        "criterion": [
            "Improves backward/forward balance",
            (
                f"Forward suppression >= "
                f"{MIN_FORWARD_SUPPRESSION_PERCENT:.0f}%"
            ),
            (
                f"Backward retention >= "
                f"{MIN_BACKWARD_RETENTION_PERCENT:.0f}%"
            ),
            (
                f"|Delta C*| <= "
                f"{MAX_ABS_DELTA_C:.0f}"
            ),
            (
                f"|Delta L*| <= "
                f"{MAX_ABS_DELTA_L:.0f}"
            ),
            (
                f"|Delta hue| <= "
                f"{MAX_ABS_DELTA_HUE_DEG:.0f} degrees"
            ),
            "Passes all primary constraints",
        ],
        "number_of_cases": [
            int(
                df_paired[
                    "improves_directionality"
                ].sum()
            ),
            int(
                df_paired[
                    "suppresses_forward_integral"
                ].sum()
            ),
            int(
                df_paired[
                    "retains_backward_integral"
                ].sum()
            ),
            int(
                df_paired[
                    "preserves_chroma"
                ].sum()
            ),
            int(
                df_paired[
                    "preserves_lightness"
                ].sum()
            ),
            int(
                df_paired[
                    "preserves_hue"
                ].sum()
            ),
            int(
                df_paired[
                    "passes_primary_tradeoff_constraints"
                ].sum()
            ),
        ],
    }
)

constraint_summary[
    "fraction_of_cases"
] = (
    constraint_summary[
        "number_of_cases"
    ]
    / len(df_paired)
)

display(
    constraint_summary.style.format(
        {
            "fraction_of_cases": "{:.1%}",
        }
    )
)

# Cell 1: Add balanced-directionality and forward/backward hue metrics

# In[76]:


# ============================================================
# BALANCED FORWARD/BACKWARD SCATTERING AND HUE METRICS
#
# The objective is not complete suppression of forward scattering.
# Instead, favourable cases should have:
#
#   1. comparable integrated forward and backward scattering;
#   2. similar forward and backward hues;
#   3. substantial retained backward scattering;
#   4. acceptable backward chroma and lightness.
# ============================================================

import numpy as np
import pandas as pd


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
            - np.asarray(
                reference_deg,
                dtype=float,
            )
            + 180.0
        )
        % 360.0
        - 180.0
    )


# ------------------------------------------------------------
# Verify required columns
# ------------------------------------------------------------

required_balance_columns = {
    "spectrum_integral_backward",
    "spectrum_integral_forward",
    "hue_deg_backward",
    "hue_deg_forward",
    "backward_integral_retention",
}

missing_balance_columns = (
    required_balance_columns
    - set(df_paired.columns)
)

if missing_balance_columns:

    raise KeyError(
        "Missing columns required for the balanced-scattering "
        f"analysis: {sorted(missing_balance_columns)}"
    )


# ------------------------------------------------------------
# Integrated forward/backward scattering balance
# ------------------------------------------------------------

df_paired[
    "backward_to_forward_integral_ratio"
] = safe_ratio(
    df_paired[
        "spectrum_integral_backward"
    ],
    df_paired[
        "spectrum_integral_forward"
    ],
)


# Signed ratio in dB:
#
#   0 dB  -> equal integrated scattering
#   <0 dB -> forward-dominant
#   >0 dB -> backward-dominant
df_paired[
    "backward_forward_integral_db"
] = (
    10.0
    * np.log10(
        df_paired[
            "backward_to_forward_integral_ratio"
        ]
    )
)


# Absolute distance from equal forward/backward scattering.
# Smaller is better.
df_paired[
    "integral_directional_imbalance_db"
] = (
    df_paired[
        "backward_forward_integral_db"
    ].abs()
)


# Symmetric bounded balance score:
#
#   1.0 -> equal forward and backward integrals
#   0.0 -> strongly imbalanced
#
# min(B/F, F/B) avoids assigning preference to either direction.
df_paired[
    "integral_directional_balance_score"
] = np.minimum(
    df_paired[
        "backward_to_forward_integral_ratio"
    ],
    safe_ratio(
        df_paired[
            "spectrum_integral_forward"
        ],
        df_paired[
            "spectrum_integral_backward"
        ],
    ),
)


# ------------------------------------------------------------
# Forward/backward colour agreement
# ------------------------------------------------------------

df_paired[
    "forward_minus_backward_hue_deg"
] = (
    circular_difference_deg(
        df_paired[
            "hue_deg_forward"
        ],
        df_paired[
            "hue_deg_backward"
        ],
    )
)

df_paired[
    "absolute_forward_backward_hue_difference_deg"
] = (
    df_paired[
        "forward_minus_backward_hue_deg"
    ].abs()
)


# Optional complete CIELAB difference between directions.
if {
    "L_forward",
    "a_forward",
    "b_forward",
    "L_backward",
    "a_backward",
    "b_backward",
}.issubset(
    df_paired.columns
):

    df_paired[
        "delta_E_ab_forward_backward"
    ] = np.sqrt(
        (
            pd.to_numeric(
                df_paired["L_forward"],
                errors="coerce",
            )
            - pd.to_numeric(
                df_paired["L_backward"],
                errors="coerce",
            )
        )
        ** 2
        + (
            pd.to_numeric(
                df_paired["a_forward"],
                errors="coerce",
            )
            - pd.to_numeric(
                df_paired["a_backward"],
                errors="coerce",
            )
        )
        ** 2
        + (
            pd.to_numeric(
                df_paired["b_forward"],
                errors="coerce",
            )
            - pd.to_numeric(
                df_paired["b_backward"],
                errors="coerce",
            )
        )
        ** 2
    )


# ------------------------------------------------------------
# Percentage form of backward retention
# ------------------------------------------------------------

df_paired[
    "backward_integral_retention_percent"
] = (
    100.0
    * pd.to_numeric(
        df_paired[
            "backward_integral_retention"
        ],
        errors="coerce",
    )
)


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

balance_metric_columns = [
    "backward_to_forward_integral_ratio",
    "backward_forward_integral_db",
    "integral_directional_imbalance_db",
    "integral_directional_balance_score",
    "absolute_forward_backward_hue_difference_deg",
    "delta_E_ab_forward_backward",
    "backward_integral_retention_percent",
]

balance_metric_columns = [
    column
    for column in balance_metric_columns
    if column in df_paired.columns
]

balance_summary_records = []

for column in balance_metric_columns:

    values = pd.to_numeric(
        df_paired[column],
        errors="coerce",
    )

    values = values.loc[
        np.isfinite(values)
    ]

    if values.empty:
        continue

    balance_summary_records.append(
        {
            "metric": column,
            "minimum": values.min(),
            "median": values.median(),
            "mean": values.mean(),
            "maximum": values.max(),
        }
    )


df_balance_summary = pd.DataFrame(
    balance_summary_records
)

display(
    df_balance_summary.style.format(
        {
            "minimum": "{:.4g}",
            "median": "{:.4g}",
            "mean": "{:.4g}",
            "maximum": "{:.4g}",
        }
    )
)

# Cell 2: Define meaningful constraint categories
#
# The previous requirement of at least 10%10\%10% forward suppression is no longer central. Replace it with ranges of directional comparability.

# In[77]:


# ============================================================
# SOFTENED, REFERENCE-RELATIVE SCREENING CRITERIA
#
# These criteria are chosen to identify improvement within the
# calculated design space. They are operational screening limits,
# not universal perceptual thresholds.
# ============================================================

MIN_DIRECTIONAL_IMPROVEMENT_DB = 1.0

MAX_FORWARD_BACKWARD_HUE_DIFFERENCE_DEG = 60.0

MIN_BACKWARD_RETENTION_PERCENT = 70.0

MAX_ABS_DELTA_C_BACKWARD = 10.0
MAX_ABS_DELTA_L_BACKWARD = 10.0
MAX_ABS_DELTA_HUE_BACKWARD_DEG = 15.0


# ------------------------------------------------------------
# Exclude nonabsorbing references from absorber optimization
# ------------------------------------------------------------

df_paired[
    "is_nonabsorbing_reference"
] = (
    df_paired[
        "case_family"
    ]
    .astype(str)
    .str.strip()
    .str.lower()
    .eq("reference")
)


# ------------------------------------------------------------
# Criteria
# ------------------------------------------------------------

df_paired[
    "improves_directionality_by_required_amount"
] = (
    pd.to_numeric(
        df_paired[
            "directionality_improvement_db"
        ],
        errors="coerce",
    )
    >= MIN_DIRECTIONAL_IMPROVEMENT_DB
)

df_paired[
    "similar_forward_backward_hue"
] = (
    pd.to_numeric(
        df_paired[
            "absolute_forward_backward_hue_difference_deg"
        ],
        errors="coerce",
    )
    <= MAX_FORWARD_BACKWARD_HUE_DIFFERENCE_DEG
)

df_paired[
    "sufficient_backward_retention"
] = (
    pd.to_numeric(
        df_paired[
            "backward_integral_retention_percent"
        ],
        errors="coerce",
    )
    >= MIN_BACKWARD_RETENTION_PERCENT
)

df_paired[
    "preserves_backward_chroma"
] = (
    pd.to_numeric(
        df_paired[
            "delta_C_backward"
        ],
        errors="coerce",
    ).abs()
    <= MAX_ABS_DELTA_C_BACKWARD
)

df_paired[
    "preserves_backward_lightness"
] = (
    pd.to_numeric(
        df_paired[
            "delta_L_backward"
        ],
        errors="coerce",
    ).abs()
    <= MAX_ABS_DELTA_L_BACKWARD
)

df_paired[
    "preserves_backward_hue"
] = (
    pd.to_numeric(
        df_paired[
            "delta_hue_backward_deg"
        ],
        errors="coerce",
    ).abs()
    <= MAX_ABS_DELTA_HUE_BACKWARD_DEG
)


df_paired[
    "passes_softened_tradeoff_constraints"
] = (
    ~df_paired[
        "is_nonabsorbing_reference"
    ]
    & df_paired[
        "improves_directionality_by_required_amount"
    ]
    & df_paired[
        "similar_forward_backward_hue"
    ]
    & df_paired[
        "sufficient_backward_retention"
    ]
    & df_paired[
        "preserves_backward_chroma"
    ]
    & df_paired[
        "preserves_backward_lightness"
    ]
    & df_paired[
        "preserves_backward_hue"
    ]
)


soft_constraint_definitions = {
    (
        f"Directionality improvement >= "
        f"{MIN_DIRECTIONAL_IMPROVEMENT_DB:.1f} dB"
    ): "improves_directionality_by_required_amount",

    (
        f"Forward/backward hue difference <= "
        f"{MAX_FORWARD_BACKWARD_HUE_DIFFERENCE_DEG:.0f} degrees"
    ): "similar_forward_backward_hue",

    (
        f"Backward retention >= "
        f"{MIN_BACKWARD_RETENTION_PERCENT:.0f}%"
    ): "sufficient_backward_retention",

    (
        f"|Delta C backward| <= "
        f"{MAX_ABS_DELTA_C_BACKWARD:.0f}"
    ): "preserves_backward_chroma",

    (
        f"|Delta L backward| <= "
        f"{MAX_ABS_DELTA_L_BACKWARD:.0f}"
    ): "preserves_backward_lightness",

    (
        f"|Delta hue backward| <= "
        f"{MAX_ABS_DELTA_HUE_BACKWARD_DEG:.0f} degrees"
    ): "preserves_backward_hue",

    "Passes all softened trade-off constraints":
        "passes_softened_tradeoff_constraints",
}


soft_constraint_records = []

for criterion, column in (
    soft_constraint_definitions.items()
):

    number_of_cases = int(
        df_paired[column].sum()
    )

    soft_constraint_records.append(
        {
            "criterion": criterion,
            "number_of_cases": number_of_cases,
            "fraction_of_cases": (
                number_of_cases
                / len(df_paired)
            ),
        }
    )


df_soft_constraint_summary = pd.DataFrame(
    soft_constraint_records
)

display(
    df_soft_constraint_summary.style.format(
        {
            "fraction_of_cases": "{:.1%}",
        }
    )
)

# In[78]:


# ============================================================
# PLOTTING-DATA DIAGNOSTIC
# ============================================================

df_paired[
    "case_family_clean"
] = (
    df_paired[
        "case_family"
    ]
    .astype(str)
    .str.strip()
    .str.lower()
    .str.replace(
        " ",
        "_",
        regex=False,
    )
    .str.replace(
        "-",
        "_",
        regex=False,
    )
)


print("Case-family values:")

display(
    df_paired[
        "case_family_clean"
    ]
    .value_counts(
        dropna=False
    )
    .rename_axis(
        "case_family_clean"
    )
    .reset_index(
        name="number_of_cases"
    )
)


plot_columns = [
    "backward_forward_integral_db",
    "directionality_improvement_db",
    "backward_integral_retention_percent",
    "absolute_forward_backward_hue_difference_deg",
    "delta_C_backward",
    "delta_L_backward",
]

finite_records = []

for wavelength_nm in sorted(
    df_paired[
        "reference_design_wavelength_nm"
    ]
    .dropna()
    .astype(int)
    .unique()
):

    wavelength_mask = np.isclose(
        pd.to_numeric(
            df_paired[
                "reference_design_wavelength_nm"
            ],
            errors="coerce",
        ),
        wavelength_nm,
    )

    row = {
        "reference_design_wavelength_nm":
            wavelength_nm,
        "total_cases": int(
            wavelength_mask.sum()
        ),
    }

    for column in plot_columns:

        values = pd.to_numeric(
            df_paired.loc[
                wavelength_mask,
                column,
            ],
            errors="coerce",
        )

        row[
            f"finite_{column}"
        ] = int(
            np.isfinite(values).sum()
        )

    finite_records.append(row)


df_plot_diagnostic = pd.DataFrame(
    finite_records
)

display(df_plot_diagnostic)

# 3. Complete corrected overview figure
#
# This version deliberately avoids fragile categorical filtering. It converts all plotting columns to numeric, plots every finite case, and uses normalized case-family labels.
#
# The horizontal axis is now directionality improvement relative to the nonabsorbing reference, rather than the absolute backward-to-forward ratio. Thus:
#
# 0 dB0\ \mathrm{dB}0 dB: no change relative to the reference;
# positive: the backward-to-forward balance improved;
# negative: the balance worsened.

# In[79]:


# ============================================================
# OVERVIEW: DIRECTIONALITY IMPROVEMENT VERSUS COLOUR PERFORMANCE
#
# Horizontal coordinate:
#   improvement of integrated backward/forward balance relative
#   to the corresponding nonabsorbing reference
#
#   0 dB  -> unchanged directional balance
#   >0 dB -> improved backward/forward balance
#   <0 dB -> worsened backward/forward balance
#
# Rows:
#   380, 580, and 760 nm reference designs
#
# Columns:
#   backward-integral retention
#   forward/backward hue difference
#   backward-colour change
# ============================================================

import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D


# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------

REFERENCE_WAVELENGTHS = sorted(
    pd.to_numeric(
        df_paired[
            "reference_design_wavelength_nm"
        ],
        errors="coerce",
    )
    .dropna()
    .astype(int)
    .unique()
    .tolist()
)

CASE_FAMILY_MARKERS = {
    "reference": "X",
    "low_index": "o",
    "all_layers": "s",
    "outer_shell": "^",
    "core": "D",
}

CASE_FAMILY_LABELS = {
    "reference": "Nonabsorbing reference",
    "low_index": "Low-index domains",
    "all_layers": "All layers",
    "outer_shell": "Additional outer shell",
    "core": "Core",
}

OVERVIEW_WIDTH_CM = 17.5
OVERVIEW_HEIGHT_CM = 12.5

SAVE_DIRECTIONAL_OVERVIEW = True

DIRECTIONAL_OVERVIEW_BASENAME = (
    "directionality_improvement_colour_overview"
)


# ------------------------------------------------------------
# Prepare a clean plotting dataframe
# ------------------------------------------------------------

df_overview = df_paired.copy()

df_overview[
    "case_family_clean"
] = (
    df_overview[
        "case_family"
    ]
    .astype(str)
    .str.strip()
    .str.lower()
    .str.replace(
        " ",
        "_",
        regex=False,
    )
    .str.replace(
        "-",
        "_",
        regex=False,
    )
)


numeric_plot_columns = [
    "reference_design_wavelength_nm",
    "absorber_k",
    "directionality_improvement_db",
    "backward_integral_retention_percent",
    "absolute_forward_backward_hue_difference_deg",
    "delta_C_backward",
]

for column in numeric_plot_columns:

    df_overview[column] = pd.to_numeric(
        df_overview[column],
        errors="coerce",
    )


required_overview_columns = {
    "reference_design_wavelength_nm",
    "case_family_clean",
    "absorber_k",
    "directionality_improvement_db",
    "backward_integral_retention_percent",
    "absolute_forward_backward_hue_difference_deg",
    "delta_C_backward",
}

missing_overview_columns = (
    required_overview_columns
    - set(df_overview.columns)
)

if missing_overview_columns:

    raise KeyError(
        "Missing overview columns: "
        f"{sorted(missing_overview_columns)}"
    )


# ------------------------------------------------------------
# Verify finite values before generating axes
# ------------------------------------------------------------

complete_plot_mask = (
    np.isfinite(
        df_overview[
            "directionality_improvement_db"
        ]
    )
    & np.isfinite(
        df_overview[
            "backward_integral_retention_percent"
        ]
    )
    & np.isfinite(
        df_overview[
            "absolute_forward_backward_hue_difference_deg"
        ]
    )
    & np.isfinite(
        df_overview[
            "delta_C_backward"
        ]
    )
)

print(
    "Cases with complete overview data:",
    int(complete_plot_mask.sum()),
    "of",
    len(df_overview),
)

if complete_plot_mask.sum() == 0:

    raise ValueError(
        "No finite rows are available for the overview plot."
    )


# ------------------------------------------------------------
# Extinction-coefficient colour normalization
# ------------------------------------------------------------

positive_k = df_overview.loc[
    np.isfinite(
        df_overview["absorber_k"]
    )
    & (
        df_overview["absorber_k"]
        > 0.0
    ),
    "absorber_k",
]

if positive_k.empty:

    k_norm = None

else:

    k_norm = mpl.colors.LogNorm(
        vmin=float(
            positive_k.min()
        ),
        vmax=float(
            positive_k.max()
        ),
    )

k_cmap = plt.get_cmap(
    "viridis"
)


# ------------------------------------------------------------
# Create axes
# ------------------------------------------------------------

cm_to_inches = 1.0 / 2.54

fig, axes = plt.subplots(
    nrows=len(
        REFERENCE_WAVELENGTHS
    ),
    ncols=3,
    figsize=(
        OVERVIEW_WIDTH_CM
        * cm_to_inches,
        OVERVIEW_HEIGHT_CM
        * cm_to_inches,
    ),
    sharex=True,
    constrained_layout=True,
)

axes = np.asarray(
    axes,
    dtype=object,
).reshape(
    len(
        REFERENCE_WAVELENGTHS
    ),
    3,
)


y_definitions = [
    {
        "column": (
            "backward_integral_retention_percent"
        ),
        "ylabel": (
            "Backward-integral\nretention [%]"
        ),
        "reference_value": 100.0,
        "criterion_value":
            MIN_BACKWARD_RETENTION_PERCENT,
    },
    {
        "column": (
            "absolute_forward_backward_hue_difference_deg"
        ),
        "ylabel": (
            r"$|\Delta h_{\mathrm{F-B}}|$ [$^\circ$]"
        ),
        "reference_value": None,
        "criterion_value": (
            MAX_FORWARD_BACKWARD_HUE_DIFFERENCE_DEG
        ),
    },
    {
        "column": "delta_C_backward",
        "ylabel": (
            r"$\Delta C_{ab,\mathrm{back}}^*$"
        ),
        "reference_value": 0.0,
        "criterion_value": None,
    },
]


# ------------------------------------------------------------
# Plot every wavelength and family
# ------------------------------------------------------------

number_of_scatter_points = 0

for row_index, wavelength_nm in enumerate(
    REFERENCE_WAVELENGTHS
):

    wavelength_mask = np.isclose(
        df_overview[
            "reference_design_wavelength_nm"
        ],
        float(wavelength_nm),
    )

    wavelength_data = (
        df_overview.loc[
            wavelength_mask
        ]
        .copy()
    )

    print(
        f"{wavelength_nm} nm:",
        len(wavelength_data),
        "cases",
    )

    for column_index, definition in enumerate(
        y_definitions
    ):

        ax = axes[
            row_index,
            column_index,
        ]

        # No change relative to the nonabsorbing reference.
        ax.axvline(
            0.0,
            color="black",
            linewidth=0.8,
            linestyle="--",
            zorder=0,
        )

        # Desired directionality improvement.
        ax.axvline(
            MIN_DIRECTIONAL_IMPROVEMENT_DB,
            color="0.55",
            linewidth=0.7,
            linestyle=":",
            zorder=0,
        )

        if (
            definition[
                "reference_value"
            ]
            is not None
        ):

            ax.axhline(
                definition[
                    "reference_value"
                ],
                color="0.55",
                linewidth=0.7,
                linestyle="--",
                zorder=0,
            )

        if (
            definition[
                "criterion_value"
            ]
            is not None
        ):

            ax.axhline(
                definition[
                    "criterion_value"
                ],
                color="0.65",
                linewidth=0.7,
                linestyle=":",
                zorder=0,
            )

        for case_family, marker in (
            CASE_FAMILY_MARKERS.items()
        ):

            family_mask = (
                wavelength_data[
                    "case_family_clean"
                ]
                == case_family
            )

            family_data = (
                wavelength_data.loc[
                    family_mask
                ]
                .copy()
            )

            if family_data.empty:
                continue

            x_values = family_data[
                "directionality_improvement_db"
            ]

            y_values = family_data[
                definition["column"]
            ]

            finite_mask = (
                np.isfinite(x_values)
                & np.isfinite(y_values)
            )

            family_data = family_data.loc[
                finite_mask
            ]

            x_values = x_values.loc[
                finite_mask
            ]

            y_values = y_values.loc[
                finite_mask
            ]

            if family_data.empty:
                continue

            number_of_scatter_points += len(
                family_data
            )

            if case_family == "reference":

                point_colours = [
                    "black"
                ] * len(family_data)

                point_sizes = np.full(
                    len(family_data),
                    55.0,
                )

                point_alpha = 1.0

            else:

                k_values = family_data[
                    "absorber_k"
                ]

                point_colours = [
                    (
                        k_cmap(
                            k_norm(
                                float(k_value)
                            )
                        )
                        if (
                            k_norm is not None
                            and np.isfinite(
                                k_value
                            )
                            and k_value > 0.0
                        )
                        else "0.65"
                    )
                    for k_value in k_values
                ]

                point_sizes = np.full(
                    len(family_data),
                    30.0,
                )

                point_alpha = 0.82

            ax.scatter(
                x_values,
                y_values,
                s=point_sizes,
                c=point_colours,
                marker=marker,
                edgecolors="black",
                linewidths=0.4,
                alpha=point_alpha,
                zorder=3,
            )

        if column_index == 0:

            ax.set_ylabel(
                (
                    f"{wavelength_nm} nm\n"
                    + definition[
                        "ylabel"
                    ]
                ),
                fontsize=8,
            )

        else:

            ax.set_ylabel(
                definition[
                    "ylabel"
                ],
                fontsize=8,
            )

        if row_index == (
            len(
                REFERENCE_WAVELENGTHS
            )
            - 1
        ):

            ax.set_xlabel(
                (
                    "Directionality improvement\n"
                    "relative to reference [dB]"
                ),
                fontsize=8,
            )

        ax.grid(
            True,
            color="0.90",
            linewidth=0.5,
            zorder=0,
        )


if number_of_scatter_points == 0:

    plt.close(fig)

    raise ValueError(
        "No scatter points were added. "
        "Inspect case_family_clean and numeric plotting columns."
    )

print(
    "Total scatter points added across all axes:",
    number_of_scatter_points,
)


# ------------------------------------------------------------
# Column titles
# ------------------------------------------------------------

axes[0, 0].set_title(
    "Useful backward\nscattering",
    fontsize=9,
)

axes[0, 1].set_title(
    "Forward/backward\nhue agreement",
    fontsize=9,
)

axes[0, 2].set_title(
    "Backward chroma\npreservation",
    fontsize=9,
)


# ------------------------------------------------------------
# Family legend
# ------------------------------------------------------------

family_handles = [
    Line2D(
        [0],
        [0],
        marker=marker,
        linestyle="None",
        markerfacecolor="white",
        markeredgecolor="black",
        markeredgewidth=0.5,
        markersize=5.5,
        label=CASE_FAMILY_LABELS[
            case_family
        ],
    )
    for case_family, marker
    in CASE_FAMILY_MARKERS.items()
]

family_legend = fig.legend(
    handles=family_handles,
    loc="outside upper center",
    ncol=3,
    frameon=True,
    fontsize=7,
)


# ------------------------------------------------------------
# Extinction-coefficient colour bar
# ------------------------------------------------------------

if k_norm is not None:

    scalar_mappable = (
        mpl.cm.ScalarMappable(
            norm=k_norm,
            cmap=k_cmap,
        )
    )

    scalar_mappable.set_array([])

    colourbar = fig.colorbar(
        scalar_mappable,
        ax=axes,
        location="right",
        shrink=0.82,
        pad=0.015,
    )

    colourbar.set_label(
        r"Extinction coefficient $k$",
        fontsize=8,
    )


# ------------------------------------------------------------
# Save and display
# ------------------------------------------------------------

if SAVE_DIRECTIONAL_OVERVIEW:

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    overview_png = (
        FIGURE_DIR
        / (
            DIRECTIONAL_OVERVIEW_BASENAME
            + ".png"
        )
    )

    overview_pdf = (
        FIGURE_DIR
        / (
            DIRECTIONAL_OVERVIEW_BASENAME
            + ".pdf"
        )
    )

    fig.canvas.draw()

    fig.savefig(
        overview_png,
        dpi=PNG_DPI,
        transparent=True,
        bbox_inches="tight",
    )

    fig.savefig(
        overview_pdf,
        transparent=True,
        bbox_inches="tight",
    )

    print("Saved:", overview_png)
    print("Saved:", overview_pdf)


plt.show()

# Cell 1: Load full checkpoints and inspect their structure

# In[80]:


# ============================================================
# LOAD FULL FORWARD-SUPPRESSION CHECKPOINTS
#
# Each checkpoint is expected to contain:
#
# {
#     "task": task_dictionary,
#     "backward": backward_sweep_result,
#     "forward": forward_sweep_result,
#     "compact": compact_dataframe,
# }
# ============================================================

from pathlib import Path
import pickle
import re
import warnings

import numpy as np
import pandas as pd

from IPython.display import display


FULL_CHECKPOINT_DIRECTORY = (
    OUTPUT_DIR
    / "full_checkpoints"
)

full_checkpoint_paths = sorted(
    FULL_CHECKPOINT_DIRECTORY.glob(
        "task_*.pkl"
    )
)

if not full_checkpoint_paths:
    raise FileNotFoundError(
        "No full checkpoint files were found in:\n"
        f"{FULL_CHECKPOINT_DIRECTORY}"
    )

print(
    "Full checkpoints:",
    len(full_checkpoint_paths),
)


def load_full_checkpoint(path):
    """
    Load one complete forward-suppression checkpoint.
    """

    with path.open("rb") as handle:
        payload = pickle.load(handle)

    required_keys = {
        "task",
        "backward",
        "forward",
        "compact",
    }

    missing_keys = (
        required_keys
        - set(payload.keys())
    )

    if missing_keys:
        raise KeyError(
            f"{path.name} is missing keys: "
            f"{sorted(missing_keys)}"
        )

    return payload


example_checkpoint = load_full_checkpoint(
    full_checkpoint_paths[0]
)

print(
    "Top-level checkpoint keys:",
    list(example_checkpoint.keys()),
)

print(
    "\nExample task:"
)

display(
    pd.DataFrame(
        [example_checkpoint["task"]]
    ).T.rename(
        columns={0: "value"}
    )
)

# Cell 2: Robust recursive array extraction helpers

# In[81]:


# ============================================================
# RECURSIVE ARRAY-EXTRACTION HELPERS
#
# These functions inspect dictionaries, objects, lists, and
# dataframes without modifying the stored checkpoint objects.
# ============================================================

def iter_nested_items(
    object_to_inspect,
    *,
    path="root",
    maximum_depth=8,
    current_depth=0,
    visited=None,
):
    """
    Yield nested leaf values together with their access paths.
    """

    if visited is None:
        visited = set()

    if current_depth > maximum_depth:
        return

    object_id = id(
        object_to_inspect
    )

    if object_id in visited:
        return

    visited.add(
        object_id
    )

    # --------------------------------------------------------
    # Dictionary
    # --------------------------------------------------------

    if isinstance(
        object_to_inspect,
        dict,
    ):

        for key, value in (
            object_to_inspect.items()
        ):

            nested_path = (
                f"{path}[{key!r}]"
            )

            yield from iter_nested_items(
                value,
                path=nested_path,
                maximum_depth=maximum_depth,
                current_depth=(
                    current_depth + 1
                ),
                visited=visited,
            )

        return

    # --------------------------------------------------------
    # Pandas dataframe
    # --------------------------------------------------------

    if isinstance(
        object_to_inspect,
        pd.DataFrame,
    ):

        for column in (
            object_to_inspect.columns
        ):

            series = (
                object_to_inspect[
                    column
                ]
            )

            yield (
                f"{path}.dataframe[{column!r}]",
                series,
            )

            for row_index, value in (
                series.items()
            ):

                if isinstance(
                    value,
                    np.ndarray,
                ):

                    yield (
                        (
                            f"{path}.dataframe"
                            f"[{column!r}]"
                            f"[{row_index!r}]"
                        ),
                        value,
                    )

        return

    # --------------------------------------------------------
    # Pandas series
    # --------------------------------------------------------

    if isinstance(
        object_to_inspect,
        pd.Series,
    ):

        yield (
            path,
            object_to_inspect,
        )

        for row_index, value in (
            object_to_inspect.items()
        ):

            if isinstance(
                value,
                np.ndarray,
            ):

                yield (
                    f"{path}[{row_index!r}]",
                    value,
                )

        return

    # --------------------------------------------------------
    # NumPy array
    # --------------------------------------------------------

    if isinstance(
        object_to_inspect,
        np.ndarray,
    ):

        yield (
            path,
            object_to_inspect,
        )

        return

    # --------------------------------------------------------
    # Lists and tuples
    # --------------------------------------------------------

    if isinstance(
        object_to_inspect,
        (list, tuple),
    ):

        for index, value in enumerate(
            object_to_inspect
        ):

            nested_path = (
                f"{path}[{index}]"
            )

            yield from iter_nested_items(
                value,
                path=nested_path,
                maximum_depth=maximum_depth,
                current_depth=(
                    current_depth + 1
                ),
                visited=visited,
            )

        return

    # --------------------------------------------------------
    # Normal Python object
    # --------------------------------------------------------

    if hasattr(
        object_to_inspect,
        "__dict__",
    ):

        for attribute, value in (
            vars(
                object_to_inspect
            ).items()
        ):

            if attribute.startswith(
                "__"
            ):
                continue

            nested_path = (
                f"{path}.{attribute}"
            )

            yield from iter_nested_items(
                value,
                path=nested_path,
                maximum_depth=maximum_depth,
                current_depth=(
                    current_depth + 1
                ),
                visited=visited,
            )


def collect_array_inventory(
    object_to_inspect,
    *,
    root_name="root",
):
    """
    Return a dataframe describing all nested NumPy arrays.
    """

    records = []

    for path, value in iter_nested_items(
        object_to_inspect,
        path=root_name,
    ):

        if not isinstance(
            value,
            np.ndarray,
        ):
            continue

        array = np.asarray(
            value
        )

        record = {
            "path": path,
            "shape": array.shape,
            "ndim": array.ndim,
            "size": array.size,
            "dtype": str(
                array.dtype
            ),
            "minimum": np.nan,
            "maximum": np.nan,
        }

        if (
            array.size > 0
            and np.issubdtype(
                array.dtype,
                np.number,
            )
        ):

            finite = array[
                np.isfinite(array)
            ]

            if finite.size > 0:

                record["minimum"] = float(
                    np.nanmin(finite)
                )

                record["maximum"] = float(
                    np.nanmax(finite)
                )

        records.append(
            record
        )

    return (
        pd.DataFrame(records)
        .sort_values(
            [
                "ndim",
                "size",
                "path",
            ]
        )
        .reset_index(drop=True)
    )


example_backward_inventory = (
    collect_array_inventory(
        example_checkpoint[
            "backward"
        ],
        root_name="backward",
    )
)

example_forward_inventory = (
    collect_array_inventory(
        example_checkpoint[
            "forward"
        ],
        root_name="forward",
    )
)

print(
    "Backward array inventory:"
)

display(
    example_backward_inventory
)

print(
    "Forward array inventory:"
)

display(
    example_forward_inventory
)

# In[82]:


# ============================================================
# REAL NUMERIC ARRAY VALIDATION
# ============================================================

def as_real_numeric_array(
    values,
    *,
    allow_negligible_imaginary=False,
    imaginary_tolerance=1e-12,
):
    """
    Convert an array to float only if the stored quantity is real.

    Complex electromagnetic-field amplitudes are rejected rather
    than silently converted by discarding their imaginary part.

    If allow_negligible_imaginary=True, a complex array is accepted
    only when its imaginary component is negligible relative to its
    real component.
    """

    array = np.asarray(values)

    if not np.issubdtype(
        array.dtype,
        np.number,
    ):
        return None

    if np.iscomplexobj(array):

        if not allow_negligible_imaginary:
            return None

        finite_real = np.real(array)[
            np.isfinite(np.real(array))
        ]

        finite_imaginary = np.imag(array)[
            np.isfinite(np.imag(array))
        ]

        if (
            finite_real.size == 0
            or finite_imaginary.size == 0
        ):
            return None

        real_scale = max(
            float(
                np.nanmax(
                    np.abs(finite_real)
                )
            ),
            1.0,
        )

        maximum_imaginary = float(
            np.nanmax(
                np.abs(finite_imaginary)
            )
        )

        if (
            maximum_imaginary
            > imaginary_tolerance
            * real_scale
        ):
            return None

        array = np.real(array)

    return np.asarray(
        array,
        dtype=float,
    )

# Cell 3: Adaptive spectral and angular-array extractor

# In[83]:


# ============================================================
# ADAPTIVE EXTRACTION OF SPECTRAL AND ANGULAR ARRAYS
# ============================================================

WAVELENGTH_KEYWORDS = [
    "wavelength",
    "lambda",
]

COLLECTED_KEYWORDS = [
    "q_collected_geom",
    "collected_geom",
    "collected",
]

THETA_KEYWORDS = [
    "theta",
    "angle",
]

DIFFERENTIAL_EFFICIENCY_KEYWORDS = [
    "dq",
    "differential",
    "d_q",
    "q_sca_angle",
    "qsca_angle",
    "angular_efficiency",
]

DIFFERENTIAL_CROSS_SECTION_KEYWORDS = [
    "differential_cross",
    "dcsca",
    "d_csca",
    "cross_section_angle",
]


def normalized_path_text(
    path,
):
    return (
        str(path)
        .lower()
        .replace(" ", "_")
    )


def score_array_path(
    path,
    *,
    required_keywords,
    preferred_keywords=None,
    rejected_keywords=None,
):
    """
    Score an array path by keyword matching.
    """

    text = normalized_path_text(
        path
    )

    score = 0

    for keyword in required_keywords:

        if keyword.lower() in text:
            score += 10

    if preferred_keywords:

        for keyword in preferred_keywords:

            if keyword.lower() in text:
                score += 2

    if rejected_keywords:

        for keyword in rejected_keywords:

            if keyword.lower() in text:
                score -= 20

    return score


def nested_arrays(
    object_to_inspect,
    *,
    root_name,
):
    """
    Return all nested NumPy arrays as path-array pairs.
    """

    arrays = []

    for path, value in iter_nested_items(
        object_to_inspect,
        path=root_name,
    ):

        if isinstance(
            value,
            np.ndarray,
        ):

            arrays.append(
                (
                    path,
                    np.asarray(value),
                )
            )

    return arrays


def select_best_1d_array(
    object_to_inspect,
    *,
    root_name,
    required_keywords,
    expected_length=None,
    minimum=None,
    maximum=None,
):
    """
    Select the highest-scoring one-dimensional array.
    """

    candidates = []

    for path, array in nested_arrays(
        object_to_inspect,
        root_name=root_name,
    ):

        squeezed = np.squeeze(
            array
        )

        if squeezed.ndim != 1:
            continue

        if (
            expected_length is not None
            and len(squeezed)
            != expected_length
        ):
            continue

        if not np.issubdtype(
            squeezed.dtype,
            np.number,
        ):
            continue

        finite = squeezed[
            np.isfinite(squeezed)
        ]

        if finite.size == 0:
            continue

        if (
            minimum is not None
            and np.nanmin(finite)
            < minimum
        ):
            continue

        if (
            maximum is not None
            and np.nanmax(finite)
            > maximum
        ):
            continue

        score = score_array_path(
            path,
            required_keywords=(
                required_keywords
            ),
        )

        candidates.append(
            (
                score,
                path,
                squeezed.astype(float),
            )
        )

    if not candidates:
        return None, None

    candidates.sort(
        key=lambda item: (
            item[0],
            len(item[2]),
        ),
        reverse=True,
    )

    score, path, array = (
        candidates[0]
    )

    return array, path


def select_wavelength_array(
    sweep_result,
    *,
    root_name,
):
    """
    Select a real one-dimensional wavelength array and return nm.

    Complex arrays are rejected.
    """

    candidates = []

    for path, array in nested_arrays(
        sweep_result,
        root_name=root_name,
    ):

        real_array = as_real_numeric_array(
            array
        )

        if real_array is None:
            continue

        squeezed = np.squeeze(
            real_array
        )

        if squeezed.ndim != 1:
            continue

        if squeezed.size < 10:
            continue

        finite = squeezed[
            np.isfinite(squeezed)
        ]

        if finite.size == 0:
            continue

        minimum_value = float(
            np.nanmin(finite)
        )

        maximum_value = float(
            np.nanmax(finite)
        )

        score = score_array_path(
            path,
            required_keywords=(
                WAVELENGTH_KEYWORDS
            ),
        )

        # Wavelengths stored in metres.
        if (
            minimum_value > 0.0
            and maximum_value < 1e-3
        ):

            wavelength_nm = (
                squeezed * 1e9
            )

            score += 4

        # Wavelengths already stored in nanometres.
        elif (
            minimum_value >= 100.0
            and maximum_value <= 5000.0
        ):

            wavelength_nm = squeezed

            score += 4

        else:
            continue

        if (
            np.nanmin(wavelength_nm)
            <= 400.0
            and np.nanmax(wavelength_nm)
            >= 750.0
        ):
            score += 4

        candidates.append(
            (
                score,
                path,
                wavelength_nm,
            )
        )

    if not candidates:
        raise KeyError(
            f"No real wavelength array could be identified "
            f"in {root_name}."
        )

    candidates.sort(
        key=lambda item: (
            item[0],
            len(item[2]),
        ),
        reverse=True,
    )

    score, path, wavelength_nm = (
        candidates[0]
    )

    return wavelength_nm, path


def select_collected_spectrum(
    sweep_result,
    wavelength_nm,
    *,
    root_name,
):
    """
    Select a real one-dimensional collected-efficiency spectrum.

    Complex field-amplitude arrays are rejected.
    """

    expected_length = len(
        wavelength_nm
    )

    candidates = []

    for path, array in nested_arrays(
        sweep_result,
        root_name=root_name,
    ):

        real_array = as_real_numeric_array(
            array
        )

        if real_array is None:
            continue

        squeezed = np.squeeze(
            real_array
        )

        if squeezed.ndim != 1:
            continue

        if len(squeezed) != expected_length:
            continue

        path_text = normalized_path_text(
            path
        )

        if (
            "wavelength" in path_text
            or "theta" in path_text
            or "angle" in path_text
        ):
            continue

        score = score_array_path(
            path,
            required_keywords=(
                COLLECTED_KEYWORDS
            ),
            preferred_keywords=[
                "q_collected_geom",
                "collected_geom",
                "efficiency",
                "spectrum",
                "integrated",
            ],
            rejected_keywords=[
                "wavelength",
                "theta",
                "angle",
                "field",
                "coefficient",
            ],
        )

        # A collected scattering efficiency should be nonnegative
        # apart from insignificant numerical noise.
        finite = squeezed[
            np.isfinite(squeezed)
        ]

        if finite.size == 0:
            continue

        negative_tolerance = (
            1e-10
            * max(
                1.0,
                float(
                    np.nanmax(
                        np.abs(finite)
                    )
                ),
            )
        )

        if np.nanmin(finite) < -negative_tolerance:
            continue

        candidates.append(
            (
                score,
                path,
                squeezed,
            )
        )

    if not candidates:
        raise KeyError(
            "No real collected spectrum with length "
            f"{expected_length} was identified in {root_name}."
        )

    candidates.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    score, path, spectrum = (
        candidates[0]
    )

    if score <= 0:
        warnings.warn(
            "The collected-spectrum path was selected primarily "
            "from its array dimensions. Verify this path:\n"
            f"{path}"
        )

    # Remove only insignificant negative numerical noise.
    spectrum = np.where(
        (
            spectrum < 0.0
        )
        & np.isclose(
            spectrum,
            0.0,
            atol=1e-10,
        ),
        0.0,
        spectrum,
    )

    return spectrum, path


def extract_directional_spectrum(
    checkpoint_payload,
    direction,
):
    """
    Extract wavelength in nm and the collected spectrum.
    """

    sweep_result = (
        checkpoint_payload[
            direction
        ]
    )

    wavelength_nm, wavelength_path = (
        select_wavelength_array(
            sweep_result,
            root_name=direction,
        )
    )

    spectrum, spectrum_path = (
        select_collected_spectrum(
            sweep_result,
            wavelength_nm,
            root_name=direction,
        )
    )

    return {
        "wavelength_nm": wavelength_nm,
        "spectrum": spectrum,
        "wavelength_path": wavelength_path,
        "spectrum_path": spectrum_path,
    }


example_backward_spectrum = (
    extract_directional_spectrum(
        example_checkpoint,
        "backward",
    )
)

example_forward_spectrum = (
    extract_directional_spectrum(
        example_checkpoint,
        "forward",
    )
)

print(
    "Selected backward wavelength path:"
)

print(
    example_backward_spectrum[
        "wavelength_path"
    ]
)

print(
    "Selected backward spectrum path:"
)

print(
    example_backward_spectrum[
        "spectrum_path"
    ]
)

print(
    "\nSelected forward wavelength path:"
)

print(
    example_forward_spectrum[
        "wavelength_path"
    ]
)

print(
    "Selected forward spectrum path:"
)

print(
    example_forward_spectrum[
        "spectrum_path"
    ]
)

# Cell 4: Build a checkpoint index

# In[84]:


# ============================================================
# INDEX ALL FULL CHECKPOINTS
# ============================================================

checkpoint_index_records = []

for checkpoint_path in full_checkpoint_paths:

    payload = load_full_checkpoint(
        checkpoint_path
    )

    task = payload["task"]

    checkpoint_index_records.append(
        {
            "checkpoint_path": (
                checkpoint_path
            ),
            "task_index": task.get(
                "task_index"
            ),
            "task_hash": task.get(
                "task_hash"
            ),
            "reference_design_wavelength_nm": (
                task.get(
                    "reference_design_wavelength_nm"
                )
            ),
            "case_family": task.get(
                "case_family"
            ),
            "scenario": task.get(
                "scenario"
            ),
            "absorber_location": task.get(
                "absorber_location"
            ),
            "absorber_k": task.get(
                "absorber_k"
            ),
            "core_thickness_factor": (
                task.get(
                    "core_thickness_factor"
                )
            ),
            "n_shells": task.get(
                "n_shells"
            ),
            "outer_shell_thickness_nm": (
                task.get(
                    "outer_shell_thickness_nm"
                )
            ),
            "predicted_diameter_nm": (
                task.get(
                    "predicted_diameter_nm"
                )
            ),
        }
    )


df_checkpoint_index = (
    pd.DataFrame(
        checkpoint_index_records
    )
)

numeric_index_columns = [
    "task_index",
    "reference_design_wavelength_nm",
    "absorber_k",
    "core_thickness_factor",
    "n_shells",
    "outer_shell_thickness_nm",
    "predicted_diameter_nm",
]

for column in numeric_index_columns:

    if column in df_checkpoint_index:

        df_checkpoint_index[
            column
        ] = pd.to_numeric(
            df_checkpoint_index[
                column
            ],
            errors="coerce",
        )


print(
    "Indexed checkpoints:",
    len(df_checkpoint_index),
)

display(
    df_checkpoint_index.head()
)

display(
    df_checkpoint_index.groupby(
        [
            "reference_design_wavelength_nm",
            "case_family",
            "scenario",
        ],
        dropna=False,
    )
    .size()
    .rename(
        "number_of_cases"
    )
    .reset_index()
)

# Cell 5: Backward and forward spectra for every absorber configuration
#
# This produces one figure per design wavelength. Each column is one absorber configuration:
#
# low-index domains;
# all layers;
# outer absorbing shell;
# core with constant shell number;
# core with approximately constant diameter.
#
# The upper row is backward collection and the lower row is forward collection.
#
# For configurations involving an additional geometrical parameter, line style identifies the geometry:
#
# outer shell: coating thickness;
# core: core-thickness factor.
#
# Line colour identifies kkk.

# In[85]:


# ============================================================
# BACKWARD AND FORWARD SPECTRA FOR EVERY ABSORBER CONFIGURATION
#
# Automatic y limits:
#   The upper limit is selected from the 99th percentile of all
#   finite plotted points for the corresponding direction and
#   reference wavelength.
#
# Manual limits:
#   Limits can be specified globally or per reference wavelength.
# ============================================================

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.lines import Line2D


# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------

SPECTRAL_REFERENCE_WAVELENGTHS = [
    380,
    580,
    760,
]

SPECTRAL_K_VALUES = [
    0.01,
    0.1,
    0.5,
    1.0,
]

SPECTRAL_FIGURE_WIDTH_CM = 18.0
SPECTRAL_FIGURE_HEIGHT_CM = 8.6

SPECTRAL_LINE_WIDTH = 0.85
SPECTRAL_REFERENCE_LINE_WIDTH = 1.4
SPECTRAL_ALPHA = 0.78

SAVE_SPECTRAL_FIGURES = True


# ------------------------------------------------------------
# Y-axis settings
# ------------------------------------------------------------

# Set to 0.99 so that 99% of all finite spectral points lie below
# the automatically selected upper limit.
SPECTRAL_Y_PERCENTILE = 0.99

# Additional headroom above the chosen percentile.
SPECTRAL_Y_HEADROOM_FACTOR = 1.05

# If True, automatic lower limits are fixed at zero.
SPECTRAL_Y_START_AT_ZERO = True


# Global manual overrides.
#
# Use None for automatic selection.
#
# Examples:
# MANUAL_BACKWARD_YLIM = (0.0, 1.5)
# MANUAL_FORWARD_YLIM = (0.0, 8.0)
MANUAL_BACKWARD_YLIM = None
MANUAL_FORWARD_YLIM = None


# Optional wavelength-specific limits.
#
# These override the global limits above.
#
# Examples:
# MANUAL_BACKWARD_YLIMS_BY_WAVELENGTH = {
#     380: (0.0, 1.5),
#     580: (0.0, 1.2),
#     760: (0.0, 1.8),
# }
#
# Use None or omit a wavelength for automatic selection.
MANUAL_BACKWARD_YLIMS_BY_WAVELENGTH = {
    380: None,
    580: None,
    760: None,
}

MANUAL_FORWARD_YLIMS_BY_WAVELENGTH = {
    380: None,
    580: None,
    760: None,
}


spectral_column_definitions = [
    {
        "title": "Low-index domains",
        "case_family": "low_index",
        "scenario": "fixed_reference_geometry",
        "geometry_column": None,
    },
    {
        "title": "All layers",
        "case_family": "all_layers",
        "scenario": "fixed_reference_geometry",
        "geometry_column": None,
    },
    {
        "title": "Additional outer shell",
        "case_family": "outer_shell",
        "scenario": "extra_outer_shell",
        "geometry_column": (
            "outer_shell_thickness_nm"
        ),
    },
    {
        "title": (
            "Core absorber\n"
            "constant shell number"
        ),
        "case_family": "core",
        "scenario": (
            "constant_shell_number"
        ),
        "geometry_column": (
            "core_thickness_factor"
        ),
    },
    {
        "title": (
            "Core absorber "
            "approximately\nconstant diameter"
        ),
        "case_family": "core",
        "scenario": "constant_diameter",
        "geometry_column": (
            "core_thickness_factor"
        ),
    },
]


K_COLOURS = {
    0.01: "#440154",
    0.1: "#31688E",
    0.5: "#35B779",
    1.0: "#FDE725",
}

GEOMETRY_LINESTYLES = [
    "-",
    "--",
    ":",
    "-.",
    (0, (5, 1)),
    (0, (3, 1, 1, 1)),
    (0, (1, 1)),
    (0, (7, 2)),
]


# ============================================================
# HELPERS
# ============================================================

def closest_dictionary_key(
    value,
    dictionary,
):
    """
    Return the numerical dictionary key nearest to value.
    """

    keys = np.asarray(
        list(dictionary.keys()),
        dtype=float,
    )

    return float(
        keys[
            np.argmin(
                np.abs(
                    keys - float(value)
                )
            )
        ]
    )


def select_reference_checkpoint(
    checkpoint_index,
    wavelength_nm,
):
    """
    Select the unique nonabsorbing reference checkpoint.
    """

    mask = (
        np.isclose(
            pd.to_numeric(
                checkpoint_index[
                    "reference_design_wavelength_nm"
                ],
                errors="coerce",
            ),
            float(wavelength_nm),
        )
        & checkpoint_index[
            "case_family"
        ]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("reference")
    )

    selected = (
        checkpoint_index.loc[
            mask
        ]
    )

    if len(selected) != 1:
        raise ValueError(
            f"Expected one {wavelength_nm} nm reference "
            f"checkpoint, found {len(selected)}."
        )

    return selected.iloc[0]


def calculate_percentile_ylim(
    spectra,
    *,
    percentile=0.99,
    headroom_factor=1.05,
    start_at_zero=True,
):
    """
    Calculate y limits containing the requested fraction of all
    finite spectral points.

    The percentile is calculated over all wavelengths and all curves
    plotted for one direction and reference wavelength.
    """

    finite_arrays = []

    for spectrum in spectra:

        values = as_real_numeric_array(
            spectrum
        )

        if values is None:
            continue

        finite = values[
            np.isfinite(values)
        ]

        if finite.size > 0:
            finite_arrays.append(
                finite
            )

    if not finite_arrays:
        return (0.0, 1.0)

    all_values = np.concatenate(
        finite_arrays
    )

    if start_at_zero:

        lower_limit = 0.0

        positive_values = all_values[
            all_values >= 0.0
        ]

        if positive_values.size == 0:
            return (0.0, 1.0)

        upper_limit = float(
            np.nanquantile(
                positive_values,
                percentile,
            )
        )

        upper_limit *= (
            headroom_factor
        )

    else:

        tail_fraction = (
            1.0 - percentile
        ) / 2.0

        lower_limit = float(
            np.nanquantile(
                all_values,
                tail_fraction,
            )
        )

        upper_limit = float(
            np.nanquantile(
                all_values,
                1.0 - tail_fraction,
            )
        )

        interval = (
            upper_limit
            - lower_limit
        )

        lower_limit -= (
            0.025 * interval
        )

        upper_limit += (
            0.025 * interval
        )

    if (
        not np.isfinite(
            upper_limit
        )
        or upper_limit <= lower_limit
    ):

        upper_limit = (
            lower_limit + 1.0
        )

    return (
        lower_limit,
        upper_limit,
    )


def resolve_spectral_ylim(
    *,
    wavelength_nm,
    direction,
    spectra,
):
    """
    Select wavelength-specific manual limits, global manual limits,
    or automatic percentile limits, in that order.
    """

    if direction == "backward":

        wavelength_specific = (
            MANUAL_BACKWARD_YLIMS_BY_WAVELENGTH.get(
                wavelength_nm
            )
        )

        global_manual = (
            MANUAL_BACKWARD_YLIM
        )

    elif direction == "forward":

        wavelength_specific = (
            MANUAL_FORWARD_YLIMS_BY_WAVELENGTH.get(
                wavelength_nm
            )
        )

        global_manual = (
            MANUAL_FORWARD_YLIM
        )

    else:

        raise ValueError(
            f"Unknown direction: {direction}"
        )

    if wavelength_specific is not None:

        selected_limits = tuple(
            map(
                float,
                wavelength_specific,
            )
        )

        source = (
            "wavelength-specific manual"
        )

    elif global_manual is not None:

        selected_limits = tuple(
            map(
                float,
                global_manual,
            )
        )

        source = "global manual"

    else:

        selected_limits = (
            calculate_percentile_ylim(
                spectra,
                percentile=(
                    SPECTRAL_Y_PERCENTILE
                ),
                headroom_factor=(
                    SPECTRAL_Y_HEADROOM_FACTOR
                ),
                start_at_zero=(
                    SPECTRAL_Y_START_AT_ZERO
                ),
            )
        )

        source = (
            f"automatic "
            f"{100 * SPECTRAL_Y_PERCENTILE:.1f}%"
        )

    if (
        len(selected_limits) != 2
        or not np.all(
            np.isfinite(
                selected_limits
            )
        )
        or selected_limits[1]
        <= selected_limits[0]
    ):

        raise ValueError(
            f"Invalid {direction} y limits for "
            f"{wavelength_nm} nm: {selected_limits}"
        )

    return (
        selected_limits,
        source,
    )


# ============================================================
# LOAD AND CACHE ALL SPECTRA NEEDED BY THE FIGURES
# ============================================================

spectral_cache = {}


for wavelength_nm in (
    SPECTRAL_REFERENCE_WAVELENGTHS
):

    reference_index_row = (
        select_reference_checkpoint(
            df_checkpoint_index,
            wavelength_nm,
        )
    )

    reference_payload = (
        load_full_checkpoint(
            Path(
                reference_index_row[
                    "checkpoint_path"
                ]
            )
        )
    )

    reference_spectra = {
        direction: (
            extract_directional_spectrum(
                reference_payload,
                direction,
            )
        )
        for direction in (
            "backward",
            "forward",
        )
    }

    wavelength_cache = {
        "reference": (
            reference_spectra
        ),
        "columns": [],
    }

    for definition in (
        spectral_column_definitions
    ):

        selection_mask = (
            np.isclose(
                pd.to_numeric(
                    df_checkpoint_index[
                        "reference_design_wavelength_nm"
                    ],
                    errors="coerce",
                ),
                float(wavelength_nm),
            )
            & df_checkpoint_index[
                "case_family"
            ]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq(
                definition[
                    "case_family"
                ]
            )
            & df_checkpoint_index[
                "scenario"
            ]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq(
                definition[
                    "scenario"
                ]
            )
        )

        selected_cases = (
            df_checkpoint_index.loc[
                selection_mask
            ]
            .copy()
        )

        geometry_column = (
            definition[
                "geometry_column"
            ]
        )

        if geometry_column is None:

            selected_cases[
                "_geometry_value"
            ] = 0.0

        else:

            selected_cases[
                "_geometry_value"
            ] = pd.to_numeric(
                selected_cases[
                    geometry_column
                ],
                errors="coerce",
            )

        geometry_values = sorted(
            selected_cases[
                "_geometry_value"
            ]
            .dropna()
            .unique()
            .tolist()
        )

        geometry_linestyle_map = {
            geometry_value: (
                GEOMETRY_LINESTYLES[
                    index
                    % len(
                        GEOMETRY_LINESTYLES
                    )
                ]
            )
            for index, geometry_value
            in enumerate(
                geometry_values
            )
        }

        column_cases = []

        for _, case_row in (
            selected_cases.sort_values(
                [
                    "_geometry_value",
                    "absorber_k",
                ]
            ).iterrows()
        ):

            payload = load_full_checkpoint(
                Path(
                    case_row[
                        "checkpoint_path"
                    ]
                )
            )

            extracted_by_direction = {
                direction: (
                    extract_directional_spectrum(
                        payload,
                        direction,
                    )
                )
                for direction in (
                    "backward",
                    "forward",
                )
            }

            column_cases.append(
                {
                    "absorber_k": float(
                        case_row[
                            "absorber_k"
                        ]
                    ),
                    "geometry_value": float(
                        case_row[
                            "_geometry_value"
                        ]
                    ),
                    "spectra": (
                        extracted_by_direction
                    ),
                }
            )

        wavelength_cache[
            "columns"
        ].append(
            {
                "definition": definition,
                "cases": column_cases,
                "geometry_linestyle_map": (
                    geometry_linestyle_map
                ),
            }
        )

    spectral_cache[
        wavelength_nm
    ] = wavelength_cache


# ============================================================
# CREATE FIGURES
# ============================================================

for wavelength_nm in (
    SPECTRAL_REFERENCE_WAVELENGTHS
):

    wavelength_cache = (
        spectral_cache[
            wavelength_nm
        ]
    )

    # --------------------------------------------------------
    # Collect all plotted values for robust y limits
    # --------------------------------------------------------

    all_spectra_by_direction = {
        "backward": [
            wavelength_cache[
                "reference"
            ][
                "backward"
            ][
                "spectrum"
            ]
        ],
        "forward": [
            wavelength_cache[
                "reference"
            ][
                "forward"
            ][
                "spectrum"
            ]
        ],
    }

    for column_cache in (
        wavelength_cache[
            "columns"
        ]
    ):

        for case in (
            column_cache[
                "cases"
            ]
        ):

            for direction in (
                "backward",
                "forward",
            ):

                all_spectra_by_direction[
                    direction
                ].append(
                    case[
                        "spectra"
                    ][
                        direction
                    ][
                        "spectrum"
                    ]
                )

    backward_ylim, backward_source = (
        resolve_spectral_ylim(
            wavelength_nm=wavelength_nm,
            direction="backward",
            spectra=(
                all_spectra_by_direction[
                    "backward"
                ]
            ),
        )
    )

    forward_ylim, forward_source = (
        resolve_spectral_ylim(
            wavelength_nm=wavelength_nm,
            direction="forward",
            spectra=(
                all_spectra_by_direction[
                    "forward"
                ]
            ),
        )
    )

    print(
        f"{wavelength_nm} nm backward ylim: "
        f"{backward_ylim} ({backward_source})"
    )

    print(
        f"{wavelength_nm} nm forward ylim: "
        f"{forward_ylim} ({forward_source})"
    )

    cm_to_inches = 1.0 / 2.54

    fig, axes = plt.subplots(
        2,
        len(
            spectral_column_definitions
        ),
        figsize=(
            SPECTRAL_FIGURE_WIDTH_CM
            * cm_to_inches,
            SPECTRAL_FIGURE_HEIGHT_CM
            * cm_to_inches,
        ),
        sharex=True,
        sharey="row",
        constrained_layout=True,
    )

    for column_index, column_cache in enumerate(
        wavelength_cache[
            "columns"
        ]
    ):

        definition = (
            column_cache[
                "definition"
            ]
        )

        geometry_linestyle_map = (
            column_cache[
                "geometry_linestyle_map"
            ]
        )

        for direction_index, direction in enumerate(
            (
                "backward",
                "forward",
            )
        ):

            ax = axes[
                direction_index,
                column_index,
            ]

            reference = (
                wavelength_cache[
                    "reference"
                ][
                    direction
                ]
            )

            ax.plot(
                reference[
                    "wavelength_nm"
                ],
                reference[
                    "spectrum"
                ],
                color="black",
                linewidth=(
                    SPECTRAL_REFERENCE_LINE_WIDTH
                ),
                label="Nonabsorbing reference",
                zorder=5,
            )

            for case in (
                column_cache[
                    "cases"
                ]
            ):

                k_value = float(
                    case[
                        "absorber_k"
                    ]
                )

                nearest_k = (
                    closest_dictionary_key(
                        k_value,
                        K_COLOURS,
                    )
                )

                geometry_value = float(
                    case[
                        "geometry_value"
                    ]
                )

                extracted = (
                    case[
                        "spectra"
                    ][
                        direction
                    ]
                )

                ax.plot(
                    extracted[
                        "wavelength_nm"
                    ],
                    extracted[
                        "spectrum"
                    ],
                    color=K_COLOURS[
                        nearest_k
                    ],
                    linestyle=(
                        geometry_linestyle_map[
                            geometry_value
                        ]
                    ),
                    linewidth=(
                        SPECTRAL_LINE_WIDTH
                    ),
                    alpha=SPECTRAL_ALPHA,
                    zorder=2,
                )

            if direction_index == 0:

                ax.set_title(
                    definition["title"],
                    fontsize=8,
                )

                ax.set_ylim(
                    *backward_ylim
                )

            else:

                ax.set_ylim(
                    *forward_ylim
                )

            if column_index == 0:

                ax.set_ylabel(
                    (
                        r"$Q_{\mathrm{back}}$"
                        if direction == "backward"
                        else r"$Q_{\mathrm{forward}}$"
                    ),
                    fontsize=8,
                )

            if direction_index == 1:

                ax.set_xlabel(
                    "Wavelength [nm]",
                    fontsize=8,
                )

            ax.set_xlim(
                380.0,
                780.0,
            )

            ax.grid(False)

    # --------------------------------------------------------
    # Legend
    # --------------------------------------------------------

    k_handles = [
        Line2D(
            [0],
            [0],
            color=K_COLOURS[
                k_value
            ],
            linewidth=1.4,
            label=rf"$k={k_value:g}$",
        )
        for k_value in SPECTRAL_K_VALUES
    ]

    reference_handle = Line2D(
        [0],
        [0],
        color="black",
        linewidth=(
            SPECTRAL_REFERENCE_LINE_WIDTH
        ),
        label="Nonabsorbing reference",
    )

    fig.legend(
        handles=[
            reference_handle,
            *k_handles,
        ],
        loc="outside lower center",
        ncol=5,
        frameon=True,
        fontsize=7,
    )

    fig.suptitle(
        (
            "Reference design "
            rf"$\lambda_0={wavelength_nm}$ nm"
        ),
        fontsize=9,
    )

    if SAVE_SPECTRAL_FIGURES:

        png_path = (
            FIGURE_DIR
            / (
                "directional_spectra_"
                f"{wavelength_nm}nm.png"
            )
        )

        pdf_path = (
            FIGURE_DIR
            / (
                "directional_spectra_"
                f"{wavelength_nm}nm.pdf"
            )
        )

        fig.savefig(
            png_path,
            dpi=PNG_DPI,
            transparent=True,
            bbox_inches="tight",
        )

        fig.savefig(
            pdf_path,
            transparent=True,
            bbox_inches="tight",
        )

        print("Saved:", png_path)
        print("Saved:", pdf_path)

    plt.show()

# Cell 6: Identify theta and differential-scattering arrays

# In[86]:


# ============================================================
# IDENTIFY ANGULAR AND DIFFERENTIAL-SCATTERING ARRAYS
# ============================================================

def select_theta_array(
    sweep_result,
    *,
    root_name,
):
    """
    Select the polar-angle array and return radians.
    """

    candidates = []

    for path, array in nested_arrays(
        sweep_result,
        root_name=root_name,
    ):

        real_array = as_real_numeric_array(
            array
        )

        if real_array is None:
            continue

        squeezed = np.squeeze(
            real_array
        )

        if squeezed.ndim != 1:
            continue

        if squeezed.size < 30:
            continue

        if not np.issubdtype(
            squeezed.dtype,
            np.number,
        ):
            continue

        finite = squeezed[
            np.isfinite(squeezed)
        ]

        if finite.size == 0:
            continue

        minimum_value = float(
            np.nanmin(finite)
        )

        maximum_value = float(
            np.nanmax(finite)
        )

        score = score_array_path(
            path,
            required_keywords=(
                THETA_KEYWORDS
            ),
        )

        if (
            minimum_value >= -1e-8
            and maximum_value
            <= np.pi + 1e-6
        ):

            theta_rad = (squeezed)

            score += 5

        elif (
            minimum_value >= -1e-8
            and maximum_value
            <= 180.0 + 1e-6
        ):

            theta_rad = np.deg2rad(squeezed)

            score += 4

        else:

            continue

        candidates.append(
            (
                score,
                path,
                theta_rad,
            )
        )

    if not candidates:

        raise KeyError(
            f"No polar-angle array was found in {root_name}."
        )

    candidates.sort(
        key=lambda item: (
            item[0],
            len(item[2]),
        ),
        reverse=True,
    )

    score, path, theta_rad = candidates[0]

    return theta_rad, path


def select_differential_array(
    sweep_result,
    *,
    root_name,
    n_wavelengths,
    n_angles,
):
    """
    Select a wavelength-by-angle differential scattering array.

    Preference is given to efficiency-like arrays. If none is found,
    differential cross-section-like names are accepted.
    """

    candidates = []

    for path, array in nested_arrays(
        sweep_result,
        root_name=root_name,
    ):

        real_array = as_real_numeric_array(
            array
        )

        if real_array is None:
            continue

        squeezed = np.squeeze(
            real_array
        )

        if squeezed.ndim != 2:
            continue

        shape = squeezed.shape

        compatible = (
            shape
            in [
                (
                    n_wavelengths,
                    n_angles,
                ),
                (
                    n_angles,
                    n_wavelengths,
                ),
            ]
        )

        if not compatible:
            continue

        path_text = normalized_path_text(
            path
        )

        efficiency_score = (
            score_array_path(
                path,
                required_keywords=(
                    DIFFERENTIAL_EFFICIENCY_KEYWORDS
                ),
                preferred_keywords=[
                    "efficiency",
                    "q",
                    "sca",
                ],
            )
        )

        cross_section_score = (
            score_array_path(
                path,
                required_keywords=(
                    DIFFERENTIAL_CROSS_SECTION_KEYWORDS
                ),
                preferred_keywords=[
                    "cross",
                    "section",
                    "sca",
                ],
            )
        )

        score = max(
            efficiency_score,
            cross_section_score,
        )

        quantity_type = (
            "differential efficiency"
            if efficiency_score
            >= cross_section_score
            else
            "differential cross-section"
        )

        if shape == (
            n_angles,
            n_wavelengths,
        ):

            oriented = (
                squeezed.T
            )

        else:

            oriented = squeezed

        candidates.append(
            (
                score,
                path,
                oriented,
                quantity_type,
            )
        )

    if not candidates:

        raise KeyError(
            "No two-dimensional differential-scattering array "
            f"with shape ({n_wavelengths}, {n_angles}) or "
            f"({n_angles}, {n_wavelengths}) was found."
        )

    candidates.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    score, path, array, quantity_type = (
        candidates[0]
    )

    return array, path, quantity_type


example_sweep_result = (
    example_checkpoint[
        "backward"
    ]
)

example_wavelength_nm, _ = (
    select_wavelength_array(
        example_sweep_result,
        root_name="example",
    )
)

example_theta_rad, example_theta_path = (
    select_theta_array(
        example_sweep_result,
        root_name="example",
    )
)

(
    example_differential,
    example_differential_path,
    example_differential_type,
) = select_differential_array(
    example_sweep_result,
    root_name="example",
    n_wavelengths=len(
        example_wavelength_nm
    ),
    n_angles=len(
        example_theta_rad
    ),
)


print(
    "Selected theta path:"
)

print(
    example_theta_path
)

print(
    "\nSelected differential-scattering path:"
)

print(
    example_differential_path
)

print(
    "\nQuantity type:"
)

print(
    example_differential_type
)

print(
    "\nDifferential array shape:"
)

print(
    example_differential.shape
)

# Cell 7: Polar plots at the backward peak and two nearby wavelengths
#
# The following settings select:
#
# nonabsorbing reference;
# weak absorption, k=0.01k=0.01k=0.01;
# moderate absorption, k=0.1k=0.1k=0.1;
# strong absorption, k=1.0k=1.0k=1.0.
#
# The default absorber example is absorption in all layers. Change POLAR_CASE_FAMILY and POLAR_SCENARIO to inspect another configuration.
#
# For outer-shell and core cases, specify POLAR_GEOMETRY_VALUE.

# In[87]:


# ============================================================
# POLAR DIFFERENTIAL-SCATTERING PLOTS
#
# Curves:
#   nonabsorbing reference
#   weak absorption
#   moderate absorption
#   strong absorption
#
# Wavelengths:
#   backward-collected peak wavelength
#   peak - wavelength offset
#   peak + wavelength offset
# ============================================================

import matplotlib.pyplot as plt


# ------------------------------------------------------------
# User settings
# ------------------------------------------------------------

POLAR_REFERENCE_WAVELENGTH_NM = 580

POLAR_CASE_FAMILY = "all_layers"
POLAR_SCENARIO = "fixed_reference_geometry"

# Used only for core or outer-shell cases.
#
# Core:
#   geometry column = core_thickness_factor
#
# Outer shell:
#   geometry column = outer_shell_thickness_nm
POLAR_GEOMETRY_VALUE = None

POLAR_K_VALUES = [
    0.01,
    0.1,
    1.0,
]

POLAR_K_LABELS = {
    0.01: "Weak absorption",
    0.1: "Moderate absorption",
    1.0: "Strong absorption",
}

POLAR_WAVELENGTH_OFFSET_NM = 50.0

POLAR_NORMALIZE_EACH_PANEL = False
POLAR_LOGARITHMIC_RADIUS = True

POLAR_WIDTH_CM = 17.5
POLAR_HEIGHT_CM = 6.4

SAVE_POLAR_FIGURE = True


POLAR_COLOURS = {
    0.0: "black",
    0.01: "#377EB8",
    0.1: "#FF7F00",
    1.0: "#E41A1C",
}


def select_absorbing_checkpoint(
    checkpoint_index,
    *,
    wavelength_nm,
    case_family,
    scenario,
    k_value,
    geometry_value=None,
):
    """
    Select one absorber checkpoint.
    """

    mask = (
        np.isclose(
            checkpoint_index[
                "reference_design_wavelength_nm"
            ],
            float(wavelength_nm),
        )
        & checkpoint_index[
            "case_family"
        ]
        .astype(str)
        .str.lower()
        .eq(
            str(case_family).lower()
        )
        & checkpoint_index[
            "scenario"
        ]
        .astype(str)
        .str.lower()
        .eq(
            str(scenario).lower()
        )
        & np.isclose(
            checkpoint_index[
                "absorber_k"
            ],
            float(k_value),
        )
    )

    selected = (
        checkpoint_index.loc[
            mask
        ]
        .copy()
    )

    if geometry_value is not None:

        if case_family == "core":

            geometry_column = (
                "core_thickness_factor"
            )

        elif case_family == "outer_shell":

            geometry_column = (
                "outer_shell_thickness_nm"
            )

        else:

            geometry_column = None

        if geometry_column is not None:

            selected = selected.loc[
                np.isclose(
                    selected[
                        geometry_column
                    ],
                    float(
                        geometry_value
                    ),
                )
            ]

    if len(selected) != 1:

        display(
            selected
        )

        raise ValueError(
            "Expected exactly one selected absorber checkpoint, "
            f"found {len(selected)} for "
            f"{wavelength_nm} nm, family={case_family}, "
            f"scenario={scenario}, k={k_value}, "
            f"geometry={geometry_value}."
        )

    return selected.iloc[0]


# ------------------------------------------------------------
# Load the nonabsorbing reference
# ------------------------------------------------------------

reference_row = select_reference_checkpoint(
    df_checkpoint_index,
    POLAR_REFERENCE_WAVELENGTH_NM,
)

reference_payload = load_full_checkpoint(
    Path(
        reference_row[
            "checkpoint_path"
        ]
    )
)

# Backward and forward payloads should contain the same full
# angular solution. Use the backward result consistently.
reference_result = (
    reference_payload[
        "backward"
    ]
)

reference_wavelength_nm, _ = (
    select_wavelength_array(
        reference_result,
        root_name="reference",
    )
)

reference_theta_rad, _ = (
    select_theta_array(
        reference_result,
        root_name="reference",
    )
)

(
    reference_differential,
    differential_path,
    differential_quantity_type,
) = select_differential_array(
    reference_result,
    root_name="reference",
    n_wavelengths=len(
        reference_wavelength_nm
    ),
    n_angles=len(
        reference_theta_rad
    ),
)


# ------------------------------------------------------------
# Identify the backward-collected peak wavelength of the
# nonabsorbing reference
# ------------------------------------------------------------

reference_collected = (
    extract_directional_spectrum(
        reference_payload,
        "backward",
    )
)

reference_peak_index = int(
    np.nanargmax(
        reference_collected[
            "spectrum"
        ]
    )
)

reference_peak_wavelength_nm = float(
    reference_collected[
        "wavelength_nm"
    ][
        reference_peak_index
    ]
)


target_wavelengths_nm = np.array(
    [
        reference_peak_wavelength_nm
        - POLAR_WAVELENGTH_OFFSET_NM,

        reference_peak_wavelength_nm,

        reference_peak_wavelength_nm
        + POLAR_WAVELENGTH_OFFSET_NM,
    ],
    dtype=float,
)

target_wavelengths_nm = np.clip(
    target_wavelengths_nm,
    np.nanmin(
        reference_wavelength_nm
    ),
    np.nanmax(
        reference_wavelength_nm
    ),
)

print(
    "Reference backward peak:",
    f"{reference_peak_wavelength_nm:.1f} nm",
)

print(
    "Polar-plot wavelengths:",
    target_wavelengths_nm,
)

print(
    "Differential quantity:",
    differential_quantity_type,
)

print(
    "Differential path:",
    differential_path,
)


# ------------------------------------------------------------
# Load reference and absorption cases
# ------------------------------------------------------------

polar_cases = [
    {
        "label": "Nonabsorbing reference",
        "k": 0.0,
        "payload": reference_payload,
    }
]

for k_value in POLAR_K_VALUES:

    selected_row = (
        select_absorbing_checkpoint(
            df_checkpoint_index,
            wavelength_nm=(
                POLAR_REFERENCE_WAVELENGTH_NM
            ),
            case_family=(
                POLAR_CASE_FAMILY
            ),
            scenario=(
                POLAR_SCENARIO
            ),
            k_value=k_value,
            geometry_value=(
                POLAR_GEOMETRY_VALUE
            ),
        )
    )

    payload = load_full_checkpoint(
        Path(
            selected_row[
                "checkpoint_path"
            ]
        )
    )

    polar_cases.append(
        {
            "label": (
                POLAR_K_LABELS[
                    k_value
                ]
                + rf", $k={k_value:g}$"
            ),
            "k": k_value,
            "payload": payload,
        }
    )


# ------------------------------------------------------------
# Extract differential arrays
# ------------------------------------------------------------

polar_case_arrays = []

for case in polar_cases:

    result = (
        case["payload"][
            "backward"
        ]
    )

    wavelength_nm, _ = (
        select_wavelength_array(
            result,
            root_name=(
                case["label"]
            ),
        )
    )

    theta_rad, _ = (
        select_theta_array(
            result,
            root_name=(
                case["label"]
            ),
        )
    )

    differential, _, _ = (
        select_differential_array(
            result,
            root_name=(
                case["label"]
            ),
            n_wavelengths=len(
                wavelength_nm
            ),
            n_angles=len(
                theta_rad
            ),
        )
    )

    polar_case_arrays.append(
        {
            **case,
            "wavelength_nm": (
                wavelength_nm
            ),
            "theta_rad": theta_rad,
            "differential": differential,
        }
    )


# ------------------------------------------------------------
# Create polar plots
# ------------------------------------------------------------

cm_to_inches = 1.0 / 2.54

fig, axes = plt.subplots(
    1,
    3,
    figsize=(
        POLAR_WIDTH_CM
        * cm_to_inches,
        POLAR_HEIGHT_CM
        * cm_to_inches,
    ),
    subplot_kw={
        "projection": "polar",
    },
    constrained_layout=True,
)

axes = np.atleast_1d(
    axes
)


for panel_index, target_wavelength_nm in enumerate(
    target_wavelengths_nm
):

    ax = axes[
        panel_index
    ]

    panel_curves = []

    for case in polar_case_arrays:

        wavelength_index = int(
            np.nanargmin(
                np.abs(
                    case[
                        "wavelength_nm"
                    ]
                    - target_wavelength_nm
                )
            )
        )

        actual_wavelength_nm = float(
            case[
                "wavelength_nm"
            ][
                wavelength_index
            ]
        )

        radial_values = (
            case[
                "differential"
            ][
                wavelength_index,
                :
            ]
        )

        radial_values = np.asarray(
            radial_values,
            dtype=float,
        )

        radial_values[
            radial_values < 0.0
        ] = np.nan

        panel_curves.append(
            {
                "case": case,
                "actual_wavelength_nm": (
                    actual_wavelength_nm
                ),
                "radial_values": (
                    radial_values
                ),
            }
        )

    if POLAR_NORMALIZE_EACH_PANEL:

        panel_maximum = np.nanmax(
            [
                np.nanmax(
                    curve[
                        "radial_values"
                    ]
                )
                for curve in panel_curves
            ]
        )

    else:

        panel_maximum = 1.0

    for curve in panel_curves:

        radial_values = (
            curve[
                "radial_values"
            ]
            / panel_maximum
        )

        k_value = float(
            curve["case"]["k"]
        )

        theta = curve["case"]["theta_rad"]

        theta_full = np.concatenate([
            theta,
            2.0 * np.pi - theta[::-1],
        ])

        radial_full = np.concatenate([
            radial_values,
            radial_values[::-1],
        ])

        ax.plot(
            theta_full,
            radial_full,
            color=POLAR_COLOURS[k_value],
            linewidth=1.2,
            label=curve["case"]["label"],
        )

    # Incident propagation direction at theta = 0.
    ax.set_theta_zero_location(
        "E"
    )

    ax.set_theta_direction(
        1
    )

    ax.set_thetagrids(
        [
            0,
            45,
            90,
            135,
            180,
            225,
            270,
            315,
        ],
        labels=[
            r"$0^\circ$",
            r"$45^\circ$",
            r"$90^\circ$",
            r"$135^\circ$",
            r"$180^\circ$",
            r"$225^\circ$",
            r"$270^\circ$",
            r"$315^\circ$",
        ],
    )

    if POLAR_LOGARITHMIC_RADIUS:

        ax.set_rscale(
            "log"
        )

    ax.set_title(
        (
            f"{target_wavelength_nm:.0f} nm"
        ),
        fontsize=9,
    )

    ax.grid(
        True,
        color="0.80",
        linewidth=0.55,
    )


# ------------------------------------------------------------
# Legend and figure description
# ------------------------------------------------------------

figure_title = (
    f"$\\lambda_0="
    f"{POLAR_REFERENCE_WAVELENGTH_NM}$ nm, "
    f"{POLAR_CASE_FAMILY.replace('_', ' ')}, "
    f"{POLAR_SCENARIO.replace('_', ' ')}"
)

if POLAR_GEOMETRY_VALUE is not None:

    figure_title += (
        f", geometry value "
        f"{POLAR_GEOMETRY_VALUE:g}"
    )

fig.suptitle(
    figure_title,
    fontsize=9,
)

fig.legend(
    handles=axes[0].get_legend_handles_labels()[0],
    labels=axes[0].get_legend_handles_labels()[1],
    loc="outside lower center",
    ncol=2,
    frameon=True,
    fontsize=7,
)


if SAVE_POLAR_FIGURE:

    geometry_suffix = (
        ""
        if POLAR_GEOMETRY_VALUE
        is None
        else (
            "_geometry_"
            f"{POLAR_GEOMETRY_VALUE:g}"
        )
    )

    polar_basename = (
        "polar_differential_scattering_"
        f"{POLAR_REFERENCE_WAVELENGTH_NM}nm_"
        f"{POLAR_CASE_FAMILY}_"
        f"{POLAR_SCENARIO}"
        f"{geometry_suffix}"
    )

    polar_png = (
        FIGURE_DIR
        / f"{polar_basename}.png"
    )

    polar_pdf = (
        FIGURE_DIR
        / f"{polar_basename}.pdf"
    )

    fig.savefig(
        polar_png,
        dpi=PNG_DPI,
        transparent=True,
        bbox_inches="tight",
    )

    fig.savefig(
        polar_pdf,
        transparent=True,
        bbox_inches="tight",
    )

    print(
        "Saved:",
        polar_png,
    )

    print(
        "Saved:",
        polar_pdf,
    )


plt.show()

# In[124]:


# ============================================================
# EXTRACT DIFFERENTIAL ARRAYS FOR PRESENTATION CASES
# ============================================================

polar_case_arrays = []

for case in presentation_polar_cases:

    result = case["payload"]["backward"]

    wavelength_nm, wavelength_path = (
        select_wavelength_array(
            result,
            root_name=case["key"],
        )
    )

    theta_rad, theta_path = (
        select_theta_array(
            result,
            root_name=case["key"],
        )
    )

    (
        differential,
        differential_path,
        differential_quantity_type,
    ) = select_differential_array(
        result,
        root_name=case["key"],
        n_wavelengths=len(
            wavelength_nm
        ),
        n_angles=len(
            theta_rad
        ),
    )

    polar_case_arrays.append(
        {
            **case,
            "wavelength_nm": wavelength_nm,
            "theta_rad": theta_rad,
            "differential": differential,
            "wavelength_path": wavelength_path,
            "theta_path": theta_path,
            "differential_path": differential_path,
            "differential_quantity_type": (
                differential_quantity_type
            ),
        }
    )

print(
    "Prepared presentation cases:",
    [
        case["label"]
        for case in polar_case_arrays
    ],
)

# In[ ]:


# ============================================================
# ALL-WAVELENGTH POLAR AND ANGULAR DIFFERENTIAL-SCATTERING PLOTS
#
# This cell uses the arrays already stored in:
#     polar_case_arrays
#
# It creates:
#   1. Polar differential scattering for all wavelengths
#   2. Polar differential scattering at selected wavelengths
#   3. Differential scattering versus theta for all wavelengths
#
# Convention:
#   theta = 0 degrees   -> forward scattering
#   theta = 180 degrees -> backward scattering
# ============================================================

from importlib import reload
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import bragg_onion._notebook_tools.scattering_plots

reload(
    bragg_onion._notebook_tools.scattering_plots
)

from bragg_onion._notebook_tools.scattering_plots import (
    plot_polar_differential_scattering,
    plot_differential_vs_theta,
)


# ============================================================
# USER SETTINGS
# ============================================================

# Select the case by its position in polar_case_arrays.
#
# With your current construction:
#   0 -> nonabsorbing reference
#   1 -> weak absorption, k = 0.01
#   2 -> moderate absorption, k = 0.1
#   3 -> strong absorption, k = 1.0
SELECTED_POLAR_CASE_INDEX = 0

# Alternatively, select by exact or partial label.
# Set to None to use SELECTED_POLAR_CASE_INDEX.
#
# Examples:
# SELECTED_POLAR_CASE_LABEL = "Nonabsorbing reference"
# Prepared presentation cases:
# ['Nonabsorbing reference',
# 'Core absorber, approximately constant diameter\n$k=1$, $\\gamma_\\mathrm{c}=50$',
# 'Core absorber, constant shell number\n$k=1$, $\\gamma_\\mathrm{c}=50$']
SELECTED_POLAR_CASE_LABEL = "Nonabsorbing reference"


# Wavelengths for the selected-wavelength polar diagram.
#
# If None, use:
#   backward reference peak - 150 nm
#   backward reference peak
#   backward reference peak + 150 nm
MANUAL_SELECTED_WAVELENGTHS_NM = None

WAVELENGTH_OFFSET_NM = 150.0


# Plot controls
PLOT_ALL_WAVELENGTH_POLAR = True
PLOT_SELECTED_WAVELENGTH_POLAR = True
PLOT_DIFFERENTIAL_VS_THETA = True

# "db" reproduces the appearance of the attached example.
Y_MODE = "db"

SHOW_SELECTED_WAVELENGTH_LEGEND = True

SAVE_SCATTERING_PLOTS = True

SCATTERING_PLOT_OUTPUT_DIRECTORY = (
    FIGURE_DIR
    / "angular_scattering"
)

SCATTERING_PLOT_OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)

SCATTERING_PLOT_DPI = 600


# ============================================================
# VALIDATE AVAILABLE CASES
# ============================================================

if (
    "polar_case_arrays" not in globals()
    or len(polar_case_arrays) == 0
):
    raise NameError(
        "polar_case_arrays is unavailable or empty. "
        "Run the preceding checkpoint-selection and differential-array "
        "extraction cells first."
    )


print("Available polar cases:")

for case_index, case in enumerate(
    polar_case_arrays
):
    print(
        f"  {case_index}: "
        f"{case.get('label', '<unlabelled case>')}"
    )


# ============================================================
# SELECT THE REQUESTED CASE
# ============================================================

if SELECTED_POLAR_CASE_LABEL is not None:

    matching_cases = [
        case
        for case in polar_case_arrays
        if (
            SELECTED_POLAR_CASE_LABEL.lower()
            in str(
                case.get(
                    "label",
                    "",
                )
            ).lower()
        )
    ]

    if len(matching_cases) != 1:

        print(
            "\nMatching labels:",
            [
                case.get(
                    "label",
                    "",
                )
                for case in matching_cases
            ],
        )

        raise ValueError(
            "The label selection must match exactly one case. "
            f"Found {len(matching_cases)} matches for "
            f"{SELECTED_POLAR_CASE_LABEL!r}."
        )

    selected_polar_case = (
        matching_cases[0]
    )

else:

    if not (
        0
        <= SELECTED_POLAR_CASE_INDEX
        < len(polar_case_arrays)
    ):
        raise IndexError(
            "SELECTED_POLAR_CASE_INDEX is outside the available "
            f"range 0 to {len(polar_case_arrays) - 1}."
        )

    selected_polar_case = (
        polar_case_arrays[
            SELECTED_POLAR_CASE_INDEX
        ]
    )


selected_case_label = str(
    selected_polar_case.get(
        "label",
        (
            "case_"
            f"{SELECTED_POLAR_CASE_INDEX}"
        ),
    )
)

print(
    "\nSelected case:",
    selected_case_label,
)


# ============================================================
# PREPARE theta, wavelength, AND DIFFERENTIAL ARRAYS
# ============================================================

theta_rad = as_real_numeric_array(
    selected_polar_case[
        "theta_rad"
    ]
)

wavelengths_nm = (
    as_real_numeric_array(
        selected_polar_case[
            "wavelength_nm"
        ]
    )
)

differential_scattering = (
    as_real_numeric_array(
        selected_polar_case[
            "differential"
        ]
    )
)


if theta_rad is None:
    raise TypeError(
        "The selected theta array is complex or nonnumeric."
    )

if wavelengths_nm is None:
    raise TypeError(
        "The selected wavelength array is complex or nonnumeric."
    )

if differential_scattering is None:
    raise TypeError(
        "The selected differential-scattering array is complex "
        "or nonnumeric."
    )


theta_rad = np.squeeze(
    theta_rad
)

wavelengths_nm = np.squeeze(
    wavelengths_nm
)

differential_scattering = np.squeeze(
    differential_scattering
)


if theta_rad.ndim != 1:
    raise ValueError(
        "theta_rad must be one-dimensional, but its shape is "
        f"{theta_rad.shape}."
    )

if wavelengths_nm.ndim != 1:
    raise ValueError(
        "wavelengths_nm must be one-dimensional, but its shape is "
        f"{wavelengths_nm.shape}."
    )

if differential_scattering.ndim != 2:
    raise ValueError(
        "The differential-scattering array must be two-dimensional, "
        f"but its shape is {differential_scattering.shape}."
    )


# Orient the differential array as:
#     wavelength index x theta index
expected_shape = (
    len(wavelengths_nm),
    len(theta_rad),
)

transposed_shape = (
    len(theta_rad),
    len(wavelengths_nm),
)

if differential_scattering.shape == expected_shape:

    dcs_geom = (
        differential_scattering.copy()
    )

elif differential_scattering.shape == transposed_shape:

    dcs_geom = (
        differential_scattering.T.copy()
    )

else:

    raise ValueError(
        "Differential-scattering array dimensions do not match "
        "the wavelength and theta arrays.\n"
        f"Differential shape: {differential_scattering.shape}\n"
        f"Expected: {expected_shape} or {transposed_shape}"
    )


theta_deg = np.rad2deg(
    theta_rad
)


# Make sure theta increases from 0 to 180 degrees.
theta_order = np.argsort(
    theta_deg
)

theta_deg = theta_deg[
    theta_order
]

dcs_geom = dcs_geom[
    :,
    theta_order,
]


# Make sure wavelength increases.
wavelength_order = np.argsort(
    wavelengths_nm
)

wavelengths_nm = wavelengths_nm[
    wavelength_order
]

dcs_geom = dcs_geom[
    wavelength_order,
    :,
]


# Remove nonphysical negative values.
#
# Small negative values may result from numerical noise. They are
# replaced by NaN rather than converted to positive values.
negative_mask = (
    dcs_geom < 0.0
)

number_of_negative_values = int(
    np.count_nonzero(
        negative_mask
    )
)

if number_of_negative_values > 0:

    print(
        "Negative differential-scattering values replaced by NaN:",
        number_of_negative_values,
    )

    dcs_geom[
        negative_mask
    ] = np.nan


if not np.isfinite(
    dcs_geom
).any():
    raise ValueError(
        "The differential-scattering array contains no finite values."
    )


print("\nPrepared angular-scattering data")
print("--------------------------------")
print("Theta range       :", theta_deg[0], "to", theta_deg[-1], "degrees")
print("Wavelength range  :", wavelengths_nm[0], "to", wavelengths_nm[-1], "nm")
print("Number of angles  :", len(theta_deg))
print("Number wavelengths:", len(wavelengths_nm))
print("DCS shape         :", dcs_geom.shape)
print("Finite fraction   :", np.isfinite(dcs_geom).mean())


# ============================================================
# SELECT THREE REPRESENTATIVE WAVELENGTHS
# ============================================================

if (
    "reference_peak_wavelength_nm"
    in globals()
    and np.isfinite(
        reference_peak_wavelength_nm
    )
):

    backward_peak_nm = float(
        reference_peak_wavelength_nm
    )

elif (
    "reference_backward_peak_nm"
    in globals()
    and np.isfinite(
        reference_backward_peak_nm
    )
):

    backward_peak_nm = float(
        reference_backward_peak_nm
    )

else:

    # Recalculate the peak from the reference backward spectrum.
    reference_row = select_reference_checkpoint(
        df_checkpoint_index,
        POLAR_REFERENCE_WAVELENGTH_NM,
    )

    reference_payload_local = (
        load_full_checkpoint(
            Path(
                reference_row[
                    "checkpoint_path"
                ]
            )
        )
    )

    reference_backward_spectrum = (
        extract_directional_spectrum(
            reference_payload_local,
            "backward",
        )
    )

    peak_index = int(
        np.nanargmax(
            reference_backward_spectrum[
                "spectrum"
            ]
        )
    )

    backward_peak_nm = float(
        reference_backward_spectrum[
            "wavelength_nm"
        ][
            peak_index
        ]
    )


if MANUAL_SELECTED_WAVELENGTHS_NM is None:

    requested_selected_wavelengths_nm = np.array(
        [
            backward_peak_nm
            - WAVELENGTH_OFFSET_NM,

            backward_peak_nm,

            backward_peak_nm
            + WAVELENGTH_OFFSET_NM,
        ],
        dtype=float,
    )

else:

    requested_selected_wavelengths_nm = np.asarray(
        MANUAL_SELECTED_WAVELENGTHS_NM,
        dtype=float,
    )


# Select the nearest actually calculated wavelength.
selected_wavelengths_nm = np.array(
    [
        wavelengths_nm[
            np.argmin(
                np.abs(
                    wavelengths_nm
                    - requested_wavelength_nm
                )
            )
        ]
        for requested_wavelength_nm in (
            requested_selected_wavelengths_nm
        )
    ],
    dtype=float,
)

selected_wavelengths_nm = np.unique(
    selected_wavelengths_nm
)


print(
    "\nReference backward peak:",
    f"{backward_peak_nm:.1f} nm",
)

print(
    "Requested selected wavelengths:",
    requested_selected_wavelengths_nm,
)

print(
    "Actual selected wavelengths:",
    selected_wavelengths_nm,
)


# ============================================================
# SAFE OUTPUT NAME
# ============================================================

safe_case_name = "".join(
    character
    if character.isalnum()
    else "_"
    for character in selected_case_label
)

while "__" in safe_case_name:
    safe_case_name = safe_case_name.replace(
        "__",
        "_",
    )

safe_case_name = safe_case_name.strip(
    "_"
).lower()


# ============================================================
# 1. POLAR PLOT WITH ALL WAVELENGTHS
# ============================================================

if PLOT_ALL_WAVELENGTH_POLAR:

    plt.close("all")

    polar_all_result = (
        plot_polar_differential_scattering(
            theta_deg,
            wavelengths_nm,
            dcs_geom,
            plot_all=True,
            y_mode=Y_MODE,
        )
    )

    current_figure = plt.gcf()

    current_figure.suptitle(
        (
            selected_case_label
            + "\nall calculated wavelengths"
        ),
        fontsize=10,
    )

    if SAVE_SCATTERING_PLOTS:

        polar_all_png = (
            SCATTERING_PLOT_OUTPUT_DIRECTORY
            / (
                "polar_all_wavelengths_"
                + safe_case_name
                + ".png"
            )
        )

        polar_all_pdf = (
            SCATTERING_PLOT_OUTPUT_DIRECTORY
            / (
                "polar_all_wavelengths_"
                + safe_case_name
                + ".pdf"
            )
        )

        current_figure.savefig(
            polar_all_png,
            dpi=SCATTERING_PLOT_DPI,
            transparent=False,
            bbox_inches="tight",
        )

        current_figure.savefig(
            polar_all_pdf,
            transparent=False,
            bbox_inches="tight",
        )

        print(
            "\nSaved:",
            polar_all_png,
        )

        print(
            "Saved:",
            polar_all_pdf,
        )

    plt.show()


# ============================================================
# 2. POLAR PLOT AT THE THREE SELECTED WAVELENGTHS
# ============================================================

if PLOT_SELECTED_WAVELENGTH_POLAR:

    plt.close("all")

    polar_selected_result = (
        plot_polar_differential_scattering(
            theta_deg,
            wavelengths_nm,
            dcs_geom,
            plot_all=False,
            selected_wavelengths_nm=(
                selected_wavelengths_nm.tolist()
            ),
            show_legend=(
                SHOW_SELECTED_WAVELENGTH_LEGEND
            ),
            y_mode=Y_MODE,
        )
    )

    current_figure = plt.gcf()

    current_figure.suptitle(
        (
            selected_case_label
            + "\n"
            + (
                "reference backward peak "
                r"and $\pm 150$ nm"
            )
        ),
        fontsize=10,
    )

    if SAVE_SCATTERING_PLOTS:

        polar_selected_png = (
            SCATTERING_PLOT_OUTPUT_DIRECTORY
            / (
                "polar_selected_wavelengths_"
                + safe_case_name
                + ".png"
            )
        )

        polar_selected_pdf = (
            SCATTERING_PLOT_OUTPUT_DIRECTORY
            / (
                "polar_selected_wavelengths_"
                + safe_case_name
                + ".pdf"
            )
        )

        current_figure.savefig(
            polar_selected_png,
            dpi=SCATTERING_PLOT_DPI,
            transparent=False,
            bbox_inches="tight",
        )

        current_figure.savefig(
            polar_selected_pdf,
            transparent=False,
            bbox_inches="tight",
        )

        print(
            "\nSaved:",
            polar_selected_png,
        )

        print(
            "Saved:",
            polar_selected_pdf,
        )

    plt.show()


# ============================================================
# 3. DIFFERENTIAL SCATTERING VERSUS THETA
# ============================================================

if PLOT_DIFFERENTIAL_VS_THETA:

    plt.close("all")

    angular_result = (
        plot_differential_vs_theta(
            theta_deg,
            wavelengths_nm,
            dcs_geom,
            plot_all=True,
            y_mode=Y_MODE,
        )
    )

    current_figure = plt.gcf()

    current_figure.suptitle(
        (
            selected_case_label
            + "\ndifferential scattering versus angle"
        ),
        fontsize=10,
    )

    if SAVE_SCATTERING_PLOTS:

        angular_png = (
            SCATTERING_PLOT_OUTPUT_DIRECTORY
            / (
                "differential_vs_theta_"
                + safe_case_name
                + ".png"
            )
        )

        angular_pdf = (
            SCATTERING_PLOT_OUTPUT_DIRECTORY
            / (
                "differential_vs_theta_"
                + safe_case_name
                + ".pdf"
            )
        )

        current_figure.savefig(
            angular_png,
            dpi=SCATTERING_PLOT_DPI,
            transparent=False,
            bbox_inches="tight",
        )

        current_figure.savefig(
            angular_pdf,
            transparent=False,
            bbox_inches="tight",
        )

        print(
            "\nSaved:",
            angular_png,
        )

        print(
            "Saved:",
            angular_pdf,
        )

    plt.show()

# For presentation selective comparison

# In[88]:


# ============================================================
# SELECT PRESENTATION CASES
#
# Cases:
#   1. Nonabsorbing 580 nm reference
#   2. Core absorber, k = 1, approximately constant diameter
#   3. Core absorber, k = 1, constant shell number
#
# By default, the largest core-thickness factor present in both
# core scenarios is selected.
#
# Manual selection can be enabled below.
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd

from IPython.display import display


# ============================================================
# USER SETTINGS
# ============================================================

PRESENTATION_REFERENCE_WAVELENGTH_NM = 580.0
PRESENTATION_ABSORBER_K = 1.0

# If None, select the largest core-thickness factor available in
# both scenarios.
MANUAL_COMMON_CORE_THICKNESS_FACTOR = None

# Alternatively, allow different core factors in the two scenarios.
# Set USE_SEPARATE_MANUAL_CORE_FACTORS = True to enable.
USE_SEPARATE_MANUAL_CORE_FACTORS = False

MANUAL_CORE_FACTOR_CONSTANT_DIAMETER = 20.0
MANUAL_CORE_FACTOR_CONSTANT_SHELL_NUMBER = 20.0


# ============================================================
# NORMALIZE CHECKPOINT METADATA
# ============================================================

df_checkpoint_selection = (
    df_checkpoint_index.copy()
)

for column in [
    "reference_design_wavelength_nm",
    "absorber_k",
    "core_thickness_factor",
    "n_shells",
    "predicted_diameter_nm",
]:

    if column in df_checkpoint_selection.columns:

        df_checkpoint_selection[
            column
        ] = pd.to_numeric(
            df_checkpoint_selection[column],
            errors="coerce",
        )


for column in [
    "case_family",
    "scenario",
]:

    if column in df_checkpoint_selection.columns:

        df_checkpoint_selection[
            f"{column}_clean"
        ] = (
            df_checkpoint_selection[column]
            .astype(str)
            .str.strip()
            .str.lower()
            .str.replace(
                " ",
                "_",
                regex=False,
            )
            .str.replace(
                "-",
                "_",
                regex=False,
            )
        )


# ============================================================
# SELECT NONABSORBING REFERENCE
# ============================================================

reference_mask_presentation = (
    np.isclose(
        df_checkpoint_selection[
            "reference_design_wavelength_nm"
        ],
        PRESENTATION_REFERENCE_WAVELENGTH_NM,
    )
    & df_checkpoint_selection[
        "case_family_clean"
    ].eq("reference")
)

reference_candidates = (
    df_checkpoint_selection.loc[
        reference_mask_presentation
    ]
)

if len(reference_candidates) != 1:

    display(reference_candidates)

    raise ValueError(
        "Expected exactly one nonabsorbing reference checkpoint "
        f"for {PRESENTATION_REFERENCE_WAVELENGTH_NM:.0f} nm, "
        f"but found {len(reference_candidates)}."
    )

presentation_reference_row = (
    reference_candidates.iloc[0]
)


# ============================================================
# FIND AVAILABLE CORE FACTORS
# ============================================================

core_base_mask = (
    np.isclose(
        df_checkpoint_selection[
            "reference_design_wavelength_nm"
        ],
        PRESENTATION_REFERENCE_WAVELENGTH_NM,
    )
    & df_checkpoint_selection[
        "case_family_clean"
    ].eq("core")
    & np.isclose(
        df_checkpoint_selection[
            "absorber_k"
        ],
        PRESENTATION_ABSORBER_K,
    )
)


constant_diameter_candidates = (
    df_checkpoint_selection.loc[
        core_base_mask
        & df_checkpoint_selection[
            "scenario_clean"
        ].eq("constant_diameter")
    ]
    .copy()
)

constant_shell_candidates = (
    df_checkpoint_selection.loc[
        core_base_mask
        & df_checkpoint_selection[
            "scenario_clean"
        ].eq("constant_shell_number")
    ]
    .copy()
)


if constant_diameter_candidates.empty:

    raise ValueError(
        "No approximately constant-diameter core cases were found "
        f"for k={PRESENTATION_ABSORBER_K:g}."
    )

if constant_shell_candidates.empty:

    raise ValueError(
        "No constant-shell-number core cases were found "
        f"for k={PRESENTATION_ABSORBER_K:g}."
    )


available_constant_diameter_factors = np.sort(
    constant_diameter_candidates[
        "core_thickness_factor"
    ]
    .dropna()
    .unique()
)

available_constant_shell_factors = np.sort(
    constant_shell_candidates[
        "core_thickness_factor"
    ]
    .dropna()
    .unique()
)

common_core_factors = np.intersect1d(
    available_constant_diameter_factors,
    available_constant_shell_factors,
)


print(
    "Constant-diameter core factors:",
    available_constant_diameter_factors,
)

print(
    "Constant-shell-number core factors:",
    available_constant_shell_factors,
)

print(
    "Common core factors:",
    common_core_factors,
)


# ============================================================
# CHOOSE CORE FACTORS
# ============================================================

if USE_SEPARATE_MANUAL_CORE_FACTORS:

    selected_constant_diameter_factor = float(
        MANUAL_CORE_FACTOR_CONSTANT_DIAMETER
    )

    selected_constant_shell_factor = float(
        MANUAL_CORE_FACTOR_CONSTANT_SHELL_NUMBER
    )

else:

    if common_core_factors.size == 0:

        raise ValueError(
            "The two scenarios have no common core-thickness factor. "
            "Enable USE_SEPARATE_MANUAL_CORE_FACTORS."
        )

    if MANUAL_COMMON_CORE_THICKNESS_FACTOR is None:

        selected_common_factor = float(
            np.nanmax(common_core_factors)
        )

    else:

        selected_common_factor = float(
            MANUAL_COMMON_CORE_THICKNESS_FACTOR
        )

        if not np.any(
            np.isclose(
                common_core_factors,
                selected_common_factor,
            )
        ):

            raise ValueError(
                "The requested common core factor is unavailable.\n"
                f"Requested: {selected_common_factor:g}\n"
                f"Available: {common_core_factors}"
            )

    selected_constant_diameter_factor = (
        selected_common_factor
    )

    selected_constant_shell_factor = (
        selected_common_factor
    )


# ============================================================
# SELECT UNIQUE CORE CHECKPOINTS
# ============================================================

def select_unique_core_checkpoint(
    candidates,
    *,
    core_factor,
    scenario_label,
):
    """
    Select exactly one core checkpoint at a requested factor.
    """

    selected = candidates.loc[
        np.isclose(
            candidates[
                "core_thickness_factor"
            ],
            float(core_factor),
        )
    ]

    if len(selected) != 1:

        display(selected)

        raise ValueError(
            f"Expected one {scenario_label} core checkpoint at "
            f"factor {core_factor:g}, found {len(selected)}."
        )

    return selected.iloc[0]


presentation_constant_diameter_row = (
    select_unique_core_checkpoint(
        constant_diameter_candidates,
        core_factor=(
            selected_constant_diameter_factor
        ),
        scenario_label="constant-diameter",
    )
)

presentation_constant_shell_row = (
    select_unique_core_checkpoint(
        constant_shell_candidates,
        core_factor=(
            selected_constant_shell_factor
        ),
        scenario_label="constant-shell-number",
    )
)


# ============================================================
# LOAD FULL CHECKPOINTS
# ============================================================

presentation_cases = [
    {
        "key": "reference",
        "label": "Nonabsorbing reference",
        "short_label": "Reference",
        "colour": "black",
        "linestyle": "-",
        "linewidth": 1.8,
        "row": presentation_reference_row,
    },
    {
        "key": "constant_diameter",
        "label": (
            "Core absorber, approximately constant diameter\n"
            rf"$k={PRESENTATION_ABSORBER_K:g}$, "
            rf"$\gamma_\mathrm{{c}}="
            f"{selected_constant_diameter_factor:g}$"
        ),
        "short_label": (
            "Core absorber\n"
            "constant diameter"
        ),
        "colour": "#0072B2",
        "linestyle": "--",
        "linewidth": 1.5,
        "row": presentation_constant_diameter_row,
    },
    {
        "key": "constant_shell_number",
        "label": (
            "Core absorber, constant shell number\n"
            rf"$k={PRESENTATION_ABSORBER_K:g}$, "
            rf"$\gamma_\mathrm{{c}}="
            f"{selected_constant_shell_factor:g}$"
        ),
        "short_label": (
            "Core absorber\n"
            "constant shell number"
        ),
        "colour": "#D55E00",
        "linestyle": "-.",
        "linewidth": 1.5,
        "row": presentation_constant_shell_row,
    },
]


for case in presentation_cases:

    case["payload"] = load_full_checkpoint(
        Path(
            case["row"][
                "checkpoint_path"
            ]
        )
    )

    case["spectra"] = {
        direction: extract_directional_spectrum(
            case["payload"],
            direction,
        )
        for direction in [
            "backward",
            "forward",
        ]
    }


# ============================================================
# CASE SUMMARY
# ============================================================

presentation_case_summary = []

for case in presentation_cases:

    row = case["row"]

    presentation_case_summary.append(
        {
            "case": case["key"],
            "label": case["label"],
            "core_thickness_factor": (
                pd.to_numeric(
                    pd.Series(
                        [
                            row.get(
                                "core_thickness_factor",
                                np.nan,
                            )
                        ]
                    ),
                    errors="coerce",
                ).iloc[0]
            ),
            "n_shells": (
                pd.to_numeric(
                    pd.Series(
                        [
                            row.get(
                                "n_shells",
                                np.nan,
                            )
                        ]
                    ),
                    errors="coerce",
                ).iloc[0]
            ),
            "predicted_diameter_um": (
                pd.to_numeric(
                    pd.Series(
                        [
                            row.get(
                                "predicted_diameter_nm",
                                np.nan,
                            )
                        ]
                    ),
                    errors="coerce",
                ).iloc[0]
                / 1000.0
            ),
            "absorber_k": (
                pd.to_numeric(
                    pd.Series(
                        [
                            row.get(
                                "absorber_k",
                                0.0,
                            )
                        ]
                    ),
                    errors="coerce",
                ).iloc[0]
            ),
            "checkpoint": Path(
                row["checkpoint_path"]
            ).name,
        }
    )


df_presentation_cases = pd.DataFrame(
    presentation_case_summary
)

display(
    df_presentation_cases.style.format(
        {
            "core_thickness_factor": "{:.1f}",
            "n_shells": "{:.0f}",
            "predicted_diameter_um": "{:.3f}",
            "absorber_k": "{:.3g}",
        }
    )
)

# Cell 2: Backward and forward spectra for the three selected cases

# In[89]:


# ============================================================
# PRESENTATION FIGURE:
# BACKWARD AND FORWARD COLLECTED SPECTRA
#
# Upper panel:
#   backward-collected efficiency
#
# Lower panel:
#   forward-collected efficiency
#
# Cases:
#   nonabsorbing reference
#   core absorber at approximately constant diameter
#   core absorber at constant shell number
# ============================================================

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ============================================================
# SETTINGS
# ============================================================

PRESENTATION_SPECTRA_WIDTH_CM = 13.5
PRESENTATION_SPECTRA_HEIGHT_CM = 10.0

PRESENTATION_XLIM_NM = (
    380.0,
    780.0,
)

# Use None for percentile-based selection.
PRESENTATION_BACKWARD_YLIM = None
PRESENTATION_FORWARD_YLIM = None

PRESENTATION_Y_PERCENTILE = 0.995
PRESENTATION_Y_HEADROOM = 1.08

SAVE_PRESENTATION_SPECTRA = True

PRESENTATION_SPECTRA_BASENAME = (
    "presentation_580nm_core_absorber_spectra"
)


# ============================================================
# DETERMINE Y LIMITS
# ============================================================

presentation_spectra_by_direction = {
    direction: [
        case["spectra"][direction]["spectrum"]
        for case in presentation_cases
    ]
    for direction in [
        "backward",
        "forward",
    ]
}


if PRESENTATION_BACKWARD_YLIM is None:

    backward_ylim = calculate_percentile_ylim(
        presentation_spectra_by_direction["backward"],
        percentile=PRESENTATION_Y_PERCENTILE,
        headroom_factor=PRESENTATION_Y_HEADROOM,
        start_at_zero=True,
    )

else:

    backward_ylim = tuple(
        map(
            float,
            PRESENTATION_BACKWARD_YLIM,
        )
    )


if PRESENTATION_FORWARD_YLIM is None:

    forward_ylim = calculate_percentile_ylim(
        presentation_spectra_by_direction["forward"],
        percentile=PRESENTATION_Y_PERCENTILE,
        headroom_factor=PRESENTATION_Y_HEADROOM,
        start_at_zero=True,
    )

else:

    forward_ylim = tuple(
        map(
            float,
            PRESENTATION_FORWARD_YLIM,
        )
    )


print(
    "Backward y limits:",
    backward_ylim,
)

print(
    "Forward y limits:",
    forward_ylim,
)


# ============================================================
# CREATE FIGURE
# ============================================================

cm_to_inches = 1.0 / 2.54

fig, axes = plt.subplots(
    2,
    1,
    figsize=(
        PRESENTATION_SPECTRA_WIDTH_CM * cm_to_inches,
        PRESENTATION_SPECTRA_HEIGHT_CM * cm_to_inches,
    ),
    sharex=True,
    constrained_layout=True,
)


for axis_index, direction in enumerate(
    [
        "backward",
        "forward",
    ]
):

    ax = axes[axis_index]

    for case in presentation_cases:

        extracted = case["spectra"][direction]

        ax.plot(
            extracted["wavelength_nm"],
            extracted["spectrum"],
            color=case["colour"],
            linestyle=case["linestyle"],
            linewidth=case["linewidth"],
            label=case["label"],
            zorder=(
                5
                if case["key"] == "reference"
                else 3
            ),
        )

    if direction == "backward":

        ax.set_ylabel(
            r"$Q_{\mathrm{back}}$"
        )

        ax.set_ylim(
            *backward_ylim
        )

        ax.set_title(
            "Backward-collected scattering"
        )

    else:

        ax.set_ylabel(
            r"$Q_{\mathrm{forward}}$"
        )

        ax.set_ylim(
            *forward_ylim
        )

        ax.set_title(
            "Forward-collected scattering"
        )

    ax.set_xlim(
        *PRESENTATION_XLIM_NM
    )

    ax.grid(
        False
    )


axes[-1].set_xlabel(
    "Wavelength [nm]"
)


# ============================================================
# LEGEND
# ============================================================

presentation_handles = [
    Line2D(
        [0],
        [0],
        color=case["colour"],
        linestyle=case["linestyle"],
        linewidth=case["linewidth"],
        label=case["label"],
    )
    for case in presentation_cases
]

fig.legend(
    handles=presentation_handles,
    loc="outside lower center",
    ncol=1,
    frameon=True,
    fontsize=8,
)


fig.suptitle(
    (
        r"$\lambda_0=580$ nm: "
        "effect of a strongly absorbing core"
    ),
    fontsize=10,
)


# ============================================================
# SAVE
# ============================================================

if SAVE_PRESENTATION_SPECTRA:

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    spectra_png = (
        FIGURE_DIR
        / (
            PRESENTATION_SPECTRA_BASENAME
            + ".png"
        )
    )

    spectra_pdf = (
        FIGURE_DIR
        / (
            PRESENTATION_SPECTRA_BASENAME
            + ".pdf"
        )
    )

    fig.savefig(
        spectra_png,
        dpi=PNG_DPI,
        transparent=True,
        bbox_inches="tight",
    )

    fig.savefig(
        spectra_pdf,
        transparent=True,
        bbox_inches="tight",
    )

    print("Saved:", spectra_png)
    print("Saved:", spectra_pdf)


plt.show()

# Cell 3: Colour matrix for backward and forward spectra
#
# This cell uses the already-calculated hex_backward and hex_forward values from df_paired. The colour patches therefore represent the CIELAB-to-sRGB conversion of the actual unnormalized spectra.

# In[ ]:


# # ---------------------------------------------------------------------------
# # FIGURE SETUP
# # ---------------------------------------------------------------------------
# COLOUR_FIGURE_WIDTH_CM = 17.5
# COLOUR_FIGURE_HEIGHT_CM = 6.4

# from matplotlib.patches import Circle
# with mpl.rc_context({
#     "figure.constrained_layout.use": False,
#     "figure.autolayout": False,
# }):

#     fig = plt.figure(
#         figsize=(
#             COLOUR_FIGURE_WIDTH_CM / 2.54,
#             COLOUR_FIGURE_HEIGHT_CM / 2.54,
#         )
#     )

#     gs = fig.add_gridspec(
#         1,
#         2,
#         width_ratios=[0.95, 1.35],
#         left=0.055,
#         right=0.985,
#         bottom=0.14,
#         top=0.86,
#         wspace=0.28,
#     )

#     ax_colour = fig.add_subplot(gs[0, 0])
#     ax_ab = fig.add_subplot(gs[0, 1])

#     # =======================================================================
#     # LEFT PANEL — DIRECTIONAL COLOURS
#     # =======================================================================

#     n_cases = len(case_detail_df)

#     ax_colour.set_xlim(0.0, 3.8)
#     ax_colour.set_ylim(-0.2, n_cases + 0.8)
#     ax_colour.axis("off")

#     ax_colour.set_title(
#         "Calculated directional colours",
#         fontsize=9,
#         fontweight="semibold",
#         pad=12,
#     )

#     ax_colour.text(
#         2.00,
#         n_cases + 0.22,
#         "Backward",
#         ha="center",
#         va="bottom",
#         fontsize=8,
#     )

#     ax_colour.text(
#         3.15,
#         n_cases + 0.22,
#         "Forward",
#         ha="center",
#         va="bottom",
#         fontsize=8,
#     )

#     x_backward = 2.00
#     x_forward = 3.15

#     patch_width = 0.74
#     patch_height = 0.58

#     for i, record in case_detail_df.iterrows():

#         y = n_cases - 1 - i + 0.35

#         # ---------------------------------------------------------------
#         # sample number
#         # ---------------------------------------------------------------

#         ax_colour.text(
#             0.15,
#             y,
#             f"{i+1}",
#             fontsize=9,
#             fontweight="bold",
#             va="center",
#         )

#         # ax_colour.text(
#         #     0.45,
#         #     y,
#         #     record["case"],
#         #     fontsize=8,
#         #     va="center",
#         # )

#         ax_colour.set_xlim(0.0, 4.8)

#         x_backward = 3.0
#         x_forward  = 4.1

#         case_label = (
#             record["case"]
#             .replace("constant diameter", "\nconstant diameter")
#             .replace("constant shell number", "\nconstant shell\nnumber")
#         )

#         ax_colour.text(
#             0.45,
#             y,
#             case_label,
#             fontsize=8,
#             va="center",
#         )
#         # ---------------------------------------------------------------
#         # guide line
#         # ---------------------------------------------------------------

#         ax_colour.plot(
#             [
#                 x_backward + patch_width / 2,
#                 x_forward - patch_width / 2,
#             ],
#             [y, y],
#             color="0.85",
#             linewidth=0.8,
#             zorder=0,
#         )

#         # ---------------------------------------------------------------
#         # backward swatch
#         # ---------------------------------------------------------------

#         ax_colour.add_patch(
#             Rectangle(
#                 (
#                     x_backward - patch_width / 2,
#                     y - patch_height / 2,
#                 ),
#                 patch_width,
#                 patch_height,
#                 facecolor=record["hex_backward"],
#                 edgecolor="black",
#                 linewidth=0.8,
#                 zorder=2,
#             )
#         )

#         # ---------------------------------------------------------------
#         # forward swatch
#         # ---------------------------------------------------------------

#         ax_colour.add_patch(
#             Rectangle(
#                 (
#                     x_forward - patch_width / 2,
#                     y - patch_height / 2,
#                 ),
#                 patch_width,
#                 patch_height,
#                 facecolor=record["hex_forward"],
#                 edgecolor="black",
#                 linewidth=0.8,
#                 zorder=2,
#             )
#         )

#     # =======================================================================
#     # RIGHT PANEL — CIELAB a*-b* PLANE
#     # =======================================================================

#     a_backward = case_detail_df["a_backward"].to_numpy(dtype=float)
#     a_forward = case_detail_df["a_forward"].to_numpy(dtype=float)

#     b_backward = case_detail_df["b_backward"].to_numpy(dtype=float)
#     b_forward = case_detail_df["b_forward"].to_numpy(dtype=float)

#     all_a = np.concatenate([a_backward, a_forward])
#     all_b = np.concatenate([b_backward, b_forward])

#     finite = np.isfinite(all_a) & np.isfinite(all_b)

#     if not np.any(finite):
#         raise ValueError(
#             "No finite CIELAB a*-b* coordinates were available."
#         )

#     # -----------------------------------------------------------------------
#     # LIMITS
#     # -----------------------------------------------------------------------

#     if MANUAL_A_LIMITS is not None:
#         a_limits = tuple(MANUAL_A_LIMITS)
#     else:
#         a_limits = None

#     if MANUAL_B_LIMITS is not None:
#         b_limits = tuple(MANUAL_B_LIMITS)
#     else:
#         b_limits = None

#     if a_limits is None or b_limits is None:

#         minimum = min(
#             np.nanmin(all_a[finite]),
#             np.nanmin(all_b[finite]),
#         )

#         maximum = max(
#             np.nanmax(all_a[finite]),
#             np.nanmax(all_b[finite]),
#         )

#         span = maximum - minimum

#         if span <= 0:
#             span = 1.0

#         centre = 0.5 * (minimum + maximum)

#         half_range = 0.5 * span * (1.0 + AB_LIMIT_HEADROOM)

#         automatic_limits = (
#             centre - half_range,
#             centre + half_range,
#         )

#         if a_limits is None:
#             a_limits = automatic_limits

#         if b_limits is None:
#             b_limits = automatic_limits

#     ax_ab.set_xlim(a_limits)
#     ax_ab.set_ylim(b_limits)

#     ax_ab.set_aspect("equal")

#     # -----------------------------------------------------------------------
#     # GRID + AXES
#     # -----------------------------------------------------------------------

#     ax_ab.grid(
#         True,
#         linestyle="--",
#         linewidth=0.5,
#         alpha=0.35,
#         zorder=0,
#     )

#     ax_ab.axhline(
#         0,
#         color="0.45",
#         linewidth=0.9,
#         zorder=1,
#     )

#     ax_ab.axvline(
#         0,
#         color="0.45",
#         linewidth=0.9,
#         zorder=1,
#     )

#     # -----------------------------------------------------------------------
#     # CHROMA CIRCLES
#     # -----------------------------------------------------------------------

#     max_radius = np.ceil(
#         np.max(np.sqrt(all_a[finite] ** 2 + all_b[finite] ** 2))
#     )

#     for r in np.arange(10, max_radius + 10, 10):

#         ax_ab.add_patch(
#             Circle(
#                 (0, 0),
#                 r,
#                 fill=False,
#                 linestyle=":",
#                 linewidth=0.5,
#                 color="0.75",
#                 zorder=0,
#             )
#         )

#     ax_ab.set_xlabel(r"$a^*$")
#     ax_ab.set_ylabel(r"$b^*$")

#     ax_ab.set_title(
#         r"CIELAB $a^*$-$b^*$ plane",
#         fontsize=9,
#         fontweight="semibold",
#         pad=8,
#     )

#     # -----------------------------------------------------------------------
#     # TRAJECTORIES
#     # -----------------------------------------------------------------------

#     trajectory_colours = [
#         "#1f77b4",
#         "#d62728",
#         "#2ca02c",
#         "#9467bd",
#         "#ff7f0e",
#         "#8c564b",
#     ]

#     for i, record in case_detail_df.iterrows():

#         a_b = record["a_backward"]
#         b_b = record["b_backward"]

#         a_f = record["a_forward"]
#         b_f = record["b_forward"]

#         if not all(np.isfinite([a_b, b_b, a_f, b_f])):
#             continue

#         traj_color = trajectory_colours[
#             i % len(trajectory_colours)
#         ]

#         # ---------------------------------------------------------------
#         # coloured directional arrow
#         # ---------------------------------------------------------------

#         ax_ab.annotate(
#             "",
#             xy=(a_f, b_f),
#             xytext=(a_b, b_b),
#             arrowprops=dict(
#                 arrowstyle="-|>",
#                 color=traj_color,
#                 linewidth=1.6,
#                 alpha=0.85,
#                 shrinkA=6,
#                 shrinkB=6,
#                 mutation_scale=12,
#             ),
#             zorder=2,
#         )

#         # ---------------------------------------------------------------
#         # backward point
#         # ---------------------------------------------------------------

#         ax_ab.scatter(
#             a_b,
#             b_b,
#             s=AB_MARKER_SIZE * 1.25,
#             marker=BACKWARD_MARKER,
#             facecolor=record["hex_backward"],
#             edgecolor="black",
#             linewidth=AB_MARKER_EDGE_WIDTH,
#             zorder=4,
#         )

#         # ---------------------------------------------------------------
#         # forward point
#         # ---------------------------------------------------------------

#         ax_ab.scatter(
#             a_f,
#             b_f,
#             s=AB_MARKER_SIZE * 1.25,
#             marker=FORWARD_MARKER,
#             facecolor=record["hex_forward"],
#             edgecolor="black",
#             linewidth=AB_MARKER_EDGE_WIDTH,
#             zorder=4,
#         )

#         # ---------------------------------------------------------------
#         # number inside marker
#         # ---------------------------------------------------------------

#         ax_ab.text(
#             a_b,
#             b_b,
#             f"{i+1}",
#             color="white",
#             fontsize=6,
#             fontweight="bold",
#             ha="center",
#             va="center",
#             zorder=5,
#         )

#     # =======================================================================
#     # LEGEND
#     # =======================================================================

#     legend_handles = [

#         Line2D(
#             [0],
#             [0],
#             marker=BACKWARD_MARKER,
#             linestyle="None",
#             markerfacecolor="white",
#             markeredgecolor="black",
#             markersize=6.5,
#             label="Backward",
#         ),

#         Line2D(
#             [0],
#             [0],
#             marker=FORWARD_MARKER,
#             linestyle="None",
#             markerfacecolor="white",
#             markeredgecolor="black",
#             markersize=6.5,
#             label="Forward",
#         ),

#         Line2D(
#             [0],
#             [0],
#             color="black",
#             linewidth=1.4,
#             label=r"Backward $\rightarrow$ Forward",
#         ),
#     ]

#     ax_ab.legend(
#         handles=legend_handles,
#         loc="upper left",
#         frameon=False,
#         fontsize=7.5,
#         handletextpad=0.5,
#     )

#     # =======================================================================
#     # FIGURE TITLE
#     # =======================================================================

#     fig.suptitle(
#         r"$\lambda_0 = 580\,\mathrm{nm}$",
#         fontsize=10,
#         y=0.96,
#     )

#     # =======================================================================
#     # SAVE
#     # =======================================================================

#     if SAVE_COLOUR_FIGURE:

#         output_png = FIGURE_DIR / (
#             f"{COLOUR_FIGURE_BASENAME}.png"
#         )

#         output_pdf = FIGURE_DIR / (
#             f"{COLOUR_FIGURE_BASENAME}.pdf"
#         )

#         fig.savefig(
#             output_png,
#             dpi=PNG_DPI,
#             bbox_inches="tight",
#         )

#         fig.savefig(
#             output_pdf,
#             bbox_inches="tight",
#         )

#         print("\nSaved:")
#         print(f"  {output_png}")
#         print(f"  {output_pdf}")

#     plt.show()

# Cell 4: Select cases for the polar plots
#
# By default, the polar plots use the same three cases. You can instead select only two cases or manually define other core factors.

# In[95]:


# ============================================================
# SELECT CASES FOR POLAR DIFFERENTIAL-SCATTERING PLOTS
#
# Defaults:
#   reference
#   core absorber, approximately constant diameter
#   core absorber, constant shell number
#
# Set POLAR_SELECTED_CASE_KEYS to any subset of:
#   "reference"
#   "constant_diameter"
#   "constant_shell_number"
# ============================================================

POLAR_SELECTED_CASE_KEYS = [
    "reference",
    "constant_diameter",
    "constant_shell_number",
]

# Example with only two cases:
#
# POLAR_SELECTED_CASE_KEYS = [
#     "reference",
#     "constant_diameter",
# ]


presentation_polar_cases = [
    case
    for case in presentation_cases
    if case["key"] in POLAR_SELECTED_CASE_KEYS
]


if len(presentation_polar_cases) not in [
    2,
    3,
]:

    raise ValueError(
        "Select exactly two or three polar-plot cases."
    )


print(
    "Polar cases:"
)

for case in presentation_polar_cases:

    print(
        "  ",
        case["label"],
    )

# Cell 5: Three polar plots at the reference peak and ±150 nm
#
# The peak wavelength is determined from the backward-collected spectrum of the nonabsorbing reference.

# In[ ]:


# ============================================================
# POLAR DIFFERENTIAL-SCATTERING COMPARISON
#
# Panels:
#   reference backward peak - 150 nm
#   reference backward peak
#   reference backward peak + 150 nm
#
# Curves:
#   two or three manually selected presentation cases
#
# Convention:
#   0 degrees   = exact forward scattering
#   180 degrees = exact backward scattering
# ============================================================

import matplotlib.pyplot as plt


# ============================================================
# SETTINGS
# ============================================================

POLAR_WAVELENGTH_OFFSET_NM = 150.0

# "absolute":
#     use the calculated differential quantity directly
#
# "panel":
#     normalize all curves in each wavelength panel by the largest
#     value within that panel
#
# "curve":
#     normalize each individual curve to its own maximum
POLAR_NORMALIZATION_MODE = "absolute"

POLAR_LOGARITHMIC_RADIUS = True

POLAR_MINIMUM_POSITIVE_FRACTION = 1e-8

POLAR_WIDTH_CM = 15.92
POLAR_HEIGHT_CM = 7

POLAR_LINE_WIDTH = 0.7

SAVE_PRESENTATION_POLAR = True

PRESENTATION_POLAR_BASENAME = (
    "presentation_580nm_core_absorber_polar"
)


# ============================================================
# EXTRACT REFERENCE PEAK
# ============================================================

reference_case = next(
    case
    for case in presentation_cases
    if case["key"] == "reference"
)

reference_backward = (
    reference_case["spectra"]["backward"]
)

reference_peak_index = int(
    np.nanargmax(
        reference_backward["spectrum"]
    )
)

reference_peak_wavelength_nm = float(
    reference_backward["wavelength_nm"][
        reference_peak_index
    ]
)


polar_target_wavelengths_nm = np.array(
    [
        reference_peak_wavelength_nm
        - POLAR_WAVELENGTH_OFFSET_NM,

        reference_peak_wavelength_nm,

        reference_peak_wavelength_nm
        + POLAR_WAVELENGTH_OFFSET_NM,
    ],
    dtype=float,
)


reference_wavelength_minimum = float(
    np.nanmin(
        reference_backward["wavelength_nm"]
    )
)

reference_wavelength_maximum = float(
    np.nanmax(
        reference_backward["wavelength_nm"]
    )
)

polar_target_wavelengths_nm = np.clip(
    polar_target_wavelengths_nm,
    reference_wavelength_minimum,
    reference_wavelength_maximum,
)


print(
    "Reference backward peak wavelength:",
    f"{reference_peak_wavelength_nm:.1f} nm",
)

print(
    "Polar target wavelengths:",
    polar_target_wavelengths_nm,
)


# ============================================================
# EXTRACT ANGULAR ARRAYS
# ============================================================

polar_array_cases = []

selected_quantity_types = set()
selected_differential_paths = set()
selected_theta_paths = set()


for case in presentation_polar_cases:

    # The full angular scattering solution should be available
    # in either directional SweepResult. Use the backward object
    # consistently.
    sweep_result = (
        case["payload"]["backward"]
    )

    wavelength_nm, wavelength_path = (
        select_wavelength_array(
            sweep_result,
            root_name=case["key"],
        )
    )

    theta_rad, theta_path = (
        select_theta_array(
            sweep_result,
            root_name=case["key"],
        )
    )

    (
        differential_array,
        differential_path,
        differential_quantity_type,
    ) = select_differential_array(
        sweep_result,
        root_name=case["key"],
        n_wavelengths=len(wavelength_nm),
        n_angles=len(theta_rad),
    )

    selected_quantity_types.add(
        differential_quantity_type
    )

    selected_differential_paths.add(
        differential_path
    )

    selected_theta_paths.add(
        theta_path
    )

    polar_array_cases.append(
        {
            **case,
            "wavelength_nm": wavelength_nm,
            "wavelength_path": wavelength_path,
            "theta_rad": theta_rad,
            "theta_path": theta_path,
            "differential": differential_array,
            "differential_path": differential_path,
            "differential_quantity_type": (
                differential_quantity_type
            ),
        }
    )


print(
    "\nSelected differential quantity types:",
    selected_quantity_types,
)

print(
    "\nSelected differential paths:"
)

for path in sorted(
    selected_differential_paths
):

    print(
        "  ",
        path,
    )


# ============================================================
# PREPARE CURVES
# ============================================================

polar_panel_curves = []


for target_wavelength_nm in (
    polar_target_wavelengths_nm
):

    panel_curves = []

    for case in polar_array_cases:

        wavelength_index = int(
            np.nanargmin(
                np.abs(
                    case["wavelength_nm"]
                    - target_wavelength_nm
                )
            )
        )

        actual_wavelength_nm = float(
            case["wavelength_nm"][
                wavelength_index
            ]
        )

        radial_values = as_real_numeric_array(
            case["differential"][
                wavelength_index,
                :
            ]
        )

        if radial_values is None:

            raise TypeError(
                "The selected differential-scattering array "
                "is complex or nonnumeric."
            )

        radial_values = np.asarray(
            radial_values,
            dtype=float,
        )

        # Remove negative numerical artefacts.
        radial_values[
            radial_values < 0.0
        ] = np.nan

        panel_curves.append(
            {
                "case": case,
                "actual_wavelength_nm": (
                    actual_wavelength_nm
                ),
                "radial_values": radial_values,
            }
        )

    polar_panel_curves.append(
        {
            "target_wavelength_nm": (
                target_wavelength_nm
            ),
            "curves": panel_curves,
        }
    )


# ============================================================
# NORMALIZATION
# ============================================================

if POLAR_NORMALIZATION_MODE not in [
    "absolute",
    "panel",
    "curve",
]:

    raise ValueError(
        "POLAR_NORMALIZATION_MODE must be "
        "'absolute', 'panel', or 'curve'."
    )


for panel in polar_panel_curves:

    if POLAR_NORMALIZATION_MODE == "panel":

        panel_maximum = float(
            np.nanmax(
                [
                    np.nanmax(
                        curve["radial_values"]
                    )
                    for curve in panel["curves"]
                ]
            )
        )

    else:

        panel_maximum = 1.0

    for curve in panel["curves"]:

        if POLAR_NORMALIZATION_MODE == "curve":

            denominator = float(
                np.nanmax(
                    curve["radial_values"]
                )
            )

        elif POLAR_NORMALIZATION_MODE == "panel":

            denominator = panel_maximum

        else:

            denominator = 1.0

        if (
            not np.isfinite(denominator)
            or denominator <= 0.0
        ):

            denominator = 1.0

        curve["radial_values_plot"] = (
            curve["radial_values"]
            / denominator
        )


# ============================================================
# CREATE POLAR FIGURE
# ============================================================

cm_to_inches = 1.0 / 2.54

fig, axes = plt.subplots(
    1,
    3,
    figsize=(
        POLAR_WIDTH_CM * cm_to_inches,
        POLAR_HEIGHT_CM * cm_to_inches,
    ),
    subplot_kw={
        "projection": "polar",
    },
    constrained_layout=True,
)

axes = np.atleast_1d(axes)


for panel_index, panel in enumerate(
    polar_panel_curves
):

    ax = axes[panel_index]

    all_positive_values = []

    for curve in panel["curves"]:

        radial_values = curve[
            "radial_values_plot"
        ].copy()

        positive = radial_values[
            np.isfinite(radial_values)
            & (radial_values > 0.0)
        ]

        if positive.size > 0:

            all_positive_values.append(
                positive
            )

        theta = curve["case"]["theta_rad"]

        # Mirror the axisymmetric theta = 0...pi result into the
        # lower half-plane for presentation.
        theta_full = np.concatenate(
            [
                theta,
                2.0 * np.pi
                - theta[-2:0:-1],
            ]
        )

        radial_full = np.concatenate(
            [
                radial_values,
                radial_values[-2:0:-1],
            ]
        )

        ax.plot(
            theta_full,
            radial_full,
            color=curve["case"]["colour"],
            linestyle=curve["case"]["linestyle"],
            linewidth=POLAR_LINE_WIDTH,
            label=curve["case"]["label"],
            alpha=0.7,
        )

    ax.set_theta_zero_location("E")

    ax.set_theta_direction(1)

    ax.set_thetagrids(
        [
            0,
            45,
            90,
            135,
            180,
            225,
            270,
            315,
        ],
        labels=[
            r"$0^\circ$",
            r"$45^\circ$",
            r"$90^\circ$",
            r"$135^\circ$",
            r"$180^\circ$",
            r"$225^\circ$",
            r"$270^\circ$",
            r"$315^\circ$",
        ],
    )

    if (
        POLAR_LOGARITHMIC_RADIUS
        and all_positive_values
    ):

        combined_positive = np.concatenate(
            all_positive_values
        )

        maximum_positive = float(
            np.nanmax(
                combined_positive
            )
        )

        minimum_positive = max(
            float(
                np.nanmin(
                    combined_positive
                )
            ),
            maximum_positive
            * POLAR_MINIMUM_POSITIVE_FRACTION,
        )

        ax.set_rscale("log")

        ax.set_ylim(
            minimum_positive,
            maximum_positive * 1.08,
        )

    ax.set_title(
        f"{panel['target_wavelength_nm']:.0f} nm",
        fontsize=9,
    )

    ax.grid(
        True,
        color="0.80",
        linewidth=0.55,
    )


# ============================================================
# LEGEND AND TITLE
# ============================================================

polar_handles = [
    Line2D(
        [0],
        [0],
        color=case["colour"],
        linestyle=case["linestyle"],
        linewidth=POLAR_LINE_WIDTH,
        label=case["label"],
    )
    for case in presentation_polar_cases
]


# fig.legend(
#     handles=polar_handles,
#     loc="outside lower center",
#     ncol=1,
#     frameon=True,
#     fontsize=7,
# )


normalization_description = {
    "absolute": "absolute differential scattering",
    "panel": "panel-normalized differential scattering",
    "curve": "individually normalized angular patterns",
}[POLAR_NORMALIZATION_MODE]


fig.suptitle(
    (
        r"$\lambda_0=580$ nm: "
        + normalization_description
    ),
    fontsize=10,
)


# ============================================================
# SAVE
# ============================================================

if SAVE_PRESENTATION_POLAR:

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    polar_png = (
        FIGURE_DIR
        / (
            PRESENTATION_POLAR_BASENAME
            + ".png"
        )
    )

    polar_pdf = (
        FIGURE_DIR
        / (
            PRESENTATION_POLAR_BASENAME
            + ".pdf"
        )
    )

    fig.savefig(
        polar_png,
        dpi=PNG_DPI,
        transparent=True,
        bbox_inches="tight",
    )

    fig.savefig(
        polar_pdf,
        transparent=True,
        bbox_inches="tight",
    )

    print("Saved:", polar_png)
    print("Saved:", polar_pdf)


plt.show()

# Near field analysis

# In[97]:


# ============================================================
# DIAGNOSTIC 1: SELECTED CASE PARAMETERS
# ============================================================

columns_to_show = [
    "task_index",
    "task_hash",
    "reference_design_wavelength_nm",
    "case_family",
    "scenario",
    "absorber_location",
    "absorber_k",
    "core_thickness_factor",
    "n_shells",
    "reference_n_shells",
    "predicted_diameter_nm",
    "reference_diameter_nm",
    "n_A",
    "n_B",
    "outer_layer",
    "actual_outer_layer",
    "t_a_nm",
    "t_b_nm",
    "core_radius_nm",
]

available_columns = [
    column
    for column in columns_to_show
    if column in df_checkpoint_index.columns
]

selected_case_rows = pd.DataFrame(
    [
        {
            "selected_case": case["key"],
            **{
                column: case["row"].get(
                    column,
                    np.nan,
                )
                for column in available_columns
            },
        }
        for case in presentation_cases
    ]
)

display(
    selected_case_rows.T
)

# In[98]:


# ============================================================
# DIAGNOSTIC 2: GEOMETRY, MATERIAL, AND FIELD API SIGNATURES
# ============================================================

import inspect

from bragg_onion.materials import ConstantDispersion
from bragg_onion.geometry import (
    build_bragg_onion_from_thicknesses,
)
from bragg_onion.fields import (
    compute_field_map,
    plot_field_magnitude,
    plot_poynting_streamlines,
)

print(
    "ConstantDispersion.from_nk:\n",
    inspect.signature(
        ConstantDispersion.from_nk
    ),
)

print(
    "\nbuild_bragg_onion_from_thicknesses:\n",
    inspect.signature(
        build_bragg_onion_from_thicknesses
    ),
)

print(
    "\ncompute_field_map:\n",
    inspect.signature(
        compute_field_map
    ),
)

print(
    "\nplot_field_magnitude:\n",
    inspect.signature(
        plot_field_magnitude
    ),
)

print(
    "\nplot_poynting_streamlines:\n",
    inspect.signature(
        plot_poynting_streamlines
    ),
)

# In[99]:


# ============================================================
# DIAGNOSTIC 3: STACK-BUILDING API
# ============================================================

import inspect

import bragg_onion
import bragg_onion.geometry
import bragg_onion.materials

try:
    import bragg_onion.stack as stack_module
except ImportError:
    stack_module = None

modules_to_inspect = [
    bragg_onion.geometry,
    bragg_onion.materials,
]

if stack_module is not None:
    modules_to_inspect.append(
        stack_module
    )

for module in modules_to_inspect:

    print(
        "\nMODULE:",
        module.__name__,
    )

    for name in sorted(
        dir(module)
    ):

        lowered = name.lower()

        if not any(
            keyword in lowered
            for keyword in [
                "stack",
                "layer",
                "profile",
                "material",
                "extinction",
                "modifier",
            ]
        ):
            continue

        object_found = getattr(
            module,
            name,
        )

        if callable(object_found):

            try:
                signature = inspect.signature(
                    object_found
                )
            except Exception:
                signature = "<signature unavailable>"

            print(
                f"{name}: {signature}"
            )

# In[100]:


# ============================================================
# DIAGNOSTIC 4: STORED CHECKPOINT OBJECT STRUCTURE
# ============================================================

reference_case = next(
    case
    for case in presentation_cases
    if case["key"] == "reference"
)

constant_diameter_case = next(
    case
    for case in presentation_cases
    if case["key"] == "constant_diameter"
)

constant_shell_case = next(
    case
    for case in presentation_cases
    if case["key"] == "constant_shell_number"
)


for case in [
    reference_case,
    constant_diameter_case,
    constant_shell_case,
]:

    print(
        "\n",
        "=" * 70,
    )

    print(
        "CASE:",
        case["key"],
    )

    print(
        "Payload keys:",
        list(
            case["payload"].keys()
        ),
    )

    for direction in [
        "backward",
        "forward",
    ]:

        result = case[
            "payload"
        ][
            direction
        ]

        print(
            f"\n{direction} result type:",
            type(result),
        )

        if hasattr(
            result,
            "__dict__",
        ):

            print(
                f"{direction} attributes:",
                sorted(
                    vars(result).keys()
                ),
            )

# In[101]:


# ============================================================
# DIAGNOSTIC 5: REBUILD HELPER
# ============================================================

import inspect

from bragg_onion._notebook_tools.field_analysis import (
    rebuild_stack_from_row,
)

print(
    inspect.signature(
        rebuild_stack_from_row
    )
)

print(
    inspect.getsource(
        rebuild_stack_from_row
    )
)

# Cell 1: Configure cases, wavelengths, planes, and plotting parameters

# In[238]:


# ============================================================
# NEAR-FIELD AND ENERGY-FLOW ANALYSIS SETTINGS
#
# Cases:
#   1. Nonabsorbing 580 nm reference
#   2. Core absorber, gamma_c = 50, k = 1,
#      approximately constant diameter
#   3. Core absorber, gamma_c = 50, k = 1,
#      constant shell number
#
# Planes:
#   xy, xz, and yz, all through the sphere centre
#
# Wavelengths:
#   reference backward-collection peak and +/- 150 nm
# ============================================================

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from IPython.display import display

from bragg_onion.materials import ConstantDispersion

from bragg_onion.geometry import (
    ExtinctionModifier,
)

from bragg_onion._notebook_tools.field_analysis import (
    rebuild_stack_from_row,
)

from bragg_onion.fields import (
    compute_field_map,
    plot_field_magnitude,
    plot_poynting_streamlines,
)


# ============================================================
# CASE SETTINGS
# ============================================================

FIELD_REFERENCE_DESIGN_WAVELENGTH_NM = 580.0

FIELD_CORE_THICKNESS_FACTOR = 50.0
FIELD_CORE_EXTINCTION_K = 1.0

FIELD_CASE_KEYS = [
    "reference",
    "constant_diameter",
    "constant_shell_number",
]


# ============================================================
# WAVELENGTH SETTINGS
# ============================================================

# Use the peak of the nonabsorbing reference backward spectrum.
FIELD_WAVELENGTH_OFFSET_NM = 150.0

# Optional explicit wavelength override.
#
# Example:
# MANUAL_FIELD_WAVELENGTHS_NM = [500.0, 550.0, 600.0]
#
# Leave as None to use peak - 150 nm, peak, peak + 150 nm.
MANUAL_FIELD_WAVELENGTHS_NM = None


# ============================================================
# SPATIAL GRID SETTINGS
# ============================================================

FIELD_PLANES = [
    "xy",
    "xz", # Propagation direction is z so yz and xz are pretty much the same
    # "yz",
]

# 301 is recommended for initial inspection.
# Increase to 501 only for selected final maps.
FIELD_NPTS = 151

# Extent of each square field map relative to the outer radius.
#
# 1.0 shows exactly the sphere.
# 1.15 adds a small surrounding-medium region.
# 1.5 shows more of the external flow.
FIELD_EXTENT_OUTER_RADIUS_FACTOR = 1.8

FIELD_N_MEDIUM = 1.0

FIELD_INCIDENT_E0 = 1.0 + 0.0j


# ============================================================
# COMPUTATION AND CACHE SETTINGS
# ============================================================

FIELD_CACHE_DIRECTORY = (
    OUTPUT_DIR
    / "field_map_cache"
)

FIELD_CACHE_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)

# Reuse an already computed FieldMapResult from the current notebook
# session. Disk caching is handled separately below.
RECOMPUTE_FIELD_MAPS = False


# ============================================================
# FIELD-MAGNITUDE PLOT SETTINGS
# ============================================================

FIELD_MAGNITUDE_SETTINGS = {
    "E_total": {
        "quantity": "E",
        "kind": "total",
        "scale": "db",
        "floor": 1e-30,
        "cmap": "turbo",
        "clip_percentile_low": 0.0,
        "clip_percentile_high": 100.0,
    },
    "E_scattered": {
        "quantity": "E",
        "kind": "scattered",
        "scale": "db",
        "floor": 1e-30,
        "cmap": "turbo",
        "clip_percentile_low": 0.0,
        "clip_percentile_high": 100.0,
    },
    "S_total": {
        "quantity": "S",
        "kind": "total",
        "scale": "db",
        "floor": 1e-30,
        "cmap": "turbo",
        "clip_percentile_low": 0.0,
        "clip_percentile_high": 100,
    },
}


# ============================================================
# POYNTING-STREAMLINE SETTINGS
#
# These dictionaries can be edited independently.
# ============================================================

POYNTING_STREAMLINE_SETTINGS = {
    "total": {
        "flow_kind": "total",
        "background_quantity": "S",
        "background_kind": "total",
        "background_scale": "db",
        "background_floor": 1e-30,
        "background_cmap": "magma",
        "background_clip_percentile_low": 0.0,
        "background_clip_percentile_high": 99.5,
        "streamline_density": 1.25,
        "streamline_color": "white",
        "streamline_linewidth": 0.70,
        "normalize_vectors": False,
        "min_speed_fraction": 0.005,
        "mask_inside_sphere": False,
        "show_boundaries": True,
    },
    "delta": {
        "flow_kind": "delta",
        "background_quantity": "S",
        "background_kind": "delta",
        "background_scale": "db",
        "background_floor": 1e-30,
        "background_cmap": "magma",
        "background_clip_percentile_low": 0.0,
        "background_clip_percentile_high": 99.5,
        "streamline_density": 1.35,
        "streamline_color": "white",
        "streamline_linewidth": 0.70,
        "normalize_vectors": True,
        "min_speed_fraction": 0.002,
        "mask_inside_sphere": False,
        "show_boundaries": True,
    },
    "scattered": {
        "flow_kind": "scattered",
        "background_quantity": "S",
        "background_kind": "scattered",
        "background_scale": "db",
        "background_floor": 1e-30,
        "background_cmap": "magma",
        "background_clip_percentile_low": 0.0,
        "background_clip_percentile_high": 99.5,
        "streamline_density": 1.35,
        "streamline_color": "white",
        "streamline_linewidth": 0.70,
        "normalize_vectors": True,
        "min_speed_fraction": 0.002,
        "mask_inside_sphere": False,
        "show_boundaries": True,
    },
}


# ============================================================
# OUTPUT SETTINGS
# ============================================================

FIELD_FIGURE_DIRECTORY = (
    FIGURE_DIR
    / "field_maps"
)

FIELD_FIGURE_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)

FIELD_PNG_DPI = 300

SAVE_FIELD_PNG = True
SAVE_FIELD_PDF = True

print(
    "Field output directory:",
    FIELD_FIGURE_DIRECTORY,
)

# Cell 2: Select and validate the three cases
#
# This cell explicitly selects γc=50\gamma_{\mathrm c}=50γc​=50 and k=1k=1k=1, rather than relying on the earlier automatic case selection.

# In[239]:


# ============================================================
# SELECT AND VALIDATE THE THREE FIELD CASES
# ============================================================

def clean_text_column(
    series,
):
    """
    Normalize categorical labels for reliable selection.
    """

    return (
        series
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(
            " ",
            "_",
            regex=False,
        )
        .str.replace(
            "-",
            "_",
            regex=False,
        )
    )


df_field_selection = (
    df_checkpoint_index.copy()
)

for column in [
    "reference_design_wavelength_nm",
    "absorber_k",
    "core_thickness_factor",
    "n_shells",
    "predicted_diameter_nm",
]:

    if column in df_field_selection.columns:

        df_field_selection[column] = (
            pd.to_numeric(
                df_field_selection[column],
                errors="coerce",
            )
        )


df_field_selection[
    "_case_family_clean"
] = clean_text_column(
    df_field_selection[
        "case_family"
    ]
)

df_field_selection[
    "_scenario_clean"
] = clean_text_column(
    df_field_selection[
        "scenario"
    ]
)


def select_exact_field_case(
    *,
    case_family,
    scenario,
    absorber_k=None,
    core_thickness_factor=None,
):
    """
    Select exactly one checkpoint matching the requested case.
    """

    mask = (
        np.isclose(
            df_field_selection[
                "reference_design_wavelength_nm"
            ],
            FIELD_REFERENCE_DESIGN_WAVELENGTH_NM,
        )
        & df_field_selection[
            "_case_family_clean"
        ].eq(case_family)
        & df_field_selection[
            "_scenario_clean"
        ].eq(scenario)
    )

    if absorber_k is not None:

        mask &= np.isclose(
            df_field_selection[
                "absorber_k"
            ],
            float(absorber_k),
        )

    if core_thickness_factor is not None:

        mask &= np.isclose(
            df_field_selection[
                "core_thickness_factor"
            ],
            float(
                core_thickness_factor
            ),
        )

    selected = (
        df_field_selection.loc[
            mask
        ]
        .copy()
    )

    if len(selected) != 1:
        display(selected)

        raise ValueError(
            "Expected exactly one field-analysis checkpoint, "
            f"found {len(selected)} for "
            f"case_family={case_family!r}, "
            f"scenario={scenario!r}, "
            f"k={absorber_k}, "
            f"core factor={core_thickness_factor}."
        )

    return selected.iloc[0]


field_case_rows = {
    "reference": select_exact_field_case(
        case_family="reference",
        scenario="reference",
    ),
    "constant_diameter": select_exact_field_case(
        case_family="core",
        scenario="constant_diameter",
        absorber_k=FIELD_CORE_EXTINCTION_K,
        core_thickness_factor=(
            FIELD_CORE_THICKNESS_FACTOR
        ),
    ),
    "constant_shell_number": (
        select_exact_field_case(
            case_family="core",
            scenario="constant_shell_number",
            absorber_k=FIELD_CORE_EXTINCTION_K,
            core_thickness_factor=(
                FIELD_CORE_THICKNESS_FACTOR
            ),
        )
    ),
}


FIELD_CASE_APPEARANCE = {
    "reference": {
        "label": "Nonabsorbing reference",
        "short_label": "Reference",
        "colour": "black",
    },
    "constant_diameter": {
        "label": (
            "Absorbing core, approximately constant diameter"
        ),
        "short_label": (
            "Core absorber\nconstant diameter"
        ),
        "colour": "#0072B2",
    },
    "constant_shell_number": {
        "label": (
            "Absorbing core, constant shell number"
        ),
        "short_label": (
            "Core absorber\nconstant shell number"
        ),
        "colour": "#D55E00",
    },
}


field_case_records = []

for case_key, row in (
    field_case_rows.items()
):

    field_case_records.append(
        {
            "case": case_key,
            "label": (
                FIELD_CASE_APPEARANCE[
                    case_key
                ][
                    "label"
                ]
            ),
            "task_index": row.get(
                "task_index",
                np.nan,
            ),
            "task_hash": row.get(
                "task_hash",
                "",
            ),
            "absorber_k": row.get(
                "absorber_k",
                np.nan,
            ),
            "core_thickness_factor": row.get(
                "core_thickness_factor",
                np.nan,
            ),
            "n_shells": row.get(
                "n_shells",
                np.nan,
            ),
            "predicted_diameter_um": (
                float(
                    row[
                        "predicted_diameter_nm"
                    ]
                )
                / 1000.0
            ),
        }
    )


df_field_cases = pd.DataFrame(
    field_case_records
)

display(
    df_field_cases.style.format(
        {
            "absorber_k": "{:.3g}",
            "core_thickness_factor": "{:.1f}",
            "n_shells": "{:.0f}",
            "predicted_diameter_um": "{:.3f}",
        },
        na_rep="not applicable",
    )
)

# Cell 3: Match the selected cases to complete calculation rows
#
# rebuild_stack_from_row() requires fields such as peak_wavelength_m, n_layers, outer_layer, and core_thickness_factor. The paired table is therefore used because it contains the calculation parameters.

# In[240]:


# ============================================================
# CONSTRUCT COMPLETE ROWS FOR FIELD-STACK RECONSTRUCTION
#
# rebuild_stack_from_row() requires:
#   peak_wavelength_m
#   n_layers
#   outer_layer
#   core_thickness_factor
#
# The forward-suppression sweep instead stores some quantities as:
#   reference_design_wavelength_nm
#   n_shells
#
# This cell combines:
#   1. checkpoint task metadata;
#   2. checkpoint-index metadata;
#   3. paired scalar results;
#
# It then creates the aliases required by rebuild_stack_from_row().
# ============================================================

import numpy as np
import pandas as pd

from pathlib import Path
from IPython.display import display


# ============================================================
# MATCH ONE CHECKPOINT TO ONE PAIRED RESULT
# ============================================================

def match_checkpoint_to_paired_row(
    checkpoint_row,
):
    """
    Match one checkpoint-index row to exactly one df_paired row.

    task_hash is preferred because it uniquely describes the
    physical calculation. task_index is used only as a fallback.
    """

    if (
        "task_hash" in df_paired.columns
        and "task_hash" in checkpoint_row.index
        and pd.notna(
            checkpoint_row["task_hash"]
        )
    ):

        selected = df_paired.loc[
            df_paired[
                "task_hash"
            ].astype(str)
            == str(
                checkpoint_row[
                    "task_hash"
                ]
            )
        ].copy()

    elif (
        "task_index" in df_paired.columns
        and "task_index" in checkpoint_row.index
        and pd.notna(
            checkpoint_row["task_index"]
        )
    ):

        selected = df_paired.loc[
            np.isclose(
                pd.to_numeric(
                    df_paired[
                        "task_index"
                    ],
                    errors="coerce",
                ),
                float(
                    checkpoint_row[
                        "task_index"
                    ]
                ),
            )
        ].copy()

    else:

        raise KeyError(
            "Neither task_hash nor task_index is available "
            "for matching the checkpoint to df_paired."
        )

    if len(selected) != 1:

        print(
            "Checkpoint identifiers:"
        )

        print(
            {
                "task_index": checkpoint_row.get(
                    "task_index",
                    np.nan,
                ),
                "task_hash": checkpoint_row.get(
                    "task_hash",
                    "",
                ),
            }
        )

        display(selected)

        raise ValueError(
            "Expected exactly one paired result row, "
            f"found {len(selected)}."
        )

    return selected.iloc[0]


# ============================================================
# CREATE ONE COMPLETE RECONSTRUCTION ROW
# ============================================================

def construct_field_reconstruction_row(
    checkpoint_row,
):
    """
    Combine checkpoint task metadata, checkpoint-index metadata,
    and paired scalar results into one reconstruction row.

    Later sources overwrite earlier sources:
      checkpoint task -> checkpoint index -> paired result

    Required aliases are then created explicitly.
    """

    checkpoint_path = Path(
        checkpoint_row[
            "checkpoint_path"
        ]
    )

    payload = load_full_checkpoint(
        checkpoint_path
    )

    task_metadata = dict(
        payload.get(
            "task",
            {},
        )
    )

    paired_row = (
        match_checkpoint_to_paired_row(
            checkpoint_row
        )
    )

    combined = {}

    # First source: original task dictionary.
    combined.update(
        task_metadata
    )

    # Second source: checkpoint-index row.
    combined.update(
        checkpoint_row.to_dict()
    )

    # Third source: scalar paired-result row.
    combined.update(
        paired_row.to_dict()
    )

    row = pd.Series(
        combined,
        dtype=object,
    )

    # --------------------------------------------------------
    # Design wavelength
    # --------------------------------------------------------

    possible_peak_columns = [
        "peak_wavelength_m",
        "reference_peak_wavelength_m",
    ]

    peak_wavelength_m = np.nan

    for column in possible_peak_columns:

        if column not in row.index:
            continue

        candidate = pd.to_numeric(
            pd.Series(
                [row[column]]
            ),
            errors="coerce",
        ).iloc[0]

        if np.isfinite(candidate):

            peak_wavelength_m = float(
                candidate
            )

            break

    if not np.isfinite(
        peak_wavelength_m
    ):

        if (
            "reference_design_wavelength_nm"
            not in row.index
        ):

            raise KeyError(
                "Neither peak_wavelength_m nor "
                "reference_design_wavelength_nm is available."
            )

        design_wavelength_nm = pd.to_numeric(
            pd.Series(
                [
                    row[
                        "reference_design_wavelength_nm"
                    ]
                ]
            ),
            errors="coerce",
        ).iloc[0]

        if not np.isfinite(
            design_wavelength_nm
        ):

            raise ValueError(
                "The design wavelength is not finite."
            )

        peak_wavelength_m = (
            float(
                design_wavelength_nm
            )
            * 1e-9
        )

    row[
        "peak_wavelength_m"
    ] = peak_wavelength_m

    # --------------------------------------------------------
    # Number of radial layers
    # --------------------------------------------------------

    n_layers = np.nan

    for column in [
        "n_layers",
        "n_shells",
        "n_layers_total",
        "reference_n_shells",
    ]:

        if column not in row.index:
            continue

        candidate = pd.to_numeric(
            pd.Series(
                [row[column]]
            ),
            errors="coerce",
        ).iloc[0]

        if np.isfinite(candidate):

            n_layers = int(
                round(
                    float(candidate)
                )
            )

            break

    if not np.isfinite(
        n_layers
    ):

        raise KeyError(
            "No usable layer count was found in "
            "n_layers, n_shells, n_layers_total, or "
            "reference_n_shells."
        )

    row["n_layers"] = n_layers

    # Preserve the new-sweep name as well.
    row["n_shells"] = n_layers

    # --------------------------------------------------------
    # Outermost Bragg layer
    # --------------------------------------------------------

    outer_layer = None

    for column in [
        "outer_layer",
        "actual_outer_layer",
        "reference_outer_layer",
    ]:

        if column not in row.index:
            continue

        candidate = str(
            row[column]
        ).strip().upper()

        if candidate in {
            "A",
            "B",
        }:

            outer_layer = candidate
            break

    if outer_layer is None:

        # All selected optimized cases were generated with A as
        # the outermost Bragg layer. This fallback is explicit.
        outer_layer = "A"

        warnings.warn(
            "No valid outer-layer label was found. "
            "Using outer_layer='A'."
        )

    row[
        "outer_layer"
    ] = outer_layer

    # --------------------------------------------------------
    # Core-thickness factor
    # --------------------------------------------------------

    if (
        "core_thickness_factor"
        not in row.index
    ):

        raise KeyError(
            "core_thickness_factor is unavailable."
        )

    core_factor = pd.to_numeric(
        pd.Series(
            [
                row[
                    "core_thickness_factor"
                ]
            ]
        ),
        errors="coerce",
    ).iloc[0]

    if (
        not np.isfinite(core_factor)
        or core_factor <= 0.0
    ):

        raise ValueError(
            "core_thickness_factor must be finite and positive."
        )

    row[
        "core_thickness_factor"
    ] = float(
        core_factor
    )

    # --------------------------------------------------------
    # Real refractive indices
    # --------------------------------------------------------

    for column in [
        "n_A",
        "n_B",
    ]:

        if column not in row.index:

            raise KeyError(
                f"{column} is unavailable for stack reconstruction."
            )

        value = pd.to_numeric(
            pd.Series(
                [row[column]]
            ),
            errors="coerce",
        ).iloc[0]

        if (
            not np.isfinite(value)
            or value <= 0.0
        ):

            raise ValueError(
                f"{column} must be finite and positive."
            )

        row[column] = float(
            value
        )

    # --------------------------------------------------------
    # Traceability
    # --------------------------------------------------------

    row[
        "checkpoint_path"
    ] = checkpoint_path

    row[
        "_source_task_index"
    ] = checkpoint_row.get(
        "task_index",
        np.nan,
    )

    row[
        "_source_task_hash"
    ] = checkpoint_row.get(
        "task_hash",
        "",
    )

    return row


# ============================================================
# CONSTRUCT THE THREE COMPLETE ROWS
# ============================================================

field_case_rows_complete = {
    case_key: (
        construct_field_reconstruction_row(
            checkpoint_row
        )
    )
    for case_key, checkpoint_row
    in field_case_rows.items()
}


# ============================================================
# STRICT VALIDATION
# ============================================================

required_rebuild_columns = [
    "peak_wavelength_m",
    "n_layers",
    "outer_layer",
    "core_thickness_factor",
    "n_A",
    "n_B",
]

validation_records = []

for case_key, row in (
    field_case_rows_complete.items()
):

    missing = [
        column
        for column in (
            required_rebuild_columns
        )
        if column not in row.index
    ]

    if missing:

        raise KeyError(
            f"{case_key} is missing reconstruction columns: "
            f"{missing}"
        )

    validation_records.append(
        {
            "case": case_key,
            "peak_wavelength_nm": (
                float(
                    row[
                        "peak_wavelength_m"
                    ]
                )
                * 1e9
            ),
            "n_layers": int(
                row[
                    "n_layers"
                ]
            ),
            "outer_layer": str(
                row[
                    "outer_layer"
                ]
            ),
            "core_thickness_factor": float(
                row[
                    "core_thickness_factor"
                ]
            ),
            "n_A": float(
                row["n_A"]
            ),
            "n_B": float(
                row["n_B"]
            ),
            "predicted_diameter_um": (
                float(
                    row.get(
                        "predicted_diameter_nm",
                        np.nan,
                    )
                )
                / 1000.0
            ),
            "task_index": row.get(
                "_source_task_index",
                np.nan,
            ),
            "task_hash": row.get(
                "_source_task_hash",
                "",
            ),
        }
    )


df_field_reconstruction_rows = (
    pd.DataFrame(
        validation_records
    )
)

display(
    df_field_reconstruction_rows.style.format(
        {
            "peak_wavelength_nm": "{:.1f}",
            "n_layers": "{:.0f}",
            "core_thickness_factor": "{:.1f}",
            "n_A": "{:.6f}",
            "n_B": "{:.6f}",
            "predicted_diameter_um": "{:.3f}",
        }
    )
)

# Cell 4: Identify the three wavelengths

# In[241]:


# ============================================================
# SELECT FIELD-INSPECTION WAVELENGTHS
# ============================================================

reference_case_payload = (
    load_full_checkpoint(
        Path(
            field_case_rows[
                "reference"
            ][
                "checkpoint_path"
            ]
        )
    )
)


reference_backward_spectrum = (
    extract_directional_spectrum(
        reference_case_payload,
        "backward",
    )
)


reference_peak_index = int(
    np.nanargmax(
        reference_backward_spectrum[
            "spectrum"
        ]
    )
)


reference_backward_peak_nm = float(
    reference_backward_spectrum[
        "wavelength_nm"
    ][
        reference_peak_index
    ]
)


if MANUAL_FIELD_WAVELENGTHS_NM is None:

    field_wavelengths_nm = np.array(
        [
            reference_backward_peak_nm
            - FIELD_WAVELENGTH_OFFSET_NM,

            reference_backward_peak_nm,

            reference_backward_peak_nm
            + FIELD_WAVELENGTH_OFFSET_NM,
        ],
        dtype=float,
    )

else:

    field_wavelengths_nm = np.asarray(
        MANUAL_FIELD_WAVELENGTHS_NM,
        dtype=float,
    )


spectral_minimum_nm = float(
    np.nanmin(
        reference_backward_spectrum[
            "wavelength_nm"
        ]
    )
)


spectral_maximum_nm = float(
    np.nanmax(
        reference_backward_spectrum[
            "wavelength_nm"
        ]
    )
)


field_wavelengths_nm = np.clip(
    field_wavelengths_nm,
    spectral_minimum_nm,
    spectral_maximum_nm,
)


print(
    "Reference backward peak:",
    f"{reference_backward_peak_nm:.1f} nm",
)

print(
    "Field wavelengths:",
    field_wavelengths_nm,
)

# Cell 5: Rebuild the three stacks with core-only absorption

# In[245]:


# ============================================================
# REBUILD GEOMETRIES AND RESOLVED STACKS
#
# Materials A and B are lossless in every case.
#
# For the two absorbing cases, k = 1 is applied only to layer
# index 0, corresponding to the core, through target="core".
# ============================================================

FIELD_FIXED_GEOMETRY_KWARGS = {}


field_stacks = {}


for case_key, row in (
    field_case_rows_complete.items()
):

    n_A = float(
        row["n_A"]
    )

    n_B = float(
        row["n_B"]
    )


    material_A_lossless = (
        ConstantDispersion.from_nk(
            name="A",
            n=n_A,
            k=0.0,
            source=(
                "Lossless high-index material "
                "for field analysis"
            ),
        )
    )


    material_B_lossless = (
        ConstantDispersion.from_nk(
            name="B",
            n=n_B,
            k=0.0,
            source=(
                "Lossless low-index material "
                "for field analysis"
            ),
        )
    )


    if case_key == "reference":

        extinction_modifiers = None

    else:

        extinction_modifiers = [
            ExtinctionModifier(
                target="core",
                mode="substitute",
                profile=float(
                    FIELD_CORE_EXTINCTION_K
                ),
                name=(
                    "absorbing_core_"
                    f"k_{FIELD_CORE_EXTINCTION_K:g}"
                ),
            )
        ]


    geometry, resolved_stack = (
        rebuild_stack_from_row(
            row,
            material_a=(
                material_A_lossless
            ),
            material_b=(
                material_B_lossless
            ),
            fixed_geometry_kwargs=(
                FIELD_FIXED_GEOMETRY_KWARGS
            ),
            extinction_modifiers=(
                extinction_modifiers
            ),
            extra_outer_shell=None,
        )
    )


    field_stacks[
        case_key
    ] = {
        "geometry": geometry,
        "stack": resolved_stack,
        "row": row,
        "extinction_modifiers": (
            extinction_modifiers
        ),
    }


# ============================================================
# STACK VALIDATION
# ============================================================

stack_validation_records = []


for case_key, stack_data in (
    field_stacks.items()
):

    geometry = stack_data[
        "geometry"
    ]

    stack = stack_data[
        "stack"
    ]


    number_of_layers = len(
        stack.layer_dispersions
    )


    stack_validation_records.append(
        {
            "case": case_key,

            "calculated_n_layers": (
                number_of_layers
            ),

            "expected_n_layers": int(
                field_case_rows_complete[
                    case_key
                ][
                    "n_layers"
                ]
            ),

            "outer_radius_um": (
                float(
                    geometry.outer_radius_m
                )
                * 1e6
                if hasattr(
                    geometry,
                    "outer_radius_m",
                )
                else np.nan
            ),

            "core_thickness_factor": float(
                field_case_rows_complete[
                    case_key
                ][
                    "core_thickness_factor"
                ]
            ),

            "core_absorber_applied": (
                case_key != "reference"
            ),

            "core_k": (
                0.0
                if case_key == "reference"
                else FIELD_CORE_EXTINCTION_K
            ),
        }
    )


df_stack_validation = pd.DataFrame(
    stack_validation_records
)


display(
    df_stack_validation.style.format(
        {
            "outer_radius_um": "{:.3f}",
            "core_thickness_factor": "{:.1f}",
            "core_k": "{:.3g}",
        }
    )
)


if not np.all(
    df_stack_validation[
        "calculated_n_layers"
    ]
    == df_stack_validation[
        "expected_n_layers"
    ]
):

    raise ValueError(
        "At least one rebuilt stack has an unexpected "
        "number of layers."
    )

# Cell 6: Compute all 27 field maps
#
# The cell stores maps in memory. At 301×301301\times301301×301, this is already a substantial computation:
#
# 3 cases×3 wavelengths×3 planes=27 maps.3\ \text{cases} \times 3\ \text{wavelengths} \times 3\ \text{planes} = 27\ \text{maps}.3 cases×3 wavelengths×3 planes=27 maps.

# In[246]:


# ============================================================
# COMPUTE FIELD MAPS WITH TIMING AND PROGRESS REPORTING
# ============================================================

import time
from datetime import datetime, timedelta

field_maps = (
    {}
    if RECOMPUTE_FIELD_MAPS
    else globals().get(
        "field_maps",
        {},
    )
)

requested_keys = [
    (
        case_key,
        float(wavelength_nm),
        plane,
    )
    for case_key in FIELD_CASE_KEYS
    for wavelength_nm in field_wavelengths_nm
    for plane in FIELD_PLANES
]

missing_keys = [
    key
    for key in requested_keys
    if (
        RECOMPUTE_FIELD_MAPS
        or key not in field_maps
    )
]

print("Total requested maps:", len(requested_keys))
print("Already completed:", len(requested_keys) - len(missing_keys))
print("Maps remaining:", len(missing_keys))

run_start_time = time.perf_counter()
completed_durations = []

for remaining_index, (
    case_key,
    wavelength_nm,
    plane,
) in enumerate(
    missing_keys,
    start=1,
):

    map_key = (
        case_key,
        wavelength_nm,
        plane,
    )

    print(
        "\n"
        f"[{remaining_index}/{len(missing_keys)}] "
        f"Starting {map_key}",
        flush=True,
    )

    print(
        "Start time:",
        datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        flush=True,
    )

    map_start_time = time.perf_counter()

    stack = field_stacks[
        case_key
    ][
        "stack"
    ]

    field_map_result = compute_field_map(
        stack=stack,
        wavelength_m=(
            wavelength_nm * 1e-9
        ),
        n_medium=FIELD_N_MEDIUM,
        plane=plane,
        npts=FIELD_NPTS,
        extent_outer_radius_factor=(
            FIELD_EXTENT_OUTER_RADIUS_FACTOR
        ),
        incident_e0=FIELD_INCIDENT_E0,
    )

    field_maps[
        map_key
    ] = field_map_result

    elapsed_seconds = (
        time.perf_counter()
        - map_start_time
    )

    completed_durations.append(
        elapsed_seconds
    )

    mean_seconds_per_map = float(
        np.mean(
            completed_durations
        )
    )

    number_still_remaining = (
        len(missing_keys)
        - remaining_index
    )

    estimated_remaining_seconds = (
        mean_seconds_per_map
        * number_still_remaining
    )

    print(
        "Completed:",
        map_key,
        flush=True,
    )

    print(
        "Map runtime:",
        str(
            timedelta(
                seconds=int(
                    elapsed_seconds
                )
            )
        ),
        flush=True,
    )

    print(
        "Mean runtime per completed map:",
        str(
            timedelta(
                seconds=int(
                    mean_seconds_per_map
                )
            )
        ),
        flush=True,
    )

    print(
        "Estimated remaining time:",
        str(
            timedelta(
                seconds=int(
                    estimated_remaining_seconds
                )
            )
        ),
        flush=True,
    )

print(
    "\nAll requested maps completed in:",
    str(
        timedelta(
            seconds=int(
                time.perf_counter()
                - run_start_time
            )
        )
    ),
)

print(
    "Number of stored maps:",
    len(field_maps),
)

# Cell 7: Plot ∣E∣|E|∣E∣ maps for all cases and center planes
#
# This produces one figure per wavelength and field representation. Columns are the three cases. Rows are the three center planes.

# In[247]:


# ============================================================
# RESTORE AND VALIDATE FIELD-MAP PLOTTING VARIABLES
# ============================================================

import numpy as np


# ------------------------------------------------------------
# Recover wavelengths from completed field-map keys when possible
# ------------------------------------------------------------

if (
    "field_wavelengths_nm" not in globals()
    or field_wavelengths_nm is None
):

    if (
        "field_maps" in globals()
        and len(field_maps) > 0
    ):

        field_wavelengths_nm = np.array(
            sorted(
                {
                    float(key[1])
                    for key in field_maps.keys()
                    if (
                        isinstance(key, tuple)
                        and len(key) == 3
                    )
                }
            ),
            dtype=float,
        )

        print(
            "Recovered field wavelengths from field_maps:",
            field_wavelengths_nm,
        )

    elif (
        "reference_backward_peak_nm" in globals()
    ):

        field_wavelengths_nm = np.array(
            [
                float(reference_backward_peak_nm) - 150.0,
                float(reference_backward_peak_nm),
                float(reference_backward_peak_nm) + 150.0,
            ],
            dtype=float,
        )

        print(
            "Recreated wavelengths from the reference peak:",
            field_wavelengths_nm,
        )

    else:

        raise NameError(
            "field_wavelengths_nm is unavailable and could not be "
            "reconstructed automatically. Rerun the wavelength-selection "
            "cell or define the wavelengths manually."
        )


# ------------------------------------------------------------
# Optional manual definition
# ------------------------------------------------------------
#
# Uncomment and replace these values if you want to select the
# wavelengths explicitly:
#
# field_wavelengths_nm = np.array(
#     [
#         530.0,
#         580.0,
#         630.0,
#     ],
#     dtype=float,
# )


field_wavelengths_nm = np.asarray(
    field_wavelengths_nm,
    dtype=float,
)

field_wavelengths_nm = np.unique(
    field_wavelengths_nm[
        np.isfinite(field_wavelengths_nm)
    ]
)

if field_wavelengths_nm.size == 0:

    raise ValueError(
        "field_wavelengths_nm contains no finite wavelengths."
    )


# ------------------------------------------------------------
# Recover available cases and planes from completed maps
# ------------------------------------------------------------

if (
    "field_maps" not in globals()
    or len(field_maps) == 0
):

    raise NameError(
        "No completed field maps are available. "
        "The computation cell must complete at least one map "
        "before the plotting cell can run."
    )


available_map_keys = [
    key
    for key in field_maps.keys()
    if (
        isinstance(key, tuple)
        and len(key) == 3
    )
]

if not available_map_keys:

    raise ValueError(
        "field_maps does not contain keys of the expected form "
        "(case_key, wavelength_nm, plane)."
    )


available_cases = sorted(
    {
        str(key[0])
        for key in available_map_keys
    }
)

available_wavelengths_nm = np.array(
    sorted(
        {
            float(key[1])
            for key in available_map_keys
        }
    ),
    dtype=float,
)

available_planes = sorted(
    {
        str(key[2])
        for key in available_map_keys
    }
)


# Use only cases and planes that were actually computed.
if "FIELD_CASE_KEYS" not in globals():

    FIELD_CASE_KEYS = available_cases

else:

    FIELD_CASE_KEYS = [
        case_key
        for case_key in FIELD_CASE_KEYS
        if case_key in available_cases
    ]


if "FIELD_PLANES" not in globals():

    FIELD_PLANES = available_planes

else:

    FIELD_PLANES = [
        plane
        for plane in FIELD_PLANES
        if plane in available_planes
    ]


# Use only requested wavelengths for which at least one map exists.
field_wavelengths_nm = np.array(
    [
        wavelength_nm
        for wavelength_nm in field_wavelengths_nm
        if np.any(
            np.isclose(
                available_wavelengths_nm,
                wavelength_nm,
            )
        )
    ],
    dtype=float,
)


if not FIELD_CASE_KEYS:

    raise ValueError(
        "None of the requested FIELD_CASE_KEYS has completed maps."
    )

if not FIELD_PLANES:

    raise ValueError(
        "None of the requested FIELD_PLANES has completed maps."
    )

if field_wavelengths_nm.size == 0:

    raise ValueError(
        "None of the requested wavelengths has completed maps."
    )


# ------------------------------------------------------------
# Report completed and missing combinations
# ------------------------------------------------------------

requested_plot_keys = [
    (
        case_key,
        float(wavelength_nm),
        plane,
    )
    for case_key in FIELD_CASE_KEYS
    for wavelength_nm in field_wavelengths_nm
    for plane in FIELD_PLANES
]

completed_plot_keys = [
    key
    for key in requested_plot_keys
    if key in field_maps
]

missing_plot_keys = [
    key
    for key in requested_plot_keys
    if key not in field_maps
]


print("\nPlotting configuration")
print("----------------------")
print("Cases       :", FIELD_CASE_KEYS)
print("Wavelengths :", field_wavelengths_nm)
print("Planes      :", FIELD_PLANES)
print("Completed   :", len(completed_plot_keys))
print("Missing     :", len(missing_plot_keys))

if missing_plot_keys:

    print("\nMissing field maps:")

    for key in missing_plot_keys:
        print("  ", key)

# In[248]:


# ============================================================
# PLOT COMPACT ELECTRIC-FIELD MAP MATRICES
#
# Rows:
# xy, xz, yz
#
# Columns:
# reference
# constant diameter
# constant shell number
# ============================================================

from matplotlib.ticker import MaxNLocator

FIELD_MAGNITUDE_KINDS_TO_PLOT = [
    "E_total",
    "E_scattered",
]

# A 3x3 matrix with nine colorbars needs more physical space.
# The result remains approximately square.
FIELD_MATRIX_WIDTH_CM = 20.0
FIELD_MATRIX_HEIGHT_CM = 20.0

# Resolution of saved PNG files.
FIELD_MATRIX_PNG_DPI = 400

# Resolution of the figure displayed inside the notebook.
FIELD_MATRIX_NOTEBOOK_DPI = 150

SAVE_FIELD_MAGNITUDE_MATRICES = True


cm_to_inches = 1.0 / 2.54

# Improves only the notebook preview.
# The saved PNG resolution is controlled separately below.
with plt.rc_context({
    "figure.dpi": FIELD_MATRIX_NOTEBOOK_DPI,
    "font.size": 7,
    "axes.titlesize": 8,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
}):

    for wavelength_nm in field_wavelengths_nm:

        for settings_key in FIELD_MAGNITUDE_KINDS_TO_PLOT:

            settings = FIELD_MAGNITUDE_SETTINGS[settings_key]

            fig, axes = plt.subplots(
                nrows=len(FIELD_PLANES),
                ncols=len(FIELD_CASE_KEYS),
                figsize=(
                    FIELD_MATRIX_WIDTH_CM * cm_to_inches,
                    FIELD_MATRIX_HEIGHT_CM * cm_to_inches,
                ),
                constrained_layout=True,
            )

            axes = np.asarray(
                axes,
                dtype=object,
            ).reshape(
                len(FIELD_PLANES),
                len(FIELD_CASE_KEYS),
            )

            # Make constrained_layout substantially more compact.
            # Values are in inches for w_pad and h_pad.
            fig.set_constrained_layout_pads(
                w_pad=0.015,
                h_pad=0.015,
                wspace=0.015,
                hspace=0.015,
            )

            for row_index, plane in enumerate(FIELD_PLANES):

                for column_index, case_key in enumerate(FIELD_CASE_KEYS):

                    ax = axes[row_index, column_index]

                    field_map = field_maps[
                        (
                            case_key,
                            float(wavelength_nm),
                            plane,
                        )
                    ]

                    # Record existing figure axes so that colorbar axes created
                    # by plot_field_magnitude can be identified afterward.
                    axes_before_plot = set(fig.axes)

                    plot_field_magnitude(
                        field_map,
                        quantity=settings["quantity"],
                        kind=settings["kind"],
                        scale=settings["scale"],
                        floor=settings["floor"],
                        cmap=settings["cmap"],
                        clip_percentile_low=(
                            settings["clip_percentile_low"]
                        ),
                        clip_percentile_high=(
                            settings["clip_percentile_high"]
                        ),
                        show_boundaries=True,
                        ax=ax,
                    )

                    # Identify colorbar axes added by plot_field_magnitude.
                    new_figure_axes = [
                        figure_ax
                        for figure_ax in fig.axes
                        if figure_ax not in axes_before_plot
                        and figure_ax is not ax
                    ]

                    # Remove the verbose title generated for every individual
                    # field map. Only column headings are retained.
                    ax.set_title("")

                    # Explicitly keep the field-map panel square.
                    ax.set_aspect(
                        "equal",
                        adjustable="box",
                    )

                    # Use only a few major ticks. This prevents overlapping
                    # labels such as -10000, 0, and 10000.
                    ax.xaxis.set_major_locator(
                        MaxNLocator(nbins=3)
                    )
                    ax.yaxis.set_major_locator(
                        MaxNLocator(nbins=3)
                    )

                    ax.minorticks_off()

                    ax.tick_params(
                        axis="both",
                        which="major",
                        direction="out",
                        length=2.5,
                        width=0.6,
                        pad=1.5,
                        labelsize=6,
                    )

                    # Store the coordinate labels created by
                    # plot_field_magnitude before selectively hiding them.
                    original_xlabel = ax.get_xlabel()
                    original_ylabel = ax.get_ylabel()

                    # Show x labels and x tick labels only on the bottom row.
                    if row_index < len(FIELD_PLANES) - 1:
                        ax.set_xlabel("")
                        ax.tick_params(
                            axis="x",
                            labelbottom=False,
                        )
                    else:
                        ax.set_xlabel(
                            original_xlabel,
                            fontsize=7,
                            labelpad=2,
                        )

                    # Show y labels and y tick labels only in the first column.
                    if column_index > 0:
                        ax.set_ylabel("")
                        ax.tick_params(
                            axis="y",
                            labelleft=False,
                        )
                    else:
                        # Plane name is placed directly in the first-column
                        # y label, avoiding an additional title or annotation.
                        ax.set_ylabel(
                            f"{plane.upper()} plane\n{original_ylabel}",
                            fontsize=7,
                            labelpad=2,
                        )

                    # Add compact column headings only to the first row.
                    if row_index == 0:
                        ax.set_title(
                            FIELD_CASE_APPEARANCE[case_key]["short_label"],
                            fontsize=8,
                            pad=3,
                            linespacing=0.9,
                        )

                    # Compact any colorbar created for this panel.
                    for colorbar_ax in new_figure_axes:

                        colorbar_ax.tick_params(
                            which="both",
                            labelsize=5.5,
                            length=2.0,
                            width=0.5,
                            pad=1.5,
                        )

                        # Use fewer colorbar ticks.
                        colorbar_ax.yaxis.set_major_locator(
                            MaxNLocator(nbins=5)
                        )

                        # Keep the colorbar quantity label only in the top row.
                        # The lower two rows represent the same quantity.
                        if row_index > 0:
                            colorbar_ax.set_ylabel("")
                        else:
                            colorbar_ax.yaxis.label.set_size(6)
                            colorbar_ax.yaxis.labelpad = 2

            field_description = (
                "total electric-field magnitude"
                if settings["kind"] == "total"
                else "scattered electric-field magnitude"
            )

            fig.suptitle(
                rf"$\lambda={wavelength_nm:.0f}\,\mathrm{{nm}}$, "
                + field_description,
                fontsize=10,
            )

            if SAVE_FIELD_MAGNITUDE_MATRICES:

                basename = (
                    f"{settings_key}_"
                    f"{wavelength_nm:.0f}nm_"
                    "three_cases_three_planes"
                )

                png_path = (
                    FIELD_FIGURE_DIRECTORY
                    / f"{basename}.png"
                )

                pdf_path = (
                    FIELD_FIGURE_DIRECTORY
                    / f"{basename}.pdf"
                )

                fig.savefig(
                    png_path,
                    dpi=FIELD_MATRIX_PNG_DPI,
                    facecolor="white",
                    transparent=False,
                    bbox_inches="tight",
                    pad_inches=0.03,
                )

                # Text, axes, and boundaries remain vector-based in the PDF.
                fig.savefig(
                    pdf_path,
                    facecolor="white",
                    transparent=False,
                    bbox_inches="tight",
                    pad_inches=0.03,
                )

                print("Saved:", png_path)
                print("Saved:", pdf_path)

            plt.show()
            plt.close(fig)

# Cell 8: Plot Poynting-flow streamline matrices
#
# Change POYNTING_FLOW_KINDS_TO_PLOT to test total, delta, or scattered flow.

# In[249]:


# ============================================================
# PLOT COMPACT POYNTING-FLOW MATRICES
#
# One figure per wavelength and flow kind.
#
# Rows:
#   xy, xz, yz
#
# Columns:
#   reference
#   constant diameter
#   constant shell number
# ============================================================

from matplotlib.ticker import MaxNLocator
import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Plot selection
# ------------------------------------------------------------

POYNTING_FLOW_KINDS_TO_PLOT = [
    "total",
    "delta",
]


# ------------------------------------------------------------
# Figure settings
# ------------------------------------------------------------

# A square 20 cm figure is large enough for a 3x3 matrix while
# remaining suitable for viewing and high-resolution export.
#
# The previous values, 15.92 * 3, produced a 47.76 cm figure.
# That is unnecessarily large and may be downscaled poorly by
# the notebook interface or image viewer.
POYNTING_MATRIX_WIDTH_CM = 20.0
POYNTING_MATRIX_HEIGHT_CM = 20.0

# Resolution of saved PNG files.
POYNTING_MATRIX_PNG_DPI = 400

# Resolution of the figure shown inside the notebook.
POYNTING_MATRIX_NOTEBOOK_DPI = 150

SAVE_POYNTING_MATRICES = True


# ------------------------------------------------------------
# Determine the wavelengths available in field_maps
# ------------------------------------------------------------

if "field_maps" not in globals():
    raise NameError(
        "`field_maps` is not defined. Run the field-map calculation/loading "
        "cell before running this plotting cell."
    )

if not field_maps:
    raise ValueError(
        "`field_maps` exists but is empty. Calculate or load the field maps first."
    )

available_field_wavelengths_nm = sorted({
    float(key[1])
    for key in field_maps
    if isinstance(key, tuple) and len(key) >= 3
})

if not available_field_wavelengths_nm:
    raise ValueError(
        "No wavelengths could be extracted from `field_maps`. "
        "Expected keys like `(case_key, wavelength_nm, plane)`."
    )

print(
    "Plotting wavelengths:",
    ", ".join(
        f"{wavelength:g} nm"
        for wavelength in available_field_wavelengths_nm
    ),
)


# ------------------------------------------------------------
# Validate other required variables
# ------------------------------------------------------------

required_names = [
    "FIELD_PLANES",
    "FIELD_CASE_KEYS",
    "FIELD_CASE_APPEARANCE",
    "POYNTING_STREAMLINE_SETTINGS",
    "FIELD_FIGURE_DIRECTORY",
    "plot_poynting_streamlines",
]

missing_names = [
    name
    for name in required_names
    if name not in globals()
]

if missing_names:
    raise NameError(
        "The following required objects are not defined: "
        + ", ".join(missing_names)
    )


# ------------------------------------------------------------
# Create the matrices
# ------------------------------------------------------------

cm_to_inches = 1.0 / 2.54

with plt.rc_context({
    "figure.dpi": POYNTING_MATRIX_NOTEBOOK_DPI,
    "font.size": 7,
    "axes.titlesize": 8,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
}):

    for wavelength_nm in available_field_wavelengths_nm:

        for flow_key in POYNTING_FLOW_KINDS_TO_PLOT:

            if flow_key not in POYNTING_STREAMLINE_SETTINGS:
                print(
                    f"Skipping unknown Poynting setting: {flow_key!r}"
                )
                continue

            settings = dict(
                POYNTING_STREAMLINE_SETTINGS[flow_key]
            )

            fig, axes = plt.subplots(
                nrows=len(FIELD_PLANES),
                ncols=len(FIELD_CASE_KEYS),
                figsize=(
                    POYNTING_MATRIX_WIDTH_CM * cm_to_inches,
                    POYNTING_MATRIX_HEIGHT_CM * cm_to_inches,
                ),
                constrained_layout=True,
            )

            axes = np.asarray(
                axes,
                dtype=object,
            ).reshape(
                len(FIELD_PLANES),
                len(FIELD_CASE_KEYS),
            )

            # Reduce whitespace between maps while still allowing room
            # for tick labels and colorbars.
            fig.set_constrained_layout_pads(
                w_pad=0.015,
                h_pad=0.015,
                wspace=0.015,
                hspace=0.015,
            )

            for row_index, plane in enumerate(FIELD_PLANES):

                for column_index, case_key in enumerate(
                    FIELD_CASE_KEYS
                ):

                    ax = axes[
                        row_index,
                        column_index,
                    ]

                    map_key = (
                        case_key,
                        float(wavelength_nm),
                        plane,
                    )

                    if map_key not in field_maps:
                        ax.set_axis_off()
                        ax.text(
                            0.5,
                            0.5,
                            "Field map\nnot available",
                            ha="center",
                            va="center",
                            transform=ax.transAxes,
                            fontsize=8,
                        )

                        print(
                            "Missing field map:",
                            map_key,
                        )
                        continue

                    field_map = field_maps[map_key]

                    # Record existing axes before plotting so colorbar axes
                    # created by plot_poynting_streamlines can be identified.
                    axes_before_plot = set(fig.axes)


                    # POYNTING_STREAMLINE_SETTINGS = {
                    #     "total": {
                    #         "flow_kind": "total",
                    #         "background_quantity": "S",
                    #         "background_kind": "total",
                    #         "background_scale": "db",
                    #         "background_floor": 1e-30,
                    #         "background_cmap": "magma",
                    #         "background_clip_percentile_low": 0.0,
                    #         "background_clip_percentile_high": 99.5,
                    #         "streamline_density": 1.25,
                    #         "streamline_color": "white",
                    #         "streamline_linewidth": 0.70,
                    #         "normalize_vectors": False,
                    #         "min_speed_fraction": 0.005,
                    #         "mask_inside_sphere": False,
                    #         "show_boundaries": True,
                    #     },

                    plot_poynting_streamlines(
                        field_map,
                        flow_kind=(
                            settings["flow_kind"]
                        ),
                        background_quantity=(
                            settings["background_quantity"]
                        ),
                        background_kind=(
                            settings["background_kind"]
                        ),
                        background_scale=(
                            settings["background_scale"]
                        ),
                        background_floor=(
                            # settings["background_floor"]
                            1e-10
                        ),
                        background_cmap=(
                            settings["background_cmap"]
                        ),
                        background_clip_percentile_low=(
                            settings[
                                "background_clip_percentile_low"
                            ]
                        ),
                        background_clip_percentile_high=(
                            settings[
                                "background_clip_percentile_high"
                            ]
                        ),
                        streamline_density=(
                            settings["streamline_density"]
                        ),
                        streamline_color=(
                            settings["streamline_color"]
                        ),
                        streamline_linewidth=(
                            settings["streamline_linewidth"]
                        ),
                        normalize_vectors=(
                            settings["normalize_vectors"]
                        ),
                        min_speed_fraction=(
                            settings["min_speed_fraction"]
                        ),
                        mask_inside_sphere=(
                            settings["mask_inside_sphere"]
                        ),
                        show_boundaries=(
                            settings["show_boundaries"]
                        ),
                        ax=ax,
                    )

                    # Find axes added by the plotting function, normally
                    # the colorbar axes.
                    new_figure_axes = [
                        figure_ax
                        for figure_ax in fig.axes
                        if figure_ax not in axes_before_plot
                        and figure_ax is not ax
                    ]

                    # Save the labels created by plot_poynting_streamlines
                    # before removing redundant labels.
                    original_xlabel = ax.get_xlabel()
                    original_ylabel = ax.get_ylabel()

                    # Remove the verbose title generated for each map.
                    # A short column heading is added below.
                    ax.set_title("")

                    # Preserve the geometrical proportions of the field map.
                    ax.set_aspect(
                        "equal",
                        adjustable="box",
                    )

                    # Use approximately three coordinate ticks per axis.
                    ax.xaxis.set_major_locator(
                        MaxNLocator(nbins=3)
                    )
                    ax.yaxis.set_major_locator(
                        MaxNLocator(nbins=3)
                    )

                    ax.minorticks_off()

                    ax.tick_params(
                        axis="both",
                        which="major",
                        direction="out",
                        length=2.5,
                        width=0.6,
                        pad=1.5,
                        labelsize=6,
                    )

                    # Only show x-axis labels and tick labels in the
                    # bottom row.
                    if row_index < len(FIELD_PLANES) - 1:
                        ax.set_xlabel("")
                        ax.tick_params(
                            axis="x",
                            labelbottom=False,
                        )
                    else:
                        ax.set_xlabel(
                            original_xlabel,
                            fontsize=7,
                            labelpad=2,
                        )

                    # Only show y-axis labels and tick labels in the
                    # first column.
                    if column_index > 0:
                        ax.set_ylabel("")
                        ax.tick_params(
                            axis="y",
                            labelleft=False,
                        )
                    else:
                        ax.set_ylabel(
                            f"{plane.upper()} plane\n"
                            f"{original_ylabel}",
                            fontsize=7,
                            labelpad=2,
                        )

                    # Add a short heading only above the top row.
                    if row_index == 0:
                        ax.set_title(
                            FIELD_CASE_APPEARANCE[
                                case_key
                            ]["short_label"],
                            fontsize=8,
                            pad=3,
                            linespacing=0.9,
                        )

                    # Compact colorbar axes generated by the function.
                    for colorbar_ax in new_figure_axes:

                        colorbar_ax.tick_params(
                            which="both",
                            labelsize=5.5,
                            length=2.0,
                            width=0.5,
                            pad=1.5,
                        )

                        colorbar_ax.yaxis.set_major_locator(
                            MaxNLocator(nbins=5)
                        )

                        # Keep the full colorbar quantity label only in
                        # the top row. Numerical ticks remain on every
                        # colorbar because color limits may differ.
                        if row_index > 0:
                            colorbar_ax.set_ylabel("")
                        else:
                            colorbar_ax.yaxis.label.set_size(6)
                            colorbar_ax.yaxis.labelpad = 2

            # Make the flow-kind description more readable.
            flow_description = {
                "total": "total Poynting flow",
                "delta": "difference in Poynting flow",
            }.get(
                flow_key,
                f"{flow_key} Poynting flow",
            )

            fig.suptitle(
                rf"$\lambda={wavelength_nm:.0f}\,\mathrm{{nm}}$, "
                + flow_description,
                fontsize=10,
            )

            if SAVE_POYNTING_MATRICES:

                basename = (
                    f"poynting_{flow_key}_"
                    f"{wavelength_nm:.0f}nm_"
                    "three_cases_three_planes"
                )

                png_path = (
                    FIELD_FIGURE_DIRECTORY
                    / f"{basename}.png"
                )

                pdf_path = (
                    FIELD_FIGURE_DIRECTORY
                    / f"{basename}.pdf"
                )

                fig.savefig(
                    png_path,
                    dpi=POYNTING_MATRIX_PNG_DPI,
                    facecolor="white",
                    transparent=False,
                    bbox_inches="tight",
                    pad_inches=0.03,
                )

                fig.savefig(
                    pdf_path,
                    facecolor="white",
                    transparent=False,
                    bbox_inches="tight",
                    pad_inches=0.03,
                )

                print("Saved:", png_path)
                print("Saved:", pdf_path)

            plt.show()

            # Prevent accumulation of large figures in notebook memory.
            plt.close(fig)

# IMPROVEMENT TESTS

# In[250]:


# ============================================================
# ROBUST POYNTING-STREAMLINE PLOTTER
#
# Improvements over plot_poynting_streamlines():
#
# 1. Uses np.hypot(U, V), avoiding overflow from U**2 + V**2.
# 2. Masks NaN and infinity before percentile calculations.
# 3. Scales vectors before normalization or streamplot.
# 4. Uses percentile-based masking on finite values only.
# 5. Optionally uses logarithmic speed weights for smoother
#    streamline integration through strongly varying fields.
# 6. Supports explicit seed points inside and outside the sphere.
# 7. Can draw a single shared background without modifying the
#    original FieldMapResult.
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Circle


def _robust_in_plane_flow(
    field_result,
    flow_kind,
):
    """
    Return finite in-plane Poynting components and robust speed.
    """

    if flow_kind == "total":

        U = np.asarray(
            field_result.plane_u_total,
            dtype=float,
        ).copy()

        V = np.asarray(
            field_result.plane_v_total,
            dtype=float,
        ).copy()

    elif flow_kind == "scattered":

        U = np.asarray(
            field_result.plane_u_sca_like,
            dtype=float,
        ).copy()

        V = np.asarray(
            field_result.plane_v_sca_like,
            dtype=float,
        ).copy()

    elif flow_kind == "delta":

        U = np.asarray(
            field_result.plane_u_delta,
            dtype=float,
        ).copy()

        V = np.asarray(
            field_result.plane_v_delta,
            dtype=float,
        ).copy()

    else:

        raise ValueError(
            "flow_kind must be 'total', 'scattered', or 'delta'."
        )

    if U.shape != V.shape:

        raise ValueError(
            f"U and V shapes differ: {U.shape} and {V.shape}."
        )

    finite_components = (
        np.isfinite(U)
        & np.isfinite(V)
    )

    U[~finite_components] = np.nan
    V[~finite_components] = np.nan

    # np.hypot is numerically stable for very large components.
    speed = np.hypot(
        U,
        V,
    )

    speed[~np.isfinite(speed)] = np.nan

    return U, V, speed


def _select_poynting_background(
    field_result,
    *,
    background_kind,
):
    """
    Select the requested Poynting magnitude.
    """

    if background_kind == "total":

        return np.asarray(
            field_result.s_total_mag,
            dtype=float,
        )

    if background_kind == "scattered":

        return np.asarray(
            field_result.s_sca_like_mag,
            dtype=float,
        )

    if background_kind == "delta":

        return np.asarray(
            field_result.s_delta_mag,
            dtype=float,
        )

    raise ValueError(
        "background_kind must be "
        "'total', 'scattered', or 'delta'."
    )


def _robust_db_background(
    values,
    *,
    floor_fraction=1e-10,
    lower_percentile=1.0,
    upper_percentile=99.5,
):
    """
    Convert a nonnegative magnitude to a relative dB scale.

    The plotted maximum is 0 dB. This is much more interpretable
    than absolute dB values exceeding 1000 due to arbitrary units.
    """

    values = np.asarray(
        values,
        dtype=float,
    )

    valid = (
        np.isfinite(values)
        & (values >= 0.0)
    )

    if not np.any(valid):

        raise ValueError(
            "The background contains no finite nonnegative values."
        )

    finite_values = values[valid]

    reference_value = float(
        np.nanpercentile(
            finite_values,
            upper_percentile,
        )
    )

    if (
        not np.isfinite(reference_value)
        or reference_value <= 0.0
    ):

        reference_value = float(
            np.nanmax(finite_values)
        )

    floor_value = max(
        reference_value
        * float(floor_fraction),
        np.finfo(float).tiny,
    )

    background_db = np.full_like(
        values,
        np.nan,
        dtype=float,
    )

    background_db[valid] = (
        10.0
        * np.log10(
            np.maximum(
                values[valid],
                floor_value,
            )
            / reference_value
        )
    )

    finite_db = background_db[
        np.isfinite(background_db)
    ]

    vmin = float(
        np.nanpercentile(
            finite_db,
            lower_percentile,
        )
    )

    vmax = 0.0

    return (
        background_db,
        vmin,
        vmax,
    )


def make_robust_streamline_seeds(
    field_result,
    *,
    outside_seed_count=31,
    inside_seed_count=25,
    include_inside=True,
    edge_fraction_min=0.05,
    edge_fraction_max=0.95,
):
    """
    Create deterministic seeds for a +z-propagating incident wave.

    For xz and yz planes, the second plotting coordinate is z.
    Seeds are placed:
      - close to the lower z boundary;
      - through a vertical line crossing the sphere interior.

    For the xy plane, the incident +z flow is perpendicular to the
    plotting plane, so in-plane streamlines are generally not useful.
    """

    plane = field_result.plane

    c1_min = float(
        field_result.coord1_m[0]
        * 1e9
    )

    c1_max = float(
        field_result.coord1_m[-1]
        * 1e9
    )

    c2_min = float(
        field_result.coord2_m[0]
        * 1e9
    )

    c2_max = float(
        field_result.coord2_m[-1]
        * 1e9
    )

    outer_radius_nm = float(
        field_result.outer_radius_m
        * 1e9
    )

    if plane == "xy":

        warnings.warn(
            "The xy plane contains Sx and Sy, while the incident "
            "power propagates along z, perpendicular to the plane. "
            "In-plane total-flow streamlines may therefore be absent "
            "or physically uninformative."
        )

        return None

    lateral_coordinates = np.linspace(
        c1_min
        + edge_fraction_min
        * (c1_max - c1_min),
        c1_min
        + edge_fraction_max
        * (c1_max - c1_min),
        outside_seed_count,
    )

    grid_spacing_nm = abs(
        float(
            field_result.coord2_m[1]
            - field_result.coord2_m[0]
        )
        * 1e9
    )

    outside_z = (
        c2_min
        + 1.5
        * grid_spacing_nm
    )

    outside_seeds = np.column_stack(
        [
            lateral_coordinates,
            np.full_like(
                lateral_coordinates,
                outside_z,
            ),
        ]
    )

    if not include_inside:

        return outside_seeds

    # Internal seeds are placed across the incident-facing half
    # of the sphere, rather than exactly on material boundaries.
    internal_lateral = np.linspace(
        -0.90 * outer_radius_nm,
        +0.90 * outer_radius_nm,
        inside_seed_count,
    )

    internal_z = np.full_like(
        internal_lateral,
        -0.85 * outer_radius_nm,
    )

    inside_seeds = np.column_stack(
        [
            internal_lateral,
            internal_z,
        ]
    )

    seeds = np.vstack(
        [
            outside_seeds,
            inside_seeds,
        ]
    )

    within_box = (
        (seeds[:, 0] > c1_min)
        & (seeds[:, 0] < c1_max)
        & (seeds[:, 1] > c2_min)
        & (seeds[:, 1] < c2_max)
    )

    return seeds[
        within_box
    ]


def plot_poynting_streamlines_robust(
    field_result,
    *,
    flow_kind="total",
    background_kind="total",
    background_cmap="magma",
    background_floor_fraction=1e-10,
    background_clip_percentile_low=1.0,
    background_clip_percentile_high=99.5,
    streamline_color="cyan",
    streamline_linewidth=0.85,
    streamline_density=1.0,
    streamline_arrowsize=1.0,
    integration_direction="forward",
    minlength=0.04,
    maxlength=8.0,
    broken_streamlines=True,
    normalize_vectors=True,
    vector_dynamic_range_decades=3.0,
    min_speed_percentile=0.5,
    mask_inside_sphere=False,
    start_points_nm=None,
    show_boundaries=True,
    show_outer_boundary=True,
    ax=None,
):
    """
    Robust Poynting streamline visualization.

    Parameters
    ----------
    vector_dynamic_range_decades
        Before streamplot, vector magnitudes are compressed to at most
        this many orders of magnitude. This prevents hot spots from
        dominating integration while preserving direction.

    min_speed_percentile
        Mask only the lowest percentile of finite in-plane speeds.
        For example, 0.5 removes the weakest 0.5% rather than using a
        fraction of an enormous maximum.

    normalize_vectors
        If True, all retained vectors are unit vectors. Recommended for
        visualizing flow topology.

    integration_direction
        "forward" is recommended when seeds are placed on the incident
        side. "both" may trace streamlines back through unrelated regions.
    """

    if ax is None:

        _, ax = plt.subplots(
            figsize=(6, 6)
        )

    U, V, speed = (
        _robust_in_plane_flow(
            field_result,
            flow_kind,
        )
    )

    finite_speed = (
        np.isfinite(speed)
        & (speed > 0.0)
    )

    if not np.any(finite_speed):

        raise ValueError(
            "No finite positive in-plane Poynting speed is available."
        )

    # --------------------------------------------------------
    # Background
    # --------------------------------------------------------

    background_magnitude = (
        _select_poynting_background(
            field_result,
            background_kind=(
                background_kind
            ),
        )
    )

    (
        background_plot,
        background_vmin,
        background_vmax,
    ) = _robust_db_background(
        background_magnitude,
        floor_fraction=(
            background_floor_fraction
        ),
        lower_percentile=(
            background_clip_percentile_low
        ),
        upper_percentile=(
            background_clip_percentile_high
        ),
    )

    extent_nm = [
        float(
            field_result.coord1_m[0]
            * 1e9
        ),
        float(
            field_result.coord1_m[-1]
            * 1e9
        ),
        float(
            field_result.coord2_m[0]
            * 1e9
        ),
        float(
            field_result.coord2_m[-1]
            * 1e9
        ),
    ]

    image = ax.imshow(
        background_plot,
        origin="lower",
        extent=extent_nm,
        cmap=background_cmap,
        vmin=background_vmin,
        vmax=background_vmax,
        aspect="equal",
    )

    colorbar = plt.colorbar(
        image,
        ax=ax,
        fraction=0.046,
        pad=0.025,
    )

    colorbar.set_label(
        (
            rf"$10\log_{{10}}"
            rf"(|S_{{\mathrm{{{background_kind}}}}}|/"
            rf"S_{{99.5}})$ [dB]"
        )
    )

    # --------------------------------------------------------
    # Robust masks
    # --------------------------------------------------------

    mask = (
        ~np.isfinite(U)
        | ~np.isfinite(V)
        | ~np.isfinite(speed)
        | (speed <= 0.0)
    )

    if mask_inside_sphere:

        mask |= (
            field_result.radial_distance_m
            <= field_result.outer_radius_m
        )

    finite_unmasked_speed = speed[
        ~mask
    ]

    if finite_unmasked_speed.size == 0:

        raise ValueError(
            "All flow vectors were masked before percentile filtering."
        )

    if (
        min_speed_percentile is not None
        and min_speed_percentile > 0.0
    ):

        speed_threshold = float(
            np.nanpercentile(
                finite_unmasked_speed,
                min_speed_percentile,
            )
        )

        mask |= (
            speed < speed_threshold
        )

    # --------------------------------------------------------
    # Rescale before streamplot
    # --------------------------------------------------------

    robust_scale = float(
        np.nanpercentile(
            speed[
                ~mask
            ],
            99.0,
        )
    )

    if (
        not np.isfinite(robust_scale)
        or robust_scale <= 0.0
    ):

        robust_scale = float(
            np.nanmax(
                speed[
                    ~mask
                ]
            )
        )

    U_scaled = (
        U / robust_scale
    )

    V_scaled = (
        V / robust_scale
    )

    speed_scaled = np.hypot(
        U_scaled,
        V_scaled,
    )

    # Compress extreme dynamic range while retaining directions.
    minimum_relative_speed = (
        10.0
        ** (
            -float(
                vector_dynamic_range_decades
            )
        )
    )

    valid_for_scaling = (
        ~mask
        & np.isfinite(speed_scaled)
        & (speed_scaled > 0.0)
    )

    if normalize_vectors:

        U_plot = np.zeros_like(
            U_scaled
        )

        V_plot = np.zeros_like(
            V_scaled
        )

        U_plot[
            valid_for_scaling
        ] = (
            U_scaled[
                valid_for_scaling
            ]
            / speed_scaled[
                valid_for_scaling
            ]
        )

        V_plot[
            valid_for_scaling
        ] = (
            V_scaled[
                valid_for_scaling
            ]
            / speed_scaled[
                valid_for_scaling
            ]
        )

    else:

        clipped_relative_speed = np.clip(
            speed_scaled,
            minimum_relative_speed,
            1.0,
        )

        # Logarithmic compression gives weak and strong regions
        # more comparable streamline integration step lengths.
        compressed_speed = (
            (
                np.log10(
                    clipped_relative_speed
                )
                - np.log10(
                    minimum_relative_speed
                )
            )
            / (
                -np.log10(
                    minimum_relative_speed
                )
            )
        )

        compressed_speed = (
            0.05
            + 0.95
            * compressed_speed
        )

        U_plot = np.zeros_like(
            U_scaled
        )

        V_plot = np.zeros_like(
            V_scaled
        )

        U_plot[
            valid_for_scaling
        ] = (
            U_scaled[
                valid_for_scaling
            ]
            / speed_scaled[
                valid_for_scaling
            ]
            * compressed_speed[
                valid_for_scaling
            ]
        )

        V_plot[
            valid_for_scaling
        ] = (
            V_scaled[
                valid_for_scaling
            ]
            / speed_scaled[
                valid_for_scaling
            ]
            * compressed_speed[
                valid_for_scaling
            ]
        )

    U_masked = np.ma.array(
        U_plot,
        mask=mask,
    )

    V_masked = np.ma.array(
        V_plot,
        mask=mask,
    )

    # --------------------------------------------------------
    # Streamlines
    # --------------------------------------------------------

    X_nm = (
        field_result.coord1_m
        * 1e9
    )

    Y_nm = (
        field_result.coord2_m
        * 1e9
    )

    streamplot_arguments = {
        "x": X_nm,
        "y": Y_nm,
        "u": U_masked,
        "v": V_masked,
        "color": streamline_color,
        "density": streamline_density,
        "linewidth": streamline_linewidth,
        "arrowsize": streamline_arrowsize,
        "arrowstyle": "->",
        "integration_direction": (
            integration_direction
        ),
        "minlength": minlength,
        "maxlength": maxlength,
        "broken_streamlines": (
            broken_streamlines
        ),
        "zorder": 5,
    }

    if start_points_nm is not None:

        start_points_nm = np.asarray(
            start_points_nm,
            dtype=float,
        )

        if (
            start_points_nm.ndim != 2
            or start_points_nm.shape[1] != 2
        ):

            raise ValueError(
                "start_points_nm must have shape (N, 2)."
            )

        streamplot_arguments[
            "start_points"
        ] = start_points_nm

    streamline_set = ax.streamplot(
        **streamplot_arguments
    )

    # --------------------------------------------------------
    # Boundaries
    # --------------------------------------------------------

    if show_boundaries:

        for radius_m in (
            field_result.radii_m
        ):

            radius_nm = float(
                radius_m
                * 1e9
            )

            circle = Circle(
                (0.0, 0.0),
                radius=radius_nm,
                fill=False,
                edgecolor="white",
                linewidth=0.25,
                alpha=0.9,
                zorder=6,
            )

            ax.add_patch(
                circle
            )

    if show_outer_boundary:

        outer_circle = Circle(
            (0.0, 0.0),
            radius=(
                field_result.outer_radius_m
                * 1e9
            ),
            fill=False,
            edgecolor="white",
            linewidth=0.9,
            alpha=0.85,
            zorder=7,
        )

        ax.add_patch(
            outer_circle
        )

    plane_labels = {
        "xy": (
            "x [nm]",
            "y [nm]",
        ),
        "xz": (
            "x [nm]",
            "z [nm]",
        ),
        "yz": (
            "y [nm]",
            "z [nm]",
        ),
    }

    ax.set_xlabel(
        plane_labels[
            field_result.plane
        ][0]
    )

    ax.set_ylabel(
        plane_labels[
            field_result.plane
        ][1]
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    return (
        ax,
        streamline_set,
        {
            "number_valid_vectors": int(
                np.count_nonzero(
                    ~mask
                )
            ),
            "number_masked_vectors": int(
                np.count_nonzero(
                    mask
                )
            ),
            "masked_fraction": float(
                np.mean(mask)
            ),
            "speed_percentile_50": float(
                np.nanpercentile(
                    finite_unmasked_speed,
                    50.0,
                )
            ),
            "speed_percentile_99": float(
                np.nanpercentile(
                    finite_unmasked_speed,
                    99.0,
                )
            ),
            "robust_scale": (
                robust_scale
            ),
        },
    )

# In[305]:


# ============================================================
# SINGLE-MAP POYNTING STREAMLINE TEST
#
# Use this cell while optimizing streamline settings.
# ============================================================

TEST_FLOW_KIND = "total"

TEST_CASE_KEY = "reference"
TEST_WAVELENGTH_NM = 502.0
TEST_PLANE = "xz"

# If the exact wavelength is unavailable, select the nearest one.
available_test_keys = [
    key
    for key in field_maps
    if (
        isinstance(key, tuple)
        and len(key) == 3
        and key[0] == TEST_CASE_KEY
        and key[2] == TEST_PLANE
    )
]

if not available_test_keys:

    raise KeyError(
        "No field map is available for the requested "
        f"case={TEST_CASE_KEY!r} and plane={TEST_PLANE!r}."
    )

test_key = min(
    available_test_keys,
    key=lambda key: abs(
        float(key[1])
        - float(
            TEST_WAVELENGTH_NM
        )
    ),
)

test_field_map = field_maps[
    test_key
]

print(
    "Testing map:",
    test_key,
)


# For +z incidence in xz or yz, seed the incident-facing lower edge
# and an internal line through the sphere.
test_seed_points_nm = (
    make_robust_streamline_seeds(
        test_field_map,
        outside_seed_count=50,
        inside_seed_count=50,
        include_inside=True,
    )
)


fig, ax = plt.subplots(
    figsize=(7.2, 6.2),
    constrained_layout=True,
)


ax, streamline_set, diagnostics = (
    plot_poynting_streamlines_robust(
        test_field_map,
        flow_kind=TEST_FLOW_KIND,
        background_kind=TEST_FLOW_KIND,
        background_cmap="magma",

        # Background: relative dB scale from -80 to 0 dB,
        # approximately, depending on the lower percentile.
        background_floor_fraction=1e-8,
        background_clip_percentile_low=0.0,
        background_clip_percentile_high=75,

        # Bright cyan or lime is usually more visible on magma
        # than white over a saturated core.
        streamline_color="#00E5FF",
        streamline_linewidth=1.2,
        streamline_density=3.0,
        streamline_arrowsize=2.0,

        # Follow energy from the incident-facing seeds toward +z.
        integration_direction="forward",

        minlength=0.025,
        maxlength=12.5,
        broken_streamlines=False,

        # Recommended for topology.
        normalize_vectors=False,

        # Retain three decades when normalize_vectors=False.
        vector_dynamic_range_decades=3.0,

        # Remove only the weakest 0.5% of finite speeds.
        min_speed_percentile=0.5,

        # Keep internal flow visible.
        mask_inside_sphere=False,

        start_points_nm=test_seed_points_nm,

        # Showing all 120 boundaries can obscure the flow.
        show_boundaries=True,
        show_outer_boundary=True,

        ax=ax,
    )
)


ax.set_title(
    (
        f"{TEST_CASE_KEY.replace('_', ' ')}, "
        f"{test_key[1]:.0f} nm, "
        f"{TEST_PLANE.upper()} plane\n"
        f"{TEST_FLOW_KIND} Poynting flow"
    ),
    fontsize=10,
)

display(
    pd.DataFrame(
        [diagnostics]
    ).style.format(
        {
            "masked_fraction": "{:.2%}",
            "speed_percentile_50": "{:.4g}",
            "speed_percentile_99": "{:.4g}",
            "robust_scale": "{:.4g}",
        }
    )
)

plt.show()

# In[254]:


# ============================================================
# ROBUST POYNTING-FLOW MATRICES
#
# Rows:
#   xz and yz planes
#
# Columns:
#   reference
#   constant diameter
#   constant shell number
#
# Set OPTIMIZATION_MODE=True to create only one figure and one
# panel for rapid parameter tuning.
# ============================================================

from matplotlib.ticker import MaxNLocator


# ------------------------------------------------------------
# Selection
# ------------------------------------------------------------

FLOW_PLANES_TO_PLOT = [
    "xz",
    # "yz",
]

POYNTING_FLOW_KINDS_TO_PLOT = [
    "total",
    "delta",
]

OPTIMIZATION_MODE = False

OPTIMIZATION_CASE_KEY = "reference"
OPTIMIZATION_PLANE = "xz"
OPTIMIZATION_FLOW_KIND = "total"

# None means nearest/first available wavelength.
OPTIMIZATION_WAVELENGTH_NM = 502.0


# ------------------------------------------------------------
# Appearance
# ------------------------------------------------------------

POYNTING_MATRIX_WIDTH_CM = 18.0
POYNTING_MATRIX_HEIGHT_CM = 11.5

POYNTING_MATRIX_PNG_DPI = 400
POYNTING_MATRIX_NOTEBOOK_DPI = 150

SAVE_POYNTING_MATRICES = True


# ------------------------------------------------------------
# Robust streamline settings
# ------------------------------------------------------------

ROBUST_POYNTING_SETTINGS = {
    "total": {
        "flow_kind": "total",
        "background_kind": "total",
        "background_cmap": "magma",
        "background_floor_fraction": 1e-8,
        "background_clip_percentile_low": 1.0,
        "background_clip_percentile_high": 99.5,
        "streamline_color": "#00E5FF",
        "streamline_linewidth": 0.75,
        "streamline_density": 1.0,
        "streamline_arrowsize": 0.9,
        "integration_direction": "forward",
        "minlength": 0.025,
        "maxlength": 12.0,
        "broken_streamlines": True,
        "normalize_vectors": True,
        "vector_dynamic_range_decades": 3.0,
        "min_speed_percentile": 0.5,
        "mask_inside_sphere": False,
        "show_boundaries": False,
        "show_outer_boundary": True,
    },
    "delta": {
        "flow_kind": "delta",
        "background_kind": "delta",
        "background_cmap": "magma",
        "background_floor_fraction": 1e-8,
        "background_clip_percentile_low": 1.0,
        "background_clip_percentile_high": 99.5,
        "streamline_color": "#00E5FF",
        "streamline_linewidth": 0.75,
        "streamline_density": 1.0,
        "streamline_arrowsize": 0.9,
        "integration_direction": "both",
        "minlength": 0.025,
        "maxlength": 12.0,
        "broken_streamlines": True,
        "normalize_vectors": True,
        "vector_dynamic_range_decades": 3.0,
        "min_speed_percentile": 0.5,
        "mask_inside_sphere": False,
        "show_boundaries": False,
        "show_outer_boundary": True,
    },
}


# ------------------------------------------------------------
# Select maps
# ------------------------------------------------------------

available_keys = [
    key
    for key in field_maps
    if (
        isinstance(key, tuple)
        and len(key) == 3
    )
]

available_wavelengths_nm = sorted(
    {
        float(key[1])
        for key in available_keys
    }
)


if OPTIMIZATION_MODE:

    selected_flow_kinds = [
        OPTIMIZATION_FLOW_KIND
    ]

    selected_planes = [
        OPTIMIZATION_PLANE
    ]

    selected_case_keys = [
        OPTIMIZATION_CASE_KEY
    ]

    selected_wavelengths_nm = [
        min(
            available_wavelengths_nm,
            key=lambda value: abs(
                value
                - float(
                    OPTIMIZATION_WAVELENGTH_NM
                )
            ),
        )
    ]

else:

    selected_flow_kinds = (
        POYNTING_FLOW_KINDS_TO_PLOT
    )

    selected_planes = [
        plane
        for plane in FLOW_PLANES_TO_PLOT
        if any(
            key[2] == plane
            for key in available_keys
        )
    ]

    selected_case_keys = list(
        FIELD_CASE_KEYS
    )

    selected_wavelengths_nm = (
        available_wavelengths_nm
    )


print(
    "Flow kinds:",
    selected_flow_kinds,
)

print(
    "Planes:",
    selected_planes,
)

print(
    "Cases:",
    selected_case_keys,
)

print(
    "Wavelengths:",
    selected_wavelengths_nm,
)


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

cm_to_inches = 1.0 / 2.54


with plt.rc_context(
    {
        "figure.dpi": (
            POYNTING_MATRIX_NOTEBOOK_DPI
        ),
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
    }
):

    for wavelength_nm in (
        selected_wavelengths_nm
    ):

        for flow_key in (
            selected_flow_kinds
        ):

            settings = dict(
                ROBUST_POYNTING_SETTINGS[
                    flow_key
                ]
            )

            fig, axes = plt.subplots(
                nrows=len(
                    selected_planes
                ),
                ncols=len(
                    selected_case_keys
                ),
                figsize=(
                    POYNTING_MATRIX_WIDTH_CM
                    * cm_to_inches,
                    POYNTING_MATRIX_HEIGHT_CM
                    * cm_to_inches,
                ),
                squeeze=False,
                constrained_layout=True,
            )

            diagnostic_records = []

            for row_index, plane in enumerate(
                selected_planes
            ):

                for column_index, case_key in enumerate(
                    selected_case_keys
                ):

                    ax = axes[
                        row_index,
                        column_index,
                    ]

                    map_key = (
                        case_key,
                        float(
                            wavelength_nm
                        ),
                        plane,
                    )

                    if map_key not in field_maps:

                        ax.set_axis_off()

                        ax.text(
                            0.5,
                            0.5,
                            "Field map\nnot available",
                            transform=ax.transAxes,
                            ha="center",
                            va="center",
                        )

                        continue

                    field_map = field_maps[
                        map_key
                    ]

                    seed_points_nm = (
                        make_robust_streamline_seeds(
                            field_map,
                            outside_seed_count=31,
                            inside_seed_count=25,
                            include_inside=True,
                        )
                    )

                    (
                        ax,
                        streamline_set,
                        diagnostics,
                    ) = (
                        plot_poynting_streamlines_robust(
                            field_map,
                            start_points_nm=(
                                seed_points_nm
                            ),
                            ax=ax,
                            **settings,
                        )
                    )

                    diagnostics.update(
                        {
                            "case": case_key,
                            "plane": plane,
                            "wavelength_nm": (
                                wavelength_nm
                            ),
                            "flow_kind": (
                                flow_key
                            ),
                        }
                    )

                    diagnostic_records.append(
                        diagnostics
                    )

                    ax.set_title("")

                    ax.set_aspect(
                        "equal",
                        adjustable="box",
                    )

                    ax.xaxis.set_major_locator(
                        MaxNLocator(
                            nbins=3
                        )
                    )

                    ax.yaxis.set_major_locator(
                        MaxNLocator(
                            nbins=3
                        )
                    )

                    if (
                        row_index
                        < len(
                            selected_planes
                        )
                        - 1
                    ):

                        ax.set_xlabel("")

                        ax.tick_params(
                            axis="x",
                            labelbottom=False,
                        )

                    if column_index > 0:

                        ax.set_ylabel("")

                        ax.tick_params(
                            axis="y",
                            labelleft=False,
                        )

                    else:

                        current_ylabel = (
                            ax.get_ylabel()
                        )

                        ax.set_ylabel(
                            (
                                plane.upper()
                                + " plane\n"
                                + current_ylabel
                            )
                        )

                    if row_index == 0:

                        ax.set_title(
                            FIELD_CASE_APPEARANCE[
                                case_key
                            ][
                                "short_label"
                            ],
                            pad=3,
                        )

            flow_description = {
                "total": (
                    "total Poynting flow"
                ),
                "delta": (
                    "incident-subtracted Poynting flow"
                ),
                "scattered": (
                    "scattered-like Poynting flow"
                ),
            }[
                flow_key
            ]

            fig.suptitle(
                (
                    rf"$\lambda="
                    f"{wavelength_nm:.0f}"
                    r"\,\mathrm{nm}$, "
                    + flow_description
                ),
                fontsize=10,
            )

            if diagnostic_records:

                display(
                    pd.DataFrame(
                        diagnostic_records
                    ).style.format(
                        {
                            "masked_fraction": "{:.2%}",
                            "speed_percentile_50": "{:.4g}",
                            "speed_percentile_99": "{:.4g}",
                            "robust_scale": "{:.4g}",
                        }
                    )
                )

            if SAVE_POYNTING_MATRICES:

                optimization_suffix = (
                    "_test"
                    if OPTIMIZATION_MODE
                    else ""
                )

                basename = (
                    f"poynting_robust_"
                    f"{flow_key}_"
                    f"{wavelength_nm:.0f}nm"
                    f"{optimization_suffix}"
                )

                png_path = (
                    FIELD_FIGURE_DIRECTORY
                    / f"{basename}.png"
                )

                pdf_path = (
                    FIELD_FIGURE_DIRECTORY
                    / f"{basename}.pdf"
                )

                fig.savefig(
                    png_path,
                    dpi=(
                        POYNTING_MATRIX_PNG_DPI
                    ),
                    facecolor="white",
                    transparent=False,
                    bbox_inches="tight",
                    pad_inches=0.03,
                )

                fig.savefig(
                    pdf_path,
                    facecolor="white",
                    transparent=False,
                    bbox_inches="tight",
                    pad_inches=0.03,
                )

                print(
                    "Saved:",
                    png_path,
                )

                print(
                    "Saved:",
                    pdf_path,
                )

            plt.show()

            plt.close(
                fig
            )

# ALL BACK TO THE BEGINNING FOR THE CROSS-SECTIONAL ANALYSIS

# In[302]:


import bragg_onion.fields

from importlib import reload

reload(
    bragg_onion.fields
)

from bragg_onion.fields import (
    plot_field_magnitude,
    plot_poynting_streamlines_safe,
    make_distributed_seeds,
)

# Part 2: Clean independent notebook setup
#
# This cell assumes only that field_maps already contains calculated maps.
#
# Cell A: Validate and select maps

# In[292]:


# ===========================================================
# FIELD AND POYNTING PLOT SELECTION
#
# Electric-field maps:
#   xz, xy, and optionally yz
#
# Poynting streamlines:
#   xz only
# ============================================================

import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.ticker import MaxNLocator
from IPython.display import display

from bragg_onion.fields import (
    plot_field_magnitude,
    plot_poynting_streamlines_safe,
    make_distributed_seeds,
    make_edge_seeds,
)


if (
    "field_maps" not in globals()
    or not field_maps
):

    raise NameError(
        "field_maps is unavailable or empty."
    )


available_field_keys = sorted(
    [
        key
        for key in field_maps
        if (
            isinstance(key, tuple)
            and len(key) == 3
        )
    ],
    key=lambda key: (
        str(key[0]),
        float(key[1]),
        str(key[2]),
    ),
)


available_field_cases = sorted(
    {
        str(key[0])
        for key in available_field_keys
    }
)

available_field_wavelengths_nm = sorted(
    {
        float(key[1])
        for key in available_field_keys
    }
)

available_field_planes = sorted(
    {
        str(key[2])
        for key in available_field_keys
    }
)


print(
    "Available cases:",
    available_field_cases,
)

print(
    "Available wavelengths:",
    available_field_wavelengths_nm,
)

print(
    "Available planes:",
    available_field_planes,
)


# ------------------------------------------------------------
# User selections
# ------------------------------------------------------------

SELECTED_FIELD_CASES = [
    case_key
    for case_key in [
        "reference",
        "constant_diameter",
        "constant_shell_number",
    ]
    if case_key
    in available_field_cases
]

SELECTED_FIELD_WAVELENGTHS_NM = (
    available_field_wavelengths_nm
)

ELECTRIC_FIELD_PLANES = [
    plane
    for plane in [
        "xy",
        "xz",
        "yz",
    ]
    if plane
    in available_field_planes
]

POYNTING_PLANES = [
    plane
    for plane in [
        "xz",
    ]
    if plane
    in available_field_planes
]


FIELD_OUTPUT_DIRECTORY = (
    FIGURE_DIR
    / "improved_field_maps"
)

FIELD_OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)


FIELD_CASE_LABELS = {
    "reference": "Reference",
    "constant_diameter": (
        "Core absorber\nconstant diameter"
    ),
    "constant_shell_number": (
        "Core absorber\nconstant shell number"
    ),
}


print(
    "\nSelected electric-field planes:",
    ELECTRIC_FIELD_PLANES,
)

print(
    "Selected Poynting planes:",
    POYNTING_PLANES,
)

# Part 3: Clean electric-field plotting cell
#
# This independently produces xyxyxy, xzxzxz, and optionally yzyzyz maps.
#
# Cell B: Electric-field matrices

# In[293]:


# ===========================================================
# ELECTRIC-FIELD MAP MATRICES
# Rows:
#   selected center planes
# Columns:
#   selected physical cases
#
# One figure per wavelength and field kind.
# ===========================================================

ELECTRIC_FIELD_KINDS = [
    "total",
    "scattered",
]

ELECTRIC_FIELD_SCALE = "db"
ELECTRIC_FIELD_CMAP = "turbo"

ELECTRIC_FIELD_FLOOR = 1e-12

ELECTRIC_FIELD_CLIP_LOW = 0.0
ELECTRIC_FIELD_CLIP_HIGH = 99.0

SHOW_ELECTRIC_FIELD_BOUNDARIES = True

ELECTRIC_FIELD_FIGURE_WIDTH_CM = 17.5

ELECTRIC_FIELD_FIGURE_HEIGHT_CM = (
    5.2
    * max(
        1,
        len(
            ELECTRIC_FIELD_PLANES
        ),
    )
)

SAVE_ELECTRIC_FIELD_FIGURES = True

ELECTRIC_FIELD_DPI = 400


cm_to_inches = 1.0 / 2.54


for wavelength_nm in (
    SELECTED_FIELD_WAVELENGTHS_NM
):

    for field_kind in (
        ELECTRIC_FIELD_KINDS
    ):

        fig, axes = plt.subplots(
            nrows=len(
                ELECTRIC_FIELD_PLANES
            ),
            ncols=len(
                SELECTED_FIELD_CASES
            ),
            figsize=(
                ELECTRIC_FIELD_FIGURE_WIDTH_CM
                * cm_to_inches,
                ELECTRIC_FIELD_FIGURE_HEIGHT_CM
                * cm_to_inches,
            ),
            squeeze=False,
            constrained_layout=True,
        )

        for row_index, plane in enumerate(
            ELECTRIC_FIELD_PLANES
        ):

            for column_index, case_key in enumerate(
                SELECTED_FIELD_CASES
            ):

                ax = axes[
                    row_index,
                    column_index,
                ]

                map_key = (
                    case_key,
                    float(
                        wavelength_nm
                    ),
                    plane,
                )

                if map_key not in field_maps:

                    ax.set_axis_off()

                    ax.text(
                        0.5,
                        0.5,
                        "Map not available",
                        transform=ax.transAxes,
                        ha="center",
                        va="center",
                    )

                    continue

                field_map = field_maps[
                    map_key
                ]

                plot_field_magnitude(
                    field_map,
                    quantity="E",
                    kind=field_kind,
                    scale=(
                        ELECTRIC_FIELD_SCALE
                    ),
                    floor=(
                        ELECTRIC_FIELD_FLOOR
                    ),
                    cmap=(
                        ELECTRIC_FIELD_CMAP
                    ),
                    clip_percentile_low=(
                        ELECTRIC_FIELD_CLIP_LOW
                    ),
                    clip_percentile_high=(
                        ELECTRIC_FIELD_CLIP_HIGH
                    ),
                    show_boundaries=(
                        SHOW_ELECTRIC_FIELD_BOUNDARIES
                    ),
                    ax=ax,
                )

                ax.set_title("")

                if row_index == 0:

                    ax.set_title(
                        FIELD_CASE_LABELS.get(
                            case_key,
                            case_key,
                        ),
                        fontsize=8,
                    )

                if column_index == 0:

                    original_ylabel = (
                        ax.get_ylabel()
                    )

                    ax.set_ylabel(
                        (
                            plane.upper()
                            + " plane\n"
                            + original_ylabel
                        ),
                        fontsize=7,
                    )

                else:

                    ax.set_ylabel("")

                    ax.tick_params(
                        axis="y",
                        labelleft=False,
                    )

                if (
                    row_index
                    < len(
                        ELECTRIC_FIELD_PLANES
                    )
                    - 1
                ):

                    ax.set_xlabel("")

                    ax.tick_params(
                        axis="x",
                        labelbottom=False,
                    )

                ax.set_aspect(
                    "equal",
                    adjustable="box",
                )

                ax.xaxis.set_major_locator(
                    MaxNLocator(
                        nbins=3
                    )
                )

                ax.yaxis.set_major_locator(
                    MaxNLocator(
                        nbins=3
                    )
                )

        field_description = {
            "total": (
                "total electric-field magnitude"
            ),
            "scattered": (
                "scattered-like electric-field magnitude"
            ),
        }[
            field_kind
        ]

        fig.suptitle(
            (
                rf"$\lambda="
                f"{wavelength_nm:.0f}"
                r"\,\mathrm{nm}$, "
                + field_description
            ),
            fontsize=10,
        )

        if SAVE_ELECTRIC_FIELD_FIGURES:

            basename = (
                f"electric_field_"
                f"{field_kind}_"
                f"{wavelength_nm:.0f}nm"
            )

            png_path = (
                FIELD_OUTPUT_DIRECTORY
                / f"{basename}.png"
            )

            pdf_path = (
                FIELD_OUTPUT_DIRECTORY
                / f"{basename}.pdf"
            )

            fig.savefig(
                png_path,
                dpi=ELECTRIC_FIELD_DPI,
                facecolor="white",
                transparent=False,
                bbox_inches="tight",
            )

            fig.savefig(
                pdf_path,
                facecolor="white",
                transparent=False,
                bbox_inches="tight",
            )

            print(
                "Saved:",
                png_path,
            )

        plt.show()

        plt.close(
            fig
        )

# Part 4: Clean Poynting streamline cell for XZ only
#
# For total flow, use a mixture of distributed inside and outside seeds. For delta flow, use distributed seeds across the entire map because the direction can locally reverse.
#
# Cell C: Poynting XZ matrices

# In[294]:


def make_total_flow_seeds(
    field_map,
    *,
    n_seeds=45,
    margin_grid_cells=2.0,
    start_fraction=0.03,
    end_fraction=0.97,
):
    """
    Place seeds along the incident-facing -z edge.

    Valid for xz and yz planes because the second plotting
    coordinate is z in both cases.
    """

    if field_map.plane not in {
        "xz",
        "yz",
    }:
        raise ValueError(
            "Incident-edge total-flow seeds are intended for "
            "xz or yz planes."
        )

    grid_spacing_nm = abs(
        float(
            field_map.coord2_m[1]
            - field_map.coord2_m[0]
        )
        * 1e9
    )

    return make_edge_seeds(
        field_map,
        edge="min_coord2",
        n_seeds=n_seeds,
        margin_nm=(
            margin_grid_cells
            * grid_spacing_nm
        ),
        start_fraction=start_fraction,
        end_fraction=end_fraction,
    )

# In[295]:


def make_flow_seeds(
    field_map,
    *,
    flow_kind,
    random_seed=12345,
):
    """
    Choose seeds according to the physical meaning of the flow.
    """

    if flow_kind == "total":

        return make_total_flow_seeds(
            field_map,
            n_seeds=50,
            margin_grid_cells=2.0,
            start_fraction=0.03,
            end_fraction=0.97,
        )

    if flow_kind in {
        "delta",
        "scattered",
    }:

        return make_distributed_seeds(
            field_map,
            n_seeds=100,
            random_seed=random_seed,
            radial_domain="all",
            avoid_boundary_fraction=0.03,
        )

    raise ValueError(
        f"Unsupported flow kind: {flow_kind!r}"
    )

# In[ ]:


# ============================================================
# POYNTING-FLOW STREAMLINE MATRICES
#
# XZ plane only.
#
# Total flow:
#   distributed seeds inside and outside the sphere
#
# Delta flow:
#   distributed seeds over the complete calculation domain
#
# The seed placement is deterministic and reproducible.
# ============================================================

POYNTING_FLOW_KINDS = [
    "total",
    "delta",
]

POYNTING_FIGURE_WIDTH_CM = 17.5
POYNTING_FIGURE_HEIGHT_CM = 6.8

POYNTING_DPI = 400

SAVE_POYNTING_FIGURES = True


POYNTING_SETTINGS = {
    "total": {
        "background_kind": "total",
        "background_normalization": "relative_db",
        "background_reference_percentile": 99.5,
        "background_dynamic_range_db": 40.0,
        "background_cmap": "inferno",
        # "background_floor_fraction": 1e-8,
        # "background_clip_percentile_low": 0.0,
        "background_clip_percentile_high": 75.0,
        "normalization_region": "outside",
        "reference_percentile": 99.5,
        "display_dynamic_range_db": 25.0,
        "mask_unreliable_inside": True,
        "streamline_color": "#00E5FF",
        "streamline_linewidth": 1.20,
        "streamline_density": 1.0,
        "streamline_arrowsize": 1.0,
        "integration_direction": "forward",
        "minlength": 0.020,
        "maxlength": 15.0,
        "broken_streamlines": True,
        "normalize_vectors": True,
        "vector_dynamic_range_decades": 4.0,
        "minimum_speed_percentile": 0.10,
        "mask_inside_sphere": False,
        "seed_individually": True,
        "show_boundaries": True,
        "show_outer_boundary": True,
    },
    "delta": {
        "background_kind": "delta",
        "background_cmap": "magma",
        "background_floor_fraction": 1e-8,
        "background_clip_percentile_low": 0.0,
        "background_clip_percentile_high": 100.0,
        "streamline_color": "#00E5FF",
        "streamline_linewidth": 1.2,
        "streamline_density": 1.0,
        "streamline_arrowsize": 1.0,
        "integration_direction": "both",
        "minlength": 0.020,
        "maxlength": 15.0,
        "broken_streamlines": True,
        "normalize_vectors": True,
        "vector_dynamic_range_decades": 4.0,
        "minimum_speed_percentile": 0.10,
        "mask_inside_sphere": False,
        "seed_individually": True,
        "show_boundaries": False,
        "show_outer_boundary": True,
    },
}


cm_to_inches = 1.0 / 2.54


for wavelength_nm in (
    SELECTED_FIELD_WAVELENGTHS_NM
):

    for flow_kind in (
        POYNTING_FLOW_KINDS
    ):

        fig, axes = plt.subplots(
            nrows=1,
            ncols=len(
                SELECTED_FIELD_CASES
            ),
            figsize=(
                POYNTING_FIGURE_WIDTH_CM
                * cm_to_inches,
                POYNTING_FIGURE_HEIGHT_CM
                * cm_to_inches,
            ),
            squeeze=False,
            constrained_layout=True,
        )

        diagnostic_records = []

        for column_index, case_key in enumerate(
            SELECTED_FIELD_CASES
        ):

            ax = axes[
                0,
                column_index,
            ]

            map_key = (
                case_key,
                float(
                    wavelength_nm
                ),
                "xz",
            )

            if map_key not in field_maps:

                ax.set_axis_off()

                ax.text(
                    0.5,
                    0.5,
                    "XZ map not available",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                )

                continue

            field_map = field_maps[
                map_key
            ]

            if flow_kind == "total":

                total_seed_points_nm = make_total_flow_seeds(
                    field_map,
                    n_seeds=50,
                    margin_grid_cells=2.0,
                )

                # inside_seeds = (
                #     make_distributed_seeds(
                #         field_map,
                #         n_seeds=70,
                #         random_seed=(
                #             1000
                #             + column_index
                #         ),
                #         radial_domain="inside",
                #         minimum_radius_fraction=0.0,
                #         maximum_radius_fraction=0.98,
                #     )
                # )

                # outside_seeds = (
                #     make_distributed_seeds(
                #         field_map,
                #         n_seeds=70,
                #         random_seed=(
                #             2000
                #             + column_index
                #         ),
                #         radial_domain="outside",
                #         minimum_radius_fraction=1.02,
                #     )
                # )

                start_points_nm = np.vstack(
                    [
                        total_seed_points_nm,
                        # inside_seeds,
                        # outside_seeds,
                    ]
                )

            else:

                start_points_nm = (
                    make_distributed_seeds(
                        field_map,
                        n_seeds=170,
                        random_seed=(
                            3000
                            + column_index
                        ),
                        radial_domain="all",
                    )
                )

            (
                ax,
                streamline_set,
                diagnostics,
            ) = (
                plot_poynting_streamlines_safe(
                    field_map,
                    flow_kind=flow_kind,
                    start_points_nm=(
                        start_points_nm
                    ),
                    ax=ax,
                    **POYNTING_SETTINGS[
                        flow_kind
                    ],
                )
            )

            diagnostics.update(
                {
                    "case": case_key,
                    "wavelength_nm": (
                        wavelength_nm
                    ),
                    "flow_kind": (
                        flow_kind
                    ),
                }
            )

            diagnostic_records.append(
                diagnostics
            )

            ax.set_title(
                FIELD_CASE_LABELS.get(
                    case_key,
                    case_key,
                ),
                fontsize=8,
            )

            if column_index > 0:

                ax.set_ylabel("")

                ax.tick_params(
                    axis="y",
                    labelleft=False,
                )

            ax.xaxis.set_major_locator(
                MaxNLocator(
                    nbins=3
                )
            )

            ax.yaxis.set_major_locator(
                MaxNLocator(
                    nbins=3
                )
            )

        flow_label = {
            "total": (
                "total Poynting flow"
            ),
            "delta": (
                "incident-subtracted Poynting flow"
            ),
        }[
            flow_kind
        ]

        fig.suptitle(
            (
                rf"$\lambda="
                f"{wavelength_nm:.0f}"
                r"\,\mathrm{nm}$, "
                + flow_label
                + ", XZ plane"
            ),
            fontsize=10,
        )

        if diagnostic_records:

            print(
                f"\n{wavelength_nm:.0f} nm, "
                f"{flow_kind}:"
            )

            display(
                pd.DataFrame(
                    diagnostic_records
                ).style.format(
                    {
                        "valid_vector_fraction": (
                            "{:.1%}"
                        ),
                        "speed_median": (
                            "{:.4g}"
                        ),
                        "speed_p99": (
                            "{:.4g}"
                        ),
                    }
                )
            )

        if SAVE_POYNTING_FIGURES:

            basename = (
                f"poynting_"
                f"{flow_kind}_"
                f"{wavelength_nm:.0f}nm_"
                "xz"
            )

            png_path = (
                FIELD_OUTPUT_DIRECTORY
                / f"{basename}.png"
            )

            pdf_path = (
                FIELD_OUTPUT_DIRECTORY
                / f"{basename}.pdf"
            )

            fig.savefig(
                png_path,
                dpi=POYNTING_DPI,
                facecolor="white",
                transparent=False,
                bbox_inches="tight",
            )

            fig.savefig(
                pdf_path,
                facecolor="white",
                transparent=False,
                bbox_inches="tight",
            )

            print(
                "Saved:",
                png_path,
            )

        plt.show()

        plt.close(
            fig
        )

# In[303]:


# ============================================================
# TOTAL POYNTING FLOW: EXTERIOR-NORMALIZED BACKGROUND
#
# Incident wave propagates along +z.
# Seeds are placed at the lower z boundary.
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

from bragg_onion.fields import (
    make_edge_seeds,
)


TOTAL_TEST_CASE = "reference"
TOTAL_TEST_WAVELENGTH_NM = 502.0
TOTAL_TEST_PLANE = "xz"


available_total_test_keys = [
    key
    for key in field_maps
    if (
        isinstance(key, tuple)
        and len(key) == 3
        and key[0] == TOTAL_TEST_CASE
        and key[2] == TOTAL_TEST_PLANE
    )
]

if not available_total_test_keys:

    raise KeyError(
        "No suitable total-flow field map was found."
    )


total_test_key = min(
    available_total_test_keys,
    key=lambda key: abs(
        float(key[1])
        - TOTAL_TEST_WAVELENGTH_NM
    ),
)

field_map = field_maps[
    total_test_key
]


grid_spacing_nm = abs(
    float(
        field_map.coord2_m[1]
        - field_map.coord2_m[0]
    )
    * 1e9
)


incident_seed_points_nm = make_edge_seeds(
    field_map,
    edge="min_coord2",
    n_seeds=35,
    margin_nm=(
        3.0
        * grid_spacing_nm
    ),
    start_fraction=0.04,
    end_fraction=0.96,
)


# ------------------------------------------------------------
# Obtain robust flow components
# ------------------------------------------------------------

U = np.asarray(
    field_map.plane_u_total,
    dtype=float,
).copy()

V = np.asarray(
    field_map.plane_v_total,
    dtype=float,
).copy()

speed = np.hypot(
    U,
    V,
)

valid_vector_mask = (
    np.isfinite(U)
    & np.isfinite(V)
    & np.isfinite(speed)
    & (speed > 0.0)
)


# Normalize directions for streamline topology.
U_plot = np.zeros_like(U)
V_plot = np.zeros_like(V)

U_plot[
    valid_vector_mask
] = (
    U[
        valid_vector_mask
    ]
    / speed[
        valid_vector_mask
    ]
)

V_plot[
    valid_vector_mask
] = (
    V[
        valid_vector_mask
    ]
    / speed[
        valid_vector_mask
    ]
)


U_plot = np.ma.array(
    U_plot,
    mask=(
        ~valid_vector_mask
    ),
)

V_plot = np.ma.array(
    V_plot,
    mask=(
        ~valid_vector_mask
    ),
)


# ------------------------------------------------------------
# Exterior-normalized relative-dB background
# ------------------------------------------------------------

(
    background_db,
    background_vmin,
    background_vmax,
    background_diagnostics,
) = _prepare_region_normalized_db_background(
    field_map,
    field_map.s_total_mag,
    normalization_region="outside",
    reference_percentile=99.5,
    display_dynamic_range_db=30.0,
    floor_dynamic_range_db=80.0,
    mask_unreliable_inside=True,
)


extent_nm = [
    float(
        field_map.coord1_m[0]
        * 1e9
    ),
    float(
        field_map.coord1_m[-1]
        * 1e9
    ),
    float(
        field_map.coord2_m[0]
        * 1e9
    ),
    float(
        field_map.coord2_m[-1]
        * 1e9
    ),
]


fig, ax = plt.subplots(
    figsize=(7.2, 6.2),
    constrained_layout=True,
)


image = ax.imshow(
    background_db,
    origin="lower",
    extent=extent_nm,
    cmap="inferno",
    vmin=background_vmin,
    vmax=background_vmax,
    aspect="equal",
)


colorbar = fig.colorbar(
    image,
    ax=ax,
    fraction=0.046,
    pad=0.025,
)

colorbar.set_label(
    r"Exterior-normalized $|\mathbf{S}|$ [dB]"
)


ax.streamplot(
    field_map.coord1_m
    * 1e9,
    field_map.coord2_m
    * 1e9,
    U_plot,
    V_plot,
    start_points=(
        incident_seed_points_nm
    ),
    color="#00E5FF",
    linewidth=0.70,
    arrowsize=0.90,
    arrowstyle="->",
    integration_direction="forward",
    minlength=0.03,
    maxlength=6.0,
    broken_streamlines=True,
    density=1.0,
    zorder=5,
)


outer_boundary = plt.Circle(
    (0.0, 0.0),
    radius=(
        field_map.outer_radius_m
        * 1e9
    ),
    fill=False,
    edgecolor="white",
    linewidth=1.0,
    zorder=6,
)

ax.add_patch(
    outer_boundary
)


ax.set_xlabel(
    "x [nm]"
)

ax.set_ylabel(
    "z [nm]"
)

ax.set_title(
    (
        f"{TOTAL_TEST_CASE}, "
        f"{total_test_key:.0f} nm\n"
        "total Poynting flow"
    )
)

ax.set_aspect(
    "equal",
    adjustable="box",
)


print(
    background_diagnostics
)

plt.show()

# In[298]:


# ============================================================
# DIAGNOSE POYNTING BACKGROUND DISTRIBUTION
# ============================================================

TEST_BACKGROUND_KIND = "total"

if TEST_BACKGROUND_KIND == "total":

    magnitude = np.asarray(
        test_field_map.s_total_mag,
        dtype=float,
    )

elif TEST_BACKGROUND_KIND == "delta":

    magnitude = np.asarray(
        test_field_map.s_delta_mag,
        dtype=float,
    )

else:

    magnitude = np.asarray(
        test_field_map.s_sca_like_mag,
        dtype=float,
    )


inside_mask = (
    test_field_map.radial_distance_m
    <= test_field_map.outer_radius_m
)

outside_mask = ~inside_mask

diagnostic_regions = {
    "all": np.ones_like(
        magnitude,
        dtype=bool,
    ),
    "inside sphere": inside_mask,
    "outside sphere": outside_mask,
}


records = []

for region_name, region_mask in (
    diagnostic_regions.items()
):

    valid = (
        region_mask
        & np.isfinite(magnitude)
        & (magnitude > 0.0)
    )

    values = magnitude[
        valid
    ]

    records.append(
        {
            "region": region_name,
            "number_of_pixels": int(
                np.count_nonzero(
                    region_mask
                )
            ),
            "finite_positive_pixels": int(
                values.size
            ),
            "finite_positive_fraction": float(
                values.size
                / np.count_nonzero(
                    region_mask
                )
            ),
            "minimum": (
                np.nanmin(values)
                if values.size
                else np.nan
            ),
            "median": (
                np.nanmedian(values)
                if values.size
                else np.nan
            ),
            "p95": (
                np.nanpercentile(
                    values,
                    95.0,
                )
                if values.size
                else np.nan
            ),
            "p99": (
                np.nanpercentile(
                    values,
                    99.0,
                )
                if values.size
                else np.nan
            ),
            "p99_5": (
                np.nanpercentile(
                    values,
                    99.5,
                )
                if values.size
                else np.nan
            ),
            "maximum": (
                np.nanmax(values)
                if values.size
                else np.nan
            ),
        }
    )


df_poynting_distribution_regions = (
    pd.DataFrame(
        records
    )
)

display(
    df_poynting_distribution_regions.style.format(
        {
            "finite_positive_fraction": "{:.2%}",
            "minimum": "{:.4g}",
            "median": "{:.4g}",
            "p95": "{:.4g}",
            "p99": "{:.4g}",
            "p99_5": "{:.4g}",
            "maximum": "{:.4g}",
        }
    )
)

# In[272]:


# ============================================================
# DIAGNOSTIC: DISTRIBUTION OF POYNTING BACKGROUND IN dB
#
# Shows the actual quantity used for colour mapping.
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def get_background_db_for_diagnostic(
    field_map,
    *,
    background_kind="total",
    background_floor_fraction=1e-8,
    background_reference_percentile=99.0,
):
    """
    Reproduce the background_db transformation used in
    plot_poynting_streamlines_safe().
    """

    if background_kind == "total":

        background = np.asarray(
            field_map.s_total_mag,
            dtype=float,
        )

    elif background_kind == "scattered":

        background = np.asarray(
            field_map.s_sca_like_mag,
            dtype=float,
        )

    elif background_kind == "delta":

        background = np.asarray(
            field_map.s_delta_mag,
            dtype=float,
        )

    else:

        raise ValueError(
            "background_kind must be "
            "'total', 'scattered', or 'delta'."
        )

    background_valid = (
        np.isfinite(background)
        & (background >= 0.0)
    )

    background_reference = float(
        np.nanpercentile(
            background[background_valid],
            background_reference_percentile,
        )
    )

    background_floor = max(
        background_reference
        * background_floor_fraction,
        np.finfo(float).tiny,
    )

    background_db = np.full_like(
        background,
        np.nan,
        dtype=float,
    )

    background_db[background_valid] = (
        10.0
        * np.log10(
            np.maximum(
                background[background_valid],
                background_floor,
            )
            / background_reference
        )
    )

    return background_db


# ------------------------------------------------------------
# Plot histograms
# ------------------------------------------------------------

for wavelength_nm in SELECTED_FIELD_WAVELENGTHS_NM:

    for flow_kind in POYNTING_FLOW_KINDS:

        settings = POYNTING_SETTINGS[flow_kind]

        background_kind = settings[
            "background_kind"
        ]

        fig, axes = plt.subplots(
            nrows=1,
            ncols=len(SELECTED_FIELD_CASES),
            figsize=(
                POYNTING_FIGURE_WIDTH_CM * cm_to_inches,
                4.5,
            ),
            squeeze=False,
            constrained_layout=True,
        )

        for column_index, case_key in enumerate(
            SELECTED_FIELD_CASES
        ):

            ax = axes[
                0,
                column_index,
            ]

            map_key = (
                case_key,
                float(wavelength_nm),
                "xz",
            )

            if map_key not in field_maps:

                ax.set_axis_off()

                continue

            field_map = field_maps[
                map_key
            ]

            background_db = (
                get_background_db_for_diagnostic(
                    field_map,
                    background_kind=background_kind,
                    background_floor_fraction=(
                        settings[
                            "background_floor_fraction"
                        ]
                    ),
                )
            )

            values = background_db[
                np.isfinite(background_db)
            ]

            # Percentiles in the actual quantity being plotted.
            percentiles = np.nanpercentile(
                values,
                [
                    1,
                    5,
                    25,
                    50,
                    75,
                    90,
                    95,
                    99,
                    99.5,
                    99.9,
                ],
            )

            # Histogram.
            ax.hist(
                values,
                bins=500,
                density=True,
                alpha=0.85,
            )

            ax.set_yscale("log")

            # Mark useful percentiles.
            for p, value in zip(
                [
                    1,
                    5,
                    25,
                    50,
                    75,
                    90,
                    95,
                    99,
                    99.5,
                    99.9,
                ],
                percentiles,
            ):

                ax.axvline(
                    value,
                    linestyle="--",
                    linewidth=0.7,
                    alpha=0.55,
                )

                # Only label the more useful upper percentiles
                # to avoid making the plot unreadable.
                if p in [90, 95, 99, 99.9]:

                    ax.text(
                        value,
                        0.96,
                        f"P{p:g}",
                        transform=(
                            ax.get_xaxis_transform()
                        ),
                        rotation=90,
                        va="top",
                        ha="right",
                        fontsize=7,
                    )

            # Current clipping limits from the plotting function.
            clip_low = np.nanpercentile(
                values,
                settings[
                    "background_clip_percentile_low"
                ],
            )

            clip_high = np.nanpercentile(
                values,
                settings[
                    "background_clip_percentile_high"
                ],
            )

            ax.axvline(
                clip_low,
                linewidth=1.4,
                alpha=0.9,
            )

            ax.axvline(
                clip_high,
                linewidth=1.4,
                alpha=0.9,
            )

            ax.set_title(
                FIELD_CASE_LABELS.get(
                    case_key,
                    case_key,
                ),
                fontsize=8,
            )

            ax.set_xlabel(
                "background intensity [dB]"
            )

            if column_index == 0:

                ax.set_ylabel(
                    "probability density"
                )

            else:

                ax.tick_params(
                    axis="y",
                    labelleft=False,
                )

            ax.xaxis.set_major_locator(
                MaxNLocator(nbins=5)
            )

        fig.suptitle(
            (
                rf"$\lambda={wavelength_nm:.0f}\,"
                r"\mathrm{nm}$, "
                f"{flow_kind} — "
                f"{background_kind} background"
            ),
            fontsize=10,
        )

        plt.show()

        plt.close(fig)

# In[274]:


fig, ax = plt.subplots(
    figsize=(8, 5),
)

sorted_values = np.sort(
    values[np.isfinite(values)]
)

cdf = (
    np.arange(1, len(sorted_values) + 1)
    / len(sorted_values)
)

upper_mask = cdf >= 0.90

ax.plot(
    sorted_values[upper_mask],
    100.0 * (1.0 - cdf[upper_mask]),
    linewidth=1.5,
)

ax.set_yscale("log")

ax.set_xlabel(
    "background intensity [dB]"
)

ax.set_ylabel(
    "fraction of pixels above value [%]"
)

ax.grid(
    which="both",
    alpha=0.2,
)

plt.show()

# In[283]:


percentile_levels = [
    0,
    0.001,
    25,
    50,
    98.2,
    98.3,
    98.4,
    98.6,
    99,
    99.5,
    99.9,
    99.95,
    99.99,
    99.999,
    100,
]

percentile_values = np.nanpercentile(
    values,
    percentile_levels,
)

percentile_table = pd.DataFrame(
    {
        "percentile": percentile_levels,
        "dB": percentile_values,
    }
)

display(
    percentile_table.style.format(
        {
            "percentile": "{:.3f}",
            "dB": "{:.8}",
        }
    )
)

# In[ ]:


# ============================================================
# CLEAN POYNTING-FLOW PLOTS
#
# Plane:
#   xz only
#
# Total flow:
#   seeds along incident-facing -z boundary
#
# Delta flow:
#   reproducible distributed seeds
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.ticker import MaxNLocator
from IPython.display import display

from bragg_onion.fields import (
    make_edge_seeds,
    make_distributed_seeds,
    plot_poynting_streamlines_safe,
)


# ------------------------------------------------------------
# User settings
# ------------------------------------------------------------

POYNTING_CASE_KEYS = [
    "reference",
    "constant_diameter",
    "constant_shell_number",
]

POYNTING_FLOW_KINDS = [
    "total",
    "delta",
]

POYNTING_PLANE = "xz"

# None means all wavelengths available for the xz maps.
POYNTING_WAVELENGTHS_NM = None

POYNTING_FIGURE_WIDTH_CM = 17.5
POYNTING_FIGURE_HEIGHT_CM = 6.8

POYNTING_PNG_DPI = 400

SAVE_CLEAN_POYNTING_FIGURES = True


POYNTING_OUTPUT_DIRECTORY = (
    FIELD_FIGURE_DIRECTORY
    / "clean_poynting"
)

POYNTING_OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)


# ------------------------------------------------------------
# Appearance and numerical settings
# ------------------------------------------------------------

CLEAN_POYNTING_SETTINGS = {
    "total": {
        "background_kind": "total",
        "background_cmap": "plasma",
        "background_floor_fraction": 1e-8,
        "background_clip_percentile_low": 0.0,
        "background_clip_percentile_high": 75.0,

        "streamline_color": "#00E5FF",
        "streamline_linewidth": 0.65,
        "streamline_density": 1.0,
        "streamline_arrowsize": 0.9,
        "integration_direction": "forward",
        "minlength": 0.02,
        "maxlength": 15.0,
        "broken_streamlines": True,

        "normalize_vectors": True,
        "vector_dynamic_range_decades": 4.0,
        "minimum_speed_percentile": 0.0,

        "mask_inside_sphere": False,
        "seed_individually": True,

        "show_boundaries": False,
        "show_outer_boundary": True,
    },

    "delta": {
        "background_kind": "delta",
        "background_cmap": "magma",
        "background_floor_fraction": 1e-8,
        "background_clip_percentile_low": 0.0,
        "background_clip_percentile_high": 75.0,

        "streamline_color": "#00E5FF",
        "streamline_linewidth": 0.65,
        "streamline_density": 1.0,
        "streamline_arrowsize": 0.9,

        "integration_direction": "both",
        "minlength": 0.02,
        "maxlength": 12.0,
        "broken_streamlines": True,

        "normalize_vectors": True,
        "vector_dynamic_range_decades": 4.0,
        "minimum_speed_percentile": 0.1,

        "mask_inside_sphere": False,
        "seed_individually": True,

        "show_boundaries": False,
        "show_outer_boundary": True,
    },
}


# ------------------------------------------------------------
# Available xz maps
# ------------------------------------------------------------

available_poynting_keys = [
    key
    for key in field_maps
    if (
        isinstance(key, tuple)
        and len(key) == 3
        and key[2] == POYNTING_PLANE
        and key[0] in POYNTING_CASE_KEYS
    )
]

available_poynting_wavelengths_nm = sorted(
    {
        float(key[1])
        for key in available_poynting_keys
    }
)

if POYNTING_WAVELENGTHS_NM is None:

    selected_poynting_wavelengths_nm = (
        available_poynting_wavelengths_nm
    )

else:

    selected_poynting_wavelengths_nm = [
        min(
            available_poynting_wavelengths_nm,
            key=lambda available: abs(
                available
                - float(requested)
            ),
        )
        for requested in POYNTING_WAVELENGTHS_NM
    ]

    selected_poynting_wavelengths_nm = sorted(
        set(
            selected_poynting_wavelengths_nm
        )
    )


print(
    "Poynting wavelengths:",
    selected_poynting_wavelengths_nm,
)


# ------------------------------------------------------------
# Seed helper
# ------------------------------------------------------------

def make_clean_poynting_seeds(
    field_map,
    *,
    flow_kind,
    random_seed,
):
    """
    Generate physically appropriate seeds for each flow type.
    """

    if flow_kind == "total":

        grid_spacing_nm = abs(
            float(
                field_map.coord2_m[1]
                - field_map.coord2_m[0]
            )
            * 1e9
        )

        return make_edge_seeds(
            field_map,
            edge="min_coord2",
            n_seeds=50,
            margin_nm=(
                2.0
                * grid_spacing_nm
            ),
            start_fraction=0.03,
            end_fraction=0.97,
        )

    return make_distributed_seeds(
        field_map,
        n_seeds=100,
        random_seed=random_seed,
        radial_domain="all",
        avoid_boundary_fraction=0.03,
    )


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

cm_to_inches = 1.0 / 2.54

for wavelength_nm in (
    selected_poynting_wavelengths_nm
):

    for flow_kind in (
        POYNTING_FLOW_KINDS
    ):

        fig, axes = plt.subplots(
            1,
            len(
                POYNTING_CASE_KEYS
            ),
            figsize=(
                POYNTING_FIGURE_WIDTH_CM
                * cm_to_inches,
                POYNTING_FIGURE_HEIGHT_CM
                * cm_to_inches,
            ),
            squeeze=False,
            constrained_layout=True,
        )

        diagnostics_records = []

        for column_index, case_key in enumerate(
            POYNTING_CASE_KEYS
        ):

            ax = axes[
                0,
                column_index,
            ]

            map_key = (
                case_key,
                float(
                    wavelength_nm
                ),
                POYNTING_PLANE,
            )

            if map_key not in field_maps:

                ax.set_axis_off()

                ax.text(
                    0.5,
                    0.5,
                    "Map not available",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                )

                continue

            field_map = field_maps[
                map_key
            ]

            seed_points_nm = (
                make_clean_poynting_seeds(
                    field_map,
                    flow_kind=flow_kind,
                    random_seed=(
                        10000
                        + column_index
                    ),
                )
            )

            (
                ax,
                streamline_set,
                diagnostics,
            ) = (
                plot_poynting_streamlines_safe(
                    field_map,
                    flow_kind=flow_kind,
                    start_points_nm=(
                        seed_points_nm
                    ),
                    ax=ax,
                    **CLEAN_POYNTING_SETTINGS[
                        flow_kind
                    ],
                )
            )

            diagnostics.update(
                {
                    "case": case_key,
                    "wavelength_nm": (
                        wavelength_nm
                    ),
                    "flow_kind": (
                        flow_kind
                    ),
                }
            )

            diagnostics_records.append(
                diagnostics
            )

            ax.set_title(
                FIELD_CASE_LABELS.get(
                    case_key,
                    case_key,
                ),
                fontsize=8,
            )

            if column_index > 0:

                ax.set_ylabel("")

                ax.tick_params(
                    axis="y",
                    labelleft=False,
                )

            ax.xaxis.set_major_locator(
                MaxNLocator(
                    nbins=3
                )
            )

            ax.yaxis.set_major_locator(
                MaxNLocator(
                    nbins=3
                )
            )

        flow_description = {
            "total": "total Poynting flow",
            "delta": (
                "incident-subtracted Poynting flow"
            ),
        }[
            flow_kind
        ]

        fig.suptitle(
            (
                rf"$\lambda="
                f"{wavelength_nm:.0f}"
                r"\,\mathrm{nm}$, "
                + flow_description
                + ", XZ plane"
            ),
            fontsize=10,
        )

        if diagnostics_records:

            display(
                pd.DataFrame(
                    diagnostics_records
                ).style.format(
                    {
                        "valid_vector_fraction": "{:.1%}",
                        "speed_median": "{:.4g}",
                        "speed_p99": "{:.4g}",
                    }
                )
            )

        if SAVE_CLEAN_POYNTING_FIGURES:

            basename = (
                f"poynting_clean_"
                f"{flow_kind}_"
                f"{wavelength_nm:.0f}nm_"
                "xz"
            )

            png_path = (
                POYNTING_OUTPUT_DIRECTORY
                / f"{basename}.png"
            )

            pdf_path = (
                POYNTING_OUTPUT_DIRECTORY
                / f"{basename}.pdf"
            )

            fig.savefig(
                png_path,
                dpi=POYNTING_PNG_DPI,
                facecolor="white",
                transparent=False,
                bbox_inches="tight",
                pad_inches=0.03,
            )

            fig.savefig(
                pdf_path,
                facecolor="white",
                transparent=False,
                bbox_inches="tight",
                pad_inches=0.03,
            )

            print(
                "Saved:",
                png_path,
            )

            print(
                "Saved:",
                pdf_path,
            )

        plt.show()

        plt.close(
            fig
        )

# Part 5: Recompute maps with more surrounding environment
#
# The environment extent is fixed during compute_field_map(). Plotting cannot reveal data outside the existing grid.
#
# Use a separate dictionary so you do not overwrite the expensive existing maps:

# In[224]:


# ============================================================
# OPTIONAL: RECALCULATE SELECTED XZ MAPS WITH A LARGER EXTENT
#
# Start with one wavelength and all three cases.
# ============================================================

ENVIRONMENT_EXTENT_FACTOR = 2.0
ENVIRONMENT_NPTS = 201

ENVIRONMENT_WAVELENGTHS_NM = [
    502.0,
]

environment_field_maps = {}


for case_key in SELECTED_FIELD_CASES:

    for requested_wavelength_nm in (
        ENVIRONMENT_WAVELENGTHS_NM
    ):

        wavelength_nm = min(
            SELECTED_FIELD_WAVELENGTHS_NM,
            key=lambda value: abs(
                float(value)
                - float(
                    requested_wavelength_nm
                )
            ),
        )

        map_key = (
            case_key,
            float(
                wavelength_nm
            ),
            "xz",
        )

        print(
            "Computing environmental map:",
            map_key,
            flush=True,
        )

        environment_field_maps[
            map_key
        ] = compute_field_map(
            stack=field_stacks[
                case_key
            ][
                "stack"
            ],
            wavelength_m=(
                float(
                    wavelength_nm
                )
                * 1e-9
            ),
            n_medium=(
                FIELD_N_MEDIUM
            ),
            plane="xz",
            npts=(
                ENVIRONMENT_NPTS
            ),
            extent_outer_radius_factor=(
                ENVIRONMENT_EXTENT_FACTOR
            ),
            incident_e0=(
                FIELD_INCIDENT_E0
            ),
        )

# In[ ]:
