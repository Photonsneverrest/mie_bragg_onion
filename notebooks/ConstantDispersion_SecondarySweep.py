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

# In[1]:


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


import matplotlib as mpl
mpl.rcdefaults()   # reset to Matplotlib defaults
plt.style.use('/mnt/c/Users/SchwarzN/OneDrive - Université de Fribourg/Institution/Other/Scripts_and_HowTo/science_paper.mplstyle')

print("Notebook helper modules imported successfully.")


# In[2]:


# CELL 3 — OUTPUT SOURCE / NOTEBOOK SETTINGS

# Windows path to the copied workstation output directory.
# This can stay in Windows form even when the notebook runs in WSL:
# load_sweep_outputs(...) will convert it automatically if needed.
SOURCE_OUTPUT_DIR_WINDOWS = (
    r"C:\Users\SchwarzN\OneDrive - Université de Fribourg"
    r"\Institution\P1_BraggSpericalPigments\Simulation\MIE_Computation"
    r"\Scattnlay_BraggOnion\outputs\run_secondary_sweep_02"
)

# Default colour view for object-colour-style plotting:
DEFAULT_COLOUR_VIEW = "raw"   # "raw", "display", or "both"

print("SOURCE_OUTPUT_DIR_WINDOWS:")
print(SOURCE_OUTPUT_DIR_WINDOWS)
print("DEFAULT_COLOUR_VIEW:", DEFAULT_COLOUR_VIEW)


# In[3]:


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


# In[4]:


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
            "design_peak_wavelength_nm": 850.0,
            "absorber_location": "none",
            "n_A": 1.3075,
            # "k": 0.001,
            "n_layers": 120,
        },
        "reason": "Suspicious hue or colour result: low-index absorber, 850 nm, nA=1.3075, k=0.001, 120 layers",
    },
    {
            "filters": {
                "design_peak_wavelength_nm": 380.0,
                "absorber_location": "none",
                "n_A": 1.605,
                # "k": 0.001,
                "n_layers": 120,
            },
            "reason": "Suspicious hue or colour result: low-index absorber, 380 nm, nA=1.605, k=0.001, 120 layers",
        },
    {
            "filters": {
                "design_peak_wavelength_nm": 850.0,
                "absorber_location": "none",
                "n_A": 1.248,
                # "k": 0.001,
                "n_layers": 91,
            },
            "reason": "Suspicious hue or colour result: low-index absorber, 850 nm, nA=1.248, k=0.001, 91 layers",
        },
    {
            "filters": {
                "design_peak_wavelength_nm": 760.0,
                "absorber_location": "none",
                "n_A": 1.010,
                # "k": 0.001,
                "n_layers": 91,
            },
            "reason": "Suspicious hue or colour result: low-index absorber, 760 nm, nA=1.010, k=0.001, 91 layers",
        },
    {
            "filters": {
                "design_peak_wavelength_nm": 680.0,
                "absorber_location": "none",
                "n_A": 1.3075,
                # "k": 0.001,
                "n_layers": 120,
            },
            "reason": "Suspicious hue or colour result: low-index absorber, 680 nm, nA=1.3075, k=0.001, 120 layers",
        },
    {
            "filters": {
                "design_peak_wavelength_nm": 380.0,
                "absorber_location": "low_index",
                "n_A": 1.605,
                "k": 0.00223607,
                "n_layers": 120,
            },
            "reason": "Suspicious hue or colour result: low-index absorber, 380 nm, nA=1.605, k=0.00223607, 120 layers",
        },
    {
            "filters": {
                "design_peak_wavelength_nm": 680.0,
                "absorber_location": "low_index",
                "n_A": 1.3075,
                # "k": 0.01320138,
                "n_layers": 120,
            },
            "reason": "Suspicious hue or colour result: low-index absorber, 680 nm, nA=1.3075, k=0.01320138, 120 layers",
        },
    {
            "filters": {
                "design_peak_wavelength_nm": 680.0,
                "absorber_location": "low_index",
                "n_A": 1.605,
                "k": 0.01320138,
                "n_layers": 104,
            },
            "reason": "Suspicious hue or colour result: low-index absorber, 680 nm, nA=1.605, k=0.01320138, 104 layers",
        },
    {
            "filters": {
                "design_peak_wavelength_nm": 680.0,
                "absorber_location": "low_index",
                "n_A": 1.605,
                "k": 0.00059038,
                "n_layers": 104,
            },
            "reason": "Suspicious hue or colour result: low-index absorber, 680 nm, nA=1.605, k=0.00059038, 104 layers",
        },
    {
            "filters": {
                "design_peak_wavelength_nm": 380.0,
                "absorber_location": "low_index",
                "n_A": 1.605,
                "k": 0.00543316,
                "n_layers": 104,
            },
            "reason": "Suspicious hue or colour result: low-index absorber, 380 nm, nA=1.605, k=0.00543316, 104 layers",
        },
    {
            "filters": {
                "design_peak_wavelength_nm": 380.0,
                "absorber_location": "low_index",
                "n_A": 1.605,
                "k": 0.01320138,
                "n_layers": 104,
            },
            "reason": "Suspicious hue or colour result: low-index absorber, 380 nm, nA=1.605, k=0.01320138, 104 layers",
        },
    {
            "filters": {
                "design_peak_wavelength_nm": 380.0,
                "absorber_location": "low_index",
                "n_A": 1.605,
                # "k": 0.01320138,
                "n_layers": 120,
            },
            "reason": "Suspicious hue or colour result: low-index absorber, 380 nm, nA=1.605, k=0.01320138, 120 layers",
        },
    {
            "filters": {
                "design_peak_wavelength_nm": 760.0,
                "absorber_location": "low_index",
                "n_A": 1.01,
                # "k": 0.01320138,
                "n_layers": 91,
            },
            "reason": "Suspicious hue or colour result: low-index absorber, 760 nm, nA=1.01, k=0.01320138, 91 layers",
        },
    {
        "filters": {
            "design_peak_wavelength_nm": 850.0,
            "absorber_location": "low_index",
            "n_A": 1.248,
            "k": 0.0001,
            "n_layers": 104,
        },
        "reason": "Suspicious hue or colour result: low-index absorber, 850 nm, nA=1.248, k=0.0001, 104 layers",
    },
    # {
    #     "filters": {
    #         "design_peak_wavelength_nm": 850.0,
    #         "absorber_location": "low_index",
    #         "n_A": 1.248,
    #         "k": 0.0001,
    #         "n_layers": 104,
    #     },
    #     "reason": "Suspicious hue or colour result: low-index absorber, 850 nm, nA=1.248, k=0.00015588, 104 layers",
    # },
    {
        "filters": {
            "design_peak_wavelength_nm": 850.0,
            "absorber_location": "low_index",
            "n_A": 1.248,
            "k": 0.00015588,
            "n_layers": 104,
        },
        "reason": "Suspicious hue or colour result: low-index absorber, 850 nm, nA=1.248, k=0.00015588, 104 layers",
    },
    {
        "filters": {
            "design_peak_wavelength_nm": 850.0,
            "absorber_location": "low_index",
            "n_A": 1.248,
            "k": 0.00024298,
            "n_layers": 104,
        },
        "reason": "Suspicious hue or colour result: low-index absorber, 850 nm, nA=1.248, k=0.00024298, 104 layers",
    },
    {
        "filters": {
            "design_peak_wavelength_nm": 850.0,
            "absorber_location": "low_index",
            "n_A": 1.248,
            "k": 0.00037875,
            "n_layers": 104,
        },
        "reason": "Suspicious hue or colour result: low-index absorber, 850 nm, nA=1.248, k=0.00037875, 104 layers",
    },
    {
        "filters": {
            "design_peak_wavelength_nm": 850.0,
            "absorber_location": "low_index",
            "n_A": 1.248,
            "k": 0.00059038,
            "n_layers": 104,
        },
        "reason": "Suspicious hue or colour result: low-index absorber, 850 nm, nA=1.248, k=0.00059038, 104 layers",
    },
    {
        "filters": {
            "design_peak_wavelength_nm": 850.0,
            "absorber_location": "low_index",
            "n_A": 1.248,
            "k": 0.00092028,
            "n_layers": 104,
        },
        "reason": "Suspicious hue or colour result: low-index absorber, 850 nm, nA=1.248, k=0.00092028, 104 layers",
    },
    {
        "filters": {
            "design_peak_wavelength_nm": 850.0,
            "absorber_location": "low_index",
            "n_A": 1.248,
            "k": 0.0014345,
            "n_layers": 104,
        },
        "reason": "Suspicious hue or colour result: low-index absorber, 850 nm, nA=1.248, k=0.0014345, 104 layers",
    },
    {
        "filters": {
            "design_peak_wavelength_nm": 850.0,
            "absorber_location": "low_index",
            "n_A": 1.248,
            "k": 0.00223607,
            "n_layers": 104,
        },
        "reason": "Suspicious hue or colour result: low-index absorber, 850 nm, nA=1.248, k=0.00223607, 104 layers",
    },
    {
        "filters": {
            "design_peak_wavelength_nm": 850.0,
            "absorber_location": "low_index",
            "n_A": 1.248,
            "k": 0.01320138,
            "n_layers": 104,
        },
        "reason": "Suspicious hue or colour result: low-index absorber, 850 nm, nA=1.248, k=0.01320138, 104 layers",
    },
    {
        "filters": {
            "design_peak_wavelength_nm": 850.0,
            "absorber_location": "low_index",
            "n_A": 1.248,
            "k": 0.0001,
            "n_layers": 79,
        },
        "reason": "Suspicious hue or colour result: low-index absorber, 850 nm, nA=1.248, k=0.0001, 79 layers",
    },
    {
        "filters": {
            "design_peak_wavelength_nm": 850.0,
            "absorber_location": "low_index",
            "n_A": 1.248,
            "k": 0.00015588,
            "n_layers": 79,
        },
        "reason": "Suspicious hue or colour result: low-index absorber, 850 nm, nA=1.248, k=0.00015588, 79 layers",
    },
    {
        "filters": {
            "design_peak_wavelength_nm": 850.0,
            "absorber_location": "low_index",
            "n_A": 1.248,
            "k": 0.00024298,
            "n_layers": 79,
        },
        "reason": "Suspicious hue or colour result: low-index absorber, 850 nm, nA=1.248, k=0.00024298, 79 layers",
    },
    {
        "filters": {
            "design_peak_wavelength_nm": 850.0,
            "absorber_location": "low_index",
            "n_A": 1.248,
            "k": 0.00037875,
            "n_layers": 79,
        },
        "reason": "Suspicious hue or colour result: low-index absorber, 850 nm, nA=1.248, k=0.00037875, 79 layers",
    },
    {
        "filters": {
            "design_peak_wavelength_nm": 850.0,
            "absorber_location": "low_index",
            "n_A": 1.248,
            "k": 0.00059038,
            "n_layers": 79,
        },
        "reason": "Suspicious hue or colour result: low-index absorber, 850 nm, nA=1.248, k=0.00059038, 79 layers",
    },
    {
        "filters": {
            "design_peak_wavelength_nm": 850.0,
            "absorber_location": "low_index",
            "n_A": 1.248,
            "k": 0.00092028,
            "n_layers": 79,
        },
        "reason": "Suspicious hue or colour result: low-index absorber, 850 nm, nA=1.248, k=0.00092028, 79 layers",
    },
    {
        "filters": {
            "design_peak_wavelength_nm": 850.0,
            "absorber_location": "low_index",
            "n_A": 1.248,
            "k": 0.0014345,
            "n_layers": 79,
        },
        "reason": "Suspicious hue or colour result: low-index absorber, 850 nm, nA=1.248, k=0.0014345, 79 layers",
    },
    {
        "filters": {
            "design_peak_wavelength_nm": 850.0,
            "absorber_location": "low_index",
            "n_A": 1.248,
            "k": 0.00223607,
            "n_layers": 79,
        },
        "reason": "Suspicious hue or colour result: low-index absorber, 850 nm, nA=1.248, k=0.00223607, 79 layers",
    },
    {
        "filters": {
            "design_peak_wavelength_nm": 850.0,
            "absorber_location": "low_index",
            "n_A": 1.248,
            "k": 0.00348553,
            "n_layers": 79,
        },
        "reason": "Suspicious hue or colour result: low-index absorber, 850 nm, nA=1.248, k=0.00348553, 79 layers",
    },
    {
        "filters": {
            "design_peak_wavelength_nm": 850.0,
            "absorber_location": "low_index",
            "n_A": 1.3075,
            "k": 0.0001,
            "n_layers": 104,
        },
        "reason": "Suspicious hue or colour result: low-index absorber, 850 nm, nA=1.3075, k=0.0014345, 104 layers",
    },
    {
        "filters": {
            "design_peak_wavelength_nm": 850.0,
            "absorber_location": "low_index",
            "n_A": 1.3075,
            "k": 0.00015588,
            "n_layers": 104,
        },
        "reason": "Suspicious hue or colour result: low-index absorber, 850 nm, nA=1.3075, k=0.0014345, 104 layers",
    },
    {
        "filters": {
            "design_peak_wavelength_nm": 850.0,
            "absorber_location": "low_index",
            "n_A": 1.3075,
            "k": 0.00024298,
            "n_layers": 104,
        },
        "reason": "Suspicious hue or colour result: low-index absorber, 850 nm, nA=1.3075, k=0.0014345, 104 layers",
    },
    {
        "filters": {
            "design_peak_wavelength_nm": 850.0,
            "absorber_location": "low_index",
            "n_A": 1.3075,
            "k": 0.00037875,
            "n_layers": 104,
        },
        "reason": "Suspicious hue or colour result: low-index absorber, 850 nm, nA=1.3075, k=0.0014345, 104 layers",
    },
    {
        "filters": {
            "design_peak_wavelength_nm": 850.0,
            "absorber_location": "low_index",
            "n_A": 1.3075,
            "k": 0.00059038,
            "n_layers": 104,
        },
        "reason": "Suspicious hue or colour result: low-index absorber, 850 nm, nA=1.3075, k=0.0014345, 104 layers",
    },
    {
        "filters": {
            "design_peak_wavelength_nm": 850.0,
            "absorber_location": "low_index",
            "n_A": 1.3075,
            "k": 0.00092028,
            "n_layers": 104,
        },
        "reason": "Suspicious hue or colour result: low-index absorber, 850 nm, nA=1.3075, k=0.0014345, 104 layers",
    },
    {
        "filters": {
            "design_peak_wavelength_nm": 850.0,
            "absorber_location": "low_index",
            "n_A": 1.3075,
            "k": 0.0014345,
            "n_layers": 104,
        },
        "reason": "Suspicious hue or colour result: low-index absorber, 850 nm, nA=1.3075, k=0.0014345, 104 layers",
    },
    {
        "filters": {
            "design_peak_wavelength_nm": 850.0,
            "absorber_location": "low_index",
            "n_A": 1.3075,
            "k": 0.00223607,
            "n_layers": 104,
        },
        "reason": "Suspicious hue or colour result: low-index absorber, 850 nm, nA=1.3075, k=0.00223607, 104 layers",
    },
    {
        "filters": {
            "design_peak_wavelength_nm": 850.0,
            "absorber_location": "low_index",
            "n_A": 1.3075,
            "k": 0.00348553,
            "n_layers": 104,
        },
        "reason": "Suspicious hue or colour result: low-index absorber, 850 nm, nA=1.3075, k=0.00348553, 104 layers",
    },
    {
        "filters": {
            "design_peak_wavelength_nm": 850.0,
            "absorber_location": "low_index",
            "n_A": 1.3075,
            "k": 0.00543316,
            "n_layers": 104,
        },
        "reason": "Suspicious hue or colour result: low-index absorber, 850 nm, nA=1.3075, k=0.00543316, 104 layers",
    },
    {
        "filters": {
            "design_peak_wavelength_nm": 850.0,
            "absorber_location": "low_index",
            "n_A": 1.3075,
            "k": 0.00846907,
            "n_layers": 104,
        },
        "reason": "Suspicious hue or colour result: low-index absorber, 850 nm, nA=1.3075, k=0.00543316, 104 layers",
    },
    {
        "filters": {
            "design_peak_wavelength_nm": 850.0,
            "absorber_location": "low_index",
            "n_A": 1.3075,
            "k": 0.01320138,
            "n_layers": 104,
        },
        "reason": "Suspicious hue or colour result: low-index absorber, 850 nm, nA=1.3075, k=0.01320138, 104 layers",
    },
    {
        "filters": {
            "design_peak_wavelength_nm": 850.0,
            "absorber_location": "low_index",
            "n_A": 1.486,
            # "k": 0.00543316,
            "n_layers": 60,
        },
        "reason": "Suspicious hue or colour result: low-index absorber, 850 nm, nA=1.486, k=0.00543316, 60 layers",
    },
    {
        "filters": {
            "design_peak_wavelength_nm": 850.0,
            "absorber_location": "low_index",
            "n_A": 1.3075,
            "k": 0.0001,
            "n_layers": 120,
        },
        "reason": "Suspicious hue or colour result: low-index absorber, 850 nm, nA=1.3075, k=0.0001, 120 layers",
    },
    {
            "filters": {
                "design_peak_wavelength_nm": 850.0,
                "absorber_location": "low_index",
                "n_A": 1.3075,
                "k": 0.00015588,
                "n_layers": 120,
            },
            "reason": "Suspicious hue or colour result: low-index absorber, 850 nm, nA=1.3075, k=0.00015588, 120 layers",
        },
    {
            "filters": {
                "design_peak_wavelength_nm": 850.0,
                "absorber_location": "low_index",
                "n_A": 1.3075,
                "k": 0.00024298,
                "n_layers": 120,
            },
            "reason": "Suspicious hue or colour result: low-index absorber, 850 nm, nA=1.3075, k=0.00024298, 120 layers",
        },
    {
            "filters": {
                "design_peak_wavelength_nm": 850.0,
                "absorber_location": "low_index",
                "n_A": 1.3075,
                "k": 0.00037875,
                "n_layers": 120,
            },
            "reason": "Suspicious hue or colour result: low-index absorber, 850 nm, nA=1.3075, k=0.00037875, 120 layers",
        },
    {
            "filters": {
                "design_peak_wavelength_nm": 850.0,
                "absorber_location": "low_index",
                "n_A": 1.3075,
                "k": 0.00059038,
                "n_layers": 120,
            },
            "reason": "Suspicious hue or colour result: low-index absorber, 850 nm, nA=1.3075, k=0.00059038, 120 layers",
        },
    {
            "filters": {
                "design_peak_wavelength_nm": 850.0,
                "absorber_location": "low_index",
                "n_A": 1.3075,
                "k": 0.00092028,
                "n_layers": 120,
            },
            "reason": "Suspicious hue or colour result: low-index absorber, 850 nm, nA=1.3075, k=0.00092028, 120 layers",
        },
    {
            "filters": {
                "design_peak_wavelength_nm": 850.0,
                "absorber_location": "low_index",
                "n_A": 1.3075,
                "k": 0.0014345,
                "n_layers": 120,
            },
            "reason": "Suspicious hue or colour result: low-index absorber, 850 nm, nA=1.3075, k=0.0014345, 120 layers",
        },
    {
            "filters": {
                "design_peak_wavelength_nm": 850.0,
                "absorber_location": "low_index",
                "n_A": 1.3075,
                "k": 0.00223607,
                "n_layers": 120,
            },
            "reason": "Suspicious hue or colour result: low-index absorber, 850 nm, nA=1.3075, k=0.00223607, 120 layers",
        },
    {
            "filters": {
                "design_peak_wavelength_nm": 850.0,
                "absorber_location": "low_index",
                "n_A": 1.3075,
                "k": 0.00348553,
                "n_layers": 120,
            },
            "reason": "Suspicious hue or colour result: low-index absorber, 850 nm, nA=1.3075, k=0.00348553, 120 layers",
        },
    {
            "filters": {
                "design_peak_wavelength_nm": 380.0,
                "absorber_location": "low_index",
                "n_A": 1.1885,
                "k": 0.0001,
                "n_layers": 60,
            },
            "reason": "Suspicious hue or colour result: low-index absorber, 380 nm, nA=1.1885, k=0.0001, 60 layers",
        },
    {
            "filters": {
                "design_peak_wavelength_nm": 380.0,
                "absorber_location": "low_index",
                "n_A": 1.1885,
                "k": 0.00015588,
                "n_layers": 60,
            },
            "reason": "Suspicious hue or colour result: low-index absorber, 380 nm, nA=1.1885, k=0.00015588, 60 layers",
        },
    {
            "filters": {
                "design_peak_wavelength_nm": 380.0,
                "absorber_location": "low_index",
                "n_A": 1.1885,
                "k": 0.00024298,
                "n_layers": 60,
            },
            "reason": "Suspicious hue or colour result: low-index absorber, 380 nm, nA=1.1885, k=0.00024298, 60 layers",
        },
    {
            "filters": {
                "design_peak_wavelength_nm": 380.0,
                "absorber_location": "low_index",
                "n_A": 1.1885,
                "k": 0.00037875,
                "n_layers": 60,
            },
            "reason": "Suspicious hue or colour result: low-index absorber, 380 nm, nA=1.1885, k=0.00037875, 60 layers",
        },
    {
            "filters": {
                "design_peak_wavelength_nm": 380.0,
                "absorber_location": "low_index",
                "n_A": 1.1885,
                "k": 0.00059038,
                "n_layers": 60,
            },
            "reason": "Suspicious hue or colour result: low-index absorber, 380 nm, nA=1.1885, k=0.00059038, 60 layers",
        },
    {
            "filters": {
                "design_peak_wavelength_nm": 380.0,
                "absorber_location": "low_index",
                "n_A": 1.1885,
                "k": 0.00092028,
                "n_layers": 60,
            },
            "reason": "Suspicious hue or colour result: low-index absorber, 380 nm, nA=1.1885, k=0.00092028, 60 layers",
        },
    {
            "filters": {
                "design_peak_wavelength_nm": 380.0,
                "absorber_location": "low_index",
                "n_A": 1.1885,
                "k": 0.0014345,
                "n_layers": 60,
            },
            "reason": "Suspicious hue or colour result: low-index absorber, 380 nm, nA=1.1885, k=0.0014345, 60 layers",
        },
    {
            "filters": {
                "design_peak_wavelength_nm": 380.0,
                "absorber_location": "low_index",
                "n_A": 1.1885,
                "k": 0.00223607,
                "n_layers": 60,
            },
            "reason": "Suspicious hue or colour result: low-index absorber, 380 nm, nA=1.1885, k=0.00223607, 60 layers",
        },
    {
            "filters": {
                "design_peak_wavelength_nm": 380.0,
                "absorber_location": "low_index",
                "n_A": 1.1885,
                "k": 0.00348553,
                "n_layers": 60,
            },
            "reason": "Suspicious hue or colour result: low-index absorber, 380 nm, nA=1.1885, k=0.00348553, 60 layers",
        },
    {
            "filters": {
                "design_peak_wavelength_nm": 380.0,
                "absorber_location": "low_index",
                "n_A": 1.1885,
                "k": 0.00543316,
                "n_layers": 60,
            },
            "reason": "Suspicious hue or colour result: low-index absorber, 380 nm, nA=1.1885, k=0.00543316, 60 layers",
        },
    {
            "filters": {
                "design_peak_wavelength_nm": 380.0,
                "absorber_location": "low_index",
                "n_A": 1.1885,
                "k": 0.00846907,
                "n_layers": 60,
            },
            "reason": "Suspicious hue or colour result: low-index absorber, 380 nm, nA=1.1885, k=0.00846907, 60 layers",
        },
    {
            "filters": {
                "design_peak_wavelength_nm": 380.0,
                "absorber_location": "low_index",
                "n_A": 1.1885,
                "k": 0.01320138,
                "n_layers": 60,
            },
            "reason": "Suspicious hue or colour result: low-index absorber, 380 nm, nA=1.1885, k=0.01320138, 60 layers",
        },
    {
            "filters": {
                "design_peak_wavelength_nm": 380.0,
                "absorber_location": "low_index",
                "n_A": 1.1885,
                "k": 0.0001,
                "n_layers": 69,
            },
            "reason": "Suspicious hue or colour result: low-index absorber, 380 nm, nA=1.1885, k=0.0001, 69 layers",
        },
    {
            "filters": {
                "design_peak_wavelength_nm": 380.0,
                "absorber_location": "low_index",
                "n_A": 1.1885,
                "k": 0.00015588,
                "n_layers": 69,
            },
            "reason": "Suspicious hue or colour result: low-index absorber, 380 nm, nA=1.1885, k=0.00015588, 69 layers",
        },
    {
            "filters": {
                "design_peak_wavelength_nm": 380.0,
                "absorber_location": "low_index",
                "n_A": 1.1885,
                "k": 0.00024298,
                "n_layers": 69,
            },
            "reason": "Suspicious hue or colour result: low-index absorber, 380 nm, nA=1.1885, k=0.00024298, 69 layers",
        },
    {
            "filters": {
                "design_peak_wavelength_nm": 380.0,
                "absorber_location": "low_index",
                "n_A": 1.1885,
                "k": 0.00037875,
                "n_layers": 69,
            },
            "reason": "Suspicious hue or colour result: low-index absorber, 380 nm, nA=1.1885, k=0.00037875, 69 layers",
        },
    {
            "filters": {
                "design_peak_wavelength_nm": 380.0,
                "absorber_location": "low_index",
                "n_A": 1.1885,
                "k": 0.00059038,
                "n_layers": 69,
            },
            "reason": "Suspicious hue or colour result: low-index absorber, 380 nm, nA=1.1885, k=0.00059038, 69 layers",
        },
    {
            "filters": {
                "design_peak_wavelength_nm": 380.0,
                "absorber_location": "low_index",
                "n_A": 1.1885,
                "k": 0.00092028,
                "n_layers": 69,
            },
            "reason": "Suspicious hue or colour result: low-index absorber, 380 nm, nA=1.1885, k=0.00092028, 69 layers",
        },
    {
            "filters": {
                "design_peak_wavelength_nm": 380.0,
                "absorber_location": "low_index",
                "n_A": 1.1885,
                "k": 0.0014345,
                "n_layers": 69,
            },
            "reason": "Suspicious hue or colour result: low-index absorber, 380 nm, nA=1.1885, k=0.0014345, 69 layers",
        },
    {
            "filters": {
                "design_peak_wavelength_nm": 380.0,
                "absorber_location": "low_index",
                "n_A": 1.1885,
                "k": 0.00223607,
                "n_layers": 69,
            },
            "reason": "Suspicious hue or colour result: low-index absorber, 380 nm, nA=1.1885, k=0.00223607, 69 layers",
        },
    {
            "filters": {
                "design_peak_wavelength_nm": 380.0,
                "absorber_location": "low_index",
                "n_A": 1.1885,
                "k": 0.00348553,
                "n_layers": 69,
            },
            "reason": "Suspicious hue or colour result: low-index absorber, 380 nm, nA=1.1885, k=0.00348553, 69 layers",
        },
    {
            "filters": {
                "design_peak_wavelength_nm": 380.0,
                "absorber_location": "low_index",
                "n_A": 1.1885,
                "k": 0.00543316,
                "n_layers": 69,
            },
            "reason": "Suspicious hue or colour result: low-index absorber, 380 nm, nA=1.1885, k=0.00543316, 69 layers",
        },
    {
            "filters": {
                "design_peak_wavelength_nm": 380.0,
                "absorber_location": "low_index",
                "n_A": 1.1885,
                "k": 0.00846907,
                "n_layers": 69,
            },
            "reason": "Suspicious hue or colour result: low-index absorber, 380 nm, nA=1.1885, k=0.00846907, 69 layers",
        },
    {
            "filters": {
                "design_peak_wavelength_nm": 380.0,
                "absorber_location": "low_index",
                "n_A": 1.1885,
                "k": 0.01320138,
                "n_layers": 69,
            },
            "reason": "Suspicious hue or colour result: low-index absorber, 380 nm, nA=1.1885, k=0.01320138, 69 layers",
        },
    {
            "filters": {
                "design_peak_wavelength_nm": 380.0,
                "absorber_location": "low_index",
                "n_A": 1.1885,
                "k": 0.0001,
                "n_layers": 79,
            },
            "reason": "Suspicious hue or colour result: low-index absorber, 380 nm, nA=1.1885, k=0.0001, 79 layers",
        },
    {
            "filters": {
                "design_peak_wavelength_nm": 380.0,
                "absorber_location": "low_index",
                "n_A": 1.1885,
                "k": 0.00015588,
                "n_layers": 79,
            },
            "reason": "Suspicious hue or colour result: low-index absorber, 380 nm, nA=1.1885, k=0.00015588, 79 layers",
        },
    {
            "filters": {
                "design_peak_wavelength_nm": 380.0,
                "absorber_location": "low_index",
                "n_A": 1.1885,
                "k": 0.00024298,
                "n_layers": 79,
            },
            "reason": "Suspicious hue or colour result: low-index absorber, 380 nm, nA=1.1885, k=0.00024298, 79 layers",
        },
    {
            "filters": {
                "design_peak_wavelength_nm": 380.0,
                "absorber_location": "low_index",
                "n_A": 1.1885,
                "k": 0.00037875,
                "n_layers": 79,
            },
            "reason": "Suspicious hue or colour result: low-index absorber, 380 nm, nA=1.1885, k=0.00037875, 79 layers",
        },
    {
            "filters": {
                "design_peak_wavelength_nm": 380.0,
                "absorber_location": "low_index",
                "n_A": 1.1885,
                "k": 0.00059038,
                "n_layers": 79,
            },
            "reason": "Suspicious hue or colour result: low-index absorber, 380 nm, nA=1.1885, k=0.00059038, 79 layers",
        },
    {
            "filters": {
                "design_peak_wavelength_nm": 380.0,
                "absorber_location": "low_index",
                "n_A": 1.1885,
                "k": 0.00092028,
                "n_layers": 79,
            },
            "reason": "Suspicious hue or colour result: low-index absorber, 380 nm, nA=1.1885, k=0.00092028, 79 layers",
        },
    {
            "filters": {
                "design_peak_wavelength_nm": 380.0,
                "absorber_location": "low_index",
                "n_A": 1.1885,
                "k": 0.0014345,
                "n_layers": 79,
            },
            "reason": "Suspicious hue or colour result: low-index absorber, 380 nm, nA=1.1885, k=0.0014345, 79 layers",
        },
    {
            "filters": {
                "design_peak_wavelength_nm": 380.0,
                "absorber_location": "low_index",
                "n_A": 1.1885,
                "k": 0.00223607,
                "n_layers": 79,
            },
            "reason": "Suspicious hue or colour result: low-index absorber, 380 nm, nA=1.1885, k=0.00223607, 79 layers",
        },
    {
            "filters": {
                "design_peak_wavelength_nm": 380.0,
                "absorber_location": "low_index",
                "n_A": 1.1885,
                "k": 0.00348553,
                "n_layers": 79,
            },
            "reason": "Suspicious hue or colour result: low-index absorber, 380 nm, nA=1.1885, k=0.00348553, 79 layers",
        },
    {
            "filters": {
                "design_peak_wavelength_nm": 380.0,
                "absorber_location": "low_index",
                "n_A": 1.1885,
                "k": 0.00543316,
                "n_layers": 79,
            },
            "reason": "Suspicious hue or colour result: low-index absorber, 380 nm, nA=1.1885, k=0.00543316, 79 layers",
        },
    {
            "filters": {
                "design_peak_wavelength_nm": 380.0,
                "absorber_location": "low_index",
                "n_A": 1.1885,
                "k": 0.00846907,
                "n_layers": 79,
            },
            "reason": "Suspicious hue or colour result: low-index absorber, 380 nm, nA=1.1885, k=0.00846907, 79 layers",
        },
    {
            "filters": {
                "design_peak_wavelength_nm": 380.0,
                "absorber_location": "low_index",
                "n_A": 1.1885,
                "k": 0.01320138,
                "n_layers": 79,
            },
            "reason": "Suspicious hue or colour result: low-index absorber, 380 nm, nA=1.1885, k=0.01320138, 79 layers",
        },
]
#     [ 0.          0.0001      0.00015588  0.00024298  0.00037875  0.00059038
#   0.00092028  0.0014345   0.00223607  0.00348553  0.00543316  0.00846907
#   0.01320138  0.02057799  0.03207646  0.05      ]


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


# In[5]:


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


# In[6]:


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


# In[7]:


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


# In[8]:


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


# In[9]:


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


# In[10]:


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

plt.grid(False, which='both', axis='both')
plt.show()


# In[11]:


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

plt.grid(False, which='both', axis='both')
plt.show()


# In[12]:


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


# In[13]:


# ============================================================
# COMPARE ABSORBING BEST CASES WITH NONABSORBING ANALOGUES
#
# Required:
#     df_best_cases
#     df_analysis
#
# The analogue has the same geometry, material refractive indices,
# design wavelength, and collection configuration, but:
#
#     absorber_location = "none"
#     k = 0
#
# Outputs:
#     df_absorption_comparison
#     df_absorption_comparison_unmatched
# ============================================================

import numpy as np
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

DESIGN_WAVELENGTH_COLUMN = (
    "design_peak_wavelength_nm"
)

ABSORBER_LOCATION_COLUMN = (
    "absorber_location"
)

ABSORBING_LOCATIONS = [
    "low_index",
    "all_layers",
    "core",
    "outer_shell",
]

NONABSORBING_LOCATION = "none"

# Tolerances used for numerical parameter matching.
MATCH_RTOL = 1e-6
MATCH_ATOL = 1e-8


# ============================================================
# SELECT MATCHING PARAMETERS
# ============================================================
#
# Only columns present in both dataframes are used.
# Add further geometry-defining columns here if necessary.

candidate_match_columns = [
    "design_peak_wavelength_nm",
    "n_A",
    "n_B",
    "n_layers",
    "outer_layer",
    "core_thickness_factor",
    "core_thickness_factor_requested",
    "collection_na",
    "collection_NA",
    "n_medium",
]

match_columns = [
    column
    for column in candidate_match_columns
    if (
        column in df_best_cases.columns
        and column in df_analysis.columns
    )
]

required_columns = {
    DESIGN_WAVELENGTH_COLUMN,
    ABSORBER_LOCATION_COLUMN,
    "eta_C",
    "eta_L",
}

missing_best = (
    required_columns
    - set(df_best_cases.columns)
)

missing_analysis = (
    required_columns
    - set(df_analysis.columns)
)

if missing_best:
    raise KeyError(
        "df_best_cases is missing required columns: "
        f"{sorted(missing_best)}"
    )

if missing_analysis:
    raise KeyError(
        "df_analysis is missing required columns: "
        f"{sorted(missing_analysis)}"
    )

if not match_columns:
    raise ValueError(
        "No common geometry columns are available for matching."
    )

print(
    "Columns used to identify nonabsorbing analogues:"
)

print(match_columns)


# ============================================================
# MATCHING HELPER
# ============================================================

def matching_mask(
    dataframe,
    reference_row,
    columns,
    *,
    rtol=MATCH_RTOL,
    atol=MATCH_ATOL,
):
    """
    Return rows matching the reference case in all selected columns.

    Numeric columns are compared with np.isclose.
    Text and categorical columns are compared exactly.
    """

    mask = np.ones(
        len(dataframe),
        dtype=bool,
    )

    for column in columns:

        reference_value = reference_row[column]

        if pd.isna(reference_value):

            mask &= dataframe[
                column
            ].isna().to_numpy()

        elif (
            pd.api.types.is_numeric_dtype(
                dataframe[column]
            )
            or isinstance(
                reference_value,
                (
                    int,
                    float,
                    np.integer,
                    np.floating,
                ),
            )
        ):

            values = pd.to_numeric(
                dataframe[column],
                errors="coerce",
            ).to_numpy(dtype=float)

            mask &= np.isclose(
                values,
                float(reference_value),
                rtol=rtol,
                atol=atol,
                equal_nan=False,
            )

        else:

            mask &= (
                dataframe[column]
                .astype(str)
                .to_numpy()
                == str(reference_value)
            )

    return mask


def calculate_d_opt(
    eta_C,
    eta_L,
):
    """
    Equal-weight distance from the normalized optimum.

    Values above unity receive neither a penalty nor an additional
    reward.
    """

    eta_C_capped = min(
        float(eta_C),
        1.0,
    )

    eta_L_capped = min(
        float(eta_L),
        1.0,
    )

    return float(
        np.hypot(
            1.0 - eta_C_capped,
            1.0 - eta_L_capped,
        )
    )


# ============================================================
# NONABSORBING REFERENCE POOL
# ============================================================

df_nonabsorbing = df_analysis.loc[
    df_analysis[
        ABSORBER_LOCATION_COLUMN
    ].astype(str)
    == NONABSORBING_LOCATION
].copy()

if df_nonabsorbing.empty:
    raise ValueError(
        "df_analysis contains no nonabsorbing reference cases."
    )


# ============================================================
# MATCH EVERY ABSORBING BEST CASE
# ============================================================

comparison_records = []
unmatched_records = []

df_absorbing_best = df_best_cases.loc[
    df_best_cases[
        ABSORBER_LOCATION_COLUMN
    ].isin(
        ABSORBING_LOCATIONS
    )
].copy()

print(
    "Absorbing selected cases:",
    len(df_absorbing_best),
)


for best_index, absorbing_row in (
    df_absorbing_best.iterrows()
):

    analogue_mask = matching_mask(
        df_nonabsorbing,
        absorbing_row,
        match_columns,
    )

    analogue_candidates = (
        df_nonabsorbing.loc[
            analogue_mask
        ].copy()
    )

    target_wavelength_nm = float(
        absorbing_row[
            DESIGN_WAVELENGTH_COLUMN
        ]
    )

    absorber_location = str(
        absorbing_row[
            ABSORBER_LOCATION_COLUMN
        ]
    )

    if analogue_candidates.empty:

        unmatched_records.append(
            {
                "best_case_index": best_index,
                "design_peak_wavelength_nm": (
                    target_wavelength_nm
                ),
                "absorber_location": (
                    absorber_location
                ),
                "reason": (
                    "No nonabsorbing analogue found"
                ),
            }
        )

        continue

    if len(analogue_candidates) > 1:

        # Identical bare cases can occur if the original sweep
        # redundantly included several k values for "none".
        #
        # Prefer the candidate with k closest to zero.
        if "k" in analogue_candidates.columns:

            analogue_candidates[
                "_absolute_k"
            ] = np.abs(
                pd.to_numeric(
                    analogue_candidates["k"],
                    errors="coerce",
                ).fillna(0.0)
            )

            analogue_candidates = (
                analogue_candidates
                .sort_values(
                    "_absolute_k"
                )
            )

        print(
            f"Warning: {len(analogue_candidates)} "
            f"nonabsorbing matches found for "
            f"{target_wavelength_nm:.0f} nm, "
            f"{absorber_location}. "
            "Using the first zero-k candidate."
        )

    reference_row = (
        analogue_candidates.iloc[0]
    )


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    eta_C_abs = float(
        absorbing_row["eta_C"]
    )

    eta_L_abs = float(
        absorbing_row["eta_L"]
    )

    eta_C_none = float(
        reference_row["eta_C"]
    )

    eta_L_none = float(
        reference_row["eta_L"]
    )

    d_opt_abs = calculate_d_opt(
        eta_C_abs,
        eta_L_abs,
    )

    d_opt_none = calculate_d_opt(
        eta_C_none,
        eta_L_none,
    )

    delta_eta_C = (
        eta_C_abs - eta_C_none
    )

    delta_eta_L = (
        eta_L_abs - eta_L_none
    )

    delta_d_opt = (
        d_opt_abs - d_opt_none
    )

    relative_eta_C_change = (
        delta_eta_C / eta_C_none
        if not np.isclose(
            eta_C_none,
            0.0,
        )
        else np.nan
    )

    relative_eta_L_change = (
        delta_eta_L / eta_L_none
        if not np.isclose(
            eta_L_none,
            0.0,
        )
        else np.nan
    )


    # --------------------------------------------------------
    # Output record
    # --------------------------------------------------------

    record = {
        "design_peak_wavelength_nm": (
            target_wavelength_nm
        ),
        "absorber_location": (
            absorber_location
        ),
        "absorber_k": float(
            absorbing_row.get(
                "k",
                np.nan,
            )
        ),

        "eta_C_absorbing": eta_C_abs,
        "eta_C_nonabsorbing": eta_C_none,
        "delta_eta_C": delta_eta_C,
        "relative_eta_C_change": (
            relative_eta_C_change
        ),

        "eta_L_absorbing": eta_L_abs,
        "eta_L_nonabsorbing": eta_L_none,
        "delta_eta_L": delta_eta_L,
        "relative_eta_L_change": (
            relative_eta_L_change
        ),

        "d_opt_absorbing": d_opt_abs,
        "d_opt_nonabsorbing": d_opt_none,
        "delta_d_opt": delta_d_opt,

        # Negative delta_d_opt means absorption improved the
        # distance to the hue-matched optimum.
        "absorption_improves_d_opt": (
            delta_d_opt < 0.0
        ),

        "absorption_improves_eta_C": (
            delta_eta_C > 0.0
        ),

        "absorption_improves_eta_L": (
            delta_eta_L > 0.0
        ),
    }

    for column in match_columns:

        record[column] = (
            absorbing_row[column]
        )

    for optional_column in [
        "diameter_nm",
        "hex",
        "hex_display",
        "L",
        "a",
        "b",
        "C",
        "hue_deg",
    ]:
        if optional_column in absorbing_row.index:
            record[
                f"{optional_column}_absorbing"
            ] = absorbing_row[
                optional_column
            ]

        if optional_column in reference_row.index:
            record[
                f"{optional_column}_nonabsorbing"
            ] = reference_row[
                optional_column
            ]

    comparison_records.append(record)


# ============================================================
# RESULTS
# ============================================================

df_absorption_comparison = pd.DataFrame(
    comparison_records
)

df_absorption_comparison_unmatched = (
    pd.DataFrame(
        unmatched_records
    )
)

if not df_absorption_comparison.empty:

    df_absorption_comparison = (
        df_absorption_comparison
        .sort_values(
            [
                "design_peak_wavelength_nm",
                "absorber_location",
            ]
        )
        .reset_index(drop=True)
    )

    display_columns = [
        "design_peak_wavelength_nm",
        "absorber_location",
        "absorber_k",
        "n_A",
        "n_layers",

        "eta_C_nonabsorbing",
        "eta_C_absorbing",
        "delta_eta_C",
        "relative_eta_C_change",

        "eta_L_nonabsorbing",
        "eta_L_absorbing",
        "delta_eta_L",
        "relative_eta_L_change",

        "d_opt_nonabsorbing",
        "d_opt_absorbing",
        "delta_d_opt",

        "absorption_improves_eta_C",
        "absorption_improves_eta_L",
        "absorption_improves_d_opt",
    ]

    display_columns = [
        column
        for column in display_columns
        if column
        in df_absorption_comparison.columns
    ]

    display(
        df_absorption_comparison[
            display_columns
        ]
        .style.format(
            {
                "absorber_k": "{:.4g}",
                "n_A": "{:.4f}",
                "eta_C_nonabsorbing": "{:.3f}",
                "eta_C_absorbing": "{:.3f}",
                "delta_eta_C": "{:+.3f}",
                "relative_eta_C_change": "{:+.1%}",
                "eta_L_nonabsorbing": "{:.3f}",
                "eta_L_absorbing": "{:.3f}",
                "delta_eta_L": "{:+.3f}",
                "relative_eta_L_change": "{:+.1%}",
                "d_opt_nonabsorbing": "{:.3f}",
                "d_opt_absorbing": "{:.3f}",
                "delta_d_opt": "{:+.3f}",
            }
        )
    )

else:

    print(
        "No absorbing selected cases could be matched."
    )


if not df_absorption_comparison_unmatched.empty:

    print(
        "\nSelected absorbing cases without a matching "
        "nonabsorbing analogue:"
    )

    display(
        df_absorption_comparison_unmatched
    )


# ============================================================
# COMPACT SUMMARY
# ============================================================

if not df_absorption_comparison.empty:

    print("\nAbsorption comparison summary")
    print("-----------------------------")

    print(
        "Cases with improved eta_C:",
        int(
            df_absorption_comparison[
                "absorption_improves_eta_C"
            ].sum()
        ),
        "/",
        len(df_absorption_comparison),
    )

    print(
        "Cases with improved eta_L:",
        int(
            df_absorption_comparison[
                "absorption_improves_eta_L"
            ].sum()
        ),
        "/",
        len(df_absorption_comparison),
    )

    print(
        "Cases with reduced d_opt:",
        int(
            df_absorption_comparison[
                "absorption_improves_d_opt"
            ].sum()
        ),
        "/",
        len(df_absorption_comparison),
    )


# In[14]:


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

plt.grid(False, which='both', axis='both')
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


# In[15]:


print(df_analysis.columns)
print(df_plot["k"].unique())
# print(df_plot["collection_na"].unique())
print(df_plot["n_A"].unique())
# print(df_plot["design_peak_wavelength_nm"].unique())


# In[16]:


# Define cases considered invalid
wrong_case_1 = (
    (df_all["design_peak_wavelength_nm"] == 850.0)
    & (df_all["absorber_location"] == "none")
    & np.isclose(df_all["n_A"].astype(float), 1.3075)
    # & np.isclose(df_all["k"].astype(float), 0.001)
    & (df_all["n_layers"] == 120)
)

# low index

wrong_case_6 = (
    (df_all["design_peak_wavelength_nm"] == 380.0)
    & (df_all["absorber_location"] == "low_index")
    & np.isclose(df_all["n_A"].astype(float), 1.605)
    & np.isclose(df_all["k"].astype(float), 0.00223607)
    & (df_all["n_layers"] == 120)
)

{
        "filters": {
            "design_peak_wavelength_nm": 850.0,
            "absorber_location": "low_index",
            "n_A": 1.3075,
            "k": 0.0001,
            "n_layers": 120,
        },
        "reason": "Suspicious hue or colour result: low-index absorber, 850 nm, nA=1.3075, k=0.0001, 120 layers",
    },

# wrong_mask = wrong_case_1 | wrong_case_2

# display(df_all.loc[wrong_mask])

# df_all_clean = df_all.loc[~wrong_mask].copy()


# In[17]:


# ============================================================
# PART C — COLOUR STRIP OPTION 2
# Multi-row strip grid:
# rows = full range of one sweep parameter
# columns = design wavelength
# ============================================================


# ------------------------------------------------------------
# User choices
# ------------------------------------------------------------
row_variable = "n_A"   # e.g. "k", "n_layers", "collection_angle_deg", "n_A"
colour_hex_col = "hex"                  # or "hex_display"

fixed_filters_grid = {
    # "collection_angle_deg": 90,   # removed automatically if row_variable matches
    # "n_layers": 104,
    # "n_A": 1.248,
    # "k": 0.00059038,
    # "requested_diameter_nm": 15000.0,
    "absorber_location": "all_layers",  # or "outer_shell", "core", "low_index", "all_layers", "none"
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

plt.grid(False, which='both', axis='both')

# Set figure size to 33.2 cm × 14.8 cm
# fig.set_size_inches(33.2 / 2.54, 14.8 / 2.54)

import os

path = "/mnt/c/Users/SchwarzN/OneDrive - Université de Fribourg/Institution/P1_BraggSpericalPigments/Simulation/MIE_Computation/Scattnlay_BraggOnion/outputs/run_secondary_sweep_02"
plt.savefig(os.path.join(path, f"ColourStrip_row_var_{grid_meta['row_variable']}_abs_loc_{fixed_filters_grid.get('absorber_location', 'any')}.png"), dpi = 600, transparent = True)
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


# In[18]:


# CIELAB a*–b* PLANE WITH RÖSCH–MACADAM MAX-CHROMA REFERENCE POINTS

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.patches import Circle
from matplotlib.lines import Line2D

import matplotlib as mpl
mpl.rcdefaults()   # reset to Matplotlib defaults
plt.style.use('/mnt/c/Users/SchwarzN/OneDrive - Université de Fribourg/Institution/Other/Scripts_and_HowTo/science_paper.mplstyle')



# ============================================================
# SETTINGS
# ============================================================

# Windows path:
# C:\Users\SchwarzN\OneDrive - Université de Fribourg\Institution\
# P1_BraggSpericalPigments\Simulation\MIE_Computation\
# Scattnlay_BraggOnion\src\bragg_onion

# WSL equivalent:
ROSCH_CSV = Path(
    "/mnt/c/Users/SchwarzN/"
    "OneDrive - Université de Fribourg/"
    "Institution/P1_BraggSpericalPigments/"
    "Simulation/MIE_Computation/"
    "Scattnlay_BraggOnion/src/bragg_onion/"
    "rosch_macadam_max_chroma_per_hue_1deg.csv"
)

# Plot only every nth reference point.
ROSCH_SAMPLE_STEP = 10

# Marker used for every Rösch–MacAdam reference point.
ROSCH_MARKER = "o"

ROSCH_MARKER_SIZE = 70
ROSCH_EDGE_COLOR = "black"
ROSCH_EDGE_WIDTH = 0.7
ROSCH_ALPHA = 1.0

# Use smoothed coordinates and smoothed display colour.
ROSCH_A_COLUMN = "a_smooth"
ROSCH_B_COLUMN = "b_smooth"
ROSCH_HEX_COLUMN = "hex_smooth"

# Set False if you only want to plot the reference markers.
PLOT_SWEEP_RESULTS = True


# ============================================================
# LOAD AND VALIDATE RÖSCH–MACADAM DATA
# ============================================================

if not ROSCH_CSV.exists():
    raise FileNotFoundError(
        "Rösch–MacAdam CSV was not found at:\n"
        f"{ROSCH_CSV}"
    )

df_rosch = pd.read_csv(ROSCH_CSV)

required_rosch_columns = {
    "hue_deg",
    ROSCH_A_COLUMN,
    ROSCH_B_COLUMN,
    ROSCH_HEX_COLUMN,
}

missing_rosch_columns = (
    required_rosch_columns - set(df_rosch.columns)
)

if missing_rosch_columns:
    raise KeyError(
        "Rösch–MacAdam CSV is missing required columns: "
        f"{sorted(missing_rosch_columns)}"
    )

for column in [
    "hue_deg",
    ROSCH_A_COLUMN,
    ROSCH_B_COLUMN,
]:
    df_rosch[column] = pd.to_numeric(
        df_rosch[column],
        errors="coerce",
    )

df_rosch = df_rosch.dropna(
    subset=[
        "hue_deg",
        ROSCH_A_COLUMN,
        ROSCH_B_COLUMN,
        ROSCH_HEX_COLUMN,
    ]
).copy()

# Sort by hue before taking every nth point.
df_rosch = (
    df_rosch
    .sort_values("hue_deg")
    .reset_index(drop=True)
)

df_rosch_sampled = (
    df_rosch
    .iloc[::ROSCH_SAMPLE_STEP]
    .copy()
)

print("Rösch–MacAdam CSV:", ROSCH_CSV)
print("Reference rows loaded:", len(df_rosch))
print("Reference rows plotted:", len(df_rosch_sampled))
print(
    "Approximate reference hue spacing:",
    f"{ROSCH_SAMPLE_STEP} degrees",
)


# ============================================================
# MARKERS FOR SWEEP DESIGN WAVELENGTHS
# ============================================================

marker_dict = {
    380.0: "o",
    470.0: "s",
    540.0: "D",
    580.0: "^",
    620.0: "v",
    680.0: "p",
    760.0: "h",
    850.0: "<",
}

design_wavelengths = sorted(
    pd.to_numeric(
        df_analysis["design_peak_wavelength_nm"],
        errors="coerce",
    )
    .dropna()
    .unique()
)

print("Design wavelengths:", design_wavelengths)


# ============================================================
# DETERMINE PLOT RANGE
# ============================================================

all_a_values = np.concatenate(
    [
        pd.to_numeric(
            df_analysis["a"],
            errors="coerce",
        ).dropna().to_numpy(dtype=float),
        df_rosch[ROSCH_A_COLUMN].to_numpy(dtype=float),
    ]
)

all_b_values = np.concatenate(
    [
        pd.to_numeric(
            df_analysis["b"],
            errors="coerce",
        ).dropna().to_numpy(dtype=float),
        df_rosch[ROSCH_B_COLUMN].to_numpy(dtype=float),
    ]
)

plot_radius = 1.08 * max(
    np.nanmax(np.abs(all_a_values)),
    np.nanmax(np.abs(all_b_values)),
    100.0,
)

chroma_ring_step = 20.0

chroma_levels = np.arange(
    chroma_ring_step,
    plot_radius + chroma_ring_step,
    chroma_ring_step,
)


# ============================================================
# PLOT
# ============================================================

cmto_inches = 1.0 / 2.54
fig, ax = plt.subplots(figsize=(10 * cmto_inches, 10 * cmto_inches))


# ------------------------------------------------------------
# Hue-angle spokes
# ------------------------------------------------------------

hue_angles = np.arange(
    0.0,
    360.0,
    30.0,
)

for hue_deg in hue_angles:

    radii = np.array(
        [0.0, plot_radius],
        dtype=float,
    )

    ax.plot(
        radii * np.cos(np.deg2rad(hue_deg)),
        radii * np.sin(np.deg2rad(hue_deg)),
        color="gray",
        linestyle="--",
        linewidth=0.5,
        alpha=0.45,
        zorder=0,
    )

    label_radius = 0.96 * plot_radius

    ax.text(
        label_radius * np.cos(np.deg2rad(hue_deg+7)),
        label_radius * np.sin(np.deg2rad(hue_deg+7)),
        f"{hue_deg:.0f}°",
        ha="center",
        va="center",
        # fontsize=8,
        color="dimgray",
    )


# ------------------------------------------------------------
# Chroma rings
# ------------------------------------------------------------

for i, chroma in enumerate(chroma_levels[:-2]):

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
    if chroma == chroma_levels[-1]:
        continue
    binary = i % 2
    # ax.text(
    #     float(chroma),
    #     3*binary,
    #     f"C*={chroma:.0f}",
    #     ha="left",
    #     va="bottom",
    #     fontsize=7,
    #     color="dimgray",
    #     # rotation=i*1
    # )


# ------------------------------------------------------------
# Sweep results
# ------------------------------------------------------------

if PLOT_SWEEP_RESULTS:

    for design_wavelength_nm in design_wavelengths:

        subset = df_analysis.loc[
            np.isclose(
                pd.to_numeric(
                    df_analysis[
                        "design_peak_wavelength_nm"
                    ],
                    errors="coerce",
                ),
                float(design_wavelength_nm),
            )
        ]

        if subset.empty:
            continue

        marker = marker_dict.get(
            float(design_wavelength_nm),
            "x",
        )

        scatter_kwargs = {
            "x": subset["a"],
            "y": subset["b"],
            "c": subset["hex"],
            "s": 50,
            "marker": marker,
            "alpha": 1.0,
            "label": f"{design_wavelength_nm:.0f} nm",
            "zorder": 2,
        }

        if marker == "x":
            scatter_kwargs.update(
                {
                    "linewidths": 0.7,
                }
            )
        else:
            scatter_kwargs.update(
                {
                    "edgecolors": "white",
                    "linewidths": 0.05,
                }
            )

        ax.scatter(**scatter_kwargs)


# ------------------------------------------------------------
# Smoothed Rösch–MacAdam maximum-chroma locus
# ------------------------------------------------------------

# Optional thin line through the full smoothed locus.
ax.plot(
    df_rosch[ROSCH_A_COLUMN],
    df_rosch[ROSCH_B_COLUMN],
    color="black",
    linewidth=1.0,
    alpha=0.65,
    label="Rösch–MacAdam maximum-chroma locus",
    zorder=4,
)

# Sampled reference points using one common selectable marker.
ax.scatter(
    df_rosch_sampled[ROSCH_A_COLUMN],
    df_rosch_sampled[ROSCH_B_COLUMN],
    c=df_rosch_sampled[ROSCH_HEX_COLUMN],
    s=ROSCH_MARKER_SIZE,
    marker=ROSCH_MARKER,
    edgecolors=ROSCH_EDGE_COLOR,
    linewidths=ROSCH_EDGE_WIDTH,
    alpha=ROSCH_ALPHA,
    label=(
        "Rösch–MacAdam maximum chroma "
        f"(every {ROSCH_SAMPLE_STEP}th point)"
    ),
    zorder=6,
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
    -1.10 * plot_radius,
    1.10 * plot_radius,
)

ax.set_ylim(
    -1.10 * plot_radius,
    1.10 * plot_radius,
)

ax.set_aspect("equal")

ax.set_xlabel("a*", ha='left')
ax.set_ylabel("b*", ha='left', rotation=0)
ax.xaxis.set_label_coords(0.0, -0.03)   # x, y in axes coordinates
ax.yaxis.set_label_coords(-0.15, 0.475)

# ax.set_title(
#     "CIELAB a*–b* plane with Rösch–MacAdam "
#     "maximum-chroma reference"
# )


# ============================================================
# LEGEND
# ============================================================

sweep_handles = []

if PLOT_SWEEP_RESULTS:

    for wavelength_nm in design_wavelengths:

        marker = marker_dict.get(
            float(wavelength_nm),
            "x",
        )

        sweep_handles.append(
            Line2D(
                [0],
                [0],
                marker=marker,
                linestyle="None",
                color="black",
                markerfacecolor=(
                    "gray"
                    if marker != "x"
                    else "none"
                ),
                markeredgecolor="black",
                markersize=8,
                label=f"{wavelength_nm:.0f} nm",
            )
        )

rosch_handle = Line2D(
    [0],
    [0],
    marker=ROSCH_MARKER,
    linestyle="-",
    color="black",
    markerfacecolor="white",
    markeredgecolor=ROSCH_EDGE_COLOR,
    markersize=8,
    label=(
        "Max-chroma\nlocus "
    ),
)

# ax.legend(
#     handles=sweep_handles + [rosch_handle],
#     loc="upper left",
#     bbox_to_anchor=(0.8, 1.0),
#     title="Design\nwavelength",
# )
ax.grid(False, which="both", axis="both")

# plt.savefig(os.path.join(path, f"CIELAB_ab_plane_RoschMacadam_max_chroma.png"), dpi = 600, transparent = True)
# plt.savefig(os.path.join(path, f"CIELAB_ab_plane_RoschMacadam_max_chroma.pdf"), transparent = True)
plt.show()


# In[40]:


# CIELAB a*–b* PLANE WITH RÖSCH–MACADAM MAX-CHROMA REFERENCE
#
# Sweep visualization options:
#   1. Plot only the outermost sweep point in each angular pizza slice.
#   2. Colour points by design wavelength rather than individual hex colour.
#   3. Increase marker size and opacity with chroma, meaning distance
#      from the CIELAB origin.

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from matplotlib.patches import Circle
from matplotlib.lines import Line2D

from matplotlib.transforms import ScaledTranslation


# ============================================================
# STYLE
# ============================================================

mpl.rcdefaults()

plt.style.use(
    "/mnt/c/Users/SchwarzN/"
    "OneDrive - Université de Fribourg/"
    "Institution/Other/Scripts_and_HowTo/"
    "science_paper.mplstyle"
)


# ============================================================
# FILE PATH
# ============================================================

ROSCH_CSV = Path(
    "/mnt/c/Users/SchwarzN/"
    "OneDrive - Université de Fribourg/"
    "Institution/P1_BraggSpericalPigments/"
    "Simulation/MIE_Computation/"
    "Scattnlay_BraggOnion/src/bragg_onion/"
    "rosch_macadam_max_chroma_per_hue_1deg.csv"
)


# ============================================================
# USER OPTIONS
# ============================================================

fig_width_cm = 15.92 / 2
fig_height_cm = 15.92 / 2

# Plot only every nth Rösch–MacAdam point.
ROSCH_SAMPLE_STEP = 10

ROSCH_MARKER = "o"
ROSCH_MARKER_SIZE = 30
ROSCH_EDGE_COLOR = "black"
ROSCH_EDGE_WIDTH = 0.35

ROSCH_A_COLUMN = "a_smooth"
ROSCH_B_COLUMN = "b_smooth"
ROSCH_HEX_COLUMN = "hex_smooth"

PLOT_SWEEP_RESULTS = True

# ------------------------------------------------------------
# Rösch–MacAdam marker alpha from optimal lightness
# ------------------------------------------------------------

# Options:
#   "lightness" -> opacity varies with L_smooth
#   "fixed"     -> use ROSCH_ALPHA for every marker
ROSCH_ALPHA_MODE = "fixed"

ROSCH_LIGHTNESS_COLUMN = "L_smooth"

ROSCH_ALPHA_MIN = 0.20
ROSCH_ALPHA_MAX = 1.00

# Fixed opacity used only when ROSCH_ALPHA_MODE == "fixed".
ROSCH_ALPHA = 1.00

# False:
#   higher L* -> more opaque
#
# True:
#   lower L* -> more opaque
ROSCH_INVERT_LIGHTNESS_ALPHA = False

# Robust percentile scaling prevents isolated values from controlling
# the complete opacity range.
ROSCH_LIGHTNESS_PERCENTILE_RANGE = (2.0, 98.0)

# ------------------------------------------------------------
# Sweep-point envelope selection
# ------------------------------------------------------------

SHOW_ONLY_OUTERMOST_PER_HUE_SLICE = True
HUE_SLICE_WIDTH_DEG = 5.0

# Selection modes:
#   "global"
#       One outermost point per hue slice across the complete sweep.
#
#   "per_wavelength"
#       One outermost point per hue slice for every design wavelength.
OUTERMOST_SELECTION_MODE = "global"

# ------------------------------------------------------------
# Highlight selected minimum-d_opt cases
# ------------------------------------------------------------

HIGHLIGHT_BEST_CASES = True

BEST_CASE_OUTER_SIZE = 65
BEST_CASE_INNER_SIZE = 85
BEST_CASE_STAR_SIZE = 18

BEST_CASE_HALO_COLOR = "none"
BEST_CASE_EDGE_COLOR = "black"
BEST_CASE_LINEWIDTH = 0.5

# If True, annotate the highlighted cases.
ANNOTATE_BEST_CASES = False

# Star displacement inside asymmetric markers, measured in points.
#
# Positive x moves right.
# Positive y moves up.
BEST_STAR_OFFSET_POINTS = {
    "^": (0.0, -1.6),    # star higher
    "v": (0.0, 1.6),   # star lower
    "<": (1.6, 0.0),    # star right
    ">": (-2.2, 0.0),   # star farther left
}

# Markers not listed above receive no displacement.
BEST_STAR_DEFAULT_OFFSET_POINTS = (0.0, 0.0)

# ------------------------------------------------------------
# Sweep-point filtering
# ------------------------------------------------------------

# If True, retain only the point with maximum C* in each angular
# sector for each design wavelength.
SHOW_ONLY_OUTERMOST_PER_HUE_SLICE = True

# Width of each angular pizza slice.
HUE_SLICE_WIDTH_DEG = 10.0

# Optional minimum chroma for plotted sweep points.
# Set to 0.0 to retain all selected outer-envelope points.
MIN_SWEEP_CHROMA = 0.0

# ------------------------------------------------------------
# Sweep-point appearance
# ------------------------------------------------------------

# Use fixed colours assigned to design wavelengths.
COLOUR_SWEEP_BY_DESIGN_WAVELENGTH = False

# Matplotlib categorical colour map.
# "tab10" provides ten distinct categorical colours.
SWEEP_COLOURMAP = "tab10"

# Sweep-point colour mode:
#   "wavelength" -> categorical colour assigned by design wavelength
#   "hex"        -> actual simulated colour from SWEEP_HEX_COLUMN
SWEEP_COLOUR_MODE = "hex"

# Use "hex" for raw calculated colours or "hex_display" for
# display-normalized colours.
SWEEP_HEX_COLUMN = "hex"

# Marker area limits. Matplotlib scatter size is an area in points squared.
SWEEP_SIZE_MIN = 8.0
SWEEP_SIZE_MAX = 65.0

# ------------------------------------------------------------
# Sweep-point alpha scaling
# ------------------------------------------------------------

# Options:
#   "lightness" -> alpha varies with CIELAB L*
#   "chroma"    -> alpha varies with distance from the origin
#   "fixed"     -> constant alpha for every sweep marker
SWEEP_ALPHA_MODE = "lightness"

SWEEP_ALPHA_MIN = 0.20
SWEEP_ALPHA_MAX = 1.00
SWEEP_ALPHA_FIXED = 0.80

# False:
#   higher L* -> more opaque
#
# True:
#   lower L* -> more opaque
INVERT_LIGHTNESS_ALPHA = False

# Robust scaling avoids isolated extreme values controlling the
# opacity range. Set to (0, 100) to use the full L* range directly.
LIGHTNESS_PERCENTILE_RANGE = (2.0, 98.0)

SWEEP_EDGE_COLOR = "black"
SWEEP_EDGE_WIDTH = 0.25

# Use one common marker instead of wavelength-dependent markers.
USE_COMMON_SWEEP_MARKER = False
COMMON_SWEEP_MARKER = "o"


# ============================================================
# LOAD AND VALIDATE RÖSCH–MACADAM DATA
# ============================================================

if not ROSCH_CSV.exists():
    raise FileNotFoundError(
        "Rösch–MacAdam CSV was not found at:\n"
        f"{ROSCH_CSV}"
    )

df_rosch = pd.read_csv(ROSCH_CSV)

required_rosch_columns = {
    "hue_deg",
    ROSCH_A_COLUMN,
    ROSCH_B_COLUMN,
    ROSCH_HEX_COLUMN,
    ROSCH_LIGHTNESS_COLUMN,
}

missing_rosch_columns = (
    required_rosch_columns - set(df_rosch.columns)
)

if missing_rosch_columns:
    raise KeyError(
        "Rösch–MacAdam CSV is missing required columns: "
        f"{sorted(missing_rosch_columns)}"
    )

for column in [
    "hue_deg",
    ROSCH_A_COLUMN,
    ROSCH_B_COLUMN,
    ROSCH_LIGHTNESS_COLUMN,
]:
    df_rosch[column] = pd.to_numeric(
        df_rosch[column],
        errors="coerce",
    )

df_rosch = (
    df_rosch
    .dropna(
        subset=[
            "hue_deg",
            ROSCH_A_COLUMN,
            ROSCH_B_COLUMN,
            ROSCH_HEX_COLUMN,
            ROSCH_LIGHTNESS_COLUMN,
        ]
    )
    .sort_values("hue_deg")
    .reset_index(drop=True)
)

df_rosch_sampled = (
    df_rosch
    .iloc[::ROSCH_SAMPLE_STEP]
    .copy()
)

# ============================================================
# CALCULATE RÖSCH–MACADAM MARKER ALPHA FROM L*
# ============================================================

valid_rosch_alpha_modes = {
    "lightness",
    "fixed",
}

if ROSCH_ALPHA_MODE not in valid_rosch_alpha_modes:
    raise ValueError(
        "ROSCH_ALPHA_MODE must be 'lightness' or 'fixed', "
        f"got {ROSCH_ALPHA_MODE!r}."
    )


if ROSCH_ALPHA_MODE == "lightness":

    rosch_lightness = pd.to_numeric(
        df_rosch_sampled[ROSCH_LIGHTNESS_COLUMN],
        errors="coerce",
    ).to_numpy(dtype=float)

    finite_rosch_lightness = rosch_lightness[
        np.isfinite(rosch_lightness)
    ]

    if finite_rosch_lightness.size == 0:
        raise ValueError(
            "No finite Rösch–MacAdam lightness values are available "
            "for alpha scaling."
        )

    rosch_lightness_low, rosch_lightness_high = np.nanpercentile(
        finite_rosch_lightness,
        ROSCH_LIGHTNESS_PERCENTILE_RANGE,
    )

    if np.isclose(
        rosch_lightness_high,
        rosch_lightness_low,
    ):
        rosch_lightness_fraction = np.ones(
            len(df_rosch_sampled),
            dtype=float,
        )

    else:
        rosch_lightness_fraction = np.clip(
            (
                rosch_lightness
                - rosch_lightness_low
            )
            / (
                rosch_lightness_high
                - rosch_lightness_low
            ),
            0.0,
            1.0,
        )

    if ROSCH_INVERT_LIGHTNESS_ALPHA:
        rosch_lightness_fraction = (
            1.0 - rosch_lightness_fraction
        )

    rosch_lightness_fraction = np.nan_to_num(
        rosch_lightness_fraction,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )

    df_rosch_sampled["marker_alpha"] = (
        ROSCH_ALPHA_MIN
        + rosch_lightness_fraction
        * (
            ROSCH_ALPHA_MAX
            - ROSCH_ALPHA_MIN
        )
    )

    print(
        "Rösch–MacAdam marker alpha based on:",
        ROSCH_LIGHTNESS_COLUMN,
    )

    print(
        "Rösch–MacAdam lightness scaling range:",
        f"{rosch_lightness_low:.2f} to "
        f"{rosch_lightness_high:.2f}",
    )


elif ROSCH_ALPHA_MODE == "fixed":

    df_rosch_sampled["marker_alpha"] = float(
        ROSCH_ALPHA
    )

# Convert every hexadecimal marker colour into RGBA so each point
# can have an individual alpha value.

rosch_marker_rgba = np.array(
    [
        mpl.colors.to_rgba(
            str(hex_colour),
            alpha=float(marker_alpha),
        )
        for hex_colour, marker_alpha in zip(
            df_rosch_sampled[ROSCH_HEX_COLUMN],
            df_rosch_sampled["marker_alpha"],
        )
    ],
    dtype=float,
)

print("Rösch–MacAdam CSV:", ROSCH_CSV)
print("Reference rows loaded:", len(df_rosch))
print("Reference rows plotted:", len(df_rosch_sampled))


# ============================================================
# PREPARE SWEEP DATA
# ============================================================

required_sweep_columns = {
    "design_peak_wavelength_nm",
    "a",
    "b",
    "hex",
}

missing_sweep_columns = (
    required_sweep_columns - set(df_analysis.columns)
)

if missing_sweep_columns:
    raise KeyError(
        "df_analysis is missing required columns: "
        f"{sorted(missing_sweep_columns)}"
    )

valid_alpha_modes = {
    "lightness",
    "chroma",
    "fixed",
}

if SWEEP_ALPHA_MODE not in valid_alpha_modes:
    raise ValueError(
        "SWEEP_ALPHA_MODE must be 'lightness', 'chroma', "
        f"or 'fixed', got {SWEEP_ALPHA_MODE!r}."
    )

df_ab = df_analysis.copy()


if SWEEP_ALPHA_MODE == "lightness":
    if "L" not in df_ab.columns:
        raise KeyError(
            "SWEEP_ALPHA_MODE='lightness' requires the column 'L'."
        )


for column in [
    "design_peak_wavelength_nm",
    "a",
    "b",
    "L",
]:
    df_ab[column] = pd.to_numeric(
        df_ab[column],
        errors="coerce",
    )

df_ab = df_ab.dropna(
    subset=[
        "design_peak_wavelength_nm",
        "a",
        "b",
    ]
).copy()

# Distance from the origin is CIELAB chroma C*.
df_ab["C_calculated"] = np.hypot(
    df_ab["a"],
    df_ab["b"],
)

# Hue angle in [0°, 360°).
df_ab["hue_calculated_deg"] = np.mod(
    np.degrees(
        np.arctan2(
            df_ab["b"],
            df_ab["a"],
        )
    ),
    360.0,
)

# Assign each point to a fixed-width angular slice.
df_ab["hue_slice_index"] = np.floor(
    df_ab["hue_calculated_deg"]
    / HUE_SLICE_WIDTH_DEG
).astype(int)

df_ab["hue_slice_start_deg"] = (
    df_ab["hue_slice_index"]
    * HUE_SLICE_WIDTH_DEG
)

df_ab["hue_slice_center_deg"] = (
    df_ab["hue_slice_start_deg"]
    + 0.5 * HUE_SLICE_WIDTH_DEG
)

valid_sweep_colour_modes = {
    "wavelength",
    "hex",
}

if SWEEP_COLOUR_MODE not in valid_sweep_colour_modes:
    raise ValueError(
        "SWEEP_COLOUR_MODE must be either "
        f"'wavelength' or 'hex', got {SWEEP_COLOUR_MODE!r}."
    )

if (
    SWEEP_COLOUR_MODE == "hex"
    and SWEEP_HEX_COLUMN not in df_ab.columns
):
    raise KeyError(
        f"SWEEP_HEX_COLUMN={SWEEP_HEX_COLUMN!r} "
        "is not present in df_analysis."
    )

if SWEEP_COLOUR_MODE == "hex":

    invalid_hex_mask = ~df_ab[
        SWEEP_HEX_COLUMN
    ].map(mpl.colors.is_color_like)

    if invalid_hex_mask.any():
        raise ValueError(
            f"{int(invalid_hex_mask.sum())} row(s) contain "
            f"invalid colours in {SWEEP_HEX_COLUMN!r}."
        )

# ============================================================
# PREPARE df_best_cases FOR HIGHLIGHTING
# ============================================================

if HIGHLIGHT_BEST_CASES:

    required_best_columns = {
        "design_peak_wavelength_nm",
        "a",
        "b",
        "d_opt",
        "eta_L",
    }

    missing_best_columns = (
        required_best_columns
        - set(df_best_cases.columns)
    )

    if missing_best_columns:
        raise KeyError(
            "df_best_cases is missing required columns: "
            f"{sorted(missing_best_columns)}"
        )

    df_best_ab = df_best_cases.copy()

    for column in [
        "design_peak_wavelength_nm",
        "a",
        "b",
        "d_opt",
        "eta_L",
    ]:
        df_best_ab[column] = pd.to_numeric(
            df_best_ab[column],
            errors="coerce",
        )

    df_best_ab = df_best_ab.dropna(
        subset=[
            "design_peak_wavelength_nm",
            "a",
            "b",
            "d_opt",
            "eta_L",
        ]
    ).copy()

    df_best_ab["C_calculated"] = np.hypot(
        df_best_ab["a"],
        df_best_ab["b"],
    )

    df_best_ab["hue_calculated_deg"] = np.mod(
        np.degrees(
            np.arctan2(
                df_best_ab["b"],
                df_best_ab["a"],
            )
        ),
        360.0,
    )

    print(
        "Best d_opt cases to highlight:",
        len(df_best_ab),
    )

else:

    df_best_ab = pd.DataFrame()


# ============================================================
# SELECT OUTERMOST POINTS IN EACH PIZZA SLICE
# ============================================================

if SHOW_ONLY_OUTERMOST_PER_HUE_SLICE:

    if OUTERMOST_SELECTION_MODE == "global":

        # One outermost point in each hue slice across all
        # design wavelengths and all parameter combinations.
        group_columns = [
            "hue_slice_index",
        ]

    elif OUTERMOST_SELECTION_MODE == "per_wavelength":

        # One outermost point in each hue slice separately
        # for every design wavelength.
        group_columns = [
            "design_peak_wavelength_nm",
            "hue_slice_index",
        ]

    else:
        raise ValueError(
            "OUTERMOST_SELECTION_MODE must be either "
            "'global' or 'per_wavelength', got "
            f"{OUTERMOST_SELECTION_MODE!r}."
        )

    outermost_indices = (
        df_ab
        .groupby(
            group_columns,
            dropna=False,
        )["C_calculated"]
        .idxmax()
    )

    df_sweep_plot = (
        df_ab.loc[outermost_indices]
        .copy()
        .sort_values(
            [
                "hue_calculated_deg",
                "design_peak_wavelength_nm",
            ]
        )
        .reset_index(drop=True)
    )

else:

    df_sweep_plot = df_ab.copy()


df_sweep_plot = df_sweep_plot.loc[
    df_sweep_plot["C_calculated"]
    >= MIN_SWEEP_CHROMA
].copy()


design_wavelengths = sorted(
    df_ab[
        "design_peak_wavelength_nm"
    ].unique()
)


print("Design wavelengths:", design_wavelengths)
print("Original sweep points:", len(df_ab))
print("Sweep points plotted:", len(df_sweep_plot))

if SHOW_ONLY_OUTERMOST_PER_HUE_SLICE:
    print(
        "Outermost-point selection mode:",
        OUTERMOST_SELECTION_MODE,
    )
    print(
        "Hue-slice width:",
        f"{HUE_SLICE_WIDTH_DEG:g} degrees",
    )


df_sweep_plot = df_sweep_plot.loc[
    df_sweep_plot["C_calculated"]
    >= MIN_SWEEP_CHROMA
].copy()


design_wavelengths = sorted(
    df_sweep_plot[
        "design_peak_wavelength_nm"
    ].unique()
)

print("Design wavelengths:", design_wavelengths)
print("Original sweep points:", len(df_ab))
print("Sweep points plotted:", len(df_sweep_plot))

if SHOW_ONLY_OUTERMOST_PER_HUE_SLICE:
    print(
        "Outermost-point selection:",
        f"one point per {HUE_SLICE_WIDTH_DEG:g}° slice",
        "and design wavelength",
    )


# ============================================================
# MARKERS AND COLOURS FOR DESIGN WAVELENGTHS
# ============================================================

default_marker_dict = {
    380.0: "o",
    470.0: "s",
    540.0: "D",
    580.0: "^",
    620.0: "v",
    680.0: "p",
    760.0: "h",
    850.0: "<",
}

legend_wavelengths = [
    380.0,
    470.0,
    540.0,
    580.0,
    620.0,
    680.0,
    760.0,
    850.0,
]

fallback_markers = [
    "o",
    "s",
    "D",
    "^",
    "v",
    "p",
    "h",
    "<",
    ">",
    "P",
]

marker_dict = {}

for index, wavelength_nm in enumerate(design_wavelengths):

    if USE_COMMON_SWEEP_MARKER:
        marker_dict[wavelength_nm] = COMMON_SWEEP_MARKER

    else:
        marker_dict[wavelength_nm] = (
            default_marker_dict.get(
                float(wavelength_nm),
                fallback_markers[
                    index % len(fallback_markers)
                ],
            )
        )


sweep_cmap = mpl.colormaps[SWEEP_COLOURMAP]

wavelength_colour_dict = {
    wavelength_nm: sweep_cmap(
        index / max(
            len(design_wavelengths) - 1,
            1,
        )
    )
    for index, wavelength_nm
    in enumerate(design_wavelengths)
}


# ============================================================
# SCALE MARKER SIZE WITH CHROMA AND ALPHA WITH SELECTED QUANTITY
# ============================================================

chroma_values = df_sweep_plot[
    "C_calculated"
].to_numpy(dtype=float)

if len(chroma_values) == 0:
    raise ValueError(
        "No sweep points remain after filtering."
    )


# ------------------------------------------------------------
# Marker size from chroma
# ------------------------------------------------------------

chroma_min = float(
    np.nanmin(chroma_values)
)

chroma_max = float(
    np.nanmax(chroma_values)
)

if np.isclose(chroma_max, chroma_min):

    df_sweep_plot["chroma_fraction"] = 1.0

else:

    df_sweep_plot["chroma_fraction"] = np.clip(
        (
            df_sweep_plot["C_calculated"]
            - chroma_min
        )
        / (
            chroma_max
            - chroma_min
        ),
        0.0,
        1.0,
    )


# Nonlinear scaling provides better visual separation.
size_fraction = (
    df_sweep_plot["chroma_fraction"]
    ** 0.75
)

df_sweep_plot["marker_size"] = (
    SWEEP_SIZE_MIN
    + size_fraction
    * (
        SWEEP_SIZE_MAX
        - SWEEP_SIZE_MIN
    )
)


# ------------------------------------------------------------
# Marker alpha
# ------------------------------------------------------------

# if SWEEP_ALPHA_MODE == "lightness":

#     lightness_values = pd.to_numeric(
#         df_sweep_plot["L"],
#         errors="coerce",
#     ).to_numpy(dtype=float)

#     finite_lightness = lightness_values[
#         np.isfinite(lightness_values)
#     ]

#     if finite_lightness.size == 0:
#         raise ValueError(
#             "No finite L* values are available for alpha scaling."
#         )

#     lightness_low, lightness_high = np.nanpercentile(
#         finite_lightness,
#         LIGHTNESS_PERCENTILE_RANGE,
#     )

#     if np.isclose(
#         lightness_high,
#         lightness_low,
#     ):
#         lightness_fraction = np.ones(
#             len(df_sweep_plot),
#             dtype=float,
#         )

#     else:
#         lightness_fraction = np.clip(
#             (
#                 lightness_values
#                 - lightness_low
#             )
#             / (
#                 lightness_high
#                 - lightness_low
#             ),
#             0.0,
#             1.0,
#         )

#     if INVERT_LIGHTNESS_ALPHA:
#         lightness_fraction = (
#             1.0 - lightness_fraction
#         )

#     # Replace nonfinite alpha values with the minimum opacity.
#     lightness_fraction = np.nan_to_num(
#         lightness_fraction,
#         nan=0.0,
#         posinf=1.0,
#         neginf=0.0,
#     )

#     df_sweep_plot["marker_alpha"] = (
#         SWEEP_ALPHA_MIN
#         + lightness_fraction
#         * (
#             SWEEP_ALPHA_MAX
#             - SWEEP_ALPHA_MIN
#         )
#     )

# ============================================================
# SCALE SWEEP-MARKER ALPHA WITH THE DISPLAYED L* RANGE
# ============================================================

if SWEEP_ALPHA_MODE == "lightness":

    lightness_values = pd.to_numeric(
        df_sweep_plot["eta_L"],
        errors="coerce",
    ).to_numpy(dtype=float)

    finite_lightness = lightness_values[
        np.isfinite(lightness_values)
    ]

    if finite_lightness.size == 0:
        raise ValueError(
            "No finite L* values are available for alpha scaling."
        )

    # Actual minimum and maximum L* values appearing in the plot.
    SWEEP_LIGHTNESS_MIN = float(
        np.nanmin(finite_lightness)
    )

    SWEEP_LIGHTNESS_MAX = float(
        np.nanmax(finite_lightness)
    )

    if np.isclose(
        SWEEP_LIGHTNESS_MAX,
        SWEEP_LIGHTNESS_MIN,
    ):
        lightness_fraction = np.ones(
            len(df_sweep_plot),
            dtype=float,
        )

    else:
        lightness_fraction = np.clip(
            (
                lightness_values
                - SWEEP_LIGHTNESS_MIN
            )
            / (
                SWEEP_LIGHTNESS_MAX
                - SWEEP_LIGHTNESS_MIN
            ),
            0.0,
            1.0,
        )

    # If inverted, low-L* markers become more opaque.
    if INVERT_LIGHTNESS_ALPHA:
        alpha_fraction = (
            1.0 - lightness_fraction
        )
    else:
        alpha_fraction = (
            lightness_fraction
        )

    alpha_fraction = np.nan_to_num(
        alpha_fraction,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )

    df_sweep_plot["marker_alpha"] = (
        SWEEP_ALPHA_MIN
        + alpha_fraction
        * (
            SWEEP_ALPHA_MAX
            - SWEEP_ALPHA_MIN
        )
    )


    print(
        "Sweep-marker L* range:",
        f"{SWEEP_LIGHTNESS_MIN:.2f} to "
        f"{SWEEP_LIGHTNESS_MAX:.2f}",
    )

elif SWEEP_ALPHA_MODE == "chroma":

    alpha_fraction = (
        df_sweep_plot["chroma_fraction"]
        ** 0.60
    )

    df_sweep_plot["marker_alpha"] = (
        SWEEP_ALPHA_MIN
        + alpha_fraction
        * (
            SWEEP_ALPHA_MAX
            - SWEEP_ALPHA_MIN
        )
    )


elif SWEEP_ALPHA_MODE == "fixed":

    df_sweep_plot["marker_alpha"] = float(
        SWEEP_ALPHA_FIXED
    )


print(
    "Sweep alpha mode:",
    SWEEP_ALPHA_MODE,
)

if SWEEP_ALPHA_MODE == "lightness":
    print(
        "Lightness scaling range:",
        # f"L* = {lightness_low:.2f} to {lightness_high:.2f}",
        f"L* = {SWEEP_LIGHTNESS_MIN:.2f} to {SWEEP_LIGHTNESS_MAX:.2f}",
    )


def calculate_lightness_alpha(eta_L):
    """Convert eta_L to the same alpha scale used for sweep markers."""

    if not np.isfinite(eta_L):
        return SWEEP_ALPHA_MIN

    if np.isclose(
        SWEEP_LIGHTNESS_MAX,
        SWEEP_LIGHTNESS_MIN,
    ):
        lightness_fraction = 1.0
    else:
        lightness_fraction = np.clip(
            (
                eta_L
                - SWEEP_LIGHTNESS_MIN
            )
            / (
                SWEEP_LIGHTNESS_MAX
                - SWEEP_LIGHTNESS_MIN
            ),
            0.0,
            1.0,
        )

    if INVERT_LIGHTNESS_ALPHA:
        alpha_fraction = 1.0 - lightness_fraction
    else:
        alpha_fraction = lightness_fraction

    return (
        SWEEP_ALPHA_MIN
        + alpha_fraction
        * (
            SWEEP_ALPHA_MAX
            - SWEEP_ALPHA_MIN
        )
    )


# ============================================================
# DETERMINE PLOT RANGE
# ============================================================

all_a_values = np.concatenate(
    [
        df_ab["a"].to_numpy(dtype=float),
        df_rosch[
            ROSCH_A_COLUMN
        ].to_numpy(dtype=float),
    ]
)

all_b_values = np.concatenate(
    [
        df_ab["b"].to_numpy(dtype=float),
        df_rosch[
            ROSCH_B_COLUMN
        ].to_numpy(dtype=float),
    ]
)

plot_radius = 1.08 * max(
    np.nanmax(np.abs(all_a_values)),
    np.nanmax(np.abs(all_b_values)),
    100.0,
)

chroma_ring_step = 20.0

chroma_levels = np.arange(
    chroma_ring_step,
    plot_radius + chroma_ring_step,
    chroma_ring_step,
)


# ============================================================
# PLOT
# ============================================================

cm_to_inches = 1.0 / 2.54

# Disable constrained layout for this figure because the external
# legend spacing is controlled manually with subplots_adjust.
with mpl.rc_context({
    "figure.constrained_layout.use": True,
}):

    fig, ax = plt.subplots(
        figsize=(
            fig_width_cm * cm_to_inches,
            fig_height_cm * cm_to_inches,
        ),
        constrained_layout=False,
    )

# ------------------------------------------------------------
# Hue-angle spokes
# ------------------------------------------------------------

hue_angles = np.arange(
    0.0,
    360.0,
    30.0,
)

for hue_deg in hue_angles:

    radii = np.array(
        [0.0, plot_radius],
        dtype=float,
    )

    ax.plot(
        radii * np.cos(
            np.deg2rad(hue_deg)
        ),
        radii * np.sin(
            np.deg2rad(hue_deg)
        ),
        color="gray",
        linestyle="--",
        linewidth=0.5,
        alpha=0.45,
        zorder=0,
    )

    label_radius = (
        0.96 * plot_radius
    )

    ax.text(
        label_radius
        * np.cos(
            np.deg2rad(hue_deg + 8)
        ),
        label_radius
        * np.sin(
            np.deg2rad(hue_deg + 8)
        ),
        f"{hue_deg:.0f}°",
        ha="center",
        va="center",
        color="dimgray",
    )


# ------------------------------------------------------------
# Chroma rings
# ------------------------------------------------------------

for chroma in chroma_levels[:-2]:

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


# ------------------------------------------------------------
# Sweep results
# ------------------------------------------------------------

if PLOT_SWEEP_RESULTS:

    for design_wavelength_nm in design_wavelengths:

        subset = df_sweep_plot.loc[
            np.isclose(
                df_sweep_plot[
                    "design_peak_wavelength_nm"
                ],
                float(design_wavelength_nm),
            )
        ].copy()

        if subset.empty:
            continue

        marker = marker_dict[
            design_wavelength_nm
        ]

        wavelength_colour = (
            wavelength_colour_dict[
                design_wavelength_nm
            ]
        )

        # Plot individually because both size and alpha vary.
        for _, row in subset.iterrows():

            if SWEEP_COLOUR_MODE == "wavelength":

                marker_colour = wavelength_colour

            elif SWEEP_COLOUR_MODE == "hex":

                marker_colour = str(
                    row[SWEEP_HEX_COLUMN]
                )

            else:
                raise RuntimeError(
                    "Unexpected SWEEP_COLOUR_MODE."
                )

            ax.scatter(
                [float(row["a"])],
                [float(row["b"])],
                facecolors=[marker_colour],
                s=float(row["marker_size"]),
                marker=marker,
                alpha=float(row["marker_alpha"]),
                edgecolors=(
                    SWEEP_EDGE_COLOR
                    if marker != "x"
                    else marker_colour
                ),
                linewidths=(
                    SWEEP_EDGE_WIDTH
                    if marker != "x"
                    else 0.7
                ),
                zorder=3,
            )

# ------------------------------------------------------------
# Highlight the selected minimum-d_opt cases
# ------------------------------------------------------------

if HIGHLIGHT_BEST_CASES:

    for _, best_row in df_best_ab.iterrows():

        wavelength_nm = float(
            best_row["design_peak_wavelength_nm"]
        )

        best_a = float(best_row["a"])
        best_b = float(best_row["b"])

        # Use the same eta_L -> alpha mapping as the sweep markers.
        if SWEEP_ALPHA_MODE == "lightness":
            best_alpha = calculate_lightness_alpha(
                float(best_row["eta_L"])
            )
        elif SWEEP_ALPHA_MODE == "fixed":
            best_alpha = SWEEP_ALPHA_FIXED
        elif SWEEP_ALPHA_MODE == "chroma":
            # If desired, implement the corresponding chroma-based
            # mapping here.
            best_alpha = 1.0
        else:
            best_alpha = 1.0

        best_marker = marker_dict.get(
            wavelength_nm,
            COMMON_SWEEP_MARKER,
        )

        # best_colour = wavelength_colour_dict.get(
        #     wavelength_nm,
        #     "black",
        # )
        if SWEEP_COLOUR_MODE == "wavelength":

            best_colour = wavelength_colour_dict.get(
                wavelength_nm,
                "black",
            )

        else:

            if SWEEP_HEX_COLUMN not in best_row.index:
                raise KeyError(
                    f"{SWEEP_HEX_COLUMN!r} is missing from "
                    "df_best_cases."
                )

            best_colour = str(
                best_row[SWEEP_HEX_COLUMN]
            )

        # White halo, ensuring visibility against the background.
        ax.scatter(
            [best_a],
            [best_b],
            s=BEST_CASE_OUTER_SIZE,
            marker=best_marker,
            facecolors="none",
            edgecolors=BEST_CASE_HALO_COLOR,
            linewidths=4.0,
            zorder=8,
        )

        # Filled wavelength-coloured marker with black outline.
        ax.scatter(
            [best_a],
            [best_b],
            s=BEST_CASE_INNER_SIZE,
            marker=best_marker,
            facecolors=[best_colour],
            edgecolors=BEST_CASE_EDGE_COLOR,
            linewidths=BEST_CASE_LINEWIDTH,
            alpha=best_alpha,
            zorder=9,
        )

        # Central star identifies the final d_opt selection.
        # Offset the star slightly inside asymmetric markers.
        star_dx_points, star_dy_points = (
            BEST_STAR_OFFSET_POINTS.get(
                best_marker,
                BEST_STAR_DEFAULT_OFFSET_POINTS,
            )
        )

        star_offset_transform = (
            ax.transData
            + ScaledTranslation(
                star_dx_points / 72.0,
                star_dy_points / 72.0,
                fig.dpi_scale_trans,
            )
        )

        ax.scatter(
            [best_a],
            [best_b],
            s=BEST_CASE_STAR_SIZE,
            marker="*",
            facecolors="white",
            edgecolors="black",
            linewidths=0.45,
            transform=star_offset_transform,
            clip_on=False,
            zorder=10,
        )

        if ANNOTATE_BEST_CASES:

            annotation_lines = [
                f"{wavelength_nm:.0f} nm",
                f"d_opt={float(best_row['d_opt']):.3f}",
            ]

            if "eta_C" in best_row.index:
                annotation_lines.append(
                    f"ηC={100 * float(best_row['eta_C']):.0f}%"
                )

            if "eta_L" in best_row.index:
                annotation_lines.append(
                    f"ηL={100 * float(best_row['eta_L']):.0f}%"
                )

            ax.annotate(
                "\n".join(annotation_lines),
                xy=(best_a, best_b),
                xytext=(5, 5),
                textcoords="offset points",
                ha="left",
                va="bottom",
                fontsize=6,
                bbox={
                    "boxstyle": "round,pad=0.2",
                    "facecolor": "white",
                    "edgecolor": "black",
                    "alpha": 0.85,
                },
                zorder=11,
            )


# ------------------------------------------------------------
# Smoothed Rösch–MacAdam maximum-chroma locus
# ------------------------------------------------------------

ax.plot(
    df_rosch[ROSCH_A_COLUMN],
    df_rosch[ROSCH_B_COLUMN],
    color="black",
    linewidth=1.0,
    alpha=0.65,
    zorder=4,
)

ax.scatter(
    df_rosch_sampled[ROSCH_A_COLUMN],
    df_rosch_sampled[ROSCH_B_COLUMN],

    # Individual RGBA colour and opacity for every marker.
    facecolors=rosch_marker_rgba,

    s=ROSCH_MARKER_SIZE,
    marker=ROSCH_MARKER,

    # Keep the marker edges fully visible.
    edgecolors=ROSCH_EDGE_COLOR,
    linewidths=ROSCH_EDGE_WIDTH,

    # Do not provide alpha= here because alpha is already included
    # individually in rosch_marker_rgba.
    zorder=6,
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
    -1.10 * plot_radius,
    1.10 * plot_radius,
)

ax.set_ylim(
    -1.10 * plot_radius,
    1.10 * plot_radius,
)

ax.set_aspect("equal")

ax.set_xlabel(
    "a*",
    ha="left",
)

ax.set_ylabel(
    "b*",
    ha="left",
    rotation=0,
)

ax.xaxis.set_label_coords(
    0.94, # 0.0
    -0.03, # -0.03
)

ax.yaxis.set_label_coords(
    -0.1, # -0.15
    0.94, # 0.475
)

ax.grid(
    False,
    which="both",
    axis="both",
)


# ============================================================
# LEGEND
# ============================================================

# sweep_handles = []

# if PLOT_SWEEP_RESULTS:

#     for wavelength_nm in design_wavelengths:

#         if SWEEP_COLOUR_MODE == "wavelength":
#             legend_facecolour = (
#                 wavelength_colour_dict[
#                     wavelength_nm
#                 ]
#             )
#         else:
#             # Actual hex colours vary point-by-point, so a neutral
#             # marker is used in the wavelength legend.
#             legend_facecolour = "white"

#         sweep_handles.append(
#             Line2D(
#                 [0],
#                 [0],
#                 marker=marker_dict[
#                     wavelength_nm
#                 ],
#                 linestyle="None",
#                 color="none",
#                 markerfacecolor=legend_facecolour,
#                 markeredgecolor="black",
#                 markeredgewidth=0.5,
#                 markersize=6,
#                 label=f"{wavelength_nm:.0f} nm",
#             )
#         )

sweep_handles = []

if PLOT_SWEEP_RESULTS:

    for wavelength_nm in legend_wavelengths:

        if SWEEP_COLOUR_MODE == "wavelength":
            legend_facecolour = wavelength_colour_dict.get(
                wavelength_nm,
                "white",
            )
        else:
            # Actual hex colours vary point-by-point, so use
            # a neutral marker in the wavelength legend.
            legend_facecolour = "white"

        sweep_handles.append(
            Line2D(
                [0],
                [0],
                marker=default_marker_dict[wavelength_nm],
                linestyle="None",
                color="none",
                markerfacecolor=legend_facecolour,
                markeredgecolor="black",
                markeredgewidth=0.5,
                markersize=6,
                label=f"{wavelength_nm:.0f}",
            )
        )


rosch_handle = Line2D(
    [0],
    [0],
    marker=ROSCH_MARKER,
    linestyle="-",
    color="black",
    markerfacecolor="white",
    markeredgecolor=ROSCH_EDGE_COLOR,
    markersize=6,
    label="Max-chroma locus",
)

best_case_handle = Line2D(
    [0],
    [0],
    marker="*",
    linestyle="None",
    color="black",
    markerfacecolor="white",
    markeredgecolor="black",
    markersize=7,
    label="Minimum-$d_{opt}$ case",
)

# Uncomment to show the legend.
# ============================================================
# COMPACT LEGEND BELOW THE PLOT
# ============================================================

# ============================================================
# COMPACT LEGENDS BELOW THE GRAPH
# ============================================================

# # Ensure short wavelength labels.
# for handle, wavelength_nm in zip(
#     sweep_handles,
#     design_wavelengths,
# ):
#     handle.set_label(
#         f"{wavelength_nm:.0f}"
#     )

rosch_handle.set_label("Max locus")

reference_handles = [rosch_handle]

if HIGHLIGHT_BEST_CASES:
    best_case_handle.set_label(
        r"Best $d_{\mathrm{opt}}$"
    )
    reference_handles.append(
        best_case_handle
    )

# ------------------------------------------------------------
# Reserve a fixed area below the axes
# ------------------------------------------------------------

# fig.subplots_adjust(
#     left=0.18,
#     right=0.97,
#     top=0.98,
#     bottom=0.27,
# )

# ------------------------------------------------------------
# Main wavelength legend: four columns
# ------------------------------------------------------------

wavelength_legend = fig.legend(
    handles=sweep_handles,
    loc="upper left",
    bbox_to_anchor=(0.15, 0.315),
    bbox_transform=fig.transFigure,
    ncol=3,
    frameon=True,
    framealpha=0.9,
    # fontsize=9,
    title=r"Design $\lambda$ [nm]",
    title_fontsize=9,
    handlelength=0.8,
    handletextpad=0.25,
    columnspacing=0.65,
    labelspacing=0.25,
    borderaxespad=0.0,
    markerscale=0.85,
)

# ------------------------------------------------------------
# Compact reference-symbol key
# ------------------------------------------------------------


reference_legend = fig.legend(
    handles=reference_handles,
    loc="upper left",
    bbox_to_anchor=(0.18, 0.92),
    bbox_transform=fig.transFigure,
    # ncol=len(reference_handles),
    ncol=1,
    frameon=True,
    framealpha=0.4,
    # fontsize=9,
    handlelength=1.0,
    handletextpad=0.30,
    columnspacing=0.85,
    labelspacing=0.20,
    borderaxespad=0.0,
    markerscale=0.85
)

# Reserve vertical space explicitly.
# This works because constrained layout was disabled when the figure
# was created.
# fig.subplots_adjust(
#     left=0.18,
#     right=0.97,
#     top=0.98,
#     bottom=0.25,
# )

# Comment off to remove extra legend for marker opacity.
# if SWEEP_ALPHA_MODE == "lightness":

#     alpha_example_values = [
#         0.0,
#         0.5,
#         1.0,
#     ]

#     alpha_example_labels = [
#         "low L*",
#         "medium L*",
#         "high L*",
#     ]

#     if INVERT_LIGHTNESS_ALPHA:
#         alpha_example_labels = [
#             "high L*",
#             "medium L*",
#             "low L*",
#         ]

#     alpha_handles = [
#         Line2D(
#             [0],
#             [0],
#             marker="o",
#             linestyle="None",
#             markerfacecolor="gray",
#             markeredgecolor="black",
#             alpha=(
#                 SWEEP_ALPHA_MIN
#                 + fraction
#                 * (
#                     SWEEP_ALPHA_MAX
#                     - SWEEP_ALPHA_MIN
#                 )
#             ),
#             markersize=6,
#             label=label,
#         )
#         for fraction, label in zip(
#             alpha_example_values,
#             alpha_example_labels,
#         )
#     ]

#     alpha_legend = ax.legend(
#         handles=alpha_handles,
#         title="Marker opacity",
#         loc="upper right",
#         bbox_to_anchor=(1.0, 1.0),
#         frameon=True,
#         fontsize=8,
#         title_fontsize=8,
#         framealpha=0.4,
#         borderpad=0.25,      # default ~0.4
#         handletextpad=0.3,   # default ~0.8
#         labelspacing=0.25,   # default ~0.5
#         borderaxespad=0.1,
#     )

#     ax.add_artist(alpha_legend)

# ============================================================
# VERTICAL MARKER-OPACITY SCALE FOR L*
# ============================================================

if SWEEP_ALPHA_MODE == "lightness":

    # Position of the opacity scale in axes coordinates:
    # [left, bottom, width, height]
    OPACITY_SCALE_BOUNDS = [
        0.82,
        0.82,
        0.035,
        0.1,
    ]

    OPACITY_SCALE_COLOUR = "gray"
    # OPACITY_SCALE_FONT_SIZE = 8
    OPACITY_SCALE_DECIMALS = 0

    # Create a small inset axis inside the main plot.
    ax_opacity = ax.inset_axes(
        OPACITY_SCALE_BOUNDS
    )

    n_alpha_levels = 256

    lightness_fraction = np.linspace(
        0.0,
        1.0,
        n_alpha_levels,
    )

    # Reproduce exactly the mapping used for the sweep markers.
    if INVERT_LIGHTNESS_ALPHA:

        opacity_fraction = (
            1.0 - lightness_fraction
        )

    else:

        opacity_fraction = (
            lightness_fraction
        )

    opacity_values = (
        SWEEP_ALPHA_MIN
        + opacity_fraction
        * (
            SWEEP_ALPHA_MAX
            - SWEEP_ALPHA_MIN
        )
    )

    # Convert the selected base colour into RGB.
    opacity_rgb = np.asarray(
        mpl.colors.to_rgb(
            OPACITY_SCALE_COLOUR
        ),
        dtype=float,
    )

    # Create an RGBA image whose alpha varies vertically.
    opacity_rgba = np.zeros(
        (
            n_alpha_levels,
            1,
            4,
        ),
        dtype=float,
    )

    opacity_rgba[:, :, 0] = opacity_rgb[0]
    opacity_rgba[:, :, 1] = opacity_rgb[1]
    opacity_rgba[:, :, 2] = opacity_rgb[2]
    opacity_rgba[:, :, 3] = opacity_values[:, None]

    # Use a white background so the opacity remains visible when
    # the complete figure is saved with transparency.
    ax_opacity.set_facecolor("white")

    ax_opacity.imshow(
        opacity_rgba,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=[
            0.0,
            1.0,
            SWEEP_LIGHTNESS_MIN * 1e2,
            SWEEP_LIGHTNESS_MAX * 1e2,
        ],
    )

    # Label only the actual minimum and maximum L* values.
    ax_opacity.set_yticks(
        [
            SWEEP_LIGHTNESS_MIN * 1e2,
            SWEEP_LIGHTNESS_MAX * 1e2,
        ]
    )

    ax_opacity.set_yticklabels(
        [
            (
                f"{SWEEP_LIGHTNESS_MIN*1e2:.{OPACITY_SCALE_DECIMALS}f}%"
            ),
            (
                f"{SWEEP_LIGHTNESS_MAX*1e2:.{OPACITY_SCALE_DECIMALS}f}%"
            ),
        ],
        # fontsize=OPACITY_SCALE_FONT_SIZE,
    )

    ax_opacity.yaxis.tick_right()
    ax_opacity.yaxis.set_label_position(
        "right"
    )

    ax_opacity.set_ylabel(
        "$\\eta$"r"$_{L^*}$",
        rotation=0,
        ha="left",
        va="bottom",
        # fontsize=OPACITY_SCALE_FONT_SIZE,
        labelpad=2,
    )

    ax_opacity.yaxis.set_label_coords(2.5, 0.3)

    ax_opacity.set_xticks([])

    ax_opacity.tick_params(
        axis="y",
        which="major",
        length=2.5,
        width=0.7,
        pad=2,
    )

    # Black frame around the opacity scale.
    for spine in ax_opacity.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.7)

    ax_opacity.set_title(
        "Opacity",
        # fontsize=OPACITY_SCALE_FONT_SIZE,
        pad=7,
        ha="left",
    )


png_path = os.path.join(
    path,
    "CIELAB_ab_plane_RoschMacadam_max_chroma_2.png",
)

pdf_path = os.path.join(
    path,
    "CIELAB_ab_plane_RoschMacadam_max_chroma_2.pdf",
)

fig.savefig(
    png_path,
    dpi=600,
    transparent=True,
    bbox_inches=None,
    pad_inches=0.0,
)

fig.savefig(
    pdf_path,
    transparent=True,
    bbox_inches=None,
    pad_inches=0.0,
)

plt.show()


# ============================================================
# OPTIONAL DIAGNOSTIC OUTPUT
# ============================================================

print("\nNumber of plotted points per wavelength:")

display(
    df_sweep_plot
    .groupby(
        "design_peak_wavelength_nm"
    )
    .size()
    .rename("number_of_plotted_points")
    .reset_index()
)


# Cell 1: Imports and reconstruction helpers

# In[20]:


# ============================================================
# FIGURE 2B/C — RECONSTRUCTION HELPERS
# ============================================================

import numpy as np
import pandas as pd

from bragg_onion.geometry import (
    ExtinctionModifier,
    ExtraOuterShellSpec,
)
from bragg_onion.materials import ConstantDispersion
from bragg_onion.sweep import (
    run_bragg_onion_sweep,
    make_plotting_aliases,
)
from bragg_onion._notebook_tools.scattering_extract import (
    extract_case_arrays,
)


def get_row_value(row, *column_names, default=None):
    """
    Return the first available, nonmissing value from a dataframe row.
    """
    for column in column_names:
        if column in row.index and pd.notna(row[column]):
            return row[column]

    if default is not None:
        return default

    raise KeyError(
        f"None of the requested columns are available: {column_names}"
    )


def build_absorber_objects(row):
    """
    Reconstruct the extinction modifier or additional outer shell
    for one selected sweep case.
    """
    location = str(row["absorber_location"])
    k = float(get_row_value(row, "k", default=0.0))

    extinction_modifiers = None
    extra_outer_shell = None

    if location == "none":
        pass

    elif location == "core":
        extinction_modifiers = [
            ExtinctionModifier(
                target="core",
                mode="substitute",
                profile=k,
                name=f"core_k_{k:.6f}",
            )
        ]

    elif location == "low_index":
        extinction_modifiers = [
            ExtinctionModifier(
                target="material_B",
                mode="substitute",
                profile=k,
                name=f"low_index_k_{k:.6f}",
            )
        ]

    elif location == "all_layers":
        extinction_modifiers = [
            ExtinctionModifier(
                target="both_materials",
                mode="substitute",
                profile=k,
                name=f"all_layers_k_{k:.6f}",
            )
        ]

    elif location == "outer_shell":
        shell_thickness_nm = float(
            get_row_value(
                row,
                "outer_shell_thickness_nm",
            )
        )

        outer_shell_n = float(
            get_row_value(
                row,
                "n_outer_shell",
                default=1.0,
            )
        )

        absorber_material = ConstantDispersion.from_nk(
            "outer_absorber",
            n=outer_shell_n,
            k=k,
        )

        extra_outer_shell = ExtraOuterShellSpec(
            thickness_m=shell_thickness_nm * 1e-9,
            material=absorber_material,
            name=(
                f"outer_shell_{shell_thickness_nm:.3f}nm_"
                f"k_{k:.6f}"
            ),
        )

    else:
        raise ValueError(
            f"Unknown absorber location: {location!r}"
        )

    return extinction_modifiers, extra_outer_shell


def build_single_case_parameter_grid(row):
    """
    Reconstruct exactly one geometry from a selected dataframe row.

    The geometry may be defined by either:
      1. an explicit number of layers, or
      2. a requested particle diameter.

    Exactly one of ``n_layers`` and ``diameter_m`` is added to the
    returned parameter grid.
    """

    # Use the stored value in metres when available.
    # Otherwise reconstruct it from the design wavelength in nanometres.
    peak_wavelength_m = float(
        get_row_value(
            row,
            "peak_wavelength_m",
            default=(
                float(
                    row["design_peak_wavelength_nm"]
                )
                * 1e-9
            ),
        )
    )

    core_thickness_factor = float(
        get_row_value(
            row,
            "core_thickness_factor",
            "core_thickness_factor_requested",
            default=1.0,
        )
    )

    outer_layer = str(
        get_row_value(
            row,
            "outer_layer",
            default="A",
        )
    )

    parameter_grid = {
        "peak_wavelength_m": [
            peak_wavelength_m
        ],
        "outer_layer": [
            outer_layer
        ],
        "core_thickness_factor": [
            core_thickness_factor
        ],
    }

    # Prefer an explicitly stored layer count.
    if (
        "n_layers" in row.index
        and pd.notna(row["n_layers"])
    ):
        parameter_grid["n_layers"] = [
            int(row["n_layers"])
        ]

    # Otherwise use the original requested diameter in metres.
    elif (
        "diameter_m" in row.index
        and pd.notna(row["diameter_m"])
    ):
        parameter_grid["diameter_m"] = [
            float(row["diameter_m"])
        ]

    # Some sweep outputs store only the requested diameter in nm.
    elif (
        "requested_diameter_nm" in row.index
        and pd.notna(row["requested_diameter_nm"])
    ):
        parameter_grid["diameter_m"] = [
            float(
                row["requested_diameter_nm"]
            )
            * 1e-9
        ]

    else:
        raise KeyError(
            "Could not reconstruct the selected geometry. "
            "The row contains neither 'n_layers', 'diameter_m', "
            "nor 'requested_diameter_nm'."
        )

    # Final consistency check required by the geometry builder.
    has_n_layers = (
        "n_layers" in parameter_grid
    )

    has_diameter = (
        "diameter_m" in parameter_grid
    )

    if has_n_layers == has_diameter:
        raise ValueError(
            "The reconstructed parameter grid must contain "
            "exactly one of 'n_layers' and 'diameter_m'. "
            f"Constructed grid: {parameter_grid}"
        )

    return parameter_grid


# Cell 2: Validate the selected cases

# In[21]:


# ============================================================
# VALIDATE df_best_cases
# ============================================================

required_best_columns = {
    "design_peak_wavelength_nm",
    "n_A",
    "n_B",
    "absorber_location",
    "eta_C",
    "eta_L",
    "d_opt",
}

missing_columns = (
    required_best_columns
    - set(df_best_cases.columns)
)

if missing_columns:
    raise KeyError(
        "df_best_cases is missing required columns: "
        f"{sorted(missing_columns)}"
    )

best_case_counts = (
    df_best_cases
    .groupby("design_peak_wavelength_nm")
    .size()
)

if not (best_case_counts == 1).all():
    raise ValueError(
        "df_best_cases must contain exactly one selected case "
        "per target wavelength.\n"
        f"{best_case_counts}"
    )

df_best_cases_rerun = (
    df_best_cases
    .sort_values("design_peak_wavelength_nm")
    .reset_index(drop=True)
    .copy()
)

summary_columns = [
    "design_peak_wavelength_nm",
    "absorber_location",
    "d_opt",
    "eta_C",
    "eta_L",
    "n_A",
    "n_B",
    "k",
    "n_layers",
    "diameter_nm",
    "core_thickness_factor",
    "outer_layer",
    "hex",
    "hex_display",
]

summary_columns = [
    column
    for column in summary_columns
    if column in df_best_cases_rerun.columns
]

display(
    df_best_cases_rerun[
        summary_columns
    ]
)


# Cell 3: Rerun all selected cases in both collection directions

# In[22]:


# # ============================================================
# # RERUN THE SELECTED minimum-d_opt CASES
# # BACKWARD AND FORWARD COLLECTION, NA = 1
# # ============================================================

# analysis_wavelengths_nm = np.arange(
#     380.0,
#     781.0,
#     1.0,
# )

# analysis_theta_deg = np.linspace(
#     0.0,
#     180.0,
#     721,
# )

# ANALYSIS_COLLECTION_NA = 1.0

# best_cases_backward = {}
# best_cases_forward = {}


# for _, row_case in df_best_cases_rerun.iterrows():

#     target_wavelength_nm = int(
#         round(
#             float(
#                 row_case[
#                     "design_peak_wavelength_nm"
#                 ]
#             )
#         )
#     )

#     key = target_wavelength_nm

#     if key in best_cases_backward:
#         raise ValueError(
#             f"Duplicate selected target wavelength: {key} nm"
#         )

#     absorber_location = str(
#         row_case["absorber_location"]
#     )

#     print(
#         f"Rerunning {target_wavelength_nm} nm design: "
#         f"location={absorber_location}, "
#         f"d_opt={float(row_case['d_opt']):.4f}"
#     )

#     # Base Bragg materials remain nonabsorbing.
#     # The selected absorption is introduced separately.
#     material_a = ConstantDispersion.from_nk(
#         "A",
#         n=float(row_case["n_A"]),
#         k=0.0,
#     )

#     material_b = ConstantDispersion.from_nk(
#         "B",
#         n=float(row_case["n_B"]),
#         k=0.0,
#     )

#     extinction_modifiers, extra_outer_shell = (
#         build_absorber_objects(row_case)
#     )

#     parameter_grid = (
#         build_single_case_parameter_grid(row_case)
#     )

#     common_kwargs = {
#         "geometry_mode": "peak_wavelength",
#         "parameter_grid": parameter_grid,
#         "material_a": material_a,
#         "material_b": material_b,
#         "n_medium": 1.0,
#         "wavelengths_m": (
#             analysis_wavelengths_nm * 1e-9
#         ),
#         "theta_rad": np.deg2rad(
#             analysis_theta_deg
#         ),
#         "collection_na": ANALYSIS_COLLECTION_NA,
#         "integration_quantity_for_colour": (
#             "q_collected_geom"
#         ),
#         "colour_wavelength_min_nm": None,
#         "colour_wavelength_max_nm": None,
#         "colour_normalize_input": False,
#         "colour_normalization": "max",
#         "extinction_modifiers": (
#             extinction_modifiers
#         ),
#         "extra_outer_shell": extra_outer_shell,
#         "store_full_results": True,
#         "progress": False,
#     }

#     sweep_backward = run_bragg_onion_sweep(
#         collection_direction="backward",
#         **common_kwargs,
#     )

#     sweep_forward = run_bragg_onion_sweep(
#         collection_direction="forward",
#         **common_kwargs,
#     )

#     if len(sweep_backward.case_results) != 1:
#         raise ValueError(
#             f"Backward rerun for {key} nm produced "
#             f"{len(sweep_backward.case_results)} cases."
#         )

#     if len(sweep_forward.case_results) != 1:
#         raise ValueError(
#             f"Forward rerun for {key} nm produced "
#             f"{len(sweep_forward.case_results)} cases."
#         )

#     backward_case_result = (
#         sweep_backward.case_results[0]
#     )

#     forward_case_result = (
#         sweep_forward.case_results[0]
#     )

#     backward_data = extract_case_arrays(
#         sweep_backward
#     )

#     forward_data = extract_case_arrays(
#         sweep_forward
#     )

#     best_cases_backward[key] = {
#         "row": row_case.copy(),
#         "sweep_case": sweep_backward,
#         "case_result": backward_case_result,
#         "data": backward_data,
#     }

#     best_cases_forward[key] = {
#         "row": row_case.copy(),
#         "sweep_case": sweep_forward,
#         "case_result": forward_case_result,
#         "data": forward_data,
#     }

#     print(
#         f"  finished backward and forward reruns "
#         f"for {target_wavelength_nm} nm"
#     )


# print(
#     f"\nStored {len(best_cases_backward)} backward and "
#     f"{len(best_cases_forward)} forward detailed cases."
# )

# print(
#     "Target wavelengths:",
#     sorted(best_cases_backward),
# )


# In[23]:


# ============================================================
# RERUN OR LOAD THE SELECTED minimum-d_opt CASES
# BACKWARD AND FORWARD COLLECTION, NA = 1
#
# Each target-wavelength case is saved independently.
# Existing compatible files are loaded and calculation is skipped.
# ============================================================

from pathlib import Path

import hashlib
import json
import os
import pickle

import numpy as np
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

RERUN_SAVE_DIRECTORY = Path(
    "/mnt/c/Users/SchwarzN/"
    "OneDrive - Université de Fribourg/"
    "Institution/P1_BraggSpericalPigments/"
    "Simulation/MIE_Computation/"
    "Scattnlay_BraggOnion/outputs/"
    "run_secondary_sweep_02/rerun_best_cases"
)

RERUN_SAVE_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)

analysis_wavelengths_nm = np.arange(
    300.0,
    851.0, # 781
    1.0,
)

analysis_theta_deg = np.linspace(
    0.0,
    180.0,
    721,
)

ANALYSIS_COLLECTION_NA = 1.0

# Change this string manually if the saved-data structure or
# calculation procedure changes in the future.
CACHE_VERSION = "best_cases_v3"

# True:
#   ignore compatible saved files and recalculate every case.
#
# False:
#   load compatible saved files when available.
FORCE_RECALCULATION = False

# If a checkpoint cannot be loaded or is incompatible, calculate
# that case again and replace the checkpoint.
RECALCULATE_INVALID_CACHE = True


# ============================================================
# CACHE HELPERS
# ============================================================

def make_json_safe(value):
    """
    Convert common NumPy and pandas values into JSON-safe values.
    """
    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if pd.isna(value):
        return None

    return value


def build_case_signature(
    row_case,
    wavelengths_nm,
    theta_deg,
    collection_na,
):
    """
    Create a reproducible signature for the selected physical case
    and numerical grids.

    A saved result is loaded only when this signature agrees with
    the currently requested calculation.
    """

    relevant_columns = [
        "design_peak_wavelength_nm",
        "peak_wavelength_m",
        "n_A",
        "n_B",
        "k",
        "absorber_location",
        "n_layers",
        "outer_layer",
        "core_thickness_factor",
        "core_thickness_factor_requested",
        "outer_shell_thickness_nm",
        "n_outer_shell",
        "diameter_m",
        "requested_diameter_nm",
        "collection_na",
        "collection_NA",
        "d_opt",
    ]

    row_parameters = {}

    for column in relevant_columns:
        if column in row_case.index:
            row_parameters[column] = make_json_safe(
                row_case[column]
            )

    signature_content = {
        "cache_version": CACHE_VERSION,
        "row_parameters": row_parameters,
        "wavelengths_nm": np.asarray(
            wavelengths_nm,
            dtype=float,
        ).tolist(),
        "theta_deg": np.asarray(
            theta_deg,
            dtype=float,
        ).tolist(),
        "collection_na": float(
            collection_na
        ),
        "n_medium": 1.0,
        "integration_quantity_for_colour": (
            "q_collected_geom"
        ),
        "colour_normalize_input": False,
        "colour_normalization": "max",
        "store_full_results": True,
    }

    signature_json = json.dumps(
        signature_content,
        sort_keys=True,
        separators=(",", ":"),
    )

    signature_hash = hashlib.sha256(
        signature_json.encode("utf-8")
    ).hexdigest()

    return signature_hash, signature_content


def checkpoint_path_for(
    target_wavelength_nm,
):
    """
    Return the checkpoint path for one selected design wavelength.
    """
    return (
        RERUN_SAVE_DIRECTORY
        / (
            f"best_case_"
            f"{int(target_wavelength_nm):04d}nm.pkl"
        )
    )


def save_checkpoint_atomic(
    checkpoint_path,
    checkpoint_data,
):
    """
    Save through a temporary file and rename atomically.

    This reduces the risk of leaving a partially written checkpoint
    if the notebook or Python process is interrupted.
    """

    temporary_path = checkpoint_path.with_suffix(
        checkpoint_path.suffix + ".tmp"
    )

    with temporary_path.open("wb") as file:
        pickle.dump(
            checkpoint_data,
            file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

        file.flush()
        os.fsync(file.fileno())

    os.replace(
        temporary_path,
        checkpoint_path,
    )


def load_checkpoint(
    checkpoint_path,
    expected_signature,
):
    """
    Load and validate one checkpoint.

    Returns
    -------
    checkpoint_data
        Loaded checkpoint when valid.

    None
        Returned when the checkpoint is incompatible or unreadable.
    """

    try:
        with checkpoint_path.open("rb") as file:
            checkpoint_data = pickle.load(
                file
            )

    except Exception as error:
        print(
            f"  Could not load checkpoint "
            f"{checkpoint_path.name}: {error}"
        )

        if not RECALCULATE_INVALID_CACHE:
            raise

        return None

    required_keys = {
        "cache_version",
        "signature",
        "target_wavelength_nm",
        "backward",
        "forward",
    }

    missing_keys = (
        required_keys
        - set(checkpoint_data)
    )

    if missing_keys:
        print(
            f"  Checkpoint {checkpoint_path.name} "
            f"is missing keys: {sorted(missing_keys)}"
        )
        return None

    if (
        checkpoint_data["cache_version"]
        != CACHE_VERSION
    ):
        print(
            f"  Checkpoint {checkpoint_path.name} "
            "has a different cache version."
        )
        return None

    if (
        checkpoint_data["signature"]
        != expected_signature
    ):
        print(
            f"  Checkpoint {checkpoint_path.name} "
            "does not match the current case or grids."
        )
        return None

    for direction in [
        "backward",
        "forward",
    ]:
        required_direction_keys = {
            "row",
            "sweep_case",
            "case_result",
            "data",
        }

        missing_direction_keys = (
            required_direction_keys
            - set(
                checkpoint_data[direction]
            )
        )

        if missing_direction_keys:
            print(
                f"  Checkpoint {checkpoint_path.name}, "
                f"{direction}, is missing keys: "
                f"{sorted(missing_direction_keys)}"
            )
            return None

    return checkpoint_data


# ============================================================
# RUN OR LOAD CASES
# ============================================================

best_cases_backward = {}
best_cases_forward = {}

number_loaded = 0
number_calculated = 0


for _, row_case in (
    df_best_cases_rerun.iterrows()
):

    target_wavelength_nm = int(
        round(
            float(
                row_case[
                    "design_peak_wavelength_nm"
                ]
            )
        )
    )

    key = target_wavelength_nm

    if (
        key in best_cases_backward
        or key in best_cases_forward
    ):
        raise ValueError(
            f"Duplicate selected target wavelength: "
            f"{key} nm"
        )

    absorber_location = str(
        row_case["absorber_location"]
    )

    checkpoint_path = (
        checkpoint_path_for(
            target_wavelength_nm
        )
    )

    signature, signature_content = (
        build_case_signature(
            row_case=row_case,
            wavelengths_nm=(
                analysis_wavelengths_nm
            ),
            theta_deg=analysis_theta_deg,
            collection_na=(
                ANALYSIS_COLLECTION_NA
            ),
        )
    )


    # --------------------------------------------------------
    # Load an existing compatible checkpoint
    # --------------------------------------------------------

    checkpoint_data = None

    if (
        checkpoint_path.exists()
        and not FORCE_RECALCULATION
    ):
        print(
            f"Checking saved "
            f"{target_wavelength_nm} nm case..."
        )

        checkpoint_data = load_checkpoint(
            checkpoint_path,
            expected_signature=signature,
        )


    if checkpoint_data is not None:

        best_cases_backward[key] = (
            checkpoint_data["backward"]
        )

        best_cases_forward[key] = (
            checkpoint_data["forward"]
        )

        number_loaded += 1

        print(
            f"  loaded backward and forward "
            f"{target_wavelength_nm} nm results "
            f"from {checkpoint_path.name}"
        )

        continue


    # --------------------------------------------------------
    # Calculate a missing or incompatible checkpoint
    # --------------------------------------------------------

    print(
        f"Rerunning {target_wavelength_nm} nm design: "
        f"location={absorber_location}, "
        f"d_opt={float(row_case['d_opt']):.4f}"
    )

    # Base Bragg materials remain nonabsorbing.
    # Absorption is introduced separately according to placement.
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

    extinction_modifiers, extra_outer_shell = (
        build_absorber_objects(
            row_case
        )
    )

    parameter_grid = (
        build_single_case_parameter_grid(
            row_case
        )
    )

    common_kwargs = {
        "geometry_mode": "peak_wavelength",
        "parameter_grid": parameter_grid,
        "material_a": material_a,
        "material_b": material_b,
        "n_medium": 1.0,
        "wavelengths_m": (
            analysis_wavelengths_nm
            * 1e-9
        ),
        "theta_rad": np.deg2rad(
            analysis_theta_deg
        ),
        "collection_na": (
            ANALYSIS_COLLECTION_NA
        ),
        "integration_quantity_for_colour": (
            "q_collected_geom"
        ),
        "colour_wavelength_min_nm": None,
        "colour_wavelength_max_nm": None,
        "colour_normalize_input": False,
        "colour_normalization": "max",
        "extinction_modifiers": (
            extinction_modifiers
        ),
        "extra_outer_shell": (
            extra_outer_shell
        ),
        "store_full_results": True,
        "progress": False,
    }

    sweep_backward = (
        run_bragg_onion_sweep(
            collection_direction="backward",
            **common_kwargs,
        )
    )

    sweep_forward = (
        run_bragg_onion_sweep(
            collection_direction="forward",
            **common_kwargs,
        )
    )

    if len(
        sweep_backward.case_results
    ) != 1:
        raise ValueError(
            f"Backward rerun for {key} nm "
            f"produced "
            f"{len(sweep_backward.case_results)} "
            "cases."
        )

    if len(
        sweep_forward.case_results
    ) != 1:
        raise ValueError(
            f"Forward rerun for {key} nm "
            f"produced "
            f"{len(sweep_forward.case_results)} "
            "cases."
        )

    backward_case_result = (
        sweep_backward.case_results[0]
    )

    forward_case_result = (
        sweep_forward.case_results[0]
    )

    backward_data = extract_case_arrays(
        sweep_backward
    )

    forward_data = extract_case_arrays(
        sweep_forward
    )

    backward_record = {
        "row": row_case.copy(),
        "sweep_case": sweep_backward,
        "case_result": (
            backward_case_result
        ),
        "data": backward_data,
    }

    forward_record = {
        "row": row_case.copy(),
        "sweep_case": sweep_forward,
        "case_result": (
            forward_case_result
        ),
        "data": forward_data,
    }

    best_cases_backward[key] = (
        backward_record
    )

    best_cases_forward[key] = (
        forward_record
    )


    # --------------------------------------------------------
    # Save both directions in one checkpoint
    # --------------------------------------------------------

    checkpoint_data = {
        "cache_version": CACHE_VERSION,
        "signature": signature,
        "signature_content": (
            signature_content
        ),
        "target_wavelength_nm": (
            target_wavelength_nm
        ),
        "backward": backward_record,
        "forward": forward_record,
    }

    save_checkpoint_atomic(
        checkpoint_path,
        checkpoint_data,
    )

    number_calculated += 1

    print(
        f"  finished and saved backward and "
        f"forward reruns for "
        f"{target_wavelength_nm} nm"
    )

    print(
        f"  checkpoint: {checkpoint_path}"
    )


# ============================================================
# FINAL VALIDATION AND SUMMARY
# ============================================================

expected_keys = {
    int(
        round(
            float(value)
        )
    )
    for value in (
        df_best_cases_rerun[
            "design_peak_wavelength_nm"
        ]
    )
}

backward_keys = set(
    best_cases_backward
)

forward_keys = set(
    best_cases_forward
)

if backward_keys != expected_keys:
    raise RuntimeError(
        "The backward results do not contain all expected cases.\n"
        f"Expected: {sorted(expected_keys)}\n"
        f"Found: {sorted(backward_keys)}"
    )

if forward_keys != expected_keys:
    raise RuntimeError(
        "The forward results do not contain all expected cases.\n"
        f"Expected: {sorted(expected_keys)}\n"
        f"Found: {sorted(forward_keys)}"
    )

print("\nRerun summary")
print("-------------")
print(
    "Loaded from disk :",
    number_loaded,
)
print(
    "Newly calculated:",
    number_calculated,
)
print(
    "Total cases      :",
    len(best_cases_backward),
)
print(
    "Save directory   :",
    RERUN_SAVE_DIRECTORY,
)
print(
    "Target wavelengths:",
    sorted(best_cases_backward),
)


# In[24]:


# Plot for the first target wavelength case the refractive index profile including the core
print("\nPlotting refractive index profile for the first target wavelength case...")
print(best_cases_backward[sorted(best_cases_backward)[0]]["row"])


# In[25]:


# ============================================================
# IDENTIFY SUSPICIOUS WAVELENGTHS FROM ENERGY CONSERVATION
#
# Required:
#     best_cases_backward
#     best_cases_forward
#
# Output:
#     suspicious_points
#     df_suspicious_points
# ============================================================

import numpy as np
import pandas as pd


# Strict tolerances remain used for detecting problematic coarse points.
PASSIVITY_ABS_TOL = 1e-8
PASSIVITY_REL_TOL = 1e-5

# More permissive tolerances are used only for accepting locally
# recalculated replacement values in numerically demanding cases.
REPLACEMENT_PASSIVITY_ABS_TOL = 1e-6
REPLACEMENT_PASSIVITY_REL_TOL = 3e-2

def passive_solution_mask(
    scattering_result,
    *,
    absolute_tolerance=PASSIVITY_ABS_TOL,
    relative_tolerance=PASSIVITY_REL_TOL,
):
    """
    Return True at wavelength points satisfying passive-energy
    consistency conditions.

    Tested conditions:
      Qsca >= 0
      Qext >= 0
      Qabs >= 0
      Qsca <= Qext
      0 <= albedo <= 1

    Small numerical deviations are accepted within tolerance.
    """

    qsca = np.asarray(
        scattering_result.qsca,
        dtype=float,
    ).ravel()

    qext = np.asarray(
        scattering_result.qext,
        dtype=float,
    ).ravel()

    qabs = np.asarray(
        scattering_result.qabs,
        dtype=float,
    ).ravel()

    albedo = np.asarray(
        scattering_result.albedo,
        dtype=float,
    ).ravel()

    scale = np.maximum.reduce(
        [
            np.abs(qsca),
            np.abs(qext),
            np.ones_like(qsca),
        ]
    )

    tolerance = (
        absolute_tolerance
        + relative_tolerance * scale
    )

    finite = (
        np.isfinite(qsca)
        & np.isfinite(qext)
        & np.isfinite(qabs)
        & np.isfinite(albedo)
    )

    return (
        finite
        & (qsca >= -tolerance)
        & (qext >= -tolerance)
        & (qabs >= -tolerance)
        & (qsca <= qext + tolerance)
        & (albedo >= -relative_tolerance)
        & (albedo <= 1.0 + relative_tolerance)
    )


# ------------------------------------------------------------
# Validate case dictionaries
# ------------------------------------------------------------

if set(best_cases_backward) != set(best_cases_forward):
    raise ValueError(
        "Forward and backward case dictionaries have different keys."
    )


# ------------------------------------------------------------
# Find suspicious wavelength points
# ------------------------------------------------------------

suspicious_points = {}
suspicious_records = []

for target_wavelength_nm in sorted(
    best_cases_backward
):

    backward_case = best_cases_backward[
        target_wavelength_nm
    ]

    scattering_result = backward_case[
        "case_result"
    ].scattering_result

    wavelengths_nm = (
        np.asarray(
            scattering_result.wavelengths_m,
            dtype=float,
        ).ravel()
        * 1e9
    )

    qsca = np.asarray(
        scattering_result.qsca,
        dtype=float,
    ).ravel()

    qext = np.asarray(
        scattering_result.qext,
        dtype=float,
    ).ravel()

    qabs = np.asarray(
        scattering_result.qabs,
        dtype=float,
    ).ravel()

    albedo = np.asarray(
        scattering_result.albedo,
        dtype=float,
    ).ravel()

    valid_mask = passive_solution_mask(
        scattering_result
    )

    invalid_indices = np.flatnonzero(
        ~valid_mask
    )

    suspicious_points[target_wavelength_nm] = (
        invalid_indices
    )

    for wavelength_index in invalid_indices:

        suspicious_records.append(
            {
                "selected_design_wavelength_nm": (
                    target_wavelength_nm
                ),
                "wavelength_index": int(
                    wavelength_index
                ),
                "wavelength_nm": float(
                    wavelengths_nm[
                        wavelength_index
                    ]
                ),
                "qsca_original": float(
                    qsca[wavelength_index]
                ),
                "qext_original": float(
                    qext[wavelength_index]
                ),
                "qabs_original": float(
                    qabs[wavelength_index]
                ),
                "albedo_original": float(
                    albedo[wavelength_index]
                ),
                "absorber_location": str(
                    backward_case["row"][
                        "absorber_location"
                    ]
                ),
            }
        )


df_suspicious_points = pd.DataFrame(
    suspicious_records
)

print("Passivity scan completed.")

for target_wavelength_nm in sorted(
    suspicious_points
):
    print(
        f"{target_wavelength_nm:.0f} nm design: "
        f"{len(suspicious_points[target_wavelength_nm])} "
        "suspicious wavelength point(s)"
    )

print(
    "\nTotal suspicious points:",
    len(df_suspicious_points),
)

if not df_suspicious_points.empty:
    display(
        df_suspicious_points.sort_values(
            [
                "selected_design_wavelength_nm",
                "wavelength_nm",
            ]
        )
    )
else:
    print(
        "All selected-case spectra satisfy the passivity checks."
    )


# In[26]:


# ============================================================
# RECALCULATE AND CORRECT SUSPICIOUS WAVELENGTH POINTS
#
# This version bypasses run_bragg_onion_sweep because the local
# 0.01 nm wavelength grid is incompatible with the ASTM colour
# conversion performed automatically by the sweep function.
#
# Required existing objects:
#     suspicious_points
#     best_cases_backward
#     best_cases_forward
#     passive_solution_mask
#     build_absorber_objects
#
# Outputs:
#     best_cases_backward_corrected
#     best_cases_forward_corrected
#     df_correction_log
# ============================================================

import copy
import numpy as np
import pandas as pd

from bragg_onion.materials import ConstantDispersion

from bragg_onion.geometry import (
    build_bragg_onion_from_peak_wavelength,
    resolve_layer_stack,
)

from bragg_onion.solver import (
    run_scattnlay_spectrum,
)

from bragg_onion.integration import (
    integrate_collection_na,
)


# ============================================================
# CORRECTION SETTINGS
# ============================================================

# Dense local wavelength batch around every suspicious point.
LOCAL_HALF_WIDTH_NM = 0.50
LOCAL_STEP_NM = 0.01

# Same angular grid used for the detailed selected-case reruns.
CORRECTION_THETA_DEG = np.linspace(
    0.0,
    180.0,
    721,
)

CORRECTION_COLLECTION_NA = 1.0
CORRECTION_N_MEDIUM = 1.0

# Stop if the replacement still violates passive-energy conditions.
REQUIRE_VALID_REPLACEMENT = True


# ============================================================
# HELPERS
# ============================================================

def row_value(
    row,
    *column_names,
    default=None,
):
    """
    Return the first available nonmissing value from a dataframe row.
    """
    for column in column_names:
        if (
            column in row.index
            and pd.notna(row[column])
        ):
            return row[column]

    if default is not None:
        return default

    raise KeyError(
        "None of these columns contain a usable value: "
        f"{column_names}"
    )


def reconstruct_selected_geometry(
    row,
    material_a,
    material_b,
    extra_outer_shell,
):
    """
    Reconstruct one selected Bragg-onion geometry.

    Supports either an explicit number of layers or a requested
    diameter, but never both.
    """

    peak_wavelength_m = float(
        row_value(
            row,
            "peak_wavelength_m",
            default=(
                float(
                    row["design_peak_wavelength_nm"]
                )
                * 1e-9
            ),
        )
    )

    core_thickness_factor = float(
        row_value(
            row,
            "core_thickness_factor",
            "core_thickness_factor_requested",
            default=1.0,
        )
    )

    outer_layer = str(
        row_value(
            row,
            "outer_layer",
            default="A",
        )
    )

    geometry_kwargs = {
        "material_a": material_a,
        "material_b": material_b,
        "peak_wavelength_m": peak_wavelength_m,
        "outer_layer": outer_layer,
        "core_thickness_factor": (
            core_thickness_factor
        ),
        "extra_outer_shell": extra_outer_shell,
    }

    if (
        "n_layers" in row.index
        and pd.notna(row["n_layers"])
    ):
        geometry_kwargs["n_layers"] = int(
            row["n_layers"]
        )

    elif (
        "diameter_m" in row.index
        and pd.notna(row["diameter_m"])
    ):
        geometry_kwargs["diameter_m"] = float(
            row["diameter_m"]
        )

    elif (
        "requested_diameter_nm" in row.index
        and pd.notna(
            row["requested_diameter_nm"]
        )
    ):
        geometry_kwargs["diameter_m"] = (
            float(
                row["requested_diameter_nm"]
            )
            * 1e-9
        )

    else:
        raise KeyError(
            "Could not reconstruct the geometry because the row "
            "contains neither n_layers nor a requested diameter."
        )

    return build_bragg_onion_from_peak_wavelength(
        **geometry_kwargs
    )


def replace_array_value(
    destination_data,
    key,
    destination_index,
    new_value,
):
    """
    Replace one value in a one-dimensional array stored in a data dict.
    """

    if key not in destination_data:
        return False

    array = np.asarray(
        destination_data[key]
    ).copy()

    if array.ndim != 1:
        return False

    if destination_index >= len(array):
        raise IndexError(
            f"Index {destination_index} is outside "
            f"{key!r}, which has length {len(array)}."
        )

    array[destination_index] = new_value
    destination_data[key] = array

    return True


def scattering_value(
    scattering_result,
    key,
    wavelength_index,
):
    """
    Extract one scalar wavelength value from ScatteringResult.
    """

    array = np.asarray(
        getattr(scattering_result, key)
    ).ravel()

    return array[wavelength_index]


def integrated_value(
    integrated_result,
    key,
    wavelength_index,
):
    """
    Extract one scalar wavelength value from an integrated result.
    """

    array = np.asarray(
        getattr(integrated_result, key)
    ).ravel()

    return array[wavelength_index]


# ============================================================
# QUANTITIES TO PATCH
# ============================================================

SCATTERING_KEYS_TO_PATCH = [
    "qsca",
    "qext",
    "qabs",
    "qbk",
    "qpr",
    "g",
    "albedo",
    "csca_m2",
    "cext_m2",
    "cabs_m2",
]

INTEGRATED_KEYS_TO_PATCH = [
    "q_collected_geom",
]


# ============================================================
# CREATE CORRECTED COPIES
# ============================================================

best_cases_backward_corrected = copy.deepcopy(
    best_cases_backward
)

best_cases_forward_corrected = copy.deepcopy(
    best_cases_forward
)

correction_records = []


# ============================================================
# CORRECT EACH SUSPICIOUS POINT
# ============================================================

for selected_design_wavelength_nm in sorted(
    suspicious_points
):

    invalid_indices = np.asarray(
        suspicious_points[
            selected_design_wavelength_nm
        ],
        dtype=int,
    )

    if len(invalid_indices) == 0:
        continue

    original_backward = best_cases_backward[
        selected_design_wavelength_nm
    ]

    original_forward = best_cases_forward[
        selected_design_wavelength_nm
    ]

    corrected_backward = (
        best_cases_backward_corrected[
            selected_design_wavelength_nm
        ]
    )

    corrected_forward = (
        best_cases_forward_corrected[
            selected_design_wavelength_nm
        ]
    )

    row_case = original_backward["row"].copy()

    original_wavelengths_nm = np.asarray(
        original_backward["data"].get(
            "wavelengths_int_nm",
            original_backward["data"][
                "wavelengths_nm"
            ],
        ),
        dtype=float,
    ).ravel()


    # --------------------------------------------------------
    # Reconstruct the exact physical case once
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

    extinction_modifiers, extra_outer_shell = (
        build_absorber_objects(
            row_case
        )
    )

    geometry = reconstruct_selected_geometry(
        row=row_case,
        material_a=material_a,
        material_b=material_b,
        extra_outer_shell=extra_outer_shell,
    )

    stack = resolve_layer_stack(
        geometry=geometry,
        material_a=material_a,
        material_b=material_b,
        extinction_modifiers=(
            extinction_modifiers
        ),
    )


    # --------------------------------------------------------
    # Process every suspicious coarse wavelength
    # --------------------------------------------------------

    for invalid_index in invalid_indices:

        suspicious_wavelength_nm = float(
            original_wavelengths_nm[
                invalid_index
            ]
        )

        print(
            f"Correcting "
            f"{selected_design_wavelength_nm:.0f} nm design "
            f"at {suspicious_wavelength_nm:.3f} nm"
        )


        # ----------------------------------------------------
        # Dense local batch
        # ----------------------------------------------------

        local_wavelengths_nm = np.arange(
            suspicious_wavelength_nm
            - LOCAL_HALF_WIDTH_NM,
            suspicious_wavelength_nm
            + LOCAL_HALF_WIDTH_NM
            + 0.5 * LOCAL_STEP_NM,
            LOCAL_STEP_NM,
            dtype=float,
        )

        # Guarantee inclusion of the exact coarse wavelength.
        local_wavelengths_nm = np.unique(
            np.append(
                local_wavelengths_nm,
                suspicious_wavelength_nm,
            )
        )

        local_wavelengths_nm.sort()

        replacement_index = int(
            np.argmin(
                np.abs(
                    local_wavelengths_nm
                    - suspicious_wavelength_nm
                )
            )
        )

        if not np.isclose(
            local_wavelengths_nm[
                replacement_index
            ],
            suspicious_wavelength_nm,
            rtol=0.0,
            atol=1e-10,
        ):
            raise RuntimeError(
                "The dense local batch does not contain "
                "the exact suspicious wavelength."
            )


        # ----------------------------------------------------
        # Direct scattering calculation
        # No colour calculation is performed.
        # ----------------------------------------------------

        local_scattering_result = (
            run_scattnlay_spectrum(
                stack=stack,
                wavelengths_m=(
                    local_wavelengths_nm
                    * 1e-9
                ),
                theta_rad=np.deg2rad(
                    CORRECTION_THETA_DEG
                ),
                n_medium=(
                    CORRECTION_N_MEDIUM
                ),
            )
        )


        # ----------------------------------------------------
        # Integrate the same scattering result in both directions
        # ----------------------------------------------------

        local_backward_integrated = (
            integrate_collection_na(
                local_scattering_result,
                collection_na=(
                    CORRECTION_COLLECTION_NA
                ),
                direction="backward",
            )
        )

        local_forward_integrated = (
            integrate_collection_na(
                local_scattering_result,
                collection_na=(
                    CORRECTION_COLLECTION_NA
                ),
                direction="forward",
            )
        )


        # ----------------------------------------------------
        # Validate replacement
        # ----------------------------------------------------

        local_valid_mask = passive_solution_mask(
            local_scattering_result,
            absolute_tolerance=(
                REPLACEMENT_PASSIVITY_ABS_TOL
            ),
            relative_tolerance=(
                REPLACEMENT_PASSIVITY_REL_TOL
            ),
        )

        replacement_is_valid = bool(
            local_valid_mask[
                replacement_index
            ]
        )

        qsca_new = float(
            scattering_value(
                local_scattering_result,
                "qsca",
                replacement_index,
            )
        )

        qext_new = float(
            scattering_value(
                local_scattering_result,
                "qext",
                replacement_index,
            )
        )

        qabs_new = float(
            scattering_value(
                local_scattering_result,
                "qabs",
                replacement_index,
            )
        )

        albedo_new = float(
            scattering_value(
                local_scattering_result,
                "albedo",
                replacement_index,
            )
        )

        replacement_energy_residual = (
            max(
                0.0,
                qsca_new - qext_new,
            )
            / max(
                abs(qext_new),
                abs(qsca_new),
                1e-30,
            )
        )

        print(
            "  Relative passivity residual:",
            f"{100 * replacement_energy_residual:.3f}%",
        )

        # ----------------------------------------------------
        # Read original values
        # ----------------------------------------------------

        qsca_old = float(
            np.asarray(
                original_backward["data"][
                    "qsca"
                ],
                dtype=float,
            ).ravel()[invalid_index]
        )

        qext_old = float(
            np.asarray(
                original_backward["data"][
                    "qext"
                ],
                dtype=float,
            ).ravel()[invalid_index]
        )

        qabs_old = float(
            np.asarray(
                original_backward["data"][
                    "qabs"
                ],
                dtype=float,
            ).ravel()[invalid_index]
        )

        albedo_old = float(
            np.asarray(
                original_backward["data"][
                    "albedo"
                ],
                dtype=float,
            ).ravel()[invalid_index]
        )

        q_backward_old = float(
            np.asarray(
                original_backward["data"][
                    "q_collected_geom"
                ],
                dtype=float,
            ).ravel()[invalid_index]
        )

        q_forward_old = float(
            np.asarray(
                original_forward["data"][
                    "q_collected_geom"
                ],
                dtype=float,
            ).ravel()[invalid_index]
        )


        # ----------------------------------------------------
        # Read corrected integrated values
        # ----------------------------------------------------

        q_backward_new = float(
            integrated_value(
                local_backward_integrated,
                "q_collected_geom",
                replacement_index,
            )
        )

        q_forward_new = float(
            integrated_value(
                local_forward_integrated,
                "q_collected_geom",
                replacement_index,
            )
        )


        if not replacement_is_valid:

            print(
                "  WARNING: dense local replacement remains "
                "unphysical."
            )

            print(
                f"  Qsca={qsca_new:.6g}, "
                f"Qext={qext_new:.6g}, "
                f"Qabs={qabs_new:.6g}, "
                f"albedo={albedo_new:.6g}"
            )

            print(
                "  Replacing this wavelength point with NaN."
            )

            # Scattering quantities shared by forward and backward results.
            keys_to_mask = [
                "qsca",
                "qext",
                "qabs",
                "qbk",
                "qpr",
                "g",
                "albedo",
                "csca_m2",
                "cext_m2",
                "cabs_m2",
            ]

            for key in keys_to_mask:

                replace_array_value(
                    corrected_backward["data"],
                    key,
                    invalid_index,
                    np.nan,
                )

                replace_array_value(
                    corrected_forward["data"],
                    key,
                    invalid_index,
                    np.nan,
                )

            # Direction-dependent collected spectra.
            replace_array_value(
                corrected_backward["data"],
                "q_collected_geom",
                invalid_index,
                np.nan,
            )

            replace_array_value(
                corrected_forward["data"],
                "q_collected_geom",
                invalid_index,
                np.nan,
            )

            correction_records.append(
                {
                    "selected_design_wavelength_nm": (
                        selected_design_wavelength_nm
                    ),
                    "wavelength_nm": (
                        suspicious_wavelength_nm
                    ),
                    "correction_status": (
                        "masked_with_nan"
                    ),
                    "replacement_valid": False,
                    "qsca_original": qsca_old,
                    "qsca_recalculated": qsca_new,
                    "qext_original": qext_old,
                    "qext_recalculated": qext_new,
                    "qabs_original": qabs_old,
                    "qabs_recalculated": qabs_new,
                    "albedo_original": albedo_old,
                    "albedo_recalculated": albedo_new,
                    "q_backward_original": (q_backward_old),
                    "q_backward_corrected": np.nan,
                    "q_forward_original": (q_forward_old),
                    "q_forward_corrected": np.nan,
                    "corrected_collection_sum": np.nan,
                    "corrected_collection_sum_over_qsca": (
                        np.nan
                    ),
                }
            )

            # Skip the normal replacement code below for this point.
            continue



        # ----------------------------------------------------
        # Patch scattering quantities in both direction dicts
        # ----------------------------------------------------

        for key in SCATTERING_KEYS_TO_PATCH:

            if not hasattr(
                local_scattering_result,
                key,
            ):
                continue

            new_value = scattering_value(
                local_scattering_result,
                key,
                replacement_index,
            )

            replace_array_value(
                corrected_backward["data"],
                key,
                invalid_index,
                new_value,
            )

            replace_array_value(
                corrected_forward["data"],
                key,
                invalid_index,
                new_value,
            )


        # ----------------------------------------------------
        # Patch direction-dependent collection spectra
        # ----------------------------------------------------

        replace_array_value(
            corrected_backward["data"],
            "q_collected_geom",
            invalid_index,
            q_backward_new,
        )

        replace_array_value(
            corrected_forward["data"],
            "q_collected_geom",
            invalid_index,
            q_forward_new,
        )


        # ----------------------------------------------------
        # Conservation check for the corrected collection
        # ----------------------------------------------------

        collection_sum = (
            q_backward_new
            + q_forward_new
        )

        collection_to_qsca_ratio = (
            collection_sum / qsca_new
            if qsca_new != 0
            else np.nan
        )


        # ----------------------------------------------------
        # Correction log
        # ----------------------------------------------------

        correction_records.append(
            {
                "selected_design_wavelength_nm": (
                    selected_design_wavelength_nm
                ),
                "wavelength_nm": (
                    suspicious_wavelength_nm
                ),
                "replacement_valid": (
                    replacement_is_valid
                ),
                "qsca_original": qsca_old,
                "qsca_corrected": qsca_new,
                "qext_original": qext_old,
                "qext_corrected": qext_new,
                "qabs_original": qabs_old,
                "qabs_corrected": qabs_new,
                "albedo_original": albedo_old,
                "albedo_corrected": (
                    albedo_new
                ),
                "q_backward_original": (
                    q_backward_old
                ),
                "q_backward_corrected": (
                    q_backward_new
                ),
                "q_forward_original": (
                    q_forward_old
                ),
                "q_forward_corrected": (
                    q_forward_new
                ),
                "corrected_collection_sum": (
                    collection_sum
                ),
                "corrected_collection_sum_over_qsca": (
                    collection_to_qsca_ratio
                ),
                "replacement_energy_residual": (
                    replacement_energy_residual
                ),
                "replacement_energy_residual_pct": (
                    100.0
                    * replacement_energy_residual
                ),
                "correction_status": "recalculated",
            }
        )

        print(
            f"  Qsca: {qsca_old:.6g} -> "
            f"{qsca_new:.6g}"
        )

        print(
            f"  Qabs: {qabs_old:.6g} -> "
            f"{qabs_new:.6g}"
        )

        print(
            f"  forward + backward over Qsca: "
            f"{collection_to_qsca_ratio:.6f}"
        )


# ============================================================
# CORRECTION SUMMARY
# ============================================================

df_correction_log = pd.DataFrame(
    correction_records
)

print(
    "\nCorrected wavelength points:",
    len(df_correction_log),
)

if not df_correction_log.empty:

    display(
        df_correction_log.sort_values(
            [
                "selected_design_wavelength_nm",
                "wavelength_nm",
            ]
        )
    )

else:

    print(
        "No suspicious wavelength points required correction."
    )


# In[27]:


# ============================================================
# VALIDATE CORRECTIONS AND ACTIVATE CORRECTED SPECTRA
#
# After this cell, continue using:
#     best_cases_backward
#     best_cases_forward
#     spectral_records
# ============================================================

import numpy as np
import pandas as pd
import matplotlib as mpl


def passive_data_mask(
    data,
    *,
    absolute_tolerance=REPLACEMENT_PASSIVITY_ABS_TOL,
    relative_tolerance=REPLACEMENT_PASSIVITY_REL_TOL,
):
    """
    Return *wo masks:

    physical_mask
     *  Finite points that satisfy passi*e-energy conditions.

    delibera*ely_masked
        Points for whic* all principal energy quantities a*e NaN.
    """

    qsca = np.asarray(
        data["qsca"],
        dtype=float,
    ).ravel()

    qext = np.asarray(
        data["qext"],
        dtype=float,
    ).ravel()

    qabs = np.asarray(
        data["qabs"],
        dtype=float,
    ).ravel()

    albedo = np.asarray(
        data["albedo"],
        dtype=float,
    ).ravel()

    deliberately_masked = (
        np.isnan(qsca)
        & np.isnan(qext)
        & np.isnan(qabs)
        & np.isnan(albedo)
    )

    finite = (
        np.isfinite(qsca)
        & np.isfinite(qext)
        & np.isfinite(qabs)
        & np.isfinite(albedo)
    )

    scale = np.maximum.reduce(
        [
            np.abs(qsca),
            np.abs(qext),
            np.ones_like(qsca),
        ]
    )

    tolerance = (
        absolute_tolerance
        + relative_tolerance * scale
    )

    physical_mask = (
        finite
        & (qsca >= -tolerance)
        & (qext >= -tolerance)
        & (qabs >= -tolerance)
        & (qsca <= qext + tolerance)
        & (
            albedo
            >= -relative_tolerance
        )
        & (
            albedo
            <= 1.0 + relative_tolerance
        )
    )

    return physical_mask, deliberately_masked


# ------------------------------------------------------------
# Validate every corrected case
# ------------------------------------------------------------

remaining_invalid_records = []

for target_wavelength_nm in sorted(
    best_cases_backward_corrected
):

    backward_data = (
        best_cases_backward_corrected[
            target_wavelength_nm
        ]["data"]
    )

    physical_mask, deliberately_masked = (
        passive_data_mask(
            backward_data,
            absolute_tolerance=(
                REPLACEMENT_PASSIVITY_ABS_TOL
            ),
            # relative_tolerance=(
            #     REPLACEMENT_PASSIVITY_REL_TOL
            # )
        )
    )

    acceptable_mask = (
        physical_mask
        | deliberately_masked
    )

    number_masked = int(
        deliberately_masked.sum()
    )

    if number_masked > 0:
        print(
            f"{target_wavelength_nm:.0f} nm design: "
            f"{number_masked} wavelength point(s) "
            "masked with NaN."
        )

    for invalid_index in np.flatnonzero(
        ~acceptable_mask
    ):

        wavelengths_nm = np.asarray(
            backward_data["wavelengths_nm"],
            dtype=float,
        ).ravel()

    for invalid_index in np.flatnonzero(
        ~acceptable_mask
    ):
        remaining_invalid_records.append(
            {
                "selected_design_wavelength_nm": (
                    target_wavelength_nm
                ),
                "wavelength_nm": float(
                    wavelengths_nm[
                        invalid_index
                    ]
                ),
                "qsca": float(
                    np.asarray(
                        backward_data["qsca"]
                    )[invalid_index]
                ),
                "qext": float(
                    np.asarray(
                        backward_data["qext"]
                    )[invalid_index]
                ),
                "qabs": float(
                    np.asarray(
                        backward_data["qabs"]
                    )[invalid_index]
                ),
                "albedo": float(
                    np.asarray(
                        backward_data["albedo"]
                    )[invalid_index]
                ),
            }
        )


df_remaining_invalid = pd.DataFrame(
    remaining_invalid_records
)

if not df_remaining_invalid.empty:
    display(df_remaining_invalid)

    raise RuntimeError(
        f"{len(df_remaining_invalid)} wavelength point(s) "
        "remain unphysical after correction."
    )


# ------------------------------------------------------------
# Activate corrected working dictionaries
# ------------------------------------------------------------

best_cases_backward = (
    best_cases_backward_corrected
)

best_cases_forward = (
    best_cases_forward_corrected
)


# ------------------------------------------------------------
# Rebuild spectral_records
# ------------------------------------------------------------

target_wavelengths = sorted(
    best_cases_backward
)

fallback_cmap = mpl.colormaps["turbo"]

fallback_colours = {
    target_wavelength_nm: fallback_cmap(
        index
        / max(
            len(target_wavelengths) - 1,
            1,
        )
    )
    for index, target_wavelength_nm
    in enumerate(target_wavelengths)
}


spectral_records = {}

for target_wavelength_nm in target_wavelengths:

    backward_case = best_cases_backward[
        target_wavelength_nm
    ]

    forward_case = best_cases_forward[
        target_wavelength_nm
    ]

    backward_data = backward_case["data"]
    forward_data = forward_case["data"]

    selected_row = backward_case["row"]

    wavelengths_nm = np.asarray(
        backward_data.get(
            "wavelengths_int_nm",
            backward_data["wavelengths_nm"],
        ),
        dtype=float,
    ).ravel()

    q_backward = np.asarray(
        backward_data[
            "q_collected_geom"
        ],
        dtype=float,
    ).ravel()

    q_forward = np.asarray(
        forward_data[
            "q_collected_geom"
        ],
        dtype=float,
    ).ravel()

    line_colour = None

    for colour_column in [
        "hex",
        "hex_display",
    ]:
        if (
            colour_column
            in selected_row.index
            and pd.notna(
                selected_row[
                    colour_column
                ]
            )
            and mpl.colors.is_color_like(
                selected_row[
                    colour_column
                ]
            )
        ):
            line_colour = str(
                selected_row[
                    colour_column
                ]
            )
            break

    if line_colour is None:
        line_colour = fallback_colours[
            target_wavelength_nm
        ]

    backward_peak_index = int(
        np.nanargmax(q_backward)
    )

    forward_peak_index = int(
        np.nanargmax(q_forward)
    )

    spectral_records[
        target_wavelength_nm
    ] = {
        "wavelengths_nm": wavelengths_nm,
        "q_backward_raw": q_backward,
        "q_forward_raw": q_forward,
        "backward_peak_wavelength_nm": float(
            wavelengths_nm[
                backward_peak_index
            ]
        ),
        "forward_peak_wavelength_nm": float(
            wavelengths_nm[
                forward_peak_index
            ]
        ),
        "backward_peak_value": float(
            q_backward[
                backward_peak_index
            ]
        ),
        "forward_peak_value": float(
            q_forward[
                forward_peak_index
            ]
        ),
        "line_colour": line_colour,
        "row": selected_row.copy(),
    }


print(
    "Correction completed successfully."
)

print(
    "All corrected selected-case spectra satisfy "
    "the passive-energy checks."
)

print(
    "Rebuilt spectral_records for:",
    sorted(spectral_records),
)


# Cell 4: Verify the reruns

# In[28]:


# ============================================================
# CONSISTENCY CHECKS
# ============================================================

backward_keys = set(
    best_cases_backward
)

forward_keys = set(
    best_cases_forward
)

if backward_keys != forward_keys:
    raise ValueError(
        "Forward and backward rerun keys differ.\n"
        f"Missing from forward: "
        f"{sorted(backward_keys - forward_keys)}\n"
        f"Missing from backward: "
        f"{sorted(forward_keys - backward_keys)}"
    )


for key in sorted(backward_keys):

    data_backward = (
        best_cases_backward[key]["data"]
    )

    data_forward = (
        best_cases_forward[key]["data"]
    )

    for quantity in [
        "qsca",
        "qext",
        "qabs",
        "g",
        "albedo",
    ]:
        if quantity not in data_backward:
            continue

        if not np.allclose(
            data_backward[quantity],
            data_forward[quantity],
            equal_nan=True,
        ):
            raise ValueError(
                f"{quantity} differs between forward and "
                f"backward reruns for {key} nm."
            )

    if np.allclose(
        data_backward["q_collected_geom"],
        data_forward["q_collected_geom"],
        equal_nan=True,
    ):
        print(
            f"Warning: forward and backward collected "
            f"spectra are identical for {key} nm."
        )


print(
    "Consistency checks passed: scattering, g, and albedo "
    "agree between the forward and backward reruns."
)


# Cell 5: Create reusable summary tables

# In[29]:


# ============================================================
# COMPACT RERUN SUMMARY TABLES
# ============================================================

backward_summary_rows = []
forward_summary_rows = []


for target_wavelength_nm in sorted(
    best_cases_backward
):

    backward_result = (
        best_cases_backward[
            target_wavelength_nm
        ]
    )

    forward_result = (
        best_cases_forward[
            target_wavelength_nm
        ]
    )

    backward_df = make_plotting_aliases(
        backward_result[
            "sweep_case"
        ].dataframe
    ).copy()

    forward_df = make_plotting_aliases(
        forward_result[
            "sweep_case"
        ].dataframe
    ).copy()

    if len(backward_df) != 1:
        raise ValueError(
            f"Expected one backward summary row for "
            f"{target_wavelength_nm} nm."
        )

    if len(forward_df) != 1:
        raise ValueError(
            f"Expected one forward summary row for "
            f"{target_wavelength_nm} nm."
        )

    backward_row = (
        backward_df.iloc[0].to_dict()
    )

    forward_row = (
        forward_df.iloc[0].to_dict()
    )

    selected_row = (
        best_cases_backward[
            target_wavelength_nm
        ]["row"]
    )

    common_metadata = {
        "selected_target_wavelength_nm": (
            target_wavelength_nm
        ),
        "selected_absorber_location": str(
            selected_row[
                "absorber_location"
            ]
        ),
        "selected_d_opt": float(
            selected_row["d_opt"]
        ),
        "selected_n_A": float(
            selected_row["n_A"]
        ),
        "selected_n_B": float(
            selected_row["n_B"]
        ),
        "selected_k": float(
            get_row_value(
                selected_row,
                "k",
                default=0.0,
            )
        ),
    }

    backward_summary_rows.append(
        {
            **backward_row,
            **common_metadata,
            "rerun_direction": "backward",
        }
    )

    forward_summary_rows.append(
        {
            **forward_row,
            **common_metadata,
            "rerun_direction": "forward",
        }
    )


df_best_backward_summary = pd.DataFrame(
    backward_summary_rows
)

df_best_forward_summary = pd.DataFrame(
    forward_summary_rows
)

display_columns = [
    "selected_target_wavelength_nm",
    "rerun_direction",
    "selected_absorber_location",
    "selected_d_opt",
    "selected_n_A",
    "selected_k",
    "n_layers",
    "diameter_nm",
    "eta_C",
    "eta_L",
    "q_collected_geom_max",
    "q_collected_geom_peak_wavelength_nm",
    "hex",
]

display_columns_backward = [
    column
    for column in display_columns
    if column
    in df_best_backward_summary.columns
]

display_columns_forward = [
    column
    for column in display_columns
    if column
    in df_best_forward_summary.columns
]

print("Backward rerun summary:")
display(
    df_best_backward_summary[
        display_columns_backward
    ]
)

print("Forward rerun summary:")
display(
    df_best_forward_summary[
        display_columns_forward
    ]
)


# In[30]:


# ============================================================
# BUILD spectral_records FROM THE DETAILED FORWARD/BACKWARD RERUNS
#
# Required existing objects:
#     best_cases_backward
#     best_cases_forward
#
# Output:
#     spectral_records
#
# This dictionary is used by the integrated-spectra plotting cell.
# ============================================================

from turtle import backward

import numpy as np
import pandas as pd
import matplotlib as mpl


# ============================================================
# VALIDATE THE RERUN DICTIONARIES
# ============================================================

if "best_cases_backward" not in globals():
    raise NameError(
        "`best_cases_backward` is missing. "
        "Run the backward rerun cell first."
    )

if "best_cases_forward" not in globals():
    raise NameError(
        "`best_cases_forward` is missing. "
        "Run the forward rerun cell first."
    )


backward_keys = set(best_cases_backward)
forward_keys = set(best_cases_forward)

if backward_keys != forward_keys:
    raise ValueError(
        "Backward and forward rerun dictionaries contain "
        "different cases.\n"
        f"Missing from forward: "
        f"{sorted(backward_keys - forward_keys)}\n"
        f"Missing from backward: "
        f"{sorted(forward_keys - backward_keys)}"
    )

if len(backward_keys) == 0:
    raise ValueError(
        "The rerun dictionaries contain no cases."
    )


# ============================================================
# HELPER
# ============================================================

def get_first_array(
    data,
    possible_names,
    *,
    case_description,
):
    """
    Return the first available one-dimensional numeric array.
    """
    for name in possible_names:

        if name in data:
            return np.asarray(
                data[name],
                dtype=float,
            ).ravel()

    raise KeyError(
        f"Could not find any of {possible_names} for "
        f"{case_description}.\n"
        f"Available keys: {sorted(data.keys())}"
    )


# ============================================================
# FALLBACK COLOURS
# ============================================================

target_wavelengths = sorted(backward_keys)

fallback_cmap = mpl.colormaps["turbo"]

fallback_colours = {
    target_wavelength_nm: fallback_cmap(
        index
        / max(
            len(target_wavelengths) - 1,
            1,
        )
    )
    for index, target_wavelength_nm
    in enumerate(target_wavelengths)
}


# ============================================================
# BUILD THE COMBINED SPECTRAL DICTIONARY
# ============================================================

spectral_records = {}

for target_wavelength_nm in target_wavelengths:

    backward_result = best_cases_backward[target_wavelength_nm]
    forward_result = best_cases_forward[target_wavelength_nm]

    backward_data = backward_result["data"]
    forward_data = forward_result["data"]

    # The selected case metadata / DataFrame row
    selected_row = backward_result["row"]


    # --------------------------------------------------------
    # Wavelength grids
    # --------------------------------------------------------

    wavelengths_backward_nm = get_first_array(
        backward_data,
        [
            "wavelengths_int_nm",
            "wavelengths_nm",
        ],
        case_description=(
            f"{target_wavelength_nm} nm backward case"
        ),
    )

    wavelengths_forward_nm = get_first_array(
        forward_data,
        [
            "wavelengths_int_nm",
            "wavelengths_nm",
        ],
        case_description=(
            f"{target_wavelength_nm} nm forward case"
        ),
    )

    if (
        wavelengths_backward_nm.shape
        != wavelengths_forward_nm.shape
        or not np.allclose(
            wavelengths_backward_nm,
            wavelengths_forward_nm,
        )
    ):
        raise ValueError(
            "Forward and backward wavelength grids differ for "
            f"the {target_wavelength_nm} nm design."
        )


    # --------------------------------------------------------
    # Integrated collection spectra
    # --------------------------------------------------------

    q_backward = get_first_array(
        backward_data,
        ["q_collected_geom"],
        case_description=(
            f"{target_wavelength_nm} nm backward case"
        ),
    )

    q_forward = get_first_array(
        forward_data,
        ["q_collected_geom"],
        case_description=(
            f"{target_wavelength_nm} nm forward case"
        ),
    )

    if len(q_backward) != len(wavelengths_backward_nm):
        raise ValueError(
            f"Backward spectrum length mismatch for "
            f"{target_wavelength_nm} nm."
        )

    if len(q_forward) != len(wavelengths_forward_nm):
        raise ValueError(
            f"Forward spectrum length mismatch for "
            f"{target_wavelength_nm} nm."
        )


    # --------------------------------------------------------
    # Determine the plotting colour
    # --------------------------------------------------------
    # Prefer the actual calculated backward-reflection colour.

    line_colour = None

    for colour_column in [
        "hex",
        "hex_display",
    ]:
        if (
            colour_column in selected_row.index
            and pd.notna(selected_row[colour_column])
            and mpl.colors.is_color_like(
                selected_row[colour_column]
            )
        ):
            line_colour = str(
                selected_row[colour_column]
            )
            break

    if line_colour is None:
        line_colour = fallback_colours[
            target_wavelength_nm
        ]


    # --------------------------------------------------------
    # Peak properties
    # --------------------------------------------------------

    backward_peak_index = int(
        np.nanargmax(q_backward)
    )

    forward_peak_index = int(
        np.nanargmax(q_forward)
    )


    # --------------------------------------------------------
    # Store the combined record
    # --------------------------------------------------------

    spectral_records[target_wavelength_nm] = {
        "wavelengths_nm": wavelengths_backward_nm,

        "q_backward_raw": q_backward,
        "q_forward_raw": q_forward,

        "backward_peak_wavelength_nm": float(
            wavelengths_backward_nm[
                backward_peak_index
            ]
        ),
        "forward_peak_wavelength_nm": float(
            wavelengths_forward_nm[
                forward_peak_index
            ]
        ),

        "backward_peak_value": float(
            q_backward[
                backward_peak_index
            ]
        ),
        "forward_peak_value": float(
            q_forward[
                forward_peak_index
            ]
        ),

        "line_colour": line_colour,
        "row": selected_row.copy(),
    }


# ============================================================
# SANITY CHECK AND SUMMARY
# ============================================================

summary_records = []

for target_wavelength_nm, record in spectral_records.items():

    row = record["row"]

    summary_records.append(
        {
            "design_wavelength_nm": target_wavelength_nm,

            "absorber_location": str(
                row["absorber_location"]
            ),

            "backward_peak_nm": (
                record["backward_peak_wavelength_nm"]
            ),

            "backward_peak_value": (
                record["backward_peak_value"]
            ),

            "forward_peak_nm": (
                record["forward_peak_wavelength_nm"]
            ),

            "forward_peak_value": (
                record["forward_peak_value"]
            ),

            "line_colour": record["line_colour"],
        }
    )

df_spectral_records_summary = pd.DataFrame(
    summary_records
).sort_values(
    "design_wavelength_nm"
).reset_index(drop=True)


print(
    f"Created spectral_records for "
    f"{len(spectral_records)} selected designs."
)

print(
    "Keys:",
    sorted(spectral_records),
)

display(
    df_spectral_records_summary
)


# In[31]:


# ============================================================
# PLOT INTEGRATED FORWARD AND/OR BACKWARD SPECTRA
#
# Required input:
#     spectral_records
#
# Options:
#     SPECTRA_TO_PLOT = "backward", "forward", or "both"
#     BOTH_LAYOUT     = "combined" or "stacked"
# ============================================================

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from matplotlib.lines import Line2D


# ============================================================
# STYLE
# ============================================================

mpl.rcdefaults()

plt.style.use(
    "/mnt/c/Users/SchwarzN/"
    "OneDrive - Université de Fribourg/"
    "Institution/Other/Scripts_and_HowTo/"
    "science_paper.mplstyle"
)


# ============================================================
# USER SETTINGS
# ============================================================

# Choose: "backward", "forward", or "both"
SPECTRA_TO_PLOT = "both"

# Used only when SPECTRA_TO_PLOT = "both".
# Choose: "combined" or "stacked"
BOTH_LAYOUT = "stacked"

# Normalize every plotted spectrum independently.
# False preserves absolute collected scattering efficiencies.
NORMALIZE_SPECTRA = False

# Axis limits. Set to None for automatic limits.
X_LIMITS = (380.0, 780.0)
# Y-axis limits
#
# Y_LIMITS is used for:
#   - backward-only plots
#   - forward-only plots
#   - combined forward/backward plots
#
# In stacked mode, BACKWARD_Y_LIMITS and FORWARD_Y_LIMITS
# override Y_LIMITS for their respective panels.
#
# Set any value to None for automatic scaling.

Y_LIMITS = None

BACKWARD_Y_LIMITS = (0.0,  None)
FORWARD_Y_LIMITS = (1.0, None)

# Axis labels. Set to None or "" to hide a label.
X_LABEL = "Wavelength [nm]"
Y_LABEL = r"$Q_{\mathrm{collected,geom}}$"

# Optional panel labels in stacked mode.
SHOW_PANEL_LABELS = False
BACKWARD_PANEL_LABEL = "(i) Backward collection"
FORWARD_PANEL_LABEL = "(ii) Forward collection"

# Line appearance
BACKWARD_LINESTYLE = "-"
FORWARD_LINESTYLE = "--"

BACKWARD_LINEWIDTH = 1.8
FORWARD_LINEWIDTH = 1.6

BACKWARD_ALPHA = 1.0
FORWARD_ALPHA = 0.90

# Optional vertical lines at the design wavelengths
SHOW_DESIGN_WAVELENGTH_LINES = False

# Optional markers at the spectral maxima
SHOW_PEAK_MARKERS = True

# Legend controls
SHOW_LEGEND = True
SHOW_WAVELENGTH_LEGEND = True
SHOW_DIRECTION_LEGEND = True
LEGEND_BELOW_FIGURE = True
WAVELENGTH_LEGEND_COLUMNS = 8
LEGEND_FONT_SIZE = 9

# Grid
SHOW_GRID = False

# Figure sizes in centimetres
SINGLE_FIGURE_SIZE_CM = (15.92 / 2, 15.92 / 4)
STACKED_FIGURE_SIZE_CM = (12.0, 12.0)

# Saving
SAVE_FIGURE = False
SAVE_PNG = True
SAVE_PDF = True

OUTPUT_DIRECTORY = Path(
    "/mnt/c/Users/SchwarzN/"
    "OneDrive - Université de Fribourg/"
    "Institution/P1_BraggSpericalPigments/"
    "Simulation/MIE_Computation/"
    "Figures"
)

FIGURE_BASENAME = "Figure_2B_integrated_spectra"
PNG_DPI = 600
TRANSPARENT_BACKGROUND = True

def interpolate_nans_for_plot(
    wavelengths_nm,
    spectrum,
):
    wavelengths_nm = np.asarray(
        wavelengths_nm,
        dtype=float,
    )

    spectrum = np.asarray(
        spectrum,
        dtype=float,
    ).copy()

    finite = np.isfinite(spectrum)

    if finite.sum() < 2:
        return spectrum

    spectrum[~finite] = np.interp(
        wavelengths_nm[~finite],
        wavelengths_nm[finite],
        spectrum[finite],
    )

    return spectrum

# ============================================================
# VALIDATION
# ============================================================

if SPECTRA_TO_PLOT not in {
    "backward",
    "forward",
    "both",
}:
    raise ValueError(
        "SPECTRA_TO_PLOT must be 'backward', 'forward', or 'both'."
    )

if BOTH_LAYOUT not in {
    "combined",
    "stacked",
}:
    raise ValueError(
        "BOTH_LAYOUT must be 'combined' or 'stacked'."
    )

if "spectral_records" not in globals():
    raise NameError(
        "`spectral_records` is missing. Run the spectrum extraction "
        "cell before this plotting cell."
    )

target_wavelengths = sorted(spectral_records)

if len(target_wavelengths) == 0:
    raise ValueError("No spectra are available in spectral_records.")

use_stacked_layout = (
    SPECTRA_TO_PLOT == "both"
    and BOTH_LAYOUT == "stacked"
)


# ============================================================
# PREPARE SPECTRA
# ============================================================

plot_records = {}

for target_wavelength_nm in target_wavelengths:

    record = spectral_records[target_wavelength_nm]

    wavelengths_nm = np.asarray(
        record["wavelengths_nm"],
        dtype=float,
    ).ravel()

    q_backward = np.asarray(
        record["q_backward_raw"],
        dtype=float,
    ).ravel()

    q_forward = np.asarray(
        record["q_forward_raw"],
        dtype=float,
    ).ravel()

    if NORMALIZE_SPECTRA:

        backward_maximum = np.nanmax(q_backward)
        forward_maximum = np.nanmax(q_forward)

        if backward_maximum > 0:
            q_backward = q_backward / backward_maximum

        if forward_maximum > 0:
            q_forward = q_forward / forward_maximum

    plot_records[target_wavelength_nm] = {
        "wavelengths_nm": wavelengths_nm,
        "q_backward": q_backward,
        "q_forward": q_forward,
        "line_colour": record["line_colour"],
    }


# ============================================================
# CREATE FIGURE
# ============================================================

cm_to_inches = 1.0 / 2.54

# Disable constrained layout inherited from the mplstyle.
mpl.rcParams["figure.constrained_layout.use"] = True

if use_stacked_layout:

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(
            STACKED_FIGURE_SIZE_CM[0] * cm_to_inches,
            STACKED_FIGURE_SIZE_CM[1] * cm_to_inches,
        ),
        sharex=True,
        constrained_layout=True,
    )

    fig.set_layout_engine(None)

    ax_backward = axes[0]
    ax_forward = axes[1]

    active_axes = [ax_backward, ax_forward]

else:

    fig, ax = plt.subplots(
        figsize=(
            SINGLE_FIGURE_SIZE_CM[0] * cm_to_inches,
            SINGLE_FIGURE_SIZE_CM[1] * cm_to_inches,
        ),
        constrained_layout=False,
    )

    fig.set_layout_engine(None)

    active_axes = [ax]


# ============================================================
# PLOT SPECTRA
# ============================================================

if use_stacked_layout:

    for target_wavelength_nm in target_wavelengths:

        record = plot_records[target_wavelength_nm]

        wavelengths_nm = record["wavelengths_nm"]
        q_backward = record["q_backward"]
        q_forward = record["q_forward"]
        line_colour = record["line_colour"]

        q_backward_plot = interpolate_nans_for_plot(
            wavelengths_nm,
            q_backward,
        )

        q_forward_plot = interpolate_nans_for_plot(
            wavelengths_nm,
            q_forward,
        )

        ax_backward.plot(
            wavelengths_nm,
            q_backward_plot,
            color=line_colour,
            linestyle=BACKWARD_LINESTYLE,
            linewidth=BACKWARD_LINEWIDTH,
            alpha=BACKWARD_ALPHA,
        )

        ax_forward.plot(
            wavelengths_nm,
            q_forward_plot,
            color=line_colour,
            linestyle=FORWARD_LINESTYLE,
            linewidth=FORWARD_LINEWIDTH,
            alpha=FORWARD_ALPHA,
        )

        if SHOW_DESIGN_WAVELENGTH_LINES:

            ax_backward.axvline(
                target_wavelength_nm,
                color=line_colour,
                linestyle=":",
                linewidth=0.6,
                alpha=0.4,
            )

            ax_forward.axvline(
                target_wavelength_nm,
                color=line_colour,
                linestyle=":",
                linewidth=0.6,
                alpha=0.4,
            )

        if SHOW_PEAK_MARKERS:

            backward_peak_index = int(
                np.nanargmax(q_backward)
            )

            forward_peak_index = int(
                np.nanargmax(q_forward)
            )

            ax_backward.scatter(
                wavelengths_nm[backward_peak_index],
                q_backward[backward_peak_index],
                s=18,
                color=line_colour,
                edgecolor="black",
                linewidth=0.4,
                zorder=5,
            )

            ax_forward.scatter(
                wavelengths_nm[forward_peak_index],
                q_forward[forward_peak_index],
                s=18,
                color=line_colour,
                edgecolor="black",
                linewidth=0.4,
                zorder=5,
            )

    if SHOW_PANEL_LABELS:

        ax_backward.text(
            0.02,
            0.95,
            BACKWARD_PANEL_LABEL,
            transform=ax_backward.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
        )

        ax_forward.text(
            0.02,
            0.95,
            FORWARD_PANEL_LABEL,
            transform=ax_forward.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
        )

    if Y_LABEL:
        ax_backward.set_ylabel(Y_LABEL)
        ax_forward.set_ylabel(Y_LABEL)

    if X_LABEL:
        ax_forward.set_xlabel(X_LABEL)


else:

    for target_wavelength_nm in target_wavelengths:

        record = plot_records[target_wavelength_nm]

        wavelengths_nm = record["wavelengths_nm"]
        q_backward = record["q_backward"]
        q_forward = record["q_forward"]
        line_colour = record["line_colour"]

        if SPECTRA_TO_PLOT in {
            "backward",
            "both",
        }:
            ax.plot(
                wavelengths_nm,
                q_backward_plot,
                color=line_colour,
                linestyle=BACKWARD_LINESTYLE,
                linewidth=BACKWARD_LINEWIDTH,
                alpha=BACKWARD_ALPHA,
            )

        if SPECTRA_TO_PLOT in {
            "forward",
            "both",
        }:
            ax.plot(
                wavelengths_nm,
                q_forward_plot,
                color=line_colour,
                linestyle=FORWARD_LINESTYLE,
                linewidth=FORWARD_LINEWIDTH,
                alpha=FORWARD_ALPHA,
            )

        if SHOW_DESIGN_WAVELENGTH_LINES:
            ax.axvline(
                target_wavelength_nm,
                color=line_colour,
                linestyle=":",
                linewidth=0.6,
                alpha=0.4,
            )

        if SHOW_PEAK_MARKERS:

            if SPECTRA_TO_PLOT in {
                "backward",
                "both",
            }:
                peak_index = int(
                    np.nanargmax(q_backward)
                )

                ax.scatter(
                    wavelengths_nm[peak_index],
                    q_backward[peak_index],
                    s=18,
                    color=line_colour,
                    edgecolor="black",
                    linewidth=0.4,
                    zorder=5,
                )

            if SPECTRA_TO_PLOT in {
                "forward",
                "both",
            }:
                peak_index = int(
                    np.nanargmax(q_forward)
                )

                ax.scatter(
                    wavelengths_nm[peak_index],
                    q_forward[peak_index],
                    s=18,
                    marker="s",
                    color=line_colour,
                    edgecolor="black",
                    linewidth=0.4,
                    zorder=5,
                )

    if X_LABEL:
        ax.set_xlabel(X_LABEL)

    if Y_LABEL:
        ax.set_ylabel(Y_LABEL)


# ============================================================
# AXIS FORMATTING
# ============================================================

# Apply the common x-axis limits and grid settings.
for current_axis in active_axes:

    if X_LIMITS is not None:
        current_axis.set_xlim(
            *X_LIMITS
        )

    current_axis.grid(
        SHOW_GRID,
        which="both",
        axis="both",
    )


# ------------------------------------------------------------
# Y-axis limits
# ------------------------------------------------------------

if use_stacked_layout:

    # Backward panel
    if BACKWARD_Y_LIMITS is not None:

        ax_backward.set_ylim(
            *BACKWARD_Y_LIMITS
        )

    elif Y_LIMITS is not None:

        ax_backward.set_ylim(
            *Y_LIMITS
        )

    elif NORMALIZE_SPECTRA:

        ax_backward.set_ylim(
            0.0,
            1.05,
        )

    else:

        ax_backward.set_ylim(
            bottom=0.0
        )


    # Forward panel
    if FORWARD_Y_LIMITS is not None:

        ax_forward.set_ylim(
            *FORWARD_Y_LIMITS
        )

    elif Y_LIMITS is not None:

        ax_forward.set_ylim(
            *Y_LIMITS
        )

    elif NORMALIZE_SPECTRA:

        ax_forward.set_ylim(
            0.0,
            1.05,
        )

    else:

        ax_forward.set_ylim(
            bottom=0.0
        )


else:

    # Single-axis plots:
    # backward only, forward only, or both combined.
    if Y_LIMITS is not None:

        ax.set_ylim(
            *Y_LIMITS
        )

    elif NORMALIZE_SPECTRA:

        ax.set_ylim(
            0.0,
            1.05,
        )

    else:

        ax.set_ylim(
            bottom=0.0
        )


# ------------------------------------------------------------
# Print applied limits for verification
# ------------------------------------------------------------

if use_stacked_layout:

    print(
        "Backward y-limits:",
        ax_backward.get_ylim(),
    )

    print(
        "Forward y-limits:",
        ax_forward.get_ylim(),
    )

else:

    print(
        "Y-limits:",
        ax.get_ylim(),
    )


# ============================================================
# LEGEND
# ============================================================

wavelength_handles = [
    Line2D(
        [0],
        [0],
        color=plot_records[
            target_wavelength_nm
        ]["line_colour"],
        linewidth=2.0,
        linestyle="-",
        label=f"{target_wavelength_nm:.0f}",
    )
    for target_wavelength_nm in target_wavelengths
]

direction_handles = []

if SPECTRA_TO_PLOT in {
    "backward",
    "both",
}:
    direction_handles.append(
        Line2D(
            [0],
            [0],
            color="black",
            linestyle=BACKWARD_LINESTYLE,
            linewidth=BACKWARD_LINEWIDTH,
            label="Backward",
        )
    )

if SPECTRA_TO_PLOT in {
    "forward",
    "both",
}:
    direction_handles.append(
        Line2D(
            [0],
            [0],
            color="black",
            linestyle=FORWARD_LINESTYLE,
            linewidth=FORWARD_LINEWIDTH,
            label="Forward",
        )
    )


legend_artists = []

if SHOW_LEGEND:

    if LEGEND_BELOW_FIGURE:

        if (
            SHOW_WAVELENGTH_LEGEND
            and wavelength_handles
        ):
            wavelength_legend = fig.legend(
                handles=wavelength_handles,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.075),
                bbox_transform=fig.transFigure,
                ncol=min(
                    WAVELENGTH_LEGEND_COLUMNS,
                    len(wavelength_handles),
                ),
                frameon=False,
                fontsize=LEGEND_FONT_SIZE,
                title=r"Design $\lambda$ [nm]",
                title_fontsize=LEGEND_FONT_SIZE,
                handlelength=1.2,
                handletextpad=0.35,
                columnspacing=0.8,
            )

            legend_artists.append(
                wavelength_legend
            )

        if (
            SHOW_DIRECTION_LEGEND
            and direction_handles
        ):
            direction_legend = fig.legend(
                handles=direction_handles,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.015),
                bbox_transform=fig.transFigure,
                ncol=len(direction_handles),
                frameon=False,
                fontsize=LEGEND_FONT_SIZE,
                handlelength=1.8,
                handletextpad=0.4,
                columnspacing=1.2,
            )

            legend_artists.append(
                direction_legend
            )

    else:

        combined_handles = []

        if SHOW_WAVELENGTH_LEGEND:
            combined_handles.extend(
                wavelength_handles
            )

        if SHOW_DIRECTION_LEGEND:
            combined_handles.extend(
                direction_handles
            )

        if combined_handles:
            inside_legend = active_axes[0].legend(
                handles=combined_handles,
                loc="best",
                frameon=False,
                fontsize=LEGEND_FONT_SIZE,
                ncol=2,
            )

            legend_artists.append(
                inside_legend
            )


# ============================================================
# FINAL LAYOUT
# ============================================================

if (
    SHOW_LEGEND
    and LEGEND_BELOW_FIGURE
):
    bottom_margin = 0.3
else:
    bottom_margin = 0.15

if use_stacked_layout:

    fig.subplots_adjust(
        left=0.16,
        right=0.98,
        top=0.98,
        bottom=bottom_margin,
        hspace=0.16,
    )

else:

    fig.subplots_adjust(
        left=0.16,
        right=0.98,
        top=0.97,
        bottom=bottom_margin,
    )


# ============================================================
# SAVE AND DISPLAY
# ============================================================

if SPECTRA_TO_PLOT == "both":
    output_suffix = (
        f"both_{BOTH_LAYOUT}"
    )
else:
    output_suffix = SPECTRA_TO_PLOT

output_basename = (
    f"{FIGURE_BASENAME}_{output_suffix}"
)

if SAVE_FIGURE:

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    if SAVE_PNG:

        png_path = (
            OUTPUT_DIRECTORY
            / f"{output_basename}.png"
        )

        fig.savefig(
            png_path,
            dpi=PNG_DPI,
            transparent=TRANSPARENT_BACKGROUND,
            bbox_inches=None,
            pad_inches=0.0,
        )

        print("Saved:", png_path)

    if SAVE_PDF:

        pdf_path = (
            OUTPUT_DIRECTORY
            / f"{output_basename}.pdf"
        )

        fig.savefig(
            pdf_path,
            transparent=TRANSPARENT_BACKGROUND,
            bbox_inches=None,
            pad_inches=0.0,
        )

        print("Saved:", pdf_path)


plt.show()


# In[32]:


# ============================================================
# BACKWARD-COLLECTED AND TOTAL-SCATTERING SPECTRA
#
# Required existing objects:
#     best_cases_backward
#     spectral_records
#
# Solid lines  : backward-collected efficiency
# Dashed lines : total scattering efficiency Q_sca
# Colours      : same design colours as spectral_records
# ============================================================

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from matplotlib.lines import Line2D


# ============================================================
# STYLE
# ============================================================

mpl.rcdefaults()

plt.style.use(
    "/mnt/c/Users/SchwarzN/"
    "OneDrive - Université de Fribourg/"
    "Institution/Other/Scripts_and_HowTo/"
    "science_paper.mplstyle"
)

mpl.rcParams["figure.constrained_layout.use"] = True
mpl.rcParams["figure.autolayout"] = False


# ============================================================
# USER SETTINGS
# ============================================================

# Curves to display
PLOT_BACKWARD_COLLECTION = True
PLOT_TOTAL_SCATTERING = False
SHOW_DESIGN_WAVELENGTH_LINES = False

# Axis limits. Use None for automatic limits.
X_LOWER_LIMIT_NM = np.asarray(best_cases_backward[design_wavelength_nm]["data"].get("wavelengths_int_nm",backward_data["wavelengths_nm"],),dtype=float,).ravel()[0]
X_LIMITS_NM = (X_LOWER_LIMIT_NM, target_wavelengths[-1]*1.01)
# X_LIMITS_NM = None
# Y_LIMITS = None
# Example:
Y_LIMITS = (0.0, 0.54)

# Labels. Use None or "" to hide.
X_LABEL = "Wavelength [nm]"
Y_LABEL = "Q$_{\mathrm{back}}$\n[-]"

# Line appearance
BACKWARD_LINESTYLE = "-"
TOTAL_SCATTERING_LINESTYLE = "-."

BACKWARD_LINEWIDTH = 1.5
TOTAL_SCATTERING_LINEWIDTH = 1.4

BACKWARD_ALPHA = 1.0
TOTAL_SCATTERING_ALPHA = 0.85

# Draw vertical lines at the design wavelengths
SHOW_DESIGN_WAVELENGTH_LINES = True

DESIGN_LINESTYLE = "--"
DESIGN_LINEWIDTH = 1.0
DESIGN_LINE_ALPHA = 0.75

# Draw vertical lines at the design wavelengths
SHOW_DESIGN_WAVELENGTH_LINES = True

# Legend
SHOW_LEGEND = False
LEGEND_BELOW_FIGURE = True
LEGEND_COLUMNS = 4
LEGEND_FONT_SIZE = 9

# Grid
SHOW_GRID = False

# Exact figure dimensions
FIGURE_WIDTH_CM = 15.92/2
FIGURE_HEIGHT_CM = 15.92/4

# Saving
SAVE_FIGURE = True
SAVE_PNG = True
SAVE_PDF = True

OUTPUT_DIRECTORY = Path(
    "/mnt/c/Users/SchwarzN/"
    "OneDrive - Université de Fribourg/"
    "Institution/P1_BraggSpericalPigments/"
    "Simulation/MIE_Computation/Scattnlay_BraggOnion/outputs/run_secondary_sweep_02/"
)

FIGURE_BASENAME = (
    "Figure_2B_backward_scattering"
)

PNG_DPI = 600
TRANSPARENT_BACKGROUND = True

default_marker_dict = {
    380.0: "o",
    470.0: "s",
    540.0: "D",
    580.0: "^",
    620.0: "v",
    680.0: "p",
    760.0: "h",
    850.0: "<",
}

# ============================================================
# VALIDATION
# ============================================================


def interpolate_nans_for_plot(
    wavelengths_nm,
    spectrum,
):
    wavelengths_nm = np.asarray(
        wavelengths_nm,
        dtype=float,
    )

    spectrum = np.asarray(
        spectrum,
        dtype=float,
    ).copy()

    finite = np.isfinite(spectrum)

    if finite.sum() < 2:
        return spectrum

    spectrum[~finite] = np.interp(
        wavelengths_nm[~finite],
        wavelengths_nm[finite],
        spectrum[finite],
    )

    return spectrum

if "best_cases_backward" not in globals():
    raise NameError(
        "`best_cases_backward` is missing."
    )

if "spectral_records" not in globals():
    raise NameError(
        "`spectral_records` is missing."
    )

if not (
    PLOT_BACKWARD_COLLECTION
    or PLOT_TOTAL_SCATTERING
):
    raise ValueError(
        "At least one spectrum must be enabled."
    )

target_wavelengths = sorted(
    set(best_cases_backward)
    & set(spectral_records)
)

if not target_wavelengths:
    raise ValueError(
        "No matching cases exist in best_cases_backward "
        "and spectral_records."
    )


# ============================================================
# CREATE FIGURE
# ============================================================

cm_to_inches = 1.0 / 2.54

fig, ax = plt.subplots(
    figsize=(
        FIGURE_WIDTH_CM * cm_to_inches,
        FIGURE_HEIGHT_CM * cm_to_inches,
    ),
    # constrained_layout=False,
)

# fig.set_layout_engine(None)


# ============================================================
# PLOT SPECTRA
# ============================================================

zorder_backward = 3
for design_wavelength_nm in target_wavelengths:

    backward_data = best_cases_backward[
        design_wavelength_nm
    ]["data"]

    spectral_record = spectral_records[
        design_wavelength_nm
    ]

    wavelengths_nm = np.asarray(
        backward_data.get(
            "wavelengths_int_nm",
            backward_data["wavelengths_nm"],
        ),
        dtype=float,
    ).ravel()

    q_backward = np.asarray(
        backward_data["q_collected_geom"],
        dtype=float,
    ).ravel()

    qsca = np.asarray(
        backward_data["qsca"],
        dtype=float,
    ).ravel()

    q_backward_plot = interpolate_nans_for_plot(
        wavelengths_nm,
        q_backward,
    )

    line_colour = spectral_record[
        "line_colour"
    ]

    if not (
        len(wavelengths_nm)
        == len(q_backward)
        == len(qsca)
    ):
        raise ValueError(
            f"Array-length mismatch for the "
            f"{design_wavelength_nm:.0f} nm design."
        )

    if PLOT_BACKWARD_COLLECTION:

        ax.plot(
            wavelengths_nm,
            q_backward_plot,
            color=line_colour,
            linestyle=BACKWARD_LINESTYLE,
            linewidth=BACKWARD_LINEWIDTH,
            alpha=BACKWARD_ALPHA,
            zorder=zorder_backward,
        )
    zorder_backward += 1

    if PLOT_TOTAL_SCATTERING:

        ax.plot(
            wavelengths_nm,
            qsca,
            color=line_colour,
            linestyle=TOTAL_SCATTERING_LINESTYLE,
            linewidth=TOTAL_SCATTERING_LINEWIDTH,
            alpha=TOTAL_SCATTERING_ALPHA,
            zorder=2,
        )

# ============================================================
# DESIGN WAVELENGTHS
# ============================================================
if SHOW_DESIGN_WAVELENGTH_LINES:

    for design_wavelength_nm in target_wavelengths:

        line_colour = spectral_records[
            design_wavelength_nm
        ]["line_colour"]

        ax.axvline(
            float(design_wavelength_nm),
            color=line_colour,
            linestyle=DESIGN_LINESTYLE,
            linewidth=DESIGN_LINEWIDTH,
            alpha=DESIGN_LINE_ALPHA,
            zorder=1,
        )

        # Label above the vertical line.
        # x is in data coordinates, y is in axes coordinates.
        ax.text(
            float(design_wavelength_nm)-35,
            0.98,
            f"{design_wavelength_nm:.0f} nm",
            transform=ax.get_xaxis_transform(),
            color=line_colour,
            ha="center",
            va="bottom",
            rotation=-30,
            fontsize=8,
            clip_on=False,
            zorder=5,
        )

# ============================================================
# ROBUST PEAK MARKERS AND HORIZONTAL ARROWS
#
# The Savitzky-Golay-smoothed spectrum is used only to identify
# the representative broad maximum. The plotted spectra remain
# unchanged.
# ============================================================

import numpy as np

from scipy.signal import savgol_filter


# ============================================================
# PEAK-DETECTION SETTINGS
# ============================================================

# Window width in wavelength points.
# For a 1 nm grid, 11 points correspond to approximately 11 nm.
SMOOTHING_WINDOW_POINTS = 51

# Usually 2 or 3 is appropriate.
SMOOTHING_POLYNOMIAL_ORDER = 3

# Optional search interval for peak detection.
# Use None to search the complete plotted wavelength interval.
PEAK_SEARCH_LIMITS_NM = None

# Example:
# PEAK_SEARCH_LIMITS_NM = (400.0, 780.0)

# If True, place the marker at the smoothed peak height.
# If False, place it on the original curve at the wavelength of
# the smoothed maximum.
MARKER_ON_SMOOTHED_CURVE = True


# ============================================================
# HELPER
# ============================================================

def smooth_spectrum_for_peak_detection(
    spectrum,
    requested_window,
    polynomial_order,
):
    """
    Smooth a one-dimensional spectrum with a valid odd window size.
    """

    spectrum = np.asarray(
        spectrum,
        dtype=float,
    ).ravel()

    if len(spectrum) < 3:
        return spectrum.copy()

    # The window must be odd and no larger than the data array.
    window = min(
        int(requested_window),
        len(spectrum),
    )

    if window % 2 == 0:
        window -= 1

    # The window must be larger than the polynomial order.
    minimum_window = int(
        polynomial_order
    ) + 2

    if minimum_window % 2 == 0:
        minimum_window += 1

    window = max(
        window,
        minimum_window,
    )

    if window > len(spectrum):

        window = (
            len(spectrum)
            if len(spectrum) % 2 == 1
            else len(spectrum) - 1
        )

    if window <= polynomial_order:
        return spectrum.copy()

    return savgol_filter(
        spectrum,
        window_length=window,
        polyorder=int(polynomial_order),
        mode="interp",
    )


# ============================================================
# PEAK MARKERS AND ARROWS
# ============================================================

zorder_arrow = 3
for design_wavelength_nm in target_wavelengths:

    backward_data = best_cases_backward[
        design_wavelength_nm
    ]["data"]

    wavelengths_nm = np.asarray(
        backward_data.get(
            "wavelengths_int_nm",
            backward_data["wavelengths_nm"],
        ),
        dtype=float,
    ).ravel()

    q_backward = np.asarray(
        backward_data["q_collected_geom"],
        dtype=float,
    ).ravel()

    line_colour = spectral_records[
        design_wavelength_nm
    ]["line_colour"]

    # Fill deliberately masked numerical points before smoothing.
    q_backward_plot = interpolate_nans_for_plot(
        wavelengths_nm,
        q_backward,
    )

    # Smooth only for robust peak identification.
    q_backward_smooth = (
        smooth_spectrum_for_peak_detection(
            q_backward_plot,
            requested_window=(
                SMOOTHING_WINDOW_POINTS
            ),
            polynomial_order=(
                SMOOTHING_POLYNOMIAL_ORDER
            ),
        )
    )

    # Restrict the peak search if requested.
    if PEAK_SEARCH_LIMITS_NM is None:

        search_mask = np.ones(
            len(wavelengths_nm),
            dtype=bool,
        )

    else:

        search_mask = (
            wavelengths_nm
            >= float(
                PEAK_SEARCH_LIMITS_NM[0]
            )
        ) & (
            wavelengths_nm
            <= float(
                PEAK_SEARCH_LIMITS_NM[1]
            )
        )

    if not search_mask.any():
        raise ValueError(
            "No wavelengths fall inside "
            f"PEAK_SEARCH_LIMITS_NM={PEAK_SEARCH_LIMITS_NM}."
        )

    search_indices = np.flatnonzero(
        search_mask
    )

    local_peak_index = int(
        np.nanargmax(
            q_backward_smooth[
                search_mask
            ]
        )
    )

    peak_index = int(
        search_indices[
            local_peak_index
        ]
    )

    peak_wavelength_nm = float(
        wavelengths_nm[
            peak_index
        ]
    )

    if MARKER_ON_SMOOTHED_CURVE:

        peak_value = float(
            q_backward_smooth[
                peak_index
            ]
        )

    else:

        peak_value = float(
            q_backward_plot[
                peak_index
            ]
        )

    marker = default_marker_dict.get(
        float(design_wavelength_nm),
        "o",
    )

    # --------------------------------------------------------
    # Peak marker
    # --------------------------------------------------------

    ax.plot(
        peak_wavelength_nm,
        peak_value,
        marker=marker,
        markersize=6,
        markerfacecolor=line_colour,
        markeredgecolor="black",
        markeredgewidth=0.8,
        linestyle="None",
        zorder=zorder_arrow,
    )

    # --------------------------------------------------------
    # Horizontal arrow from design wavelength to robust peak
    # --------------------------------------------------------

    ax.annotate(
        "",
        xy=(
            peak_wavelength_nm,
            peak_value,
        ),
        xytext=(
            float(design_wavelength_nm),
            peak_value,
        ),
        arrowprops={
            "arrowstyle": "-|>",
            "edgecolor": "black",
            "fc": "black",
            "linewidth": 0.75,
            "alpha": 0.75,
            "shrinkA": 0,
            "shrinkB": 4,
        },
        zorder=zorder_arrow,
    )
    zorder_arrow += 1

# ============================================================
# AXES
# ============================================================

if X_LIMITS_NM is not None:
    ax.set_xlim(*X_LIMITS_NM)

if Y_LIMITS is not None:
    ax.set_ylim(*Y_LIMITS)
else:
    ax.set_ylim(bottom=0.0)

ax.set_yticks(np.array([0.0, 0.5]))
if X_LABEL:
    ax.set_xlabel(X_LABEL)

if Y_LABEL:
    ax.set_ylabel(Y_LABEL)
    ax.yaxis.set_label_coords(-0.03, 0.5)

ax.grid(
    SHOW_GRID,
    which="both",
    axis="both",
)


# ============================================================
# LEGEND
# ============================================================

wavelength_handles = [
    Line2D(
        [0],
        [0],
        color=spectral_records[
            design_wavelength_nm
        ]["line_colour"],
        linestyle="-",
        linewidth=2.0,
        label=f"{design_wavelength_nm:.0f}",
    )
    for design_wavelength_nm in target_wavelengths
]

quantity_handles = []

if PLOT_BACKWARD_COLLECTION:

    quantity_handles.append(
        Line2D(
            [0],
            [0],
            color="black",
            linestyle=BACKWARD_LINESTYLE,
            linewidth=BACKWARD_LINEWIDTH,
            label=r"$Q_{\mathrm{back}}$",
        )
    )

if PLOT_TOTAL_SCATTERING:

    quantity_handles.append(
        Line2D(
            [0],
            [0],
            color="black",
            linestyle=TOTAL_SCATTERING_LINESTYLE,
            linewidth=TOTAL_SCATTERING_LINEWIDTH,
            label=r"$Q_{\mathrm{sca}}$",
        )
    )


legend_artists = []

if SHOW_LEGEND and LEGEND_BELOW_FIGURE:

    wavelength_legend = fig.legend(
        handles=wavelength_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.075),
        bbox_transform=fig.transFigure,
        ncol=min(
            LEGEND_COLUMNS,
            len(wavelength_handles),
        ),
        frameon=False,
        # fontsize=LEGEND_FONT_SIZE,
        title=r"Design $\lambda$ [nm]",
        # title_fontsize=LEGEND_FONT_SIZE,
        handlelength=1.2,
        handletextpad=0.35,
        columnspacing=0.8,
        labelspacing=0.25,
    )

    legend_artists.append(
        wavelength_legend
    )

    quantity_legend = fig.legend(
        handles=quantity_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        bbox_transform=fig.transFigure,
        ncol=len(quantity_handles),
        frameon=False,
        # fontsize=LEGEND_FONT_SIZE,
        handlelength=1.8,
        handletextpad=0.4,
        columnspacing=1.2,
    )

    legend_artists.append(
        quantity_legend
    )

elif SHOW_LEGEND:

    ax.legend(
        handles=(
            wavelength_handles
            + quantity_handles
        ),
        loc="best",
        frameon=False,
        # fontsize=LEGEND_FONT_SIZE,
        ncol=2,
    )


# ============================================================
# LAYOUT
# ============================================================

bottom_margin = (
    0.25
    if SHOW_LEGEND and LEGEND_BELOW_FIGURE
    else 0.15
)

# fig.subplots_adjust(
#     left=0.16,
#     right=0.98,
#     top=0.97,
#     bottom=bottom_margin,
# )
fig.set_constrained_layout_pads(
    w_pad=0.02,
    h_pad=0.02,
    wspace=0.02,
    hspace=0.02,
)

# ============================================================
# REDUCE THE RIGHT FIGURE MARGIN
# ============================================================

# First let constrained layout calculate the positions of all artists.
fig.canvas.draw()

# Store the position produced by constrained layout.
axes_position = ax.get_position()

# Desired right edge in normalized figure coordinates.
# 1.0 is the absolute figure edge.
RIGHT_FIGURE_EDGE = 0.995

# Freeze the current constrained-layout result so that Matplotlib
# does not overwrite the manual axis-position adjustment at save time.
fig.set_layout_engine(None)

# Extend only the right edge of the plotting area.
ax.set_position(
    [
        axes_position.x0,
        axes_position.y0,
        RIGHT_FIGURE_EDGE - axes_position.x0,
        axes_position.height,
    ]
)

# Render the final positions before saving.
fig.canvas.draw()

# ============================================================
# SAVE AND DISPLAY
# ============================================================

if SAVE_FIGURE:

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    if SAVE_PNG:

        png_path = (
            OUTPUT_DIRECTORY
            / f"{FIGURE_BASENAME}.png"
        )

        fig.savefig(
            png_path,
            dpi=PNG_DPI,
            transparent=TRANSPARENT_BACKGROUND,
            bbox_inches=None,
            pad_inches=0.0,
        )

        print("Saved:", png_path)

    if SAVE_PDF:

        pdf_path = (
            OUTPUT_DIRECTORY
            / f"{FIGURE_BASENAME}.pdf"
        )

        fig.savefig(
            pdf_path,
            transparent=TRANSPARENT_BACKGROUND,
            bbox_inches=None,
            pad_inches=0.0,
        )

        print("Saved:", pdf_path)

plt.show()


# Figure 2c:
#
# Sphere material properties and geometry

# In[33]:


# ============================================================
# FIGURE 2C: PARAMETERS OF THE SELECTED BEST CASES
#
# For each target wavelength:
#   Bar 1: sphere diameter, annotated with number of shells
#          including the core, n_shells = n_layers
#   Bar 2: refractive-index ratio n_A / n_B
#   Bar 3: extinction coefficient k
#
# Each bar family has its own independent, hidden y-axis.
# Actual values are written on or above the bars.
#
# Bottom row:
#   calculated colour rectangle
#   target design wavelength
#   calculated CIELAB hue angle
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from matplotlib.patches import Rectangle, Patch


# ============================================================
# STYLE
# ============================================================

mpl.rcdefaults()

plt.style.use(
    "/mnt/c/Users/SchwarzN/"
    "OneDrive - Université de Fribourg/"
    "Institution/Other/Scripts_and_HowTo/"
    "science_paper.mplstyle"
)

mpl.rcParams["figure.constrained_layout.use"] = True


# ============================================================
# USER SETTINGS
# ============================================================

# Colour shown in the bottom rectangles.
HEX_COLUMN = "hex"
# Alternative:
# HEX_COLUMN = "hex_display"

# Bar colours
DIAMETER_COLOUR = "black" # "#777777"
RI_RATIO_COLOUR = "#888888" # #FBC15E

# Hatch patterns for extinction bars
ABSORBER_HATCHES = {
    "low_index": "///",
    "all_layers": "\\\\\\",
    "core": "xxx",
    "outer_shell": "...",
}

ABSORBER_LABELS = {
    "low_index": "LI",
    "all_layers": "All",
    "core": "Core",
    "outer_shell": "Outer shell",
    "none": "No absorber",
}

# Extinction bars have a blank background.
EXTINCTION_FACE_COLOUR = "white"
EXTINCTION_EDGE_COLOUR = "black"
EXTINCTION_HATCH_LINEWIDTH = 0.8

# Bar geometry
BAR_WIDTH = 0.22
BAR_EDGE_COLOUR = "black"
BAR_EDGE_WIDTH = 0.6

# Horizontal offsets within each target-wavelength group
DIAMETER_OFFSET = -0.25
RI_RATIO_OFFSET = 0.00
EXTINCTION_OFFSET = 0.25

# Annotation formatting
DIAMETER_DECIMALS = 1
RI_RATIO_DECIMALS = 2
K_SIGNIFICANT_DIGITS = 2
ANNOTATION_FONT_SIZE = 8

# True writes values vertically, which is useful for compact figures.
ROTATE_BAR_ANNOTATIONS = True

# Independent axis headroom
DIAMETER_HEADROOM = 1.22
RI_RATIO_HEADROOM = 1.22
EXTINCTION_HEADROOM = 1.28

# Optional manual limits. These axes remain hidden.
DIAMETER_Y_LIMITS_UM = None
RI_RATIO_Y_LIMITS = None
EXTINCTION_Y_LIMITS = None

# Examples:
# DIAMETER_Y_LIMITS_UM = (0.0, 45.0)
# RI_RATIO_Y_LIMITS = (0.0, 2.5)
# EXTINCTION_Y_LIMITS = (0.0, 0.015)

# Colour-strip geometry
COLOUR_RECTANGLE_WIDTH = 0.82
COLOUR_RECTANGLE_HEIGHT = 0.62

# Figure size
FIGURE_WIDTH_CM = 15.92 / 2
FIGURE_HEIGHT_CM = 15.92 / 4

# Legend
SHOW_LEGEND = True
LEGEND_FONT_SIZE = 9
LEGEND_COLUMNS = 3

# Saving
SAVE_FIGURE = True
SAVE_PNG = True
SAVE_PDF = True

OUTPUT_DIRECTORY = Path(
    "/mnt/c/Users/SchwarzN/"
    "OneDrive - Université de Fribourg/"
    "Institution/P1_BraggSpericalPigments/"
    "Simulation/MIE_Computation/"
    "Scattnlay_BraggOnion/outputs/"
    "run_secondary_sweep_02/"
)

FIGURE_BASENAME = (
    "Figure_2C_selected_case_parameters"
)

PNG_DPI = 600
TRANSPARENT_BACKGROUND = True

# Rösch–MacAdam maximum-chroma reference
ROSCH_CSV = Path(
    "/mnt/c/Users/SchwarzN/"
    "OneDrive - Université de Fribourg/"
    "Institution/P1_BraggSpericalPigments/"
    "Simulation/MIE_Computation/"
    "Scattnlay_BraggOnion/src/bragg_onion/"
    "rosch_macadam_max_chroma_per_hue_1deg.csv"
)

ROSCH_HUE_COLUMN = "hue_deg"
ROSCH_HEX_COLUMN = "hex_smooth"

# Reduced distance between neighbouring design groups.
# Smaller values place the groups closer together.
CASE_SPACING = 0.85

# Rectangle geometry
COLOUR_RECTANGLE_WIDTH = 0.72
COLOUR_RECTANGLE_HEIGHT = 0.30

# Vertical positions of the two colour rows
CALCULATED_RECTANGLE_Y = 0.08
OPTIMAL_RECTANGLE_Y = 0.52

# Labels at the left of the two colour rows
SHOW_COLOUR_ROW_LABELS = True
COLOUR_ROW_LABEL_FONT_SIZE = 8

# Minimum visible height used only to reveal the hatch pattern.
# The numerical annotation still reports the true value.
MIN_VISIBLE_K_BAR_FRACTION = 0.05

# ============================================================
# PREPARE DATA
# ============================================================

required_columns = {
    "design_peak_wavelength_nm",
    "n_A",
    "n_B",
    "n_layers",
    "absorber_location",
    "k",
    HEX_COLUMN,
}

missing_columns = (
    required_columns
    - set(df_best_cases.columns)
)

if missing_columns:
    raise KeyError(
        "df_best_cases is missing required columns: "
        f"{sorted(missing_columns)}"
    )


df_parameter_panel = (
    df_best_cases
    .sort_values("design_peak_wavelength_nm")
    .reset_index(drop=True)
    .copy()
)

# ============================================================
# LOAD THE RÖSCH–MACADAM HUE REFERENCE
# ============================================================

if not ROSCH_CSV.exists():
    raise FileNotFoundError(
        "Rösch–MacAdam CSV was not found at:\n"
        f"{ROSCH_CSV}"
    )

df_rosch_reference = pd.read_csv(
    ROSCH_CSV
)

required_rosch_columns = {
    ROSCH_HUE_COLUMN,
    ROSCH_HEX_COLUMN,
}

missing_rosch_columns = (
    required_rosch_columns
    - set(df_rosch_reference.columns)
)

if missing_rosch_columns:
    raise KeyError(
        "The Rösch–MacAdam CSV is missing columns: "
        f"{sorted(missing_rosch_columns)}"
    )

df_rosch_reference[
    ROSCH_HUE_COLUMN
] = pd.to_numeric(
    df_rosch_reference[
        ROSCH_HUE_COLUMN
    ],
    errors="coerce",
)

df_rosch_reference = (
    df_rosch_reference
    .dropna(
        subset=[
            ROSCH_HUE_COLUMN,
            ROSCH_HEX_COLUMN,
        ]
    )
    .sort_values(
        ROSCH_HUE_COLUMN
    )
    .reset_index(drop=True)
)

invalid_rosch_colours = ~df_rosch_reference[
    ROSCH_HEX_COLUMN
].map(
    mpl.colors.is_color_like
)

if invalid_rosch_colours.any():
    raise ValueError(
        "The Rösch–MacAdam CSV contains invalid colours "
        f"in {ROSCH_HEX_COLUMN!r}."
    )


def circular_hue_distance_deg(
    hue_1_deg,
    hue_2_deg,
):
    """
    Return the shortest angular distance between hue angles.
    """
    return np.abs(
        (
            np.asarray(
                hue_1_deg,
                dtype=float,
            )
            - np.asarray(
                hue_2_deg,
                dtype=float,
            )
            + 180.0
        )
        % 360.0
        - 180.0
    )


def nearest_rosch_colour(
    target_hue_deg,
):
    """
    Return the Rösch–MacAdam row nearest to the requested
    CIELAB hue angle.
    """

    hue_distances = circular_hue_distance_deg(
        df_rosch_reference[
            ROSCH_HUE_COLUMN
        ].to_numpy(dtype=float),
        float(target_hue_deg),
    )

    nearest_index = int(
        np.nanargmin(
            hue_distances
        )
    )

    return df_rosch_reference.iloc[
        nearest_index
    ]


# Determine the hue-matched optimum for every selected design.
rosch_matches = [
    nearest_rosch_colour(
        hue_deg
    )
    for hue_deg in df_parameter_panel[
        "hue_deg"
    ]
]

df_parameter_panel[
    "rosch_hue_deg"
] = [
    float(
        match[
            ROSCH_HUE_COLUMN
        ]
    )
    for match in rosch_matches
]

df_parameter_panel[
    "rosch_hex"
] = [
    str(
        match[
            ROSCH_HEX_COLUMN
        ]
    )
    for match in rosch_matches
]

df_parameter_panel[
    "rosch_hue_difference_deg"
] = [
    float(
        circular_hue_distance_deg(
            calculated_hue,
            reference_hue,
        )
    )
    for calculated_hue, reference_hue in zip(
        df_parameter_panel[
            "hue_deg"
        ],
        df_parameter_panel[
            "rosch_hue_deg"
        ],
    )
]

print(
    "Maximum hue difference to the nearest "
    "Rösch–MacAdam reference point:",
    f"{df_parameter_panel['rosch_hue_difference_deg'].max():.3f}°",
)

# ------------------------------------------------------------
# Diameter in micrometres
# ------------------------------------------------------------

if "diameter_nm" in df_parameter_panel.columns:

    df_parameter_panel["diameter_um"] = (
        pd.to_numeric(
            df_parameter_panel["diameter_nm"],
            errors="coerce",
        )
        / 1000.0
    )

elif "diameter_m" in df_parameter_panel.columns:

    df_parameter_panel["diameter_um"] = (
        pd.to_numeric(
            df_parameter_panel["diameter_m"],
            errors="coerce",
        )
        * 1e6
    )

elif "outer_radius_m" in df_parameter_panel.columns:

    df_parameter_panel["diameter_um"] = (
        2.0
        * pd.to_numeric(
            df_parameter_panel["outer_radius_m"],
            errors="coerce",
        )
        * 1e6
    )

elif "outer_radius_nm" in df_parameter_panel.columns:

    df_parameter_panel["diameter_um"] = (
        2.0
        * pd.to_numeric(
            df_parameter_panel["outer_radius_nm"],
            errors="coerce",
        )
        / 1000.0
    )

else:
    raise KeyError(
        "No diameter information was found. Expected one of "
        "'diameter_nm', 'diameter_m', 'outer_radius_m', "
        "or 'outer_radius_nm'."
    )


# ------------------------------------------------------------
# Number of shells including the core
# ------------------------------------------------------------
#
# The geometry class currently calls this n_layers, but physically
# this is the total number of A/B shells including the core.

df_parameter_panel["n_shells"] = pd.to_numeric(
    df_parameter_panel["n_layers"],
    errors="coerce",
)


# ------------------------------------------------------------
# Refractive-index ratio
# ------------------------------------------------------------

df_parameter_panel["n_ratio"] = (
    pd.to_numeric(
        df_parameter_panel["n_A"],
        errors="coerce",
    )
    / pd.to_numeric(
        df_parameter_panel["n_B"],
        errors="coerce",
    )
)


# ------------------------------------------------------------
# Extinction coefficient
# ------------------------------------------------------------

df_parameter_panel["k_plot"] = (
    1e3
    * pd.to_numeric(
        df_parameter_panel["k"],
        errors="coerce",
    ).fillna(0.0)
)

nonabsorbing_mask = (
    df_parameter_panel["absorber_location"]
    .astype(str)
    .eq("none")
)

# No extinction bar is drawn for nonabsorbing cases.
df_parameter_panel.loc[
    nonabsorbing_mask,
    "k_plot",
] = 0.0


# ------------------------------------------------------------
# CIELAB hue angle
# ------------------------------------------------------------

if "hue_deg" in df_parameter_panel.columns:

    df_parameter_panel["hue_plot_deg"] = np.mod(
        pd.to_numeric(
            df_parameter_panel["hue_deg"],
            errors="coerce",
        ),
        360.0,
    )

elif {
    "a",
    "b",
}.issubset(df_parameter_panel.columns):

    df_parameter_panel["hue_plot_deg"] = np.mod(
        np.degrees(
            np.arctan2(
                pd.to_numeric(
                    df_parameter_panel["b"],
                    errors="coerce",
                ),
                pd.to_numeric(
                    df_parameter_panel["a"],
                    errors="coerce",
                ),
            )
        ),
        360.0,
    )

else:
    raise KeyError(
        "Hue annotation requires either 'hue_deg' or "
        "both 'a' and 'b'."
    )


# ------------------------------------------------------------
# Validate colours
# ------------------------------------------------------------

invalid_hex_mask = ~df_parameter_panel[
    HEX_COLUMN
].map(mpl.colors.is_color_like)

if invalid_hex_mask.any():
    raise ValueError(
        "Invalid colours found in "
        f"{HEX_COLUMN!r}:\n"
        f"{df_parameter_panel.loc[invalid_hex_mask, HEX_COLUMN]}"
    )


# ============================================================
# CREATE FIGURE
# ============================================================

cm_to_inches = 1.0 / 2.54

fig = plt.figure(
    figsize=(
        FIGURE_WIDTH_CM * cm_to_inches,
        FIGURE_HEIGHT_CM * cm_to_inches,
    ),
    constrained_layout=True,
)

# fig.set_layout_engine(None)

grid_spec = fig.add_gridspec(
    nrows=2,
    ncols=1,
    height_ratios=[
        4.2,
        1.25,
    ],
    hspace=0.0,
)

# Main axis for the diameter bars
ax_diameter = fig.add_subplot(
    grid_spec[0]
)

# Overlaid axes with independent scales
ax_ri = ax_diameter.twinx()
ax_k = ax_diameter.twinx()

# Colour-strip axis
ax_colour = fig.add_subplot(
    grid_spec[1],
    sharex=ax_diameter,
)


# ============================================================
# X POSITIONS
# ============================================================

n_cases = len(df_parameter_panel)

x_positions = (
    np.arange(
        n_cases,
        dtype=float,
    )
    * CASE_SPACING
)

diameter_x = (
    x_positions
    + DIAMETER_OFFSET
)

ri_x = (
    x_positions
    + RI_RATIO_OFFSET
)

k_x = (
    x_positions
    + EXTINCTION_OFFSET
)


# ============================================================
# INDEPENDENT HIDDEN Y-AXIS LIMITS
# ============================================================

diameter_values = df_parameter_panel[
    "diameter_um"
].to_numpy(dtype=float)

ri_values = df_parameter_panel[
    "n_ratio"
].to_numpy(dtype=float)

k_values = df_parameter_panel[
    "k_plot"
].to_numpy(dtype=float)


if DIAMETER_Y_LIMITS_UM is None:

    diameter_upper = max(
        np.nanmax(diameter_values)
        * DIAMETER_HEADROOM,
        1.0,
    )

    ax_diameter.set_ylim(
        0.0,
        diameter_upper,
    )

else:

    ax_diameter.set_ylim(
        *DIAMETER_Y_LIMITS_UM
    )


if RI_RATIO_Y_LIMITS is None:

    ri_upper = max(
        np.nanmax(ri_values)
        * RI_RATIO_HEADROOM,
        1.0,
    )

    ax_ri.set_ylim(
        0.0,
        ri_upper,
    )

else:

    ax_ri.set_ylim(
        *RI_RATIO_Y_LIMITS
    )


maximum_k = float(
    np.nanmax(k_values)
)

if EXTINCTION_Y_LIMITS is None:

    k_upper = (
        maximum_k * EXTINCTION_HEADROOM
        if maximum_k > 0.0
        else 1.0
    )

    ax_k.set_ylim(
        0.0,
        k_upper,
    )

else:

    ax_k.set_ylim(
        *EXTINCTION_Y_LIMITS
    )


# ============================================================
# BAR 1: DIAMETER
# ANNOTATION ALSO REPORTS NUMBER OF SHELLS
# ============================================================

diameter_bars = ax_diameter.bar(
    diameter_x,
    diameter_values,
    width=BAR_WIDTH,
    color=DIAMETER_COLOUR,
    edgecolor=BAR_EDGE_COLOUR,
    linewidth=BAR_EDGE_WIDTH,
    zorder=3,
)


# ============================================================
# DIAMETER AND SHELL-COUNT ANNOTATIONS
#
# Diameter:
#   written horizontally in white inside the grey bar
#
# Number of shells:
#   written horizontally immediately above the grey bar
# ============================================================

diameter_axis_height = (
    ax_diameter.get_ylim()[1]
    - ax_diameter.get_ylim()[0]
)

shell_label_offset = (
    0.03 * diameter_axis_height
)

for bar, diameter_um, n_shells in zip(
    diameter_bars,
    diameter_values,
    df_parameter_panel["n_shells"],
):

    bar_center_x = (
        bar.get_x()
        + bar.get_width() / 2.0
    )

    bar_height = float(
        bar.get_height()
    )

    # --------------------------------------------------------
    # Diameter value inside the grey bar
    # --------------------------------------------------------

    ax_diameter.text(
        bar_center_x+0.01,

        # Place the text near the upper part of the bar.
        0.96 * bar_height,

        f"{int(round(n_shells))}",

        ha="center",
        va="top",
        rotation=90,
        fontsize=ANNOTATION_FONT_SIZE,
        color="white",
        clip_on=True,
        zorder=6,
    )

    # --------------------------------------------------------
    # Number of shells immediately above the grey bar
    # --------------------------------------------------------

    ax_diameter.text(
        bar_center_x,
        bar_height + shell_label_offset,
        f"{diameter_um:.{DIAMETER_DECIMALS}f}",
        ha="center",
        va="bottom",
        rotation=90,
        fontsize=ANNOTATION_FONT_SIZE,
        color="black",
        clip_on=False,
        zorder=6,
    )


# ============================================================
# BAR 2: REFRACTIVE-INDEX RATIO n_A / n_B
# ============================================================

ri_bars = ax_ri.bar(
    ri_x,
    ri_values,
    width=BAR_WIDTH,
    color=RI_RATIO_COLOUR,
    edgecolor=BAR_EDGE_COLOUR,
    linewidth=BAR_EDGE_WIDTH,
    zorder=4,
)

for bar, n_ratio in zip(
    ri_bars,
    ri_values,
):

    ax_ri.text(
        bar.get_x()
        + bar.get_width() / 2.0,
        bar.get_height()
        + 0.025
        * ax_ri.get_ylim()[1],
        f"{n_ratio:.{RI_RATIO_DECIMALS}f}",
        ha="center",
        va="bottom",
        rotation=(
            90
            if ROTATE_BAR_ANNOTATIONS
            else 0
        ),
        fontsize=ANNOTATION_FONT_SIZE,
        color="black",
        clip_on=False,
        zorder=6,
    )


# ============================================================
# BAR 3: EXTINCTION COEFFICIENT
# Hatch pattern indicates absorber placement
# ============================================================

for x_value, (_, row) in zip(
    k_x,
    df_parameter_panel.iterrows(),
):

    location = str(
        row["absorber_location"]
    )

    k_value = float(
        row["k_plot"]
    )

    # No bar for a nonabsorbing case.
    if (
        location == "none"
        or np.isclose(k_value, 0.0)
    ):
        continue

    hatch_pattern = ABSORBER_HATCHES.get(
        location,
        "",
    )

    k_bar = ax_k.bar(
        x_value,
        k_value,
        width=BAR_WIDTH,
        facecolor=EXTINCTION_FACE_COLOUR,
        edgecolor=EXTINCTION_EDGE_COLOUR,
        linewidth=BAR_EDGE_WIDTH,
        hatch=hatch_pattern,
        zorder=5,
    )[0]

    ax_k.text(
        k_bar.get_x()
        + k_bar.get_width() / 2.0,
        k_bar.get_height()
        + 0.025
        * ax_k.get_ylim()[1],
        f"{k_value:.2f}",
        ha="center",
        va="bottom",
        rotation=(
            90
            if ROTATE_BAR_ANNOTATIONS
            else 0
        ),
        fontsize=ANNOTATION_FONT_SIZE,
        color="black",
        clip_on=False,
        zorder=6,
    )
    # True bar height
    true_k_height = k_value
    # Minimum visible height in axis units
    minimum_visible_height = (
    MIN_VISIBLE_K_BAR_FRACTION
    * (
    ax_k.get_ylim()[1]
    - ax_k.get_ylim()[0]
    )

    )
    # if true_k_height < minimum_visible_height:

    #     # Hatch is not supported reliably by scatter markers,
    #     # so overlay a small Rectangle instead.
    #     hatch_indicator = Rectangle(
    #         (
    #             x_value - BAR_WIDTH / 2.0,
    #             6.5 * minimum_visible_height,
    #         ),
    #         BAR_WIDTH,
    #         minimum_visible_height*2,
    #         facecolor="white",
    #         edgecolor="black",
    #         linewidth=0.7,
    #         hatch=hatch_pattern,
    #         zorder=7,
    #     )

    #     ax_k.add_patch(hatch_indicator)
# ============================================================
# HIDE ALL Y AXES AND MAIN-PANEL X AXIS
# ============================================================

for axis in [
    ax_diameter,
    ax_ri,
    ax_k,
]:

    axis.set_yticks([])
    axis.set_ylabel("")

    axis.tick_params(
        axis="y",
        which="both",
        left=False,
        right=False,
        labelleft=False,
        labelright=False,
    )

    axis.tick_params(
        axis="x",
        which="both",
        bottom=False,
        top=False,
        labelbottom=False,
    )

    axis.grid(
        False,
        which="both",
        axis="both",
    )


# Hide all spines except the lower black separator.
for spine_name in [
    "left",
    "right",
    "top",
]:
    ax_diameter.spines[
        spine_name
    ].set_visible(False)

ax_diameter.spines[
    "bottom"
].set_visible(True)

ax_diameter.spines[
    "bottom"
].set_color("black")

ax_diameter.spines[
    "bottom"
].set_linewidth(1.0)


# Hide all overlaid-axis spines.
for axis in [
    ax_ri,
    ax_k,
]:
    for spine in axis.spines.values():
        spine.set_visible(False)

# ============================================================
# CALCULATED AND OPTIMAL COLOUR RECTANGLES
# ============================================================

for x_position, (_, row) in zip(
    x_positions,
    df_parameter_panel.iterrows(),
):

    # --------------------------------------------------------
    # Lower rectangle: calculated colour
    # --------------------------------------------------------

    calculated_rectangle = Rectangle(
        (
            x_position
            - COLOUR_RECTANGLE_WIDTH / 2.0,
            CALCULATED_RECTANGLE_Y,
        ),
        COLOUR_RECTANGLE_WIDTH,
        COLOUR_RECTANGLE_HEIGHT,
        facecolor=str(
            row[HEX_COLUMN]
        ),
        edgecolor="black",
        linewidth=0.65,
        zorder=3,
    )

    ax_colour.add_patch(
        calculated_rectangle
    )

    # --------------------------------------------------------
    # Upper rectangle: hue-matched Rösch–MacAdam optimum
    # --------------------------------------------------------

    optimal_rectangle = Rectangle(
        (
            x_position
            - COLOUR_RECTANGLE_WIDTH / 2.0,
            OPTIMAL_RECTANGLE_Y,
        ),
        COLOUR_RECTANGLE_WIDTH,
        COLOUR_RECTANGLE_HEIGHT,
        facecolor=str(
            row["rosch_hex"]
        ),
        edgecolor="black",
        linewidth=0.65,
        zorder=3,
    )

    ax_colour.add_patch(
        optimal_rectangle
    )


ax_colour.set_ylim(
    0.0,
    0.90,
)

ax_colour.set_yticks([])
ax_colour.set_ylabel("")

ax_colour.grid(
    False,
    which="both",
    axis="both",
)

for spine_name in [
    "left",
    "right",
    "top",
    "bottom",
]:
    ax_colour.spines[spine_name].set_visible(True)
    ax_colour.spines[spine_name].set_color("black")
    ax_colour.spines[spine_name].set_linewidth(0.5)
    ax_colour.spines[spine_name].set_zorder(10)


# ------------------------------------------------------------
# Colour-row labels at the left
# ------------------------------------------------------------

if SHOW_COLOUR_ROW_LABELS:

    ax_colour.text(
        -0.025,
        OPTIMAL_RECTANGLE_Y
        + COLOUR_RECTANGLE_HEIGHT / 2.0,
        "Locus",
        transform=ax_colour.get_yaxis_transform(),
        ha="right",
        va="center",
        fontsize=COLOUR_ROW_LABEL_FONT_SIZE,
        clip_on=False,
    )

    ax_colour.text(
        -0.025,
        CALCULATED_RECTANGLE_Y
        + COLOUR_RECTANGLE_HEIGHT / 2.0,
        "Best",
        transform=ax_colour.get_yaxis_transform(),
        ha="right",
        va="center",
        fontsize=COLOUR_ROW_LABEL_FONT_SIZE,
        clip_on=False,
    )

# ============================================================
# BOTTOM LABELS: TARGET WAVELENGTH AND CIELAB HUE
# ============================================================

bottom_labels = [
    (
        f"{float(row['design_peak_wavelength_nm']):.0f}"
        "\n"
        f"{float(row['hue_plot_deg']):.0f}°"
    )
    for _, row
    in df_parameter_panel.iterrows()
]

ax_colour.set_xticks(
    x_positions
)

ax_colour.set_xticklabels(
    bottom_labels,
    ha="center",
)

# Disable all minor ticks.
ax_colour.minorticks_off()

ax_colour.tick_params(
    axis="x",
    which="minor",
    bottom=False,
    top=False,
)

# Keep the major positions without visible tick marks.
ax_colour.tick_params(
    axis="x",
    which="major",
    bottom=True,
    top=False,
    length=0,
    width=0,
    pad=5,
)


# ------------------------------------------------------------
# Descriptor to the left of the bottom annotations
# ------------------------------------------------------------

ax_colour.text(
    -0.025,
    -0.3,
    # "Target $\lambda$ [nm]\nHue angle [°]",
    "[nm]\n[°]",
    transform=ax_colour.transAxes,
    ha="right",
    va="top",
    fontsize=ANNOTATION_FONT_SIZE,
    clip_on=False,
)


# ============================================================
# SHARED X LIMITS
# ============================================================

x_minimum = (
    x_positions[0] - 0.52
)

x_maximum = (
    x_positions[-1] + 0.52
)

for axis in [
    ax_diameter,
    ax_ri,
    ax_k,
    ax_colour,
]:
    axis.set_xlim(
        x_minimum,
        x_maximum,
    )


# ============================================================
# LEGEND
# ============================================================

legend_handles = [
    Patch(
        facecolor=DIAMETER_COLOUR,
        edgecolor=BAR_EDGE_COLOUR,
        linewidth=BAR_EDGE_WIDTH,
        label=(
            # r"$N_{\mathrm{shells}}$ |"
            "N |"
            # "\n"
            " $\u2300$ "
            r"[$\mu$m]"
        ),
    ),
    Patch(
        facecolor=RI_RATIO_COLOUR,
        edgecolor=BAR_EDGE_COLOUR,
        linewidth=BAR_EDGE_WIDTH,
        label=r"$n_A/n_B$",
    ),
]

locations_present = set(
    df_parameter_panel[
        "absorber_location"
    ].astype(str)
)

for location in [
    "low_index",
    "all_layers",
    "core",
    "outer_shell",
]:

    if location not in locations_present:
        continue

    legend_handles.append(
        Patch(
            facecolor=EXTINCTION_FACE_COLOUR,
            edgecolor=EXTINCTION_EDGE_COLOUR,
            linewidth=BAR_EDGE_WIDTH,
            hatch=ABSORBER_HATCHES.get(
                location,
                "",
            ),
            label=(
                r"$10^3 k$: "
                + ABSORBER_LABELS[
                    location
                ]
            ),
        )
    )

if SHOW_LEGEND:

    fig.legend(
        handles=legend_handles,
        loc="outside upper center",
        # bbox_to_anchor=(0.5, 1.1), # y = 1.10
        # bbox_transform=fig.transFigure,
        # ncol=min(
        #     LEGEND_COLUMNS,
        #     len(legend_handles),
        # ),
        ncol = 4,
        frameon=True,
        fontsize=LEGEND_FONT_SIZE,
        borderpad=0.1,
        handlelength=1.0,
        handletextpad=0.4,
        columnspacing=0.9,
        labelspacing=0.3,
        borderaxespad=0.0,
    )


# ============================================================
# LAYOUT
# ============================================================

fig.set_constrained_layout_pads(
    w_pad=0.02,
    h_pad=0.02,
    wspace=0.02,
    hspace=0.02,
)


# ============================================================
# SAVE AND DISPLAY
# ============================================================

if SAVE_FIGURE:

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    if SAVE_PNG:

        png_path = (
            OUTPUT_DIRECTORY
            / f"{FIGURE_BASENAME}.png"
        )

        fig.savefig(
            png_path,
            dpi=PNG_DPI,
            transparent=TRANSPARENT_BACKGROUND,
            bbox_inches=None,
            pad_inches=0.0,
        )

        print("Saved:", png_path)

    if SAVE_PDF:

        pdf_path = (
            OUTPUT_DIRECTORY
            / f"{FIGURE_BASENAME}.pdf"
        )

        fig.savefig(
            pdf_path,
            transparent=TRANSPARENT_BACKGROUND,
            bbox_inches=None,
            pad_inches=0.0,
        )

        print("Saved:", pdf_path)


plt.show()


# ============================================================
# PARAMETER SUMMARY
# ============================================================

display(
    df_parameter_panel[
        [
            "design_peak_wavelength_nm",
            "diameter_um",
            "n_shells",
            "n_A",
            "n_B",
            "n_ratio",
            "absorber_location",
            "k_plot",
            "hue_plot_deg",
            "rosch_hue_deg",
            "rosch_hue_difference_deg",
            HEX_COLUMN,
            "rosch_hex",
        ]
    ]
)


# In[ ]:


from pathlib import Path

# Save best cases to a Parquet file for future reference.

BEST_CASES_PATH = Path(
    "/mnt/c/Users/SchwarzN/"
    "OneDrive - Université de Fribourg/"
    "Institution/P1_BraggSpericalPigments/"
    "Simulation/MIE_Computation/"
    "Scattnlay_BraggOnion/outputs/"
    "run_secondary_sweep_02/"
    "df_best_cases.parquet"
)

df_best_cases.to_parquet(
    BEST_CASES_PATH,
    index=False,
)

print("Saved:", BEST_CASES_PATH)


# In[34]:


# ============================================================
# HIGH-RESOLUTION TEST AROUND THE 850 nm DESIGN SPIKE
#
# Required existing objects:
#   df_best_cases
#   build_absorber_objects
#
# This cell:
#   1. retrieves the selected 850 nm design;
#   2. reconstructs its geometry and resolved layer stack;
#   3. locates the strongest peak in the previous coarse result;
#   4. reruns the scattering solver around that peak at high resolution.
# ============================================================

import numpy as np
import pandas as pd

from bragg_onion.materials import ConstantDispersion
from bragg_onion.geometry import (
    build_bragg_onion_from_peak_wavelength,
    resolve_layer_stack,
)
from bragg_onion.solver import (
    run_scattnlay_spectrum,
)


# ============================================================
# USER SETTINGS
# ============================================================

TARGET_DESIGN_WAVELENGTH_NM = 850.0

# Spectral interval around the detected coarse peak.
PEAK_WINDOW_HALF_WIDTH_NM = 3.0

# High-resolution wavelength spacing.
HIGH_RESOLUTION_STEP_NM = 0.01

# Angular resolution from 0° to 180°.
N_THETA_POINTS = 721

N_MEDIUM = 1.0


# ============================================================
# RETRIEVE THE SELECTED 850 nm CASE
# ============================================================

selected_rows = df_best_cases.loc[
    np.isclose(
        pd.to_numeric(
            df_best_cases["design_peak_wavelength_nm"],
            errors="coerce",
        ),
        TARGET_DESIGN_WAVELENGTH_NM,
    )
]

if len(selected_rows) != 1:
    raise ValueError(
        f"Expected exactly one selected case for "
        f"{TARGET_DESIGN_WAVELENGTH_NM:.0f} nm, "
        f"but found {len(selected_rows)}."
    )

row_850 = selected_rows.iloc[0].copy()

print("Selected 850 nm design:")
display(
    row_850[
        [
            column
            for column in [
                "design_peak_wavelength_nm",
                "absorber_location",
                "n_A",
                "n_B",
                "k",
                "n_layers",
                "outer_layer",
                "core_thickness_factor",
                "eta_C",
                "eta_L",
                "d_opt",
            ]
            if column in row_850.index
        ]
    ]
)


# ============================================================
# RECONSTRUCT MATERIALS
# ============================================================

material_a_850 = ConstantDispersion.from_nk(
    "A",
    n=float(row_850["n_A"]),
    k=0.0,
)

material_b_850 = ConstantDispersion.from_nk(
    "B",
    n=float(row_850["n_B"]),
    k=0.0,
)

extinction_modifiers_850, extra_outer_shell_850 = (
    build_absorber_objects(row_850)
)


# ============================================================
# RECONSTRUCT GEOMETRY
# ============================================================

if (
    "n_layers" not in row_850.index
    or pd.isna(row_850["n_layers"])
):
    raise KeyError(
        "This test cell expects an explicit n_layers value "
        "in df_best_cases."
    )

geometry_850 = build_bragg_onion_from_peak_wavelength(
    material_a=material_a_850,
    material_b=material_b_850,
    peak_wavelength_m=float(
        row_850["peak_wavelength_m"]
    ),
    n_layers=int(row_850["n_layers"]),
    outer_layer=str(
        row_850.get("outer_layer", "A")
    ),
    core_thickness_factor=float(
        row_850.get(
            "core_thickness_factor",
            1.0,
        )
    ),
    extra_outer_shell=extra_outer_shell_850,
)

print(
    "Stored peak wavelength:",
    float(row_850["peak_wavelength_m"]) * 1e9,
    "nm",
)

print(
    "Design peak wavelength:",
    float(row_850["design_peak_wavelength_nm"]),
    "nm",
)

# ============================================================
# RESOLVE THE EXACT LAYER STACK
# ============================================================

stack_850 = resolve_layer_stack(
    geometry=geometry_850,
    material_a=material_a_850,
    material_b=material_b_850,
    extinction_modifiers=extinction_modifiers_850,
)

print("\nResolved stack:")
print("  geometry type :", type(geometry_850))
print("  stack type    :", type(stack_850))


# ============================================================
# LOCATE THE SUSPECTED PEAK FROM THE PREVIOUS RERUN
# ============================================================

if (
    "spectral_records" in globals()
    and int(TARGET_DESIGN_WAVELENGTH_NM)
    in spectral_records
):
    coarse_record = spectral_records[
        int(TARGET_DESIGN_WAVELENGTH_NM)
    ]

    coarse_wavelengths_nm = np.asarray(
        coarse_record["wavelengths_nm"],
        dtype=float,
    )

    coarse_backward_spectrum = np.asarray(
        coarse_record["q_backward_raw"],
        dtype=float,
    )

    coarse_peak_index = int(
        np.nanargmax(coarse_backward_spectrum)
    )

    suspected_peak_nm = float(
        coarse_wavelengths_nm[
            coarse_peak_index
        ]
    )

else:
    # Fallback if spectral_records is unavailable.
    suspected_peak_nm = float(
        TARGET_DESIGN_WAVELENGTH_NM
    )

print(
    "\nHigh-resolution scan center:",
    f"{suspected_peak_nm:.4f} nm",
)


# ============================================================
# HIGH-RESOLUTION GRIDS
# ============================================================

high_res_wavelengths_nm = np.arange(
    suspected_peak_nm - PEAK_WINDOW_HALF_WIDTH_NM,
    suspected_peak_nm + PEAK_WINDOW_HALF_WIDTH_NM
    + 0.5 * HIGH_RESOLUTION_STEP_NM,
    HIGH_RESOLUTION_STEP_NM,
)

high_res_theta_deg = np.linspace(
    0.0,
    180.0,
    N_THETA_POINTS,
)

high_res_theta_rad = np.deg2rad(
    high_res_theta_deg
)

print(
    "Wavelength range:",
    f"{high_res_wavelengths_nm[0]:.4f} to "
    f"{high_res_wavelengths_nm[-1]:.4f} nm",
)

print(
    "Wavelength points:",
    len(high_res_wavelengths_nm),
)

print(
    "Angular points:",
    len(high_res_theta_deg),
)


# ============================================================
# RUN HIGH-RESOLUTION SCATTERING CALCULATION
# ============================================================

scattering_result_850_high_res = (
    run_scattnlay_spectrum(
        stack=stack_850,
        wavelengths_m=(
            high_res_wavelengths_nm
            * 1e-9
        ),
        theta_rad=high_res_theta_rad,
        n_medium=N_MEDIUM,
    )
)

print("\nHigh-resolution scattering calculation completed.")


# ============================================================
# INSPECT AVAILABLE RESULT FIELDS
# ============================================================

available_result_fields = [
    name
    for name in dir(
        scattering_result_850_high_res
    )
    if not name.startswith("_")
]

print("\nAvailable ScatteringResult fields:")
print(available_result_fields)


# ============================================================
# IDENTIFY WAVELENGTH-ANGLE ARRAYS
# ============================================================
#
# This automatically lists two-dimensional arrays that could
# represent angular scattering quantities such as S11 or
# differential scattering cross section.

angular_array_candidates = {}

for field_name in available_result_fields:

    try:
        value = getattr(
            scattering_result_850_high_res,
            field_name,
        )

        array = np.asarray(value)

    except Exception:
        continue

    if array.ndim != 2:
        continue

    if array.shape == (
        len(high_res_wavelengths_nm),
        len(high_res_theta_deg),
    ):
        angular_array_candidates[
            field_name
        ] = array

    elif array.shape == (
        len(high_res_theta_deg),
        len(high_res_wavelengths_nm),
    ):
        angular_array_candidates[
            field_name
        ] = array.T


print("\nCandidate wavelength × angle arrays:")

for field_name, array in angular_array_candidates.items():
    print(
        f"  {field_name}: "
        f"shape={array.shape}, "
        f"min={np.nanmin(array):.4e}, "
        f"max={np.nanmax(array):.4e}"
    )

if not angular_array_candidates:
    raise ValueError(
        "No wavelength × polar-angle arrays were found in "
        "ScatteringResult. Inspect the printed field names."
    )


# In[35]:


# ============================================================
# PLOT HIGH-RESOLUTION SPECTRAL CONTINUITY AND ANGULAR ORIGIN
#
# Upper panel:
#   angle-integrated/angularly summed diagnostic spectrum
#
# Middle panel:
#   wavelength × polar-angle intensity map
#
# Lower panel:
#   angular profiles at the spike and on both sides
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from matplotlib.lines import Line2D


# ============================================================
# USER SETTINGS
# ============================================================

# Set to None for automatic selection.
#
# After running Cell 1, you may explicitly enter one printed field,
# for example:
#
# ANGULAR_QUANTITY_NAME = "S11"
#
ANGULAR_QUANTITY_NAME = None

# Logarithmic display is recommended for large dynamic ranges.
USE_LOG_SCALE = True

LOG_FLOOR_RELATIVE = 1e-8

# Number of neighbouring wavelength points used on either side
# of the strongest high-resolution spike.
PROFILE_OFFSET_POINTS = 10

FIGURE_SIZE_CM = (13.0, 15.0)

X_LIMITS_NM = (
    float(high_res_wavelengths_nm[0]),
    float(high_res_wavelengths_nm[-1]),
)

ANGLE_LIMITS_DEG = (0.0, 180.0)


# ============================================================
# SELECT ANGULAR QUANTITY
# ============================================================

preferred_angular_fields = [
    "differential_cross_section_m2_sr",
    "differential_cross_section",
    "S11",
    "s11",
    "intensity",
    "angular_intensity",
]


if ANGULAR_QUANTITY_NAME is not None:

    if (
        ANGULAR_QUANTITY_NAME
        not in angular_array_candidates
    ):
        raise KeyError(
            f"{ANGULAR_QUANTITY_NAME!r} was not found.\n"
            f"Available candidates: "
            f"{list(angular_array_candidates)}"
        )

    angular_quantity_name = (
        ANGULAR_QUANTITY_NAME
    )


else:

    angular_quantity_name = None

    for preferred_name in preferred_angular_fields:

        if preferred_name in angular_array_candidates:
            angular_quantity_name = preferred_name
            break

    if angular_quantity_name is None:
        angular_quantity_name = next(
            iter(angular_array_candidates)
        )


angular_quantity = np.asarray(
    angular_array_candidates[
        angular_quantity_name
    ],
    dtype=float,
)

print(
    "Angular quantity selected:",
    angular_quantity_name,
)

print(
    "Selected quantity shape:",
    angular_quantity.shape,
)


# ============================================================
# BUILD AN ANGLE-INTEGRATED DIAGNOSTIC SPECTRUM
# ============================================================
#
# For an axisymmetric angular quantity, solid-angle weighting is
# proportional to sin(theta). This diagnostic is used to identify
# continuity and the strongest wavelength.
#
# It is not assumed to replace the package's formally normalized
# forward/backward collection integration.

sin_theta = np.sin(
    high_res_theta_rad
)

angle_integrated_diagnostic = (
    2.0
    * np.pi
    * np.trapezoid(
        angular_quantity
        * sin_theta[None, :],
        high_res_theta_rad,
        axis=1,
    )
)


spike_index = int(
    np.nanargmax(
        angle_integrated_diagnostic
    )
)

spike_wavelength_nm = float(
    high_res_wavelengths_nm[
        spike_index
    ]
)

left_index = max(
    0,
    spike_index - PROFILE_OFFSET_POINTS,
)

right_index = min(
    len(high_res_wavelengths_nm) - 1,
    spike_index + PROFILE_OFFSET_POINTS,
)

selected_profile_indices = [
    left_index,
    spike_index,
    right_index,
]

print(
    "Strongest high-resolution feature:",
    f"{spike_wavelength_nm:.5f} nm",
)


# ============================================================
# PREPARE MAP VALUES
# ============================================================

map_values = np.abs(
    angular_quantity
)

if USE_LOG_SCALE:

    map_maximum = float(
        np.nanmax(map_values)
    )

    map_floor = max(
        map_maximum
        * LOG_FLOOR_RELATIVE,
        np.finfo(float).tiny,
    )

    displayed_map = (
        10.0
        * np.log10(
            np.maximum(
                map_values,
                map_floor,
            )
            / map_maximum
        )
    )

    colourbar_label = (
        "Relative angular intensity [dB]"
    )

else:

    displayed_map = map_values

    colourbar_label = angular_quantity_name


# ============================================================
# CREATE FIGURE
# ============================================================

cm_to_inches = 1.0 / 2.54

mpl.rcParams[
    "figure.constrained_layout.use"
] = False

fig, axes = plt.subplots(
    3,
    1,
    figsize=(
        FIGURE_SIZE_CM[0] * cm_to_inches,
        FIGURE_SIZE_CM[1] * cm_to_inches,
    ),
    constrained_layout=False,
)

fig.set_layout_engine(None)

ax_spectrum = axes[0]
ax_map = axes[1]
ax_angle = axes[2]


# ============================================================
# PANEL 1 — HIGH-RESOLUTION SPECTRAL CONTINUITY
# ============================================================

ax_spectrum.plot(
    high_res_wavelengths_nm,
    angle_integrated_diagnostic,
    color="black",
    linewidth=1.4,
)

ax_spectrum.axvline(
    spike_wavelength_nm,
    color="tab:red",
    linestyle="--",
    linewidth=0.9,
)

ax_spectrum.scatter(
    [spike_wavelength_nm],
    [
        angle_integrated_diagnostic[
            spike_index
        ]
    ],
    s=25,
    facecolor="tab:red",
    edgecolor="black",
    linewidth=0.5,
    zorder=5,
)

ax_spectrum.set_xlim(
    *X_LIMITS_NM
)

ax_spectrum.set_ylabel(
    "Angle-integrated\nscattering diagnostic"
)

ax_spectrum.grid(
    False,
    which="both",
    axis="both",
)


# ============================================================
# PANEL 2 — WAVELENGTH/ANGLE MAP
# ============================================================

mesh = ax_map.pcolormesh(
    high_res_wavelengths_nm,
    high_res_theta_deg,
    displayed_map.T,
    shading="auto",
    cmap="magma",
)

ax_map.axvline(
    spike_wavelength_nm,
    color="cyan",
    linestyle="--",
    linewidth=0.9,
)

ax_map.set_xlim(
    *X_LIMITS_NM
)

ax_map.set_ylim(
    *ANGLE_LIMITS_DEG
)

ax_map.set_ylabel(
    r"Polar angle $\theta$ [deg]"
)

colourbar = fig.colorbar(
    mesh,
    ax=ax_map,
    pad=0.02,
)

colourbar.set_label(
    colourbar_label
)

ax_map.grid(
    False,
    which="both",
    axis="both",
)


# ============================================================
# PANEL 3 — ANGULAR PROFILES AROUND THE SPIKE
# ============================================================

profile_colours = [
    "tab:blue",
    "tab:red",
    "tab:green",
]

profile_linestyles = [
    "--",
    "-",
    ":",
]

for profile_index, colour, linestyle in zip(
    selected_profile_indices,
    profile_colours,
    profile_linestyles,
):

    profile = np.abs(
        angular_quantity[
            profile_index,
            :,
        ]
    )

    if USE_LOG_SCALE:

        profile_maximum = max(
            float(np.nanmax(profile)),
            np.finfo(float).tiny,
        )

        profile_plot = (
            10.0
            * np.log10(
                np.maximum(
                    profile,
                    profile_maximum
                    * LOG_FLOOR_RELATIVE,
                )
                / profile_maximum
            )
        )

    else:

        profile_plot = profile

    ax_angle.plot(
        high_res_theta_deg,
        profile_plot,
        color=colour,
        linestyle=linestyle,
        linewidth=1.4,
        label=(
            f"{high_res_wavelengths_nm[profile_index]:.4f} nm"
        ),
    )


ax_angle.axvline(
    90.0,
    color="0.6",
    linestyle="--",
    linewidth=0.7,
)

ax_angle.set_xlim(
    *ANGLE_LIMITS_DEG
)

ax_angle.set_xlabel(
    r"Polar angle $\theta$ [deg]"
)

if USE_LOG_SCALE:

    ax_angle.set_ylabel(
        "Relative angular\nintensity [dB]"
    )

else:

    ax_angle.set_ylabel(
        angular_quantity_name
    )

ax_angle.legend(
    frameon=False,
    loc="best",
)

ax_angle.grid(
    False,
    which="both",
    axis="both",
)


# ============================================================
# LAYOUT AND DISPLAY
# ============================================================

fig.subplots_adjust(
    left=0.15,
    right=0.98,
    top=0.98,
    bottom=0.08,
    hspace=0.25,
)
