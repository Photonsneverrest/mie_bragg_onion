from __future__ import annotations

# colour_solid_plotting.py
#
# Plot colours in CIELAB space together with a Rosch-MacAdam colour solid.
#
# Key features:
# - load Rosch-MacAdam colour solid points from CSV
# - robustly detect CIELAB columns (avoids confusion with RGB blue channel)
# - plot the solid in interactive Plotly 3D
# - colour the solid using stored RGB or hex values
# - optionally add max-chroma-per-hue markers
# - overlay one or more computed colour points from spectrum_colour_props.py
#
# Expected default files in the same directory as this module:
# - rosch_macadam_colour_solid_1nm.csv
# - rosch_macadam_max_chroma_per_hue_1deg.csv
#
# Conventions:
# - x-axis: a*
# - y-axis: b*
# - z-axis: L*
#
# Notes:
# - This module does not generate gifs.
# - It returns Plotly figures so you can show(), write_html(), or save them externally.

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_SOLID_CSV = MODULE_DIR / "rosch_macadam_colour_solid_1nm.csv"
DEFAULT_MAX_CHROMA_CSV = MODULE_DIR / "rosch_macadam_max_chroma_per_hue_1deg.csv"


__all__ = [
    "load_rosch_macadam_colour_solid",
    "load_rosch_macadam_max_chroma",
    "plot_rosch_macadam_colour_solid",
    "add_colour_properties_point",
    "add_cielab_point",
    "plot_colour_in_rosch_macadam_solid",
]


# ============================================================
# Path handling
# ============================================================

def _resolve_existing_path(
    csv_path: str | Path | None,
    default_path: Path,
    description: str,
) -> Path:
    """
    Resolve a CSV path robustly.

    Resolution order:
    1. If csv_path is None -> use default_path
    2. If csv_path is absolute -> use it directly
    3. If csv_path is relative -> try:
       - relative to current working directory
       - relative to MODULE_DIR

    Returns
    -------
    Path
        Existing resolved path
    """
    if csv_path is None:
        if default_path.exists():
            return default_path.resolve()
        raise FileNotFoundError(
            f"{description} not found at default location: {default_path}"
        )

    path = Path(csv_path)

    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.append(path)               # relative to current working directory
        candidates.append(MODULE_DIR / path) # relative to this module

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    tried = "\n".join(f" - {c}" for c in candidates)
    raise FileNotFoundError(
        f"{description} not found. Tried:\n{tried}"
    )


# ============================================================
# Small helpers
# ============================================================

def _normalize_name(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _downsample_indices(n: int, max_points: int) -> np.ndarray:
    """Return approximately uniformly spaced indices for downsampling."""
    if n <= max_points:
        return np.arange(n)
    return np.linspace(0, n - 1, max_points).astype(int)


def _rgb_to_hex(rgb: np.ndarray) -> list[str]:
    """Convert RGB array in [0, 1] to list of hex strings."""
    rgb = np.asarray(rgb, dtype=float)
    rgb = np.clip(rgb, 0.0, 1.0)
    rgb_255 = np.round(rgb * 255.0).astype(int)
    return [f"#{r:02X}{g:02X}{b:02X}" for r, g, b in rgb_255]


def _find_exact_or_normalized_column(
    df: pd.DataFrame,
    candidates: list[str],
    *,
    required: bool = True,
) -> str | None:
    """
    Find a column by trying exact names first, then normalized-name matching.
    """
    # Exact name priority
    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    # Normalized fallback
    norm_map = {_normalize_name(col): col for col in df.columns}
    for candidate in candidates:
        key = _normalize_name(candidate)
        if key in norm_map:
            return norm_map[key]

    if required:
        raise ValueError(
            f"Could not find any of the required columns {candidates} in "
            f"available columns: {list(df.columns)}"
        )
    return None


def _candidate_columns_by_tokens(
    df: pd.DataFrame,
    include_tokens: list[str],
    *,
    exclude_tokens: list[str] | None = None,
) -> list[str]:
    """
    Find columns whose normalized names contain all include_tokens and none of exclude_tokens.
    """
    exclude_tokens = exclude_tokens or []
    matches: list[str] = []

    for col in df.columns:
        norm = _normalize_name(col)
        if all(tok in norm for tok in include_tokens) and not any(tok in norm for tok in exclude_tokens):
            matches.append(col)

    return matches


def _choose_numeric_column_with_largest_range(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """
    Among numeric candidates, choose the one with the largest data range.
    Useful for distinguishing CIELAB b* from RGB blue channel.
    """
    best_col = None
    best_range = -np.inf

    for col in candidates:
        try:
            values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        except Exception:
            continue

        finite = np.isfinite(values)
        if not np.any(finite):
            continue

        rng = float(np.nanmax(values[finite]) - np.nanmin(values[finite]))
        if rng > best_range:
            best_range = rng
            best_col = col

    return best_col


def _detect_lab_columns(df: pd.DataFrame) -> tuple[str, str, str]:
    """
    Detect CIELAB L*, a*, b* columns robustly.

    This function strongly prioritizes explicit Lab-like names and avoids
    confusing the b* axis with an RGB blue channel.
    """
    # Strong priority: explicit Lab names
    L_candidates = [
        "L_smooth", "L_star", "L*", "Lab_L", "lab_L", "LabL", "L_ref", "L",
    ]
    a_candidates = [
        "a_smooth", "a_star", "a*", "Lab_a", "lab_a", "Laba", "a_ref", "a",
    ]
    b_candidates = [
        "b_smooth", "b_star", "b*", "Lab_b", "lab_b", "Labb", "b_ref", "b_lab", "b",
    ]

    L_col = _find_exact_or_normalized_column(df, L_candidates, required=False)
    a_col = _find_exact_or_normalized_column(df, a_candidates, required=False)
    b_col = _find_exact_or_normalized_column(df, b_candidates, required=False)

    # If explicit names failed, try token-based discovery
    if L_col is None:
        token_matches = _candidate_columns_by_tokens(df, ["l"], exclude_tokens=["rgb", "red", "green", "blue", "hex"])
        L_col = _choose_numeric_column_with_largest_range(df, token_matches)

    if a_col is None:
        token_matches = _candidate_columns_by_tokens(df, ["a"], exclude_tokens=["rgb", "red", "green", "blue", "hex"])
        a_col = _choose_numeric_column_with_largest_range(df, token_matches)

    if b_col is None:
        # explicitly exclude likely RGB blue columns
        token_matches = _candidate_columns_by_tokens(df, ["b"], exclude_tokens=["rgb", "blue", "hex"])
        b_col = _choose_numeric_column_with_largest_range(df, token_matches)

    if L_col is None or a_col is None or b_col is None:
        raise ValueError(
            "Could not robustly detect CIELAB columns. "
            f"Available columns: {list(df.columns)}"
        )

    # Heuristic safety check:
    # If chosen b column has tiny range, but another b-like numeric column has much larger range,
    # prefer the larger-range one. This often fixes confusion with RGB blue channel.
    b_values = pd.to_numeric(df[b_col], errors="coerce").to_numpy(dtype=float)
    b_finite = np.isfinite(b_values)
    b_range = float(np.nanmax(b_values[b_finite]) - np.nanmin(b_values[b_finite])) if np.any(b_finite) else 0.0

    alternative_b_candidates = [
        col for col in df.columns
        if col != b_col and "b" in _normalize_name(col) and "blue" not in _normalize_name(col) and "rgb" not in _normalize_name(col)
    ]
    alt_b = _choose_numeric_column_with_largest_range(df, alternative_b_candidates)

    if alt_b is not None:
        alt_values = pd.to_numeric(df[alt_b], errors="coerce").to_numpy(dtype=float)
        alt_finite = np.isfinite(alt_values)
        alt_range = float(np.nanmax(alt_values[alt_finite]) - np.nanmin(alt_values[alt_finite])) if np.any(alt_finite) else 0.0

        # If the current b range is suspiciously tiny compared with an alternative, switch
        if b_range < 2.0 and alt_range > 10.0:
            b_col = alt_b

    return L_col, a_col, b_col


def _detect_colour_columns(df: pd.DataFrame) -> tuple[str | None, str | None, str | None, str | None]:
    """
    Detect colour columns for plotting:
    - preferred: hex
    - fallback: r, g, b columns
    """
    hex_candidates = ["hex_smooth", "hex", "HEX"]

    # Prefer explicit RGB-like naming over generic 'b'
    r_candidates = ["r_srgb", "r_rgb", "red", "R", "r"]
    g_candidates = ["g_srgb", "g_rgb", "green", "G", "g"]
    b_candidates = ["b_srgb", "b_rgb", "blue", "B", "b.1"]  # avoid generic 'b' here on purpose

    hex_col = _find_exact_or_normalized_column(df, hex_candidates, required=False)
    r_col = _find_exact_or_normalized_column(df, r_candidates, required=False)
    g_col = _find_exact_or_normalized_column(df, g_candidates, required=False)
    b_col = _find_exact_or_normalized_column(df, b_candidates, required=False)

    return hex_col, r_col, g_col, b_col


# ============================================================
# CSV loading
# ============================================================

def load_rosch_macadam_colour_solid(
    csv_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Load Rosch-MacAdam colour solid CSV.

    Expected columns should include at least CIELAB L*, a*, b*,
    and ideally either:
    - hex
    or
    - RGB columns
    """
    path = _resolve_existing_path(
        csv_path=csv_path,
        default_path=DEFAULT_SOLID_CSV,
        description="Rosch-MacAdam colour solid CSV",
    )

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Rosch-MacAdam colour solid CSV is empty: {path}")

    return df


def load_rosch_macadam_max_chroma(
    csv_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Load max-chroma-per-hue CSV for optional overlay markers.
    """
    path = _resolve_existing_path(
        csv_path=csv_path,
        default_path=DEFAULT_MAX_CHROMA_CSV,
        description="Rosch-MacAdam max-chroma CSV",
    )

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Rosch-MacAdam max-chroma CSV is empty: {path}")

    return df


# ============================================================
# Plotting the solid
# ============================================================

def plot_rosch_macadam_colour_solid(
    solid_df: pd.DataFrame,
    *,
    title: str = "Rosch–MacAdam Colour Solid (CIELAB)",
    max_points: int = 60000,
    marker_size: float = 1.5,
    opacity: float = 0.10,
) -> go.Figure:
    """
    Plot the Rosch-MacAdam colour solid in CIELAB space.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    L_col, a_col, b_col = _detect_lab_columns(solid_df)
    hex_col, r_col, g_col, b_rgb_col = _detect_colour_columns(solid_df)

    hue_col = _find_exact_or_normalized_column(
        solid_df,
        ["hue_deg", "hue", "h"],
        required=False,
    )
    chroma_col = _find_exact_or_normalized_column(
        solid_df,
        ["chroma", "C", "C_star", "C*", "C_smooth"],
        required=False,
    )

    L = pd.to_numeric(solid_df[L_col], errors="coerce").to_numpy(dtype=float)
    a = pd.to_numeric(solid_df[a_col], errors="coerce").to_numpy(dtype=float)
    b = pd.to_numeric(solid_df[b_col], errors="coerce").to_numpy(dtype=float)

    valid = np.isfinite(L) & np.isfinite(a) & np.isfinite(b)
    if not np.any(valid):
        raise ValueError(
            f"No valid finite CIELAB points found using columns "
            f"L='{L_col}', a='{a_col}', b='{b_col}'."
        )

    L = L[valid]
    a = a[valid]
    b = b[valid]

    n_points = L.size
    idx = _downsample_indices(n_points, max_points=max_points)

    L = L[idx]
    a = a[idx]
    b = b[idx]

    # Resolve colours
    if hex_col is not None:
        hex_all = solid_df[hex_col].astype(str).to_numpy()
        hex_all = hex_all[valid]
        colors = hex_all[idx]
    elif r_col is not None and g_col is not None and b_rgb_col is not None:
        rgb_all = solid_df[[r_col, g_col, b_rgb_col]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        rgb_all = rgb_all[valid]
        colors = _rgb_to_hex(rgb_all[idx])
    else:
        colors = ["rgba(120,120,120,0.6)"] * len(idx)

    custom_columns: list[np.ndarray] = []
    hover_lines = [
        "a*: %{x:.2f}",
        "b*: %{y:.2f}",
        "L*: %{z:.2f}",
    ]

    if hue_col is not None:
        hue_all = pd.to_numeric(solid_df[hue_col], errors="coerce").to_numpy(dtype=float)
        hue_all = hue_all[valid]
        hue = hue_all[idx]
        custom_columns.append(hue)
        hover_lines.append("h*: %{customdata[0]:.2f}°")

    if chroma_col is not None:
        chroma_all = pd.to_numeric(solid_df[chroma_col], errors="coerce").to_numpy(dtype=float)
        chroma_all = chroma_all[valid]
        chroma = chroma_all[idx]
        custom_columns.append(chroma)
        custom_index = len(custom_columns) - 1
        hover_lines.append(f"C*: %{{customdata[{custom_index}]:.2f}}")

    customdata = np.column_stack(custom_columns) if custom_columns else None
    hovertemplate = "<br>".join(hover_lines) + "<extra></extra>"

    fig = go.Figure(
        data=go.Scatter3d(
            x=a,
            y=b,
            z=L,
            mode="markers",
            marker=dict(
                size=marker_size,
                color=colors,
                opacity=opacity,
            ),
            customdata=customdata,
            hovertemplate=hovertemplate,
            name="Rosch–MacAdam solid",
        )
    )

    fig.update_layout(
        title=title,
        template="plotly_white",
        showlegend=True,
        margin=dict(l=0, r=0, t=40, b=0),
        scene=dict(
            xaxis_title="a*",
            yaxis_title="b*",
            zaxis_title="L*",
            aspectmode="cube",
        ),
    )

    return fig


# ============================================================
# Add overlays
# ============================================================

def add_cielab_point(
    fig: go.Figure,
    *,
    L: float,
    a: float,
    b: float,
    name: str = "Colour point",
    color: str = "black",
    size: float = 8.0,
    symbol: str = "diamond",
    line_color: str = "white",
    line_width: float = 1.0,
    extra_hover_lines: list[str] | None = None,
) -> go.Figure:
    """
    Add one CIELAB point to an existing Plotly figure.
    """
    hover_lines = [
        "a*: %{x:.2f}",
        "b*: %{y:.2f}",
        "L*: %{z:.2f}",
    ]
    if extra_hover_lines:
        hover_lines.extend(extra_hover_lines)

    hovertemplate = "<br>".join(hover_lines) + "<extra></extra>"

    fig.add_trace(
        go.Scatter3d(
            x=[float(a)],
            y=[float(b)],
            z=[float(L)],
            mode="markers",
            marker=dict(
                size=size,
                color=color,
                symbol=symbol,
                line=dict(color=line_color, width=line_width),
            ),
            name=name,
            hovertemplate=hovertemplate,
        )
    )
    return fig


def add_colour_properties_point(
    fig: go.Figure,
    colour_properties: dict[str, Any],
    *,
    name: str = "Computed colour",
    size: float = 10.0,
    symbol: str = "diamond",
    fallback_color: str = "black",
    line_color: str = "white",
    line_width: float = 1.0,
) -> go.Figure:
    """
    Add a colour point from the output of spectrum_colour_props.compute_color_properties().
    """
    if "CIELAB" not in colour_properties:
        raise ValueError("colour_properties must contain a 'CIELAB' entry.")

    lab = colour_properties["CIELAB"]
    L = float(lab["L"])
    a = float(lab["a"])
    b = float(lab["b"])

    point_color = fallback_color
    if "Hex" in colour_properties and "hex" in colour_properties["Hex"]:
        point_color = str(colour_properties["Hex"]["hex"])

    extra_hover = []
    if "CIELAB" in colour_properties:
        extra_hover.append(f"C*: {float(lab['C']):.2f}")
        extra_hover.append(f"h*: {float(lab['hue_deg']):.2f}°")

    if "Performance in Rosch-MacAdam Solid" in colour_properties:
        perf = colour_properties["Performance in Rosch-MacAdam Solid"]
        extra_hover.append(f"eta_C: {float(perf['eta_C']):.3f}")
        extra_hover.append(f"eta_L: {float(perf['eta_L']):.3f}")
        extra_hover.append(f"eta_Y: {float(perf['eta_Y']):.3f}")

    return add_cielab_point(
        fig,
        L=L,
        a=a,
        b=b,
        name=name,
        color=point_color,
        size=size,
        symbol=symbol,
        line_color=line_color,
        line_width=line_width,
        extra_hover_lines=extra_hover,
    )


def _add_max_chroma_markers(
    fig: go.Figure,
    max_chroma_df: pd.DataFrame,
    *,
    name: str = "Max chroma per hue",
    size: float = 4.0,
) -> go.Figure:
    """
    Add max-chroma-per-hue markers to the figure.
    """
    L_col = _find_exact_or_normalized_column(max_chroma_df, ["L_smooth", "L", "L*"])
    a_col = _find_exact_or_normalized_column(max_chroma_df, ["a_smooth", "a", "a*"])
    b_col = _find_exact_or_normalized_column(max_chroma_df, ["b_smooth", "b", "b*"])
    hex_col = _find_exact_or_normalized_column(max_chroma_df, ["hex_smooth", "hex"], required=False)
    hue_col = _find_exact_or_normalized_column(max_chroma_df, ["hue_deg", "hue"], required=False)

    L = pd.to_numeric(max_chroma_df[L_col], errors="coerce").to_numpy(dtype=float)
    a = pd.to_numeric(max_chroma_df[a_col], errors="coerce").to_numpy(dtype=float)
    b = pd.to_numeric(max_chroma_df[b_col], errors="coerce").to_numpy(dtype=float)

    valid = np.isfinite(L) & np.isfinite(a) & np.isfinite(b)
    L = L[valid]
    a = a[valid]
    b = b[valid]

    if hex_col is not None:
        colors = max_chroma_df[hex_col].astype(str).to_numpy()[valid]
    else:
        colors = ["black"] * len(L)

    if hue_col is not None:
        hue = pd.to_numeric(max_chroma_df[hue_col], errors="coerce").to_numpy(dtype=float)[valid]
        customdata = np.column_stack([hue])
        hovertemplate = (
            "a*: %{x:.2f}<br>"
            "b*: %{y:.2f}<br>"
            "L*: %{z:.2f}<br>"
            "h*: %{customdata[0]:.2f}°"
            "<extra></extra>"
        )
    else:
        customdata = None
        hovertemplate = (
            "a*: %{x:.2f}<br>"
            "b*: %{y:.2f}<br>"
            "L*: %{z:.2f}<extra></extra>"
        )

    fig.add_trace(
        go.Scatter3d(
            x=a,
            y=b,
            z=L,
            mode="markers",
            marker=dict(
                size=size,
                color=colors,
                opacity=1.0,
                line=dict(color="black", width=0.8),
            ),
            customdata=customdata,
            hovertemplate=hovertemplate,
            name=name,
        )
    )

    return fig


# ============================================================
# High-level convenience function
# ============================================================

def plot_colour_in_rosch_macadam_solid(
    colour_properties: dict[str, Any],
    *,
    solid_csv_path: str | Path | None = None,
    max_chroma_csv_path: str | Path | None = None,
    show_max_chroma: bool = True,
    title: str = "Computed Colour in Rosch–MacAdam Colour Solid",
    solid_max_points: int = 60000,
    solid_marker_size: float = 1.5,
    solid_opacity: float = 0.10,
    point_name: str = "Computed colour",
    point_size: float = 10.0,
) -> go.Figure:
    """
    High-level function:
    - loads Rosch-MacAdam solid
    - plots it
    - optionally overlays max-chroma markers
    - adds the computed colour point
    """
    solid_df = load_rosch_macadam_colour_solid(solid_csv_path)

    fig = plot_rosch_macadam_colour_solid(
        solid_df,
        title=title,
        max_points=solid_max_points,
        marker_size=solid_marker_size,
        opacity=solid_opacity,
    )

    if show_max_chroma:
        max_chroma_df = load_rosch_macadam_max_chroma(max_chroma_csv_path)
        fig = _add_max_chroma_markers(fig, max_chroma_df)

    fig = add_colour_properties_point(
        fig,
        colour_properties,
        name=point_name,
        size=point_size,
    )

    return fig