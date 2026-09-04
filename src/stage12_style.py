"""Stage 12 unified publication-figure style for the Conformal UQ project.

Single source of truth for rcParams, palettes, and the dual colour/linestyle
encoding used by every Stage 12 figure. Colour AND line style AND marker all
encode the same contrast so figures remain readable under colour-vision
deficiency and in greyscale print.

Backend: Python/matplotlib exclusively (project-wide convention).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Frozen project constants (identical to Stage 10/11 convention)
# ---------------------------------------------------------------------------
DATASETS = [
    "openml_3_kr_vs_kp",
    "openml_24_mushroom",
    "openml_1486_nomao",
    "openml_1489_phoneme",
    "openml_1590_adult",
    "openml_4534_phishingwebsite",
    "openml_23512_higgs",
    "openml_23517_numerai28_6",
]
SEEDS = [104729, 130363, 155921, 196613, 262147, 318281, 374209, 419893, 481517, 552721]
MODELS = ["logistic_regression", "xgboost", "tabpfn"]
CP_METHODS = ["global_split_cp", "class_conditional_cp"]
M_VALUES = [10, 20, 50, 100]

MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "xgboost": "XGBoost",
    "tabpfn": "TabPFN",
}
CP_LABELS = {
    "global_split_cp": "Global Split CP",
    "class_conditional_cp": "Class-Conditional CP",
}

# ---------------------------------------------------------------------------
# Colour / linestyle / marker encoding (Okabe-Ito, colour-blind safe)
# ---------------------------------------------------------------------------
# CP-method contrast: hue + linetype + marker (triple coding).
METHOD_STYLE = {
    "global_split_cp": {
        "color": "#0072B2",       # blue
        "linestyle": "-",
        "marker": "o",
        "label": CP_LABELS["global_split_cp"],
    },
    "class_conditional_cp": {
        "color": "#D55E00",       # vermillion
        "linestyle": (0, (5, 2)),  # dashed
        "marker": "s",
        "label": CP_LABELS["class_conditional_cp"],
    },
}

# Fig. 4 metric contrast within each pipeline panel: linetype + marker fill.
VARIABILITY_STYLE = {
    "sd": {
        "color": "#0072B2",
        "linestyle": "-",
        "marker": "o",
        "markerfill": True,
        "label": "Across-seed SD",
    },
    "iqr": {
        "color": "#0072B2",
        "linestyle": (0, (3, 1.5)),  # dash-dot
        "marker": "o",
        "markerfill": False,
        "label": "Across-seed IQR",
    },
}

# Individual-dataset thin curves behind the median curves.
DATASET_THIN_STYLE = {
    "color": "#9E9E9E",
    "linewidth": 0.7,
    "alpha": 0.85,
    "marker": ".",
    "markersize": 2.2,
}

# Reference lines (nominal coverage / threshold geometry).
REFERENCE_LINE_STYLE = {
    "color": "#555555",
    "linestyle": (0, (1, 1)),  # dotted
    "linewidth": 0.9,
}

# Boundary/main-comparison region shading.
REGION_SHADE = {"color": "0.94", "lw": 0, "zorder": 0}

# ---------------------------------------------------------------------------
# rcParams (Nature-style: 7 pt Times New Roman serif, editable text in SVG/PDF,
# no top/right spines). Math text uses the STIX font set, which is designed to
# match Times visually, so $q_{\mathrm{minority}}$-style labels do not clash.
# ---------------------------------------------------------------------------
RC_PARAMS = {
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif", "serif"],
    "mathtext.fontset": "stix",
    "font.size": 7,
    "axes.titlesize": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6,
    "legend.title_fontsize": 6.5,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "svg.fonttype": "none",   # keep text editable in SVG
    "pdf.fonttype": 42,       # editable TrueType text in PDF
    "ps.fonttype": 42,
    "axes.grid": False,
    "savefig.bbox": "tight",
}

# Single-column 90 mm, double-column 190 mm (Nature 3-column grid analogues).
FIG_SIZE_IN = {
    "fig1": (3.54, 3.05),
    "fig2": (3.54, 3.05),
    "fig3": (3.54, 3.05),
    "fig4": (7.48, 2.60),
}
PNG_DPI = 600

# Aggregation contract mirrored from the preregistered Stage 11 analysis.
AGGREGATION_RULE = (
    "Statistical unit = dataset (n=8). Dataset-level value = mean over the "
    "3 predictive pipelines x 10 frozen seeds (30 CP cells per dataset x "
    "CP method x m); bold curves show the median across the 8 datasets; thin "
    "grey curves show individual datasets. Seed cells are never treated as "
    "independent units."
)


def apply_style() -> None:
    """Apply the unified rcParams to matplotlib."""
    import matplotlib as mpl

    mpl.rcParams.update(RC_PARAMS)
    # Times New Roman is a Microsoft core font; when absent (e.g. Linux CI),
    # fall back gracefully instead of crashing on findfont.
    import matplotlib.font_manager as fm

    available = {f.name for f in fm.fontManager.ttflist}
    if not any(name in available for name in ("Times New Roman", "Times")):
        mpl.rcParams["font.serif"] = ["Nimbus Roman", "DejaVu Serif", "serif"]


def style_config_payload() -> dict:
    """Machine-readable export of the full style contract."""
    return {
        "style_id": "conformal_uq_stage12_publication_v1.0",
        "backend": "python/matplotlib",
        "rc_params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in RC_PARAMS.items()},
        "png_dpi": PNG_DPI,
        "figure_sizes_inches": FIG_SIZE_IN,
        "method_style": {
            cp: {**st, "linestyle": list(st["linestyle"]) if isinstance(st["linestyle"], tuple) else st["linestyle"]}
            for cp, st in METHOD_STYLE.items()
        },
        "variability_style": {
            k: {**st, "linestyle": list(st["linestyle"]) if isinstance(st["linestyle"], tuple) else st["linestyle"]}
            for k, st in VARIABILITY_STYLE.items()
        },
        "dataset_thin_style": DATASET_THIN_STYLE,
        "reference_line_style": {"color": REFERENCE_LINE_STYLE["color"], "linestyle": "dotted", "linewidth": REFERENCE_LINE_STYLE["linewidth"]},
        "region_shade": REGION_SHADE,
        "dual_encoding_note": (
            "Every categorical contrast is encoded simultaneously in colour, "
            "line style, and (where applicable) marker fill, so figures remain "
            "readable under colour-vision deficiency and in greyscale print."
        ),
        "m_grid": {
            "10": "boundary diagnostic",
            "20": "near-boundary diagnostic",
            "50": "main comparison",
            "100": "main comparison",
        },
        "aggregation_rule": AGGREGATION_RULE,
    }
