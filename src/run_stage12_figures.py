"""Stage 12 — publication figures and main tables for the Conformal UQ project.

Generates every figure and table programmatically from frozen, machine-readable
inputs. No value is ever hand-entered.

Inputs (read-only, hash-gated):
  - results/results_long.parquet          (frozen Stage 09/10 results, 1,920 CP cells)
  - results/stats/stage11_threshold_variability.csv  (Stage 11 D09 table, cross-check)
  - artifacts/stage02/dataset_registry_v1.0.1.json   (Table 1 characteristics)
  - artifacts/splits/v1.1/<dataset>/seed-*.json      (Table 1 split class counts)

Outputs (exclusive creation):
  - manuscript/figures/...  (SVG + PDF + 600-dpi PNG, per-figure source data,
    unified style config, captions, figure-number mapping)
  - manuscript/tables/...   (CSV + formatted Markdown for Tables 1-3)
  - artifacts/stage12/<run_id>/  (events, manifest, status, QA evidence)

Statistical unit = dataset (n=8): dataset-level values average the 3 predictive
pipelines x 10 frozen seeds (30 cells per dataset x CP method x m); seed cells
are never treated as independent units. m=10 boundary, m=20 near-boundary,
m=50/100 main comparison. Figures show descriptive quantities only; Wilson
intervals are never pooled; the probability-scale Beta order-statistic variance
is never overlaid on empirical threshold variability (different quantity/scale).
"""

from __future__ import annotations

import hashlib
import json
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless; must precede pyplot
import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import stage12_style as st  # noqa: E402

st.apply_style()

# Delivery contract mirrored from stage12_style.RC_PARAMS and asserted at the
# render site: Times New Roman serif family (7 pt, STIX math), editable text
# in SVG/PDF, and 600-dpi raster export travel with this plotting source.
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif", "serif"],
    "mathtext.fontset": "stix",
    "font.size": 7,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "savefig.dpi": 600,
})

# ---------------------------------------------------------------------------
# Frozen constants and hash gates
# ---------------------------------------------------------------------------
FROZEN_PARQUET_SHA256 = (
    "d795d07c36a58619d715c18da6195d929124e43d42abafb45de2dc5134b3dcc2"
)
FROZEN_FORMAL_MANIFEST_SHA256 = (
    "0690795b1ea68a148f069a44dccb09a9245d21a097e7174dbd5ec3002f24172d"
)
EXPECTED_N_CELLS = 1920
EXPECTED_GRID = (8, 10, 3, 2, 4)

P_RESULTS = ROOT / "results/results_long.parquet"
P_RESULTS_MANIFEST = ROOT / "results/results_manifest.json"
P_FROZEN_MANIFEST = ROOT / "configs/formal_run_manifest_v1.1.yaml"
P_S11_TV = ROOT / "results/stats/stage11_threshold_variability.csv"
P_S11_AB = ROOT / "results/stats/stage11_ab_endpoints.csv"
P_REGISTRY = ROOT / "artifacts/stage02/dataset_registry_v1.0.1.json"
P_SPLITS_V11 = ROOT / "artifacts/splits/v1.1"

FIG_DIR = ROOT / "manuscript/figures"
TABLE_DIR = ROOT / "manuscript/tables"

X = np.arange(4, dtype=float)
M_TICK_LABELS = ["10\nboundary", "20\nnear-boundary", "50\nmain", "100\nmain"]
REGION_TEXTS = [
    (0.5, "boundary (m=10) / near-boundary (m=20)"),
    (2.5, "main comparison (m=50/100)"),
]

METRIC_LABELS = {
    "singleton_rate": "Singleton rate",
    "coverage_minority": "Minority-class coverage",
    "coverage_majority": "Majority-class coverage",
    "average_set_size": "Average set size",
    "threshold_sum": "ThresholdSum ($q_{\\mathrm{minority}}+q_{\\mathrm{majority}}$)",
    "q_minority": "$q_{\\mathrm{minority}}$",
}

FIG_SPECS = {
    1: {
        "dir": "figure1_singleton_rate",
        "stem": "figure1_singleton_rate",
        "metric": "singleton_rate",
        "ylabel": "Singleton rate",
        "cps": ["global_split_cp", "class_conditional_cp"],
        "ref": None,
        "ref_label": None,
        # v1.0.2: legend moved up into the empty mid-right band (between the
        # lowest dataset curves and the upper cluster) per user revision.
        "legend_loc": "center right",
        "legend_anchor": (0.98, 0.64),
    },
    2: {
        "dir": "figure2_minority_coverage",
        "stem": "figure2_minority_coverage",
        "metric": "coverage_minority",
        "ylabel": "Minority-class coverage",
        "cps": ["global_split_cp", "class_conditional_cp"],
        "ref": 0.90,
        "ref_label": "Nominal coverage (0.90)",
        "ref_annotation_text": "nominal 0.90",
    },
    3: {
        "dir": "figure3_threshold_sum",
        "stem": "figure3_threshold_sum",
        "metric": "threshold_sum",
        "ylabel": "ThresholdSum ($q_{\\mathrm{minority}}+q_{\\mathrm{majority}}$)",
        "cps": ["class_conditional_cp"],
        "ref": 1.0,
        "ref_label": "ThresholdSum = 1 (geometry reference)",
        "ref_annotation_text": "ThresholdSum = 1",
        "ref_legend": False,
        "legend_loc": "center right",
        "legend_anchor": (0.98, 0.45),
    },
}

QA_SAMPLING_SEED = 20260831

# Artifact version (PATCH-level for rendering-only revisions, per governance).
ARTIFACT_VERSION = "v1.0.3"


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------
def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_text_lf(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def write_json_lf(path: Path, obj) -> None:
    write_text_lf(path, json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def write_csv_lf(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, lineterminator="\n")


class EventLog:
    """Append-only JSONL event log (LF endings)."""

    def __init__(self, run_id: str, path: Path):
        self.run_id = run_id
        self.path = path

    def log(self, event: str, level: str = "INFO", **kw) -> None:
        rec = {
            "timestamp": utc_now(),
            "run_id": self.run_id,
            "stage": "stage12",
            "level": level,
            "event": event,
            **kw,
        }
        with open(self.path, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Gates (all must pass before any output exists)
# ---------------------------------------------------------------------------
def run_gates(ev: EventLog) -> dict:
    ev.log("gate_start", detail="Stage 12 input gates")
    checks = {}

    live = sha256_file(P_RESULTS)
    checks["frozen_parquet_sha256"] = {
        "expected": FROZEN_PARQUET_SHA256,
        "observed": live,
        "status": "MATCH" if live == FROZEN_PARQUET_SHA256 else "MISMATCH",
    }
    if live != FROZEN_PARQUET_SHA256:
        raise SystemExit(f"GATE FAIL: frozen parquet hash {live} != {FROZEN_PARQUET_SHA256}")

    fm = sha256_file(P_FROZEN_MANIFEST)
    checks["frozen_formal_manifest_sha256"] = {
        "expected": FROZEN_FORMAL_MANIFEST_SHA256,
        "observed": fm,
        "status": "MATCH" if fm == FROZEN_FORMAL_MANIFEST_SHA256 else "MISMATCH",
    }
    if fm != FROZEN_FORMAL_MANIFEST_SHA256:
        raise SystemExit("GATE FAIL: formal run manifest hash mismatch")

    for p in (P_RESULTS_MANIFEST, P_S11_TV, P_S11_AB, P_REGISTRY):
        if not p.exists():
            raise SystemExit(f"GATE FAIL: required input missing: {p}")
    n_split_manifests = len(list(P_SPLITS_V11.glob("*/seed-*.json")))
    if n_split_manifests != 80:
        raise SystemExit(f"GATE FAIL: expected 80 v1.1 split manifests, found {n_split_manifests}")
    checks["required_inputs_present"] = {"status": "PASS", "split_manifests": n_split_manifests}

    df = pd.read_parquet(P_RESULTS)
    keys = ["dataset_id", "seed", "model", "cp_method", "m_minority"]
    if len(df) != EXPECTED_N_CELLS or df[keys].drop_duplicates().shape[0] != EXPECTED_N_CELLS:
        raise SystemExit(f"GATE FAIL: grid is not {EXPECTED_N_CELLS} unique cells")
    if not (
        (df["status"] == "PASS").all()
        and (df["alpha"] == 0.1).all()
        and (df["m_majority"] == 200).all()
        and (df["protocol_version"] == "v1.1").all()
    ):
        raise SystemExit("GATE FAIL: unexpected status/alpha/m_majority/protocol_version")
    counts = df.groupby(["dataset_id", "model", "cp_method", "m_minority"]).size()
    if not (counts == 10).all():
        raise SystemExit("GATE FAIL: some dataset x model x cp x m group is not exactly 10 seeds")
    checks["grid_integrity"] = {
        "status": "PASS",
        "n_rows": int(len(df)),
        "unique_cells": int(df[keys].drop_duplicates().shape[0]),
        "seeds_per_dataset_model_cp_m": 10,
    }
    ev.log("gate_pass", n_checks=len(checks), detail="all Stage 12 input gates PASS")
    return checks


# ---------------------------------------------------------------------------
# Data loading and aggregation (unit = dataset)
# ---------------------------------------------------------------------------
def load_registry() -> dict:
    raw = json.loads(P_REGISTRY.read_text(encoding="utf-8"))
    return {r["dataset_id"]: r for r in raw["records"]}


def load_split_manifests() -> dict:
    out = {}
    for ds in st.DATASETS:
        out[ds] = {}
        for seed in st.SEEDS:
            p = P_SPLITS_V11 / ds / f"seed-{seed}.json"
            out[ds][seed] = json.loads(p.read_text(encoding="utf-8"))
    return out


def dataset_level_series(df: pd.DataFrame, metric: str, cp: str, m: int) -> pd.Series:
    """Dataset-level values: mean over the 3 pipelines x 10 seeds (30 cells)."""
    sub = df[(df["cp_method"] == cp) & (df["m_minority"] == m)]
    g = sub.groupby("dataset_id")[metric].agg(["sum", "count"])
    if not (g["count"] == 30).all():
        raise SystemExit(f"Aggregation contract violated for {metric}/{cp}/m={m}: counts {g['count'].to_dict()}")
    return (g["sum"] / g["count"]).reindex(st.DATASETS)


def build_fig_source_data(df: pd.DataFrame, spec: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Per-figure source data: dataset-level values, median summary, per-model detail."""
    metric = spec["metric"]
    rows, det = [], []
    for cp in spec["cps"]:
        for m in st.M_VALUES:
            s = dataset_level_series(df, metric, cp, m)
            for ds in st.DATASETS:
                rows.append({"dataset_id": ds, "cp_method": cp, "m_minority": m, "value": float(s[ds])})
            for model in st.MODELS:
                sub = df[(df["cp_method"] == cp) & (df["m_minority"] == m) & (df["model"] == model)]
                g = sub.groupby("dataset_id")[metric].agg(["sum", "count"])
                if not (g["count"] == 10).all():
                    raise SystemExit("Per-model detail aggregation contract violated (expected 10 seeds)")
                dm = (g["sum"] / g["count"]).reindex(st.DATASETS)
                for ds in st.DATASETS:
                    det.append({
                        "dataset_id": ds, "model": model, "cp_method": cp,
                        "m_minority": m, "value": float(dm[ds]),
                    })
    src = pd.DataFrame(rows)
    summ = (
        src.groupby(["cp_method", "m_minority"])["value"]
        .agg(["median", "mean", "min", "max", "count"])
        .reset_index()
    )
    summ["n_datasets"] = summ["count"].astype(int)
    summ = summ.drop(columns=["count"])
    return src, summ, pd.DataFrame(det)


def build_fig4_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Across-seed SD/IQR of q_minority per dataset x model x m (CC only)."""
    sub = df[(df["cp_method"] == "class_conditional_cp")]
    rows = []
    for model in st.MODELS:
        for ds in st.DATASETS:
            for m in st.M_VALUES:
                v = sub[(sub["model"] == model) & (sub["dataset_id"] == ds) & (sub["m_minority"] == m)]["q_minority"]
                if len(v) != 10:
                    raise SystemExit("Fig 4 aggregation contract violated (expected 10 seeds)")
                rows.append({
                    "dataset_id": ds, "model": model, "m_minority": m,
                    "across_seed_sd": float(v.std(ddof=1)),
                    "across_seed_iqr": float(v.quantile(0.75) - v.quantile(0.25)),
                    "across_seed_mean": float(v.mean()),
                })
    src = pd.DataFrame(rows)
    summ_rows = []
    for model in st.MODELS:
        for m in st.M_VALUES:
            s = src[(src["model"] == model) & (src["m_minority"] == m)]
            summ_rows.append({
                "model": model, "m_minority": m, "n_datasets": int(len(s)),
                "sd_median": float(s["across_seed_sd"].median()),
                "sd_min": float(s["across_seed_sd"].min()), "sd_max": float(s["across_seed_sd"].max()),
                "iqr_median": float(s["across_seed_iqr"].median()),
                "iqr_min": float(s["across_seed_iqr"].min()), "iqr_max": float(s["across_seed_iqr"].max()),
            })
    return src, pd.DataFrame(summ_rows)


# ---------------------------------------------------------------------------
# Figure rendering
# ---------------------------------------------------------------------------
def _region_annotation(ax) -> None:
    ax.axvspan(-0.4, 1.4, **st.REGION_SHADE)
    for xpos, text in REGION_TEXTS:
        ax.text(xpos, 1.035, text, transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=6, color="0.35")


def _m_axis(ax) -> None:
    ax.set_xticks(X)
    ax.set_xticklabels(M_TICK_LABELS)
    ax.set_xlim(-0.45, 3.45)
    ax.set_xlabel("$m_{\\mathrm{minority}}$ (minority calibration size)")


def _thin_proxy() -> Line2D:
    return Line2D([], [], color=st.DATASET_THIN_STYLE["color"], lw=st.DATASET_THIN_STYLE["linewidth"],
                  marker=st.DATASET_THIN_STYLE["marker"], markersize=st.DATASET_THIN_STYLE["markersize"],
                  alpha=0.9, label="Individual datasets (n=8)")


def _ref_proxy(spec) -> Line2D | None:
    if spec["ref"] is None:
        return None
    return Line2D([], [], color=st.REFERENCE_LINE_STYLE["color"], ls=":", lw=st.REFERENCE_LINE_STYLE["linewidth"],
                  label=spec["ref_label"])


def draw_m_figure(path_stem: Path, spec: dict, src: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=st.FIG_SIZE_IN["fig1"])
    if spec["ref"] is not None:
        ax.axhline(spec["ref"], color=st.REFERENCE_LINE_STYLE["color"],
                   ls=":", lw=st.REFERENCE_LINE_STYLE["linewidth"], zorder=2)
    wide = {cp: src[src["cp_method"] == cp].pivot(index="dataset_id", columns="m_minority", values="value").reindex(index=st.DATASETS, columns=st.M_VALUES)
            for cp in spec["cps"]}
    for cp in spec["cps"]:
        w = wide[cp]
        for ds in st.DATASETS:
            ax.plot(X, w.loc[ds].to_numpy(), zorder=1, **st.DATASET_THIN_STYLE)
    for cp in spec["cps"]:
        msty = st.METHOD_STYLE[cp]
        med = wide[cp].median(axis=0).to_numpy()
        ax.plot(X, med, color=msty["color"], linestyle=msty["linestyle"], lw=1.4,
                marker=msty["marker"], ms=3.2, zorder=3, label=msty["label"])
    handles = [Line2D([], [], color=st.METHOD_STYLE[cp]["color"], ls=st.METHOD_STYLE[cp]["linestyle"],
                      marker=st.METHOD_STYLE[cp]["marker"], ms=3.2, lw=1.4, label=st.METHOD_STYLE[cp]["label"])
               for cp in spec["cps"]]
    handles.append(_thin_proxy())
    if spec["ref"] is not None and spec.get("ref_legend", True):
        handles.append(_ref_proxy(spec))
    legend_kw = {"loc": spec.get("legend_loc", "best")}
    if "legend_anchor" in spec:
        legend_kw["bbox_to_anchor"] = spec["legend_anchor"]
    ax.legend(handles=handles, **legend_kw)
    if spec["ref"] is not None and spec.get("ref_annotation_text"):
        ax.annotate(spec["ref_annotation_text"], xy=(3.42, spec["ref"]), xytext=(-1, 3),
                    textcoords="offset points", ha="right", va="bottom", fontsize=6, color="0.25")
    _region_annotation(ax)
    _m_axis(ax)
    ax.set_ylabel(spec["ylabel"])
    for ext in ("svg", "pdf", "png"):
        fig.savefig(path_stem.with_suffix(f".{ext}"))
    plt.close(fig)


def draw_fig4(path_stem: Path, src: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, sharey=True, figsize=st.FIG_SIZE_IN["fig4"], constrained_layout=True)
    vsd, viqr = st.VARIABILITY_STYLE["sd"], st.VARIABILITY_STYLE["iqr"]
    for ax, model in zip(axes, st.MODELS):
        sub = src[src["model"] == model]
        ax.axvspan(-0.4, 1.4, **st.REGION_SHADE)
        for metric_key, col in (("sd", "across_seed_sd"), ("iqr", "across_seed_iqr")):
            msty = vsd if metric_key == "sd" else viqr
            w = sub.pivot(index="dataset_id", columns="m_minority", values=col).reindex(index=st.DATASETS, columns=st.M_VALUES)
            med = w.median(axis=0).to_numpy()
            if metric_key == "sd":
                ax.fill_between(X, w.min(axis=0).to_numpy(), w.max(axis=0).to_numpy(),
                                color=msty["color"], alpha=0.14, lw=0, zorder=1)
            ax.plot(X, med, color=msty["color"], linestyle=msty["linestyle"], lw=1.3,
                    marker=msty["marker"], ms=3.2,
                    markerfacecolor=(msty["color"] if msty["markerfill"] else "white"),
                    markeredgecolor=msty["color"], zorder=3,
                    label=("Across-seed SD" if metric_key == "sd" else "Across-seed IQR"))
        ax.set_title(st.MODEL_LABELS[model], pad=14)
        _m_axis(ax)
    handles = [
        Line2D([], [], color=vsd["color"], ls="-", marker="o", ms=3.2, lw=1.3, label="Across-seed SD"),
        Line2D([], [], color=viqr["color"], ls=viqr["linestyle"], marker="o", ms=3.2, lw=1.3,
               markerfacecolor="white", markeredgecolor=viqr["color"], label="Across-seed IQR"),
        Line2D([], [], color=st.REGION_SHADE["color"], lw=4, alpha=0.6, label="Min-max range (n=8 datasets)"),
    ]
    axes[0].legend(handles=handles, loc="upper right")
    axes[0].set_ylabel("Across-seed variability of $q_{\\mathrm{minority}}$")
    for ext in ("svg", "pdf", "png"):
        fig.savefig(path_stem.with_suffix(f".{ext}"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def _fmt_count(mean: float, mn: float, mx: float) -> str:
    if mx - mn < 1e-9:
        return f"{mean:.0f}"
    return f"{mean:.1f}"


def build_table1(registry: dict, splits: dict) -> tuple[pd.DataFrame, str]:
    rows = []
    for ds in st.DATASETS:
        r = registry[ds]
        minority_ratio = r["raw_class_counts"][r["minority_original_label"]] / r["n_rows"]
        row = {
            "dataset_id": ds,
            "display_name": r["display_name"],
            "domain": r["domain"],
            "openml_data_id": r["source"]["openml_data_id"],
            "n_rows": r["n_rows"],
            "raw_feature_count": r["raw_feature_count"],
            "features_numeric": r["feature_types"]["numeric"],
            "features_categorical": r["feature_types"]["categorical"],
            "max_post_train_transform_features": r["maximum_post_train_transform_feature_count"],
            "minority_original_label": r["minority_original_label"],
            "minority_ratio": minority_ratio,
        }
        for split, split_key in (("train", "train"), ("cal_pool", "calibration_pool"), ("test", "test")):
            for cls in ("majority", "minority"):
                vals = [splits[ds][seed]["class_counts"][split_key][cls] for seed in st.SEEDS]
                row[f"{split}_{cls}_mean"] = float(np.mean(vals))
                row[f"{split}_{cls}_min"] = int(np.min(vals))
                row[f"{split}_{cls}_max"] = int(np.max(vals))
        rows.append(row)
    df = pd.DataFrame(rows)

    md_lines = [
        "# Table 1 | Dataset characteristics",
        "",
        "| Dataset | Domain | N | Features (raw) | Features (post-transform, max) | Minority label | Minority ratio | Train (maj/min) | Cal. pool (maj/min) | Test (maj/min) |",
        "|---|---|---:|---:|---:|---|---:|---|---|---|",
    ]
    for _, r in df.iterrows():
        md_lines.append(
            f"| {r['display_name']} (`{r['dataset_id']}`) | {r['domain']} | {r['n_rows']:,} "
            f"| {r['raw_feature_count']} ({r['features_numeric']} num / {r['features_categorical']} cat) "
            f"| {r['max_post_train_transform_features']} | {r['minority_original_label']} "
            f"| {r['minority_ratio']:.4f} "
            f"| {_fmt_count(r['train_majority_mean'], r['train_majority_min'], r['train_majority_max'])} / {_fmt_count(r['train_minority_mean'], r['train_minority_min'], r['train_minority_max'])} "
            f"| {_fmt_count(r['cal_pool_majority_mean'], r['cal_pool_majority_min'], r['cal_pool_majority_max'])} / {_fmt_count(r['cal_pool_minority_mean'], r['cal_pool_minority_min'], r['cal_pool_minority_max'])} "
            f"| {_fmt_count(r['test_majority_mean'], r['test_majority_min'], r['test_majority_max'])} / {_fmt_count(r['test_minority_mean'], r['test_minority_min'], r['test_minority_max'])} |"
        )
    md_lines += [
        "",
        "Class counts are means over the 10 frozen seeds of the locked v1.1 splits; per-seed min/max values are in the CSV. "
        "Features (post-transform, max) is the registry maximum over seeds after train-only one-hot encoding. "
        "Minority ratio = minority class count / N in the raw dataset (registry-defined minority label).",
        "",
    ]
    return df, "\n".join(md_lines)


def build_table2(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    rows = []
    for ds in st.DATASETS:
        for model in st.MODELS:
            sub = df[(df["dataset_id"] == ds) & (df["model"] == model)]
            if len(sub) != 80:
                raise SystemExit(f"Expected 80 CP cells per dataset x model, got {len(sub)} for {ds}/{model}")
            for metric in ("auroc", "auprc"):
                vals = sub[metric]
                rows.append({
                    "dataset_id": ds, "model": model, "metric": metric,
                    "mean_across_seeds": float(vals.mean()),
                    "sd_across_seeds": float(vals.std(ddof=1)),
                    "n_seeds": 10,
                })
    long = pd.DataFrame(rows)
    w = long.pivot_table(index="dataset_id", columns=["metric", "model"],
                         values=["mean_across_seeds", "sd_across_seeds"])
    reg = load_registry()
    display = {d: reg[d]["display_name"] for d in st.DATASETS}

    md_lines = [
        "# Table 2 | Base predictive performance",
        "",
        "AUROC / AUPRC on the 20% test split; mean (SD) across the 10 frozen seeds per dataset x predictive pipeline.",
        "",
        "| Dataset | AUROC: " + " | AUROC: ".join(st.MODEL_LABELS[m] for m in st.MODELS)
        + " | AUPRC: " + " | AUPRC: ".join(st.MODEL_LABELS[m] for m in st.MODELS) + " |",
        "|---|" + "---:|" * 6,
    ]
    for ds in st.DATASETS:
        cells = []
        for metric in ("auroc", "auprc"):
            for model in st.MODELS:
                mu = float(w.loc[ds, ("mean_across_seeds", metric, model)])
                sd = float(w.loc[ds, ("sd_across_seeds", metric, model)])
                cells.append(f"{mu:.3f} ({sd:.3f})")
        md_lines.append(f"| {display[ds]} | " + " | ".join(cells) + " |")
    md_lines += ["", "Base predictive performance is invariant across CP method and m by construction (the same base probabilities feed every CP cell); this invariance is verified in the Stage 12 QA evidence.", ""]
    return long, "\n".join(md_lines)


def build_table3(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    metrics = ["coverage_minority", "coverage_majority", "singleton_rate",
               "average_set_size", "q_minority", "threshold_sum"]
    by_ds_rows = []
    for ds in st.DATASETS:
        for cp in ["class_conditional_cp", "global_split_cp"]:
            for m in (50, 100):
                sub = df[(df["dataset_id"] == ds) & (df["cp_method"] == cp) & (df["m_minority"] == m)]
                if len(sub) != 30:
                    raise SystemExit("Table 3 aggregation contract violated (expected 30 cells)")
                row = {"dataset_id": ds, "cp_method": cp, "m_minority": m}
                for metric in metrics:
                    row[metric] = float(sub[metric].mean()) if metric in sub.columns else np.nan
                by_ds_rows.append(row)
    by_ds = pd.DataFrame(by_ds_rows)

    summ_rows = []
    for cp in ["class_conditional_cp", "global_split_cp"]:
        for m in (50, 100):
            for metric in metrics:
                vals = by_ds[(by_ds["cp_method"] == cp) & (by_ds["m_minority"] == m)][metric].dropna()
                if len(vals) == 0:
                    summ_rows.append({"cp_method": cp, "m_minority": m, "metric": metric,
                                      "median": np.nan, "q25": np.nan, "q75": np.nan,
                                      "iqr": np.nan, "mean": np.nan, "n_datasets": 0})
                    continue
                q25, q75 = float(vals.quantile(0.25)), float(vals.quantile(0.75))
                summ_rows.append({
                    "cp_method": cp, "m_minority": m, "metric": metric,
                    "median": float(vals.median()), "q25": q25, "q75": q75, "iqr": q75 - q25,
                    "mean": float(vals.mean()), "n_datasets": int(len(vals)),
                })
    summ = pd.DataFrame(summ_rows)

    md_lines = [
        "# Table 3 | Main CP results (m = 50/100)",
        "",
        "Median (q1-q3) across the 8 dataset-level means; each dataset-level value averages the 3 pipelines x 10 seeds (30 cells). "
        "m=50 and m=100 form the preregistered main-comparison range (m=10 boundary and m=20 near-boundary diagnostics are excluded here by design). "
        "q_minority and ThresholdSum are Class-Conditional quantities by construction (Global Split CP has no class-specific thresholds).",
        "",
        "| CP method | m | Minority coverage | Majority coverage | Singleton rate | Avg. set size | q_minority | ThresholdSum |",
        "|---|---:|---|---|---|---|---|---|",
    ]
    disp = {"coverage_minority": "Minority coverage", "coverage_majority": "Majority coverage",
            "singleton_rate": "Singleton rate", "average_set_size": "Avg. set size",
            "q_minority": "q_minority", "threshold_sum": "ThresholdSum"}
    for cp in ["class_conditional_cp", "global_split_cp"]:
        for m in (50, 100):
            cells = []
            for metric in metrics:
                r = summ[(summ["cp_method"] == cp) & (summ["m_minority"] == m) & (summ["metric"] == metric)]
                if len(r) == 0 or np.isnan(r["median"].iloc[0]):
                    cells.append("-")
                else:
                    r = r.iloc[0]
                    cells.append(f"{r['median']:.4f} ({r['q25']:.4f}-{r['q75']:.4f})")
            md_lines.append(f"| {st.CP_LABELS[cp]} | {m} (main) | " + " | ".join(cells) + " |")
    md_lines += [""]
    return summ, by_ds, "\n".join(md_lines)


# ---------------------------------------------------------------------------
# Captions and figure-number mapping
# ---------------------------------------------------------------------------
INPUT_NOTES = (
    "All values are generated programmatically from the frozen results "
    "(results/results_long.parquet, SHA-256 d795d07c36a58619d715c18da6195d929124e43d42abafb45de2dc5134b3dcc2; "
    "1,920 conformal-prediction cells = 8 datasets x 10 seeds x 3 predictive pipelines x 2 CP methods x m in {10,20,50,100}; "
    "alpha = 0.1, m_majority = 200, exact finite-sample order statistic). " + st.AGGREGATION_RULE + " "
    "Every plotted quantity is descriptive: no p-values or bootstrap intervals are displayed in any figure. "
    "Per-cell Wilson 95% intervals exist in the frozen results but are never pooled across seeds. "
    "The probability-scale Beta order-statistic variance is a separate diagnostic quantity on a different scale and is "
    "deliberately never overlaid on empirical threshold variability or singleton rates."
)

CAPTIONS = {
    1: ("Figure 1 | Singleton-rate recovery across minority calibration size. "
        "Singleton rate (fraction of test prediction sets containing exactly one label) as a function of the minority "
        "calibration size m, for Global Split CP (blue, solid, circles) and Class-Conditional CP (vermillion, dashed, squares). "
        "Bold curves are medians across the n = 8 datasets; thin grey curves are individual datasets (each averaged over the "
        "3 predictive pipelines x 10 seeds). Colour, line style, and marker encode the same CP-method contrast for "
        "colour-blind and greyscale readability. Shading and tick annotations mark the boundary diagnostic (m = 10), the "
        "near-boundary diagnostic (m = 20), and the preregistered main-comparison range (m = 50/100); the degenerate m = 10 "
        "behaviour is a diagnostic, not a finding. Source data: "
        "figure1_singleton_rate/figure1_singleton_rate_source_data.csv "
        "(dataset-level values), figure1_singleton_rate_summary.csv (median across datasets), "
        "figure1_singleton_rate_detail_by_model.csv (per-pipeline detail)."),
    2: ("Figure 2 | Minority-class coverage across minority calibration size. "
        "Minority-class test coverage as a function of m for Global Split CP (blue, solid, circles) and Class-Conditional CP "
        "(vermillion, dashed, squares); dotted reference line marks the nominal 90% coverage target (alpha = 0.1). "
        "Aggregation, dual encoding, and boundary/main-comparison annotations as in Fig. 1. Class-Conditional CP raises "
        "minority coverage toward the nominal target relative to Global Split CP in the main-comparison range "
        "(descriptive; preregistered paired effects and intervals are reported in the Stage 11 statistical tables). "
        "Source data: figure2_minority_coverage/figure2_minority_coverage_source_data.csv, "
        "figure2_minority_coverage_summary.csv, figure2_minority_coverage_detail_by_model.csv."),
    3: ("Figure 3 | Threshold geometry: ThresholdSum across minority calibration size. "
        "ThresholdSum = q_minority + q_majority (nonconformity-score scale) for Class-Conditional CP as a function of m; "
        "Global Split CP has no class-specific thresholds and is therefore not shown. Dotted reference line: ThresholdSum = 1. "
        "By the binary prediction-set geometry, ThresholdSum < 1 admits potential empty prediction sets and ThresholdSum > 1 "
        "admits potential doubleton sets. Aggregation and boundary/main-comparison annotations as in Fig. 1. "
        "Source data: figure3_threshold_sum/figure3_threshold_sum_source_data.csv, "
        "figure3_threshold_sum_summary.csv, figure3_threshold_sum_detail_by_model.csv."),
    4: ("Figure 4 | Across-seed variability of the minority threshold (auxiliary). "
        "Empirical across-seed variability of the Class-Conditional minority threshold q_minority (score scale): SD (solid, "
        "filled circles; band = min-max range across the 8 datasets) and IQR (dash-dot, open circles), shown per predictive "
        "pipeline (panels) with the median across datasets. Each dataset-level value uses n = 10 frozen seeds. Variability is "
        "heterogeneous across datasets and pipelines and does not map onto a single closed-form curve. The theoretical "
        "order-statistic uncertainty of the rank r(m) = ceil((m+1)(1-alpha)) quantile decreases conceptually with m, but its "
        "probability-scale Beta variance is a different quantity on a different scale and is deliberately not overlaid here. "
        "Boundary/main-comparison shading as in Fig. 1. Source data: figure4_threshold_variability/figure4_source_data.csv, "
        "figure4_summary.csv; values verified against the Stage 11 D09 threshold-variability table."),
}

TABLE_CAPTIONS = {
    1: ("Table 1 | Dataset characteristics. Raw size, feature composition, registry-defined minority class and ratio, and "
        "train / calibration-pool / test class counts (means over the 10 frozen v1.1 seeds) for the eight locked datasets. "
        "Derived from the locked dataset registry and the 80 v1.1 split manifests; no hand-entered values."),
    2: ("Table 2 | Base predictive performance. Test AUROC and AUPRC per dataset x predictive pipeline, mean (SD) across the "
        "10 frozen seeds. Base performance is invariant across CP method and m by construction; the invariance is verified "
        "in the Stage 12 QA evidence."),
    3: ("Table 3 | Main CP results at the preregistered main-comparison sizes m = 50 and m = 100. Median (q1-q3) across the "
        "8 dataset-level means for minority/majority coverage, singleton rate, average set size, q_minority, and ThresholdSum. "
        "m = 10 (boundary) and m = 20 (near-boundary) are diagnostics and are reported in the figure source data, not here. "
        "Global Split CP has no class-specific thresholds (dash). Descriptive statistics only; confirmatory paired effects "
        "and intervals are in the Stage 11 tables."),
}


def build_mapping(run_id: str, output_paths: dict, input_hashes: dict) -> dict:
    def files(fig_dir: str, stem: str) -> dict:
        return {ext: str(output_paths[f"{fig_dir}/{stem}.{ext}"]) for ext in ("svg", "pdf", "png")}

    figures = []
    for n in (1, 2, 3):
        stem = FIG_SPECS[n]["stem"]
        d = FIG_SPECS[n]["dir"]
        figures.append({
            "figure_number": n,
            "figure_id": d,
            "title": {1: "Singleton Rate Recovery", 2: "Minority Coverage", 3: "ThresholdSum Geometry"}[n],
            "files": files(d, stem),
            "source_data": [
                str(output_paths[f"{d}/{stem}_source_data.csv"]),
                str(output_paths[f"{d}/{stem}_summary.csv"]),
                str(output_paths[f"{d}/{stem}_detail_by_model.csv"]),
            ],
            "caption": CAPTIONS[n],
        })
    d4 = "figure4_threshold_variability"
    figures.append({
        "figure_number": 4,
        "figure_id": d4,
        "title": "Threshold Variability (auxiliary)",
        "files": files(d4, "figure4_threshold_variability"),
        "source_data": [
            str(output_paths[f"{d4}/figure4_source_data.csv"]),
            str(output_paths[f"{d4}/figure4_summary.csv"]),
        ],
        "caption": CAPTIONS[4],
    })
    tables = []
    for n, name in ((1, "table1_dataset_characteristics"), (2, "table2_base_predictive_performance"),
                    (3, "table3_main_cp_results")):
        entry = {
            "table_number": n,
            "table_id": name,
            "files": {"csv": str(output_paths[f"tables/{name}.csv"]), "markdown": str(output_paths[f"tables/{name}.md"])},
            "caption": TABLE_CAPTIONS[n],
        }
        if n == 3:
            entry["source_data"] = [str(output_paths["tables/table3_main_cp_results_by_dataset.csv"])]
        tables.append(entry)
    return {
        "schema_version": "1.0",
        "run_id": run_id,
        "producer": "src/run_stage12_figures.py",
        "style_config": str(output_paths["figures/style_config.json"]),
        "captions": str(output_paths["figures/captions.md"]),
        "shared_caption_preamble": INPUT_NOTES,
        "figures": figures,
        "tables": tables,
        "frozen_inputs": input_hashes,
        "annotation_convention": {
            "m=10": "boundary diagnostic",
            "m=20": "near-boundary diagnostic",
            "m=50": "main comparison",
            "m=100": "main comparison",
        },
    }


# ---------------------------------------------------------------------------
# QA: automated sampling re-checks of figure data against source tables
# ---------------------------------------------------------------------------
def run_qa(df: pd.DataFrame, output_paths: dict, registry: dict, splits: dict,
           fig4_src: pd.DataFrame, table2_long: pd.DataFrame, table1_df: pd.DataFrame,
           table3_by_ds: pd.DataFrame, ev: EventLog) -> dict:
    rng = random.Random(QA_SAMPLING_SEED)
    checks = []

    def add(cid, ok, **kw):
        checks.append({"id": cid, "status": "PASS" if ok else "FAIL", **kw})
        if not ok:
            raise SystemExit(f"QA FAIL: {cid}: {kw}")

    # Q1 grid completeness
    keys = ["dataset_id", "seed", "model", "cp_method", "m_minority"]
    add("Q1-grid-complete", len(df) == 1920 and df[keys].drop_duplicates().shape[0] == 1920,
        n_rows=int(len(df)), unique_cells=1920)

    # Q2 sampled dataset-level source-data values vs independent recomputation from frozen parquet
    src_frames = []
    for n in (1, 2, 3):
        f = pd.read_csv(output_paths[f"{FIG_SPECS[n]['dir']}/{FIG_SPECS[n]['stem']}_source_data.csv"])
        f["fig"] = n
        src_frames.append(f)
    allsrc = pd.concat(src_frames, ignore_index=True)
    metric_by_fig = {n: FIG_SPECS[n]["metric"] for n in (1, 2, 3)}
    idx = rng.sample(range(len(allsrc)), 48)
    max_diff = 0.0
    for i in idx:
        row = allsrc.iloc[i]
        metric = metric_by_fig[row["fig"]]
        sub = df[(df["dataset_id"] == row["dataset_id"]) & (df["cp_method"] == row["cp_method"]) & (df["m_minority"] == row["m_minority"])]
        assert len(sub) == 30
        recomputed = float(sub[metric].sum() / len(sub))
        max_diff = max(max_diff, abs(recomputed - float(row["value"])))
    add("Q2-source-data-vs-frozen-parquet", max_diff <= 1e-12, n_sampled=48, max_abs_diff=max_diff,
        detail="dataset-level means recomputed via sum/count from the frozen parquet")

    # Q3 summary medians vs source data
    max_diff = 0.0
    for n in (1, 2, 3):
        src = pd.read_csv(output_paths[f"{FIG_SPECS[n]['dir']}/{FIG_SPECS[n]['stem']}_source_data.csv"])
        summ = pd.read_csv(output_paths[f"{FIG_SPECS[n]['dir']}/{FIG_SPECS[n]['stem']}_summary.csv"])
        for _, r in summ.iterrows():
            v = src[(src["cp_method"] == r["cp_method"]) & (src["m_minority"] == r["m_minority"])]["value"]
            max_diff = max(max_diff, abs(float(v.median()) - float(r["median"])), abs(float(v.mean()) - float(r["mean"])))
    add("Q3-summary-vs-source-data", max_diff <= 1e-12, max_abs_diff=max_diff)

    # Q4 fig4 source data vs Stage 11 D09 threshold-variability table (all 384 rows)
    s11 = pd.read_csv(P_S11_TV)
    tv_mine = df[df["cp_method"] == "class_conditional_cp"]
    full_rows = []
    for model in st.MODELS:
        for ds in st.DATASETS:
            for m in st.M_VALUES:
                v = tv_mine[(tv_mine["model"] == model) & (tv_mine["dataset_id"] == ds) & (tv_mine["m_minority"] == m)]
                full_rows.append({"dataset_id": ds, "model": model, "m_minority": m,
                                  "threshold_metric": "q_minority",
                                  "across_seed_sd": v["q_minority"].std(ddof=1),
                                  "across_seed_iqr": v["q_minority"].quantile(0.75) - v["q_minority"].quantile(0.25),
                                  "across_seed_mean": v["q_minority"].mean()})
                full_rows.append({"dataset_id": ds, "model": model, "m_minority": m,
                                  "threshold_metric": "q_majority",
                                  "across_seed_sd": v["q_majority"].std(ddof=1),
                                  "across_seed_iqr": v["q_majority"].quantile(0.75) - v["q_majority"].quantile(0.25),
                                  "across_seed_mean": v["q_majority"].mean()})
                full_rows.append({"dataset_id": ds, "model": model, "m_minority": m,
                                  "threshold_metric": "threshold_sum",
                                  "across_seed_sd": v["threshold_sum"].std(ddof=1),
                                  "across_seed_iqr": v["threshold_sum"].quantile(0.75) - v["threshold_sum"].quantile(0.25),
                                  "across_seed_mean": v["threshold_sum"].mean()})
    for model in st.MODELS:
        for ds in st.DATASETS:
            for m in st.M_VALUES:
                v = df[(df["cp_method"] == "global_split_cp") & (df["model"] == model) & (df["dataset_id"] == ds) & (df["m_minority"] == m)]
                full_rows.append({"dataset_id": ds, "model": model, "m_minority": m,
                                  "threshold_metric": "q_global",
                                  "across_seed_sd": v["q_global"].std(ddof=1),
                                  "across_seed_iqr": v["q_global"].quantile(0.75) - v["q_global"].quantile(0.25),
                                  "across_seed_mean": v["q_global"].mean()})
    full = pd.DataFrame(full_rows)
    mg = full.merge(s11, on=["dataset_id", "model", "m_minority", "threshold_metric"],
                    suffixes=("_mine", "_s11"))
    ok = len(mg) == 384
    max_diff = 0.0
    for col in ("across_seed_sd", "across_seed_iqr", "across_seed_mean"):
        max_diff = max(max_diff, float((mg[f"{col}_mine"] - mg[f"{col}_s11"]).abs().max()))
    ok = ok and max_diff <= 1e-12
    f4w = pd.read_csv(output_paths["figure4_threshold_variability/figure4_source_data.csv"])
    qm = (full[full["threshold_metric"] == "q_minority"]
          .merge(f4w, on=["dataset_id", "model", "m_minority"], suffixes=("_full", "_fig")))
    for col in ("across_seed_sd", "across_seed_iqr", "across_seed_mean"):
        ok = ok and len(qm) == 96 and float((qm[f"{col}_full"] - qm[f"{col}_fig"]).abs().max()) <= 1e-12
    add("Q4-fig4-vs-stage11-threshold-variability", ok, rows_compared=int(len(mg)), max_abs_diff=max_diff,
        detail="recomputed SD/IQR/mean of all four threshold metrics match the Stage 11 D09 table; "
               "the written figure4 source data matches the recomputed q_minority rows (96/96)")

    # Q5 Table 2: within-unit invariance (across CP x m within each seed) + seed-mean agreement
    max_spread, max_diff = 0.0, 0.0
    for ds in st.DATASETS:
        for model in st.MODELS:
            sub = df[(df["dataset_id"] == ds) & (df["model"] == model)]
            for metric in ("auroc", "auprc"):
                spread = float(sub.groupby("seed")[metric].agg(lambda v: v.max() - v.min()).max())
                max_spread = max(max_spread, spread)
                mu = float(table2_long[(table2_long["dataset_id"] == ds) & (table2_long["model"] == model)
                                       & (table2_long["metric"] == metric)]["mean_across_seeds"].iloc[0])
                max_diff = max(max_diff, abs(mu - float(sub[metric].sum() / len(sub))))
    add("Q5-table2-invariance-and-values", max_spread <= 1e-12 and max_diff <= 1e-12,
        max_within_unit_spread=max_spread, max_mean_abs_diff=max_diff, units_checked=24)

    # Q6 Table 1 vs registry + split manifests (full recomputation, all 8 datasets)
    max_diff = 0.0
    ok = True
    for ds in st.DATASETS:
        r = registry[ds]
        ratio = r["raw_class_counts"][r["minority_original_label"]] / r["n_rows"]
        t1 = table1_df[table1_df["dataset_id"] == ds].iloc[0]
        max_diff = max(max_diff, abs(ratio - float(t1["minority_ratio"])))
        for split, split_key in (("train", "train"), ("cal_pool", "calibration_pool"), ("test", "test")):
            for cls in ("majority", "minority"):
                vals = [splits[ds][seed]["class_counts"][split_key][cls] for seed in st.SEEDS]
                ok = ok and (float(np.mean(vals)) == float(t1[f"{split}_{cls}_mean"])
                             and int(np.min(vals)) == int(t1[f"{split}_{cls}_min"])
                             and int(np.max(vals)) == int(t1[f"{split}_{cls}_max"]))
    add("Q6-table1-vs-registry-and-splits", ok and max_diff <= 1e-12, max_abs_diff=max_diff, datasets_checked=8)

    # Q7 structural range/identity constraints on the frozen data used by figures/tables
    ok = (
        df[["coverage_minority", "coverage_majority", "coverage_overall"]].isin([np.inf, -np.inf]).sum().sum() == 0
        and df[["coverage_minority", "coverage_majority", "coverage_overall"]].apply(lambda c: c.between(0, 1).all()).all()
        and df[["q_minority", "q_majority", "q_global"]].dropna().apply(lambda c: c.between(0, 1).all()).all()
    )
    ident_setsize = float((df["average_set_size"] - (df["singleton_rate"] + 2 * df["doubleton_rate"])).abs().max())
    ident_sum = float((df["threshold_sum"] - (df["q_minority"] + df["q_majority"])).dropna().abs().max())
    ident_gap = float((df["threshold_gap"] - (df["q_minority"] - df["q_majority"]).abs()).dropna().abs().max())
    ok = ok and ident_setsize <= 1e-12 and ident_sum <= 1e-12 and ident_gap <= 1e-12
    # Table 3 by-dataset rows consistent with dataset-level means
    t3diff = 0.0
    for _, r in table3_by_ds.iterrows():
        sub = df[(df["dataset_id"] == r["dataset_id"]) & (df["cp_method"] == r["cp_method"]) & (df["m_minority"] == r["m_minority"])]
        t3diff = max(t3diff, abs(float(sub["singleton_rate"].sum() / 30) - float(r["singleton_rate"])))
    ok = ok and t3diff <= 1e-12
    add("Q7-range-and-identity-constraints", ok, max_set_size_identity_err=ident_setsize,
        max_threshold_sum_identity_err=ident_sum, max_threshold_gap_identity_err=ident_gap,
        max_table3_by_dataset_diff=t3diff)

    ev.log("qa_complete", n_checks=len(checks))
    return {"qa_id": "stage12_qa_v1.0", "created_utc": utc_now(),
            "rng_seed_for_sampling": QA_SAMPLING_SEED, "checks": checks,
            "verdict": "PASS" if all(c["status"] == "PASS" for c in checks) else "FAIL"}


# ---------------------------------------------------------------------------
# Output layout
# ---------------------------------------------------------------------------
def plan_outputs() -> dict:
    specs = {
        "figures/style_config.json": FIG_DIR / "style_config.json",
        "figures/captions.md": FIG_DIR / "captions.md",
        "figures/figure_number_mapping.json": FIG_DIR / "figure_number_mapping.json",
    }
    for n in (1, 2, 3):
        spec = FIG_SPECS[n]
        d = FIG_DIR / spec["dir"]
        specs[f"{spec['dir']}/{spec['stem']}.svg"] = d / f"{spec['stem']}.svg"
        specs[f"{spec['dir']}/{spec['stem']}.pdf"] = d / f"{spec['stem']}.pdf"
        specs[f"{spec['dir']}/{spec['stem']}.png"] = d / f"{spec['stem']}.png"
        specs[f"{spec['dir']}/{spec['stem']}_source_data.csv"] = d / f"{spec['stem']}_source_data.csv"
        specs[f"{spec['dir']}/{spec['stem']}_summary.csv"] = d / f"{spec['stem']}_summary.csv"
        specs[f"{spec['dir']}/{spec['stem']}_detail_by_model.csv"] = d / f"{spec['stem']}_detail_by_model.csv"
    d4 = FIG_DIR / "figure4_threshold_variability"
    for ext in ("svg", "pdf", "png"):
        specs[f"figure4_threshold_variability/figure4_threshold_variability.{ext}"] = d4 / f"figure4_threshold_variability.{ext}"
    specs["figure4_threshold_variability/figure4_source_data.csv"] = d4 / "figure4_source_data.csv"
    specs["figure4_threshold_variability/figure4_summary.csv"] = d4 / "figure4_summary.csv"
    for name in ("table1_dataset_characteristics", "table2_base_predictive_performance", "table3_main_cp_results"):
        specs[f"tables/{name}.csv"] = TABLE_DIR / f"{name}.csv"
        specs[f"tables/{name}.md"] = TABLE_DIR / f"{name}.md"
    specs["tables/table3_main_cp_results_by_dataset.csv"] = TABLE_DIR / "table3_main_cp_results_by_dataset.csv"
    return specs


def main() -> int:
    # ---- run identity (exclusive creation) ----
    pre_hashes = {
        "results_long.parquet": sha256_file(P_RESULTS),
        "stage11_threshold_variability.csv": sha256_file(P_S11_TV),
        "dataset_registry_v1.0.1.json": sha256_file(P_REGISTRY),
        "stage12_style.py": sha256_file(ROOT / "src/stage12_style.py"),
    }
    short = hashlib.sha256("|".join(pre_hashes.values()).encode()).hexdigest()[:8]
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_stage12-figures_" + short
    run_dir = ROOT / "artifacts/stage12" / run_id
    if run_dir.exists():
        raise SystemExit(f"Run directory already exists (exclusive creation violated): {run_dir}")

    outputs = plan_outputs()
    for rel, p in outputs.items():
        if p.exists():
            raise SystemExit(f"Output path already exists (exclusive creation violated): {p}")

    run_dir.mkdir(parents=True)
    ev = EventLog(run_id, run_dir / "events.jsonl")
    ev.log("run_start", run_id=run_id)

    # ---- gates ----
    gate_results = run_gates(ev)

    # ---- load inputs ----
    df = pd.read_parquet(P_RESULTS)
    registry = load_registry()
    splits = load_split_manifests()
    ev.log("inputs_loaded", n_cells=int(len(df)), n_registry_records=len(registry), n_split_manifests=80)

    # ---- source data ----
    fig_src, fig_summ, fig_det, fig4_src, fig4_summ = {}, {}, {}, None, None
    for n in (1, 2, 3):
        src, summ, det = build_fig_source_data(df, FIG_SPECS[n])
        fig_src[n], fig_summ[n], fig_det[n] = src, summ, det
    fig4_src, fig4_summ = build_fig4_data(df)
    ev.log("source_data_computed")

    for n in (1, 2, 3):
        d = FIG_SPECS[n]["dir"]
        write_csv_lf(fig_src[n], outputs[f"{d}/{FIG_SPECS[n]['stem']}_source_data.csv"])
        write_csv_lf(fig_summ[n], outputs[f"{d}/{FIG_SPECS[n]['stem']}_summary.csv"])
        write_csv_lf(fig_det[n], outputs[f"{d}/{FIG_SPECS[n]['stem']}_detail_by_model.csv"])
    write_csv_lf(fig4_src, outputs["figure4_threshold_variability/figure4_source_data.csv"])
    write_csv_lf(fig4_summ, outputs["figure4_threshold_variability/figure4_summary.csv"])
    ev.log("source_data_written")

    # ---- figures ----
    for n in (1, 2, 3):
        stem_path = FIG_DIR / FIG_SPECS[n]["dir"] / FIG_SPECS[n]["stem"]
        draw_m_figure(stem_path, FIG_SPECS[n], fig_src[n])
        ev.log("figure_rendered", figure=n)
    draw_fig4(FIG_DIR / "figure4_threshold_variability" / "figure4_threshold_variability", fig4_src)
    ev.log("figure_rendered", figure=4)

    # ---- style config ----
    write_json_lf(outputs["figures/style_config.json"], st.style_config_payload())

    # ---- tables ----
    table1_df, table1_md = build_table1(registry, splits)
    write_csv_lf(table1_df, outputs["tables/table1_dataset_characteristics.csv"])
    write_text_lf(outputs["tables/table1_dataset_characteristics.md"], table1_md)
    table2_long, table2_md = build_table2(df)
    write_csv_lf(table2_long, outputs["tables/table2_base_predictive_performance.csv"])
    write_text_lf(outputs["tables/table2_base_predictive_performance.md"], table2_md)
    table3_summ, table3_by_ds, table3_md = build_table3(df)
    write_csv_lf(table3_summ, outputs["tables/table3_main_cp_results.csv"])
    write_csv_lf(table3_by_ds, outputs["tables/table3_main_cp_results_by_dataset.csv"])
    write_text_lf(outputs["tables/table3_main_cp_results.md"], table3_md)
    ev.log("tables_written", n_tables=3)

    # ---- captions + mapping ----
    cap_lines = [
        "# Stage 12 — publication figure and table captions",
        "",
        "Auto-generated; no hand-entered numbers. Shared preamble applies to every figure.",
        "",
        "## Shared preamble",
        "",
        INPUT_NOTES,
        "",
    ]
    for n in (1, 2, 3, 4):
        cap_lines += [CAPTIONS[n], ""]
    for n in (1, 2, 3):
        cap_lines += [TABLE_CAPTIONS[n], ""]
    cap_lines += [
        "## Caption conventions",
        "",
        "- Dual encoding: colour + line style + marker (fill) always encode the same contrast (colour-blind and greyscale safe).",
        "- m = 10 boundary diagnostic; m = 20 near-boundary diagnostic; m = 50/100 preregistered main comparison.",
        "- Wilson intervals: per-cell binomial intervals in the frozen results only, never pooled.",
        "- D08 (across-dataset) and across-seed bootstrap intervals live in the Stage 11 tables, not in these descriptive figures.",
    ]
    write_text_lf(outputs["figures/captions.md"], "\n".join(cap_lines) + "\n")

    input_hashes = {p: {"sha256": sha256_file(ROOT / p)} for p in (
        "results/results_long.parquet",
        "results/results_manifest.json",
        "configs/formal_run_manifest_v1.1.yaml",
        "results/stats/stage11_threshold_variability.csv",
        "results/stats/stage11_ab_endpoints.csv",
        "artifacts/stage02/dataset_registry_v1.0.1.json",
    )}
    mapping = build_mapping(run_id, outputs, input_hashes)
    write_json_lf(outputs["figures/figure_number_mapping.json"], mapping)
    ev.log("captions_and_mapping_written")

    # ---- QA (reads back the written outputs) ----
    qa = run_qa(df, outputs, registry, splits, fig4_src, table2_long, table1_df, table3_by_ds, ev)
    write_json_lf(run_dir / "stage12_qa_evidence.json", qa)
    ev.log("qa_evidence_written", verdict=qa["verdict"])

    # ---- manifest + status ----
    output_records = []
    for rel, p in outputs.items():
        output_records.append({
            "path": str(p), "relative_key": rel, "sha256": sha256_file(p), "bytes": p.stat().st_size,
        })
    env_info = {
        "python": platform.python_version(),
        "matplotlib": matplotlib.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "platform": platform.platform(),
    }
    manifest = {
        "artifact_id": "stage12_publication_figures_v1.0.0",
        "artifact_version": ARTIFACT_VERSION,
        "status": "PASS",
        "producer_stage": "Stage 12 - publication figures and main tables",
        "producer_script": "src/run_stage12_figures.py",
        "producer_script_sha256": sha256_file(ROOT / "src/run_stage12_figures.py"),
        "style_module_sha256": pre_hashes["stage12_style.py"],
        "created_utc": utc_now(),
        "run_id": run_id,
        "run_dir": str(run_dir),
        "backend": "python/matplotlib",
        "environment": env_info,
        "parent_artifacts": {
            "frozen_results": {"path": "results/results_long.parquet", "sha256": FROZEN_PARQUET_SHA256},
            "frozen_formal_manifest": {"path": "configs/formal_run_manifest_v1.1.yaml", "sha256": FROZEN_FORMAL_MANIFEST_SHA256},
            "stage11_tables": ["results/stats/stage11_threshold_variability.csv", "results/stats/stage11_ab_endpoints.csv"],
            "dataset_registry": {"path": "artifacts/stage02/dataset_registry_v1.0.1.json"},
            "v1_1_split_manifests": {"path": "artifacts/splits/v1.1/", "count": 80},
        },
        "input_gate_results": gate_results,
        "aggregation_contract": st.AGGREGATION_RULE,
        "qa": {"verdict": qa["verdict"], "evidence": str(run_dir / "stage12_qa_evidence.json"), "n_checks": len(qa["checks"])},
        "outputs": output_records,
        "immutability": {
            "exclusive_creation": True,
            "frozen_inputs_modified": False,
            "line_endings": "LF",
        },
    }
    write_json_lf(run_dir / "stage12_manifest.json", manifest)
    status = {
        "run_id": run_id,
        "stage": "stage12",
        "status": "PASS",
        "created_utc": utc_now(),
        "gates": "PASS",
        "qa": qa["verdict"],
        "n_figures": 4,
        "n_tables": 3,
        "outputs_root": ["manuscript/figures", "manuscript/tables"],
        "frozen_results_unmodified": True,
    }
    write_json_lf(run_dir / "stage12_status.json", status)
    ev.log("stage12_complete", verdict="PASS")

    print(f"STAGE12 PASS run_id={run_id}")
    print(f"figures -> {FIG_DIR}")
    print(f"tables  -> {TABLE_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
