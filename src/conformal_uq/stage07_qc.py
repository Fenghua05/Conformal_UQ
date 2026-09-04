"""Independent Stage 07 record checks and diagnostic-only matplotlib figures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .conformal import CP_METHODS, M_MINORITY, evaluate_split_cp, select_nested_calibration_subsets
from .data import load_dataset_registry, load_locked_dataset, registry_record
from .prediction_cache import read_valid_cache
from .results_schema import validate_results_records
from .split import make_stratified_split


DIAGNOSTIC_FIGURES = ("overall_coverage", "classwise_coverage", "set_decomposition", "threshold_geometry")
PALETTE = {"global_split_cp": "#0F4D92", "class_conditional_cp": "#B64342", "minority": "#9A4D8E", "majority": "#42949E", "singleton": "#3775BA", "empty": "#E9A6A1", "doubleton": "#8BCF8B"}


def _minority_label(root: Path, dataset_id: str, registry_path: Path) -> int:
    record = registry_record(load_dataset_registry(root, registry_path), dataset_id)
    return int(record["label_mapping_to_protocol_binary"][str(record["minority_original_label"])])


def _figure_path(root: Path, name: str) -> list[str]:
    root.mkdir(parents=True, exist_ok=True)
    return [str(root / f"{name}.{ext}") for ext in ("png", "pdf")]


def _save(fig: Any, root: Path, name: str) -> list[str]:
    paths = _figure_path(root, name)
    for item in paths:
        fig.savefig(item, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return paths


def _style(ax: Any, ylabel: str) -> None:
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.22, linewidth=0.7)


def diagnostic_figures(records: Iterable[dict[str, Any]], output_dir: Path) -> dict[str, list[str]]:
    frame = pd.DataFrame(list(records))
    figures: dict[str, list[str]] = {}
    x = np.asarray(M_MINORITY)
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    for (model, method), part in frame.groupby(["model", "cp_method"], sort=True):
        means = part.groupby("m_minority")["coverage_overall"].mean().reindex(M_MINORITY)
        ax.plot(x, means, marker="o", linewidth=1.8, color=PALETTE[method], linestyle="-" if model == "logistic_regression" else ("--" if model == "xgboost" else ":"), label=f"{model} · {method}")
    ax.axhline(0.9, color="black", linewidth=1.0, linestyle="--", label="nominal 0.90")
    ax.set_xscale("symlog", linthresh=10); ax.set_xticks(x); ax.set_xticklabels([str(v) for v in x]); ax.set_ylim(0, 1); ax.set_xlabel("minority calibration size m")
    _style(ax, "mean overall coverage"); ax.legend(fontsize=7, ncol=2, frameon=False)
    figures["overall_coverage"] = _save(fig, output_dir, "overall_coverage")

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), sharey=True)
    for ax, field, label, color in zip(axes, ("coverage_minority", "coverage_majority"), ("minority-class coverage", "majority-class coverage"), (PALETTE["minority"], PALETTE["majority"])):
        for (model, method), part in frame.groupby(["model", "cp_method"], sort=True):
            means = part.groupby("m_minority")[field].mean().reindex(M_MINORITY)
            ax.plot(x, means, marker="o", linewidth=1.8, color=color, alpha=0.95 if method == "class_conditional_cp" else 0.45, linestyle="-" if model == "logistic_regression" else ("--" if model == "xgboost" else ":"), label=f"{model} · {method}")
        ax.axhline(0.9, color="black", linewidth=1, linestyle="--"); ax.set_xscale("symlog", linthresh=10); ax.set_xticks(x); ax.set_xticklabels([str(v) for v in x]); ax.set_ylim(0, 1); ax.set_xlabel("m"); _style(ax, label)
    axes[1].legend(fontsize=6.5, frameon=False)
    figures["classwise_coverage"] = _save(fig, output_dir, "classwise_coverage")

    fig, ax = plt.subplots(figsize=(8.6, 4.7))
    grouped = frame.groupby(["cp_method", "m_minority"])[["empty_rate", "singleton_rate", "doubleton_rate"]].mean().reindex(pd.MultiIndex.from_product([CP_METHODS, M_MINORITY], names=["cp_method", "m_minority"]))
    positions = np.arange(len(grouped)); bottom = np.zeros(len(grouped))
    for field, label, color in (("empty_rate", "empty", PALETTE["empty"]), ("singleton_rate", "singleton", PALETTE["singleton"]), ("doubleton_rate", "doubleton", PALETTE["doubleton"])):
        ax.bar(positions, grouped[field].to_numpy(), bottom=bottom, width=0.78, color=color, edgecolor="black", linewidth=0.45, label=label); bottom += grouped[field].to_numpy()
    ax.set_xticks(positions); ax.set_xticklabels([f"{'G' if method == 'global_split_cp' else 'CC'}\nm={m}" for method, m in grouped.index], fontsize=8); ax.set_ylim(0, 1); _style(ax, "mean prediction-set rate"); ax.legend(frameon=False, ncol=3, fontsize=8)
    figures["set_decomposition"] = _save(fig, output_dir, "set_decomposition")

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), sharex=True)
    conditional = frame[frame["cp_method"] == "class_conditional_cp"]
    for model, part in conditional.groupby("model", sort=True):
        means = part.groupby("m_minority")[["threshold_gap", "threshold_sum"]].mean().reindex(M_MINORITY)
        linestyle = "-" if model == "logistic_regression" else ("--" if model == "xgboost" else ":")
        axes[0].plot(x, means["threshold_gap"], marker="o", linestyle=linestyle, label=model)
        axes[1].plot(x, means["threshold_sum"], marker="o", linestyle=linestyle, label=model)
    for ax, field in zip(axes, ("mean |q_minority − q_majority|", "mean q_minority + q_majority")):
        ax.set_xscale("symlog", linthresh=10); ax.set_xticks(x); ax.set_xticklabels([str(v) for v in x]); ax.set_xlabel("m"); _style(ax, field); ax.legend(frameon=False, fontsize=8)
    figures["threshold_geometry"] = _save(fig, output_dir, "threshold_geometry")
    return figures


def independent_qc(root: Path, records: list[dict[str, Any]], lock: dict[str, Any]) -> dict[str, Any]:
    errors = validate_results_records(records)
    expected = len(lock["pilot_dataset_ids"]) * len(lock["seeds"]) * 3 * len(CP_METHODS) * len(M_MINORITY)
    pair_groups: dict[tuple[str, int, str, int], list[dict[str, Any]]] = {}
    for record in records:
        pair_groups.setdefault((record["dataset_id"], record["seed"], record["model"], record["m_minority"]), []).append(record)
    subset_identity = all(len(group) == 2 and len({row["subset_hash"] for row in group}) == 1 for group in pair_groups.values())
    rank_m10 = all(row["rank_minority"] == 10 for row in records if row["cp_method"] == "class_conditional_cp" and row["m_minority"] == 10)
    rank_m20 = all(row["rank_minority"] == 19 for row in records if row["cp_method"] == "class_conditional_cp" and row["m_minority"] == 20)
    nested_subset = True
    recomputation = True
    for dataset_id in lock["pilot_dataset_ids"]:
        for seed in lock["seeds"]:
            minority = _minority_label(root, dataset_id, root / lock["registry_path"])
            for model in ("logistic_regression", "xgboost", "tabpfn"):
                matching = [row for row in records if row["dataset_id"] == dataset_id and row["seed"] == seed and row["model"] == model]
                if len(matching) != 8:
                    recomputation = False; continue
                first = matching[0]
                cache_dir = root / "artifacts" / "caches" / f"cfg-{first['config_hash'][:12]}" / f"code-{first['code_hash'][:12]}" / dataset_id / f"seed-{seed}" / model
                table = load_locked_dataset(root, dataset_id, registry_path=root / lock["registry_path"])
                split = make_stratified_split(table, seed, protocol_version="v1.0")
                manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
                cached = read_valid_cache(cache_dir, manifest["provenance"], {"calibration_pool": split.ids.calibration_pool, "test": split.ids.test}, {"calibration_pool": table.subset_labels(split.ids.calibration_pool).to_numpy(dtype="int8"), "test": table.subset_labels(split.ids.test).to_numpy(dtype="int8")})
                subsets = select_nested_calibration_subsets(cached["ids"]["calibration_pool"], cached["labels"]["calibration_pool"], protocol_version="v1.0", dataset_id=dataset_id, base_seed=seed, minority_label=minority)
                nested_subset = nested_subset and all(set(subsets.minority_ids_by_m[a]).issubset(subsets.minority_ids_by_m[b]) for a, b in zip(M_MINORITY, M_MINORITY[1:]))
                for row in matching:
                    direct = evaluate_split_cp(cached["ids"]["calibration_pool"], cached["labels"]["calibration_pool"], cached["probabilities"]["calibration_pool"], cached["labels"]["test"], cached["probabilities"]["test"], subsets, m_minority=row["m_minority"], cp_method=row["cp_method"])
                    for field in ("subset_hash", "rank_global", "rank_minority", "rank_majority", "covered_count_overall", "covered_count_minority", "covered_count_majority"):
                        recomputation = recomputation and direct[field] == row[field]
                    for field in ("q_global", "q_minority", "q_majority", "threshold_gap", "threshold_sum", "coverage_overall", "coverage_minority", "coverage_majority", "singleton_rate", "empty_rate", "doubleton_rate", "average_set_size"):
                        a, b = direct[field], row[field]
                        recomputation = recomputation and ((a is None and b is None) or (a is not None and b is not None and abs(float(a) - float(b)) <= 1e-12))
    global_rows = [row for row in records if row["cp_method"] == "global_split_cp"]
    systematic: list[dict[str, Any]] = []
    for (model, m), rows in pd.DataFrame(global_rows).groupby(["model", "m_minority"], sort=True):
        flags = [not (row["coverage_overall_wilson_low"] <= 0.9 <= row["coverage_overall_wilson_high"]) for _, row in rows.iterrows()]
        systematic.append({"model": model, "m_minority": int(m), "flagged_cells": int(sum(flags)), "total_cells": int(len(flags)), "flagged_fraction": float(np.mean(flags)), "implementation_review_required": bool(np.mean(flags) > 0.5)})
    return {"status": "PASS" if not errors and len(records) == expected and subset_identity and rank_m10 and rank_m20 and nested_subset and recomputation else "FAIL", "expected_cells": expected, "validated_cells": len(records), "schema_errors": errors, "checks": {"m10_max_score_rank": rank_m10, "m20_19th_order_statistic_rank": rank_m20, "nested_subset": nested_subset, "cp_subset_identity": subset_identity, "q_threshold_sum_and_geometry": recomputation, "classwise_coverage_and_wilson": not errors, "probability_mapping_from_validated_caches": True}, "global_coverage_diagnostic": {"nominal": 0.9, "groups": systematic, "implementation_review_required": any(item["implementation_review_required"] for item in systematic), "interpretation_rule": "A flag requires implementation review first; it is not a scientific finding."}}
