"""Stage 11 — preregistered statistical analysis on the frozen Stage 09/10 results.

Read-only with respect to every frozen artifact. Verifies all frozen hashes
BEFORE any analysis; if any verification fails, writes a failure record and no
analysis output. All Stage 11 outputs are new files created exclusively.

Preregistered frame (protocol v1.1 §8; STAGE_01_DECISION_TABLE D08/D09/D10):
  - Unit of inference: the DATASET (8 datasets). Within each dataset, the paired
    seed-level contrasts are aggregated first: d_j = mean over complete paired
    contrasts (3 pipelines x 10 seeds for comparisons A/B; 10 seeds for the
    comparison-C pipeline pairs). The 8x10 seed cells are never treated as 80
    independent research units.
  - Confirmatory effect: median(d_j) over the 8 datasets; two-sided 95% CI =
    20,000-replicate percentile bootstrap of the 8 whole-dataset effects with a
    D01-derived RNG seed (convention identical to Stage 10, documented below).
  - Auxiliary tests: exact two-sided Wilcoxon signed-rank on the eight dataset
    effects (zeros discarded and counted), Holm-corrected within each family.
    Confirmatory families: A (RQ1-A primaries, k=2) and B (RQ2-B, k=10).
    RQ3/comparison-C p-values and the six A-secondary p-values are computed
    only because the Stage 11 instruction requests Wilcoxon/Holm reporting for
    A/B/C; they are flagged exploratory_not_preregistered and excluded from
    confirmatory claims (D10 limits formal testing to A and B).
  - D09: Wilson intervals are within-cell (test-set binomial) intervals and are
    never pooled across seeds; across-seed outputs are descriptive
    (n/mean/median/SD ddof=1/IQR). The across-seed bootstrap CI is an
    explicitly flagged exploratory descriptive supplement, distinct from both
    Wilson and D08 intervals.

Outputs (exclusive creation; nothing overwritten):
  results/stats/            machine-readable tables + stage11_manifest.json
  artifacts/stage11/<run>/  evidence, events, status, analysis plan
  reports/                  STAGE11_ANALYSIS_REPORT.md
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
STATS_DIR = RESULTS_DIR / "stats"
ARTIFACTS_DIR = ROOT / "artifacts"
STAGE11_ROOT = ARTIFACTS_DIR / "stage11"
FAILURES_DIR = ARTIFACTS_DIR / "failures"
REPORTS_DIR = ROOT / "reports"

FROZEN_MANIFEST_SHA256 = "0690795b1ea68a148f069a44dccb09a9245d21a097e7174dbd5ec3002f24172d"
RESULTS_MANIFEST_V1_SHA256 = "c116e1f1cb69abfad84e11fc641ac6a3ab11bb80eb43357dbe7431534ef7efae"
STAGE10_RUN_DIR = "20260831T101122Z_stage10-results-freeze_63eb04da"
STAGE10_EVIDENCE_REL = f"artifacts/stage10/{STAGE10_RUN_DIR}/results_qc_evidence.json"
STAGE10_HANDOFF_REL = "handoffs/STAGE_10_HANDOFF.md"
RESULTS_LONG_REL = "results/results_long.parquet"

Z95 = 1.959963984540054
SEEDS = [104729, 130363, 155921, 196613, 262147, 318281, 374209, 419893, 481517, 552721]
DATASETS = [
    "openml_3_kr_vs_kp", "openml_24_mushroom", "openml_1486_nomao", "openml_1489_phoneme",
    "openml_1590_adult", "openml_4534_phishingwebsite", "openml_23512_higgs",
    "openml_23517_numerai28_6",
]
MODELS = ["logistic_regression", "xgboost", "tabpfn"]
CP_METHODS = ["global_split_cp", "class_conditional_cp"]
M_VALUES = [10, 20, 50, 100]
KEY_COLS = ["dataset_id", "seed", "model", "cp_method", "m_minority"]
N_BOOT = 20000

CONFIRMATORY_METRICS = ["coverage_minority", "coverage_majority", "coverage_disparity",
                        "singleton_rate", "average_set_size", "empty_rate", "doubleton_rate"]
DESCRIPTIVE_METRICS = CONFIRMATORY_METRICS + [
    "coverage_overall", "auroc", "auprc"]
CC_ONLY_DESCRIPTIVE = ["q_minority", "q_majority", "threshold_gap", "threshold_sum"]
GLOBAL_ONLY_DESCRIPTIVE = ["q_global"]

# ---------------------------------------------------------------- families ----
# Family A confirmatory: the two preregistered RQ1-A primary endpoints
# (identical tags/definitions to the Stage 10 determinability report).
FAMILY_A = [
    ("rq1a_cc_m100_minus_m50_singleton_rate", "singleton_rate",
     ("class_conditional_cp", 100), ("class_conditional_cp", 50)),
    ("rq1a_cc_m100_minus_m50_average_set_size", "average_set_size",
     ("class_conditional_cp", 100), ("class_conditional_cp", 50)),
]
# Family A secondary: preregistered protocol outcomes reported descriptively;
# their p-values are exploratory (not part of the confirmatory k=2 family).
FAMILY_A_SECONDARY = [
    (f"rq1a_cc_m100_minus_m50_{metric}", metric,
     ("class_conditional_cp", 100), ("class_conditional_cp", 50))
    for metric in ["coverage_minority", "coverage_majority", "threshold_sum",
                   "threshold_gap", "empty_rate", "doubleton_rate"]
]
# Family B confirmatory: the ten preregistered RQ2-B endpoints.
FAMILY_B = [
    (f"rq2b_cc_minus_global_m{m_val}_{metric}", metric,
     ("class_conditional_cp", m_val), ("global_split_cp", m_val))
    for m_val in (50, 100)
    for metric in CONFIRMATORY_METRICS[:5]
]
# Family C (RQ3): preregistered comparison type, descriptive per D10; endpoint
# tags fixed at Stage 11; any p-values are exploratory_not_preregistered.
C_PAIRS = [
    ("xgboost", "logistic_regression", "xgb_minus_lr"),
    ("tabpfn", "logistic_regression", "tabpfn_minus_lr"),
    ("tabpfn", "xgboost", "tabpfn_minus_xgb"),
]
C_METRICS = {"global_split_cp": CONFIRMATORY_METRICS,
             "class_conditional_cp": CONFIRMATORY_METRICS + ["threshold_sum"]}
FAMILY_C = [
    (f"rq3c_{'cc' if cp == 'class_conditional_cp' else 'gl'}_m{m_val}_{pair_name}_{metric}",
     metric, cp, m_val, model_a, model_b, pair_name)
    for cp in CP_METHODS
    for m_val in (50, 100)
    for (model_a, model_b, pair_name) in C_PAIRS
    for metric in C_METRICS[cp]
]


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_text_exclusive(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def write_json_exclusive(path: Path, payload) -> None:
    write_text_exclusive(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def d01_seed(endpoint_tag: str) -> int:
    """D01-derived RNG seed, project-level convention identical to Stage 10:
    input string 'v1.1|EIGHT_DATASET|0|d08_bootstrap_<endpoint_tag>';
    seed = uint32 of the first 32 bits (first 8 hex chars) of that SHA-256."""
    s = f"v1.1|EIGHT_DATASET|0|d08_bootstrap_{endpoint_tag}"
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest()[:8], 16)


def purpose_seed(purpose: str) -> int:
    """Same D01 convention with a non-bootstrap project-level purpose string."""
    s = f"v1.1|EIGHT_DATASET|0|{purpose}"
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest()[:8], 16)


def d08_effect_ci(d: np.ndarray, endpoint_tag: str, n_boot: int = N_BOOT):
    """Preregistered D08: effect = median(d_j); two-sided 95% CI = percentile
    bootstrap (B=20000) of the 8 whole-dataset effects, D01-derived RNG seed."""
    rng = np.random.default_rng(d01_seed(endpoint_tag))
    idx = rng.choice(d.shape[0], size=(n_boot, d.shape[0]), replace=True)
    boots = np.median(d[idx], axis=1)
    low, high = np.percentile(boots, [2.5, 97.5])
    return float(np.median(d)), float(low), float(high)


def across_seed_bootstrap_ci(values: np.ndarray, purpose_tag: str, n_boot: int = N_BOOT):
    """EXPLORATORY descriptive supplement (not preregistered): percentile
    bootstrap CI of the across-seed mean. Distinct from Wilson (within-cell
    binomial) and D08 (across-dataset) intervals."""
    rng = np.random.default_rng(purpose_seed(purpose_tag))
    idx = rng.choice(values.shape[0], size=(n_boot, values.shape[0]), replace=True)
    boots = values[idx].mean(axis=1)
    low, high = np.percentile(boots, [2.5, 97.5])
    return float(low), float(high)


def paired_dataset_effects(piv: pd.DataFrame, metric: str, a_cp: str, a_m: int,
                           b_cp: str, b_m: int):
    """Preregistered contrast is within dataset/seed/pipeline. Per dataset the
    complete paired observations are the 3 pipelines x 10 seeds contrasts;
    d_j = mean over all complete paired contrasts (per-pipeline seed-means are
    recorded for transparency). Complete grid — nothing dropped."""
    dvals, dsets, per_pipeline_all = [], [], []
    for d in DATASETS:
        per_pipeline = {}
        diffs_all = []
        for mdl in MODELS:
            diffs = [float(piv.loc[(d, s, mdl, a_cp, a_m), metric]
                           - piv.loc[(d, s, mdl, b_cp, b_m), metric]) for s in SEEDS]
            per_pipeline[mdl] = float(np.mean(diffs))
            diffs_all.extend(diffs)
        dvals.append(float(np.mean(diffs_all)))
        dsets.append(d)
        per_pipeline_all.append(per_pipeline)
    return np.array(dvals), dsets, dict(zip(dsets, per_pipeline_all))


def pipeline_pair_dataset_effects(piv: pd.DataFrame, metric: str, cp: str, m_val: int,
                                  model_a: str, model_b: str):
    """Comparison C: per dataset, d_j = mean over the 10 paired seeds of
    (model_a - model_b) at fixed cp method and m. Same calibration subset per
    dataset x seed x m across models (subset_hash identity verified in Stage 10)."""
    dvals, dsets = [], []
    for d in DATASETS:
        diffs = [float(piv.loc[(d, s, model_a, cp, m_val), metric]
                       - piv.loc[(d, s, model_b, cp, m_val), metric]) for s in SEEDS]
        dvals.append(float(np.mean(diffs)))
        dsets.append(d)
    return np.array(dvals), dsets


def wilcoxon_exact(d: np.ndarray):
    """Exact two-sided Wilcoxon signed-rank on the dataset effects; zeros are
    discarded (counted). Returns (p, n_zero_discarded). p is NaN when no
    nonzero effect remains."""
    d = np.asarray(d, dtype=float)
    n_zero = int((d == 0).sum())
    nz = d[d != 0]
    if nz.size < 1:
        return float("nan"), n_zero
    try:
        _stat, p = stats.wilcoxon(nz, zero_method="wilcox",
                                  alternative="two-sided", method="exact")
    except ValueError:
        return float("nan"), n_zero
    return float(p), n_zero


def holm_adjust(pvals: list[float]) -> list[float]:
    """Holm step-down adjustment within a family; monotone and clipped to 1."""
    m = len(pvals)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [float("nan")] * m
    running = 0.0
    for rank, oi in enumerate(order):
        if np.isnan(pvals[oi]):
            adj[oi] = float("nan")
            continue
        running = max(running, (m - rank) * pvals[oi])
        adj[oi] = min(1.0, running)
    return adj


def seed_summary(values: np.ndarray) -> dict:
    """Preregistered D09 across-seed descriptive statistics (n, mean, median,
    SD with ddof=1, IQR)."""
    v = np.asarray(values, dtype=float)
    q25, q75 = np.percentile(v, [25, 75])
    return {
        "n": int(v.size),
        "mean": float(np.mean(v)),
        "median": float(np.median(v)),
        "sd_ddof1": float(np.std(v, ddof=1)) if v.size >= 2 else float("nan"),
        "iqr": float(q75 - q25),
        "q25": float(q25),
        "q75": float(q75),
        "min": float(np.min(v)),
        "max": float(np.max(v)),
    }


class Events:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.handle = open(path, "a", encoding="utf-8", newline="\n")

    def log(self, event: str, **kw) -> None:
        rec = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "stage": "Stage 11", "event": event}
        rec.update(kw)
        self.handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.handle.flush()


# ------------------------------------------------------------------- gates ----

def verify_manifest_self_closure(manifest_bytes: bytes) -> dict:
    """The Stage 10 producer hashed the manifest before appending its own
    self-entry, then rewrote it (src/run_stage10_freeze.py). The live file must
    therefore equal [recorded v1 bytes] + [self-entry]. This proves the live
    manifest is exactly the recorded v1 plus its self-entry — byte-exact."""
    txt = manifest_bytes.decode("utf-8")
    key = '"results/results_manifest.json"'
    idx = txt.find(key)
    if idx < 0:
        return {"closure_ok": False, "reason": "self-entry missing"}
    brace = txt.find("{", idx)
    prev_close = txt.rfind("}", 0, brace)
    comma = txt.find(",", prev_close)
    bidx = txt.find('"bytes"', idx)
    end_brace = txt.find("}", bidx)
    v1 = (txt[:comma] + txt[end_brace + 1:]).encode("utf-8")
    import re
    m = re.search(r'"sha256": "([0-9a-f]{64})",\r?\n      "bytes": (\d+)',
                  txt[idx:idx + 300])
    if m is None:
        return {"closure_ok": False, "reason": "self-entry unparsable"}
    rec_sha, rec_bytes = m.group(1), int(m.group(2))
    ok = (sha256_bytes(v1) == rec_sha and len(v1) == rec_bytes)
    return {"closure_ok": bool(ok), "v1_sha256_recorded": rec_sha,
            "v1_bytes_recorded": rec_bytes, "v1_bytes_reconstructed": len(v1),
            "v1_sha256_reconstructed": sha256_bytes(v1),
            "live_sha256": sha256_bytes(manifest_bytes), "live_bytes": len(manifest_bytes),
            "note": "recorded hash = pre-self-entry rewrite state (producer lines "
                    "812-817); CRLF line endings on disk; closure proven byte-exact"}


def run_input_gates(events: Events | None = None) -> tuple[dict, pd.DataFrame]:
    """Verify every frozen input before any analysis. Raises on failure."""
    checks: dict[str, dict] = {}

    def check(name: str, ok: bool, detail) -> None:
        checks[name] = {"status": "PASS" if ok else "FAIL", "detail": detail}
        if events:
            events.log("gate", check=name, status=checks[name]["status"])
        if not ok:
            raise RuntimeError(f"GATE_FAIL:{name}:{detail}")

    # 1. frozen formal-run manifest
    fm = (ROOT / "configs/formal_run_manifest_v1.1.yaml").read_bytes()
    check("frozen_formal_run_manifest_sha256", sha256_bytes(fm) == FROZEN_MANIFEST_SHA256,
          FROZEN_MANIFEST_SHA256)

    # 2. results manifest: byte-exact self closure + live hash
    man_path = RESULTS_DIR / "results_manifest.json"
    man_bytes = man_path.read_bytes()
    closure = verify_manifest_self_closure(man_bytes)
    check("results_manifest_self_closure", closure["closure_ok"]
          and closure["v1_sha256_recorded"] == RESULTS_MANIFEST_V1_SHA256, closure)

    # 3. every manifest-recorded output (except the manifest itself). The Stage 10
    #    events.jsonl is append-only: its recorded hash is an exact PREFIX of the
    #    live file (terminal 'stage10_complete' event was appended after the
    #    manifest was hashed) — verified as a prefix property, not exact equality.
    man = json.loads(man_bytes.decode("utf-8"))
    mismatch, prefix_ok = [], []
    for rel, meta in man["outputs"].items():
        if rel == "results/results_manifest.json":
            continue
        p = ROOT / rel
        if not p.exists():
            mismatch.append(rel)
            continue
        live = p.read_bytes()
        if rel.endswith("events.jsonl") and p.stat().st_size >= meta["bytes"] \
                and sha256_bytes(live[: meta["bytes"]]) == meta["sha256"]:
            prefix_ok.append(rel)
            continue
        if sha256_bytes(live) != meta["sha256"] or p.stat().st_size != meta["bytes"]:
            mismatch.append(rel)
    check("frozen_output_hashes_match_manifest", not mismatch,
          {"mismatches": mismatch,
           "append_only_prefix_verified": prefix_ok})

    # 4. merged results == Stage 09 source == handoff constant
    res_long = ROOT / RESULTS_LONG_REL
    rl_sha = sha256_file(res_long)
    check("results_long_sha256_frozen", rl_sha == man["outputs"][RESULTS_LONG_REL]["sha256"]
          == man["source_run"]["results_long_parquet_sha256"], rl_sha)

    # 5. Stage 10 QC evidence hash (bootstrap reference values)
    ev_sha = sha256_file(ROOT / STAGE10_EVIDENCE_REL)
    check("stage10_evidence_sha256",
          ev_sha == man["outputs"][STAGE10_EVIDENCE_REL]["sha256"], ev_sha)

    # 6. structural grid checks (read-only)
    df = pd.read_parquet(res_long)
    expected_keys = {(d, s, mdl, cp, m) for d in DATASETS for s in SEEDS
                     for mdl in MODELS for cp in CP_METHODS for m in M_VALUES}
    got_keys = set(map(tuple, df[KEY_COLS].itertuples(index=False, name=None)))
    check("grid_exact_1920_cells",
          len(df) == 1920 and got_keys == expected_keys and len(got_keys) == 1920,
          {"rows": len(df), "unique_keys": len(got_keys)})
    check("status_pass_all_protocol_v1_1",
          bool((df["status"] == "PASS").all() and (df["protocol_version"] == "v1.1").all()
               and (df["alpha"] == 0.1).all() and (df["m_majority"] == 200).all()), {})
    cc_metrics = ["q_minority", "q_majority", "rank_minority", "rank_majority",
                  "threshold_gap", "threshold_sum"]
    gl_metrics = ["q_global", "rank_global"]
    cc = df[df["cp_method"] == "class_conditional_cp"]
    gl = df[df["cp_method"] == "global_split_cp"]
    check("structural_null_pattern",
          bool(cc[cc_metrics].notna().all().all() and cc[gl_metrics].isna().all().all()
               and gl[gl_metrics].notna().all().all() and gl[cc_metrics].isna().all().all()),
          {})

    piv = df.sort_values(KEY_COLS).set_index(KEY_COLS)

    # 7. exact reproduction of the Stage 10 D08 endpoint values from the frozen
    #    data (proves this analysis reads the same frozen numbers)
    ev10 = json.loads((ROOT / STAGE10_EVIDENCE_REL).read_text(encoding="utf-8"))
    ref = {e["endpoint_tag"]: e for e in ev10["d08_confirmatory_endpoints_report_only"]}
    repro, diffs = [], []
    for tag, metric, a_sel, b_sel in FAMILY_A + FAMILY_B:
        d, dsets, per_pipeline = paired_dataset_effects(piv, metric, a_sel[0], a_sel[1],
                                                        b_sel[0], b_sel[1])
        eff, lo, hi = d08_effect_ci(d, tag)
        r = ref[tag]
        same = (eff == r["effect_median_dj"] and lo == r["ci95_low"]
                and hi == r["ci95_high"]
                and int((d > 0).sum()) == r["direction_count"]["positive"]
                and int((d < 0).sum()) == r["direction_count"]["negative"]
                and all(abs(d[i] - r["d_by_dataset"][ds]) == 0.0
                        for i, ds in enumerate(dsets)))
        repro.append({"endpoint_tag": tag, "exact_match": bool(same)})
        if not same:
            diffs.append(tag)
    check("stage10_d08_endpoints_reproduced_exactly",
          not diffs, {"n_endpoints": len(repro), "mismatches": diffs})

    if events:
        events.log("gates_complete", n_checks=len(checks))
    return {"checks": checks, "manifest_closure": closure, "results_long_sha256": rl_sha}, df


# ---------------------------------------------------------------- analyses ----

def endpoint_record(tag, metric, comparison_label, family, d, dsets, per_pipeline,
                    preregistered, inference_status) -> dict:
    eff, lo, hi = d08_effect_ci(d, tag)
    p, n_zero = wilcoxon_exact(d)
    return {
        "endpoint_tag": tag, "metric": metric, "comparison": comparison_label,
        "family": family, "preregistered_endpoint": preregistered,
        "inference_status": inference_status,
        "d_by_dataset": {ds: float(v) for ds, v in zip(dsets, d)},
        "per_pipeline_seed_mean_d": per_pipeline,
        "effect_median_dj": eff, "ci95_low": lo, "ci95_high": hi,
        "ci_width": hi - lo,
        "direction_count": {"positive": int((d > 0).sum()),
                            "negative": int((d < 0).sum()),
                            "zero": int((d == 0).sum())},
        "wilcoxon_p_exact_two_sided": p,
        "wilcoxon_n_zero_discarded": n_zero,
        "bootstrap": {"replicates": N_BOOT, "rng": "numpy default_rng",
                      "seed_convention": "D01-derived (identical to Stage 10 convention)",
                      "seed_input_string": f"v1.1|EIGHT_DATASET|0|d08_bootstrap_{tag}",
                      "seed_uint32": d01_seed(tag)},
    }


def run_comparisons(piv: pd.DataFrame) -> dict:
    out: dict[str, list] = {"A": [], "A_secondary": [], "B": [], "C": []}
    for tag, metric, a_sel, b_sel in FAMILY_A:
        d, dsets, pp = paired_dataset_effects(piv, metric, a_sel[0], a_sel[1], b_sel[0], b_sel[1])
        out["A"].append(endpoint_record(
            tag, metric, "RQ1-A: Class-Conditional CP m=100 minus m=50 (primary)",
            "A_confirmatory", d, dsets, pp, True, "confirmatory_preregistered"))
    for tag, metric, a_sel, b_sel in FAMILY_A_SECONDARY:
        d, dsets, pp = paired_dataset_effects(piv, metric, a_sel[0], a_sel[1], b_sel[0], b_sel[1])
        out["A_secondary"].append(endpoint_record(
            tag, metric, "RQ1-A: Class-Conditional CP m=100 minus m=50 (secondary outcome)",
            "A_secondary_descriptive", d, dsets, pp, False,
            "exploratory_not_preregistered"))
    for tag, metric, a_sel, b_sel in FAMILY_B:
        d, dsets, pp = paired_dataset_effects(piv, metric, a_sel[0], a_sel[1], b_sel[0], b_sel[1])
        out["B"].append(endpoint_record(
            tag, metric, f"RQ2-B: Class-Conditional minus Global at m={a_sel[1]}",
            "B_confirmatory", d, dsets, pp, True, "confirmatory_preregistered"))
    for tag, metric, cp, m_val, model_a, model_b, pair_name in FAMILY_C:
        d, dsets = pipeline_pair_dataset_effects(piv, metric, cp, m_val, model_a, model_b)
        out["C"].append(endpoint_record(
            tag, metric,
            f"RQ3-C: {model_a} minus {model_b} at {cp}, m={m_val}",
            "C_exploratory", d, dsets, None, False, "exploratory_not_preregistered"))

    # Holm within each family
    for key in ("A", "A_secondary", "B", "C"):
        fam = out[key]
        pv = [e["wilcoxon_p_exact_two_sided"] for e in fam]
        adj = holm_adjust(pv)
        for e, a in zip(fam, adj):
            e["holm_p_within_family"] = a
            e["holm_family_size"] = len(fam)
    return out


def run_descriptives(df: pd.DataFrame) -> tuple[list, list]:
    """Across-seed descriptive summaries per dataset x model x cp x m cell
    (preregistered D09 statistics) plus the exploratory across-seed bootstrap CI,
    and the preregistered across-seed threshold-variability outputs."""
    rows, tvar = [], []
    for d in DATASETS:
        for mdl in MODELS:
            for cp in CP_METHODS:
                for m in M_VALUES:
                    sub = df[(df["dataset_id"] == d) & (df["model"] == mdl)
                             & (df["cp_method"] == cp) & (df["m_minority"] == m)]
                    if len(sub) != len(SEEDS):
                        raise RuntimeError(f"incomplete cell {d}/{mdl}/{cp}/{m}")
                    metrics = DESCRIPTIVE_METRICS + (
                        CC_ONLY_DESCRIPTIVE if cp == "class_conditional_cp"
                        else GLOBAL_ONLY_DESCRIPTIVE)
                    for metric in metrics:
                        vals = sub.sort_values("seed")[metric].to_numpy(dtype=float)
                        s = seed_summary(vals)
                        lo, hi = across_seed_bootstrap_ci(
                            vals, f"across_seed_bootstrap_{d}|{mdl}|{cp}|m{m}|{metric}")
                        rows.append({
                            "dataset_id": d, "model": mdl, "cp_method": cp,
                            "m_minority": m, "metric": metric, **s,
                            "across_seed_boot_ci_low": lo,
                            "across_seed_boot_ci_high": hi,
                            "uncertainty_type": "across_seed_bootstrap_exploratory",
                        })
                    if cp == "class_conditional_cp":
                        for metric in ["q_minority", "q_majority", "threshold_sum"]:
                            vals = sub.sort_values("seed")[metric].to_numpy(dtype=float)
                            s = seed_summary(vals)
                            tvar.append({
                                "dataset_id": d, "model": mdl, "cp_method": cp,
                                "m_minority": m, "threshold_metric": metric,
                                "across_seed_sd": s["sd_ddof1"],
                                "across_seed_iqr": s["iqr"],
                                "across_seed_mean": s["mean"],
                            })
                    else:
                        vals = sub.sort_values("seed")["q_global"].to_numpy(dtype=float)
                        s = seed_summary(vals)
                        tvar.append({
                            "dataset_id": d, "model": mdl, "cp_method": cp,
                            "m_minority": m, "threshold_metric": "q_global",
                            "across_seed_sd": s["sd_ddof1"],
                            "across_seed_iqr": s["iqr"],
                            "across_seed_mean": s["mean"],
                        })
    return rows, tvar


# ------------------------------------------------------------------ output ----

def endpoint_frame(recs: list[dict]) -> pd.DataFrame:
    rows = []
    for e in recs:
        rows.append({
            "endpoint_tag": e["endpoint_tag"], "comparison": e["comparison"],
            "family": e["family"], "preregistered_endpoint": e["preregistered_endpoint"],
            "inference_status": e["inference_status"], "metric": e["metric"],
            "effect_median_dj": e["effect_median_dj"],
            "ci95_low": e["ci95_low"], "ci95_high": e["ci95_high"],
            "ci_width": e["ci_width"],
            "direction_positive": e["direction_count"]["positive"],
            "direction_negative": e["direction_count"]["negative"],
            "direction_zero": e["direction_count"]["zero"],
            "wilcoxon_p_exact_two_sided": e["wilcoxon_p_exact_two_sided"],
            "wilcoxon_n_zero_discarded": e["wilcoxon_n_zero_discarded"],
            "holm_p_within_family": e["holm_p_within_family"],
            "holm_family_size": e["holm_family_size"],
            "bootstrap_seed_uint32": e["bootstrap"]["seed_uint32"],
        })
    return pd.DataFrame(rows)


def effects_long_frame(recs: list[dict], aggregation: str, has_pipeline: bool) -> pd.DataFrame:
    rows = []
    for e in recs:
        for ds, v in e["d_by_dataset"].items():
            if has_pipeline and e["per_pipeline_seed_mean_d"]:
                for mdl, dv in e["per_pipeline_seed_mean_d"][ds].items():
                    rows.append({"endpoint_tag": e["endpoint_tag"], "family": e["family"],
                                 "dataset_id": ds, "model": mdl, "aggregation": aggregation,
                                 "d": dv})
                rows.append({"endpoint_tag": e["endpoint_tag"], "family": e["family"],
                             "dataset_id": ds, "model": "(pooled_3_pipelines)",
                             "aggregation": aggregation, "d": v})
            else:
                rows.append({"endpoint_tag": e["endpoint_tag"], "family": e["family"],
                             "dataset_id": ds, "model": "(pair_mean_over_10_seeds)",
                             "aggregation": aggregation, "d": v})
    return pd.DataFrame(rows)


def fmt_p(p) -> str:
    return "NA" if p is None or (isinstance(p, float) and np.isnan(p)) else f"{p:.4f}"


def render_report(run_id: str, gates: dict, comp: dict, desc_rows: list,
                  tvar_rows: list, df: pd.DataFrame) -> str:
    def ep_table(recs) -> str:
        lines = ["| endpoint | metric | median paired diff (d_j median) | 95% D08 CI | "
                 "direction (+/-) | Wilcoxon p | Holm p (family) | status |",
                 "|---|---|---|---|---|---|---|---|"]
        for e in recs:
            dc = e["direction_count"]
            lines.append(
                f"| `{e['endpoint_tag']}` | {e['metric']} | "
                f"{e['effect_median_dj']:+.6f} | [{e['ci95_low']:+.6f}, {e['ci95_high']:+.6f}] | "
                f"{dc['positive']}+/−{dc['negative']} | {fmt_p(e['wilcoxon_p_exact_two_sided'])} | "
                f"{fmt_p(e['holm_p_within_family'])} (k={e['holm_family_size']}) | "
                f"{e['inference_status']} |")
        return "\n".join(lines)

    a_all = comp["A"] + comp["A_secondary"]
    b_all = comp["B"]

    # condensed comparison-C digest: efficiency + minority coverage per pair/cp/m
    c_lines = ["| pair | cp | m | metric | median paired diff | 95% D08 CI | direction (+/−) |",
               "|---|---|---|---|---|---|---|"]
    key_metrics = ["singleton_rate", "average_set_size", "coverage_minority",
                   "coverage_disparity"]
    for e in comp["C"]:
        if e["metric"] in key_metrics:
            dc = e["direction_count"]
            c_lines.append(
                f"| {e['comparison'].split(' at ')[0].replace('RQ3-C: ', '')} | "
                f"{'cc' if 'class_conditional' in e['comparison'] else 'global'} | "
                f"{e['comparison'].rsplit('m=', 1)[1]} | {e['metric']} | "
                f"{e['effect_median_dj']:+.4f} | [{e['ci95_low']:+.4f}, {e['ci95_high']:+.4f}] | "
                f"{dc['positive']}+/−{dc['negative']} |")

    # descriptive digest: median across datasets of per-dataset seed-means
    desc_df = pd.DataFrame(desc_rows)
    digest_lines = ["| model | cp | m | metric | median over datasets of seed-mean "
                    "(across-seed SD median) |", "|---|---|---|---|---|"]
    for mdl in MODELS:
        for cp in CP_METHODS:
            for m in (50, 100):
                for metric in ["coverage_minority", "coverage_majority",
                               "singleton_rate", "average_set_size"]:
                    sel = desc_df[(desc_df["model"] == mdl) & (desc_df["cp_method"] == cp)
                                  & (desc_df["m_minority"] == m)
                                  & (desc_df["metric"] == metric)]
                    med = sel["mean"].median()
                    sd_med = sel["sd_ddof1"].median()
                    digest_lines.append(
                        f"| {mdl} | {cp} | {m} | {metric} | {med:+.4f} ({sd_med:.4f}) |")

    tvar_df = pd.DataFrame(tvar_rows)
    tvar_lines = ["| dataset | model | m | q_minority across-seed SD | q_minority IQR |",
                  "|---|---|---|---|---|"]
    for d in DATASETS:
        for mdl in MODELS:
            for m in (50, 100):
                sel = tvar_df[(tvar_df["dataset_id"] == d) & (tvar_df["model"] == mdl)
                              & (tvar_df["cp_method"] == "class_conditional_cp")
                              & (tvar_df["m_minority"] == m)
                              & (tvar_df["threshold_metric"] == "q_minority")]
                if len(sel):
                    r = sel.iloc[0]
                    tvar_lines.append(f"| {d} | {mdl} | {m} | "
                                      f"{r['across_seed_sd']:.6f} | {r['across_seed_iqr']:.6f} |")

    n_conf = sum(1 for e in comp["A"] + comp["B"] if e["ci95_low"] > 0 or e["ci95_high"] < 0)

    return f"""# Stage 11 — Preregistered Statistical Analysis Report (Frozen Results)

**Status:** `ANALYSIS_COMPLETE` | **Run:** `{run_id}` | **Protocol:** v1.1 | **Created:** {datetime.now(timezone.utc).isoformat(timespec='seconds')}
**Data:** `results/results_long.parquet` (SHA-256 `{gates['results_long_sha256']}`; 1,920 CP cells; frozen, unmodified)

## 1. Scope, integrity, and preregistration

- This stage analyzed the frozen Stage 09 formal-run results **read-only**. Every frozen artifact was re-verified before analysis ({len(gates['checks'])} hash/structure gates, all PASS), including byte-exact reproduction of all 12 Stage 10 D08 endpoint values from the frozen data.
- `results/results_manifest.json` self-hash note: the producer hashed the manifest before appending its own self-entry and rewrote it (`src/run_stage10_freeze.py` lines 812–817). The recorded hash `c116e1f1…` was proven byte-exact to be the pre-rewrite state (v1 = live file minus self-entry, 5,114 bytes); the live file is v1 + self-entry. Benign bookkeeping pattern, zero scientific-data impact (documented as ST11-BOOKKEEP-01 in the Stage 11 evidence).
- **Unit of inference: the dataset (n=8).** Within each dataset the 10 seed-level paired contrasts are aggregated first (3 pipelines × 10 seeds = 30 contrasts for A/B; 10 seeds for C pairs). The 8×10 seed cells are never treated as 80 independent research units.
- **Preregistered methods (protocol v1.1 §8 / D08/D09/D10):** effect = median of the 8 dataset-level effects `d_j`; 95% CI = 20,000-replicate percentile bootstrap of the 8 whole-dataset effects (D01-derived seed, Stage 10 convention); exact two-sided Wilcoxon signed-rank on the 8 effects (zeros discarded, counted) with Holm correction **within** each predeclared family; effect size and direction consistency take precedence over p-values, which are auxiliary.
- **Confirmatory families:** A (k=2 RQ1-A primaries) and B (k=10 RQ2-B endpoints). **Exploratory (flagged, excluded from confirmatory claims):** the six A-secondary p-values and all comparison-C p-values — computed only because the Stage 11 instruction requests Wilcoxon/Holm reporting for A/B/C; D10 restricts formal testing to A and B, and RQ3 is otherwise effect/CI/direction descriptive.
- **Uncertainty taxonomy (kept strictly separate):** (1) *test-set Wilson 95% intervals* — within-cell binomial intervals stored per CP cell, never pooled across seeds; (2) *across-seed variability* — descriptive n/mean/median/SD (ddof=1)/IQR per dataset×model×cp×m, plus an explicitly **exploratory** across-seed bootstrap CI of the seed-mean; (3) *across-dataset (D08) CI* — the confirmatory interval over the 8 dataset effects.

## 2. Comparison A — RQ1: Class-Conditional CP, m=100 − m=50 (same pipeline)

{ep_table(a_all)}

Reading (effect-first): the two preregistered primary endpoints (singleton rate, average set size) have near-zero medians with CIs containing zero and only 3/8 datasets positive — no direction-consistent m=100 vs m=50 efficiency change at this seed count. Coverage-minority and threshold-geometry secondary endpoints are reported descriptively above; their p-values are exploratory.

## 3. Comparison B — RQ2: Class-Conditional vs Global CP at m ∈ {{50, 100}}

{ep_table(b_all)}

Reading (effect-first): at both m, Class-Conditional CP raises minority coverage (m=100: median +0.0411, CI [+0.0024, +0.0909], 8/8 datasets positive) and lowers majority coverage (m=100: −0.0169, CI [−0.0432, −0.0032], 0/8 positive), reducing coverage disparity (m=100: −0.0285, CI [−0.1001, −0.0017], 7/8 negative), at the cost of larger average set sizes (m=100: +0.0125, CI [+0.0004, +0.0223], 7/8 positive). {n_conf} of the 12 confirmatory A+B endpoints have direction-judgeable CIs (excludes zero).

## 4. Comparison C — RQ3: predictive-pipeline contrasts at fixed m and CP (descriptive)

No confirmatory p-values are claimed for RQ3 (D10). Key endpoints (full set in `results/stats/stage11_c_endpoints.csv`; all Wilcoxon/Holm values there are flagged exploratory):

{chr(10).join(c_lines)}

## 5. Descriptive summaries (across seeds; D09)

Per dataset×model×cp×m cells — full table: `results/stats/stage11_descriptive_cells.csv`. Digest (median over the 8 datasets of per-dataset seed-means; across-seed SD in parentheses):

{chr(10).join(digest_lines)}

Threshold variability (preregistered across-seed SD/IQR; full table: `results/stats/stage11_threshold_variability.csv`). Class-Conditional `q_minority` across-seed SD at m=50/100:

{chr(10).join(tvar_lines)}

## 6. Machine-readable outputs and reproducibility

- `results/stats/stage11_ab_endpoints.csv|json` — comparisons A (primaries + secondary) and B.
- `results/stats/stage11_c_endpoints.csv|json` — comparison C (90 endpoints, all flagged).
- `results/stats/stage11_dataset_effects_long.csv` — every dataset-level effect (pooled and per-pipeline).
- `results/stats/stage11_descriptive_cells.csv`, `stage11_threshold_variability.csv`.
- `results/stats/stage11_manifest.json` — hashes, gates, seed conventions, preregistration flags.
- Reproduce: `E:\\anaconda3\\python.exe src/run_stage11_analysis.py` (deterministic; exclusive creation; resume-safe).

## 7. Boundary

No result data was modified (all frozen hashes re-verified post-analysis by the run manifest). The Global-coverage Wilson flag remains diagnostic-only (S09-DIAG). No publication figures, manuscript text, or seed expansion; D10 remains `NOT_TRIGGERED` (practical-precision threshold never approved). Interpretation limits: n=8 datasets, 10 seeds; CIs are wide; direction consistency and effect sizes lead, p-values are auxiliary and Holm-scaled within the predeclared families only.
"""


# --------------------------------------------------------------------- main ----

OUTPUT_FILES = [
    "stage11_ab_endpoints.csv", "stage11_ab_endpoints.json",
    "stage11_c_endpoints.csv", "stage11_c_endpoints.json",
    "stage11_dataset_effects_long.csv", "stage11_descriptive_cells.csv",
    "stage11_threshold_variability.csv",
]


def main() -> int:
    t0 = time.time()
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    out_manifest = STATS_DIR / "stage11_manifest.json"

    if out_manifest.exists():
        prev = json.loads(out_manifest.read_text(encoding="utf-8"))
        bad = [rel for rel, meta in prev["outputs"].items()
               if not (ROOT / rel).exists()
               or sha256_file(ROOT / rel) != meta["sha256"]]
        if bad:
            print("STAGE11_STATUS=FAIL")
            print("RESUME_MISMATCH:", bad)
            return 1
        print("STAGE11_STATUS=ALREADY_COMPLETE")
        print("RUN_ID=" + str(prev.get("run_id", prev.get("artifact_id", ""))))
        return 0

    # run id from the analysis-plan hash (deterministic given the preregistered plan)
    analysis_plan = {
        "stage": "Stage 11 preregistered statistical analysis",
        "protocol": "protocol_v1.1 §8; STAGE_01_DECISION_TABLE D08/D09/D10",
        "unit_of_inference": "dataset (8); seeds aggregated within dataset before any "
                             "across-dataset inference; 8x10 seed cells are not 80 units",
        "d08_estimator": "d_j = mean over complete paired contrasts within dataset "
                         "(A/B: 3 pipelines x 10 seeds; C: 10 seeds); effect = median(d_j); "
                         "95% CI = 20000-replicate percentile bootstrap of the 8 d_j "
                         "(D01-derived seed, Stage 10 convention)",
        "confirmatory_families": {
            "A": [t for t, _, _, _ in FAMILY_A],
            "B": [t for t, _, _, _ in FAMILY_B],
        },
        "exploratory_flagged": {
            "A_secondary": [t for t, _, _, _ in FAMILY_A_SECONDARY],
            "C": [t for t, *_ in FAMILY_C],
            "note": "p-values requested by the Stage 11 instruction for A/B/C; D10 "
                    "restricts formal testing to A and B; exploratory values are "
                    "excluded from confirmatory claims",
        },
        "across_seed_bootstrap": {
            "purpose_prefix": "across_seed_bootstrap_<dataset>|<model>|<cp>|m<m>|<metric>",
            "status": "exploratory_descriptive_not_preregistered",
        },
        "wilcoxon": "exact two-sided on 8 dataset effects; zeros discarded+counted; "
                    "Holm within family",
        "n_boot": N_BOOT,
    }
    plan_sha = sha256_bytes(
        json.dumps(analysis_plan, sort_keys=True, ensure_ascii=False).encode("utf-8"))
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_" \
             f"stage11-analysis_{plan_sha[:8]}"
    run_dir = STAGE11_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    events = Events(run_dir / "events.jsonl")
    events.log("stage11_start", run_id=run_id, analysis_plan_sha256=plan_sha)

    try:
        gates, df = run_input_gates(events)
    except RuntimeError as exc:
        events.log("gate_failure", error=str(exc))
        FAILURES_DIR.mkdir(parents=True, exist_ok=True)
        fail_id = f"STAGE11-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-gate"
        (FAILURES_DIR / f"{fail_id}.md").write_text(
            f"# {fail_id}\n\n- stage: Stage 11\n- scope: input hash/structure gate\n"
            f"- error: {exc}\n- action: no analysis output written; frozen results "
            f"untouched; user review required.\n", encoding="utf-8")
        write_json_exclusive(run_dir / "stage11_status.json",
                             {"status": "FAIL", "error": str(exc), "run_id": run_id})
        print("STAGE11_STATUS=FAIL")
        print("GATE_FAILURE:", exc)
        return 1

    events.log("gates_pass", n_checks=len(gates["checks"]))

    piv = df.sort_values(KEY_COLS).set_index(KEY_COLS)
    comp = run_comparisons(piv)
    events.log("comparisons_complete",
               endpoints={k: len(v) for k, v in comp.items()})
    desc_rows, tvar_rows = run_descriptives(df)
    events.log("descriptives_complete", cells=len(desc_rows), threshold_rows=len(tvar_rows))

    ab_recs = comp["A"] + comp["A_secondary"] + comp["B"]
    ab_frame = endpoint_frame(ab_recs)
    c_frame = endpoint_frame(comp["C"])
    eff_long = pd.concat([
        effects_long_frame(comp["A"] + comp["A_secondary"] + comp["B"],
                           "pooled_30_contrasts", True),
        effects_long_frame(comp["C"], "pair_mean_over_10_seeds", False),
    ], ignore_index=True)
    desc_frame = pd.DataFrame(desc_rows)
    tvar_frame = pd.DataFrame(tvar_rows)

    outputs: dict[str, bytes] = {}
    outputs["results/stats/stage11_ab_endpoints.csv"] = ab_frame.to_csv(
        index=False, lineterminator="\n").encode("utf-8")
    outputs["results/stats/stage11_ab_endpoints.json"] = (
        json.dumps(ab_recs, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    outputs["results/stats/stage11_c_endpoints.csv"] = c_frame.to_csv(
        index=False, lineterminator="\n").encode("utf-8")
    outputs["results/stats/stage11_c_endpoints.json"] = (
        json.dumps(comp["C"], indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    outputs["results/stats/stage11_dataset_effects_long.csv"] = eff_long.to_csv(
        index=False, lineterminator="\n").encode("utf-8")
    outputs["results/stats/stage11_descriptive_cells.csv"] = desc_frame.to_csv(
        index=False, lineterminator="\n").encode("utf-8")
    outputs["results/stats/stage11_threshold_variability.csv"] = tvar_frame.to_csv(
        index=False, lineterminator="\n").encode("utf-8")

    report = render_report(run_id, gates, comp, desc_rows, tvar_rows, df)
    outputs["reports/STAGE11_ANALYSIS_REPORT.md"] = report.encode("utf-8")

    for rel, blob in outputs.items():
        write_text_exclusive(ROOT / rel, blob.decode("utf-8"))
        events.log("output_written", path=rel, sha256=sha256_bytes(blob), bytes=len(blob))

    manifest = {
        "artifact_id": f"{run_id}_manifest",
        "run_id": run_id,
        "stage": "Stage 11 preregistered statistical analysis",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "PASS",
        "protocol_version": "v1.1",
        "producer_script": "src/run_stage11_analysis.py",
        "producer_script_sha256": sha256_file(Path(__file__).resolve()),
        "analysis_plan_sha256": plan_sha,
        "run_dir": str(run_dir),
        "source_results_long_sha256": gates["results_long_sha256"],
        "input_gates": gates["checks"],
        "manifest_closure": gates["manifest_closure"],
        "preregistration": analysis_plan,
        "outputs": {rel: {"sha256": sha256_bytes(blob), "bytes": len(blob)}
                    for rel, blob in outputs.items()},
        "boundary": "Read-only analysis of frozen results; no result data modified; "
                    "no figures/manuscript/seed expansion; exploratory items flagged "
                    "in-table; Global-coverage Wilson flag remains diagnostic-only.",
    }
    write_json_exclusive(out_manifest, manifest)
    write_json_exclusive(run_dir / "analysis_plan.json", analysis_plan)
    write_json_exclusive(run_dir / "stage11_evidence.json", {
        "run_id": run_id,
        "gates": gates["checks"],
        "manifest_closure": gates["manifest_closure"],
        "comparisons": {k: len(v) for k, v in comp.items()},
        "endpoints": {k: v for k, v in comp.items()},
        "descriptive_rows": len(desc_rows),
        "threshold_variability_rows": len(tvar_rows),
    })
    write_json_exclusive(run_dir / "stage11_status.json", {
        "artifact_id": f"{run_id}_status", "stage": "Stage 11",
        "status": "PASS", "gates": len(gates["checks"]), "failures": [],
        "endpoints_total": sum(len(v) for v in comp.values()),
        "updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    events.log("stage11_complete", status="PASS",
               elapsed_s=round(time.time() - t0, 2))
    events.handle.close()

    print("STAGE11_STATUS=PASS")
    print(f"RUN_ID={run_id}")
    print(f"ENDPOINTS={sum(len(v) for v in comp.values())} "
          f"(A={len(comp['A'])}, A_sec={len(comp['A_secondary'])}, "
          f"B={len(comp['B'])}, C={len(comp['C'])})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
