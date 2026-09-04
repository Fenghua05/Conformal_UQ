"""Stage 02 public binary-tabular dataset registry and feasibility screen.

This program is intentionally limited to metadata/data validation.  It does not
fit predictive models, construct conformal scores, or read experiment results.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import sys
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "dataset_screening_v1.0.json"
RAW_DIR = ROOT / "data" / "stage02_raw" / "openml"
OUT_DIR = ROOT / "artifacts" / "stage02"
REPORT_DIR = ROOT / "reports"

# Candidate discovery is independent of any model or CP outcome.  The source IDs
# are OpenML data IDs; each API response records its immutable OpenML file ID.
CANDIDATES: list[dict[str, Any]] = [
    {"id": "openml_3_kr_vs_kp", "openml_id": 3, "name": "kr-vs-kp", "target": "class", "domain": "chess endgame", "source_note": "OpenML versioned public dataset; original UCI Chess (King-Rook vs King-Pawn)."},
    {"id": "openml_24_mushroom", "openml_id": 24, "name": "mushroom", "target": "class", "domain": "mushroom morphology", "source_note": "OpenML versioned public dataset; original UCI Mushroom."},
    {"id": "openml_1111_kddcup09_appetency", "openml_id": 1111, "name": "KDDCup09_appetency", "target": "target", "domain": "marketing response", "source_note": "OpenML versioned KDD Cup 2009 benchmark."},
    {"id": "openml_1461_bank_marketing", "openml_id": 1461, "name": "bank-marketing", "target": "y", "domain": "bank marketing", "source_note": "OpenML versioned public dataset; original UCI Bank Marketing.", "known_hard_risk": "The call-duration field is only known after a marketing call and is an explicit target-leakage field in the source documentation."},
    {"id": "openml_1486_nomao", "openml_id": 1486, "name": "nomao", "target": "class", "domain": "entity matching", "source_note": "OpenML versioned public benchmark."},
    {"id": "openml_1489_phoneme", "openml_id": 1489, "name": "phoneme", "target": "Class", "domain": "speech", "source_note": "OpenML versioned public benchmark."},
    {"id": "openml_1590_adult", "openml_id": 1590, "name": "adult", "target": "class", "domain": "census income", "source_note": "OpenML versioned public dataset; original UCI Adult/Census Income."},
    {"id": "openml_4135_amazon_employee_access", "openml_id": 4135, "name": "Amazon_employee_access", "target": "ACTION", "domain": "workplace access", "source_note": "OpenML versioned public benchmark."},
    {"id": "openml_4534_phishingwebsite", "openml_id": 4534, "name": "PhishingWebsites", "target": "Result", "domain": "web security", "source_note": "OpenML versioned public dataset; original UCI Phishing Websites."},
    {"id": "openml_23512_higgs", "openml_id": 23512, "name": "higgs", "target": "class", "domain": "particle-physics simulation", "source_note": "OpenML versioned public benchmark subset of HIGGS."},
    {"id": "openml_40536_speeddating", "openml_id": 40536, "name": "SpeedDating", "target": "match", "domain": "speed dating", "source_note": "OpenML versioned public benchmark.", "known_hard_risk": "Multiple post-interaction ratings/decisions are obvious outcome-proximal features for match; not admissible without a uniform pre-event feature definition."},
    {"id": "openml_41146_sylvine", "openml_id": 41146, "name": "sylvine", "target": "class", "domain": "automl benchmark", "source_note": "OpenML versioned AutoML benchmark; transformed source provenance is documented but not primary-domain raw data."},
    {"id": "openml_41150_miniboone", "openml_id": 41150, "name": "MiniBooNE", "target": "class", "domain": "particle-physics simulation", "source_note": "OpenML versioned public dataset; original UCI MiniBooNE."}
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def derive_seed(protocol_version: str, dataset_id: str, base_seed: int, purpose: str) -> tuple[str, int]:
    canonical = f"{protocol_version}|{dataset_id}|{base_seed}|{purpose}"
    return canonical, int.from_bytes(hashlib.sha256(canonical.encode("utf-8")).digest()[:4], "big")


def fetch_url(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "conformal-uq-stage02-registry/1.0"})
    with urllib.request.urlopen(request, timeout=180) as response:
        return response.read()


def metadata_for(data_id: int) -> dict[str, Any]:
    payload = fetch_url(f"https://www.openml.org/api/v1/json/data/{data_id}")
    return json.loads(payload.decode("utf-8"))


def scalar_at(mapping: Any, *names: str) -> Any:
    """Find a metadata scalar despite OpenML's namespace-prefixed response keys."""
    if isinstance(mapping, dict):
        for key, value in mapping.items():
            clean = key.split(":")[-1]
            if clean in names:
                return value
            found = scalar_at(value, *names)
            if found is not None:
                return found
    if isinstance(mapping, list):
        for value in mapping:
            found = scalar_at(value, *names)
            if found is not None:
                return found
    return None


def ensure_openml_arff(candidate: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    metadata = metadata_for(candidate["openml_id"])
    file_id = scalar_at(metadata, "file_id")
    if file_id is None:
        raise RuntimeError(f"OpenML metadata has no file_id for {candidate['id']}")
    path = RAW_DIR / f"{candidate['openml_id']}_{file_id}.arff"
    if not path.exists():
        path.write_bytes(fetch_url(f"https://www.openml.org/data/v1/download/{file_id}"))
    return path, metadata


def decode_cell(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def load_arff(path: Path) -> pd.DataFrame:
    records, _ = arff.loadarff(str(path))
    frame = pd.DataFrame(records)
    for column in frame.columns:
        if frame[column].dtype == object:
            frame[column] = frame[column].map(decode_cell)
    return frame.replace("?", pd.NA)


def resolve_target(frame: pd.DataFrame, requested: str) -> str:
    exact = [column for column in frame.columns if column == requested]
    if exact:
        return exact[0]
    matches = [column for column in frame.columns if column.casefold() == requested.casefold()]
    if len(matches) == 1:
        return matches[0]
    raise KeyError(f"Target {requested!r} not found; columns begin {list(frame.columns)[:10]}")


def normalise_binary_target(series: pd.Series) -> tuple[pd.Series, dict[str, int], str, dict[str, int]]:
    labels = series.map(decode_cell).astype(str).str.strip()
    counts = labels.value_counts(dropna=False).to_dict()
    if len(counts) != 2:
        raise ValueError(f"Expected exactly two target labels; observed {counts}")
    ordered = sorted(counts.items(), key=lambda item: (item[1], str(item[0])))
    minority_label, majority_label = ordered[0][0], ordered[1][0]
    mapping = {str(majority_label): 0, str(minority_label): 1}
    return labels.map(mapping).astype("int8"), mapping, str(minority_label), {str(k): int(v) for k, v in counts.items()}


def is_categorical(series: pd.Series) -> bool:
    return not pd.api.types.is_numeric_dtype(series)


def feature_width_after_train_fit(frame: pd.DataFrame, train_indices: np.ndarray) -> int:
    train = frame.iloc[train_indices]
    total = 0
    for column in frame.columns:
        if is_categorical(frame[column]):
            total += int(train[column].astype("string").fillna("__MISSING__").nunique())
        else:
            total += 1
    return total


def class_count(values: pd.Series) -> dict[str, int]:
    outcome = values.value_counts().to_dict()
    return {"majority": int(outcome.get(0, 0)), "minority": int(outcome.get(1, 0))}


def split_records(dataset_id: str, features: pd.DataFrame, y: pd.Series, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    row_indices = np.arange(len(y))
    for base_seed in cfg["seeds"]:
        test_canonical, test_seed = derive_seed(cfg["protocol_version"], dataset_id, base_seed, "stratified_test_split")
        train_cal, test = train_test_split(row_indices, test_size=cfg["split"]["test"], stratify=y, random_state=test_seed)
        cal_canonical, cal_seed = derive_seed(cfg["protocol_version"], dataset_id, base_seed, "stratified_calibration_split")
        train, calibration = train_test_split(train_cal, test_size=0.25, stratify=y.iloc[train_cal], random_state=cal_seed)
        train_counts, cal_counts, test_counts = class_count(y.iloc[train]), class_count(y.iloc[calibration]), class_count(y.iloc[test])
        width = feature_width_after_train_fit(features, train)
        minimums = cfg["minimum_counts"]
        passed = (cal_counts["minority"] >= minimums["calibration_minority"] and cal_counts["majority"] >= minimums["calibration_majority"] and test_counts["minority"] >= minimums["test_minority"] and width <= cfg["feature_cap_after_train_only_transform"])
        records.append({
            "base_seed": base_seed,
            "test_split_seed": test_seed,
            "test_split_canonical_input": test_canonical,
            "calibration_split_seed": cal_seed,
            "calibration_split_canonical_input": cal_canonical,
            "n_train": int(len(train)), "n_calibration_pool": int(len(calibration)), "n_test": int(len(test)),
            "train_class_counts": train_counts, "calibration_pool_class_counts": cal_counts, "test_class_counts": test_counts,
            "post_train_transform_feature_count": width,
            "pass_cal_minority": cal_counts["minority"] >= minimums["calibration_minority"],
            "pass_cal_majority": cal_counts["majority"] >= minimums["calibration_majority"],
            "pass_test_minority": test_counts["minority"] >= minimums["test_minority"],
            "pass_feature_cap": width <= cfg["feature_cap_after_train_only_transform"],
            "pass": passed,
        })
    return records


def risk_audit(candidate: dict[str, Any], features: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
    names = [str(column) for column in features.columns]
    name_flags = [name for name in names if re.search(r"(^|[_ -])(id|identifier|index|unnamed)([_ -]|$)", name, flags=re.I)]
    near_unique = [name for name in names if features[name].nunique(dropna=False) / max(1, len(features)) > 0.995]
    exact_duplicates = int(pd.concat([features.reset_index(drop=True), y.rename("__target__")], axis=1).duplicated().sum())
    feature_duplicates = int(features.duplicated().sum())
    known = candidate.get("known_hard_risk")
    automatic_hard = bool(known)
    return {
        "exact_duplicate_rows_including_target": exact_duplicates,
        "duplicate_feature_rows": feature_duplicates,
        "explicit_id_columns": name_flags,
        "near_unique_possible_id_columns": near_unique,
        "known_time_or_target_leakage": known or "No documented field-level issue identified in this screen; manual source review remains required.",
        "hard_integrity_failure": automatic_hard,
        "complex_domain_preprocessing": "No" if not automatic_hard else "Not assessed after hard leakage flag.",
    }


def tabpfn_status(n_rows: int, maximum_width: int) -> dict[str, Any]:
    # Current official default documentation (retrieved 2026-08-29): TabPFN-3
    # supports 1M x 200, 100k x 2k, or 1k x 20k. We use the largest training
    # split rather than the entire dataset because only train is model context.
    n_train = int(np.floor(n_rows * 0.60))
    within = (n_train <= 1_000_000 and maximum_width <= 200) or (n_train <= 100_000 and maximum_width <= 2_000) or (n_train <= 1_000 and maximum_width <= 20_000)
    return {
        "official_version_checked": "TabPFN-3 current default (official Prior Labs README, accessed 2026-08-29)",
        "screened_train_rows": n_train,
        "maximum_screened_features": maximum_width,
        "within_documented_input_envelope": within,
        "cpu_caveat": "Official current README says CPU use is only moderate up to 5,000 samples by default; hardware/model lock is deferred to authorized Stage 03.",
        "no_truncation_or_subsampling": True,
    }


def screen_candidate(candidate: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    path, metadata = ensure_openml_arff(candidate)
    frame = load_arff(path)
    target = resolve_target(frame, candidate["target"])
    y, mapping, minority_label, raw_counts = normalise_binary_target(frame[target])
    features = frame.drop(columns=[target])
    splits = split_records(candidate["id"], features, y, cfg)
    audit = risk_audit(candidate, features, y)
    max_width = max(record["post_train_transform_feature_count"] for record in splits)
    tabpfn = tabpfn_status(len(frame), max_width)
    all_seed_pass = all(record["pass"] for record in splits)
    feature_types = {"numeric": int(sum(not is_categorical(features[c]) for c in features.columns)), "categorical": int(sum(is_categorical(features[c]) for c in features.columns))}
    licence = scalar_at(metadata, "licence", "license")
    version = scalar_at(metadata, "version")
    description = scalar_at(metadata, "description")
    eligible = all_seed_pass and not audit["hard_integrity_failure"] and tabpfn["within_documented_input_envelope"]
    return {
        "dataset_id": candidate["id"], "display_name": candidate["name"], "status": "eligible_proposed" if eligible else "excluded_or_not_admissible",
        "selection_status": "PROPOSED_PENDING_USER_CONFIRMATION" if eligible else "NOT_ELIGIBLE",
        "source": {"type": "OpenML versioned dataset", "priority": 1, "openml_data_id": candidate["openml_id"], "openml_version": version, "source_url": f"https://www.openml.org/d/{candidate['openml_id']}", "download_file_id": scalar_at(metadata, "file_id"), "raw_local_path": str(path.relative_to(ROOT)), "raw_sha256": sha256_path(path), "accessed_utc": utc_now(), "licence": licence or "Not supplied in retrieved OpenML metadata; public-access status only. User confirmation required before any final lock."},
        "source_note": candidate["source_note"], "metadata_description_excerpt": (description or "")[:500], "domain": candidate["domain"],
        "n_rows": int(len(frame)), "raw_feature_count": int(features.shape[1]), "feature_types": feature_types,
        "missing_values": int(features.isna().sum().sum()), "rows_with_missing_values": int(features.isna().any(axis=1).sum()),
        "target": target, "label_mapping_to_protocol_binary": mapping, "minority_original_label": minority_label, "raw_class_counts": raw_counts,
        "split_checks": splits, "all_seed_feasible": all_seed_pass, "maximum_post_train_transform_feature_count": max_width,
        "integrity_audit": audit, "tabpfn_input_screen": tabpfn,
        "eligibility_reason": "All protocol feasibility checks passed; proposed only, no user lock." if eligible else "Failed one or more frozen feasibility/integrity/TabPFN-input checks; see fields above.",
    }


def rank_and_number(records: list[dict[str, Any]]) -> None:
    eligible = sorted((record for record in records if record["status"] == "eligible_proposed"), key=lambda record: record["source"]["openml_data_id"])
    for position, record in enumerate(eligible, start=1):
        record["frozen_selection_rank"] = position
        if position <= 8:
            record["proposed_role"] = "primary"
        elif position <= 12:
            record["proposed_role"] = "replacement"
        else:
            record["proposed_role"] = "reserve_after_required_replacements"
    for record in records:
        if record["status"] != "eligible_proposed":
            record["frozen_selection_rank"] = None
            record["proposed_role"] = "excluded"


def concise_counts(record: dict[str, Any]) -> str:
    first = record["split_checks"][0]
    return f"train M/m {first['train_class_counts']['majority']}/{first['train_class_counts']['minority']}; cal {first['calibration_pool_class_counts']['majority']}/{first['calibration_pool_class_counts']['minority']}; test {first['test_class_counts']['majority']}/{first['test_class_counts']['minority']}"


def render_markdown(registry: dict[str, Any]) -> str:
    lines = [
        "# Dataset Registry v1.0 — Stage 02",
        "",
        "**Status:** `PROPOSED_PENDING_USER_CONFIRMATION`  ",
        "**Scope:** public, binary, tabular screening only; no model fitting and no CP results were read.  ",
        f"**Generated:** {registry['generated_utc']}",
        "",
        "## Frozen feasibility rule",
        "",
        "For every candidate and every one of the ten protocol seeds, the program executes a two-stage stratified split: 20% test, then 25% of the remaining 80% as calibration pool. It requires calibration minority ≥100, calibration majority ≥200, test minority ≥75, and train-only transformed feature width ≤500. No split proportions, m values, row truncation, or subsampling were changed.",
        "",
        "## Proposed ranking",
        "",
        "| Rank | Role | Dataset | Source / ID | N | Raw features (num/cat) | Minority label / count | First-seed class counts (majority/minority) | Max transformed features | Licence |",
        "|---:|---|---|---|---:|---|---|---|---:|---|",
    ]
    for record in sorted(registry["records"], key=lambda r: (r["frozen_selection_rank"] is None, r["frozen_selection_rank"] or 999999, r["dataset_id"])):
        if record["status"] != "eligible_proposed":
            continue
        ft = record["feature_types"]
        lines.append(f"| {record['frozen_selection_rank']} | {record['proposed_role']} | {record['display_name']} | OpenML {record['source']['openml_data_id']} / v{record['source']['openml_version']} | {record['n_rows']:,} | {record['raw_feature_count']} ({ft['numeric']}/{ft['categorical']}) | {record['minority_original_label']} / {min(record['raw_class_counts'].values()):,} | {concise_counts(record)} | {record['maximum_post_train_transform_feature_count']} | {record['source']['licence']} |")
    lines.extend(["", "## Excluded candidates and audit flags", "", "| Dataset | Source / ID | Reason | Duplicate rows (full/features) | ID / leakage flags |", "|---|---|---|---:|---|"])
    for record in registry["records"]:
        if record["status"] == "eligible_proposed":
            continue
        audit = record["integrity_audit"]
        lines.append(f"| {record['display_name']} | OpenML {record['source']['openml_data_id']} | {record['eligibility_reason']} | {audit['exact_duplicate_rows_including_target']}/{audit['duplicate_feature_rows']} | {audit['known_time_or_target_leakage']} |")
    lines.extend([
        "", "## TabPFN current official constraint check", "",
        "Current official Prior Labs documentation was checked on 2026-08-29. The current default TabPFN-3 documents a trade-off envelope of 1,000,000 rows × 200 features, 100,000 rows × 2,000 features, or 1,000 rows × 20,000 features. Each record stores its train-context size and maximum train-only transformed feature count. The frozen protocol's stricter ≤500 transformed-feature rule remains unchanged. CPU execution is not assumed feasible beyond 5,000 samples; Stage 03 must separately lock package/checkpoint/device, and no data are truncated to work around a limit.",
        "", "## Required user decision", "",
        "Confirm the proposed eight primary datasets and the next four eligible records as ordered replacements. Until that confirmation, every record is only `PROPOSED_PENDING_USER_CONFIRMATION` and `dataset_lock_v1.0.md` is not a lock authorization.",
        "",
    ])
    return "\n".join(lines)


def write_outputs(registry: dict[str, Any]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "dataset_registry_v1.0.json").write_text(json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT_DIR / "feasibility_checks_v1.0.json").write_text(json.dumps({"protocol": registry["protocol"], "records": [{"dataset_id": r["dataset_id"], "split_checks": r["split_checks"], "all_seed_feasible": r["all_seed_feasible"]} for r in registry["records"]]}, indent=2, ensure_ascii=False), encoding="utf-8")
    fields = ["dataset_id", "display_name", "status", "selection_status", "frozen_selection_rank", "proposed_role", "n_rows", "raw_feature_count", "maximum_post_train_transform_feature_count", "all_seed_feasible", "minority_original_label", "raw_class_counts"]
    with (OUT_DIR / "dataset_registry_v1.0.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in registry["records"]:
            writer.writerow({field: json.dumps(record[field], ensure_ascii=False) if isinstance(record[field], (dict, list)) else record[field] for field in fields})
    markdown = render_markdown(registry)
    (REPORT_DIR / "dataset_registry_v1.0.md").write_text(markdown, encoding="utf-8")
    primary = [r for r in registry["records"] if r.get("proposed_role") == "primary"]
    replacements = [r for r in registry["records"] if r.get("proposed_role") == "replacement"]
    lock_lines = ["# Dataset Lock v1.0", "", "**Status:** `PROPOSED_PENDING_USER_CONFIRMATION — NOT LOCKED`", "", "This document is a review proposal generated under frozen protocol `conformal-uq-stage1-v1.0.0`. It is not a user authorization and does not permit Stage 03.", "", "## Proposed primary eight", ""]
    lock_lines += [f"{i}. `{r['dataset_id']}` — {r['display_name']} (OpenML data ID {r['source']['openml_data_id']}, version {r['source']['openml_version']})" for i, r in enumerate(primary, 1)]
    lock_lines += ["", "## Ordered replacements", ""]
    lock_lines += [f"{i}. `{r['dataset_id']}` — {r['display_name']} (OpenML data ID {r['source']['openml_data_id']}, version {r['source']['openml_version']})" for i, r in enumerate(replacements, 1)]
    lock_lines += ["", "## Replacement rule", "", "A replacement is permitted only before outcome analysis for source/licence/label-semantic failure, all-seed infeasibility, transformed-feature-cap failure, TabPFN-input incompatibility, or irreparable data-integrity failure. It must follow the displayed order and may never be driven by outcome direction, effect size, CI width, p-value, or model ranking.", "", "## Confirmation required", "", "User must explicitly confirm this exact primary/replacement list before its status can become `LOCKED`.", ""]
    (ROOT / "protocols" / "dataset_lock_v1.0.md").write_text("\n".join(lock_lines), encoding="utf-8")


def verify(registry: dict[str, Any], cfg: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    if registry["protocol"]["seeds"] != cfg["seeds"]:
        issues.append("Seed list differs from frozen config.")
    for record in registry["records"]:
        if len(record["split_checks"]) != 10:
            issues.append(f"{record['dataset_id']}: missing split checks.")
        for split in record["split_checks"]:
            if split["n_train"] + split["n_calibration_pool"] + split["n_test"] != record["n_rows"]:
                issues.append(f"{record['dataset_id']} seed {split['base_seed']}: split row total mismatch.")
        if record["status"] == "eligible_proposed" and record["selection_status"] != "PROPOSED_PENDING_USER_CONFIRMATION":
            issues.append(f"{record['dataset_id']}: an eligible record was silently locked.")
    return issues


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-and-screen", action="store_true")
    parser.add_argument("--render-artifacts", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    registry_path = OUT_DIR / "dataset_registry_v1.0.json"
    if args.download_and_screen:
        records: list[dict[str, Any]] = []
        for candidate in CANDIDATES:
            print(f"Screening {candidate['id']}", flush=True)
            try:
                records.append(screen_candidate(candidate, cfg))
            except Exception as exc:  # Explicit failed acquisition is an auditable exclusion, not a silent skip.
                records.append({"dataset_id": candidate["id"], "display_name": candidate["name"], "status": "excluded_or_not_admissible", "selection_status": "NOT_ELIGIBLE", "frozen_selection_rank": None, "proposed_role": "excluded", "screen_failure": repr(exc), "source": {"type": "OpenML versioned dataset", "priority": 1, "openml_data_id": candidate["openml_id"], "source_url": f"https://www.openml.org/d/{candidate['openml_id']}"}})
        rank_and_number(records)
        registry = {"artifact_id": "dataset_registry_v1.0", "version": cfg["artifact_version"], "status": "PROPOSED_PENDING_USER_CONFIRMATION", "generated_utc": utc_now(), "protocol": cfg, "records": records}
        write_outputs(registry)
    if args.render_artifacts:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        write_outputs(registry)
    if args.verify:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        issues = verify(registry, cfg)
        print(json.dumps({"status": "PASS" if not issues else "FAIL", "issues": issues}, ensure_ascii=False))
        return 0 if not issues else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
