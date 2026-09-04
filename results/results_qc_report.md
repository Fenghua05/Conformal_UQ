# Stage 10 — Full-Grid Results Freeze QC Report

**Status:** `PASS` | **Created:** 20260831T101122Z | **Stage 10 run:** `20260831T101122Z_stage10-results-freeze_63eb04da`
**Source run:** `20260831T091426Z_stage09-formal_8abaf7bebe64` (single complete 8-dataset formal run; no batch split exists — user-confirmed option A)
**Frozen manifest:** `configs/formal_run_manifest_v1.1.yaml` SHA-256 `0690795b1ea68a148f069a44dccb09a9245d21a097e7174dbd5ec3002f24172d`

## 1. Scope and grid verification

- CP result cells: **1920 / 1920** (`dataset x seed x model x cp_method x m_minority`); unique keys, zero duplicates, zero missing/extra.
- Base prediction units: **240 / 240** (`dataset x seed x model`, one immutable probability cache each). CP cells (8 per unit) are distinguished from base units.
- All cells `protocol_version=v1.1`, `results_schema_version=v1.1.0`, `status=PASS`, `alpha=0.1`, `m_majority=200`.
- Lineage: LR/XGBoost cells bind config `40f29139...` / code `cb25b48d...`; TabPFN cells bind config `cee5c7d7...` / code `8be59da8...`; environment `32bdba72...` constant; frozen-manifest hash constant in all rows.

## 2. Hash-bound input verification (live re-hash)

- `configs/formal_run_manifest_v1.1.yaml`: MATCH
- `protocols/protocol_v1.1.md`: MATCH
- `protocols/dataset_lock_v1.0.md`: MATCH
- `configs/stage04_splits_v1.1.yaml`: MATCH
- `configs/stage05_lr_xgboost_v1.1.yaml`: MATCH
- `configs/stage05b_tabpfn_v1.1.yaml`: MATCH
- `decisions/pilot_decision_stage07_v1.1.json`: MATCH
- `environment/environment_lock_v1.0.json`: MATCH
- `configs/results_long.schema.json`: MATCH
- `artifacts/stage08_v11/20260831T085012Z_pilot_independent_audit/independent_audit.json`: MATCH
- `artifacts/stage02/dataset_registry_v1.0.1.json`: MATCH
- `decisions/D08-004_FORMAL_RUN_GO_RECEIPT.json`: MATCH
- `decisions/D08-003_V11_CACHE_AND_PILOT_BUDGET_RECEIPT.json`: MATCH
- `artifacts/stage08_v11_cloud/cache_intake_20260831T081631Z/intake_audit.json`: MATCH

## 3. Invariant and identity checks

- PASS — frozen_manifest_and_input_hashes_live_match
- PASS — run_status_pass_1920_of_1920
- PASS — run_qc_all_checks_true
- PASS — run_manifest_binds_frozen_manifest
- PASS — cell_file_count_1920
- PASS — cell_json_parse_ok
- PASS — cell_keys_unique_no_duplicates
- PASS — cell_grid_exact_no_missing_no_extra
- PASS — base_prediction_units_240
- PASS — protocol_version_v1_1_all
- PASS — results_schema_version_v1_1_0_all
- PASS — alpha_0_1_all
- PASS — m_majority_200_all
- PASS — status_pass_all
- PASS — run_id_constant
- PASS — frozen_manifest_sha256_constant_all_rows
- PASS — environment_hash_constant
- PASS — config_code_lineage_per_model_family
- PASS — created_utc_within_run_window
- PASS — exact_order_statistic_ranks_all_cells
- PASS — structural_null_pattern_exact
- PASS — no_unexpected_missing_values
- PASS — threshold_geometry_gap_sum_identities_cc
- PASS — thresholds_in_unit_interval
- PASS — subset_hash_identical_across_models_and_cp_methods
- PASS — split_hash_equals_locked_v1_1_split_manifests
- PASS — class_counts_and_feasibility_identities
- PASS — coverage_identities_and_disparity
- PASS — set_decomposition_identities
- PASS — wilson_intervals_reproduced_all_5760_intervals
- PASS — wilson_bounds_contain_estimate
- PASS — metric_ranges_valid
- PASS — auroc_auprc_invariant_across_cp_and_m
- PASS — all_240_caches_rehashed_and_provenance_bound
- PASS — dataset_hash_constant_per_dataset
- PASS — label_mapping_hash_constant_per_dataset
- PASS — model_hash_constant_per_base_unit
- PASS — results_long_rows_1920
- PASS — results_long_parquet_matches_cell_records
- PASS — results_long_csv_parquet_key_sets_equal
- PASS — results_long_csv_parquet_values_agree
- PASS — global_coverage_diagnostic_reproduced
- PASS — results_long_parquet_merged_sha256_identical_to_source
- PASS — completeness_matrix_all_1920_complete

- Exact order statistics reproduced for all cells (CC ranks {10:10, 20:19, 50:46, 100:91} and majority 181; Global ranks {10:190, 20:199, 50:226, 100:271}).
- `subset_hash` identical across all 3 models and both CP methods within every `dataset x seed x m` (canonical membership).
- `split_hash` equals the locked v1.1 split manifest for all 80 `dataset x seed`.
- All **240** cache NPZ files re-hashed; each equals its manifest `cache_sha256` and every descendant cell's `prediction_cache_hash`; cache-manifest provenance (split/model/dataset hash, AUROC/AUPRC) matches all descendant cells.
- AUROC/AUPRC invariant across CP method and m within every `dataset x seed x model`.
- Coverage identities, disparity, set decomposition (`empty+singleton+doubleton=1`, `average_set_size=singleton+2*doubleton`), and threshold geometry identities (`gap=abs(q_minority-q_majority)`, `sum=q_minority+q_majority`) verified on all rows.
- All **5,760** Wilson intervals recomputed with the producing implementation's exact formula (`src/conformal_uq/metrics.py`, plain Wilson, no continuity correction, [0,1] endpoint clipping; max abs err 0.00e+00); bounds contain estimates.
- Structural null pattern exact (Global rows null in `q_minority, q_majority, rank_minority, rank_majority, threshold_gap, threshold_sum`; CC rows null in `q_global, rank_global`); no unexpected missing values.
- `results_long.parquet` and `.csv` key sets and values agree (max numeric abs err 4.44e-16).
- Global-coverage Wilson diagnostic reproduced from aggregates and matches run QC exactly; remains **diagnostic-only** (no calculation discrepancy; not a scientific finding).

## 4. Authoritative merge

- `results/results_long.parquet` written once (exclusive creation); SHA-256 identical to the immutable source `artifacts/runs/20260831T091426Z_stage09-formal_8abaf7bebe64/results_long.parquet`. Originals preserved; nothing overwritten.

## 5. D08 confirmatory endpoints at core m=50/100 — 10-seed CI determinability (report-only)

Preregistered D08 estimator: per-dataset seed-mean paired effect `d_j`; confirmatory effect = median of the 8 `d_j`; two-sided 95% CI = 20,000-replicate percentile bootstrap of the 8 whole-dataset effects with a D01-derived RNG seed (convention documented in the evidence file). 8 of 12 endpoints have a direction-judgeable 95% CI (excludes zero).

- `rq1a_cc_m100_minus_m50_singleton_rate` (RQ1-A: Class-Conditional CP m=100 minus m=50 (primary)): -0.000841 [-0.002072, +0.005734] w=0.007806 dir=3+/5- judgeable=NO
- `rq1a_cc_m100_minus_m50_average_set_size` (RQ1-A: Class-Conditional CP m=100 minus m=50 (primary)): -0.000841 [-0.005734, +0.002771] w=0.008505 dir=3+/5- judgeable=NO
- `rq2b_cc_minus_global_m50_coverage_minority` (RQ2-B: Class-Conditional minus Global at m=50): +0.047165 [+0.005705, +0.126709] w=0.121004 dir=7+/1- judgeable=YES
- `rq2b_cc_minus_global_m50_coverage_majority` (RQ2-B: Class-Conditional minus Global at m=50): -0.011765 [-0.031457, -0.000752] w=0.030705 dir=0+/8- judgeable=YES
- `rq2b_cc_minus_global_m50_coverage_disparity` (RQ2-B: Class-Conditional minus Global at m=50): -0.022957 [-0.118824, -0.001032] w=0.117792 dir=1+/7- judgeable=YES
- `rq2b_cc_minus_global_m50_singleton_rate` (RQ2-B: Class-Conditional minus Global at m=50): -0.015840 [-0.065652, +0.015031] w=0.080683 dir=3+/5- judgeable=NO
- `rq2b_cc_minus_global_m50_average_set_size` (RQ2-B: Class-Conditional minus Global at m=50): +0.022908 [+0.002359, +0.070151] w=0.067792 dir=7+/1- judgeable=YES
- `rq2b_cc_minus_global_m100_coverage_minority` (RQ2-B: Class-Conditional minus Global at m=100): +0.041143 [+0.002397, +0.090852] w=0.088455 dir=8+/0- judgeable=YES
- `rq2b_cc_minus_global_m100_coverage_majority` (RQ2-B: Class-Conditional minus Global at m=100): -0.016877 [-0.043237, -0.003194] w=0.040044 dir=0+/8- judgeable=YES
- `rq2b_cc_minus_global_m100_coverage_disparity` (RQ2-B: Class-Conditional minus Global at m=100): -0.028515 [-0.100144, -0.001651] w=0.098493 dir=1+/7- judgeable=YES
- `rq2b_cc_minus_global_m100_singleton_rate` (RQ2-B: Class-Conditional minus Global at m=100): -0.006608 [-0.016713, +0.001054] w=0.017767 dir=3+/5- judgeable=NO
- `rq2b_cc_minus_global_m100_average_set_size` (RQ2-B: Class-Conditional minus Global at m=100): +0.012457 [+0.000369, +0.022263] w=0.021894 dir=7+/1- judgeable=YES

## 6. D10 20-seed expansion condition — verdict

- Preregistered rule (D10/O-06): 10->20 expansion may be **proposed** only for a core m=50/100 primary endpoint whose 95% D08 CI contains zero **and** whose width exceeds a user-approved practical-precision threshold; never p-value-motivated.
- Practical-precision threshold status: **NOT APPROVED** (explicitly an open parameter in D10; without it, expansion remains pending and cannot auto-run).
- **Verdict: `NOT_TRIGGERED`.** No seed expansion was executed, proposed for execution, or authorized; seeds remain fixed at 10. No p-value was computed or used.

## 7. Provenance

- Stage 10 script SHA-256: `63eb04da7dd97580dcb4e1d79cc5d7b9af8fe883d3660d27db2bf5c64ecd1456`
- Evidence: `artifacts/stage10/20260831T101122Z_stage10-results-freeze_63eb04da/results_qc_evidence.json`; events: `events.jsonl`
- Results manifest: `results/results_manifest.json` (artifact hashes below)

## 8. Boundary

No scientific interpretation, aggregate inference beyond the preregistered determinability report, publication figures, or manuscript work is included. The Global-coverage Wilson flag stays diagnostic-only. Next gates require separate user authorization.
