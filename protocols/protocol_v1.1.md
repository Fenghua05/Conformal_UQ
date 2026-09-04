# Protocol v1.1 — Controlled Minority Calibration Scarcity

**Status:** `APPROVED_CONDITIONAL_ON_CLOUD_PREFLIGHT`  
**Protocol ID:** `conformal-uq-stage1-v1.1.0`  
**Approved at:** `2026-08-31T00:00:00+08:00`  
**Authority:** v2 implementation plan (`831193ADA5A1CDBF1DA52EC54F3916466D54AF9CFEC9563D3E80EF1AFD8F0058`); v4 is background only; approved decision D08-001.  
**Approval boundary:** The user approved the full-context TabPFN protocol change and preparation of a preflight package. Cloud preflight execution still requires a separately approved numeric budget; the formal run remains unapproved.

## 1. Scope and contribution boundary

This is a controlled-resampling empirical study of IID/exchangeable imbalanced binary tabular classification. It characterizes the pathway

`m_minority → finite-sample rank → (q0,q1) → binary set geometry → prediction-set efficiency`.

It does **not** propose a new CP method, score, estimator, fairness method, synthetic baseline, distribution-shift analysis, multiclass study, or causal architecture comparison. RQ3 is a comparison of model-appropriate predictive pipelines.

## 2. Research questions and hypotheses

- **RQ1 / H1:** Under Class-Conditional CP, quantify threshold, geometry, and efficiency changes as `m_minority` changes. Primary same-pipeline comparison is `m=100 − m=50`; efficiency is expected to improve beyond boundary conditions but need not be monotone.
- **RQ2 / H2:** At fixed m and identical calibration IDs, compare Global Split CP with Class-Conditional CP for class-wise coverage, disparity, and efficiency. Classwise validity may cost efficiency.
- **RQ3 / H3:** At identical dataset/seed/subset/method/score conditions, describe whether thresholds, geometry, and efficiency differ across LR, XGBoost, and TabPFN. AUROC/AUPRC ordering need not match efficiency ordering.

`m=10` is a boundary diagnostic and `m=20` a near-boundary diagnostic. Pipeline conclusions center on same-m `m=50,100` contrasts. Global CP is never interpreted as a cross-m pure method effect.

## 3. Data design and admission

- Target: 8 public, version-recorded binary tabular datasets, with 4 ordered replacements.
- Split every dataset × top-level seed once by stratified `60% train / 20% calibration pool / 20% test`.
- Every candidate must pass **all ten** locked splits: calibration-pool minority `≥100`, calibration-pool majority `≥200`, and test-minority `≥75`.
- Candidate source/ranking rule: OpenML versioned task, then UCI versioned archive, then documented public benchmark; rank only by (1) documented binary labels, (2) licence/access, (3) all-seed feasibility, (4) tabular/non-text/non-image suitability, (5) post-transform feature count `≤500`, then (6) source ID ascending. Lock first eight that pass and next four as replacements.
- A replacement is allowed only before outcome analysis for source/licence/label-semantic failure, all-seed infeasibility, feature-cap failure, or irreparable data-integrity failure. Never replace for result direction, effect size, CI width, p-value, or model ranking.

Dataset names remain intentionally unresolved until authorized Stage 02 registry verification; this is a conditional, precommitted selection rule, not a gap to fill post hoc.

## 4. Randomness, split and calibration subsets

Top-level seeds are exactly:

`[104729, 130363, 155921, 196613, 262147, 318281, 374209, 419893, 481517, 552721]`.

Derive a purpose-specific 32-bit seed as the first 32 bits of `SHA256(protocol_version + "|" + dataset_id + "|" + base_seed + "|" + purpose)`. Record both canonical input string and resulting unsigned integer. Do not use runtime-dependent language hashes.

For each dataset × seed:

1. Fit every preprocessing component on train only.
2. Train/predict each base pipeline once; calibration pool and test are never fitting/tuning data.
3. Fix one 200-sample majority calibration subset.
4. Draw nested minority subsets `S10 ⊂ S20 ⊂ S50 ⊂ S100` of sizes `10,20,50,100`.
5. For a fixed m, Global and Class-Conditional CP must use exactly the same calibration IDs and saved subset hash.

`alpha=0.1`; `m_majority=200`; `m_minority ∈ {10,20,50,100}`.

## 5. Predictive pipelines

### 5.1 Shared preprocessing rule

Use train-only `ColumnTransformer`: numerical median imputation and `StandardScaler` for LR; categorical most-frequent imputation and `OneHotEncoder(handle_unknown='ignore')`; no target encoding. XGBoost consumes the same semantic transformed features but need not be scaled. Any transformed feature count above 500 makes a candidate ineligible.

### 5.2 Logistic Regression

No HPO: `C=1.0`, L2 penalty, `lbfgs`, `max_iter=2000`, `class_weight=None`, and the derived seed where accepted. It outputs class probabilities.

### 5.3 XGBoost

No HPO and no early stopping: `objective='binary:logistic'`, `eval_metric='logloss'`, `n_estimators=300`, `max_depth=6`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`, `min_child_weight=1`, `reg_lambda=1`, `reg_alpha=0`, `scale_pos_weight=1`, `tree_method='hist'`, `n_jobs=1`, and derived seed. It outputs class probabilities.

### 5.4 TabPFN full-context conditional lock

The fixed deployment is AutoDL Ubuntu 22.04 on an RTX 4090 (24 GB), CUDA, TabPFN `8.5.0`, and checkpoint SHA-256 `d0d865d54dfbc524f5703104be90620182dca7e5fb2c16de72e9959ea18f3988`. No version/checkpoint substitution is permitted.

For every dataset × seed, TabPFN must use the entire fixed train partition: no context truncation, sampling, or subsampling. The project-side safety limit is 100,000 train rows and 2,000 transformed features; the existing Stage 04 feature cap of 500 remains stricter. `ignore_pretraining_limits` remains exactly `false`.

Before any v1.1 base cache, the user-operated cloud compatibility preflight must pass on seed 104729 for Higgs, Numerai, and Adult. It must confirm the fixed runtime/checkpoint, transform shape, finite aligned `[0,1]` probabilities, resource envelope, and a repeated Higgs prediction with maximum absolute difference `≤1e-10`. It creates no CP output, cache, or formal result. If it fails, preserve evidence and stop for a user decision; never silently substitute, omit, downgrade, sample, truncate, or override pretraining limits.

## 6. Conformal methods and mathematics

Only two methods are allowed: Global Split CP and label/Class-Conditional (Mondrian) Split CP. The only score is `s(x,y)=1-p_y(x)` and a candidate label is included iff `s(x,y) ≤ q_y`.

For n calibration scores, use the exact rank `r=ceil((n+1)(1-alpha))`; use sorted `S_(r)` when `r≤n`, otherwise the augmented-quantile value `+∞`. Do not replace this with an interpolated quantile. On the registered minority grid, ranks are 10, 19, 46, and 91 respectively.

For binary `p=p1(x)`, inclusion is `1∈C(x)` iff `p≥1-q1` and `0∈C(x)` iff `p≤q0`.

- Empty interval: `q0 < p < 1-q1`, positive length iff `q0+q1<1`.
- Doubleton interval: `1-q1 ≤ p ≤ q0`, positive length iff `q0+q1>1`; at equality its boundary point remains a doubleton due to `≤`.

`ThresholdSum=q0+q1` is a geometry diagnostic. It does not alone determine empty/doubleton rates, which also depend on test predicted-probability distribution. Under a continuous class-specific score CDF, the Beta probability-scale order-statistic quantity is a diagnostic only; it is never used as singleton-rate or empirical-threshold-variance baseline.

## 7. Outcomes and QC

Base predictive outcomes: AUROC and AUPRC. Within dataset × seed × pipeline, both must be invariant across CP method and m.

CP outcomes: overall/minority/majority coverage; coverage disparity; singleton rate; average set size; empty and doubleton rates; `q_global`, `q_minority`, `q_majority`, threshold gap/sum; and across-seed threshold SD/IQR.

Required QC includes: exact ranks; Global/Class-Conditional subset identity; empty+singleton+doubleton=1; binary geometry consistency; predictive metric invariance; train-only preprocessing/no label leakage; and Global marginal-coverage sanity diagnostics.

## 8. Uncertainty and confirmatory analysis

For every test coverage cell, report covered count, class-specific denominator, estimate, and two-sided 95% Wilson score interval with `z=1.959963984540054` and no continuity correction.

Across seeds, report descriptive n, mean, median, SD (`ddof=1` when n≥2), and IQR. Do not pool seed-level Wilson intervals or treat seeds as independent datasets.

For a pre-specified paired comparison and metric, calculate the seed-mean within-dataset effect `d_j=mean_s(metric_A-metric_B)` on complete paired seeds. The confirmatory effect is `median(d_j)` over the eight datasets; report all eight `d_j` values and direction count. Its two-sided 95% CI is a 20,000-replicate percentile bootstrap of the eight whole-dataset effects, with a D01-derived RNG seed.

Auxiliary tests are exact two-sided Wilcoxon signed-rank tests on the nonzero eight-dataset effects; report discarded zeros and use Holm correction within each predeclared family. Formal testing is limited to RQ1 comparison A and RQ2 comparison B; RQ3 uses effect/CI/direction descriptive reporting unless separately approved later.

The seed count is fixed at 10. There is **no automatic 10→20 expansion**. Any expansion is a substantive, separately user-approved protocol deviation; it can never be motivated by a p-value.

## 9. Missing cells, failure, retry, and exclusions

Classify every failure as `data`, `environment`, `resource`, `bug`, or `protocol_question`.

- One identical-config retry is allowed only for a documented transient environment/resource failure.
- Deterministic failures receive no blind retry.
- A confirmed bug requires a regression test, new code/config hash, explicit stale-descendant map, and rerun only of the affected scope.
- Do not impute missing cells. A comparison includes a dataset only if all required paired cells are complete. Report incomplete comparisons in an availability table.
- A failed base-prediction unit makes all its descendant CP cells missing.
- An unresolved protocol question pauses work for the user.

## 10. Pilot rule

After Stage 02 locks the registry and before model training, select pilot datasets objectively: the highest-ranked eligible moderate-imbalance dataset (`0.10≤minority ratio≤0.30`) and severe-but-feasible dataset (`0.02≤minority ratio<0.10`). If either stratum is empty, choose the first two registry-ranked eligible datasets and log the exception. Pilot selection cannot use model/CP outcomes.

## 11. Protocol deviations and versioning

Documentation-only corrections (path spelling, prose, nonsemantic log metadata) may be logged without scientific reapproval. A change to data, split, seed, pipeline, CP method, score, m, outcome, comparison, CI, exclusion, version, or feasibility/compute rule is substantive: stop affected work; write a proposed deviation with rationale, alternatives, affected hashes and stale/rerun map; obtain explicit user approval; issue the required version increment; never mix outputs across versions.

## 12. Authorization boundary

This protocol change authorizes only the creation of v1.1 controlled documents and a credential-free, user-operated cloud compatibility-preflight package. It does not authorize cloud preflight execution, package/checkpoint download, device substitution, runtime/storage spending, cache generation, pilot/full runs, or formal-manifest freezing. A numeric cloud preflight budget must be explicitly approved before the three-unit preflight may run. After an independently audited preflight PASS, the user must separately authorize the full-cache budget; after a version-consistent pilot and Stage 08 audit, the user must separately give explicit `go` before a formal run manifest may be frozen.

## 13. v1.1 lineage and rerun rule

`v1.1` changes the canonical protocol-version input to all purpose-specific SHA-256 seed derivations. Therefore all v1.0 splits, preprocessing audits, probability caches, pilot results, and Stage 08 audit remain immutable historical evidence only and cannot be mixed with v1.1. Regenerate 80 splits, 240 base-prediction caches, a 480-cell outcome-blind v1.1 pilot, and its independent Stage 08 audit before considering the 1,920-cell formal design.
