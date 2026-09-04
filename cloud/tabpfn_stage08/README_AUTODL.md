# Stage 08 v1.1 TabPFN 全上下文兼容性预检包

本包仅用于在已锁定的 AutoDL RTX 4090 / TabPFN 8.5.0 / checkpoint 环境中，准备三项**全训练集上下文兼容性预检**。它不包含、也不允许生成概率缓存、CP 输出、`results_long`、图或正式 run manifest。

> **当前阶段（2026-08-31 起）：下方“80-unit v1.1 TabPFN 概率缓存包”章节是本指南的现行操作内容。** 三单元预检（D08-002）已完成并通过独立审计，其历史章节保留在下方仅作参考；不要再重复运行预检命令。

## 已绑定的执行边界

已授权的预算为**最长 12 小时、云端存储最多 50 GB**，且只适用于以下四次拟合：Higgs、Numerai、Adult，以及一次 Higgs 重复。执行者只能使用包含 `D08-002_CLOUD_PREFLIGHT_BUDGET_RECEIPT.json` 的授权包；脚本会在导入 TabPFN 前验证该回执与配置的哈希绑定。

始终不得：

- 改变 TabPFN `8.5.0`、checkpoint、CUDA 设备或预处理；
- 设置 `ignore_pretraining_limits=True`；
- 截断、抽样或子采样训练集；
- 生成 `predictions.npz`、CP 或正式结果；
- 以任何方式启动全量实验。

## 本地生成授权包

在项目根目录运行：

```powershell
E:\anaconda3\python.exe cloud\tabpfn_stage08\01_build_upload_bundle.py --config configs\stage08_tabpfn_full_context_preflight_v1.1.yaml --budget-receipt decisions\D08-002_CLOUD_PREFLIGHT_BUDGET_RECEIPT.json --output-dir artifacts\stage08_v11_transfer\preflight_execution_authorized_01
```

上传生成的 `stage08_v11_full_context_preflight_upload.tar.gz` 时，不要上传 checkpoint、访问令牌、云端凭据或现有缓存。

## AutoDL 执行（仅授权包）

以下示例假定你已将授权包上传到 AutoDL 的 `/root/autodl-tmp/`。先定位并校验文件；显示的哈希必须为 `0829592c78aaac194342dbd928b609d7b6cf3baf2df69af9cfea703e8515fc95`：

```bash
cd /root/autodl-tmp
ls -lh stage08_v11_full_context_preflight_upload.tar.gz
sha256sum stage08_v11_full_context_preflight_upload.tar.gz
```

创建一个专用目录、解压并进入包根目录。不要把它解压到现有 cache、checkpoint 或正式实验目录中：

```bash
mkdir -p /root/autodl-tmp/stage08_v11_preflight_20260831
tar -xzf /root/autodl-tmp/stage08_v11_full_context_preflight_upload.tar.gz -C /root/autodl-tmp/stage08_v11_preflight_20260831
cd /root/autodl-tmp/stage08_v11_preflight_20260831/conformal_uq_stage08_preflight_upload
pwd
ls -lh decisions/D08-002_CLOUD_PREFLIGHT_BUDGET_RECEIPT.json configs/stage08_tabpfn_full_context_preflight_v1.1.yaml
```

确认上述路径存在后，在**既有、已锁定的 TabPFN 8.5.0 / RTX 4090 环境**中运行；`preflight_20260831` 必须是一个尚不存在的输出目录：

```bash
python cloud/tabpfn_stage08/00_full_context_preflight.py \
  --config configs/stage08_tabpfn_full_context_preflight_v1.1.yaml \
  --budget-receipt decisions/D08-002_CLOUD_PREFLIGHT_BUDGET_RECEIPT.json \
  --output-dir artifacts/stage08_v11_cloud/preflight_20260831
```

结束后只回传该输出目录中的 `preflight_manifest.json`、`events.jsonl` 和（如有）`failure_records/`，以及云端运行日志。不要在同一会话继续启动 cache、pilot、CP 或正式实验。

---

# 80-unit v1.1 TabPFN 概率缓存包（D08-003 授权；当前阶段）

本包在已锁定的 AutoDL Ubuntu 22.04 / RTX 4090 24 GB / CUDA / TabPFN 8.5.0 / checkpoint 环境中，生成**恰好 80 个 v1.1 TabPFN 基础概率缓存**（8 个锁定数据集 × 10 个冻结种子）。它由 `decisions/D08-003_V11_CACHE_AND_PILOT_BUDGET_RECEIPT.json` 授权（最长 12 小时 / 云端存储最多 50 GB），且只允许概率缓存：**不允许** CP、pilot、`results_long`、图或正式 run manifest。

不变约束（与预检一致）：

- 只使用本包自带的 `configs/stage05b_tabpfn_v1.1.yaml` 锁与 D08-003 回执；脚本在导入 TabPFN 之前先验证回执与锁的哈希绑定；
- checkpoint 固定为 `/root/autodl-fs/tabpfn-model-cache/tabpfn-v3-classifier-v3_default.ckpt`（SHA-256 `d0d865d54dfbc524f5703104be90620182dca7e5fb2c16de72e9959ea18f3988`）；
- 每个 unit 在**完整固定 v1.1 训练分区**上拟合：不截断、不抽样、不子采样，`ignore_pretraining_limits` 保持 false；
- 遵守 `100000 行 × 2000 特征`安全上限；
- 已完成的缓存不会被覆盖；可断点续跑（用一个新的 `--output-dir` 重新执行同一命令，已验证缓存会被复用）；
- 不得在同一会话继续执行 CP、pilot 或正式实验。

## 1. 上传并校验

将 `stage08_v11_tabpfn_cache_upload.tar.gz` 上传到 AutoDL 的 `/root/autodl-tmp/`，然后核对其 SHA-256（必须与你收到的本地 `archive_receipt.json` 中的 `archive_sha256` 完全一致）：

```bash
cd /root/autodl-tmp
ls -lh stage08_v11_tabpfn_cache_upload.tar.gz
sha256sum stage08_v11_tabpfn_cache_upload.tar.gz
```

## 2. 解压到全新目录并进入包根

```bash
mkdir -p /root/autodl-tmp/stage08_v11_cache_20260831
tar -xzf /root/autodl-tmp/stage08_v11_tabpfn_cache_upload.tar.gz -C /root/autodl-tmp/stage08_v11_cache_20260831
cd /root/autodl-tmp/stage08_v11_cache_20260831/conformal_uq_stage08_v11_cache_upload
pwd
ls -lh configs/stage05b_tabpfn_v1.1.yaml decisions/D08-003_V11_CACHE_AND_PILOT_BUDGET_RECEIPT.json
```

## 3. 生成 80 个缓存（唯一授权的生成命令）

`cache_run_20260831` 必须是一个尚不存在的输出目录：

```bash
python cloud/tabpfn_stage08/02_generate_v11_tabpfn_caches.py \
  --lock configs/stage05b_tabpfn_v1.1.yaml \
  --receipt decisions/D08-003_V11_CACHE_AND_PILOT_BUDGET_RECEIPT.json \
  --output-dir artifacts/stage08_v11_cloud/cache_run_20260831
```

完成后确认 `summary.json` 中 `"status": "PASS"`、`"completed_units": 80`。若中途失败，保留现场并把 `summary.json`、`events.jsonl` 与 `artifacts/failures/` 中的新失败记录一并回传；不要删除任何缓存或失败记录。

## 4. 校验并打包回传

```bash
python cloud/tabpfn_stage08/03_verify_and_pack_v11_caches.py \
  --lock configs/stage05b_tabpfn_v1.1.yaml \
  --receipt decisions/D08-003_V11_CACHE_AND_PILOT_BUDGET_RECEIPT.json \
  --run-dir artifacts/stage08_v11_cloud/cache_run_20260831 \
  --output-dir /root/autodl-tmp/stage08_v11_cache_return_20260831
```

## 5. 只回传以下内容

- `stage08_v11_tabpfn_cache_return.tar.gz` 及其 SHA-256（`sha256sum` 输出）；
- 同目录下脚本打印/生成的 `archive_receipt.json`。

本地独立审计通过 240 缓存门之前，不要在云端做任何其他事情。
