"""Build the credential-free, D08-003-authorized v1.1 TabPFN cache upload archive.

Runs locally WITHOUT TabPFN, a GPU, or network access.  It validates the
final v1.1 TabPFN cache lock and the hash-bound D08-003 receipt, verifies the
eight raw sources and all eighty v1.1 split manifests, stages exactly the
controlled inputs the cloud runner validates (including the environment
lock), records drift-proof bundle config/code hashes, and produces one
immutable archive plus its receipt for the user-operated AutoDL run.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from conformal_uq.provenance import sha256_path

RUNNER_PATH = Path(__file__).resolve().parent / "02_generate_v11_tabpfn_caches.py"
_spec = importlib.util.spec_from_file_location("v11_tabpfn_cache_runner_shared", RUNNER_PATH)
assert _spec is not None and _spec.loader is not None
runner = importlib.util.module_from_spec(_spec)
sys.modules["v11_tabpfn_cache_runner_shared"] = runner
_spec.loader.exec_module(runner)

CANONICAL_LOCK_PATH = runner.CANONICAL_LOCK_PATH
CANONICAL_RECEIPT_PATH = runner.CANONICAL_RECEIPT_PATH
AUTHORIZED_TABPFN_UNITS = runner.AUTHORIZED_TABPFN_UNITS
STAGE_ROOT_NAME = "conformal_uq_stage08_v11_cache_upload"
ARCHIVE_NAME = "stage08_v11_tabpfn_cache_upload.tar.gz"

PROHIBITED_MEMBER_MARKERS = (
    "predictions.npz", "artifacts/caches/", "artifacts/runs/", "artifacts/splits/v1.0/",
    "results_long", "formal_run_manifest", "figures/", "figures\\",
    "credential", "token", "secret", "password", ".pem", ".ckpt", ".pt", ".pth", ".bin",
    "__pycache__", ".pyc", ".npz",
)


def require_absent_directory(path: Path) -> None:
    """Reject any output directory that already exists (immutable packages)."""
    if path.exists():
        raise FileExistsError(f"Output directory already exists and cannot be overwritten: {path}")


def scan_prohibited_members(names: list[str]) -> list[str]:
    """Flag cache, credential, checkpoint, CP, and formal members by relative name."""
    flagged: list[str] = []
    for name in names:
        normalized = name.replace("\\", "/").lower()
        if any(marker in normalized for marker in PROHIBITED_MEMBER_MARKERS):
            flagged.append(name)
    return flagged


def controlled_upload_inputs(lock: Mapping[str, Any], registry: Mapping[str, Any]) -> list[Path]:
    """Every controlled input the cloud runner validates, in a stable order."""
    records = {record["dataset_id"]: record for record in registry["records"]}
    locked_ids = list(lock["registry"]["locked_primary_ids"])
    inputs = [
        Path("src"),
        Path("cloud/tabpfn_stage08"),
        Path("cloud/tabpfn_stage05b/requirements-tabpfn.lock"),
        Path("configs/stage05b_tabpfn_v1.1.yaml"),
        Path("configs/stage04_splits_v1.1.yaml"),
        Path("decisions/D08-003_V11_CACHE_AND_PILOT_BUDGET_RECEIPT.json"),
        Path("protocols/protocol_v1.1.md"),
        Path("protocols/dataset_lock_v1.0.md"),
        Path("environment/environment_lock_v1.0.json"),
        Path(str(lock["registry"]["path"]).replace("\\", "/")),
    ]
    for dataset_id in locked_ids:
        if dataset_id not in records:
            raise ValueError(f"Locked dataset {dataset_id} is absent from the registry.")
        inputs.append(Path(str(records[dataset_id]["source"]["raw_local_path"]).replace("\\", "/")))
        for seed in lock["seeds"]:
            inputs.append(Path("artifacts/splits/v1.1") / dataset_id / f"seed-{seed}.json")
    return inputs


def verify_raw_sources(root: Path, lock: Mapping[str, Any], registry: Mapping[str, Any]) -> dict[str, str]:
    """Verify every locked raw source against its registry SHA-256 before upload."""
    records = {record["dataset_id"]: record for record in registry["records"]}
    observed: dict[str, str] = {}
    for dataset_id in lock["registry"]["locked_primary_ids"]:
        record = records[dataset_id]
        raw_path = root / str(record["source"]["raw_local_path"]).replace("\\", "/")
        if not raw_path.is_file():
            raise FileNotFoundError(f"Locked raw source is absent: {raw_path}")
        observed_hash = sha256_path(raw_path)
        if observed_hash != str(record["source"]["raw_sha256"]):
            raise ValueError(f"{dataset_id}: raw source SHA-256 differs from the locked registry value.")
        observed[dataset_id] = observed_hash
    return observed


def verify_split_manifests(root: Path, lock: Mapping[str, Any]) -> int:
    """Verify all eighty v1.1 split manifests are present and structurally intact."""
    count = 0
    for dataset_id in lock["registry"]["locked_primary_ids"]:
        for seed in lock["seeds"]:
            path = root / "artifacts/splits/v1.1" / dataset_id / f"seed-{seed}.json"
            if not path.is_file():
                raise FileNotFoundError(f"v1.1 split manifest is absent: {path}")
            manifest = json.loads(path.read_text(encoding="utf-8"))
            split_ids = manifest.get("split_ids")
            if (
                manifest.get("dataset_id") != dataset_id
                or manifest.get("base_seed") != seed
                or not isinstance(manifest.get("split_hash"), str) or len(manifest["split_hash"]) != 64
                or not isinstance(manifest.get("raw_sha256"), str) or len(manifest["raw_sha256"]) != 64
                or not isinstance(split_ids, dict) or set(split_ids) != {"train", "calibration_pool", "test"}
            ):
                raise ValueError(f"v1.1 split manifest is structurally invalid: {path}")
            count += 1
    if count != AUTHORIZED_TABPFN_UNITS:
        raise ValueError(f"Expected {AUTHORIZED_TABPFN_UNITS} v1.1 split manifests, found {count}.")
    return count


def _copy_controlled(root: Path, stage_root: Path, relative: Path) -> None:
    source, destination = root / relative, stage_root / relative
    if source.is_dir():
        shutil.copytree(source, destination, dirs_exist_ok=False, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def staged_config_hash(stage_root: Path) -> str:
    """Deterministic v1.1 TabPFN cache config hash over the staged lock files."""
    interim = hashlib.sha256()
    interim.update((stage_root / "configs/stage04_splits_v1.1.yaml").read_bytes())
    interim.update((stage_root / "configs/stage05b_tabpfn_v1.1.yaml").read_bytes())
    final = hashlib.sha256((stage_root / "configs/stage05b_tabpfn_v1.1.yaml").read_bytes()).hexdigest()
    return hashlib.sha256(bytes.fromhex(interim.hexdigest()) + bytes.fromhex(final)).hexdigest()


def staged_code_hash(stage_root: Path) -> str:
    """Hash the staged src and stage08 cloud code exactly as the runner does.

    Uses the same platform-independent POSIX-relative-string ordering as
    ``v11_tabpfn_code_hash`` so the recorded value matches the cloud runner.
    """
    entries = [
        (str(path.relative_to(stage_root)).replace("\\", "/"), path)
        for directory in (stage_root / "src", stage_root / "cloud" / "tabpfn_stage08")
        for path in directory.rglob("*.py")
    ]
    digest = hashlib.sha256()
    for relative, path in sorted(entries, key=lambda item: item[0]):
        digest.update(relative.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _inventory(stage_root: Path) -> list[dict[str, str | int]]:
    return [
        {"path": str(path.relative_to(stage_root)).replace("\\", "/"), "bytes": path.stat().st_size, "sha256": sha256_path(path)}
        for path in sorted(stage_root.rglob("*")) if path.is_file()
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the D08-003-authorized v1.1 TabPFN cache upload archive.")
    parser.add_argument("--lock", type=Path, default=CANONICAL_LOCK_PATH)
    parser.add_argument("--receipt", type=Path, default=CANONICAL_RECEIPT_PATH)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    require_absent_directory(output_dir)
    lock = runner.load_and_validate_v11_cache_lock(ROOT, args.lock)
    authorization = runner.load_validated_d08_003_authorization(ROOT, args.receipt)
    receipt = authorization["receipt"]
    registry = json.loads((ROOT / str(lock["registry"]["path"]).replace("\\", "/")).read_text(encoding="utf-8"))
    if registry.get("status") not in {"LOCKED_BY_USER_CONFIRMATION", "LOCKED"}:
        raise ValueError("The dataset registry is not status-consistent and locked.")
    raw_hashes = verify_raw_sources(ROOT, lock, registry)
    split_count = verify_split_manifests(ROOT, lock)
    controlled = controlled_upload_inputs(lock, registry)
    output_dir.mkdir(parents=True)
    stage_root = output_dir / STAGE_ROOT_NAME
    stage_root.mkdir()
    for relative in controlled:
        source = ROOT / relative
        if not source.exists():
            raise FileNotFoundError(f"Controlled package input is absent: {relative}")
        _copy_controlled(ROOT, stage_root, relative)
    member_names = [str(path.relative_to(stage_root)).replace("\\", "/") for path in stage_root.rglob("*")]
    prohibited = scan_prohibited_members(member_names)
    if prohibited:
        raise ValueError(f"Refusing to package prohibited members: {prohibited}")
    receipt_relative = PurePosixPath("decisions/D08-003_V11_CACHE_AND_PILOT_BUDGET_RECEIPT.json")
    receipt_members = [name for name in member_names if name.endswith(receipt_relative.name)]
    if len(receipt_members) != 1:
        raise ValueError("The D08-003 budget receipt must appear exactly once in the package.")
    bundle_config_sha256 = staged_config_hash(stage_root)
    bundle_code_sha256 = staged_code_hash(stage_root)
    inventory = {
        "artifact_id": "stage08_v11_tabpfn_cache_upload",
        "stage": "Stage 08 / Task 5",
        "created_utc": runner.utc_now(),
        "stage_root": STAGE_ROOT_NAME,
        "scope": "D08_003_AUTHORIZED_80_UNIT_TABPFN_PROBABILITY_CACHE_UPLOAD",
        "protocol_version": "v1.1",
        "authorized_tabpfn_units": AUTHORIZED_TABPFN_UNITS,
        "datasets": len(lock["registry"]["locked_primary_ids"]),
        "seeds_per_dataset": len(lock["seeds"]),
        "split_manifest_count": split_count,
        "budget": {
            "maximum_wall_clock_hours": receipt["maximum_wall_clock_hours"],
            "maximum_cloud_storage_gb": receipt["maximum_cloud_storage_gb"],
        },
        "bundle_config_sha256": bundle_config_sha256,
        "bundle_code_sha256": bundle_code_sha256,
        "d08_003_receipt": {
            "path": str(receipt_relative),
            "sha256": sha256_path(ROOT / "decisions/D08-003_V11_CACHE_AND_PILOT_BUDGET_RECEIPT.json"),
        },
        "tabpfn_cache_lock_sha256": authorization["final_lock_sha256"],
        "split_lock_sha256": sha256_path(ROOT / runner.CANONICAL_SPLIT_LOCK_PATH),
        "environment_lock_sha256": sha256_path(ROOT / runner.CANONICAL_ENVIRONMENT_LOCK_PATH),
        "raw_source_sha256": raw_hashes,
        "formal_run_manifest_authorized": False,
        "full_experiment_authorized": False,
        "cp_evaluated": False,
        "pilot_outputs": False,
        "included": [str(path).replace("\\", "/") for path in controlled],
        "excluded": [
            "existing probability caches", "CP/pilot/formal outputs", "results_long",
            "figures", "checkpoint binaries", "credentials/tokens", "v1.0 artifacts",
        ],
        "files": _inventory(stage_root),
    }
    (stage_root / "upload_inventory.json").write_text(json.dumps(inventory, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    archive_path = output_dir / ARCHIVE_NAME
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(stage_root, arcname=stage_root.name)
    archive_receipt = {
        "artifact_id": "stage08_v11_tabpfn_cache_upload_archive",
        "created_utc": runner.utc_now(),
        "archive": archive_path.name,
        "archive_sha256": sha256_path(archive_path),
        "archive_bytes": archive_path.stat().st_size,
        "stage_root": STAGE_ROOT_NAME,
        "scope": inventory["scope"],
        "cloud_execution_authorized": True,
        "authorized_tabpfn_units": AUTHORIZED_TABPFN_UNITS,
        "bundle_config_sha256": bundle_config_sha256,
        "bundle_code_sha256": bundle_code_sha256,
        "tabpfn_cache_lock_sha256": authorization["final_lock_sha256"],
        "budget_receipt": inventory["d08_003_receipt"],
        "budget": inventory["budget"],
        "formal_run_manifest_authorized": False,
        "full_experiment_authorized": False,
    }
    (output_dir / "archive_receipt.json").write_text(json.dumps(archive_receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(archive_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
