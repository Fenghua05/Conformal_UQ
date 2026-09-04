"""Build a credential-free Stage 08 v1.1 cloud-preflight archive."""

from __future__ import annotations

import argparse
import json
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_item(source: Path, target_root: Path) -> None:
    relative = source.relative_to(ROOT)
    destination = target_root / relative
    if source.is_dir():
        shutil.copytree(source, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def inventory(root: Path) -> list[dict[str, str | int]]:
    return [{"path": str(path.relative_to(root)).replace("\\", "/"), "bytes": path.stat().st_size, "sha256": sha256(path)} for path in sorted(root.rglob("*")) if path.is_file()]


def load_budget_receipt(path: Path, config_path: Path) -> dict:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict):
        raise ValueError("Budget receipt must be a JSON object.")
    if receipt.get("status") != "APPROVED_FOR_STAGE08_CLOUD_PREFLIGHT_EXECUTION":
        raise ValueError("Budget receipt does not authorize Stage 08 cloud preflight execution.")
    if receipt.get("config_sha256") != sha256(config_path):
        raise ValueError("Budget receipt is not bound to this exact preflight configuration.")
    for field in ("maximum_wall_clock_hours", "maximum_cloud_storage_gb"):
        value = receipt.get(field)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"Budget receipt field {field} must be a positive number.")
    return receipt


def controlled_common_inputs(config: dict) -> list[Path]:
    """Return all non-data inputs whose provenance the runner validates."""
    return [
        Path("src"),
        Path("cloud/tabpfn_stage08"),
        Path("cloud/tabpfn_stage05b/requirements-tabpfn.lock"),
        Path("configs/stage08_tabpfn_full_context_preflight_v1.1.yaml"),
        Path("protocols/protocol_v1.1.md"),
        Path("protocols/dataset_lock_v1.0.md"),
        Path("decisions/D08-001_APPROVAL_RECEIPT.md"),
        Path("environment/environment_lock_v1.0.json"),
        Path(config["registry_path"]),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--budget-receipt", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config_path, output_dir = args.config.resolve(), args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError("Output directory already exists and cannot be overwritten.")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("artifact_status") != "APPROVED_PACKAGE_PREPARATION_ONLY" or config.get("execution_gate", {}).get("numeric_cloud_budget_required_before_execution") is not True:
        raise ValueError("Refusing to package an uncontrolled or execution-authorized configuration.")
    budget_path = args.budget_receipt.resolve() if args.budget_receipt else None
    budget = load_budget_receipt(budget_path, config_path) if budget_path else None
    registry_path = ROOT / config["registry_path"]
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    records = {row["dataset_id"]: row for row in registry["records"]}
    unit_ids = [unit["dataset_id"] for unit in config["units"]]
    if len(unit_ids) != 3 or len(set(unit_ids)) != 3 or any(unit_id not in records for unit_id in unit_ids):
        raise ValueError("Preflight package requires exactly the approved three registry datasets.")
    output_dir.mkdir(parents=True)
    stage_root = output_dir / "conformal_uq_stage08_preflight_upload"
    stage_root.mkdir()
    common = controlled_common_inputs(config)
    controlled = common + [Path(records[dataset_id]["source"]["raw_local_path"].replace("\\", "/")) for dataset_id in unit_ids]
    if budget_path:
        controlled.append(budget_path.relative_to(ROOT))
    for relative in controlled:
        source = ROOT / relative
        if not source.exists():
            raise FileNotFoundError(f"Controlled package input is absent: {relative}")
        copy_item(source, stage_root)
    generated_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    prohibited = ["credentials/tokens", "checkpoint binaries", "prediction caches", "predictions.npz", "CP cells", "results_long", "figures", "formal run manifests"]
    scope = "THREE_UNIT_CLOUD_PREFLIGHT_AUTHORIZED_BY_BOUND_NUMERIC_BUDGET_ONLY" if budget else "PACKAGE_ONLY_CLOUD_EXECUTION_REQUIRES_SEPARATE_NUMERIC_BUDGET_APPROVAL"
    metadata = {"artifact_id": "stage08_v11_full_context_preflight_upload", "created_utc": generated_utc, "scope": scope, "config_sha256": sha256(config_path), "protocol_sha256": sha256(ROOT / "protocols/protocol_v1.1.md"), "approval_sha256": sha256(ROOT / "decisions/D08-001_APPROVAL_RECEIPT.md"), "units": config["units"], "included_roots": [str(item).replace("\\", "/") for item in controlled], "excluded": prohibited}
    if budget_path and budget:
        metadata["budget_receipt"] = {"path": str(budget_path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(budget_path), "maximum_wall_clock_hours": budget["maximum_wall_clock_hours"], "maximum_cloud_storage_gb": budget["maximum_cloud_storage_gb"]}
    (stage_root / "upload_inventory.json").write_text(json.dumps({**metadata, "files": inventory(stage_root)}, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    archive_path = output_dir / "stage08_v11_full_context_preflight_upload.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(stage_root, arcname=stage_root.name)
    receipt = {"artifact_id": "stage08_v11_full_context_preflight_archive", "created_utc": generated_utc, "archive": archive_path.name, "archive_sha256": sha256(archive_path), "archive_bytes": archive_path.stat().st_size, "scope": metadata["scope"], "cloud_execution_authorized": budget is not None}
    if budget_path and budget:
        receipt["budget_receipt"] = metadata["budget_receipt"]
    (output_dir / "archive_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(archive_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
