"""Revalidate all 80 v1.1 TabPFN caches and pack one credential-free return archive.

Runs ONLY on the locked AutoDL runtime after ``02_generate_v11_tabpfn_caches.py``
reported a complete PASS.  Without importing TabPFN, it re-reads every cache
against the recomputed provenance, requires exactly one cfg/code tree with
exactly 80 complete units, rejects any CP/pilot/formal output, and packs the
caches plus control files and run evidence into one archive whose receipt the
user returns to the local auditor.
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
from typing import Any, Iterable, Mapping

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from conformal_uq.data import load_locked_dataset
from conformal_uq.paths import cache_path
from conformal_uq.prediction_cache import read_valid_cache
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
PROHIBITED_OUTPUT_MARKERS = (
    "results_long", "formal_run_manifest", "cp_cell", "pilot_cell", "figures",
)


def validate_generator_summary(summary: Mapping[str, Any], *, config_hash: str, code_hash: str, environment_hash: str) -> None:
    """Accept only a complete PASS summary of this exact code/config lineage."""
    if summary.get("status") != "PASS":
        raise ValueError("The generator run did not finish with status PASS; preserve its failure records and stop.")
    if summary.get("protocol_version") != "v1.1" or summary.get("scope") != "D08_003_V11_TABPFN_PROBABILITY_CACHES_ONLY":
        raise ValueError("The generator summary is not the v1.1 cache-only run record.")
    if summary.get("expected_units") != AUTHORIZED_TABPFN_UNITS or summary.get("completed_units") != AUTHORIZED_TABPFN_UNITS:
        raise ValueError(f"The generator summary must report exactly {AUTHORIZED_TABPFN_UNITS} completed units.")
    if summary.get("config_hash") != config_hash or summary.get("code_hash") != code_hash or summary.get("environment_hash") != environment_hash:
        raise ValueError("The generator summary lineage differs from the recomputed config/code/environment hashes.")
    for key in ("cp_evaluated", "pilot_outputs", "formal_run_manifest_created", "full_experiment_executed"):
        if summary.get(key) is not False:
            raise ValueError(f"The generator summary must keep {key} false.")


def validate_exact_v11_tabpfn_cache_relative_entries(
    entries: set[PurePosixPath], config_hash: str, code_hash: str, expected_units: Iterable[tuple[str, int]],
) -> list[str]:
    """Require one exact cfg/code tree and only the complete expected cache units."""
    expected_units = set(expected_units)
    cfg, code = f"cfg-{config_hash[:12]}", f"code-{code_hash[:12]}"
    expected: set[PurePosixPath] = {PurePosixPath(cfg), PurePosixPath(cfg, code)}
    for dataset_id, seed in expected_units:
        unit = PurePosixPath(cfg, code, dataset_id, f"seed-{seed}", "tabpfn")
        expected.update({unit.parents[2], unit.parents[1], unit.parents[0], unit})
        expected.update({unit / "manifest.json", unit / "predictions.npz"})
    errors: list[str] = []
    for entry in sorted(entries):
        if entry not in expected:
            if len(entry.parts) >= 2 and (entry.parts[0] != cfg or entry.parts[1] != code):
                errors.append(f"foreign cfg/code tree entry under the v1.1 cache root: {entry}")
            elif len(entry.parts) >= 5:
                errors.append(f"incomplete or unexpected cache directory/file under the expected tree: {entry}")
            else:
                errors.append(f"unexpected v1.1 cache path: {entry}")
    complete_units = 0
    for dataset_id, seed in sorted(expected_units):
        unit = PurePosixPath(cfg, code, dataset_id, f"seed-{seed}", "tabpfn")
        if (unit / "manifest.json") in entries and (unit / "predictions.npz") in entries:
            complete_units += 1
        else:
            errors.append(f"incomplete expected v1.1 TabPFN cache directory: {unit}")
    if complete_units != len(expected_units):
        errors.append(f"complete cache directory count mismatch: observed={complete_units} expected={len(expected_units)}")
    return errors


def scan_prohibited_outputs(paths: Iterable[Path]) -> list[str]:
    """Flag any CP/pilot/formal output file inside the run or cache roots."""
    flagged: list[str] = []
    for path in paths:
        if not path.is_file():
            continue
        name = path.name.lower()
        relative = str(path).replace("\\", "/").lower()
        if any(marker in name or marker in relative for marker in PROHIBITED_OUTPUT_MARKERS):
            flagged.append(str(path))
    return flagged


def _inventory(root: Path) -> list[dict[str, str | int]]:
    return [
        {"path": str(path.relative_to(root)).replace("\\", "/"), "bytes": path.stat().st_size, "sha256": sha256_path(path)}
        for path in sorted(root.rglob("*")) if path.is_file()
    ]


def _copy(source_root: Path, destination_root: Path, relative: PurePosixPath) -> None:
    source, destination = source_root / Path(*relative.parts), destination_root / Path(*relative.parts)
    if source.is_dir():
        shutil.copytree(source, destination, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify and pack the 80 v1.1 TabPFN caches into one return archive.")
    parser.add_argument("--lock", type=Path, default=CANONICAL_LOCK_PATH)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run_root = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not (run_root / "summary.json").is_file():
        raise FileNotFoundError(f"Generator run summary is absent: {run_root / 'summary.json'}")
    if output_dir.exists():
        raise FileExistsError(f"Return output directory already exists: {output_dir}")
    lock = runner.load_and_validate_v11_cache_lock(ROOT, args.lock)
    authorization = runner.load_validated_d08_003_authorization(ROOT, args.receipt)
    config_hash = runner.v11_tabpfn_config_hash(ROOT, lock)
    code_hash = runner.v11_tabpfn_code_hash(ROOT)
    environment_hash = sha256_path(ROOT / runner.CANONICAL_ENVIRONMENT_LOCK_PATH)
    split_lock_sha256 = sha256_path(ROOT / runner.CANONICAL_SPLIT_LOCK_PATH)
    tabpfn_cache_lock_sha256 = sha256_path(ROOT / CANONICAL_LOCK_PATH)
    summary = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))
    validate_generator_summary(summary, config_hash=config_hash, code_hash=code_hash, environment_hash=environment_hash)
    units = runner.expected_v11_tabpfn_units(lock)
    cache_root = ROOT / lock["paths"]["cache_root"]
    verified: list[dict[str, str | int]] = []
    tables: dict[str, Any] = {}
    for dataset_id, base_seed in units:
        if dataset_id not in tables:
            tables[dataset_id] = load_locked_dataset(ROOT, dataset_id, registry_path=ROOT / lock["registry"]["path"])
        table = tables[dataset_id]
        split = runner.locked_v11_split(ROOT, lock, table, base_seed)
        provenance = runner.v11_tabpfn_cache_provenance(
            config_hash=config_hash, code_hash=code_hash, environment_hash=environment_hash,
            table=table, split=split, base_seed=base_seed,
            split_lock_sha256=split_lock_sha256, tabpfn_cache_lock_sha256=tabpfn_cache_lock_sha256,
        )
        destination = cache_path(cache_root, config_hash, code_hash, dataset_id, base_seed, "tabpfn")
        read_valid_cache(
            destination, provenance,
            {"calibration_pool": split.ids.calibration_pool, "test": split.ids.test},
            {
                "calibration_pool": table.subset_labels(split.ids.calibration_pool).to_numpy(dtype="int8", copy=True),
                "test": table.subset_labels(split.ids.test).to_numpy(dtype="int8", copy=True),
            },
        )
        verified.append({"dataset_id": dataset_id, "seed": base_seed, "model": "tabpfn"})
    entries = {PurePosixPath(path.relative_to(cache_root).as_posix()) for path in cache_root.rglob("*")}
    errors = validate_exact_v11_tabpfn_cache_relative_entries(entries, config_hash, code_hash, set(units))
    if errors:
        raise ValueError("Cache tree validation failed: " + "; ".join(errors))
    prohibited = scan_prohibited_outputs(list(run_root.rglob("*")) + list(cache_root.rglob("*")))
    if prohibited:
        raise ValueError(f"Prohibited CP/pilot/formal outputs found: {prohibited}")
    stage_root = output_dir / "stage08_v11_tabpfn_cache_return"
    stage_root.mkdir(parents=True)
    controls = [
        PurePosixPath("configs/stage05b_tabpfn_v1.1.yaml"),
        PurePosixPath("configs/stage04_splits_v1.1.yaml"),
        PurePosixPath("decisions/D08-003_V11_CACHE_AND_PILOT_BUDGET_RECEIPT.json"),
        PurePosixPath("protocols/protocol_v1.1.md"),
        PurePosixPath("protocols/dataset_lock_v1.0.md"),
        PurePosixPath("environment/environment_lock_v1.0.json"),
        PurePosixPath(lock["registry"]["path"].replace("\\", "/")),
    ]
    for relative in controls:
        _copy(ROOT, stage_root, relative)
    for dataset_id, base_seed in units:
        cache_relative = PurePosixPath(*cache_path(Path("artifacts/caches/v1.1"), config_hash, code_hash, dataset_id, base_seed, "tabpfn").parts)
        _copy(ROOT, stage_root, cache_relative)
    run_relative = run_root.relative_to(ROOT)
    for path in sorted(run_root.rglob("*")):
        if path.is_file():
            _copy(ROOT, stage_root, PurePosixPath(*path.relative_to(ROOT).parts))
    for failure in summary.get("failures", []):
        failure_relative = PurePosixPath(failure["failure_record"].replace("\\", "/"))
        if (ROOT / failure_relative).is_file():
            _copy(ROOT, stage_root, failure_relative)
    inventory = {
        "artifact_id": "stage08_v11_tabpfn_cache_return_inventory",
        "stage": "Stage 08 / Task 5",
        "status": "PASS",
        "created_utc": runner.utc_now(),
        "source_run": str(run_relative).replace("\\", "/"),
        "scope": "D08_003_V11_TABPFN_PROBABILITY_CACHES_ONLY",
        "config_hash": config_hash, "code_hash": code_hash, "environment_hash": environment_hash,
        "split_lock_sha256": split_lock_sha256,
        "tabpfn_cache_lock_sha256": tabpfn_cache_lock_sha256,
        "d08_003_cache_lock_sha256": authorization["final_lock_sha256"],
        "expected_units": AUTHORIZED_TABPFN_UNITS,
        "verified_units": verified,
        "cp_evaluated": False, "pilot_outputs": False, "formal_run_manifest_created": False,
        "excluded": ["raw data", "credentials", "tokens", "checkpoint binaries", "CP/results_long outputs", "figures"],
        "files": _inventory(stage_root),
    }
    (stage_root / "return_inventory.json").write_text(json.dumps(inventory, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    archive_path = output_dir / "stage08_v11_tabpfn_cache_return.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(stage_root, arcname=stage_root.name)
    receipt = {
        "artifact_id": "stage08_v11_tabpfn_cache_return_archive",
        "created_utc": runner.utc_now(),
        "archive": archive_path.name,
        "archive_sha256": sha256_path(archive_path),
        "archive_bytes": archive_path.stat().st_size,
        "inventory_sha256": sha256_path(stage_root / "return_inventory.json"),
        "source_run": str(run_relative).replace("\\", "/"),
        "verified_units": len(verified),
        "config_hash": config_hash, "code_hash": code_hash,
        "d08_003_cache_lock_sha256": authorization["final_lock_sha256"],
        "cp_evaluated": False, "pilot_outputs": False, "formal_run_manifest_created": False,
    }
    (output_dir / "archive_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
