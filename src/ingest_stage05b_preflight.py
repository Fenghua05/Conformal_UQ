"""Safely preserve a returned Stage 05B cloud preflight package as a controlled input."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path


EXPECTED_MEMBERS = {
    "artifacts/stage05b_cloud/preflight_02/events.jsonl",
    "artifacts/stage05b_cloud/preflight_02/preflight.json",
    "artifacts/stage05b_cloud/preflight_02/pip_freeze.txt",
    "artifacts/stage05b_cloud/preflight_02/gpu_inventory.txt",
}


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite preflight input directory: {args.output_dir}")
    with tarfile.open(args.archive, "r:gz") as archive:
        members = {member.name for member in archive.getmembers() if member.isfile()}
        if members != EXPECTED_MEMBERS:
            raise ValueError(f"Unexpected archive members: {sorted(members)}")
        args.output_dir.mkdir(parents=True)
        for name in sorted(EXPECTED_MEMBERS):
            source = archive.extractfile(name)
            if source is None:
                raise ValueError(f"Unable to read expected archive member: {name}")
            target = args.output_dir / Path(name).name
            target.write_bytes(source.read())
    preflight_path = args.output_dir / "preflight.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    required = {"status", "tabpfn_version", "checkpoint_sha256", "checkpoint_path", "code_hash", "cuda_available"}
    missing = sorted(required.difference(preflight))
    if missing or preflight["status"] != "PASS" or preflight["cuda_available"] is not True:
        raise ValueError(f"Preflight is not a passing CUDA record; missing={missing}")
    receipt = {
        "artifact_id": "stage05b_preflight_input_receipt_v1.0",
        "source_archive": str(args.archive),
        "source_archive_sha256": sha256_path(args.archive),
        "files": {path.name: sha256_path(path) for path in sorted(args.output_dir.iterdir()) if path.is_file()},
    }
    (args.output_dir / "intake_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(args.output_dir / "intake_receipt.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
