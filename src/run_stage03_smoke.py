"""Execute Stage 03 toy-only integration checks; never a research experiment."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conformal_uq.config import canonical_json, load_config
from conformal_uq.identity import run_id, sha256_text
from conformal_uq.logging import write_event
from conformal_uq.paths import create_immutable_run_dir
from conformal_uq.provenance import write_json
from conformal_uq.results_schema import validate_results_record
from conformal_uq.toy import make_toy_contract, toy_results_record


def main() -> int:
    config_path = ROOT / "configs" / "stage03_base_v1.0.yaml"
    config = load_config(config_path)
    config_hash = sha256_text(canonical_json(config))
    code_hash = sha256_text((ROOT / "src" / "conformal_uq" / "toy.py").read_text(encoding="utf-8"))
    evidence_root = ROOT / config["paths"]["stage03_evidence_root"] / "smoke"
    smoke_run_id = run_id("stage03-smoke", config_hash)
    run_dir = create_immutable_run_dir(evidence_root, smoke_run_id)
    table, split = make_toy_contract()
    write_event(
        run_dir / "events.jsonl", run_id=smoke_run_id, stage="03", level="INFO",
        event="toy_smoke_started", config_hash=config_hash,
        message="Toy-only smoke validation started; no real data, model fitting, or checkpoint access.",
    )
    record = toy_results_record(config_hash, code_hash, smoke_run_id)
    errors = validate_results_record(record)
    if errors:
        raise RuntimeError(f"Toy results record violates results schema contract: {errors}")
    payload = {
        "artifact_id": "stage03_toy_smoke",
        "status": "TOY_ONLY_NOT_RESEARCH_RESULT",
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "run_id": smoke_run_id,
        "config_hash": config_hash,
        "code_hash": code_hash,
        "toy_sample_count": len(table.sample_ids),
        "toy_split_sizes": {"train": len(split.train), "calibration_pool": len(split.calibration_pool), "test": len(split.test)},
        "results_record": record,
        "prohibition": "This evidence is synthetic smoke validation only and must never be reported as a research experiment or manuscript result.",
    }
    write_json(run_dir / "toy_smoke_evidence.json", payload)
    write_event(
        run_dir / "events.jsonl", run_id=smoke_run_id, stage="03", level="INFO",
        event="toy_smoke_passed", config_hash=config_hash,
        message="Toy-only smoke validation passed.",
    )
    print(json.dumps({"status": "PASS", "run_dir": str(run_dir), "marker": payload["status"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
