import sys
import unittest
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conformal_uq.identity import derive_seed, run_id
from conformal_uq.paths import ImmutablePathError, create_immutable_run_dir


class IdentityTests(unittest.TestCase):
    def test_registry_seed_derivation_is_stable(self) -> None:
        canonical, derived = derive_seed("v1.0", "openml_3_kr_vs_kp", 104729, "stratified_test_split")
        self.assertEqual(canonical, "v1.0|openml_3_kr_vs_kp|104729|stratified_test_split")
        self.assertEqual(derived, 3604429721)

    def test_run_directory_cannot_be_reused(self) -> None:
        fixed = run_id("stage03", "a" * 64, datetime(2026, 8, 30, tzinfo=timezone.utc))
        root = ROOT / "tmp" / f"test_identity_{uuid.uuid4().hex}"
        create_immutable_run_dir(root, fixed)
        with self.assertRaises(ImmutablePathError):
            create_immutable_run_dir(root, fixed)
