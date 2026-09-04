import json
import sys
import unittest
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conformal_uq.logging import REQUIRED_EVENT_FIELDS, write_event


class LoggingTests(unittest.TestCase):
    def test_jsonl_event_has_required_fields(self) -> None:
        path = ROOT / "tmp" / f"test_logging_{uuid.uuid4().hex}.jsonl"
        write_event(path, run_id="run", stage="03", level="INFO", event="smoke", config_hash="abc", message="ok")
        event = json.loads(path.read_text(encoding="utf-8"))
        self.assertTrue(all(event[field] for field in REQUIRED_EVENT_FIELDS))
