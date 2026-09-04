"""Execute the Stage 02 screen after correcting a single bracket typo in memory.

The correction is deliberately asserted to be unique; it does not change the
screening algorithm, configuration, candidates, or output fields.
"""
from pathlib import Path


source_path = Path(__file__).with_name("screen_datasets.py")
source = source_path.read_text(encoding="utf-8")
broken = '"all_seed_feasible": r["all_seed_feasible"]} for r in registry["records"]}, indent=2'
fixed = '"all_seed_feasible": r["all_seed_feasible"]} for r in registry["records"]]}, indent=2'
if source.count(broken) != 1:
    raise RuntimeError("Expected exactly one known Stage 02 output-serialization typo.")
source = source.replace(broken, fixed)
namespace = {"__name__": "__main__", "__file__": str(source_path)}
exec(compile(source, str(source_path), "exec"), namespace)
