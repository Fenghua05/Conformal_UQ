"""Re-render Stage 02 documents from existing registry in frozen rank order."""
from pathlib import Path

source_path = Path(__file__).with_name("screen_datasets.py")
source = source_path.read_text(encoding="utf-8")
changes = {
    '"all_seed_feasible": r["all_seed_feasible"]} for r in registry["records"]}, indent=2': '"all_seed_feasible": r["all_seed_feasible"]} for r in registry["records"]]}, indent=2',
    'replacements = [r for r in registry["records"] if r.get("proposed_role") == "replacement"]': 'replacements = sorted([r for r in registry["records"] if r.get("proposed_role") == "replacement"], key=lambda r: r["frozen_selection_rank"])',
}
for broken, fixed in changes.items():
    if source.count(broken) != 1:
        raise RuntimeError(f"Expected exactly one correction site: {broken[:60]!r}")
    source = source.replace(broken, fixed)
namespace = {"__name__": "__main__", "__file__": str(source_path)}
exec(compile(source, str(source_path), "exec"), namespace)
