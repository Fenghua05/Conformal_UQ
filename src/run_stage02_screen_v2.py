"""Executable Stage 02 screen with audited in-memory typo/mapping corrections.

The source file remains the reviewable implementation; this launcher applies
only corrections demonstrated by the first screen attempt: one JSON bracket,
four OpenML target names, and robust rendering of explicit acquisition failures.
"""
from pathlib import Path


source_path = Path(__file__).with_name("screen_datasets.py")
source = source_path.read_text(encoding="utf-8")
replacements = {
    '"all_seed_feasible": r["all_seed_feasible"]} for r in registry["records"]}, indent=2': '"all_seed_feasible": r["all_seed_feasible"]} for r in registry["records"]]}, indent=2',
    '"target": "target", "domain": "marketing response"': '"target": "APPETENCY", "domain": "marketing response"',
    '"target": "y", "domain": "bank marketing"': '"target": "Class", "domain": "bank marketing"',
    '"target": "ACTION", "domain": "workplace access"': '"target": "target", "domain": "workplace access"',
    '"target": "class", "domain": "particle-physics simulation", "source_note": "OpenML versioned public dataset; original UCI MiniBooNE."': '"target": "signal", "domain": "particle-physics simulation", "source_note": "OpenML versioned public dataset; original UCI MiniBooNE."',
    'isinstance(record[field], (dict, list)) else record[field] for field in fields': 'isinstance(record.get(field), (dict, list)) else record.get(field) for field in fields',
    'audit = record["integrity_audit"]\n        lines.append(f"| {record[\'display_name\']} | OpenML {record[\'source\'][\'openml_data_id\']} | {record[\'eligibility_reason\']} | {audit[\'exact_duplicate_rows_including_target\']}/{audit[\'duplicate_feature_rows\']} | {audit[\'known_time_or_target_leakage\']} |")': 'audit = record.get("integrity_audit", {})\n        lines.append(f"| {record[\'display_name\']} | OpenML {record[\'source\'][\'openml_data_id\']} | {record.get(\'eligibility_reason\', record.get(\'screen_failure\', \'Unspecified acquisition failure\'))} | {audit.get(\'exact_duplicate_rows_including_target\', \'NA\')}/{audit.get(\'duplicate_feature_rows\', \'NA\')} | {audit.get(\'known_time_or_target_leakage\', \'Acquisition failed before field audit.\')} |")',
}
for broken, fixed in replacements.items():
    if source.count(broken) != 1:
        raise RuntimeError(f"Expected exactly one in-memory correction site: {broken[:60]!r}")
    source = source.replace(broken, fixed)
namespace = {"__name__": "__main__", "__file__": str(source_path)}
exec(compile(source, str(source_path), "exec"), namespace)
