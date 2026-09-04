"""Final Stage 02 screen over the declared small/medium public candidate pool."""
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
    '\n]\n\n\ndef utc_now': ',\n    {"id": "openml_23517_numerai28_6", "openml_id": 23517, "name": "numerai28.6", "target": "attribute_21", "domain": "financial benchmark", "source_note": "OpenML versioned public benchmark; no era/time field is admitted as a feature in this screen."},\n    {"id": "openml_40701_churn", "openml_id": 40701, "name": "churn", "target": "class", "domain": "customer churn", "source_note": "OpenML versioned public benchmark."},\n    {"id": "openml_41143_jasmine", "openml_id": 41143, "name": "jasmine", "target": "class", "domain": "automl benchmark", "source_note": "OpenML versioned AutoML benchmark."}\n]\n\n\ndef utc_now',
}
for broken, fixed in replacements.items():
    if source.count(broken) != 1:
        raise RuntimeError(f"Expected exactly one in-memory correction site: {broken[:60]!r}")
    source = source.replace(broken, fixed)
namespace = {"__name__": "__main__", "__file__": str(source_path)}
exec(compile(source, str(source_path), "exec"), namespace)
