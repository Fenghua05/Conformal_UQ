# Conformal UQ under controlled minority calibration scarcity

This repository contains the public reproducibility package for a study of label-conditional conformal prediction when minority-class calibration data are scarce.

The scientific analysis is frozen at the released result grid. The repository contains code, tests, locked protocols, configuration schemas, derived aggregate results, figure/table sources, and the latest editable manuscript materials.

## Repository contents

- `src/` — analysis and reproducibility code.
- `tests/` — unit and integrity tests.
- `protocols/` — the released protocol and dataset lock.
- `configs/` — the selected run manifests and schemas.
- `results/` — aggregate derived results and quality-control summaries.
- `manuscript/figures/` and `manuscript/tables/` — figure/table sources and exports.
- `manuscript/manuscript_polished_v1.5.md` — the latest manuscript source.
- `manuscript/manuscript_polished_v1.5.docx` — the latest editable Word manuscript.
- `cloud/tabpfn_stage08/` — user-operated TabPFN cloud-run helper and guide.

## Data and models

Raw datasets, downloaded archives, probability caches, trained checkpoints, and local execution logs are intentionally not stored here. The analysis uses the locked public OpenML records specified by the protocol: dataset IDs 3, 24, 1486, 1489, 1590, 4534, 23512, and 23517. Retrieve them from OpenML according to `protocols/dataset_lock_v1.0.md` before running data-dependent workflows.

The TabPFN checkpoint is also not redistributed. The cloud helper documents the user-operated execution path and records the expected model provenance.

## Reproduction

After installing the dependencies appropriate to the selected workflow:

```text
python -m unittest discover -s tests -v
```

The scripts under `src/` document the staged workflow. Data-dependent runs expect local raw data and generated caches in the ignored workspace paths described by the protocols and configuration files.

## Manuscript and results

The released aggregate result table contains cell-level metrics and provenance fields, not sample-level records or prediction caches. Figures can be regenerated from the public result/source files using the scripts documented in `src/` and the figure metadata under `manuscript/figures/`.

The manuscript source and editable Word derivative are provided for review and reuse of the released research record. A public repository URL and archival DOI remain author-supplied publication metadata.

## License

No open-source license has been selected for this release yet. Permission is required for reuse beyond viewing, citation, and evaluation of the published research record.
