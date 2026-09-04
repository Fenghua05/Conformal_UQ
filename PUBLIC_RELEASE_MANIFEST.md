# Public release manifest

Release scope: public reproducibility package, prepared 2026-09-04.

## Included

- Analysis source and tests: `src/`, `tests/`.
- Selected cloud helper: `cloud/tabpfn_stage08/`.
- Released protocols: `protocols/protocol_v1.1.md`, `protocols/dataset_lock_v1.0.md`.
- Selected reproducibility configurations and schemas under `configs/`.
- Aggregate derived results under `results/`.
- Figure and table sources/exports under `manuscript/figures/` and `manuscript/tables/`.
- Latest manuscript source under `manuscript/`.
- Root dependency lock and public documentation.

## Excluded by design

Raw data and downloaded archives; generated artifacts; probability caches; model checkpoints; local logs; temporary files; internal project state; handoff and decision records; literature working files; superseded delivery versions; and editor/OS metadata.

No credentials, tokens, private keys, or environment files are part of the release set.

## Path correction

The public manuscript copy is placed at `manuscript/manuscript_polished_v1.5.md` so its figure links resolve against `manuscript/figures/`. The source copy in the internal delivery workspace is not published.

The v1.5 PDF and Word derivatives are not included because they are not final versions. Only the confirmed v1.5 Markdown source is published as manuscript text.
