# Public release manifest

Release scope: public reproducibility package, prepared 2026-09-04.

## Included

- Analysis source and tests: `src/`, `tests/`.
- Selected cloud helper: `cloud/tabpfn_stage08/`.
- Released protocols: `protocols/protocol_v1.1.md`, `protocols/dataset_lock_v1.0.md`.
- Selected reproducibility configurations and schemas under `configs/`.
- Aggregate derived results under `results/`.
- Figure and table sources/exports under `manuscript/figures/` and `manuscript/tables/`.
- Latest manuscript source and editable Word derivative under `manuscript/`.
- Root dependency lock and public documentation.

## Excluded by design

Raw data and downloaded archives; generated artifacts; probability caches; model checkpoints; local logs; temporary files; internal project state; handoff and decision records; literature working files; superseded delivery versions; and editor/OS metadata.

No credentials, tokens, private keys, or environment files are part of the release set.

## Path correction

The public manuscript copy is placed at `manuscript/manuscript_polished_v1.5.md` so its figure links resolve against `manuscript/figures/`. The source copy in the internal delivery workspace is not published.

The v1.5 PDF and PDF-matched Word derivative referenced by internal state were not present at their declared paths during this audit, so they are not included. The confirmed v1.5 Markdown and editable Word files are the published manuscript materials.
