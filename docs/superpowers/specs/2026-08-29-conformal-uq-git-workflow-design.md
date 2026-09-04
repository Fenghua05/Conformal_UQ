# Conformal UQ Git Workflow Design

## Goal

Create a lightweight, auditable Git workflow for this repository that keeps `main` aligned with the latest accepted project stage and backs it up to the configured GitHub repository.

## Approved approach

Use direct, atomic stage commits on `main` rather than long-lived stage branches or pull-request gates.

- A completed and QC-verified stage is committed once with a message such as `stage-00: initialize project controls`.
- `main` contains only the latest accepted stage state. Work that has not passed its stage gate is not committed as a completed-stage checkpoint.
- `origin` will point to `https://github.com/Fenghua05/Conformal_UQ.git`; after the first successful push, `main` tracks `origin/main`.
- No remote configuration, push, or pull occurs before the workflow files and safeguards below are in place.

## Repository contents

Tracked:

- Authority protocols, project state, decision log, handoffs, documentation, source code, tests, configurations, environment-lock metadata, small machine-readable registries, manifests, QC reports, and manuscript source.

Ignored by default:

- Raw/downloaded data; processed caches; probability caches; trained models/checkpoints; local logs; temporary files; notebook checkpoints; Python build/virtual-environment files; editor/OS metadata; generated large result tables and render products.

An approved small report, figure source, or final result manifest may be force-added only when its provenance and SHA-256 are recorded. Raw data, credentials, secrets, and private tokens must never be force-added.

## Operational rules

1. At each stage boundary, inspect `git status`, run the stage-appropriate QC, update `PROJECT_STATE.md`, `DECISION_LOG.md` when needed, and produce the required handoff before committing.
2. The commit message format is `stage-XX: <accepted scope>` for accepted stages; `fix(stage-XX): <verified correction>` for a scoped correction; `docs: <documentation-only change>` for non-protocol documentation.
3. Before pushing, verify the staged file list excludes ignored and sensitive outputs. Push only after a successful local commit.
4. Before starting a new stage, fetch/pull only when a remote exists and the local tree is clean. Any remote divergence is reviewed before merging; no force push, history rewrite, or silent conflict resolution.
5. A protocol change requires an approved `DECISION_LOG.md` entry and must identify invalidated downstream artifacts in `PROJECT_STATE.md` before it is committed.
6. The repository uses the existing Git author identity already configured on the machine. Credential handling remains delegated to GitHub/Git Credential Manager; no token is stored in repository files.

## Initial implementation sequence

1. Initialize the repository with branch `main`.
2. Add a root `.gitignore` implementing the approved default exclusions.
3. Add a concise `README.md` describing recovery inputs and the Stage 00 / Stage 01 boundary.
4. Record the Git workflow decision in the decision log and state file.
5. Create the initial Stage 00 checkpoint commit.
6. Add `origin`, inspect the remote safely, and push `main` without force.
7. Verify local/remote tracking, clean working tree, and the absence of unintended tracked files.

## Error handling and validation

- If the remote repository already has a divergent history, stop before merging or overwriting it and report the state for a user decision.
- If authentication is unavailable, preserve the completed local commit and report the exact next action; do not place credentials in files or command history.
- If a sensitive or generated file appears in the staged list, remove it from staging, strengthen `.gitignore` if appropriate, and recheck before committing.
- Completion evidence requires: a local `main` repository, valid `origin` URL, successful non-force push (or a documented authentication/divergence blocker), `main` tracking status, clean tree, and an updated state/decision/handoff record.

## Scope exclusions

This workflow does not create a GitHub repository, change GitHub repository settings, add collaborators, set branch protection, install software, download data, or begin Stage 01.
