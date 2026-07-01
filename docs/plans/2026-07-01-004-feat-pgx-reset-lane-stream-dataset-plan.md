---
title: Pgx Reset Lane Stream Dataset - Plan
artifact_contract: ce-unified-plan/v1
artifact_readiness: implementation-ready
execution: code
product_contract_source: ce-plan-bootstrap
created: 2026-07-01
---

# Pgx Reset Lane Stream Dataset - Plan

## Goal Capsule

- **Objective:** Add a fixed-horizon Pgx generation mode where each JAX lane resets immediately after terminal and host post-processing emits completed episodes.
- **Product authority:** User direction on 2026-07-01 to target large datasets with `100000` steps across `16` lanes and eventual `10 GB` per game.
- **Execution profile:** Dataset-generation performance work with remote CUDA validation and ignored generated artifacts.
- **Open blockers:** None for the generator. The full 50 GB dataset should only run after confirming disk budget and chunk sizing.

## Product Contract

### Requirements

- R1. The generator must support fixed step budgets across parallel lanes, e.g. `100000` steps x `16` lanes.
- R2. A lane must reset to a fresh game on the step after terminal, rather than idling until every lane terminates.
- R3. Saved output must still be grouped as variable-length episode records so existing writer/audit/export tools continue to work.
- R4. The stream generator must be exposed as a separate CLI path from complete-episode generation.
- R5. Incomplete tail episodes should be dropped by default, with an option to include them for frame-maximizing dataset runs.
- R6. Generated datasets, caches, and baselines must remain ignored by source control.

### Acceptance Evidence

- AE1. Unit tests cover reset-stream post-processing into complete episodes.
- AE2. A local smoke run writes and audits a stream-generated artifact.
- AE3. A remote Aesop smoke run validates CUDA/JAX execution for reset lanes.
- AE4. Dataset target math is documented for the 10 GB per game / 50 GB total goal.

## Implementation Units

### U1. Reset-Lane Stream Rollout

- **Goal:** Add a JAX scanned lane rollout that resets terminal lanes and emits frame/action/reward/terminal/episode metadata.
- **Files:** `dataset_creation/nanovision_dataset/pgx_source.py`, `tests/dataset_creation/test_pgx_source.py`
- **Approach:** Use batched Pgx state, vectorized model/action selection, vectorized `env.step`, and per-lane reset state selection. Carry lane episode IDs and seeds through the scan.
- **Verification:** `uv run pytest tests/dataset_creation/test_pgx_source.py`

### U2. Stream CLI

- **Goal:** Expose the reset-lane path as `generate-stream`.
- **Files:** `dataset_creation/nanovision_dataset/cli.py`, `tests/dataset_creation/test_cli.py`, `dataset_creation/README.md`
- **Approach:** Add `--steps`, `--lanes`, `--include-incomplete`, cache/baseline options, and write the existing artifact format through `write_run`.
- **Verification:** `uv run pytest tests/dataset_creation/test_cli.py`

### U3. Remote Pilot

- **Goal:** Prove the target shape with a small remote run before attempting large dataset chunks.
- **Files:** ignored `artifacts/datasets/`
- **Approach:** Run at least one representative game with reset lanes on Aesop and audit the artifact locally or remotely.
- **Verification:** Remote output reports episode count, frame count, elapsed time, and audit success.

## Verification Contract

| Gate | Command or Evidence | Purpose |
| --- | --- | --- |
| Local tests | `uv run pytest tests/dataset_creation` | Protect all dataset helpers. |
| Local stream smoke | `uv run nanovision-dataset generate-stream --games breakout --steps 64 --lanes 4 ...` | Catch stream shape and writer errors. |
| Remote stream pilot | Aesop `generate-stream` run | Confirm CUDA/JAX path works for reset lanes. |

## Dataset Target

- A `100000` step x `16` lane run emits up to `1,600,000` frames before dropping incomplete tails.
- Current frame storage is `uint8` grayscale, so raw frames are about `100` bytes each before compression and metadata.
- `10 GB` raw frames per game is roughly `100,000,000` frames, or about sixty-three `100000 x 16` chunks per game before compression.
- Full `50 GB` raw target across five games is roughly `500,000,000` frames.
- The remote machine currently has enough headroom for the target, but large runs should be chunked per game and audited after each chunk.

## Definition Of Done

- `generate-stream` can produce reset-lane Pgx datasets.
- Stream outputs are saved as existing episode records.
- Tests and local smoke pass.
- Remote smoke passes on Aesop.
- Changes are committed and pushed.
