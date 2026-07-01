---
title: Batched Pgx Rollout Dataset - Plan
artifact_contract: ce-unified-plan/v1
artifact_readiness: implementation-ready
execution: code
product_contract_source: ce-plan-bootstrap
created: 2026-07-01
---

# Batched Pgx Rollout Dataset - Plan

## Goal Capsule

- **Objective:** Make Pgx MinAtar dataset generation efficient enough for large multi-episode datasets by batching rollouts in JAX.
- **Product authority:** User direction on 2026-07-01 after single-episode scanned rollouts still showed too much per-episode overhead.
- **Execution profile:** Python CLI/package change with local tests and remote Aesop throughput verification.
- **Open blockers:** None. Remote CUDA/JAX remains the authoritative performance path.

## Product Contract

### Requirements

- R1. Pgx baseline generation must avoid rebuilding the environment, loading the model, and creating the jitted rollout function for every episode.
- R2. Pgx generation must batch multiple episode seeds for the same game through one JAX call where practical.
- R3. Saved artifacts must keep the existing frame/action/reward/terminal contract and variable-length episode records after terminal trimming.
- R4. The CLI must expose batch size for Pgx generation without changing random-policy behavior.
- R5. Batch generation must keep using the persistent JAX cache path and ignored artifact directories.
- R6. Verification must include a remote throughput benchmark for multi-episode generation on Aesop.

### Scope Boundaries

- No training loop is included.
- No new storage format is required in this pass.
- No cross-game single-JAX-call batching is required because Pgx models and observation channels differ by game.

### Acceptance Evidence

- AE1. Local tests cover batch seed handling, terminal trimming, grayscale parity, and CLI metadata.
- AE2. A small local smoke generation still writes a valid audited artifact.
- AE3. Remote Aesop can generate a multi-episode Pgx batch for at least one game and report materially better throughput than one-process single-episode calls.

## Planning Contract

### Key Technical Decisions

- **KTD1. Cache per-game runtime objects in `PgxBaselineSource`.** Reuse Pgx env, baseline model, and jitted rollout functions across episodes for the same source instance.
- **KTD2. Batch within each game, not across games.** Per-game batching keeps static shapes and model parameters simple while still removing the dominant per-episode overhead.
- **KTD3. Use `jax.vmap` over seed keys.** Keep the scanned single-episode rollout as the primitive, then vectorize it over a batch of PRNG keys.
- **KTD4. Trim on host after one device transfer per batch.** The compiled function returns fixed-length arrays; Python trims each episode at the first terminal before writing existing `EpisodeRecord`s.

## Implementation Units

### U1. Batched Pgx Runtime

- **Goal:** Add per-game cached runtime and vectorized rollout support.
- **Files:** `dataset_creation/nanovision_dataset/pgx_source.py`, `tests/dataset_creation/test_pgx_source.py`
- **Approach:** Add `batch_size`; cache a runtime object keyed by game; build one jitted batched rollout function per game/max_steps; generate seeds in chunks and convert batch outputs into records.
- **Test Scenarios:** episode seeds remain deterministic and unique; batch outputs are trimmed at first terminal; JAX grayscale still matches NumPy grayscale.
- **Verification:** `uv run pytest tests/dataset_creation/test_pgx_source.py`

### U2. CLI And Docs

- **Goal:** Expose batch size and document the large-dataset generation path.
- **Files:** `dataset_creation/nanovision_dataset/cli.py`, `dataset_creation/README.md`, `tests/dataset_creation/test_cli.py`
- **Approach:** Add `--batch-size` for `pgx-baseline`; include it in manifest settings; explain that batching is per game.
- **Test Scenarios:** CLI help lists batch size; Pgx manifest records batch size; random generation still works unchanged.
- **Verification:** `uv run pytest tests/dataset_creation/test_cli.py`

### U3. Remote Throughput Check

- **Goal:** Verify the new path on the CUDA host before treating it as the dataset path.
- **Files:** ignored `artifacts/datasets/`
- **Approach:** Sync source to Aesop; run a multi-episode Pgx generation for representative games; compare frames/sec and wall time against recent single-episode numbers.
- **Verification:** Remote command output reports episode count, frame count, elapsed seconds, and frames/sec.

## Verification Contract

| Gate | Command or Evidence | Purpose |
| --- | --- | --- |
| Local tests | `uv run pytest tests/dataset_creation` | Protect writer, CLI, grayscale, and Pgx helper behavior. |
| Local smoke | `uv run nanovision-dataset generate --games breakout --episodes 2 --max-steps 20 --policy pgx-baseline --batch-size 2 ...` | Catch JIT/batch shape errors. |
| Remote throughput | Aesop multi-episode generation benchmark | Confirm batching improves dataset generation throughput. |

## Definition Of Done

- Pgx generation supports configurable per-game batch size.
- The default Pgx path reuses per-game runtime objects and batches episodes.
- Existing artifact format remains compatible with audit/export tools.
- Local tests and a local smoke pass.
- Remote Aesop benchmark demonstrates the batched path.
- Changes are committed and pushed; generated datasets remain ignored.
