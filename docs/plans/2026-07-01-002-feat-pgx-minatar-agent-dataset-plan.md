---
title: Pgx MinAtar Agent Dataset - Plan
artifact_contract: ce-unified-plan/v1
artifact_readiness: implementation-ready
execution: code
product_contract_source: ce-plan-bootstrap
created: 2026-07-01
---

# Pgx MinAtar Agent Dataset - Plan

## Goal Capsule

- **Objective:** Add a trained-agent rollout source for the existing MinAtar frame dataset tools using Pgx pretrained baselines.
- **Product authority:** User direction on 2026-07-01 to stop random-only rollouts and connect to existing pretrained agents, starting with a small set of games.
- **Execution profile:** Small Python integration plus remote CUDA/JAX validation, with generated datasets kept out of source control.
- **Open blockers:** None for implementation. Heavy rollout generation should run on the Aesop CUDA machine because local JAX generation already caused an OOM.

## Product Contract

### Requirements

- R1. Dataset generation must support a `pgx-baseline` policy option alongside the existing random MinAtar source.
- R2. The Pgx source must produce the same saved frame contract as the current writer: normalized grayscale frames, actions, rewards, terminals, and manifest metadata.
- R3. Version 1 should target three games for trained-agent samples before expanding to all five games.
- R4. Generated checkpoints, datasets, previews, credentials, and virtual environments must remain out of source control.
- R5. The workflow must include a remote CUDA/JAX path for rollout generation so local machine memory pressure does not block sample creation.
- R6. The phone-preview HTML viewer should be able to inspect generated Pgx sample runs after remote artifacts are copied back or regenerated locally from saved frames.
- R7. Pgx generation must support a persistent JAX/XLA compilation cache so repeated remote runs do not pay full compile cost every time.

### Scope Boundaries

- No model training work is included.
- No control-policy changes are included beyond using Pgx pretrained baseline logits.
- No large dataset is required for this pass; the deliverable is a verified path and small inspectable samples.
- Remote secrets and connection details stay in ignored local files or existing private tooling, never in committed docs or source.

### Acceptance Evidence

- AE1. A local focused test confirms illegal Pgx actions are masked before selecting an action.
- AE2. `nanovision-dataset generate --policy pgx-baseline` can create at least one small valid run.
- AE3. A remote command on the CUDA/JAX machine reports the available NVIDIA GPU and JAX devices before remote rollout generation is trusted.
- AE4. At least three Pgx baseline game samples can be generated or attempted on the remote machine with failures recorded concretely.
- AE5. Local audit and HTML export can inspect at least one Pgx-generated sample run.
- AE6. A repeated remote Pgx generation using the same cache directory leaves persistent cache files on disk and can reuse them on the second run.

## Planning Contract

### Key Technical Decisions

- **KTD1. Keep Pgx behind a source adapter.** The CLI should choose between `MinAtarSource` and `PgxBaselineSource`; writer, audit, and viewer code should not care which policy produced the episode records.
- **KTD2. Use greedy legal-action selection for v1.** Deterministic argmax over legal baseline logits is enough for dataset creation and avoids sampling variability while we prove the pipeline.
- **KTD3. Run heavy JAX/Pgx generation remotely.** Local smoke tests can validate interfaces, but sample generation should happen on the Aesop CUDA host after a JAX device probe.
- **KTD4. Treat CUDA JAX as an environment concern.** The repo can pin the Pgx-compatible JAX API version, but CUDA-enabled `jaxlib` installation may need remote-specific setup rather than a source-control dependency that breaks local CPU users.
- **KTD5. Enable persistent JAX compilation cache by default for Pgx.** Use an ignored local cache path and keep the directory configurable for remote runs.

### Risks

- **Pgx/JAX version drift:** Pgx baselines may fail on newer JAX versions, so the tested version must be pinned or documented.
- **CUDA wheel mismatch:** A remote environment can silently fall back to CPU if the wrong JAX package is installed; validation must print `jax.devices()`.
- **Action-space mismatch:** Pgx MinAtar action indices may not match native MinAtar action counts; manifests should identify the policy/source clearly enough to avoid mixing semantics blindly.
- **Remote artifact size:** Only small samples should be copied back for preview until the dataset contract stabilizes.
- **Cache trust boundary:** JAX compilation caches are executable trust surfaces, so cache directories must stay local to trusted users and out of source control.

## Implementation Units

### U1. Pgx Source Adapter

- **Goal:** Add a Pgx baseline rollout source that returns existing `EpisodeRecord` objects.
- **Files:** `dataset_creation/nanovision_dataset/pgx_source.py`, `dataset_creation/nanovision_dataset/cli.py`, `tests/dataset_creation/test_pgx_source.py`
- **Approach:** Load `pgx`, `jax`, and `jax.numpy` lazily; create `pgx.make("minatar-<game>")`; load the matching pretrained baseline; project observations through the existing grayscale helper; select legal actions from baseline logits.
- **Test Scenarios:** legal-action masking chooses the best legal action; invalid `max_steps` and `episodes` fail consistently with the random source; a tiny Pgx smoke run can be executed when dependencies are installed.
- **Verification:** `uv run pytest tests/dataset_creation/test_pgx_source.py`

### U2. CLI And Metadata

- **Goal:** Expose Pgx baseline generation without changing existing random-generation commands.
- **Files:** `dataset_creation/nanovision_dataset/cli.py`, `dataset_creation/README.md`, `tests/dataset_creation/test_cli.py`
- **Approach:** Add `--policy random|pgx-baseline`, `--baseline-dir`, and manifest settings that identify policy, baseline directory, and action-space caveats.
- **Test Scenarios:** generate help lists the policy and cache flags; random generation still works; Pgx settings appear in manifests for Pgx runs.
- **Verification:** `uv run pytest tests/dataset_creation/test_cli.py`

### U3. Remote CUDA/JAX Runner

- **Goal:** Prove and document a safe remote execution path for small Pgx sample generation.
- **Files:** `dataset_creation/README.md`, optional helper under `dataset_creation/scripts/`
- **Approach:** Reuse the existing private remote connection pattern without committing credentials; create or sync a Nanovision checkout on Aesop; install a Pgx-compatible JAX environment; print `nvidia-smi` and `jax.devices()` before generation.
- **Test Scenarios:** remote command can reach the host; remote Python imports `jax` and `pgx`; remote JAX reports a CUDA device or the failure is captured before generation.
- **Verification:** remote probe output includes GPU name and JAX device list.

### U4. Sample Generation And Preview

- **Goal:** Produce small trained-agent samples for three games and make them inspectable in the HTML viewer.
- **Files:** ignored `artifacts/datasets/phone-preview/`, no committed generated data
- **Approach:** Generate bounded episodes for three games with `--policy pgx-baseline`; audit the saved runs; export HTML viewers; update only ignored preview index artifacts if useful.
- **Test Scenarios:** each generated sample has `frames.npz` and `manifest.json`; audit reports valid frame shape and numeric range; HTML viewer opens from the phone preview server.
- **Verification:** `nanovision-dataset audit <run-dir>` succeeds for each generated sample and at least one HTML export exists.

## Verification Contract

| Gate | Command or Evidence | Purpose |
| --- | --- | --- |
| Unit tests | `uv run pytest tests/dataset_creation` | Protect existing dataset behavior and Pgx helper behavior. |
| CLI help | `uv run nanovision-dataset generate --help` | Confirm public CLI flags are discoverable. |
| Local tiny smoke | One `pgx-baseline` run with low `--max-steps` if local memory allows | Confirm adapter works before remote scaling. |
| Remote CUDA probe | Aesop command showing `nvidia-smi` and `jax.devices()` | Confirm rollout generation will use the intended GPU path. |
| Remote cache probe | Two repeated Pgx generations with the same `--jax-cache-dir` | Confirm persistent cache files are written and reused. |
| Sample audit | `nanovision-dataset audit` on generated samples | Confirm saved artifacts match the dataset contract. |

## Definition Of Done

- Pgx baseline policy is available through the dataset CLI.
- Focused tests cover the Pgx action-selection behavior and existing tests still pass.
- Documentation explains how to generate Pgx baseline samples and where checkpoint artifacts live.
- Pgx generation enables and records a configurable JAX compilation cache path.
- Remote CUDA/JAX probing has been attempted and the result is reported.
- Three-game Pgx sample generation is completed on the remote machine or the concrete remote blocker is documented.
- No generated datasets, checkpoints, credentials, virtual environments, or cache files are staged.
