# Dataset Creation

This folder contains the first Nano Pixel RL dataset path: MinAtar rollouts converted into saved grayscale frame backlogs.

The v1 artifact is model-facing visual data, not tokenization or training data.
Frames are saved as normalized `0..1` grayscale arrays and inspection helpers read those saved arrays back for audit and visual QA.
Background pixels are `0.0`; active object channels are projected to stable grayscale brightness values so different objects remain visually separable without storing a multi-channel state image.

## Generate A Tiny Backlog

```powershell
uv run nanovision-dataset generate --games breakout --episodes 1 --seed 0 --out artifacts/datasets/smoke
```

Use Pgx pretrained baselines for non-random trajectories:

```powershell
uv run nanovision-dataset generate --games breakout --episodes 1 --seed 0 --policy pgx-baseline --out artifacts/datasets/pgx-smoke
```

Pgx baseline checkpoints download into `artifacts/pgx-baselines` by default.
JAX/XLA compilations are cached in `artifacts/jax-cache` by default.
Override it with `--jax-cache-dir` when running repeated remote generations from another working directory.
Pgx baseline rollouts run as a JIT-compiled `jax.lax.scan`, then copy completed episode arrays back once for writing.
Pgx runs use Pgx's MinAtar action indexing; do not mix action labels with native MinAtar random-policy runs without checking the manifest `action_space`.
That directory is rebuildable and ignored by version control.

For CUDA rollout generation, use a separate Linux remote environment rather than the local Windows lock.
The Pgx baseline checkpoints currently require JAX `0.4.35`; newer CUDA 13 JAX releases can see the GPU but cannot load the baseline pickle.
On the Aesop host, the verified setup used `jax[cuda12]==0.4.35`, `pgx==2.6.0`, and a venv-local `nvidia/cuda_nvcc/__init__.py` package-file workaround before `jax.devices()` reported `CudaDevice(id=0)`.

## Audit A Backlog

```powershell
uv run nanovision-dataset audit artifacts/datasets/smoke
```

## Export An HTML Viewer

```powershell
uv run nanovision-dataset export-html artifacts/datasets/smoke --out artifacts/datasets/smoke/viewer.html
```

The HTML viewer is self-contained, so it can be served with a local or phone preview server.

## Export A GIF Snippet

```powershell
uv run nanovision-dataset export-snippet artifacts/datasets/smoke --out artifacts/datasets/smoke/snippet.gif
```

## Export A Contact Sheet

```powershell
uv run nanovision-dataset export-contact-sheet artifacts/datasets/smoke --out artifacts/datasets/smoke/contact_sheet.png
```

## Artifact Shape

- `manifest.json` records games, seeds, policy source, frame counts, episode counts, frame shape, numeric range, and bundle paths.
- `frames.npz` stores `frames` as `N x 10 x 10` float32 grayscale values plus aligned metadata arrays.

Generated dataset outputs under `artifacts/` are rebuildable and intentionally ignored by version control.
