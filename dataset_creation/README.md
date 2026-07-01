# Dataset Creation

This folder contains the first Nano Pixel RL dataset path: MinAtar rollouts converted into saved grayscale frame backlogs.

The v1 artifact is model-facing visual data, not tokenization or training data.
Frames are saved as normalized `0..1` grayscale arrays and inspection helpers read those saved arrays back for audit and visual QA.

## Generate A Tiny Backlog

```powershell
uv run nanovision-dataset generate --games breakout --episodes 1 --seed 0 --out artifacts/datasets/smoke
```

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
