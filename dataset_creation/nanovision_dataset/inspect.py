from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from nanovision_dataset.grayscale import validate_frames
from nanovision_dataset.writer import load_bundle, load_manifest


def audit_run(run_dir: Path | str) -> dict[str, Any]:
    manifest = load_manifest(run_dir)
    bundle = load_bundle(run_dir)
    frames = bundle["frames"]
    validate_frames(frames, expected_shape=tuple(manifest["frame_shape"]))
    _validate_aligned_metadata(bundle)

    games = bundle["games"]
    per_game = {
        str(game): int((games == game).sum())
        for game in sorted(set(games.tolist()))
    }
    audit = {
        "ok": True,
        "frame_shape": [int(frames.shape[1]), int(frames.shape[2])],
        "dtype": str(frames.dtype),
        "value_min": float(frames.min()),
        "value_max": float(frames.max()),
        "frame_count": int(frames.shape[0]),
        "episode_count": _episode_count(bundle),
        "seed_min": int(bundle["seeds"].min()),
        "seed_max": int(bundle["seeds"].max()),
        "games": sorted(per_game.keys()),
        "per_game_frame_count": per_game,
        "per_game": _per_game_summary(bundle),
    }
    _validate_manifest_matches(manifest, audit)
    return audit


def export_snippet(run_dir: Path | str, output_path: Path | str, max_frames: int = 64, scale: int = 16) -> Path:
    if max_frames < 1:
        raise ValueError("max_frames must be positive")
    images = _load_frame_images(run_dir, max_frames=max_frames, scale=scale)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(output, save_all=True, append_images=images[1:], duration=100, loop=0)
    return output


def export_contact_sheet(
    run_dir: Path | str,
    output_path: Path | str,
    max_frames: int = 64,
    columns: int = 8,
    scale: int = 16,
) -> Path:
    if max_frames < 1:
        raise ValueError("max_frames must be positive")
    if columns < 1:
        raise ValueError("columns must be positive")
    images = _load_frame_images(run_dir, max_frames=max_frames, scale=scale)
    width, height = images[0].size
    rows = int(np.ceil(len(images) / columns))
    sheet = Image.new("L", (columns * width, rows * height), color=0)
    for index, image in enumerate(images):
        x = (index % columns) * width
        y = (index // columns) * height
        sheet.paste(image, (x, y))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)
    return output


def export_html_viewer(run_dir: Path | str, output_path: Path | str, max_frames: int = 512) -> Path:
    if max_frames < 1:
        raise ValueError("max_frames must be positive")
    manifest = load_manifest(run_dir)
    bundle = load_bundle(run_dir)
    frames = bundle["frames"][:max_frames]
    validate_frames(frames)
    frame_payload = (np.clip(frames, 0.0, 1.0) * 255).astype(np.uint8).reshape((frames.shape[0], -1)).tolist()
    payload = {
        "manifest": manifest,
        "frameShape": [int(frames.shape[1]), int(frames.shape[2])],
        "frames": frame_payload,
        "metadata": {
            "games": bundle["games"][:max_frames].astype(str).tolist(),
            "episodes": bundle["episodes"][:max_frames].astype(int).tolist(),
            "timesteps": bundle["timesteps"][:max_frames].astype(int).tolist(),
            "actions": bundle["actions"][:max_frames].astype(int).tolist(),
            "rewards": bundle["rewards"][:max_frames].astype(float).tolist(),
            "terminals": bundle["terminals"][:max_frames].astype(bool).tolist(),
        },
    }
    html = _render_html_viewer(payload)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    return output


def _load_frame_images(run_dir: Path | str, max_frames: int, scale: int) -> list[Image.Image]:
    if scale < 1:
        raise ValueError("scale must be positive")
    bundle = load_bundle(run_dir)
    frames = bundle["frames"][:max_frames]
    validate_frames(frames)
    return [_frame_to_image(frame, scale=scale) for frame in frames]


def _frame_to_image(frame: np.ndarray, scale: int) -> Image.Image:
    pixels = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
    image = Image.fromarray(pixels, mode="L")
    if scale != 1:
        image = image.resize((image.width * scale, image.height * scale), resample=Image.Resampling.NEAREST)
    return image


def _render_html_viewer(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>NanoVision Frame Viewer</title>
  <style>
    :root {{
      color-scheme: dark;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #101114;
      color: #f3f4f6;
    }}
    body {{
      margin: 0;
      min-height: 100vh;
      display: grid;
      grid-template-rows: auto 1fr auto;
    }}
    header, footer {{
      padding: 12px 16px;
      background: #181a1f;
      border-color: #2a2d35;
    }}
    header {{
      border-bottom: 1px solid #2a2d35;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      flex-wrap: wrap;
    }}
    h1 {{
      font-size: 16px;
      margin: 0;
      font-weight: 650;
    }}
    .meta {{
      color: #c4c8d0;
      font-size: 13px;
    }}
    main {{
      display: grid;
      place-items: center;
      padding: 16px;
    }}
    canvas {{
      image-rendering: pixelated;
      image-rendering: crisp-edges;
      width: min(92vw, 76vh);
      height: min(92vw, 76vh);
      background: #000;
      border: 1px solid #3a3f4b;
    }}
    footer {{
      border-top: 1px solid #2a2d35;
      display: grid;
      gap: 10px;
    }}
    .controls {{
      display: flex;
      gap: 8px;
      align-items: center;
      flex-wrap: wrap;
    }}
    button {{
      min-height: 36px;
      border: 1px solid #4b5563;
      background: #252a33;
      color: #f9fafb;
      padding: 0 12px;
      font-size: 14px;
    }}
    button:active {{
      background: #374151;
    }}
    input[type="range"] {{
      flex: 1;
      min-width: 160px;
    }}
    label {{
      display: inline-flex;
      gap: 6px;
      align-items: center;
      color: #d1d5db;
      font-size: 13px;
    }}
    output {{
      min-width: 92px;
      font-variant-numeric: tabular-nums;
      color: #e5e7eb;
    }}
  </style>
</head>
<body>
  <header>
    <h1>NanoVision Frame Viewer</h1>
    <div class="meta" id="meta"></div>
  </header>
  <main>
    <canvas id="frame"></canvas>
  </main>
  <footer>
    <div class="controls">
      <button id="play">Play</button>
      <button id="prev" aria-label="Previous frame">Prev</button>
      <button id="next" aria-label="Next frame">Next</button>
      <input id="scrub" type="range" min="0" value="0">
      <output id="index"></output>
    </div>
    <div class="controls">
      <label>FPS <input id="fps" type="range" min="1" max="30" value="10"></label>
      <output id="fpsValue">10</output>
      <span class="meta" id="stepMeta"></span>
    </div>
  </footer>
  <script>
    const data = {data};
    const canvas = document.getElementById("frame");
    const ctx = canvas.getContext("2d");
    const [height, width] = data.frameShape;
    canvas.width = width;
    canvas.height = height;
    const image = ctx.createImageData(width, height);
    const scrub = document.getElementById("scrub");
    const indexOut = document.getElementById("index");
    const meta = document.getElementById("meta");
    const stepMeta = document.getElementById("stepMeta");
    const fps = document.getElementById("fps");
    const fpsValue = document.getElementById("fpsValue");
    const play = document.getElementById("play");
    let frameIndex = 0;
    let timer = null;

    scrub.max = String(data.frames.length - 1);
    meta.textContent = `${{data.frames.length}} frames | ${{data.manifest.games.join(", ")}} | ${{width}}x${{height}}`;

    function draw(index) {{
      frameIndex = Math.max(0, Math.min(data.frames.length - 1, index));
      const frame = data.frames[frameIndex];
      for (let i = 0; i < frame.length; i++) {{
        const value = frame[i];
        const offset = i * 4;
        image.data[offset] = value;
        image.data[offset + 1] = value;
        image.data[offset + 2] = value;
        image.data[offset + 3] = 255;
      }}
      ctx.putImageData(image, 0, 0);
      scrub.value = String(frameIndex);
      indexOut.value = `${{frameIndex + 1}} / ${{data.frames.length}}`;
      const m = data.metadata;
      stepMeta.textContent = `${{m.games[frameIndex]}} episode ${{m.episodes[frameIndex]}}, t=${{m.timesteps[frameIndex]}}, action=${{m.actions[frameIndex]}}, reward=${{m.rewards[frameIndex]}}, done=${{m.terminals[frameIndex]}}`;
    }}

    function stop() {{
      if (timer) {{
        window.clearInterval(timer);
        timer = null;
      }}
      play.textContent = "Play";
    }}

    function start() {{
      stop();
      play.textContent = "Pause";
      timer = window.setInterval(() => draw((frameIndex + 1) % data.frames.length), 1000 / Number(fps.value));
    }}

    play.addEventListener("click", () => timer ? stop() : start());
    document.getElementById("prev").addEventListener("click", () => {{ stop(); draw(frameIndex - 1); }});
    document.getElementById("next").addEventListener("click", () => {{ stop(); draw(frameIndex + 1); }});
    scrub.addEventListener("input", () => {{ stop(); draw(Number(scrub.value)); }});
    fps.addEventListener("input", () => {{
      fpsValue.value = fps.value;
      if (timer) start();
    }});
    draw(0);
  </script>
</body>
</html>
"""


def _validate_aligned_metadata(bundle: dict[str, np.ndarray]) -> None:
    required = ["frames", "games", "episodes", "timesteps", "seeds", "actions", "rewards", "terminals"]
    missing = [key for key in required if key not in bundle]
    if missing:
        raise ValueError(f"bundle missing required arrays: {', '.join(missing)}")
    lengths = {key: len(bundle[key]) for key in required}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"bundle arrays have inconsistent lengths: {lengths}")


def _validate_manifest_matches(manifest: dict[str, Any], audit: dict[str, Any]) -> None:
    checks = {
        "bundles": ["frames.npz"],
        "dtype": audit["dtype"],
        "episode_count": audit["episode_count"],
        "frame_count": audit["frame_count"],
        "frame_shape": audit["frame_shape"],
        "games": audit["games"],
        "seed_min": audit["seed_min"],
        "seed_max": audit["seed_max"],
        "value_min": audit["value_min"],
        "value_max": audit["value_max"],
        "per_game": audit["per_game"],
    }
    for key, actual in checks.items():
        if manifest.get(key) != actual:
            raise ValueError(f"manifest {key}={manifest.get(key)!r} does not match bundle {actual!r}")


def _episode_count(bundle: dict[str, np.ndarray]) -> int:
    identities = zip(
        bundle["games"].astype(str).tolist(),
        bundle["episodes"].astype(int).tolist(),
        bundle["seeds"].astype(int).tolist(),
        strict=True,
    )
    return len(set(identities))


def _per_game_summary(bundle: dict[str, np.ndarray]) -> dict[str, dict[str, int]]:
    games = bundle["games"].astype(str)
    summary: dict[str, dict[str, int]] = {}
    for game in sorted(set(games.tolist())):
        mask = games == game
        identities = zip(
            games[mask].tolist(),
            bundle["episodes"][mask].astype(int).tolist(),
            bundle["seeds"][mask].astype(int).tolist(),
            strict=True,
        )
        summary[game] = {
            "episode_count": len(set(identities)),
            "frame_count": int(mask.sum()),
        }
    return summary
