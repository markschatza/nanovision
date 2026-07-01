from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from nanovision_dataset.grayscale import validate_frames
from nanovision_dataset.minatar_source import EpisodeRecord


BUNDLE_NAME = "frames.npz"
MANIFEST_NAME = "manifest.json"


@dataclass(frozen=True)
class RunArtifact:
    run_dir: Path
    manifest_path: Path
    bundle_path: Path
    frame_count: int
    episode_count: int


def write_run(
    output_dir: Path | str,
    records: Iterable[EpisodeRecord],
    policy_source: str,
    settings: dict[str, Any],
) -> RunArtifact:
    run_dir = Path(output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    records = list(records)
    if not records:
        raise ValueError("cannot write an empty run")

    arrays = flatten_records(records)
    validate_frames(arrays["frames"])

    bundle_path = run_dir / BUNDLE_NAME
    np.savez_compressed(bundle_path, **arrays)

    manifest = build_manifest(records, arrays, policy_source, settings, bundle_path.name)
    manifest_path = run_dir / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return RunArtifact(
        run_dir=run_dir,
        manifest_path=manifest_path,
        bundle_path=bundle_path,
        frame_count=int(arrays["frames"].shape[0]),
        episode_count=len(records),
    )


def flatten_records(records: list[EpisodeRecord]) -> dict[str, np.ndarray]:
    frames: list[np.ndarray] = []
    games: list[str] = []
    episodes: list[int] = []
    timesteps: list[int] = []
    seeds: list[int] = []
    actions: list[int] = []
    rewards: list[float] = []
    terminals: list[bool] = []

    for record in records:
        _validate_record(record)
        for timestep in range(len(record.frames)):
            frames.append(record.frames[timestep])
            games.append(record.game)
            episodes.append(record.episode)
            timesteps.append(timestep)
            seeds.append(record.seed)
            actions.append(int(record.actions[timestep]))
            rewards.append(float(record.rewards[timestep]))
            terminals.append(bool(record.terminals[timestep]))

    return {
        "frames": np.asarray(frames, dtype=np.float32),
        "games": np.asarray(games),
        "episodes": np.asarray(episodes, dtype=np.int32),
        "timesteps": np.asarray(timesteps, dtype=np.int32),
        "seeds": np.asarray(seeds, dtype=np.int64),
        "actions": np.asarray(actions, dtype=np.int16),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "terminals": np.asarray(terminals, dtype=bool),
    }


def build_manifest(
    records: list[EpisodeRecord],
    arrays: dict[str, np.ndarray],
    policy_source: str,
    settings: dict[str, Any],
    bundle_name: str,
) -> dict[str, Any]:
    frames = arrays["frames"]
    per_game: dict[str, dict[str, int]] = {}
    for game in sorted(set(arrays["games"].tolist())):
        mask = arrays["games"] == game
        game_episode_count = sum(1 for record in records if record.game == game)
        per_game[game] = {
            "frame_count": int(mask.sum()),
            "episode_count": game_episode_count,
        }

    return {
        "format_version": 1,
        "policy_source": policy_source,
        "settings": settings,
        "games": sorted(per_game.keys()),
        "seed_min": int(arrays["seeds"].min()),
        "seed_max": int(arrays["seeds"].max()),
        "episode_count": len(records),
        "frame_count": int(frames.shape[0]),
        "frame_shape": [int(frames.shape[1]), int(frames.shape[2])],
        "dtype": str(frames.dtype),
        "value_min": float(frames.min()),
        "value_max": float(frames.max()),
        "bundles": [bundle_name],
        "per_game": per_game,
    }


def load_manifest(run_dir: Path | str) -> dict[str, Any]:
    manifest_path = Path(run_dir) / MANIFEST_NAME
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def load_bundle(run_dir: Path | str) -> dict[str, np.ndarray]:
    bundle_path = Path(run_dir) / BUNDLE_NAME
    with np.load(bundle_path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _validate_record(record: EpisodeRecord) -> None:
    lengths = {
        "frames": len(record.frames),
        "actions": len(record.actions),
        "rewards": len(record.rewards),
        "terminals": len(record.terminals),
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(f"episode metadata lengths differ: {lengths}")
    if lengths["frames"] == 0:
        raise ValueError("episode contains no frames")
    validate_frames(record.frames)


def artifact_to_dict(artifact: RunArtifact) -> dict[str, Any]:
    data = asdict(artifact)
    return {key: str(value) if isinstance(value, Path) else value for key, value in data.items()}
