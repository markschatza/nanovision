import json

import numpy as np
import pytest

from nanovision_dataset.minatar_source import EpisodeRecord
from nanovision_dataset.writer import load_bundle, load_manifest, write_run


def _record(game: str = "breakout", episode: int = 0) -> EpisodeRecord:
    return EpisodeRecord(
        game=game,
        episode=episode,
        seed=episode + 10,
        frames=np.zeros((2, 10, 10), dtype=np.float32),
        actions=np.asarray([1, 2], dtype=np.int16),
        rewards=np.asarray([0.0, 1.0], dtype=np.float32),
        terminals=np.asarray([False, True]),
    )


def test_write_run_writes_manifest_and_bundle(tmp_path) -> None:
    artifact = write_run(tmp_path, [_record()], policy_source="random", settings={"episodes": 1})

    assert artifact.frame_count == 2
    assert artifact.episode_count == 1
    assert artifact.bundle_path.exists()
    assert artifact.manifest_path.exists()

    manifest = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))
    assert manifest["frame_count"] == 2
    assert manifest["per_game"]["breakout"]["episode_count"] == 1

    bundle = load_bundle(tmp_path)
    assert bundle["frames"].shape == (2, 10, 10)
    assert bundle["games"].tolist() == ["breakout", "breakout"]


def test_load_manifest_reads_json(tmp_path) -> None:
    write_run(tmp_path, [_record()], policy_source="random", settings={})

    manifest = load_manifest(tmp_path)

    assert manifest["policy_source"] == "random"


def test_write_run_rejects_inconsistent_metadata_lengths(tmp_path) -> None:
    bad = EpisodeRecord(
        game="breakout",
        episode=0,
        seed=1,
        frames=np.zeros((2, 10, 10), dtype=np.float32),
        actions=np.asarray([1], dtype=np.int16),
        rewards=np.asarray([0.0, 1.0], dtype=np.float32),
        terminals=np.asarray([False, True]),
    )

    with pytest.raises(ValueError):
        write_run(tmp_path, [bad], policy_source="random", settings={})


def test_write_run_rejects_empty_records(tmp_path) -> None:
    with pytest.raises(ValueError):
        write_run(tmp_path, [], policy_source="random", settings={})
