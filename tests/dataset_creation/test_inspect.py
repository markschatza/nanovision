import json

import numpy as np
import pytest

from nanovision_dataset.inspect import audit_run, export_contact_sheet, export_html_viewer, export_snippet
from nanovision_dataset.minatar_source import EpisodeRecord
from nanovision_dataset.writer import write_run


def _write_sample(tmp_path):
    frames = np.zeros((3, 10, 10), dtype=np.float32)
    frames[1, 2, 3] = 1.0
    record = EpisodeRecord(
        game="breakout",
        episode=0,
        seed=3,
        frames=frames,
        actions=np.asarray([0, 1, 2], dtype=np.int16),
        rewards=np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
        terminals=np.asarray([False, False, True]),
    )
    return write_run(tmp_path, [record], policy_source="random", settings={})


def test_audit_reports_saved_bundle_stats(tmp_path) -> None:
    _write_sample(tmp_path)

    audit = audit_run(tmp_path)

    assert audit["ok"] is True
    assert audit["frame_shape"] == [10, 10]
    assert audit["dtype"] == "uint8"
    assert audit["frame_encoding"] == "uint8_0_255"
    assert audit["frame_count"] == 3
    assert audit["per_game_frame_count"] == {"breakout": 3}


def test_export_snippet_and_contact_sheet_read_saved_frames(tmp_path) -> None:
    _write_sample(tmp_path)

    gif = export_snippet(tmp_path, tmp_path / "snippet.gif", max_frames=2, scale=2)
    sheet = export_contact_sheet(tmp_path, tmp_path / "sheet.png", max_frames=2, columns=2, scale=2)

    assert gif.exists()
    assert sheet.exists()


def test_export_html_viewer_embeds_saved_frames(tmp_path) -> None:
    _write_sample(tmp_path)

    html = export_html_viewer(tmp_path, tmp_path / "viewer.html", max_frames=2)
    content = html.read_text(encoding="utf-8")

    assert "NanoVision Frame Viewer" in content
    assert '"frames":[[' in content
    assert "255" in content
    assert "breakout" in content


def test_audit_flags_manifest_mismatch(tmp_path) -> None:
    artifact = _write_sample(tmp_path)
    manifest = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))
    manifest["frame_count"] = 999
    artifact.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError):
        audit_run(tmp_path)


def test_audit_flags_manifest_coverage_mismatch(tmp_path) -> None:
    artifact = _write_sample(tmp_path)
    manifest = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))
    manifest["games"] = ["asterix", "breakout"]
    artifact.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError):
        audit_run(tmp_path)


def test_audit_flags_manifest_frame_encoding_mismatch(tmp_path) -> None:
    artifact = _write_sample(tmp_path)
    manifest = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))
    manifest["frame_encoding"] = "normalized_float_0_1"
    artifact.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError):
        audit_run(tmp_path)


def test_audit_counts_seeded_episode_identity(tmp_path) -> None:
    frames = np.zeros((1, 10, 10), dtype=np.float32)
    records = [
        EpisodeRecord(
            game="breakout",
            episode=0,
            seed=1,
            frames=frames,
            actions=np.asarray([0], dtype=np.int16),
            rewards=np.asarray([0.0], dtype=np.float32),
            terminals=np.asarray([True]),
        ),
        EpisodeRecord(
            game="breakout",
            episode=0,
            seed=2,
            frames=frames,
            actions=np.asarray([1], dtype=np.int16),
            rewards=np.asarray([0.0], dtype=np.float32),
            terminals=np.asarray([True]),
        ),
    ]
    write_run(tmp_path, records, policy_source="random", settings={})

    audit = audit_run(tmp_path)

    assert audit["episode_count"] == 2
    assert audit["per_game"]["breakout"]["episode_count"] == 2


def test_export_rejects_bad_limits(tmp_path) -> None:
    _write_sample(tmp_path)

    with pytest.raises(ValueError):
        export_snippet(tmp_path, tmp_path / "bad.gif", max_frames=0)
