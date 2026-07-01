import json

import numpy as np

from nanovision_dataset.cli import main
from nanovision_dataset.minatar_source import EpisodeRecord
from nanovision_dataset.writer import write_run


def _write_sample(tmp_path):
    record = EpisodeRecord(
        game="breakout",
        episode=0,
        seed=3,
        frames=np.zeros((2, 10, 10), dtype=np.float32),
        actions=np.asarray([0, 1], dtype=np.int16),
        rewards=np.asarray([0.0, 1.0], dtype=np.float32),
        terminals=np.asarray([False, True]),
    )
    write_run(tmp_path, [record], policy_source="random", settings={})


def test_audit_command_outputs_summary(tmp_path, capsys) -> None:
    _write_sample(tmp_path)

    assert main(["audit", str(tmp_path)]) == 0
    output = json.loads(capsys.readouterr().out)

    assert output["ok"] is True
    assert output["frame_count"] == 2


def test_generate_command_creates_tiny_run(tmp_path, capsys) -> None:
    output_dir = tmp_path / "generated"

    assert main(
        [
            "generate",
            "--games",
            "breakout",
            "--episodes",
            "1",
            "--seed",
            "0",
            "--max-steps",
            "2",
            "--out",
            str(output_dir),
        ]
    ) == 0
    output = json.loads(capsys.readouterr().out)

    assert output["artifact"]["frame_count"] == 2
    assert (output_dir / "frames.npz").exists()
    assert (output_dir / "manifest.json").exists()


def test_export_commands_create_visual_artifacts(tmp_path, capsys) -> None:
    _write_sample(tmp_path)

    html = tmp_path / "viewer.html"
    gif = tmp_path / "snippet.gif"
    sheet = tmp_path / "sheet.png"

    assert main(["export-html", str(tmp_path), "--out", str(html), "--max-frames", "1"]) == 0
    assert main(["export-snippet", str(tmp_path), "--out", str(gif), "--max-frames", "1"]) == 0
    assert main(["export-contact-sheet", str(tmp_path), "--out", str(sheet), "--max-frames", "1"]) == 0

    assert html.exists()
    assert gif.exists()
    assert sheet.exists()
    assert "output" in capsys.readouterr().out
