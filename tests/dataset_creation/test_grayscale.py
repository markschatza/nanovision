import numpy as np
import pytest

from nanovision_dataset.grayscale import project_to_grayscale, validate_frame, validate_frames


def test_empty_object_planes_produce_black_frame() -> None:
    obs = np.zeros((10, 10, 3), dtype=bool)

    frame = project_to_grayscale(obs)

    assert frame.shape == (10, 10)
    assert frame.dtype == np.float32
    assert np.all(frame == 0.0)


def test_any_occupied_channel_produces_white_pixel() -> None:
    obs = np.zeros((10, 10, 3), dtype=np.float32)
    obs[2, 4, 1] = 1.0
    obs[2, 4, 2] = 1.0

    frame = project_to_grayscale(obs)

    assert frame[2, 4] == 1.0
    assert frame.sum() == 1.0


@pytest.mark.parametrize(
    "bad_obs",
    [
        np.zeros((10, 10), dtype=np.float32),
        np.zeros((10, 10, 0), dtype=np.float32),
    ],
)
def test_project_rejects_invalid_observation_shape(bad_obs: np.ndarray) -> None:
    with pytest.raises(ValueError):
        project_to_grayscale(bad_obs)


def test_project_rejects_non_finite_observation() -> None:
    obs = np.zeros((10, 10, 1), dtype=np.float32)
    obs[0, 0, 0] = np.nan

    with pytest.raises(ValueError):
        project_to_grayscale(obs)


def test_validate_frame_rejects_bad_shape_and_range() -> None:
    with pytest.raises(ValueError):
        validate_frame(np.zeros((9, 10), dtype=np.float32))

    with pytest.raises(ValueError):
        validate_frame(np.full((10, 10), 2.0, dtype=np.float32))


def test_validate_frames_rejects_bad_rank() -> None:
    with pytest.raises(ValueError):
        validate_frames(np.zeros((10, 10), dtype=np.float32))
