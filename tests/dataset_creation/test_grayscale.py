import numpy as np
import pytest

from nanovision_dataset.grayscale import (
    channel_weights,
    frame_pixels_uint8,
    frames_to_uint8,
    project_to_grayscale,
    validate_frame,
    validate_frames,
)


def test_empty_object_planes_produce_black_frame() -> None:
    obs = np.zeros((10, 10, 3), dtype=bool)

    frame = project_to_grayscale(obs)

    assert frame.shape == (10, 10)
    assert frame.dtype == np.float32
    assert np.all(frame == 0.0)


def test_occupied_channels_produce_distinct_grayscale_values() -> None:
    obs = np.zeros((10, 10, 3), dtype=np.float32)
    obs[1, 1, 0] = 1.0
    obs[2, 2, 1] = 1.0
    obs[3, 3, 2] = 1.0

    frame = project_to_grayscale(obs)

    assert frame[1, 1] == pytest.approx(0.25)
    assert frame[2, 2] == pytest.approx(0.625)
    assert frame[3, 3] == pytest.approx(1.0)


def test_overlapping_channels_keep_brightest_active_value() -> None:
    obs = np.zeros((10, 10, 3), dtype=np.float32)
    obs[2, 4, 0] = 1.0
    obs[2, 4, 2] = 1.0

    frame = project_to_grayscale(obs)

    assert frame[2, 4] == 1.0
    assert frame.sum() == 1.0


def test_single_channel_uses_white_for_occupied_pixels() -> None:
    obs = np.zeros((10, 10, 1), dtype=np.float32)
    obs[2, 4, 0] = 1.0

    frame = project_to_grayscale(obs)

    assert frame[2, 4] == 1.0


def test_channel_weights_validate_inputs() -> None:
    np.testing.assert_allclose(channel_weights(3), np.asarray([0.25, 0.625, 1.0], dtype=np.float32))

    with pytest.raises(ValueError):
        channel_weights(0)

    with pytest.raises(ValueError):
        channel_weights(2, min_value=0.0)


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


def test_validate_frame_accepts_uint8_storage_values() -> None:
    stats = validate_frame(np.full((10, 10), 255, dtype=np.uint8))

    assert stats.dtype == "uint8"
    assert stats.max_value == 255.0


def test_frames_to_uint8_quantizes_normalized_float_frames() -> None:
    frames = np.zeros((1, 10, 10), dtype=np.float32)
    frames[0, 0, 1] = 0.5
    frames[0, 0, 2] = 1.0

    quantized = frames_to_uint8(frames)

    assert quantized.dtype == np.uint8
    assert quantized[0, 0, 0] == 0
    assert quantized[0, 0, 1] == 128
    assert quantized[0, 0, 2] == 255


def test_frame_pixels_uint8_preserves_uint8_frames() -> None:
    frame = np.zeros((10, 10), dtype=np.uint8)
    frame[0, 1] = 128
    frame[0, 2] = 255

    pixels = frame_pixels_uint8(frame)

    assert pixels.dtype == np.uint8
    np.testing.assert_array_equal(pixels, frame)


def test_validate_frames_rejects_bad_rank() -> None:
    with pytest.raises(ValueError):
        validate_frames(np.zeros((10, 10), dtype=np.float32))
