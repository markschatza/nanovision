from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FrameStats:
    shape: tuple[int, int]
    dtype: str
    min_value: float
    max_value: float


def project_to_grayscale(observation: np.ndarray) -> np.ndarray:
    """Project MinAtar object planes to object-agnostic occupancy grayscale."""
    array = np.asarray(observation)
    if array.ndim != 3:
        raise ValueError(f"expected observation rank 3, got shape {array.shape}")
    if array.shape[2] < 1:
        raise ValueError("expected at least one object channel")
    if not np.isfinite(array).all():
        raise ValueError("observation contains non-finite values")

    occupied = np.any(array > 0, axis=2)
    frame = occupied.astype(np.float32)
    validate_frame(frame)
    return frame


def validate_frame(frame: np.ndarray, expected_shape: tuple[int, int] | None = (10, 10)) -> FrameStats:
    array = np.asarray(frame)
    if array.ndim != 2:
        raise ValueError(f"expected grayscale frame rank 2, got shape {array.shape}")
    if expected_shape is not None and tuple(array.shape) != expected_shape:
        raise ValueError(f"expected frame shape {expected_shape}, got {array.shape}")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"expected numeric frame dtype, got {array.dtype}")
    if not np.isfinite(array).all():
        raise ValueError("frame contains non-finite values")
    min_value = float(array.min()) if array.size else 0.0
    max_value = float(array.max()) if array.size else 0.0
    if min_value < 0.0 or max_value > 1.0:
        raise ValueError(f"expected frame values in [0, 1], got [{min_value}, {max_value}]")
    return FrameStats(
        shape=tuple(int(dim) for dim in array.shape),
        dtype=str(array.dtype),
        min_value=min_value,
        max_value=max_value,
    )


def validate_frames(frames: np.ndarray, expected_shape: tuple[int, int] | None = (10, 10)) -> None:
    array = np.asarray(frames)
    if array.ndim != 3:
        raise ValueError(f"expected frames rank 3, got shape {array.shape}")
    for frame in array:
        validate_frame(frame, expected_shape=expected_shape)
