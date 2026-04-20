from __future__ import annotations

import numpy as np


def build_egomotion_window(frames: np.ndarray, end_index: int, span: int = 16) -> np.ndarray:
    """Build a fixed-length history window ending at end_index.

    If there are fewer than ``span`` frames available on the left, the earliest
    available frame is repeated to pad the window.

    Example for span=4:
    - end_index=0 -> [I0, I0, I0, I0]
    - end_index=1 -> [I0, I0, I0, I1]
    - end_index=3 -> [I0, I1, I2, I3]

    Args:
        frames: Array of shape [T, ...] containing ordered frame history.
        end_index: Inclusive index of the current frame.
        span: Number of frames in the output window.

    Returns:
        Array of shape [span, ...].
    """
    frames = np.asarray(frames)

    if frames.ndim < 1:
        raise ValueError(f"Expected frames with rank >= 1, got shape {frames.shape}")
    if frames.shape[0] == 0:
        raise ValueError("Cannot build EgoMotion window from an empty frame array")
    if span <= 0:
        raise ValueError(f"span must be positive, got {span}")
    if end_index < 0 or end_index >= frames.shape[0]:
        raise IndexError(f"end_index {end_index} out of range for {frames.shape[0]} frames")

    start_index = max(0, end_index - span + 1)
    window = frames[start_index : end_index + 1]

    missing = span - window.shape[0]
    if missing <= 0:
        return window

    first_frame = window[:1]
    left_pad = np.repeat(first_frame, missing, axis=0)
    return np.concatenate([left_pad, window], axis=0)


def append_frame_and_build_window(history: list[np.ndarray], frame: np.ndarray, span: int = 16) -> np.ndarray:
    """Append a frame to history and return the padded fixed-length window.

    This helper is convenient for online inference with a Python-side frame
    buffer. The returned window uses the same earliest-frame replication policy
    as ``build_egomotion_window``.
    """
    history.append(np.asarray(frame))
    stacked = np.stack(history, axis=0)
    return build_egomotion_window(stacked, end_index=stacked.shape[0] - 1, span=span)
