import numpy as np

from openpi.shared import egomotion_window


def test_build_window_repeats_first_frame_at_sequence_start():
    frames = np.arange(3 * 2, dtype=np.float32).reshape(3, 2)
    window = egomotion_window.build_egomotion_window(frames, end_index=0, span=4)
    expected = np.stack([frames[0], frames[0], frames[0], frames[0]], axis=0)
    np.testing.assert_array_equal(window, expected)


def test_build_window_left_pads_when_history_is_short():
    frames = np.arange(4 * 2, dtype=np.float32).reshape(4, 2)
    window = egomotion_window.build_egomotion_window(frames, end_index=1, span=4)
    expected = np.stack([frames[0], frames[0], frames[0], frames[1]], axis=0)
    np.testing.assert_array_equal(window, expected)


def test_build_window_uses_latest_full_span_when_history_is_long_enough():
    frames = np.arange(6 * 2, dtype=np.float32).reshape(6, 2)
    window = egomotion_window.build_egomotion_window(frames, end_index=5, span=4)
    expected = frames[2:6]
    np.testing.assert_array_equal(window, expected)


def test_append_frame_and_build_window_matches_padding_rule():
    history = [np.array([1.0, 2.0], dtype=np.float32)]
    frame = np.array([3.0, 4.0], dtype=np.float32)
    window = egomotion_window.append_frame_and_build_window(history, frame, span=4)
    expected = np.stack([history[0], history[0], history[0], frame], axis=0)
    np.testing.assert_array_equal(window, expected)