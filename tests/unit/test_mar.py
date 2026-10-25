"""Unit tests for the Mouth Aspect Ratio (MAR) computation."""
from __future__ import annotations

import numpy as np
import pytest

from drowsiness.detectors.geometric import mouth_aspect_ratio


def _closed_mouth(width: float = 10.0) -> np.ndarray:
    """Lips touching → MAR ≈ 0.

    Order: [upper_inner, lower_inner, left_corner, right_corner].
    """
    return np.array([
        [0, 0], [0, 0], [-width / 2, 0], [width / 2, 0],
    ], dtype=np.float64)


def _open_mouth(width: float = 10.0, height: float = 5.0) -> np.ndarray:
    """Mouth opened with vertical/horizontal ratio of 0.5."""
    return np.array([
        [0, -height / 2], [0, height / 2], [-width / 2, 0], [width / 2, 0],
    ], dtype=np.float64)


def test_mar_closed_mouth_is_zero():
    assert mouth_aspect_ratio(_closed_mouth()) == pytest.approx(0.0, abs=1e-9)


def test_mar_open_mouth_half():
    """Vertical = 5, horizontal = 10 → MAR = 0.5."""
    assert mouth_aspect_ratio(_open_mouth(10.0, 5.0)) == pytest.approx(0.5, abs=1e-9)


def test_mar_yawn_threshold_example():
    """A typical yawn: vertical ≈ 5.5, horizontal ≈ 10 → MAR ≈ 0.55."""
    pts = np.array([
        [0, -2.75], [0, 2.75], [-5, 0], [5, 0],
    ], dtype=np.float64)
    assert mouth_aspect_ratio(pts) == pytest.approx(0.55, abs=1e-9)


def test_mar_large_yawn():
    """A large yawn: vertical = 8, horizontal = 10 → MAR = 0.8."""
    pts = np.array([
        [0, -4], [0, 4], [-5, 0], [5, 0],
    ], dtype=np.float64)
    assert mouth_aspect_ratio(pts) == pytest.approx(0.8, abs=1e-9)


def test_mar_handles_3d_input():
    pts2d = _open_mouth()
    pts3d = np.column_stack([pts2d, np.zeros(4)])
    assert mouth_aspect_ratio(pts3d) == pytest.approx(
        mouth_aspect_ratio(pts2d), abs=1e-9
    )


def test_mar_returns_zero_on_degenerate_input():
    short = np.array([[0, 0], [1, 1]], dtype=np.float64)
    assert mouth_aspect_ratio(short) == 0.0


def test_mar_returns_zero_when_mouth_width_is_zero():
    """All mouth landmarks on a vertical line → MAR = 0."""
    pts = np.array([
        [0, -5], [0, 5], [0, 0], [0, 0],
    ], dtype=np.float64)
    assert mouth_aspect_ratio(pts) == 0.0


def test_mar_scales_linearly():
    """Scaling all coordinates by k should NOT change MAR."""
    base = _open_mouth()
    scaled = base * 3.14
    assert mouth_aspect_ratio(scaled) == pytest.approx(
        mouth_aspect_ratio(base), abs=1e-9
    )
