"""Unit tests for the Eye Aspect Ratio computation.

These tests use **synthetic landmark arrays with known geometry** so the
math is verifiable without any image content — no mocks of MediaPipe,
no real face images.  Each test asserts an exact numeric value derived
from the EAR formula, not a range.
"""
from __future__ import annotations

import numpy as np
import pytest

from drowsiness.detectors.geometric import eye_aspect_ratio


# ---------------------------------------------------------------------------
# Hand-computed landmark positions
# ---------------------------------------------------------------------------
def _square_eye(side: float = 10.0) -> np.ndarray:
    """Eye landmarks for a perfectly square eye.

    Layout (per the EAR convention):
      p0 = (-5, 0)   outer corner
      p1 = (-2, 5)   top-outer
      p2 = ( 2, 5)   top-inner
      p3 = ( 5, 0)   inner corner
      p4 = ( 2,-5)   bottom-inner
      p5 = (-2,-5)   bottom-outer

    Vertical distances: ||p1-p5|| = ||p2-p4|| = 10
    Horizontal distance: ||p0-p3|| = 10
    EAR = (10 + 10) / (2 * 10) = 1.0
    """
    return np.array([
        [-5, 0], [-2, 5], [2, 5], [5, 0], [2, -5], [-2, -5],
    ], dtype=np.float64) * (side / 10.0)


def _flat_eye(width: float = 10.0) -> np.ndarray:
    """Eye landmarks where all points lie on a single horizontal line.

    EAR = 0 / (2 * 10) = 0.0  (perfectly closed)
    """
    x = width / 2
    return np.array([
        [-x, 0], [-x / 2, 0], [x / 2, 0], [x, 0], [x / 2, 0], [-x / 2, 0],
    ], dtype=np.float64)


def _wide_open_eye() -> np.ndarray:
    """A wide-open eye: vertical distance much larger than horizontal.

    p1-p5 distance = 12, p2-p4 distance = 12, p0-p3 distance = 6.
    EAR = (12 + 12) / (2 * 6) = 2.0
    """
    return np.array([
        [-3, 0], [-1, 6], [1, 6], [3, 0], [1, -6], [-1, -6],
    ], dtype=np.float64)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_ear_open_eye_is_one():
    """Square eye → EAR = 1.0 (the theoretical maximum for this geometry)."""
    assert eye_aspect_ratio(_square_eye()) == pytest.approx(1.0, abs=1e-9)


def test_ear_closed_eye_is_zero():
    """Flat (closed) eye → EAR = 0.0."""
    assert eye_aspect_ratio(_flat_eye()) == pytest.approx(0.0, abs=1e-9)


def test_ear_wide_open_eye_is_two():
    """Wide-open eye (vertical > horizontal) → EAR = 2.0."""
    assert eye_aspect_ratio(_wide_open_eye()) == pytest.approx(2.0, abs=1e-9)


def test_ear_realistic_open_value():
    """Realistic open eye → EAR ≈ 0.30 (canonical open-eye value).

    p1-p5 = 3 (vertical, top-outer to bottom-outer)
    p2-p4 = 3 (vertical, top-inner to bottom-inner)
    p0-p3 = 10 (horizontal, outer corner to inner corner)
    EAR = (3 + 3) / (2 * 10) = 0.30
    """
    pts = np.array([
        [-5, 0], [-2, 1.5], [2, 1.5], [5, 0], [2, -1.5], [-2, -1.5],
    ], dtype=np.float64)
    assert eye_aspect_ratio(pts) == pytest.approx(0.30, abs=1e-9)


def test_ear_realistic_closed_value():
    """Realistic closed eye → EAR ≈ 0.10.

    p1-p5 = 1, p2-p4 = 1, p0-p3 = 10.
    EAR = (1 + 1) / (2 * 10) = 0.10
    """
    pts = np.array([
        [-5, 0], [-2, 0.5], [2, 0.5], [5, 0], [2, -0.5], [-2, -0.5],
    ], dtype=np.float64)
    assert eye_aspect_ratio(pts) == pytest.approx(0.10, abs=1e-9)


def test_ear_handles_3d_input():
    """3-D landmark arrays (x, y, z) should still work — only x and y used."""
    pts2d = _square_eye()
    pts3d = np.column_stack([pts2d, np.zeros(6)])
    assert eye_aspect_ratio(pts3d) == pytest.approx(
        eye_aspect_ratio(pts2d), abs=1e-9
    )


def test_ear_returns_zero_on_degenerate_input():
    """Too-few landmarks → return 0.0 (defensive)."""
    short = np.array([[0, 0], [1, 1]], dtype=np.float64)
    assert eye_aspect_ratio(short) == 0.0


def test_ear_returns_zero_on_zero_horizontal():
    """Outer corner == inner corner (d3 = 0) → division by zero is guarded → 0.0."""
    pts = np.array([
        [0, -5], [-1, 3], [1, 3], [0, 0], [1, -3], [-1, -3],
    ], dtype=np.float64)
    # p0 and p3 are both (0,0)-ish — actually we set p0=(0,-5) and p3=(0,0).
    # Let's make them identical so d3 is exactly 0.
    pts = np.array([
        [0, 0], [-1, 3], [1, 3], [0, 0], [1, -3], [-1, -3],
    ], dtype=np.float64)
    assert eye_aspect_ratio(pts) == 0.0


def test_ear_scales_linearly():
    """Scaling all coordinates by k should NOT change EAR (it's a ratio)."""
    base = _square_eye()
    scaled = base * 7.3
    assert eye_aspect_ratio(scaled) == pytest.approx(
        eye_aspect_ratio(base), abs=1e-9,
    )
