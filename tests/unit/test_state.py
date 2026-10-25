"""Unit tests for the DrowsinessTracker state machine.

These tests feed the tracker **synthetic Detection objects** with known
EAR / MAR / face_found values, so the temporal behaviour (PERCLOS, yawn
rate, alarm thresholds) can be verified deterministically without any
image processing.
"""
from __future__ import annotations

import time

import pytest

from drowsiness.config import Config, EyeConfig, MouthConfig
from drowsiness.detectors.base import Detection
from drowsiness.state import DrowsinessTracker


def _det(*, t=None, face=True, ear=None, mar=None,
         eyes_closed=False, yawning=False, idx=0) -> Detection:
    """Build a Detection with the given fields."""
    return Detection(
        timestamp=t if t is not None else time.time(),
        face_found=face,
        ear=ear,
        mar=mar,
        eyes_closed=eyes_closed,
        yawning=yawning,
        frame_index=idx,
    )


# ---------------------------------------------------------------------------
# Eye-alarm tests
# ---------------------------------------------------------------------------
def test_eye_alarm_fires_after_threshold_frames():
    """Closed eyes for >= closed_frame_threshold frames raises the alarm."""
    cfg = Config(eye=EyeConfig(closed_frame_threshold=5))
    tr = DrowsinessTracker(cfg)
    # First 4 frames: eyes closed but below threshold.
    for i in range(4):
        st = tr.update(_det(eyes_closed=True, ear=0.05, idx=i))
        assert not st.eye_alarm, f"Alarm should not fire at frame {i}"
    # 5th frame: reaches threshold.
    st = tr.update(_det(eyes_closed=True, ear=0.05, idx=4))
    assert st.eye_alarm
    assert st.consecutive_closed_frames == 5


def test_eye_alarm_resets_when_eye_opens():
    """A single open-eye frame resets the closed-frame counter."""
    cfg = Config(eye=EyeConfig(closed_frame_threshold=5))
    tr = DrowsinessTracker(cfg)
    for i in range(4):
        tr.update(_det(eyes_closed=True, ear=0.05, idx=i))
    # Open eye for one frame.
    tr.update(_det(eyes_closed=False, ear=0.35, idx=4))
    # Should have reset.
    assert tr.state.consecutive_closed_frames == 0
    assert not tr.state.eye_alarm
    # Closing again now needs another 5 frames.
    for i in range(5, 9):
        st = tr.update(_det(eyes_closed=True, ear=0.05, idx=i))
        assert not st.eye_alarm
    st = tr.update(_det(eyes_closed=True, ear=0.05, idx=9))
    assert st.eye_alarm


def test_perclos_computed_over_window():
    """PERCLOS = (# closed-eye frames in window) / (total frames in window)."""
    cfg = Config(eye=EyeConfig(perclos_window_seconds=10.0,
                                perclos_threshold=0.5,
                                closed_frame_threshold=999))  # disable eye-alarm
    tr = DrowsinessTracker(cfg)
    t0 = time.time()
    # 5 closed-eye frames at t0..t0+4, then 5 open-eye frames at t0+5..t0+9.
    for i in range(5):
        tr.update(_det(t=t0 + i, eyes_closed=True, ear=0.05, idx=i))
    for i in range(5, 10):
        tr.update(_det(t=t0 + i, eyes_closed=False, ear=0.35, idx=i))
    assert tr.state.perclos == pytest.approx(0.5, abs=1e-6)
    assert tr.state.perclos_alarm  # threshold was 0.5


def test_perclos_window_expires_old_entries():
    """Old entries outside the window should not count toward PERCLOS."""
    cfg = Config(eye=EyeConfig(perclos_window_seconds=2.0,
                                perclos_threshold=99.0,
                                closed_frame_threshold=999))
    tr = DrowsinessTracker(cfg)
    t0 = time.time()
    # 10 closed-eye frames at t0 (all in window).
    for i in range(10):
        tr.update(_det(t=t0, eyes_closed=True, ear=0.05, idx=i))
    assert tr.state.perclos == pytest.approx(1.0, abs=1e-6)
    # A frame at t0+10 — old entries should expire.
    tr.update(_det(t=t0 + 10.0, eyes_closed=False, ear=0.35, idx=10))
    assert tr.state.perclos == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Yawn-alarm tests
# ---------------------------------------------------------------------------
def test_yawn_count_increments_on_rising_edge():
    """A yawn counts as ONE event, not N frames."""
    cfg = Config(mouth=MouthConfig(yawn_frame_threshold=3,
                                    yawn_rate_per_minute=999,
                                    yawn_window_seconds=60.0))
    tr = DrowsinessTracker(cfg)
    t0 = time.time()
    # 5 frames of yawning (above threshold of 3) then 1 frame of no yawn.
    for i in range(5):
        tr.update(_det(t=t0 + i, yawning=True, mar=0.7, idx=i))
    tr.update(_det(t=t0 + 5, yawning=False, mar=0.1, idx=5))
    assert tr.state.yawn_count_window == 1


def test_multiple_yawns_are_counted_separately():
    """Two distinct yawn events → count = 2."""
    cfg = Config(mouth=MouthConfig(yawn_frame_threshold=2,
                                    yawn_rate_per_minute=999,
                                    yawn_window_seconds=60.0))
    tr = DrowsinessTracker(cfg)
    t0 = time.time()
    # First yawn: 3 frames closed, then 1 open.
    for i in range(3):
        tr.update(_det(t=t0 + i, yawning=True, mar=0.7, idx=i))
    tr.update(_det(t=t0 + 3, yawning=False, mar=0.1, idx=3))
    # Second yawn: 3 frames closed, then 1 open.
    for i in range(4, 7):
        tr.update(_det(t=t0 + i, yawning=True, mar=0.7, idx=i))
    tr.update(_det(t=t0 + 7, yawning=False, mar=0.1, idx=7))
    assert tr.state.yawn_count_window == 2


def test_yawn_below_threshold_does_not_count():
    """Brief mouth openings (< yawn_frame_threshold frames) don't count."""
    cfg = Config(mouth=MouthConfig(yawn_frame_threshold=10,
                                    yawn_rate_per_minute=999,
                                    yawn_window_seconds=60.0))
    tr = DrowsinessTracker(cfg)
    t0 = time.time()
    for i in range(5):  # 5 < 10 threshold
        tr.update(_det(t=t0 + i, yawning=True, mar=0.7, idx=i))
    tr.update(_det(t=t0 + 5, yawning=False, mar=0.1, idx=5))
    assert tr.state.yawn_count_window == 0


def test_yawn_rate_triggers_alarm():
    """3 yawns in a 60 s window → yawn_rate_per_min = 3 → alarm fires."""
    cfg = Config(mouth=MouthConfig(yawn_frame_threshold=2,
                                    yawn_rate_per_minute=3,
                                    yawn_window_seconds=60.0))
    tr = DrowsinessTracker(cfg)
    t0 = time.time()
    for yawn_idx in range(3):
        offset = yawn_idx * 5.0
        for i in range(3):
            tr.update(_det(t=t0 + offset + i, yawning=True, mar=0.7, idx=yawn_idx * 4 + i))
        tr.update(_det(t=t0 + offset + 3, yawning=False, mar=0.1, idx=yawn_idx * 4 + 3))
    assert tr.state.yawn_count_window == 3
    assert tr.state.yawn_rate_per_min == pytest.approx(3.0, abs=1e-6)
    assert tr.state.yawn_alarm


# ---------------------------------------------------------------------------
# No-face / edge cases
# ---------------------------------------------------------------------------
def test_no_face_does_not_crash_or_reset_counters():
    """If a frame has no face, the tracker should not crash or reset state."""
    cfg = Config()
    tr = DrowsinessTracker(cfg)
    # Build up some state.
    for i in range(3):
        tr.update(_det(eyes_closed=True, ear=0.05, idx=i))
    assert tr.state.consecutive_closed_frames == 3
    # Now a frame with no face.
    tr.update(_det(face=False, ear=None, mar=None, idx=3))
    # Counter should NOT have reset.
    assert tr.state.consecutive_closed_frames == 3
    # But alarm should NOT have fired either (we couldn't measure).
    assert not tr.state.eye_alarm


def test_reset_clears_all_state():
    cfg = Config(eye=EyeConfig(closed_frame_threshold=5))
    tr = DrowsinessTracker(cfg)
    for i in range(3):
        tr.update(_det(eyes_closed=True, ear=0.05, idx=i))
    assert tr.state.consecutive_closed_frames == 3
    tr.reset()
    assert tr.state.consecutive_closed_frames == 0
    assert tr.state.frames_processed == 0
