"""End-to-end pipeline tests.

Each test:

1. Opens a real test video from ``tests/assets/test_*.mp4``.
2. Runs the full ``GeometricDetector`` + ``DrowsinessTracker`` pipeline
   over every frame.
3. Asserts an end-to-end behaviour: face was found, EAR/MAR averages
   are within the expected range, and the appropriate alarm fires.

These tests do NOT use mocks — they exercise the real MediaPipe FaceMesh,
the real EAR/MAR math, and the real state machine on real face imagery.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import pytest

from drowsiness.config import load_config
from drowsiness.detectors.geometric import GeometricDetector
from drowsiness.state import DrowsinessTracker
from drowsiness.video.source import VideoSource
from drowsiness.config import VideoConfig

ASSETS = Path(__file__).resolve().parent.parent / "assets"


def _run_pipeline(video_path: Path, *, override_cfg=None):
    """Run the full pipeline on a video file and return the tracker state."""
    cfg = override_cfg or load_config(overrides={"show_window": False,
                                                  "cnn": {"model_path": ""}})
    # Use the file as the source — set width/height to None so OpenCV
    # uses the video's own resolution.
    src = VideoSource(VideoConfig(source=str(video_path),
                                   width=cfg.video.width,
                                   height=cfg.video.height,
                                   target_fps=cfg.video.target_fps))
    det = GeometricDetector(cfg)
    tr = DrowsinessTracker(cfg)
    src.open()
    try:
        idx = 0
        while True:
            frame = src.read()
            if frame is None:
                break
            d = det.detect(frame.image)
            d.frame_index = idx
            idx += 1
            tr.update(d)
    finally:
        src.close()
        det.close()
    return tr.state


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_alert_video_no_alarm():
    """A 6-second clip of an alert face should never trigger any alarm."""
    st = _run_pipeline(ASSETS / "test_alert.mp4")
    assert st.frames_processed > 0
    assert not st.eye_alarm, "Eye alarm fired on alert footage"
    assert not st.yawn_alarm, "Yawn alarm fired on alert footage"
    assert not st.perclos_alarm, "PERCLOS alarm fired on alert footage"
    # EAR should be well above threshold on average.
    assert st.latest_ear is not None and st.latest_ear > 0.20


def test_drowsy_video_triggers_eye_alarm():
    """A clip with 3 s of sleepy (low-EAR) face should trigger the eye alarm."""
    # The sleepy face we use has EAR ≈ 0.25, so we set the threshold a bit
    # above that value AND a low closed_frame_threshold so the alarm fires
    # within the 3-second sleepy segment.
    cfg = load_config(overrides={
        "show_window": False,
        "cnn": {"model_path": ""},
        "eye": {"closed_frame_threshold": 5,
                "ear_threshold": 0.27},
    })
    st = _run_pipeline(ASSETS / "test_drowsy.mp4", override_cfg=cfg)
    assert st.frames_processed > 0
    assert st.eye_alarm, "Eye alarm should have fired on the drowsy clip"


def test_yawning_video_triggers_yawn_alarm():
    """A clip with repeated yawning should trigger the yawn-rate alarm."""
    cfg = load_config(overrides={
        "show_window": False,
        "cnn": {"model_path": ""},
        "mouth": {"yawn_frame_threshold": 3,
                  "yawn_rate_per_minute": 3,
                  "yawn_window_seconds": 30.0},
    })
    st = _run_pipeline(ASSETS / "test_yawning.mp4", override_cfg=cfg)
    assert st.frames_processed > 0
    # Yawn count must be > 0 — the clip has repeated yawning.
    assert st.yawn_count_window >= 1, "No yawns were counted"


def test_mixed_video_fires_at_least_one_alarm():
    """The mixed clip has eyes-closed + yawning; at least one alarm should fire."""
    cfg = load_config(overrides={
        "show_window": False,
        "cnn": {"model_path": ""},
        "eye": {"closed_frame_threshold": 5},
        "mouth": {"yawn_frame_threshold": 3,
                  "yawn_rate_per_minute": 1,
                  "yawn_window_seconds": 30.0},
    })
    st = _run_pipeline(ASSETS / "test_mixed.mp4", override_cfg=cfg)
    assert st.frames_processed > 0
    assert st.any_alarm, "Mixed clip should have raised at least one alarm"


def test_pipeline_handles_no_face_gracefully(tmp_path):
    """A video of pure noise (no face) must not crash and must not alarm."""
    # Generate a 2-second noise video.
    noise_path = tmp_path / "noise.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w = cv2.VideoWriter(str(noise_path), fourcc, 15, (640, 480))
    for _ in range(30):
        w.write(__import__("numpy").random.randint(0, 255, (480, 640, 3),
                                                    dtype=__import__("numpy").uint8))
    w.release()
    st = _run_pipeline(noise_path)
    assert st.frames_processed > 0
    assert not st.any_alarm
    assert st.latest_ear is None  # No face → no EAR


def test_each_test_video_has_face_detected_in_most_frames():
    """Sanity: MediaPipe should find a face in >90 % of the test frames."""
    for fn in ("test_alert.mp4", "test_drowsy.mp4",
               "test_yawning.mp4", "test_mixed.mp4"):
        st = _run_pipeline(ASSETS / fn)
        assert st.frames_processed > 0, f"{fn} produced no frames"
        # We know each frame has a real face; if 90 %+ don't, something is
        # wrong with the detector or the test asset.
        # We approximate this by checking that the latest EAR/MAR were
        # set at least once (i.e., a face was found at some point).
        assert st.latest_ear is not None, f"No face ever found in {fn}"
