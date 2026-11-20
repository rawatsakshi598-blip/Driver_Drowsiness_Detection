"""Unit tests for the new real-time alerts: prolonged yawn, face-not-visible
distraction, looking-away, low/bright lighting, out-of-focus.

Feeds the tracker synthetic Detections with known brightness/sharpness/
straightness values so behaviour is deterministic without a camera.
"""
from __future__ import annotations

import time

from drowsiness.config import (
    Config,
    DistractionConfig,
    FocusConfig,
    LightingConfig,
    MouthConfig,
)
from drowsiness.detectors.base import Detection
from drowsiness.state import DrowsinessTracker


def _det(*, t, face=True, ear=0.30, mar=0.10,
         eyes_closed=False, yawning=False,
         brightness=128.0, sharpness=500.0, straightness=1.0) -> Detection:
    return Detection(
        timestamp=t,
        face_found=face,
        ear=ear,
        mar=mar,
        eyes_closed=eyes_closed,
        yawning=yawning,
        mean_brightness=brightness,
        sharpness=sharpness,
        face_straightness=straightness,
    )


# ---------------------------------------------------------------------------
# Prolonged yawn
# ---------------------------------------------------------------------------
def test_prolonged_yawn_fires_after_max_yawn_seconds():
    cfg = Config(mouth=MouthConfig(max_yawn_seconds=2.0))
    tr = DrowsinessTracker(cfg)
    tr.update(_det(t=0.0, yawning=True, mar=0.9))
    st = tr.update(_det(t=1.0, yawning=True, mar=0.9))
    assert not st.prolonged_yawn_alarm
    st = tr.update(_det(t=2.5, yawning=True, mar=0.9))
    assert st.prolonged_yawn_alarm
    assert any(a["code"] == "PROLONGED_YAWN" for a in st.alerts)


def test_prolonged_yawn_clears_when_mouth_closes():
    cfg = Config(mouth=MouthConfig(max_yawn_seconds=1.0))
    tr = DrowsinessTracker(cfg)
    tr.update(_det(t=0.0, yawning=True, mar=0.9))
    st = tr.update(_det(t=2.0, yawning=True, mar=0.9))
    assert st.prolonged_yawn_alarm
    st = tr.update(_det(t=2.5, yawning=False, mar=0.1))
    assert not st.prolonged_yawn_alarm
    assert all(a["code"] != "PROLONGED_YAWN" for a in st.alerts)


# ---------------------------------------------------------------------------
# Face missing / distraction
# ---------------------------------------------------------------------------
def test_face_missing_fires_after_grace_period_and_clears():
    cfg = Config(distraction=DistractionConfig(no_face_seconds=2.0))
    tr = DrowsinessTracker(cfg)
    # Face visible at t=0.
    tr.update(_det(t=100.0, face=True))
    # 1 s without face — under the limit, no alert yet.
    st = tr.update(_det(t=101.0, face=False))
    assert not st.face_missing_alarm
    # 2.5 s without face — critical alert.
    st = tr.update(_det(t=102.5, face=False))
    assert st.face_missing_alarm
    codes = [a["code"] for a in st.alerts]
    severities = {a["code"]: a["severity"] for a in st.alerts}
    assert "FACE_NOT_VISIBLE" in codes
    assert severities["FACE_NOT_VISIBLE"] == "critical"
    # Face returns — alert clears.
    st = tr.update(_det(t=103.0, face=True))
    assert not st.face_missing_alarm


def test_face_missing_disabled_does_not_fire():
    cfg = Config(distraction=DistractionConfig(
        no_face_seconds=0.5, enabled=False))
    tr = DrowsinessTracker(cfg)
    tr.update(_det(t=100.0, face=True))
    st = tr.update(_det(t=200.0, face=False))
    assert not st.face_missing_alarm


# ---------------------------------------------------------------------------
# Looking away (head yaw straightness)
# ---------------------------------------------------------------------------
def test_looking_away_fires_after_consecutive_frames_and_clears():
    cfg = Config(distraction=DistractionConfig(
        look_away_straightness=0.5, look_away_seconds=0.15))
    tr = DrowsinessTracker(cfg)
    tr.update(_det(t=0.0, straightness=0.2))
    tr.update(_det(t=0.1, straightness=0.2))
    st = tr.update(_det(t=0.2, straightness=0.2))
    assert st.looking_away_alarm
    assert any(a["code"] == "LOOKING_AWAY" for a in st.alerts)
    st = tr.update(_det(t=0.3, straightness=0.95))
    assert not st.looking_away_alarm


# ---------------------------------------------------------------------------
# Lighting
# ---------------------------------------------------------------------------
def test_low_light_fires_after_consecutive_dark_frames():
    cfg = Config(lighting=LightingConfig(
        min_brightness=60.0, consecutive_frames=3))
    tr = DrowsinessTracker(cfg)
    tr.update(_det(t=0.0, brightness=30.0))
    tr.update(_det(t=0.1, brightness=30.0))
    st = tr.update(_det(t=0.2, brightness=30.0))
    assert st.lighting_alarm
    assert st.low_light_alarm
    alert = next(a for a in st.alerts if a["code"] == "LOW_LIGHT")
    assert alert["severity"] == "warning"
    # Brightness recovers — alert clears.
    st = tr.update(_det(t=0.3, brightness=140.0))
    assert not st.lighting_alarm


def test_bright_light_fires_after_consecutive_bright_frames():
    cfg = Config(lighting=LightingConfig(
        max_brightness=240.0, consecutive_frames=2))
    tr = DrowsinessTracker(cfg)
    tr.update(_det(t=0.0, brightness=250.0))
    st = tr.update(_det(t=0.1, brightness=252.0))
    assert st.bright_light_alarm
    assert any(a["code"] == "BRIGHT_LIGHT" for a in st.alerts)


def test_lighting_disabled_never_fires():
    cfg = Config(lighting=LightingConfig(
        enabled=False, min_brightness=250.0, max_brightness=0.0,
        consecutive_frames=1))
    tr = DrowsinessTracker(cfg)
    st = tr.update(_det(t=0.0, brightness=10.0))
    assert not st.lighting_alarm


# ---------------------------------------------------------------------------
# Focus / blur
# ---------------------------------------------------------------------------
def test_out_of_focus_fires_after_consecutive_blurry_frames_and_clears():
    cfg = Config(focus=FocusConfig(min_sharpness=100.0, consecutive_frames=3))
    tr = DrowsinessTracker(cfg)
    tr.update(_det(t=0.0, sharpness=10.0))
    tr.update(_det(t=0.1, sharpness=10.0))
    st = tr.update(_det(t=0.2, sharpness=10.0))
    assert st.out_of_focus_alarm
    assert any(a["code"] == "OUT_OF_FOCUS" for a in st.alerts)
    st = tr.update(_det(t=0.3, sharpness=500.0))
    assert not st.out_of_focus_alarm


# ---------------------------------------------------------------------------
# Combined / housekeeping
# ---------------------------------------------------------------------------
def test_alerts_list_and_any_alert_property():
    cfg = Config(
        lighting=LightingConfig(min_brightness=60.0, consecutive_frames=1),
        focus=FocusConfig(min_sharpness=100.0, consecutive_frames=1),
    )
    tr = DrowsinessTracker(cfg)
    st = tr.update(_det(t=0.0, brightness=20.0, sharpness=5.0))
    assert st.any_alert
    codes = {a["code"] for a in st.alerts}
    assert {"LOW_LIGHT", "OUT_OF_FOCUS"} <= codes
    # Drowsiness alarms remain separate from alerts.
    assert not st.any_alarm


def test_healthy_frame_produces_no_alerts():
    cfg = Config()
    tr = DrowsinessTracker(cfg)
    st = tr.update(_det(t=0.0))
    assert st.alerts == []
    assert not st.any_alert
    assert not st.any_issue


def test_reset_clears_alert_counters():
    cfg = Config(lighting=LightingConfig(
        min_brightness=60.0, consecutive_frames=1))
    tr = DrowsinessTracker(cfg)
    st = tr.update(_det(t=0.0, brightness=10.0))
    assert st.lighting_alarm
    tr.reset()
    st = tr.update(_det(t=1.0, brightness=200.0))
    assert not st.lighting_alarm
    assert st.alerts == []
