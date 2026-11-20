"""Unit tests for the unified AudioManager.

Deterministic: uses ``_sync=True`` (no worker thread) and a fake
``pygame`` module injected into ``sys.modules`` so the real code path —
state machine, priority pre-emption, channel handling — is exercised
without an audio device.
"""
from __future__ import annotations

import sys
import types

import pytest

from drowsiness.config import (
    AlarmConfig,
    AudioAlertConfig,
    AudioConfig,
    Config,
    DistractionConfig,
    LightingConfig,
)
from drowsiness.detectors.base import Detection
from drowsiness.state import DrowsinessTracker


# ---------------------------------------------------------------------------
# Fake pygame
# ---------------------------------------------------------------------------
class _FakeChannel:
    def __init__(self):
        self.plays = 0
        self.stops = 0
        self._busy = False
        self.volume = None

    def get_busy(self):
        return self._busy

    def stop(self):
        self.stops += 1
        self._busy = False

    def set_volume(self, v):
        self.volume = v

    def play(self, snd):
        self.plays += 1
        self._busy = True


class _FakeSound:
    def __init__(self, path=None, buffer=None):
        self.path = path
        self.buffer = buffer
        self.volume = None

    def set_volume(self, v):
        self.volume = v


def _fake_pygame_module():
    module = types.ModuleType("pygame")
    mixer = types.SimpleNamespace()
    mixer.init = lambda **kw: None
    mixer.quit = lambda: None
    mixer.set_num_channels = lambda n: None
    mixer.Sound = _FakeSound
    mixer.Channel = lambda i: _FakeChannel()
    module.mixer = mixer
    return module


def _make_manager(monkeypatch, audio_cfg=None):
    monkeypatch.setitem(sys.modules, "pygame", _fake_pygame_module())
    from drowsiness.alarms.audio_manager import AudioManager
    cfg = audio_cfg or AudioConfig()
    return AudioManager(cfg, AlarmConfig(sound_path="definitely_missing.mp3"),
                        _sync=True)


# ---------------------------------------------------------------------------
# State machine: INACTIVE -> PENDING -> ACTIVE
# ---------------------------------------------------------------------------
def test_pending_then_active_after_min_seconds(monkeypatch):
    mgr = _make_manager(monkeypatch, AudioConfig(
        look_away=AudioAlertConfig(priority=60, repeat_seconds=2.0,
                                   min_seconds=1.0, volume=0.8,
                                   frequency_hz=660.0),
    ))
    assert mgr.update("look_away", True, now=100.0) is False  # PENDING
    assert mgr.beeps_emitted == 0
    assert mgr.update("look_away", True, now=100.4) is False  # still PENDING
    assert mgr.beeps_emitted == 0
    assert mgr.update("look_away", True, now=101.1) is True   # ACTIVE
    assert mgr.beeps_emitted == 1                             # beep on ACTIVE


def test_active_repeats_at_repeat_seconds(monkeypatch):
    mgr = _make_manager(monkeypatch)
    mgr.update("yawn", True, now=0.0)
    assert mgr.beeps_emitted == 1                     # ACTIVE entry beep
    mgr.update("yawn", True, now=0.5)                 # too soon
    assert mgr.beeps_emitted == 1
    mgr.update("yawn", True, now=2.5)                 # repeat_seconds=2.0 reached
    assert mgr.beeps_emitted == 2


def test_condition_end_returns_to_inactive_and_clears(monkeypatch):
    mgr = _make_manager(monkeypatch)
    mgr.update("low_light", True, now=0.0)
    assert mgr.is_active()
    mgr.update("low_light", False, now=0.1)
    assert not mgr.is_active()
    assert mgr.state_of("low_light") == "INACTIVE"
    # And it can fire again from scratch.
    mgr.update("low_light", True, now=1.0)
    assert mgr.state_of("low_light") == "ACTIVE"
    assert mgr.beeps_emitted == 2


def test_unknown_alert_type_rejected(monkeypatch):
    mgr = _make_manager(monkeypatch)
    with pytest.raises(ValueError):
        mgr.update("migraine", True, now=0.0)


def test_disabled_manager_is_silent(monkeypatch):
    mgr = _make_manager(monkeypatch, AudioConfig(enabled=False))
    assert mgr.update("drowsy", True, now=0.0) is False
    assert mgr.beeps_emitted == 0
    assert not mgr.is_active()
    # trigger() still returns an event for /alarm/test, just no beep
    ev = mgr.trigger("manual_test", 0.0)
    assert ev.reason == "manual_test"
    assert mgr.beeps_emitted == 0


# ---------------------------------------------------------------------------
# Priority & pre-emption
# ---------------------------------------------------------------------------
def test_pick_highest_priority(monkeypatch):
    mgr = _make_manager(monkeypatch)
    assert mgr._pick_highest_priority({"low_light": True, "yawn": True}) == "yawn"
    assert mgr._pick_highest_priority(
        {"low_light": True, "look_away": True, "yawn": True}) == "look_away"
    assert mgr._pick_highest_priority(
        {"drowsy": True, "look_away": True}) == "drowsy"


def test_higher_priority_preempts_lower_channel(monkeypatch):
    mgr = _make_manager(monkeypatch)
    mgr.update("low_light", True, now=0.0)          # low_light channel busy
    mgr.update("drowsy", True, now=0.05)            # drowsy pre-empts it
    low_light_chan = mgr._channels["low_light"]
    assert low_light_chan.stops >= 1


def test_lower_priority_does_not_preempt_higher(monkeypatch):
    mgr = _make_manager(monkeypatch)
    mgr.update("drowsy", True, now=0.0)             # drowsy channel busy
    mgr.update("low_light", True, now=0.05)         # low_light must yield
    drowsy_chan = mgr._channels["drowsy"]
    assert drowsy_chan.stops == 0


def test_same_type_beeps_do_not_stack(monkeypatch):
    mgr = _make_manager(monkeypatch, AudioConfig(
        look_away=AudioAlertConfig(priority=60, repeat_seconds=1.0,
                                   min_seconds=0.0, volume=0.8,
                                   frequency_hz=660.0),
    ))
    mgr.update("look_away", True, now=0.0)
    mgr.update("look_away", True, now=1.5)          # repeat interval elapsed
    chan = mgr._channels["look_away"]
    assert chan.plays == 2
    assert chan.stops >= 1                           # previous stopped first


def test_log_only_mode_when_no_backend(monkeypatch):
    monkeypatch.setitem(sys.modules, "pygame", None)  # import fails
    from drowsiness.alarms.audio_manager import AudioManager
    mgr = AudioManager(AudioConfig(), None, _sync=True)
    assert mgr.update("drowsy", True, now=0.0) is True
    assert mgr.beeps_emitted == 1                    # counted, not crashed
    assert mgr.last_event() is None


def test_manual_start_and_stop(monkeypatch):
    mgr = _make_manager(monkeypatch)
    mgr.start("look_away")
    assert mgr.state_of("look_away") == "ACTIVE"
    # Forced ACTIVE survives a False condition frame until stop().
    assert mgr.update("look_away", False, now=1000.0) is True
    mgr.stop("look_away")
    assert mgr.state_of("look_away") == "INACTIVE"
    mgr.stop_all()


def test_trigger_returns_event_and_counts_beep(monkeypatch):
    mgr = _make_manager(monkeypatch)
    ev = mgr.trigger("manual_test", 0.0)
    assert ev.reason == "manual_test"
    assert mgr.last_event() is ev
    assert mgr.beeps_emitted == 1
    assert mgr.state_of("drowsy") == "INACTIVE"      # compat trigger has no state


# ---------------------------------------------------------------------------
# Tracker integration: yawn confirm event + LOW_LIGHT hysteresis
# ---------------------------------------------------------------------------
def _det(t, *, yawning=False, brightness=128.0, sharpness=500.0,
         straightness=0.95):
    return Detection(timestamp=t, face_found=True, ear=0.3, mar=0.9 if yawning else 0.1,
                     eyes_closed=False, yawning=yawning,
                     mean_brightness=brightness, sharpness=sharpness,
                     face_straightness=straightness)


def test_yawn_just_confirmed_fires_once_within_half_second():
    cfg = Config()
    tr = DrowsinessTracker(cfg)
    confirmed_frames = []
    for i in range(10):
        st = tr.update(_det(t=i * (1 / 15), yawning=True))
        if st.yawn_just_confirmed:
            confirmed_frames.append(i)
    # 0.45 s confirm at 15 FPS = frame ~7; the event fires exactly once.
    assert confirmed_frames == [7], f"confirm fired on frames {confirmed_frames}"
    st = tr.update(_det(t=10 * (1 / 15), yawning=False))
    assert st.yawn_just_confirmed is False
    assert st.yawn_count_window == 1                 # one yawn counted


def test_low_light_hysteresis_enters_below_60_exits_above_65():
    cfg = Config(lighting=LightingConfig(min_brightness=60.0,
                                         recover_brightness=65.0,
                                         consecutive_frames=3))
    tr = DrowsinessTracker(cfg)
    for i in range(3):
        tr.update(_det(t=i * 0.1, brightness=55.0))
    st = tr.update(_det(t=0.3, brightness=55.0))
    assert st.low_light_alarm
    # 62 is above the entry threshold but below recovery — must NOT clear.
    st = tr.update(_det(t=0.4, brightness=62.0))
    assert st.low_light_alarm
    # Above recover_brightness — clears.
    st = tr.update(_det(t=0.5, brightness=70.0))
    assert not st.low_light_alarm


def test_look_away_fires_after_one_second_sustained():
    cfg = Config(distraction=DistractionConfig(
        look_away_straightness=0.58, look_away_seconds=1.0))
    tr = DrowsinessTracker(cfg)
    t0 = 100.0
    st = None
    for i in range(13):  # 13 frames at 15 FPS ≈ 0.87 s — not yet
        st = tr.update(_det(t=t0 + i / 15, straightness=0.3))
    assert not st.looking_away_alarm
    for i in range(13, 20):  # passes 1.0 s
        st = tr.update(_det(t=t0 + i / 15, straightness=0.3))
    assert st.looking_away_alarm
    # Facing straight again clears immediately.
    st = tr.update(_det(t=t0 + 20 / 15, straightness=0.95))
    assert not st.looking_away_alarm
