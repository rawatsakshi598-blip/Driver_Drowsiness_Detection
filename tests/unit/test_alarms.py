"""Unit tests for the alarm subsystem.

The tests use a non-playing ``SoundAlarm`` (``enabled=false``) so they
don't make any sound during the test run, while still exercising all the
trigger/cooldown/is_active logic.
"""
from __future__ import annotations

import time

import pytest

from drowsiness.alarms.sound_alarm import SoundAlarm
from drowsiness.config import AlarmConfig


def _disabled_alarm(cooldown=2.0) -> SoundAlarm:
    """Build an alarm with sound disabled (no audio output during tests)."""
    return SoundAlarm(AlarmConfig(enabled=False, cooldown_seconds=cooldown))


def test_initial_state_is_inactive():
    al = _disabled_alarm()
    assert not al.is_active()
    assert al.last_event() is None


def test_trigger_activates_alarm():
    al = _disabled_alarm()
    ev = al.trigger("eye_closure", 0.05)
    assert al.is_active()
    assert ev is not None
    assert ev.reason == "eye_closure"
    assert ev.metric_value == 0.05


def test_reset_clears_active_state():
    al = _disabled_alarm()
    al.trigger("test")
    assert al.is_active()
    al.reset()
    assert not al.is_active()


def test_cooldown_suppresses_retrigger():
    """Within the cooldown window, a second trigger returns the same event."""
    al = _disabled_alarm(cooldown=10.0)
    ev1 = al.trigger("first", 0.1)
    time.sleep(0.05)
    ev2 = al.trigger("second", 0.2)
    assert ev1 is ev2  # same event, second was suppressed


def test_cooldown_releases_after_window():
    """After cooldown_seconds, a new trigger should produce a new event."""
    al = _disabled_alarm(cooldown=0.2)
    ev1 = al.trigger("first")
    time.sleep(0.3)
    ev2 = al.trigger("second")
    assert ev1 is not ev2
    assert ev2.reason == "second"


def test_last_event_records_latest():
    al = _disabled_alarm()
    al.trigger("a")
    # Wait beyond cooldown (default 2s is OK), then trigger again.
    time.sleep(2.1)
    al.trigger("b")
    assert al.last_event() is not None
    assert al.last_event().reason == "b"
