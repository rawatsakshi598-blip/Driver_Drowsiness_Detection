"""Sound-based alarm using pygame's mixer (no native deps, cross-platform).

Plays ``data/alarm.mp3`` on a background thread so the alarm never blocks
the detection loop.  Cooldown is enforced to avoid retriggering while the
driver is still drowsy.
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from typing import Optional

from ..config import AlarmConfig
from .base import Alarm, AlarmEvent

log = logging.getLogger(__name__)


class SoundAlarm(Alarm):
    """Plays an audio alarm with cooldown."""

    def __init__(self, config: AlarmConfig):
        self._cfg = config
        self._active = False
        self._last_event: Optional[AlarmEvent] = None
        self._last_trigger_ts: float = 0.0
        self._mixer = None
        self._mixer_lock = threading.Lock()
        self._initialised = False

    # ------------------------------------------------------------------ #
    # Lazy init (so importing this module never loads pygame)
    # ------------------------------------------------------------------ #
    def _init_mixer(self) -> None:
        if self._initialised or not self._cfg.enabled:
            self._initialised = True
            return
        try:
            # pygame's mixer is more reliable than playsound across
            # platforms and supports non-blocking playback.
            import pygame  # noqa: WPS433
            pygame.mixer.init()
            self._mixer = pygame.mixer
            # Load the sound once; cheap to replay.
            self._sound = self._mixer.Sound(self._cfg.sound_path)
            log.info("Alarm sound loaded: %s", self._cfg.sound_path)
        except Exception as exc:
            log.warning(
                "SoundAlarm disabled — could not init mixer (%s). "
                "Alarms will be logged only.", exc,
            )
            self._mixer = None
        self._initialised = True

    # ------------------------------------------------------------------ #
    # Alarm interface
    # ------------------------------------------------------------------ #
    def trigger(self, reason: str, metric_value: float = 0.0) -> AlarmEvent:
        now = time.time()
        # Cooldown: don't retrigger within the cooldown window.
        if now - self._last_trigger_ts < self._cfg.cooldown_seconds and self._active:
            log.debug("Alarm trigger suppressed by cooldown (reason=%s)", reason)
            return self._last_event  # type: ignore[return-value]
        self._init_mixer()
        self._last_trigger_ts = now
        self._active = True
        event = AlarmEvent(
            timestamp=datetime.now(),
            reason=reason,
            metric_value=metric_value,
        )
        self._last_event = event
        log.warning("ALARM TRIGGERED: %s (value=%.3f)", reason, metric_value)
        # Play the sound in a daemon thread — never block the main loop.
        if self._mixer is not None:
            t = threading.Thread(target=self._play_async, daemon=True)
            t.start()
        return event

    def _play_async(self) -> None:
        try:
            self._sound.play()
            # Block this background thread until the sound finishes so we
            # don't queue overlapping plays.
            while self._mixer.get_busy():
                time.sleep(0.05)
        except Exception as exc:  # pragma: no cover - defensive
            log.exception("SoundAlarm playback failed: %s", exc)

    def reset(self) -> None:
        if self._active:
            log.info("Alarm reset")
        self._active = False

    def is_active(self) -> bool:
        return self._active

    def last_event(self) -> Optional[AlarmEvent]:
        return self._last_event
