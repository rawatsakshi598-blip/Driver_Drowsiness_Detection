"""Unified, non-blocking audio manager for ALL alert types.

Every alert in the system routes through this single pipeline — no alert
is allowed to call ``pygame`` / ``winsound`` directly (the legacy
:class:`SoundAlarm` remains only for backwards compatibility and tests).

Design
------
* **Non-blocking**: beep decisions are enqueued onto a priority-sorted
  pending table; a single daemon worker thread plays them.  The video
  loop never waits on audio.  (With ``_sync=True`` — used by tests — the
  play happens inline so behaviour is deterministic.)
* **Priority**: ``drowsy > look_away > yawn > low_light`` (configurable).
  When several alerts are pending, only the highest-priority one plays;
  lower-priority ones are dropped.  Before a sound plays, any *sounding*
  lower-priority channels are stopped (pre-emption).  A lower-priority
  alert never blocks or delays a higher-priority one.  Same-type beeps
  stop the previous one first — no stacking, no overlapping audio.
* **Per-type settings** (priority, ``repeat_seconds``, ``min_seconds``,
  volume, tone frequency) live in one ``audio:`` CONFIG block in
  config.yaml — see :class:`~drowsiness.config.AudioConfig`.
* **State machine**: each alert type is ``INACTIVE`` -> ``PENDING``
  (condition true but under its ``min_seconds`` hold-off) -> ``ACTIVE``
  (beeps on entry and every ``repeat_seconds`` while it stays ACTIVE).
  The condition clearing at any point returns the type to ``INACTIVE``.
  Note: for most types the tracker already gates the condition in time
  (e.g. ``distraction.look_away_seconds``), so ``min_seconds`` defaults
  to 0; raise it to add an extra hold-off.
* **Backends**: pygame.mixer everywhere (cross-platform, non-blocking);
  ``winsound`` as a Windows fallback (its blocking Beep runs inside the
  worker thread, so the main loop still never waits); log-only mode if
  neither is available — audio failure never affects detection.
* **Tones** are synthesised with numpy at startup (one PCM buffer per
  alert type).  DROWSY plays ``alarm.sound_path`` when the file exists.
* **DEBUG**: set ``audio.debug: true`` for one log line per transition
  (``ALERT_TRANSITION type=... -> ... value=... threshold=... t=...``).
  Off by default.

Public API (per spec): ``start(alert_type)``, ``stop(alert_type)``,
``stop_all()``, plus SoundAlarm-compatible ``trigger()`` / ``reset()`` /
``is_active()`` / ``last_event()`` so existing call sites keep working.
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..config import AlarmConfig, AudioConfig
from .base import Alarm, AlarmEvent

log = logging.getLogger(__name__)

TYPE_DROWSY = "drowsy"
TYPE_LOOK_AWAY = "look_away"
TYPE_YAWN = "yawn"
TYPE_LOW_LIGHT = "low_light"
ALERT_TYPES = (TYPE_DROWSY, TYPE_LOOK_AWAY, TYPE_YAWN, TYPE_LOW_LIGHT)

_INACTIVE = "INACTIVE"
_PENDING = "PENDING"
_ACTIVE = "ACTIVE"

# Beep patterns per alert type: list of (tone_seconds, gap_seconds).
# DROWSY is only used when data/alarm.mp3 is missing.
_PATTERNS = {
    TYPE_DROWSY: [(0.10, 0.06), (0.10, 0.06), (0.10, 0.0)],   # urgent triple
    TYPE_LOOK_AWAY: [(0.13, 0.09), (0.13, 0.0)],             # medium double
    TYPE_YAWN: [(0.07, 0.06), (0.07, 0.0)],                  # short double
    TYPE_LOW_LIGHT: [(0.30, 0.0)],                           # single soft
}

_SAMPLE_RATE = 22050


class AudioManager(Alarm):
    """Single shared audio pipeline for all alert types (see module doc)."""

    def __init__(self, cfg: AudioConfig,
                 alarm_cfg: Optional[AlarmConfig] = None,
                 *, _sync: bool = False):
        self._cfg = cfg
        self._alarm_cfg = alarm_cfg
        self._state = {t: _INACTIVE for t in ALERT_TYPES}
        self._condition_since: dict[str, Optional[float]] = {t: None for t in ALERT_TYPES}
        self._last_beep: dict[str, float] = {t: 0.0 for t in ALERT_TYPES}
        self._forced: dict[str, bool] = {t: False for t in ALERT_TYPES}
        self._pending: dict[str, bool] = {}
        self._cond = threading.Condition()
        self._state_lock = threading.Lock()
        self._stop_worker = False
        self._beeps_emitted = 0
        self._last_event: Optional[AlarmEvent] = None
        self._sync = _sync
        # Serialises ALL SDL/mixer access onto one path; SDL is not
        # reliably thread-safe, so every mixer call happens under this lock.
        self._mixer_lock = threading.Lock()
        # Python-side "channel busy until" timestamps (monotonic-ish wall
        # clock) — used instead of SDL get_busy() so the worker never
        # touches the mixer just to *wait*.
        self._busy_until = {t: 0.0 for t in ALERT_TYPES}
        self._durations = {t: self._pattern_duration(t) for t in ALERT_TYPES}
        # Backend state
        self._mixer_ok = False
        self._winsound = None
        self._pygame = None
        self._sounds: dict[str, object] = {}
        self._channels: dict[str, object] = {}
        if self._cfg.enabled:
            self._init_output()
            if not self._sync:
                self._worker = threading.Thread(
                    target=self._worker_loop, name="alert-audio", daemon=True,
                )
                self._worker.start()
        else:
            log.info("AudioManager: all alert audio disabled (audio.enabled=false)")

    # ------------------------------------------------------------------ #
    # Backend initialisation (once, at startup)
    # ------------------------------------------------------------------ #
    def _init_output(self) -> None:
        # Preferred: pygame.mixer everywhere (non-blocking, channels for
        # per-type pre-emption, plays our numpy-synthesised PCM).
        try:
            import pygame
            pygame.mixer.init(frequency=_SAMPLE_RATE, size=-16, channels=1, buffer=512)
            pygame.mixer.set_num_channels(max(8, len(ALERT_TYPES)))
            self._pygame = pygame
            for i, alert_type in enumerate(ALERT_TYPES):
                self._sounds[alert_type] = self._build_sound(alert_type)
                self._channels[alert_type] = pygame.mixer.Channel(i)
            with self._mixer_lock:
                for t in ALERT_TYPES:
                    try:
                        d = float(self._sounds[t].get_length())
                        if d > 0:
                            self._durations[t] = d
                    except Exception:
                        pass  # keep the pattern-duration fallback
            self._mixer_ok = True
            log.info("AudioManager: pygame mixer initialised (%d alert channels) "
                     "(durations=%s)", len(ALERT_TYPES),
                     {t: round(self._durations[t], 2) for t in ALERT_TYPES})
            return
        except Exception as exc:
            log.warning("AudioManager: pygame mixer unavailable (%s)", exc)
        # Windows fallback: winsound (blocking calls run in the worker thread).
        try:
            import platform
            if platform.system() == "Windows":
                import winsound
                self._winsound = winsound
                log.info("AudioManager: using winsound fallback")
                return
        except Exception as exc:
            log.warning("AudioManager: winsound fallback unavailable (%s)", exc)
        log.warning(
            "AudioManager: no audio backend available — beeps will be "
            "logged only. Detection is unaffected."
        )

    def _pattern_duration(self, alert_type: str) -> float:
        return sum(on_s + gap_s for on_s, gap_s in _PATTERNS[alert_type])

    def _settings(self, alert_type: str):
        return getattr(self._cfg, alert_type)

    def _build_sound(self, alert_type: str):
        """Build (or load) the PCM sound for one alert type. Called once."""
        pygame = self._pygame
        # DROWSY: reuse the configured alarm file when it exists.
        if alert_type == TYPE_DROWSY and self._alarm_cfg is not None:
            path = Path(self._alarm_cfg.sound_path)
            try:
                if path.exists():
                    snd = pygame.mixer.Sound(str(path))
                    snd.set_volume(self._settings(alert_type).volume)
                    log.info("AudioManager: DROWSY sound = %s", path)
                    return snd
                log.info("AudioManager: %s not found — synthesising tone", path)
            except Exception as exc:
                log.warning("AudioManager: could not load %s (%s) — synthesising", path, exc)
        return pygame.mixer.Sound(buffer=self._synth_pcm(alert_type).tobytes())

    def _synth_pcm(self, alert_type: str):
        import numpy as np

        s = self._settings(alert_type)
        freq = float(s.frequency_hz)
        amplitude = int(32767 * 0.9 * max(0.0, min(1.0, s.volume)))
        pcm_parts = []
        for on_s, gap_s in _PATTERNS[alert_type]:
            n = max(1, int(_SAMPLE_RATE * on_s))
            t = np.arange(n, dtype=np.float32) / _SAMPLE_RATE
            tone = amplitude * np.sin(2 * np.pi * freq * t)
            # 8 ms attack ramp + exponential-ish decay over last third.
            attack = max(1, int(_SAMPLE_RATE * 0.008))
            tone[:attack] *= np.linspace(0.0, 1.0, attack, dtype=np.float32)
            decay_start = int(n * 0.66)
            if n > decay_start:
                tone[decay_start:] *= np.linspace(1.0, 0.0, n - decay_start, dtype=np.float32)
            pcm_parts.append(tone.astype(np.int16))
            if gap_s > 0:
                pcm_parts.append(np.zeros(int(_SAMPLE_RATE * gap_s), dtype=np.int16))
        return np.concatenate(pcm_parts)

    # ------------------------------------------------------------------ #
    # Worker (daemon thread) — priority pre-emption, no audio overlap
    # ------------------------------------------------------------------ #
    def _worker_loop(self) -> None:
        while True:
            with self._cond:
                self._cond.wait(timeout=0.25)
                pending = self._pending
                self._pending = {}
                stop = self._stop_worker
            if pending:
                # Play every pending alert, highest priority first.  Each
                # beep still pre-empts any lower-priority *sounding*
                # channel.  A started batch always drains to completion;
                # the stop flag is only honoured BETWEEN batches, so
                # shutdown() never loses or half-plays an alert.
                for alert_type in sorted(
                    pending, key=lambda t: self._settings(t).priority,
                    reverse=True,
                ):
                    self._wait_for_higher_priority_channels(alert_type)
                    self._play(alert_type)
                continue
            if stop:
                return

    def _wait_for_higher_priority_channels(self, alert_type: str,
                                           cap_seconds: float = 2.0) -> None:
        """Worker-thread only: wait (capped) for higher-priority channels.

        Keeps audio non-overlapping while never letting a *lower* alert
        block a *higher* one (higher ones play first in the drain loop).
        """
        prio = self._settings(alert_type).priority
        deadline = time.time() + cap_seconds
        while time.time() < deadline:
            wait_for = max(
                (self._busy_until[o] for o in ALERT_TYPES
                 if self._settings(o).priority > prio),
                default=0.0,
            )
            if wait_for <= time.time():
                return
            time.sleep(min(0.02, wait_for - time.time() + 0.005))

    def _pick_highest_priority(self, pending) -> str:
        """Pure helper (unit-testable): pick the winning alert type."""
        return max(pending, key=lambda t: self._settings(t).priority)

    def _preempt_lower_priority(self, alert_type: str,
                                now: Optional[float] = None) -> None:
        now = time.time() if now is None else now
        prio = self._settings(alert_type).priority
        for other in ALERT_TYPES:
            if other == alert_type:
                continue
            if self._settings(other).priority >= prio:
                continue  # same/higher priority: never touched
            if self._busy_until[other] <= now:
                continue  # not sounding
            chan = self._channels.get(other)
            try:
                with self._mixer_lock:
                    if chan is not None:
                        chan.stop()
            except Exception as exc:  # pragma: no cover - defensive
                log.debug("AudioManager: stop(%s) failed: %s", other, exc)
            self._busy_until[other] = 0.0

    def _play(self, alert_type: str) -> None:
        """Play one beep (or pattern). Never raises."""
        try:
            now = time.time()
            self._preempt_lower_priority(alert_type, now)
            if self._mixer_ok:
                chan = self._channels[alert_type]
                try:
                    with self._mixer_lock:
                        if self._busy_until[alert_type] > now:
                            chan.stop()  # no same-type stacking
                        chan.set_volume(self._settings(alert_type).volume)
                        chan.play(self._sounds[alert_type])
                except Exception:
                    # If the configured mp3 can't play in this environment
                    # (e.g. headless dummy SDL), degrade DROWSY to the
                    # synthesised tone rather than going silent.
                    if alert_type != TYPE_DROWSY:
                        raise
                    log.warning(
                        "AudioManager: DROWSY mp3 failed to play — "
                        "falling back to synthesised tone",
                    )
                    snd = self._pygame.mixer.Sound(
                        buffer=self._synth_pcm(TYPE_DROWSY).tobytes())
                    snd.set_volume(self._settings(TYPE_DROWSY).volume)
                    self._sounds[TYPE_DROWSY] = snd
                    self._durations[TYPE_DROWSY] = self._pattern_duration(TYPE_DROWSY)
                    with self._mixer_lock:
                        chan.play(snd)
                self._busy_until[alert_type] = now + self._durations.get(alert_type, 0.5)
            elif self._winsound is not None:
                s = self._settings(alert_type)
                for on_s, gap_s in _PATTERNS[alert_type]:
                    self._winsound.Beep(int(s.frequency_hz), max(1, int(on_s * 1000)))
                    if gap_s > 0:
                        time.sleep(gap_s)
            else:  # log-only mode
                log.info("AudioManager: [no backend] beep for %s", alert_type)
            self._beeps_emitted += 1
        except Exception:
            log.exception("AudioManager: failed to play %s (detection unaffected)", alert_type)

    # ------------------------------------------------------------------ #
    # Per-frame state machine: INACTIVE -> PENDING -> ACTIVE
    # ------------------------------------------------------------------ #
    def update(self, alert_type: str, condition: bool, *,
               value: Optional[float] = None,
               threshold: Optional[float] = None,
               now: Optional[float] = None) -> bool:
        """Feed the current frame's condition for ``alert_type``.

        Returns True while the alert type is ACTIVE.  Cheap (no I/O);
        safe to call every frame from the video loop.
        """
        if alert_type not in self._state:
            raise ValueError(f"Unknown alert type: {alert_type!r}")
        now = time.time() if now is None else float(now)
        s = self._settings(alert_type)
        if not self._cfg.enabled or not s.enabled:
            condition = False
        with self._state_lock:
            state = self._state[alert_type]
            if condition or self._forced[alert_type]:
                if state == _INACTIVE:
                    state = _PENDING
                    self._condition_since[alert_type] = now
                    self._state[alert_type] = state
                    self._dbg(alert_type, state, value, threshold, now)
                if state == _PENDING:
                    since = self._condition_since[alert_type]
                    if since is None or (now - since) >= s.min_seconds:
                        state = _ACTIVE
                        self._state[alert_type] = state
                        self._dbg(alert_type, state, value, threshold, now)
                        self._emit(alert_type, now, value, threshold)
                elif state == _ACTIVE:
                    if now - self._last_beep[alert_type] >= s.repeat_seconds:
                        self._emit(alert_type, now, value, threshold)
            else:
                if state != _INACTIVE:
                    self._state[alert_type] = _INACTIVE
                    self._condition_since[alert_type] = None
                    self._stop_channel(alert_type)  # silence promptly on recovery
                    self._dbg(alert_type, _INACTIVE, value, threshold, now)
            return self._state[alert_type] == _ACTIVE

    def start(self, alert_type: str) -> None:
        """Force ``alert_type`` ACTIVE and beep immediately (manual override)."""
        if alert_type not in self._state:
            raise ValueError(f"Unknown alert type: {alert_type!r}")
        now = time.time()
        with self._state_lock:
            self._forced[alert_type] = True
            self._condition_since[alert_type] = self._condition_since[alert_type] or now
            prev = self._state[alert_type]
            self._state[alert_type] = _ACTIVE
            if prev != _ACTIVE:
                self._dbg(alert_type, _ACTIVE, None, None, now)
        self._emit(alert_type, now, None, None)

    def stop(self, alert_type: str) -> None:
        """Clear ``alert_type`` (manual override off); silences it."""
        if alert_type not in self._state:
            raise ValueError(f"Unknown alert type: {alert_type!r}")
        with self._state_lock:
            self._forced[alert_type] = False
            self._state[alert_type] = _INACTIVE
            self._condition_since[alert_type] = None
        self._stop_channel(alert_type)
        self._dbg(alert_type, _INACTIVE, None, None, time.time())

    def stop_all(self) -> None:
        """Silence and clear every alert type."""
        for alert_type in ALERT_TYPES:
            self.stop(alert_type)

    def shutdown(self) -> None:
        """Stop the worker thread, drain it, then release the mixer."""
        self.stop_all()
        with self._cond:
            self._stop_worker = True
            self._cond.notify()
        worker = getattr(self, "_worker", None)
        if worker is not None:
            worker.join(timeout=3.0)
        if self._mixer_ok and self._pygame is not None:
            try:
                with self._mixer_lock:
                    self._pygame.mixer.quit()
            except Exception as exc:  # pragma: no cover - defensive
                log.debug("AudioManager: mixer quit failed: %s", exc)

    # ------------------------------------------------------------------ #
    # SoundAlarm-compatible API (so existing call sites keep working)
    # ------------------------------------------------------------------ #
    def trigger(self, reason: str, metric_value: float = 0.0) -> AlarmEvent:
        """Force a DROWSY-priority beep and return the event (test/compat)."""
        event = AlarmEvent(
            timestamp=datetime.now(), reason=reason, metric_value=metric_value,
        )
        self._last_event = event
        if self._cfg.enabled:
            log.warning("ALARM TRIGGERED: %s (value=%.3f)", reason, metric_value)
            self._emit(TYPE_DROWSY, time.time(), metric_value, None)
        else:
            log.warning("ALARM TRIGGERED (audio disabled): %s (value=%.3f)",
                        reason, metric_value)
        return event

    def reset(self) -> None:
        """Compatibility shim for SoundAlarm.reset() — clears all alerts."""
        self.stop_all()

    def is_active(self) -> bool:
        return any(state == _ACTIVE for state in self._state.values())

    def last_event(self) -> Optional[AlarmEvent]:
        return self._last_event

    @property
    def beeps_emitted(self) -> int:
        """Number of beeps actually played (decision counter for tests)."""
        return self._beeps_emitted

    def state_of(self, alert_type: str) -> str:
        return self._state.get(alert_type, _INACTIVE)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _emit(self, alert_type: str, now: float,
              value: Optional[float], threshold: Optional[float]) -> None:
        self._last_beep[alert_type] = now
        if self._sync:
            self._play(alert_type)
            return
        with self._cond:
            self._pending[alert_type] = True
            self._cond.notify()

    def _stop_channel(self, alert_type: str) -> None:
        self._busy_until[alert_type] = 0.0
        if not self._mixer_ok:
            return
        chan = self._channels.get(alert_type)
        try:
            with self._mixer_lock:
                if chan is not None:
                    chan.stop()
        except Exception as exc:  # pragma: no cover - defensive
            log.debug("AudioManager: stop(%s) failed: %s", alert_type, exc)

    def _dbg(self, alert_type: str, state: str,
             value: Optional[float], threshold: Optional[float],
             now: float) -> None:
        """One log line per transition — only when audio.debug is true."""
        if not self._cfg.debug:
            return
        val = "n/a" if value is None else f"{value:.3f}"
        thr = "n/a" if threshold is None else f"{threshold:.3f}"
        log.info(
            "ALERT_TRANSITION type=%s -> %s value=%s threshold=%s t=%.3f",
            alert_type, state, val, thr, now,
        )
