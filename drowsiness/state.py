"""Stateful drowsiness tracking — converts per-frame ``Detection``s into
fatigue state and decides when to raise alarms.

This is the temporal brain of the system.  Each ``Detection`` is a
single-frame snapshot; this class integrates them across time, computing:

* Consecutive closed-eye frames (immediate alarm trigger)
* PERCLOS — % eye closure over a rolling window (NHTSA standard)
* Yawn rate — yawns per minute (rolling window)
* Alarm cooldown enforcement

All windows are implemented as monotonic deques so we never re-scan the
whole history.
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional

from .config import Config
from .detectors.base import Detection

log = logging.getLogger(__name__)


@dataclass
class FatigueState:
    """Snapshot of the fatigue state machine — exposed via the status API."""
    # Consecutive closed-eye frames right now.
    consecutive_closed_frames: int = 0
    # Consecutive yawning frames right now.
    consecutive_yawn_frames: int = 0
    # PERCLOS over the rolling window.
    perclos: float = 0.0
    # Yawn count over the rolling window.
    yawn_count_window: int = 0
    # Yawn rate (yawns/min) over the rolling window.
    yawn_rate_per_min: float = 0.0
    # Whether the eye-alarm condition is currently firing.
    eye_alarm: bool = False
    # Whether the yawn-alarm condition is currently firing.
    yawn_alarm: bool = False
    # Whether the PERCLOS-alarm condition is currently firing.
    perclos_alarm: bool = False
    # Latest EAR/MAR values for monitoring.
    latest_ear: Optional[float] = None
    latest_mar: Optional[float] = None
    # Total frames processed.
    frames_processed: int = 0
    # Wall-clock start time.
    started_at: float = field(default_factory=time.time)
    # FPS based on frames processed.
    fps: float = 0.0

    @property
    def any_alarm(self) -> bool:
        return self.eye_alarm or self.yawn_alarm or self.perclos_alarm

    @property
    def alarm_reason(self) -> Optional[str]:
        if self.eye_alarm:
            return "eye_closure"
        if self.yawn_alarm:
            return "yawn_burst"
        if self.perclos_alarm:
            return "perclos"
        return None


class DrowsinessTracker:
    """Stateful tracker — receives per-frame ``Detection``s and emits alarms.

    Threading: not thread-safe; intended to be called from the orchestrator's
    main loop.
    """

    def __init__(self, config: Config):
        self._cfg = config
        self._state = FatigueState()
        # Rolling window of (timestamp, eyes_closed) tuples for PERCLOS.
        window_s = config.eye.perclos_window_seconds
        self._eye_window: Deque[tuple[float, bool]] = deque()
        # Rolling window of yawn events (timestamps of yawn onsets only).
        yawn_window_s = config.mouth.yawn_window_seconds
        self._yawn_window: Deque[float] = deque()
        self._last_yawn_active = False

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    @property
    def state(self) -> FatigueState:
        return self._state

    def update(self, detection: Detection) -> FatigueState:
        """Update the state machine with one frame's detection."""
        now = detection.timestamp
        cfg = self._cfg
        st = self._state
        st.frames_processed += 1
        st.latest_ear = detection.ear
        st.latest_mar = detection.mar

        # If no face found, we still want to expire old window entries
        # but we should NOT clear state (the driver may be leaning away
        # briefly).  Just decay the windows.
        if not detection.face_found:
            self._expire_windows(now)
            st.fps = self._compute_fps()
            return st

        # ---------------- Eye closure state ----------------
        if detection.eyes_closed:
            st.consecutive_closed_frames += 1
        else:
            st.consecutive_closed_frames = 0
        self._eye_window.append((now, detection.eyes_closed))
        self._expire_windows(now)
        st.perclos = self._compute_perclos(now)

        # ---------------- Yawn state ----------------
        if detection.yawning:
            st.consecutive_yawn_frames += 1
        else:
            # Detect yawn onset only (rising edge) so we count yawns, not frames.
            if st.consecutive_yawn_frames >= cfg.mouth.yawn_frame_threshold:
                self._yawn_window.append(now)
            st.consecutive_yawn_frames = 0
        self._expire_yawn_window(now)
        st.yawn_count_window = len(self._yawn_window)
        window_minutes = cfg.mouth.yawn_window_seconds / 60.0
        st.yawn_rate_per_min = (
            st.yawn_count_window / window_minutes if window_minutes > 0 else 0.0
        )

        # ---------------- Alarms ----------------
        st.eye_alarm = st.consecutive_closed_frames >= cfg.eye.closed_frame_threshold
        st.yawn_alarm = st.yawn_rate_per_min >= cfg.mouth.yawn_rate_per_minute
        st.perclos_alarm = st.perclos >= cfg.eye.perclos_threshold

        st.fps = self._compute_fps()
        return st

    def reset(self) -> None:
        """Reset all counters (e.g. when the driver is changed)."""
        self._state = FatigueState()
        self._eye_window.clear()
        self._yawn_window.clear()

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _expire_windows(self, now: float) -> None:
        window = self._cfg.eye.perclos_window_seconds
        while self._eye_window and (now - self._eye_window[0][0]) > window:
            self._eye_window.popleft()

    def _expire_yawn_window(self, now: float) -> None:
        window = self._cfg.mouth.yawn_window_seconds
        while self._yawn_window and (now - self._yawn_window[0]) > window:
            self._yawn_window.popleft()

    def _compute_perclos(self, now: float) -> float:
        if not self._eye_window:
            return 0.0
        closed = sum(1 for _, c in self._eye_window if c)
        return closed / len(self._eye_window)

    def _compute_fps(self) -> float:
        elapsed = time.time() - self._state.started_at
        if elapsed <= 0:
            return 0.0
        return self._state.frames_processed / elapsed
