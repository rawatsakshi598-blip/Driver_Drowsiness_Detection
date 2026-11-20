"""Stateful drowsiness tracking — converts per-frame ``Detection``s into
fatigue state and decides when to raise alarms.

This is the temporal brain of the system.  Each ``Detection`` is a
single-frame snapshot; this class integrates them across time, computing:

* Consecutive closed-eye frames (immediate alarm trigger)
* PERCLOS — % eye closure over a rolling window (NHTSA standard)
* Yawn rate — yawns per minute (rolling window)
* **Prolonged yawn duration** — a single yawn held too long (new)
* **Face presence / distraction** — face missing or looking away (new)
* **Lighting & focus environment checks** — too dark, washed out, blurry (new)
* Alarm cooldown enforcement

All windows are implemented as monotonic deques so we never re-scan the
whole history.

Alerts (``FatigueState.alerts``) are edge-triggered log messages plus an
active-alert list that the orchestrator renders on the HUD and exposes via
the status API — so every issue is surfaced the moment it appears.
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional

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
    # True ONLY on the frame a yawn becomes confirmed (rising-edge event
    # used by the audio manager; cleared on every other frame).
    yawn_just_confirmed: bool = False
    # Latest EAR/MAR values for monitoring.
    latest_ear: Optional[float] = None
    latest_mar: Optional[float] = None
    # Latest frame-level environment signals for monitoring.
    latest_brightness: Optional[float] = None
    latest_sharpness: Optional[float] = None
    latest_face_straightness: Optional[float] = None
    # Seconds the face has been continuously missing (0 = face visible now).
    consecutive_no_face_seconds: float = 0.0
    # New environment / attention alarms.
    prolonged_yawn_alarm: bool = False
    face_missing_alarm: bool = False
    looking_away_alarm: bool = False
    lighting_alarm: bool = False
    low_light_alarm: bool = False
    bright_light_alarm: bool = False
    out_of_focus_alarm: bool = False
    # Currently-active alerts.  Each entry:
    # {"code": str, "severity": "critical"|"warning", "message": str, "value": float|None}
    alerts: List[dict] = field(default_factory=list)
    # Total frames processed.
    frames_processed: int = 0
    # Wall-clock start time.
    started_at: float = field(default_factory=time.time)
    # FPS based on frames processed.
    fps: float = 0.0

    @property
    def any_alarm(self) -> bool:
        """True only for drowsiness alarms (unchanged legacy semantics)."""
        return self.eye_alarm or self.yawn_alarm or self.perclos_alarm

    @property
    def any_alert(self) -> bool:
        """True when any of the new distraction/environment alerts fire."""
        return (
            self.prolonged_yawn_alarm
            or self.face_missing_alarm
            or self.looking_away_alarm
            or self.lighting_alarm
            or self.out_of_focus_alarm
        )

    @property
    def any_issue(self) -> bool:
        """True when anything (drowsiness alarm or alert) is active."""
        return self.any_alarm or self.any_alert

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
        # --- New attention / environment bookkeeping ---
        # Wall-clock time the face was last seen (None = never seen).
        self._last_face_seen_at: Optional[float] = None
        # Wall-clock time the *current* yawn started (None = not yawning).
        self._yawn_started_at: Optional[float] = None
        # True once the current yawn has been confirmed (counted once).
        self._yawn_confirmed = False
        # Wall-clock time the current look-away began (None = facing road).
        self._look_away_since: Optional[float] = None
        self._dark_frames = 0
        self._bright_frames = 0
        self._blurry_frames = 0
        # Codes currently logged as active (for edge-triggered logging).
        self._active_alert_codes: set[str] = set()

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
        st.yawn_just_confirmed = False
        st.latest_ear = detection.ear
        st.latest_mar = detection.mar
        st.latest_brightness = detection.mean_brightness
        st.latest_sharpness = detection.sharpness
        st.latest_face_straightness = detection.face_straightness

        # ---------------- Frame-level environment checks ----------------
        self._update_lighting(st, detection)
        self._update_focus(st, detection)

        if not detection.face_found:
            # No face in this frame: the driver may have leaned away or
            # left the wheel.  We expire old window entries, track how
            # long the face has been missing, and raise a FACE_NOT_VISIBLE
            # alert once the configured grace period elapses.
            self._expire_windows(now)
            last_seen = self._last_face_seen_at
            if last_seen is None:
                # Both branches must use the same clock.  detection.timestamp
                # and started_at are both wall-clock in production; if a
                # synthetic test feeds tiny timestamps we clamp at 0.
                last_seen = st.started_at if now >= st.started_at else now
            st.consecutive_no_face_seconds = max(0.0, now - last_seen)
            st.face_missing_alarm = (
                cfg.distraction.enabled
                and st.consecutive_no_face_seconds >= cfg.distraction.no_face_seconds
            )
            # Attention signals that require a face are cleared.
            st.looking_away_alarm = False
            self._look_away_since = None
            self._yawn_confirmed = False
            # Suppress the prolonged-yawn alarm while the face is gone,
            # but keep the yawn-streak bookkeeping paused rather than
            # reset — a transient one-frame detection glitch must not
            # destroy an ongoing yawn streak.
            st.prolonged_yawn_alarm = False
            st.fps = self._compute_fps()
            st.alerts = self._build_alerts(st, now)
            self._log_alert_transitions(st.alerts)
            return st

        # Face is visible — reset presence bookkeeping.
        self._last_face_seen_at = now
        st.consecutive_no_face_seconds = 0.0
        st.face_missing_alarm = False

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
            # Prolonged-yawn / confirm bookkeeping: track when this yawn began.
            if self._yawn_started_at is None:
                self._yawn_started_at = now
            yawn_duration = now - self._yawn_started_at
            # Time-based confirm (fps-independent): yawn confirmed once the
            # mouth has stayed open for yawn_confirm_seconds.  Fires within
            # ~0.5 s of a real yawn and is immune to FPS drops.
            if not self._yawn_confirmed and yawn_duration >= cfg.mouth.yawn_confirm_seconds:
                self._yawn_confirmed = True
                st.yawn_just_confirmed = True
            st.prolonged_yawn_alarm = (
                yawn_duration >= cfg.mouth.max_yawn_seconds
            )
        else:
            # Count one yawn per confirmed event (rising-confirm edge).
            if self._yawn_confirmed:
                self._yawn_window.append(now)
            st.consecutive_yawn_frames = 0
            self._yawn_started_at = None
            self._yawn_confirmed = False
            st.prolonged_yawn_alarm = False
        self._expire_yawn_window(now)
        st.yawn_count_window = len(self._yawn_window)
        window_minutes = cfg.mouth.yawn_window_seconds / 60.0
        st.yawn_rate_per_min = (
            st.yawn_count_window / window_minutes if window_minutes > 0 else 0.0
        )

        # ---------------- Look-away (head yaw) ----------------
        self._update_look_away(st, detection, now)

        # ---------------- Alarms ----------------
        st.eye_alarm = st.consecutive_closed_frames >= cfg.eye.closed_frame_threshold
        st.yawn_alarm = st.yawn_rate_per_min >= cfg.mouth.yawn_rate_per_minute
        st.perclos_alarm = st.perclos >= cfg.eye.perclos_threshold

        st.fps = self._compute_fps()
        st.alerts = self._build_alerts(st, now)
        self._log_alert_transitions(st.alerts)
        return st

    def reset(self) -> None:
        """Reset all counters (e.g. when the driver is changed)."""
        self._state = FatigueState()
        self._eye_window.clear()
        self._yawn_window.clear()
        self._last_face_seen_at = None
        self._yawn_started_at = None
        self._yawn_confirmed = False
        self._look_away_since = None
        self._dark_frames = 0
        self._bright_frames = 0
        self._blurry_frames = 0
        self._active_alert_codes = set()

    # ------------------------------------------------------------------ #
    # Environment checks
    # ------------------------------------------------------------------ #
    def _update_lighting(self, st: FatigueState, detection: Detection) -> None:
        cfg = self._cfg.lighting
        b = detection.mean_brightness
        if not cfg.enabled or b is None:
            self._dark_frames = 0
            self._bright_frames = 0
            st.low_light_alarm = False
            st.bright_light_alarm = False
            st.lighting_alarm = False
            return
        # Hysteresis: once LOW_LIGHT latched, brightness must rise above
        # recover_brightness (not back above min_brightness) to clear —
        # otherwise a frame hovering at 60/61 flickers the alert.
        if st.low_light_alarm:
            if b >= cfg.recover_brightness:
                self._dark_frames = 0
                st.low_light_alarm = False
        else:
            self._dark_frames = (self._dark_frames + 1) if b < cfg.min_brightness else 0
            st.low_light_alarm = self._dark_frames >= cfg.consecutive_frames
        self._bright_frames = (self._bright_frames + 1) if b > cfg.max_brightness else 0
        st.bright_light_alarm = self._bright_frames >= cfg.consecutive_frames
        st.lighting_alarm = st.low_light_alarm or st.bright_light_alarm

    def _update_focus(self, st: FatigueState, detection: Detection) -> None:
        cfg = self._cfg.focus
        s = detection.sharpness
        if not cfg.enabled or s is None:
            self._blurry_frames = 0
            st.out_of_focus_alarm = False
            return
        self._blurry_frames = (self._blurry_frames + 1) if s < cfg.min_sharpness else 0
        st.out_of_focus_alarm = self._blurry_frames >= cfg.consecutive_frames

    def _update_look_away(self, st: FatigueState, detection: Detection,
                          now: float) -> None:
        cfg = self._cfg.distraction
        s = detection.face_straightness
        if not cfg.enabled or s is None:
            self._look_away_since = None
            st.looking_away_alarm = False
            return
        if s < cfg.look_away_straightness:
            # Time-based (fps-independent): fires after look_away_seconds of
            # sustained head-turn, not after N frames.
            if self._look_away_since is None:
                self._look_away_since = now
            st.looking_away_alarm = (now - self._look_away_since) >= cfg.look_away_seconds
        else:
            self._look_away_since = None
            st.looking_away_alarm = False

    # ------------------------------------------------------------------ #
    # Alert construction + edge-triggered logging
    # ------------------------------------------------------------------ #
    def _build_alerts(self, st: FatigueState, now: float) -> List[dict]:
        cfg = self._cfg
        alerts: List[dict] = []
        if st.face_missing_alarm:
            alerts.append({
                "code": "FACE_NOT_VISIBLE",
                "severity": "critical",
                "message": (
                    f"Face not visible for {st.consecutive_no_face_seconds:.1f}s "
                    f"(limit {cfg.distraction.no_face_seconds:.1f}s)"
                ),
                "value": round(st.consecutive_no_face_seconds, 2),
            })
        if st.looking_away_alarm:
            alerts.append({
                "code": "LOOKING_AWAY",
                "severity": "warning",
                "message": "Driver looking away from the road",
                "value": (
                    None if st.latest_face_straightness is None
                    else round(st.latest_face_straightness, 3)
                ),
            })
        if st.prolonged_yawn_alarm:
            duration = (
                max(0.0, now - self._yawn_started_at)
                if self._yawn_started_at is not None else 0.0
            )
            alerts.append({
                "code": "PROLONGED_YAWN",
                "severity": "warning",
                "message": (
                    f"Yawn held for too long (limit {cfg.mouth.max_yawn_seconds:.1f}s)"
                ),
                "value": round(duration, 2),
            })
        if st.lighting_alarm:
            bri = st.latest_brightness
            if st.low_light_alarm:
                alerts.append({
                    "code": "LOW_LIGHT",
                    "severity": "warning",
                    "message": (
                        f"Low light (brightness {bri if bri is not None else 0.0:.0f}/255, min {cfg.lighting.min_brightness:.0f})"
                    ),
                    "value": None if bri is None else round(bri, 1),
                })
            if st.bright_light_alarm:
                alerts.append({
                    "code": "BRIGHT_LIGHT",
                    "severity": "warning",
                    "message": (
                        f"Overexposed (brightness {bri if bri is not None else 255.0:.0f}/255, max {cfg.lighting.max_brightness:.0f})"
                    ),
                    "value": None if bri is None else round(bri, 1),
                })
        if st.out_of_focus_alarm:
            sharp = st.latest_sharpness
            alerts.append({
                "code": "OUT_OF_FOCUS",
                "severity": "warning",
                "message": (
                    f"Blurry frame (sharpness {sharp if sharp is not None else 0.0:.1f}, min {cfg.focus.min_sharpness:.1f})"
                ),
                "value": None if sharp is None else round(sharp, 1),
            })
        return alerts

    def _log_alert_transitions(self, alerts: List[dict]) -> None:
        current = {a["code"] for a in alerts}
        by_code = {a["code"]: a for a in alerts}
        for code in sorted(current - self._active_alert_codes):
            a = by_code[code]
            if a["severity"] == "critical":
                log.error("ALERT [%s]: %s", code, a["message"])
            else:
                log.warning("ALERT [%s]: %s", code, a["message"])
        for code in sorted(self._active_alert_codes - current):
            log.info("Alert cleared: %s", code)
        self._active_alert_codes = current

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
