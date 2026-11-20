"""Main orchestrator — ties video source, detectors, tracker, alarms together.

Run as::

    python -m drowsiness.app --source 0
    python -m drowsiness.app --source video.mp4
    python -m drowsiness.app --source 0 --no-cnn
"""
from __future__ import annotations

import argparse
import logging
import time
from dataclasses import asdict
from typing import Optional

import cv2
import numpy as np

from .alarms.sound_alarm import SoundAlarm
from .config import Config, load_config
from .detectors.base import Detection
from .detectors.cnn import CNNVerifier
from .detectors.geometric import GeometricDetector
from .state import DrowsinessTracker
from .video.source import VideoSource

log = logging.getLogger(__name__)


class DrowsinessMonitor:
    """High-level orchestrator."""

    def __init__(self, config: Config):
        self._cfg = config
        self._source = VideoSource(config.video)
        self._detector = GeometricDetector(config)
        self._tracker = DrowsinessTracker(config)
        self._alarm = SoundAlarm(config.alarm)
        self._cnn: Optional[CNNVerifier] = None
        self._stop = False
        self._last_cnn_frame = -1
        # Latest snapshot of the state — exposed for the Flask status API.
        from .state import FatigueState  # local import to avoid cycle
        self._latest_state = None

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def __enter__(self):
        self._source.open()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()
        self._source.close()
        self._detector.close()
        if self._cnn is not None:
            self._cnn.close()

    def stop(self) -> None:
        self._stop = True

    # ------------------------------------------------------------------ #
    # Optional CNN verifier
    # ------------------------------------------------------------------ #
    def enable_cnn(self) -> None:
        """Lazily enable the CNN verifier."""
        if self._cnn is None:
            self._cnn = CNNVerifier(self._cfg)

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    def run(self) -> None:
        """Run the drowsiness monitoring loop until ``stop()`` or end-of-stream."""
        cfg = self._cfg
        target_dt = 1.0 / cfg.video.target_fps
        next_frame_idx = 0
        if cfg.cnn.model_path and cfg.cnn.model_path.strip():
            try:
                self.enable_cnn()
            except Exception as exc:
                log.warning("CNN verifier unavailable, continuing without: %s", exc)

        log.info(
            "Alerts enabled: lighting=%s focus=%s distraction=%s "
            "prolonged_yawn(>%.1fs) look_away(<%.2f for %d frames)",
            cfg.lighting.enabled, cfg.focus.enabled, cfg.distraction.enabled,
            cfg.mouth.max_yawn_seconds,
            cfg.distraction.look_away_straightness, cfg.distraction.look_away_frames,
        )
        deadline = (cfg.run_for_seconds or 0.0)
        start = time.perf_counter()
        try:
            while not self._stop:
                if deadline and (time.perf_counter() - start) > deadline:
                    break
                frame = self._source.read()
                if frame is None:
                    log.info("End of stream reached.")
                    break
                frame_perf_t0 = time.perf_counter()
                # Pace to target FPS — drop the frame if we are behind.
                detection = self._detector.detect(frame.image)
                detection.frame_index = next_frame_idx
                next_frame_idx += 1

                # Optionally run the CNN verifier every Nth frame.
                if (
                    self._cnn is not None
                    and detection.face_found
                    and (detection.frame_index % cfg.cnn.run_every_n_frames) == 0
                ):
                    # Crop face ROI from the bbox.
                    x, y, w, h = detection.face_bbox
                    pad = max(8, int(0.1 * max(w, h)))
                    x0 = max(0, x - pad)
                    y0 = max(0, y - pad)
                    x1 = min(frame.image.shape[1], x + w + pad)
                    y1 = min(frame.image.shape[0], y + h + pad)
                    face_crop = frame.image[y0:y1, x0:x1]
                    if face_crop.size > 0:
                        cnn_det = self._cnn.classify_face(
                            face_crop, bbox=detection.face_bbox,
                        )
                        detection.cnn_class = cnn_det.cnn_class
                        detection.cnn_confidence = cnn_det.cnn_confidence

                # Update tracker + alarms.
                state = self._tracker.update(detection)
                self._latest_state = state
                self._maybe_trigger_alarm(state)

                # Render overlay (optional).
                if cfg.show_window:
                    self._render_overlay(frame.image, detection, state)
                    cv2.imshow("Drowsiness Detector", frame.image)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                # Pace.
                # Use perf_counter (monotonic) consistently — frame.timestamp
                # is wall-clock and must not be mixed with perf_counter.
                elapsed_since_frame = time.perf_counter() - frame_perf_t0
                remaining = target_dt - elapsed_since_frame
                if remaining > 0:
                    time.sleep(remaining)
        finally:
            if cfg.show_window:
                cv2.destroyAllWindows()

    # ------------------------------------------------------------------ #
    # Public introspection (used by Flask status API)
    # ------------------------------------------------------------------ #
    @property
    def latest_state(self):
        return self._latest_state

    @property
    def is_cnn_enabled(self) -> bool:
        return self._cnn is not None

    def status(self) -> dict:
        st = self._latest_state
        if st is None:
            return {"running": True, "frames_processed": 0, "any_alarm": False}
        return {
            "running": not self._stop,
            "frames_processed": st.frames_processed,
            "fps": round(st.fps, 2),
            "ear": st.latest_ear,
            "mar": st.latest_mar,
            "consecutive_closed_frames": st.consecutive_closed_frames,
            "consecutive_yawn_frames": st.consecutive_yawn_frames,
            "perclos": round(st.perclos, 4),
            "yawn_count_window": st.yawn_count_window,
            "yawn_rate_per_min": round(st.yawn_rate_per_min, 3),
            "eye_alarm": st.eye_alarm,
            "yawn_alarm": st.yawn_alarm,
            "perclos_alarm": st.perclos_alarm,
            "any_alarm": st.any_alarm,
            "alarm_reason": st.alarm_reason,
            # --- New: distraction / environment alerts (real-time) ---
            "face_missing_alarm": st.face_missing_alarm,
            "looking_away_alarm": st.looking_away_alarm,
            "prolonged_yawn_alarm": st.prolonged_yawn_alarm,
            "lighting_alarm": st.lighting_alarm,
            "low_light_alarm": st.low_light_alarm,
            "bright_light_alarm": st.bright_light_alarm,
            "out_of_focus_alarm": st.out_of_focus_alarm,
            "brightness": st.latest_brightness,
            "sharpness": st.latest_sharpness,
            "face_straightness": st.latest_face_straightness,
            "consecutive_no_face_seconds": round(st.consecutive_no_face_seconds, 2),
            "any_alert": st.any_alert,
            "any_issue": st.any_issue,
            "alerts": st.alerts,
            "alarm_active": self._alarm.is_active(),
            "cnn_enabled": self.is_cnn_enabled,
        }

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _maybe_trigger_alarm(self, state) -> None:
        sound_worthy_alert = (
            state.face_missing_alarm or state.prolonged_yawn_alarm
        )
        want_sound = state.any_alarm or (
            sound_worthy_alert and self._cfg.distraction.alarm_with_sound
        )
        if not want_sound:
            self._alarm.reset()
            return
        # Pick a reason + numeric metric for the alarm log.
        if state.eye_alarm:
            reason, metric = "eye_closure", state.consecutive_closed_frames
        elif state.yawn_alarm:
            reason, metric = "yawn_burst", state.yawn_rate_per_min
        elif state.perclos_alarm:
            reason, metric = "perclos", state.perclos
        elif state.face_missing_alarm:
            reason, metric = "face_not_visible", state.consecutive_no_face_seconds
        else:  # prolonged yawn
            value = next(
                (a.get("value") for a in state.alerts if a["code"] == "PROLONGED_YAWN"),
                0.0,
            ) or 0.0
            reason, metric = "prolonged_yawn", float(value)
        self._alarm.trigger(str(reason), float(metric))

    def _render_overlay(self, image: np.ndarray, det: Detection, state) -> None:
        """Draw face bbox, EAR/MAR readout, live alerts, and alarm banners."""
        def fmt(v, pattern="{:.3f}"):
            # None-safe formatting — must never crash on a no-face frame.
            return pattern.format(v) if isinstance(v, (int, float)) else "n/a"

        h, w = image.shape[:2]
        if det.face_found and det.face_bbox is not None:
            x, y, bw, bh = det.face_bbox
            color = (0, 0, 255) if state.any_issue else (0, 255, 0)
            cv2.rectangle(image, (x, y), (x + bw, y + bh), color, 2)

        def _tag_light():
            if state.low_light_alarm:
                return "  LOW LIGHT!"
            if state.bright_light_alarm:
                return "  TOO BRIGHT!"
            return ""

        def _tag_face():
            if state.face_missing_alarm:
                return "  NOT VISIBLE!"
            if state.looking_away_alarm:
                return "  LOOKING AWAY!"
            return ""

        lines = [
            f"FPS: {state.fps:.1f}",
            f"EAR: {fmt(det.ear)}" + (
                " CLOSED" if det.eyes_closed else (" open" if det.face_found else " (no face)")
            ),
            f"MAR: {fmt(det.mar)}" + (
                " YAWN" if det.yawning else ""
            ) + (
                "  PROLONGED!" if state.prolonged_yawn_alarm else ""
            ),
            f"Closed frames: {state.consecutive_closed_frames}",
            f"PERCLOS: {state.perclos:.3f}",
            f"Yawn rate/min: {state.yawn_rate_per_min:.2f}",
            f"Light: {fmt(state.latest_brightness, '{:.0f}')}/255{_tag_light()}",
            f"Sharpness: {fmt(state.latest_sharpness, '{:.0f}')}"
            + ("  BLURRY!" if state.out_of_focus_alarm else ""),
            f"Face straight: {fmt(state.latest_face_straightness, '{:.2f}')}{_tag_face()}",
            f"No-face time: {state.consecutive_no_face_seconds:.1f}s",
        ]
        if det.cnn_class is not None:
            lines.append(f"CNN: {det.cnn_class} ({fmt(det.cnn_confidence, '{:.2f}')})")
        for i, line in enumerate(lines):
            cv2.putText(
                image, line, (10, 22 + i * 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA,
            )

        # ---- Live alert banners, stacked from the bottom up ----
        # Critical drowsiness stays red/white; distraction-critical alerts
        # also red/white; warnings are amber/black.
        banners = list(state.alerts)  # already built with code/severity/message
        if state.any_alarm:
            banners.append({
                "code": "DROWSINESS",
                "severity": "critical",
                "message": f"DROWSINESS ALERT: {(state.alarm_reason or 'fatigue').upper()}",
            })
        banner_h = 46
        y = h
        for banner in banners[-3:]:  # at most 3 rows
            critical = banner["severity"] == "critical"
            bg = (0, 0, 255) if critical else (0, 165, 255)
            fg = (255, 255, 255) if critical else (0, 0, 0)
            cv2.rectangle(image, (0, y - banner_h), (w, y), bg, -1)
            text = f"*** {banner['code']}: {banner['message']}"
            if len(text) > 72:
                text = text[:72]
            cv2.putText(
                image, text, (10, y - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, fg, 2, cv2.LINE_AA,
            )
            y -= banner_h


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point — use ``python -m drowsiness.app``."""
    p = argparse.ArgumentParser(prog="drowsiness", description=__doc__)
    p.add_argument(
        "--source", default=None,
        help="Video source: 0 for webcam, or a file/URL path.",
    )
    p.add_argument("--config", default=None, help="Path to config.yaml.")
    p.add_argument("--no-window", action="store_true",
                  help="Disable the cv2 window (headless runs).")
    p.add_argument("--no-cnn", action="store_true",
                  help="Disable the CNN verifier (geometric detector only).")
    p.add_argument("--run-for", type=float, default=0.0,
                  help="Auto-stop after N seconds (0 = forever).")
    p.add_argument("--verbose", action="store_true",
                  help="Verbose logging.")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    overrides = {}
    if args.source is not None:
        try:
            source = int(args.source)
        except ValueError:
            source = args.source
        overrides["video"] = {"source": source}
    if args.no_window:
        overrides["show_window"] = False
    if args.no_cnn:
        overrides["cnn"] = {"model_path": ""}
    if args.run_for:
        overrides["run_for_seconds"] = float(args.run_for)

    cfg = load_config(args.config, overrides=overrides if overrides else None)
    # Respect the config.debug flag for log level.
    if cfg.debug and not args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    with DrowsinessMonitor(cfg) as mon:
        try:
            mon.run()
        except KeyboardInterrupt:
            log.info("Interrupted by user.")
        # Print a final summary so the user can see what happened.
        st = mon.latest_state
        if st is not None:
            log.info(
                "Final state: frames=%d fps=%.1f ear=%s mar=%s "
                "closed_frames=%d yawns=%d perclos=%.3f "
                "eye_alarm=%s yawn_alarm=%s perclos_alarm=%s",
                st.frames_processed, st.fps,
                f"{st.latest_ear:.3f}" if st.latest_ear is not None else "n/a",
                f"{st.latest_mar:.3f}" if st.latest_mar is not None else "n/a",
                st.consecutive_closed_frames,
                st.yawn_count_window,
                st.perclos,
                st.eye_alarm, st.yawn_alarm, st.perclos_alarm,
            )
            if st.any_alarm:
                log.warning("At least one alarm fired during the run.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
